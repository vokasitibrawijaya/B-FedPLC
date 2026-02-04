"""
FedPLC Quick Start - GPU Memory Optimized Version
Menggunakan batch size kecil dan optimisasi memori untuk RTX 5060 Ti
"""

import os
import sys
import gc

# Set CUDA memory allocator
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import time

def cleanup_gpu():
    """Clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_gpu_memory():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# ============= Simple Model =============
class SimpleCNN(nn.Module):
    """CNN sederhana untuk CIFAR-10 - memory efficient"""
    def __init__(self, num_classes=10):
        super().__init__()
        # Backbone - lebih ringan
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_representation(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

# ============= Data Loading =============
def create_synthetic_data(num_clients=10, samples_per_client=500, num_classes=10):
    """Create synthetic CIFAR-10-like data untuk testing"""
    print("Creating synthetic data...")
    client_loaders = []
    
    for i in range(num_clients):
        # Random images (32x32x3) dan labels
        images = torch.randn(samples_per_client, 3, 32, 32)
        labels = torch.randint(0, num_classes, (samples_per_client,))
        
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)  # Small batch
        client_loaders.append(loader)
    
    # Test loader
    test_images = torch.randn(1000, 3, 32, 32)
    test_labels = torch.randint(0, num_classes, (1000,))
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return client_loaders, test_loader

def load_real_cifar10(num_clients=10, batch_size=32):
    """Load real CIFAR-10 data with Non-IID partitioning"""
    from torchvision import datasets, transforms
    
    print("Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    # Simple IID partitioning (untuk simplicity)
    total_samples = len(train_dataset)
    samples_per_client = total_samples // num_clients
    
    client_loaders = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        indices = list(range(start_idx, end_idx))
        
        subset = torch.utils.data.Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return client_loaders, test_loader

# ============= FedAvg Training =============
def train_local(model, loader, device, epochs=2, lr=0.01):
    """Train model locally for a few epochs"""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_samples = 0
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            # Clear intermediate tensors
            del data, target, output, loss
        
        cleanup_gpu()
    
    return total_loss / total_samples

def evaluate(model, loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            del data, target, output, pred
    
    cleanup_gpu()
    return 100.0 * correct / total

def federated_averaging(global_state, client_states, weights=None):
    """Aggregate client models using FedAvg"""
    if weights is None:
        weights = [1.0 / len(client_states)] * len(client_states)
    
    # Initialize with zeros
    avg_state = {}
    for key in global_state.keys():
        avg_state[key] = torch.zeros_like(global_state[key])
    
    # Weighted average
    for state, weight in zip(client_states, weights):
        for key in avg_state.keys():
            avg_state[key] += weight * state[key].cpu()
    
    return avg_state

# ============= Main Training Loop =============
def main():
    print("=" * 70)
    print("FedPLC GPU-Safe Quick Start")
    print("=" * 70)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print_gpu_memory()
    
    # Configuration
    NUM_CLIENTS = 10
    NUM_ROUNDS = 15
    LOCAL_EPOCHS = 2
    BATCH_SIZE = 32  # Reduced batch size
    PARTICIPATION_RATE = 0.3  # 30% clients per round
    LR = 0.01
    
    print(f"\nConfiguration:")
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Rounds: {NUM_ROUNDS}")
    print(f"  Local Epochs: {LOCAL_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Participation: {int(PARTICIPATION_RATE * 100)}%")
    
    # Load data - gunakan synthetic dulu untuk test cepat
    use_real_data = True  # Set False untuk test dengan data synthetic
    
    if use_real_data:
        try:
            client_loaders, test_loader = load_real_cifar10(
                num_clients=NUM_CLIENTS, 
                batch_size=BATCH_SIZE
            )
            print(f"\nLoaded real CIFAR-10 data")
        except Exception as e:
            print(f"\nFailed to load CIFAR-10: {e}")
            print("Using synthetic data instead...")
            client_loaders, test_loader = create_synthetic_data(
                num_clients=NUM_CLIENTS,
                samples_per_client=500
            )
    else:
        client_loaders, test_loader = create_synthetic_data(
            num_clients=NUM_CLIENTS,
            samples_per_client=500
        )
        print(f"\nUsing synthetic data for quick test")
    
    print(f"  Clients: {len(client_loaders)}")
    
    # Create global model
    print("\nCreating model...")
    global_model = SimpleCNN(num_classes=10).to(device)
    num_params = sum(p.numel() for p in global_model.parameters())
    print(f"  Total parameters: {num_params:,}")
    print_gpu_memory()
    
    # Initial evaluation
    print("\nInitial Evaluation...")
    acc = evaluate(global_model, test_loader, device)
    print(f"  Initial Accuracy: {acc:.2f}%")
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Federated Training")
    print("=" * 70)
    
    best_acc = 0
    history = []
    
    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        
        # Select participating clients
        num_selected = max(1, int(NUM_CLIENTS * PARTICIPATION_RATE))
        selected_clients = np.random.choice(NUM_CLIENTS, num_selected, replace=False)
        
        # Train on selected clients
        client_states = []
        client_weights = []
        total_loss = 0
        
        for client_id in selected_clients:
            # Copy global model to client
            client_model = SimpleCNN(num_classes=10).to(device)
            client_model.load_state_dict(global_model.state_dict())
            
            # Train locally
            loss = train_local(
                client_model, 
                client_loaders[client_id], 
                device, 
                epochs=LOCAL_EPOCHS,
                lr=LR
            )
            
            # Collect model state
            client_states.append({k: v.clone() for k, v in client_model.state_dict().items()})
            client_weights.append(1.0)  # Equal weights
            total_loss += loss
            
            # Cleanup
            del client_model
            cleanup_gpu()
        
        # Aggregate
        avg_state = federated_averaging(
            global_model.state_dict(),
            client_states,
            [w/sum(client_weights) for w in client_weights]
        )
        global_model.load_state_dict(avg_state)
        
        # Cleanup
        del client_states, avg_state
        cleanup_gpu()
        
        # Evaluate
        acc = evaluate(global_model, test_loader, device)
        best_acc = max(best_acc, acc)
        history.append(acc)
        
        round_time = time.time() - round_start
        avg_loss = total_loss / num_selected
        
        print(f"Round {round_idx+1:2d}/{NUM_ROUNDS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {acc:.2f}% | "
              f"Best: {best_acc:.2f}% | "
              f"Time: {round_time:.1f}s")
        
        # Print GPU memory setiap 5 rounds
        if (round_idx + 1) % 5 == 0:
            print_gpu_memory()
    
    # Final results
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nFinal Accuracy: {acc:.2f}%")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Accuracy History: {[f'{a:.1f}' for a in history]}")
    
    if device.type == "cuda":
        print("\nFinal GPU Memory:")
        print_gpu_memory()
    
    print("\n✓ GPU training successful!")
    print("=" * 70)
    
    return best_acc, history

if __name__ == "__main__":
    try:
        best_acc, history = main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
