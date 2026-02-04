"""
FedPLC - Robust GPU Version with CUDA Error Handling
Federated Learning dengan CIFAR-10 pada RTX 5060 Ti
"""

import os
import gc
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import time
import sys

# Config
NUM_CLIENTS = 10
NUM_ROUNDS = 15
LOCAL_EPOCHS = 2
BATCH_SIZE = 32

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.gap(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def main():
    print('='*60, flush=True)
    print('FedPLC - Federated Learning with GPU (Robust)', flush=True)  
    print('='*60, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}', flush=True)
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB', flush=True)
        
        # Warm up CUDA
        print('Warming up CUDA...', flush=True)
        x = torch.randn(1, 3, 32, 32, device=device)
        y = x * 2
        del x, y
        cleanup()
        print('CUDA warm-up complete', flush=True)

    print(f'\nConfig: {NUM_CLIENTS} clients, {NUM_ROUNDS} rounds, batch={BATCH_SIZE}', flush=True)

    # Load CIFAR-10
    print('\nLoading CIFAR-10...', flush=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10('./data', train=False, transform=transform)

    # Partition to clients (simple IID)
    samples_per_client = len(train_data) // NUM_CLIENTS
    client_loaders = []
    for i in range(NUM_CLIENTS):
        indices = list(range(i * samples_per_client, (i+1) * samples_per_client))
        subset = Subset(train_data, indices)
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=0, pin_memory=False)
        client_loaders.append(loader)

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=0, pin_memory=False)
    print(f'Data loaded: {NUM_CLIENTS} clients, {samples_per_client} samples each', flush=True)

    # Create global model
    global_model = SimpleCNN().to(device)
    print(f'Model params: {sum(p.numel() for p in global_model.parameters()):,}', flush=True)

    def train_client(model, loader, epochs, lr=0.01):
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        crit = nn.CrossEntropyLoss()
        
        for ep in range(epochs):
            for data, target in loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                opt.zero_grad()
                output = model(data)
                loss = crit(output, target)
                loss.backward()
                opt.step()
                
                del data, target, output, loss
        
        # Get state dict on CPU
        state = {}
        for k, v in model.state_dict().items():
            state[k] = v.detach().cpu().clone()
        
        return state

    def evaluate(model, loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(data)
                pred = output.argmax(1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                del data, target, output, pred
        cleanup()
        return 100.0 * correct / total

    def fed_avg(states):
        avg = {}
        for key in states[0].keys():
            stacked = torch.stack([s[key].float() for s in states])
            avg[key] = stacked.mean(0)
            del stacked
        return avg

    # Initial eval
    init_acc = evaluate(global_model, test_loader)
    print(f'Initial accuracy: {init_acc:.2f}%', flush=True)

    # Federated Training
    print('\n' + '='*60, flush=True)
    print('Starting Federated Training', flush=True)
    print('='*60, flush=True)

    best_acc = 0
    history = []
    start_time = time.time()
    
    for round_idx in range(NUM_ROUNDS):
        t0 = time.time()
        
        # Select 30% clients
        num_selected = max(1, int(NUM_CLIENTS * 0.3))
        selected = np.random.choice(NUM_CLIENTS, num_selected, replace=False)
        
        # Train clients
        client_states = []
        for cid in selected:
            # Create new model for each client
            client_model = SimpleCNN().to(device)
            client_model.load_state_dict(global_model.state_dict())
            
            try:
                state = train_client(client_model, client_loaders[cid], LOCAL_EPOCHS)
                client_states.append(state)
            except RuntimeError as e:
                print(f'Warning: Client {cid} failed: {e}', flush=True)
                continue
            finally:
                del client_model
                cleanup()
        
        if len(client_states) == 0:
            print(f'Round {round_idx+1}: No successful client training', flush=True)
            continue
            
        # Aggregate
        avg_state = fed_avg(client_states)
        global_model.load_state_dict(avg_state)
        del client_states, avg_state
        cleanup()
        
        # Evaluate
        acc = evaluate(global_model, test_loader)
        best_acc = max(best_acc, acc)
        history.append(acc)
        
        elapsed = time.time() - t0
        print(f'Round {round_idx+1:2d}/{NUM_ROUNDS} | Acc: {acc:.2f}% | Best: {best_acc:.2f}% | Time: {elapsed:.1f}s', flush=True)

    total_time = time.time() - start_time
    
    print('\n' + '='*60, flush=True)
    print('TRAINING COMPLETE!', flush=True)
    print('='*60, flush=True)
    print(f'Final Accuracy: {history[-1]:.2f}%', flush=True)
    print(f'Best Accuracy: {best_acc:.2f}%', flush=True)
    print(f'Total Time: {total_time:.1f}s ({total_time/NUM_ROUNDS:.1f}s per round)', flush=True)
    print(f'Accuracy History: {[f"{a:.1f}" for a in history]}', flush=True)
    print('='*60, flush=True)
    print('\n*** GPU FEDERATED LEARNING SUCCESS! ***', flush=True)
    
    # Save results
    results = {
        'best_acc': best_acc,
        'final_acc': history[-1],
        'history': history,
        'total_time': total_time,
        'config': {
            'num_clients': NUM_CLIENTS,
            'num_rounds': NUM_ROUNDS,
            'local_epochs': LOCAL_EPOCHS,
            'batch_size': BATCH_SIZE
        }
    }
    
    torch.save(results, 'fedplc_results.pt')
    print(f'\nResults saved to fedplc_results.pt', flush=True)
    
    return best_acc, history

if __name__ == "__main__":
    try:
        best_acc, history = main()
        sys.exit(0)
    except Exception as e:
        print(f'\nERROR: {e}', flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
