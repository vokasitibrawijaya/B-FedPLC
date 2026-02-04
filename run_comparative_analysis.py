"""
Comparative Analysis: B-FedPLC vs Baseline FL Algorithms
=========================================================
Compares:
1. FedAvg - Standard Federated Averaging
2. FedProx - Proximal Federated Optimization
3. SCAFFOLD - Stochastic Controlled Averaging
4. B-FedPLC - Our proposed method (Blockchain + PARL + LDCA)

For doctoral dissertation (S3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import json
import copy
from datetime import datetime
from pathlib import Path


# ============================================================================
# Model Definition (same for all methods)
# ============================================================================
class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================================
# Data Partitioning
# ============================================================================
def dirichlet_partition(dataset, num_clients, alpha=0.5, num_classes=10):
    """Partition dataset using Dirichlet distribution for Non-IID"""
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        
        class_splits = np.split(class_indices, proportions)
        
        for client_id, indices in enumerate(class_splits):
            client_indices[client_id].extend(indices)
    
    return client_indices


# ============================================================================
# FedAvg Implementation
# ============================================================================
class FedAvg:
    """Standard Federated Averaging"""
    
    def __init__(self, num_clients, device):
        self.num_clients = num_clients
        self.device = device
        self.name = "FedAvg"
    
    def train_client(self, model, train_loader, epochs=3, lr=0.01):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(epochs):
            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        return model
    
    def aggregate(self, global_model, client_models, weights):
        global_state = global_model.state_dict()
        
        for key in global_state.keys():
            if 'num_batches_tracked' in key:
                continue
            weighted_sum = torch.zeros_like(global_state[key], dtype=torch.float32)
            for client_model, w in zip(client_models, weights):
                weighted_sum += w * client_model.state_dict()[key].float()
            global_state[key] = weighted_sum.to(global_state[key].dtype)
        
        global_model.load_state_dict(global_state)
        return global_model


# ============================================================================
# FedProx Implementation
# ============================================================================
class FedProx:
    """Federated Optimization with Proximal Term"""
    
    def __init__(self, num_clients, device, mu=0.01):
        self.num_clients = num_clients
        self.device = device
        self.mu = mu  # Proximal term weight
        self.name = f"FedProx(mu={mu})"
    
    def train_client(self, model, train_loader, global_model, epochs=3, lr=0.01):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        global_params = {name: param.clone() for name, param in global_model.named_parameters()}
        
        for _ in range(epochs):
            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(data)
                
                # CE loss
                loss = criterion(outputs, labels)
                
                # Proximal term
                prox_term = 0
                for name, param in model.named_parameters():
                    prox_term += ((param - global_params[name]) ** 2).sum()
                loss += (self.mu / 2) * prox_term
                
                loss.backward()
                optimizer.step()
        
        return model
    
    def aggregate(self, global_model, client_models, weights):
        # Same as FedAvg
        global_state = global_model.state_dict()
        
        for key in global_state.keys():
            if 'num_batches_tracked' in key:
                continue
            weighted_sum = torch.zeros_like(global_state[key], dtype=torch.float32)
            for client_model, w in zip(client_models, weights):
                weighted_sum += w * client_model.state_dict()[key].float()
            global_state[key] = weighted_sum.to(global_state[key].dtype)
        
        global_model.load_state_dict(global_state)
        return global_model


# ============================================================================
# SCAFFOLD Implementation
# ============================================================================
class SCAFFOLD:
    """Stochastic Controlled Averaging for Federated Learning"""
    
    def __init__(self, num_clients, device, model_template):
        self.num_clients = num_clients
        self.device = device
        self.name = "SCAFFOLD"
        
        # Initialize control variates
        self.c_global = {name: torch.zeros_like(param) 
                        for name, param in model_template.named_parameters()}
        self.c_local = [{name: torch.zeros_like(param) 
                        for name, param in model_template.named_parameters()} 
                       for _ in range(num_clients)]
    
    def train_client(self, model, train_loader, global_model, client_idx, 
                    epochs=3, lr=0.01):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        c_global = self.c_global
        c_local = self.c_local[client_idx]
        
        num_steps = 0
        for _ in range(epochs):
            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Apply SCAFFOLD correction
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad.data += c_global[name].to(self.device) - c_local[name].to(self.device)
                
                optimizer.step()
                num_steps += 1
        
        # Update local control variate
        new_c_local = {}
        for name, param in model.named_parameters():
            global_param = dict(global_model.named_parameters())[name]
            new_c_local[name] = c_local[name] - c_global[name] + \
                               (global_param.data - param.data) / (num_steps * lr)
        
        # Compute delta c
        delta_c = {name: new_c_local[name] - c_local[name] for name in c_local}
        self.c_local[client_idx] = new_c_local
        
        return model, delta_c
    
    def aggregate(self, global_model, client_models, weights, delta_cs):
        # Aggregate models (FedAvg style)
        global_state = global_model.state_dict()
        
        for key in global_state.keys():
            if 'num_batches_tracked' in key:
                continue
            weighted_sum = torch.zeros_like(global_state[key], dtype=torch.float32)
            for client_model, w in zip(client_models, weights):
                weighted_sum += w * client_model.state_dict()[key].float()
            global_state[key] = weighted_sum.to(global_state[key].dtype)
        
        global_model.load_state_dict(global_state)
        
        # Update global control variate
        for name in self.c_global:
            delta_sum = torch.zeros_like(self.c_global[name])
            for delta_c, w in zip(delta_cs, weights):
                delta_sum += w * delta_c[name]
            self.c_global[name] += delta_sum * len(delta_cs) / self.num_clients
        
        return global_model


# ============================================================================
# Evaluation
# ============================================================================
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


# ============================================================================
# Run Single Algorithm
# ============================================================================
def run_algorithm(algorithm, config, train_dataset, test_loader, client_indices, device):
    """Run a single FL algorithm"""
    
    print(f"\n{'='*60}")
    print(f"Running: {algorithm.name}")
    print(f"{'='*60}")
    
    # Initialize global model
    global_model = SimpleCNN(num_classes=10).to(device)
    
    history = {'accuracy': [], 'rounds': []}
    best_accuracy = 0
    start_time = time.time()
    
    for round_num in range(1, config['num_rounds'] + 1):
        # Select clients
        num_selected = max(1, int(config['num_clients'] * config['client_fraction']))
        selected_clients = np.random.choice(
            config['num_clients'], num_selected, replace=False
        )
        
        client_models = []
        weights = []
        delta_cs = []  # For SCAFFOLD
        total_samples = sum(len(client_indices[c]) for c in selected_clients)
        
        for client_idx in selected_clients:
            # Create client model
            client_model = copy.deepcopy(global_model)
            
            # Client data
            indices = client_indices[client_idx]
            subset = Subset(train_dataset, indices)
            loader = DataLoader(subset, batch_size=config['batch_size'], shuffle=True)
            
            # Train based on algorithm type
            if isinstance(algorithm, FedAvg):
                client_model = algorithm.train_client(
                    client_model, loader, 
                    epochs=config['local_epochs'], 
                    lr=config['learning_rate']
                )
            elif isinstance(algorithm, FedProx):
                client_model = algorithm.train_client(
                    client_model, loader, global_model,
                    epochs=config['local_epochs'], 
                    lr=config['learning_rate']
                )
            elif isinstance(algorithm, SCAFFOLD):
                client_model, delta_c = algorithm.train_client(
                    client_model, loader, global_model, client_idx,
                    epochs=config['local_epochs'], 
                    lr=config['learning_rate']
                )
                delta_cs.append(delta_c)
            
            client_models.append(client_model)
            weights.append(len(indices) / total_samples)
        
        # Aggregate
        if isinstance(algorithm, SCAFFOLD):
            global_model = algorithm.aggregate(global_model, client_models, weights, delta_cs)
        else:
            global_model = algorithm.aggregate(global_model, client_models, weights)
        
        # Evaluate
        accuracy = evaluate(global_model, test_loader, device)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        
        history['rounds'].append(round_num)
        history['accuracy'].append(accuracy)
        
        if round_num % 10 == 0 or round_num == 1:
            print(f"Round {round_num:3d}/{config['num_rounds']} | "
                  f"Acc: {accuracy:.2f}% | Best: {best_accuracy:.2f}%")
    
    total_time = time.time() - start_time
    
    return {
        'name': algorithm.name,
        'history': history,
        'best_accuracy': best_accuracy,
        'final_accuracy': history['accuracy'][-1],
        'training_time': total_time
    }


# ============================================================================
# Main Comparative Analysis
# ============================================================================
def run_comparative_analysis():
    print("="*70)
    print("Comparative Analysis: B-FedPLC vs Baseline FL Algorithms")
    print("="*70)
    
    config = {
        'num_clients': 50,
        'num_rounds': 100,
        'client_fraction': 0.2,
        'local_epochs': 3,
        'batch_size': 32,
        'learning_rate': 0.01,
        'dirichlet_alpha': 0.5
    }
    
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Load CIFAR-10
    print("\nLoading CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Partition data (same for all algorithms - fair comparison)
    print(f"\nPartitioning data (Dirichlet alpha={config['dirichlet_alpha']})...")
    np.random.seed(42)  # Fixed seed for reproducibility
    client_indices = dirichlet_partition(
        train_dataset, config['num_clients'], config['dirichlet_alpha']
    )
    
    # Initialize algorithms
    model_template = SimpleCNN(num_classes=10).to(device)
    
    algorithms = [
        FedAvg(config['num_clients'], device),
        FedProx(config['num_clients'], device, mu=0.01),
        SCAFFOLD(config['num_clients'], device, model_template),
    ]
    
    # Run all algorithms
    results = []
    
    for algo in algorithms:
        np.random.seed(42)  # Same random seed for fair comparison
        torch.manual_seed(42)
        
        result = run_algorithm(
            algo, config, train_dataset, test_loader, client_indices, device
        )
        results.append(result)
    
    # Load B-FedPLC results
    print("\n" + "="*60)
    print("Loading B-FedPLC results...")
    print("="*60)
    
    try:
        with open('b_fedplc_results.json', 'r') as f:
            b_fedplc = json.load(f)
        
        results.append({
            'name': 'B-FedPLC',
            'history': {
                'rounds': b_fedplc['history']['rounds'],
                'accuracy': b_fedplc['history']['accuracy']
            },
            'best_accuracy': b_fedplc['best_accuracy'],
            'final_accuracy': b_fedplc['final_accuracy'],
            'training_time': b_fedplc['training_time_minutes'] * 60
        })
        print(f"B-FedPLC: Best {b_fedplc['best_accuracy']:.2f}%")
    except FileNotFoundError:
        print("Warning: B-FedPLC results not found. Run run_b_fedplc.py first.")
    
    # Summary
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS RESULTS")
    print("="*70)
    print(f"\n{'Algorithm':<20} {'Best Acc':<12} {'Final Acc':<12} {'Time (min)':<12}")
    print("-"*56)
    
    for r in results:
        print(f"{r['name']:<20} {r['best_accuracy']:<12.2f} "
              f"{r['final_accuracy']:<12.2f} {r['training_time']/60:<12.1f}")
    
    # Save results
    save_data = {
        'config': config,
        'results': [{
            'name': r['name'],
            'history': r['history'],
            'best_accuracy': r['best_accuracy'],
            'final_accuracy': r['final_accuracy'],
            'training_time_minutes': r['training_time'] / 60
        } for r in results],
        'timestamp': datetime.now().isoformat()
    }
    
    with open('comparative_analysis_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: comparative_analysis_results.json")
    
    return results


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    results = run_comparative_analysis()
    
    print("\n" + "="*70)
    print("Comparative analysis completed!")
    print("Run: python visualize_comparison.py to generate comparison plots")
    print("="*70)
