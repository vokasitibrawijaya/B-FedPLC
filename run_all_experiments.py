"""
Comprehensive B-FedPLC Experiments for Dissertation
Runs all experiments sequentially:
1. Ablation Study
2. Scalability Analysis
3. Non-IID Sensitivity
4. Security Analysis (Byzantine Fault Tolerance)
5. Communication Efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import time
import copy
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
BASE_CONFIG = {
    'num_clients': 50,
    'num_rounds': 50,  # Reduced for faster experiments
    'client_fraction': 0.2,
    'local_epochs': 3,
    'batch_size': 32,
    'learning_rate': 0.01,
    'dirichlet_alpha': 0.5,
    'seed': 42
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Model Definition
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ============================================================
# Data Loading and Partitioning
# ============================================================
def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

def dirichlet_partition(dataset, num_clients, alpha, seed=42):
    np.random.seed(seed)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))
    
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = proportions / proportions.sum()
        
        splits = (proportions * len(class_indices)).astype(int)
        splits[-1] = len(class_indices) - splits[:-1].sum()
        
        start = 0
        for client_id, num_samples in enumerate(splits):
            client_indices[client_id].extend(class_indices[start:start+num_samples].tolist())
            start += num_samples
    
    return client_indices

# ============================================================
# LDCA Clustering
# ============================================================
def compute_label_distribution(dataset, indices, num_classes=10):
    labels = [dataset[i][1] for i in indices]
    dist = np.zeros(num_classes)
    for l in labels:
        dist[l] += 1
    return dist / (dist.sum() + 1e-8)

def ldca_clustering(client_distributions, threshold=0.3):
    num_clients = len(client_distributions)
    communities = []
    assigned = set()
    
    for i in range(num_clients):
        if i in assigned:
            continue
        community = [i]
        assigned.add(i)
        
        for j in range(i+1, num_clients):
            if j in assigned:
                continue
            similarity = 1 - 0.5 * np.sum(np.abs(client_distributions[i] - client_distributions[j]))
            if similarity > threshold:
                community.append(j)
                assigned.add(j)
        
        communities.append(community)
    
    return communities

# ============================================================
# Training Functions
# ============================================================
def train_client(model, dataloader, epochs, lr, use_parl=True, global_model=None, mu=0.01):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    if use_parl and global_model:
        global_params = {n: p.clone() for n, p in global_model.named_parameters()}
    
    for _ in range(epochs):
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            if use_parl and global_model:
                parl_loss = 0
                for name, param in model.named_parameters():
                    parl_loss += ((param - global_params[name]) ** 2).sum()
                loss += (mu / 2) * parl_loss
            
            loss.backward()
            optimizer.step()
    
    return model.state_dict()

def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return 100. * correct / total

def aggregate_models(global_model, client_models, weights=None):
    if weights is None:
        weights = [1.0 / len(client_models)] * len(client_models)
    
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
        for w, client_model in zip(weights, client_models):
            global_dict[key] += w * client_model[key].float()
    
    global_model.load_state_dict(global_dict)
    return global_model

# ============================================================
# Experiment 1: Ablation Study
# ============================================================
def run_ablation_study(trainset, testset, testloader, client_indices):
    print("\n" + "="*70)
    print("EXPERIMENT 1: ABLATION STUDY")
    print("="*70)
    
    results = {}
    configs = [
        ("Full B-FedPLC", True, True),
        ("Without LDCA", False, True),
        ("Without PARL", True, False),
        ("Without Both", False, False),
    ]
    
    for name, use_ldca, use_parl in configs:
        print(f"\n--- Running: {name} ---")
        
        model = SimpleCNN().to(DEVICE)
        history = []
        best_acc = 0
        start_time = time.time()
        
        # Compute communities if using LDCA
        if use_ldca:
            client_dists = [compute_label_distribution(trainset, idx) for idx in client_indices]
            communities = ldca_clustering(client_dists)
        
        for round_num in range(BASE_CONFIG['num_rounds']):
            # Select clients
            num_selected = max(1, int(BASE_CONFIG['num_clients'] * BASE_CONFIG['client_fraction']))
            selected = np.random.choice(BASE_CONFIG['num_clients'], num_selected, replace=False)
            
            client_models = []
            client_weights = []
            
            for client_id in selected:
                client_model = SimpleCNN().to(DEVICE)
                client_model.load_state_dict(model.state_dict())
                
                indices = client_indices[client_id]
                subset = Subset(trainset, indices)
                loader = DataLoader(subset, batch_size=BASE_CONFIG['batch_size'], shuffle=True)
                
                state = train_client(
                    client_model, loader, 
                    BASE_CONFIG['local_epochs'], 
                    BASE_CONFIG['learning_rate'],
                    use_parl=use_parl,
                    global_model=model if use_parl else None
                )
                
                client_models.append(state)
                client_weights.append(len(indices))
            
            # Normalize weights
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]
            
            # Aggregate
            model = aggregate_models(model, client_models, client_weights)
            
            # Evaluate
            acc = evaluate(model, testloader)
            history.append(acc)
            best_acc = max(best_acc, acc)
            
            if (round_num + 1) % 10 == 0:
                print(f"Round {round_num+1:3d}/{BASE_CONFIG['num_rounds']} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
        
        elapsed = time.time() - start_time
        results[name] = {
            'best_accuracy': best_acc,
            'final_accuracy': history[-1],
            'history': history,
            'time_minutes': elapsed / 60
        }
        print(f"Completed: {name} - Best: {best_acc:.2f}% in {elapsed/60:.1f} min")
    
    return results

# ============================================================
# Experiment 2: Scalability Analysis
# ============================================================
def run_scalability_analysis(trainset, testset, testloader):
    print("\n" + "="*70)
    print("EXPERIMENT 2: SCALABILITY ANALYSIS")
    print("="*70)
    
    results = {}
    client_counts = [10, 30, 50, 100]
    
    for num_clients in client_counts:
        print(f"\n--- Running with {num_clients} clients ---")
        
        # Re-partition data
        client_indices = dirichlet_partition(trainset, num_clients, BASE_CONFIG['dirichlet_alpha'])
        
        model = SimpleCNN().to(DEVICE)
        history = []
        best_acc = 0
        start_time = time.time()
        
        for round_num in range(BASE_CONFIG['num_rounds']):
            num_selected = max(1, int(num_clients * BASE_CONFIG['client_fraction']))
            selected = np.random.choice(num_clients, num_selected, replace=False)
            
            client_models = []
            client_weights = []
            
            for client_id in selected:
                client_model = SimpleCNN().to(DEVICE)
                client_model.load_state_dict(model.state_dict())
                
                indices = client_indices[client_id]
                subset = Subset(trainset, indices)
                loader = DataLoader(subset, batch_size=BASE_CONFIG['batch_size'], shuffle=True)
                
                state = train_client(
                    client_model, loader,
                    BASE_CONFIG['local_epochs'],
                    BASE_CONFIG['learning_rate'],
                    use_parl=True,
                    global_model=model
                )
                
                client_models.append(state)
                client_weights.append(len(indices))
            
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]
            
            model = aggregate_models(model, client_models, client_weights)
            acc = evaluate(model, testloader)
            history.append(acc)
            best_acc = max(best_acc, acc)
            
            if (round_num + 1) % 10 == 0:
                print(f"Round {round_num+1:3d}/{BASE_CONFIG['num_rounds']} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
        
        elapsed = time.time() - start_time
        results[f"{num_clients}_clients"] = {
            'num_clients': num_clients,
            'best_accuracy': best_acc,
            'final_accuracy': history[-1],
            'history': history,
            'time_minutes': elapsed / 60
        }
        print(f"Completed: {num_clients} clients - Best: {best_acc:.2f}%")
    
    return results

# ============================================================
# Experiment 3: Non-IID Sensitivity
# ============================================================
def run_noniid_sensitivity(trainset, testset, testloader):
    print("\n" + "="*70)
    print("EXPERIMENT 3: NON-IID SENSITIVITY ANALYSIS")
    print("="*70)
    
    results = {}
    alpha_values = [0.1, 0.3, 0.5, 1.0]  # Lower = more Non-IID
    
    for alpha in alpha_values:
        print(f"\n--- Running with Dirichlet alpha={alpha} ---")
        
        # Re-partition with different alpha
        client_indices = dirichlet_partition(trainset, BASE_CONFIG['num_clients'], alpha)
        
        model = SimpleCNN().to(DEVICE)
        history = []
        best_acc = 0
        start_time = time.time()
        
        for round_num in range(BASE_CONFIG['num_rounds']):
            num_selected = max(1, int(BASE_CONFIG['num_clients'] * BASE_CONFIG['client_fraction']))
            selected = np.random.choice(BASE_CONFIG['num_clients'], num_selected, replace=False)
            
            client_models = []
            client_weights = []
            
            for client_id in selected:
                client_model = SimpleCNN().to(DEVICE)
                client_model.load_state_dict(model.state_dict())
                
                indices = client_indices[client_id]
                subset = Subset(trainset, indices)
                loader = DataLoader(subset, batch_size=BASE_CONFIG['batch_size'], shuffle=True)
                
                state = train_client(
                    client_model, loader,
                    BASE_CONFIG['local_epochs'],
                    BASE_CONFIG['learning_rate'],
                    use_parl=True,
                    global_model=model
                )
                
                client_models.append(state)
                client_weights.append(len(indices))
            
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]
            
            model = aggregate_models(model, client_models, client_weights)
            acc = evaluate(model, testloader)
            history.append(acc)
            best_acc = max(best_acc, acc)
            
            if (round_num + 1) % 10 == 0:
                print(f"Round {round_num+1:3d}/{BASE_CONFIG['num_rounds']} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
        
        elapsed = time.time() - start_time
        results[f"alpha_{alpha}"] = {
            'alpha': alpha,
            'best_accuracy': best_acc,
            'final_accuracy': history[-1],
            'history': history,
            'time_minutes': elapsed / 60
        }
        print(f"Completed: alpha={alpha} - Best: {best_acc:.2f}%")
    
    return results

# ============================================================
# Experiment 4: Security Analysis (Byzantine Fault Tolerance)
# ============================================================
def run_security_analysis(trainset, testset, testloader, client_indices):
    print("\n" + "="*70)
    print("EXPERIMENT 4: SECURITY ANALYSIS (Byzantine Fault Tolerance)")
    print("="*70)
    
    results = {}
    byzantine_fractions = [0.0, 0.1, 0.2, 0.3]
    
    for byz_frac in byzantine_fractions:
        print(f"\n--- Running with {int(byz_frac*100)}% Byzantine clients ---")
        
        model = SimpleCNN().to(DEVICE)
        history = []
        best_acc = 0
        attacks_detected = 0
        start_time = time.time()
        
        num_byzantine = int(BASE_CONFIG['num_clients'] * byz_frac)
        byzantine_clients = set(np.random.choice(BASE_CONFIG['num_clients'], num_byzantine, replace=False))
        
        for round_num in range(BASE_CONFIG['num_rounds']):
            num_selected = max(1, int(BASE_CONFIG['num_clients'] * BASE_CONFIG['client_fraction']))
            selected = np.random.choice(BASE_CONFIG['num_clients'], num_selected, replace=False)
            
            client_models = []
            client_weights = []
            client_norms = []
            
            for client_id in selected:
                client_model = SimpleCNN().to(DEVICE)
                client_model.load_state_dict(model.state_dict())
                
                indices = client_indices[client_id]
                subset = Subset(trainset, indices)
                loader = DataLoader(subset, batch_size=BASE_CONFIG['batch_size'], shuffle=True)
                
                state = train_client(
                    client_model, loader,
                    BASE_CONFIG['local_epochs'],
                    BASE_CONFIG['learning_rate'],
                    use_parl=True,
                    global_model=model
                )
                
                # Byzantine attack: random noise or sign flip
                if client_id in byzantine_clients:
                    for key in state:
                        if 'weight' in key or 'bias' in key:
                            # Random noise attack
                            state[key] = state[key] + torch.randn_like(state[key]) * 10
                
                # Compute update norm for anomaly detection
                update_norm = 0
                global_state = model.state_dict()
                for key in state:
                    if 'weight' in key or 'bias' in key:
                        update_norm += ((state[key] - global_state[key]) ** 2).sum().item()
                update_norm = np.sqrt(update_norm)
                
                client_models.append(state)
                client_weights.append(len(indices))
                client_norms.append(update_norm)
            
            # Simple anomaly detection: filter out updates with extreme norms
            median_norm = np.median(client_norms)
            threshold = median_norm * 3
            
            filtered_models = []
            filtered_weights = []
            for i, (m, w, norm) in enumerate(zip(client_models, client_weights, client_norms)):
                if norm < threshold:
                    filtered_models.append(m)
                    filtered_weights.append(w)
                else:
                    attacks_detected += 1
            
            if len(filtered_models) == 0:
                filtered_models = client_models
                filtered_weights = client_weights
            
            total_weight = sum(filtered_weights)
            filtered_weights = [w / total_weight for w in filtered_weights]
            
            model = aggregate_models(model, filtered_models, filtered_weights)
            acc = evaluate(model, testloader)
            history.append(acc)
            best_acc = max(best_acc, acc)
            
            if (round_num + 1) % 10 == 0:
                print(f"Round {round_num+1:3d}/{BASE_CONFIG['num_rounds']} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
        
        elapsed = time.time() - start_time
        results[f"byzantine_{int(byz_frac*100)}pct"] = {
            'byzantine_fraction': byz_frac,
            'best_accuracy': best_acc,
            'final_accuracy': history[-1],
            'history': history,
            'attacks_detected': attacks_detected,
            'time_minutes': elapsed / 60
        }
        print(f"Completed: {int(byz_frac*100)}% Byzantine - Best: {best_acc:.2f}%, Attacks Detected: {attacks_detected}")
    
    return results

# ============================================================
# Experiment 5: Communication Efficiency
# ============================================================
def run_communication_efficiency(trainset, testset, testloader, client_indices):
    print("\n" + "="*70)
    print("EXPERIMENT 5: COMMUNICATION EFFICIENCY ANALYSIS")
    print("="*70)
    
    results = {}
    
    # Compute model size
    model = SimpleCNN()
    model_size_bytes = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per float32
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    print(f"Model size: {model_size_mb:.2f} MB ({model_size_bytes:,} bytes)")
    
    algorithms = ['FedAvg', 'B-FedPLC']
    
    for algo in algorithms:
        print(f"\n--- Running: {algo} ---")
        
        model = SimpleCNN().to(DEVICE)
        history = []
        best_acc = 0
        total_bytes_transferred = 0
        start_time = time.time()
        
        for round_num in range(BASE_CONFIG['num_rounds']):
            num_selected = max(1, int(BASE_CONFIG['num_clients'] * BASE_CONFIG['client_fraction']))
            selected = np.random.choice(BASE_CONFIG['num_clients'], num_selected, replace=False)
            
            # Download: server -> clients
            total_bytes_transferred += model_size_bytes * num_selected
            
            client_models = []
            client_weights = []
            
            for client_id in selected:
                client_model = SimpleCNN().to(DEVICE)
                client_model.load_state_dict(model.state_dict())
                
                indices = client_indices[client_id]
                subset = Subset(trainset, indices)
                loader = DataLoader(subset, batch_size=BASE_CONFIG['batch_size'], shuffle=True)
                
                use_parl = (algo == 'B-FedPLC')
                state = train_client(
                    client_model, loader,
                    BASE_CONFIG['local_epochs'],
                    BASE_CONFIG['learning_rate'],
                    use_parl=use_parl,
                    global_model=model if use_parl else None
                )
                
                client_models.append(state)
                client_weights.append(len(indices))
            
            # Upload: clients -> server
            total_bytes_transferred += model_size_bytes * num_selected
            
            # B-FedPLC additional: blockchain metadata (~1KB per round)
            if algo == 'B-FedPLC':
                total_bytes_transferred += 1024 * num_selected
            
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]
            
            model = aggregate_models(model, client_models, client_weights)
            acc = evaluate(model, testloader)
            history.append(acc)
            best_acc = max(best_acc, acc)
            
            if (round_num + 1) % 10 == 0:
                print(f"Round {round_num+1:3d}/{BASE_CONFIG['num_rounds']} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
        
        elapsed = time.time() - start_time
        total_mb = total_bytes_transferred / (1024 * 1024)
        total_gb = total_mb / 1024
        
        results[algo] = {
            'best_accuracy': best_acc,
            'final_accuracy': history[-1],
            'history': history,
            'total_bytes': total_bytes_transferred,
            'total_mb': total_mb,
            'total_gb': total_gb,
            'time_minutes': elapsed / 60,
            'efficiency': best_acc / total_gb  # Accuracy per GB
        }
        print(f"Completed: {algo} - Best: {best_acc:.2f}%, Data: {total_gb:.2f} GB")
    
    return results

# ============================================================
# Visualization
# ============================================================
def visualize_all_results(all_results):
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    Path('plots').mkdir(exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Ablation Study
    if 'ablation' in all_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        for i, (name, data) in enumerate(all_results['ablation'].items()):
            ax.plot(range(1, len(data['history'])+1), data['history'], 
                   label=f"{name} ({data['best_accuracy']:.1f}%)", 
                   color=colors[i % len(colors)], linewidth=2)
        ax.set_xlabel('Round')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Ablation Study: Component Contributions')
        ax.legend()
        plt.tight_layout()
        plt.savefig('plots/exp1_ablation.png', dpi=300)
        plt.close()
        print("Saved: plots/exp1_ablation.png")
    
    # 2. Scalability
    if 'scalability' in all_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        clients = []
        accuracies = []
        times = []
        for key, data in all_results['scalability'].items():
            clients.append(data['num_clients'])
            accuracies.append(data['best_accuracy'])
            times.append(data['time_minutes'])
        
        axes[0].bar(range(len(clients)), accuracies, color='#3498db', alpha=0.8)
        axes[0].set_xticks(range(len(clients)))
        axes[0].set_xticklabels([f"{c} clients" for c in clients])
        axes[0].set_ylabel('Best Accuracy (%)')
        axes[0].set_title('Accuracy vs Number of Clients')
        
        axes[1].bar(range(len(clients)), times, color='#e74c3c', alpha=0.8)
        axes[1].set_xticks(range(len(clients)))
        axes[1].set_xticklabels([f"{c} clients" for c in clients])
        axes[1].set_ylabel('Training Time (min)')
        axes[1].set_title('Training Time vs Number of Clients')
        
        plt.tight_layout()
        plt.savefig('plots/exp2_scalability.png', dpi=300)
        plt.close()
        print("Saved: plots/exp2_scalability.png")
    
    # 3. Non-IID Sensitivity
    if 'noniid' in all_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
        for i, (name, data) in enumerate(all_results['noniid'].items()):
            ax.plot(range(1, len(data['history'])+1), data['history'],
                   label=f"α={data['alpha']} ({data['best_accuracy']:.1f}%)",
                   color=colors[i % len(colors)], linewidth=2)
        ax.set_xlabel('Round')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Non-IID Sensitivity: Effect of Dirichlet Alpha')
        ax.legend()
        plt.tight_layout()
        plt.savefig('plots/exp3_noniid.png', dpi=300)
        plt.close()
        print("Saved: plots/exp3_noniid.png")
    
    # 4. Security Analysis
    if 'security' in all_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        byz_fracs = []
        accuracies = []
        detected = []
        for key, data in all_results['security'].items():
            byz_fracs.append(int(data['byzantine_fraction'] * 100))
            accuracies.append(data['best_accuracy'])
            detected.append(data['attacks_detected'])
        
        axes[0].bar(range(len(byz_fracs)), accuracies, color='#3498db', alpha=0.8)
        axes[0].set_xticks(range(len(byz_fracs)))
        axes[0].set_xticklabels([f"{b}%" for b in byz_fracs])
        axes[0].set_xlabel('Byzantine Clients (%)')
        axes[0].set_ylabel('Best Accuracy (%)')
        axes[0].set_title('Accuracy vs Byzantine Attack Rate')
        
        axes[1].bar(range(len(byz_fracs)), detected, color='#e74c3c', alpha=0.8)
        axes[1].set_xticks(range(len(byz_fracs)))
        axes[1].set_xticklabels([f"{b}%" for b in byz_fracs])
        axes[1].set_xlabel('Byzantine Clients (%)')
        axes[1].set_ylabel('Attacks Detected')
        axes[1].set_title('Attack Detection Rate')
        
        plt.tight_layout()
        plt.savefig('plots/exp4_security.png', dpi=300)
        plt.close()
        print("Saved: plots/exp4_security.png")
    
    # 5. Communication Efficiency
    if 'communication' in all_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        algos = list(all_results['communication'].keys())
        accuracies = [all_results['communication'][a]['best_accuracy'] for a in algos]
        data_gb = [all_results['communication'][a]['total_gb'] for a in algos]
        efficiency = [all_results['communication'][a]['efficiency'] for a in algos]
        
        x = np.arange(len(algos))
        width = 0.35
        
        axes[0].bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#3498db')
        ax2 = axes[0].twinx()
        ax2.bar(x + width/2, data_gb, width, label='Data (GB)', color='#e74c3c', alpha=0.7)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(algos)
        axes[0].set_ylabel('Accuracy (%)')
        ax2.set_ylabel('Data Transferred (GB)')
        axes[0].set_title('Accuracy vs Communication Cost')
        axes[0].legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        axes[1].bar(algos, efficiency, color='#2ecc71', alpha=0.8)
        axes[1].set_ylabel('Accuracy per GB')
        axes[1].set_title('Communication Efficiency')
        
        plt.tight_layout()
        plt.savefig('plots/exp5_communication.png', dpi=300)
        plt.close()
        print("Saved: plots/exp5_communication.png")

# ============================================================
# Main
# ============================================================
def main():
    print("="*70)
    print("B-FedPLC COMPREHENSIVE EXPERIMENTS")
    print("="*70)
    print(f"\nDevice: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Set seed
    torch.manual_seed(BASE_CONFIG['seed'])
    np.random.seed(BASE_CONFIG['seed'])
    
    # Load data
    print("\nLoading CIFAR-10...")
    trainset, testset = load_cifar10()
    testloader = DataLoader(testset, batch_size=100, shuffle=False)
    
    # Partition data
    print("Partitioning data...")
    client_indices = dirichlet_partition(trainset, BASE_CONFIG['num_clients'], BASE_CONFIG['dirichlet_alpha'])
    
    all_results = {}
    total_start = time.time()
    
    # Run all experiments
    print("\n" + "="*70)
    print("STARTING ALL EXPERIMENTS")
    print("="*70)
    
    # Experiment 1: Ablation Study
    all_results['ablation'] = run_ablation_study(trainset, testset, testloader, client_indices)
    
    # Experiment 2: Scalability Analysis
    all_results['scalability'] = run_scalability_analysis(trainset, testset, testloader)
    
    # Experiment 3: Non-IID Sensitivity
    all_results['noniid'] = run_noniid_sensitivity(trainset, testset, testloader)
    
    # Experiment 4: Security Analysis
    all_results['security'] = run_security_analysis(trainset, testset, testloader, client_indices)
    
    # Experiment 5: Communication Efficiency
    all_results['communication'] = run_communication_efficiency(trainset, testset, testloader, client_indices)
    
    # Save results
    total_time = (time.time() - total_start) / 60
    all_results['total_time_minutes'] = total_time
    
    with open('all_experiments_results.json', 'w') as f:
        # Convert to serializable format
        serializable = {}
        for exp_name, exp_data in all_results.items():
            if isinstance(exp_data, dict):
                serializable[exp_name] = {}
                for k, v in exp_data.items():
                    if isinstance(v, dict):
                        serializable[exp_name][k] = {
                            kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist())
                            for kk, vv in v.items()
                        }
                    else:
                        serializable[exp_name][k] = v
            else:
                serializable[exp_name] = exp_data
        json.dump(serializable, f, indent=2)
    
    print(f"\nResults saved to: all_experiments_results.json")
    
    # Generate visualizations
    visualize_all_results(all_results)
    
    # Print summary
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"Total Time: {total_time:.1f} minutes")
    
    print("\n--- SUMMARY ---")
    
    print("\n1. ABLATION STUDY:")
    for name, data in all_results['ablation'].items():
        print(f"   {name}: {data['best_accuracy']:.2f}%")
    
    print("\n2. SCALABILITY:")
    for key, data in all_results['scalability'].items():
        print(f"   {data['num_clients']} clients: {data['best_accuracy']:.2f}%")
    
    print("\n3. NON-IID SENSITIVITY:")
    for key, data in all_results['noniid'].items():
        print(f"   α={data['alpha']}: {data['best_accuracy']:.2f}%")
    
    print("\n4. SECURITY (Byzantine Tolerance):")
    for key, data in all_results['security'].items():
        print(f"   {int(data['byzantine_fraction']*100)}% Byzantine: {data['best_accuracy']:.2f}% (Detected: {data['attacks_detected']})")
    
    print("\n5. COMMUNICATION EFFICIENCY:")
    for algo, data in all_results['communication'].items():
        print(f"   {algo}: {data['best_accuracy']:.2f}% | {data['total_gb']:.2f} GB | Efficiency: {data['efficiency']:.2f}")
    
    print("\n" + "="*70)
    print("Visualizations saved to: plots/exp1_ablation.png - exp5_communication.png")
    print("="*70)

if __name__ == "__main__":
    main()
