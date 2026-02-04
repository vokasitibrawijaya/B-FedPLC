"""
Phase 2: Multi-Dataset Experiments for B-FedPLC Paper
======================================================
Extends Byzantine aggregation experiments to CIFAR-10.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt

# IEEE Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================
# CIFAR-10 CNN Model (more complex than MNIST)
# ============================================================
class CIFAR10CNN(nn.Module):
    """CNN for CIFAR-10 (3x32x32 images, 10 classes)"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(self.pool1(x))
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout2(self.pool2(x))
        
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x


class MNISTModel(nn.Module):
    """Same MNIST model as before for consistency"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


# ============================================================
# Data Loading
# ============================================================
def load_cifar10(num_clients=20):
    """Load CIFAR-10 and partition for FL"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
    
    # IID partition
    indices = np.random.permutation(len(train_dataset))
    split_size = len(train_dataset) // num_clients
    client_indices = [indices[i*split_size:(i+1)*split_size] for i in range(num_clients)]
    
    client_loaders = [
        DataLoader(Subset(train_dataset, idx), batch_size=32, shuffle=True)
        for idx in client_indices
    ]
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    return client_loaders, test_loader


def load_mnist(num_clients=20):
    """Load MNIST and partition for FL"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    indices = np.random.permutation(len(train_dataset))
    split_size = len(train_dataset) // num_clients
    client_indices = [indices[i*split_size:(i+1)*split_size] for i in range(num_clients)]
    
    client_loaders = [
        DataLoader(Subset(train_dataset, idx), batch_size=32, shuffle=True)
        for idx in client_indices
    ]
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    return client_loaders, test_loader


# ============================================================
# Byzantine Attacks
# ============================================================
def sign_flip_attack(gradient, scale=1.0):
    """Sign-flipping attack"""
    return {k: -scale * v for k, v in gradient.items()}


def gaussian_noise_attack(gradient, std=1.0):
    """Gaussian noise attack"""
    return {k: torch.randn_like(v) * std for k, v in gradient.items()}


def label_flip_attack(gradient, scale=2.0):
    """Label-flipping style attack - amplified wrong direction"""
    return {k: -scale * v + torch.randn_like(v) * 0.1 for k, v in gradient.items()}


# ============================================================
# Aggregation Methods
# ============================================================
def fedavg(gradients):
    """Simple averaging"""
    avg = {}
    for key in gradients[0].keys():
        stacked = torch.stack([g[key] for g in gradients])
        if stacked.dtype in [torch.int32, torch.int64, torch.long]:
            # For integer types (like num_batches_tracked), use the first value
            avg[key] = gradients[0][key]
        else:
            avg[key] = stacked.mean(dim=0)
    return avg


def trimmed_mean(gradients, trim_ratio=0.1):
    """Trimmed mean aggregation"""
    avg = {}
    n = len(gradients)
    trim_count = int(n * trim_ratio)
    
    for key in gradients[0].keys():
        stacked = torch.stack([g[key] for g in gradients])
        
        # Handle integer types
        if stacked.dtype in [torch.int32, torch.int64, torch.long]:
            avg[key] = gradients[0][key]
            continue
            
        sorted_vals, _ = torch.sort(stacked, dim=0)
        if trim_count > 0:
            trimmed = sorted_vals[trim_count:-trim_count]
        else:
            trimmed = sorted_vals
        avg[key] = trimmed.mean(dim=0)
    return avg


def krum(gradients, f=None):
    """Krum: select most representative gradient"""
    n = len(gradients)
    if f is None:
        f = (n - 3) // 2
    
    # Flatten gradients
    flat = []
    for g in gradients:
        flat.append(torch.cat([v.flatten() for v in g.values()]))
    flat = torch.stack(flat)
    
    # Compute pairwise distances
    scores = []
    for i in range(n):
        dists = torch.norm(flat - flat[i], dim=1)
        dists_sorted, _ = torch.sort(dists)
        score = dists_sorted[1:n-f-1].sum()  # Exclude self and f largest
        scores.append(score)
    
    selected = torch.argmin(torch.tensor(scores))
    return gradients[selected]


def multi_krum(gradients, k=None, f=None):
    """Multi-Krum: average k most representative gradients"""
    n = len(gradients)
    if f is None:
        f = (n - 3) // 2
    if k is None:
        k = max(1, n - f - 2)
    
    flat = []
    for g in gradients:
        flat.append(torch.cat([v.flatten() for v in g.values()]))
    flat = torch.stack(flat)
    
    scores = []
    for i in range(n):
        dists = torch.norm(flat - flat[i], dim=1)
        dists_sorted, _ = torch.sort(dists)
        score = dists_sorted[1:n-f-1].sum()
        scores.append(score)
    
    _, indices = torch.topk(torch.tensor(scores), k, largest=False)
    selected_grads = [gradients[i] for i in indices]
    
    return fedavg(selected_grads)


# ============================================================
# Training Functions
# ============================================================
def train_client(model, loader, epochs=1, lr=0.01):
    """Train on client data and return gradient"""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    initial_params = {k: v.clone() for k, v in model.state_dict().items()}
    
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()
    
    # Compute pseudo-gradient
    gradient = {}
    for k, v in model.state_dict().items():
        gradient[k] = initial_params[k] - v
    
    # Restore initial params
    model.load_state_dict(initial_params)
    
    return gradient


def evaluate(model, test_loader):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


def run_fl_experiment(dataset_name, num_clients=20, num_rounds=30, byzantine_frac=0.3,
                      aggregation='multi_krum', attack='sign_flip', seed=42):
    """Run single FL experiment"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data and model
    if dataset_name == 'mnist':
        client_loaders, test_loader = load_mnist(num_clients)
        model = MNISTModel().to(DEVICE)
    else:  # cifar10
        client_loaders, test_loader = load_cifar10(num_clients)
        model = CIFAR10CNN().to(DEVICE)
    
    # Determine Byzantine clients
    num_byz = int(num_clients * byzantine_frac)
    byz_clients = set(np.random.choice(num_clients, num_byz, replace=False))
    
    # Select aggregation function
    agg_funcs = {
        'fedavg': fedavg,
        'trimmed_mean': trimmed_mean,
        'krum': krum,
        'multi_krum': multi_krum
    }
    aggregate = agg_funcs[aggregation]
    
    # Select attack function
    attack_funcs = {
        'sign_flip': sign_flip_attack,
        'gaussian': gaussian_noise_attack,
        'label_flip': label_flip_attack
    }
    attack_fn = attack_funcs[attack]
    
    accuracies = []
    
    for round_idx in range(num_rounds):
        gradients = []
        
        for client_id in range(num_clients):
            gradient = train_client(model, client_loaders[client_id])
            
            if client_id in byz_clients:
                gradient = attack_fn(gradient)
            
            gradients.append(gradient)
        
        # Aggregate
        aggregated = aggregate(gradients)
        
        # Update global model
        new_state = {}
        for k, v in model.state_dict().items():
            new_state[k] = v - aggregated[k]
        model.load_state_dict(new_state)
        
        # Evaluate
        acc = evaluate(model, test_loader)
        accuracies.append(acc)
        
        if (round_idx + 1) % 5 == 0:
            print(f"  Round {round_idx+1}/{num_rounds}: {acc:.2f}%")
    
    return accuracies


# ============================================================
# Main Experiment Suite
# ============================================================
def run_phase2_experiments():
    """Run complete Phase 2: Multi-Dataset experiments"""
    
    print("=" * 70)
    print("PHASE 2: MULTI-DATASET BYZANTINE RESILIENCE EXPERIMENTS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print()
    
    os.makedirs('plots', exist_ok=True)
    
    results = {
        'mnist': {},
        'cifar10': {}
    }
    
    datasets_config = ['mnist', 'cifar10']
    methods = ['fedavg', 'trimmed_mean', 'krum', 'multi_krum']
    byz_fractions = [0.0, 0.2, 0.3]
    seeds = [42, 123, 456]  # 3 seeds for efficiency
    
    # Experiment parameters
    num_rounds_mnist = 30
    num_rounds_cifar = 50  # CIFAR needs more rounds
    
    for dataset in datasets_config:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*70}")
        
        num_rounds = num_rounds_mnist if dataset == 'mnist' else num_rounds_cifar
        results[dataset] = {'curves': {}, 'final': {}}
        
        for method in methods:
            results[dataset]['curves'][method] = {}
            results[dataset]['final'][method] = {}
            
            for byz in byz_fractions:
                print(f"\n[{dataset.upper()}] {method} @ {int(byz*100)}% Byzantine")
                
                all_curves = []
                final_accs = []
                
                for seed in seeds:
                    print(f"  Seed {seed}...")
                    accs = run_fl_experiment(
                        dataset_name=dataset,
                        num_rounds=num_rounds,
                        byzantine_frac=byz,
                        aggregation=method,
                        attack='sign_flip',
                        seed=seed
                    )
                    all_curves.append(accs)
                    final_accs.append(accs[-1])
                    print(f"    Final: {accs[-1]:.2f}%")
                
                mean_acc = np.mean(final_accs)
                std_acc = np.std(final_accs)
                print(f"  => Mean: {mean_acc:.2f}% ± {std_acc:.2f}%")
                
                results[dataset]['curves'][method][str(byz)] = all_curves
                results[dataset]['final'][method][str(byz)] = final_accs
    
    # Save results
    results_path = 'phase2_results.json'
    
    # Convert to JSON-serializable format
    json_results = {}
    for dataset in results:
        json_results[dataset] = {
            'curves': {},
            'final': {}
        }
        for method in results[dataset]['curves']:
            json_results[dataset]['curves'][method] = {}
            json_results[dataset]['final'][method] = {}
            for byz in results[dataset]['curves'][method]:
                json_results[dataset]['curves'][method][byz] = [
                    [float(x) for x in curve] 
                    for curve in results[dataset]['curves'][method][byz]
                ]
                json_results[dataset]['final'][method][byz] = [
                    float(x) for x in results[dataset]['final'][method][byz]
                ]
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n✓ Results saved: {results_path}")
    
    # Generate plots
    generate_phase2_plots(json_results)
    
    print(f"\n{'='*70}")
    print("PHASE 2 COMPLETE!")
    print(f"{'='*70}")
    
    return json_results


def generate_phase2_plots(results):
    """Generate comparison plots for Phase 2"""
    
    # Plot 1: Dataset comparison (MNIST vs CIFAR-10)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = ['fedavg', 'trimmed_mean', 'krum', 'multi_krum']
    method_names = ['FedAvg', 'Trimmed Mean', 'Krum', 'Multi-Krum']
    colors = ['#D62728', '#2CA02C', '#1F77B4', '#9467BD']
    
    for idx, (dataset, ax) in enumerate(zip(['mnist', 'cifar10'], axes)):
        byz_levels = ['0.0', '0.2', '0.3']
        byz_labels = ['0%', '20%', '30%']
        x = np.arange(len(byz_levels))
        width = 0.18
        offsets = [-1.5, -0.5, 0.5, 1.5]
        
        for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
            means = []
            stds = []
            for byz in byz_levels:
                vals = results[dataset]['final'][method][byz]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            
            ax.bar(x + offsets[i]*width, means, width, yerr=stds,
                   label=name, color=color, edgecolor='black', linewidth=0.5, capsize=2)
        
        ax.set_xlabel('Byzantine Fraction', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title(f'{dataset.upper()} - Byzantine Resilience', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(byz_labels)
        ax.set_ylim([0, 110])
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/phase2_dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: plots/phase2_dataset_comparison.png")
    
    # Plot 2: Learning curves comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for row, dataset in enumerate(['mnist', 'cifar10']):
        for col, byz in enumerate(['0.0', '0.3']):
            ax = axes[row, col]
            
            for method, name, color in zip(methods, method_names, colors):
                curves = results[dataset]['curves'][method][byz]
                mean_curve = np.mean(curves, axis=0)
                std_curve = np.std(curves, axis=0)
                rounds = np.arange(1, len(mean_curve) + 1)
                
                ax.plot(rounds, mean_curve, color=color, linewidth=2, label=name)
                ax.fill_between(rounds, mean_curve - std_curve, mean_curve + std_curve,
                               color=color, alpha=0.2)
            
            byz_pct = int(float(byz) * 100)
            ax.set_xlabel('Round', fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontweight='bold')
            ax.set_title(f'{dataset.upper()} - {byz_pct}% Byzantine', fontweight='bold')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig('plots/phase2_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: plots/phase2_learning_curves.png")
    
    # Plot 3: Summary table visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Create table data
    table_data = [['Dataset', 'Method', '0% Byz', '20% Byz', '30% Byz']]
    
    for dataset in ['mnist', 'cifar10']:
        for method, name in zip(methods, method_names):
            row = [dataset.upper(), name]
            for byz in ['0.0', '0.2', '0.3']:
                vals = results[dataset]['final'][method][byz]
                mean = np.mean(vals)
                std = np.std(vals)
                row.append(f'{mean:.1f}±{std:.1f}')
            table_data.append(row)
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        color = '#E8F0FE' if i % 2 == 1 else 'white'
        for j in range(5):
            table[(i, j)].set_facecolor(color)
    
    ax.set_title('Phase 2: Multi-Dataset Byzantine Resilience Results\n(Mean ± Std, 3 seeds)',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig('plots/phase2_summary_table.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Saved: plots/phase2_summary_table.png")


if __name__ == "__main__":
    run_phase2_experiments()
