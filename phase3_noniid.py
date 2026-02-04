"""
Phase 3: Non-IID Data Distribution Experiments for IEEE Access Paper
=====================================================================

This script extends Phase 2 by testing Byzantine resilience under 
heterogeneous (Non-IID) data distributions using Dirichlet partitioning.

Key Features:
- Dirichlet distribution with α = {0.1, 0.5, 1.0}
  - α=0.1: Highly Non-IID (extreme heterogeneity)
  - α=0.5: Moderately Non-IID (realistic FL scenario)
  - α=1.0: Slightly Non-IID (approaching IID)
- MNIST + CIFAR-10 datasets
- 4 Byzantine aggregation methods
- 3 Byzantine levels (0%, 20%, 30%)
- 3 random seeds per configuration

Total Experiments: 2 datasets × 3 α values × 4 methods × 3 Byz levels × 3 seeds = 216 experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple
import time

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==================== Data Loading with Dirichlet Distribution ====================

def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, seed: int = 42) -> List[np.ndarray]:
    """
    Partition data using Dirichlet distribution for Non-IID setting.
    
    Args:
        labels: Array of data labels
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
               - Small α (e.g., 0.1): Highly Non-IID (each client has few classes)
               - Large α (e.g., 10): Nearly IID (each client has balanced classes)
        seed: Random seed
        
    Returns:
        List of indices for each client
    """
    np.random.seed(seed)
    num_classes = len(np.unique(labels))
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    
    for c_idx, c_indices in enumerate(class_indices):
        np.random.shuffle(c_indices)
        splits = (label_distribution[c_idx] * len(c_indices)).astype(int)
        splits[0] += len(c_indices) - splits.sum()  # Adjust for rounding
        
        idx = 0
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(c_indices[idx:idx+split])
            idx += split
    
    return [np.array(indices) for indices in client_indices]

def load_mnist_noniid(alpha: float, num_clients: int = 10, seed: int = 42):
    """Load MNIST with Non-IID Dirichlet partitioning."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Get labels
    train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    
    # Dirichlet partitioning
    client_indices = dirichlet_partition(train_labels, num_clients, alpha, seed)
    
    train_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        train_loaders.append(DataLoader(subset, batch_size=32, shuffle=True))
    
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loaders, test_loader

def load_cifar10_noniid(alpha: float, num_clients: int = 10, seed: int = 42):
    """Load CIFAR-10 with Non-IID Dirichlet partitioning."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    # Get labels
    train_labels = np.array(train_dataset.targets)
    
    # Dirichlet partitioning
    client_indices = dirichlet_partition(train_labels, num_clients, alpha, seed)
    
    train_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        train_loaders.append(DataLoader(subset, batch_size=32, shuffle=True))
    
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loaders, test_loader

# ==================== Model Definitions ====================

class MNISTModel(nn.Module):
    """Simple CNN for MNIST."""
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CIFAR10CNN(nn.Module):
    """Deeper CNN for CIFAR-10."""
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ==================== Byzantine Aggregation Methods ====================

def fedavg(gradients: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """FedAvg: Simple averaging."""
    avg = {}
    for key in gradients[0].keys():
        stacked = torch.stack([g[key].float() for g in gradients])
        # Handle integer tensors (e.g., num_batches_tracked in BatchNorm)
        if stacked.dtype in [torch.int32, torch.int64, torch.long]:
            avg[key] = gradients[0][key]
        else:
            avg[key] = stacked.mean(dim=0)
    return avg

def trimmed_mean(gradients: List[Dict[str, torch.Tensor]], beta: float = 0.1) -> Dict[str, torch.Tensor]:
    """Trimmed Mean: Remove top/bottom β fraction."""
    avg = {}
    for key in gradients[0].keys():
        stacked = torch.stack([g[key].float() for g in gradients])
        if stacked.dtype in [torch.int32, torch.int64, torch.long]:
            avg[key] = gradients[0][key]
        else:
            k = int(len(gradients) * beta)
            if k > 0:
                sorted_vals, _ = torch.sort(stacked, dim=0)
                trimmed = sorted_vals[k:-k] if k < len(gradients)//2 else sorted_vals
                avg[key] = trimmed.mean(dim=0)
            else:
                avg[key] = stacked.mean(dim=0)
    return avg

def krum(gradients: List[Dict[str, torch.Tensor]], f: int = None) -> Dict[str, torch.Tensor]:
    """Krum: Select gradient closest to others."""
    if f is None:
        f = (len(gradients) - 3) // 2
    
    n = len(gradients)
    m = n - f - 2
    
    # Flatten gradients
    flat_grads = []
    for g in gradients:
        flat = torch.cat([v.flatten().float() for v in g.values()])
        flat_grads.append(flat)
    flat_grads = torch.stack(flat_grads)
    
    # Compute pairwise distances
    distances = torch.cdist(flat_grads, flat_grads)
    
    # For each gradient, sum distances to m closest neighbors
    scores = []
    for i in range(n):
        sorted_dists, _ = torch.sort(distances[i])
        scores.append(sorted_dists[1:m+1].sum())  # Skip self (distance 0)
    
    # Select gradient with minimum score
    selected = torch.argmin(torch.tensor(scores))
    return gradients[selected]

def multi_krum(gradients: List[Dict[str, torch.Tensor]], f: int = None, k: int = None) -> Dict[str, torch.Tensor]:
    """Multi-Krum: Average k selected gradients."""
    if f is None:
        f = (len(gradients) - 3) // 2
    if k is None:
        k = len(gradients) - f - 2
    
    n = len(gradients)
    m = n - f - 2
    
    # Flatten gradients
    flat_grads = []
    for g in gradients:
        flat = torch.cat([v.flatten().float() for v in g.values()])
        flat_grads.append(flat)
    flat_grads = torch.stack(flat_grads)
    
    # Compute pairwise distances
    distances = torch.cdist(flat_grads, flat_grads)
    
    # For each gradient, sum distances to m closest neighbors
    scores = []
    for i in range(n):
        sorted_dists, _ = torch.sort(distances[i])
        scores.append(sorted_dists[1:m+1].sum())
    
    # Select k gradients with minimum scores
    _, selected_indices = torch.topk(torch.tensor(scores), k, largest=False)
    selected_grads = [gradients[i] for i in selected_indices]
    
    # Average selected gradients
    return fedavg(selected_grads)

# ==================== Attack Implementations ====================

def sign_flip_attack(gradients: List[Dict[str, torch.Tensor]], 
                     byzantine_clients: List[int]) -> List[Dict[str, torch.Tensor]]:
    """Sign-flipping attack: Byzantine clients flip gradient signs."""
    attacked = []
    for i, grad in enumerate(gradients):
        if i in byzantine_clients:
            attacked.append({k: -v for k, v in grad.items()})
        else:
            attacked.append(grad)
    return attacked

# ==================== Federated Learning ====================

def train_local(model, train_loader, epochs=1, lr=0.01):
    """Train model on local data."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    
    # Return full state dict (includes BatchNorm running stats)
    return {name: param.data.clone() for name, param in model.state_dict().items()}

def evaluate(model, test_loader):
    """Evaluate model on test data."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
    
    return 100.0 * correct / total

def run_fl_experiment(dataset_name: str, alpha: float, aggregation_method: str, 
                     byzantine_fraction: float, seed: int, rounds: int) -> Dict:
    """Run a single FL experiment."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data
    if dataset_name == "mnist":
        train_loaders, test_loader = load_mnist_noniid(alpha, num_clients=10, seed=seed)
        model = MNISTModel().to(device)
    else:  # cifar10
        train_loaders, test_loader = load_cifar10_noniid(alpha, num_clients=10, seed=seed)
        model = CIFAR10CNN().to(device)
    
    num_clients = len(train_loaders)
    num_byzantine = int(num_clients * byzantine_fraction)
    byzantine_clients = list(range(num_byzantine))
    
    # Aggregation function
    agg_funcs = {
        'fedavg': fedavg,
        'trimmed_mean': trimmed_mean,
        'krum': krum,
        'multi_krum': multi_krum
    }
    aggregate = agg_funcs[aggregation_method]
    
    # Initial evaluation
    accuracies = [evaluate(model, test_loader)]
    
    print(f"  [{dataset_name.upper()} α={alpha} {aggregation_method} Byz={byzantine_fraction:.0%} seed={seed}] "
          f"Round 0: {accuracies[0]:.2f}%")
    
    # FL training
    for round_num in range(1, rounds + 1):
        # Local training
        local_weights = []
        for client_id, train_loader in enumerate(train_loaders):
            client_model = type(model)().to(device)
            client_model.load_state_dict(model.state_dict())
            weights = train_local(client_model, train_loader)
            local_weights.append(weights)
        
        # Compute gradients (only for trainable parameters, not running stats)
        gradients = []
        global_params = {k: v for k, v in model.state_dict().items() if 'running' not in k and 'num_batches_tracked' not in k}
        for weights in local_weights:
            local_params = {k: v for k, v in weights.items() if 'running' not in k and 'num_batches_tracked' not in k}
            grad = {k: local_params[k] - global_params[k] for k in local_params.keys()}
            gradients.append(grad)
        
        # Apply Byzantine attack
        if num_byzantine > 0:
            gradients = sign_flip_attack(gradients, byzantine_clients)
        
        # Aggregate
        if aggregation_method in ['krum', 'multi_krum']:
            agg_grad = aggregate(gradients, f=(num_clients-3)//2)
        else:
            agg_grad = aggregate(gradients)
        
        # Update global model (merge aggregated gradients with running stats)
        current_state = model.state_dict()
        new_state = {}
        for k in current_state.keys():
            if k in agg_grad:
                new_state[k] = current_state[k] + agg_grad[k]
            else:
                # Keep running stats from first client (or use current)
                new_state[k] = local_weights[0][k] if k in local_weights[0] else current_state[k]
        model.load_state_dict(new_state)
        
        # Evaluate
        acc = evaluate(model, test_loader)
        accuracies.append(acc)
        
        if round_num % 10 == 0 or round_num == rounds:
            print(f"  [{dataset_name.upper()} α={alpha} {aggregation_method} Byz={byzantine_fraction:.0%} seed={seed}] "
                  f"Round {round_num}: {acc:.2f}%")
    
    return {
        'accuracies': accuracies,
        'final_accuracy': accuracies[-1]
    }

# ==================== Main Experiment Runner ====================

def run_phase3_experiments():
    """Run all Phase 3 experiments."""
    results = {
        'mnist': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        'cifar10': defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    }
    
    datasets = ['mnist', 'cifar10']
    alphas = [0.1, 0.5, 1.0]  # Non-IID levels
    methods = ['fedavg', 'trimmed_mean', 'krum', 'multi_krum']
    byzantine_fractions = [0.0, 0.2, 0.3]
    seeds = [42, 123, 456]
    
    total_experiments = len(datasets) * len(alphas) * len(methods) * len(byzantine_fractions) * len(seeds)
    current = 0
    
    print(f"\n{'='*80}")
    print(f"PHASE 3: NON-IID DATA DISTRIBUTION EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Total experiments: {total_experiments}")
    print(f"Configuration: 2 datasets × 3 α × 4 methods × 3 Byz levels × 3 seeds\n")
    
    start_time = time.time()
    
    for dataset in datasets:
        rounds = 30 if dataset == 'mnist' else 50
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*80}\n")
        
        for alpha in alphas:
            print(f"\n--- Alpha = {alpha} (Non-IID level) ---")
            
            for method in methods:
                for byz_frac in byzantine_fractions:
                    for seed in seeds:
                        current += 1
                        print(f"\n[{current}/{total_experiments}] Running experiment...")
                        
                        result = run_fl_experiment(
                            dataset_name=dataset,
                            alpha=alpha,
                            aggregation_method=method,
                            byzantine_fraction=byz_frac,
                            seed=seed,
                            rounds=rounds
                        )
                        
                        results[dataset][alpha][method][byz_frac].append(result['accuracies'])
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED!")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"{'='*80}\n")
    
    # Save results
    output_file = Path('phase3_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")
    
    return results

# ==================== Visualization ====================

def generate_phase3_plots(results: Dict):
    """Generate comprehensive Phase 3 visualizations."""
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    print("\nGenerating Phase 3 visualizations...")
    
    # Plot 1: Non-IID Impact Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    datasets = ['mnist', 'cifar10']
    alphas = [0.1, 0.5, 1.0]
    methods = ['fedavg', 'trimmed_mean', 'krum', 'multi_krum']
    
    for ds_idx, dataset in enumerate(datasets):
        for byz_idx, byz_frac in enumerate([0.0, 0.3]):
            ax = axes[ds_idx, byz_idx]
            
            x = np.arange(len(alphas))
            width = 0.2
            
            for m_idx, method in enumerate(methods):
                means = []
                stds = []
                for alpha in alphas:
                    accs = [res[-1] for res in results[dataset][alpha][method][byz_frac]]
                    means.append(np.mean(accs))
                    stds.append(np.std(accs))
                
                ax.bar(x + m_idx * width, means, width, label=method.replace('_', ' ').title(),
                      yerr=stds, capsize=5)
            
            ax.set_xlabel('Dirichlet α (Non-IID Level)', fontsize=11)
            ax.set_ylabel('Test Accuracy (%)', fontsize=11)
            ax.set_title(f'{dataset.upper()}: Byzantine={int(byz_frac*100)}%', fontsize=12, fontweight='bold')
            ax.set_xticks(x + 1.5 * width)
            ax.set_xticklabels([f'α={a}\n({"High" if a==0.1 else "Med" if a==0.5 else "Low"} Non-IID)' 
                                for a in alphas])
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'phase3_noniid_impact.png', dpi=300, bbox_inches='tight')
    print("  ✓ Generated: phase3_noniid_impact.png")
    plt.close()
    
    # Plot 2: Learning Curves for Different α
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for ds_idx, dataset in enumerate(datasets):
        for alpha_idx, alpha in enumerate(alphas):
            ax = axes[ds_idx, alpha_idx]
            
            for method in methods:
                # Get curves for 30% Byzantine
                curves = results[dataset][alpha][method][0.3]
                mean_curve = np.mean(curves, axis=0)
                std_curve = np.std(curves, axis=0)
                
                rounds = len(mean_curve)
                x = range(rounds)
                
                ax.plot(x, mean_curve, label=method.replace('_', ' ').title(), linewidth=2)
                ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
            
            ax.set_xlabel('Round', fontsize=10)
            ax.set_ylabel('Test Accuracy (%)', fontsize=10)
            ax.set_title(f'{dataset.upper()}: α={alpha} (Byz=30%)', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'phase3_learning_curves.png', dpi=300, bbox_inches='tight')
    print("  ✓ Generated: phase3_learning_curves.png")
    plt.close()
    
    # Plot 3: Summary Table
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['Dataset', 'Alpha', 'Method', 'Byz 0%', 'Byz 20%', 'Byz 30%', 'Degradation'])
    
    for dataset in datasets:
        for alpha in alphas:
            for method in methods:
                row = [dataset.upper(), f'{alpha}', method.replace('_', ' ').title()]
                
                accs = []
                for byz_frac in [0.0, 0.2, 0.3]:
                    results_list = results[dataset][alpha][method][byz_frac]
                    final_accs = [res[-1] for res in results_list]
                    mean_acc = np.mean(final_accs)
                    std_acc = np.std(final_accs)
                    accs.append(mean_acc)
                    row.append(f'{mean_acc:.2f}±{std_acc:.2f}')
                
                degradation = accs[0] - accs[2]
                row.append(f'{degradation:.2f}%')
                table_data.append(row)
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.1, 0.08, 0.15, 0.15, 0.15, 0.15, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    plt.title('Phase 3: Non-IID Experiments Summary (mean±std over 3 seeds)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig(plots_dir / 'phase3_summary_table.png', dpi=300, bbox_inches='tight')
    print("  ✓ Generated: phase3_summary_table.png")
    plt.close()
    
    # Plot 4: Data Distribution Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for alpha_idx, alpha in enumerate([0.1, 0.5, 1.0]):
        ax = axes[alpha_idx]
        
        # Simulate Dirichlet distribution visualization
        np.random.seed(42)
        num_clients = 10
        num_classes = 10
        label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
        
        im = ax.imshow(label_distribution, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('Client', fontsize=11)
        ax.set_ylabel('Class', fontsize=11)
        ax.set_title(f'α={alpha} ({"High" if alpha==0.1 else "Med" if alpha==0.5 else "Low"} Non-IID)', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(range(num_clients))
        ax.set_xticklabels(range(num_clients))
        ax.set_yticks(range(num_classes))
        ax.set_yticklabels(range(num_classes))
        
        plt.colorbar(im, ax=ax, label='Proportion')
    
    plt.suptitle('Data Distribution Heterogeneity (Dirichlet Partitioning)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / 'phase3_data_distribution.png', dpi=300, bbox_inches='tight')
    print("  ✓ Generated: phase3_data_distribution.png")
    plt.close()
    
    print("\n✅ All Phase 3 visualizations generated!")

# ==================== Main ====================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PHASE 3: NON-IID DATA DISTRIBUTION EXPERIMENTS")
    print("="*80)
    print("\nThis experiment tests Byzantine resilience under heterogeneous data distributions.")
    print("Using Dirichlet distribution with α = {0.1, 0.5, 1.0}")
    print("  - α=0.1: Highly Non-IID (extreme heterogeneity)")
    print("  - α=0.5: Moderately Non-IID (realistic FL)")
    print("  - α=1.0: Slightly Non-IID (approaching IID)")
    print("\nTotal: 2 datasets × 3 α × 4 methods × 3 Byz × 3 seeds = 216 experiments")
    print("="*80 + "\n")
    
    # Run experiments
    results = run_phase3_experiments()
    
    # Generate plots
    generate_phase3_plots(results)
    
    print("\n" + "="*80)
    print("PHASE 3 COMPLETE!")
    print("="*80)
    print("\nKey Findings:")
    print("  1. Check phase3_noniid_impact.png for Non-IID impact analysis")
    print("  2. Check phase3_learning_curves.png for convergence under Non-IID")
    print("  3. Check phase3_summary_table.png for detailed results")
    print("  4. Check phase3_data_distribution.png for heterogeneity visualization")
    print("\nNext: Phase 4 - Blockchain Integration")
    print("="*80 + "\n")
