"""
B-FedPLC Multiple Seeds Experiment
==================================
Run each experiment 3 times with different random seeds
to calculate mean ± std for statistical validation.

Author: Dissertation Research
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import json
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# Configuration
# ============================================================
SEEDS = [42, 123, 456]  # 3 different seeds
NUM_ROUNDS = 50
NUM_CLIENTS = 30
CLIENT_FRACTION = 0.2
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.01
DIRICHLET_ALPHA = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Model Definition
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ============================================================
# Utility Functions
# ============================================================
def set_seed(seed):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data():
    """Load CIFAR-10 dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False)
    
    return trainset, testloader

def dirichlet_partition(trainset, num_clients, alpha, seed):
    """Partition data using Dirichlet distribution"""
    set_seed(seed)
    labels = np.array(trainset.targets)
    num_classes = 10
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        splits = np.split(class_indices, proportions)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())
    
    return client_indices

def evaluate(model, testloader):
    """Evaluate model accuracy"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_client(model, train_loader, epochs=3, lr=0.01):
    """Train on single client"""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    return model.state_dict()

def fedavg_aggregate(global_model, client_states, client_sizes):
    """FedAvg aggregation"""
    total_size = sum(client_sizes)
    global_state = global_model.state_dict()
    
    for key in global_state.keys():
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
        for state, size in zip(client_states, client_sizes):
            global_state[key] += (size / total_size) * state[key].float()
    
    global_model.load_state_dict(global_state)
    return global_model

# ============================================================
# Single Experiment Run
# ============================================================
def run_single_experiment(seed, trainset, testloader, client_indices, config):
    """Run a single FL experiment with given seed"""
    set_seed(seed)
    
    # Initialize model
    global_model = SimpleCNN().to(device)
    
    # Training loop
    history = []
    best_acc = 0
    
    for round_num in range(config['rounds']):
        # Select clients
        num_selected = max(1, int(config['num_clients'] * config['client_fraction']))
        selected = random.sample(range(config['num_clients']), num_selected)
        
        # Train selected clients
        client_states = []
        client_sizes = []
        
        for client_id in selected:
            # Create client data loader
            indices = client_indices[client_id]
            if len(indices) == 0:
                continue
            subset = Subset(trainset, indices)
            loader = DataLoader(subset, batch_size=config['batch_size'], shuffle=True)
            
            # Train client
            client_model = SimpleCNN().to(device)
            client_model.load_state_dict(global_model.state_dict())
            state = train_client(client_model, loader, config['local_epochs'], config['lr'])
            
            client_states.append(state)
            client_sizes.append(len(indices))
        
        # Aggregate
        if client_states:
            global_model = fedavg_aggregate(global_model, client_states, client_sizes)
        
        # Evaluate
        acc = evaluate(global_model, testloader)
        history.append(acc)
        best_acc = max(best_acc, acc)
    
    return {
        'history': history,
        'best_accuracy': best_acc,
        'final_accuracy': history[-1]
    }

def run_bfedplc_experiment(seed, trainset, testloader, client_indices, config, byzantine_fraction=0.0):
    """Run B-FedPLC experiment with Byzantine detection"""
    set_seed(seed)
    
    global_model = SimpleCNN().to(device)
    history = []
    best_acc = 0
    attacks_detected = 0
    
    for round_num in range(config['rounds']):
        num_selected = max(1, int(config['num_clients'] * config['client_fraction']))
        selected = random.sample(range(config['num_clients']), num_selected)
        
        # Determine Byzantine clients
        num_byzantine = int(len(selected) * byzantine_fraction)
        byzantine_clients = set(random.sample(selected, num_byzantine)) if num_byzantine > 0 else set()
        
        client_states = []
        client_sizes = []
        client_norms = []
        
        for client_id in selected:
            indices = client_indices[client_id]
            if len(indices) == 0:
                continue
            subset = Subset(trainset, indices)
            loader = DataLoader(subset, batch_size=config['batch_size'], shuffle=True)
            
            client_model = SimpleCNN().to(device)
            client_model.load_state_dict(global_model.state_dict())
            state = train_client(client_model, loader, config['local_epochs'], config['lr'])
            
            # Byzantine attack: random noise
            if client_id in byzantine_clients:
                for key in state:
                    state[key] = torch.randn_like(state[key]) * 10
            
            # Calculate update norm for detection
            norm = sum((state[k] - global_model.state_dict()[k]).float().norm().item() 
                      for k in state.keys())
            
            client_states.append(state)
            client_sizes.append(len(indices))
            client_norms.append(norm)
        
        # Byzantine detection using median-based filtering
        if client_states and byzantine_fraction > 0:
            median_norm = np.median(client_norms)
            std_norm = np.std(client_norms) + 1e-6
            
            filtered_states = []
            filtered_sizes = []
            
            for i, (state, size, norm) in enumerate(zip(client_states, client_sizes, client_norms)):
                if abs(norm - median_norm) <= 3 * std_norm:
                    filtered_states.append(state)
                    filtered_sizes.append(size)
                else:
                    attacks_detected += 1
            
            client_states = filtered_states if filtered_states else client_states[:1]
            client_sizes = filtered_sizes if filtered_sizes else client_sizes[:1]
        
        if client_states:
            global_model = fedavg_aggregate(global_model, client_states, client_sizes)
        
        acc = evaluate(global_model, testloader)
        history.append(acc)
        best_acc = max(best_acc, acc)
    
    return {
        'history': history,
        'best_accuracy': best_acc,
        'final_accuracy': history[-1],
        'attacks_detected': attacks_detected
    }

# ============================================================
# Main Experiment Functions
# ============================================================
def experiment_baseline(trainset, testloader, seeds):
    """Baseline comparison: FedAvg vs B-FedPLC"""
    print("\n" + "="*70)
    print("EXPERIMENT: BASELINE COMPARISON (FedAvg vs B-FedPLC)")
    print("="*70)
    
    results = {'FedAvg': [], 'B-FedPLC': []}
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        client_indices = dirichlet_partition(trainset, NUM_CLIENTS, DIRICHLET_ALPHA, seed)
        
        config = {
            'rounds': NUM_ROUNDS,
            'num_clients': NUM_CLIENTS,
            'client_fraction': CLIENT_FRACTION,
            'local_epochs': LOCAL_EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LEARNING_RATE
        }
        
        # FedAvg
        print("Running FedAvg...", end=" ")
        result = run_single_experiment(seed, trainset, testloader, client_indices, config)
        results['FedAvg'].append(result)
        print(f"Best: {result['best_accuracy']:.2f}%")
        
        # B-FedPLC
        print("Running B-FedPLC...", end=" ")
        result = run_bfedplc_experiment(seed, trainset, testloader, client_indices, config)
        results['B-FedPLC'].append(result)
        print(f"Best: {result['best_accuracy']:.2f}%")
    
    return results

def experiment_scalability(trainset, testloader, seeds):
    """Scalability with different client counts"""
    print("\n" + "="*70)
    print("EXPERIMENT: SCALABILITY ANALYSIS")
    print("="*70)
    
    client_counts = [10, 30, 50, 100]
    results = {f"{n}_clients": [] for n in client_counts}
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        
        for num_clients in client_counts:
            client_indices = dirichlet_partition(trainset, num_clients, DIRICHLET_ALPHA, seed)
            
            config = {
                'rounds': NUM_ROUNDS,
                'num_clients': num_clients,
                'client_fraction': CLIENT_FRACTION,
                'local_epochs': LOCAL_EPOCHS,
                'batch_size': BATCH_SIZE,
                'lr': LEARNING_RATE
            }
            
            print(f"Running {num_clients} clients...", end=" ")
            result = run_bfedplc_experiment(seed, trainset, testloader, client_indices, config)
            results[f"{num_clients}_clients"].append(result)
            print(f"Best: {result['best_accuracy']:.2f}%")
    
    return results

def experiment_noniid(trainset, testloader, seeds):
    """Non-IID sensitivity with different alpha values"""
    print("\n" + "="*70)
    print("EXPERIMENT: NON-IID SENSITIVITY")
    print("="*70)
    
    alphas = [0.1, 0.3, 0.5, 1.0]
    results = {f"alpha_{a}": [] for a in alphas}
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        
        for alpha in alphas:
            client_indices = dirichlet_partition(trainset, NUM_CLIENTS, alpha, seed)
            
            config = {
                'rounds': NUM_ROUNDS,
                'num_clients': NUM_CLIENTS,
                'client_fraction': CLIENT_FRACTION,
                'local_epochs': LOCAL_EPOCHS,
                'batch_size': BATCH_SIZE,
                'lr': LEARNING_RATE
            }
            
            print(f"Running alpha={alpha}...", end=" ")
            result = run_bfedplc_experiment(seed, trainset, testloader, client_indices, config)
            results[f"alpha_{alpha}"].append(result)
            print(f"Best: {result['best_accuracy']:.2f}%")
    
    return results

def experiment_security(trainset, testloader, seeds):
    """Byzantine fault tolerance"""
    print("\n" + "="*70)
    print("EXPERIMENT: SECURITY ANALYSIS (Byzantine Tolerance)")
    print("="*70)
    
    byzantine_fractions = [0.0, 0.1, 0.2, 0.3]
    results = {f"byzantine_{int(b*100)}pct": [] for b in byzantine_fractions}
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        client_indices = dirichlet_partition(trainset, NUM_CLIENTS, DIRICHLET_ALPHA, seed)
        
        config = {
            'rounds': NUM_ROUNDS,
            'num_clients': NUM_CLIENTS,
            'client_fraction': CLIENT_FRACTION,
            'local_epochs': LOCAL_EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LEARNING_RATE
        }
        
        for byz_frac in byzantine_fractions:
            print(f"Running {int(byz_frac*100)}% Byzantine...", end=" ")
            result = run_bfedplc_experiment(seed, trainset, testloader, client_indices, config, byz_frac)
            results[f"byzantine_{int(byz_frac*100)}pct"].append(result)
            print(f"Best: {result['best_accuracy']:.2f}%, Detected: {result['attacks_detected']}")
    
    return results

# ============================================================
# Statistical Analysis
# ============================================================
def compute_statistics(results):
    """Compute mean ± std for all experiments"""
    stats_results = {}
    
    for key, runs in results.items():
        best_accs = [r['best_accuracy'] for r in runs]
        final_accs = [r['final_accuracy'] for r in runs]
        
        stats_results[key] = {
            'best_mean': np.mean(best_accs),
            'best_std': np.std(best_accs),
            'final_mean': np.mean(final_accs),
            'final_std': np.std(final_accs),
            'n_runs': len(runs)
        }
        
        if 'attacks_detected' in runs[0]:
            attacks = [r['attacks_detected'] for r in runs]
            stats_results[key]['attacks_mean'] = np.mean(attacks)
            stats_results[key]['attacks_std'] = np.std(attacks)
    
    return stats_results

def perform_ttest(results, key1, key2):
    """Perform t-test between two configurations"""
    accs1 = [r['best_accuracy'] for r in results[key1]]
    accs2 = [r['best_accuracy'] for r in results[key2]]
    
    t_stat, p_value = stats.ttest_ind(accs1, accs2)
    
    return {
        'key1': key1,
        'key2': key2,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

# ============================================================
# Visualization
# ============================================================
def plot_with_error_bars(stats_results, title, xlabel, ylabel, filename, x_labels=None):
    """Create bar plot with error bars"""
    plt.figure(figsize=(10, 6))
    
    keys = list(stats_results.keys())
    means = [stats_results[k]['best_mean'] for k in keys]
    stds = [stats_results[k]['best_std'] for k in keys]
    
    if x_labels is None:
        x_labels = keys
    
    x = np.arange(len(keys))
    bars = plt.bar(x, means, yerr=stds, capsize=5, color='steelblue', edgecolor='black', alpha=0.8)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(x, x_labels, rotation=45, ha='right')
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'plots/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: plots/{filename}")

def plot_convergence_with_std(all_results, title, filename):
    """Plot convergence curves with shaded std region"""
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for (key, runs), color in zip(all_results.items(), colors):
        # Stack histories
        histories = np.array([r['history'] for r in runs])
        mean_hist = np.mean(histories, axis=0)
        std_hist = np.std(histories, axis=0)
        
        rounds = np.arange(1, len(mean_hist) + 1)
        
        plt.plot(rounds, mean_hist, label=key, color=color, linewidth=2)
        plt.fill_between(rounds, mean_hist - std_hist, mean_hist + std_hist, 
                        color=color, alpha=0.2)
    
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: plots/{filename}")

# ============================================================
# Main
# ============================================================
def main():
    print("="*70)
    print("B-FedPLC MULTIPLE SEEDS EXPERIMENT")
    print("Statistical Validation with Mean ± Std")
    print("="*70)
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Seeds: {SEEDS}")
    print(f"Rounds per experiment: {NUM_ROUNDS}")
    
    # Load data
    print("\nLoading CIFAR-10...")
    trainset, testloader = load_data()
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    all_results = {}
    all_stats = {}
    start_time = time.time()
    
    # Run experiments
    print("\n" + "="*70)
    print("RUNNING ALL EXPERIMENTS WITH MULTIPLE SEEDS")
    print("="*70)
    
    # 1. Baseline comparison
    results = experiment_baseline(trainset, testloader, SEEDS)
    all_results['baseline'] = results
    all_stats['baseline'] = compute_statistics(results)
    
    # 2. Scalability
    results = experiment_scalability(trainset, testloader, SEEDS)
    all_results['scalability'] = results
    all_stats['scalability'] = compute_statistics(results)
    
    # 3. Non-IID
    results = experiment_noniid(trainset, testloader, SEEDS)
    all_results['noniid'] = results
    all_stats['noniid'] = compute_statistics(results)
    
    # 4. Security
    results = experiment_security(trainset, testloader, SEEDS)
    all_results['security'] = results
    all_stats['security'] = compute_statistics(results)
    
    total_time = (time.time() - start_time) / 60
    
    # Print summary
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY (Mean ± Std)")
    print("="*70)
    
    for exp_name, stats in all_stats.items():
        print(f"\n{exp_name.upper()}:")
        print("-" * 50)
        for key, values in stats.items():
            print(f"  {key}: {values['best_mean']:.2f} ± {values['best_std']:.2f}%")
    
    # Statistical tests
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTS (t-test, α=0.05)")
    print("="*70)
    
    # Test FedAvg vs B-FedPLC
    if 'FedAvg' in all_results['baseline'] and 'B-FedPLC' in all_results['baseline']:
        ttest = perform_ttest(all_results['baseline'], 'FedAvg', 'B-FedPLC')
        print(f"\nFedAvg vs B-FedPLC:")
        print(f"  t-statistic: {ttest['t_statistic']:.4f}")
        print(f"  p-value: {ttest['p_value']:.4f}")
        print(f"  Significant (p<0.05): {'Yes' if ttest['significant'] else 'No'}")
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS WITH ERROR BARS")
    print("="*70)
    
    # Baseline plot
    plot_with_error_bars(
        all_stats['baseline'],
        'Baseline Comparison: FedAvg vs B-FedPLC (n=3)',
        'Method', 'Best Accuracy (%)',
        'multiseed_baseline.png'
    )
    
    # Scalability plot
    plot_with_error_bars(
        all_stats['scalability'],
        'Scalability Analysis (n=3)',
        'Number of Clients', 'Best Accuracy (%)',
        'multiseed_scalability.png',
        ['10', '30', '50', '100']
    )
    
    # Non-IID plot
    plot_with_error_bars(
        all_stats['noniid'],
        'Non-IID Sensitivity (n=3)',
        'Dirichlet Alpha', 'Best Accuracy (%)',
        'multiseed_noniid.png',
        ['α=0.1', 'α=0.3', 'α=0.5', 'α=1.0']
    )
    
    # Security plot
    plot_with_error_bars(
        all_stats['security'],
        'Byzantine Fault Tolerance (n=3)',
        'Byzantine Fraction', 'Best Accuracy (%)',
        'multiseed_security.png',
        ['0%', '10%', '20%', '30%']
    )
    
    # Convergence plots
    plot_convergence_with_std(
        all_results['baseline'],
        'Convergence: FedAvg vs B-FedPLC (Mean ± Std)',
        'multiseed_convergence_baseline.png'
    )
    
    plot_convergence_with_std(
        all_results['scalability'],
        'Convergence by Client Count (Mean ± Std)',
        'multiseed_convergence_scalability.png'
    )
    
    plot_convergence_with_std(
        all_results['noniid'],
        'Convergence by Non-IID Level (Mean ± Std)',
        'multiseed_convergence_noniid.png'
    )
    
    plot_convergence_with_std(
        all_results['security'],
        'Convergence under Byzantine Attack (Mean ± Std)',
        'multiseed_convergence_security.png'
    )
    
    # Save results
    results_file = 'multiseed_results.json'
    
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    save_data = {
        'seeds': SEEDS,
        'config': {
            'rounds': NUM_ROUNDS,
            'clients': NUM_CLIENTS,
            'client_fraction': CLIENT_FRACTION,
            'local_epochs': LOCAL_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        },
        'results': convert_to_serializable(all_results),
        'statistics': convert_to_serializable(all_stats),
        'total_time_minutes': total_time
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    print("\n" + "="*70)
    print(f"ALL EXPERIMENTS COMPLETED!")
    print(f"Total Time: {total_time:.2f} minutes")
    print("="*70)
    
    # Final summary table
    print("\n" + "="*70)
    print("FINAL RESULTS TABLE (for dissertation)")
    print("="*70)
    print("\n| Experiment | Configuration | Best Acc (Mean±Std) |")
    print("|------------|---------------|---------------------|")
    
    for exp_name, stats in all_stats.items():
        for key, values in stats.items():
            print(f"| {exp_name} | {key} | {values['best_mean']:.2f} ± {values['best_std']:.2f}% |")

if __name__ == "__main__":
    main()
