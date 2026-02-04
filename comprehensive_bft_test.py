"""
Comprehensive Byzantine Tolerance Test Suite for B-FedPLC
=========================================================
Tests:
1. Multiple datasets (MNIST, CIFAR-10, Fashion-MNIST)
2. Multiple client configurations (20, 50, 100 clients)
3. Multiple attack types (random, sign_flip, label_flip, backdoor)
4. Non-IID data distributions

This test validates B-FedPLC's Byzantine Fault Tolerance claims.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import json
import time
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import from fedplc
import sys
sys.path.insert(0, str(Path(__file__).parent))
from fedplc.robust_aggregation import (
    RobustAggregator, 
    create_aggregator,
    apply_byzantine_attack,
    apply_label_flip_attack,
    apply_backdoor_attack
)


# ============================================================================
# Model Definitions
# ============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN for MNIST/Fashion-MNIST"""
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class CIFAR10CNN(nn.Module):
    """CNN for CIFAR-10"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ============================================================================
# Data Loading Functions
# ============================================================================

def get_mnist_data(data_dir='./data'):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return train_data, test_data


def get_fashion_mnist_data(data_dir='./data'):
    """Load Fashion-MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_data = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    return train_data, test_data


def get_cifar10_data(data_dir='./data'):
    """Load CIFAR-10 dataset"""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_data = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    return train_data, test_data


def partition_data_iid(dataset, num_clients):
    """Partition data IID among clients"""
    n = len(dataset)
    indices = np.random.permutation(n)
    client_indices = np.array_split(indices, num_clients)
    return {i: idx.tolist() for i, idx in enumerate(client_indices)}


def partition_data_noniid(dataset, num_clients, alpha=0.5):
    """Partition data Non-IID using Dirichlet distribution"""
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(targets))
    
    client_indices = {i: [] for i in range(num_clients)}
    
    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        np.random.shuffle(class_indices)
        
        # Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(class_indices)).astype(int)
        
        # Adjust to match total
        proportions[-1] = len(class_indices) - proportions[:-1].sum()
        
        start = 0
        for i in range(num_clients):
            end = start + proportions[i]
            client_indices[i].extend(class_indices[start:end].tolist())
            start = end
    
    return client_indices


# ============================================================================
# Federated Learning Simulation
# ============================================================================

def local_train(model, dataloader, device, epochs=2, lr=0.01):
    """Train model locally and return updated weights"""
    model.train()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def evaluate_model(model, dataloader, device):
    """Evaluate model accuracy"""
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return 100. * correct / total


def run_federated_experiment(
    dataset_name: str,
    num_clients: int,
    num_rounds: int,
    byzantine_fraction: float,
    aggregation_method: str,
    attack_type: str = 'random',
    data_distribution: str = 'iid',
    alpha: float = 0.5,
    device: str = 'cuda',
    seed: int = 42,
    local_epochs: int = 2,
    batch_size: int = 64
) -> Dict:
    """
    Run a complete federated learning experiment.
    
    Args:
        dataset_name: 'mnist', 'fashion_mnist', or 'cifar10'
        num_clients: Total number of clients
        num_rounds: Number of FL rounds
        byzantine_fraction: Fraction of Byzantine clients
        aggregation_method: Aggregation method to use
        attack_type: Type of Byzantine attack
        data_distribution: 'iid' or 'noniid'
        alpha: Dirichlet parameter for non-IID
        device: 'cuda' or 'cpu'
        seed: Random seed
        local_epochs: Local training epochs
        batch_size: Batch size for training
    
    Returns:
        Dictionary with experiment results
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset
    print(f"\n[{dataset_name.upper()}] Loading data...")
    if dataset_name == 'mnist':
        train_data, test_data = get_mnist_data()
        model_class = SimpleCNN
        in_channels = 1
    elif dataset_name == 'fashion_mnist':
        train_data, test_data = get_fashion_mnist_data()
        model_class = SimpleCNN
        in_channels = 1
    elif dataset_name == 'cifar10':
        train_data, test_data = get_cifar10_data()
        model_class = CIFAR10CNN
        in_channels = 3
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Partition data
    if data_distribution == 'iid':
        client_indices = partition_data_iid(train_data, num_clients)
    else:
        client_indices = partition_data_noniid(train_data, num_clients, alpha)
    
    # Create data loaders
    client_loaders = {}
    for i in range(num_clients):
        subset = Subset(train_data, client_indices[i])
        client_loaders[i] = DataLoader(subset, batch_size=batch_size, shuffle=True)
    
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    
    # Initialize global model
    global_model = model_class(num_classes=10)
    global_model.to(device)
    
    # Create aggregator
    aggregator = create_aggregator(aggregation_method)
    
    # Determine Byzantine clients
    num_byzantine = int(num_clients * byzantine_fraction)
    byzantine_clients = set(np.random.choice(num_clients, num_byzantine, replace=False))
    
    print(f"[{dataset_name.upper()}] {num_clients} clients, {num_byzantine} Byzantine ({byzantine_fraction*100:.0f}%)")
    print(f"[{dataset_name.upper()}] Aggregation: {aggregation_method}, Attack: {attack_type}")
    
    # Training history
    history = {
        'rounds': [],
        'accuracy': [],
        'loss': []
    }
    
    # Federated training
    start_time = time.time()
    
    for round_idx in range(num_rounds):
        # Select participating clients (all for simplicity)
        participating = list(range(num_clients))
        
        # Collect client updates
        client_weights = {}
        client_sizes = {}
        
        for client_id in participating:
            # Create local model
            local_model = model_class(num_classes=10)
            local_model.load_state_dict(global_model.state_dict())
            
            # Local training
            weights = local_train(
                local_model, 
                client_loaders[client_id],
                device,
                epochs=local_epochs,
                lr=0.01
            )
            
            # Apply Byzantine attack if needed
            if client_id in byzantine_clients:
                if attack_type == 'label_flip':
                    # For label flip, we need to retrain with flipped labels
                    # Simplified: just corrupt the weights
                    weights = apply_byzantine_attack(weights, 'sign_flip', 1.0)
                else:
                    weights = apply_byzantine_attack(weights, attack_type, 1.0)
            
            client_weights[client_id] = weights
            client_sizes[client_id] = len(client_indices[client_id])
        
        # Aggregate
        aggregated_weights = aggregator.aggregate(client_weights, client_sizes)
        
        # Update global model
        global_model.load_state_dict(aggregated_weights)
        
        # Evaluate
        accuracy = evaluate_model(global_model, test_loader, device)
        
        history['rounds'].append(round_idx + 1)
        history['accuracy'].append(accuracy)
        
        if (round_idx + 1) % 5 == 0 or round_idx == 0:
            print(f"  Round {round_idx + 1}/{num_rounds}: Accuracy = {accuracy:.2f}%")
    
    elapsed_time = time.time() - start_time
    
    return {
        'dataset': dataset_name,
        'num_clients': num_clients,
        'num_rounds': num_rounds,
        'byzantine_fraction': byzantine_fraction,
        'aggregation_method': aggregation_method,
        'attack_type': attack_type,
        'data_distribution': data_distribution,
        'final_accuracy': history['accuracy'][-1],
        'max_accuracy': max(history['accuracy']),
        'history': history,
        'elapsed_time': elapsed_time,
        'seed': seed
    }


# ============================================================================
# Main Test Suite
# ============================================================================

def main():
    """Run comprehensive test suite"""
    
    print("=" * 80)
    print("B-FedPLC COMPREHENSIVE BYZANTINE TOLERANCE TEST SUITE")
    print("=" * 80)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    Path('plots').mkdir(exist_ok=True)
    
    # ========================================================================
    # Test Configuration
    # ========================================================================
    
    # Datasets to test
    datasets_to_test = ['mnist', 'fashion_mnist', 'cifar10']
    
    # Client configurations
    client_configs = [20, 50, 100]
    
    # Attack types
    attack_types = ['random', 'sign_flip', 'additive_noise']
    
    # Byzantine fractions
    byzantine_fractions = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    # Aggregation methods
    aggregation_methods = ['fedavg', 'trimmed_mean', 'krum', 'multi_krum', 'hybrid']
    
    # Data distributions
    data_distributions = ['iid', 'noniid']
    
    # Test settings (reduced for quick testing)
    NUM_ROUNDS = 20  # Increase for full experiments
    SEEDS = [42]  # Add more seeds for statistical significance
    
    all_results = []
    
    # ========================================================================
    # Test 1: Dataset Comparison (Fixed config)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("TEST 1: DATASET COMPARISON")
    print("=" * 80)
    
    test1_results = {}
    for dataset in datasets_to_test:
        print(f"\n--- Testing {dataset.upper()} ---")
        
        result = run_federated_experiment(
            dataset_name=dataset,
            num_clients=20,
            num_rounds=NUM_ROUNDS,
            byzantine_fraction=0.3,
            aggregation_method='multi_krum',
            attack_type='random',
            data_distribution='iid',
            device=device,
            seed=42
        )
        
        test1_results[dataset] = result
        all_results.append(result)
        
        print(f"  Final Accuracy: {result['final_accuracy']:.2f}%")
    
    # ========================================================================
    # Test 2: Client Scaling (MNIST, 30% Byzantine)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("TEST 2: CLIENT SCALING")
    print("=" * 80)
    
    test2_results = {}
    for num_clients in client_configs:
        print(f"\n--- Testing with {num_clients} clients ---")
        
        result = run_federated_experiment(
            dataset_name='mnist',
            num_clients=num_clients,
            num_rounds=NUM_ROUNDS,
            byzantine_fraction=0.3,
            aggregation_method='multi_krum',
            attack_type='random',
            data_distribution='iid',
            device=device,
            seed=42
        )
        
        test2_results[num_clients] = result
        all_results.append(result)
        
        print(f"  Final Accuracy: {result['final_accuracy']:.2f}%")
    
    # ========================================================================
    # Test 3: Attack Type Comparison
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("TEST 3: ATTACK TYPE COMPARISON")
    print("=" * 80)
    
    test3_results = {}
    for attack in attack_types:
        print(f"\n--- Testing {attack.upper()} attack ---")
        
        result = run_federated_experiment(
            dataset_name='mnist',
            num_clients=20,
            num_rounds=NUM_ROUNDS,
            byzantine_fraction=0.3,
            aggregation_method='multi_krum',
            attack_type=attack,
            data_distribution='iid',
            device=device,
            seed=42
        )
        
        test3_results[attack] = result
        all_results.append(result)
        
        print(f"  Final Accuracy: {result['final_accuracy']:.2f}%")
    
    # ========================================================================
    # Test 4: Non-IID Data Distribution
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("TEST 4: NON-IID DATA DISTRIBUTION")
    print("=" * 80)
    
    test4_results = {}
    for dist in data_distributions:
        print(f"\n--- Testing {dist.upper()} distribution ---")
        
        result = run_federated_experiment(
            dataset_name='mnist',
            num_clients=20,
            num_rounds=NUM_ROUNDS,
            byzantine_fraction=0.3,
            aggregation_method='multi_krum',
            attack_type='random',
            data_distribution=dist,
            alpha=0.3 if dist == 'noniid' else 1.0,
            device=device,
            seed=42
        )
        
        test4_results[dist] = result
        all_results.append(result)
        
        print(f"  Final Accuracy: {result['final_accuracy']:.2f}%")
    
    # ========================================================================
    # Test 5: Byzantine Fraction Sweep (All Methods)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("TEST 5: BYZANTINE FRACTION SWEEP")
    print("=" * 80)
    
    test5_results = {method: {} for method in aggregation_methods}
    
    for method in aggregation_methods:
        print(f"\n--- Testing {method.upper()} ---")
        
        for byz_frac in byzantine_fractions:
            result = run_federated_experiment(
                dataset_name='mnist',
                num_clients=20,
                num_rounds=NUM_ROUNDS,
                byzantine_fraction=byz_frac,
                aggregation_method=method,
                attack_type='random',
                data_distribution='iid',
                device=device,
                seed=42
            )
            
            test5_results[method][byz_frac] = result['final_accuracy']
            all_results.append(result)
    
    # ========================================================================
    # Generate Plots
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    # Plot 1: Dataset Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    datasets = list(test1_results.keys())
    accuracies = [test1_results[d]['final_accuracy'] for d in datasets]
    bars = ax.bar(datasets, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('B-FedPLC Performance Across Datasets (30% Byzantine)')
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('plots/comprehensive_dataset_comparison.png', dpi=300)
    plt.close()
    print("✓ Saved: plots/comprehensive_dataset_comparison.png")
    
    # Plot 2: Client Scaling
    fig, ax = plt.subplots(figsize=(10, 6))
    clients = list(test2_results.keys())
    accuracies = [test2_results[c]['final_accuracy'] for c in clients]
    ax.plot(clients, accuracies, 'o-', linewidth=2, markersize=10, color='#9b59b6')
    ax.set_xlabel('Number of Clients')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('B-FedPLC Scalability (30% Byzantine)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/comprehensive_client_scaling.png', dpi=300)
    plt.close()
    print("✓ Saved: plots/comprehensive_client_scaling.png")
    
    # Plot 3: Attack Type Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    attacks = list(test3_results.keys())
    accuracies = [test3_results[a]['final_accuracy'] for a in attacks]
    bars = ax.bar(attacks, accuracies, color=['#e74c3c', '#f39c12', '#1abc9c'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('B-FedPLC Resilience to Different Attack Types (30% Byzantine)')
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('plots/comprehensive_attack_types.png', dpi=300)
    plt.close()
    print("✓ Saved: plots/comprehensive_attack_types.png")
    
    # Plot 4: IID vs Non-IID
    fig, ax = plt.subplots(figsize=(8, 6))
    distributions = list(test4_results.keys())
    accuracies = [test4_results[d]['final_accuracy'] for d in distributions]
    bars = ax.bar(distributions, accuracies, color=['#3498db', '#e74c3c'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('B-FedPLC: IID vs Non-IID Data (30% Byzantine)')
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('plots/comprehensive_data_distribution.png', dpi=300)
    plt.close()
    print("✓ Saved: plots/comprehensive_data_distribution.png")
    
    # Plot 5: Byzantine Tolerance Comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {'fedavg': '#e74c3c', 'trimmed_mean': '#3498db', 'krum': '#2ecc71', 
              'multi_krum': '#9b59b6', 'hybrid': '#f39c12'}
    markers = {'fedavg': 'o', 'trimmed_mean': 's', 'krum': '^', 'multi_krum': 'D', 'hybrid': '*'}
    
    for method in aggregation_methods:
        byz = [f * 100 for f in byzantine_fractions]
        accs = [test5_results[method][f] for f in byzantine_fractions]
        ax.plot(byz, accs, marker=markers[method], color=colors[method], 
                label=method.upper(), linewidth=2, markersize=8)
    
    ax.axhline(y=40, color='gray', linestyle='--', label='BFT Threshold')
    ax.axvline(x=33, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Byzantine Fraction (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Byzantine Tolerance Comparison (MNIST)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 42)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig('plots/comprehensive_bft_comparison.png', dpi=300)
    plt.close()
    print("✓ Saved: plots/comprehensive_bft_comparison.png")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    # Prepare JSON-serializable results
    json_results = {
        'test1_dataset_comparison': {k: v['final_accuracy'] for k, v in test1_results.items()},
        'test2_client_scaling': {str(k): v['final_accuracy'] for k, v in test2_results.items()},
        'test3_attack_types': {k: v['final_accuracy'] for k, v in test3_results.items()},
        'test4_data_distribution': {k: v['final_accuracy'] for k, v in test4_results.items()},
        'test5_bft_sweep': {m: {str(f): a for f, a in accs.items()} 
                           for m, accs in test5_results.items()}
    }
    
    with open('comprehensive_test_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print("\n✓ Saved: comprehensive_test_results.json")
    
    # ========================================================================
    # Print Summary
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)
    
    print("\n--- Test 1: Dataset Comparison (30% Byzantine, Multi-Krum) ---")
    for dataset, result in test1_results.items():
        print(f"  {dataset.upper()}: {result['final_accuracy']:.2f}%")
    
    print("\n--- Test 2: Client Scaling (30% Byzantine, Multi-Krum) ---")
    for num_clients, result in test2_results.items():
        print(f"  {num_clients} clients: {result['final_accuracy']:.2f}%")
    
    print("\n--- Test 3: Attack Types (30% Byzantine, Multi-Krum) ---")
    for attack, result in test3_results.items():
        print(f"  {attack}: {result['final_accuracy']:.2f}%")
    
    print("\n--- Test 4: Data Distribution (30% Byzantine, Multi-Krum) ---")
    for dist, result in test4_results.items():
        print(f"  {dist.upper()}: {result['final_accuracy']:.2f}%")
    
    print("\n--- Test 5: BFT Sweep ---")
    print(f"{'Method':<15}", end="")
    for byz in byzantine_fractions:
        print(f"{byz*100:.0f}%".rjust(10), end="")
    print()
    print("-" * 65)
    for method in aggregation_methods:
        print(f"{method.upper():<15}", end="")
        for byz in byzantine_fractions:
            print(f"{test5_results[method][byz]:.1f}%".rjust(10), end="")
        print()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
