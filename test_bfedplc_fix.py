"""
Quick Test untuk Memverifikasi Perbaikan Byzantine Detection
Test B-FedPLC dengan Byzantine attacks dan compare dengan baseline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import random
import sys
import os

# Import functions from phase7
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase7_sota_comparison import (
    MNISTNet, get_dataset, dirichlet_split, 
    apply_byzantine_attack, local_train, evaluate, apply_update,
    bfedplc_aggregation, multi_krum, fedavg
)

def quick_test_bfedplc(num_rounds=10, byzantine_fraction=0.2, seed=42, verbose=True):
    """Quick test B-FedPLC dengan Byzantine attacks"""
    
    print("=" * 70)
    print("QUICK TEST: B-FedPLC Byzantine Detection Fix")
    print("=" * 70)
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Configuration:")
    print(f"  - Rounds: {num_rounds}")
    print(f"  - Byzantine Fraction: {byzantine_fraction*100:.0f}%")
    print(f"  - Seed: {seed}")
    print()
    
    # Get dataset
    train_dataset, test_dataset = get_dataset('MNIST')
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Split data
    num_clients = 10
    client_datasets = dirichlet_split(train_dataset, num_clients, alpha=0.5)
    client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) 
                      for ds in client_datasets]
    
    # Initialize model
    global_model = MNISTNet().to(device)
    
    # Byzantine clients
    n_byzantine = int(num_clients * byzantine_fraction)
    byzantine_indices = list(range(n_byzantine))
    
    print(f"Byzantine Setup:")
    print(f"  - Total Clients: {num_clients}")
    print(f"  - Byzantine Clients: {n_byzantine} (indices: {byzantine_indices})")
    print()
    
    # Training loop
    accuracies = []
    detection_stats = []
    
    print("Starting Training...")
    print("-" * 70)
    
    for round_num in range(1, num_rounds + 1):
        # Local training
        updates = []
        import copy
        for client_id in range(num_clients):
            client_model = copy.deepcopy(global_model)
            update = local_train(client_model, client_loaders[client_id], 
                               type('Config', (), {'lr': 0.01, 'local_epochs': 2})(), device)
            updates.append(update)
        
        # Apply Byzantine attack
        if n_byzantine > 0:
            updates = apply_byzantine_attack(updates, byzantine_indices, 'sign_flip')
        
        # Aggregate dengan B-FedPLC (dengan verbose)
        if verbose and round_num <= 3:
            print(f"\nRound {round_num} - B-FedPLC Aggregation:")
        
        global_update = bfedplc_aggregation(
            updates, 
            n_byzantine=n_byzantine, 
            num_clusters=3, 
            attack_type='sign_flip',
            verbose=verbose and round_num <= 3
        )
        
        # Fallback jika gagal
        if global_update is None:
            if verbose:
                print(f"  Warning: B-FedPLC returned None, using Multi-Krum fallback")
            global_update = multi_krum(updates, n_byzantine)
        
        # Apply update
        if global_update is not None:
            apply_update(global_model, global_update)
        
        # Evaluate
        acc = evaluate(global_model, test_loader, device)
        accuracies.append(acc)
        
        if round_num % 2 == 0 or round_num == 1:
            print(f"Round {round_num:2d}/{num_rounds}: Accuracy = {acc:.2f}%")
    
    print("-" * 70)
    print(f"\nResults:")
    print(f"  Initial Accuracy: {accuracies[0]:.2f}%")
    print(f"  Final Accuracy:   {accuracies[-1]:.2f}%")
    print(f"  Best Accuracy:    {max(accuracies):.2f}%")
    print(f"  Average Accuracy: {np.mean(accuracies):.2f}%")
    
    # Compare dengan baseline
    baseline = 9.8  # Old result
    improvement = accuracies[-1] - baseline
    improvement_pct = (improvement / baseline) * 100 if baseline > 0 else 0
    
    print(f"\nComparison with Baseline:")
    print(f"  Baseline (Old):   {baseline:.2f}%")
    print(f"  Current Result:   {accuracies[-1]:.2f}%")
    print(f"  Improvement:      {improvement:+.2f}% ({improvement_pct:+.1f}%)")
    
    # Success criteria
    target = 95.0
    print(f"\nSuccess Criteria:")
    print(f"  Target:           ≥{target:.1f}%")
    print(f"  Current:         {accuracies[-1]:.2f}%")
    
    if accuracies[-1] >= target:
        print(f"  Status:           ✅ SUCCESS (≥{target}%)")
    elif accuracies[-1] >= 90.0:
        print(f"  Status:           ⚠️  GOOD (≥90% but <{target}%)")
    elif accuracies[-1] >= 50.0:
        print(f"  Status:           ⚠️  PARTIAL (≥50% but <90%)")
    else:
        print(f"  Status:           ❌ FAILED (<50%)")
    
    print("=" * 70)
    
    return {
        'accuracies': accuracies,
        'final_accuracy': accuracies[-1],
        'best_accuracy': max(accuracies),
        'baseline': baseline,
        'improvement': improvement,
        'success': accuracies[-1] >= target
    }

if __name__ == "__main__":
    # Run test
    result = quick_test_bfedplc(
        num_rounds=10,
        byzantine_fraction=0.2,
        seed=42,
        verbose=True
    )
    
    # Exit code based on success
    if result['success']:
        print("\n✅ TEST PASSED: B-FedPLC fix is working!")
        sys.exit(0)
    elif result['final_accuracy'] >= 50.0:
        print("\n⚠️  TEST PARTIAL: Improvement detected but not at target yet")
        sys.exit(1)
    else:
        print("\n❌ TEST FAILED: Fix not working as expected")
        sys.exit(2)
