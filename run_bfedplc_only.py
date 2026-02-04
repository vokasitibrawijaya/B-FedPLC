"""
Run hanya B-FedPLC test untuk verifikasi cepat
Lebih cepat daripada run semua methods
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_sota_comparison import (
    Config, get_dataset, dirichlet_split, apply_byzantine_attack,
    local_train, evaluate, apply_update, bfedplc_aggregation,
    MNISTNet, CIFAR10Net
)
import torch
import numpy as np
import random
import copy
from torch.utils.data import DataLoader
from datetime import datetime

def run_bfedplc_test(dataset_name='MNIST', byzantine_fraction=0.2, attack_type='sign_flip', 
                     num_rounds=30, seed=42, verbose=True):
    """Run B-FedPLC test only"""
    
    print("=" * 70)
    print(f"B-FedPLC TEST - {dataset_name} with {byzantine_fraction*100:.0f}% Byzantine")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    config = Config()
    config.num_rounds = num_rounds
    device = config.device
    
    print(f"Device: {device}")
    print(f"Rounds: {num_rounds}")
    print(f"Byzantine: {byzantine_fraction*100:.0f}%")
    print(f"Attack: {attack_type}\n")
    
    # Get dataset
    train_dataset, test_dataset = get_dataset(dataset_name)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Split data
    client_datasets = dirichlet_split(train_dataset, config.num_clients, config.dirichlet_alpha)
    client_loaders = [DataLoader(ds, batch_size=config.batch_size, shuffle=True) 
                      for ds in client_datasets]
    
    # Initialize model
    if dataset_name == 'MNIST':
        global_model = MNISTNet().to(device)
    else:
        global_model = CIFAR10Net().to(device)
    
    # Byzantine clients
    n_byzantine = int(config.num_clients * byzantine_fraction)
    byzantine_indices = list(range(n_byzantine))
    
    print(f"Byzantine clients: {n_byzantine} (indices: {byzantine_indices})\n")
    print("Starting training...\n")
    
    # Training loop
    accuracies = []
    
    for round_num in range(1, config.num_rounds + 1):
        # Local training
        updates = []
        for client_id in range(config.num_clients):
            client_model = copy.deepcopy(global_model)
            update = local_train(client_model, client_loaders[client_id], config, device)
            updates.append(update)
        
        # Apply Byzantine attack
        if n_byzantine > 0:
            updates = apply_byzantine_attack(updates, byzantine_indices, attack_type)
        
        # Aggregate dengan B-FedPLC
        global_update = bfedplc_aggregation(
            updates,
            n_byzantine=n_byzantine,
            num_clusters=config.num_clusters,
            attack_type=attack_type,
            verbose=verbose and (round_num <= 3 or round_num % 10 == 0)
        )
        
        # Fallback
        if global_update is None:
            from phase7_sota_comparison import multi_krum
            global_update = multi_krum(updates, n_byzantine)
        
        # Apply update
        if global_update is not None:
            apply_update(global_model, global_update)
        
        # Evaluate
        acc = evaluate(global_model, test_loader, device)
        accuracies.append(acc)
        
        if round_num % 5 == 0 or round_num == 1:
            print(f"Round {round_num:3d}/{num_rounds}: Accuracy = {acc:.2f}%")
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Initial Accuracy: {accuracies[0]:.2f}%")
    print(f"Final Accuracy:   {accuracies[-1]:.2f}%")
    print(f"Best Accuracy:    {max(accuracies):.2f}%")
    print(f"Average Accuracy: {np.mean(accuracies):.2f}%")
    
    # Compare with baseline
    baseline = 9.8
    improvement = accuracies[-1] - baseline
    improvement_pct = (improvement / baseline) * 100 if baseline > 0 else 0
    
    print(f"\nComparison:")
    print(f"  Baseline (Old):   {baseline:.2f}%")
    print(f"  Current Result:   {accuracies[-1]:.2f}%")
    print(f"  Improvement:      {improvement:+.2f}% ({improvement_pct:+.1f}%)")
    
    # Success criteria
    target = 95.0
    print(f"\nSuccess Criteria:")
    print(f"  Target:           >={target:.1f}%")
    print(f"  Current:          {accuracies[-1]:.2f}%")
    
    if accuracies[-1] >= target:
        print(f"  Status:           [SUCCESS] (>={target}%)")
        success = True
    elif accuracies[-1] >= 90.0:
        print(f"  Status:           [GOOD] (≥90% but <{target}%)")
        success = False
    elif accuracies[-1] >= 50.0:
        print(f"  Status:           [PARTIAL] (≥50% but <90%)")
        success = False
    else:
        print(f"  Status:           [FAIL] (<50%)")
        success = False
    
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return {
        'final_accuracy': accuracies[-1],
        'best_accuracy': max(accuracies),
        'success': success
    }

if __name__ == "__main__":
    print("\nRunning B-FedPLC Quick Test...")
    print("This will test B-FedPLC with Byzantine attacks to verify the fix.\n")
    
    # Test MNIST with 20% Byzantine
    result = run_bfedplc_test(
        dataset_name='MNIST',
        byzantine_fraction=0.2,
        attack_type='sign_flip',
        num_rounds=30,
        seed=42,
        verbose=True
    )
    
    if result['success']:
        print("\n[SUCCESS] B-FedPLC fix is working! Accuracy ≥95%")
        sys.exit(0)
    elif result['final_accuracy'] >= 50:
        print(f"\n[PARTIAL] Improvement detected ({result['final_accuracy']:.2f}%) but not at target yet")
        sys.exit(1)
    else:
        print(f"\n[FAIL] Fix not working as expected ({result['final_accuracy']:.2f}%)")
        sys.exit(2)
