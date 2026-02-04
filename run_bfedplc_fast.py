"""
Run B-FedPLC test dengan rounds lebih sedikit untuk verifikasi cepat
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_sota_comparison import (
    Config, get_dataset, dirichlet_split, apply_byzantine_attack,
    local_train, evaluate, apply_update, bfedplc_aggregation,
    MNISTNet
)
import torch
import numpy as np
import random
import copy
from torch.utils.data import DataLoader
from datetime import datetime

def run_bfedplc_fast_test():
    """Run B-FedPLC dengan 15 rounds untuk verifikasi cepat"""
    
    print("=" * 70)
    print("B-FedPLC FAST TEST - MNIST with 20% Byzantine")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Set seeds
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    config = Config()
    config.num_rounds = 15  # Reduced for faster test
    device = config.device
    
    print(f"Device: {device}")
    print(f"Rounds: {config.num_rounds} (reduced for quick test)")
    print(f"Byzantine: 20%")
    print(f"Attack: sign_flip\n")
    
    # Get dataset
    train_dataset, test_dataset = get_dataset('MNIST')
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Split data
    client_datasets = dirichlet_split(train_dataset, config.num_clients, config.dirichlet_alpha)
    client_loaders = [DataLoader(ds, batch_size=config.batch_size, shuffle=True) 
                      for ds in client_datasets]
    
    # Initialize model
    global_model = MNISTNet().to(device)
    
    # Byzantine clients
    n_byzantine = int(config.num_clients * 0.2)
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
        updates = apply_byzantine_attack(updates, byzantine_indices, 'sign_flip')
        
        # Aggregate dengan B-FedPLC
        global_update = bfedplc_aggregation(
            updates,
            n_byzantine=n_byzantine,
            num_clusters=config.num_clusters,
            attack_type='sign_flip',
            verbose=(round_num <= 2)  # Only verbose for first 2 rounds
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
        
        if round_num % 3 == 0 or round_num == 1:
            print(f"Round {round_num:3d}/{config.num_rounds}: Accuracy = {acc:.2f}%")
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Initial Accuracy: {accuracies[0]:.2f}%")
    print(f"Final Accuracy:   {accuracies[-1]:.2f}%")
    print(f"Best Accuracy:    {max(accuracies):.2f}%")
    
    # Compare with baseline
    baseline = 9.8
    improvement = accuracies[-1] - baseline
    
    print(f"\nComparison:")
    print(f"  Baseline (Old):   {baseline:.2f}%")
    print(f"  Current Result:   {accuracies[-1]:.2f}%")
    print(f"  Improvement:      {improvement:+.2f}%")
    
    # Success criteria
    target = 95.0
    print(f"\nSuccess Criteria:")
    print(f"  Target:           >={target:.1f}%")
    print(f"  Current:          {accuracies[-1]:.2f}%")
    
    if accuracies[-1] >= target:
        print(f"  Status:           [SUCCESS] (>={target}%)")
        success = True
    elif accuracies[-1] >= 50.0:
        print(f"  Status:           [PARTIAL] (≥50% but <{target}%)")
        success = False
    else:
        print(f"  Status:           [FAIL] (<50%)")
        success = False
    
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return {
        'final_accuracy': accuracies[-1],
        'best_accuracy': max(accuracies),
        'success': success,
        'accuracies': accuracies
    }

if __name__ == "__main__":
    result = run_bfedplc_fast_test()
    
    if result['success']:
        print("\n[SUCCESS] B-FedPLC fix is working! Accuracy ≥95%")
        sys.exit(0)
    elif result['final_accuracy'] >= 50:
        print(f"\n[PARTIAL] Improvement detected ({result['final_accuracy']:.2f}%)")
        print("Note: With more rounds (30+), accuracy should improve further")
        sys.exit(1)
    else:
        print(f"\n[FAIL] Fix not working as expected ({result['final_accuracy']:.2f}%)")
        sys.exit(2)
