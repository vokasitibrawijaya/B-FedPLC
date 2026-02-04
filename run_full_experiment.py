"""
Full FedPLC Experiment - Following Paper Specifications
Dataset: CIFAR-10, SVHN, Fashion-MNIST
Clients: 100
Rounds: 200
With Concept Drift simulation
"""

import os
import sys
import time
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fedplc.config import ExperimentConfig
from fedplc.data import create_federated_dataloaders, ConceptDriftSimulator
from fedplc.models import create_model
from fedplc.server import FedPLCServer, FedPLCClient
from fedplc.utils import set_seed


def run_experiment(config: ExperimentConfig, drift_type: str = 'none'):
    """Run single FedPLC experiment"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config.seed)
    
    print(f"\n{'='*70}", flush=True)
    print(f"Running FedPLC Experiment", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Dataset: {config.dataset}", flush=True)
    print(f"Clients: {config.num_clients}", flush=True)
    print(f"Rounds: {config.num_rounds}", flush=True)
    print(f"Local Epochs: {config.local_epochs}", flush=True)
    print(f"Warmup Rounds: {config.warmup_rounds}", flush=True)
    print(f"Drift Type: {drift_type}", flush=True)
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)
    print(f"{'='*70}\n", flush=True)
    
    # Create data loaders
    print("[1/4] Creating federated dataloaders...", flush=True)
    train_loaders, test_loader, partitioner = create_federated_dataloaders(
        dataset_name=config.dataset,
        num_clients=config.num_clients,
        alpha=config.alpha,
        batch_size=config.batch_size,
        data_dir=config.data_dir
    )
    
    # Create model
    print("[2/4] Creating model...", flush=True)
    global_model = create_model(config.dataset).to(device)
    
    # Create server and clients
    print("[3/4] Initializing server and clients...", flush=True)
    server = FedPLCServer(
        config=config,
        model=global_model,
        test_loader=test_loader
    )
    
    clients = []
    for i in range(config.num_clients):
        client = FedPLCClient(
            client_id=i,
            config=config,
            dataloader=train_loaders[i]
        )
        clients.append(client)
        if (i + 1) % 20 == 0:
            print(f"  Created {i+1}/{config.num_clients} clients...", flush=True)
    
    # Setup drift simulator if needed
    drift_simulator = None
    if drift_type != 'none':
        drift_simulator = ConceptDriftSimulator(
            drift_type=drift_type,
            drift_round=config.drift_round,
            num_classes=config.num_classes
        )
    
    # Training loop
    print("\n[4/4] Starting training...", flush=True)
    results = {
        'accuracy': [],
        'loss': [],
        'communities': [],
        'config': {
            'dataset': config.dataset,
            'num_clients': config.num_clients,
            'num_rounds': config.num_rounds,
            'alpha': config.alpha,
            'drift_type': drift_type,
            'warmup_rounds': config.warmup_rounds,
            'similarity_threshold': config.similarity_threshold
        }
    }
    
    best_acc = 0.0
    start_time = time.time()
    
    pbar = tqdm(range(1, config.num_rounds + 1), desc="Training", ncols=100)
    
    for round_num in pbar:
        round_start = time.time()
        
        # Check for concept drift
        if drift_simulator is not None:
            label_mapping = drift_simulator.get_label_mapping(round_num)
        else:
            label_mapping = None
        
        # Select clients for this round
        selected_indices = server.select_clients()
        
        # Determine if warmup phase
        is_warmup = round_num <= config.warmup_rounds
        server.is_warmup = is_warmup
        
        # Distribute global model and collect updates
        global_state = server.get_global_model_state()
        global_prototypes = server.get_global_prototypes()
        
        client_updates = {}
        client_data_sizes = {}
        
        for idx in selected_indices:
            client = clients[idx]
            
            # Send global model to client
            client.receive_global_model(global_state)
            
            # Train locally
            update = client.train(
                global_prototypes=global_prototypes,
                warmup=is_warmup
            )
            
            client_updates[idx] = update
            client_data_sizes[idx] = update['data_size']
        
        # Server aggregation
        server.aggregate_round(client_updates, client_data_sizes)
        
        # Update round counter
        server.step_round()
        
        # Evaluate
        acc, loss = server.evaluate()
        
        results['accuracy'].append(acc)
        results['loss'].append(loss)
        
        if acc > best_acc:
            best_acc = acc
        
        # Log round
        server.log_round(acc, loss)
        
        # Update progress bar
        phase = "WARMUP" if is_warmup else "LDCA"
        drift_info = f" | DRIFT@{config.drift_round}" if (drift_type != 'none' and round_num >= config.drift_round) else ""
        pbar.set_postfix_str(f"[{phase}] Acc: {acc:.2f}% | Best: {best_acc:.2f}%{drift_info}")
        
        # Periodic logging
        if round_num % 10 == 0 or round_num == config.num_rounds:
            round_time = time.time() - round_start
            print(f"\n  Round {round_num}/{config.num_rounds}: Acc={acc:.2f}%, Loss={loss:.4f}, "
                  f"Best={best_acc:.2f}%, Time={round_time:.1f}s")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Final Accuracy: {results['accuracy'][-1]:.2f}%")
    print(f"Total Time: {total_time/60:.1f} minutes")
    print(f"{'='*70}\n")
    
    results['best_accuracy'] = best_acc
    results['total_time'] = total_time
    
    return results


def save_results(results, experiment_name, results_dir='./results'):
    """Save experiment results"""
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON
    json_path = os.path.join(results_dir, f'{experiment_name}_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump({
            'config': results['config'],
            'best_accuracy': results['best_accuracy'],
            'final_accuracy': results['accuracy'][-1],
            'total_time': results['total_time'],
            'accuracy_history': results['accuracy'],
            'loss_history': results['loss']
        }, f, indent=2)
    
    # Save plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['accuracy'])
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Test Accuracy - {experiment_name}')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(results['loss'])
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title(f'Test Loss - {experiment_name}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{experiment_name}_{timestamp}.png'), dpi=150)
    plt.close()
    
    print(f"Results saved to {results_dir}/")
    return json_path


def main():
    """Run full experiments according to paper specifications"""
    
    print("="*70)
    print("FedPLC Full Experiment Suite")
    print("Replicating Paper: 'Personalized Federated Learning with")
    print("Label-wise Clustering and Adaptive Aggregation'")
    print("="*70)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("\nWARNING: No GPU detected! Training will be very slow.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    else:
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Experiment configurations following paper
    experiments = [
        # Main experiment: CIFAR-10 without drift
        {
            'name': 'cifar10_no_drift',
            'dataset': 'cifar10',
            'drift_type': 'none',
            'num_rounds': 200,
            'num_clients': 100,
        },
        # CIFAR-10 with Abrupt Drift at round 100
        {
            'name': 'cifar10_abrupt_drift',
            'dataset': 'cifar10',
            'drift_type': 'abrupt',
            'num_rounds': 200,
            'num_clients': 100,
        },
        # CIFAR-10 with Incremental Drift
        {
            'name': 'cifar10_incremental_drift',
            'dataset': 'cifar10',
            'drift_type': 'incremental',
            'num_rounds': 200,
            'num_clients': 100,
        },
    ]
    
    all_results = {}
    
    for exp in experiments:
        print(f"\n{'#'*70}")
        print(f"# Experiment: {exp['name']}")
        print(f"{'#'*70}")
        
        config = ExperimentConfig(
            dataset=exp['dataset'],
            num_clients=exp['num_clients'],
            num_rounds=exp['num_rounds'],
            local_epochs=5,
            batch_size=64,
            alpha=0.5,
            warmup_rounds=30,
            similarity_threshold=0.85,
            parl_weight=0.1,
            drift_round=100,
            participation_rate=0.1,
            seed=42
        )
        
        results = run_experiment(config, drift_type=exp['drift_type'])
        save_results(results, exp['name'])
        all_results[exp['name']] = results
        
        # Give GPU a break between experiments
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'Experiment':<30} {'Best Acc':<12} {'Final Acc':<12} {'Time':<10}")
    print("-"*70)
    for name, res in all_results.items():
        print(f"{name:<30} {res['best_accuracy']:.2f}%{'':<6} "
              f"{res['accuracy'][-1]:.2f}%{'':<6} {res['total_time']/60:.1f} min")
    print("="*70)


if __name__ == '__main__':
    main()
