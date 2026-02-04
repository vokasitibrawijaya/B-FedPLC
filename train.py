"""
Main training script for FedPLC replication
"""

import os
import sys
import argparse
import time
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedplc.config import ExperimentConfig
from fedplc.data import (
    create_federated_dataloaders, 
    ConceptDriftSimulator,
    NonIIDPartitioner
)
from fedplc.models import create_model
from fedplc.server import FedPLCServer, FedPLCClient
from fedplc.utils import set_seed, setup_logging, save_results, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='FedPLC Replication')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'fmnist', 'svhn'],
                        help='Dataset to use')
    
    # Federated Learning
    parser.add_argument('--num_clients', type=int, default=100,
                        help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=200,
                        help='Number of communication rounds')
    parser.add_argument('--local_epochs', type=int, default=5,
                        help='Number of local training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--participation_rate', type=float, default=0.1,
                        help='Client participation rate per round')
    
    # Non-IID
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet concentration parameter')
    
    # PARL
    parser.add_argument('--parl_weight', type=float, default=0.1,
                        help='Weight for PARL alignment loss')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for InfoNCE')
    parser.add_argument('--warmup_rounds', type=int, default=30,
                        help='Warmup rounds before LDCA')
    
    # LDCA
    parser.add_argument('--similarity_threshold', type=float, default=0.85,
                        help='Similarity threshold for community detection')
    parser.add_argument('--resolution', type=float, default=1.0,
                        help='Louvain resolution parameter')
    
    # Concept Drift
    parser.add_argument('--drift_type', type=str, default='none',
                        choices=['none', 'abrupt', 'incremental'],
                        help='Type of concept drift')
    parser.add_argument('--drift_round', type=int, default=100,
                        help='Round when drift occurs')
    
    # Training
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # System
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Results directory')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Checkpoint save interval')
    
    return parser.parse_args()


def create_config(args) -> ExperimentConfig:
    """Create config from command line arguments"""
    return ExperimentConfig(
        dataset=args.dataset,
        num_classes=10,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        participation_rate=args.participation_rate,
        alpha=args.alpha,
        hidden_dim=512 if args.dataset in ['cifar10', 'svhn'] else 256,
        parl_weight=args.parl_weight,
        temperature=args.temperature,
        warmup_rounds=args.warmup_rounds,
        similarity_threshold=args.similarity_threshold,
        resolution=args.resolution,
        drift_type=args.drift_type,
        drift_round=args.drift_round,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed,
        results_dir=args.results_dir
    )


def run_experiment(config: ExperimentConfig):
    """
    Run FedPLC experiment
    
    This implements Algorithm 1 from the paper:
    1. Warmup phase with standard FL
    2. LDCA community detection after warmup
    3. Label-wise aggregation within communities
    """
    print("\n" + "="*70)
    print("FedPLC: Federated Learning with Prototype-anchored Learning")
    print("and Label-wise Dynamic Community Adaptation")
    print("="*70)
    
    # Set seed
    set_seed(config.seed)
    
    # Setup logging
    logger = setup_logging(os.path.join(config.results_dir, 'logs'))
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.results_dir, f"{config.dataset}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\nExperiment Configuration:")
    print(f"  Dataset: {config.dataset}")
    print(f"  Clients: {config.num_clients}")
    print(f"  Rounds: {config.num_rounds}")
    print(f"  Local epochs: {config.local_epochs}")
    print(f"  Alpha (Non-IID): {config.alpha}")
    print(f"  PARL weight: {config.parl_weight}")
    print(f"  Warmup rounds: {config.warmup_rounds}")
    print(f"  Similarity threshold: {config.similarity_threshold}")
    print(f"  Drift type: {config.drift_type}")
    
    # Create data loaders
    print("\n[1/4] Creating federated data loaders...")
    client_loaders, test_loader, partitioner = create_federated_dataloaders(
        dataset_name=config.dataset,
        num_clients=config.num_clients,
        alpha=config.alpha,
        batch_size=config.batch_size,
        data_dir='./data',
        seed=config.seed
    )
    
    # Setup concept drift simulator
    drift_simulator = None
    if config.drift_type != 'none':
        print(f"\n[1.5/4] Setting up {config.drift_type} drift simulator...")
        drift_simulator = ConceptDriftSimulator(
            partitioner=partitioner,
            drift_type=config.drift_type,
            seed=config.seed
        )
    
    # Create global model
    print("\n[2/4] Creating global model...")
    global_model = create_model(
        dataset=config.dataset,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        use_labelwise=True
    )
    
    # Initialize server
    print("\n[3/4] Initializing server...")
    server = FedPLCServer(
        config=config,
        model=global_model,
        test_loader=test_loader
    )
    
    # Initialize clients
    print("\n[4/4] Initializing clients...")
    clients = {}
    for client_id in tqdm(range(config.num_clients), desc="Creating clients"):
        clients[client_id] = FedPLCClient(
            client_id=client_id,
            config=config,
            dataloader=client_loaders[client_id],
            model=global_model
        )
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    results = {
        'config': vars(config),
        'rounds': [],
        'accuracy': [],
        'loss': [],
        'community_stats': [],
        'client_accuracies': []
    }
    
    best_accuracy = 0.0
    
    for round_idx in range(1, config.num_rounds + 1):
        round_start = time.time()
        
        # Check if drift should be applied
        if drift_simulator is not None:
            if config.drift_type == 'abrupt' and round_idx == config.drift_round:
                print(f"\n[Round {round_idx}] Applying ABRUPT drift...")
                affected_clients = list(range(config.num_clients // 2))
                drift_simulator.apply_abrupt_drift(affected_clients)
            
            elif config.drift_type == 'incremental':
                if round_idx in config.incremental_drift_rounds:
                    print(f"\n[Round {round_idx}] Applying INCREMENTAL drift...")
                    affected_clients = list(range(config.num_clients // 2))
                    drift_simulator.apply_incremental_drift(affected_clients)
        
        # Select clients for this round
        selected_clients = server.select_clients()
        
        # Get global model and prototypes
        global_state = server.get_global_model_state()
        global_prototypes = server.get_global_prototypes()
        
        # Client training
        client_updates = {}
        client_data_sizes = {}
        
        for client_id in selected_clients:
            # Send global model to client
            clients[client_id].receive_global_model(
                server.get_client_model(client_id)
            )
            
            # Local training
            updates = clients[client_id].train(
                global_prototypes=global_prototypes,
                warmup=server.is_warmup
            )
            
            client_updates[client_id] = updates
            client_data_sizes[client_id] = updates['data_size']
        
        # Server aggregation
        server.aggregate_round(client_updates, client_data_sizes)
        
        # Advance round
        server.step_round()
        
        # Evaluate
        accuracy, loss = server.evaluate()
        
        # Log results
        server.log_round(accuracy, loss)
        
        results['rounds'].append(round_idx)
        results['accuracy'].append(accuracy)
        results['loss'].append(loss)
        
        round_time = time.time() - round_start
        
        # Print progress
        warmup_str = "[WARMUP]" if server.is_warmup else "[LDCA]"
        print(f"Round {round_idx:3d}/{config.num_rounds} {warmup_str} | "
              f"Acc: {accuracy:.2f}% | Loss: {loss:.4f} | "
              f"Time: {round_time:.1f}s")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_checkpoint(
                {
                    'round': round_idx,
                    'model_state': server.global_model.state_dict(),
                    'accuracy': accuracy,
                    'loss': loss
                },
                run_dir,
                'best_model.pth'
            )
        
        # Periodic checkpoint
        if round_idx % config.save_interval == 0:
            server.save_state(
                os.path.join(run_dir, f'checkpoint_round_{round_idx}.pth')
            )
        
        # Print community statistics periodically
        if not server.is_warmup and round_idx % 20 == 0:
            server.ldca_manager.print_summary()
    
    # Final evaluation
    print("\n" + "="*70)
    print("Training Complete")
    print("="*70)
    
    final_accuracy, final_loss = server.evaluate()
    print(f"\nFinal Results:")
    print(f"  Best Accuracy: {best_accuracy:.2f}%")
    print(f"  Final Accuracy: {final_accuracy:.2f}%")
    print(f"  Final Loss: {final_loss:.4f}")
    
    # Save final results
    results['best_accuracy'] = best_accuracy
    results['final_accuracy'] = final_accuracy
    results['final_loss'] = final_loss
    
    save_results(results, run_dir, 'results.json')
    server.save_state(os.path.join(run_dir, 'final_state.pth'))
    
    print(f"\nResults saved to: {run_dir}")
    
    return results


def main():
    args = parse_args()
    config = create_config(args)
    
    # Run experiment
    results = run_experiment(config)
    
    return results


if __name__ == '__main__':
    main()
