"""
Quick start script for FedPLC training
Runs a short experiment to demonstrate the implementation
"""

import os
import sys

# Fix for Windows multiprocessing
if __name__ == '__main__':
    import sys
    import warnings
    warnings.filterwarnings('ignore')
    
    import torch
    import numpy as np
    from tqdm import tqdm
    
    # Force stdout flush
    sys.stdout.reconfigure(line_buffering=True)
    
    # Add current directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from fedplc.config import ExperimentConfig
    from fedplc.data import create_federated_dataloaders
    from fedplc.models import create_model
    from fedplc.server import FedPLCServer, FedPLCClient
    from fedplc.utils import set_seed
    
    print("="*70)
    print("FedPLC Quick Start - Short Experiment")
    print("="*70)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Quick experiment config - very small for fast testing
    config = ExperimentConfig(
        dataset='cifar10',
        num_clients=10,           # Reduced for quick test
        num_rounds=15,            # Short run
        local_epochs=1,           # Fewer epochs
        batch_size=32,
        participation_rate=0.3,   # 30% participation
        warmup_rounds=5,
        hidden_dim=128,           # Smaller model
        alpha=0.5,
        parl_weight=0.1,
        similarity_threshold=0.85,
        seed=42
    )
    
    set_seed(config.seed)
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {config.dataset}")
    print(f"  Clients: {config.num_clients}")
    print(f"  Rounds: {config.num_rounds}")
    print(f"  Warmup: {config.warmup_rounds}")
    print(f"  Alpha (Non-IID): {config.alpha}")
    
    # Create data loaders
    print("\n[1/3] Loading data...")
    client_loaders, test_loader, partitioner = create_federated_dataloaders(
        dataset_name=config.dataset,
        num_clients=config.num_clients,
        alpha=config.alpha,
        batch_size=config.batch_size,
        data_dir='./data',
        seed=config.seed
    )
    
    # Create model
    print("\n[2/3] Creating model...")
    global_model = create_model(
        dataset=config.dataset,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        use_labelwise=True
    )
    
    # Initialize server
    print("\n[3/3] Initializing server and clients...")
    server = FedPLCServer(config, global_model, test_loader)
    
    # Create clients
    clients = {}
    for client_id in range(config.num_clients):
        clients[client_id] = FedPLCClient(
            client_id=client_id,
            config=config,
            dataloader=client_loaders[client_id],
            model=global_model
        )
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70 + "\n")
    
    best_acc = 0.0
    
    for round_idx in range(1, config.num_rounds + 1):
        # Select clients
        selected = server.select_clients()
        
        # Get global state
        global_prototypes = server.get_global_prototypes()
        
        # Client training
        client_updates = {}
        client_data_sizes = {}
        
        for client_id in selected:
            clients[client_id].receive_global_model(
                server.get_client_model(client_id)
            )
            updates = clients[client_id].train(
                global_prototypes=global_prototypes,
                warmup=server.is_warmup
            )
            client_updates[client_id] = updates
            client_data_sizes[client_id] = updates['data_size']
        
        # Aggregate
        server.aggregate_round(client_updates, client_data_sizes)
        server.step_round()
        
        # Evaluate
        acc, loss = server.evaluate()
        server.log_round(acc, loss)
        
        if acc > best_acc:
            best_acc = acc
        
        # Print progress
        phase = "WARMUP" if server.is_warmup else "LDCA"
        print(f"Round {round_idx:2d}/{config.num_rounds} [{phase:6s}] | "
              f"Acc: {acc:5.2f}% | Loss: {loss:.4f} | Best: {best_acc:.2f}%")
    
    # Summary
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nFinal Accuracy: {acc:.2f}%")
    print(f"Best Accuracy:  {best_acc:.2f}%")
    
    # Show community stats if LDCA was used
    if not server.is_warmup:
        print("\nCommunity Statistics:")
        server.ldca_manager.print_summary()
