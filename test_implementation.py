"""
Quick test script to verify FedPLC implementation
Runs a short experiment to check all components work correctly
"""

import os
import sys
import torch
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fedplc.config import ExperimentConfig
from fedplc.data import create_federated_dataloaders, ConceptDriftSimulator
from fedplc.models import create_model
from fedplc.server import FedPLCServer, FedPLCClient
from fedplc.utils import set_seed


def test_data_loading():
    """Test data loading and partitioning"""
    print("\n" + "="*50)
    print("Testing Data Loading")
    print("="*50)
    
    client_loaders, test_loader, partitioner = create_federated_dataloaders(
        dataset_name='cifar10',
        num_clients=10,
        alpha=0.5,
        batch_size=32,
        seed=42
    )
    
    print(f"✓ Created {len(client_loaders)} client loaders")
    print(f"✓ Test loader has {len(test_loader.dataset)} samples")
    
    # Check data distribution
    stats = partitioner.get_statistics()
    print(f"✓ Partitioning stats: {stats}")
    
    # Test a batch
    for client_id in [0, 1]:
        data, target = next(iter(client_loaders[client_id]))
        print(f"✓ Client {client_id}: batch shape {data.shape}, labels {target.unique().tolist()}")
    
    return True


def test_model_creation():
    """Test model creation"""
    print("\n" + "="*50)
    print("Testing Model Creation")
    print("="*50)
    
    # Test CIFAR-10 model
    model = create_model(
        dataset='cifar10',
        hidden_dim=512,
        num_classes=10,
        use_labelwise=True
    )
    
    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    logits, features = model(x)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Features shape: {features.shape}")
    print(f"✓ Logits shape: {logits.shape}")
    
    # Test label-wise classifier
    for label in range(3):
        weights = model.get_label_classifier_params(label)
        print(f"✓ Label {label} classifier has {len(weights)} parameters")
    
    return True


def test_parl_training():
    """Test PARL local training"""
    print("\n" + "="*50)
    print("Testing PARL Training")
    print("="*50)
    
    from fedplc.parl import LocalTrainer, DecoupledTrainer
    
    # Create minimal setup
    client_loaders, test_loader, _ = create_federated_dataloaders(
        dataset_name='cifar10',
        num_clients=5,
        alpha=0.5,
        batch_size=32,
        seed=42
    )
    
    model = create_model('cifar10', hidden_dim=256, num_classes=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainer = LocalTrainer(
        model=model,
        dataloader=client_loaders[0],
        device=device,
        local_epochs=1,
        parl_weight=0.1
    )
    
    # Train
    stats = trainer.train(warmup=False)
    
    print(f"✓ Training loss: {stats['loss']:.4f}")
    print(f"✓ CE loss: {stats['ce_loss']:.4f}")
    print(f"✓ Alignment loss: {stats['align_loss']:.4f}")
    print(f"✓ Accuracy: {stats['accuracy']:.2f}%")
    print(f"✓ Local prototypes shape: {stats['local_prototypes'].shape}")
    
    return True


def test_ldca():
    """Test LDCA community detection"""
    print("\n" + "="*50)
    print("Testing LDCA Community Detection")
    print("="*50)
    
    from fedplc.ldca import LDCAManager
    
    ldca = LDCAManager(
        num_clients=10,
        num_classes=5,
        threshold=0.8,
        resolution=1.0
    )
    
    # Create fake classifier weights
    client_clf_weights = {}
    for client_id in range(10):
        client_clf_weights[client_id] = {}
        for label in range(5):
            # Create similar weights for clients in same "group"
            group = client_id // 3
            base = torch.randn(64, 128) * 0.1
            noise = torch.randn(64, 128) * (0.01 if group == client_id // 3 else 0.5)
            
            client_clf_weights[client_id][label] = {
                '0.weight': base + noise,
                '0.bias': torch.randn(64)
            }
    
    # Update communities
    ldca.update_communities(client_clf_weights)
    
    # Print results
    ldca.print_summary()
    
    stats = ldca.get_statistics()
    print(f"\n✓ Community statistics computed for {len(stats)} labels")
    
    return True


def test_fedplc_round():
    """Test full FedPLC round"""
    print("\n" + "="*50)
    print("Testing Full FedPLC Round")
    print("="*50)
    
    # Minimal config
    config = ExperimentConfig(
        dataset='cifar10',
        num_clients=5,
        num_rounds=3,
        local_epochs=1,
        batch_size=32,
        participation_rate=0.6,
        warmup_rounds=1,
        hidden_dim=256,
        seed=42
    )
    
    set_seed(config.seed)
    
    # Create data
    client_loaders, test_loader, _ = create_federated_dataloaders(
        dataset_name=config.dataset,
        num_clients=config.num_clients,
        alpha=0.5,
        batch_size=config.batch_size,
        seed=config.seed
    )
    
    # Create model and server
    model = create_model(
        dataset=config.dataset,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes
    )
    
    server = FedPLCServer(config, model, test_loader)
    
    # Create clients
    clients = {
        i: FedPLCClient(i, config, client_loaders[i], model)
        for i in range(config.num_clients)
    }
    
    # Run rounds
    for round_idx in range(1, config.num_rounds + 1):
        print(f"\n--- Round {round_idx} ---")
        
        # Select clients
        selected = server.select_clients()
        print(f"Selected clients: {selected}")
        
        # Training
        client_updates = {}
        client_data_sizes = {}
        
        for client_id in selected:
            clients[client_id].receive_global_model(
                server.get_client_model(client_id)
            )
            updates = clients[client_id].train(
                global_prototypes=server.get_global_prototypes(),
                warmup=server.is_warmup
            )
            client_updates[client_id] = updates
            client_data_sizes[client_id] = updates['data_size']
        
        # Aggregate
        server.aggregate_round(client_updates, client_data_sizes)
        server.step_round()
        
        # Evaluate
        acc, loss = server.evaluate()
        print(f"✓ Accuracy: {acc:.2f}%, Loss: {loss:.4f}")
        print(f"✓ Warmup: {server.is_warmup}")
    
    return True


def test_concept_drift():
    """Test concept drift simulation"""
    print("\n" + "="*50)
    print("Testing Concept Drift Simulation")
    print("="*50)
    
    from fedplc.data import NonIIDPartitioner, get_dataset
    
    dataset = get_dataset('cifar10', train=True)
    partitioner = NonIIDPartitioner(dataset, num_clients=10, alpha=0.5)
    
    drift_sim = ConceptDriftSimulator(partitioner, drift_type='abrupt')
    
    # Check original labels
    client_0_labels = partitioner.targets[partitioner.client_indices[0][:10]]
    print(f"Original labels (client 0): {client_0_labels}")
    
    # Apply drift
    drift_sim.apply_abrupt_drift(client_ids=[0, 1, 2])
    
    # Check after drift
    new_labels = partitioner.targets[partitioner.client_indices[0][:10]]
    print(f"After drift (client 0): {new_labels}")
    
    status = drift_sim.get_drift_status()
    print(f"✓ Drift status: {status}")
    
    # Reset
    drift_sim.reset_drift()
    reset_labels = partitioner.targets[partitioner.client_indices[0][:10]]
    print(f"After reset (client 0): {reset_labels}")
    
    return True


def main():
    print("\n" + "="*70)
    print("FedPLC Implementation Test Suite")
    print("="*70)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("PARL Training", test_parl_training),
        ("LDCA Community Detection", test_ldca),
        ("Full FedPLC Round", test_fedplc_round),
        ("Concept Drift", test_concept_drift),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = sum(1 for _, s, _ in results if s)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
