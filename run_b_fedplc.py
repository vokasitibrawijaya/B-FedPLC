"""
B-FedPLC Training Script
Runs the complete B-FedPLC experiment with Blockchain + IPFS integration

This script demonstrates:
1. FedPLC training with PARL + LDCA
2. Blockchain audit trail for all model updates
3. IPFS decentralized model storage
4. Smart contract aggregation rules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
import copy

# Import B-FedPLC
from b_fedplc import BFedPLC, CryptoUtils


# ============================================================================
# Model Definition
# ============================================================================
class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================================
# Data Partitioning
# ============================================================================
def dirichlet_partition(dataset, num_clients, alpha=0.5, num_classes=10):
    """Partition dataset using Dirichlet distribution for Non-IID"""
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    client_indices = [[] for _ in range(num_clients)]
    label_distributions = []
    
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        
        class_splits = np.split(class_indices, proportions)
        
        for client_id, indices in enumerate(class_splits):
            client_indices[client_id].extend(indices)
    
    # Compute label distributions
    for client_id in range(num_clients):
        indices = client_indices[client_id]
        client_labels = labels[indices]
        dist = np.bincount(client_labels, minlength=num_classes).astype(float)
        dist = dist / (dist.sum() + 1e-8)
        label_distributions.append(dist)
    
    return client_indices, label_distributions


# ============================================================================
# PARL Loss
# ============================================================================
class PARLLoss(nn.Module):
    """Prototype-Anchored Representation Learning Loss"""
    def __init__(self, num_classes=10, feature_dim=256, lambda_parl=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_parl = lambda_parl
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.prototypes)
    
    def forward(self, logits, labels, features=None):
        ce_loss = F.cross_entropy(logits, labels)
        
        if features is not None and self.lambda_parl > 0:
            batch_protos = self.prototypes[labels]
            parl_loss = F.mse_loss(features, batch_protos)
            return ce_loss + self.lambda_parl * parl_loss
        
        return ce_loss


# ============================================================================
# Client Training
# ============================================================================
def train_client(model, train_loader, criterion, optimizer, device, epochs=1):
    """Train a client model"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels, None)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    
    return accuracy, avg_loss


def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


# ============================================================================
# Community-Aware Aggregation
# ============================================================================
def community_aggregate(global_model, client_models, communities, client_ids_map, 
                       weights, bfedplc):
    """
    Aggregate models with community awareness
    Uses smart contract weights and community structure
    """
    global_state = global_model.state_dict()
    
    # Get aggregation weights from smart contract
    agg_weights = bfedplc.contract.compute_weights(
        bfedplc.blockchain.pending_updates
    )
    
    # Aggregate
    for key in global_state.keys():
        if 'num_batches_tracked' in key:
            continue
            
        weighted_sum = torch.zeros_like(global_state[key], dtype=torch.float32)
        total_weight = 0
        
        for client_idx, client_model in client_models.items():
            client_id = bfedplc.client_ids[client_idx]
            w = agg_weights.get(client_id, weights[client_idx])
            weighted_sum += w * client_model.state_dict()[key].float()
            total_weight += w
        
        if total_weight > 0:
            global_state[key] = (weighted_sum / total_weight).to(global_state[key].dtype)
    
    global_model.load_state_dict(global_state)
    return global_model


# ============================================================================
# Main B-FedPLC Training
# ============================================================================
def run_b_fedplc(config):
    """Run B-FedPLC experiment"""
    
    print("="*70)
    print("B-FedPLC: Blockchain-Enabled Federated Learning")
    print("with Prototype-Anchored Learning & Dynamic Community Adaptation")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Clients: {config['num_clients']}")
    print(f"  Rounds: {config['num_rounds']}")
    print(f"  PARL Weight: {config['parl_weight']}")
    print(f"  LDCA Threshold: {config['similarity_threshold']}")
    print(f"  Blockchain Difficulty: {config['blockchain_difficulty']}")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Load CIFAR-10
    print("\nLoading CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Partition data
    print(f"\nPartitioning data (Dirichlet alpha={config['dirichlet_alpha']})...")
    client_indices, label_distributions = dirichlet_partition(
        train_dataset, config['num_clients'], config['dirichlet_alpha']
    )
    
    # Initialize B-FedPLC
    print("\nInitializing B-FedPLC...")
    bfedplc = BFedPLC(
        num_clients=config['num_clients'],
        num_classes=10,
        parl_weight=config['parl_weight'],
        similarity_threshold=config['similarity_threshold'],
        warmup_rounds=config['warmup_rounds'],
        blockchain_difficulty=config['blockchain_difficulty']
    )
    
    # Register label distributions
    for client_idx, dist in enumerate(label_distributions):
        bfedplc.update_label_distribution(client_idx, dist)
    
    # Initialize global model
    global_model = SimpleCNN(num_classes=10).to(device)
    print(f"\nModel: SimpleCNN ({sum(p.numel() for p in global_model.parameters()):,} params)")
    
    # Training history
    history = {
        'rounds': [],
        'accuracy': [],
        'loss': [],
        'communities': [],
        'blocks': [],
        'ipfs_size': [],
        'blockchain_verified': []
    }
    
    best_accuracy = 0
    start_time = time.time()
    
    # Training loop
    print("\n" + "="*70)
    print("Starting B-FedPLC Training")
    print("="*70 + "\n")
    
    for round_num in range(1, config['num_rounds'] + 1):
        round_start = time.time()
        
        # Select clients
        num_selected = max(1, int(config['num_clients'] * config['client_fraction']))
        selected_clients = np.random.choice(
            config['num_clients'], num_selected, replace=False
        )
        
        # Compute communities (LDCA)
        if round_num >= config['warmup_rounds']:
            bfedplc.compute_communities()
        
        # Client training
        client_models = {}
        client_weights = {}
        total_samples = sum(len(client_indices[c]) for c in selected_clients)
        
        for client_idx in selected_clients:
            # Create client model
            client_model = copy.deepcopy(global_model)
            client_model.to(device)
            
            # Client data
            indices = client_indices[client_idx]
            subset = Subset(train_dataset, indices)
            loader = DataLoader(subset, batch_size=config['batch_size'], shuffle=True)
            
            # PARL loss
            criterion = PARLLoss(
                num_classes=10, 
                feature_dim=256, 
                lambda_parl=config['parl_weight']
            ).to(device)
            
            optimizer = torch.optim.SGD(
                client_model.parameters(),
                lr=config['learning_rate'],
                momentum=0.9,
                weight_decay=1e-4
            )
            
            # Train
            acc, loss = train_client(
                client_model, loader, criterion, optimizer, 
                device, config['local_epochs']
            )
            
            # Submit update to blockchain
            success, msg = bfedplc.submit_update(
                client_idx=client_idx,
                model_state=client_model.state_dict(),
                accuracy=acc,
                loss=loss,
                data_size=len(indices),
                round_number=round_num
            )
            
            if success:
                client_models[client_idx] = client_model
                client_weights[client_idx] = len(indices) / total_samples
        
        # Aggregate with community awareness
        global_model = community_aggregate(
            global_model, client_models, bfedplc.communities,
            bfedplc.client_ids, client_weights, bfedplc
        )
        
        # Evaluate
        accuracy = evaluate(global_model, test_loader, device)
        
        # Finalize round -> Create blockchain block
        block = bfedplc.finalize_round(round_num, global_model.state_dict(), accuracy)
        
        # Verify blockchain integrity
        verified, verify_msg = bfedplc.verify_integrity()
        
        # Update best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(global_model.state_dict(), 'b_fedplc_best_model.pt')
        
        # Record history
        stats = bfedplc.get_statistics()
        history['rounds'].append(round_num)
        history['accuracy'].append(accuracy)
        history['loss'].append(np.mean([u.loss for u in block.updates]) if block.updates else 0)
        history['communities'].append(len(bfedplc.communities))
        history['blocks'].append(block.to_dict())
        history['ipfs_size'].append(stats['blockchain']['ipfs_stats']['total_size_mb'])
        history['blockchain_verified'].append(verified)
        
        round_time = time.time() - round_start
        
        # Print progress
        print(f"Round {round_num:3d}/{config['num_rounds']} | "
              f"Acc: {accuracy:.2f}% | "
              f"Best: {best_accuracy:.2f}% | "
              f"Comm: {len(bfedplc.communities):2d} | "
              f"Block #{block.index} | "
              f"IPFS: {stats['blockchain']['ipfs_stats']['total_size_mb']:.1f}MB | "
              f"Chain: {'OK' if verified else 'ERR'} | "
              f"Time: {round_time:.1f}s")
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("B-FedPLC Training Complete!")
    print("="*70)
    print(f"\nResults:")
    print(f"  Best Accuracy: {best_accuracy:.2f}%")
    print(f"  Final Accuracy: {accuracy:.2f}%")
    print(f"  Training Time: {total_time/60:.1f} minutes")
    
    final_stats = bfedplc.get_statistics()
    print(f"\nBlockchain Statistics:")
    print(f"  Chain Length: {final_stats['blockchain']['chain_length']} blocks")
    print(f"  Total Updates: {final_stats['blockchain']['total_updates']}")
    print(f"  IPFS Storage: {final_stats['blockchain']['ipfs_stats']['total_size_mb']:.2f} MB")
    print(f"  Chain Verified: {bfedplc.verify_integrity()[0]}")
    
    print(f"\nCommunity Statistics:")
    print(f"  Final Communities: {final_stats['communities']['count']}")
    print(f"  Community Sizes: {final_stats['communities']['sizes']}")
    
    # Export blockchain
    bfedplc.export_blockchain('b_fedplc_blockchain.json')
    print(f"\nBlockchain exported to: b_fedplc_blockchain.json")
    
    # Save results
    results = {
        'config': config,
        'best_accuracy': best_accuracy,
        'final_accuracy': accuracy,
        'training_time_minutes': total_time / 60,
        'history': {
            'rounds': history['rounds'],
            'accuracy': history['accuracy'],
            'loss': history['loss'],
            'communities': history['communities'],
            'ipfs_size': history['ipfs_size']
        },
        'blockchain_stats': final_stats['blockchain'],
        'community_stats': final_stats['communities'],
        'timestamp': datetime.now().isoformat()
    }
    
    with open('b_fedplc_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: b_fedplc_results.json")
    
    return results, bfedplc


# ============================================================================
# Configuration and Entry Point
# ============================================================================
if __name__ == "__main__":
    config = {
        # FL Configuration
        'num_clients': 50,
        'num_rounds': 100,
        'client_fraction': 0.2,
        'local_epochs': 3,
        'batch_size': 32,
        'learning_rate': 0.01,
        
        # Data Configuration
        'dirichlet_alpha': 0.5,
        
        # PARL Configuration
        'parl_weight': 0.1,
        
        # LDCA Configuration  
        'similarity_threshold': 0.85,
        'warmup_rounds': 10,
        
        # Blockchain Configuration
        'blockchain_difficulty': 2
    }
    
    results, bfedplc = run_b_fedplc(config)
    
    print("\n" + "="*70)
    print("B-FedPLC experiment completed successfully!")
    print("="*70)
