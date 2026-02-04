"""
FedPLC Concept Drift Experiment
Tests LDCA's ability to adapt to dynamic data distribution changes

Concept Drift Scenarios:
1. Label Swap: Swap labels for subset of clients mid-training
2. Client Addition: Add new clients with different distributions
3. Distribution Shift: Gradually change Non-IID alpha
4. Sudden Drift: Abrupt change in client data distributions

Author: FedPLC Replication Study
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy
import time
import json
from collections import defaultdict
from pathlib import Path

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*70)
print("FedPLC Concept Drift Experiment")
print("="*70)
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Base settings
    num_clients = 50
    num_rounds = 100
    local_epochs = 3
    batch_size = 64
    lr = 0.01
    participation_rate = 0.2
    
    # Non-IID settings
    alpha_initial = 0.5  # Initial Dirichlet alpha
    
    # PARL settings
    parl_weight = 0.1
    temperature = 0.07
    warmup_rounds = 15
    
    # LDCA settings  
    similarity_threshold = 0.85
    ldca_update_interval = 5  # Recompute communities every N rounds
    
    # Concept drift settings
    drift_round = 50  # When drift occurs
    drift_type = "label_swap"  # Options: label_swap, distribution_shift, client_addition
    drift_intensity = 0.3  # Fraction of clients affected


config = Config()


# ============================================================================
# Simple CNN Model (for faster training)
# ============================================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.projector = nn.Linear(128 * 4 * 4, 128)
    
    def forward(self, x):
        features = self.features(x)
        flat = features.view(features.size(0), -1)
        return self.classifier(flat)
    
    def get_features(self, x):
        features = self.features(x)
        flat = features.view(features.size(0), -1)
        return self.projector(flat)


# ============================================================================
# Data Partitioning with Concept Drift Support
# ============================================================================
class ConceptDriftDataManager:
    def __init__(self, num_clients, alpha):
        self.num_clients = num_clients
        self.alpha = alpha
        self.client_indices = {}
        self.client_label_maps = {}  # For label swapping
        self.original_indices = {}
        
        # Load CIFAR-10
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        self.test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        
        self.test_loader = DataLoader(self.test_dataset, batch_size=100, shuffle=False)
        
        # Initial partitioning
        self._partition_data(alpha)
        
        # Initialize identity label maps
        for i in range(num_clients):
            self.client_label_maps[i] = {j: j for j in range(10)}
    
    def _partition_data(self, alpha):
        """Partition data using Dirichlet distribution"""
        targets = np.array(self.train_dataset.targets)
        num_classes = 10
        
        # Dirichlet distribution for each class
        class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
        
        self.client_indices = {i: [] for i in range(self.num_clients)}
        
        for c in range(num_classes):
            np.random.shuffle(class_indices[c])
            proportions = np.random.dirichlet([alpha] * self.num_clients)
            proportions = (proportions * len(class_indices[c])).astype(int)
            
            # Adjust for rounding
            diff = len(class_indices[c]) - proportions.sum()
            proportions[0] += diff
            
            start = 0
            for i in range(self.num_clients):
                self.client_indices[i].extend(
                    class_indices[c][start:start + proportions[i]].tolist()
                )
                start += proportions[i]
        
        # Store original for reference
        self.original_indices = copy.deepcopy(self.client_indices)
    
    def apply_label_swap_drift(self, affected_clients, swap_pairs=None):
        """
        Apply label swap drift to specified clients
        swap_pairs: list of (old_label, new_label) tuples
        """
        if swap_pairs is None:
            # Default: swap 0↔1, 2↔3, etc.
            swap_pairs = [(0, 1), (2, 3), (4, 5)]
        
        for client_id in affected_clients:
            label_map = {j: j for j in range(10)}
            for old, new in swap_pairs:
                label_map[old] = new
                label_map[new] = old
            self.client_label_maps[client_id] = label_map
        
        return len(affected_clients), swap_pairs
    
    def apply_distribution_shift(self, affected_clients, new_alpha=0.1):
        """
        Re-partition data for affected clients with different alpha
        (More extreme Non-IID)
        """
        targets = np.array(self.train_dataset.targets)
        num_classes = 10
        
        # Get indices for affected clients
        class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
        
        for client_id in affected_clients:
            # Clear existing indices
            self.client_indices[client_id] = []
            
            # Assign data with more extreme Non-IID (smaller alpha)
            dominant_classes = np.random.choice(num_classes, 2, replace=False)
            
            for c in range(num_classes):
                available = [idx for idx in class_indices[c] 
                           if idx not in sum(self.client_indices.values(), [])]
                
                if c in dominant_classes:
                    # Take more samples from dominant classes
                    n_samples = min(len(available), 400)
                else:
                    # Take fewer from others
                    n_samples = min(len(available), 50)
                
                if n_samples > 0:
                    selected = np.random.choice(available, n_samples, replace=False)
                    self.client_indices[client_id].extend(selected.tolist())
        
        return len(affected_clients), new_alpha
    
    def get_client_loader(self, client_id):
        """Get DataLoader with potentially modified labels"""
        indices = self.client_indices[client_id]
        label_map = self.client_label_maps[client_id]
        
        # Create subset with label mapping
        class LabelMappedDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, indices, label_map):
                self.dataset = dataset
                self.indices = indices
                self.label_map = label_map
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                img, label = self.dataset[self.indices[idx]]
                return img, self.label_map[label]
        
        mapped_dataset = LabelMappedDataset(self.train_dataset, indices, label_map)
        return DataLoader(mapped_dataset, batch_size=config.batch_size, 
                         shuffle=True, drop_last=len(indices) > config.batch_size)
    
    def get_client_label_distribution(self, client_id):
        """Get label distribution for a client"""
        indices = self.client_indices[client_id]
        label_map = self.client_label_maps[client_id]
        
        dist = np.zeros(10)
        for idx in indices:
            original_label = self.train_dataset.targets[idx]
            mapped_label = label_map[original_label]
            dist[mapped_label] += 1
        
        return dist / (dist.sum() + 1e-8)


# ============================================================================
# LDCA - Label-wise Dynamic Community Adaptation
# ============================================================================
class LDCA:
    def __init__(self, num_clients, num_classes=10, threshold=0.85):
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.threshold = threshold
        self.label_distributions = {}
        self.communities = {0: list(range(num_clients))}
        self.community_history = []
    
    def update_distribution(self, client_id, distribution):
        self.label_distributions[client_id] = distribution
    
    def compute_communities(self):
        """Compute communities based on label distribution similarity"""
        if len(self.label_distributions) < 2:
            return self.communities
        
        # Compute similarity matrix
        clients = list(self.label_distributions.keys())
        n = len(clients)
        similarity = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                dist_i = self.label_distributions[clients[i]]
                dist_j = self.label_distributions[clients[j]]
                # Cosine similarity
                similarity[i, j] = np.dot(dist_i, dist_j) / (
                    np.linalg.norm(dist_i) * np.linalg.norm(dist_j) + 1e-8)
        
        # Hierarchical clustering
        visited = [False] * n
        communities = {}
        comm_id = 0
        
        for i in range(n):
            if visited[i]:
                continue
            
            # Start new community
            community = [clients[i]]
            visited[i] = True
            
            # Add similar clients
            for j in range(n):
                if not visited[j] and similarity[i, j] >= self.threshold:
                    community.append(clients[j])
                    visited[j] = True
            
            communities[comm_id] = community
            comm_id += 1
        
        self.communities = communities
        self.community_history.append(len(communities))
        return communities
    
    def get_community_prototypes(self, client_id, prototypes_dict):
        """Get community prototypes for a client"""
        for comm_id, members in self.communities.items():
            if client_id in members:
                # Average prototypes from community members
                comm_prototypes = []
                for member in members:
                    if member in prototypes_dict and member != client_id:
                        comm_prototypes.append(prototypes_dict[member])
                
                if comm_prototypes:
                    return torch.stack(comm_prototypes).mean(dim=0)
        return None


# ============================================================================
# Training Functions
# ============================================================================
def train_client(model, data_loader, prototypes, community_prototypes, 
                use_parl=False, epochs=3, device='cuda'):
    """Train a client with optional PARL loss"""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, 
                                momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_parl = 0
    parl_count = 0
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            ce_loss = criterion(output, target)
            
            # PARL loss
            parl_loss = torch.tensor(0.0, device=device)
            if use_parl and community_prototypes is not None:
                features = model.get_features(data)
                
                # Compute class prototypes
                for c in range(10):
                    mask = (target == c)
                    if mask.sum() > 0:
                        class_features = features[mask].mean(dim=0)
                        if c < community_prototypes.size(0):
                            comm_proto = community_prototypes[c]
                            sim = F.cosine_similarity(
                                class_features.unsqueeze(0),
                                comm_proto.unsqueeze(0)
                            )
                            parl_loss = parl_loss + (1 - sim.squeeze())
                            parl_count += 1
                
                if parl_count > 0:
                    parl_loss = parl_loss / parl_count
            
            loss = ce_loss + config.parl_weight * parl_loss
            loss.backward()
            optimizer.step()
            
            total_loss += ce_loss.item()
            total_parl += parl_loss.item() if isinstance(parl_loss, torch.Tensor) else parl_loss
    
    n_batches = len(data_loader) * epochs
    return total_loss / max(n_batches, 1), total_parl / max(n_batches, 1)


def compute_prototypes(model, data_loader, device='cuda'):
    """Compute class prototypes from client data"""
    model.eval()
    class_features = defaultdict(list)
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            features = model.get_features(data)
            
            for c in range(10):
                mask = (target == c)
                if mask.sum() > 0:
                    class_features[c].append(features[mask])
    
    prototypes = torch.zeros(10, 128, device=device)
    for c in range(10):
        if class_features[c]:
            prototypes[c] = torch.cat(class_features[c], dim=0).mean(dim=0)
    
    return prototypes


def evaluate(model, test_loader, device='cuda'):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return 100.0 * correct / total


def federated_averaging(global_model, client_models, client_sizes):
    """FedAvg aggregation"""
    total_size = sum(client_sizes)
    
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
        for i, model in enumerate(client_models):
            weight = client_sizes[i] / total_size
            global_dict[key] += weight * model.state_dict()[key].float()
    
    global_model.load_state_dict(global_dict)
    return global_model


# ============================================================================
# Main Experiment
# ============================================================================
def run_concept_drift_experiment(drift_type="label_swap"):
    """Run concept drift experiment"""
    
    print(f"\n{'='*70}")
    print(f"Concept Drift Experiment: {drift_type.upper()}")
    print(f"{'='*70}")
    
    # Initialize
    print("\nInitializing...")
    data_manager = ConceptDriftDataManager(config.num_clients, config.alpha_initial)
    
    print(f"Config: {config.num_clients} clients, {config.num_rounds} rounds")
    print(f"Drift at round {config.drift_round}, intensity: {config.drift_intensity}")
    
    # Model
    global_model = SimpleCNN(num_classes=10).to(device)
    ldca = LDCA(config.num_clients, threshold=config.similarity_threshold)
    
    # History
    history = {
        'accuracy': [],
        'loss': [],
        'parl_loss': [],
        'communities': [],
        'drift_round': config.drift_round,
        'drift_type': drift_type
    }
    
    best_acc = 0
    client_prototypes = {}
    drift_applied = False
    
    print(f"\nInitial accuracy: {evaluate(global_model, data_manager.test_loader, device):.2f}%")
    
    print(f"\n{'='*70}")
    print("Starting Training with Concept Drift")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    for round_idx in range(1, config.num_rounds + 1):
        round_start = time.time()
        
        # =====================================================================
        # Apply Concept Drift at specified round
        # =====================================================================
        if round_idx == config.drift_round and not drift_applied:
            print(f"\n{'!'*70}")
            print(f"[DRIFT] Applying {drift_type} drift at round {round_idx}")
            
            num_affected = int(config.num_clients * config.drift_intensity)
            affected_clients = np.random.choice(
                config.num_clients, num_affected, replace=False)
            
            if drift_type == "label_swap":
                n, pairs = data_manager.apply_label_swap_drift(
                    affected_clients, swap_pairs=[(0, 1), (2, 3), (4, 5)])
                print(f"[DRIFT] Swapped labels for {n} clients: {pairs}")
            
            elif drift_type == "distribution_shift":
                n, new_alpha = data_manager.apply_distribution_shift(
                    affected_clients, new_alpha=0.1)
                print(f"[DRIFT] Shifted distribution for {n} clients to alpha={new_alpha}")
            
            drift_applied = True
            print(f"[DRIFT] Affected clients: {sorted(affected_clients)[:10]}...")
            print(f"{'!'*70}\n")
        
        # =====================================================================
        # Client Selection
        # =====================================================================
        num_selected = max(1, int(config.num_clients * config.participation_rate))
        selected_clients = np.random.choice(
            config.num_clients, num_selected, replace=False)
        
        # =====================================================================
        # Determine Phase
        # =====================================================================
        use_parl = round_idx > config.warmup_rounds
        
        # Update LDCA communities periodically after warmup
        if use_parl and (round_idx - config.warmup_rounds) % config.ldca_update_interval == 1:
            # Update label distributions
            for client_id in range(config.num_clients):
                dist = data_manager.get_client_label_distribution(client_id)
                ldca.update_distribution(client_id, dist)
            
            old_comm = len(ldca.communities)
            ldca.compute_communities()
            new_comm = len(ldca.communities)
            
            if old_comm != new_comm:
                print(f"\n[LDCA] Communities updated: {old_comm} -> {new_comm}")
        
        # =====================================================================
        # Local Training
        # =====================================================================
        client_models = []
        client_sizes = []
        total_loss = 0
        total_parl = 0
        
        for client_id in selected_clients:
            # Copy global model
            client_model = copy.deepcopy(global_model)
            
            # Get data loader
            data_loader = data_manager.get_client_loader(client_id)
            if len(data_loader) == 0:
                continue
            
            # Get community prototypes if using PARL
            comm_prototypes = None
            if use_parl and client_id in client_prototypes:
                comm_prototypes = ldca.get_community_prototypes(
                    client_id, client_prototypes)
            
            # Train
            loss, parl_loss = train_client(
                client_model, data_loader,
                client_prototypes.get(client_id),
                comm_prototypes,
                use_parl=use_parl,
                epochs=config.local_epochs,
                device=device
            )
            
            # Update prototypes
            client_prototypes[client_id] = compute_prototypes(
                client_model, data_loader, device)
            
            client_models.append(client_model)
            client_sizes.append(len(data_manager.client_indices[client_id]))
            total_loss += loss
            total_parl += parl_loss
        
        if not client_models:
            continue
        
        # =====================================================================
        # Aggregation
        # =====================================================================
        global_model = federated_averaging(global_model, client_models, client_sizes)
        
        # =====================================================================
        # Evaluation
        # =====================================================================
        acc = evaluate(global_model, data_manager.test_loader, device)
        
        if acc > best_acc:
            best_acc = acc
        
        avg_loss = total_loss / len(client_models)
        avg_parl = total_parl / len(client_models)
        num_comm = len(ldca.communities)
        
        history['accuracy'].append(acc)
        history['loss'].append(avg_loss)
        history['parl_loss'].append(avg_parl)
        history['communities'].append(num_comm)
        
        round_time = time.time() - round_start
        
        # Phase label
        if round_idx <= config.warmup_rounds:
            phase = "Warmup"
        elif round_idx == config.drift_round:
            phase = "DRIFT!"
        else:
            phase = "PARL+LDCA"
        
        # Print progress
        if round_idx <= 5 or round_idx % 10 == 0 or round_idx == config.drift_round:
            drift_marker = " *** DRIFT ***" if round_idx == config.drift_round else ""
            print(f"Round {round_idx:3d}/{config.num_rounds} [{phase:8s}] | "
                  f"Loss: {avg_loss:.4f} | PARL: {avg_parl:.4f} | "
                  f"Acc: {acc:.2f}% | Best: {best_acc:.2f}% | "
                  f"Comm: {num_comm} | {round_time:.1f}s{drift_marker}")
    
    total_time = time.time() - start_time
    
    # =========================================================================
    # Results Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    
    # Analyze pre/post drift
    pre_drift_acc = history['accuracy'][:config.drift_round-1]
    post_drift_acc = history['accuracy'][config.drift_round:]
    
    # Find recovery point (where accuracy returns to pre-drift levels)
    pre_drift_best = max(pre_drift_acc) if pre_drift_acc else 0
    recovery_round = None
    for i, acc in enumerate(post_drift_acc):
        if acc >= pre_drift_best * 0.95:  # 95% of pre-drift best
            recovery_round = config.drift_round + i + 1
            break
    
    print(f"\nResults Summary:")
    print(f"  Best Accuracy: {best_acc:.2f}%")
    print(f"  Final Accuracy: {history['accuracy'][-1]:.2f}%")
    print(f"  Total Time: {total_time/60:.1f} minutes")
    print(f"\nConcept Drift Analysis:")
    print(f"  Drift Type: {drift_type}")
    print(f"  Drift Round: {config.drift_round}")
    print(f"  Pre-drift Best: {pre_drift_best:.2f}%")
    print(f"  Post-drift Min: {min(post_drift_acc):.2f}%")
    print(f"  Post-drift Best: {max(post_drift_acc):.2f}%")
    print(f"  Accuracy Drop: {pre_drift_best - min(post_drift_acc):.2f}%")
    if recovery_round:
        print(f"  Recovery Round: {recovery_round} ({recovery_round - config.drift_round} rounds after drift)")
    else:
        print(f"  Recovery: Did not fully recover to pre-drift levels")
    print(f"  Final Communities: {history['communities'][-1]}")
    
    # Save results
    results = {
        'config': {
            'num_clients': config.num_clients,
            'num_rounds': config.num_rounds,
            'drift_round': config.drift_round,
            'drift_type': drift_type,
            'drift_intensity': config.drift_intensity,
        },
        'best_accuracy': best_acc,
        'final_accuracy': history['accuracy'][-1],
        'pre_drift_best': pre_drift_best,
        'post_drift_min': min(post_drift_acc),
        'post_drift_best': max(post_drift_acc),
        'recovery_round': recovery_round,
        'history': history,
        'total_time': total_time
    }
    
    filename = f'concept_drift_{drift_type}_results.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")
    
    return results


# ============================================================================
# Visualization for Concept Drift
# ============================================================================
def visualize_concept_drift(results, save_path='plots/concept_drift.png'):
    """Create visualization showing concept drift impact and recovery"""
    import matplotlib.pyplot as plt
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    history = results['history']
    drift_round = results['config']['drift_round']
    drift_type = results['config']['drift_type']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    rounds = list(range(1, len(history['accuracy']) + 1))
    
    # Accuracy plot with drift marked
    ax1 = axes[0, 0]
    ax1.plot(rounds, history['accuracy'], 'b-', linewidth=2, label='Accuracy')
    ax1.axvline(x=drift_round, color='red', linestyle='--', linewidth=2, 
                label=f'Drift ({drift_type})')
    ax1.axvspan(1, drift_round, alpha=0.1, color='green', label='Pre-drift')
    ax1.axvspan(drift_round, len(rounds), alpha=0.1, color='orange', label='Post-drift')
    
    if results.get('recovery_round'):
        ax1.axvline(x=results['recovery_round'], color='green', linestyle=':', 
                   linewidth=2, label=f"Recovery (Round {results['recovery_round']})")
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy with Concept Drift')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Community evolution
    ax2 = axes[0, 1]
    ax2.fill_between(rounds, history['communities'], alpha=0.4, color='green')
    ax2.plot(rounds, history['communities'], 'g-', linewidth=2)
    ax2.axvline(x=drift_round, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Number of Communities')
    ax2.set_title('LDCA Community Adaptation')
    ax2.grid(True, alpha=0.3)
    
    # Loss curves
    ax3 = axes[1, 0]
    ax3.plot(rounds, history['loss'], 'b-', linewidth=1.5, label='CE Loss')
    ax3.plot(rounds, history['parl_loss'], 'r-', linewidth=1.5, label='PARL Loss')
    ax3.axvline(x=drift_round, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Losses')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Pre vs Post drift comparison
    ax4 = axes[1, 1]
    pre_acc = history['accuracy'][:drift_round-1]
    post_acc = history['accuracy'][drift_round:]
    
    bp = ax4.boxplot([pre_acc, post_acc], patch_artist=True, 
                     tick_labels=['Pre-Drift', 'Post-Drift'])
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightsalmon')
    
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Accuracy Distribution: Pre vs Post Drift')
    
    # Add stats
    stats_text = f"Pre-drift mean: {np.mean(pre_acc):.1f}%\nPost-drift mean: {np.mean(post_acc):.1f}%"
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax4.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'FedPLC Concept Drift Analysis: {drift_type.replace("_", " ").title()}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    Path('plots').mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


# ============================================================================
# Run All Drift Scenarios
# ============================================================================
if __name__ == "__main__":
    # Run label swap drift experiment
    print("\n" + "="*70)
    print("RUNNING CONCEPT DRIFT EXPERIMENTS")
    print("="*70)
    
    # Experiment 1: Label Swap
    results_label_swap = run_concept_drift_experiment(drift_type="label_swap")
    visualize_concept_drift(results_label_swap, 'plots/concept_drift_label_swap.png')
    
    # Experiment 2: Distribution Shift
    print("\n" + "="*70)
    results_dist_shift = run_concept_drift_experiment(drift_type="distribution_shift")
    visualize_concept_drift(results_dist_shift, 'plots/concept_drift_distribution.png')
    
    print("\n" + "="*70)
    print("ALL CONCEPT DRIFT EXPERIMENTS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  📊 plots/concept_drift_label_swap.png")
    print("  📊 plots/concept_drift_distribution.png")
    print("  📄 concept_drift_label_swap_results.json")
    print("  📄 concept_drift_distribution_shift_results.json")
