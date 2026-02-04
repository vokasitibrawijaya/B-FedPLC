"""
FedPLC Concept Drift Experiment - Simplified Version
Tests LDCA's ability to adapt to dynamic data distribution changes

Author: FedPLC Replication Study
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
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
print("FedPLC Concept Drift Experiment (Simplified)")
print("="*70)
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# Simple CNN Model
# ============================================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
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
# Data Preparation - Pre-load to tensors for speed
# ============================================================================
def prepare_data(num_clients=30, alpha=0.5):
    """Prepare CIFAR-10 data as tensors"""
    print("Loading CIFAR-10...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    
    # Convert to tensors
    print("Converting to tensors...")
    train_data = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    train_labels = torch.tensor(train_dataset.targets)
    
    test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    test_labels = torch.tensor(test_dataset.targets)
    
    # Partition using Dirichlet
    print(f"Partitioning (alpha={alpha})...")
    client_indices = partition_data(train_labels, num_clients, alpha)
    
    return train_data, train_labels, test_data, test_labels, client_indices


def partition_data(labels, num_clients, alpha):
    """Dirichlet partitioning"""
    num_classes = 10
    client_indices = {i: [] for i in range(num_clients)}
    
    for c in range(num_classes):
        class_idx = torch.where(labels == c)[0].numpy()
        np.random.shuffle(class_idx)
        
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(class_idx)).astype(int)
        proportions[-1] = len(class_idx) - proportions[:-1].sum()
        
        start = 0
        for i in range(num_clients):
            client_indices[i].extend(class_idx[start:start + proportions[i]].tolist())
            start += proportions[i]
    
    return client_indices


# ============================================================================
# Training Functions
# ============================================================================
def train_client(model, data, labels, indices, community_proto=None, use_parl=False, epochs=2):
    """Train client"""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Get client data
    client_data = data[indices].to(device)
    client_labels = labels[indices].to(device)
    
    if len(client_data) == 0:
        return 0, 0
    
    # Mini-batches
    batch_size = min(64, len(client_data))
    total_loss = 0
    total_parl = 0
    n_batches = 0
    
    for epoch in range(epochs):
        perm = torch.randperm(len(client_data))
        for i in range(0, len(client_data), batch_size):
            idx = perm[i:i+batch_size]
            batch_data = client_data[idx]
            batch_labels = client_labels[idx]
            
            optimizer.zero_grad()
            output = model(batch_data)
            ce_loss = criterion(output, batch_labels)
            
            # PARL loss
            parl_loss = torch.tensor(0.0, device=device)
            if use_parl and community_proto is not None:
                features = model.get_features(batch_data)
                batch_proto = features.mean(dim=0)
                sim = F.cosine_similarity(batch_proto.unsqueeze(0), community_proto.unsqueeze(0))
                parl_loss = (1 - sim.squeeze()) * 0.1
            
            loss = ce_loss + parl_loss
            loss.backward()
            optimizer.step()
            
            total_loss += ce_loss.item()
            total_parl += parl_loss.item() if isinstance(parl_loss, torch.Tensor) else parl_loss
            n_batches += 1
    
    return total_loss / max(n_batches, 1), total_parl / max(n_batches, 1)


def compute_prototype(model, data, labels, indices):
    """Compute client prototype"""
    model.eval()
    with torch.no_grad():
        client_data = data[indices].to(device)
        if len(client_data) == 0:
            return None
        features = model.get_features(client_data)
        return features.mean(dim=0)


def evaluate(model, test_data, test_labels):
    """Evaluate model"""
    model.eval()
    with torch.no_grad():
        test_data = test_data.to(device)
        test_labels = test_labels.to(device)
        output = model(test_data)
        pred = output.argmax(dim=1)
        acc = pred.eq(test_labels).float().mean().item() * 100
    return acc


def get_label_distribution(labels, indices, label_map=None):
    """Get label distribution for client"""
    if label_map is None:
        label_map = {i: i for i in range(10)}
    
    dist = np.zeros(10)
    for idx in indices:
        original_label = labels[idx].item()
        mapped_label = label_map[original_label]
        dist[mapped_label] += 1
    return dist / (dist.sum() + 1e-8)


def compute_communities(distributions, threshold=0.85):
    """Compute communities based on label distribution similarity"""
    clients = list(distributions.keys())
    n = len(clients)
    
    if n < 2:
        return {0: clients}
    
    # Similarity matrix
    similarity = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_i = distributions[clients[i]]
            dist_j = distributions[clients[j]]
            similarity[i, j] = np.dot(dist_i, dist_j) / (
                np.linalg.norm(dist_i) * np.linalg.norm(dist_j) + 1e-8)
    
    # Clustering
    visited = [False] * n
    communities = {}
    comm_id = 0
    
    for i in range(n):
        if visited[i]:
            continue
        community = [clients[i]]
        visited[i] = True
        
        for j in range(n):
            if not visited[j] and similarity[i, j] >= threshold:
                community.append(clients[j])
                visited[j] = True
        
        communities[comm_id] = community
        comm_id += 1
    
    return communities


# ============================================================================
# Concept Drift Functions
# ============================================================================
def apply_label_swap(labels, indices, affected_clients, swap_pairs=None):
    """Apply label swap to affected clients"""
    if swap_pairs is None:
        swap_pairs = [(0, 1), (2, 3), (4, 5)]
    
    # Create label maps
    label_maps = {}
    for c in range(len(indices)):
        if c in affected_clients:
            label_map = {i: i for i in range(10)}
            for old, new in swap_pairs:
                label_map[old] = new
                label_map[new] = old
            label_maps[c] = label_map
        else:
            label_maps[c] = {i: i for i in range(10)}
    
    return label_maps


# ============================================================================
# Main Experiment
# ============================================================================
def run_experiment(drift_type="label_swap"):
    """Run concept drift experiment"""
    
    # Config
    num_clients = 30
    num_rounds = 60
    warmup_rounds = 15
    drift_round = 35
    drift_intensity = 0.3
    participation = 0.3
    
    print(f"\n{'='*70}")
    print(f"Experiment: {drift_type.upper()}")
    print(f"{'='*70}")
    print(f"Clients: {num_clients}, Rounds: {num_rounds}")
    print(f"Drift at round {drift_round}, intensity: {drift_intensity*100:.0f}%")
    
    # Prepare data
    train_data, train_labels, test_data, test_labels, client_indices = prepare_data(num_clients)
    
    # Model
    model = SimpleCNN().to(device)
    
    # Label maps (identity initially)
    label_maps = {c: {i: i for i in range(10)} for c in range(num_clients)}
    
    # History
    history = {'accuracy': [], 'loss': [], 'parl_loss': [], 'communities': []}
    
    best_acc = 0
    prototypes = {}
    communities = {0: list(range(num_clients))}
    drift_applied = False
    
    print(f"\nInitial accuracy: {evaluate(model, test_data, test_labels):.2f}%")
    print(f"\n{'='*70}")
    print("Training")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    for round_idx in range(1, num_rounds + 1):
        round_start = time.time()
        
        # Apply drift
        if round_idx == drift_round and not drift_applied:
            print(f"\n{'!'*70}")
            print(f"[DRIFT] Applying {drift_type} at round {round_idx}")
            
            num_affected = int(num_clients * drift_intensity)
            affected = np.random.choice(num_clients, num_affected, replace=False).tolist()
            
            if drift_type == "label_swap":
                label_maps = apply_label_swap(train_labels, client_indices, affected)
                print(f"[DRIFT] Swapped labels for {num_affected} clients")
            
            drift_applied = True
            print(f"{'!'*70}\n")
        
        # Select clients
        num_selected = max(1, int(num_clients * participation))
        selected = np.random.choice(num_clients, num_selected, replace=False)
        
        # Determine phase
        use_parl = round_idx > warmup_rounds
        
        # Update communities
        if use_parl and (round_idx - warmup_rounds) % 5 == 1:
            distributions = {}
            for c in range(num_clients):
                distributions[c] = get_label_distribution(
                    train_labels, client_indices[c], label_maps[c])
            
            old_comm = len(communities)
            communities = compute_communities(distributions)
            new_comm = len(communities)
            
            if old_comm != new_comm:
                print(f"\n[LDCA] Communities: {old_comm} -> {new_comm}")
        
        # Local training
        client_models = []
        client_sizes = []
        total_loss = 0
        total_parl = 0
        
        for c in selected:
            client_model = copy.deepcopy(model)
            
            # Get community prototype
            comm_proto = None
            if use_parl:
                for comm_id, members in communities.items():
                    if c in members:
                        member_protos = [prototypes[m] for m in members if m in prototypes and m != c]
                        if member_protos:
                            comm_proto = torch.stack(member_protos).mean(dim=0)
                        break
            
            # Apply label mapping during training
            mapped_labels = train_labels.clone()
            for idx in client_indices[c]:
                orig = train_labels[idx].item()
                mapped_labels[idx] = label_maps[c][orig]
            
            loss, parl_loss = train_client(
                client_model, train_data, mapped_labels, client_indices[c],
                comm_proto, use_parl=use_parl, epochs=2
            )
            
            # Update prototype
            prototypes[c] = compute_prototype(client_model, train_data, mapped_labels, client_indices[c])
            
            client_models.append(client_model)
            client_sizes.append(len(client_indices[c]))
            total_loss += loss
            total_parl += parl_loss
        
        # Aggregate
        total_size = sum(client_sizes)
        global_dict = model.state_dict()
        for key in global_dict:
            global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
            for i, cm in enumerate(client_models):
                w = client_sizes[i] / total_size
                global_dict[key] += w * cm.state_dict()[key].float()
        model.load_state_dict(global_dict)
        
        # Evaluate
        acc = evaluate(model, test_data, test_labels)
        best_acc = max(best_acc, acc)
        
        avg_loss = total_loss / len(client_models)
        avg_parl = total_parl / len(client_models)
        
        history['accuracy'].append(acc)
        history['loss'].append(avg_loss)
        history['parl_loss'].append(avg_parl)
        history['communities'].append(len(communities))
        
        round_time = time.time() - round_start
        
        # Phase label
        if round_idx <= warmup_rounds:
            phase = "Warmup"
        elif round_idx == drift_round:
            phase = "DRIFT!"
        else:
            phase = "PARL+LDCA"
        
        # Print
        if round_idx <= 5 or round_idx % 5 == 0 or round_idx == drift_round:
            marker = " *** DRIFT ***" if round_idx == drift_round else ""
            print(f"Round {round_idx:2d}/{num_rounds} [{phase:8s}] | "
                  f"Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Best: {best_acc:.2f}% | "
                  f"Comm: {len(communities)} | {round_time:.1f}s{marker}")
    
    total_time = time.time() - start_time
    
    # Analysis
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    pre_drift = history['accuracy'][:drift_round-1]
    post_drift = history['accuracy'][drift_round:]
    
    pre_drift_best = max(pre_drift) if pre_drift else 0
    post_drift_min = min(post_drift) if post_drift else 0
    post_drift_best = max(post_drift) if post_drift else 0
    
    # Find recovery
    recovery = None
    for i, acc in enumerate(post_drift):
        if acc >= pre_drift_best * 0.95:
            recovery = drift_round + i + 1
            break
    
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Final Accuracy: {history['accuracy'][-1]:.2f}%")
    print(f"Time: {total_time/60:.1f} min")
    print(f"\nConcept Drift Analysis:")
    print(f"  Pre-drift Best: {pre_drift_best:.2f}%")
    print(f"  Post-drift Drop: {pre_drift_best - post_drift_min:.2f}%")
    print(f"  Post-drift Best: {post_drift_best:.2f}%")
    if recovery:
        print(f"  Recovery: Round {recovery} ({recovery - drift_round} rounds)")
    else:
        print(f"  Recovery: Not fully recovered")
    
    # Save
    results = {
        'config': {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'drift_round': drift_round,
            'drift_type': drift_type,
        },
        'best_accuracy': best_acc,
        'final_accuracy': history['accuracy'][-1],
        'pre_drift_best': pre_drift_best,
        'post_drift_min': post_drift_min,
        'post_drift_best': post_drift_best,
        'accuracy_drop': pre_drift_best - post_drift_min,
        'recovery_round': recovery,
        'history': history,
        'total_time': total_time
    }
    
    filename = f'concept_drift_{drift_type}_results.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {filename}")
    
    return results


# ============================================================================
# Visualization
# ============================================================================
def visualize_drift(results, save_path='plots/concept_drift.png'):
    """Visualize concept drift experiment"""
    import matplotlib.pyplot as plt
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    history = results['history']
    drift_round = results['config']['drift_round']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    rounds = list(range(1, len(history['accuracy']) + 1))
    
    # Accuracy
    ax1 = axes[0, 0]
    ax1.plot(rounds, history['accuracy'], 'b-', linewidth=2)
    ax1.axvline(drift_round, color='red', linestyle='--', linewidth=2, label='Drift')
    ax1.axvspan(1, drift_round, alpha=0.1, color='green')
    ax1.axvspan(drift_round, len(rounds), alpha=0.1, color='orange')
    if results.get('recovery_round'):
        ax1.axvline(results['recovery_round'], color='green', linestyle=':', 
                   linewidth=2, label=f"Recovery")
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy with Concept Drift')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Communities
    ax2 = axes[0, 1]
    ax2.fill_between(rounds, history['communities'], alpha=0.4, color='green')
    ax2.plot(rounds, history['communities'], 'g-', linewidth=2)
    ax2.axvline(drift_round, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Communities')
    ax2.set_title('LDCA Community Adaptation')
    ax2.grid(True, alpha=0.3)
    
    # Loss
    ax3 = axes[1, 0]
    ax3.plot(rounds, history['loss'], 'b-', linewidth=1.5, label='CE Loss')
    ax3.plot(rounds, history['parl_loss'], 'r-', linewidth=1.5, label='PARL Loss')
    ax3.axvline(drift_round, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Losses')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Box comparison
    ax4 = axes[1, 1]
    pre = history['accuracy'][:drift_round-1]
    post = history['accuracy'][drift_round:]
    bp = ax4.boxplot([pre, post], patch_artist=True, tick_labels=['Pre-Drift', 'Post-Drift'])
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightsalmon')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Pre vs Post Drift')
    ax4.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f"FedPLC Concept Drift: {results['config']['drift_type'].replace('_', ' ').title()}", 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    Path('plots').mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    # Run label swap experiment
    results = run_experiment(drift_type="label_swap")
    visualize_drift(results, 'plots/concept_drift_label_swap.png')
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
