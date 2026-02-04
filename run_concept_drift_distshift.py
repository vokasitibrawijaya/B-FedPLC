"""
FedPLC Concept Drift Experiment - Distribution Shift
Continuation from label_swap experiment

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
print("FedPLC Concept Drift Experiment - Distribution Shift")
print("="*70)
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# Configuration
# ============================================================================
class Config:
    num_clients = 30
    num_rounds = 60
    local_epochs = 3
    batch_size = 64
    lr = 0.01
    participation_rate = 0.2

    # Non-IID settings
    alpha_initial = 0.5
    alpha_drifted = 0.1  # More extreme Non-IID after drift

    # PARL settings
    parl_weight = 0.1
    warmup_rounds = 15

    # LDCA settings
    similarity_threshold = 0.85
    ldca_update_interval = 5

    # Concept drift settings
    drift_round = 35
    drift_type = "distribution_shift"
    drift_intensity = 0.3  # 30% clients affected


config = Config()


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
# Data Preparation
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


def partition_data(labels, num_clients, alpha, seed=42):
    """Dirichlet partitioning"""
    np.random.seed(seed)
    num_classes = 10
    client_indices = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        class_idx = torch.where(labels == c)[0].numpy()
        np.random.shuffle(class_idx)

        proportions = np.random.dirichlet([alpha] * num_clients)
        splits = (proportions * len(class_idx)).astype(int)
        splits[-1] = len(class_idx) - splits[:-1].sum()

        start = 0
        for i, n in enumerate(splits):
            client_indices[i].extend(class_idx[start:start+n].tolist())
            start += n

    return client_indices


def apply_distribution_shift(train_labels, client_indices, affected_clients, new_alpha=0.1, seed=None):
    """Re-partition data for affected clients with more extreme Non-IID"""
    if seed is not None:
        np.random.seed(seed)

    num_clients = len(client_indices)
    num_classes = 10

    # Collect all indices from affected clients
    affected_indices = []
    for cid in affected_clients:
        affected_indices.extend(client_indices[cid])
        client_indices[cid] = []

    affected_indices = np.array(affected_indices)
    np.random.shuffle(affected_indices)

    # Get labels for affected indices
    affected_labels = train_labels[affected_indices].numpy()

    # Re-partition with new alpha (more extreme)
    for c in range(num_classes):
        class_mask = affected_labels == c
        class_idx = affected_indices[class_mask]
        np.random.shuffle(class_idx)

        # New Dirichlet with lower alpha (more skewed)
        proportions = np.random.dirichlet([new_alpha] * len(affected_clients))
        splits = (proportions * len(class_idx)).astype(int)
        splits[-1] = len(class_idx) - splits[:-1].sum()

        start = 0
        for i, cid in enumerate(affected_clients):
            client_indices[cid].extend(class_idx[start:start+splits[i]].tolist())
            start += splits[i]

    return client_indices


# ============================================================================
# LDCA Clustering
# ============================================================================
class LDCA:
    def __init__(self, num_clients, num_classes=10, threshold=0.85):
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.threshold = threshold
        self.distributions = {i: np.ones(num_classes) / num_classes for i in range(num_clients)}
        self.communities = [[i] for i in range(num_clients)]  # Start with single communities

    def update_distribution(self, client_id, distribution):
        self.distributions[client_id] = distribution

    def compute_similarity(self, dist1, dist2):
        return 1 - 0.5 * np.sum(np.abs(dist1 - dist2))

    def compute_communities(self):
        """Greedy community detection based on distribution similarity"""
        assigned = set()
        communities = []

        for i in range(self.num_clients):
            if i in assigned:
                continue

            community = [i]
            assigned.add(i)

            for j in range(i+1, self.num_clients):
                if j in assigned:
                    continue

                sim = self.compute_similarity(self.distributions[i], self.distributions[j])
                if sim >= self.threshold:
                    community.append(j)
                    assigned.add(j)

            communities.append(community)

        self.communities = communities
        return communities

    def get_community_prototypes(self, client_id, client_prototypes):
        """Get averaged prototypes from same community"""
        for comm in self.communities:
            if client_id in comm:
                valid_protos = [client_prototypes[c] for c in comm
                              if c in client_prototypes and client_prototypes[c] is not None]
                if valid_protos:
                    return torch.stack(valid_protos).mean(0)
        return None


def get_label_distribution(train_labels, indices, num_classes=10):
    """Compute label distribution for given indices"""
    labels = train_labels[indices].numpy()
    dist = np.zeros(num_classes)
    for l in labels:
        dist[l] += 1
    return dist / (dist.sum() + 1e-8)


# ============================================================================
# Training Functions
# ============================================================================
def train_client(model, dataloader, local_proto, comm_proto, use_parl=True,
                epochs=3, lr=0.01, parl_weight=0.1):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    total_parl = 0
    num_batches = 0

    for epoch in range(epochs):
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model(data)
            ce_loss = criterion(output, labels)

            parl_loss = torch.tensor(0.0, device=device)
            if use_parl and comm_proto is not None and local_proto is not None:
                features = model.get_features(data)
                local_proto = local_proto.to(device)
                comm_proto = comm_proto.to(device)

                # PARL: maximize similarity to community prototypes
                feat_mean = features.mean(0)
                sim_local = F.cosine_similarity(feat_mean.unsqueeze(0), local_proto.unsqueeze(0))
                sim_comm = F.cosine_similarity(feat_mean.unsqueeze(0), comm_proto.unsqueeze(0))
                parl_loss = parl_weight * (1 - 0.5 * (sim_local + sim_comm))

            loss = ce_loss + parl_loss
            loss.backward()
            optimizer.step()

            total_loss += ce_loss.item()
            total_parl += parl_loss.item() if isinstance(parl_loss, torch.Tensor) else parl_loss
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_parl = total_parl / max(num_batches, 1)
    return model, avg_loss, avg_parl


def compute_prototypes(model, dataloader):
    """Compute mean feature as prototype"""
    model.eval()
    all_features = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            feat = model.get_features(data)
            all_features.append(feat)

    if all_features:
        all_features = torch.cat(all_features, 0)
        return all_features.mean(0).cpu()
    return None


def federated_averaging(global_model, client_models, client_sizes):
    """Weighted averaging of client models"""
    total_size = sum(client_sizes)
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
        for model, size in zip(client_models, client_sizes):
            weight = size / total_size
            global_dict[key] += weight * model.state_dict()[key].float()

    global_model.load_state_dict(global_dict)
    return global_model


def evaluate(model, test_loader):
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
# Main Experiment
# ============================================================================
def run_experiment():
    print("\n" + "="*70)
    print(f"Experiment: DISTRIBUTION_SHIFT")
    print("="*70)
    print(f"Clients: {config.num_clients}, Rounds: {config.num_rounds}")
    print(f"Drift at round {config.drift_round}, intensity: {int(config.drift_intensity*100)}%")
    print(f"Alpha: {config.alpha_initial} -> {config.alpha_drifted} (after drift)")

    # Prepare data
    train_data, train_labels, test_data, test_labels, client_indices = prepare_data(
        config.num_clients, config.alpha_initial)

    # Create test loader
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Initialize model
    global_model = SimpleCNN().to(device)

    # Initialize LDCA
    ldca = LDCA(config.num_clients, threshold=config.similarity_threshold)
    for cid in range(config.num_clients):
        dist = get_label_distribution(train_labels, client_indices[cid])
        ldca.update_distribution(cid, dist)

    # Initial evaluation
    initial_acc = evaluate(global_model, test_loader)
    print(f"\nInitial accuracy: {initial_acc:.2f}%")

    # Training state
    client_prototypes = {}
    history = {'accuracy': [], 'loss': [], 'parl_loss': [], 'communities': []}
    best_acc = 0
    drift_applied = False

    print("\n" + "="*70)
    print("Training")
    print("="*70)

    start_time = time.time()

    for round_idx in range(1, config.num_rounds + 1):
        round_start = time.time()

        # Apply drift at specified round
        if round_idx == config.drift_round and not drift_applied:
            print(f"\n{'!'*70}")
            print(f"[DRIFT] Applying distribution_shift at round {round_idx}")

            num_affected = int(config.num_clients * config.drift_intensity)
            affected_clients = list(np.random.choice(
                config.num_clients, num_affected, replace=False))

            # Apply distribution shift
            client_indices = apply_distribution_shift(
                train_labels, client_indices, affected_clients,
                new_alpha=config.alpha_drifted, seed=round_idx)

            # Update LDCA distributions
            for cid in affected_clients:
                dist = get_label_distribution(train_labels, client_indices[cid])
                ldca.update_distribution(cid, dist)

            print(f"[DRIFT] Shifted {num_affected} clients from alpha={config.alpha_initial} to alpha={config.alpha_drifted}")
            print(f"[DRIFT] Affected clients: {sorted(affected_clients)}")
            drift_applied = True
            print(f"{'!'*70}\n")

        # Client selection
        num_selected = max(1, int(config.num_clients * config.participation_rate))
        selected_clients = np.random.choice(config.num_clients, num_selected, replace=False)

        # Determine phase
        use_parl = round_idx > config.warmup_rounds

        # Update communities periodically
        if use_parl and (round_idx - config.warmup_rounds) % config.ldca_update_interval == 1:
            # Update all distributions
            for cid in range(config.num_clients):
                dist = get_label_distribution(train_labels, client_indices[cid])
                ldca.update_distribution(cid, dist)

            old_comm = len(ldca.communities)
            ldca.compute_communities()
            new_comm = len(ldca.communities)

            if old_comm != new_comm:
                print(f"\n[LDCA] Communities: {old_comm} -> {new_comm}")

        # Local training
        client_models = []
        client_sizes = []
        total_loss = 0
        total_parl = 0

        for cid in selected_clients:
            if len(client_indices[cid]) < 10:
                continue

            # Create client dataloader
            indices = client_indices[cid]
            client_data = train_data[indices]
            client_labels = train_labels[indices]
            client_dataset = TensorDataset(client_data, client_labels)
            client_loader = DataLoader(client_dataset, batch_size=config.batch_size, shuffle=True)

            # Get prototypes
            local_proto = client_prototypes.get(cid)
            comm_proto = ldca.get_community_prototypes(cid, client_prototypes) if use_parl else None

            # Train
            client_model = copy.deepcopy(global_model)
            client_model, loss, parl_loss = train_client(
                client_model, client_loader, local_proto, comm_proto,
                use_parl=use_parl, epochs=config.local_epochs,
                lr=config.lr, parl_weight=config.parl_weight
            )

            # Update prototype
            client_prototypes[cid] = compute_prototypes(client_model, client_loader)

            client_models.append(client_model)
            client_sizes.append(len(indices))
            total_loss += loss
            total_parl += parl_loss

        if not client_models:
            continue

        # Aggregate
        global_model = federated_averaging(global_model, client_models, client_sizes)

        # Evaluate
        acc = evaluate(global_model, test_loader)
        best_acc = max(best_acc, acc)

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
        if round_idx <= 5 or round_idx % 5 == 0 or round_idx == config.drift_round:
            drift_marker = " *** DRIFT ***" if round_idx == config.drift_round else ""
            print(f"Round {round_idx:3d}/{config.num_rounds} [{phase:8s}] | "
                  f"Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Best: {best_acc:.2f}% | "
                  f"Comm: {num_comm} | {round_time:.1f}s{drift_marker}")

    total_time = time.time() - start_time

    # Results summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Final Accuracy: {history['accuracy'][-1]:.2f}%")
    print(f"Time: {total_time/60:.1f} min")

    # Drift analysis
    pre_drift = history['accuracy'][:config.drift_round-1]
    post_drift = history['accuracy'][config.drift_round:]

    pre_drift_best = max(pre_drift) if pre_drift else 0
    post_drift_min = min(post_drift) if post_drift else 0
    post_drift_best = max(post_drift) if post_drift else 0
    accuracy_drop = pre_drift_best - post_drift_min

    # Recovery round
    recovery_round = None
    for i, acc in enumerate(post_drift):
        if acc >= pre_drift_best * 0.95:
            recovery_round = config.drift_round + i + 1
            break

    print(f"\nConcept Drift Analysis:")
    print(f"  Pre-drift Best: {pre_drift_best:.2f}%")
    print(f"  Post-drift Drop: {accuracy_drop:.2f}%")
    print(f"  Post-drift Best: {post_drift_best:.2f}%")
    if recovery_round:
        print(f"  Recovery: Round {recovery_round} ({recovery_round - config.drift_round} rounds)")
    else:
        print(f"  Recovery: Not reached 95% of pre-drift performance")

    # Save results
    results = {
        'config': {
            'num_clients': config.num_clients,
            'num_rounds': config.num_rounds,
            'drift_round': config.drift_round,
            'drift_type': config.drift_type,
            'drift_intensity': config.drift_intensity,
            'alpha_initial': config.alpha_initial,
            'alpha_drifted': config.alpha_drifted,
        },
        'best_accuracy': best_acc,
        'final_accuracy': history['accuracy'][-1],
        'pre_drift_best': pre_drift_best,
        'post_drift_min': post_drift_min,
        'post_drift_best': post_drift_best,
        'accuracy_drop': accuracy_drop,
        'recovery_round': recovery_round,
        'history': history,
        'total_time': total_time
    }

    filename = f'concept_drift_{config.drift_type}_results.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {filename}")

    # Visualization
    visualize_drift(results, f'plots/concept_drift_{config.drift_type}.png')

    return results


def visualize_drift(results, save_path):
    """Create visualization"""
    import matplotlib.pyplot as plt

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
        ax1.axvline(results['recovery_round'], color='green', linestyle=':', linewidth=2, label='Recovery')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy with Distribution Shift')
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
    ax4.set_title('Pre vs Post Drift Distribution')
    ax4.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f"FedPLC Concept Drift: Distribution Shift (α: {results['config']['alpha_initial']} → {results['config']['alpha_drifted']})",
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
    results = run_experiment()

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
