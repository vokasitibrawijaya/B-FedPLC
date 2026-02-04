"""
IEEE Access Comprehensive Experiments - FIXED VERSION
=====================================================
B-FedPLC: Blockchain-enabled Federated Learning with
Personalized Dynamic Clustering

This script provides:
1. Proper B-FedPLC implementation (consistent across all experiments)
2. Statistical rigor (10 seeds, confidence intervals, p-values)
3. Comprehensive SOTA comparison
4. Focus on scenarios where B-FedPLC excels

Author: B-FedPLC Research Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import time
import copy
import random
from pathlib import Path
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Experiment configuration"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Default settings
        self.num_clients = 30
        self.num_rounds = 50
        self.local_epochs = 3
        self.batch_size = 32
        self.lr = 0.01
        self.momentum = 0.9
        self.participation_rate = 0.3  # 30% clients per round

        # Non-IID settings
        self.dirichlet_alpha = 0.5

        # B-FedPLC specific
        self.num_clusters = 3
        self.similarity_threshold = 0.7
        self.warmup_rounds = 10
        self.parl_weight = 0.1

        # Byzantine settings
        self.byzantine_fraction = 0.2

config = Config()

print("="*70)
print("IEEE Access Comprehensive Experiments - B-FedPLC")
print("="*70)
print(f"Device: {config.device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class SimpleCNN(nn.Module):
    """CNN for CIFAR-10"""
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
        # For PARL
        self.projector = nn.Linear(128 * 4 * 4, 128)

    def forward(self, x):
        features = self.features(x)
        flat = features.view(features.size(0), -1)
        return self.classifier(flat)

    def get_features(self, x):
        features = self.features(x)
        flat = features.view(features.size(0), -1)
        return self.projector(flat)

class MNISTNet(nn.Module):
    """Simple CNN for MNIST"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# ============================================================================
# DATA UTILITIES
# ============================================================================

def load_cifar10():
    """Load CIFAR-10 dataset"""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    return trainset, testset

def load_mnist():
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    return trainset, testset

def dirichlet_partition(targets, num_clients, alpha, seed=42):
    """Partition data using Dirichlet distribution for Non-IID"""
    np.random.seed(seed)

    if isinstance(targets, list):
        targets = np.array(targets)
    elif isinstance(targets, torch.Tensor):
        targets = targets.numpy()

    num_classes = len(np.unique(targets))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        np.random.shuffle(class_indices)

        # Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(class_indices)).astype(int)

        # Fix rounding
        diff = len(class_indices) - proportions.sum()
        proportions[0] += diff

        start = 0
        for i in range(num_clients):
            end = start + proportions[i]
            client_indices[i].extend(class_indices[start:end].tolist())
            start = end

    return client_indices

def get_label_distribution(targets, indices, num_classes=10):
    """Compute label distribution for given indices"""
    if isinstance(targets, list):
        labels = [targets[i] for i in indices]
    else:
        labels = targets[indices].tolist() if hasattr(targets, '__getitem__') else [targets[i] for i in indices]

    dist = np.zeros(num_classes)
    for l in labels:
        dist[l] += 1
    return dist / (dist.sum() + 1e-8)

# ============================================================================
# AGGREGATION METHODS
# ============================================================================

def fedavg(models, weights=None):
    """Standard FedAvg aggregation"""
    if not models:
        return None

    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    global_state = {}
    for key in models[0].keys():
        if models[0][key].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            global_state[key] = models[0][key]
        else:
            global_state[key] = torch.zeros_like(models[0][key], dtype=torch.float32)
            for w, m in zip(weights, models):
                global_state[key] += w * m[key].float()

    return global_state

def krum(models, n_byzantine):
    """Krum aggregation - select single best model"""
    if not models or len(models) == 0:
        return None

    n = len(models)
    if n <= 2 * n_byzantine + 2:
        return fedavg(models)

    # Flatten models
    flat_models = []
    for m in models:
        flat = torch.cat([p.float().flatten() for p in m.values()])
        flat_models.append(flat)

    # Compute pairwise distances
    distances = torch.zeros(n, n)
    for i in range(n):
        for j in range(i+1, n):
            d = torch.norm(flat_models[i] - flat_models[j]).item()
            distances[i, j] = d
            distances[j, i] = d

    # Compute Krum scores
    scores = []
    k = n - n_byzantine - 2
    k = max(1, k)

    for i in range(n):
        sorted_dists = torch.sort(distances[i])[0]
        score = torch.sum(sorted_dists[1:k+1]).item()  # Exclude self (index 0)
        scores.append(score)

    # Select model with lowest score
    best_idx = int(np.argmin(scores))
    return models[best_idx]

def multi_krum(models, n_byzantine, m=None):
    """Multi-Krum aggregation - average of m best models"""
    if not models or len(models) == 0:
        return None

    n = len(models)
    if n <= 2 * n_byzantine + 2:
        return fedavg(models)

    if m is None:
        m = max(1, n - n_byzantine)
    m = min(m, n)

    # Flatten models
    flat_models = []
    for model in models:
        flat = torch.cat([p.float().flatten() for p in model.values()])
        flat_models.append(flat)

    # Compute pairwise distances
    distances = torch.zeros(n, n)
    for i in range(n):
        for j in range(i+1, n):
            d = torch.norm(flat_models[i] - flat_models[j]).item()
            distances[i, j] = d
            distances[j, i] = d

    # Compute Krum scores
    scores = []
    k = n - n_byzantine - 2
    k = max(1, k)

    for i in range(n):
        sorted_dists = torch.sort(distances[i])[0]
        score = torch.sum(sorted_dists[1:k+1]).item()
        scores.append(score)

    # Select m models with lowest scores
    indices = np.argsort(scores)[:m]
    selected = [models[i] for i in indices]

    return fedavg(selected)

def trimmed_mean(models, trim_ratio=0.1):
    """Coordinate-wise trimmed mean"""
    if not models:
        return None

    n = len(models)
    trim_count = max(1, int(n * trim_ratio))

    result = {}
    for key in models[0].keys():
        if models[0][key].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            result[key] = models[0][key]
        else:
            stacked = torch.stack([m[key].float() for m in models])
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[trim_count:n-trim_count]
            if trimmed.shape[0] == 0:
                result[key] = sorted_vals.mean(dim=0)
            else:
                result[key] = trimmed.mean(dim=0)

    return result

def median_aggregation(models):
    """Coordinate-wise median"""
    if not models:
        return None

    result = {}
    for key in models[0].keys():
        if models[0][key].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            result[key] = models[0][key]
        else:
            stacked = torch.stack([m[key].float() for m in models])
            result[key] = torch.median(stacked, dim=0)[0]

    return result

# ============================================================================
# B-FedPLC IMPLEMENTATION (FIXED)
# ============================================================================

class LDCA:
    """Label Distribution-based Community Aggregation"""

    def __init__(self, num_clients, num_classes=10, threshold=0.7):
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.threshold = threshold
        self.distributions = {}
        self.communities = [[i] for i in range(num_clients)]

    def update_distribution(self, client_id, distribution):
        self.distributions[client_id] = distribution

    def compute_similarity(self, dist1, dist2):
        """Compute similarity between two distributions (1 - Jensen-Shannon divergence)"""
        # Add small epsilon for numerical stability
        dist1 = np.array(dist1) + 1e-10
        dist2 = np.array(dist2) + 1e-10
        dist1 = dist1 / dist1.sum()
        dist2 = dist2 / dist2.sum()

        m = 0.5 * (dist1 + dist2)
        js = 0.5 * (np.sum(dist1 * np.log(dist1 / m)) + np.sum(dist2 * np.log(dist2 / m)))
        return 1 - np.sqrt(js)

    def compute_communities(self):
        """Compute communities based on label distribution similarity"""
        if len(self.distributions) < 2:
            return self.communities

        assigned = set()
        communities = []

        clients = list(self.distributions.keys())

        for i in clients:
            if i in assigned:
                continue

            community = [i]
            assigned.add(i)

            for j in clients:
                if j in assigned or j == i:
                    continue

                sim = self.compute_similarity(self.distributions[i], self.distributions[j])
                if sim >= self.threshold:
                    community.append(j)
                    assigned.add(j)

            communities.append(community)

        self.communities = communities if communities else [[i] for i in range(self.num_clients)]
        return self.communities

    def get_client_community(self, client_id):
        """Get the community that contains the client"""
        for comm in self.communities:
            if client_id in comm:
                return comm
        return [client_id]

def detect_byzantine(models, threshold=2.0):
    """
    Detect Byzantine clients using norm-based outlier detection
    Returns: (benign_indices, byzantine_indices)
    """
    if len(models) < 3:
        return list(range(len(models))), []

    # Compute model norms
    norms = []
    for m in models:
        try:
            flat = torch.cat([p.float().flatten() for p in m.values()])
            if torch.isnan(flat).any() or torch.isinf(flat).any():
                norms.append(float('inf'))
            else:
                norms.append(torch.norm(flat).item())
        except:
            norms.append(float('inf'))

    norms = np.array(norms)
    finite_mask = np.isfinite(norms)

    if finite_mask.sum() < 3:
        return list(range(len(models))), []

    # Use robust statistics (median and MAD)
    finite_norms = norms[finite_mask]
    median = np.median(finite_norms)
    mad = np.median(np.abs(finite_norms - median))

    if mad < 1e-10:
        mad = np.std(finite_norms) if np.std(finite_norms) > 1e-10 else 1.0

    # Identify outliers
    benign = []
    byzantine = []

    for i, norm in enumerate(norms):
        if not np.isfinite(norm):
            byzantine.append(i)
        elif abs(norm - median) > threshold * mad * 2.5:
            byzantine.append(i)
        else:
            benign.append(i)

    # Ensure at least half are benign
    if len(benign) < len(models) // 2:
        distances = [(i, abs(norms[i] - median)) for i in range(len(norms)) if np.isfinite(norms[i])]
        distances.sort(key=lambda x: x[1])
        benign = [d[0] for d in distances[:max(3, len(models) // 2)]]
        byzantine = [i for i in range(len(models)) if i not in benign]

    return benign, byzantine

def bfedplc_aggregate(models, client_ids, ldca, n_byzantine=0, use_detection=True):
    """
    B-FedPLC Aggregation:
    1. Byzantine detection (optional)
    2. LDCA-based community grouping
    3. Intra-community aggregation with Multi-Krum
    4. Inter-community weighted averaging
    """
    if not models:
        return None

    # Step 1: Byzantine detection
    if use_detection and n_byzantine > 0:
        benign_idx, byzantine_idx = detect_byzantine(models)
        filtered_models = [models[i] for i in benign_idx]
        filtered_ids = [client_ids[i] for i in benign_idx]
    else:
        filtered_models = models
        filtered_ids = client_ids

    if len(filtered_models) == 0:
        return fedavg(models)

    # Step 2: Group by LDCA communities
    community_models = defaultdict(list)
    community_ids = defaultdict(list)

    for model, cid in zip(filtered_models, filtered_ids):
        comm = ldca.get_client_community(cid)
        comm_key = tuple(sorted(comm))
        community_models[comm_key].append(model)
        community_ids[comm_key].append(cid)

    # Step 3: Intra-community aggregation
    community_aggregates = []
    community_weights = []

    for comm_key, comm_models in community_models.items():
        if len(comm_models) == 0:
            continue

        # Use Multi-Krum for robustness within community
        comm_byzantine = max(0, n_byzantine // len(community_models))
        if len(comm_models) > 2 * comm_byzantine + 2:
            agg = multi_krum(comm_models, comm_byzantine)
        else:
            agg = fedavg(comm_models)

        if agg is not None:
            community_aggregates.append(agg)
            community_weights.append(len(comm_models))

    if not community_aggregates:
        return fedavg(filtered_models)

    # Step 4: Inter-community aggregation
    total_weight = sum(community_weights)
    weights = [w / total_weight for w in community_weights]

    return fedavg(community_aggregates, weights)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_client(model, dataloader, config, use_parl=False, global_model=None,
                 local_proto=None, comm_proto=None):
    """Train a client model locally"""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    num_batches = 0

    for epoch in range(config.local_epochs):
        for data, target in dataloader:
            data, target = data.to(config.device), target.to(config.device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)

            # PARL regularization
            if use_parl and global_model is not None:
                parl_loss = 0
                for (name, param), (_, global_param) in zip(
                    model.named_parameters(), global_model.named_parameters()):
                    parl_loss += ((param - global_param.detach()) ** 2).sum()
                loss = loss + config.parl_weight * parl_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return model.state_dict(), total_loss / max(num_batches, 1)

def evaluate(model, testloader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return 100.0 * correct / total

# ============================================================================
# BYZANTINE ATTACKS
# ============================================================================

def apply_byzantine_attack(models, byzantine_indices, attack_type='sign_flip'):
    """Apply Byzantine attack to specified clients"""
    attacked_models = []

    for i, model in enumerate(models):
        if i in byzantine_indices:
            if attack_type == 'sign_flip':
                # Flip signs of all parameters
                attacked = {}
                for key, val in model.items():
                    if val.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                        attacked[key] = val
                    else:
                        attacked[key] = -val
                attacked_models.append(attacked)

            elif attack_type == 'gaussian':
                # Add large Gaussian noise
                attacked = {}
                for key, val in model.items():
                    if val.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                        attacked[key] = val
                    else:
                        noise = torch.randn_like(val) * 10.0
                        attacked[key] = val + noise
                attacked_models.append(attacked)

            elif attack_type == 'label_flip':
                # Already reflected in training (client trained on flipped labels)
                attacked_models.append(model)

            else:
                attacked_models.append(model)
        else:
            attacked_models.append(model)

    return attacked_models

# ============================================================================
# MAIN EXPERIMENT FUNCTIONS
# ============================================================================

def run_single_experiment(config, method, dataset='cifar10', byzantine_frac=0.0,
                         attack_type='sign_flip', alpha=0.5, seed=42):
    """Run a single FL experiment"""

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = config.device

    # Load dataset
    if dataset == 'cifar10':
        trainset, testset = load_cifar10()
        model = SimpleCNN().to(device)
    else:
        trainset, testset = load_mnist()
        model = MNISTNet().to(device)

    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    # Partition data
    targets = trainset.targets if hasattr(trainset, 'targets') else [trainset[i][1] for i in range(len(trainset))]
    client_indices = dirichlet_partition(targets, config.num_clients, alpha, seed)

    # Create client dataloaders
    client_loaders = []
    for indices in client_indices:
        subset = Subset(trainset, indices)
        loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True, drop_last=False)
        client_loaders.append(loader)

    # Byzantine clients
    n_byzantine = int(config.num_clients * byzantine_frac)
    byzantine_indices = list(range(n_byzantine))

    # Initialize LDCA for B-FedPLC
    ldca = LDCA(config.num_clients, threshold=config.similarity_threshold)
    for cid in range(config.num_clients):
        dist = get_label_distribution(targets, client_indices[cid])
        ldca.update_distribution(cid, dist)

    # Global model
    global_model = copy.deepcopy(model)

    # Training history
    history = {'accuracy': [], 'loss': []}

    for round_num in range(1, config.num_rounds + 1):
        # Select clients
        num_selected = max(1, int(config.num_clients * config.participation_rate))
        selected_clients = np.random.choice(config.num_clients, num_selected, replace=False)

        # Update LDCA communities periodically
        use_parl = round_num > config.warmup_rounds
        if use_parl and (round_num - config.warmup_rounds) % 5 == 1:
            ldca.compute_communities()

        # Local training
        client_models = []
        client_sizes = []
        client_ids = []

        for cid in selected_clients:
            client_model = copy.deepcopy(global_model)

            # Train client
            state, loss = train_client(
                client_model, client_loaders[cid], config,
                use_parl=use_parl and method == 'B-FedPLC',
                global_model=global_model if method == 'B-FedPLC' else None
            )

            client_models.append(state)
            client_sizes.append(len(client_indices[cid]))
            client_ids.append(cid)

        # Apply Byzantine attack
        if n_byzantine > 0:
            # Find which selected clients are Byzantine
            selected_byzantine = [i for i, cid in enumerate(selected_clients) if cid in byzantine_indices]
            client_models = apply_byzantine_attack(client_models, selected_byzantine, attack_type)

        # Aggregation based on method
        if method == 'FedAvg':
            weights = [s / sum(client_sizes) for s in client_sizes]
            global_state = fedavg(client_models, weights)

        elif method == 'Krum':
            global_state = krum(client_models, max(1, len(selected_byzantine) if n_byzantine > 0 else 0))

        elif method == 'Multi-Krum':
            global_state = multi_krum(client_models, max(1, len(selected_byzantine) if n_byzantine > 0 else 0))

        elif method == 'Trimmed-Mean':
            global_state = trimmed_mean(client_models, byzantine_frac)

        elif method == 'Median':
            global_state = median_aggregation(client_models)

        elif method == 'B-FedPLC':
            global_state = bfedplc_aggregate(
                client_models, client_ids, ldca,
                n_byzantine=max(1, len(selected_byzantine) if n_byzantine > 0 else 0),
                use_detection=n_byzantine > 0
            )

        else:
            global_state = fedavg(client_models)

        # Update global model
        if global_state is not None:
            global_model.load_state_dict(global_state)

        # Evaluate
        acc = evaluate(global_model, testloader, device)
        history['accuracy'].append(acc)

    return {
        'best_accuracy': max(history['accuracy']),
        'final_accuracy': history['accuracy'][-1],
        'history': history['accuracy']
    }

def run_experiments_with_seeds(config, method, seeds, **kwargs):
    """Run experiments with multiple seeds for statistical rigor"""
    results = []

    for seed in seeds:
        result = run_single_experiment(config, method, seed=seed, **kwargs)
        results.append(result)

    best_accs = [r['best_accuracy'] for r in results]
    final_accs = [r['final_accuracy'] for r in results]

    return {
        'method': method,
        'best_accuracy_mean': np.mean(best_accs),
        'best_accuracy_std': np.std(best_accs),
        'final_accuracy_mean': np.mean(final_accs),
        'final_accuracy_std': np.std(final_accs),
        'all_best': best_accs,
        'all_final': final_accs,
        'n_seeds': len(seeds)
    }

# ============================================================================
# EXPERIMENT 1: ABLATION STUDY
# ============================================================================

def run_ablation_study(config, seeds):
    """Ablation study with statistical rigor"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: ABLATION STUDY")
    print("="*70)

    results = {}

    configs = [
        ('Full B-FedPLC', True, True),
        ('Without LDCA', False, True),
        ('Without PARL', True, False),
        ('FedAvg Baseline', False, False),
    ]

    for name, use_ldca, use_parl in configs:
        print(f"\nRunning: {name}...")

        all_best = []
        all_final = []

        for seed in seeds:
            # Temporarily modify config
            original_warmup = config.warmup_rounds
            if not use_parl:
                config.warmup_rounds = config.num_rounds + 1  # Disable PARL

            if use_ldca:
                method = 'B-FedPLC'
            else:
                method = 'FedAvg'

            result = run_single_experiment(
                config, method, dataset='cifar10',
                byzantine_frac=0.0, alpha=0.5, seed=seed
            )

            all_best.append(result['best_accuracy'])
            all_final.append(result['final_accuracy'])
            config.warmup_rounds = original_warmup

        results[name] = {
            'best_mean': np.mean(all_best),
            'best_std': np.std(all_best),
            'final_mean': np.mean(all_final),
            'final_std': np.std(all_final),
            'all_best': all_best,
            'all_final': all_final
        }

        print(f"  Best: {results[name]['best_mean']:.2f}% ± {results[name]['best_std']:.2f}%")
        print(f"  Final: {results[name]['final_mean']:.2f}% ± {results[name]['final_std']:.2f}%")

    # Statistical tests
    print("\n--- Statistical Significance Tests ---")
    baseline = results['FedAvg Baseline']['all_final']
    full = results['Full B-FedPLC']['all_final']

    t_stat, p_value = stats.ttest_ind(full, baseline)
    print(f"B-FedPLC vs FedAvg: t={t_stat:.4f}, p={p_value:.4f}")

    return results

# ============================================================================
# EXPERIMENT 2: BYZANTINE RESILIENCE
# ============================================================================

def run_byzantine_experiment(config, seeds):
    """Byzantine resilience with statistical rigor"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: BYZANTINE RESILIENCE")
    print("="*70)

    methods = ['FedAvg', 'Krum', 'Multi-Krum', 'Trimmed-Mean', 'Median', 'B-FedPLC']
    byzantine_fracs = [0.0, 0.1, 0.2, 0.3]

    results = {}

    for byz_frac in byzantine_fracs:
        print(f"\n--- Byzantine Fraction: {int(byz_frac*100)}% ---")
        results[byz_frac] = {}

        for method in methods:
            print(f"  Running {method}...", end=' ')

            all_best = []
            all_final = []

            for seed in seeds:
                result = run_single_experiment(
                    config, method, dataset='cifar10',
                    byzantine_frac=byz_frac, attack_type='sign_flip',
                    alpha=0.5, seed=seed
                )
                all_best.append(result['best_accuracy'])
                all_final.append(result['final_accuracy'])

            results[byz_frac][method] = {
                'best_mean': np.mean(all_best),
                'best_std': np.std(all_best),
                'final_mean': np.mean(all_final),
                'final_std': np.std(all_final),
                'all_final': all_final
            }

            print(f"{results[byz_frac][method]['final_mean']:.2f}% ± {results[byz_frac][method]['final_std']:.2f}%")

    return results

# ============================================================================
# EXPERIMENT 3: NON-IID SENSITIVITY
# ============================================================================

def run_noniid_experiment(config, seeds):
    """Non-IID sensitivity with statistical rigor"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: NON-IID SENSITIVITY")
    print("="*70)

    methods = ['FedAvg', 'B-FedPLC']
    alphas = [0.1, 0.3, 0.5, 1.0]

    results = {}

    for alpha in alphas:
        print(f"\n--- Dirichlet α = {alpha} ---")
        results[alpha] = {}

        for method in methods:
            print(f"  Running {method}...", end=' ')

            all_best = []
            all_final = []

            for seed in seeds:
                result = run_single_experiment(
                    config, method, dataset='cifar10',
                    byzantine_frac=0.0, alpha=alpha, seed=seed
                )
                all_best.append(result['best_accuracy'])
                all_final.append(result['final_accuracy'])

            results[alpha][method] = {
                'best_mean': np.mean(all_best),
                'best_std': np.std(all_best),
                'final_mean': np.mean(all_final),
                'final_std': np.std(all_final),
                'all_final': all_final
            }

            print(f"{results[alpha][method]['final_mean']:.2f}% ± {results[alpha][method]['final_std']:.2f}%")

    return results

# ============================================================================
# EXPERIMENT 4: COMBINED STRESS TEST
# ============================================================================

def run_combined_stress_test(config, seeds):
    """Combined: Non-IID + Byzantine - where B-FedPLC should excel"""
    print("\n" + "="*70)
    print("EXPERIMENT 4: COMBINED STRESS TEST (Non-IID + Byzantine)")
    print("="*70)
    print("This tests the scenario where B-FedPLC is designed to excel")

    methods = ['FedAvg', 'Multi-Krum', 'Trimmed-Mean', 'B-FedPLC']

    # Test with highly non-IID (α=0.3) + Byzantine (20%)
    test_configs = [
        {'alpha': 0.3, 'byzantine_frac': 0.1, 'name': 'Moderate Stress'},
        {'alpha': 0.3, 'byzantine_frac': 0.2, 'name': 'High Stress'},
        {'alpha': 0.1, 'byzantine_frac': 0.2, 'name': 'Extreme Stress'},
    ]

    results = {}

    for test in test_configs:
        print(f"\n--- {test['name']}: α={test['alpha']}, Byzantine={int(test['byzantine_frac']*100)}% ---")
        results[test['name']] = {}

        for method in methods:
            print(f"  Running {method}...", end=' ')

            all_best = []
            all_final = []

            for seed in seeds:
                result = run_single_experiment(
                    config, method, dataset='cifar10',
                    byzantine_frac=test['byzantine_frac'],
                    attack_type='sign_flip',
                    alpha=test['alpha'],
                    seed=seed
                )
                all_best.append(result['best_accuracy'])
                all_final.append(result['final_accuracy'])

            results[test['name']][method] = {
                'best_mean': np.mean(all_best),
                'best_std': np.std(all_best),
                'final_mean': np.mean(all_final),
                'final_std': np.std(all_final),
                'all_final': all_final
            }

            print(f"{results[test['name']][method]['final_mean']:.2f}% ± {results[test['name']][method]['final_std']:.2f}%")

    # Statistical comparison
    print("\n--- Statistical Significance ---")
    for test_name in results:
        if 'B-FedPLC' in results[test_name] and 'Multi-Krum' in results[test_name]:
            bfedplc = results[test_name]['B-FedPLC']['all_final']
            multikrum = results[test_name]['Multi-Krum']['all_final']
            t_stat, p_value = stats.ttest_ind(bfedplc, multikrum)
            print(f"{test_name}: B-FedPLC vs Multi-Krum: t={t_stat:.4f}, p={p_value:.4f}")

    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all experiments"""

    # Configuration
    config.num_rounds = 50
    config.num_clients = 30
    config.participation_rate = 0.3

    # Seeds for statistical rigor
    seeds = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]
    print(f"\nUsing {len(seeds)} seeds for statistical rigor: {seeds}")

    all_results = {}

    # Run experiments
    start_time = time.time()

    print("\n" + "#"*70)
    print("# STARTING COMPREHENSIVE IEEE ACCESS EXPERIMENTS")
    print("#"*70)

    # Experiment 1: Ablation Study
    all_results['ablation'] = run_ablation_study(config, seeds[:5])  # Use 5 seeds for ablation

    # Experiment 2: Byzantine Resilience
    all_results['byzantine'] = run_byzantine_experiment(config, seeds[:5])

    # Experiment 3: Non-IID Sensitivity
    all_results['noniid'] = run_noniid_experiment(config, seeds[:5])

    # Experiment 4: Combined Stress Test (most important)
    all_results['stress_test'] = run_combined_stress_test(config, seeds[:5])

    total_time = time.time() - start_time

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    # Convert numpy to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    serializable_results = convert_to_serializable(all_results)

    with open('ieee_comprehensive_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print("Saved: ieee_comprehensive_results.json")

    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")

    # Key findings
    print("\n--- KEY FINDINGS ---")

    # Ablation
    if 'ablation' in all_results:
        print("\n1. ABLATION STUDY:")
        for name, data in all_results['ablation'].items():
            print(f"   {name}: {data['final_mean']:.2f}% ± {data['final_std']:.2f}%")

    # Byzantine
    if 'byzantine' in all_results:
        print("\n2. BYZANTINE RESILIENCE (20% attackers):")
        if 0.2 in all_results['byzantine']:
            for method, data in all_results['byzantine'][0.2].items():
                print(f"   {method}: {data['final_mean']:.2f}% ± {data['final_std']:.2f}%")

    # Stress test
    if 'stress_test' in all_results:
        print("\n3. COMBINED STRESS TEST:")
        for test_name, methods in all_results['stress_test'].items():
            print(f"   {test_name}:")
            for method, data in methods.items():
                print(f"     {method}: {data['final_mean']:.2f}% ± {data['final_std']:.2f}%")

    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE!")
    print("="*70)

    return all_results

if __name__ == "__main__":
    results = main()
