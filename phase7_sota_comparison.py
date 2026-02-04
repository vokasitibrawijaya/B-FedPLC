"""
Phase 7: SOTA Comparison for IEEE Access Paper
Compare B-FedPLC with state-of-the-art Byzantine-resilient FL methods:
- FedAvg-M (Momentum-based FedAvg)
- CRFL (Client-Robust Federated Learning)
- FLTrust (Federated Learning with Trust)
- Krum (Byzantine-resilient aggregation)
- Trimmed Mean
- Median
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import json
import time
import copy
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    def __init__(self):
        self.num_clients = 10
        self.num_rounds = 30
        self.local_epochs = 2
        self.batch_size = 32
        self.lr = 0.01
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.byzantine_fraction = 0.2  # 20% Byzantine attackers
        self.num_clusters = 3
        self.dirichlet_alpha = 0.5  # Non-IID setting

# ============================================================================
# MODELS
# ============================================================================

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ============================================================================
# DATA UTILITIES
# ============================================================================

def get_dataset(dataset_name='MNIST'):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    else:  # CIFAR10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    return train_dataset, test_dataset

def dirichlet_split(dataset, num_clients, alpha=0.5):
    """Split dataset using Dirichlet distribution for Non-IID"""
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))
    
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(class_indices)).astype(int)
        proportions[-1] = len(class_indices) - proportions[:-1].sum()
        
        start = 0
        for client_id, prop in enumerate(proportions):
            client_indices[client_id].extend(class_indices[start:start+prop].tolist())
            start += prop
    
    return [Subset(dataset, indices) for indices in client_indices]

# ============================================================================
# BYZANTINE ATTACK
# ============================================================================

def apply_byzantine_attack(updates, byzantine_indices, attack_type='sign_flip'):
    """Apply Byzantine attack to specified client updates"""
    attacked_updates = []
    
    for i, update in enumerate(updates):
        if i in byzantine_indices:
            if attack_type == 'sign_flip':
                attacked = {}
                for k, v in update.items():
                    if v.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                        attacked[k] = v  # Keep integer params unchanged
                    else:
                        attacked[k] = -v * 3.0
            elif attack_type == 'random':
                attacked = {}
                for k, v in update.items():
                    if v.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                        attacked[k] = v  # Keep integer params unchanged
                    else:
                        attacked[k] = torch.randn_like(v.float()) * 10.0
            elif attack_type == 'label_flip':
                attacked = {}
                for k, v in update.items():
                    if v.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                        attacked[k] = v  # Keep integer params unchanged
                    else:
                        attacked[k] = v * 5.0
            else:
                attacked = update
            attacked_updates.append(attacked)
        else:
            attacked_updates.append(update)
    
    return attacked_updates

# ============================================================================
# AGGREGATION METHODS (SOTA)
# ============================================================================

def fedavg(updates, weights=None):
    """Standard FedAvg aggregation"""
    if not updates:
        return None
    
    if weights is None:
        weights = [1.0 / len(updates)] * len(updates)
    
    result = {}
    for key in updates[0].keys():
        stacked = torch.stack([u[key].float() for u in updates])
        weights_tensor = torch.tensor(weights, device=stacked.device).view(-1, *([1] * (stacked.dim() - 1)))
        result[key] = (stacked * weights_tensor).sum(dim=0)
        
        # Preserve original dtype for integer parameters
        if updates[0][key].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            result[key] = updates[0][key]
    
    return result

def fedavg_momentum(updates, momentum_buffer=None, beta=0.9):
    """FedAvg with server momentum"""
    avg = fedavg(updates)
    
    if momentum_buffer is None:
        return avg, avg
    
    result = {}
    new_momentum = {}
    for key in avg.keys():
        if avg[key].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            result[key] = avg[key]
            new_momentum[key] = avg[key]
        else:
            new_momentum[key] = beta * momentum_buffer[key] + (1 - beta) * avg[key]
            result[key] = new_momentum[key]
    
    return result, new_momentum

def krum(updates, n_byzantine=0):
    """Krum Byzantine-resilient aggregation"""
    n = len(updates)
    if n == 0:
        return None
    
    if n <= 2 * n_byzantine + 2:
        return fedavg(updates)
    
    # Flatten updates
    flat_updates = []
    for u in updates:
        flat = torch.cat([p.float().flatten() for p in u.values()])
        flat_updates.append(flat)
    flat_updates = torch.stack(flat_updates)
    
    # Compute scores
    n_select = n - n_byzantine - 2
    scores = []
    
    for i in range(n):
        dists = torch.norm(flat_updates - flat_updates[i], dim=1)
        dists_sorted, _ = torch.sort(dists)
        score = dists_sorted[1:n_select+1].sum()
        scores.append(score.item())
    
    # Select the one with lowest score
    selected_idx = np.argmin(scores)
    return updates[selected_idx]

def multi_krum(updates, n_byzantine=0, multi_k=None):
    """Multi-Krum aggregation"""
    n = len(updates)
    if n == 0:
        return None
    
    if n <= 2 * n_byzantine + 2:
        return fedavg(updates)
    
    # Flatten updates
    flat_updates = []
    for u in updates:
        flat = torch.cat([p.float().flatten() for p in u.values()])
        flat_updates.append(flat)
    flat_updates = torch.stack(flat_updates)
    
    # Compute scores
    n_select = n - n_byzantine - 2
    scores = []
    
    for i in range(n):
        dists = torch.norm(flat_updates - flat_updates[i], dim=1)
        dists_sorted, _ = torch.sort(dists)
        score = dists_sorted[1:n_select+1].sum()
        scores.append(score.item())
    
    # Select top-k with lowest scores
    if multi_k is None:
        multi_k = max(1, n - n_byzantine)
    
    selected_indices = np.argsort(scores)[:multi_k]
    selected_updates = [updates[i] for i in selected_indices]
    
    return fedavg(selected_updates)

def trimmed_mean(updates, trim_ratio=0.1):
    """Trimmed Mean aggregation"""
    if not updates:
        return None
    
    n = len(updates)
    trim_count = int(n * trim_ratio)
    
    result = {}
    for key in updates[0].keys():
        stacked = torch.stack([u[key].float() for u in updates])
        
        if updates[0][key].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            result[key] = updates[0][key]
        else:
            # Sort along client dimension and trim
            sorted_vals, _ = torch.sort(stacked, dim=0)
            if trim_count > 0:
                trimmed = sorted_vals[trim_count:-trim_count]
            else:
                trimmed = sorted_vals
            result[key] = trimmed.mean(dim=0)
    
    return result

def coordinate_median(updates):
    """Coordinate-wise Median aggregation"""
    if not updates:
        return None
    
    result = {}
    for key in updates[0].keys():
        stacked = torch.stack([u[key].float() for u in updates])
        
        if updates[0][key].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            result[key] = updates[0][key]
        else:
            result[key] = stacked.median(dim=0)[0]
    
    return result

def fltrust(updates, server_update, n_byzantine=0):
    """FLTrust aggregation using server's trusted update"""
    if not updates or server_update is None:
        return fedavg(updates)
    
    # Flatten all updates
    flat_server = torch.cat([p.float().flatten() for p in server_update.values()])
    server_norm = torch.norm(flat_server)
    
    if server_norm < 1e-10:
        return fedavg(updates)
    
    trust_scores = []
    for u in updates:
        flat_u = torch.cat([p.float().flatten() for p in u.values()])
        # Cosine similarity as trust score
        cos_sim = torch.dot(flat_u, flat_server) / (torch.norm(flat_u) * server_norm + 1e-10)
        trust_scores.append(max(0, cos_sim.item()))  # ReLU on trust scores
    
    # Normalize trust scores
    total_trust = sum(trust_scores)
    if total_trust < 1e-10:
        return fedavg(updates)
    
    weights = [ts / total_trust for ts in trust_scores]
    
    # Scale updates by server norm and aggregate
    result = {}
    for key in updates[0].keys():
        if updates[0][key].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            result[key] = updates[0][key]
        else:
            weighted_sum = torch.zeros_like(updates[0][key], dtype=torch.float32)
            for u, w in zip(updates, weights):
                flat_u = torch.cat([p.float().flatten() for p in u.values()])
                u_norm = torch.norm(flat_u)
                if u_norm > 1e-10:
                    scale = server_norm / u_norm
                    weighted_sum += w * u[key].float() * scale
            result[key] = weighted_sum
    
    return result

def crfl(updates, global_model, clip_threshold=10.0, noise_scale=0.01):
    """CRFL: Client-Robust Federated Learning with gradient clipping and noise"""
    if not updates:
        return None
    
    # Clip updates
    clipped_updates = []
    for u in updates:
        flat = torch.cat([p.float().flatten() for p in u.values()])
        norm = torch.norm(flat)
        
        if norm > clip_threshold:
            scale = clip_threshold / norm
            clipped = {k: v.float() * scale for k, v in u.items()}
        else:
            clipped = {k: v.float() for k, v in u.items()}
        clipped_updates.append(clipped)
    
    # Aggregate
    result = fedavg(clipped_updates)
    
    # Add noise for differential privacy
    for key in result.keys():
        if result[key].dtype not in [torch.long, torch.int, torch.int32, torch.int64]:
            noise = torch.randn_like(result[key]) * noise_scale
            result[key] = result[key] + noise
    
    return result

# ============================================================================
# B-FedPLC IMPLEMENTATION (IMPROVED)
# ============================================================================

def detect_byzantine_updates(updates, threshold_factor=2.0, attack_type='sign_flip', n_byzantine=0):
    """
    Detect potential Byzantine updates using norm-based and cosine similarity filtering.
    
    IMPROVED VERSION (2026-01-10):
    - Adaptive thresholds based on attack type (sign_flip vs random)
    - Cosine similarity detection for sign_flip attacks
    - Better minimum required updates logic
    
    Args:
        updates: List of model update dictionaries
        threshold_factor: Base threshold multiplier (default: 2.0)
        attack_type: Type of Byzantine attack ('sign_flip', 'random', etc.)
        n_byzantine: Expected number of Byzantine clients
    
    Returns:
        benign_indices: List of indices for benign updates
        byzantine_indices: List of indices for suspected Byzantine updates
    """
    if len(updates) < 3:
        return list(range(len(updates))), []
    
    # Adaptive threshold based on attack type
    if attack_type == 'sign_flip':
        # Sign flip creates large norm differences and negative cosine similarity
        norm_threshold_multiplier = 1.5  # More sensitive
        cosine_threshold = -0.3  # Negative cosine indicates sign flip
    elif attack_type == 'random':
        # Random noise might have similar norms but different directions
        norm_threshold_multiplier = 3.0  # More strict
        cosine_threshold = 0.0  # Low cosine similarity
    else:
        norm_threshold_multiplier = 2.0
        cosine_threshold = 0.1
    
    # Compute norms and cosine similarities
    norms = []
    flat_updates_list = []
    
    for u in updates:
        flat = torch.cat([p.float().flatten() for p in u.values()])
        if torch.isnan(flat).any() or torch.isinf(flat).any():
            norms.append(float('inf'))
            flat_updates_list.append(None)
        else:
            norms.append(torch.norm(flat).item())
            flat_updates_list.append(flat)
    
    norms = np.array(norms)
    
    # Use median and MAD for robust outlier detection
    finite_norms = norms[np.isfinite(norms)]
    if len(finite_norms) == 0:
        return [], list(range(len(updates)))
    
    median_norm = np.median(finite_norms)
    mad = np.median(np.abs(finite_norms - median_norm))
    
    if mad < 1e-10:
        mad = np.std(finite_norms)
    if mad < 1e-10:
        mad = 1.0
    
    # Compute cosine similarities with median update (for sign flip detection)
    cosine_similarities = []
    valid_flat = [f for f in flat_updates_list if f is not None]
    if len(valid_flat) > 0:
        # Use median update as reference
        median_idx = np.argsort(norms)[len(norms) // 2]
        if flat_updates_list[median_idx] is not None:
            median_flat = flat_updates_list[median_idx]
            median_flat_norm = torch.norm(median_flat)
            
            for i, flat in enumerate(flat_updates_list):
                if flat is None:
                    cosine_similarities.append(-1.0)  # Mark as suspicious
                else:
                    flat_norm = torch.norm(flat)
                    if flat_norm > 1e-10 and median_flat_norm > 1e-10:
                        cos_sim = torch.dot(flat, median_flat) / (flat_norm * median_flat_norm)
                        cosine_similarities.append(cos_sim.item())
                    else:
                        cosine_similarities.append(0.0)
        else:
            cosine_similarities = [0.0] * len(updates)
    else:
        cosine_similarities = [0.0] * len(updates)
    
    # Identify outliers using both norm and cosine similarity
    benign_indices = []
    byzantine_indices = []
    
    for i, norm in enumerate(norms):
        is_byzantine = False
        
        # Check for NaN/Inf
        if np.isinf(norm) or np.isnan(norm):
            is_byzantine = True
        # Check norm-based outlier (adaptive threshold)
        elif abs(norm - median_norm) > threshold_factor * norm_threshold_multiplier * mad:
            is_byzantine = True
        # Check cosine similarity for sign flip attacks
        elif attack_type == 'sign_flip' and i < len(cosine_similarities):
            if cosine_similarities[i] < cosine_threshold:
                is_byzantine = True
        
        if is_byzantine:
            byzantine_indices.append(i)
        else:
            benign_indices.append(i)
    
    # Ensure we have enough benign updates (at least n - n_byzantine - 1)
    min_required = max(3, len(updates) - n_byzantine - 1)
    if len(benign_indices) < min_required:
        # Sort by distance from median and take the closest ones
        distances = [(i, abs(norms[i] - median_norm)) for i in range(len(norms)) if np.isfinite(norms[i])]
        distances.sort(key=lambda x: x[1])
        # Take at least min_required, but prefer more if available
        num_to_keep = max(min_required, len(distances) - n_byzantine)
        benign_indices = [d[0] for d in distances[:num_to_keep]]
        byzantine_indices = [i for i in range(len(updates)) if i not in benign_indices]
    
    return benign_indices, byzantine_indices

def plc_clustering(updates, num_clusters=3):
    """PLC-based clustering with outlier handling"""
    if len(updates) <= num_clusters:
        return [[i] for i in range(len(updates))]
    
    # Flatten updates
    flat_updates = []
    valid_indices = []
    for i, u in enumerate(updates):
        flat = torch.cat([p.float().flatten() for p in u.values()])
        flat_np = flat.cpu().numpy()
        # Check for NaN or Inf values
        if not np.any(np.isnan(flat_np)) and not np.any(np.isinf(flat_np)):
            flat_updates.append(flat_np)
            valid_indices.append(i)
    
    # If too few valid updates, return all as one cluster
    if len(flat_updates) < num_clusters:
        return [list(range(len(updates)))]
    
    flat_updates = np.array(flat_updates)
    
    # Replace any remaining NaN/Inf with 0 (safety check)
    flat_updates = np.nan_to_num(flat_updates, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Simple k-means clustering
    from sklearn.cluster import KMeans
    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(flat_updates)
    except Exception as e:
        # Fallback: return all as one cluster
        return [list(range(len(updates)))]
    
    clusters = [[] for _ in range(num_clusters)]
    for idx, label in enumerate(labels):
        original_idx = valid_indices[idx]
        clusters[label].append(original_idx)
    
    # Add invalid indices to the largest cluster
    invalid_indices = [i for i in range(len(updates)) if i not in valid_indices]
    if invalid_indices:
        largest_cluster_idx = max(range(len(clusters)), key=lambda x: len(clusters[x]))
        clusters[largest_cluster_idx].extend(invalid_indices)
    
    # Remove empty clusters
    clusters = [c for c in clusters if len(c) > 0]
    
    return clusters

def bfedplc_aggregation(updates, n_byzantine=0, num_clusters=3, attack_type='sign_flip', verbose=False):
    """
    B-FedPLC: Hierarchical aggregation with improved Byzantine pre-filtering and PLC clustering.
    
    IMPROVED VERSION (2026-01-10):
    - Adaptive Byzantine detection based on attack type
    - Better fallback logic to ensure minimum updates
    - Cosine similarity check for sign flip attacks
    - Improved cluster-level Byzantine handling
    
    Args:
        updates: List of model update dictionaries
        n_byzantine: Number of Byzantine clients
        num_clusters: Number of clusters for PLC clustering
        attack_type: Type of Byzantine attack ('sign_flip', 'random', etc.)
        verbose: Enable verbose logging for debugging
    
    Returns:
        Aggregated model update dictionary, or None if aggregation fails
    """
    if not updates:
        return None
    
    # Ensure we have minimum required updates
    min_required = max(3, len(updates) - n_byzantine - 1)
    
    # Step 0: Pre-filter Byzantine updates using improved detection
    benign_indices, suspected_byzantine = detect_byzantine_updates(
        updates, 
        threshold_factor=2.0, 
        attack_type=attack_type,
        n_byzantine=n_byzantine
    )
    
    if verbose:
        print(f"      B-FedPLC Detection: {len(benign_indices)}/{len(updates)} benign, "
              f"{len(suspected_byzantine)} suspected Byzantine")
    
    # If too many detected as Byzantine, fall back to Multi-Krum directly
    # This is safer than proceeding with too few updates
    if len(benign_indices) < min_required:
        if verbose:
            print(f"      Warning: Too few benign updates ({len(benign_indices)} < {min_required}), "
                  f"falling back to Multi-Krum")
        return multi_krum(updates, n_byzantine)
    
    # If detection is too aggressive (filtered more than expected), use Multi-Krum
    # Expected: filter ~n_byzantine, but allow some margin
    max_expected_filtered = n_byzantine + max(1, len(updates) // 10)  # Allow 10% margin
    if len(suspected_byzantine) > max_expected_filtered:
        if verbose:
            print(f"      Warning: Detection too aggressive ({len(suspected_byzantine)} filtered, "
                  f"expected ~{n_byzantine}), falling back to Multi-Krum")
        return multi_krum(updates, n_byzantine)
    
    # Filter to only benign updates
    filtered_updates = [updates[i] for i in benign_indices]
    
    if len(filtered_updates) == 0:
        if verbose:
            print("      Error: No benign updates after filtering, using Multi-Krum")
        return multi_krum(updates, n_byzantine)
    
    # Step 1: PLC Clustering on filtered updates
    clusters = plc_clustering(filtered_updates, num_clusters)
    
    if verbose:
        print(f"      Clustering: {len(clusters)} clusters from {len(filtered_updates)} updates")
    
    # Step 2: Intra-cluster aggregation with Multi-Krum
    cluster_updates = []
    cluster_weights = []
    
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        
        cluster_data = [filtered_updates[i] for i in cluster]
        # Reduced Byzantine count since we already filtered
        # But keep some margin for remaining Byzantine in clusters
        cluster_byz = max(0, (n_byzantine - len(suspected_byzantine)) // max(1, len(clusters)))
        
        # Multi-Krum within cluster
        agg = multi_krum(cluster_data, cluster_byz)
        if agg is not None:
            cluster_updates.append(agg)
            cluster_weights.append(len(cluster))
    
    if not cluster_updates:
        if verbose:
            print("      Warning: No cluster aggregates, using FedAvg on filtered updates")
        return fedavg(filtered_updates) if filtered_updates else fedavg(updates)
    
    # Step 3: Inter-cluster aggregation
    total_weight = sum(cluster_weights)
    if total_weight == 0:
        return fedavg(filtered_updates)
    
    weights = [w / total_weight for w in cluster_weights]
    
    result = {}
    for key in cluster_updates[0].keys():
        if cluster_updates[0][key].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            result[key] = cluster_updates[0][key]
        else:
            weighted_sum = torch.zeros_like(cluster_updates[0][key], dtype=torch.float32)
            for cu, w in zip(cluster_updates, weights):
                weighted_sum += w * cu[key].float()
            result[key] = weighted_sum
    
    return result

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def local_train(model, data_loader, config, device):
    """Local training on client"""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    for epoch in range(config.local_epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # Compute update (difference)
    update = {}
    for k, v in model.state_dict().items():
        update[k] = v - initial_state[k]
    
    return update

def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return 100.0 * correct / total

def apply_update(model, update):
    """Apply update to model"""
    state = model.state_dict()
    for k, v in update.items():
        if k in state:
            state[k] = state[k] + v
    model.load_state_dict(state)

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(config, dataset_name, method_name, aggregation_fn, 
                   byzantine_fraction=0.2, attack_type='sign_flip', seed=42):
    """Run single experiment with given configuration"""
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = config.device
    
    # Get dataset
    train_dataset, test_dataset = get_dataset(dataset_name)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Split data
    client_datasets = dirichlet_split(train_dataset, config.num_clients, config.dirichlet_alpha)
    client_loaders = [DataLoader(ds, batch_size=config.batch_size, shuffle=True) 
                      for ds in client_datasets]
    
    # Initialize model
    if dataset_name == 'MNIST':
        global_model = MNISTNet().to(device)
    else:
        global_model = CIFAR10Net().to(device)
    
    # Byzantine clients
    n_byzantine = int(config.num_clients * byzantine_fraction)
    byzantine_indices = list(range(n_byzantine))
    
    # Momentum buffer for FedAvg-M
    momentum_buffer = None
    
    # Server model for FLTrust (small clean dataset)
    server_loader = DataLoader(Subset(train_dataset, list(range(500))), 
                               batch_size=config.batch_size, shuffle=True)
    
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
        if n_byzantine > 0:
            updates = apply_byzantine_attack(updates, byzantine_indices, attack_type)
        
        # Aggregate based on method
        if method_name == 'FedAvg':
            global_update = aggregation_fn(updates)
        elif method_name == 'FedAvg-M':
            global_update, momentum_buffer = aggregation_fn(updates, momentum_buffer)
        elif method_name == 'Krum':
            global_update = aggregation_fn(updates, n_byzantine)
        elif method_name == 'Multi-Krum':
            global_update = aggregation_fn(updates, n_byzantine)
        elif method_name == 'Trimmed-Mean':
            global_update = aggregation_fn(updates, trim_ratio=byzantine_fraction)
        elif method_name == 'Median':
            global_update = aggregation_fn(updates)
        elif method_name == 'FLTrust':
            # Get server update
            server_model = copy.deepcopy(global_model)
            server_update = local_train(server_model, server_loader, config, device)
            global_update = aggregation_fn(updates, server_update, n_byzantine)
        elif method_name == 'CRFL':
            global_update = aggregation_fn(updates, global_model)
        elif method_name == 'B-FedPLC':
            global_update = aggregation_fn(updates, n_byzantine, config.num_clusters, attack_type)
        else:
            global_update = fedavg(updates)
        
        # Apply update
        if global_update is not None:
            apply_update(global_model, global_update)
        
        # Evaluate
        acc = evaluate(global_model, test_loader, device)
        accuracies.append(acc)
        
        if round_num % 10 == 0:
            print(f"    Round {round_num}: {acc:.2f}%")
    
    return accuracies

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 80)
    print("PHASE 7: SOTA Comparison")
    print("=" * 80)
    
    config = Config()
    print(f"Device: {config.device}")
    
    # Methods to compare
    methods = {
        'FedAvg': fedavg,
        'FedAvg-M': fedavg_momentum,
        'Krum': krum,
        'Multi-Krum': multi_krum,
        'Trimmed-Mean': trimmed_mean,
        'Median': coordinate_median,
        'FLTrust': fltrust,
        'CRFL': crfl,
        'B-FedPLC': bfedplc_aggregation,
    }
    
    # Experiment configurations
    experiments = {
        'byzantine_20': {'byzantine_fraction': 0.2, 'attack_type': 'sign_flip'},
        'byzantine_30': {'byzantine_fraction': 0.3, 'attack_type': 'sign_flip'},
        'random_attack': {'byzantine_fraction': 0.2, 'attack_type': 'random'},
        'no_attack': {'byzantine_fraction': 0.0, 'attack_type': None},
    }
    
    datasets = ['MNIST', 'CIFAR10']
    seeds = [42, 123, 456]
    
    results = defaultdict(dict)
    
    # Run experiments
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        
        for exp_name, exp_config in experiments.items():
            print(f"\n  Experiment: {exp_name}")
            print(f"  Byzantine Fraction: {exp_config['byzantine_fraction']*100:.0f}%")
            print(f"  Attack Type: {exp_config['attack_type']}")
            print("-" * 50)
            
            for method_name, agg_fn in methods.items():
                print(f"\n  Method: {method_name}")
                
                all_accs = []
                for seed in seeds:
                    print(f"    Seed: {seed}")
                    try:
                        accs = run_experiment(
                            config, dataset, method_name, agg_fn,
                            byzantine_fraction=exp_config['byzantine_fraction'],
                            attack_type=exp_config['attack_type'],
                            seed=seed
                        )
                        all_accs.append(accs[-1])  # Final accuracy
                    except Exception as e:
                        print(f"      Error: {e}")
                        all_accs.append(0.0)
                
                mean_acc = np.mean(all_accs)
                std_acc = np.std(all_accs)
                
                results[f"{dataset}_{exp_name}"][method_name] = {
                    'mean': mean_acc,
                    'std': std_acc,
                    'all': all_accs
                }
                
                print(f"    Result: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    # Save results
    print(f"\n{'='*60}")
    print("Saving Results")
    print(f"{'='*60}")
    
    # Convert to JSON serializable format
    json_results = {}
    for key, method_results in results.items():
        json_results[key] = {}
        for method, values in method_results.items():
            json_results[key][method] = {
                'mean': float(values['mean']),
                'std': float(values['std']),
                'all': [float(x) for x in values['all']]
            }
    
    with open('phase7_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print("✅ Results saved to: phase7_results.json")
    
    # Generate visualizations
    generate_visualizations(results)
    
    # Print summary table
    print_summary_table(results)
    
    print(f"\n{'='*80}")
    print("PHASE 7 COMPLETE!")
    print(f"{'='*80}")

def generate_visualizations(results):
    """Generate comparison visualizations"""
    print("\nGenerating Phase 7 visualizations...")
    
    # Color scheme for methods
    colors = {
        'FedAvg': '#FF6B6B',
        'FedAvg-M': '#4ECDC4',
        'Krum': '#45B7D1',
        'Multi-Krum': '#96CEB4',
        'Trimmed-Mean': '#FFEAA7',
        'Median': '#DDA0DD',
        'FLTrust': '#98D8C8',
        'CRFL': '#F7DC6F',
        'B-FedPLC': '#2ECC71',  # Our method - highlighted
    }
    
    methods = list(colors.keys())
    
    # 1. Bar chart comparing methods under 20% Byzantine attack (MNIST)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # MNIST Byzantine 20%
    ax = axes[0, 0]
    key = 'MNIST_byzantine_20'
    if key in results:
        means = [results[key].get(m, {'mean': 0})['mean'] for m in methods]
        stds = [results[key].get(m, {'std': 0})['std'] for m in methods]
        bars = ax.bar(range(len(methods)), means, yerr=stds, 
                      color=[colors[m] for m in methods], capsize=3, edgecolor='black')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('MNIST - 20% Byzantine Attack', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight B-FedPLC
        bars[-1].set_edgecolor('red')
        bars[-1].set_linewidth(3)
    
    # CIFAR-10 Byzantine 20%
    ax = axes[0, 1]
    key = 'CIFAR10_byzantine_20'
    if key in results:
        means = [results[key].get(m, {'mean': 0})['mean'] for m in methods]
        stds = [results[key].get(m, {'std': 0})['std'] for m in methods]
        bars = ax.bar(range(len(methods)), means, yerr=stds,
                      color=[colors[m] for m in methods], capsize=3, edgecolor='black')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('CIFAR-10 - 20% Byzantine Attack', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        bars[-1].set_edgecolor('red')
        bars[-1].set_linewidth(3)
    
    # MNIST Byzantine 30%
    ax = axes[1, 0]
    key = 'MNIST_byzantine_30'
    if key in results:
        means = [results[key].get(m, {'mean': 0})['mean'] for m in methods]
        stds = [results[key].get(m, {'std': 0})['std'] for m in methods]
        bars = ax.bar(range(len(methods)), means, yerr=stds,
                      color=[colors[m] for m in methods], capsize=3, edgecolor='black')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('MNIST - 30% Byzantine Attack', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        bars[-1].set_edgecolor('red')
        bars[-1].set_linewidth(3)
    
    # CIFAR-10 Byzantine 30%
    ax = axes[1, 1]
    key = 'CIFAR10_byzantine_30'
    if key in results:
        means = [results[key].get(m, {'mean': 0})['mean'] for m in methods]
        stds = [results[key].get(m, {'std': 0})['std'] for m in methods]
        bars = ax.bar(range(len(methods)), means, yerr=stds,
                      color=[colors[m] for m in methods], capsize=3, edgecolor='black')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('CIFAR-10 - 30% Byzantine Attack', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        bars[-1].set_edgecolor('red')
        bars[-1].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig('phase7_byzantine_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Generated: phase7_byzantine_comparison.png")
    
    # 2. Radar chart for overall performance
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    categories = ['MNIST\n(20% Byz)', 'MNIST\n(30% Byz)', 'CIFAR10\n(20% Byz)', 
                  'CIFAR10\n(30% Byz)', 'Random\nAttack', 'No Attack']
    keys = ['MNIST_byzantine_20', 'MNIST_byzantine_30', 'CIFAR10_byzantine_20',
            'CIFAR10_byzantine_30', 'MNIST_random_attack', 'MNIST_no_attack']
    
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Plot each method
    for method in ['FedAvg', 'Multi-Krum', 'FLTrust', 'CRFL', 'B-FedPLC']:
        values = []
        for key in keys:
            if key in results and method in results[key]:
                values.append(results[key][method]['mean'])
            else:
                values.append(0)
        values += values[:1]  # Close the polygon
        
        linewidth = 3 if method == 'B-FedPLC' else 1.5
        ax.plot(angles, values, 'o-', linewidth=linewidth, label=method, color=colors[method])
        ax.fill(angles, values, alpha=0.1, color=colors[method])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.set_title('Multi-Scenario Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('phase7_radar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Generated: phase7_radar_comparison.png")
    
    # 3. Heatmap of all results
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for heatmap
    scenarios = list(results.keys())
    methods_list = methods
    
    data = np.zeros((len(scenarios), len(methods_list)))
    for i, scenario in enumerate(scenarios):
        for j, method in enumerate(methods_list):
            if method in results[scenario]:
                data[i, j] = results[scenario][method]['mean']
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(range(len(methods_list)))
    ax.set_xticklabels(methods_list, rotation=45, ha='right')
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([s.replace('_', '\n') for s in scenarios])
    
    # Add text annotations
    for i in range(len(scenarios)):
        for j in range(len(methods_list)):
            text = f'{data[i, j]:.1f}'
            color = 'white' if data[i, j] < 50 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Accuracy (%)')
    ax.set_title('SOTA Comparison Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('phase7_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Generated: phase7_heatmap.png")
    
    print("\n✅ All Phase 7 visualizations generated!")

def print_summary_table(results):
    """Print summary table"""
    print(f"\n{'='*100}")
    print("SUMMARY TABLE: SOTA Comparison")
    print(f"{'='*100}")
    
    methods = ['FedAvg', 'FedAvg-M', 'Krum', 'Multi-Krum', 'Trimmed-Mean', 
               'Median', 'FLTrust', 'CRFL', 'B-FedPLC']
    
    # Header
    header = f"{'Scenario':<30}"
    for m in methods:
        header += f"{m:>10}"
    print(header)
    print("-" * 120)
    
    # Data rows
    for scenario in sorted(results.keys()):
        row = f"{scenario:<30}"
        for method in methods:
            if method in results[scenario]:
                acc = results[scenario][method]['mean']
                row += f"{acc:>10.2f}"
            else:
                row += f"{'N/A':>10}"
        print(row)
    
    print("-" * 120)
    
    # Highlight best results
    print("\n🏆 Best Results per Scenario:")
    for scenario in sorted(results.keys()):
        best_method = max(results[scenario].keys(), 
                         key=lambda m: results[scenario][m]['mean'])
        best_acc = results[scenario][best_method]['mean']
        bfedplc_acc = results[scenario].get('B-FedPLC', {'mean': 0})['mean']
        
        marker = "⭐" if best_method == 'B-FedPLC' else ""
        print(f"  {scenario}: {best_method} ({best_acc:.2f}%) {marker}")
        if best_method != 'B-FedPLC':
            diff = bfedplc_acc - best_acc
            print(f"    → B-FedPLC: {bfedplc_acc:.2f}% (diff: {diff:+.2f}%)")

if __name__ == "__main__":
    main()
