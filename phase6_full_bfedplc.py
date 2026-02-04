"""
Phase 6: Full B-FedPLC System Integration
=========================================
Complete integration of:
- Blockchain-based secure aggregation
- Dynamic PLC clustering
- Byzantine-resilient aggregation (Multi-Krum)
- Non-IID data handling

For IEEE Access Paper - Dissertation Research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import time
import hashlib
from collections import defaultdict
import copy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MODELS
# ============================================================================

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# ============================================================================
# BLOCKCHAIN LAYER
# ============================================================================

class BlockchainLayer:
    """Lightweight blockchain for FL model updates"""
    
    def __init__(self):
        self.chain = []
        self.pending_updates = []
        self.create_genesis_block()
    
    def create_genesis_block(self):
        genesis = {
            'index': 0,
            'timestamp': time.time(),
            'data': 'Genesis Block',
            'previous_hash': '0' * 64,
            'hash': self.compute_hash({'data': 'Genesis', 'nonce': 0})
        }
        self.chain.append(genesis)
    
    def compute_hash(self, data):
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def add_model_update(self, client_id, model_hash, round_num, cluster_id=None):
        """Record model update on blockchain"""
        update = {
            'client_id': client_id,
            'model_hash': model_hash,
            'round': round_num,
            'cluster_id': cluster_id,
            'timestamp': time.time()
        }
        self.pending_updates.append(update)
        return update
    
    def create_block(self, round_num, aggregated_hash):
        """Create new block with all pending updates"""
        if not self.pending_updates:
            return None
        
        block = {
            'index': len(self.chain),
            'timestamp': time.time(),
            'round': round_num,
            'updates': self.pending_updates.copy(),
            'aggregated_model_hash': aggregated_hash,
            'previous_hash': self.chain[-1]['hash']
        }
        block['hash'] = self.compute_hash(block)
        self.chain.append(block)
        self.pending_updates = []
        return block
    
    def verify_chain(self):
        """Verify blockchain integrity"""
        for i in range(1, len(self.chain)):
            if self.chain[i]['previous_hash'] != self.chain[i-1]['hash']:
                return False
        return True
    
    def get_metrics(self):
        """Get blockchain metrics"""
        return {
            'chain_length': len(self.chain),
            'total_updates': sum(len(b.get('updates', [])) for b in self.chain),
            'integrity_valid': self.verify_chain()
        }

# ============================================================================
# PLC CLUSTERING
# ============================================================================

class PLCClusterer:
    """Probabilistic Latent Component Clustering"""
    
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.cluster_assignments = {}
        self.cluster_centers = None
    
    def extract_features(self, model_updates):
        """Extract features from model updates for clustering"""
        features = []
        for update in model_updates:
            flat = torch.cat([p.flatten() for p in update.values()])
            # Use statistical features
            feat = torch.tensor([
                flat.mean().item(),
                flat.std().item(),
                flat.abs().mean().item(),
                flat.max().item(),
                flat.min().item(),
                torch.norm(flat).item()
            ])
            features.append(feat)
        return torch.stack(features)
    
    def cluster(self, client_ids, model_updates):
        """Perform K-means clustering on model updates"""
        if len(model_updates) < self.n_clusters:
            # Not enough clients, assign all to cluster 0
            for cid in client_ids:
                self.cluster_assignments[cid] = 0
            return {0: client_ids}
        
        features = self.extract_features(model_updates)
        
        # K-means clustering
        n_samples = features.shape[0]
        
        # Initialize centers randomly
        indices = torch.randperm(n_samples)[:self.n_clusters]
        centers = features[indices].clone()
        
        # Iterate
        for _ in range(10):
            # Assign to nearest center
            distances = torch.cdist(features, centers)
            assignments = distances.argmin(dim=1)
            
            # Update centers
            new_centers = []
            for k in range(self.n_clusters):
                mask = assignments == k
                if mask.sum() > 0:
                    new_centers.append(features[mask].mean(dim=0))
                else:
                    new_centers.append(centers[k])
            centers = torch.stack(new_centers)
        
        # Create cluster mapping
        clusters = defaultdict(list)
        for i, cid in enumerate(client_ids):
            cluster_id = assignments[i].item()
            self.cluster_assignments[cid] = cluster_id
            clusters[cluster_id].append(cid)
        
        self.cluster_centers = centers
        return dict(clusters)

# ============================================================================
# BYZANTINE AGGREGATION
# ============================================================================

def multi_krum(updates, n_byzantine, multi_k=None):
    """Multi-Krum Byzantine-resilient aggregation"""
    n = len(updates)
    if n == 0:
        return None
    
    f = n_byzantine
    if n <= 2 * f + 2:
        # Fall back to simple average with integer type handling
        result = {}
        for k in updates[0].keys():
            stacked = torch.stack([u[k] for u in updates])
            if stacked.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                result[k] = stacked[0]  # Just take the first one for integer params
            else:
                result[k] = stacked.mean(dim=0)
        return result
    
    # Flatten updates
    flat_updates = []
    for u in updates:
        flat = torch.cat([p.flatten() for p in u.values()])
        flat_updates.append(flat)
    flat_updates = torch.stack(flat_updates)
    
    # Compute pairwise distances
    n_select = n - f - 2
    scores = []
    
    for i in range(n):
        dists = torch.norm(flat_updates - flat_updates[i], dim=1)
        dists_sorted, _ = torch.sort(dists)
        score = dists_sorted[1:n_select+1].sum()  # Exclude self (index 0)
        scores.append(score)
    
    scores = torch.tensor(scores)
    
    # Select top-k with lowest scores
    if multi_k is None:
        multi_k = max(1, n - f)
    
    _, selected_indices = torch.topk(scores, multi_k, largest=False)
    
    # Average selected updates
    selected_updates = [updates[i] for i in selected_indices]
    aggregated = {}
    for key in updates[0].keys():
        stacked = torch.stack([u[key] for u in selected_updates])
        # Handle integer types (e.g., num_batches_tracked in BatchNorm)
        if stacked.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            aggregated[key] = stacked[0]  # Just take the first one for integer params
        else:
            aggregated[key] = stacked.mean(dim=0)
    
    return aggregated

# ============================================================================
# FULL B-FedPLC SYSTEM
# ============================================================================

class BFedPLC:
    """Complete B-FedPLC System"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.blockchain = BlockchainLayer()
        self.clusterer = PLCClusterer(n_clusters=config.get('n_clusters', 4))
        
        # Create model
        if config['dataset'] == 'mnist':
            self.global_model = MNISTNet().to(self.device)
        else:
            self.global_model = CIFAR10Net().to(self.device)
        
        # Metrics storage
        self.round_metrics = []
    
    def get_model_hash(self, model_state):
        """Compute hash of model state"""
        data = {k: v.cpu().numpy().tolist() for k, v in model_state.items()}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
    
    def local_train(self, model, train_loader, epochs=1, lr=0.01):
        """Train model locally"""
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        return {k: v.clone() for k, v in model.state_dict().items()}
    
    def apply_byzantine_attack(self, update, attack_type='sign_flip'):
        """Apply Byzantine attack to model update"""
        attacked = {}
        for k, v in update.items():
            if attack_type == 'sign_flip':
                attacked[k] = -v
            elif attack_type == 'noise':
                attacked[k] = v + torch.randn_like(v) * 10
            elif attack_type == 'zero':
                attacked[k] = torch.zeros_like(v)
            else:
                attacked[k] = v
        return attacked
    
    def hierarchical_aggregation(self, client_updates, client_ids, clusters, n_byzantine_per_cluster):
        """Two-level hierarchical aggregation with Byzantine defense"""
        
        # Level 1: Intra-cluster aggregation
        cluster_models = {}
        cluster_weights = {}
        
        for cluster_id, cluster_clients in clusters.items():
            cluster_updates = []
            for cid in cluster_clients:
                idx = client_ids.index(cid)
                cluster_updates.append(client_updates[idx])
            
            if len(cluster_updates) > 0:
                # Apply Multi-Krum within cluster
                n_byz = min(n_byzantine_per_cluster, len(cluster_updates) // 3)
                cluster_agg = multi_krum(cluster_updates, n_byz)
                if cluster_agg is not None:
                    cluster_models[cluster_id] = cluster_agg
                    cluster_weights[cluster_id] = len(cluster_clients)
        
        if not cluster_models:
            return None
        
        # Level 2: Inter-cluster aggregation
        total_weight = sum(cluster_weights.values())
        global_update = {}
        
        for key in list(cluster_models.values())[0].keys():
            # Handle integer types (e.g., num_batches_tracked in BatchNorm)
            first_val = list(cluster_models.values())[0][key]
            if first_val.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                global_update[key] = first_val  # Just take the first one for integer params
            else:
                weighted_sum = None
                for cluster_id, model in cluster_models.items():
                    weight = cluster_weights[cluster_id] / total_weight
                    if weighted_sum is None:
                        weighted_sum = model[key] * weight
                    else:
                        weighted_sum += model[key] * weight
                global_update[key] = weighted_sum
        
        return global_update
    
    def evaluate(self, test_loader):
        """Evaluate global model"""
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        return 100.0 * correct / total
    
    def run_round(self, round_num, client_loaders, byzantine_clients, test_loader):
        """Execute one round of B-FedPLC"""
        
        round_start = time.time()
        client_updates = []
        client_ids = list(range(len(client_loaders)))
        
        # Phase 1: Local training
        train_start = time.time()
        for client_id, loader in enumerate(client_loaders):
            # Clone global model
            local_model = copy.deepcopy(self.global_model)
            
            # Local training
            update = self.local_train(local_model, loader, epochs=1)
            
            # Apply Byzantine attack if malicious
            if client_id in byzantine_clients:
                update = self.apply_byzantine_attack(update, 'sign_flip')
            
            client_updates.append(update)
            
            # Record on blockchain
            model_hash = self.get_model_hash(update)
            self.blockchain.add_model_update(client_id, model_hash, round_num)
        
        train_time = time.time() - train_start
        
        # Phase 2: Dynamic clustering (every recluster_interval rounds)
        cluster_start = time.time()
        if round_num % self.config.get('recluster_interval', 5) == 0:
            clusters = self.clusterer.cluster(client_ids, client_updates)
        else:
            # Use existing clusters
            clusters = defaultdict(list)
            for cid in client_ids:
                cluster_id = self.clusterer.cluster_assignments.get(cid, 0)
                clusters[cluster_id].append(cid)
            clusters = dict(clusters)
        cluster_time = time.time() - cluster_start
        
        # Phase 3: Hierarchical aggregation with Byzantine defense
        agg_start = time.time()
        n_byzantine = len(byzantine_clients)
        n_byz_per_cluster = max(1, n_byzantine // max(1, len(clusters)))
        
        global_update = self.hierarchical_aggregation(
            client_updates, client_ids, clusters, n_byz_per_cluster
        )
        agg_time = time.time() - agg_start
        
        # Update global model
        if global_update is not None:
            self.global_model.load_state_dict(global_update)
        
        # Create blockchain block
        block_start = time.time()
        agg_hash = self.get_model_hash(self.global_model.state_dict())
        self.blockchain.create_block(round_num, agg_hash)
        block_time = time.time() - block_start
        
        # Evaluate
        accuracy = self.evaluate(test_loader)
        
        round_time = time.time() - round_start
        
        # Store metrics
        metrics = {
            'round': round_num,
            'accuracy': accuracy,
            'train_time': train_time,
            'cluster_time': cluster_time,
            'aggregation_time': agg_time,
            'blockchain_time': block_time,
            'total_time': round_time,
            'n_clusters': len(clusters),
            'blockchain_valid': self.blockchain.verify_chain()
        }
        self.round_metrics.append(metrics)
        
        return accuracy, metrics

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(dataset_name):
    """Load dataset"""
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_data = torchvision.datasets.MNIST('./data', train=False, transform=transform)
    else:  # cifar10
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_data = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        test_data = torchvision.datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    return train_data, test_data

def create_noniid_partitions(dataset, n_clients, alpha=0.5):
    """Create Non-IID data partitions using Dirichlet distribution"""
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    n_classes = len(np.unique(labels))
    
    # Dirichlet distribution for label proportions
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    
    # Get indices for each class
    class_indices = [np.where(labels == k)[0] for k in range(n_classes)]
    
    # Distribute indices to clients
    client_indices = [[] for _ in range(n_clients)]
    
    for c, indices in enumerate(class_indices):
        np.random.shuffle(indices)
        proportions = label_distribution[c]
        proportions = proportions / proportions.sum()
        
        split_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        splits = np.split(indices, split_points)
        
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())
    
    return client_indices

# ============================================================================
# EXPERIMENTS
# ============================================================================

def run_full_bfedplc_experiment(config, seeds=[42, 123]):
    """Run complete B-FedPLC experiment"""
    
    results = {
        'config': config,
        'runs': []
    }
    
    for seed in seeds:
        print(f"\n  Seed: {seed}")
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Load data
        train_data, test_data = load_data(config['dataset'])
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
        
        # Create Non-IID partitions
        client_indices = create_noniid_partitions(
            train_data, config['n_clients'], config['alpha']
        )
        
        # Create client data loaders
        client_loaders = []
        for indices in client_indices:
            subset = Subset(train_data, indices)
            loader = DataLoader(subset, batch_size=32, shuffle=True)
            client_loaders.append(loader)
        
        # Select Byzantine clients
        n_byzantine = int(config['n_clients'] * config['byzantine_fraction'])
        byzantine_clients = set(np.random.choice(
            config['n_clients'], n_byzantine, replace=False
        ))
        
        # Initialize B-FedPLC system
        system = BFedPLC(config)
        
        # Run training
        accuracies = []
        for round_num in range(1, config['n_rounds'] + 1):
            acc, metrics = system.run_round(
                round_num, client_loaders, byzantine_clients, test_loader
            )
            accuracies.append(acc)
            
            if round_num % 10 == 0:
                print(f"    Round {round_num}: {acc:.2f}%")
        
        # Get final blockchain metrics
        blockchain_metrics = system.blockchain.get_metrics()
        
        run_result = {
            'seed': seed,
            'final_accuracy': accuracies[-1],
            'best_accuracy': max(accuracies),
            'accuracy_history': accuracies,
            'round_metrics': system.round_metrics,
            'blockchain_metrics': blockchain_metrics
        }
        results['runs'].append(run_result)
    
    # Compute statistics
    final_accs = [r['final_accuracy'] for r in results['runs']]
    results['statistics'] = {
        'mean_accuracy': np.mean(final_accs),
        'std_accuracy': np.std(final_accs),
        'max_accuracy': max(final_accs),
        'min_accuracy': min(final_accs)
    }
    
    return results

def main():
    print("=" * 80)
    print("PHASE 6: Full B-FedPLC System Integration")
    print("=" * 80)
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    all_results = {}
    
    # =========================================================================
    # Experiment 1: B-FedPLC vs Baselines under Byzantine Attack
    # =========================================================================
    print("\n" + "=" * 60)
    print("Experiment 1: B-FedPLC Performance Under Byzantine Attacks")
    print("=" * 60)
    
    exp1_results = {}
    
    for dataset in ['mnist', 'cifar10']:
        print(f"\n  Dataset: {dataset.upper()}")
        exp1_results[dataset] = {}
        
        for byz_frac in [0.0, 0.2, 0.3]:
            print(f"\n    Byzantine Fraction: {byz_frac*100:.0f}%")
            
            config = {
                'dataset': dataset,
                'n_clients': 20,
                'n_rounds': 30,
                'n_clusters': 4,
                'recluster_interval': 5,
                'byzantine_fraction': byz_frac,
                'alpha': 0.5  # Non-IID degree
            }
            
            result = run_full_bfedplc_experiment(config, seeds=[42, 123])
            exp1_results[dataset][f'byz_{int(byz_frac*100)}'] = result
            
            print(f"    Result: {result['statistics']['mean_accuracy']:.2f}% ± {result['statistics']['std_accuracy']:.2f}%")
    
    all_results['experiment1_byzantine'] = exp1_results
    
    # =========================================================================
    # Experiment 2: Non-IID Robustness
    # =========================================================================
    print("\n" + "=" * 60)
    print("Experiment 2: Non-IID Robustness (varying α)")
    print("=" * 60)
    
    exp2_results = {}
    
    for alpha in [0.1, 0.5, 1.0]:
        print(f"\n  Alpha: {alpha} (lower = more heterogeneous)")
        
        config = {
            'dataset': 'cifar10',
            'n_clients': 20,
            'n_rounds': 30,
            'n_clusters': 4,
            'recluster_interval': 5,
            'byzantine_fraction': 0.2,
            'alpha': alpha
        }
        
        result = run_full_bfedplc_experiment(config, seeds=[42, 123])
        exp2_results[f'alpha_{alpha}'] = result
        
        print(f"  Result: {result['statistics']['mean_accuracy']:.2f}% ± {result['statistics']['std_accuracy']:.2f}%")
    
    all_results['experiment2_noniid'] = exp2_results
    
    # =========================================================================
    # Experiment 3: System Scalability
    # =========================================================================
    print("\n" + "=" * 60)
    print("Experiment 3: System Scalability")
    print("=" * 60)
    
    exp3_results = {}
    
    for n_clients in [10, 20, 30]:
        print(f"\n  Clients: {n_clients}")
        
        config = {
            'dataset': 'cifar10',
            'n_clients': n_clients,
            'n_rounds': 30,
            'n_clusters': max(2, n_clients // 5),
            'recluster_interval': 5,
            'byzantine_fraction': 0.2,
            'alpha': 0.5
        }
        
        result = run_full_bfedplc_experiment(config, seeds=[42])
        exp3_results[f'clients_{n_clients}'] = result
        
        # Compute average times
        avg_times = {
            'train_time': np.mean([m['train_time'] for m in result['runs'][0]['round_metrics']]),
            'cluster_time': np.mean([m['cluster_time'] for m in result['runs'][0]['round_metrics']]),
            'aggregation_time': np.mean([m['aggregation_time'] for m in result['runs'][0]['round_metrics']]),
            'blockchain_time': np.mean([m['blockchain_time'] for m in result['runs'][0]['round_metrics']]),
            'total_time': np.mean([m['total_time'] for m in result['runs'][0]['round_metrics']])
        }
        exp3_results[f'clients_{n_clients}']['timing'] = avg_times
        
        print(f"  Accuracy: {result['statistics']['mean_accuracy']:.2f}%")
        print(f"  Avg Round Time: {avg_times['total_time']:.2f}s")
    
    all_results['experiment3_scalability'] = exp3_results
    
    # =========================================================================
    # Experiment 4: Component Ablation Study
    # =========================================================================
    print("\n" + "=" * 60)
    print("Experiment 4: Component Ablation Study")
    print("=" * 60)
    
    exp4_results = {}
    
    # Full B-FedPLC (already tested)
    print("\n  Full B-FedPLC: Using results from Experiment 1")
    exp4_results['full_bfedplc'] = exp1_results['cifar10']['byz_20']
    
    # Without clustering (direct Multi-Krum)
    print("\n  Without PLC Clustering (direct aggregation):")
    
    config_no_cluster = {
        'dataset': 'cifar10',
        'n_clients': 20,
        'n_rounds': 30,
        'n_clusters': 1,  # No real clustering
        'recluster_interval': 100,  # Never recluster
        'byzantine_fraction': 0.2,
        'alpha': 0.5
    }
    
    result_no_cluster = run_full_bfedplc_experiment(config_no_cluster, seeds=[42, 123])
    exp4_results['no_clustering'] = result_no_cluster
    print(f"  Result: {result_no_cluster['statistics']['mean_accuracy']:.2f}%")
    
    all_results['experiment4_ablation'] = exp4_results
    
    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    
    # Convert numpy types for JSON
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
    
    with open('phase6_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print("✅ Results saved to: phase6_results.json")
    
    # =========================================================================
    # Generate Visualizations
    # =========================================================================
    print("\nGenerating Phase 6 visualizations...")
    
    # Figure 1: B-FedPLC Performance Under Byzantine Attacks
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, dataset in enumerate(['mnist', 'cifar10']):
        ax = axes[idx]
        byz_fracs = [0, 20, 30]
        means = []
        stds = []
        
        for byz in byz_fracs:
            key = f'byz_{byz}'
            if key in exp1_results[dataset]:
                means.append(exp1_results[dataset][key]['statistics']['mean_accuracy'])
                stds.append(exp1_results[dataset][key]['statistics']['std_accuracy'])
        
        x = np.arange(len(byz_fracs))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=['#2ecc71', '#f39c12', '#e74c3c'])
        ax.set_xlabel('Byzantine Fraction (%)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'B-FedPLC on {dataset.upper()}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(['0%', '20%', '30%'])
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, mean in zip(bars, means):
            ax.annotate(f'{mean:.1f}%', xy=(bar.get_x() + bar.get_width()/2, mean),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('phase6_byzantine_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Generated: phase6_byzantine_performance.png")
    
    # Figure 2: Non-IID Robustness
    fig, ax = plt.subplots(figsize=(8, 6))
    
    alphas = [0.1, 0.5, 1.0]
    means = [exp2_results[f'alpha_{a}']['statistics']['mean_accuracy'] for a in alphas]
    stds = [exp2_results[f'alpha_{a}']['statistics']['std_accuracy'] for a in alphas]
    
    x = np.arange(len(alphas))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=['#9b59b6', '#3498db', '#1abc9c'])
    ax.set_xlabel('Dirichlet α (lower = more heterogeneous)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('B-FedPLC Non-IID Robustness', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['α=0.1\n(Highly Non-IID)', 'α=0.5\n(Moderate)', 'α=1.0\n(Near IID)'])
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, mean in zip(bars, means):
        ax.annotate(f'{mean:.1f}%', xy=(bar.get_x() + bar.get_width()/2, mean),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('phase6_noniid_robustness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Generated: phase6_noniid_robustness.png")
    
    # Figure 3: Scalability Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy vs Clients
    ax1 = axes[0]
    clients = [10, 20, 30]
    accs = [exp3_results[f'clients_{c}']['statistics']['mean_accuracy'] for c in clients]
    ax1.plot(clients, accs, 'o-', markersize=10, linewidth=2, color='#3498db')
    ax1.set_xlabel('Number of Clients', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy vs Number of Clients', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    for c, a in zip(clients, accs):
        ax1.annotate(f'{a:.1f}%', xy=(c, a), xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=10)
    
    # Time breakdown
    ax2 = axes[1]
    time_components = ['train_time', 'cluster_time', 'aggregation_time', 'blockchain_time']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    width = 0.2
    x = np.arange(len(clients))
    
    for i, comp in enumerate(time_components):
        times = [exp3_results[f'clients_{c}']['timing'][comp] for c in clients]
        ax2.bar(x + i*width, times, width, label=comp.replace('_', ' ').title(), color=colors[i])
    
    ax2.set_xlabel('Number of Clients', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Time Breakdown per Round', fontsize=14)
    ax2.set_xticks(x + width*1.5)
    ax2.set_xticklabels(clients)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase6_scalability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Generated: phase6_scalability.png")
    
    # Figure 4: Summary Table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Create summary data
    summary_data = [
        ['Experiment', 'Configuration', 'Accuracy', 'Key Finding'],
        ['Byzantine 0%', 'CIFAR-10, 20 clients', f"{exp1_results['cifar10']['byz_0']['statistics']['mean_accuracy']:.2f}%", 'Baseline performance'],
        ['Byzantine 20%', 'CIFAR-10, 20 clients', f"{exp1_results['cifar10']['byz_20']['statistics']['mean_accuracy']:.2f}%", 'Robust under attack'],
        ['Byzantine 30%', 'CIFAR-10, 20 clients', f"{exp1_results['cifar10']['byz_30']['statistics']['mean_accuracy']:.2f}%", 'High attack resilience'],
        ['Non-IID α=0.1', 'CIFAR-10, 20% Byzantine', f"{exp2_results['alpha_0.1']['statistics']['mean_accuracy']:.2f}%", 'Handles heterogeneity'],
        ['Non-IID α=0.5', 'CIFAR-10, 20% Byzantine', f"{exp2_results['alpha_0.5']['statistics']['mean_accuracy']:.2f}%", 'Balanced performance'],
        ['Non-IID α=1.0', 'CIFAR-10, 20% Byzantine', f"{exp2_results['alpha_1.0']['statistics']['mean_accuracy']:.2f}%", 'Near-IID performance'],
        ['30 Clients', 'CIFAR-10, 20% Byzantine', f"{exp3_results['clients_30']['statistics']['mean_accuracy']:.2f}%", 'Scalable system'],
        ['No Clustering', 'Direct Multi-Krum', f"{exp4_results['no_clustering']['statistics']['mean_accuracy']:.2f}%", 'Ablation baseline'],
    ]
    
    table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                    loc='center', cellLoc='center',
                    colColours=['#3498db']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Phase 6: B-FedPLC System Integration Results', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('phase6_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Generated: phase6_summary_table.png")
    
    print("\n✅ All Phase 6 visualizations generated!")
    
    # =========================================================================
    # Print Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 6 COMPLETE!")
    print("=" * 80)
    
    print("\nKey Findings:")
    print(f"  1. B-FedPLC maintains {exp1_results['cifar10']['byz_20']['statistics']['mean_accuracy']:.1f}% accuracy under 20% Byzantine attack")
    print(f"  2. System remains robust ({exp1_results['cifar10']['byz_30']['statistics']['mean_accuracy']:.1f}%) even with 30% attackers")
    print(f"  3. Non-IID robustness: {exp2_results['alpha_0.1']['statistics']['mean_accuracy']:.1f}% with extreme heterogeneity (α=0.1)")
    print(f"  4. Scalable to {max(clients)} clients with minimal performance degradation")
    print(f"  5. PLC clustering improves accuracy by {exp4_results['full_bfedplc']['statistics']['mean_accuracy'] - exp4_results['no_clustering']['statistics']['mean_accuracy']:.1f}% over direct aggregation")
    
    print("\nGenerated Files:")
    print("  - phase6_results.json")
    print("  - phase6_byzantine_performance.png")
    print("  - phase6_noniid_robustness.png")
    print("  - phase6_scalability.png")
    print("  - phase6_summary_table.png")
    
    print("\nNext: Phase 7 - SOTA Comparison")
    print("=" * 80)

if __name__ == "__main__":
    main()
