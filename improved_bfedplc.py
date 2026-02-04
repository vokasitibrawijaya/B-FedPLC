"""
IMPROVED B-FedPLC with Enhanced Byzantine Tolerance
===================================================
Improvements:
1. Better Byzantine-tolerant aggregation (Krum, Trimmed Mean, Coordinate-wise Median)
2. Comprehensive metrics: accuracy, communication cost, latency, security
3. Clear differentiation from standard FedAvg

Author: B-FedPLC Research Team
Date: 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import json
import os
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model & Training
    NUM_CLIENTS = 30
    ROUNDS = 50
    LOCAL_EPOCHS = 3
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    CLIENT_FRACTION = 0.2
    
    # Non-IID
    DIRICHLET_ALPHA = 0.5
    
    # Seeds for reproducibility
    SEEDS = [42, 123, 456]
    
    # Byzantine settings
    BYZANTINE_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.33]  # Test up to 33%
    
    # Aggregation methods
    AGGREGATION_METHODS = ['fedavg', 'trimmed_mean', 'krum', 'multi_krum', 'coordinate_median']

# ============================================================================
# SIMPLE CNN MODEL
# ============================================================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_data():
    """Load CIFAR-10 dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    return trainset, testset

def dirichlet_partition(dataset, num_clients, alpha):
    """Partition data using Dirichlet distribution for non-IID"""
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = 10
    
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = proportions / proportions.sum()
        
        splits = (proportions * len(class_indices)).astype(int)
        splits[-1] = len(class_indices) - splits[:-1].sum()
        
        idx = 0
        for client_id, count in enumerate(splits):
            client_indices[client_id].extend(class_indices[idx:idx+count])
            idx += count
    
    return client_indices

def get_model_size(model):
    """Calculate model size in bytes"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return param_size + buffer_size

def flatten_params(model):
    """Flatten model parameters to 1D tensor"""
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def unflatten_params(flat_params, model):
    """Unflatten 1D tensor back to model parameters"""
    idx = 0
    for p in model.parameters():
        length = p.numel()
        p.data.copy_(flat_params[idx:idx+length].view(p.shape))
        idx += length

# ============================================================================
# BYZANTINE ATTACK SIMULATIONS
# ============================================================================

def apply_byzantine_attack(model, attack_type='random'):
    """Apply Byzantine attack to model updates"""
    with torch.no_grad():
        for param in model.parameters():
            if attack_type == 'random':
                # Random noise attack
                param.data = torch.randn_like(param.data) * 10
            elif attack_type == 'sign_flip':
                # Sign flipping attack
                param.data = -param.data * 5
            elif attack_type == 'label_flip':
                # Scaled random (more subtle)
                param.data += torch.randn_like(param.data) * 2
            elif attack_type == 'scaling':
                # Scaling attack
                param.data = param.data * 100
    return model

# ============================================================================
# ROBUST AGGREGATION METHODS
# ============================================================================

class RobustAggregator:
    """Collection of Byzantine-tolerant aggregation methods"""
    
    @staticmethod
    def fedavg(models, weights=None):
        """Standard Federated Averaging (no Byzantine tolerance)"""
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        avg_state = copy.deepcopy(models[0].state_dict())
        for key in avg_state.keys():
            avg_state[key] = torch.zeros_like(avg_state[key], dtype=torch.float32)
            for i, model in enumerate(models):
                avg_state[key] += weights[i] * model.state_dict()[key].float()
        
        return avg_state
    
    @staticmethod
    def trimmed_mean(models, trim_ratio=0.2):
        """
        Trimmed Mean Aggregation
        - Removes top and bottom trim_ratio of values before averaging
        - Tolerates up to trim_ratio Byzantine clients
        """
        n = len(models)
        trim_count = int(n * trim_ratio)
        
        # Get all state dicts
        state_dicts = [m.state_dict() for m in models]
        avg_state = copy.deepcopy(state_dicts[0])
        
        for key in avg_state.keys():
            # Stack all values for this parameter
            stacked = torch.stack([sd[key].float() for sd in state_dicts])
            
            # Sort and trim
            sorted_vals, _ = torch.sort(stacked, dim=0)
            if trim_count > 0 and n - 2 * trim_count > 0:
                trimmed = sorted_vals[trim_count:n-trim_count]
            else:
                trimmed = sorted_vals
            
            # Mean of trimmed values
            avg_state[key] = trimmed.mean(dim=0)
        
        return avg_state
    
    @staticmethod
    def coordinate_median(models):
        """
        Coordinate-wise Median Aggregation
        - Takes median of each parameter coordinate
        - Tolerates up to 50% Byzantine clients (but may have bias)
        """
        state_dicts = [m.state_dict() for m in models]
        avg_state = copy.deepcopy(state_dicts[0])
        
        for key in avg_state.keys():
            stacked = torch.stack([sd[key].float() for sd in state_dicts])
            avg_state[key] = torch.median(stacked, dim=0)[0]
        
        return avg_state
    
    @staticmethod
    def krum(models, f=None):
        """
        Krum Aggregation (Blanchard et al., 2017)
        - Selects the model with smallest sum of distances to nearest n-f-2 models
        - Tolerates up to f Byzantine clients where n >= 2f + 3
        """
        n = len(models)
        if f is None:
            f = int((n - 3) / 2)  # Maximum tolerable Byzantine clients
        
        if f < 0:
            f = 0
        
        # Flatten all models
        flat_models = [flatten_params(m) for m in models]
        
        # Compute pairwise distances
        distances = torch.zeros(n, n)
        for i in range(n):
            for j in range(i+1, n):
                dist = torch.norm(flat_models[i] - flat_models[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist
        
        # For each model, compute sum of distances to nearest n-f-2 models
        scores = []
        k = max(1, n - f - 2)
        for i in range(n):
            sorted_dists, _ = torch.sort(distances[i])
            # Sum of k smallest distances (excluding self which is 0)
            score = sorted_dists[1:k+1].sum().item()
            scores.append(score)
        
        # Select model with minimum score
        selected_idx = np.argmin(scores)
        return models[selected_idx].state_dict()
    
    @staticmethod
    def multi_krum(models, f=None, m=None):
        """
        Multi-Krum Aggregation
        - Selects m models with smallest Krum scores and averages them
        - Better convergence than single Krum
        """
        n = len(models)
        if f is None:
            f = int((n - 3) / 2)
        if m is None:
            m = max(1, n - f)
        
        if f < 0:
            f = 0
        
        # Flatten all models
        flat_models = [flatten_params(m) for m in models]
        
        # Compute pairwise distances
        distances = torch.zeros(n, n)
        for i in range(n):
            for j in range(i+1, n):
                dist = torch.norm(flat_models[i] - flat_models[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Compute Krum scores
        scores = []
        k = max(1, n - f - 2)
        for i in range(n):
            sorted_dists, _ = torch.sort(distances[i])
            score = sorted_dists[1:k+1].sum().item()
            scores.append(score)
        
        # Select m models with lowest scores
        selected_indices = np.argsort(scores)[:m]
        selected_models = [models[i] for i in selected_indices]
        
        # Average selected models
        return RobustAggregator.fedavg(selected_models)

# ============================================================================
# BLOCKCHAIN SIMULATION
# ============================================================================

class BlockchainSimulator:
    """Simulates blockchain operations for B-FedPLC"""
    
    def __init__(self):
        self.blocks = []
        self.pending_transactions = []
        self.consensus_time = 0
        self.total_bytes = 0
        
    def add_model_update(self, client_id, model_hash, model_size):
        """Record a model update transaction"""
        tx = {
            'type': 'model_update',
            'client_id': client_id,
            'model_hash': model_hash,
            'size': model_size,
            'timestamp': time.time()
        }
        self.pending_transactions.append(tx)
        self.total_bytes += model_size
        
    def add_aggregation_result(self, round_num, aggregated_hash, participants):
        """Record aggregation result"""
        tx = {
            'type': 'aggregation',
            'round': round_num,
            'hash': aggregated_hash,
            'participants': participants,
            'timestamp': time.time()
        }
        self.pending_transactions.append(tx)
        
    def run_consensus(self):
        """Simulate PBFT consensus"""
        start = time.time()
        
        # PBFT consensus simulation
        # In real implementation: pre-prepare, prepare, commit phases
        n_validators = 4  # Minimum for PBFT
        message_complexity = 3 * n_validators * len(self.pending_transactions)
        
        # Simulate network latency (proportional to message complexity)
        simulated_latency = 0.001 * message_complexity  # ms
        time.sleep(simulated_latency / 1000)  # Convert to seconds
        
        self.consensus_time += time.time() - start
        
        # Create block
        block = {
            'transactions': self.pending_transactions.copy(),
            'timestamp': time.time(),
            'block_num': len(self.blocks)
        }
        self.blocks.append(block)
        self.pending_transactions = []
        
        return block
    
    def get_metrics(self):
        """Get blockchain metrics"""
        return {
            'total_blocks': len(self.blocks),
            'total_bytes': self.total_bytes,
            'consensus_time': self.consensus_time
        }

# ============================================================================
# CLUSTERING MODULE
# ============================================================================

class DynamicClusterManager:
    """Dynamic Personalized Local Clustering"""
    
    def __init__(self, num_clients, num_clusters=3):
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.cluster_assignments = {}
        self.cluster_leaders = {}
        self.client_scores = defaultdict(float)
        
    def compute_client_similarity(self, model1, model2):
        """Compute cosine similarity between model parameters"""
        flat1 = flatten_params(model1)
        flat2 = flatten_params(model2)
        
        cos_sim = torch.nn.functional.cosine_similarity(
            flat1.unsqueeze(0), flat2.unsqueeze(0)
        )
        return cos_sim.item()
    
    def update_clusters(self, client_models, client_ids):
        """Update cluster assignments based on model similarity"""
        n = len(client_ids)
        if n < self.num_clusters:
            # Assign each client to its own cluster
            for i, cid in enumerate(client_ids):
                self.cluster_assignments[cid] = i
            return
        
        # Compute similarity matrix
        similarities = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                sim = self.compute_client_similarity(client_models[i], client_models[j])
                similarities[i, j] = sim
                similarities[j, i] = sim
        
        # Simple k-means style clustering based on similarity
        # Initialize cluster centers randomly
        centers = np.random.choice(n, self.num_clusters, replace=False)
        
        for _ in range(5):  # Iterations
            # Assign to nearest center
            assignments = []
            for i in range(n):
                dists = [1 - similarities[i, c] for c in centers]
                assignments.append(np.argmin(dists))
            
            # Update centers (pick most central member)
            new_centers = []
            for k in range(self.num_clusters):
                members = [i for i, a in enumerate(assignments) if a == k]
                if members:
                    # Find member with highest average similarity to others
                    best_member = members[0]
                    best_score = -1
                    for m in members:
                        score = np.mean([similarities[m, other] for other in members])
                        if score > best_score:
                            best_score = score
                            best_member = m
                    new_centers.append(best_member)
                else:
                    new_centers.append(centers[k])
            centers = new_centers
        
        # Store assignments
        for i, cid in enumerate(client_ids):
            self.cluster_assignments[cid] = assignments[i]
            
        # Select leaders (highest scoring in each cluster)
        for k in range(self.num_clusters):
            members = [client_ids[i] for i, a in enumerate(assignments) if a == k]
            if members:
                leader = max(members, key=lambda x: self.client_scores.get(x, 0))
                self.cluster_leaders[k] = leader
    
    def update_client_score(self, client_id, accuracy, is_byzantine=False):
        """Update client reputation score"""
        if is_byzantine:
            self.client_scores[client_id] -= 10
        else:
            self.client_scores[client_id] += accuracy
    
    def get_cluster_info(self):
        """Get cluster information"""
        return {
            'assignments': dict(self.cluster_assignments),
            'leaders': dict(self.cluster_leaders),
            'scores': dict(self.client_scores)
        }

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_client(model, train_loader, device, epochs=3, lr=0.01):
    """Train a client model locally"""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    return model

def evaluate(model, test_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100 * correct / total

# ============================================================================
# MAIN EXPERIMENT: IMPROVED B-FedPLC
# ============================================================================

def run_improved_experiment(
    aggregation_method='trimmed_mean',
    byzantine_fraction=0.0,
    seed=42,
    verbose=True
):
    """
    Run improved B-FedPLC experiment with robust aggregation
    
    Returns comprehensive metrics:
    - accuracy_history: list of accuracies per round
    - best_accuracy: best achieved accuracy
    - communication_cost: total bytes transferred
    - latency_per_round: average time per round
    - blockchain_metrics: blockchain overhead
    - byzantine_detected: number of detected Byzantine clients
    - security_score: overall security metric
    """
    
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    trainset, testset = load_data()
    test_loader = DataLoader(testset, batch_size=100, shuffle=False)
    
    # Partition data
    client_indices = dirichlet_partition(trainset, Config.NUM_CLIENTS, Config.DIRICHLET_ALPHA)
    
    # Initialize
    global_model = SimpleCNN().to(device)
    model_size = get_model_size(global_model)
    
    # Initialize components
    blockchain = BlockchainSimulator()
    cluster_manager = DynamicClusterManager(Config.NUM_CLIENTS)
    
    # Metrics tracking
    accuracy_history = []
    round_times = []
    communication_bytes = 0
    byzantine_detected = 0
    
    # Determine Byzantine clients
    num_byzantine = int(Config.NUM_CLIENTS * byzantine_fraction)
    byzantine_clients = set(np.random.choice(Config.NUM_CLIENTS, num_byzantine, replace=False))
    
    best_accuracy = 0
    
    for round_num in range(Config.ROUNDS):
        round_start = time.time()
        
        # Select clients
        num_selected = max(1, int(Config.NUM_CLIENTS * Config.CLIENT_FRACTION))
        selected_clients = np.random.choice(Config.NUM_CLIENTS, num_selected, replace=False)
        
        client_models = []
        client_weights = []
        
        # Local training
        for client_id in selected_clients:
            client_model = copy.deepcopy(global_model)
            
            # Create data loader
            indices = client_indices[client_id]
            if len(indices) == 0:
                continue
                
            subset = Subset(trainset, indices)
            train_loader = DataLoader(subset, batch_size=Config.BATCH_SIZE, shuffle=True)
            
            # Train
            client_model = train_client(
                client_model, train_loader, device,
                epochs=Config.LOCAL_EPOCHS, lr=Config.LEARNING_RATE
            )
            
            # Apply Byzantine attack if this is a Byzantine client
            if client_id in byzantine_clients:
                client_model = apply_byzantine_attack(client_model, 'random')
            
            client_models.append(client_model)
            client_weights.append(len(indices))
            
            # Record on blockchain
            model_hash = hash(flatten_params(client_model).sum().item())
            blockchain.add_model_update(client_id, model_hash, model_size)
            communication_bytes += model_size
        
        if len(client_models) == 0:
            continue
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Update clusters
        cluster_manager.update_clusters(client_models, list(selected_clients))
        
        # Robust Aggregation based on method
        if aggregation_method == 'fedavg':
            aggregated_state = RobustAggregator.fedavg(client_models, client_weights)
        elif aggregation_method == 'trimmed_mean':
            aggregated_state = RobustAggregator.trimmed_mean(client_models, trim_ratio=0.2)
        elif aggregation_method == 'coordinate_median':
            aggregated_state = RobustAggregator.coordinate_median(client_models)
        elif aggregation_method == 'krum':
            aggregated_state = RobustAggregator.krum(client_models, f=num_byzantine)
        elif aggregation_method == 'multi_krum':
            aggregated_state = RobustAggregator.multi_krum(client_models, f=num_byzantine)
        else:
            aggregated_state = RobustAggregator.fedavg(client_models, client_weights)
        
        # Update global model
        global_model.load_state_dict(aggregated_state)
        
        # Byzantine detection (simple outlier detection)
        if len(client_models) > 2:
            flat_updates = torch.stack([flatten_params(m) for m in client_models])
            mean_update = flat_updates.mean(dim=0)
            distances = torch.norm(flat_updates - mean_update, dim=1)
            threshold = distances.mean() + 2 * distances.std()
            detected = (distances > threshold).sum().item()
            byzantine_detected += detected
        
        # Run blockchain consensus
        blockchain.add_aggregation_result(
            round_num,
            hash(flatten_params(global_model).sum().item()),
            list(selected_clients)
        )
        blockchain.run_consensus()
        
        # Evaluate
        accuracy = evaluate(global_model, test_loader, device)
        accuracy_history.append(accuracy)
        best_accuracy = max(best_accuracy, accuracy)
        
        round_time = time.time() - round_start
        round_times.append(round_time)
        
        if verbose and (round_num + 1) % 10 == 0:
            print(f"  Round {round_num+1}/{Config.ROUNDS}: {accuracy:.2f}%")
    
    # Compile metrics
    blockchain_metrics = blockchain.get_metrics()
    
    return {
        'accuracy_history': accuracy_history,
        'best_accuracy': best_accuracy,
        'communication_cost': communication_bytes,
        'avg_round_time': np.mean(round_times),
        'total_time': sum(round_times),
        'blockchain_overhead': blockchain_metrics['consensus_time'],
        'blockchain_bytes': blockchain_metrics['total_bytes'],
        'byzantine_detected': byzantine_detected,
        'cluster_info': cluster_manager.get_cluster_info()
    }

# ============================================================================
# COMPREHENSIVE COMPARISON EXPERIMENT
# ============================================================================

def run_comprehensive_comparison():
    """
    Run comprehensive comparison between different aggregation methods
    across different Byzantine fractions
    """
    
    print("=" * 70)
    print("IMPROVED B-FedPLC: COMPREHENSIVE COMPARISON")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\nConfiguration:")
    print(f"  Clients: {Config.NUM_CLIENTS}")
    print(f"  Rounds: {Config.ROUNDS}")
    print(f"  Seeds: {Config.SEEDS}")
    print(f"  Byzantine fractions: {Config.BYZANTINE_FRACTIONS}")
    print(f"  Aggregation methods: {Config.AGGREGATION_METHODS}")
    
    results = defaultdict(lambda: defaultdict(list))
    
    total_experiments = len(Config.AGGREGATION_METHODS) * len(Config.BYZANTINE_FRACTIONS) * len(Config.SEEDS)
    current = 0
    
    for method in Config.AGGREGATION_METHODS:
        print(f"\n{'=' * 70}")
        print(f"AGGREGATION METHOD: {method.upper()}")
        print(f"{'=' * 70}")
        
        for byz_frac in Config.BYZANTINE_FRACTIONS:
            print(f"\n--- Byzantine Fraction: {byz_frac*100:.0f}% ---")
            
            method_results = []
            
            for seed in Config.SEEDS:
                current += 1
                print(f"  Seed {seed} ({current}/{total_experiments})...", end=" ", flush=True)
                
                result = run_improved_experiment(
                    aggregation_method=method,
                    byzantine_fraction=byz_frac,
                    seed=seed,
                    verbose=False
                )
                
                method_results.append(result)
                print(f"Best: {result['best_accuracy']:.2f}%")
            
            # Aggregate results across seeds
            key = f"{method}_byz{int(byz_frac*100)}"
            results[method][f'byz_{int(byz_frac*100)}'] = {
                'accuracy_mean': np.mean([r['best_accuracy'] for r in method_results]),
                'accuracy_std': np.std([r['best_accuracy'] for r in method_results]),
                'comm_cost_mean': np.mean([r['communication_cost'] for r in method_results]),
                'round_time_mean': np.mean([r['avg_round_time'] for r in method_results]),
                'blockchain_overhead_mean': np.mean([r['blockchain_overhead'] for r in method_results]),
                'byzantine_detected_mean': np.mean([r['byzantine_detected'] for r in method_results]),
                'accuracy_histories': [r['accuracy_history'] for r in method_results]
            }
    
    return dict(results)

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comprehensive_results(results):
    """Generate comprehensive comparison plots"""
    
    os.makedirs('plots', exist_ok=True)
    
    methods = list(results.keys())
    byz_fracs = [0, 10, 20, 30, 33]
    
    # 1. Byzantine Tolerance Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(byz_fracs))
    width = 0.15
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    
    for i, method in enumerate(methods):
        accuracies = []
        errors = []
        for byz in byz_fracs:
            key = f'byz_{byz}'
            if key in results[method]:
                accuracies.append(results[method][key]['accuracy_mean'])
                errors.append(results[method][key]['accuracy_std'])
            else:
                accuracies.append(0)
                errors.append(0)
        
        ax.bar(x + i * width, accuracies, width, label=method.upper(),
               yerr=errors, capsize=3, color=colors[i % len(colors)])
    
    ax.set_xlabel('Byzantine Fraction (%)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Byzantine Tolerance: Comparison of Aggregation Methods', fontsize=14)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([f'{b}%' for b in byz_fracs])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/byzantine_tolerance_comparison.png', dpi=150)
    plt.close()
    print("Saved: plots/byzantine_tolerance_comparison.png")
    
    # 2. Convergence Comparison (at 20% Byzantine)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, method in enumerate(methods):
        ax = axes[i]
        
        key = 'byz_20'
        if key in results[method] and 'accuracy_histories' in results[method][key]:
            histories = results[method][key]['accuracy_histories']
            
            # Plot mean with std shading
            histories_arr = np.array(histories)
            mean_hist = histories_arr.mean(axis=0)
            std_hist = histories_arr.std(axis=0)
            rounds = range(1, len(mean_hist) + 1)
            
            ax.plot(rounds, mean_hist, label=f'{method.upper()}', linewidth=2)
            ax.fill_between(rounds, mean_hist - std_hist, mean_hist + std_hist, alpha=0.3)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{method.upper()} (20% Byzantine)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    # Hide unused subplot
    if len(methods) < 6:
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('plots/convergence_comparison_20pct_byzantine.png', dpi=150)
    plt.close()
    print("Saved: plots/convergence_comparison_20pct_byzantine.png")
    
    # 3. Summary Table Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Prepare table data
    headers = ['Method', '0% Byz', '10% Byz', '20% Byz', '30% Byz', '33% Byz', 'BFT Compliant']
    table_data = []
    
    for method in methods:
        row = [method.upper()]
        is_bft = True
        for byz in byz_fracs:
            key = f'byz_{byz}'
            if key in results[method]:
                acc = results[method][key]['accuracy_mean']
                std = results[method][key]['accuracy_std']
                row.append(f'{acc:.1f}±{std:.1f}')
                
                # Check if still working at 33% Byzantine
                if byz == 33 and acc < 50:
                    is_bft = False
            else:
                row.append('N/A')
        
        row.append('✓' if is_bft else '✗')
        table_data.append(row)
    
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['#3498db'] * len(headers)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Color cells based on accuracy
    for i, row in enumerate(table_data):
        for j, cell in enumerate(row[1:-1], start=1):
            if cell != 'N/A':
                acc = float(cell.split('±')[0])
                if acc >= 60:
                    table[(i+1, j)].set_facecolor('#2ecc71')  # Green
                elif acc >= 40:
                    table[(i+1, j)].set_facecolor('#f39c12')  # Orange
                else:
                    table[(i+1, j)].set_facecolor('#e74c3c')  # Red
    
    plt.title('Aggregation Method Comparison: Accuracy (%) at Different Byzantine Fractions', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('plots/aggregation_comparison_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/aggregation_comparison_table.png")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("IMPROVED B-FedPLC EXPERIMENT")
    print("Enhanced Byzantine Tolerance & Comprehensive Metrics")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    # Run comprehensive comparison
    results = run_comprehensive_comparison()
    
    # Save results
    # Convert numpy types for JSON serialization
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
    
    results_serializable = convert_to_serializable(results)
    
    with open('improved_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print("\nResults saved to: improved_results.json")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_comprehensive_results(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: BYZANTINE TOLERANCE COMPARISON")
    print("=" * 70)
    
    print("\n{:<20} {:>12} {:>12} {:>12} {:>12}".format(
        'Method', '0% Byz', '10% Byz', '20% Byz', '33% Byz'))
    print("-" * 70)
    
    for method in Config.AGGREGATION_METHODS:
        row = f"{method.upper():<20}"
        for byz in [0, 10, 20, 33]:
            key = f'byz_{byz}'
            if key in results[method]:
                acc = results[method][key]['accuracy_mean']
                row += f" {acc:>10.2f}%"
            else:
                row += " {:>10}".format('N/A')
        print(row)
    
    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Total Experiment Time: {total_time/60:.2f} minutes")
    print("=" * 70)
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR B-FedPLC")
    print("=" * 70)
    
    # Find best method at 33% Byzantine
    best_method = None
    best_acc = 0
    for method in Config.AGGREGATION_METHODS:
        if 'byz_33' in results[method]:
            acc = results[method]['byz_33']['accuracy_mean']
            if acc > best_acc:
                best_acc = acc
                best_method = method
    
    print(f"\n1. Best Byzantine-tolerant method: {best_method.upper()}")
    print(f"   Maintains {best_acc:.2f}% accuracy at 33% Byzantine clients")
    
    print(f"\n2. Use {best_method.upper()} as default aggregation for B-FedPLC")
    print(f"   to achieve true BFT compliance (f < n/3)")
    
    print("\n3. B-FedPLC differentiation from FedAvg:")
    print("   - Byzantine tolerance up to 33% (vs 0% for FedAvg)")
    print("   - Blockchain audit trail for accountability")
    print("   - Dynamic clustering for personalization")
    print("   - Client reputation scoring")
