"""
FedPLC Full Experiment - Replication of Paper
100 clients, 200 rounds, Non-IID (α=0.5), PARL + LDCA
"""

import os
import gc
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import time
import json
from collections import defaultdict

# ============= Configuration =============
class Config:
    # Dataset
    dataset = "cifar10"
    num_classes = 10
    
    # Federated Learning (Paper settings)
    num_clients = 100        # N = 100 clients
    num_rounds = 200         # Total rounds
    local_epochs = 5         # E = 5 local epochs
    batch_size = 64          # Batch size
    participation_rate = 0.1 # 10% clients per round
    
    # Non-IID (Dirichlet)
    alpha = 0.5              # Lower = more heterogeneous
    
    # PARL settings
    parl_weight = 0.1        # λ_parl
    temperature = 0.07       # Temperature for contrastive
    warmup_rounds = 30       # Warmup before LDCA
    
    # LDCA settings  
    similarity_threshold = 0.85  # τ threshold
    
    # Training
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# ============= Model (ResNet-style for CIFAR) =============
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class FedPLCModel(nn.Module):
    """ResNet-style model with separate representation and classifier"""
    def __init__(self, num_classes=10, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Feature extractor (representation)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head
        self.classifier = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_planes, planes, num_blocks, stride):
        layers = [BasicBlock(in_planes, planes, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)
    
    def get_representation(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)
    
    def forward(self, x):
        rep = self.get_representation(x)
        return self.classifier(rep)

# ============= Non-IID Data Partitioning =============
def dirichlet_partition(dataset, num_clients, alpha, num_classes=10):
    """Partition data using Dirichlet distribution for Non-IID"""
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        
        # Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = proportions / proportions.sum()
        
        # Assign to clients
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        splits = np.split(class_indices, proportions)
        
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())
    
    # Shuffle each client's data
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
    
    return client_indices

# ============= PARL Loss (Prototype-Anchored Representation Learning) =============
def compute_prototypes(model, loader, device, num_classes=10):
    """Compute class prototypes from representations"""
    model.eval()
    class_features = defaultdict(list)
    
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            rep = model.get_representation(data)
            
            for i, label in enumerate(target):
                class_features[label.item()].append(rep[i].cpu())
    
    prototypes = {}
    for c in range(num_classes):
        if class_features[c]:
            prototypes[c] = torch.stack(class_features[c]).mean(0)
    
    return prototypes

def parl_loss(rep, target, global_prototypes, temperature=0.07, device='cuda'):
    """PARL alignment loss - align local representations to global prototypes"""
    if not global_prototypes:
        return torch.tensor(0.0, device=device)
    
    loss = 0.0
    count = 0
    
    for i, label in enumerate(target):
        c = label.item()
        if c in global_prototypes:
            proto = global_prototypes[c].to(device)
            # Cosine similarity loss
            sim = F.cosine_similarity(rep[i].unsqueeze(0), proto.unsqueeze(0))
            loss += (1 - sim)
            count += 1
    
    return loss / max(count, 1)

# ============= LDCA (Label-wise Dynamic Community Adaptation) =============
def compute_label_distribution(indices, dataset):
    """Compute label distribution for a client"""
    labels = [dataset[i][1] for i in indices]
    dist = np.zeros(10)
    for l in labels:
        dist[l] += 1
    return dist / (dist.sum() + 1e-8)

def compute_similarity_matrix(client_distributions):
    """Compute cosine similarity between client label distributions"""
    n = len(client_distributions)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            dot = np.dot(client_distributions[i], client_distributions[j])
            norm_i = np.linalg.norm(client_distributions[i])
            norm_j = np.linalg.norm(client_distributions[j])
            sim_matrix[i, j] = dot / (norm_i * norm_j + 1e-8)
    
    return sim_matrix

def louvain_community_detection(sim_matrix, threshold=0.85):
    """Simple community detection based on similarity threshold"""
    n = len(sim_matrix)
    communities = {}
    visited = set()
    community_id = 0
    
    for i in range(n):
        if i in visited:
            continue
        
        # Find similar clients
        community = [i]
        visited.add(i)
        
        for j in range(i + 1, n):
            if j not in visited and sim_matrix[i, j] >= threshold:
                community.append(j)
                visited.add(j)
        
        communities[community_id] = community
        community_id += 1
    
    return communities

# ============= Training Functions =============
def train_client(model, loader, optimizer, global_prototypes, config, warmup=True):
    """Train a single client with PARL loss"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_parl = 0
    samples = 0
    
    for epoch in range(config.local_epochs):
        for data, target in loader:
            data, target = data.to(config.device), target.to(config.device)
            
            optimizer.zero_grad()
            
            # Get representation and output
            rep = model.get_representation(data)
            output = model.classifier(rep)
            
            # Classification loss
            ce_loss = criterion(output, target)
            
            # PARL loss (after warmup)
            if not warmup and global_prototypes:
                p_loss = parl_loss(rep, target, global_prototypes, config.temperature, config.device)
                loss = ce_loss + config.parl_weight * p_loss
                total_parl += p_loss.item() * data.size(0)
            else:
                loss = ce_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += ce_loss.item() * data.size(0)
            samples += data.size(0)
    
    return {
        'loss': total_loss / samples,
        'parl_loss': total_parl / samples if total_parl > 0 else 0,
        'state': {k: v.cpu().clone() for k, v in model.state_dict().items()}
    }

def evaluate(model, loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return 100.0 * correct / total

def fed_avg(global_state, client_states, weights=None):
    """Federated Averaging"""
    if weights is None:
        weights = [1.0 / len(client_states)] * len(client_states)
    
    avg_state = {}
    for key in global_state.keys():
        avg_state[key] = sum(w * s[key].float() for w, s in zip(weights, client_states))
    
    return avg_state

def community_aggregate(global_state, client_states, client_ids, communities):
    """Community-based aggregation (LDCA)"""
    # Find which community each client belongs to
    client_to_community = {}
    for comm_id, members in communities.items():
        for m in members:
            client_to_community[m] = comm_id
    
    # Group client states by community
    community_states = defaultdict(list)
    for cid, state in zip(client_ids, client_states):
        comm = client_to_community.get(cid, 0)
        community_states[comm].append(state)
    
    # Average within each community, then average across communities
    comm_avg_states = []
    for comm_id, states in community_states.items():
        comm_avg = {}
        for key in global_state.keys():
            comm_avg[key] = sum(s[key].float() for s in states) / len(states)
        comm_avg_states.append(comm_avg)
    
    # Final average across communities
    final_state = {}
    for key in global_state.keys():
        final_state[key] = sum(s[key] for s in comm_avg_states) / len(comm_avg_states)
    
    return final_state

# ============= Main Experiment =============
def main():
    print("=" * 70, flush=True)
    print("FedPLC Full Experiment - Paper Replication", flush=True)
    print("=" * 70, flush=True)
    
    # Device info
    print(f"\nDevice: {config.device}", flush=True)
    if config.device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB", flush=True)
    
    # Configuration
    print(f"\nConfiguration:", flush=True)
    print(f"  Dataset: {config.dataset}", flush=True)
    print(f"  Clients: {config.num_clients}", flush=True)
    print(f"  Rounds: {config.num_rounds}", flush=True)
    print(f"  Local Epochs: {config.local_epochs}", flush=True)
    print(f"  Batch Size: {config.batch_size}", flush=True)
    print(f"  Participation: {config.participation_rate*100:.0f}%", flush=True)
    print(f"  Non-IID Alpha: {config.alpha}", flush=True)
    print(f"  Warmup Rounds: {config.warmup_rounds}", flush=True)
    print(f"  PARL Weight: {config.parl_weight}", flush=True)
    print(f"  LDCA Threshold: {config.similarity_threshold}", flush=True)
    
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    print("\nLoading CIFAR-10...", flush=True)
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
    
    # Non-IID partition
    print("Creating Non-IID partitions (Dirichlet)...", flush=True)
    client_indices = dirichlet_partition(train_dataset, config.num_clients, config.alpha)
    
    # Compute label distributions for LDCA
    client_distributions = []
    for indices in client_indices:
        dist = compute_label_distribution(indices, train_dataset)
        client_distributions.append(dist)
    
    # Compute similarity matrix
    sim_matrix = compute_similarity_matrix(client_distributions)
    
    # Create client dataloaders
    client_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)
    
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Print partition stats
    sizes = [len(indices) for indices in client_indices]
    print(f"Data partitioned: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.0f}", flush=True)
    
    # Create global model
    print("\nCreating model...", flush=True)
    global_model = FedPLCModel(num_classes=config.num_classes).to(config.device)
    num_params = sum(p.numel() for p in global_model.parameters())
    print(f"Model parameters: {num_params:,}", flush=True)
    
    # Initial evaluation
    init_acc = evaluate(global_model, test_loader, config.device)
    print(f"Initial accuracy: {init_acc:.2f}%", flush=True)
    
    # Training
    print("\n" + "=" * 70, flush=True)
    print("Starting Federated Training with PARL + LDCA", flush=True)
    print("=" * 70, flush=True)
    
    history = {
        'accuracy': [],
        'loss': [],
        'parl_loss': [],
        'communities': []
    }
    
    best_acc = 0
    global_prototypes = {}
    communities = {0: list(range(config.num_clients))}  # Initially all in one community
    
    start_time = time.time()
    
    for round_idx in range(config.num_rounds):
        round_start = time.time()
        warmup = round_idx < config.warmup_rounds
        
        # Select participating clients
        num_selected = max(1, int(config.num_clients * config.participation_rate))
        selected_clients = np.random.choice(config.num_clients, num_selected, replace=False)
        
        # LDCA: Update communities after warmup
        if not warmup and round_idx == config.warmup_rounds:
            print(f"\n[LDCA] Computing communities at round {round_idx+1}...", flush=True)
            communities = louvain_community_detection(sim_matrix, config.similarity_threshold)
            print(f"[LDCA] Found {len(communities)} communities", flush=True)
        
        # Train selected clients
        client_states = []
        total_loss = 0
        total_parl = 0
        
        for cid in selected_clients:
            # Create client model
            client_model = FedPLCModel(num_classes=config.num_classes).to(config.device)
            client_model.load_state_dict(global_model.state_dict())
            
            optimizer = torch.optim.SGD(
                client_model.parameters(),
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
            
            # Train
            result = train_client(
                client_model, 
                client_loaders[cid], 
                optimizer, 
                global_prototypes,
                config,
                warmup=warmup
            )
            
            client_states.append(result['state'])
            total_loss += result['loss']
            total_parl += result['parl_loss']
            
            del client_model, optimizer
            gc.collect()
            torch.cuda.empty_cache()
        
        # Aggregate
        if warmup:
            # Standard FedAvg during warmup
            avg_state = fed_avg(global_model.state_dict(), client_states)
        else:
            # Community-based aggregation after warmup
            avg_state = community_aggregate(
                global_model.state_dict(), 
                client_states, 
                selected_clients, 
                communities
            )
        
        global_model.load_state_dict(avg_state)
        
        # Update global prototypes after warmup
        if not warmup and (round_idx - config.warmup_rounds) % 10 == 0:
            # Compute prototypes from a subset of data
            proto_loader = DataLoader(
                Subset(train_dataset, list(range(0, len(train_dataset), 10))),
                batch_size=128, shuffle=False, num_workers=0
            )
            global_prototypes = compute_prototypes(global_model, proto_loader, config.device)
        
        # Evaluate
        acc = evaluate(global_model, test_loader, config.device)
        best_acc = max(best_acc, acc)
        
        # Record history
        avg_loss = total_loss / num_selected
        avg_parl = total_parl / num_selected
        history['accuracy'].append(acc)
        history['loss'].append(avg_loss)
        history['parl_loss'].append(avg_parl)
        history['communities'].append(len(communities))
        
        round_time = time.time() - round_start
        
        # Print progress
        phase = "Warmup" if warmup else "PARL+LDCA"
        if (round_idx + 1) % 10 == 0 or round_idx < 5:
            print(f"Round {round_idx+1:3d}/{config.num_rounds} [{phase:8s}] | "
                  f"Loss: {avg_loss:.4f} | PARL: {avg_parl:.4f} | "
                  f"Acc: {acc:.2f}% | Best: {best_acc:.2f}% | "
                  f"Comm: {len(communities)} | Time: {round_time:.1f}s", flush=True)
        
        # Cleanup
        del client_states, avg_state
        gc.collect()
        torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    
    # Final results
    print("\n" + "=" * 70, flush=True)
    print("TRAINING COMPLETE!", flush=True)
    print("=" * 70, flush=True)
    print(f"Final Accuracy: {history['accuracy'][-1]:.2f}%", flush=True)
    print(f"Best Accuracy: {best_acc:.2f}%", flush=True)
    print(f"Total Time: {total_time/60:.1f} minutes ({total_time/config.num_rounds:.1f}s per round)", flush=True)
    print(f"Final Communities: {len(communities)}", flush=True)
    
    # Save results
    results = {
        'config': {
            'dataset': config.dataset,
            'num_clients': config.num_clients,
            'num_rounds': config.num_rounds,
            'local_epochs': config.local_epochs,
            'batch_size': config.batch_size,
            'alpha': config.alpha,
            'parl_weight': config.parl_weight,
            'warmup_rounds': config.warmup_rounds,
            'similarity_threshold': config.similarity_threshold,
        },
        'best_accuracy': best_acc,
        'final_accuracy': history['accuracy'][-1],
        'history': history,
        'total_time': total_time,
        'communities': {str(k): v for k, v in communities.items()}
    }
    
    # Save as JSON
    with open('fedplc_full_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    torch.save({
        'model_state': global_model.state_dict(),
        'config': results['config'],
        'best_accuracy': best_acc
    }, 'fedplc_full_model.pt')
    
    print(f"\nResults saved to fedplc_full_results.json", flush=True)
    print(f"Model saved to fedplc_full_model.pt", flush=True)
    print("=" * 70, flush=True)
    
    return best_acc, history

if __name__ == "__main__":
    try:
        best_acc, history = main()
    except Exception as e:
        print(f"\nERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
