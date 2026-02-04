"""
FedPLC Medium Experiment - 50 rounds test
Versi lebih pendek untuk verifikasi implementasi
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
NUM_CLIENTS = 50          # Reduced for faster test
NUM_ROUNDS = 50           # 50 rounds
LOCAL_EPOCHS = 3          # Reduced
BATCH_SIZE = 64
PARTICIPATION_RATE = 0.2  # 20% per round
ALPHA = 0.5               # Non-IID
WARMUP_ROUNDS = 15        # Warmup before LDCA
PARL_WEIGHT = 0.1
SIMILARITY_THRESHOLD = 0.85
LR = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============= Simple CNN Model =============
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)
    
    def get_representation(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)
    
    def forward(self, x):
        rep = self.get_representation(x)
        return self.classifier(rep)

# ============= Non-IID Partitioning =============
def dirichlet_partition(dataset, num_clients, alpha, num_classes=10):
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        splits = np.split(class_indices, proportions)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())
    
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    
    return client_indices

# ============= LDCA Community Detection =============
def compute_label_distribution(indices, dataset):
    labels = [dataset[i][1] for i in indices]
    dist = np.zeros(10)
    for l in labels:
        dist[l] += 1
    return dist / (dist.sum() + 1e-8)

def compute_similarity_matrix(distributions):
    n = len(distributions)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dot = np.dot(distributions[i], distributions[j])
            norm = np.linalg.norm(distributions[i]) * np.linalg.norm(distributions[j])
            sim[i, j] = dot / (norm + 1e-8)
    return sim

def detect_communities(sim_matrix, threshold=0.85):
    n = len(sim_matrix)
    communities = {}
    visited = set()
    comm_id = 0
    
    for i in range(n):
        if i in visited:
            continue
        community = [i]
        visited.add(i)
        for j in range(i + 1, n):
            if j not in visited and sim_matrix[i, j] >= threshold:
                community.append(j)
                visited.add(j)
        communities[comm_id] = community
        comm_id += 1
    
    return communities

# ============= Training =============
def train_client(model, loader, global_prototypes, warmup=True):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_parl = 0
    samples = 0
    
    for _ in range(LOCAL_EPOCHS):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            rep = model.get_representation(data)
            output = model.classifier(rep)
            ce_loss = criterion(output, target)
            
            # PARL loss after warmup
            parl_loss = torch.tensor(0.0, device=device)
            if not warmup and global_prototypes:
                parl_count = 0
                for i, label in enumerate(target):
                    c = label.item()
                    if c in global_prototypes:
                        proto = global_prototypes[c].to(device)
                        sim = F.cosine_similarity(rep[i].unsqueeze(0), proto.unsqueeze(0))
                        parl_loss = parl_loss + (1 - sim.squeeze())
                        parl_count += 1
                if parl_count > 0:
                    parl_loss = parl_loss / parl_count
            
            loss = ce_loss + PARL_WEIGHT * parl_loss
            loss.backward()
            optimizer.step()
            
            total_loss += ce_loss.item() * data.size(0)
            total_parl += parl_loss.item() * data.size(0)
            samples += data.size(0)
    
    return {
        'loss': total_loss / samples,
        'parl': total_parl / samples,
        'state': {k: v.cpu().clone() for k, v in model.state_dict().items()}
    }

def compute_prototypes(model, loader, num_classes=10):
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

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total

def aggregate(global_state, client_states, client_ids, communities):
    # Community-based aggregation
    client_to_comm = {}
    for cid, members in communities.items():
        for m in members:
            client_to_comm[m] = cid
    
    comm_states = defaultdict(list)
    for cid, state in zip(client_ids, client_states):
        comm = client_to_comm.get(cid, 0)
        comm_states[comm].append(state)
    
    # Average within communities
    comm_avgs = []
    for states in comm_states.values():
        avg = {}
        for key in global_state.keys():
            avg[key] = sum(s[key].float() for s in states) / len(states)
        comm_avgs.append(avg)
    
    # Average across communities
    final = {}
    for key in global_state.keys():
        final[key] = sum(s[key] for s in comm_avgs) / len(comm_avgs)
    return final

# ============= Main =============
def main():
    print("=" * 60, flush=True)
    print("FedPLC Medium Experiment (50 rounds)", flush=True)
    print("=" * 60, flush=True)
    
    print(f"\nDevice: {device}", flush=True)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    
    print(f"\nConfig: {NUM_CLIENTS} clients, {NUM_ROUNDS} rounds", flush=True)
    print(f"Warmup: {WARMUP_ROUNDS}, PARL weight: {PARL_WEIGHT}", flush=True)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    print("\nLoading CIFAR-10...", flush=True)
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10('./data', train=False, transform=test_transform)
    
    # Partition
    client_indices = dirichlet_partition(train_data, NUM_CLIENTS, ALPHA)
    
    # Compute distributions for LDCA
    distributions = [compute_label_distribution(idx, train_data) for idx in client_indices]
    sim_matrix = compute_similarity_matrix(distributions)
    
    # Create loaders
    client_loaders = []
    for indices in client_indices:
        subset = Subset(train_data, indices)
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        client_loaders.append(loader)
    
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    sizes = [len(idx) for idx in client_indices]
    print(f"Data: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.0f}", flush=True)
    
    # Model
    global_model = SimpleCNN().to(device)
    print(f"Model params: {sum(p.numel() for p in global_model.parameters()):,}", flush=True)
    
    init_acc = evaluate(global_model, test_loader)
    print(f"Initial accuracy: {init_acc:.2f}%", flush=True)
    
    # Training
    print("\n" + "=" * 60, flush=True)
    print("Training with PARL + LDCA", flush=True)
    print("=" * 60, flush=True)
    
    history = {'accuracy': [], 'loss': [], 'parl': []}
    best_acc = 0
    global_prototypes = {}
    communities = {0: list(range(NUM_CLIENTS))}
    
    start = time.time()
    
    for r in range(NUM_ROUNDS):
        t0 = time.time()
        warmup = r < WARMUP_ROUNDS
        
        # Select clients
        n_sel = max(1, int(NUM_CLIENTS * PARTICIPATION_RATE))
        selected = np.random.choice(NUM_CLIENTS, n_sel, replace=False)
        
        # LDCA after warmup
        if not warmup and r == WARMUP_ROUNDS:
            print(f"\n[LDCA] Computing communities...", flush=True)
            communities = detect_communities(sim_matrix, SIMILARITY_THRESHOLD)
            print(f"[LDCA] Found {len(communities)} communities", flush=True)
        
        # Train
        client_states = []
        total_loss = total_parl = 0
        
        for cid in selected:
            client_model = SimpleCNN().to(device)
            client_model.load_state_dict(global_model.state_dict())
            
            result = train_client(client_model, client_loaders[cid], global_prototypes, warmup)
            client_states.append(result['state'])
            total_loss += result['loss']
            total_parl += result['parl']
            
            del client_model
            gc.collect()
            torch.cuda.empty_cache()
        
        # Aggregate
        if warmup:
            avg = {}
            for key in global_model.state_dict().keys():
                avg[key] = sum(s[key].float() for s in client_states) / len(client_states)
        else:
            avg = aggregate(global_model.state_dict(), client_states, selected, communities)
        
        global_model.load_state_dict(avg)
        
        # Update prototypes
        if not warmup and (r - WARMUP_ROUNDS) % 5 == 0:
            proto_loader = DataLoader(
                Subset(train_data, list(range(0, len(train_data), 20))),
                batch_size=128, shuffle=False, num_workers=0
            )
            global_prototypes = compute_prototypes(global_model, proto_loader)
        
        # Evaluate
        acc = evaluate(global_model, test_loader)
        best_acc = max(best_acc, acc)
        
        history['accuracy'].append(acc)
        history['loss'].append(total_loss / n_sel)
        history['parl'].append(total_parl / n_sel)
        
        phase = "Warmup" if warmup else "PARL+LDCA"
        print(f"Round {r+1:2d}/{NUM_ROUNDS} [{phase:9s}] | "
              f"Loss: {total_loss/n_sel:.4f} | PARL: {total_parl/n_sel:.4f} | "
              f"Acc: {acc:.2f}% | Best: {best_acc:.2f}% | "
              f"Comm: {len(communities)} | {time.time()-t0:.1f}s", flush=True)
        
        del client_states, avg
        gc.collect()
        torch.cuda.empty_cache()
    
    total_time = time.time() - start
    
    print("\n" + "=" * 60, flush=True)
    print("COMPLETE!", flush=True)
    print("=" * 60, flush=True)
    print(f"Final: {history['accuracy'][-1]:.2f}%", flush=True)
    print(f"Best: {best_acc:.2f}%", flush=True)
    print(f"Time: {total_time/60:.1f} min", flush=True)
    print(f"Communities: {len(communities)}", flush=True)
    
    # Save
    results = {
        'best_accuracy': best_acc,
        'final_accuracy': history['accuracy'][-1],
        'history': history,
        'total_time': total_time,
        'num_communities': len(communities)
    }
    
    with open('fedplc_50rounds_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save(global_model.state_dict(), 'fedplc_50rounds_model.pt')
    print("\nSaved to fedplc_50rounds_results.json", flush=True)
    
    return best_acc

if __name__ == "__main__":
    main()
