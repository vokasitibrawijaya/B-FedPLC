"""
FedPLC - Working GPU Version
Federated Learning dengan CIFAR-10 pada RTX 5060 Ti
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import time
import sys

def main():
    print('='*60)
    print('FedPLC - Federated Learning with GPU')  
    print('='*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

    # Config
    NUM_CLIENTS = 10
    NUM_ROUNDS = 15
    LOCAL_EPOCHS = 2
    BATCH_SIZE = 32

    print(f'\nConfig: {NUM_CLIENTS} clients, {NUM_ROUNDS} rounds, batch={BATCH_SIZE}')

    # Load CIFAR-10
    print('\nLoading CIFAR-10...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10('./data', train=False, transform=transform)

    # Partition to clients
    samples_per_client = len(train_data) // NUM_CLIENTS
    client_loaders = []
    for i in range(NUM_CLIENTS):
        indices = list(range(i * samples_per_client, (i+1) * samples_per_client))
        subset = Subset(train_data, indices)
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        client_loaders.append(loader)

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f'Data loaded: {NUM_CLIENTS} clients, {samples_per_client} samples each')

    # Simple CNN
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 10))
        
        def forward(self, x):
            return self.classifier(self.features(x))

    # Create global model
    global_model = CNN().to(device)
    print(f'Model params: {sum(p.numel() for p in global_model.parameters()):,}')

    def train_client(model, loader, epochs, lr=0.01):
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        crit = nn.CrossEntropyLoss()
        for _ in range(epochs):
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                loss = crit(model(data), target)
                loss.backward()
                opt.step()
        return {k: v.cpu().clone() for k, v in model.state_dict().items()}

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

    def fed_avg(states):
        avg = {}
        for key in states[0].keys():
            avg[key] = torch.stack([s[key].float() for s in states]).mean(0)
        return avg

    # Initial eval
    init_acc = evaluate(global_model, test_loader)
    print(f'Initial accuracy: {init_acc:.2f}%')

    # Federated Training
    print('\n' + '='*60)
    print('Starting Federated Training')
    print('='*60)

    best_acc = 0
    history = []
    start_time = time.time()
    
    for round_idx in range(NUM_ROUNDS):
        t0 = time.time()
        
        # Select 30% clients
        selected = np.random.choice(NUM_CLIENTS, 3, replace=False)
        
        # Train clients
        client_states = []
        for cid in selected:
            client_model = CNN().to(device)
            client_model.load_state_dict(global_model.state_dict())
            state = train_client(client_model, client_loaders[cid], LOCAL_EPOCHS)
            client_states.append(state)
            del client_model
            torch.cuda.empty_cache()
        
        # Aggregate
        avg_state = fed_avg(client_states)
        global_model.load_state_dict(avg_state)
        
        # Evaluate
        acc = evaluate(global_model, test_loader)
        best_acc = max(best_acc, acc)
        history.append(acc)
        
        print(f'Round {round_idx+1:2d}/{NUM_ROUNDS} | Acc: {acc:.2f}% | Best: {best_acc:.2f}% | Time: {time.time()-t0:.1f}s')
        sys.stdout.flush()

    total_time = time.time() - start_time
    
    print('\n' + '='*60)
    print('TRAINING COMPLETE!')
    print('='*60)
    print(f'Final Accuracy: {acc:.2f}%')
    print(f'Best Accuracy: {best_acc:.2f}%')
    print(f'Total Time: {total_time:.1f}s')
    print(f'Accuracy History: {[f"{a:.1f}" for a in history]}')
    print('='*60)
    print('\n*** GPU FEDERATED LEARNING SUCCESS! ***')
    
    return best_acc, history

if __name__ == "__main__":
    main()
