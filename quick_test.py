"""
Quick Test Experiment - Tests just 1 round to verify everything works
"""
import sys
import os
import traceback
from datetime import datetime

LOG_FILE = "quick_test.log"

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()

# Clear log
open(LOG_FILE, "w").close()

log("="*60)
log("QUICK TEST - Verifying experiment setup")
log("="*60)

try:
    log("1. Importing libraries...")
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Subset
    import numpy as np
    log(f"   PyTorch: {torch.__version__}")
    log(f"   CUDA: {torch.cuda.is_available()}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"   Device: {device}")

    log("2. Loading CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    log(f"   Train samples: {len(trainset)}")
    log(f"   Test samples: {len(testset)}")

    log("3. Creating simple CNN model...")
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 64 * 8 * 8)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    model = SimpleCNN().to(device)
    log(f"   Model created on {device}")

    log("4. Quick training test (100 samples, 1 epoch)...")
    small_trainset = Subset(trainset, list(range(100)))
    trainloader = DataLoader(small_trainset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    log(f"   Training loss: {total_loss/len(trainloader):.4f}")

    log("5. Quick evaluation test...")
    small_testset = Subset(testset, list(range(200)))
    testloader = DataLoader(small_testset, batch_size=64)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    log(f"   Test accuracy: {accuracy:.2f}%")

    log("="*60)
    log("QUICK TEST PASSED!")
    log("="*60)
    log("Your system is ready to run the full experiment.")
    log("")
    log("To run the full experiment, use:")
    log("  python ieee_comprehensive_experiment.py")
    log("")
    log("Estimated time: 4-8 hours")

except Exception as e:
    log(f"ERROR: {e}")
    log(traceback.format_exc())
    sys.exit(1)
