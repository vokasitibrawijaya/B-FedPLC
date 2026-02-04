"""
Quick Test: Verify IEEE Comprehensive Experiment Setup
======================================================
This runs a minimal version of the experiment to check:
- CUDA availability
- Data loading works
- Model training works
- File writing works

Run this BEFORE starting the full 8-hour experiment!
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import json
import os
from pathlib import Path

def test_setup():
    print("="*70)
    print("QUICK SETUP VERIFICATION TEST")
    print("="*70)
    print()

    # Test 1: CUDA
    print("1. Testing CUDA availability...")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device('cuda')
        print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print(f"   ⚠ CUDA not available, using CPU (will be VERY slow)")
    print()

    # Test 2: Data Loading
    print("2. Testing data loading...")
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        print(f"   ✓ CIFAR-10 loaded: {len(trainset)} samples")

        # Test small batch
        from torch.utils.data import DataLoader, Subset
        small_set = Subset(trainset, list(range(100)))
        loader = DataLoader(small_set, batch_size=32, shuffle=True)
        batch = next(iter(loader))
        print(f"   ✓ Data loader works: batch shape {batch[0].shape}")
    except Exception as e:
        print(f"   ✗ Data loading failed: {e}")
        return False
    print()

    # Test 3: Model Creation
    print("3. Testing model creation...")
    try:
        class TestCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.fc = nn.Linear(16 * 32 * 32, 10)

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = x.view(x.size(0), -1)
                return self.fc(x)

        model = TestCNN().to(device)
        print(f"   ✓ Model created and moved to {device}")

        # Test forward pass
        x = torch.randn(4, 3, 32, 32).to(device)
        y = model(x)
        print(f"   ✓ Forward pass works: output shape {y.shape}")
    except Exception as e:
        print(f"   ✗ Model creation failed: {e}")
        return False
    print()

    # Test 4: Training
    print("4. Testing training loop...")
    try:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for i, (inputs, labels) in enumerate(loader):
            if i >= 2:  # Just test 2 batches
                break

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"   ✓ Training loop works: loss = {loss.item():.4f}")
    except Exception as e:
        print(f"   ✗ Training failed: {e}")
        return False
    print()

    # Test 5: File Writing
    print("5. Testing file writing...")
    try:
        test_data = {
            "test": "success",
            "cuda": cuda_available,
            "device": str(device)
        }

        with open('_test_output.json', 'w') as f:
            json.dump(test_data, f, indent=2)

        print(f"   ✓ JSON file writing works")

        # Clean up
        if os.path.exists('_test_output.json'):
            os.remove('_test_output.json')
            print(f"   ✓ File cleanup works")
    except Exception as e:
        print(f"   ✗ File writing failed: {e}")
        return False
    print()

    # Test 6: Disk Space
    print("6. Checking disk space...")
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (2**30)
        print(f"   ✓ Free disk space: {free_gb:.1f} GB")
        if free_gb < 5:
            print(f"   ⚠ Warning: Less than 5GB free (recommended: 10GB+)")
    except Exception as e:
        print(f"   ⚠ Could not check disk space: {e}")
    print()

    # Final Summary
    print("="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print()
    print("✅ All tests passed! Ready to run full experiment.")
    print()
    print("To start the comprehensive experiment:")
    print("  Option 1: python ieee_comprehensive_experiment.py")
    print("  Option 2: python run_experiment_now.py")
    print("  Option 3: RUN_COMPREHENSIVE_EXPERIMENT.bat")
    print()
    print("Estimated time: 4-8 hours")
    print("Make sure your computer stays on!")
    print()
    print("="*70)

    return True

if __name__ == "__main__":
    try:
        success = test_setup()
        if not success:
            print()
            print("❌ Some tests failed. Fix errors before running full experiment.")
            exit(1)
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
