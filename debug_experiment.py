"""Debug version of FedPLC experiment"""
import sys
print("=== DEBUG START ===", flush=True)

try:
    print("Step 1: Importing os...", flush=True)
    import os
    print("Step 1: OK", flush=True)
    
    print("Step 2: Importing time...", flush=True)
    import time
    print("Step 2: OK", flush=True)
    
    print("Step 3: Importing torch...", flush=True)
    import torch
    print(f"Step 3: OK - PyTorch {torch.__version__}", flush=True)
    print(f"CUDA: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    
    print("Step 4: Importing numpy...", flush=True)
    import numpy as np
    print("Step 4: OK", flush=True)
    
    print("Step 5: Importing tqdm...", flush=True)
    from tqdm import tqdm
    print("Step 5: OK", flush=True)
    
    print("Step 6: Adding path...", flush=True)
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    print("Step 6: OK", flush=True)
    
    print("Step 7: Importing config...", flush=True)
    from fedplc.config import ExperimentConfig
    print("Step 7: OK", flush=True)
    
    print("Step 8: Importing data...", flush=True)
    from fedplc.data import create_federated_dataloaders, ConceptDriftSimulator
    print("Step 8: OK", flush=True)
    
    print("Step 9: Importing models...", flush=True)
    from fedplc.models import create_model
    print("Step 9: OK", flush=True)
    
    print("Step 10: Importing server...", flush=True)
    from fedplc.server import FedPLCServer, FedPLCClient
    print("Step 10: OK", flush=True)
    
    print("Step 11: Importing utils...", flush=True)
    from fedplc.utils import set_seed
    print("Step 11: OK", flush=True)
    
    print("\n=== ALL IMPORTS SUCCESSFUL ===", flush=True)
    
    print("\nStep 12: Creating config...", flush=True)
    config = ExperimentConfig(
        dataset='cifar10',
        num_clients=10,  # Small number for testing
        num_rounds=5,
        local_epochs=1,
        batch_size=32,
        participation_rate=0.3,
        data_dir='./data'
    )
    print("Step 12: OK", flush=True)
    
    print("\nStep 13: Setting seed...", flush=True)
    set_seed(42)
    print("Step 13: OK", flush=True)
    
    print("\nStep 14: Creating dataloaders...", flush=True)
    train_loaders, test_loader, partitioner = create_federated_dataloaders(
        dataset_name=config.dataset,
        num_clients=config.num_clients,
        alpha=config.alpha,
        batch_size=config.batch_size,
        data_dir=config.data_dir
    )
    print(f"Step 14: OK - {len(train_loaders)} train loaders created", flush=True)
    
    print("\nStep 15: Creating model...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model = create_model(config.dataset).to(device)
    print(f"Step 15: OK - Model on {device}", flush=True)
    
    print("\nStep 16: Creating server...", flush=True)
    server = FedPLCServer(
        config=config,
        model=global_model,
        test_loader=test_loader
    )
    print("Step 16: OK", flush=True)
    
    print("\nStep 17: Creating clients...", flush=True)
    clients = []
    for i in range(config.num_clients):
        client = FedPLCClient(
            client_id=i,
            config=config,
            dataloader=train_loaders[i]
        )
        clients.append(client)
        print(f"  Client {i+1}/{config.num_clients} created", flush=True)
    print("Step 17: OK", flush=True)
    
    print("\n=== SETUP COMPLETE ===", flush=True)
    print("Ready to run training loop!", flush=True)
    
except Exception as e:
    print(f"\n!!! ERROR: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== DEBUG END ===", flush=True)
