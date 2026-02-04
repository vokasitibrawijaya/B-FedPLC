"""Quick test script"""
import sys
print("Script started!", flush=True)
sys.stdout.flush()

print("Importing torch...", flush=True)
import torch
print(f"PyTorch version: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    
print("All imports OK!", flush=True)
