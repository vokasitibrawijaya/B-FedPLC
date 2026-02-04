"""
Simple experiment runner that logs everything to file
"""
import sys
import os
import traceback
from datetime import datetime

# Log file
LOG_FILE = "experiment_progress.log"

def log(msg):
    """Write to both console and log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()

# Clear log file
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("")

log("="*60)
log("STARTING IEEE COMPREHENSIVE EXPERIMENT")
log("="*60)

try:
    log("Step 1: Importing PyTorch...")
    import torch
    log(f"  PyTorch version: {torch.__version__}")
    log(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.get_device_name(0)}")

    log("Step 2: Importing experiment module...")
    from ieee_comprehensive_experiment import main, config
    log(f"  Config loaded, device: {config.device}")

    log("Step 3: Starting main experiment...")
    log("  This will take 4-8 hours...")
    log("-"*60)

    results = main()

    log("-"*60)
    log("Step 4: Experiment completed!")
    log("  Results saved to: ieee_comprehensive_results.json")
    log("="*60)
    log("SUCCESS!")
    log("="*60)

except Exception as e:
    log(f"ERROR: {e}")
    log(traceback.format_exc())
    sys.exit(1)
