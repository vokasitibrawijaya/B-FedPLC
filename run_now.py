"""
Simplified IEEE Experiment Runner
Runs the comprehensive experiment with live output
"""
import sys
import os

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

print("=" * 70, flush=True)
print("STARTING IEEE COMPREHENSIVE EXPERIMENT", flush=True)
print("=" * 70, flush=True)
print(flush=True)

# Import and run
try:
    print("Importing experiment module...", flush=True)
    from ieee_comprehensive_experiment import main, config
    
    print(f"Device: {config.device}", flush=True)
    print("Starting experiments...", flush=True)
    print(flush=True)
    
    results = main()
    
    print(flush=True)
    print("=" * 70, flush=True)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 70, flush=True)
    print("Results saved to: ieee_comprehensive_results.json", flush=True)
    
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
