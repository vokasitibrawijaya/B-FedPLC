"""
Simple launcher for IEEE Comprehensive Experiment
This will run in the foreground with live output
"""
import subprocess
import sys

print("="*70)
print("STARTING IEEE COMPREHENSIVE EXPERIMENT")
print("="*70)
print()
print("This will take 4-8 hours to complete.")
print("Output will be displayed here in real-time.")
print()
print("Press Ctrl+C to cancel if needed.")
print()
print("="*70)
print()

# Run the experiment
try:
    result = subprocess.run(
        [sys.executable, "ieee_comprehensive_experiment.py"],
        check=True
    )
    print()
    print("="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*70)
except KeyboardInterrupt:
    print()
    print("="*70)
    print("EXPERIMENT CANCELLED BY USER")
    print("="*70)
except Exception as e:
    print()
    print("="*70)
    print(f"ERROR: {e}")
    print("="*70)
