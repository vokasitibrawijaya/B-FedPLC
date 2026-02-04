"""
Monitor IEEE Comprehensive Experiment Progress
"""
import os
import json
import time
from pathlib import Path
from datetime import datetime

def check_progress():
    print("="*70)
    print("IEEE COMPREHENSIVE EXPERIMENT MONITOR")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check if output file exists
    output_file = Path("ieee_comprehensive_output.txt")
    if output_file.exists():
        size = output_file.stat().st_size
        print(f"✓ Output file: {size:,} bytes")

        # Read last 30 lines
        try:
            with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if lines:
                    print("\nLast 30 lines of output:")
                    print("-" * 70)
                    for line in lines[-30:]:
                        print(line.rstrip())
                else:
                    print("⏳ Output file is empty (experiment initializing...)")
        except Exception as e:
            print(f"Error reading output: {e}")
    else:
        print("❌ Output file not found")
        print("   Run: RUN_COMPREHENSIVE_EXPERIMENT.bat")

    print()

    # Check if results file exists
    results_file = Path("ieee_comprehensive_results.json")
    if results_file.exists():
        print("✓ Results file created!")
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
                print(f"   Experiments completed: {len(results)}")
        except:
            print("   (file exists but may still be writing)")
    else:
        print("⏳ Results file not created yet (experiment still running)")

    print()
    print("="*70)
    print("To monitor continuously, run this script every 10 minutes")
    print("="*70)

if __name__ == "__main__":
    check_progress()
