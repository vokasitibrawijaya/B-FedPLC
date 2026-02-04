"""Quick status check for running experiments"""
import os
from pathlib import Path
import time

print("\n" + "="*70)
print("IEEE ACCESS EXPERIMENT STATUS CHECK")
print("="*70)

# Check if experiment is running
output_file = Path("experiment_live_output.txt")
result_file = Path("ieee_comprehensive_results.json")

if result_file.exists():
    print("\n✅ EXPERIMENT COMPLETED!")
    print(f"   Results saved to: {result_file}")

    # Show file size and modification time
    import json
    with open(result_file, 'r') as f:
        results = json.load(f)

    print(f"\n📊 RESULTS SUMMARY:")
    for exp_name in results:
        print(f"   - {exp_name}")

elif output_file.exists():
    print("\n🔄 EXPERIMENT IN PROGRESS...")

    # Read and show last lines
    with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    print(f"\n📝 Last 40 lines of output:\n")
    for line in lines[-40:]:
        print(line.rstrip())

    print(f"\n💾 Total output size: {len(lines)} lines")

else:
    print("\n⏳ WAITING FOR EXPERIMENT TO START...")
    print("   No output file found yet.")

print("\n" + "="*70)
