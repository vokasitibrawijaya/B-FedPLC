"""Monitor experiment progress"""
import time
import os
from pathlib import Path

output_file = Path("experiment_output_new.txt")
result_file = Path("ieee_comprehensive_results.json")

print("Monitoring IEEE Access Experiments...")
print("=" * 60)

last_size = 0
check_count = 0

while check_count < 120:  # Monitor for up to 10 minutes
    check_count += 1

    # Check output file
    if output_file.exists():
        size = output_file.stat().st_size
        if size > last_size:
            print(f"\n[{time.strftime('%H:%M:%S')}] Output file updated: {size} bytes")
            # Read last 20 lines
            with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if lines:
                    print("Last 10 lines:")
                    for line in lines[-10:]:
                        print(f"  {line.rstrip()}")
            last_size = size

    # Check result file
    if result_file.exists():
        print(f"\n✓ Results file created: {result_file}")
        print("Experiment completed successfully!")
        break

    time.sleep(5)  # Check every 5 seconds

print("\nMonitoring ended.")
