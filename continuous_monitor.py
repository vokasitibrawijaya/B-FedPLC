"""
Continuous Progress Tracker for IEEE Experiments
Checks every 30 seconds and provides status updates
"""
import time
import os
from pathlib import Path
from datetime import datetime

def check_status():
    result_file = Path("ieee_comprehensive_results.json")
    output_file = Path("experiment_live_output.txt")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status Check:")
    print("-" * 60)

    if result_file.exists():
        print("✅ EXPERIMENT COMPLETED!")
        print(f"   Results file: {result_file}")
        print(f"   Size: {result_file.stat().st_size / 1024:.2f} KB")
        return True

    if output_file.exists():
        size = output_file.stat().st_size
        print(f"🔄 Experiment running... Output: {size / 1024:.2f} KB")

        # Try to read and show progress
        try:
            with open(output_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            # Find recent experiment markers
            recent = []
            for line in reversed(lines[-50:]):
                clean_line = line.strip()
                if any(keyword in clean_line for keyword in ['Running:', 'EXPERIMENT', 'Best:', 'Final:', '---']):
                    recent.insert(0, clean_line)
                    if len(recent) >= 10:
                        break

            if recent:
                print("\n📝 Recent progress:")
                for line in recent[-10:]:
                    if line:
                        print(f"   {line}")
        except Exception as e:
            print(f"   (Could not read details: {e})")
    else:
        print("⏳ Waiting for experiment to start...")

    return False

def main():
    print("="*70)
    print("IEEE ACCESS EXPERIMENT CONTINUOUS MONITOR")
    print("="*70)
    print("Checking every 30 seconds... Press Ctrl+C to stop")

    completed = False
    iterations = 0
    max_iterations = 120  # Max 1 hour monitoring

    while not completed and iterations < max_iterations:
        completed = check_status()

        if not completed:
            iterations += 1
            print(f"\n⏰ Next check in 30 seconds... ({iterations}/{max_iterations})")
            time.sleep(30)
        else:
            print("\n🎉 Monitoring complete!")
            break

    if iterations >= max_iterations:
        print("\n⚠️ Max monitoring time reached (1 hour)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
