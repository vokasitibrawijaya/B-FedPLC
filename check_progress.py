"""Quick progress checker"""
import json
import os
from datetime import datetime

def check_progress():
    print(f"\n{'='*70}")
    print(f"PROGRESS CHECK - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Check file
    if not os.path.exists('phase7_results.json'):
        print("Status: Results file not created yet")
        print("Experiment may be starting...")
        return
    
    # Read results
    with open('phase7_results.json', 'r') as f:
        data = json.load(f)
    
    scenarios = list(data.keys())
    print(f"Scenarios completed: {len(scenarios)}/8 (expected)")
    print(f"Scenarios: {', '.join(scenarios[:4])}...")
    
    # Check B-FedPLC
    bfedplc_results = {}
    for s in scenarios:
        if 'B-FedPLC' in data[s]:
            bfedplc_results[s] = data[s]['B-FedPLC']
    
    print(f"\nB-FedPLC Results Found: {len(bfedplc_results)}")
    
    if bfedplc_results:
        print("\nB-FedPLC Accuracy:")
        baseline = 9.8
        for scenario in sorted(bfedplc_results.keys()):
            result = bfedplc_results[scenario]
            if 'mean' in result:
                mean = result['mean']
                std = result.get('std', 0)
                improvement = mean - baseline
                status = "[SUCCESS]" if mean >= 95 else "[PARTIAL]" if mean >= 50 else "[FAIL]"
                print(f"  {scenario:30s}: {mean:6.2f}% ± {std:.2f}% (improvement: {improvement:+.2f}%) {status}")
    
    # Check if experiment might be running (file recently modified)
    mtime = os.path.getmtime('phase7_results.json')
    age_seconds = datetime.now().timestamp() - mtime
    age_minutes = age_seconds / 60
    
    print(f"\nFile Status:")
    print(f"  Last modified: {age_minutes:.1f} minutes ago")
    if age_minutes < 2:
        print(f"  Status: [ACTIVE] - File recently updated, experiment likely running")
    elif age_minutes < 10:
        print(f"  Status: [RECENT] - File updated recently")
    else:
        print(f"  Status: [STALE] - File not updated recently, experiment may have finished")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    check_progress()
