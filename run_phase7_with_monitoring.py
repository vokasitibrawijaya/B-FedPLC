"""
Run Phase 7 experiment dengan monitoring progress
Backup old results dan run experiment baru
"""

import os
import json
import shutil
from datetime import datetime

def backup_old_results():
    """Backup old results file"""
    if os.path.exists('phase7_results.json'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f'phase7_results_OLD_{timestamp}.json'
        shutil.copy('phase7_results.json', backup_name)
        print(f"✅ Backed up old results to: {backup_name}")
        return backup_name
    return None

def check_progress_periodically():
    """Check progress every 2 minutes"""
    import time
    import subprocess
    
    print("\n" + "="*70)
    print("STARTING PHASE 7 EXPERIMENT WITH MONITORING")
    print("="*70)
    
    # Backup old results
    backup_old_results()
    
    print("\nStarting experiment...")
    print("This will take 30-60 minutes. Progress will be checked every 2 minutes.\n")
    
    # Start experiment in background
    process = subprocess.Popen(
        ['python', 'phase7_sota_comparison.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print(f"Process started with PID: {process.pid}")
    print("\nMonitoring progress...\n")
    
    check_count = 0
    while True:
        time.sleep(120)  # Check every 2 minutes
        check_count += 1
        
        # Check if process is still running
        if process.poll() is not None:
            print(f"\n{'='*70}")
            print("EXPERIMENT FINISHED!")
            print(f"{'='*70}\n")
            
            # Get output
            stdout, stderr = process.communicate()
            if stdout:
                print("STDOUT (last 20 lines):")
                print('\n'.join(stdout.split('\n')[-20:]))
            if stderr:
                print("\nSTDERR:")
                print(stderr)
            
            break
        
        # Check progress
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Check #{check_count} - Process still running...")
        
        if os.path.exists('phase7_results.json'):
            try:
                with open('phase7_results.json', 'r') as f:
                    data = json.load(f)
                
                scenarios = len(data)
                print(f"  Scenarios completed: {scenarios}/8")
                
                # Check B-FedPLC
                bfedplc_count = sum(1 for s in data.values() if 'B-FedPLC' in s)
                print(f"  B-FedPLC results: {bfedplc_count}/8")
                
                # Show latest B-FedPLC result
                for scenario, methods in data.items():
                    if 'B-FedPLC' in methods and 'mean' in methods['B-FedPLC']:
                        mean = methods['B-FedPLC']['mean']
                        print(f"  Latest: {scenario} = {mean:.2f}%")
                        break
            except:
                print("  (Results file exists but not yet readable)")
        else:
            print("  (Results file not created yet)")
    
    # Final check
    print("\n" + "="*70)
    print("FINAL RESULTS CHECK")
    print("="*70)
    
    if os.path.exists('phase7_results.json'):
        with open('phase7_results.json', 'r') as f:
            data = json.load(f)
        
        print("\nB-FedPLC Final Results:")
        baseline = 9.8
        for scenario in sorted(data.keys()):
            if 'B-FedPLC' in data[scenario]:
                result = data[scenario]['B-FedPLC']
                if 'mean' in result:
                    mean = result['mean']
                    improvement = mean - baseline
                    status = "[SUCCESS]" if mean >= 95 else "[PARTIAL]" if mean >= 50 else "[FAIL]"
                    print(f"  {scenario:30s}: {mean:6.2f}% (improvement: {improvement:+.2f}%) {status}")

if __name__ == "__main__":
    try:
        check_progress_periodically()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        print("Experiment may still be running in background")
