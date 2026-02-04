"""
Monitor Progress untuk Phase 7 Experiment
Check status, progress, dan hasil sementara
"""

import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

def check_process_running():
    """Check if phase7 experiment is still running"""
    try:
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
            capture_output=True,
            text=True,
            shell=True
        )
        # Check if phase7_sota_comparison.py is in the command line
        # This is a simple check - in reality we'd need to check the full command
        return 'python.exe' in result.stdout
    except:
        return False

def check_results_file():
    """Check if results file exists and get last modified time"""
    results_file = Path('phase7_results.json')
    if results_file.exists():
        mtime = results_file.stat().st_mtime
        size = results_file.stat().st_size
        return {
            'exists': True,
            'modified': datetime.fromtimestamp(mtime),
            'size': size,
            'size_kb': size / 1024
        }
    return {'exists': False}

def parse_results():
    """Parse results file if exists"""
    results_file = Path('phase7_results.json')
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Count completed experiments
        total_scenarios = len(data)
        completed_methods = {}
        
        for scenario, methods in data.items():
            for method, result in methods.items():
                if method not in completed_methods:
                    completed_methods[method] = 0
                if 'mean' in result:
                    completed_methods[method] += 1
        
        # Get B-FedPLC results specifically
        bfedplc_results = {}
        for scenario, methods in data.items():
            if 'B-FedPLC' in methods:
                bfedplc_results[scenario] = methods['B-FedPLC']
        
        return {
            'total_scenarios': total_scenarios,
            'completed_methods': completed_methods,
            'bfedplc_results': bfedplc_results,
            'data': data
        }
    except Exception as e:
        return {'error': str(e)}

def estimate_progress(results_data):
    """Estimate progress based on completed experiments"""
    if not results_data or 'error' in results_data:
        return None
    
    # Expected: 8 scenarios (2 datasets × 4 experiment types)
    expected_scenarios = 8
    completed = results_data['total_scenarios']
    
    # Expected: 9 methods per scenario
    total_expected = expected_scenarios * 9
    
    # Count completed
    total_completed = sum(results_data['completed_methods'].values())
    
    progress_pct = (total_completed / total_expected) * 100 if total_expected > 0 else 0
    
    return {
        'scenarios_completed': completed,
        'scenarios_expected': expected_scenarios,
        'experiments_completed': total_completed,
        'experiments_expected': total_expected,
        'progress_percent': progress_pct
    }

def print_status():
    """Print current status"""
    print("\n" + "=" * 70)
    print(f"PROGRESS MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Check process
    process_running = check_process_running()
    print(f"\nProcess Status: {'[RUNNING]' if process_running else '[NOT DETECTED]'}")
    
    # Check results file
    file_info = check_results_file()
    if file_info['exists']:
        print(f"\nResults File:")
        print(f"  Status: EXISTS")
        print(f"  Size: {file_info['size_kb']:.2f} KB")
        print(f"  Last Modified: {file_info['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Parse results
        results = parse_results()
        if results and 'error' not in results:
            progress = estimate_progress(results)
            
            print(f"\nProgress:")
            if progress:
                print(f"  Scenarios: {progress['scenarios_completed']}/{progress['scenarios_expected']}")
                print(f"  Experiments: {progress['experiments_completed']}/{progress['experiments_expected']}")
                print(f"  Progress: {progress['progress_percent']:.1f}%")
            
            # Show B-FedPLC results
            if results['bfedplc_results']:
                print(f"\nB-FedPLC Results (Partial):")
                for scenario, result in list(results['bfedplc_results'].items())[:4]:
                    if 'mean' in result:
                        mean_acc = result['mean']
                        std_acc = result.get('std', 0)
                        print(f"  {scenario:30s}: {mean_acc:6.2f}% ± {std_acc:.2f}%")
            
            # Show comparison with baseline
            if results['bfedplc_results']:
                print(f"\nKey Metrics:")
                baseline = 9.8
                for scenario in ['MNIST_byzantine_20', 'CIFAR10_byzantine_20']:
                    if scenario in results['bfedplc_results']:
                        mean_acc = results['bfedplc_results'][scenario]['mean']
                        improvement = mean_acc - baseline
                        status = "[SUCCESS]" if mean_acc >= 95 else "[PARTIAL]" if mean_acc >= 50 else "[FAIL]"
                        print(f"  {scenario:30s}: {mean_acc:6.2f}% (improvement: {improvement:+.2f}%) {status}")
        elif results and 'error' in results:
            print(f"\nError parsing results: {results['error']}")
    else:
        print(f"\nResults File: NOT YET CREATED")
        print("  Experiment may still be in early stages...")
    
    print("\n" + "=" * 70)

def main():
    """Main monitoring loop"""
    print("Starting Progress Monitor for Phase 7 Experiment")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            print_status()
            print("\nNext check in 30 seconds... (Ctrl+C to stop)")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        print_status()  # Final status

if __name__ == "__main__":
    main()
