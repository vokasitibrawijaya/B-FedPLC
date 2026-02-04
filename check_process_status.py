"""
Simple process monitor - check if experiment is making progress
"""
import psutil
import time

def find_experiment_processes():
    """Find all Python processes that might be running experiments"""
    experiment_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'ieee_comprehensive_experiment' in cmdline or 'experiment' in cmdline:
                    experiment_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return experiment_procs

def main():
    print("="*70)
    print("EXPERIMENT PROCESS MONITOR")
    print("="*70)
    
    procs = find_experiment_processes()
    
    if not procs:
        print("\n❌ No experiment processes found running!")
        print("\nTo start experiments:")
        print("  python ieee_comprehensive_experiment.py")
    else:
        print(f"\n✅ Found {len(procs)} experiment process(es):\n")
        for proc in procs:
            try:
                info = proc.as_dict(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info', 'create_time'])
                print(f"PID: {info['pid']}")
                print(f"Name: {info['name']}")
                print(f"Command: {' '.join(info['cmdline'][:3]) if info['cmdline'] else 'N/A'}")
                print(f"CPU: {proc.cpu_percent(interval=1):.1f}%")
                print(f"Memory: {info['memory_info'].rss / 1024 / 1024:.2f} MB")
                
                # Calculate runtime
                runtime = time.time() - info['create_time']
                hours = int(runtime // 3600)
                minutes = int((runtime % 3600) // 60)
                print(f"Runtime: {hours}h {minutes}m")
                print("-" * 50)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    
    # Check for results file
    import os
    if os.path.exists('ieee_comprehensive_results.json'):
        print("\n🎉 RESULTS FILE EXISTS!")
        size = os.path.getsize('ieee_comprehensive_results.json')
        print(f"   Size: {size / 1024:.2f} KB")
    else:
        print("\n⏳ Results file not created yet...")
        print("   Experiment still running")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
