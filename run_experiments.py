"""
Script to run all experiments for FedPLC replication
Includes baseline comparisons and ablation studies
"""

import os
import subprocess
import sys
from datetime import datetime


def run_command(cmd: str, description: str):
    """Run a command and print status"""
    print("\n" + "="*70)
    print(f"Running: {description}")
    print("="*70)
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"[ERROR] {description} failed with return code {result.returncode}")
    else:
        print(f"[SUCCESS] {description} completed")
    
    return result.returncode


def main():
    # Base command
    python = sys.executable
    
    experiments = []
    
    # =========================================================================
    # 1. Main Experiments (Table 1 in paper)
    # =========================================================================
    
    # CIFAR-10 experiments
    experiments.append({
        'name': 'CIFAR-10 - No Drift',
        'cmd': f'{python} train.py --dataset cifar10 --num_rounds 200 --drift_type none --alpha 0.5'
    })
    
    experiments.append({
        'name': 'CIFAR-10 - Abrupt Drift',
        'cmd': f'{python} train.py --dataset cifar10 --num_rounds 200 --drift_type abrupt --drift_round 100 --alpha 0.5'
    })
    
    experiments.append({
        'name': 'CIFAR-10 - Incremental Drift',
        'cmd': f'{python} train.py --dataset cifar10 --num_rounds 200 --drift_type incremental --alpha 0.5'
    })
    
    # Fashion-MNIST experiments
    experiments.append({
        'name': 'Fashion-MNIST - No Drift',
        'cmd': f'{python} train.py --dataset fmnist --num_rounds 200 --drift_type none --alpha 0.5'
    })
    
    experiments.append({
        'name': 'Fashion-MNIST - Abrupt Drift',
        'cmd': f'{python} train.py --dataset fmnist --num_rounds 200 --drift_type abrupt --drift_round 100 --alpha 0.5'
    })
    
    # SVHN experiments
    experiments.append({
        'name': 'SVHN - No Drift',
        'cmd': f'{python} train.py --dataset svhn --num_rounds 200 --drift_type none --alpha 0.5'
    })
    
    experiments.append({
        'name': 'SVHN - Abrupt Drift',
        'cmd': f'{python} train.py --dataset svhn --num_rounds 200 --drift_type abrupt --drift_round 100 --alpha 0.5'
    })
    
    # =========================================================================
    # 2. Ablation Studies
    # =========================================================================
    
    # Different alpha values (heterogeneity levels)
    for alpha in [0.1, 0.3, 0.5, 1.0]:
        experiments.append({
            'name': f'Ablation - Alpha={alpha}',
            'cmd': f'{python} train.py --dataset cifar10 --num_rounds 200 --drift_type none --alpha {alpha}'
        })
    
    # Different similarity thresholds
    for threshold in [0.7, 0.8, 0.85, 0.9, 0.95]:
        experiments.append({
            'name': f'Ablation - Threshold={threshold}',
            'cmd': f'{python} train.py --dataset cifar10 --num_rounds 200 --similarity_threshold {threshold}'
        })
    
    # Different PARL weights
    for parl_weight in [0.0, 0.05, 0.1, 0.2, 0.5]:
        experiments.append({
            'name': f'Ablation - PARL_weight={parl_weight}',
            'cmd': f'{python} train.py --dataset cifar10 --num_rounds 200 --parl_weight {parl_weight}'
        })
    
    # =========================================================================
    # Run experiments
    # =========================================================================
    
    print("\n" + "="*70)
    print("FedPLC Replication - Experiment Suite")
    print(f"Total experiments: {len(experiments)}")
    print("="*70)
    
    # Ask which experiments to run
    print("\nAvailable experiments:")
    for i, exp in enumerate(experiments):
        print(f"  [{i+1}] {exp['name']}")
    
    print("\nOptions:")
    print("  Enter experiment numbers (e.g., '1,2,3' or '1-5')")
    print("  Enter 'all' to run all experiments")
    print("  Enter 'main' to run main experiments (1-7)")
    print("  Enter 'ablation' to run ablation studies (8+)")
    print("  Enter 'q' to quit")
    
    choice = input("\nYour choice: ").strip().lower()
    
    if choice == 'q':
        print("Exiting...")
        return
    
    selected = []
    
    if choice == 'all':
        selected = list(range(len(experiments)))
    elif choice == 'main':
        selected = list(range(7))
    elif choice == 'ablation':
        selected = list(range(7, len(experiments)))
    else:
        # Parse selection
        parts = choice.split(',')
        for part in parts:
            if '-' in part:
                start, end = part.split('-')
                selected.extend(range(int(start)-1, int(end)))
            else:
                selected.append(int(part)-1)
    
    # Validate selection
    selected = [s for s in selected if 0 <= s < len(experiments)]
    
    if len(selected) == 0:
        print("No valid experiments selected. Exiting...")
        return
    
    print(f"\nRunning {len(selected)} experiments...")
    
    # Run selected experiments
    results = []
    for idx in selected:
        exp = experiments[idx]
        start_time = datetime.now()
        
        ret_code = run_command(exp['cmd'], exp['name'])
        
        elapsed = datetime.now() - start_time
        results.append({
            'name': exp['name'],
            'success': ret_code == 0,
            'elapsed': str(elapsed)
        })
    
    # Summary
    print("\n" + "="*70)
    print("Experiment Summary")
    print("="*70)
    
    successful = sum(1 for r in results if r['success'])
    print(f"\nCompleted: {successful}/{len(results)} experiments")
    
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"  {status} {r['name']} ({r['elapsed']})")


if __name__ == '__main__':
    main()
