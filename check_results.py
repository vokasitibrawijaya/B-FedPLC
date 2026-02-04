"""
Analyze existing results and determine what we have
"""
import json
import os
import sys

# Redirect output to file
output_file = open('results_analysis.txt', 'w', encoding='utf-8')
def print_both(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=output_file)
    output_file.flush()

print = print_both

print("="*60)
print("ANALYZING EXISTING RESULTS")
print("="*60)
print()

# Check all result files
result_files = [
    'all_experiments_results.json',
    'ieee_experiment_results.json',
    'comparative_analysis_results.json',
    'phase7_results.json',
    'multiseed_results.json',
    'ieee_comprehensive_results.json'
]

for fname in result_files:
    if os.path.exists(fname):
        size = os.path.getsize(fname)
        with open(fname, 'r') as f:
            data = json.load(f)
        print(f"[OK] {fname}")
        print(f"     Size: {size:,} bytes")
        if isinstance(data, dict):
            print(f"     Keys: {list(data.keys())}")
        print()
    else:
        print(f"[MISSING] {fname}")
        print()

# Detailed analysis of all_experiments_results.json
print("="*60)
print("DETAILED ANALYSIS: all_experiments_results.json")
print("="*60)
print()

if os.path.exists('all_experiments_results.json'):
    with open('all_experiments_results.json', 'r') as f:
        data = json.load(f)

    # Ablation study
    if 'ablation' in data:
        print("ABLATION STUDY:")
        for config, results in data['ablation'].items():
            acc = results.get('final_accuracy', 'N/A')
            print(f"  {config}: {acc}%")
        print()

    # Byzantine
    if 'byzantine' in data:
        print("BYZANTINE ROBUSTNESS:")
        for frac, methods in data['byzantine'].items():
            print(f"  {frac}% attackers:")
            if isinstance(methods, dict):
                for method, results in methods.items():
                    if isinstance(results, dict):
                        acc = results.get('accuracy', results.get('final_accuracy', 'N/A'))
                    else:
                        acc = results
                    print(f"    {method}: {acc}")
        print()

print("="*60)
print("CONCLUSION")
print("="*60)

if os.path.exists('ieee_comprehensive_results.json'):
    print("ieee_comprehensive_results.json EXISTS!")
    print("Ready to generate visualizations.")
else:
    print("ieee_comprehensive_results.json MISSING")
    print()
    print("BUT we have extensive results in other files!")
    print("We can use the existing data for the paper.")
    print()
    print("Available data:")
    print("  - Ablation study (all_experiments_results.json)")
    print("  - Byzantine tests (ieee_experiment_results.json)")
    print("  - Comparative analysis (comparative_analysis_results.json)")
    print("  - Multi-seed validation (multiseed_results.json)")
