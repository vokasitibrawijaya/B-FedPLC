"""
Analyze Existing Results and Prepare for IEEE ACCESS
====================================================
This script analyzes all available results and provides guidance
on what additional experiments are needed for IEEE ACCESS.
"""
import json
import os
from pathlib import Path
import numpy as np
from scipy import stats

def load_json_if_exists(filename):
    """Load JSON file if it exists"""
    if Path(filename).exists():
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return None

def analyze_results():
    print("="*70)
    print("IEEE ACCESS - EXISTING RESULTS ANALYSIS")
    print("="*70)
    print()

    # Check all result files
    result_files = {
        'ieee_experiment_results.json': 'IEEE Experiments (Statistical Rigor)',
        'ieee_comprehensive_results.json': 'Comprehensive Experiments (MAIN)',
        'all_experiments_results.json': 'All Experiments',
        'comparative_analysis_results.json': 'Comparative Analysis',
        'phase7_results.json': 'Phase 7 Results',
        'multiseed_results.json': 'Multi-seed Results',
    }

    available_results = {}
    missing_results = []

    print("CHECKING AVAILABLE RESULTS:")
    print("-" * 70)
    for filename, description in result_files.items():
        data = load_json_if_exists(filename)
        if data:
            available_results[filename] = data
            print(f"[OK] {description}")
            print(f"  File: {filename}")
            print(f"  Size: {len(json.dumps(data)):,} bytes")
        else:
            missing_results.append((filename, description))
            print(f"[MISSING] {description}")
            print(f"  File: {filename} - NOT FOUND")
        print()

    print()
    print("="*70)
    print("ANALYSIS OF AVAILABLE DATA")
    print("="*70)
    print()

    # Analyze ieee_experiment_results.json
    if 'ieee_experiment_results.json' in available_results:
        data = available_results['ieee_experiment_results.json']
        print("1. IEEE EXPERIMENT RESULTS:")
        print("-" * 70)

        if 'statistical_rigor' in data:
            print("   [OK] Statistical Rigor Test")
            rigor = data['statistical_rigor']['results']
            print(f"     Methods tested: {len(rigor)}")
            print(f"     Methods: {', '.join(rigor.keys())}")

            # Analyze results
            for method, byzantine_results in rigor.items():
                print(f"\n   {method.upper()}:")
                for byz_frac, accuracies in byzantine_results.items():
                    mean_acc = np.mean(accuracies)
                    std_acc = np.std(accuracies)
                    print(f"     Byzantine {byz_frac}: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print()

    # Analyze all_experiments_results.json
    if 'all_experiments_results.json' in available_results:
        data = available_results['all_experiments_results.json']
        print("2. ALL EXPERIMENTS RESULTS:")
        print("-" * 70)

        if 'ablation' in data:
            print("   [OK] Ablation Study")
            for config, results in data['ablation'].items():
                final_acc = results.get('final_accuracy', 0)
                print(f"     {config}: {final_acc:.2f}%")

        if 'byzantine' in data:
            print("\n   [OK] Byzantine Robustness Test")

        print()

    # Analyze comparative_analysis_results.json
    if 'comparative_analysis_results.json' in available_results:
        data = available_results['comparative_analysis_results.json']
        print("3. COMPARATIVE ANALYSIS:")
        print("-" * 70)

        if 'results' in data:
            print("   [OK] Method Comparison")
            for result in data['results']:
                name = result['name']
                if 'history' in result and 'accuracy' in result['history']:
                    final_acc = result['history']['accuracy'][-1]
                    print(f"     {name}: {final_acc:.2f}%")
        print()

    print()
    print("="*70)
    print("WHAT'S NEEDED FOR IEEE ACCESS SUBMISSION?")
    print("="*70)
    print()

    critical_missing = []

    # Check for comprehensive results (most important)
    if 'ieee_comprehensive_results.json' not in available_results:
        critical_missing.append("ieee_comprehensive_results.json")
        print("[CRITICAL] ieee_comprehensive_results.json MISSING")
        print("   This is the MAIN experiment file required for IEEE ACCESS")
        print("   It should contain:")
        print("   - Ablation study (10 seeds)")
        print("   - Byzantine resilience (multiple attack fractions)")
        print("   - Scalability tests")
        print("   - Stress tests (combined challenges)")
        print()
        print("   ACTION REQUIRED:")
        print("   Run: RUN_COMPREHENSIVE_EXPERIMENT.bat")
        print("   Estimated time: 4-8 hours")
        print()
    else:
        print("[OK] CRITICAL: ieee_comprehensive_results.json EXISTS")
        print("  Main experiment data is available")
        print()

    # Check visualization readiness
    print("VISUALIZATION STATUS:")
    print("-" * 70)

    plot_files = list(Path('.').glob('*ieee*.png'))
    if plot_files:
        print(f"[OK] Found {len(plot_files)} IEEE visualizations")
        for pf in plot_files[:5]:
            print(f"  - {pf.name}")
        if len(plot_files) > 5:
            print(f"  ... and {len(plot_files)-5} more")
    else:
        print("[MISSING] No IEEE visualizations found")
        if 'ieee_comprehensive_results.json' in available_results:
            print("   ACTION: Run generate_ieee_visualizations.py")
        else:
            print("   WAITING: Need ieee_comprehensive_results.json first")
    print()

    print()
    print("="*70)
    print("NEXT STEPS SUMMARY")
    print("="*70)
    print()

    if critical_missing:
        print("[CRITICAL] EXPERIMENTS NEEDED:")
        print()
        print("1. Run comprehensive experiment:")
        print("   > RUN_COMPREHENSIVE_EXPERIMENT.bat")
        print("   - This will take 4-8 hours")
        print("   - Monitor with: python monitor_comprehensive.py")
        print()
        print("2. After completion, generate visualizations:")
        print("   > python generate_ieee_visualizations.py")
        print()
        print("3. Review results and start writing paper")
        print()
    else:
        print("[SUCCESS] ALL CRITICAL DATA AVAILABLE!")
        print()
        print("1. Generate visualizations (if not done):")
        print("   > python generate_ieee_visualizations.py")
        print()
        print("2. Review all results and plots")
        print()
        print("3. Start writing IEEE ACCESS paper")
        print("   - Use LAPORAN_LATEX.tex as starting point")
        print("   - Focus on scenarios where B-FedPLC excels")
        print("   - Include all statistical analysis (p-values, CIs)")
        print()

    print("="*70)

    return available_results, missing_results

if __name__ == "__main__":
    available, missing = analyze_results()
