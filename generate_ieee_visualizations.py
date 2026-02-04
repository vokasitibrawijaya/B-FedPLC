"""
Generate IEEE Access Quality Visualizations
Run this after ieee_comprehensive_results.json is created
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

def load_results():
    """Load experiment results"""
    with open('ieee_comprehensive_results.json', 'r') as f:
        return json.load(f)

def plot_ablation_study(results):
    """Plot ablation study results"""
    if 'ablation' not in results:
        return

    data = results['ablation']
    configs = list(data.keys())
    means = [data[c]['final_mean'] for c in configs]
    stds = [data[c]['final_std'] for c in configs]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                   color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])

    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Component Contributions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])

    # Add value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 1, f'{m:.1f}%±{s:.1f}%',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('ieee_ablation_study.png', dpi=300, bbox_inches='tight')
    plt.savefig('ieee_ablation_study.pdf', bbox_inches='tight')
    print("✅ Saved: ieee_ablation_study.png/pdf")
    plt.close()

def plot_byzantine_resilience(results):
    """Plot Byzantine resilience across attack intensities"""
    if 'byzantine' not in results:
        return

    data = results['byzantine']
    fractions = sorted([float(k) for k in data.keys()])
    methods = list(data[fractions[0]].keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in methods:
        means = [data[f][method]['final_mean'] for f in fractions]
        stds = [data[f][method]['final_std'] for f in fractions]

        ax.plot([int(f*100) for f in fractions], means, marker='o',
                label=method, linewidth=2, markersize=8)
        ax.fill_between([int(f*100) for f in fractions],
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2)

    ax.set_xlabel('Byzantine Fraction (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Byzantine Resilience Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig('ieee_byzantine_resilience.png', dpi=300, bbox_inches='tight')
    plt.savefig('ieee_byzantine_resilience.pdf', bbox_inches='tight')
    print("✅ Saved: ieee_byzantine_resilience.png/pdf")
    plt.close()

def plot_noniid_sensitivity(results):
    """Plot Non-IID sensitivity"""
    if 'noniid' not in results:
        return

    data = results['noniid']
    alphas = sorted([float(k) for k in data.keys()])
    methods = list(data[alphas[0]].keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(alphas))
    width = 0.35

    for i, method in enumerate(methods):
        means = [data[a][method]['final_mean'] for a in alphas]
        stds = [data[a][method]['final_std'] for a in alphas]

        ax.bar(x + i*width, means, width, yerr=stds,
               label=method, capsize=5, alpha=0.8)

    ax.set_xlabel('Dirichlet α (Data Heterogeneity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Non-IID Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([f'{a:.1f}' for a in alphas])
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig('ieee_noniid_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.savefig('ieee_noniid_sensitivity.pdf', bbox_inches='tight')
    print("✅ Saved: ieee_noniid_sensitivity.png/pdf")
    plt.close()

def plot_stress_test_heatmap(results):
    """Plot combined stress test as heatmap"""
    if 'stress_test' not in results:
        return

    data = results['stress_test']
    scenarios = list(data.keys())
    methods = list(data[scenarios[0]].keys())

    # Create matrix
    matrix = np.zeros((len(methods), len(scenarios)))
    for i, method in enumerate(methods):
        for j, scenario in enumerate(scenarios):
            matrix[i, j] = data[scenario][method]['final_mean']

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # Set ticks
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_yticklabels(methods, fontsize=10)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")

    # Add values
    for i in range(len(methods)):
        for j in range(len(scenarios)):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=9)

    ax.set_title('Combined Stress Test: Accuracy Heatmap', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Test Accuracy (%)', fontsize=11)

    plt.tight_layout()
    plt.savefig('ieee_stress_test_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig('ieee_stress_test_heatmap.pdf', bbox_inches='tight')
    print("✅ Saved: ieee_stress_test_heatmap.png/pdf")
    plt.close()

def plot_stress_test_comparison(results):
    """Bar plot comparison for stress test"""
    if 'stress_test' not in results:
        return

    data = results['stress_test']
    scenarios = list(data.keys())
    methods = list(data[scenarios[0]].keys())

    fig, axes = plt.subplots(1, len(scenarios), figsize=(15, 5))

    if len(scenarios) == 1:
        axes = [axes]

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        means = [data[scenario][m]['final_mean'] for m in methods]
        stds = [data[scenario][m]['final_std'] for m in methods]

        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)

        # Color B-FedPLC differently
        if 'B-FedPLC' in methods:
            bfedplc_idx = methods.index('B-FedPLC')
            bars[bfedplc_idx].set_color('#2E86AB')
            bars[bfedplc_idx].set_alpha(1.0)

        ax.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(scenario, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 100])

        # Add value labels
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 1, f'{m:.1f}%',
                    ha='center', va='bottom', fontsize=8)

    plt.suptitle('Combined Stress Test: Method Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('ieee_stress_test_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('ieee_stress_test_comparison.pdf', bbox_inches='tight')
    print("✅ Saved: ieee_stress_test_comparison.png/pdf")
    plt.close()

def generate_summary_table(results):
    """Generate LaTeX table for paper"""
    print("\n" + "="*70)
    print("LATEX TABLE FOR PAPER")
    print("="*70)

    # Stress test table (most important)
    if 'stress_test' in results:
        print("\n% Combined Stress Test Results")
        print("\\begin{table}[htbp]")
        print("\\centering")
        print("\\caption{Combined Stress Test: Accuracy under Non-IID + Byzantine Attack}")
        print("\\label{tab:stress_test}")
        print("\\begin{tabular}{lccc}")
        print("\\hline")
        print("Method & Moderate & High & Extreme \\\\")
        print("& (α=0.3, 10\\%) & (α=0.3, 20\\%) & (α=0.1, 20\\%) \\\\")
        print("\\hline")

        data = results['stress_test']
        scenarios = list(data.keys())
        methods = list(data[scenarios[0]].keys())

        for method in methods:
            row = [method]
            for scenario in scenarios:
                mean = data[scenario][method]['final_mean']
                std = data[scenario][method]['final_std']
                row.append(f"${mean:.2f} \\pm {std:.2f}$")
            print(" & ".join(row) + " \\\\")

        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")

    print("\n" + "="*70)

def main():
    """Generate all visualizations"""
    print("="*70)
    print("IEEE ACCESS - VISUALIZATION GENERATOR")
    print("="*70)

    # Check if results file exists
    if not Path('ieee_comprehensive_results.json').exists():
        print("\n❌ Error: ieee_comprehensive_results.json not found!")
        print("   Run ieee_comprehensive_experiment.py first.")
        return

    print("\n📊 Loading results...")
    results = load_results()

    print("\n🎨 Generating visualizations...")

    # Generate all plots
    plot_ablation_study(results)
    plot_byzantine_resilience(results)
    plot_noniid_sensitivity(results)
    plot_stress_test_heatmap(results)
    plot_stress_test_comparison(results)

    # Generate LaTeX table
    generate_summary_table(results)

    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS GENERATED!")
    print("="*70)
    print("\nFiles created:")
    print("  - ieee_ablation_study.png/pdf")
    print("  - ieee_byzantine_resilience.png/pdf")
    print("  - ieee_noniid_sensitivity.png/pdf")
    print("  - ieee_stress_test_heatmap.png/pdf")
    print("  - ieee_stress_test_comparison.png/pdf")
    print("\nReady for IEEE Access submission!")

if __name__ == "__main__":
    main()
