"""
Visualization for Comparative Analysis
Generates publication-quality comparison plots
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results():
    """Load comparative analysis results"""
    with open('comparative_analysis_results.json', 'r') as f:
        return json.load(f)


def plot_accuracy_comparison(data, save_dir='plots'):
    """Plot accuracy curves for all algorithms"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    markers = ['o', 's', '^', 'D']
    
    for i, result in enumerate(data['results']):
        rounds = result['history']['rounds']
        accuracy = result['history']['accuracy']
        
        # Plot with markers at intervals
        ax.plot(rounds, accuracy, 
               color=colors[i % len(colors)],
               linewidth=2,
               label=f"{result['name']} (Best: {result['best_accuracy']:.2f}%)",
               marker=markers[i % len(markers)],
               markevery=10,
               markersize=6)
    
    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Comparative Analysis: FL Algorithms on CIFAR-10 (Non-IID)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(rounds))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/comparison_accuracy.png")


def plot_convergence_comparison(data, save_dir='plots'):
    """Plot convergence speed comparison"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Find rounds to reach certain accuracy thresholds
    thresholds = [50, 60, 65, 70]
    
    algo_names = [r['name'] for r in data['results']]
    x = np.arange(len(thresholds))
    width = 0.2
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    for i, result in enumerate(data['results']):
        accuracy = result['history']['accuracy']
        rounds_to_threshold = []
        
        for thresh in thresholds:
            found = False
            for r, acc in enumerate(accuracy):
                if acc >= thresh:
                    rounds_to_threshold.append(r + 1)
                    found = True
                    break
            if not found:
                rounds_to_threshold.append(len(accuracy))  # Never reached
        
        offset = (i - len(data['results'])/2 + 0.5) * width
        bars = ax.bar(x + offset, rounds_to_threshold, width, 
                     label=result['name'], color=colors[i % len(colors)], alpha=0.8)
    
    ax.set_xlabel('Accuracy Threshold (%)', fontsize=12)
    ax.set_ylabel('Rounds to Reach Threshold', fontsize=12)
    ax.set_title('Convergence Speed Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}%' for t in thresholds])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/comparison_convergence.png")


def plot_final_comparison(data, save_dir='plots'):
    """Plot final accuracy and training time comparison"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    algo_names = [r['name'] for r in data['results']]
    best_acc = [r['best_accuracy'] for r in data['results']]
    final_acc = [r['final_accuracy'] for r in data['results']]
    train_time = [r['training_time_minutes'] for r in data['results']]
    
    x = np.arange(len(algo_names))
    width = 0.35
    
    # Best accuracy comparison
    bars1 = axes[0].bar(x - width/2, best_acc, width, label='Best Accuracy', 
                       color=colors, alpha=0.8)
    bars2 = axes[0].bar(x + width/2, final_acc, width, label='Final Accuracy',
                       color=colors, alpha=0.5)
    
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Accuracy Comparison', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(algo_names, rotation=15, ha='right')
    axes[0].legend()
    axes[0].set_ylim(60, 80)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Training time comparison
    bars3 = axes[1].bar(x, train_time, width*2, color=colors, alpha=0.8)
    axes[1].set_ylabel('Training Time (minutes)', fontsize=12)
    axes[1].set_title('Training Time Comparison', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(algo_names, rotation=15, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        axes[1].annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/comparison_final.png")


def plot_comprehensive_comparison(data, save_dir='plots'):
    """Create comprehensive comparison figure"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    markers = ['o', 's', '^', 'D']
    
    # 2x2 layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy curves (top-left, spans full width)
    ax1 = fig.add_subplot(gs[0, :])
    for i, result in enumerate(data['results']):
        rounds = result['history']['rounds']
        accuracy = result['history']['accuracy']
        ax1.plot(rounds, accuracy, 
                color=colors[i], linewidth=2,
                label=f"{result['name']}",
                marker=markers[i], markevery=10, markersize=5)
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Learning Curves Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Best accuracy bar chart (bottom-left)
    ax2 = fig.add_subplot(gs[1, 0])
    algo_names = [r['name'] for r in data['results']]
    best_acc = [r['best_accuracy'] for r in data['results']]
    bars = ax2.bar(range(len(algo_names)), best_acc, color=colors, alpha=0.8)
    ax2.set_ylabel('Best Accuracy (%)')
    ax2.set_title('Best Accuracy Achieved', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(algo_names)))
    ax2.set_xticklabels(algo_names, rotation=15, ha='right')
    ax2.set_ylim(60, 80)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add labels
    for bar, acc in zip(bars, best_acc):
        ax2.annotate(f'{acc:.2f}%',
                    xy=(bar.get_x() + bar.get_width()/2, acc),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Training efficiency (bottom-right)
    ax3 = fig.add_subplot(gs[1, 1])
    train_time = [r['training_time_minutes'] for r in data['results']]
    efficiency = [acc/time for acc, time in zip(best_acc, train_time)]
    bars = ax3.bar(range(len(algo_names)), efficiency, color=colors, alpha=0.8)
    ax3.set_ylabel('Accuracy per Minute')
    ax3.set_title('Training Efficiency', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(algo_names)))
    ax3.set_xticklabels(algo_names, rotation=15, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Main title
    fig.suptitle('Comparative Analysis: B-FedPLC vs Baseline FL Algorithms\n'
                'CIFAR-10, Non-IID (Dirichlet alpha=0.5), 50 Clients',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig(f'{save_dir}/comparison_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/comparison_comprehensive.png")


def create_latex_table(data, save_dir='plots'):
    """Generate LaTeX table for dissertation"""
    Path(save_dir).mkdir(exist_ok=True)
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Comparative Analysis Results on CIFAR-10 with Non-IID Data Distribution}
\label{tab:comparison}
\begin{tabular}{lcccc}
\toprule
\textbf{Algorithm} & \textbf{Best Acc. (\%)} & \textbf{Final Acc. (\%)} & \textbf{Time (min)} & \textbf{Efficiency} \\
\midrule
"""
    
    for r in data['results']:
        efficiency = r['best_accuracy'] / r['training_time_minutes']
        latex += f"{r['name']} & {r['best_accuracy']:.2f} & {r['final_accuracy']:.2f} & "
        latex += f"{r['training_time_minutes']:.1f} & {efficiency:.2f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(f'{save_dir}/comparison_table.tex', 'w') as f:
        f.write(latex)
    
    print(f"Saved: {save_dir}/comparison_table.tex")


def generate_all_visualizations():
    """Generate all comparison visualizations"""
    print("="*60)
    print("Comparison Visualization Generator")
    print("="*60 + "\n")
    
    try:
        data = load_results()
        print("Loaded comparative analysis results")
    except FileNotFoundError:
        print("ERROR: comparative_analysis_results.json not found!")
        print("Please run run_comparative_analysis.py first.")
        return
    
    print(f"\nAlgorithms compared: {len(data['results'])}")
    for r in data['results']:
        print(f"  - {r['name']}: {r['best_accuracy']:.2f}%")
    
    print("\nGenerating visualizations...")
    
    plot_accuracy_comparison(data)
    plot_convergence_comparison(data)
    plot_final_comparison(data)
    plot_comprehensive_comparison(data)
    create_latex_table(data)
    
    print("\n" + "="*60)
    print("All comparison visualizations generated!")
    print("="*60)


if __name__ == "__main__":
    generate_all_visualizations()
