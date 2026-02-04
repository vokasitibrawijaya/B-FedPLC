"""
FedPLC Experiment Visualization Script
Generates publication-quality plots for dissertation

Author: FedPLC Replication Study
Date: January 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

# Color palette
COLORS = {
    'fedplc': '#2E86AB',      # Blue
    'warmup': '#A23B72',      # Magenta
    'ldca': '#F18F01',        # Orange
    'best': '#C73E1D',        # Red
    'community': '#3A7D44',   # Green
    'loss': '#6B4E71',        # Purple
    'parl': '#E94F37',        # Coral
}


def load_results(filepath='fedplc_full_results.json'):
    """Load experiment results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_accuracy_curve(results, save_path='plots/accuracy_curve.png'):
    """Plot accuracy curve with warmup and LDCA phases highlighted"""
    
    accuracy = results['history']['accuracy']
    rounds = list(range(1, len(accuracy) + 1))
    warmup_end = results['config']['warmup_rounds']
    best_acc = results['best_accuracy']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create phase backgrounds
    ax.axvspan(1, warmup_end, alpha=0.15, color=COLORS['warmup'], 
               label='Warmup Phase')
    ax.axvspan(warmup_end, len(accuracy), alpha=0.15, color=COLORS['ldca'], 
               label='PARL+LDCA Phase')
    
    # Plot accuracy
    ax.plot(rounds, accuracy, color=COLORS['fedplc'], linewidth=2.5, 
            label='FedPLC Accuracy', zorder=5)
    
    # Smoothed trend line
    window = 10
    smoothed = np.convolve(accuracy, np.ones(window)/window, mode='valid')
    smoothed_rounds = rounds[window//2:len(rounds)-window//2+1]
    ax.plot(smoothed_rounds, smoothed, color=COLORS['fedplc'], 
            linewidth=3, linestyle='--', alpha=0.7, label='Smoothed Trend')
    
    # Mark best accuracy
    best_round = accuracy.index(best_acc) + 1
    ax.scatter([best_round], [best_acc], color=COLORS['best'], s=150, 
               zorder=10, marker='*', edgecolors='white', linewidth=2)
    ax.annotate(f'Best: {best_acc:.2f}%\n(Round {best_round})', 
                xy=(best_round, best_acc), xytext=(best_round+15, best_acc-3),
                fontsize=12, fontweight='bold', color=COLORS['best'],
                arrowprops=dict(arrowstyle='->', color=COLORS['best'], lw=1.5))
    
    # Mark LDCA transition
    ax.axvline(x=warmup_end+1, color=COLORS['community'], linestyle=':', 
               linewidth=2, alpha=0.8)
    ax.annotate('LDCA\nActivated', xy=(warmup_end+1, 30), fontsize=10,
                color=COLORS['community'], ha='center', fontweight='bold')
    
    # Labels and formatting
    ax.set_xlabel('Communication Round', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('FedPLC Training Progress: Accuracy over Communication Rounds',
                fontsize=16, fontweight='bold', pad=15)
    
    ax.set_xlim(1, len(accuracy))
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Add config info
    config_text = (f"Config: {results['config']['num_clients']} clients, "
                  f"α={results['config']['alpha']}, "
                  f"PARL λ={results['config']['parl_weight']}")
    ax.text(0.02, 0.98, config_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    Path('plots').mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_loss_curves(results, save_path='plots/loss_curves.png'):
    """Plot training loss and PARL loss curves"""
    
    loss = results['history']['loss']
    parl_loss = results['history']['parl_loss']
    rounds = list(range(1, len(loss) + 1))
    warmup_end = results['config']['warmup_rounds']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Training Loss
    ax1.axvspan(1, warmup_end, alpha=0.15, color=COLORS['warmup'])
    ax1.axvspan(warmup_end, len(loss), alpha=0.15, color=COLORS['ldca'])
    ax1.plot(rounds, loss, color=COLORS['loss'], linewidth=2, label='Cross-Entropy Loss')
    ax1.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_title('Training Loss over Communication Rounds', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(loss) * 1.1)
    
    # PARL Loss
    ax2.axvspan(1, warmup_end, alpha=0.15, color=COLORS['warmup'], label='Warmup (No PARL)')
    ax2.axvspan(warmup_end, len(parl_loss), alpha=0.15, color=COLORS['ldca'], label='PARL+LDCA Active')
    ax2.plot(rounds, parl_loss, color=COLORS['parl'], linewidth=2, label='PARL Loss')
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Communication Round', fontsize=14, fontweight='bold')
    ax2.set_ylabel('PARL Loss', fontsize=14, fontweight='bold')
    ax2.set_title('Prototype-Anchored Representation Learning (PARL) Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path('plots').mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_community_evolution(results, save_path='plots/community_evolution.png'):
    """Plot community count evolution over rounds"""
    
    communities = results['history']['communities']
    rounds = list(range(1, len(communities) + 1))
    warmup_end = results['config']['warmup_rounds']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Background phases
    ax.axvspan(1, warmup_end, alpha=0.2, color=COLORS['warmup'])
    ax.axvspan(warmup_end, len(communities), alpha=0.2, color=COLORS['ldca'])
    
    # Plot communities
    ax.fill_between(rounds, communities, alpha=0.4, color=COLORS['community'])
    ax.plot(rounds, communities, color=COLORS['community'], linewidth=2.5, 
            marker='o', markersize=3, label='Number of Communities')
    
    # Annotate key transitions
    ldca_start = warmup_end
    ldca_communities = communities[ldca_start] if ldca_start < len(communities) else communities[-1]
    ax.annotate(f'LDCA: {ldca_communities} communities', 
                xy=(ldca_start+1, ldca_communities), 
                xytext=(ldca_start+20, ldca_communities+5),
                fontsize=11, fontweight='bold', color=COLORS['community'],
                arrowprops=dict(arrowstyle='->', color=COLORS['community']))
    
    # Labels
    ax.set_xlabel('Communication Round', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Communities', fontsize=14, fontweight='bold')
    ax.set_title('LDCA Community Evolution: Dynamic Clustering Adaptation',
                fontsize=16, fontweight='bold', pad=15)
    
    ax.set_xlim(1, len(communities))
    ax.set_ylim(0, max(communities) + 10)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add phase labels
    ax.text(warmup_end/2, max(communities)+5, 'Warmup\n(Single Community)', 
            ha='center', fontsize=11, color=COLORS['warmup'], fontweight='bold')
    ax.text((warmup_end + len(communities))/2, max(communities)+5, 
            'LDCA Active\n(Dynamic Communities)', 
            ha='center', fontsize=11, color=COLORS['ldca'], fontweight='bold')
    
    Path('plots').mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_accuracy_vs_communities(results, save_path='plots/accuracy_vs_communities.png'):
    """Plot accuracy against community count to show correlation"""
    
    accuracy = results['history']['accuracy']
    communities = results['history']['communities']
    warmup_end = results['config']['warmup_rounds']
    
    # Only use data after LDCA activation
    acc_ldca = accuracy[warmup_end:]
    comm_ldca = communities[warmup_end:]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by round number
    rounds_ldca = list(range(warmup_end+1, len(accuracy)+1))
    scatter = ax.scatter(comm_ldca, acc_ldca, c=rounds_ldca, cmap='viridis', 
                        s=80, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax, label='Communication Round')
    
    ax.set_xlabel('Number of Communities', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy vs Community Count (LDCA Phase)',
                fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    
    # Add trend info
    final_comm = communities[-1]
    final_acc = accuracy[-1]
    ax.annotate(f'Final: {final_acc:.1f}%\n({final_comm} communities)', 
                xy=(final_comm, final_acc), xytext=(final_comm-5, final_acc-8),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'))
    
    Path('plots').mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_phase_comparison(results, save_path='plots/phase_comparison.png'):
    """Compare accuracy distribution in warmup vs LDCA phases"""
    
    accuracy = results['history']['accuracy']
    warmup_end = results['config']['warmup_rounds']
    
    warmup_acc = accuracy[:warmup_end]
    ldca_acc = accuracy[warmup_end:]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot comparison
    bp = ax1.boxplot([warmup_acc, ldca_acc], patch_artist=True,
                     labels=['Warmup Phase', 'PARL+LDCA Phase'])
    bp['boxes'][0].set_facecolor(COLORS['warmup'])
    bp['boxes'][1].set_facecolor(COLORS['ldca'])
    for box in bp['boxes']:
        box.set_alpha(0.7)
    
    ax1.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Accuracy Distribution by Training Phase', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Statistics annotation
    stats_warmup = f"Warmup:\nMean: {np.mean(warmup_acc):.1f}%\nMax: {np.max(warmup_acc):.1f}%"
    stats_ldca = f"PARL+LDCA:\nMean: {np.mean(ldca_acc):.1f}%\nMax: {np.max(ldca_acc):.1f}%"
    ax1.text(1, np.min(warmup_acc)-5, stats_warmup, ha='center', fontsize=10)
    ax1.text(2, np.min(ldca_acc)-5, stats_ldca, ha='center', fontsize=10)
    
    # Histogram comparison
    ax2.hist(warmup_acc, bins=15, alpha=0.6, color=COLORS['warmup'], 
             label=f'Warmup (n={len(warmup_acc)})', edgecolor='white')
    ax2.hist(ldca_acc, bins=20, alpha=0.6, color=COLORS['ldca'], 
             label=f'PARL+LDCA (n={len(ldca_acc)})', edgecolor='white')
    
    ax2.axvline(np.mean(warmup_acc), color=COLORS['warmup'], linestyle='--', 
                linewidth=2, label=f'Warmup Mean: {np.mean(warmup_acc):.1f}%')
    ax2.axvline(np.mean(ldca_acc), color=COLORS['ldca'], linestyle='--', 
                linewidth=2, label=f'LDCA Mean: {np.mean(ldca_acc):.1f}%')
    
    ax2.set_xlabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax2.set_title('Accuracy Histogram by Phase', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path('plots').mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_summary_dashboard(results, save_path='plots/fedplc_dashboard.png'):
    """Create a comprehensive dashboard with all key metrics"""
    
    accuracy = results['history']['accuracy']
    loss = results['history']['loss']
    parl_loss = results['history']['parl_loss']
    communities = results['history']['communities']
    rounds = list(range(1, len(accuracy) + 1))
    warmup_end = results['config']['warmup_rounds']
    
    fig = plt.figure(figsize=(16, 12))
    
    # Grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main accuracy plot (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.axvspan(1, warmup_end, alpha=0.15, color=COLORS['warmup'])
    ax1.axvspan(warmup_end, len(accuracy), alpha=0.15, color=COLORS['ldca'])
    ax1.plot(rounds, accuracy, color=COLORS['fedplc'], linewidth=2)
    ax1.set_title('Accuracy Progression', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, alpha=0.3)
    
    # Summary metrics box
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    metrics_text = f"""
    FedPLC Results Summary
    ─────────────────────
    
    Best Accuracy: {results['best_accuracy']:.2f}%
    Final Accuracy: {results['final_accuracy']:.2f}%
    
    Configuration:
    • Clients: {results['config']['num_clients']}
    • Rounds: {results['config']['num_rounds']}
    • Non-IID α: {results['config']['alpha']}
    • PARL λ: {results['config']['parl_weight']}
    • Warmup: {results['config']['warmup_rounds']} rounds
    • Communities: {communities[-1]}
    """
    ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Loss plot
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(rounds, loss, color=COLORS['loss'], linewidth=1.5)
    ax3.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Loss')
    ax3.grid(True, alpha=0.3)
    
    # PARL loss plot
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(rounds, parl_loss, color=COLORS['parl'], linewidth=1.5)
    ax4.axvline(warmup_end, color='gray', linestyle=':', alpha=0.7)
    ax4.set_title('PARL Loss', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('PARL Loss')
    ax4.grid(True, alpha=0.3)
    
    # Community evolution
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.fill_between(rounds, communities, alpha=0.4, color=COLORS['community'])
    ax5.plot(rounds, communities, color=COLORS['community'], linewidth=1.5)
    ax5.set_title('Communities', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Round')
    ax5.set_ylabel('Count')
    ax5.grid(True, alpha=0.3)
    
    # Accuracy improvement over phases
    ax6 = fig.add_subplot(gs[2, :2])
    warmup_acc = accuracy[:warmup_end]
    ldca_acc = accuracy[warmup_end:]
    
    phases = ['Initial', 'Warmup End', 'LDCA Start', 'Round 100', 'Round 150', 'Final', 'Best']
    values = [
        accuracy[0], 
        max(warmup_acc), 
        accuracy[warmup_end] if warmup_end < len(accuracy) else accuracy[-1],
        accuracy[99] if len(accuracy) > 99 else accuracy[-1],
        accuracy[149] if len(accuracy) > 149 else accuracy[-1],
        accuracy[-1],
        results['best_accuracy']
    ]
    
    bars = ax6.bar(phases, values, color=[COLORS['warmup'], COLORS['warmup'], 
                   COLORS['ldca'], COLORS['ldca'], COLORS['ldca'], 
                   COLORS['fedplc'], COLORS['best']], alpha=0.8, edgecolor='white')
    ax6.set_title('Key Milestones', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Accuracy (%)')
    ax6.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    ax6.tick_params(axis='x', rotation=30)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Improvement metrics
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    improvement_warmup = max(warmup_acc) - accuracy[0]
    improvement_ldca = max(ldca_acc) - max(warmup_acc)
    improvement_total = results['best_accuracy'] - accuracy[0]
    
    imp_text = f"""
    Accuracy Improvements
    ─────────────────────
    
    Warmup Phase:
    +{improvement_warmup:.1f}% ({accuracy[0]:.1f}% → {max(warmup_acc):.1f}%)
    
    PARL+LDCA Phase:
    +{improvement_ldca:.1f}% ({max(warmup_acc):.1f}% → {max(ldca_acc):.1f}%)
    
    Total Improvement:
    +{improvement_total:.1f}%
    """
    ax7.text(0.1, 0.9, imp_text, transform=ax7.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    fig.suptitle('FedPLC Experiment Dashboard - Paper Replication Results', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    Path('plots').mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def generate_all_visualizations():
    """Generate all visualization plots"""
    
    print("="*60)
    print("FedPLC Visualization Generator")
    print("="*60)
    
    # Load results
    print("\nLoading results...")
    results = load_results('fedplc_full_results.json')
    print(f"✓ Loaded results: {results['config']['num_rounds']} rounds, "
          f"{results['config']['num_clients']} clients")
    print(f"✓ Best accuracy: {results['best_accuracy']:.2f}%")
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plot_accuracy_curve(results)
    plot_loss_curves(results)
    plot_community_evolution(results)
    plot_accuracy_vs_communities(results)
    plot_phase_comparison(results)
    plot_summary_dashboard(results)
    
    print("\n" + "="*60)
    print("All visualizations saved to 'plots/' directory")
    print("="*60)
    
    # List generated files
    print("\nGenerated files:")
    for f in Path('plots').glob('*.png'):
        print(f"  📊 {f}")


if __name__ == "__main__":
    generate_all_visualizations()
