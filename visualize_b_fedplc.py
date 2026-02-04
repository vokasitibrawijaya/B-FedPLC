"""
B-FedPLC Visualization Generator
Creates publication-quality visualizations for blockchain metrics
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime


def load_results():
    """Load B-FedPLC results"""
    with open('b_fedplc_results.json', 'r') as f:
        return json.load(f)


def load_blockchain():
    """Load blockchain data"""
    with open('b_fedplc_blockchain.json', 'r') as f:
        return json.load(f)


def plot_training_metrics(results, save_dir='plots'):
    """Plot training accuracy and loss"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    history = results['history']
    rounds = history['rounds']
    
    # Accuracy plot
    axes[0].plot(rounds, history['accuracy'], 'b-', linewidth=2, label='Test Accuracy')
    axes[0].axhline(y=results['best_accuracy'], color='r', linestyle='--', 
                   label=f'Best: {results["best_accuracy"]:.2f}%')
    axes[0].set_xlabel('Round', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('B-FedPLC: Test Accuracy over Rounds', fontsize=14)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(rounds, history['loss'], 'g-', linewidth=2, label='Training Loss')
    axes[1].set_xlabel('Round', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('B-FedPLC: Training Loss over Rounds', fontsize=14)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/b_fedplc_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/b_fedplc_training_metrics.png")


def plot_community_evolution(results, save_dir='plots'):
    """Plot community dynamics"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    history = results['history']
    rounds = history['rounds']
    communities = history['communities']
    
    ax.fill_between(rounds, communities, alpha=0.3, color='blue')
    ax.plot(rounds, communities, 'b-', linewidth=2, marker='o', markersize=3)
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Number of Communities', fontsize=12)
    ax.set_title('B-FedPLC: LDCA Community Evolution', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add annotation for final communities
    ax.annotate(f'Final: {communities[-1]} communities',
               xy=(rounds[-1], communities[-1]),
               xytext=(-80, 20), textcoords='offset points',
               fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/b_fedplc_communities.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/b_fedplc_communities.png")


def plot_blockchain_metrics(results, save_dir='plots'):
    """Plot blockchain-specific metrics"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    history = results['history']
    rounds = history['rounds']
    ipfs_size = history['ipfs_size']
    
    # IPFS storage growth
    axes[0].fill_between(rounds, ipfs_size, alpha=0.3, color='purple')
    axes[0].plot(rounds, ipfs_size, 'purple', linewidth=2)
    axes[0].set_xlabel('Round', fontsize=12)
    axes[0].set_ylabel('IPFS Storage (MB)', fontsize=12)
    axes[0].set_title('B-FedPLC: Decentralized Storage Growth', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Blockchain growth
    block_nums = list(range(1, len(rounds) + 1))
    axes[1].bar(rounds, block_nums, color='orange', alpha=0.7, width=1)
    axes[1].set_xlabel('Round', fontsize=12)
    axes[1].set_ylabel('Blockchain Height', fontsize=12)
    axes[1].set_title('B-FedPLC: Blockchain Growth', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/b_fedplc_blockchain_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/b_fedplc_blockchain_metrics.png")


def plot_system_overview(results, save_dir='plots'):
    """Create comprehensive system overview"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(16, 12))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    history = results['history']
    rounds = history['rounds']
    
    # 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rounds, history['accuracy'], 'b-', linewidth=2)
    ax1.axhline(y=results['best_accuracy'], color='r', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(f'Test Accuracy (Best: {results["best_accuracy"]:.2f}%)')
    ax1.grid(True, alpha=0.3)
    
    # Communities
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(rounds, history['communities'], alpha=0.3, color='green')
    ax2.plot(rounds, history['communities'], 'g-', linewidth=2)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Communities')
    ax2.set_title('LDCA Community Dynamics')
    ax2.grid(True, alpha=0.3)
    
    # IPFS Storage
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(rounds, history['ipfs_size'], alpha=0.3, color='purple')
    ax3.plot(rounds, history['ipfs_size'], 'purple', linewidth=2)
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Storage (MB)')
    ax3.set_title('IPFS Decentralized Storage')
    ax3.grid(True, alpha=0.3)
    
    # Combined metrics (normalized)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Normalize for comparison
    acc_norm = np.array(history['accuracy']) / 100
    comm_norm = np.array(history['communities']) / max(history['communities'])
    ipfs_norm = np.array(history['ipfs_size']) / max(history['ipfs_size'])
    
    ax4.plot(rounds, acc_norm, 'b-', linewidth=2, label='Accuracy (norm)')
    ax4.plot(rounds, comm_norm, 'g-', linewidth=2, label='Communities (norm)')
    ax4.plot(rounds, ipfs_norm, 'purple', linewidth=2, label='Storage (norm)')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Normalized Value')
    ax4.set_title('Combined System Metrics')
    ax4.legend(loc='center right')
    ax4.grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle('B-FedPLC: Blockchain-Enabled Federated Learning System Overview', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.savefig(f'{save_dir}/b_fedplc_system_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/b_fedplc_system_overview.png")


def plot_comparison_baseline(results, baseline_file='fedplc_full_results.json', save_dir='plots'):
    """Compare B-FedPLC with baseline FedPLC"""
    Path(save_dir).mkdir(exist_ok=True)
    
    # Load baseline if exists
    try:
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
    except FileNotFoundError:
        print(f"Baseline file not found: {baseline_file}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # B-FedPLC data
    bf_rounds = results['history']['rounds']
    bf_acc = results['history']['accuracy']
    
    # Baseline data
    base_rounds = baseline['history']['rounds']
    base_acc = baseline['history']['accuracy']
    
    # Truncate to same length
    min_len = min(len(bf_rounds), len(base_rounds))
    
    # Accuracy comparison
    axes[0].plot(bf_rounds[:min_len], bf_acc[:min_len], 'b-', linewidth=2, 
                label=f'B-FedPLC (Best: {results["best_accuracy"]:.2f}%)')
    axes[0].plot(base_rounds[:min_len], base_acc[:min_len], 'r--', linewidth=2,
                label=f'FedPLC (Best: {baseline["best_accuracy"]:.2f}%)')
    axes[0].set_xlabel('Round', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('B-FedPLC vs FedPLC: Accuracy Comparison', fontsize=14)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Community comparison
    bf_comm = results['history']['communities']
    base_comm = baseline['history']['communities']
    
    axes[1].plot(bf_rounds[:min_len], bf_comm[:min_len], 'b-', linewidth=2,
                label='B-FedPLC Communities')
    axes[1].plot(base_rounds[:min_len], base_comm[:min_len], 'r--', linewidth=2,
                label='FedPLC Communities')
    axes[1].set_xlabel('Round', fontsize=12)
    axes[1].set_ylabel('Number of Communities', fontsize=12)
    axes[1].set_title('Community Evolution Comparison', fontsize=14)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/b_fedplc_vs_fedplc.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/b_fedplc_vs_fedplc.png")


def generate_all_visualizations():
    """Generate all B-FedPLC visualizations"""
    print("="*60)
    print("B-FedPLC Visualization Generator")
    print("="*60 + "\n")
    
    try:
        results = load_results()
        print("Loaded B-FedPLC results")
    except FileNotFoundError:
        print("ERROR: b_fedplc_results.json not found!")
        print("Please run run_b_fedplc.py first.")
        return
    
    print("\nGenerating visualizations...\n")
    
    plot_training_metrics(results)
    plot_community_evolution(results)
    plot_blockchain_metrics(results)
    plot_system_overview(results)
    plot_comparison_baseline(results)
    
    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print("="*60)


if __name__ == "__main__":
    generate_all_visualizations()
