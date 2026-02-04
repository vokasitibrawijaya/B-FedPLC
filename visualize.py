"""
Visualization utilities for FedPLC experiments
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import argparse


def load_results(results_dir: str) -> Dict:
    """Load results from JSON file"""
    results_file = os.path.join(results_dir, 'results.json')
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_accuracy_curve(results: Dict, 
                       save_path: Optional[str] = None,
                       title: str = "FedPLC Training Progress"):
    """Plot accuracy over rounds"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rounds = results['rounds']
    accuracy = results['accuracy']
    
    ax.plot(rounds, accuracy, 'b-', linewidth=2, label='Test Accuracy')
    
    # Mark warmup period
    config = results.get('config', {})
    warmup_rounds = config.get('warmup_rounds', 30)
    ax.axvline(x=warmup_rounds, color='g', linestyle='--', 
               label=f'Warmup End (Round {warmup_rounds})')
    
    # Mark drift round if applicable
    drift_type = config.get('drift_type', 'none')
    if drift_type != 'none':
        drift_round = config.get('drift_round', 100)
        ax.axvline(x=drift_round, color='r', linestyle='--',
                   label=f'Drift Applied (Round {drift_round})')
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_loss_curve(results: Dict,
                    save_path: Optional[str] = None):
    """Plot loss over rounds"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rounds = results['rounds']
    loss = results['loss']
    
    ax.plot(rounds, loss, 'r-', linewidth=2)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_comparison(results_list: List[Dict],
                    labels: List[str],
                    save_path: Optional[str] = None,
                    title: str = "Method Comparison"):
    """Compare multiple experiment results"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    
    for results, label, color in zip(results_list, labels, colors):
        rounds = results['rounds']
        accuracy = results['accuracy']
        ax.plot(rounds, accuracy, linewidth=2, label=label, color=color)
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_ablation_study(results_dict: Dict[str, Dict],
                        param_name: str,
                        save_path: Optional[str] = None):
    """Plot ablation study results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    param_values = list(results_dict.keys())
    final_accuracies = [r['final_accuracy'] for r in results_dict.values()]
    best_accuracies = [r['best_accuracy'] for r in results_dict.values()]
    
    # Bar plot of final accuracies
    ax1 = axes[0]
    x = np.arange(len(param_values))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, final_accuracies, width, label='Final Accuracy')
    bars2 = ax1.bar(x + width/2, best_accuracies, width, label='Best Accuracy')
    
    ax1.set_xlabel(param_name, fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'Ablation Study: {param_name}', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_values)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Learning curves
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_dict)))
    
    for (param_val, results), color in zip(results_dict.items(), colors):
        ax2.plot(results['rounds'], results['accuracy'],
                 label=f'{param_name}={param_val}', color=color, linewidth=1.5)
    
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Learning Curves', fontsize=14)
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_drift_recovery(results_no_drift: Dict,
                        results_with_drift: Dict,
                        drift_round: int = 100,
                        save_path: Optional[str] = None):
    """Visualize concept drift and recovery"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(results_no_drift['rounds'], results_no_drift['accuracy'],
            'b-', linewidth=2, label='No Drift')
    ax.plot(results_with_drift['rounds'], results_with_drift['accuracy'],
            'r-', linewidth=2, label='With Drift')
    
    # Mark drift point
    ax.axvline(x=drift_round, color='gray', linestyle='--', linewidth=2)
    ax.text(drift_round + 2, ax.get_ylim()[1] - 5, 'Drift Applied',
            fontsize=10, color='gray')
    
    # Shade recovery period
    ax.axvspan(drift_round, drift_round + 30, alpha=0.2, color='red',
               label='Recovery Period')
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Concept Drift Impact and Recovery', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def generate_latex_table(results_dict: Dict[str, Dict],
                         caption: str = "Experiment Results",
                         label: str = "tab:results") -> str:
    """Generate LaTeX table from results"""
    
    # Header
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += "\\begin{tabular}{lcc}\n"
    latex += "\\toprule\n"
    latex += "Method & Best Accuracy (\\%) & Final Accuracy (\\%) \\\\\n"
    latex += "\\midrule\n"
    
    # Data rows
    for name, results in results_dict.items():
        best_acc = results.get('best_accuracy', 0)
        final_acc = results.get('final_accuracy', 0)
        latex += f"{name} & {best_acc:.2f} & {final_acc:.2f} \\\\\n"
    
    # Footer
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex


def main():
    parser = argparse.ArgumentParser(description='Visualize FedPLC results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing results.json')
    parser.add_argument('--output_dir', type=str, default='./figures',
                        help='Directory to save figures')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and plot results
    results = load_results(args.results_dir)
    
    dataset = results.get('config', {}).get('dataset', 'unknown')
    
    plot_accuracy_curve(
        results,
        save_path=os.path.join(args.output_dir, f'{dataset}_accuracy.png'),
        title=f'FedPLC on {dataset.upper()}'
    )
    
    plot_loss_curve(
        results,
        save_path=os.path.join(args.output_dir, f'{dataset}_loss.png')
    )
    
    print(f"\nFigures saved to {args.output_dir}")


if __name__ == '__main__':
    main()
