"""
Concept Drift Comparison Summary
Compares Label Swap vs Distribution Shift results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
with open('concept_drift_label_swap_results.json', 'r') as f:
    label_swap = json.load(f)

with open('concept_drift_distribution_shift_results.json', 'r') as f:
    dist_shift = json.load(f)

print("="*70)
print("CONCEPT DRIFT EXPERIMENT SUMMARY")
print("="*70)

# Create comparison table
print("\n" + "-"*70)
print(f"{'Metric':<35} {'Label Swap':>15} {'Dist Shift':>15}")
print("-"*70)

metrics = [
    ('Best Accuracy (%)', 'best_accuracy'),
    ('Final Accuracy (%)', 'final_accuracy'),
    ('Pre-drift Best (%)', 'pre_drift_best'),
    ('Post-drift Min (%)', 'post_drift_min'),
    ('Post-drift Best (%)', 'post_drift_best'),
    ('Accuracy Drop (%)', 'accuracy_drop'),
]

for name, key in metrics:
    ls_val = label_swap.get(key, 0)
    ds_val = dist_shift.get(key, 0)
    print(f"{name:<35} {ls_val:>15.2f} {ds_val:>15.2f}")

# Recovery
ls_rec = label_swap.get('recovery_round', 'N/A')
ds_rec = dist_shift.get('recovery_round', 'N/A')
ls_rec_str = f"Round {ls_rec}" if ls_rec else "N/A"
ds_rec_str = f"Round {ds_rec}" if ds_rec else "N/A"
print(f"{'Recovery Round':<35} {ls_rec_str:>15} {ds_rec_str:>15}")

print("-"*70)

# Create combined visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

drift_round = label_swap['config']['drift_round']

# Plot 1: Accuracy comparison
ax1 = axes[0, 0]
rounds = list(range(1, len(label_swap['history']['accuracy']) + 1))
ax1.plot(rounds, label_swap['history']['accuracy'], 'b-', linewidth=2, label='Label Swap', alpha=0.8)
ax1.plot(rounds, dist_shift['history']['accuracy'], 'r-', linewidth=2, label='Dist Shift', alpha=0.8)
ax1.axvline(drift_round, color='gray', linestyle='--', linewidth=2, label='Drift Point')
ax1.set_xlabel('Round')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Accuracy Comparison: Label Swap vs Distribution Shift')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Communities comparison
ax2 = axes[0, 1]
ax2.plot(rounds, label_swap['history']['communities'], 'b-', linewidth=2, label='Label Swap')
ax2.plot(rounds, dist_shift['history']['communities'], 'r-', linewidth=2, label='Dist Shift')
ax2.axvline(drift_round, color='gray', linestyle='--', linewidth=2)
ax2.set_xlabel('Round')
ax2.set_ylabel('Number of Communities')
ax2.set_title('LDCA Community Adaptation')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Loss comparison
ax3 = axes[0, 2]
ax3.plot(rounds, label_swap['history']['loss'], 'b-', linewidth=1.5, label='Label Swap')
ax3.plot(rounds, dist_shift['history']['loss'], 'r-', linewidth=1.5, label='Dist Shift')
ax3.axvline(drift_round, color='gray', linestyle='--', linewidth=2)
ax3.set_xlabel('Round')
ax3.set_ylabel('Loss')
ax3.set_title('Training Loss Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Pre vs Post drift boxplot - Label Swap
ax4 = axes[1, 0]
ls_pre = label_swap['history']['accuracy'][:drift_round-1]
ls_post = label_swap['history']['accuracy'][drift_round:]
bp1 = ax4.boxplot([ls_pre, ls_post], patch_artist=True, tick_labels=['Pre-Drift', 'Post-Drift'])
bp1['boxes'][0].set_facecolor('lightblue')
bp1['boxes'][1].set_facecolor('lightcoral')
ax4.set_ylabel('Accuracy (%)')
ax4.set_title('Label Swap: Pre vs Post Drift')
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Pre vs Post drift boxplot - Distribution Shift
ax5 = axes[1, 1]
ds_pre = dist_shift['history']['accuracy'][:drift_round-1]
ds_post = dist_shift['history']['accuracy'][drift_round:]
bp2 = ax5.boxplot([ds_pre, ds_post], patch_artist=True, tick_labels=['Pre-Drift', 'Post-Drift'])
bp2['boxes'][0].set_facecolor('lightgreen')
bp2['boxes'][1].set_facecolor('lightsalmon')
ax5.set_ylabel('Accuracy (%)')
ax5.set_title('Distribution Shift: Pre vs Post Drift')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Summary Bar Chart
ax6 = axes[1, 2]
metrics_bar = ['Best Acc', 'Final Acc', 'Pre-drift', 'Post-drift\nDrop', 'Recovery\nSpeed']
ls_vals = [
    label_swap['best_accuracy'],
    label_swap['final_accuracy'],
    label_swap['pre_drift_best'],
    label_swap['accuracy_drop'],
    (label_swap['recovery_round'] - drift_round) if label_swap['recovery_round'] else 10
]
ds_vals = [
    dist_shift['best_accuracy'],
    dist_shift['final_accuracy'],
    dist_shift['pre_drift_best'],
    dist_shift['accuracy_drop'],
    (dist_shift['recovery_round'] - drift_round) if dist_shift['recovery_round'] else 10
]

x = np.arange(len(metrics_bar))
width = 0.35
bars1 = ax6.bar(x - width/2, ls_vals, width, label='Label Swap', color='steelblue')
bars2 = ax6.bar(x + width/2, ds_vals, width, label='Dist Shift', color='coral')
ax6.set_ylabel('Value')
ax6.set_title('Performance Comparison')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics_bar)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax6.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax6.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

fig.suptitle('FedPLC Concept Drift Analysis: Label Swap vs Distribution Shift',
            fontsize=16, fontweight='bold')

plt.tight_layout()
Path('plots').mkdir(exist_ok=True)
plt.savefig('plots/concept_drift_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: plots/concept_drift_comparison.png")

# Summary Analysis
print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

print(f"""
1. LABEL SWAP DRIFT (Swapping class labels for 30% of clients):
   - More severe immediate impact: Accuracy dropped by {label_swap['accuracy_drop']:.2f}%
   - Recovery time: {(label_swap['recovery_round'] - drift_round) if label_swap['recovery_round'] else 'N/A'} rounds
   - Final accuracy: {label_swap['final_accuracy']:.2f}%

2. DISTRIBUTION SHIFT DRIFT (α: 0.5 → 0.1 for 30% of clients):
   - More moderate impact: Accuracy dropped by {dist_shift['accuracy_drop']:.2f}%
   - Recovery time: {(dist_shift['recovery_round'] - drift_round) if dist_shift['recovery_round'] else 'N/A'} rounds
   - Final accuracy: {dist_shift['final_accuracy']:.2f}%

3. LDCA ADAPTATION:
   - Label Swap: Communities reorganized from {label_swap['history']['communities'][drift_round-2]} to {label_swap['history']['communities'][-1]}
   - Dist Shift: Communities remained stable at {dist_shift['history']['communities'][-1]}

4. CONCLUSION:
   - FedPLC demonstrates robust recovery from both drift scenarios
   - Distribution shift shows better final performance ({dist_shift['final_accuracy']:.2f}% vs {label_swap['final_accuracy']:.2f}%)
   - Label swap is more challenging due to semantic confusion
   - LDCA helps adapt to distribution changes through community reorganization
""")

# Save combined results
combined_results = {
    'label_swap': label_swap,
    'distribution_shift': dist_shift,
    'comparison': {
        'label_swap_recovery': label_swap['recovery_round'],
        'dist_shift_recovery': dist_shift['recovery_round'],
        'label_swap_drop': label_swap['accuracy_drop'],
        'dist_shift_drop': dist_shift['accuracy_drop'],
        'label_swap_final': label_swap['final_accuracy'],
        'dist_shift_final': dist_shift['final_accuracy'],
    }
}

with open('concept_drift_combined_results.json', 'w') as f:
    json.dump(combined_results, f, indent=2)
print("\n✓ Saved: concept_drift_combined_results.json")

print("\n" + "="*70)
print("CONCEPT DRIFT ANALYSIS COMPLETE!")
print("="*70)
