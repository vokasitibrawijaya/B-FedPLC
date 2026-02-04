"""
Generate B-FedPLC Architecture Diagram for IEEE Access Paper
This script creates a professional architecture diagram showing:
1. PARL (Prototype-Anchored Regularization Layer)
2. LDCA (Label Distribution-based Community Adaptation)
3. Blockchain-IPFS Integration
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Arrow, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
colors = {
    'client': '#3498db',
    'server': '#e74c3c',
    'blockchain': '#f39c12',
    'ipfs': '#9b59b6',
    'community': '#2ecc71',
    'parl': '#1abc9c',
    'ldca': '#e67e22',
    'arrow': '#34495e',
    'text': '#2c3e50',
    'bg_light': '#ecf0f1'
}

# Title
ax.text(7, 9.5, 'B-FedPLC: Blockchain-Enabled Federated Learning Architecture',
        fontsize=14, fontweight='bold', ha='center', color=colors['text'])

# ====== LEFT SIDE: CLIENTS ======
client_y_positions = [7, 5.5, 4, 2.5]
for i, y in enumerate(client_y_positions):
    # Client box
    client_box = FancyBboxPatch((0.5, y-0.4), 2.5, 0.8,
                                  boxstyle="round,pad=0.05,rounding_size=0.1",
                                  facecolor=colors['client'], edgecolor='black', linewidth=1.5)
    ax.add_patch(client_box)
    ax.text(1.75, y, f'Client {i+1}', fontsize=10, ha='center', va='center',
            color='white', fontweight='bold')

    # Local data
    ax.text(1.75, y-0.6, f'Local Data $D_{i+1}$', fontsize=8, ha='center',
            va='center', color=colors['text'], style='italic')

# PARL component
parl_box = FancyBboxPatch((0.3, 0.8), 2.9, 0.8,
                           boxstyle="round,pad=0.05,rounding_size=0.1",
                           facecolor=colors['parl'], edgecolor='black', linewidth=1.5)
ax.add_patch(parl_box)
ax.text(1.75, 1.2, 'PARL Module', fontsize=10, ha='center', va='center',
        color='white', fontweight='bold')
ax.text(1.75, 0.5, 'Prototype-Anchored\nRegularization', fontsize=7, ha='center',
        va='center', color=colors['text'])

# ====== MIDDLE: AGGREGATION SERVER ======
# Server box
server_box = FancyBboxPatch((5, 4), 4, 3,
                             boxstyle="round,pad=0.1,rounding_size=0.2",
                             facecolor=colors['server'], edgecolor='black', linewidth=2)
ax.add_patch(server_box)
ax.text(7, 6.5, 'Aggregation Server', fontsize=12, ha='center', va='center',
        color='white', fontweight='bold')

# LDCA inside server
ldca_box = FancyBboxPatch((5.3, 5.2), 3.4, 1,
                           boxstyle="round,pad=0.05",
                           facecolor=colors['ldca'], edgecolor='white', linewidth=1)
ax.add_patch(ldca_box)
ax.text(7, 5.7, 'LDCA', fontsize=10, ha='center', va='center',
        color='white', fontweight='bold')
ax.text(7, 5.3, 'Dynamic Community\nAdaptation', fontsize=7, ha='center',
        va='center', color='white')

# Byzantine Detection inside server
byz_box = FancyBboxPatch((5.3, 4.2), 3.4, 0.8,
                          boxstyle="round,pad=0.05",
                          facecolor='#c0392b', edgecolor='white', linewidth=1)
ax.add_patch(byz_box)
ax.text(7, 4.6, 'Byzantine Detection', fontsize=9, ha='center', va='center',
        color='white', fontweight='bold')

# Communities visualization
for i, (x, y) in enumerate([(5.8, 3.2), (7, 3.2), (8.2, 3.2)]):
    comm_circle = plt.Circle((x, y), 0.25, color=colors['community'], ec='black')
    ax.add_patch(comm_circle)
    ax.text(x, y, f'C{i+1}', fontsize=8, ha='center', va='center', color='white')
ax.text(7, 2.7, 'Communities', fontsize=8, ha='center', va='center', color=colors['text'])

# ====== RIGHT SIDE: BLOCKCHAIN + IPFS ======
# Blockchain
blockchain_box = FancyBboxPatch((10.5, 5), 3, 2.5,
                                 boxstyle="round,pad=0.1,rounding_size=0.2",
                                 facecolor=colors['blockchain'], edgecolor='black', linewidth=2)
ax.add_patch(blockchain_box)
ax.text(12, 7, 'Blockchain', fontsize=12, ha='center', va='center',
        color='white', fontweight='bold')

# Blocks inside blockchain
for i, y in enumerate([6.3, 5.8, 5.3]):
    block = FancyBboxPatch((10.8, y-0.15), 0.8, 0.3,
                            boxstyle="round,pad=0.02",
                            facecolor='#d68910', edgecolor='black', linewidth=0.5)
    ax.add_patch(block)
    ax.text(11.2, y, f'B{i}', fontsize=7, ha='center', va='center', color='black')

# Merkle tree text
ax.text(12.5, 5.8, 'Merkle\nTree', fontsize=8, ha='center', va='center',
        color='white')

# IPFS
ipfs_box = FancyBboxPatch((10.5, 2), 3, 2,
                           boxstyle="round,pad=0.1,rounding_size=0.2",
                           facecolor=colors['ipfs'], edgecolor='black', linewidth=2)
ax.add_patch(ipfs_box)
ax.text(12, 3.5, 'IPFS Storage', fontsize=12, ha='center', va='center',
        color='white', fontweight='bold')
ax.text(12, 2.8, 'Decentralized\nModel Storage', fontsize=8, ha='center',
        va='center', color='white')
ax.text(12, 2.2, 'CID: Qm...', fontsize=7, ha='center', va='center',
        color='#d2b4de', family='monospace')

# ====== ARROWS ======
# Client to Server arrows
for y in client_y_positions:
    ax.annotate('', xy=(5, 5.5), xytext=(3, y),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

# Server to Blockchain
ax.annotate('', xy=(10.5, 6), xytext=(9, 5.5),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
ax.text(9.75, 6.2, 'Audit\nLog', fontsize=7, ha='center', va='center', color=colors['text'])

# Server to IPFS
ax.annotate('', xy=(10.5, 3), xytext=(9, 4.5),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
ax.text(9.75, 3.4, 'Model\nWeights', fontsize=7, ha='center', va='center', color=colors['text'])

# Blockchain to IPFS (reference)
ax.annotate('', xy=(12, 4), xytext=(12, 5),
            arrowprops=dict(arrowstyle='<->', color=colors['arrow'], lw=1.5))
ax.text(12.5, 4.5, 'CID\nRef', fontsize=7, ha='left', va='center', color=colors['text'])

# PARL to Clients
ax.annotate('', xy=(1.75, 2.1), xytext=(1.75, 0.9),
            arrowprops=dict(arrowstyle='<->', color=colors['parl'], lw=1.5))

# ====== LEGEND ======
legend_elements = [
    mpatches.Patch(facecolor=colors['client'], edgecolor='black', label='Local Clients'),
    mpatches.Patch(facecolor=colors['server'], edgecolor='black', label='Aggregation Server'),
    mpatches.Patch(facecolor=colors['parl'], edgecolor='black', label='PARL (Personalization)'),
    mpatches.Patch(facecolor=colors['ldca'], edgecolor='black', label='LDCA (Clustering)'),
    mpatches.Patch(facecolor=colors['blockchain'], edgecolor='black', label='Blockchain Audit'),
    mpatches.Patch(facecolor=colors['ipfs'], edgecolor='black', label='IPFS Storage'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9)

# ====== ANNOTATIONS ======
# Key equations
ax.text(1.75, 8.3, r'$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{PARL} \cdot \mathcal{L}_{PARL}$',
        fontsize=9, ha='center', va='center', color=colors['text'],
        bbox=dict(boxstyle='round', facecolor=colors['bg_light'], edgecolor='gray'))

ax.text(7, 8, r'$s_{ij} = \frac{q_i \cdot q_j}{\|q_i\| \cdot \|q_j\|}$',
        fontsize=9, ha='center', va='center', color=colors['text'],
        bbox=dict(boxstyle='round', facecolor=colors['bg_light'], edgecolor='gray'))

ax.text(12, 8.3, r'$H_{root} = Merkle(h_1, ..., h_n)$',
        fontsize=9, ha='center', va='center', color=colors['text'],
        bbox=dict(boxstyle='round', facecolor=colors['bg_light'], edgecolor='gray'))

# Flow labels
ax.text(4, 7.2, '1. Local\nTraining', fontsize=8, ha='center', va='center',
        color=colors['arrow'], fontweight='bold')
ax.text(4, 3.5, '2. Submit\nUpdates', fontsize=8, ha='center', va='center',
        color=colors['arrow'], fontweight='bold')
ax.text(7, 1.8, '3. Aggregate', fontsize=8, ha='center', va='center',
        color=colors['arrow'], fontweight='bold')

plt.tight_layout()
plt.savefig('bfedplc_architecture.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('bfedplc_architecture.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Architecture diagram saved as bfedplc_architecture.png and bfedplc_architecture.pdf")
plt.show()
