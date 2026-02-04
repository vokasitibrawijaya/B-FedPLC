# FedPLC: Federated Learning with Prototype-anchored Learning and Community Detection
# B-FedPLC: Blockchain-Enabled Federated Learning with Dynamic Clustering
# Includes Byzantine Fault Tolerant aggregation (Multi-Krum, Hybrid)

__version__ = "2.0.0"
__author__ = "Rachmad Andri Atmoko"

from .config import ExperimentConfig
from .robust_aggregation import (
    RobustAggregator,
    create_aggregator,
    apply_byzantine_attack,
    apply_label_flip_attack,
    apply_backdoor_attack
)

__all__ = [
    'ExperimentConfig',
    'RobustAggregator',
    'create_aggregator',
    'apply_byzantine_attack',
    'apply_label_flip_attack',
    'apply_backdoor_attack',
]
