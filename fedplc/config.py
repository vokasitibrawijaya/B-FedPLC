"""
Configuration file for FedPLC experiments
Based on paper: FedPLC - Federated Learning with Prototype-anchored Learning and Community Detection
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for FedPLC experiments"""
    
    # Dataset settings
    dataset: str = "cifar10"  # Options: "cifar10", "fmnist", "svhn"
    num_classes: int = 10
    
    # Federated Learning settings
    num_clients: int = 100  # N = 100 clients
    num_rounds: int = 200  # Total communication rounds
    local_epochs: int = 5  # E = 5 local epochs
    batch_size: int = 64  # Local batch size
    participation_rate: float = 0.1  # 10% client participation per round
    
    # Non-IID settings (Dirichlet distribution)
    alpha: float = 0.5  # Dirichlet concentration parameter (lower = more heterogeneous)
    
    # Model architecture
    hidden_dim: int = 512  # Representation dimension
    
    # PARL settings (Prototype-Anchored Representation Learning)
    parl_weight: float = 0.1  # λ_parl weight for alignment loss
    temperature: float = 0.07  # Temperature for InfoNCE loss
    warmup_rounds: int = 30  # Warmup rounds before community detection
    
    # LDCA settings (Label-wise Dynamic Community Adaptation)
    similarity_threshold: float = 0.85  # τ threshold (0.8-0.9 optimal)
    resolution: float = 1.0  # Louvain resolution parameter
    
    # Byzantine Fault Tolerance settings
    aggregation_method: str = "multi_krum"  # Options: "fedavg", "trimmed_mean", "krum", "multi_krum", "hybrid", "hybrid_aggressive"
    byzantine_fraction: float = 0.0  # Expected fraction of Byzantine clients
    trim_ratio: float = 0.2  # Trim ratio for trimmed mean
    selection_ratio: float = 0.6  # Selection ratio for multi-krum/hybrid
    
    # Concept Drift settings
    drift_type: str = "abrupt"  # Options: "abrupt", "incremental", "none"
    drift_round: int = 100  # Round when drift occurs
    incremental_drift_rounds: List[int] = field(default_factory=lambda: [100, 120, 140])
    drift_severity: float = 0.5  # Proportion of labels affected
    
    # Training settings
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # Hardware settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # Logging
    log_interval: int = 10
    save_interval: int = 50
    results_dir: str = "./results"
    checkpoints_dir: str = "./checkpoints"
    data_dir: str = "./data"
    
    # Random seed for reproducibility
    seed: int = 42


# Preset configurations for different experiments
CIFAR10_CONFIG = ExperimentConfig(
    dataset="cifar10",
    num_classes=10,
    hidden_dim=512,
)

FMNIST_CONFIG = ExperimentConfig(
    dataset="fmnist",
    num_classes=10,
    hidden_dim=256,
)

SVHN_CONFIG = ExperimentConfig(
    dataset="svhn",
    num_classes=10,
    hidden_dim=512,
)

# Concept drift experiment configurations
ABRUPT_DRIFT_CONFIG = ExperimentConfig(
    drift_type="abrupt",
    drift_round=100,
    drift_severity=0.5,
)

INCREMENTAL_DRIFT_CONFIG = ExperimentConfig(
    drift_type="incremental",
    incremental_drift_rounds=[100, 120, 140],
    drift_severity=0.3,
)
