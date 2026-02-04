"""
Robust Aggregation Methods for B-FedPLC
=====================================
Byzantine-resilient aggregation methods including:
- FedAvg (baseline)
- Trimmed Mean
- Krum
- Multi-Krum
- Hybrid (Krum + Trimmed Mean)

These methods provide Byzantine Fault Tolerance (BFT) for federated learning.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict


class RobustAggregator:
    """
    Byzantine-resilient aggregation for federated learning.
    
    Supports multiple aggregation strategies:
    - 'fedavg': Standard federated averaging (no Byzantine protection)
    - 'trimmed_mean': Coordinate-wise trimmed mean
    - 'krum': Select single most representative update
    - 'multi_krum': Select top-k most representative updates
    - 'hybrid': Krum selection + Trimmed Mean aggregation
    - 'hybrid_aggressive': More aggressive filtering
    """
    
    SUPPORTED_METHODS = ['fedavg', 'trimmed_mean', 'krum', 'multi_krum', 'hybrid', 'hybrid_aggressive']
    
    def __init__(self, 
                 method: str = 'multi_krum',
                 trim_ratio: float = 0.2,
                 selection_ratio: float = 0.6,
                 num_byzantine: Optional[int] = None):
        """
        Initialize robust aggregator.
        
        Args:
            method: Aggregation method to use
            trim_ratio: Fraction of extreme values to trim (for trimmed_mean)
            selection_ratio: Fraction of clients to select (for multi_krum)
            num_byzantine: Expected number of Byzantine clients (for krum)
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown method: {method}. Supported: {self.SUPPORTED_METHODS}")
        
        self.method = method
        self.trim_ratio = trim_ratio
        self.selection_ratio = selection_ratio
        self.num_byzantine = num_byzantine
        
        # Statistics for logging
        self.last_scores = None
        self.last_selected = None
    
    def aggregate(self,
                  client_weights: Dict[int, Dict[str, torch.Tensor]],
                  client_data_sizes: Optional[Dict[int, int]] = None) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates using the configured method.
        
        Args:
            client_weights: Dict mapping client_id -> model state dict
            client_data_sizes: Optional dict mapping client_id -> local data size
        
        Returns:
            Aggregated model state dict
        """
        if len(client_weights) == 0:
            raise ValueError("No client weights provided")
        
        if self.method == 'fedavg':
            return self._fedavg(client_weights, client_data_sizes)
        elif self.method == 'trimmed_mean':
            return self._trimmed_mean(client_weights, client_data_sizes)
        elif self.method == 'krum':
            return self._krum(client_weights, multi=False)
        elif self.method == 'multi_krum':
            return self._multi_krum(client_weights, client_data_sizes)
        elif self.method == 'hybrid':
            return self._hybrid(client_weights, client_data_sizes, aggressive=False)
        elif self.method == 'hybrid_aggressive':
            return self._hybrid(client_weights, client_data_sizes, aggressive=True)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _fedavg(self,
                client_weights: Dict[int, Dict[str, torch.Tensor]],
                client_data_sizes: Optional[Dict[int, int]] = None) -> Dict[str, torch.Tensor]:
        """
        Standard Federated Averaging.
        
        Simple weighted average - vulnerable to Byzantine attacks.
        """
        client_ids = list(client_weights.keys())
        n = len(client_ids)
        
        # Compute weights
        if client_data_sizes is not None:
            total_size = sum(client_data_sizes.get(c, 1) for c in client_ids)
            weights = {c: client_data_sizes.get(c, 1) / total_size for c in client_ids}
        else:
            weights = {c: 1.0 / n for c in client_ids}
        
        # Initialize aggregated weights
        first_client = client_ids[0]
        aggregated = OrderedDict()
        
        for key in client_weights[first_client].keys():
            aggregated[key] = torch.zeros_like(
                client_weights[first_client][key], 
                dtype=torch.float32
            )
            for client_id in client_ids:
                aggregated[key] += weights[client_id] * client_weights[client_id][key].float()
        
        return aggregated
    
    def _trimmed_mean(self,
                      client_weights: Dict[int, Dict[str, torch.Tensor]],
                      client_data_sizes: Optional[Dict[int, int]] = None) -> Dict[str, torch.Tensor]:
        """
        Coordinate-wise Trimmed Mean.
        
        For each parameter coordinate, sort values and remove top/bottom trim_ratio.
        """
        client_ids = list(client_weights.keys())
        n = len(client_ids)
        trim_count = int(n * self.trim_ratio)
        
        first_client = client_ids[0]
        aggregated = OrderedDict()
        
        for key in client_weights[first_client].keys():
            # Stack all client values for this parameter
            stacked = torch.stack([
                client_weights[c][key].float() for c in client_ids
            ], dim=0)  # Shape: (n_clients, *param_shape)
            
            if trim_count > 0 and n > 2 * trim_count:
                # Sort along client dimension
                sorted_vals, _ = torch.sort(stacked, dim=0)
                # Trim top and bottom
                trimmed = sorted_vals[trim_count:n - trim_count]
                # Average remaining
                aggregated[key] = trimmed.mean(dim=0)
            else:
                # Not enough clients to trim, use mean
                aggregated[key] = stacked.mean(dim=0)
        
        return aggregated
    
    def _compute_krum_scores(self,
                             client_weights: Dict[int, Dict[str, torch.Tensor]]) -> Dict[int, float]:
        """
        Compute Krum scores for each client.
        
        Lower score = more similar to other clients = more trustworthy.
        """
        client_ids = list(client_weights.keys())
        n = len(client_ids)
        
        # Estimate number of Byzantine clients
        if self.num_byzantine is not None:
            f = self.num_byzantine
        else:
            f = max(1, int(n * 0.3))  # Assume up to 30% Byzantine
        
        # Flatten client weights into vectors
        client_vectors = {}
        for client_id in client_ids:
            params = []
            for key in client_weights[client_id].keys():
                params.append(client_weights[client_id][key].float().flatten())
            client_vectors[client_id] = torch.cat(params)
        
        # Compute pairwise distances
        distances = {}
        for i, ci in enumerate(client_ids):
            distances[ci] = {}
            for j, cj in enumerate(client_ids):
                if i != j:
                    dist = torch.norm(client_vectors[ci] - client_vectors[cj]).item()
                    distances[ci][cj] = dist
        
        # Compute Krum scores (sum of n-f-2 closest neighbors)
        scores = {}
        k = max(1, n - f - 2)  # Number of neighbors to consider
        
        for ci in client_ids:
            neighbor_dists = sorted(distances[ci].values())
            scores[ci] = sum(neighbor_dists[:k])
        
        self.last_scores = scores
        return scores
    
    def _krum(self,
              client_weights: Dict[int, Dict[str, torch.Tensor]],
              multi: bool = False) -> Dict[str, torch.Tensor]:
        """
        Krum aggregation.
        
        Selects the client with lowest Krum score (most representative).
        """
        scores = self._compute_krum_scores(client_weights)
        
        # Select client with lowest score
        best_client = min(scores, key=scores.get)
        self.last_selected = [best_client]
        
        return OrderedDict({
            k: v.clone() for k, v in client_weights[best_client].items()
        })
    
    def _multi_krum(self,
                    client_weights: Dict[int, Dict[str, torch.Tensor]],
                    client_data_sizes: Optional[Dict[int, int]] = None) -> Dict[str, torch.Tensor]:
        """
        Multi-Krum aggregation.
        
        Selects top-k clients by Krum score and averages their updates.
        """
        scores = self._compute_krum_scores(client_weights)
        client_ids = list(client_weights.keys())
        n = len(client_ids)
        
        # Select top-k clients
        k = max(1, int(n * self.selection_ratio))
        sorted_clients = sorted(scores.keys(), key=lambda c: scores[c])
        selected_clients = sorted_clients[:k]
        self.last_selected = selected_clients
        
        # Average selected clients
        selected_weights = {c: client_weights[c] for c in selected_clients}
        selected_sizes = None
        if client_data_sizes is not None:
            selected_sizes = {c: client_data_sizes.get(c, 1) for c in selected_clients}
        
        return self._fedavg(selected_weights, selected_sizes)
    
    def _hybrid(self,
                client_weights: Dict[int, Dict[str, torch.Tensor]],
                client_data_sizes: Optional[Dict[int, int]] = None,
                aggressive: bool = False) -> Dict[str, torch.Tensor]:
        """
        Hybrid aggregation: Krum selection + Trimmed Mean.
        
        1. Use Krum scores to filter out suspicious clients
        2. Apply Trimmed Mean on remaining clients
        
        Args:
            aggressive: If True, use tighter selection (50%) and higher trim (30%)
        """
        scores = self._compute_krum_scores(client_weights)
        client_ids = list(client_weights.keys())
        n = len(client_ids)
        
        # Selection parameters
        if aggressive:
            selection_ratio = 0.5
            trim_ratio = 0.3
        else:
            selection_ratio = self.selection_ratio
            trim_ratio = self.trim_ratio
        
        # Step 1: Krum-based selection
        k = max(3, int(n * selection_ratio))  # At least 3 for trimmed mean
        sorted_clients = sorted(scores.keys(), key=lambda c: scores[c])
        selected_clients = sorted_clients[:k]
        self.last_selected = selected_clients
        
        # Step 2: Trimmed Mean on selected clients
        selected_weights = {c: client_weights[c] for c in selected_clients}
        
        first_client = selected_clients[0]
        aggregated = OrderedDict()
        m = len(selected_clients)
        trim_count = int(m * trim_ratio)
        
        for key in selected_weights[first_client].keys():
            stacked = torch.stack([
                selected_weights[c][key].float() for c in selected_clients
            ], dim=0)
            
            if trim_count > 0 and m > 2 * trim_count:
                sorted_vals, _ = torch.sort(stacked, dim=0)
                trimmed = sorted_vals[trim_count:m - trim_count]
                aggregated[key] = trimmed.mean(dim=0)
            else:
                aggregated[key] = stacked.mean(dim=0)
        
        return aggregated
    
    def get_statistics(self) -> Dict:
        """Get statistics from last aggregation."""
        return {
            'method': self.method,
            'last_scores': self.last_scores,
            'last_selected': self.last_selected,
            'num_selected': len(self.last_selected) if self.last_selected else 0
        }


def create_aggregator(method: str = 'multi_krum', **kwargs) -> RobustAggregator:
    """
    Factory function to create aggregator with preset configurations.
    
    Args:
        method: Aggregation method
        **kwargs: Additional configuration
    
    Returns:
        Configured RobustAggregator
    """
    defaults = {
        'fedavg': {'trim_ratio': 0.0, 'selection_ratio': 1.0},
        'trimmed_mean': {'trim_ratio': 0.2, 'selection_ratio': 1.0},
        'krum': {'trim_ratio': 0.0, 'selection_ratio': 0.0},
        'multi_krum': {'trim_ratio': 0.0, 'selection_ratio': 0.6},
        'hybrid': {'trim_ratio': 0.2, 'selection_ratio': 0.6},
        'hybrid_aggressive': {'trim_ratio': 0.3, 'selection_ratio': 0.5},
    }
    
    config = defaults.get(method, {})
    config.update(kwargs)
    
    return RobustAggregator(method=method, **config)


# ============================================================================
# Attack Simulation Functions
# ============================================================================

def apply_byzantine_attack(weights: Dict[str, torch.Tensor],
                           attack_type: str = 'random',
                           attack_strength: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    Apply Byzantine attack to model weights.
    
    Args:
        weights: Model state dict
        attack_type: Type of attack ('random', 'sign_flip', 'label_flip', 'backdoor')
        attack_strength: Strength multiplier for the attack
    
    Returns:
        Corrupted model weights
    """
    corrupted = OrderedDict()
    
    for key, value in weights.items():
        # Skip non-float tensors (e.g., batch norm running stats)
        if not value.is_floating_point():
            corrupted[key] = value.clone()
            continue
            
        if attack_type == 'random':
            # Replace with random noise
            corrupted[key] = torch.randn_like(value) * attack_strength
        
        elif attack_type == 'sign_flip':
            # Flip the sign of all gradients
            corrupted[key] = -value * attack_strength
        
        elif attack_type == 'scale':
            # Scale weights by large factor
            corrupted[key] = value * (10 * attack_strength)
        
        elif attack_type == 'additive_noise':
            # Add Gaussian noise
            noise = torch.randn_like(value) * torch.std(value) * attack_strength
            corrupted[key] = value + noise
        
        elif attack_type == 'zero':
            # Zero out all weights
            corrupted[key] = torch.zeros_like(value)
        
        else:
            # Default: random noise
            corrupted[key] = torch.randn_like(value) * attack_strength
    
    return corrupted


def apply_label_flip_attack(labels: torch.Tensor,
                            num_classes: int = 10,
                            flip_ratio: float = 1.0) -> torch.Tensor:
    """
    Apply label flipping attack.
    
    Args:
        labels: Original labels tensor
        num_classes: Number of classes
        flip_ratio: Fraction of labels to flip
    
    Returns:
        Corrupted labels
    """
    corrupted_labels = labels.clone()
    n = len(labels)
    num_flip = int(n * flip_ratio)
    
    # Randomly select indices to flip
    flip_indices = np.random.choice(n, size=num_flip, replace=False)
    
    # Flip to random different class
    for idx in flip_indices:
        original = corrupted_labels[idx].item()
        new_label = (original + np.random.randint(1, num_classes)) % num_classes
        corrupted_labels[idx] = new_label
    
    return corrupted_labels


def apply_backdoor_attack(data: torch.Tensor,
                          labels: torch.Tensor,
                          target_label: int = 0,
                          poison_ratio: float = 0.1,
                          trigger_pattern: str = 'square') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply backdoor attack by adding trigger pattern to data.
    
    Args:
        data: Input data tensor (batch, channels, height, width)
        labels: Labels tensor
        target_label: Target label for backdoor
        poison_ratio: Fraction of data to poison
        trigger_pattern: Type of trigger ('square', 'cross', 'random')
    
    Returns:
        Tuple of (poisoned_data, poisoned_labels)
    """
    poisoned_data = data.clone()
    poisoned_labels = labels.clone()
    
    n = len(data)
    num_poison = int(n * poison_ratio)
    poison_indices = np.random.choice(n, size=num_poison, replace=False)
    
    for idx in poison_indices:
        # Add trigger pattern
        if trigger_pattern == 'square':
            # Small white square in corner
            poisoned_data[idx, :, -5:, -5:] = 1.0
        elif trigger_pattern == 'cross':
            # Cross pattern
            poisoned_data[idx, :, -5:, -3:-2] = 1.0
            poisoned_data[idx, :, -3:-2, -5:] = 1.0
        elif trigger_pattern == 'random':
            # Random noise pattern
            poisoned_data[idx, :, -5:, -5:] = torch.rand(data.shape[1], 5, 5)
        
        # Change label to target
        poisoned_labels[idx] = target_label
    
    return poisoned_data, poisoned_labels
