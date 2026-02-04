"""
Utility functions for FedPLC implementation
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str = "./logs") -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"fedplc_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def average_weights(weights_list: List[Dict[str, torch.Tensor]], 
                    weights_coefficients: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
    """
    Average model weights for federated aggregation
    
    Args:
        weights_list: List of model state dictionaries
        weights_coefficients: Optional weights for weighted averaging
    
    Returns:
        Averaged state dictionary
    """
    if weights_coefficients is None:
        weights_coefficients = [1.0 / len(weights_list)] * len(weights_list)
    
    averaged_weights = {}
    
    for key in weights_list[0].keys():
        averaged_weights[key] = torch.zeros_like(weights_list[0][key], dtype=torch.float32)
        for i, weights in enumerate(weights_list):
            averaged_weights[key] += weights_coefficients[i] * weights[key].float()
    
    return averaged_weights


def compute_similarity(w1: Dict[str, torch.Tensor], 
                       w2: Dict[str, torch.Tensor],
                       keys: Optional[List[str]] = None) -> float:
    """
    Compute cosine similarity between two sets of model weights
    
    Args:
        w1, w2: Model state dictionaries
        keys: Optional list of keys to compare (if None, use all keys)
    
    Returns:
        Cosine similarity value
    """
    if keys is None:
        keys = list(w1.keys())
    
    vec1 = torch.cat([w1[k].flatten() for k in keys])
    vec2 = torch.cat([w2[k].flatten() for k in keys])
    
    similarity = torch.nn.functional.cosine_similarity(
        vec1.unsqueeze(0), vec2.unsqueeze(0)
    ).item()
    
    return similarity


def split_model_weights(state_dict: Dict[str, torch.Tensor], 
                        representation_keys: List[str]) -> Tuple[Dict, Dict]:
    """
    Split model weights into representation layer and classifier head
    
    Args:
        state_dict: Full model state dictionary
        representation_keys: Keys belonging to representation layer
    
    Returns:
        Tuple of (representation_weights, classifier_weights)
    """
    repr_weights = {k: v for k, v in state_dict.items() if any(rk in k for rk in representation_keys)}
    clf_weights = {k: v for k, v in state_dict.items() if k not in repr_weights}
    
    return repr_weights, clf_weights


def save_checkpoint(state: Dict, 
                    checkpoint_dir: str, 
                    filename: str = "checkpoint.pth"):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict:
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint


def save_results(results: Dict, 
                 results_dir: str, 
                 filename: str = "results.json"):
    """Save experiment results to JSON"""
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    
    # Convert numpy arrays to lists
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results = convert_to_serializable(results)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {filepath}")


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop
