"""
Dataset utilities for FedPLC
Implements Non-IID data partitioning using Dirichlet distribution
Supports CIFAR-10, Fashion-MNIST, and SVHN datasets
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import Dict, List, Tuple, Optional
import os


def get_transforms(dataset_name: str, train: bool = True) -> transforms.Compose:
    """
    Get data transforms for different datasets
    
    Args:
        dataset_name: Name of dataset ("cifar10", "fmnist", "svhn")
        train: Whether for training or testing
    
    Returns:
        Compose transform
    """
    if dataset_name == "cifar10":
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616]
                )
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616]
                )
            ])
    
    elif dataset_name == "fmnist":
        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.2860], std=[0.3530])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.2860], std=[0.3530])
            ])
    
    elif dataset_name == "svhn":
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4377, 0.4438, 0.4728],
                    std=[0.1980, 0.2010, 0.1970]
                )
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4377, 0.4438, 0.4728],
                    std=[0.1980, 0.2010, 0.1970]
                )
            ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return transform


def get_dataset(dataset_name: str, 
                data_dir: str = "./data", 
                train: bool = True) -> Dataset:
    """
    Load dataset with appropriate transforms
    
    Args:
        dataset_name: Name of dataset
        data_dir: Directory to store/load data
        train: Whether to load train or test set
    
    Returns:
        Dataset object
    """
    transform = get_transforms(dataset_name, train)
    
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
        )
    
    elif dataset_name == "fmnist":
        dataset = datasets.FashionMNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
        )
    
    elif dataset_name == "svhn":
        split = "train" if train else "test"
        dataset = datasets.SVHN(
            root=data_dir,
            split=split,
            download=True,
            transform=transform
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def get_targets(dataset: Dataset) -> np.ndarray:
    """Extract targets/labels from dataset"""
    if hasattr(dataset, 'targets'):
        return np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        return np.array(dataset.labels)
    else:
        raise AttributeError("Dataset has no targets or labels attribute")


class NonIIDPartitioner:
    """
    Partitions dataset among clients using Dirichlet distribution
    to simulate Non-IID data distribution
    """
    
    def __init__(self, 
                 dataset: Dataset,
                 num_clients: int,
                 alpha: float = 0.5,
                 min_samples: int = 10,
                 seed: int = 42):
        """
        Args:
            dataset: PyTorch dataset to partition
            num_clients: Number of clients to partition among
            alpha: Dirichlet concentration parameter (lower = more Non-IID)
            min_samples: Minimum samples per client
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.alpha = alpha
        self.min_samples = min_samples
        self.seed = seed
        
        np.random.seed(seed)
        
        self.targets = get_targets(dataset)
        self.num_classes = len(np.unique(self.targets))
        
        self.client_indices = self._partition()
    
    def _partition(self) -> Dict[int, List[int]]:
        """
        Partition dataset using Dirichlet distribution
        
        Returns:
            Dictionary mapping client_id -> list of sample indices
        """
        client_indices = {i: [] for i in range(self.num_clients)}
        
        # Group indices by class
        class_indices = {}
        for c in range(self.num_classes):
            class_indices[c] = np.where(self.targets == c)[0].tolist()
        
        # For each class, distribute samples to clients using Dirichlet
        for c in range(self.num_classes):
            indices = class_indices[c]
            np.random.shuffle(indices)
            
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            
            # Calculate number of samples for each client
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            
            # Split indices
            split_indices = np.split(indices, proportions)
            
            for client_id, idx in enumerate(split_indices):
                client_indices[client_id].extend(idx.tolist())
        
        # Ensure minimum samples per client
        for client_id in range(self.num_clients):
            if len(client_indices[client_id]) < self.min_samples:
                # Add random samples from other clients
                all_indices = list(range(len(self.dataset)))
                existing = set(client_indices[client_id])
                available = [i for i in all_indices if i not in existing]
                additional = np.random.choice(
                    available, 
                    size=self.min_samples - len(client_indices[client_id]),
                    replace=False
                )
                client_indices[client_id].extend(additional.tolist())
        
        return client_indices
    
    def get_client_data(self, client_id: int) -> Subset:
        """Get subset of dataset for a specific client"""
        return Subset(self.dataset, self.client_indices[client_id])
    
    def get_client_loader(self, 
                          client_id: int, 
                          batch_size: int = 64,
                          shuffle: bool = True,
                          num_workers: int = 0) -> DataLoader:
        """Get DataLoader for a specific client"""
        import torch
        subset = self.get_client_data(client_id)
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_label_distribution(self, client_id: int) -> Dict[int, int]:
        """Get label distribution for a specific client"""
        indices = self.client_indices[client_id]
        labels = self.targets[indices]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def get_all_distributions(self) -> Dict[int, Dict[int, int]]:
        """Get label distributions for all clients"""
        return {
            client_id: self.get_label_distribution(client_id)
            for client_id in range(self.num_clients)
        }
    
    def get_statistics(self) -> Dict:
        """Get partitioning statistics"""
        num_samples = [len(self.client_indices[i]) for i in range(self.num_clients)]
        
        return {
            "num_clients": self.num_clients,
            "num_classes": self.num_classes,
            "alpha": self.alpha,
            "total_samples": len(self.dataset),
            "min_samples_per_client": min(num_samples),
            "max_samples_per_client": max(num_samples),
            "avg_samples_per_client": np.mean(num_samples),
            "std_samples_per_client": np.std(num_samples)
        }


class ConceptDriftSimulator:
    """
    Simulates concept drift by permuting labels or shifting data distribution
    """
    
    def __init__(self, 
                 partitioner: NonIIDPartitioner,
                 drift_type: str = "abrupt",
                 seed: int = 42):
        """
        Args:
            partitioner: NonIIDPartitioner object
            drift_type: Type of drift ("abrupt" or "incremental")
            seed: Random seed
        """
        self.partitioner = partitioner
        self.drift_type = drift_type
        self.seed = seed
        
        np.random.seed(seed)
        
        self.num_classes = partitioner.num_classes
        self.original_targets = partitioner.targets.copy()
        
        # Generate label permutation for drift
        self.drift_permutation = self._generate_permutation()
        
        # Track drift state
        self.drift_applied = {i: False for i in range(partitioner.num_clients)}
        self.drift_progress = {i: 0.0 for i in range(partitioner.num_clients)}
    
    def _generate_permutation(self) -> Dict[int, int]:
        """Generate a random label permutation for drift simulation"""
        labels = list(range(self.num_classes))
        shuffled = labels.copy()
        np.random.shuffle(shuffled)
        
        # Ensure some labels are actually changed
        while all(a == b for a, b in zip(labels, shuffled)):
            np.random.shuffle(shuffled)
        
        return dict(zip(labels, shuffled))
    
    def apply_abrupt_drift(self, 
                           client_ids: List[int],
                           affected_labels: Optional[List[int]] = None):
        """
        Apply abrupt drift to selected clients
        
        Args:
            client_ids: List of client IDs to apply drift
            affected_labels: Labels to permute (if None, use random subset)
        """
        if affected_labels is None:
            # Select random subset of labels to affect
            num_affected = max(1, self.num_classes // 2)
            affected_labels = np.random.choice(
                self.num_classes, 
                size=num_affected, 
                replace=False
            ).tolist()
        
        for client_id in client_ids:
            indices = self.partitioner.client_indices[client_id]
            
            for idx in indices:
                original_label = self.original_targets[idx]
                if original_label in affected_labels:
                    # Apply permutation
                    self.partitioner.targets[idx] = self.drift_permutation[original_label]
            
            self.drift_applied[client_id] = True
            self.drift_progress[client_id] = 1.0
        
        print(f"Abrupt drift applied to {len(client_ids)} clients")
        print(f"Affected labels: {affected_labels}")
    
    def apply_incremental_drift(self,
                                client_ids: List[int],
                                progress_increment: float = 0.33,
                                affected_labels: Optional[List[int]] = None):
        """
        Apply incremental drift to selected clients
        
        Args:
            client_ids: List of client IDs
            progress_increment: How much to increase drift (0-1)
            affected_labels: Labels to permute
        """
        if affected_labels is None:
            num_affected = max(1, self.num_classes // 2)
            affected_labels = np.random.choice(
                self.num_classes,
                size=num_affected,
                replace=False
            ).tolist()
        
        for client_id in client_ids:
            # Update progress
            old_progress = self.drift_progress[client_id]
            new_progress = min(1.0, old_progress + progress_increment)
            self.drift_progress[client_id] = new_progress
            
            # Calculate how many samples to flip based on progress
            indices = self.partitioner.client_indices[client_id]
            
            for idx in indices:
                original_label = self.original_targets[idx]
                if original_label in affected_labels:
                    # Probabilistically apply drift based on progress
                    if np.random.random() < new_progress:
                        self.partitioner.targets[idx] = self.drift_permutation[original_label]
                    else:
                        self.partitioner.targets[idx] = original_label
            
            self.drift_applied[client_id] = new_progress > 0
        
        avg_progress = np.mean([self.drift_progress[c] for c in client_ids])
        print(f"Incremental drift updated for {len(client_ids)} clients")
        print(f"Average drift progress: {avg_progress:.2%}")
    
    def reset_drift(self, client_ids: Optional[List[int]] = None):
        """Reset drift for specified clients (or all if None)"""
        if client_ids is None:
            client_ids = list(range(self.partitioner.num_clients))
        
        for client_id in client_ids:
            indices = self.partitioner.client_indices[client_id]
            for idx in indices:
                self.partitioner.targets[idx] = self.original_targets[idx]
            
            self.drift_applied[client_id] = False
            self.drift_progress[client_id] = 0.0
        
        print(f"Drift reset for {len(client_ids)} clients")
    
    def get_drift_status(self) -> Dict:
        """Get current drift status"""
        return {
            "drift_type": self.drift_type,
            "num_clients_affected": sum(self.drift_applied.values()),
            "avg_drift_progress": np.mean(list(self.drift_progress.values())),
            "permutation": self.drift_permutation
        }


def create_federated_dataloaders(
    dataset_name: str,
    num_clients: int = 100,
    alpha: float = 0.5,
    batch_size: int = 64,
    data_dir: str = "./data",
    seed: int = 42
) -> Tuple[Dict[int, DataLoader], DataLoader, NonIIDPartitioner]:
    """
    Create federated data loaders for all clients
    
    Args:
        dataset_name: Name of dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration
        batch_size: Batch size
        data_dir: Data directory
        seed: Random seed
    
    Returns:
        Tuple of (client_loaders, test_loader, partitioner)
    """
    import torch
    
    # Load datasets
    train_dataset = get_dataset(dataset_name, data_dir, train=True)
    test_dataset = get_dataset(dataset_name, data_dir, train=False)
    
    # Partition training data
    partitioner = NonIIDPartitioner(
        dataset=train_dataset,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed
    )
    
    # Check if CUDA is available for pin_memory
    use_pin_memory = torch.cuda.is_available()
    num_workers = 0  # Use 0 for Windows compatibility
    
    # Create client dataloaders
    client_loaders = {}
    for client_id in range(num_clients):
        client_loaders[client_id] = partitioner.get_client_loader(
            client_id=client_id,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    print(f"Created federated dataloaders for {num_clients} clients")
    print(f"Dataset: {dataset_name}, Alpha: {alpha}")
    stats = partitioner.get_statistics()
    print(f"Samples per client: min={stats['min_samples_per_client']}, "
          f"max={stats['max_samples_per_client']}, "
          f"avg={stats['avg_samples_per_client']:.1f}")
    
    return client_loaders, test_loader, partitioner
