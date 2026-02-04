"""
PARL: Prototype-Anchored Representation Learning
Implements the local training mechanism for FedPLC

Key components:
1. Prototype-based feature alignment using InfoNCE loss
2. Decoupled training for representation and classifier
3. Local prototype maintenance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import numpy as np


class PARLLoss(nn.Module):
    """
    Prototype-Anchored Representation Learning Loss
    
    Combines:
    1. Cross-entropy loss for classification
    2. InfoNCE-style alignment loss to anchor features to class prototypes
    
    L_total = L_ce + λ * L_align
    
    where L_align is the prototype alignment loss
    """
    
    def __init__(self, 
                 temperature: float = 0.07,
                 parl_weight: float = 0.1):
        """
        Args:
            temperature: Temperature parameter for InfoNCE
            parl_weight: Weight for alignment loss (λ)
        """
        super().__init__()
        
        self.temperature = temperature
        self.parl_weight = parl_weight
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self,
                logits: torch.Tensor,
                features: torch.Tensor,
                labels: torch.Tensor,
                prototypes: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss
        
        Args:
            logits: (batch_size, num_classes) classification logits
            features: (batch_size, hidden_dim) feature embeddings
            labels: (batch_size,) ground truth labels
            prototypes: (num_classes, hidden_dim) class prototypes
        
        Returns:
            Tuple of (total_loss, loss_dict with individual components)
        """
        # Classification loss
        loss_ce = self.ce_loss(logits, labels)
        
        # Prototype alignment loss (InfoNCE-style)
        loss_align = self.compute_alignment_loss(features, labels, prototypes)
        
        # Combined loss
        total_loss = loss_ce + self.parl_weight * loss_align
        
        loss_dict = {
            'total': total_loss.item(),
            'ce': loss_ce.item(),
            'align': loss_align.item()
        }
        
        return total_loss, loss_dict
    
    def compute_alignment_loss(self,
                               features: torch.Tensor,
                               labels: torch.Tensor,
                               prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE-style alignment loss
        
        Encourages features to be close to their class prototype
        and far from other class prototypes
        
        L_align = -log(exp(sim(z, p+)/τ) / Σ exp(sim(z, p)/τ))
        """
        # Normalize features and prototypes
        features_norm = F.normalize(features, dim=1)
        prototypes_norm = F.normalize(prototypes, dim=1)
        
        # Compute similarity to all prototypes
        # (batch_size, num_classes)
        similarity = torch.mm(features_norm, prototypes_norm.t()) / self.temperature
        
        # Create labels for contrastive loss
        # Each sample should be similar to its class prototype
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class LocalTrainer:
    """
    Local training class for FedPLC clients
    Implements PARL training procedure
    """
    
    def __init__(self,
                 model: nn.Module,
                 dataloader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 local_epochs: int = 5,
                 parl_weight: float = 0.1,
                 temperature: float = 0.07,
                 prototype_momentum: float = 0.9):
        """
        Args:
            model: FedPLC model
            dataloader: Client's local data loader
            device: Training device
            learning_rate: Learning rate
            momentum: SGD momentum
            weight_decay: L2 regularization
            local_epochs: Number of local training epochs
            parl_weight: Weight for PARL alignment loss
            temperature: Temperature for InfoNCE
            prototype_momentum: EMA momentum for prototype updates
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.local_epochs = local_epochs
        self.prototype_momentum = prototype_momentum
        
        # Loss function
        self.criterion = PARLLoss(
            temperature=temperature,
            parl_weight=parl_weight
        )
        
        # Optimizer
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Local prototypes
        self.local_prototypes = None
    
    def train(self, 
              global_prototypes: Optional[torch.Tensor] = None,
              warmup: bool = False) -> Dict:
        """
        Perform local training
        
        Args:
            global_prototypes: Global prototypes from server (if available)
            warmup: Whether in warmup phase (use only CE loss)
        
        Returns:
            Dictionary with training statistics
        """
        self.model.train()
        self.model.to(self.device)
        
        # Use global prototypes if provided, otherwise use model's prototypes
        if global_prototypes is not None:
            prototypes = global_prototypes.to(self.device)
        else:
            prototypes = self.model.prototypes.to(self.device)
        
        # Initialize local prototypes
        self.local_prototypes = torch.zeros_like(prototypes)
        prototype_counts = torch.zeros(prototypes.size(0), device=self.device)
        
        total_loss = 0.0
        total_ce = 0.0
        total_align = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(self.dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                logits, features = self.model(data)
                
                if warmup:
                    # During warmup, only use classification loss
                    loss = F.cross_entropy(logits, target)
                    loss_dict = {
                        'total': loss.item(),
                        'ce': loss.item(),
                        'align': 0.0
                    }
                else:
                    # Use PARL loss
                    loss, loss_dict = self.criterion(
                        logits, features, target, prototypes
                    )
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update local prototypes
                with torch.no_grad():
                    for c in range(prototypes.size(0)):
                        mask = target == c
                        if mask.sum() > 0:
                            class_features = features[mask].mean(dim=0)
                            
                            if prototype_counts[c] == 0:
                                self.local_prototypes[c] = class_features
                            else:
                                # EMA update
                                self.local_prototypes[c] = (
                                    self.prototype_momentum * self.local_prototypes[c] +
                                    (1 - self.prototype_momentum) * class_features
                                )
                            
                            prototype_counts[c] += 1
                
                # Statistics
                epoch_loss += loss_dict['total']
                total_ce += loss_dict['ce']
                total_align += loss_dict['align']
                
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            total_loss += epoch_loss / len(self.dataloader)
        
        accuracy = 100. * correct / total
        
        return {
            'loss': total_loss / self.local_epochs,
            'ce_loss': total_ce / (self.local_epochs * len(self.dataloader)),
            'align_loss': total_align / (self.local_epochs * len(self.dataloader)),
            'accuracy': accuracy,
            'local_prototypes': self.local_prototypes.cpu()
        }
    
    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get current model state dictionary"""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def get_representation_state(self) -> Dict[str, torch.Tensor]:
        """Get representation network state"""
        return self.model.get_representation_params()
    
    def get_classifier_state(self) -> Dict[str, torch.Tensor]:
        """Get classifier state"""
        return self.model.get_classifier_params()


class DecoupledTrainer(LocalTrainer):
    """
    Decoupled local trainer for FedPLC
    Trains representation and classifier separately
    """
    
    def __init__(self, *args, 
                 repr_lr: float = 0.01,
                 clf_lr: float = 0.01,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        # Separate optimizers for representation and classifier
        self.repr_optimizer = torch.optim.SGD(
            self.model.representation.parameters(),
            lr=repr_lr,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 1e-4)
        )
        
        self.clf_optimizer = torch.optim.SGD(
            self.model.classifier.parameters(),
            lr=clf_lr,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 1e-4)
        )
    
    def train_representation(self,
                             global_prototypes: Optional[torch.Tensor] = None) -> Dict:
        """
        Train only representation network with PARL loss
        Classifier is frozen
        """
        self.model.train()
        self.model.to(self.device)
        
        # Freeze classifier
        for param in self.model.classifier.parameters():
            param.requires_grad = False
        
        prototypes = (global_prototypes.to(self.device) 
                     if global_prototypes is not None 
                     else self.model.prototypes.to(self.device))
        
        total_loss = 0.0
        
        for epoch in range(self.local_epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.repr_optimizer.zero_grad()
                
                # Forward pass
                features = self.model.get_features(data)
                
                # PARL alignment loss only
                loss = self.criterion.compute_alignment_loss(
                    features, target, prototypes
                )
                
                loss.backward()
                self.repr_optimizer.step()
                
                total_loss += loss.item()
        
        # Unfreeze classifier
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
        return {
            'repr_loss': total_loss / (self.local_epochs * len(self.dataloader))
        }
    
    def train_classifier(self) -> Dict:
        """
        Train only classifier with cross-entropy loss
        Representation is frozen
        """
        self.model.train()
        self.model.to(self.device)
        
        # Freeze representation
        for param in self.model.representation.parameters():
            param.requires_grad = False
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.local_epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.clf_optimizer.zero_grad()
                
                # Forward pass
                with torch.no_grad():
                    features = self.model.get_features(data)
                
                logits = self.model.classify(features)
                
                loss = F.cross_entropy(logits, target)
                loss.backward()
                self.clf_optimizer.step()
                
                total_loss += loss.item()
                
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        # Unfreeze representation
        for param in self.model.representation.parameters():
            param.requires_grad = True
        
        return {
            'clf_loss': total_loss / (self.local_epochs * len(self.dataloader)),
            'accuracy': 100. * correct / total
        }
    
    def train_decoupled(self,
                        global_prototypes: Optional[torch.Tensor] = None,
                        warmup: bool = False) -> Dict:
        """
        Full decoupled training: representation then classifier
        """
        if warmup:
            # During warmup, train jointly
            return super().train(global_prototypes, warmup=True)
        
        # Train representation
        repr_stats = self.train_representation(global_prototypes)
        
        # Train classifier
        clf_stats = self.train_classifier()
        
        # Update local prototypes after training
        self._update_local_prototypes()
        
        return {
            **repr_stats,
            **clf_stats,
            'local_prototypes': self.local_prototypes.cpu()
        }
    
    def _update_local_prototypes(self):
        """Update local prototypes after training"""
        self.model.eval()
        
        self.local_prototypes = torch.zeros(
            self.model.num_classes, 
            self.model.hidden_dim,
            device=self.device
        )
        prototype_counts = torch.zeros(self.model.num_classes, device=self.device)
        
        with torch.no_grad():
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                features = self.model.get_features(data)
                
                for c in range(self.model.num_classes):
                    mask = target == c
                    if mask.sum() > 0:
                        class_features = features[mask].sum(dim=0)
                        self.local_prototypes[c] += class_features
                        prototype_counts[c] += mask.sum()
        
        # Normalize
        for c in range(self.model.num_classes):
            if prototype_counts[c] > 0:
                self.local_prototypes[c] /= prototype_counts[c]
