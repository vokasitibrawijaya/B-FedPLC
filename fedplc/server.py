"""
FedPLC Server Implementation
Coordinates federated learning with PARL and LDCA
Includes Robust Byzantine-Resilient Aggregation (Multi-Krum)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy
from collections import defaultdict

from .models import FedPLCModel, create_model
from .ldca import LDCAManager, GlobalAggregator
from .utils import average_weights, compute_similarity
from .robust_aggregation import RobustAggregator, create_aggregator


class FedPLCServer:
    """
    Federated Learning Server for FedPLC
    
    Responsibilities:
    1. Coordinate client selection
    2. Manage global model and prototypes
    3. Perform LDCA community detection
    4. Aggregate representation layers globally
    5. Aggregate classifier heads by community
    """
    
    def __init__(self,
                 config,
                 model: FedPLCModel,
                 test_loader: DataLoader,
                 aggregation_method: str = 'multi_krum'):
        """
        Args:
            config: ExperimentConfig object
            model: Global FedPLC model
            test_loader: Test data loader for evaluation
            aggregation_method: Robust aggregation method 
                               ('fedavg', 'trimmed_mean', 'krum', 'multi_krum', 'hybrid', 'hybrid_aggressive')
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Global model
        self.global_model = model.to(self.device)
        self.test_loader = test_loader
        
        # Initialize robust aggregator (default: multi_krum for BFT compliance)
        self.aggregation_method = aggregation_method
        self.robust_aggregator = create_aggregator(aggregation_method)
        print(f"[B-FedPLC] Using robust aggregation: {aggregation_method.upper()}")
        
        # Initialize managers
        self.ldca_manager = LDCAManager(
            num_clients=config.num_clients,
            num_classes=config.num_classes,
            threshold=config.similarity_threshold,
            resolution=config.resolution
        )
        
        self.global_aggregator = GlobalAggregator(config.num_clients)
        
        # Global prototypes
        self.global_prototypes = torch.zeros(
            config.num_classes, config.hidden_dim
        ).to(self.device)
        
        # Training history
        self.history = {
            'round': [],
            'accuracy': [],
            'loss': [],
            'community_stats': []
        }
        
        # Current round
        self.current_round = 0
        self.is_warmup = True
    
    def select_clients(self, 
                       num_clients: Optional[int] = None) -> List[int]:
        """
        Select clients for current round
        
        Args:
            num_clients: Number of clients to select
        
        Returns:
            List of selected client IDs
        """
        if num_clients is None:
            num_clients = int(self.config.num_clients * self.config.participation_rate)
        
        selected = np.random.choice(
            self.config.num_clients,
            size=min(num_clients, self.config.num_clients),
            replace=False
        )
        
        return selected.tolist()
    
    def get_global_model_state(self) -> Dict[str, torch.Tensor]:
        """Get current global model state"""
        return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
    
    def get_global_prototypes(self) -> torch.Tensor:
        """Get current global prototypes"""
        return self.global_prototypes.clone()
    
    def aggregate_round(self,
                        client_updates: Dict[int, Dict],
                        client_data_sizes: Dict[int, int]):
        """
        Aggregate client updates for a round using robust aggregation.
        
        Args:
            client_updates: Dict[client_id] -> {
                'repr_weights': representation weights,
                'clf_weights': Dict[label] -> classifier weights,
                'prototypes': local prototypes,
                'stats': training statistics
            }
            client_data_sizes: Dict[client_id] -> local data size
        """
        if len(client_updates) == 0:
            return
        
        # 1. Aggregate representation layers using ROBUST AGGREGATION
        repr_weights = {
            c: updates['repr_weights']
            for c, updates in client_updates.items()
        }
        
        # Use robust aggregator instead of simple averaging
        global_repr = self.robust_aggregator.aggregate(repr_weights, client_data_sizes)
        
        # Update global model representation
        self._update_representation(global_repr)
        
        # 2. If not in warmup, perform LDCA community-based aggregation
        if not self.is_warmup:
            # Collect label-wise classifier weights
            label_clf_weights = {
                c: updates['clf_weights']
                for c, updates in client_updates.items()
            }
            
            # Update communities based on classifier similarity
            self.ldca_manager.update_communities(label_clf_weights)
            
            # Aggregate by community
            label_data_sizes = self._compute_label_data_sizes(client_updates)
            aggregated_clf = self.ldca_manager.aggregate_label_wise(
                label_clf_weights, label_data_sizes
            )
            
            # Store community-aggregated classifiers for distribution
            self.community_clf_weights = aggregated_clf
        else:
            # During warmup, use global averaging for classifiers
            self._aggregate_classifier_global(client_updates, client_data_sizes)
        
        # 3. Aggregate prototypes
        client_prototypes = {
            c: updates['prototypes']
            for c, updates in client_updates.items()
        }
        
        self.global_prototypes = self.global_aggregator.aggregate_prototypes(
            client_prototypes
        ).to(self.device)
        
        # Update model's internal prototypes
        self.global_model.prototypes.copy_(self.global_prototypes)
    
    def _update_representation(self, repr_weights: Dict[str, torch.Tensor]):
        """Update global model's representation network"""
        current_state = self.global_model.state_dict()
        
        for key, value in repr_weights.items():
            if key in current_state:
                current_state[key] = value.to(self.device)
        
        self.global_model.load_state_dict(current_state)
    
    def _aggregate_classifier_global(self,
                                     client_updates: Dict[int, Dict],
                                     client_data_sizes: Dict[int, int]):
        """Global averaging for classifier during warmup"""
        clf_weights_list = []
        coefficients = []
        
        total_data = sum(client_data_sizes.values())
        
        for client_id, updates in client_updates.items():
            # Flatten label-wise weights to full classifier
            full_clf_weights = {}
            for label, label_weights in updates['clf_weights'].items():
                for name, param in label_weights.items():
                    key = f"classifiers.{label}.{name}"
                    full_clf_weights[key] = param
            
            clf_weights_list.append(full_clf_weights)
            coefficients.append(client_data_sizes[client_id] / total_data)
        
        # Average classifier weights
        if len(clf_weights_list) > 0:
            avg_clf = average_weights(clf_weights_list, coefficients)
            
            # Update classifier in global model
            current_state = self.global_model.state_dict()
            for key, value in avg_clf.items():
                full_key = f"classifier.{key}"
                if full_key in current_state:
                    current_state[full_key] = value.to(self.device)
            
            self.global_model.load_state_dict(current_state)
    
    def _compute_label_data_sizes(self,
                                  client_updates: Dict[int, Dict]) -> Dict[int, Dict[int, int]]:
        """Compute per-label data sizes for weighted aggregation"""
        # This would be computed from actual client data distributions
        # For simplicity, assume equal contribution
        label_sizes = {}
        for client_id in client_updates.keys():
            label_sizes[client_id] = {
                c: 1 for c in range(self.config.num_classes)
            }
        return label_sizes
    
    def get_client_model(self, client_id: int) -> Dict[str, torch.Tensor]:
        """
        Get model weights for a specific client
        
        After LDCA, each client may have different classifier heads
        based on their community membership
        """
        # Start with global model
        model_state = self.get_global_model_state()
        
        # If not in warmup and we have community-specific classifiers
        if not self.is_warmup and hasattr(self, 'community_clf_weights'):
            if client_id in self.community_clf_weights:
                # Update classifier with community-aggregated weights
                for label, weights in self.community_clf_weights[client_id].items():
                    for name, param in weights.items():
                        key = f"classifier.classifiers.{label}.{name}"
                        if key in model_state:
                            model_state[key] = param
        
        return model_state
    
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate global model on test set
        
        Returns:
            Tuple of (accuracy, loss)
        """
        self.global_model.eval()
        
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                logits, _ = self.global_model(data)
                loss = criterion(logits, target)
                
                total_loss += loss.item() * data.size(0)
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / total
        
        return accuracy, avg_loss
    
    def step_round(self):
        """Advance to next round and update warmup status"""
        self.current_round += 1
        
        if self.current_round >= self.config.warmup_rounds:
            if self.is_warmup:
                print(f"\n[Round {self.current_round}] Warmup completed. "
                      f"Starting LDCA community detection.")
            self.is_warmup = False
    
    def log_round(self, accuracy: float, loss: float):
        """Log round statistics"""
        self.history['round'].append(self.current_round)
        self.history['accuracy'].append(accuracy)
        self.history['loss'].append(loss)
        
        if not self.is_warmup:
            stats = self.ldca_manager.get_statistics()
            self.history['community_stats'].append(stats)
    
    def save_state(self, path: str):
        """Save server state"""
        state = {
            'round': self.current_round,
            'model_state': self.global_model.state_dict(),
            'prototypes': self.global_prototypes,
            'history': self.history,
            'is_warmup': self.is_warmup
        }
        torch.save(state, path)
        print(f"Server state saved to {path}")
    
    def load_state(self, path: str):
        """Load server state"""
        state = torch.load(path, map_location=self.device)
        
        self.current_round = state['round']
        self.global_model.load_state_dict(state['model_state'])
        self.global_prototypes = state['prototypes'].to(self.device)
        self.history = state['history']
        self.is_warmup = state['is_warmup']
        
        print(f"Server state loaded from {path}")


class FedPLCClient:
    """
    Federated Learning Client for FedPLC
    
    Responsibilities:
    1. Local training with PARL
    2. Extract and return label-wise classifier weights
    3. Maintain local prototypes
    """
    
    def __init__(self,
                 client_id: int,
                 config,
                 dataloader: DataLoader,
                 model: Optional[FedPLCModel] = None):
        """
        Args:
            client_id: Unique client identifier
            config: ExperimentConfig
            dataloader: Local data loader
            model: Optional initial model
        """
        self.client_id = client_id
        self.config = config
        self.device = torch.device(config.device)
        self.dataloader = dataloader
        
        # Initialize local model
        if model is None:
            self.model = create_model(
                dataset=config.dataset,
                hidden_dim=config.hidden_dim,
                num_classes=config.num_classes,
                use_labelwise=True
            )
        else:
            self.model = copy.deepcopy(model)
        
        self.model.to(self.device)
        
        # Local data size
        self.data_size = len(dataloader.dataset)
        
        # Local prototypes
        self.local_prototypes = None
    
    def receive_global_model(self, model_state: Dict[str, torch.Tensor]):
        """Receive and load global model weights"""
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in model_state.items()}
        )
    
    def train(self,
              global_prototypes: Optional[torch.Tensor] = None,
              warmup: bool = False) -> Dict:
        """
        Perform local training
        
        Args:
            global_prototypes: Global prototypes from server
            warmup: Whether in warmup phase
        
        Returns:
            Dictionary with update information
        """
        from .parl import DecoupledTrainer
        
        trainer = DecoupledTrainer(
            model=self.model,
            dataloader=self.dataloader,
            device=self.device,
            learning_rate=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
            local_epochs=self.config.local_epochs,
            parl_weight=self.config.parl_weight,
            temperature=self.config.temperature,
            repr_lr=self.config.learning_rate,
            clf_lr=self.config.learning_rate
        )
        
        # Perform training
        if warmup:
            stats = trainer.train(global_prototypes, warmup=True)
        else:
            stats = trainer.train_decoupled(global_prototypes, warmup=False)
        
        self.local_prototypes = stats.get('local_prototypes', trainer.local_prototypes)
        
        # Extract weights
        repr_weights = self._get_representation_weights()
        clf_weights = self._get_labelwise_classifier_weights()
        
        return {
            'repr_weights': repr_weights,
            'clf_weights': clf_weights,
            'prototypes': self.local_prototypes,
            'stats': stats,
            'data_size': self.data_size
        }
    
    def _get_representation_weights(self) -> Dict[str, torch.Tensor]:
        """Extract representation network weights"""
        weights = {}
        for name, param in self.model.representation.named_parameters():
            weights[f"representation.{name}"] = param.cpu().clone()
        return weights
    
    def _get_labelwise_classifier_weights(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Extract label-wise classifier weights"""
        clf_weights = {}
        
        for label in range(self.config.num_classes):
            label_weights = {}
            for name, param in self.model.classifier.classifiers[label].named_parameters():
                label_weights[name] = param.cpu().clone()
            clf_weights[label] = label_weights
        
        return clf_weights
    
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate local model on local data
        
        Returns:
            Tuple of (accuracy, loss)
        """
        self.model.eval()
        
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                logits, _ = self.model(data)
                loss = criterion(logits, target)
                
                total_loss += loss.item() * data.size(0)
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        
        return accuracy, avg_loss
    
    def get_label_distribution(self) -> Dict[int, int]:
        """Get local data label distribution"""
        label_counts = defaultdict(int)
        
        for _, target in self.dataloader:
            for t in target.numpy():
                label_counts[int(t)] += 1
        
        return dict(label_counts)
