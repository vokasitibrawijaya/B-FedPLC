"""
Baseline Federated Learning Algorithms for Comparison
Implements: FedAvg, FedProx, FLASH, FedDrift
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import copy
import numpy as np


class FedAvgClient:
    """
    FedAvg Client - Baseline federated averaging
    McMahan et al., "Communication-Efficient Learning of Deep Networks"
    """
    
    def __init__(self,
                 client_id: int,
                 model: nn.Module,
                 dataloader: DataLoader,
                 device: torch.device,
                 lr: float = 0.01,
                 local_epochs: int = 5):
        
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.dataloader = dataloader
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        self.data_size = len(dataloader.dataset)
    
    def train(self, global_weights: Dict[str, torch.Tensor]) -> Dict:
        """Local training"""
        # Load global weights
        self.model.load_state_dict(global_weights)
        self.model.to(self.device)
        self.model.train()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.local_epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                if hasattr(self.model, 'get_features'):
                    output, _ = self.model(data)
                else:
                    output = self.model(data)
                
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return {
            'weights': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'data_size': self.data_size,
            'loss': total_loss / (self.local_epochs * len(self.dataloader)),
            'accuracy': 100. * correct / total
        }


class FedProxClient(FedAvgClient):
    """
    FedProx Client - Adds proximal term for heterogeneous data
    Li et al., "Federated Optimization in Heterogeneous Networks"
    """
    
    def __init__(self, *args, mu: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu
    
    def train(self, global_weights: Dict[str, torch.Tensor]) -> Dict:
        """Local training with proximal term"""
        # Store global weights for proximal term
        global_params = {k: v.clone() for k, v in global_weights.items()}
        
        # Load global weights
        self.model.load_state_dict(global_weights)
        self.model.to(self.device)
        self.model.train()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.local_epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                if hasattr(self.model, 'get_features'):
                    output, _ = self.model(data)
                else:
                    output = self.model(data)
                
                # Classification loss
                loss = criterion(output, target)
                
                # Proximal term: (mu/2) * ||w - w_global||^2
                proximal_term = 0.0
                for name, param in self.model.named_parameters():
                    if name in global_params:
                        proximal_term += ((param - global_params[name].to(self.device)) ** 2).sum()
                
                loss = loss + (self.mu / 2) * proximal_term
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return {
            'weights': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'data_size': self.data_size,
            'loss': total_loss / (self.local_epochs * len(self.dataloader)),
            'accuracy': 100. * correct / total
        }


class FLASHClient(FedAvgClient):
    """
    FLASH Client - Federated Learning with Adaptive Sample Harvesting
    Uses importance sampling based on loss
    """
    
    def __init__(self, *args, harvest_ratio: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.harvest_ratio = harvest_ratio
    
    def train(self, global_weights: Dict[str, torch.Tensor]) -> Dict:
        """Local training with importance sampling"""
        self.model.load_state_dict(global_weights)
        self.model.to(self.device)
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.local_epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    if hasattr(self.model, 'get_features'):
                        output, _ = self.model(data)
                    else:
                        output = self.model(data)
                    
                    sample_losses = criterion(output, target)
                
                # Select high-loss samples (important samples)
                num_samples = int(len(sample_losses) * self.harvest_ratio)
                _, indices = torch.topk(sample_losses, min(num_samples, len(sample_losses)))
                
                # Train on selected samples
                self.model.train()
                optimizer.zero_grad()
                
                selected_data = data[indices]
                selected_target = target[indices]
                
                if hasattr(self.model, 'get_features'):
                    output, _ = self.model(selected_data)
                else:
                    output = self.model(selected_data)
                
                loss = criterion(output, selected_target).mean()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += selected_target.size(0)
                correct += predicted.eq(selected_target).sum().item()
        
        return {
            'weights': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'data_size': self.data_size,
            'loss': total_loss / (self.local_epochs * len(self.dataloader)),
            'accuracy': 100. * correct / max(total, 1)
        }


class FedDriftClient(FedAvgClient):
    """
    FedDrift Client - Federated Learning with Drift Detection
    Uses local drift detection to trigger model adaptation
    """
    
    def __init__(self, *args, window_size: int = 50, drift_threshold: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.loss_history = []
    
    def detect_drift(self) -> bool:
        """Simple drift detection based on loss increase"""
        if len(self.loss_history) < 2 * self.window_size:
            return False
        
        old_window = np.mean(self.loss_history[-2*self.window_size:-self.window_size])
        new_window = np.mean(self.loss_history[-self.window_size:])
        
        # Drift detected if loss increases significantly
        return (new_window - old_window) / old_window > self.drift_threshold
    
    def train(self, global_weights: Dict[str, torch.Tensor]) -> Dict:
        """Local training with drift detection"""
        self.model.load_state_dict(global_weights)
        self.model.to(self.device)
        self.model.train()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                if hasattr(self.model, 'get_features'):
                    output, _ = self.model(data)
                else:
                    output = self.model(data)
                
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            # Record loss for drift detection
            self.loss_history.append(epoch_loss / len(self.dataloader))
        
        # Detect drift
        drift_detected = self.detect_drift()
        
        return {
            'weights': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'data_size': self.data_size,
            'loss': total_loss / (self.local_epochs * len(self.dataloader)),
            'accuracy': 100. * correct / total,
            'drift_detected': drift_detected
        }


class BaselineServer:
    """
    Generic server for baseline FL algorithms
    """
    
    def __init__(self,
                 model: nn.Module,
                 test_loader: DataLoader,
                 device: torch.device,
                 num_clients: int = 100):
        
        self.global_model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.num_clients = num_clients
        
        self.history = {
            'accuracy': [],
            'loss': []
        }
    
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
    
    def aggregate(self, client_updates: List[Dict]):
        """FedAvg aggregation"""
        total_data = sum(u['data_size'] for u in client_updates)
        
        # Weighted average
        avg_weights = {}
        for key in client_updates[0]['weights'].keys():
            avg_weights[key] = sum(
                u['weights'][key] * (u['data_size'] / total_data)
                for u in client_updates
            )
        
        self.global_model.load_state_dict(avg_weights)
    
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate global model"""
        self.global_model.eval()
        
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if hasattr(self.global_model, 'get_features'):
                    output, _ = self.global_model(data)
                else:
                    output = self.global_model(data)
                
                loss = criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / total
        
        self.history['accuracy'].append(accuracy)
        self.history['loss'].append(avg_loss)
        
        return accuracy, avg_loss
    
    def select_clients(self, num_selected: int) -> List[int]:
        """Random client selection"""
        return np.random.choice(
            self.num_clients,
            size=min(num_selected, self.num_clients),
            replace=False
        ).tolist()


def run_baseline_experiment(algorithm: str,
                           config,
                           client_loaders: Dict[int, DataLoader],
                           test_loader: DataLoader,
                           model: nn.Module) -> Dict:
    """
    Run baseline federated learning experiment
    
    Args:
        algorithm: 'fedavg', 'fedprox', 'flash', or 'feddrift'
        config: ExperimentConfig
        client_loaders: Dictionary of client dataloaders
        test_loader: Test dataloader
        model: Model architecture
    
    Returns:
        Results dictionary
    """
    from tqdm import tqdm
    
    device = torch.device(config.device)
    
    # Initialize server
    server = BaselineServer(
        model=model,
        test_loader=test_loader,
        device=device,
        num_clients=config.num_clients
    )
    
    # Initialize clients
    ClientClass = {
        'fedavg': FedAvgClient,
        'fedprox': FedProxClient,
        'flash': FLASHClient,
        'feddrift': FedDriftClient
    }[algorithm]
    
    clients = {}
    for client_id in range(config.num_clients):
        if algorithm == 'fedprox':
            clients[client_id] = ClientClass(
                client_id=client_id,
                model=model,
                dataloader=client_loaders[client_id],
                device=device,
                lr=config.learning_rate,
                local_epochs=config.local_epochs,
                mu=0.01
            )
        else:
            clients[client_id] = ClientClass(
                client_id=client_id,
                model=model,
                dataloader=client_loaders[client_id],
                device=device,
                lr=config.learning_rate,
                local_epochs=config.local_epochs
            )
    
    # Training loop
    results = {
        'algorithm': algorithm,
        'rounds': [],
        'accuracy': [],
        'loss': []
    }
    
    num_selected = int(config.num_clients * config.participation_rate)
    
    for round_idx in tqdm(range(1, config.num_rounds + 1), desc=f'{algorithm.upper()}'):
        # Select clients
        selected = server.select_clients(num_selected)
        
        # Get global weights
        global_weights = server.get_global_weights()
        
        # Client training
        updates = []
        for client_id in selected:
            update = clients[client_id].train(global_weights)
            updates.append(update)
        
        # Aggregate
        server.aggregate(updates)
        
        # Evaluate
        accuracy, loss = server.evaluate()
        
        results['rounds'].append(round_idx)
        results['accuracy'].append(accuracy)
        results['loss'].append(loss)
        
        if round_idx % 20 == 0:
            print(f"  Round {round_idx}: Acc={accuracy:.2f}%, Loss={loss:.4f}")
    
    results['final_accuracy'] = results['accuracy'][-1]
    results['best_accuracy'] = max(results['accuracy'])
    
    return results
