"""
Model architectures for FedPLC
Implements decoupled architecture with:
- Shared Representation Layer (φ): Feature extractor
- Label-wise Classifier Head (ψ): Personalized classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional


class BasicBlock(nn.Module):
    """Basic residual block for ResNet"""
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RepresentationNetwork(nn.Module):
    """
    Shared Representation Network (φ)
    Feature extractor based on ResNet-like architecture
    """
    
    def __init__(self, 
                 in_channels: int = 3,
                 hidden_dim: int = 512,
                 num_blocks: List[int] = [2, 2, 2, 2]):
        super().__init__()
        
        self.in_channels = 64
        self.hidden_dim = hidden_dim
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        # Adaptive pooling and projection
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, hidden_dim)
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class SimpleCNN(nn.Module):
    """
    Simple CNN for Fashion-MNIST (single channel input)
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class LabelWiseClassifier(nn.Module):
    """
    Label-wise Classifier Head (ψ)
    Separate classifier for each label, allowing label-level community aggregation
    """
    
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Create separate classifier for each class
        # Each classifier outputs a single logit for its class
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(num_classes)
        ])
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, hidden_dim) feature embeddings
        
        Returns:
            logits: (batch_size, num_classes) classification logits
        """
        logits = []
        for classifier in self.classifiers:
            logits.append(classifier(features))
        
        return torch.cat(logits, dim=1)
    
    def get_label_classifier(self, label: int) -> nn.Module:
        """Get classifier head for a specific label"""
        return self.classifiers[label]
    
    def get_label_weights(self, label: int) -> Dict[str, torch.Tensor]:
        """Get weights for a specific label's classifier"""
        return {
            name: param.clone()
            for name, param in self.classifiers[label].named_parameters()
        }
    
    def set_label_weights(self, label: int, weights: Dict[str, torch.Tensor]):
        """Set weights for a specific label's classifier"""
        state_dict = self.classifiers[label].state_dict()
        for name, param in weights.items():
            if name in state_dict:
                state_dict[name] = param
        self.classifiers[label].load_state_dict(state_dict)


class SimpleClassifier(nn.Module):
    """
    Simple linear classifier head (alternative to label-wise)
    """
    
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class FedPLCModel(nn.Module):
    """
    Complete FedPLC model combining representation network and classifier
    Supports decoupled training and prototype-anchored learning
    """
    
    def __init__(self,
                 dataset: str = "cifar10",
                 hidden_dim: int = 512,
                 num_classes: int = 10,
                 use_labelwise_classifier: bool = True):
        super().__init__()
        
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_labelwise_classifier = use_labelwise_classifier
        
        # Initialize representation network based on dataset
        if dataset in ["cifar10", "svhn"]:
            self.representation = RepresentationNetwork(
                in_channels=3,
                hidden_dim=hidden_dim
            )
        elif dataset == "fmnist":
            self.representation = SimpleCNN(hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Initialize classifier
        if use_labelwise_classifier:
            self.classifier = LabelWiseClassifier(hidden_dim, num_classes)
        else:
            self.classifier = SimpleClassifier(hidden_dim, num_classes)
        
        # Class prototypes for PARL
        self.register_buffer(
            'prototypes',
            torch.zeros(num_classes, hidden_dim)
        )
        self.prototype_counts = torch.zeros(num_classes)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
        
        Returns:
            Tuple of (logits, features)
        """
        features = self.representation(x)
        logits = self.classifier(features)
        return logits, features
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification"""
        return self.representation(x)
    
    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Classify from features"""
        return self.classifier(features)
    
    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor, momentum: float = 0.9):
        """
        Update class prototypes using exponential moving average
        
        Args:
            features: (batch_size, hidden_dim) feature embeddings
            labels: (batch_size,) class labels
            momentum: EMA momentum
        """
        with torch.no_grad():
            for c in range(self.num_classes):
                mask = labels == c
                if mask.sum() > 0:
                    class_features = features[mask].mean(dim=0)
                    
                    if self.prototype_counts[c] == 0:
                        self.prototypes[c] = class_features
                    else:
                        self.prototypes[c] = (
                            momentum * self.prototypes[c] +
                            (1 - momentum) * class_features
                        )
                    
                    self.prototype_counts[c] += mask.sum().item()
    
    def get_representation_params(self) -> Dict[str, torch.Tensor]:
        """Get parameters of representation network"""
        return {
            f"representation.{name}": param.clone()
            for name, param in self.representation.named_parameters()
        }
    
    def get_classifier_params(self) -> Dict[str, torch.Tensor]:
        """Get parameters of classifier"""
        return {
            f"classifier.{name}": param.clone()
            for name, param in self.classifier.named_parameters()
        }
    
    def get_label_classifier_params(self, label: int) -> Dict[str, torch.Tensor]:
        """Get parameters of classifier for a specific label"""
        if not self.use_labelwise_classifier:
            raise ValueError("Label-wise classifier not enabled")
        return self.classifier.get_label_weights(label)
    
    def set_label_classifier_params(self, label: int, weights: Dict[str, torch.Tensor]):
        """Set parameters of classifier for a specific label"""
        if not self.use_labelwise_classifier:
            raise ValueError("Label-wise classifier not enabled")
        self.classifier.set_label_weights(label, weights)
    
    def load_representation(self, state_dict: Dict[str, torch.Tensor]):
        """Load representation network weights"""
        filtered = {
            k.replace("representation.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("representation.")
        }
        self.representation.load_state_dict(filtered, strict=False)
    
    def load_classifier(self, state_dict: Dict[str, torch.Tensor]):
        """Load classifier weights"""
        filtered = {
            k.replace("classifier.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("classifier.")
        }
        self.classifier.load_state_dict(filtered, strict=False)


def create_model(dataset: str = "cifar10",
                 hidden_dim: int = 512,
                 num_classes: int = 10,
                 use_labelwise: bool = True) -> FedPLCModel:
    """
    Factory function to create FedPLC model
    
    Args:
        dataset: Dataset name ("cifar10", "fmnist", "svhn")
        hidden_dim: Hidden dimension for features
        num_classes: Number of output classes
        use_labelwise: Whether to use label-wise classifier
    
    Returns:
        FedPLCModel instance
    """
    model = FedPLCModel(
        dataset=dataset,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        use_labelwise_classifier=use_labelwise
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    repr_params = sum(p.numel() for p in model.representation.parameters())
    clf_params = sum(p.numel() for p in model.classifier.parameters())
    
    print(f"Model created for {dataset}")
    print(f"Total parameters: {total_params:,}")
    print(f"Representation parameters: {repr_params:,}")
    print(f"Classifier parameters: {clf_params:,}")
    
    return model
