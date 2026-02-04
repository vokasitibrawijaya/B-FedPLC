"""
LDCA: Label-wise Dynamic Community Adaptation
Implements community detection using Louvain algorithm for FedPLC

Key components:
1. Build similarity graph based on classifier head similarity
2. Detect communities using Louvain algorithm
3. Label-wise aggregation within communities
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import networkx as nx
from community import community_louvain  # python-louvain package


class SimilarityGraph:
    """
    Builds and maintains similarity graph between clients
    for each label based on classifier head similarity
    """
    
    def __init__(self, 
                 num_clients: int,
                 num_classes: int,
                 threshold: float = 0.85):
        """
        Args:
            num_clients: Number of clients
            num_classes: Number of classes/labels
            threshold: Similarity threshold τ for edge creation
        """
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.threshold = threshold
        
        # Create separate graph for each label
        self.graphs: Dict[int, nx.Graph] = {
            c: nx.Graph() for c in range(num_classes)
        }
        
        # Initialize nodes
        for c in range(num_classes):
            self.graphs[c].add_nodes_from(range(num_clients))
    
    def compute_similarity(self,
                           w1: Dict[str, torch.Tensor],
                           w2: Dict[str, torch.Tensor]) -> float:
        """
        Compute cosine similarity between two sets of weights
        """
        # Flatten and concatenate all parameters
        vec1 = torch.cat([v.flatten().float() for v in w1.values()])
        vec2 = torch.cat([v.flatten().float() for v in w2.values()])
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            vec1.unsqueeze(0), vec2.unsqueeze(0)
        ).item()
        
        return similarity
    
    def update_graph(self,
                     label: int,
                     client_weights: Dict[int, Dict[str, torch.Tensor]]):
        """
        Update similarity graph for a specific label
        
        Args:
            label: Class label
            client_weights: Dictionary mapping client_id -> classifier weights for this label
        """
        # Clear existing edges
        self.graphs[label].clear_edges()
        
        client_ids = list(client_weights.keys())
        
        # Compute pairwise similarities
        for i, client_i in enumerate(client_ids):
            for client_j in client_ids[i+1:]:
                sim = self.compute_similarity(
                    client_weights[client_i],
                    client_weights[client_j]
                )
                
                # Add edge if similarity exceeds threshold
                if sim >= self.threshold:
                    self.graphs[label].add_edge(client_i, client_j, weight=sim)
    
    def update_all_graphs(self,
                          all_client_weights: Dict[int, Dict[int, Dict[str, torch.Tensor]]]):
        """
        Update similarity graphs for all labels
        
        Args:
            all_client_weights: Dict[client_id][label] -> weights
        """
        for label in range(self.num_classes):
            # Extract weights for this label from all clients
            label_weights = {}
            for client_id, label_dict in all_client_weights.items():
                if label in label_dict:
                    label_weights[client_id] = label_dict[label]
            
            if len(label_weights) > 0:
                self.update_graph(label, label_weights)
    
    def get_graph(self, label: int) -> nx.Graph:
        """Get graph for a specific label"""
        return self.graphs[label]
    
    def get_adjacency_matrix(self, label: int) -> np.ndarray:
        """Get adjacency matrix for a label's graph"""
        return nx.to_numpy_array(self.graphs[label])


class LouvainCommunityDetector:
    """
    Detects communities in similarity graphs using Louvain algorithm
    """
    
    def __init__(self, resolution: float = 1.0):
        """
        Args:
            resolution: Resolution parameter for Louvain algorithm
                       Higher = more communities, Lower = fewer communities
        """
        self.resolution = resolution
    
    def detect_communities(self, graph: nx.Graph) -> Dict[int, int]:
        """
        Detect communities in a graph
        
        Args:
            graph: NetworkX graph
        
        Returns:
            Dictionary mapping node_id -> community_id
        """
        if graph.number_of_edges() == 0:
            # If no edges, each node is its own community
            return {node: node for node in graph.nodes()}
        
        # Apply Louvain algorithm
        communities = community_louvain.best_partition(
            graph,
            resolution=self.resolution,
            random_state=42
        )
        
        return communities
    
    def get_communities_list(self, 
                             partition: Dict[int, int]) -> Dict[int, List[int]]:
        """
        Convert partition to list of communities
        
        Args:
            partition: Node -> community mapping
        
        Returns:
            Community -> list of nodes mapping
        """
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        return dict(communities)


class LDCAManager:
    """
    Label-wise Dynamic Community Adaptation Manager
    Orchestrates community detection and label-wise aggregation
    """
    
    def __init__(self,
                 num_clients: int,
                 num_classes: int,
                 threshold: float = 0.85,
                 resolution: float = 1.0):
        """
        Args:
            num_clients: Number of clients
            num_classes: Number of classes
            threshold: Similarity threshold for edge creation
            resolution: Louvain resolution parameter
        """
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.threshold = threshold
        
        # Initialize components
        self.similarity_graph = SimilarityGraph(
            num_clients, num_classes, threshold
        )
        self.community_detector = LouvainCommunityDetector(resolution)
        
        # Store current communities for each label
        self.label_communities: Dict[int, Dict[int, List[int]]] = {
            c: {0: list(range(num_clients))}  # Initially all clients in one community
            for c in range(num_classes)
        }
        
        # Store client -> community mapping
        self.client_communities: Dict[int, Dict[int, int]] = {
            client_id: {c: 0 for c in range(num_classes)}
            for client_id in range(num_clients)
        }
    
    def update_communities(self,
                           client_classifier_weights: Dict[int, Dict[int, Dict[str, torch.Tensor]]]):
        """
        Update communities based on new classifier weights
        
        Args:
            client_classifier_weights: Dict[client_id][label] -> weights for label classifier
        """
        # Update similarity graphs
        self.similarity_graph.update_all_graphs(client_classifier_weights)
        
        # Detect communities for each label
        for label in range(self.num_classes):
            graph = self.similarity_graph.get_graph(label)
            partition = self.community_detector.detect_communities(graph)
            
            # Update community structures
            self.label_communities[label] = self.community_detector.get_communities_list(partition)
            
            # Update client -> community mapping
            for client_id, comm_id in partition.items():
                self.client_communities[client_id][label] = comm_id
    
    def get_community_members(self, label: int, community_id: int) -> List[int]:
        """Get list of client IDs in a community for a label"""
        return self.label_communities[label].get(community_id, [])
    
    def get_client_community(self, client_id: int, label: int) -> int:
        """Get community ID for a client-label pair"""
        return self.client_communities[client_id][label]
    
    def get_same_community_clients(self, client_id: int, label: int) -> List[int]:
        """Get all clients in the same community for a label"""
        comm_id = self.get_client_community(client_id, label)
        return self.get_community_members(label, comm_id)
    
    def aggregate_label_wise(self,
                             client_weights: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
                             data_sizes: Optional[Dict[int, Dict[int, int]]] = None
                             ) -> Dict[int, Dict[int, Dict[str, torch.Tensor]]]:
        """
        Perform label-wise community aggregation
        
        Args:
            client_weights: Dict[client_id][label] -> weights
            data_sizes: Optional Dict[client_id][label] -> number of samples
        
        Returns:
            Dict[client_id][label] -> aggregated weights
        """
        aggregated = defaultdict(dict)
        
        for label in range(self.num_classes):
            # Get communities for this label
            communities = self.label_communities[label]
            
            for comm_id, members in communities.items():
                if len(members) == 0:
                    continue
                
                # Collect weights from community members
                member_weights = []
                member_coefficients = []
                
                for client_id in members:
                    if client_id in client_weights and label in client_weights[client_id]:
                        member_weights.append(client_weights[client_id][label])
                        
                        # Use data size as weight if provided
                        if data_sizes and client_id in data_sizes and label in data_sizes[client_id]:
                            member_coefficients.append(data_sizes[client_id][label])
                        else:
                            member_coefficients.append(1.0)
                
                if len(member_weights) == 0:
                    continue
                
                # Normalize coefficients
                total = sum(member_coefficients)
                member_coefficients = [c / total for c in member_coefficients]
                
                # Average weights
                avg_weights = self._average_weights(member_weights, member_coefficients)
                
                # Assign to all community members
                for client_id in members:
                    aggregated[client_id][label] = avg_weights
        
        return dict(aggregated)
    
    def _average_weights(self,
                         weights_list: List[Dict[str, torch.Tensor]],
                         coefficients: List[float]) -> Dict[str, torch.Tensor]:
        """Average a list of weight dictionaries"""
        if len(weights_list) == 0:
            return {}
        
        avg = {}
        for key in weights_list[0].keys():
            avg[key] = sum(
                coef * w[key].float()
                for w, coef in zip(weights_list, coefficients)
            )
        
        return avg
    
    def get_modularity(self, label: int) -> float:
        """
        Compute modularity of current partition for a label
        
        Modularity Q = (1/2m) Σ [A_ij - k_i*k_j/2m] δ(c_i, c_j)
        """
        graph = self.similarity_graph.get_graph(label)
        
        if graph.number_of_edges() == 0:
            return 0.0
        
        partition = {
            node: self.client_communities[node][label]
            for node in graph.nodes()
        }
        
        return community_louvain.modularity(partition, graph)
    
    def get_statistics(self) -> Dict:
        """Get community detection statistics"""
        stats = {}
        
        for label in range(self.num_classes):
            communities = self.label_communities[label]
            num_communities = len(communities)
            community_sizes = [len(members) for members in communities.values()]
            
            stats[f'label_{label}'] = {
                'num_communities': num_communities,
                'modularity': self.get_modularity(label),
                'avg_community_size': np.mean(community_sizes) if community_sizes else 0,
                'min_community_size': min(community_sizes) if community_sizes else 0,
                'max_community_size': max(community_sizes) if community_sizes else 0
            }
        
        return stats
    
    def print_summary(self):
        """Print summary of current communities"""
        print("\n" + "="*60)
        print("LDCA Community Summary")
        print("="*60)
        
        for label in range(self.num_classes):
            communities = self.label_communities[label]
            modularity = self.get_modularity(label)
            
            print(f"\nLabel {label}: {len(communities)} communities (Q={modularity:.4f})")
            for comm_id, members in communities.items():
                if len(members) <= 10:
                    print(f"  Community {comm_id}: {members}")
                else:
                    print(f"  Community {comm_id}: {len(members)} members")


class GlobalAggregator:
    """
    Global aggregation for representation layer
    Combines representations from all clients
    """
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
    
    def aggregate_representations(self,
                                  client_repr_weights: Dict[int, Dict[str, torch.Tensor]],
                                  data_sizes: Optional[Dict[int, int]] = None
                                  ) -> Dict[str, torch.Tensor]:
        """
        Aggregate representation layer weights from all clients
        
        Args:
            client_repr_weights: Dict[client_id] -> representation weights
            data_sizes: Optional Dict[client_id] -> local data size
        
        Returns:
            Aggregated representation weights
        """
        if len(client_repr_weights) == 0:
            return {}
        
        # Compute aggregation coefficients
        if data_sizes:
            total_data = sum(data_sizes.get(c, 0) for c in client_repr_weights.keys())
            coefficients = {
                c: data_sizes.get(c, 0) / total_data
                for c in client_repr_weights.keys()
            }
        else:
            coefficients = {
                c: 1.0 / len(client_repr_weights)
                for c in client_repr_weights.keys()
            }
        
        # Initialize aggregated weights
        first_client = list(client_repr_weights.keys())[0]
        aggregated = {
            key: torch.zeros_like(val, dtype=torch.float32)
            for key, val in client_repr_weights[first_client].items()
        }
        
        # Weighted average
        for client_id, weights in client_repr_weights.items():
            coef = coefficients[client_id]
            for key, val in weights.items():
                aggregated[key] += coef * val.float()
        
        return aggregated
    
    def aggregate_prototypes(self,
                             client_prototypes: Dict[int, torch.Tensor],
                             prototype_counts: Optional[Dict[int, torch.Tensor]] = None
                             ) -> torch.Tensor:
        """
        Aggregate prototypes from all clients
        
        Args:
            client_prototypes: Dict[client_id] -> (num_classes, hidden_dim) prototypes
            prototype_counts: Optional Dict[client_id] -> per-class sample counts
        
        Returns:
            Aggregated global prototypes
        """
        if len(client_prototypes) == 0:
            return None
        
        first_client = list(client_prototypes.keys())[0]
        num_classes, hidden_dim = client_prototypes[first_client].shape
        
        # Initialize
        aggregated = torch.zeros(num_classes, hidden_dim)
        counts = torch.zeros(num_classes)
        
        # Weighted sum (weighted by sample counts if available)
        for client_id, prototypes in client_prototypes.items():
            if prototype_counts and client_id in prototype_counts:
                client_counts = prototype_counts[client_id]
            else:
                client_counts = torch.ones(num_classes)
            
            for c in range(num_classes):
                if client_counts[c] > 0:
                    aggregated[c] += prototypes[c] * client_counts[c]
                    counts[c] += client_counts[c]
        
        # Normalize
        for c in range(num_classes):
            if counts[c] > 0:
                aggregated[c] /= counts[c]
        
        return aggregated
