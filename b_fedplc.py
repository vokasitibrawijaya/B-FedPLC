"""
B-FedPLC: Blockchain-Enabled Federated Learning with 
Prototype-Anchored Learning and Dynamic Community Adaptation

This implementation integrates:
1. FedPLC (PARL + LDCA) - Base federated learning algorithm
2. Blockchain - For model verification, audit trail, and consensus
3. IPFS - For decentralized model storage

Innovation for Doctoral Dissertation (S3)

Author: B-FedPLC Research
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hashlib
import json
import time
import copy
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import threading
import queue

# ============================================================================
# Cryptographic Utilities
# ============================================================================
class CryptoUtils:
    """Cryptographic utilities for blockchain operations"""
    
    @staticmethod
    def hash_data(data: bytes) -> str:
        """Compute SHA-256 hash of data"""
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def hash_model(model_state: dict) -> str:
        """Compute hash of model state dict"""
        # Serialize model weights to bytes
        buffer = []
        for key in sorted(model_state.keys()):
            tensor = model_state[key]
            buffer.append(key.encode())
            buffer.append(tensor.cpu().numpy().tobytes())
        return CryptoUtils.hash_data(b''.join(buffer))
    
    @staticmethod
    def merkle_root(hashes: List[str]) -> str:
        """Compute Merkle root from list of hashes"""
        if not hashes:
            return CryptoUtils.hash_data(b'empty')
        
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])  # Duplicate last if odd
            
            new_hashes = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i+1]
                new_hashes.append(CryptoUtils.hash_data(combined.encode()))
            hashes = new_hashes
        
        return hashes[0]
    
    @staticmethod
    def generate_client_id() -> str:
        """Generate unique client identifier"""
        return hashlib.sha256(
            f"{time.time()}-{np.random.random()}".encode()
        ).hexdigest()[:16]


# ============================================================================
# IPFS Simulator (For local testing - replace with actual IPFS in production)
# ============================================================================
class IPFSSimulator:
    """
    Simulates IPFS for decentralized storage
    In production, replace with actual IPFS client (ipfshttpclient)
    """
    
    def __init__(self, storage_dir: str = "./ipfs_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.index: Dict[str, dict] = {}  # CID -> metadata
        self._lock = threading.Lock()
    
    def add(self, data: bytes, metadata: dict = None) -> str:
        """Add data to IPFS, returns CID (Content Identifier)"""
        cid = f"Qm{CryptoUtils.hash_data(data)[:44]}"  # Simulated IPFS CID
        
        with self._lock:
            # Store data
            filepath = self.storage_dir / f"{cid}.bin"
            with open(filepath, 'wb') as f:
                f.write(data)
            
            # Store metadata
            self.index[cid] = {
                'size': len(data),
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
        
        return cid
    
    def get(self, cid: str) -> Optional[bytes]:
        """Retrieve data from IPFS by CID"""
        filepath = self.storage_dir / f"{cid}.bin"
        if filepath.exists():
            with open(filepath, 'rb') as f:
                return f.read()
        return None
    
    def pin(self, cid: str) -> bool:
        """Pin content to prevent garbage collection"""
        if cid in self.index:
            self.index[cid]['pinned'] = True
            return True
        return False
    
    def get_stats(self) -> dict:
        """Get storage statistics"""
        total_size = sum(info['size'] for info in self.index.values())
        return {
            'num_objects': len(self.index),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024)
        }


# ============================================================================
# Blockchain Data Structures
# ============================================================================
@dataclass
class ModelUpdate:
    """Represents a client model update"""
    client_id: str
    round_number: int
    model_hash: str
    ipfs_cid: str
    accuracy: float
    loss: float
    data_size: int
    community_id: int
    timestamp: float = field(default_factory=time.time)
    signature: str = ""  # Digital signature (simplified)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def compute_hash(self) -> str:
        data = f"{self.client_id}{self.round_number}{self.model_hash}{self.ipfs_cid}"
        return CryptoUtils.hash_data(data.encode())


@dataclass
class Block:
    """Blockchain block containing model updates"""
    index: int
    timestamp: float
    round_number: int
    updates: List[ModelUpdate]
    global_model_hash: str
    global_model_cid: str
    merkle_root: str
    previous_hash: str
    communities: Dict[int, List[str]]  # community_id -> client_ids
    accuracy: float
    nonce: int = 0
    hash: str = ""
    
    def compute_hash(self) -> str:
        """Compute block hash"""
        block_data = (
            f"{self.index}{self.timestamp}{self.round_number}"
            f"{self.global_model_hash}{self.global_model_cid}"
            f"{self.merkle_root}{self.previous_hash}{self.nonce}"
        )
        return CryptoUtils.hash_data(block_data.encode())
    
    def mine(self, difficulty: int = 2):
        """Simple proof-of-work mining (simplified for FL context)"""
        prefix = '0' * difficulty
        while not self.hash.startswith(prefix):
            self.nonce += 1
            self.hash = self.compute_hash()
    
    def to_dict(self) -> dict:
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'round_number': self.round_number,
            'num_updates': len(self.updates),
            'global_model_hash': self.global_model_hash,
            'global_model_cid': self.global_model_cid,
            'merkle_root': self.merkle_root,
            'previous_hash': self.previous_hash,
            'communities': self.communities,
            'accuracy': self.accuracy,
            'nonce': self.nonce,
            'hash': self.hash
        }


# ============================================================================
# Blockchain Implementation
# ============================================================================
class FederatedBlockchain:
    """
    Blockchain for Federated Learning
    
    Features:
    - Immutable audit trail of all model updates
    - Merkle tree for efficient verification
    - IPFS integration for model storage
    - Community tracking (LDCA)
    """
    
    def __init__(self, ipfs: IPFSSimulator, difficulty: int = 2):
        self.chain: List[Block] = []
        self.pending_updates: List[ModelUpdate] = []
        self.ipfs = ipfs
        self.difficulty = difficulty
        self._lock = threading.Lock()
        
        # Create genesis block
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Create the genesis block"""
        genesis = Block(
            index=0,
            timestamp=time.time(),
            round_number=0,
            updates=[],
            global_model_hash="0" * 64,
            global_model_cid="",
            merkle_root="0" * 64,
            previous_hash="0" * 64,
            communities={},
            accuracy=0.0
        )
        genesis.hash = genesis.compute_hash()
        self.chain.append(genesis)
    
    def add_update(self, update: ModelUpdate):
        """Add a model update to pending pool"""
        with self._lock:
            self.pending_updates.append(update)
    
    def create_block(self, round_number: int, global_model_state: dict,
                    communities: Dict[int, List[str]], accuracy: float) -> Block:
        """Create a new block from pending updates"""
        
        # Store global model in IPFS
        model_bytes = self._serialize_model(global_model_state)
        global_cid = self.ipfs.add(model_bytes, {
            'type': 'global_model',
            'round': round_number,
            'accuracy': accuracy
        })
        self.ipfs.pin(global_cid)
        
        # Compute model hash
        global_hash = CryptoUtils.hash_model(global_model_state)
        
        # Compute Merkle root of updates
        update_hashes = [u.compute_hash() for u in self.pending_updates]
        merkle = CryptoUtils.merkle_root(update_hashes) if update_hashes else "0" * 64
        
        # Create block
        block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            round_number=round_number,
            updates=self.pending_updates.copy(),
            global_model_hash=global_hash,
            global_model_cid=global_cid,
            merkle_root=merkle,
            previous_hash=self.chain[-1].hash,
            communities={k: list(v) for k, v in communities.items()},
            accuracy=accuracy
        )
        
        # Mine block (simplified PoW)
        block.mine(self.difficulty)
        
        # Add to chain
        with self._lock:
            self.chain.append(block)
            self.pending_updates = []
        
        return block
    
    def _serialize_model(self, state_dict: dict) -> bytes:
        """Serialize model state dict to bytes"""
        import io
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return buffer.getvalue()
    
    def _deserialize_model(self, data: bytes) -> dict:
        """Deserialize bytes to model state dict"""
        import io
        buffer = io.BytesIO(data)
        return torch.load(buffer, weights_only=True)
    
    def get_model_from_ipfs(self, cid: str) -> Optional[dict]:
        """Retrieve model from IPFS"""
        data = self.ipfs.get(cid)
        if data:
            return self._deserialize_model(data)
        return None
    
    def verify_chain(self) -> Tuple[bool, str]:
        """Verify blockchain integrity"""
        for i in range(1, len(self.chain)):
            block = self.chain[i]
            prev_block = self.chain[i-1]
            
            # Verify hash
            if block.hash != block.compute_hash():
                return False, f"Block {i} hash mismatch"
            
            # Verify chain link
            if block.previous_hash != prev_block.hash:
                return False, f"Block {i} previous hash mismatch"
            
            # Verify Merkle root
            update_hashes = [u.compute_hash() for u in block.updates]
            expected_merkle = CryptoUtils.merkle_root(update_hashes) if update_hashes else "0" * 64
            if block.merkle_root != expected_merkle:
                return False, f"Block {i} Merkle root mismatch"
        
        return True, "Chain valid"
    
    def get_audit_trail(self, client_id: str = None) -> List[dict]:
        """Get audit trail, optionally filtered by client"""
        trail = []
        for block in self.chain[1:]:  # Skip genesis
            block_info = block.to_dict()
            if client_id:
                block_info['updates'] = [
                    u.to_dict() for u in block.updates 
                    if u.client_id == client_id
                ]
            else:
                block_info['updates'] = [u.to_dict() for u in block.updates]
            trail.append(block_info)
        return trail
    
    def get_statistics(self) -> dict:
        """Get blockchain statistics"""
        total_updates = sum(len(b.updates) for b in self.chain)
        return {
            'chain_length': len(self.chain),
            'total_updates': total_updates,
            'latest_round': self.chain[-1].round_number if self.chain else 0,
            'latest_accuracy': self.chain[-1].accuracy if self.chain else 0,
            'ipfs_stats': self.ipfs.get_stats()
        }


# ============================================================================
# Smart Contract Simulator (For Aggregation Rules)
# ============================================================================
class AggregationContract:
    """
    Smart Contract for FL Aggregation Rules
    Enforces aggregation policies and validates updates
    """
    
    def __init__(self, min_updates: int = 3, max_staleness: int = 5):
        self.min_updates = min_updates
        self.max_staleness = max_staleness
        self.reputation_scores: Dict[str, float] = {}
        self.update_history: Dict[str, List[float]] = defaultdict(list)
    
    def validate_update(self, update: ModelUpdate, current_round: int) -> Tuple[bool, str]:
        """Validate a model update"""
        # Check staleness
        if current_round - update.round_number > self.max_staleness:
            return False, "Update too stale"
        
        # Check data size
        if update.data_size < 10:
            return False, "Insufficient data"
        
        return True, "Valid"
    
    def compute_weights(self, updates: List[ModelUpdate]) -> Dict[str, float]:
        """Compute aggregation weights based on reputation and data size"""
        weights = {}
        total_weight = 0
        
        for update in updates:
            # Base weight from data size
            base_weight = update.data_size
            
            # Reputation multiplier
            rep = self.reputation_scores.get(update.client_id, 1.0)
            
            # Combined weight
            weight = base_weight * rep
            weights[update.client_id] = weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def update_reputation(self, client_id: str, contribution_quality: float):
        """Update client reputation based on contribution quality"""
        # EMA of contribution quality
        alpha = 0.3
        current = self.reputation_scores.get(client_id, 1.0)
        self.reputation_scores[client_id] = alpha * contribution_quality + (1 - alpha) * current
        
        # Clamp to [0.1, 2.0]
        self.reputation_scores[client_id] = max(0.1, min(2.0, self.reputation_scores[client_id]))
    
    def get_reputation(self, client_id: str) -> float:
        return self.reputation_scores.get(client_id, 1.0)


# ============================================================================
# B-FedPLC Main Class
# ============================================================================
class BFedPLC:
    """
    B-FedPLC: Blockchain-Enabled FedPLC
    
    Combines:
    - FedPLC (PARL + LDCA)
    - Blockchain for immutable audit trail
    - IPFS for decentralized model storage
    - Smart contracts for aggregation rules
    """
    
    def __init__(self, 
                 num_clients: int,
                 num_classes: int = 10,
                 parl_weight: float = 0.1,
                 similarity_threshold: float = 0.85,
                 warmup_rounds: int = 15,
                 blockchain_difficulty: int = 2):
        
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.parl_weight = parl_weight
        self.similarity_threshold = similarity_threshold
        self.warmup_rounds = warmup_rounds
        
        # Initialize components
        self.ipfs = IPFSSimulator("./b_fedplc_ipfs")
        self.blockchain = FederatedBlockchain(self.ipfs, blockchain_difficulty)
        self.contract = AggregationContract()
        
        # Client management
        self.client_ids = {i: CryptoUtils.generate_client_id() for i in range(num_clients)}
        self.client_prototypes: Dict[str, torch.Tensor] = {}
        self.label_distributions: Dict[str, np.ndarray] = {}
        
        # LDCA communities
        self.communities: Dict[int, List[str]] = {0: list(self.client_ids.values())}
        
        # Metrics
        self.metrics = {
            'rounds': [],
            'accuracy': [],
            'loss': [],
            'communities': [],
            'blockchain_size': [],
            'ipfs_size': []
        }
    
    def update_label_distribution(self, client_idx: int, distribution: np.ndarray):
        """Update client's label distribution for LDCA"""
        client_id = self.client_ids[client_idx]
        self.label_distributions[client_id] = distribution
    
    def compute_communities(self) -> Dict[int, List[str]]:
        """LDCA: Compute communities based on label distribution similarity"""
        if len(self.label_distributions) < 2:
            return self.communities
        
        clients = list(self.label_distributions.keys())
        n = len(clients)
        
        # Compute similarity matrix
        similarity = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_i = self.label_distributions[clients[i]]
                dist_j = self.label_distributions[clients[j]]
                similarity[i, j] = np.dot(dist_i, dist_j) / (
                    np.linalg.norm(dist_i) * np.linalg.norm(dist_j) + 1e-8)
        
        # Hierarchical clustering
        visited = [False] * n
        communities = {}
        comm_id = 0
        
        for i in range(n):
            if visited[i]:
                continue
            
            community = [clients[i]]
            visited[i] = True
            
            for j in range(n):
                if not visited[j] and similarity[i, j] >= self.similarity_threshold:
                    community.append(clients[j])
                    visited[j] = True
            
            communities[comm_id] = community
            comm_id += 1
        
        self.communities = communities
        return communities
    
    def get_community_prototype(self, client_idx: int) -> Optional[torch.Tensor]:
        """Get community prototype for PARL"""
        client_id = self.client_ids[client_idx]
        
        for comm_id, members in self.communities.items():
            if client_id in members:
                member_protos = [
                    self.client_prototypes[m] for m in members 
                    if m in self.client_prototypes and m != client_id
                ]
                if member_protos:
                    return torch.stack(member_protos).mean(dim=0)
        return None
    
    def submit_update(self, client_idx: int, model_state: dict, 
                     accuracy: float, loss: float, data_size: int,
                     round_number: int) -> Tuple[bool, str]:
        """
        Submit model update to blockchain
        
        Returns:
            (success, message)
        """
        client_id = self.client_ids[client_idx]
        
        # Find client's community
        community_id = 0
        for cid, members in self.communities.items():
            if client_id in members:
                community_id = cid
                break
        
        # Store model in IPFS
        model_bytes = self._serialize_model(model_state)
        ipfs_cid = self.ipfs.add(model_bytes, {
            'type': 'client_model',
            'client_id': client_id,
            'round': round_number
        })
        
        # Create update
        update = ModelUpdate(
            client_id=client_id,
            round_number=round_number,
            model_hash=CryptoUtils.hash_model(model_state),
            ipfs_cid=ipfs_cid,
            accuracy=accuracy,
            loss=loss,
            data_size=data_size,
            community_id=community_id
        )
        
        # Validate via smart contract
        valid, msg = self.contract.validate_update(update, round_number)
        if not valid:
            return False, msg
        
        # Add to blockchain pending pool
        self.blockchain.add_update(update)
        
        return True, f"Update accepted: {ipfs_cid[:20]}..."
    
    def finalize_round(self, round_number: int, global_model_state: dict,
                      accuracy: float) -> Block:
        """
        Finalize a training round by creating a new block
        """
        # Create block
        block = self.blockchain.create_block(
            round_number=round_number,
            global_model_state=global_model_state,
            communities=self.communities,
            accuracy=accuracy
        )
        
        # Update metrics
        stats = self.blockchain.get_statistics()
        self.metrics['rounds'].append(round_number)
        self.metrics['accuracy'].append(accuracy)
        self.metrics['communities'].append(len(self.communities))
        self.metrics['blockchain_size'].append(stats['chain_length'])
        self.metrics['ipfs_size'].append(stats['ipfs_stats']['total_size_mb'])
        
        # Update reputations based on contribution
        for update in block.updates:
            # Simple quality metric: accuracy relative to global
            quality = min(2.0, update.accuracy / max(accuracy, 0.1))
            self.contract.update_reputation(update.client_id, quality)
        
        return block
    
    def _serialize_model(self, state_dict: dict) -> bytes:
        """Serialize model state dict"""
        import io
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return buffer.getvalue()
    
    def verify_integrity(self) -> Tuple[bool, str]:
        """Verify blockchain integrity"""
        return self.blockchain.verify_chain()
    
    def get_audit_trail(self, client_idx: int = None) -> List[dict]:
        """Get audit trail"""
        client_id = self.client_ids[client_idx] if client_idx is not None else None
        return self.blockchain.get_audit_trail(client_id)
    
    def get_statistics(self) -> dict:
        """Get B-FedPLC statistics"""
        return {
            'blockchain': self.blockchain.get_statistics(),
            'communities': {
                'count': len(self.communities),
                'sizes': {k: len(v) for k, v in self.communities.items()}
            },
            'reputations': dict(self.contract.reputation_scores),
            'metrics': self.metrics
        }
    
    def export_blockchain(self, filepath: str):
        """Export blockchain to JSON"""
        chain_data = {
            'exported_at': datetime.now().isoformat(),
            'chain': [block.to_dict() for block in self.blockchain.chain],
            'statistics': self.get_statistics()
        }
        with open(filepath, 'w') as f:
            json.dump(chain_data, f, indent=2, default=str)


# ============================================================================
# Export for use in training
# ============================================================================
__all__ = [
    'BFedPLC',
    'FederatedBlockchain',
    'IPFSSimulator',
    'AggregationContract',
    'ModelUpdate',
    'Block',
    'CryptoUtils'
]

if __name__ == "__main__":
    print("="*70)
    print("B-FedPLC: Blockchain-Enabled Federated Learning")
    print("="*70)
    
    # Quick test
    print("\nInitializing B-FedPLC...")
    bfedplc = BFedPLC(num_clients=10)
    
    print(f"✓ IPFS initialized: {bfedplc.ipfs.storage_dir}")
    print(f"✓ Blockchain genesis block created")
    print(f"✓ {len(bfedplc.client_ids)} clients registered")
    
    # Test blockchain
    print("\nTesting blockchain...")
    valid, msg = bfedplc.verify_integrity()
    print(f"✓ Chain integrity: {msg}")
    
    stats = bfedplc.get_statistics()
    print(f"✓ Blockchain stats: {stats['blockchain']}")
    
    print("\n" + "="*70)
    print("B-FedPLC module ready for integration!")
    print("="*70)
