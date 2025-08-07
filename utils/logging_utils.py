import logging
import torch
import numpy as np
from typing import Any, Optional

class TripletLogger:
    """Enhanced logging utility for triplet training with configurable detail levels"""
    
    def __init__(self, logger_name: str = "triplet_training", log_level: int = logging.INFO):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_batch_info(self, batch_idx: int, total_batches: int, labels: torch.Tensor):
        """Log basic batch information"""
        self.logger.info(f"Batch {batch_idx}/{total_batches} - Labels: {labels.tolist() if len(labels) <= 10 else f'{labels[:5].tolist()}...({len(labels)} total)'}")
    
    def log_embeddings_debug(self, embeddings: torch.Tensor, labels: torch.Tensor, prefix: str = ""):
        """Log detailed embedding information (use sparingly - debug level only)"""
        if self.logger.isEnabledFor(logging.DEBUG):
            k = min(5, len(labels))
            self.logger.debug(f"{prefix}Embeddings shape: {embeddings.shape}")
            self.logger.debug(f"{prefix}Labels[:{k}]: {labels[:k].tolist()}")
            self.logger.debug(f"{prefix}Embedding stats - mean: {embeddings.mean().item():.4f}, std: {embeddings.std().item():.4f}")
    
    def log_triplet_stats(self, triplets: torch.Tensor, losses: torch.Tensor, margin: float):
        """Log triplet mining and loss statistics"""
        if len(triplets) == 0:
            self.logger.warning("No valid triplets found!")
            return
            
        num_valid = (losses > 0).sum().item()
        self.logger.info(f"Triplets: {len(triplets)} total, {num_valid} non-zero loss")
        
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Loss stats: mean={losses.mean().item():.6f}, max={losses.max().item():.6f}, margin={margin}")
    
    def log_distances_debug(self, ap_distances: torch.Tensor, an_distances: torch.Tensor, k: int = 5):
        """Log anchor-positive and anchor-negative distance information"""
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"AP distances[:{k}]: {ap_distances[:k].tolist()}")
            self.logger.debug(f"AN distances[:{k}]: {an_distances[:k].tolist()}")
            self.logger.debug(f"Distance ratio (AP/AN): {(ap_distances.mean() / an_distances.mean()).item():.4f}")

# Usage example:
# triplet_logger = TripletLogger("triplet_loss", logging.DEBUG)
# triplet_logger.log_batch_info(batch_idx, len(dataloader), target)
# triplet_logger.log_triplet_stats(triplets, losses, self.margin)