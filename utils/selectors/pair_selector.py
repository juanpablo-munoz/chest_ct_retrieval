import numpy as np
import torch
from itertools import combinations
from ..distances import pdist

class PairSelector:
    def get_pairs(self, embeddings, labels):
        raise NotImplementedError

class AllPositivePairSelector(PairSelector):
    def __init__(self, balance=True):
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().numpy()
        all_pairs = torch.LongTensor(list(combinations(range(len(labels)), 2)))
        pos_mask = labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]
        neg_mask = ~pos_mask
        pos_pairs = all_pairs[pos_mask]
        neg_pairs = all_pairs[neg_mask]
        if self.balance:
            neg_pairs = neg_pairs[torch.randperm(len(neg_pairs))[:len(pos_pairs)]]
        return pos_pairs, neg_pairs

class HardNegativePairSelector(PairSelector):
    def __init__(self, cpu=True):
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        D = pdist(embeddings)
        labels = labels.cpu().numpy()
        all_pairs = torch.LongTensor(list(combinations(range(len(labels)), 2)))
        pos_mask = labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]
        neg_mask = ~pos_mask
        pos_pairs = all_pairs[pos_mask]
        neg_pairs = all_pairs[neg_mask]
        dists = D[neg_pairs[:, 0], neg_pairs[:, 1]].cpu().numpy()
        top_neg = np.argpartition(dists, len(pos_pairs))[:len(pos_pairs)]
        return pos_pairs, neg_pairs[torch.LongTensor(top_neg)]
