import numpy as np
import torch
from itertools import combinations
from ..distances import query_dataset_dist
from ..compatibility import determine_negative_compatibles
from typing import Callable, Optional, Dict, List

class TripletSelector:
    def get_triplets(self, embeddings, labels):
        raise NotImplementedError

class AllTripletSelector(TripletSelector):
    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().numpy()
        triplets = []
        for label in set(labels):
            pos_idx = np.where(labels == label)[0]
            if len(pos_idx) < 2:
                continue
            neg_idx = np.where(labels != label)[0]
            for a, p in combinations(pos_idx, 2):
                for n in neg_idx:
                    triplets.append([a, p, n])
        return torch.LongTensor(triplets)

def hardest_negative(loss_values): return np.argmax(loss_values) if loss_values.max() > 0 else None
def random_hard_negative(loss_values): 
    mask = np.where(loss_values > 0)[0]
    return np.random.choice(mask) if len(mask) else None
def semihard_negative(loss_values, margin): 
    mask = np.where((loss_values < margin) & (loss_values > 0))[0]
    return np.random.choice(mask) if len(mask) else None

class FunctionNegativeTripletSelector(TripletSelector):
    def __init__(self, margin, negative_selection_fn, cpu=True):
        self.margin = margin
        self.fn = negative_selection_fn
        self.cpu = cpu

    def get_triplets(
        self,
        query_embeddings: torch.Tensor,
        query_labels: torch.Tensor,
        db_embeddings: torch.Tensor,
        db_labels: torch.Tensor,
        negative_compatibles_dict: Dict[int, List[int]],
        print_log: bool = False,
    ) -> torch.LongTensor:

        in_batch = torch.equal(query_embeddings, db_embeddings)

        if self.cpu:
            query_embeddings = query_embeddings.cpu()
            db_embeddings = db_embeddings.cpu()

        distance_matrix = query_dataset_dist(query_embeddings, db_embeddings)
        query_labels = query_labels.cpu().numpy()
        db_labels = db_labels.cpu().numpy()

        triplets = []

        for label in np.unique(query_labels):
            q_mask = query_labels == label
            db_pos_mask = db_labels == label

            q_indices = np.where(q_mask)[0]
            db_pos_indices = np.where(db_pos_mask)[0]

            if in_batch and len(db_pos_indices) <= 1:
                continue
            elif not in_batch and len(db_pos_indices) == 0:
                continue

            neg_labels = negative_compatibles_dict.get(label, [])
            db_neg_mask = np.isin(db_labels, neg_labels)
            db_neg_indices = np.where(db_neg_mask)[0]

            anchor_positives = np.array(
                np.meshgrid(q_indices, db_pos_indices)
            ).T.reshape(-1, 2)

            if in_batch:
                anchor_positives = anchor_positives[
                    anchor_positives[:, 0] != anchor_positives[:, 1]
                ]

            for a_idx, p_idx in anchor_positives:
                ap_dist = distance_matrix[a_idx, p_idx].item()
                an_dists = distance_matrix[a_idx, db_neg_indices]
                loss_vals = ap_dist - an_dists + self.margin
                loss_vals = loss_vals.numpy()

                n_sel = self.negative_selection_fn(loss_vals)
                if n_sel is not None:
                    n_idx = db_neg_indices[n_sel]
                    triplets.append([a_idx, p_idx, n_idx])
                else:
                    fallback = random_hard_negative(loss_vals)
                    if fallback is not None:
                        triplets.append([a_idx, p_idx, db_neg_indices[fallback]])

        if not triplets and db_neg_indices.size > 0:
            # fallback case (return one default triplet)
            a_idx, p_idx = anchor_positives[0]
            n_idx = db_neg_indices[0]
            triplets.append([a_idx, p_idx, n_idx])

        return torch.LongTensor(triplets)

def HardestNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin, hardest_negative, cpu)

def RandomNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin, random_hard_negative, cpu)

def SemihardNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin, lambda x: semihard_negative(x, margin), cpu)
