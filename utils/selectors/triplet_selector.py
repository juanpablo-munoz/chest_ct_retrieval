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
    if len(mask):
        return np.random.choice(mask)
    else:
        return random_hard_negative(loss_values)

class FunctionNegativeTripletSelector(TripletSelector):
    def __init__(self, margin, negative_selection_fn, cpu=True):
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn
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

        if print_log:
            print('FunctionNegativeTripletSelector.get_triplets()')
            k = min(10, len(distance_matrix))
            if k < len(distance_matrix):
                print(f'(Distance matrix has length {len(distance_matrix)} which is too long! Printing only its first {k} elements.)')
            print(f'distance_matrix[:{k}]:')
            print(distance_matrix[:k])

        query_labels = query_labels.cpu().numpy()
        db_labels = db_labels.cpu().numpy()

        triplets = []

        for label in np.unique(query_labels):
            q_mask = query_labels == label
            db_pos_mask = db_labels == label

            q_indices = np.where(q_mask)[0]
            db_pos_indices = np.where(db_pos_mask)[0]

            # if triplets are being mined from the batch itself (query and database are the same)
            # then omit labels for which there's only one sample
            if in_batch and len(db_pos_indices) <= 1:
                continue

            # else, if the triplets are being mined from a query on a database (query and database are different)
            # then omit query samples for which there are no same-label samples in the database
            elif not in_batch and len(db_pos_indices) == 0:
                continue

            neg_labels = negative_compatibles_dict[label]
            db_neg_mask = np.isin(db_labels, neg_labels)
            db_neg_indices = np.where(db_neg_mask)[0]

            # All anchor-positive pairs
            anchor_positives = np.array(
                np.meshgrid(q_indices, db_pos_indices)
            ).T.reshape(-1, 2)

            # if triplets are being mined from the batch itself (query and database are the same)
            # then prune anchor_positives pairs in which the anchor and positive are both the same object
            if in_batch:
                anchor_positives = anchor_positives[
                    anchor_positives[:, 0] != anchor_positives[:, 1]
                ]
            
            anchor_negatives = np.array(np.meshgrid(q_indices, db_pos_indices)).T.reshape(-1, 2)  # All anchor-negative pairs

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            an_distances = distance_matrix[anchor_negatives[:, 0], anchor_negatives[:, 1]]
            if print_log:
                k = min(10, len(ap_distances))
                if k < len(ap_distances):
                    print(f'(Pair and distance lists are of length {len(ap_distances)} which is too long! Printing only the first {k} elements of each.)')
                print('query_label:', label)
                print(f'anchor_positives[:{k}] [query_index, db_index]:')
                print(anchor_positives[:k])
                print(f'ap_distances[:{k}]:')
                print(ap_distances[:k])
                print(f'anchor_negatives[:{k}] [query_index, db_index]:')
                print(anchor_negatives[:k])
                print(f'an_distances[:{k}]:')
                print(an_distances[:k])


            for a_idx, p_idx in anchor_positives:
                # Avoid including malformed anchor-positive pairs if there's any
                # This case may occur only when mining triplets from the batch itself (query and db are the same)
                if in_batch and a_idx == p_idx:
                    # skip pairs in which the anchor and positive are both the same object
                    print(f'Skipping same-sample positive pair: ({a_idx}, {p_idx})')
                    continue
                ap_dist = distance_matrix[a_idx, p_idx].item()
                an_dists = distance_matrix[a_idx, db_neg_indices]
                loss_vals = ap_dist - an_dists + self.margin
                loss_vals = loss_vals.cpu().detach().numpy()
                
                if print_log:
                    print(f'for anchor_positive=({a_idx}, {p_idx}) with distance={round(ap_dist, 4)}: loss_values={loss_vals}')
                
                n_sel = self.negative_selection_fn(loss_vals)
                if n_sel is not None:
                    n_idx = db_neg_indices[n_sel]
                    if print_log:
                        print(f'Semi-hard negative for anchor_positive ({a_idx}, {p_idx}) is: {n_idx}')
                    triplets.append([a_idx, p_idx, n_idx])
                else:
                    fallback = random_hard_negative(loss_vals)
                    if fallback is not None:
                        if print_log:
                            print(f'Random hard negative for anchor_positive ({a_idx}, {p_idx}) is: {n_idx}')
                        triplets.append([a_idx, p_idx, db_neg_indices[fallback]])

        if not triplets and db_neg_indices.size > 0:
            # fallback case (return one default triplet)
            n_idx = db_neg_indices[0]
            a_idx, p_idx = anchor_positives[0]
            if print_log:
                print(f'(fallback) negative index in batch is: {n_idx}')
                print(f'(fallback) triplet to be returned: {[a_idx, p_idx, n_idx]}')
            triplets.append([a_idx, p_idx, n_idx])

        return torch.LongTensor(triplets)

def HardestNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin, hardest_negative, cpu)

def RandomNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin, random_hard_negative, cpu)

def SemihardNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin, lambda x: semihard_negative(x, margin), cpu)
