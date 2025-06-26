import numpy as np
from math import ceil
import torch
from torch.utils.data import DataLoader


class TripletDataLoader(DataLoader):
    def __init__(self, dataset, n_classes, n_samples, **kwargs):
        self.rng = np.random.default_rng(0)
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_classes * self.n_samples
        self.triplets = []
        self.label_to_indices = dataset.label_to_indices
        self.labels_set = list(set(self.label_to_indices.keys()))
        self.indices_to_labels = {idx: label for label in self.labels_set for idx in self.label_to_indices[label]}
        self.length = -1
        super().__init__(dataset=dataset, batch_size=self.batch_size, **kwargs)

    def __len__(self):
        if self.length < 0:
            return ceil(len(self.dataset) / self.batch_size)
        else:
            return self.length

    def generate_batches_from_triplets(self, triplets):
        # TODO: sort triplets so consecutive triplets' negatives are of the same class
        # hierarchical sort: first by the first elements' classes, then, by the third elements' classes
        batches_list = []
        self.triplets = triplets.cpu().numpy()
        to_be_sampled_mask = np.array([True]*len(self.triplets))
        #print('self.triplets:', self.triplets)
        triplet_idx = 0
        batch_labels_list = []
        batch_buffer = []
        while to_be_sampled_mask.any():
            #to_be_iterated_triplets = self.triplets[to_be_sampled_mask]
            pending_indices = np.where(to_be_sampled_mask)[0]
            starting_triplet = self.triplets[pending_indices[0]]
            starting_anchor, starting_positive, starting_negative = starting_triplet
            #print('to_be_iterated_triplets:',to_be_iterated_triplets)
            #print('starting_triplet:',starting_triplet)
            # assuming valid labels: starting_anchor and starting_positive both correspond to the same label
            # starting_anchor (as well as starting_positive) and starting_negative should belong to different labels
            #anchor_positive_label = self.indices_to_labels[starting_anchor]
            #negative_label = self.indices_to_labels[starting_negative]
            same_anchor_mask = (self.triplets[:, 0] == starting_anchor)
            same_positive_mask = (self.triplets[:, 1] == starting_positive)
            same_anchor_positive_mask = same_anchor_mask & same_positive_mask
            same_anchor_positive_to_be_sampled_mask = to_be_sampled_mask & same_anchor_positive_mask
            #print('same anchor:',same_anchor_mask)
            #print('same positive:',same_positive_mask)
            #print('same anchor & positive:',same_anchor_positive_mask)
            #same_anchor_negative_mask = (to_be_iterated_triplets[:, 0] == starting_anchor) & (to_be_iterated_triplets[:, 2] == starting_negative)
            #batchable_triplets = same_anchor_positive_mask & same_anchor_negative_mask
            #same_anchor_positive_triplets_indices = np.where(same_anchor_positive_mask)[0]
            to_be_sampled_triplets_indices = np.where(same_anchor_positive_to_be_sampled_mask)[0]
            # iterate all triplets with the same anchor-positive
            starting_anchor, starting_positive, starting_negative = starting_triplet
            #print(f'to be iterated for pair {[starting_anchor, starting_positive]}: {to_be_iterated_triplets[same_anchor_positive_triplets_indices]}')
            #for next_triplet_idx in same_anchor_positive_triplets_indices:
            for next_triplet_idx in to_be_sampled_triplets_indices:
                to_be_sampled_mask[next_triplet_idx] = False
                next_anchor, next_positive, next_negative = self.triplets[next_triplet_idx]
                #next_anchor_positive_label = self.indices_to_labels[next_anchor]
                #next_negative_label = self.indices_to_labels[next_negative]
                new_batch_buffer = list(set(batch_buffer + [next_anchor, next_positive, next_negative]))
                if len(new_batch_buffer) <= self.batch_size:
                    batch_buffer = new_batch_buffer
                else:
                    batches_list.append(sorted(list(set(batch_buffer))))
                    batch_buffer = [next_anchor, next_positive, next_negative]
        #optimized_batches_list = self.optimize_batches(batches_list)
        #self.length = len(optimized_batches_list)
        #return optimized_batches_list
        self.length = len(batches_list)
        return batches_list
    
    def optimize_batches(self, batch_list):
        optimized_batch_list = []
        batch_buffer = []
        for batch_indices in batch_list:
            new_batch_buffer = list(set(batch_buffer + batch_indices))
            if len(list(set(new_batch_buffer))) <= self.batch_size:
                batch_buffer = new_batch_buffer
            else:
                optimized_batch_list.append(batch_buffer)
                batch_buffer = batch_indices
        return optimized_batch_list
    

    def load_from_batches(self, batch_list, sample_size=None):
        self.rng.shuffle(batch_list)
        for batch_indices in batch_list[:sample_size]:
            batch = [self.dataset[i] for i in batch_indices]
            batch_samples = torch.tensor(np.array([s for (s, _) in batch]))
            batch_labels = torch.tensor([l for (_, l) in batch])
            yield batch_samples, batch_labels
            