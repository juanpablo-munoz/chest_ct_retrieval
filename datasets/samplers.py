import numpy as np
from torch.utils.data.sampler import BatchSampler


class BalancedBatchSampler(BatchSampler):
    """
    Samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, vector_labels_dict, n_classes, n_samples, multilabel=False):
        self.rng = np.random.default_rng(seed=0)
        self.vector_labels_dict = vector_labels_dict
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.indices_to_labels = {idx: label for label in self.labels_set for idx in self.label_to_indices[label]}
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes
        self.multilabel = multilabel
        if self.multilabel:
            self.n_classes = 2 # multilabel compatible with 2-class sampling at the moment
            self.compatible_labels = dict()
            for current_label_id in self.labels_set:
                current_label_vector = self.vector_labels_dict[current_label_id]
                compatible_labels = []
                for candidate_label_id, candidate_label_vector in self.vector_labels_dict.items():
                    if current_label_id == candidate_label_id:
                        continue
                    same_class_detection = np.array([x and y for x, y in zip(current_label_vector, candidate_label_vector)])
                    # same_class_detection: binary array. Value at index i determines whether current_label and candidate_label both indicate "positive" for the i-th class
                    any_same_class_detection = same_class_detection.any()
                    detect_only_different_classes = not any_same_class_detection
                    if detect_only_different_classes:
                        # if current_label_vector and candidate_label_vector correpsond to the detection of sets of mutually exclusive classes, then they are "negative-compatible"
                        compatible_labels.append(candidate_label_id)
                self.compatible_labels[current_label_id] = compatible_labels
            #print('self.compatible_labels:', self.compatible_labels)
            self.valid_label_pairs = []
            for k, v in self.compatible_labels.items():
                self.valid_label_pairs.extend([k, l] for l in v if l > k)
            #print('self.labels_set:', self.labels_set)
            #print('self.label_to_indices:', self.label_to_indices)
            #print('self.valid_label_pairs:', self.valid_label_pairs)

    def __len__(self):
        #return self.n_dataset // self.batch_size
        size = sum([min(len(self.label_to_indices[left_hand]), len(self.label_to_indices[right_hand]))//self.n_samples for left_hand, right_hand in self.valid_label_pairs])
        #size = sum([max(len(self.label_to_indices[left_hand]), len(self.label_to_indices[right_hand]))//self.n_samples for left_hand, right_hand in self.valid_label_pairs])
        return size
     

    def __iter__(self):
        self.count = 0
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        for class_ in self.label_to_indices:
            self.rng.shuffle(self.label_to_indices[class_])
        self.rng.shuffle(self.valid_label_pairs)
        #while self.count + self.batch_size < self.n_dataset:
        for class_pair in self.valid_label_pairs:
            # if pair_exhausted_left_hand and pair_exhausted_right_hand -> need to continue onto the next label pair
            pair_exhausted_left_hand = False
            pair_exhausted_right_hand = False
            if self.multilabel:
                #label_pair_index = self.rng.choice(len(self.valid_label_pairs), 1, replace=False)
                #classes = self.rng.choice(self.valid_label_pairs)
                classes = class_pair
                classes = self.rng.choice(classes, len(classes), replace=False)
            else:
                classes = self.rng.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            while not pair_exhausted_left_hand and not pair_exhausted_right_hand: # samples pairs until all samples are yielded for any left and right-hand classes, undersampling on the most represented class if necessary
            #while not pair_exhausted_left_hand or not pair_exhausted_right_hand: # samples pairs until all samples are yielded for both left and right-hand classes, oversampling on the least represented class if necessary 
                class_left_hand, class_right_hand = classes
                indices.extend(
                    self.label_to_indices[class_left_hand][
                        self.used_label_indices_count[class_left_hand]:
                        self.used_label_indices_count[class_left_hand] + self.n_samples
                    ]
                )
                indices.extend(
                    self.label_to_indices[class_right_hand][
                        self.used_label_indices_count[class_right_hand]:
                        self.used_label_indices_count[class_right_hand] + self.n_samples
                    ]
                )
                self.used_label_indices_count[class_left_hand] += self.n_samples
                if self.used_label_indices_count[class_left_hand] + self.n_samples > len(self.label_to_indices[class_left_hand]):
                    self.rng.shuffle(self.label_to_indices[class_left_hand])
                    self.used_label_indices_count[class_left_hand] = 0
                    pair_exhausted_left_hand = True
                self.used_label_indices_count[class_right_hand] += self.n_samples
                if self.used_label_indices_count[class_right_hand] + self.n_samples > len(self.label_to_indices[class_right_hand]):
                    self.rng.shuffle(self.label_to_indices[class_right_hand])
                    self.used_label_indices_count[class_right_hand] = 0
                    pair_exhausted_right_hand = True
                yield indices
                self.count += len(indices)
                indices = []
