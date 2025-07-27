from typing import List, Dict, Optional
import numpy as np
from datasets.constants import PROXIMITY_VECTOR_LABELS_FOR_TRAINING

class LabelVectorHelper:
    def __init__(self):
        self.proximity_vector_labels_dict: Dict[int, List[int]] = PROXIMITY_VECTOR_LABELS_FOR_TRAINING
    
    def get_label_vector(self, class_id: int) -> List[int]:
        if class_id in self.proximity_vector_labels_dict:
            return self.proximity_vector_labels_dict[class_id]
        else:
            return None

    def get_class_id(self, label_vector: List[int]) -> Optional[int]:
        for k, v in self.proximity_vector_labels_dict.items():
            try:
                if (label_vector == v) or (hasattr(label_vector, 'all') and np.array_equal(label_vector, v)):
                    return k
            except Exception as e:
                print(f"k: {k}, v: {v}, label_vector: {label_vector}\n{e}")
        print(label_vector, 'returned None!')
        return None

    def build_pair_indices(self, labels: np.ndarray):
        pos, neg = {k: [] for k in self.proximity_vector_labels_dict}, {k: [] for k in self.proximity_vector_labels_dict}
        for k, vec in self.proximity_vector_labels_dict.items():
            for idx, lbl in enumerate(labels):
                if lbl == k:
                    pos[k].append(idx)
                else:
                    label_vector = self.proximity_vector_labels_dict[lbl]
                    classes_in_common = np.logical_and(vec, label_vector)
                    no_classes_in_common = not np.array(classes_in_common).any()
                    if no_classes_in_common:
                        # A data sample pair (a, n) is a valid negative pair if their labels have no classes in common
                        # ex. a=[1, 0, 0, 0] and n=[0, 1, 0, 0] is a valid negative pair; a=[0, 1, 1, 0] and n=[0, 0, 1, 1] is not
                        neg[k].append(idx)
        return pos, neg
