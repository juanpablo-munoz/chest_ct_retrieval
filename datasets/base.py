from typing import List, Dict, Optional
import numpy as np
from chest_ct_retrieval.datasets.constants import PROXIMITY_VECTOR_LABELS_FOR_TRAINING

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
                    neg[k].append(idx)
        return pos, neg
