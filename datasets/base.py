from typing import List, Dict, Optional
import numpy as np
from datasets.constants import PROXIMITY_VECTOR_LABELS

class LabelVectorHelper:
    proximity_vector_labels_dict: Dict[int, List[int]] = PROXIMITY_VECTOR_LABELS

    def get_class_id(self, label_vector: List[int]) -> Optional[int]:
        for k, v in self.proximity_vector_labels_dict.items():
            if label_vector == v or (hasattr(label_vector, 'all') and np.array_equal(label_vector, v)):
                return k
        return None

    def build_pair_indices(self, labels: np.ndarray):
        pos, neg = {k: [] for k in PROXIMITY_VECTOR_LABELS}, {k: [] for k in PROXIMITY_VECTOR_LABELS}
        for k, vec in PROXIMITY_VECTOR_LABELS.items():
            for idx, lbl in enumerate(labels):
                if np.array_equal(lbl, vec):
                    pos[k].append(idx)
                else:
                    neg[k].append(idx)
        return pos, neg
