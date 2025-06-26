import numpy as np

def determine_negative_compatibles(vector_labels_dict):
    compatible_labels = dict()
    for current_label_id, current_label_vector in vector_labels_dict.items():
        compatible_labels_list = []
        for candidate_label_id, candidate_label_vector in vector_labels_dict.items():
            if current_label_id == candidate_label_id:
                continue
            same_class = np.array([x and y for x, y in zip(current_label_vector, candidate_label_vector)])
            if not same_class.any():
                compatible_labels_list.append(candidate_label_id)
        compatible_labels[current_label_id] = compatible_labels_list
    return compatible_labels
