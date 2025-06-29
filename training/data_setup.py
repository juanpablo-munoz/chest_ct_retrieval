import os
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, default_collate

from datasets.ct_volume_dataset import ProximityCTEmbeddingTripletDataset
from datasets.constants import PROXIMITY_VECTOR_LABELS_FOR_TRAINING
from datasets.samplers import ProximityMultilabelBalancedBatchSampler, TripletDataLoader
from utils.compatibility import determine_negative_compatibles

def get_class_id(label_vector):
    for k, v in PROXIMITY_VECTOR_LABELS_FOR_TRAINING.items():
        if np.array_equal(label_vector, v):
            return k
    return None

def load_dataset(volume_dir, seed, train_frac):
    paths = sorted(Path(volume_dir).glob("*.npz"))
    labels = []

    for p in paths:
        info = os.path.basename(p).replace(".npz", "").split("_")
        label_vector = list(map(int, info[2:6]))  # sin_anomalias, condensacion, nodulos, quistes
        labels.append(label_vector)

    labels_tensor = torch.LongTensor([get_class_id(l) for l in labels])
    x_train, x_test, y_train, y_test = train_test_split(
        paths, labels_tensor, train_size=train_frac, stratify=labels_tensor, random_state=seed
    )

    train_set = ProximityCTEmbeddingTripletDataset(x_train, y_train, preprocessed=True)
    test_set = ProximityCTEmbeddingTripletDataset(x_test, y_test, preprocessed=True)
    return train_set, test_set, determine_negative_compatibles(PROXIMITY_VECTOR_LABELS_FOR_TRAINING)

def create_loaders(train_set, test_set, n_classes, n_samples, cuda):
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

    sampler_train = ProximityMultilabelBalancedBatchSampler(
        train_set.labels_list, PROXIMITY_VECTOR_LABELS_FOR_TRAINING, n_classes, n_samples, multilabel=True
    )
    sampler_test = ProximityMultilabelBalancedBatchSampler(
        test_set.labels_list, PROXIMITY_VECTOR_LABELS_FOR_TRAINING, n_classes, n_samples, multilabel=True
    )

    batch_size = n_classes * n_samples

    return {
        "train_eval": DataLoader(train_set, batch_size=batch_size, shuffle=False, **kwargs),
        "test_eval": DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs),
        "train_triplet": DataLoader(train_set, batch_sampler=sampler_train, **kwargs),
        "test_triplet": DataLoader(test_set, batch_sampler=sampler_test, **kwargs),
        "all_triplet_train": TripletDataLoader(train_set, n_classes=n_classes, n_samples=n_samples, **kwargs),
        "all_triplet_test": TripletDataLoader(test_set, n_classes=n_classes, n_samples=n_samples, **kwargs),
    }
