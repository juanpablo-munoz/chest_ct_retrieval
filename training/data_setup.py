import os
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from chest_ct_retrieval.datasets.ct_volume_dataset import ProximityCTTripletDataset, ProximityPrerocessedCTTripletDataset
from chest_ct_retrieval.datasets.embedding_dataset import ProximityCTEmbeddingTripletDataset
from chest_ct_retrieval.datasets.constants import PROXIMITY_VECTOR_LABELS_FOR_TRAINING
from chest_ct_retrieval.datasets.samplers import BalancedBatchSampler
from chest_ct_retrieval.datasets.loaders import TripletDataLoader
from chest_ct_retrieval.utils.compatibility import determine_negative_compatibles

def get_class_id(label_vector):
    for k, v in PROXIMITY_VECTOR_LABELS_FOR_TRAINING.items():
        if np.array_equal(label_vector, v):
            return k
    return None

def load_dataset(volume_dir, seed, train_frac):
    paths = sorted(Path(volume_dir).glob("*.npz"))
    labels = []

    # samples_path_list = [[]]*len(volume_dir)
    # labels_list = [[]]*len(volume_dir)
    # for i, p in enumerate(paths):
    #     _, fname = os.path.split(p)
    #     info = fname.split('.')[0:-1]
    #     info = ''.join(info)
    #     info = info.split('_')
    #     fid = int(info[0])
    #     vol_id = info[1]
    #     sin_anomalias = int(info[2])
    #     condensacion = int(info[3])
    #     nodulos = int(info[4])
    #     quistes = int(info[5])
    #     samples_path_list[ fid - 1 ] = p
    #     labels_list[ fid - 1 ] = [sin_anomalias, condensacion, nodulos, quistes]
    # labels_as_classes = torch.LongTensor([get_class_id(l) for l in labels_list])

    # negative_compatibles_dict = determine_negative_compatibles(PROXIMITY_VECTOR_LABELS_FOR_TRAINING)

    for p in paths:
        info = os.path.basename(p).replace(".npz", "").split("_")
        label_vector = list(map(int, info[2:6]))  # sin_anomalias, condensacion, nodulos, quistes
        labels.append(label_vector)

    labels_tensor = torch.LongTensor([get_class_id(l) for l in labels])
    x_train, x_test, y_train, y_test = train_test_split(
        paths, labels_tensor, train_size=train_frac, stratify=labels_tensor, random_state=seed
    )

    train_set = ProximityPrerocessedCTTripletDataset(x_train, train=True)
    test_set = ProximityPrerocessedCTTripletDataset(x_test, train=False)
    return train_set, test_set, determine_negative_compatibles(PROXIMITY_VECTOR_LABELS_FOR_TRAINING)

def create_loaders(train_set, test_set, n_classes, n_samples, cuda):
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

    sampler_train = BalancedBatchSampler(
        train_set.labels_list, PROXIMITY_VECTOR_LABELS_FOR_TRAINING, n_classes, n_samples, multilabel=True
    )
    sampler_test = BalancedBatchSampler(
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
