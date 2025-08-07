import os
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets.base import LabelVectorHelper
from datasets.ct_volume_dataset_local import ProximityPreprocessedCTDataset, ProximityPrerocessedCTTripletDataset, ProximityZarrPreprocessedCTTripletDataset
from datasets.constants import PROXIMITY_VECTOR_LABELS_FOR_TRAINING
from datasets.samplers import BalancedBatchSampler
from datasets.loaders import TripletDataLoader
from utils.compatibility import determine_negative_compatibles

import kornia.augmentation as K
import torch.nn.functional as F
from utils.transforms import RandomGaussianNoise3D

gpu_aug = K.AugmentationSequential(
    K.RandomAffine3D(degrees=(5, 5, 5), scale=(0.95, 1.05), p=0.5),
    RandomGaussianNoise3D(mean=0.0, std=0.01, p=0.5),
    data_keys=["input"]
).to("cuda")

def collate_tensor_batch(batch, apply_gpu_aug=False):
    """Optimized collation function with uint8 support and GPU-ready processing"""
    samples = []
    transposed_target = []

    for sample, target in batch:
        samples.append(sample)
        transposed_target.append(target)

    # Convert uint8 data to float32 tensor
    samples = torch.tensor(np.array(samples), dtype=torch.float16)  # [B, D, 1, H, W]
    transposed_target = torch.tensor(transposed_target)

    # Normalize uint8 data (0-255) to [0,1] then apply ImageNet normalization
    samples = samples / 255.0
    samples = (samples - 0.449) / 0.226

    return samples, transposed_target

class CollateFn:
    def __init__(self, apply_gpu_aug=False):
        self.apply_gpu_aug = apply_gpu_aug

    def __call__(self, batch):
        return collate_tensor_batch(batch, apply_gpu_aug=self.apply_gpu_aug)

def collate_tensor_batch_vector_labels(batch):
    """Collation function with vector labels for triplet training"""
    label_vector_helper = LabelVectorHelper()
    samples = []
    transposed_target = []
    for sample, target in batch:
        samples.append(sample)
        transposed_target.append(target)
    
    # Convert uint8 data to float32 and normalize
    samples = torch.tensor(np.array(samples), dtype=torch.float16)
    transposed_target = torch.tensor(transposed_target)
    
    # Normalize uint8 data (0-255) to [0,1] then apply ImageNet normalization
    samples = samples / 255.0
    samples = (samples - 0.449) / 0.226
    
    vector_labels = [label_vector_helper.get_label_vector(lbl) for lbl in transposed_target]
    return samples, vector_labels

def get_class_id(label_vector):
    for k, v in PROXIMITY_VECTOR_LABELS_FOR_TRAINING.items():
        if np.array_equal(label_vector, v):
            return k
    return None

def load_dataset(volume_dir, seed, train_frac, augmentations_arg):
    paths = sorted(Path(volume_dir).glob("*.npz"))
    #paths = sorted(Path(volume_dir).glob("*.zarr.zip")) # Zarr volumes come in ZIP format.
    labels = []
    label_vector_helper = LabelVectorHelper()

    for p in paths:
        info = os.path.basename(p).replace(".npz", "").split("_")
        #info = os.path.basename(p).replace(".zarr.zip", "").split("_")  # Zarr volumes come in ZIP format.
        label_vector = list(map(int, info[2:6]))  # sin_anomalias, condensacion, nodulos, quistes
        labels.append(label_vector)

    labels_tensor = torch.LongTensor([label_vector_helper.get_class_id(l) for l in labels])
    x_train, x_test, y_train, y_test = train_test_split(
        paths, labels_tensor, train_size=train_frac, stratify=labels_tensor, random_state=seed
    )

    train_set = ProximityPrerocessedCTTripletDataset(x_train, y_train, train=True, augmentations=augmentations_arg)
    test_set = ProximityPrerocessedCTTripletDataset(x_test, y_test, train=False, augmentations=False)
    #train_set = ProximityZarrPreprocessedCTTripletDataset(x_train, y_train, train=True, augment=True)
    #test_set = ProximityZarrPreprocessedCTTripletDataset(x_test, y_test, train=False, augment=True)
    return train_set, test_set, determine_negative_compatibles(PROXIMITY_VECTOR_LABELS_FOR_TRAINING)

def load_dataset_microf1(volume_dir, seed, train_frac, train_eval_frac, augmentations_arg):
    paths = sorted(Path(volume_dir).glob("*.npz"))
    labels = []
    label_vector_helper = LabelVectorHelper()

    for p in paths:
        info = os.path.basename(p).replace(".npz", "").split("_")
        label_vector = list(map(int, info[2:6]))  # sin_anomalias, condensacion, nodulos, quistes
        labels.append(label_vector)

    labels_tensor = torch.LongTensor([label_vector_helper.get_class_id(l) for l in labels])
    x_train, x_test, y_train, y_test = train_test_split(
        paths, labels_tensor, train_size=train_frac, stratify=labels_tensor, random_state=seed
    )

    # Create stratified subset from training data for evaluation (subset of train, not separate split)
    x_train_eval, _, y_train_eval, _ = train_test_split(
        x_train, y_train, train_size=train_eval_frac, stratify=y_train, random_state=seed+1
    )

    train_set = ProximityPreprocessedCTDataset(x_train, y_train, train=True, augmentations=augmentations_arg)
    train_eval_set = ProximityPreprocessedCTDataset(x_train_eval, y_train_eval, train=False, augmentations=False)  # No augmentations for eval
    test_set = ProximityPreprocessedCTDataset(x_test, y_test, train=False, augmentations=False)
    return train_set, train_eval_set, test_set, determine_negative_compatibles(PROXIMITY_VECTOR_LABELS_FOR_TRAINING)


def create_loaders(train_set, test_set, n_classes, n_samples, cuda):
    """Create optimized data loaders for triplet training with GPU-ready processing"""
    kwargs = {'num_workers': 2, 'prefetch_factor': 1, 'persistent_workers': True, 'pin_memory': True} if cuda else {}

    sampler_train = BalancedBatchSampler(
        train_set.labels, PROXIMITY_VECTOR_LABELS_FOR_TRAINING, n_classes, n_samples, multilabel=True
    )
    sampler_test = BalancedBatchSampler(
        test_set.labels, PROXIMITY_VECTOR_LABELS_FOR_TRAINING, n_classes, n_samples, multilabel=True
    )

    batch_size = n_classes * n_samples

    return {
        "train_eval": DataLoader(train_set, collate_fn=CollateFn(apply_gpu_aug=False), batch_size=batch_size, shuffle=False, **kwargs),
        "test_eval": DataLoader(test_set, collate_fn=CollateFn(apply_gpu_aug=False), batch_size=batch_size, shuffle=False, **kwargs),
        "train_triplet": DataLoader(train_set, collate_fn=CollateFn(apply_gpu_aug=False), batch_sampler=sampler_train, **kwargs),
        "test_triplet": DataLoader(test_set, collate_fn=CollateFn(apply_gpu_aug=False), batch_sampler=sampler_test, **kwargs),
        "all_triplet_train": TripletDataLoader(train_set, n_classes=n_classes, n_samples=n_samples, **kwargs),
        "all_triplet_test": TripletDataLoader(test_set, n_classes=n_classes, n_samples=n_samples, **kwargs),
    }

def create_loaders_microf1(train_set, train_eval_set, test_set, batch_size, cuda):
    kwargs = {'num_workers': 2, 'prefetch_factor': 1, 'persistent_workers': True, 'pin_memory': True} if cuda else {}

    return {
        "train": DataLoader(train_set, collate_fn=CollateFn(apply_gpu_aug=False), batch_size=batch_size, **kwargs),
        "train_eval": DataLoader(train_eval_set, collate_fn=CollateFn(apply_gpu_aug=False), batch_size=batch_size, shuffle=False, **kwargs),
        "test": DataLoader(test_set, collate_fn=CollateFn(apply_gpu_aug=False), batch_size=batch_size, shuffle=False, **kwargs),
    }
