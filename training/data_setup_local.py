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

def collate_tensor_batch(batch):
    samples = []
    transposed_target = []
    for sample, target in batch:
        samples.append(sample)
        transposed_target.append(target)
    samples = torch.tensor(np.array(samples))
    #transposed_target = np.array(transposed_target).transpose().tolist()
    transposed_target = torch.tensor(transposed_target)
    return samples, transposed_target

def collate_tensor_batch_vector_labels(batch):
    label_vector_helper = LabelVectorHelper()
    samples = []
    transposed_target = []
    for sample, target in batch:
        samples.append(sample)
        transposed_target.append(target)
    samples = torch.tensor(np.array(samples))
    #transposed_target = np.array(transposed_target).transpose().tolist()
    transposed_target = torch.tensor(transposed_target)
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

def load_dataset_microf1(volume_dir, seed, train_frac, augmentations_arg, device=None):
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

    train_set = ProximityPreprocessedCTDataset(x_train, y_train, train=True, augmentations=augmentations_arg, device=device)
    test_set = ProximityPreprocessedCTDataset(x_test, y_test, train=False, augmentations=False, device=device)
    return train_set, test_set, determine_negative_compatibles(PROXIMITY_VECTOR_LABELS_FOR_TRAINING)


def create_loaders(train_set, test_set, n_classes, n_samples, cuda):
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

    sampler_train = BalancedBatchSampler(
        train_set.labels, PROXIMITY_VECTOR_LABELS_FOR_TRAINING, n_classes, n_samples, multilabel=True
    )
    sampler_test = BalancedBatchSampler(
        test_set.labels, PROXIMITY_VECTOR_LABELS_FOR_TRAINING, n_classes, n_samples, multilabel=True
    )

    batch_size = n_classes * n_samples

    return {
        "train_eval": DataLoader(train_set, collate_fn=collate_tensor_batch, batch_size=batch_size, shuffle=False, **kwargs),
        "test_eval": DataLoader(test_set, collate_fn=collate_tensor_batch, batch_size=batch_size, shuffle=False, **kwargs),
        "train_triplet": DataLoader(train_set, collate_fn=collate_tensor_batch, batch_sampler=sampler_train, **kwargs),
        "test_triplet": DataLoader(test_set, collate_fn=collate_tensor_batch, batch_sampler=sampler_test, **kwargs),
        "all_triplet_train": TripletDataLoader(train_set, n_classes=n_classes, n_samples=n_samples, **kwargs),
        "all_triplet_test": TripletDataLoader(test_set, n_classes=n_classes, n_samples=n_samples, **kwargs),
    }

def collate_tensor_batch_gpu_optimized(batch):
    """Optimized collate function with GPU preprocessing integration"""
    samples = []
    targets = []
    for sample, target in batch:
        samples.append(sample)
        targets.append(target)
    
    # Stack samples and targets efficiently
    samples = torch.stack([torch.from_numpy(s) if isinstance(s, np.ndarray) else s for s in samples])
    targets = torch.stack([torch.from_numpy(t) if isinstance(t, np.ndarray) else t for t in targets])
    
    return samples, targets

def collate_with_gpu_preprocessing(batch, device=None, dataset=None):
    """Collate function that applies GPU preprocessing to the batch"""
    from datasets.ct_volume_dataset_local import ProximityPreprocessedCTDataset
    
    samples = []
    targets = []
    for sample, target in batch:
        samples.append(sample)
        targets.append(target)
    
    # Stack raw samples and targets
    samples = torch.stack([torch.from_numpy(s) if isinstance(s, np.ndarray) else s for s in samples])
    targets = torch.stack([torch.from_numpy(t) if isinstance(t, np.ndarray) else t for t in targets])
    
    # Apply GPU preprocessing if device is provided
    if isinstance(device, torch.cuda.device) and device != 'cpu':
        samples = ProximityPreprocessedCTDataset.gpu_preprocess_batch(samples, 'cuda')
        
        # Apply GPU augmentations if dataset supports it
        if dataset is not None and hasattr(dataset, 'apply_gpu_augmentations'):
            samples = dataset.apply_gpu_augmentations(samples)
    
    return samples, targets

def create_loaders_microf1(train_set, test_set, batch_size, cuda, gpu_preprocessing=False, device=None, **kwargs):
    # Optimize for parallel loading and prefetching
    if cuda:
        kwargs = {
            'num_workers': 2,  # Parallel workers for data loading
            'pin_memory': True,  # Pin memory for faster GPU transfer
            'prefetch_factor': 1,  # Prefetch next batches
            'persistent_workers': True,  # Keep workers alive between epochs
        }
    else:
        kwargs = {'num_workers': 2}

    # Choose collate function based on GPU preprocessing option
    if gpu_preprocessing and device is not None:
        from functools import partial
        train_collate_fn = partial(collate_with_gpu_preprocessing, device=device, dataset=train_set)
        test_collate_fn = partial(collate_with_gpu_preprocessing, device=device, dataset=test_set)
    else:
        train_collate_fn = collate_tensor_batch_gpu_optimized
        test_collate_fn = collate_tensor_batch_gpu_optimized

    return {
        "train": DataLoader(
            train_set, 
            collate_fn=train_collate_fn, 
            batch_size=batch_size, 
            shuffle=True,  # Add shuffling for better training
            drop_last=True,  # Drop incomplete batches for consistent GPU memory usage
            **kwargs
        ),
        "test": DataLoader(
            test_set, 
            collate_fn=test_collate_fn, 
            batch_size=batch_size, 
            shuffle=False,
            drop_last=False,
            **kwargs
        ),
    }

def create_loaders_microf1_optimized(train_set, test_set, batch_size, cuda, device=None, jupyter_friendly=True):
    """Optimized loader creation with best practices for large data"""
    # Set device for GPU preprocessing
    if hasattr(train_set, 'device') and device is not None:
        train_set.device = device
        test_set.device = device
    
    # Configure multiprocessing based on environment
    if jupyter_friendly:
        # Jupyter-friendly configuration (avoids multiprocessing issues)
        if cuda and device is not None:
            kwargs = {
                'num_workers': 0,  # Disable multiprocessing for Jupyter
                'pin_memory': True,
                # Prefetch and persistent workers not needed for num_workers=0
            }
        else:
            kwargs = {'num_workers': 0}
    else:
        # Full multiprocessing configuration for standalone scripts
        if cuda and device is not None:
            kwargs = {
                'num_workers': 2,  # Moderate workers for stability
                'pin_memory': True,
                'prefetch_factor': 1,  # Conservative prefetch
                'persistent_workers': True,
            }
        else:
            kwargs = {'num_workers': 2}

    return create_loaders_microf1(train_set, test_set, batch_size, cuda, 
                                  gpu_preprocessing=True, device=device, **kwargs)

def create_loaders_microf1_standalone(train_set, test_set, batch_size, cuda, device=None):
    """High-performance loader for standalone scripts (non-Jupyter)"""
    return create_loaders_microf1_optimized(train_set, test_set, batch_size, cuda, 
                                           device=device, jupyter_friendly=False)
