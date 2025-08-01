import numpy as np
import torchio as tio
from torch.utils.data import Dataset
import pandas as pd
import os
from datasets.base import LabelVectorHelper
from datasets.constants import PROXIMITY_VECTOR_LABELS

import zarr
import fsspec
import torch
import torch.nn.functional as F
import torchio as tio
from torchvision.transforms import v2
try:
    import kornia
    import kornia.augmentation as K
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("Warning: Kornia not available. GPU augmentations will be disabled.")

class ProximityZarrPreprocessedCTTripletDataset(Dataset):
    def __init__(self, zarr_path_list, labels_list, train=True, augment=False):
        self.train = train
        self.augment = augment
        self.paths = zarr_path_list
        self.labels = np.array(labels_list)
        self.label_vector_helper = LabelVectorHelper()
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in sorted(set(self.labels))}
        self.positive_pairs_dict, self.negative_pairs_dict = self.label_vector_helper.build_pair_indices(self.labels)

        if not self.train:
            rng = np.random.default_rng(seed=0)
            self.test_triplets = [
                [i, rng.choice(self.get_positives(i)), rng.choice(self.get_negatives(i))]
                for i in range(len(self))
            ]

        # Define deterministic preprocessing
        self.preprocess = tio.Compose([
            tio.Resample(1),
            #tio.RescaleIntensity(out_min_max=(-1, 1)),
        ])
        #self.normalize = v2.Normalize(mean=[0.449], std=[0.226])

        # Optional augmentation
        self.augmentations = tio.Compose([
            tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
            tio.RandomNoise(mean=0, std=(0, 0.1)),
        ]) if augment else None

    def rand_flip(self, ctvol):
        """Flip <ctvol> along a random axis with 50% probability"""
        if np.random.randint(low=0,high=100) < 50:
            chosen_axis = np.random.randint(low=0,high=3) #0, 1, and 2 are axis options
            ctvol =  np.flip(ctvol, axis=chosen_axis)
        return ctvol

    def rand_rotate(self, ctvol):
        """Rotate <ctvol> some random amount axially with 50% probability"""
        if np.random.randint(low=0,high=100) < 50:
            chosen_k = np.random.randint(low=0,high=4)
            ctvol = np.rot90(ctvol, k=chosen_k, axes=(1,2))
        return ctvol

    def __len__(self):
        return len(self.paths)

    def get_positives(self, anchor_idx):
        return self.positive_pairs_dict[self.labels[anchor_idx]]

    def get_negatives(self, anchor_idx):
        return self.negative_pairs_dict[self.labels[anchor_idx]]

    def __getitem__(self, index):
        if self.train:
            anchor_path = self.paths[index]
            anchor_label = self.labels[index]
            anchor_label_vector = self.label_vector_helper.get_label_vector(anchor_label)
            volume = self._load_volume(anchor_path)
        else:
            a, p, n = self.test_triplets[index]
            volume = self._load_volume(self.paths[a])
            anchor_label_vector = self.label_vector_helper.get_label_vector(self.labels[a])
        return volume, anchor_label_vector

    def _load_volume(self, path):
        store = fsspec.get_mapper(f"zip://{path}", write=False)
        z = zarr.open(store, mode="r")
        vol = torch.tensor(z[:]).unsqueeze(0).float()  # [1, D, H, W]

        subject = tio.Subject(t1=tio.ScalarImage(tensor=vol))
        subject = self.preprocess(subject)
        if self.augment and self.train:
            subject = self.augmentations(subject)
            vol = subject['t1']['data']  # Still [1, D, H, W]
            subject['t1']['data'] = self.rand_rotate(self.rand_flip(vol))
        vol = subject['t1']['data']  # Still [1, D, H, W]
        #vol = self.normalize(vol)  # torchvision v2 transform
        vol = vol - 0.449 # Subtract ImageNet mean as we are using pretrained ResNet18 weights in the Proximity300x300 network
        return vol


class SliceWise2DAugmentation(torch.nn.Module):
    def __init__(self, p_affine=0.5, p_noise=0.5, p_hflip=0.5, p_vflip=0.5):
        super().__init__()
        self.affine = K.RandomAffine(
            degrees=5.0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            p=p_affine,
            same_on_batch=True  # ensures same transform for each slice in volume
        )
        self.noise = K.RandomGaussianNoise(mean=0., std=0.05, p=p_noise)
        self.hflip = K.RandomHorizontalFlip(p=p_hflip)
        self.vflip = K.RandomVerticalFlip(p=p_vflip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)  # → [B×D, C, H, W]

        # Apply the same transform per volume: group of D slices
        x = self.affine(x)
        x = self.noise(x)
        x = self.hflip(x)
        x = self.vflip(x)

        x = x.reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4)  # → [B, C, D, H, W]
        return x


class ProximityPreprocessedCTDataset(Dataset):
    def __init__(self, embeddings_path_list, labels_list, train=True, augmentations=False, device=None):
        self.rng = np.random.default_rng(seed=0)
        self.train = train
        self.paths = embeddings_path_list
        self.labels = labels_list
        self.augmentations = augmentations
        ##self.kornia_transforms = SliceWise2DAugmentation(p_affine=0.5, p_noise=0.5, p_hflip=0.5, p_vflip=0.5)
        self.device = device or torch.device('cpu')
        self.label_vector_helper = LabelVectorHelper()
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in sorted(set(self.labels))}

        self.labels = np.array(self.labels)
        self.positive_pairs_dict, self.negative_pairs_dict = self.label_vector_helper.build_pair_indices(self.labels)

        if not self.train:
            self.test_triplets = [
                [i, int(self.rng.choice(self.get_positives(i))), int(self.rng.choice(self.get_negatives(i)))]
                for i in range(len(self))
            ]

        # Setup GPU-based augmentations using Kornia
        if self.augmentations and KORNIA_AVAILABLE:
            self.gpu_augmentations = SliceWise2DAugmentation(p_affine=0.5, p_noise=0.5, p_hflip=0.5, p_vflip=0.5)
            # self.gpu_augmentations = K.AugmentationSequential(
            #     K.RandomAffine3D(
            #         degrees=((-1., 1.), (-10., 10.), -10, 10),
            #         translate=(0.1, 0.1, 0.1),
            #         scale=(0.9, 1.1),
            #         p=0.5
            #     ),
            #     K.RandomGaussianNoise3D(mean=0., std=0.05, p=0.5),
            #     K.RandomHorizontalFlip3D(p=0.5),
            #     K.RandomVerticalFlip3D(p=0.5),
            #     data_keys=["input"],
            # )
        else:
            self.gpu_augmentations = None

    def __getitem__(self, index):
        if self.train:
            anchor_path = self.paths[index]
            anchor_label = self.labels[index]
            anchor_label_vector = self.label_vector_helper.get_label_vector(anchor_label)
        else:
            a_idx, p_idx, n_idx = self.test_triplets[index]
            anchor_path = self.paths[a_idx]
            anchor_label = self.labels[a_idx]
            anchor_label_vector = self.label_vector_helper.get_label_vector(anchor_label)
        
        # Load volume with minimal CPU preprocessing
        anchor = self._load_volume_minimal(anchor_path)
        return anchor, np.array(anchor_label_vector)

    def __len__(self):
        return len(self.paths)

    def get_positives(self, anchor_idx):
        return self.positive_pairs_dict[self.labels[anchor_idx]]

    def get_negatives(self, anchor_idx):
        return self.negative_pairs_dict[self.labels[anchor_idx]]

    def _load_volume_minimal(self, path):
        """Load volume with minimal CPU preprocessing, defer heavy work to GPU"""
        with np.load(path) as data:
            vol = data['volume']
            # Minimal CPU preprocessing - just reshape and convert to tensor
            vol = np.transpose(vol, axes=[3, 0, 1, 2])  # [1, H, W, D] to [D, 1, H, W]
            vol = torch.from_numpy(vol).float()
            return vol  # Return raw tensor for GPU processing

    @staticmethod
    def gpu_preprocess_batch(batch_tensors, device, target_hw_size=(100, 100)):
        """Apply preprocessing on GPU for entire batch"""
        # Move to GPU
        if not batch_tensors.is_cuda:
            batch_tensors = batch_tensors.to(device)
        
        # Input format: [B, D, 1, H, W] where D=300 (keep constant), H=300, W=300
        B, D, C, H, W = batch_tensors.shape
        
        # Resize only height and width dimensions, keep depth (D) constant
        # For 2D interpolation on each slice, we need to reshape to [B*D, C, H, W]
        batch_tensors = batch_tensors.view(B * D, C, H, W)  # [B*D, 1, H, W]
        
        # Apply 2D bilinear interpolation to resize only H and W
        batch_tensors = F.interpolate(
            batch_tensors,  # [B*D, 1, H, W]
            size=target_hw_size,  # (150, 150) - only height and width
            mode='bilinear',
            align_corners=False
        )  # Output: [B*D, 1, 150, 150]
        
        # Reshape back to original format: [B, D, 1, H_new, W_new]
        _, C, H_new, W_new = batch_tensors.shape
        batch_tensors = batch_tensors.view(B, D, C, H_new, W_new)  # [B, 300, 1, 150, 150]
        
        # Rescale intensity to [0, 1]
        batch_flat = batch_tensors.flatten(1)  # [B, D*1*H*W]
        batch_min = batch_flat.min(dim=1, keepdim=True)[0]  # [B, 1]
        batch_max = batch_flat.max(dim=1, keepdim=True)[0]  # [B, 1]
        
        # Reshape min/max to broadcast correctly
        batch_min = batch_min.view(B, 1, 1, 1, 1)  # [B, 1, 1, 1, 1]
        batch_max = batch_max.view(B, 1, 1, 1, 1)  # [B, 1, 1, 1, 1]
        
        batch_tensors = (batch_tensors - batch_min) / (batch_max - batch_min + 1e-8)
        
        # Apply ImageNet normalization for ResNet18 compatibility
        batch_tensors = (batch_tensors - 0.449) / 0.226
        
        return batch_tensors

    def apply_gpu_augmentations(self, batch_tensors):
        """Apply GPU-based augmentations using Kornia"""
        if self.gpu_augmentations is not None and self.train:
            # Input format: [B, D, 1, H, W]
            # Kornia expects [B, C, D, H, W] format
            B, D, C, H, W = batch_tensors.shape
            batch_tensors = batch_tensors.permute(0, 2, 1, 3, 4)  # [B, D, 1, H, W] -> [B, 1, D, H, W]
            
            batch_tensors = self.gpu_augmentations(batch_tensors)
            
            # Convert back to expected format: [B, D, 1, H, W]
            batch_tensors = batch_tensors.permute(0, 2, 1, 3, 4)  # [B, 1, D, H, W] -> [B, D, 1, H, W]
        
        return batch_tensors
            

class ProximityPrerocessedCTTripletDataset(Dataset):
    def __init__(self, embeddings_path_list, labels_list, train=True, augmentations=False):
        self.train = train
        self.paths = embeddings_path_list
        self.labels = labels_list
        self.augmentations = augmentations
        #self.names = []
        self.label_vector_helper = LabelVectorHelper()
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in sorted(set(self.labels))}

        # for p in embeddings_path_list:
        #     with np.load(p) as data:
        #         self.labels.append(data['label'])
        #         self.names.append(data['name'])

        
        self.labels = np.array(self.labels)
        self.positive_pairs_dict, self.negative_pairs_dict = self.label_vector_helper.build_pair_indices(self.labels)

        if not self.train:
            rng = np.random.default_rng(seed=0)
            self.test_triplets = [
                [i, int(rng.choice(self.get_positives(i))), int(rng.choice(self.get_negatives(i)))]
                for i in range(len(self))
            ]

        # Define deterministic preprocessing
        self.preprocess = tio.Compose([
            tio.Resample(1),
            tio.Resize([224, 224, -1], image_interpolation='nearest'),
            tio.RescaleIntensity(out_min_max=(0, 1))
        ])

        # Optional augmentation
        self.tio_transforms = tio.Compose([
            tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
            tio.RandomNoise(mean=0, std=(0, 0.05)),
        ])

    def rand_flip(self, ctvol):
        """Flip <ctvol> along a random axis with 50% probability"""
        if np.random.randint(low=0,high=100) < 50:
            chosen_axis = np.random.randint(low=0,high=3) #0, 1, and 2 are axis options
            ctvol =  np.flip(ctvol, axis=chosen_axis)
        return ctvol

    def rand_rotate(self, ctvol):
        """Rotate <ctvol> some random amount axially with 50% probability"""
        if np.random.randint(low=0,high=100) < 50:
            chosen_k = np.random.randint(low=0,high=4)
            #ctvol = np.rot90(ctvol, k=chosen_k, axes=(1,2))
        return ctvol

    def __getitem__(self, index):
        if self.train:
            anchor_path = self.paths[index]
            anchor_label = self.labels[index]
            #anchor_label_vector = self.label_vector_helper.get_label_vector(anchor_label)
            #class_id = self.label_vector_helper.get_class_id(anchor_label)
        else:
            a_idx, p_idx, n_idx = self.test_triplets[index]
            anchor_path = self.paths[a_idx]
            anchor_label = self.labels[a_idx]
        anchor = self._load_volume(anchor_path)
        #return anchor, anchor_label_vector
        return anchor, anchor_label
    
    def __getitem__2(self, index):
        if self.train:
            anchor_path = self.paths[index]
            anchor_label = self.labels[index]
            anchor_label_vector = self.label_vector_helper.get_label_vector(anchor_label)
            #class_id = self.label_vector_helper.get_class_id(anchor_label)
            class_id = anchor_label
            positives = [i for i in self.positive_pairs_dict[class_id] if i != index]
            negatives = self.negative_pairs_dict[class_id]
            pos_index = np.random.choice(positives)
            neg_index = np.random.choice(negatives)
            anchor = self._load_volume(anchor_path)
            pos = self._load_volume(self.paths[pos_index])
            neg = self._load_volume(self.paths[neg_index])
        else:
            a, p, n = self.test_triplets[index]
            anchor = self._load_volume(self.paths[a])
            pos = self._load_volume(self.paths[p])
            neg = self._load_volume(self.paths[n])
        return (anchor, pos, neg), anchor_label_vector

    def __len__(self):
        return len(self.paths)

    def get_positives(self, anchor_idx):
        #return self.positive_pairs_dict[self.label_vector_helper.get_class_id(self.labels[anchor_idx])]
        return self.positive_pairs_dict[self.labels[anchor_idx]]

    def get_negatives(self, anchor_idx):
        #return self.negative_pairs_dict[self.label_vector_helper.get_class_id(self.labels[anchor_idx])]
        return self.negative_pairs_dict[self.labels[anchor_idx]]

    def _load_volume(self, path):
            with np.load(path) as data:
                tio_image = tio.ScalarImage(tensor=data['volume'], affine=np.eye(4))
                tio_image = self.preprocess(tio_image)
                vol = tio_image["data"].squeeze(0).numpy()
                if self.augmentations and self.train:
                    tio_image = self.tio_transforms(tio_image)
                    vol = self.rand_rotate(self.rand_flip(tio_image["data"].squeeze(0).numpy()))
                vol -= 0.449 # Subtract ImageNet mean as we are using pretrained ResNet18 weights in the Proximity300x300 network
                return vol[np.newaxis] # shape [1, D, H, W]
            

class ProximityCTTripletDataset(tio.SubjectsDataset):
    def __init__(self, ct_base_path, ct_image_ids, ct_labels_path, train=True):
        
        self.labels_df = pd.read_csv(
            ct_labels_path, 
            header=0, 
            #index_col=0, 
            dtype={'CT': str, 'condensacion': int, 'nodulos': int, 'quistes': int}
        )

        random_state = np.random.RandomState(0)
        
        # Mark those samples that have non-binary label values as not valid
        self.labels_df['labels_are_valid'] = self.labels_df.apply(self._determine_valid_labels, axis=1)

        s_list = []
        for s_id in ct_image_ids:
            label_attribs = self._get_ct_label_data(s_id)
            s = tio.Subject(
                t1 = tio.ScalarImage(os.path.join(ct_base_path, s_id)),
                image_path = os.path.join(ct_base_path, s_id),
                **label_attribs
            )
            s_list.append(s)

        # Transforms and augmentations to apply to images in the dataset
        s_transforms = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(3),
            tio.CropOrPad(target_shape=(130, 130, 84), padding_mode='minimum'),
            tio.RescaleIntensity(out_min_max=(0, 1)),
            tio.ZNormalization() # TODO: apply exact z-norm required by ResNet-18
            
        ])
        '''
        s_transforms = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(1),
            tio.CropOrPad(target_shape=(290, 290, 249), padding_mode='minimum'),
            tio.RescaleIntensity(out_min_max=(0, 1)),
            tio.ZNormalization() # TODO: apply exact z-norm required by ResNet-18
            
        ])
        '''

        # Filter out those samples with invalid labels
        s_list = [s for s in s_list if s['labels_are_valid']]
        
        super().__init__(s_list, transform=s_transforms, load_getitem=False)

        self.positive_pairs_dict, self.negative_pairs_dict = self._get_positive_negative_pairs_indices()

        self.train = train

        if not self.train:
            triplets = [[i,
                         random_state.choice(self.get_positives_indices(i)),
                         random_state.choice(self.get_negatives_indices(i)),
                         ]
                        for i in range(len(s_list))]
            self.test_triplets = triplets
        #print(self.__getitem__(0, verbose=True)) # For testing purposes
    
    def _determine_valid_labels(self, row):
        if (row['condensacion'] == 1 or row['condensacion'] == 0) \
        and (row['nodulos'] == 1 or row['nodulos'] == 0) \
        and (row['quistes'] == 1 or row['quistes'] == 0):
            return True
        return False

    def _get_ct_label_data(self, ct_code):
        result = self.labels_df.loc[self.labels_df['CT'] == ct_code+'AN'].to_dict('records')
        if result:
            r = result[0]
            r['labels_as_vector'] = self._get_label_vector(r)
            return r
        else:
            return {'labels_are_valid': False}

    def _get_label_vector(self, labels_dict):
        c = labels_dict['condensacion']
        n = labels_dict['nodulos']
        q = labels_dict['quistes']
        return [c, n, q]
    
    def _collate(self, sample):
        data = sample['t1']['data'].squeeze(0).numpy()
        labels = sample['labels_as_vector']
        return data, labels

    def get_labels(self, idx_list):
        l = []
        if not hasattr(idx_list, '__iter__'):
            idx_list = [idx_list]
        for s in self.dry_iter():
            l.append(s['labels_as_vector'])
        l = np.array(l)
        return l[idx_list]

    def _get_positive_negative_pairs_indices(self):
        positive_pairs_dict = dict((i, []) for i in range(8))
        negative_pairs_dict = dict((i, []) for i in range(8))
        for current_label_id, current_label_vector in PROXIMITY_VECTOR_LABELS.items():
            for sample_index, sample in enumerate(self.dry_iter()):
                sample_label = sample['labels_as_vector']
                #is_positive_pair = (current_label_vector == sample_label).all()
                is_positive_pair = current_label_vector == sample_label
                if is_positive_pair:
                    positive_pairs_dict[current_label_id].append(sample_index)
                else:
                    negative_pairs_dict[current_label_id].append(sample_index)
        return positive_pairs_dict, negative_pairs_dict


    def get_positives_indices(self, anchor_idx):
        #print('anchor_idx:', anchor_idx)
        #print('self.get_labels(anchor_idx):', self.get_labels(anchor_idx))
        return self.positive_pairs_dict[LabelVectorHelper.get_class_id(self.get_labels(anchor_idx))]

    def get_negatives_indices(self, anchor_idx):
        return self.negative_pairs_dict[LabelVectorHelper.get_class_id(self.get_labels(anchor_idx))]
    
    def __getitem__(self, index, verbose=False):
        if self.train:
            # item 1: anchor
            sample1 = super().__getitem__(index)
            #sample1 = {'t1':sample1['t1'], 'labels_as_vector':sample1['labels_as_vector']}
            img1, label1 = self._collate(sample1)
            class1 = LabelVectorHelper.get_class_id(label1)
            positives_list = self.positive_pairs_dict[class1]
            negatives_list = self.negative_pairs_dict[class1]
            if verbose:
                print(f'\nItem of index {index} has labels {label1}')
                positives_labels = self.get_labels(positives_list)
                print(f'Positive items\' indices are {positives_list} of labels {positives_labels}')
                negatives_labels = self.get_labels(negatives_list)
                print(f'Negative items\' indices are {negatives_list} of labels {negatives_labels}')
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(positives_list)
            negative_index = np.random.choice(negatives_list)
            negative_labels = self.get_labels(negative_index)
            
            # item 2: positive sample
            img2, _ = self._collate(super().__getitem__(positive_index))
    
            # item 3: negative sample
            img3, _ = self._collate(super().__getitem__(negative_index))
        else:
            img1, _ = self._collate(super().__getitem__(self.test_triplets[index][0]))
            img2, _ = self._collate(super().__getitem__(self.test_triplets[index][1]))
            img3, _ = self._collate(super().__getitem__(self.test_triplets[index][2]))

        #img1 = Image.fromarray(img1.numpy(), mode='L')
        #img2 = Image.fromarray(img2.numpy(), mode='L')
        #img3 = Image.fromarray(img3.numpy(), mode='L')
        #if self.transform is not None:
        #    img1 = self.transform(img1)
        #    img2 = self.transform(img2)
        #    img3 = self.transform(img3)
        return (img1, img2, img3), []

