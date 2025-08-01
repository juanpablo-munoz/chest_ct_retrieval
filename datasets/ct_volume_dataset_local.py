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
import torchio as tio
from torchvision.transforms import v2

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


class ProximityPreprocessedCTDataset(Dataset):
    def __init__(self, embeddings_path_list, labels_list, train=True, augmentations=False):
        self.rng = np.random.default_rng(seed=0)
        self.train = train
        self.paths = embeddings_path_list
        self.labels = labels_list
        self.augmentations = augmentations
        #self.names = []
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

        # Define deterministic preprocessing
        self.preprocess = tio.Compose([
            tio.Resample(1),
            tio.Resize([-1, 150, 150], image_interpolation='nearest'),
            tio.RescaleIntensity(out_min_max=(0, 1))
        ])

        # Optional augmentation
        self.tio_transforms = tio.Compose([
            tio.RandomAffine(scales=(0.9, 1.1), degrees=10, p=0.5),
            tio.RandomNoise(mean=0, std=(0, 0.05), p=0.5),
        ])

    def rand_flip(self, ctvol):
        """Flip <ctvol> along a random axis with 50% probability"""
        if self.rng.random() < 0.5:
            chosen_axis = np.random.randint(low=0,high=3) #0, 1, and 2 are axis options
            ctvol =  np.flip(ctvol, axis=chosen_axis)
        return ctvol

    def rand_rotate(self, ctvol):
        """Rotate <ctvol> some random amount axially with 50% probability"""
        if self.rng.random() < 0.5:
            chosen_k = self.rng.choice([0, 1, 2, 3])
            #ctvol = np.rot90(ctvol, k=chosen_k, axes=(1,2))
        return ctvol

    def __getitem__(self, index):
        if self.train:
            anchor_path = self.paths[index]
            anchor_label = self.labels[index]
            anchor_label_vector = self.label_vector_helper.get_label_vector(anchor_label)
            #class_id = self.label_vector_helper.get_class_id(anchor_label)
        else:
            a_idx, p_idx, n_idx = self.test_triplets[index]
            anchor_path = self.paths[a_idx]
            anchor_label = self.labels[a_idx]
        anchor = self._load_volume(anchor_path)
        return anchor, anchor_label_vector
        #return anchor, anchor_label

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
                vol = data['volume']
                vol = np.transpose(vol, axes=[3, 0, 1, 2]) # transpose [1, H, W, D] to [D, 1, H, W]
                tio_image = tio.ScalarImage(tensor=vol, affine=np.eye(4))
                tio_image = self.preprocess(tio_image)
                if self.augmentations and self.train:
                    tio_image = self.tio_transforms(tio_image)
                    vol = self.rand_rotate(self.rand_flip(tio_image["data"].squeeze(0).numpy()))
                vol -= 0.449 # Subtract ImageNet mean as we are using pretrained ResNet18 weights in the Proximity300x300 network
                vol /= 0.226 # Divide by ImageNet std to complete Z-Normalization
                vol = vol.astype(np.float32)
                return vol # shape [ D, 1, H, W]
            

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
                vol = ( vol - 0.449 ) / 0.226 # Subtract ImageNet mean and divide by its std as we are using pretrained ResNet18 weights in the network
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

