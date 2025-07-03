import numpy as np
import torchio as tio
import pandas as pd
import os
from chest_ct_retrieval.datasets.base import LabelVectorHelper
from chest_ct_retrieval.datasets.constants import PROXIMITY_VECTOR_LABELS


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

