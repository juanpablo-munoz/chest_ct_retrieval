import numpy as np
from torch.utils.data import Dataset
from datasets.base import LabelVectorHelper

class ProximityCTEmbeddingDataset(Dataset):
    def __init__(self, embeddings_path_list):
        self.embeddings = []
        self.labels = []
        self.names = []

        for p in embeddings_path_list:
            with np.load(p) as data:
                self.embeddings.append(data['embedding'])
                self.labels.append(data['label'])
                self.names.append(data['name'])

        self.embeddings = np.array(self.embeddings)
        self.labels = np.array(self.labels)

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]

    def __len__(self):
        return len(self.embeddings)


class ProximityCTEmbeddingTripletDataset(Dataset):
    def __init__(self, embeddings_path_list, train=True):
        self.train = train
        self.embeddings = []
        self.labels = []
        self.names = []

        for p in embeddings_path_list:
            with np.load(p) as data:
                self.embeddings.append(data['embedding'])
                self.labels.append(data['label'])
                self.names.append(data['name'])

        self.embeddings = np.array(self.embeddings)
        self.labels = np.array(self.labels)

        self.positive_pairs_dict, self.negative_pairs_dict = LabelVectorHelper.build_pair_indices(self.labels)

        if not self.train:
            rng = np.random.default_rng(seed=0)
            self.test_triplets = [
                [i, rng.choice(self.get_positives(i)), rng.choice(self.get_negatives(i))]
                for i in range(len(self))
            ]

    def __getitem__(self, index):
        if self.train:
            anchor = self.embeddings[index]
            anchor_label = self.labels[index]
            class_id = LabelVectorHelper.get_class_id(anchor_label)
            positives = [i for i in self.positive_pairs_dict[class_id] if i != index]
            negatives = self.negative_pairs_dict[class_id]
            pos = self.embeddings[np.random.choice(positives)]
            neg = self.embeddings[np.random.choice(negatives)]
        else:
            a, p, n = self.test_triplets[index]
            anchor, pos, neg = self.embeddings[a], self.embeddings[p], self.embeddings[n]
        return (anchor, pos, neg), []

    def __len__(self):
        return len(self.embeddings)

    def get_positives(self, anchor_idx):
        return self.positive_pairs_dict[LabelVectorHelper.get_class_id(self.labels[anchor_idx])]

    def get_negatives(self, anchor_idx):
        return self.negative_pairs_dict[LabelVectorHelper.get_class_id(self.labels[anchor_idx])]
