from itertools import combinations, permutations

import numpy as np
import torch

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        cuda = torch.cuda.is_available()
        model.eval()
        embeddings = []
        #labels = np.zeros((len(dataloader.dataset), label_size))
        labels = []
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings.extend(model(images))
            labels.extend(target)
            k += len(images)
            #break
        embeddings = torch.stack(embeddings)
        labels = torch.tensor(labels)
        #print(embeddings)
        #print(labels)
        return embeddings, labels

def determine_negative_compatibles(vector_labels_dict):
    compatible_labels = dict()
    for current_label_id in vector_labels_dict.keys():
        current_label_vector = vector_labels_dict[current_label_id]
        compatible_labels_list = []
        for candidate_label_id, candidate_label_vector in vector_labels_dict.items():
            if current_label_id == candidate_label_id:
                continue
            same_class_detection = np.array([x and y for x, y in zip(current_label_vector, candidate_label_vector)])
            # same_class_detection: binary array. Value at index i determines whether current_label and candidate_label both indicate "positive" for the i-th class
            any_same_class_detection = same_class_detection.any()
            detect_only_different_classes = not any_same_class_detection
            if detect_only_different_classes:
                # if current_label_vector and candidate_label_vector correpsond to the detection of sets of mutually exclusive classes, then they are "negative-compatible"
                compatible_labels_list.append(candidate_label_id)
        compatible_labels[current_label_id] = compatible_labels_list
    #print('self.compatible_labels:', self.compatible_labels)
    #valid_label_pairs = []
    #for k, v in compatible_labels.items():
    #    valid_label_pairs.extend([k, l] for l in v if l > k)
    #print('self.labels_set:', self.labels_set)
    #print('self.label_to_indices:', self.label_to_indices)
    #print('self.valid_label_pairs:', self.valid_label_pairs)
    return compatible_labels

def pdist(vectors):
    #distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(dim=1).view(-1, 1)
    distance_matrix = torch.cdist(vectors, vectors, p=2)**2 # distance metric: squared euclidean distance
    return distance_matrix

def query_dataset_dist(query_tensor, dataset_tensor):
    #distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(dim=1).view(-1, 1)
    distance_matrix = torch.cdist(query_tensor, dataset_tensor, p=2)**2 # distance metric: squared euclidean distance
    return distance_matrix

def psim_cosine(vectors):
    cos_sim = torch.nn.functional.cosine_similarity
    similarity_matrix = cos_sim(vectors[None,:,:], vectors[:,None,:], dim=-1)
    return similarity_matrix

def pdist_dot(vectors):
    distance_matrix = torch.bmm(vectors[None,:,:], vectors[:,None,:].permute(1, 2, 0))
    return distance_matrix



class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn
        
        
    def get_triplets(self, query_embeddings, query_labels, db_embeddings, db_labels, negative_compatibles_dict, print_log):
        triplet_mining_in_batch = False
        if torch.equal(query_embeddings, db_embeddings):
            triplet_mining_in_batch = True
        if self.cpu:
            query_embeddings = query_embeddings.cpu()
            db_embeddings = db_embeddings.cpu()
        
        #TODO: make make the distance metric used to calculate distance_metric setteable via a parameter
        distance_matrix = query_dataset_dist(query_embeddings, db_embeddings)
        #distance_matrix = pdist(embeddings)
        #distance_matrix = psim_cosine(embeddings).cpu()
        #distance_matrix = pdist_dot(embeddings).cpu()
        
        if print_log:
            print('FunctionNegativeTripletSelector.get_triplets()')
            k = min(10, len(distance_matrix))
            if k < len(distance_matrix):
                print(f'(Distance matrix has length {len(distance_matrix)} which is too long! Printing only its first {k} elements.)')
            print(f'distance_matrix[:{k}]:')
            print(distance_matrix[:k])

        #labels = np.array([self.get_class_id(e) for e in labels.cpu().data.numpy()])
        query_labels = query_labels.detach().cpu().numpy()
        db_labels = db_labels.detach().cpu().numpy()
        
        triplets = []

        for query_label in set(query_labels):
            label_triplets = []
            query_label_mask = (query_labels == query_label)
            db_positive_label_mask = (db_labels == query_label)
            query_label_indices = np.where(query_label_mask)[0]
            db_positive_label_indices = np.where(db_positive_label_mask)[0]

            # if triplets are being mined from the batch itself (query and database are the same)
            # then omit labels for which there's only one sample
            if triplet_mining_in_batch and len(db_positive_label_indices) == 1:
                continue
            # if the triplets are being mined from a query on a database (query and database are different)
            # then omit query samples for which there are no same-label samples in the database
            elif not triplet_mining_in_batch and len(db_positive_label_indices) < 1:
                continue
            
            negative_compatible_labels = negative_compatibles_dict[query_label]
            db_negatives_mask = np.array([False] * len(db_labels))
            for negative_compatible_label in negative_compatible_labels:
                db_negatives_mask = db_negatives_mask | (db_labels == negative_compatible_label)
            db_negative_indices = np.where(db_negatives_mask)[0]
            #anchor_positives = np.array(list(combinations(label_indices, 2)))  # All anchor-positive pairs
            anchor_positives = np.array(np.meshgrid(query_label_indices, db_positive_label_indices)).T.reshape(-1, 2)  # All anchor-positive pairs
            # if triplets are being mined from the batch itself (query and database are the same)
            # then prune anchor_positives pairs in which the anchor and positive are both the same object
            if triplet_mining_in_batch:
                anchor_positives = np.array([ap_pair for ap_pair in anchor_positives if ap_pair[0] != ap_pair[1]])
            #if triplet_mining_in_batch:
                # if triplets are being mined from the batch itself (query and database are the same)
                # then omit anchor-positive pairs with the same index because these correspond to malformed pairs in which an item is paired with itself
                #anchor_positives = np.array([ap for ap in anchor_positives if ap[0] != ap[1]])
            anchor_negatives = np.array(np.meshgrid(query_label_indices, db_negative_indices)).T.reshape(-1, 2)  # All anchor-negative pairs
            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            an_distances = distance_matrix[anchor_negatives[:, 0], anchor_negatives[:, 1]]
            if print_log:
                k = min(10, len(ap_distances))
                if k < len(ap_distances):
                    print(f'(Pair and distance lists are of length {len(ap_distances)} which is too long! Printing only the first {k} elements of each.)')
                print('query_label:', query_label)
                print(f'anchor_positives[:{k}] [query_index, db_index]:')
                print(anchor_positives[:k])
                print(f'ap_distances[:{k}]:')
                print(ap_distances[:k])
                print(f'anchor_negatives[:{k}] [query_index, db_index]:')
                print(anchor_negatives[:k])
                print(f'an_distances[:{k}]:')
                print(an_distances[:k])

            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                # Avoid including malformed anchor-positive pairs if there's any
                # This case may occur only when mining triplets from the batch itself (query and db are the same)
                if triplet_mining_in_batch and anchor_positive[0] == anchor_positive[1]:
                    # skip pairs in which the anchor and positive are both the same object
                    continue
                all_an_distances = distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(db_negative_indices)]
                loss_values =  ap_distance - all_an_distances + self.margin
                loss_values = loss_values.data.cpu().numpy()
                if print_log:
                    print(f'for anchor_positive={anchor_positive} with distance={round(ap_distance.item(), 4)}: loss_values={loss_values}')
                mined_negative = self.negative_selection_fn(loss_values)
                if mined_negative is not None:
                    mined_negative = db_negative_indices[mined_negative]
                    if print_log:
                        print(f'Semi-hard negative for anchor_positive {anchor_positive} is: {mined_negative}')
                    triplets.append([anchor_positive[0], anchor_positive[1], mined_negative])
                else:
                    mined_negative = random_hard_negative(loss_values)
                    if mined_negative is not None:
                        if print_log:
                            print(f'Random hard negative for anchor_positive {anchor_positive} is: {mined_negative}')
                        triplets.append([anchor_positive[0], anchor_positive[1], mined_negative])
            #         label_triplets.append([anchor_positive[0], anchor_positive[1], mined_negative])
            # if len(label_triplets) > 0:
            #     triplets.extend(label_triplets)
                

        if len(triplets) == 0:
            if print_log:
                print(f'(default) negative index in batch is: {db_negative_indices[0]}')
                print(f'(default) triplet to be returned: {[anchor_positive[0], anchor_positive[1], db_negative_indices[0]]}')
            triplets.append([anchor_positive[0], anchor_positive[1], db_negative_indices[0]])

        
        
        #triplets = [torch.LongTensor(t) for t in triplets]
        triplets = np.array(triplets)
        return torch.LongTensor(triplets)
        #return triplets


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)
