import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        #distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        #distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        distance_positive = nn.CosineSimilarity(dim=1, eps=1e-6)(anchor, positive)
        distance_negative = nn.CosineSimilarity(dim=1, eps=1e-6)(anchor, negative)
        #losses = F.relu(distance_positive - distance_negative + self.margin)
        losses = F.relu(self.margin + distance_negative - distance_positive)
        #print('anchor.size():', anchor.size())
        #print('(anchor - positive).pow(2).size():', (anchor - positive).pow(2).size())
        #print('distance_positive.size():', distance_positive.size())
        return losses.mean() if size_average else losses.sum()

class GradedMicroF1Loss(nn.Module):
    def __init__(self, smooth=1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred (Tensor): Raw logits of shape [B, C]
            y_true (Tensor): Target relevance vectors of shape [B, C]
        Returns:
            Tensor: Scalar loss value
        """
        # Apply sigmoid to convert logits to probabilities
        y_pred = torch.sigmoid(y_pred)

        # Optional: normalize to unit vectors for cosine similarity instead
        # y_pred = F.normalize(y_pred, dim=1)
        # y_true = F.normalize(y_true, dim=1)

        # Soft true positives
        tp = (y_pred * y_true).sum(dim=1)
        fp = (y_pred * (1 - y_true)).sum(dim=1)
        fn = ((1 - y_pred) * y_true).sum(dim=1)

        precision = tp / (tp + fp + self.smooth)
        recall = tp / (tp + fn + self.smooth)
        f1 = 2 * precision * recall / (precision + recall + self.smooth)

        return 1.0 - f1.mean()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector, negative_compatibles_dict, print_interval=0):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        self.negative_compatibles_dict = negative_compatibles_dict
        if print_interval < 1:
            self.print_interval = float('inf')
            self.print_counter = 1
        else:
            self.print_interval = print_interval
            self.print_counter = 0
        
        self.print_counter += 1

    def forward(self, query_embeddings, query_target, db_embeddings, db_target):
        print_log = False
        if self.print_counter % self.print_interval == 0:
            print_log = True
        self.print_counter += 1
        if print_log:
            print('OnlineTripletLoss.forward()')
            k = min(10, len(query_target))
            if k < len(query_target):
                print(f'(Dataset elements are of length {len(query_target)} and {len(db_target)} which is too long! Printing only the first {k} elements of each.)')
            print(f'query_embeddings[:{k}]:')
            print(query_embeddings[:k])
            print(f'query_target[:{k}]:')
            print(query_target[:k])
            print(f'db_embeddings[:{k}]:')
            print(db_embeddings[:k])
            print(f'db_target[:{k}]:')
            print(db_target[:k])
            
        triplets = self.triplet_selector.get_triplets(query_embeddings, query_target, db_embeddings, db_target, self.negative_compatibles_dict, print_log)

        if query_embeddings.is_cuda or db_embeddings.is_cuda:
            #triplets = [t.cuda() for t in triplets]
            triplets = triplets.cuda()

        ap_distances = (db_embeddings[triplets[:, 0]] - db_embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (db_embeddings[triplets[:, 0]] - db_embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        

        #ap_sims = F.cosine_similarity(embeddings[triplets[:, 0]], embeddings[triplets[:, 1]], dim=-1)
        #an_sims = F.cosine_similarity(embeddings[triplets[:, 0]], embeddings[triplets[:, 2]], dim=-1)
        #losses = F.relu(self.margin + an_sims - ap_sims)

        if print_log:
            print('OnlineTripletLoss.forward()')
            k = min(10, len(losses))
            if k < len(losses):
                print(f'(Loss elements are of length {len(losses)} which is too long! Printing only the first {k} items of each element.)')
            print(f'label_triplets[:{k}]:\n', triplets[:k])
            #print(f'an_sims[:{k}]:\n', an_sims[:k])
            #print(f'ap_sims[:{k}]:\n', ap_sims[:k])
            #print(f'F.relu(self.margin + an_sims[:{k}] - ap_sims[:{k}]):\n', losses)
            print(f'ap_distances[:{k}]:\n', ap_distances[:k])
            print(f'an_distances[:{k}]:\n', an_distances[:k])
            print(f'F.relu(ap_distances[:{k}] - an_distances[:{k}] + self.margin):\n', losses)
        
        #losses_per_label_mean = [l.mean() for l in losses]
        #return losses_per_label_mean, triplets
        return losses.mean(), triplets
