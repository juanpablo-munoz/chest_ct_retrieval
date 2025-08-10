from torch import optim
from torch.optim import lr_scheduler

from models.networks import Proximity100x100, Proximity300x300
from losses.losses_local import OnlineTripletLoss, GradedMicroF1Loss, HybridBCEGradedMicroF1Loss
from utils.selectors.triplet_selector import SemihardNegativeTripletSelector, HardestNegativeTripletSelector
from datasets.base import LabelVectorHelper

def initialize_model_triplets(embedding_size, margin, lr, weight_decay, negative_compatibles_dict, print_interval, cuda):
    """Initialize Proximity100x100 model in embedding mode for triplet training with optimized configuration"""
    model = Proximity100x100(embedding_size=embedding_size, num_classes=None, task='embedding')
    if cuda:
        model.cuda()

    label_vector_helper = LabelVectorHelper()

    loss_fn = OnlineTripletLoss(
        margin=margin,
        triplet_selector=SemihardNegativeTripletSelector(margin, label_vector_helper),
        negative_compatibles_dict=negative_compatibles_dict,
        print_interval=print_interval,
    )

    # Use same optimizer configuration as micro-F1 training for consistency
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    return model, loss_fn, optimizer, scheduler

def initialize_model_micro_f1(embedding_size, lr, weight_decay, cuda):
    model = Proximity100x100(embedding_size=embedding_size, num_classes=4, task='classification')
    if cuda:
        model.cuda()

    loss_fn = GradedMicroF1Loss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    return model, loss_fn, optimizer, scheduler

def initialize_model_bce_micro_f1(embedding_size, lr, weight_decay, y_true_all, cuda):
    model = Proximity100x100(embedding_size=embedding_size, num_classes=4, task='classification')
    if cuda:
        model.cuda()

    loss_fn = HybridBCEGradedMicroF1Loss(y_true_all, alpha=0.5, beta=0.5, smooth=1e-7)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    return model, loss_fn, optimizer, scheduler
