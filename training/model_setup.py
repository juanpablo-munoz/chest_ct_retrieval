from torch import optim
from torch.optim import lr_scheduler

from chest_ct_retrieval.models.networks import Proximity300x300
from chest_ct_retrieval.losses.losses import OnlineTripletLoss
from chest_ct_retrieval.utils.selectors.triplet_selector import SemihardNegativeTripletSelector

def initialize_model(embedding_size, margin, lr, weight_decay, negative_compatibles_dict, print_interval, cuda):
    model = Proximity300x300(embedding_size=embedding_size)
    if cuda:
        model = model.cuda()

    loss_fn = OnlineTripletLoss(
        margin=margin,
        triplet_selector=SemihardNegativeTripletSelector(margin),
        negative_compatibles_dict=negative_compatibles_dict,
        print_interval=print_interval,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    return model, loss_fn, optimizer, scheduler
