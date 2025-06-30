from eval.metric_loader import initialize_metrics
from utils.config import load_config
from utils.seed import set_seed
import torch
from training.setup import setup_training_run
(
    model,
    loss_fn,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    train_eval_loader,
    val_eval_loader,
    full_train_loader,
    full_val_loader,
    run_dirs,
    log_interval,
    start_epoch,
) = setup_training_run(cfg, cuda)
from eval.metric_loader import load_metrics
from training.trainer import Trainer

metric_cfg_list = cfg["training"].get("metrics", [])
metrics = initialize_metrics(metric_cfg_list)

cfg = load_config("configs/base.yaml")

set_seed(cfg["training"]["seed"])

cuda = torch.cuda.is_available()

metrics = load_metrics(cfg)

trainer = Trainer(
    train_loader,
    val_loader,
    train_eval_loader,
    val_eval_loader,
    full_train_loader,
    full_val_loader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    cfg["training"]["n_epochs"],
    cuda,
    log_interval,
    run_dirs["checkpoints"],
    run_dirs["logs"],
    cfg["training"]["train_full_loader_switch"],
    metrics,
    start_epoch,
)

trainer.fit()
