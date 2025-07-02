from datetime import datetime
import os
import torch
from config.config import load_config
from eval.metric_loader import load_metrics
from training.data_setup import load_dataset, create_loaders
from training.model_setup import initialize_model
from training.setup import setup_training_run
from training.trainer import Trainer
from utils.seed import set_seed



cfg = load_config("configs/base.yaml")

set_seed(cfg["training"]["seed"])

cuda_available = torch.cuda.is_available()

run_dirs = setup_training_run(cfg)

checkpoints_dir = run_dirs["checkpoints"]
tensorboard_dir = run_dirs["logs"]

train_set, test_set, neg_compatibles = load_dataset(cfg["volume_dir"], cfg["seed"], cfg["train_fraction"])

p_model, p_loss_fn, p_optimizer, p_scheduler = initialize_model(
    embedding_size=cfg["embedding_size"],
    margin=cfg["margin"],
    lr=cfg["learning_rate"],
    weight_decay=cfg["weight_decay"],
    negative_compatibles_dict=neg_compatibles,
    print_interval=10,
    cuda=cuda
)
loaders = create_loaders(train_set, test_set, cfg["n_classes"], cfg["n_samples"], cuda)

p_metrics = load_metrics(cfg)

trainer = Trainer(
    train_loader=loaders["train_triplet"],
    val_loader=loaders["test_triplet"],
    train_eval_loader=loaders["train_eval"],
    val_eval_loader=loaders["test_eval"],
    train_full_loader=loaders["all_triplet_train"],
    val_full_loader=loaders["all_triplet_test"],
    model=p_model,
    loss_fn=p_loss_fn,
    optimizer=p_optimizer,
    scheduler=p_scheduler,
    n_epochs=cfg["training"]["n_epochs"],
    cuda=cuda_available,
    log_interval=cfg["logging"]["log_interval"],
    checkpoint_dir=checkpoints_dir,
    tensorboard_logs_dir=tensorboard_dir,
    train_full_loader_switch=cfg["train_full_loader_switch"],
    metrics=p_metrics,
    start_epoch=0,
)
trainer.fit()
