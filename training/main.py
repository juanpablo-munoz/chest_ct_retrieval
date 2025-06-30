import os
from datetime import datetime
from training.trainer import Trainer
from training.data_setup import prepare_training_data
from training.setup import setup_training_run

def main(cfg):
    data = prepare_training_data(cfg)
    (
        triplet_train_loader, triplet_test_loader,
        train_eval_loader, test_eval_loader,
        all_triplet_train_loader, all_triplet_test_loader,
        model, loss_fn, optimizer, scheduler
    ) = data

    run_dirs = setup_training_run(cfg)
    run_checkpoints_subdir = run_dirs['checkpoints']
    run_logs_subdir = run_dirs['logs']

    trainer = Trainer(
        triplet_train_loader,
        triplet_test_loader,
        train_eval_loader,
        test_eval_loader,
        all_triplet_train_loader,
        all_triplet_test_loader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        cfg["training"]["n_epochs"],
        cfg["training"]["cuda"],
        cfg["training"]["log_interval"],
        run_checkpoints_subdir,
        run_logs_subdir,
        cfg["training"].get("train_full_loader_switch", False),
        metrics=cfg["training"]["metrics"],
        start_epoch=cfg["training"].get("start_epoch", 0),
    )

    trainer.fit()
