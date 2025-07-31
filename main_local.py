import torch
from config.config import load_config
from utils.seed import set_seed
from training.setup import setup_training_run
from training.environment import configure_environment
from eval.metric_loader import load_metrics
from training.data_setup import load_dataset, create_loaders
from training.model_setup import initialize_model
from training.trainer_local import Trainer


cuda_available = torch.cuda.is_available()

cfg = load_config("config/base_local.yaml")

set_seed(cfg["training"]["seed"])

run_dirs = setup_training_run(cfg["paths"]["dr2156"]["triplet_runs"])

checkpoints_dir = run_dirs["checkpoints"]
tensorboard_dir = run_dirs["logs"]

configure_environment(cfg)

train_set, test_set, neg_compatibles = load_dataset(
    cfg["paths"]["dr2156"]["preprocessed_300_int8"], 
    cfg["training"]["seed"], 
    float(cfg["dataset"]["train_frac"]),
    augmentations_arg=cfg["training"]["augmentations"]
)

p_model, p_loss_fn, p_optimizer, p_scheduler = initialize_model(
    embedding_size=int(cfg["model"]["embedding_size"]),
    margin=float(cfg["loss"]["margin"]),
    lr=float(cfg["training"]["optimizer"]["lr"]),
    weight_decay=float(cfg["training"]["optimizer"]["weight_decay"]),
    negative_compatibles_dict=neg_compatibles,
    print_interval=int(cfg["logging"]["log_interval"]),
    cuda=cuda_available
)

loaders = create_loaders(
    train_set,
    test_set,
    cfg["training"]["batch"]["n_classes"],
    cfg["training"]["batch"]["n_samples"],
    cuda_available
)

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
    train_full_loader_switch=cfg["training"]["train_full_loader_switch"],
    metrics=p_metrics,
    start_epoch=0,
    accumulation_steps=3
)
trainer.fit()
