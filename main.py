import torch
import argparse
from config.config import load_config
from utils.seed import set_seed
from training.setup import setup_training_run
from training.environment import configure_environment
from eval.metric_loader import load_metrics
import torch.multiprocessing as mp


def run_triplet_training(cfg, cuda_available, cuda_device=None):
    """Run triplet-based training pipeline"""
    from training.data_setup_local import load_dataset, create_loaders
    from training.model_setup_local import initialize_model_triplets
    from training.trainer_local import Trainer
    
    print("=== TRIPLET TRAINING MODE ===")
    
    # Setup directories
    run_dirs = setup_training_run(cfg["paths"]["dr2156"]["triplet_runs"])
    checkpoints_dir = run_dirs["checkpoints"]
    tensorboard_dir = run_dirs["logs"]
    
    # Load dataset
    train_set, test_set, neg_compatibles = load_dataset(
        cfg["paths"]["dr2156"]["preprocessed_270_uint8"], 
        cfg["training"]["seed"], 
        float(cfg["dataset"]["train_frac"]),
        augmentations_arg=cfg["training"]["augmentations"]
    )
    
    # Initialize model
    p_model, p_loss_fn, p_optimizer, p_scheduler = initialize_model_triplets(
        embedding_size=int(cfg["model"]["embedding_size"]),
        margin=float(cfg["loss"]["margin"]),
        lr=float(cfg["training"]["optimizer"]["lr"]),
        weight_decay=float(cfg["training"]["optimizer"]["weight_decay"]),
        negative_compatibles_dict=neg_compatibles,
        print_interval=int(cfg["logging"]["log_interval"]),
        cuda=cuda_available
    )
    
    # Create loaders
    loaders = create_loaders(
        train_set,
        test_set,
        cfg["training"]["batch"]["n_classes"],
        cfg["training"]["batch"]["n_samples"],
        cuda_available
    )
    
    # Load metrics
    p_metrics = load_metrics(cfg)
    
    # Create trainer
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
        accumulation_steps=cfg.get("training", {}).get("accumulation_steps", 1)
    )
    
    return trainer


def run_microf1_training(cfg, cuda_available, cuda_device=None):
    """Run micro-F1 based training pipeline"""
    from training.data_setup_local import load_dataset_microf1, create_loaders_microf1
    from training.model_setup_local import initialize_model_micro_f1
    from training.trainer_local_microf1 import Trainer
    
    print("=== MICRO-F1 TRAINING MODE ===")
    
    # Detect Jupyter environment if not specified
    
    # Setup directories
    run_dirs = setup_training_run(cfg["paths"]["dr2156"]["microf1_runs"])
    checkpoints_dir = run_dirs["checkpoints"]
    tensorboard_dir = run_dirs["logs"]
    
    # Load dataset with device parameter
    train_set, train_eval_set, test_set, neg_compatibles = load_dataset_microf1(
        #cfg["paths"]["dr2156"]["preprocessed_300_int8"],
        cfg["paths"]["dr2156"]["preprocessed_270_uint8"],
        cfg["training"]["seed"], 
        float(cfg["dataset"]["train_frac"]),
        float(cfg["dataset"]["train_eval_frac"]),
        augmentations_arg=cfg["training"]["augmentations"]
    )
    
    # Initialize model
    p_model, p_loss_fn, p_optimizer, p_scheduler = initialize_model_micro_f1(
        embedding_size=int(cfg["model"]["embedding_size"]),
        lr=float(cfg["training"]["optimizer"]["lr"]),
        weight_decay=float(cfg["training"]["optimizer"]["weight_decay"]),
        cuda=cuda_available
    )
    
    
    loaders = create_loaders_microf1(
        train_set,
        train_eval_set,
        test_set, 
        cfg["training"]["batch"]["batch_size"],
        cuda_available
    )
    
    # Load metrics
    p_metrics = load_metrics(cfg)
    
    # Create trainer
    trainer = Trainer(
        train_loader=loaders["train"],
        train_eval_loader=loaders["train_eval"],
        val_loader=loaders["test"],
        model=p_model,
        loss_fn=p_loss_fn,
        optimizer=p_optimizer,
        scheduler=p_scheduler,
        n_epochs=cfg["training"]["n_epochs"],
        cuda=cuda_available,
        log_interval=cfg["logging"]["log_interval"],
        checkpoint_dir=checkpoints_dir,
        tensorboard_logs_dir=tensorboard_dir,
        metrics=p_metrics,
        start_epoch=0,
        accumulation_steps=cfg.get("training", {}).get("accumulation_steps", 3)
    )
    
    return trainer


def main(training_mode="triplet", config_path="config/base.yaml", optimized_loaders=False, jupyter_mode=None):
    """Main training function with configurable pipeline"""
    
    # Setup
    cuda_available = torch.cuda.is_available()
    cuda_device = torch.device(0) if cuda_available else torch.device('cpu')
    
    # Load config - support both local and standard configs
    if training_mode == "microf1" and "base_local.yaml" not in config_path:
        config_path = config_path.replace("base.yaml", "base_local.yaml")
    
    cfg = load_config(config_path)
    
    set_seed(cfg["training"]["seed"])
    configure_environment(cfg)
    
    print(f"CUDA Available: {cuda_available}")
    print(f"Device: {cuda_device}")
    print(f"Training Mode: {training_mode}")
    print(f"Config: {config_path}")
    print(f"Optimized Loaders: {optimized_loaders}")
    
    # Run appropriate training pipeline
    if training_mode == "triplet":
        trainer = run_triplet_training(cfg, cuda_available, cuda_device)
    elif training_mode == "microf1":
        trainer = run_microf1_training(cfg, cuda_available, cuda_device)
    else:
        raise ValueError(f"Unknown training mode: {training_mode}. Use 'triplet' or 'microf1'")
    
    # Start training
    trainer.fit()
    
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proximity CT Training Pipeline")
    parser.add_argument("--mode", choices=["triplet", "microf1"], default="triplet",
                        help="Training mode: triplet or microf1")
    parser.add_argument("--config", default="config/base.yaml",
                        help="Path to config file")
    parser.add_argument("--no-optimized-loaders", action="store_true",
                        help="Disable optimized data loaders")
    
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    
    main(
        training_mode=args.mode,
        config_path=args.config,
        optimized_loaders=not args.no_optimized_loaders
    )
