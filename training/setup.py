import os
from datetime import datetime

def setup_training_run(cfg):
    """
    Creates timestamped run directory structure:
    - runs/run_<timestamp>/
        - checkpoints/
        - logs/
    
    Returns:
        dict with paths: {'run_dir', 'checkpoints', 'logs'}
    """
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M')
    run_subdir = f"run_{timestamp_str}"
    base_runs_path = cfg["paths"]["runs"]
    run_dir = os.path.join(base_runs_path, run_subdir)
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    logs_dir = os.path.join(run_dir, 'logs')

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    return {
        "run_dir": run_dir,
        "checkpoints": checkpoints_dir,
        "logs": logs_dir
    }
