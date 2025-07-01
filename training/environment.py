import os
import sys
from pathlib import Path

def configure_environment(cfg):
    # CUDA debug/debugging features
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Add project root to PYTHONPATH
    project_root = Path(cfg.paths.project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
