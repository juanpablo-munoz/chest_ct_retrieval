# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research thesis project that implements a chest CT image retrieval system using both triplet learning and micro-F1 classification for medical image similarity. Its main features are:

- **Dual Training Modes**: Supports both triplet learning and micro-F1 classification approaches
- **Adaptive 3D CNN Architecture**: Proximity100x100 model with automatic layer size calculation
- **GPU-Optimized Pipeline**: Kornia-based augmentations and mixed precision training
- **Medical Dataset**: Works with DR2156 chest CT scans with multi-class medical anomaly labels
- **Advanced Evaluation**: Comprehensive metrics including F1-scores, mAP@k, and medical relevance scoring

## Core Architecture

### Main Components

- **`models/networks.py`**: Neural network architectures
  - `Proximity100x100`: Adaptive model with automatic flattened_size calculation and dual-task support (embedding/classification)
  - `Proximity300x300`: Legacy model for 300x300 fixed inputs
  - Built-in `get_embeddings()` method with mixed precision support

- **`training/`**: Dual training pipeline system
  - `trainer_local_microf1.py`: Micro-F1 classification training with separate train_eval_loader for faster embedding extraction
  - `data_setup_local.py`: GPU-optimized data loading with Kornia augmentations and memory-efficient preprocessing
  - `model_setup_local.py`: Model initialization supporting both training modes

- **`losses/losses_local.py`**: Enhanced loss functions
  - `GradedMicroF1Loss`: Differentiable micro-F1 loss with sigmoid activations
  - `OnlineTripletLoss`: Improved triplet mining with compatibility checks

- **`eval/metrics.py`**: Comprehensive evaluation system
  - `AllMetrics`: Unified metric calculation (Precision@k, Recall@k, mAP@k, NDCG@k, F1-scores)
  - Real-time TensorBoard logging with PR curves
  - Medical relevance scoring using Jaccard similarity

- **`datasets/`**: Enhanced data handling
  - `ct_volume_dataset_local.py`: Optimized dataset classes with memory-efficient loading
  - Support for stratified train/eval splits and uint8 volume storage

### Configuration System

- **`config/base_local.yaml`**: Micro-F1 training configuration
  - Model: Proximity100x100 with 1024-dim embeddings
  - Data: 270x270 preprocessed volumes (uint8 format)
  - Training: Mixed precision with GPU augmentations
  - `train_eval_frac`: Separate evaluation subset size (default: 0.1)

- **`config/base.yaml`**: Traditional triplet learning configuration
  - Model: Proximity300x300 with balanced batch sampling
  - Training: n_classes × n_samples batch structure

## Common Development Commands

### Training Commands

```bash
# Micro-F1 classification training (recommended)
python main.py --mode microf1 --config config/base_local.yaml

# Traditional triplet learning
python main.py --mode triplet --config config/base.yaml

# Disable GPU optimizations for debugging
python main.py --mode microf1 --no-optimized-loaders
```

### Key Training Parameters

**Micro-F1 Mode (config/base_local.yaml):**
- `model.embedding_size`: 1024 (high-dimensional embeddings)
- `training.batch.batch_size`: 8 (simple batch structure)
- `dataset.train_eval_frac`: 0.1 (subset for embedding extraction)
- `paths.dr2156.preprocessed_270_uint8`: Memory-efficient uint8 volumes

**Triplet Mode (config/base.yaml):**
- `training.batch.n_classes`: 2 (classes per batch)
- `training.batch.n_samples`: 3 (samples per class)
- `loss.margin`: 0.2 (triplet loss margin)

## Data Pipeline Architecture

### Micro-F1 Pipeline
```
Raw DICOM → 270x270x300 uint8 volumes → GPU Augmentation (Kornia) → 
ResNet18 Features → 3D CNN Reduction → 1024-dim Embeddings → 
Classification Head → Micro-F1 Loss
```

### GPU-Based Augmentation
```python
gpu_aug = K.AugmentationSequential(
    K.RandomAffine3D(degrees=(5, 5, 5), scale=(0.95, 1.05), p=0.5),
    RandomGaussianNoise3D(mean=0.0, std=0.01, p=0.5)
).to("cuda")
```

## Advanced Training Features

### Two-Phase Training (Micro-F1 Mode)
1. **Training Phase**: Gradient updates on full training set with augmentations
2. **Embedding Phase**: Fast embedding extraction on train_eval subset (no augmentations)

### Memory Optimization
- **Mixed Precision Training**: Automatic loss scaling with GradScaler
- **Uint8 Volume Storage**: Reduced memory footprint for large datasets  
- **Gradient Accumulation**: Effective larger batch sizes with limited GPU memory
- **Separate Train-Eval Loader**: Faster metrics calculation during training

### Model Architecture Innovations
- **Automatic Layer Sizing**: Proximity100x100 calculates flattened_size based on input dimensions
- **Dual-Task Support**: Single model handles both embedding and classification tasks
- **Mixed Precision Methods**: Built-in autocast support in get_embeddings()

## Model Training Flow

### Micro-F1 Mode
1. Load configuration from `config/base_local.yaml`
2. Create stratified train/train_eval/test splits with train_eval_frac
3. Initialize Proximity100x100 model with classification head
4. Setup GradedMicroF1Loss and mixed precision training
5. Train with gradient accumulation and GPU augmentations
6. Extract embeddings on train_eval subset for metrics
7. Save checkpoints based on best micro-F1 or mAP@k scores

### Advanced Checkpointing
- **Metric-Based Selection**: Automatically saves best model based on micro-F1 > mAP@k priority
- **Timestamped Checkpoints**: Format: `microf1_YYYYMMDD_epoch=XXX_metric=X.XXXX.pth`
- **Final Checkpoint**: Guaranteed save at training completion

## Important Implementation Notes

- **Input Format**: Model expects `[batch_size, 300, 1, height, width]` with automatic resizing to 270x150x150
- **Mixed Precision**: Essential for GPU memory management with large volumes
- **Train-Eval Split**: Uses stratified subset of training data (not separate validation split)
- **GPU Augmentations**: All data preprocessing happens on GPU for maximum efficiency
- **Environment Setup**: Requires CUDA with multiprocessing spawn method for Windows compatibility
- **Development Focus**: Primary development now occurs in main.py training loops rather than notebooks

## Evaluation System

### Comprehensive Metrics
- **Retrieval Metrics**: Precision@k, Recall@k, mAP@k, NDCG@k for k=[1,3,5,10,20]
- **Classification Metrics**: Micro/Macro/Weighted F1-scores for medical anomaly detection
- **Medical Relevance**: Jaccard similarity scoring based on multi-class label vectors
- **Per-Class Analysis**: Individual class performance tracking for medical anomalies

### Real-Time Monitoring
- **TensorBoard Integration**: Automatic logging of losses, metrics, and PR curves
- **Progress Tracking**: TQDM progress bars with batch-level loss reporting
- **Memory Monitoring**: CUDA memory debugging with environment variables