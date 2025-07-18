# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research thesis project that implements a chest CT image retrieval system using triplet learning for medical image similarity. Its main features are:

- **Triplet Learning**: Uses contrastive/triplet loss to learn embeddings for CT volume similarity
- **3D CNN Architecture**: Proximity300x300 model combining ResNet18 features with 3D convolutions
- **Medical Dataset**: Works with a dataset (DR2156) of chest CT scans paired with radiology reports and multi-class labels that distill medical anomalies of reports
- **Embedding-based Retrieval**: Generates embeddings for similarity-based medical image retrieval

## Core Architecture

### Main Components

- **`datasets/`**: Dataset loading and preprocessing for CT volumes
  - `ct_volume_dataset.py`: Main dataset class for preprocessed CT triplets
  - `base.py`: Label vector utilities and pair/triplet construction helpers
  - `constants.py`: Proximity vector labels and class definitions

- **`models/networks.py`**: Neural network architectures
  - `Proximity300x300`: Primary model combining ResNet18 + 3D CNN for CT volume processing

- **`losses/losses.py`**: Loss functions for metric learning
  - `TripletLoss`, `ContrastiveLoss`, `OnlineTripletLoss` implementations

- **`training/`**: Training pipeline and utilities
  - `trainer.py`: Main training loop with metrics and checkpointing
  - `data_setup.py`: Data loading and batch creation
  - `model_setup.py`: Model, optimizer, and scheduler initialization

- **`eval/`**: Evaluation metrics (NDCG, recall, custom triplet metrics)

- **`utils/`**: Utility functions including distance metrics, selectors, and embedding tools

- **`notebooks/`**: Jupyter notebooks in which experiments and development is done.
  - `notebooks/ct_embeddings_resnet.ipynb`: DICOM data preprocessing experiments and pipeline
  - `notebooks/Exp_1_TripletDR2156.ipynb`: Main training flow development
 Other notebook files contain different experiments that are not immidiately relevant but required for the final thesis report

### Configuration System

- **`config/base.yaml`**: Main configuration file defining:
  - Model parameters (embedding_size: 128, margin: 0.2)
  - Training settings (50 epochs, Adam optimizer, lr: 1e-5)
  - Data paths and preprocessing settings
  - Metrics and logging configuration

- **`config/config.py`**: YAML configuration loader

## Common Development Commands

### Training

```bash
# Main training script
python main.py
```

### Key Training Parameters (config/base.yaml)

- `model.embedding_size`: Embedding dimension (default: 128)
- `loss.margin`: Triplet loss margin (default: 0.2)
- `training.n_epochs`: Training epochs (default: 50)
- `training.batch.n_classes`: Classes per batch (default: 2)
- `training.batch.n_samples`: Samples per class (default: 4)
- `paths.dr2156.preprocessed_300`: Path to preprocessed CT data

## Data Pipeline

1. **Raw CT Data**: DICOM files in `data/DR2156/DR2156_DICOM/`
2. **Preprocessed Volumes**: 300x300 preprocessed CT slices
3. **Embeddings**: ResNet18-based embeddings for triplet construction
4. **Triplet Dataset**: Anchor-positive-negative triplets for training

## Model Training Flow

1. Load configuration from `config/base.yaml`
2. Setup training environment and directories (`runs/`)
3. Initialize dataset on the preprocessed volumes
4. Create model (Proximity300x300) with embedding layers
5. Setup triplet loss and optimizer
6. Train with batch sampling (n_classes × n_samples per batch)
7. Evaluate using NDCG, recall, and custom triplet metrics
8. Save checkpoints and TensorBoard logs

## Important Notes

- Model expects 5D input: `[batch_size, 300, 3, height, width]` (300 CT slices)
- Uses mixed precision training with autocast
- Implements online triplet mining during training
- Supports both fixed test triplets and dynamic training triplets
- Uses multi-class labels for medical relevance scoring during evaluation
- This project is under development and is still not feature-complete
- All development occurs inside Jupyter notebooks located in `notebooks/`