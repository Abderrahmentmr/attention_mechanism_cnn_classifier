# Attention-Augmented CNN for Cat vs. Dog Classification

A small convolutional neural network for binary image classification (cats vs. dogs, sampled from CIFAR-10), enhanced with a **custom QKV self-attention mechanism layer** inserted between convolutional blocks.

## Overview

Standard CNNs rely purely on local convolutional filters, which can struggle to relate distant regions of an image. This project adds a lightweight **query-key-value (QKV) self-attention mechanism** (the same core idea behind Transformers) directly on top of convolutional feature maps, letting the network learn which spatial regions of an image are most relevant to each other before making a prediction.

**Architecture:**
- Conv2D (64 filters) → BatchNorm → MaxPooling
- Conv2D (128 filters) → BatchNorm
- **Custom QKV self-attention block** (residual: attended features added back to the original feature map)
- MaxPooling → Flatten
- Dense(128) → BatchNorm → Dropout
- Dense(1, sigmoid) — binary output

## Dataset

- Source: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), filtered to classes `cat` and `dog` only.
- Currently uses a subset (700 train / 150 validation images) with on-the-fly data augmentation (rotation, shift, horizontal flip).

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
python attention_cnn_classifier.py
```

This will:
1. Download CIFAR-10 (via `tf.keras.datasets`) and filter cat/dog images.
2. Train the attention-augmented CNN for 50 epochs.
3. Plot training/validation accuracy and loss curves.
4. Generate a confusion matrix on the validation set.
5. Display sample predictions.

## Results

*(Add your accuracy/loss curves and confusion matrix screenshots here once you have a run you're happy with — this is the first thing recruiters/reviewers will look at.)*

| Metric | Value |
|---|---|
| Validation accuracy | TBD |
| Validation loss | TBD |

## Status & Roadmap

This is an early, working version of the project. Planned improvements:

- [ ] Train on the full cat/dog subset of CIFAR-10 (not just 1000 images) with a proper train/val/test split
- [ ] optimise it to be able to use/read and classify self uploaded data
- [ ] Replace `ImageDataGenerator` with `tf.data` + Keras preprocessing layers
- [ ] Add `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau` callbacks
- [ ] Add precision/recall/F1/ROC-AUC metrics
- [ ] Visualize the learned attention maps over input images (interpretability)
- [ ] Compare against a baseline CNN without the attention block
- [ ] Refactor into modules (`data.py`, `model.py`, `train.py`, `evaluate.py`)

## Why this project?

This was built to explore how self-attention mechanisms can be adapted to convolutional feature maps for small-scale image classification, and to understand their effect on both accuracy and interpretability.

## Author

TAMAMRA Abderrahmane

