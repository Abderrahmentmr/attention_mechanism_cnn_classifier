# Attention-Augmented CNN for Cat vs. Dog Classification

A small CNN for binary image classification (cats vs. dogs), enhanced with a **custom QKV self-attention layer** inserted between convolutional blocks — the same core mechanism behind Transformers, adapted here to convolutional feature maps.

## Building the Model with TensorFlow

**1. Load and preprocess the data.** Images are loaded from the Cats vs. Dogs dataset, resized, and normalized to `[0, 1]`.

**2. Build the CNN.** Using TensorFlow's Functional API, the network stacks:
- [`Conv2D`](https://www.geeksforgeeks.org/deep-learning/convolutional-layer/) blocks with [ReLU activation](https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/) and `BatchNormalization`
- A custom **QKV self-attention layer** (see below) placed between conv blocks
- `Dense` layers with `Dropout` for regularization
- A final `Dense(1)` layer with [sigmoid activation](https://www.geeksforgeeks.org/deep-learning/sigmoid-activation-function/) for binary output

**3. Add self-attention.** Standard CNNs only look at local neighborhoods (whatever fits inside a filter's kernel), so they can struggle to relate distant parts of an image. The custom `QKVAttentionLayer` computes Query, Key, and Value projections from the feature map, then lets every spatial position attend to every other position — helping the network learn *which regions matter to each other* before making a prediction. Its output is added back to the original features (a residual connection), so attention refines the representation rather than replacing it.

**4. Compile and train.** The model is compiled with the [Adam optimizer](https://www.geeksforgeeks.org/deep-learning/adam-optimizer/) and [binary crossentropy loss](https://www.geeksforgeeks.org/deep-learning/binary-cross-entropy-log-loss-for-binary-classification/), then trained with data augmentation (random flips, rotation, translation) and `EarlyStopping`/`ReduceLROnPlateau` callbacks.

**5. Install dependencies:**
```bash
pip install -r requirements.txt
```

**6. Run it:**
```bash
python attention_cnn_classifier_colab.py
```
This will download the dataset, train the model, plot accuracy/loss curves, and generate a confusion matrix + sample predictions on the test set.

## Architecture at a Glance

```
Input (96×96×3)
  → Conv2D(64) → BatchNorm → MaxPooling
  → Conv2D(128) → BatchNorm
  → QKV Self-Attention (residual add)
  → MaxPooling → Flatten
  → Dense(128) → BatchNorm → Dropout
  → Dense(1, sigmoid)
```

## Results

<img width="1268" height="589" alt="image" src="https://github.com/user-attachments/assets/e7b32afc-f52b-4f74-8db9-1445453f28d7" />

<img width="633" height="560" alt="image" src="https://github.com/user-attachments/assets/0e2f3da8-edf9-4041-8f76-4566bed3fc65" />


## Roadmap

- [ ] Train on the full dataset with a proper train/val/test split
- [ ] Support classifying self-uploaded images
- [ ] Add precision/recall/F1/ROC-AUC metrics
- [ ] Visualize attention maps over input images (interpretability)
- [ ] Compare against a baseline CNN without attention
- [ ] Refactor into modules (`data.py`, `model.py`, `train.py`, `evaluate.py`)

## Why This Project?

Built to explore how self-attention — the mechanism behind Transformers — can be adapted to convolutional feature maps for small-scale image classification, and to see its effect on both accuracy and interpretability.

## Author

TAMAMRA Abderrahmane
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

<img width="1268" height="589" alt="image" src="https://github.com/user-attachments/assets/e7b32afc-f52b-4f74-8db9-1445453f28d7" />

<img width="633" height="560" alt="image" src="https://github.com/user-attachments/assets/0e2f3da8-edf9-4041-8f76-4566bed3fc65" />


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

