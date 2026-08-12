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
## Why This Project?

Built to explore how self-attention (the mechanism behind Transformers) can be adapted to convolutional feature maps for small-scale image classification, and to see its effect on both accuracy and interpretability.

## Author

TAMAMRA Abderrahmane


