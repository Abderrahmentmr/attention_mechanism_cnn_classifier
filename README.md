# Cats vs Dogs Classifier / CNN with Self-Attention

A convolutional neural network that classifies images as **cat** or **dog**, built in TensorFlow/Keras.It uses a CNN to extract features from the images and a custom self-attention mechanism to help the model focus on important parts of the image.

---

##  Dataset

- **Source:** [Kaggle — Microsoft Cats vs Dogs Dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset), based on Microsoft's original Asirra dataset.
- **Classes:** `Cat`, `Dog` — labeled automatically from folder names.
- **Size:** ~25,000 images total, roughly balanced between the two classes. A small number of files are corrupted and are automatically skipped during loading.
- **Image size:** all images resized to **128 × 128** pixels.
- **Split:** 80% training / 20% validation, using a fixed random seed so the split is reproducible.
- **Preprocessing:** pixel values are rescaled to the 0–1 range inside the model. During training only, images are randomly flipped and slightly rotated to reduce overfitting.

> Note: this version uses a train/validation split only. A held-out test set would make the final evaluation stronger — see [Future Improvements](#9-future-improvements).

---

##  Data Loading

```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    image_folder,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
)
```

This function scans the dataset folder, treats each subfolder (`Cat`, `Dog`) as a class label, and returns ready-to-use batches of `(image, label)` pairs. Shuffling **before** splitting is important here — it ensures both classes are properly mixed into both the training and validation sets, instead of one subset accidentally ending up with mostly one class.

---

##  Model Architecture

```
Input Image (128×128×3)
        ↓
Rescaling + Augmentation (flip, rotate)
        ↓
Conv2D (32) → BatchNorm → MaxPool
        ↓
Conv2D (64) → BatchNorm → MaxPool
        ↓
Conv2D (128) → BatchNorm → MaxPool
        ↓
   ┌─────────────────┐
   │  Self-Attention  │
   │      (QKV)       │
   └─────────────────┘
        ↓  (added back via skip connection)
Global Average Pooling
        ↓
Dense (128) → BatchNorm → Dropout
        ↓
Dense (1, sigmoid) → Cat / Dog
```

**Why this design:**
- **Conv2D layers** extract visual features/edges, textures, shapes.
- **Self-attention layer** lets the model weigh which regions of the feature map matter most (e.g. the animal's face) instead of treating the whole image equally.
- **Skip connection** adds the attention output back onto the original features, so useful information from the CNN isn't lost.
- **Dense + sigmoid** turns the final features into a single probability: closer to 0 means cat, closer to 1 means dog.

---

## Training

| Setting | Value |
|---|---|
| Optimizer | Adam (learning rate 0.0005) |
| Loss function | Binary cross-entropy |
| Batch size | 16 |
| Epochs | 10 |

**Accuracy / loss over training:**

<img width="1200" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/1e1f133e-34f8-44b1-bb52-f2f46b874812" />




---

## Results

Final epoch:

| Metric | Result |
|---|---|
| Training accuracy | 81.95% |
| Validation accuracy | 82.49% |
| Validation loss | 0.387 |

**Confusion matrix:**

<img width="600" height="500" alt="Figure_CM" src="https://github.com/user-attachments/assets/6fb142cc-2f4a-44ee-b4fa-731ccff9e4f2" />



---

##  Example Predictions

<img width="1536" height="752" alt="Figure_Test RES" src="https://github.com/user-attachments/assets/53217702-5947-4826-98ba-f1e055f683f3" />

---

##  About the Attention Mechanism

The attention layer computes three projections of the feature map **Query**, **Key**, and **Value** ,then uses them to figure out how much each spatial location should "pay attention to" every other location. In plain terms: it lets the model learn *where to look* in the image, rather than processing every pixel region with equal importance.


---

## How to Run It

```bash
git clone <your-repo-url>
cd <your-repo>
pip install tensorflow kagglehub scikit-learn matplotlib seaborn numpy
python main.py
```

The dataset downloads automatically the first time via `kagglehub` (requires a free Kaggle account and API key — see [Kaggle's API docs](https://www.kaggle.com/docs/api)).

---

## Future Improvements

- Add a proper train / validation / **test** split for a cleaner final evaluation.
- Save the trained model (`model.save(...)`) and deploy it behind a simple web app so anyone can upload an image and get a live prediction.
- Visualize the attention maps to show *where* the model is looking.


---
## Author
Tamamra Abderrahmane
