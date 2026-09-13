# Cats vs Dogs Classifier — CNN with Self-Attention

A convolutional neural network that classifies images as **cat** or **dog**, built in TensorFlow/Keras. It uses a CNN to extract features from the images, plus a custom self-attention layer that helps the model focus on the important parts of the image instead of treating every pixel equally.

## Dataset

- **Source:** [Kaggle — Microsoft Cats vs Dogs Dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset), based on Microsoft's original Asirra dataset.
- **Classes:** `Cat`, `Dog`, labeled automatically from folder names.
- **Size:** ~25,000 images, roughly balanced. A handful of files are corrupted and get skipped automatically during loading.
- **Image size:** everything resized to 128×128.
- **Split:** 80/20 train/validation with a fixed seed for reproducibility.
- **Preprocessing:** pixels rescaled to 0–1 inside the model; during training only, images are randomly flipped and slightly rotated to cut down on overfitting.

Right now I'm only doing a train/val split — no separate test set yet. I'll add one later (see [Future Improvements](#future-improvements)).

## Data Loading

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

This loads images from the `Cat`/`Dog` subfolders and uses the folder names as labels. Shuffling before the split matters — otherwise one subset can end up mostly one class.

## Model Architecture

```
Input Image (128×128×3)
  → Rescaling + augmentation (flip, rotate)
  → Conv2D(32) → BatchNorm → MaxPool
  → Conv2D(64) → BatchNorm → MaxPool
  → Conv2D(128) → BatchNorm → MaxPool
  → Self-attention (Q/K/V), added back via a skip connection
  → Global Average Pooling
  → Dense(128) → BatchNorm → Dropout
  → Dense(1, sigmoid) → cat / dog
```

The conv layers pull out edges, textures, and shapes as usual. The self-attention layer sits on top and lets the model weigh which regions of the feature map actually matter — the animal's face, say — rather than scanning the whole image with equal weight. I add its output back onto the original features with a skip connection so nothing the CNN already learned gets lost, then pool and run it through a couple of dense layers to get a single sigmoid output (closer to 0 = cat, closer to 1 = dog).

## Training

| Setting | Value |
|---|---|
| Optimizer | Adam (learning rate 0.0005) |
| Loss function | Binary cross-entropy |
| Batch size | 16 |
| Epochs | 10 |

**Accuracy / loss over training:**

<img width="1200" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/1e1f133e-34f8-44b1-bb52-f2f46b874812" />

## Results

Final epoch:

| Metric | Result |
|---|---|
| Training accuracy | 81.95% |
| Validation accuracy | 82.49% |
| Validation loss | 0.387 |

**Confusion matrix:**

<img width="600" height="500" alt="Figure_CM" src="https://github.com/user-attachments/assets/6fb142cc-2f4a-44ee-b4fa-731ccff9e4f2" />

## Example Predictions

<img width="1536" height="752" alt="Figure_Test RES" src="https://github.com/user-attachments/assets/53217702-5947-4826-98ba-f1e055f683f3" />

## About the Attention Mechanism

The attention layer computes three projections of the feature map — Query, Key, and Value — then uses them to work out how much each spatial location should attend to every other location. Basically, it lets the model learn where to look instead of processing every region with the same importance.

## How to Run It

```bash
git clone https://github.com/Abderrahmentmr/attention_mechanism_cnn_classifier.git
cd attention_mechanism_cnn_classifier
pip install tensorflow kagglehub scikit-learn matplotlib seaborn numpy
python main.py
```

The dataset downloads automatically the first time via `kagglehub` (needs a free Kaggle account and API key — see [Kaggle's API docs](https://www.kaggle.com/docs/api)).

## Future Improvements

- Add a proper train/validation/test split for a cleaner final evaluation.
- Save the trained model (`model.save(...)`) and deploy it behind a simple web app for live predictions.
- Visualize the attention maps to show where the model is actually looking.

---

## Author
Tamamra Abderrahmane
