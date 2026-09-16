# Cats vs Dogs Classifier — CNN with QKV Self-Attention

An end-to-end image-classification project built with TensorFlow/Keras. The model classifies an uploaded image as a **cat** or **dog** using a convolutional neural network (CNN) enhanced with a custom Query-Key-Value (QKV) self-attention layer. A small Streamlit web interface makes the trained model easy to try with new images.

This project was built to practise the complete machine-learning workflow: loading real-world data, preprocessing images, training and evaluating a neural network, saving the trained model, and serving predictions through a simple user-facing application.

## Highlights

- **Binary image classification** for cats and dogs using approximately 25,000 images.
- **Custom QKV self-attention** layer added to CNN features through a residual connection.
- **Reproducible data split** using a fixed random seed.
- **Robust data pipeline** with corrupted-image handling and prefetching.
- **Interactive Streamlit interface** for uploading a photo and receiving a prediction with confidence scores.
- **Training diagnostics**: accuracy/loss curves, confusion matrix, and sample predictions.

## Dataset

- **Source:** [Microsoft Cats vs Dogs Dataset on Kaggle](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset), based on Microsoft's Asirra dataset.
- **Classes:** `Cat` and `Dog`. Labels are created from the image-folder names, in alphabetical order (`Cat = 0`, `Dog = 1`).
- **Size:** roughly 25,000 images and approximately balanced classes.
- **Image size:** every image is resized to **128 × 128** pixels.
- **Split:** 80% training and 20% validation, with `SEED = 42` for repeatability.
- **Preprocessing:** pixel values are rescaled from 0–255 to 0–1 inside the model. Training images are randomly flipped and slightly rotated for augmentation.

Some images in the source dataset are corrupted. The `tf.data` pipeline skips those files so that training can continue without failing.

## Model Architecture

```text
Input image (128 × 128 × 3)
  → Rescaling + training-only augmentation
  → Conv2D(32)  → BatchNorm → MaxPool
  → Conv2D(64)  → BatchNorm → MaxPool
  → Conv2D(128) → BatchNorm → MaxPool
  → QKV self-attention + residual/skip connection
  → Global Average Pooling
  → Dense(128) → BatchNorm → Dropout
  → Dense(1, sigmoid) → Cat or Dog
```

### Why these choices?

| Choice | Reason |
|---|---|
| CNN feature extractor | Convolutions learn useful local patterns such as edges, fur texture, ears, and faces. |
| QKV self-attention | Lets each spatial feature location weigh information from other locations, helping the model consider relationships across an image. |
| Attention after three pooling stages | The feature map is reduced to 16 × 16 (256 locations), making attention practical without the cost of applying it directly to all image pixels. |
| Residual connection | Adds attended features back to the original CNN features, preserving useful local information. |
| `GlobalAveragePooling2D` | Reduces model size compared with `Flatten`, which helps limit overfitting. |
| Batch normalization and dropout | Improve training stability and help regularize the classifier. |
| In-model rescaling | Ensures training and Streamlit inference use the same input normalization. |

## How QKV Attention Works

The custom layer produces three learned projections of the CNN feature map:

- **Query (Q):** what a location is looking for.
- **Key (K):** what information each location offers.
- **Value (V):** the information passed forward.

The layer compares every query with every key, turns those similarity scores into attention weights with softmax, and uses the weights to combine the values. The result lets the network emphasise the regions that are most useful for its decision, rather than treating each region independently.

Because `QKVAttentionLayer` is a custom Keras layer, its definition is required both when the model is trained and when the Streamlit app reloads the saved `.keras` model. The app loads it with `custom_objects` and then uses the learned attention weights during inference; it does not train the model again.

## Training Setup

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Loss function | Binary cross-entropy |
| Batch size | 16 |
| Reported experiment | 10 epochs |
| Random seed | 42 |

### Accuracy and loss

<img width="1200" height="500" alt="Training and validation accuracy and loss curves" src="https://github.com/user-attachments/assets/1e1f133e-34f8-44b1-bb52-f2f46b874812" />

## Results

Results below are from the reported 10-epoch training run.

| Metric | Result |
|---|---:|
| Training accuracy | 81.95% |
| Validation accuracy | 82.49% |
| Validation loss | 0.387 |

### Confusion matrix

<img width="600" height="500" alt="Confusion matrix for cat and dog predictions" src="https://github.com/user-attachments/assets/6fb142cc-2f4a-44ee-b4fa-731ccff9e4f2" />

### Example validation predictions

<img width="1200" alt="Example cat and dog predictions" src="https://github.com/user-attachments/assets/53217702-5947-4826-98ba-f1e055f683f3" />

## Streamlit Web App

The app, **Paws & Predict**, loads the trained model once when it starts. A visitor can upload a JPG, PNG, or WEBP image and receive:

- the predicted class (`Cat` or `Dog`),
- the model confidence, and
- the probability for both classes.

The interface has a compact preview so that the uploaded image and result remain visible together.

### Web app screenshots

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/ae04fbd3-5657-4e9d-87a7-e1f7c6c8c32b" />

When the images are ready, add them below:

<!--
<p align="center">
  <img src="docs/images/streamlit-upload.png" width="45%" alt="Paws and Predict upload screen" />
  <img src="docs/images/streamlit-prediction.png" width="45%" alt="Paws and Predict prediction result" />
</p>
-->

## Run Locally

### 1. Clone and install dependencies

```bash
git clone https://github.com/Abderrahmentmr/attention_mechanism_cnn_classifier.git
cd attention_mechanism_cnn_classifier
python -m venv .venv
```

Activate the environment, then install the requirements:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r files/requirements.txt
```

### 2. Train and save the model

```bash
python files/attention_cnn_classifier.py
```

The script downloads the Kaggle dataset, trains the network, saves `cats_dogs_model.keras` in the repository root, and displays the evaluation plots. Kaggle access requires a free account and API credentials; see the [Kaggle API documentation](https://www.kaggle.com/docs/api).

### 3. Start the web app

```bash
streamlit run files/streamlit_app.py
```

Open the local URL shown by Streamlit, upload an image, and view the prediction.

## Project Structure

```text
attention_mechanism_cnn_classifier/
├── README.md                         # Project overview, results, and setup
├── docs/
│   └── images/                       # Streamlit screenshots for this README
└── files/
    ├── attention_cnn_classifier.py   # Data loading, model training, and evaluation
    ├── streamlit_app.py               # Streamlit inference interface
    ├── requirements.txt               # Python dependencies
    └── .gitignore                     # Dataset, model, environment, and cache exclusions
```

The downloaded dataset and trained `.keras` model are intentionally excluded from version control. This keeps the repository light and avoids committing large generated artifacts; running the training script recreates the model locally.

## Limitations and Next Steps

- Add a held-out test set for final, unbiased evaluation.
- Compare the attention-augmented CNN with a CNN baseline to measure the attention layer's contribution.
- Add `EarlyStopping` and `ModelCheckpoint` callbacks to retain the best validation model.
- Report precision, recall, F1-score, and ROC-AUC in addition to accuracy.
- Visualise attention maps to make the model's focus easier to interpret.
- Package the app for cloud deployment and store the trained model as a release or deployment artifact.

## Author

Tamamra Abderrahmane
