from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import kagglehub
from sklearn.metrics import confusion_matrix


# 1. Configuration
DATASET_NAME = "shaunthesheep/microsoft-catsvsdogs-dataset"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

tf.keras.utils.set_random_seed(SEED)


# 2. Load data
dataset_path = kagglehub.dataset_download(DATASET_NAME)
image_folder = Path(dataset_path) / "PetImages"

if not image_folder.exists():
    raise FileNotFoundError("The PetImages folder was not found in the Kaggle dataset.")

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

validation_ds = tf.keras.utils.image_dataset_from_directory(
    image_folder,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
)

class_names = train_ds.class_names

if len(class_names) != 2 or class_names != validation_ds.class_names:
    raise ValueError(
        "Both train and validation must have the same two class folders."
    )

print("Classes found:", class_names)

train_ds = train_ds.shuffle(1000, seed=SEED).ignore_errors().prefetch(tf.data.AUTOTUNE)
validation_ds = validation_ds.ignore_errors().prefetch(tf.data.AUTOTUNE)


# 3. QKV attention layer
class QKVAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.query = tf.keras.layers.Conv2D(filters, 1, padding="same")
        self.key = tf.keras.layers.Conv2D(filters, 1, padding="same")
        self.value = tf.keras.layers.Conv2D(filters, 1, padding="same")

    def call(self, inputs):
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)

        batch_size = tf.shape(query)[0]
        height = tf.shape(query)[1]
        width = tf.shape(query)[2]

        query = tf.reshape(query, [batch_size, -1, self.filters])
        key = tf.reshape(key, [batch_size, -1, self.filters])
        value = tf.reshape(value, [batch_size, -1, self.filters])

        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.filters, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(attention_weights, value)

        return tf.reshape(output, [batch_size, height, width, self.filters])


# 4. Build model
augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
    ],
    name="augmentation",
)

inputs = tf.keras.layers.Input(shape=IMAGE_SIZE + (3,))
x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
x = augmentation(x)

x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D()(x)

attention = QKVAttentionLayer(128)(x)
x = tf.keras.layers.Add()([x, attention])

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# 5. Train model
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
)


# 6. Show results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history.history["accuracy"], label="Training")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Training")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.show()


probabilities = model.predict(validation_ds, verbose=0).ravel()
predicted_labels = (probabilities >= 0.5).astype(int)

true_labels = []
for _, labels in validation_ds:
    true_labels.extend(labels.numpy().astype(int).ravel())

matrix = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(6, 5))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Prediction")
plt.ylabel("True label")
plt.title("Confusion matrix")
plt.tight_layout()
plt.show()

images, labels = next(iter(validation_ds))
sample_probabilities = model.predict(images, verbose=0).ravel()
sample_predictions = (sample_probabilities >= 0.5).astype(int)

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for index, axis in enumerate(axes.flat):
    if index >= len(images):
        axis.axis("off")
        continue

    true_label = int(labels[index].numpy()[0])
    predicted_label = sample_predictions[index]
    axis.imshow(images[index].numpy().astype("uint8"))
    axis.set_title(
        f"Predicted: {class_names[predicted_label]}\n"
        f"True: {class_names[true_label]}"
    )
    axis.axis("off")

plt.tight_layout()
plt.show()
