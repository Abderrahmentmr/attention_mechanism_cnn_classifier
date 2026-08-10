import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

# Load the CIFAR-10 dataset
(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

# Keep only cat images (class 3) and dog images (class 5)
mask = np.logical_or(y_train.flatten() == 3, y_train.flatten() == 5)
x_filtered = x_train[mask]
y_filtered = y_train[mask]

# Shuffle and select 1000 images
indices = np.random.permutation(len(x_filtered))
x_filtered = x_filtered[indices]
y_filtered = y_filtered[indices]

x_subset = x_filtered[:1000]
y_subset = y_filtered[:1000]

# Remap labels: Cat -> 0, Dog -> 1
y_subset = np.where(y_subset == 3, 0, 1)

# Split into training and validation sets
x_train_final = x_subset[:700] / 255.0  # Normalization
y_train_final = y_subset[:700]
x_val_final = x_subset[700:850] / 255.0
y_val_final = y_subset[700:850]

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Custom QKV attention layer
class QKVAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=1, **kwargs):
        super(QKVAttentionLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        # Create the convolution layers once
        self.convQ = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.convK = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.convV = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, inputs):
        Q = self.convQ(inputs)
        K = self.convK(inputs)
        V = self.convV(inputs)

        # Get dynamic batch size and spatial dimensions
        batch_size = tf.shape(Q)[0]
        H = tf.shape(Q)[1]
        W = tf.shape(Q)[2]

        # Flatten spatial dimensions: (B, H*W, filters)
        Q_flat = tf.reshape(Q, [batch_size, -1, self.filters])
        K_flat = tf.reshape(K, [batch_size, -1, self.filters])
        V_flat = tf.reshape(V, [batch_size, -1, self.filters])

        # Compute attention scores: matrix product of Q and K^T
        scores = tf.matmul(Q_flat, K_flat, transpose_b=True)  # (B, H*W, H*W)

        # Stabilization: divide by sqrt(filters)
        dk = tf.cast(self.filters, tf.float32)
        scores = scores / tf.math.sqrt(dk)

        # Apply softmax to get attention weights
        attention_weights = self.softmax(scores)

        # Weight V by the attention weights
        attended_features = tf.matmul(attention_weights, V_flat)  # (B, H*W, filters)

        # Reshape back to the original spatial shape
        output = tf.reshape(attended_features, [batch_size, H, W, self.filters])
        return output

    def get_config(self):
        config = super(QKVAttentionLayer, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size
        })
        return config

# CNN model definition enhanced with the QKV attention module
inputs = tf.keras.layers.Input(shape=(32, 32, 3))
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
# Apply the QKV attention block via the custom layer
x_att = QKVAttentionLayer(filters=128)(x)
# Combine original features with attention-weighted features
x = tf.keras.layers.Add()([x, x_att])
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model with data augmentation
history = model.fit(
    train_datagen.flow(x_train_final, y_train_final, batch_size=32),
    epochs=50,
    validation_data=(x_val_final, y_val_final)
)

# Plot accuracy and loss curves
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy (train)', color='blue')
plt.plot(history.history['val_accuracy'], label='Accuracy (val)', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy curve')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss (train)', color='blue')
plt.plot(history.history['val_loss'], label='Loss (val)', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss curve')
plt.legend()

plt.show()

# Evaluate the model on the validation images
predictions = model.predict(x_val_final)
predicted_labels = (predictions > 0.5).astype(int).flatten()

# Confusion matrix
conf_matrix = confusion_matrix(y_val_final, predicted_labels)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel('Prediction')
plt.ylabel('Ground truth')
plt.title('Confusion matrix')
plt.show()

# Display visual results
fig, axes = plt.subplots(15, 10, figsize=(20, 30))
axes = axes.flatten()
for i in range(150):
    axes[i].imshow(x_val_final[i])
    axes[i].set_title(f"Predicted: {'Dog' if predicted_labels[i] == 1 else 'Cat'}\nTrue: {'Dog' if y_val_final[i] == 1 else 'Cat'}")
    axes[i].axis("off")
plt.tight_layout()
plt.show()
