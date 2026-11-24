#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a CNN model on MNIST dataset for character recognition."""

import numpy as np
import tensorflow as tf
from tensorflow import keras  # pylint: disable=E0611
from tensorflow.keras import layers  # pylint: disable=E0611, E0401
import matplotlib.pyplot as plt


def load_mnist_images_all(filename):
    """Load all images from an MNIST IDX file."""
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        print(f"Magic number: {hex(magic)}")
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')

        # Read all images at once
        buffer = f.read(num_images * rows * cols)
        images = np.frombuffer(
            buffer,
            dtype=np.uint8
        ).reshape(
            num_images,
            rows,
            cols
        )

    return images


def load_mnist_labels_all(filename):
    """Load all labels from an MNIST IDX file."""
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        print(f"Magic number: {hex(magic)}")
        num_labels = int.from_bytes(f.read(4), 'big')

        # Read all labels at once
        buffer = f.read(num_labels)
        labels = np.frombuffer(buffer, dtype=np.uint8)

    return labels


def prepare_data():
    """Load and preprocess MNIST data."""
    print("Loading MNIST dataset from IDX files...")

    # Paths to MNIST IDX files
    train_images_path = './mnist/MNIST_ORG/train-images.idx3-ubyte'
    train_labels_path = './mnist/MNIST_ORG/train-labels.idx1-ubyte'
    test_images_path = './mnist/MNIST_ORG/t10k-images.idx3-ubyte'
    test_labels_path = './mnist/MNIST_ORG/t10k-labels.idx1-ubyte'

    # Load data
    x_train = load_mnist_images_all(train_images_path)
    y_train = load_mnist_labels_all(train_labels_path)
    x_test = load_mnist_images_all(test_images_path)
    y_test = load_mnist_labels_all(test_labels_path)

    print(f"Training set: {x_train.shape[0]} images")
    print(f"Test set: {x_test.shape[0]} images")
    print(f"Image shape: {x_train.shape[1:]} (28x28)")

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Expand dimensions to add channel dimension (for CNN)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(f"Preprocessed image shape: {x_train.shape[1:]}")

    return x_train, y_train, x_test, y_test


def create_model(input_shape=(28, 28, 1), num_classes=10):
    """Create a CNN model for MNIST digit classification."""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # First convolutional block
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional block
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def train_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=15,
        batch_size=128):
    """Compile and train the model."""
    print("\nCompiling model...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel summary:")
    model.summary()

    print(f"\nTraining for {epochs} epochs...")
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )

    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    return history


def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("\nTraining history saved to 'training_history.png'")
    plt.show()


def make_predictions(model, x_test, y_test, num_samples=10):
    """Make predictions and visualize results."""
    predictions = model.predict(x_test[:num_samples])
    predicted_classes = np.argmax(predictions, axis=1)

    # fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    _, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(num_samples):
        axes[i].imshow(x_test[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(
            f'True: {y_test[i]}, Pred: {predicted_classes[i]}',
            color='green' if y_test[i] == predicted_classes[i] else 'red'
        )
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)
    print("Predictions saved to 'predictions.png'")
    plt.show()


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("MNIST Character Recognition Training")
    print("=" * 60)

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load and prepare data
    x_train, y_train, x_test, y_test = prepare_data()

    # Create model
    print("\nCreating model...")
    model = create_model()

    # Train model
    history = train_model(model, x_train, y_train, x_test, y_test, epochs=15)

    # Plot training history
    plot_training_history(history)

    # Make and visualize predictions
    print("\nGenerating predictions...")
    make_predictions(model, x_test, y_test, num_samples=10)

    # Save the model
    model_path = 'mnist_model.keras'
    print(f"\nSaving model to '{model_path}'...")
    model.save(model_path)
    print("Model saved successfully!")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
