#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict a digit from MNIST test set using the trained model."""

import sys
import random
import numpy as np
from tensorflow import keras  # pylint: disable=E0611
import matplotlib.pyplot as plt


def load_mnist_image(filename, idx):
    """Load a single image from an MNIST IDX file."""
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        print(f"Magic number: {hex(magic)}")
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')

        # Validate index
        if idx < 0 or idx >= num_images:
            raise ValueError(f"Index {idx} out of range [0, {num_images-1}]")

        # Skip to the desired image
        f.seek(16 + idx * rows * cols)
        img = np.frombuffer(f.read(rows * cols), dtype=np.uint8).reshape((rows, cols))

    return img


def load_mnist_label(filename, idx):
    """Load a single label from an MNIST IDX file."""
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        print(f"Magic number: {hex(magic)}")
        num_labels = int.from_bytes(f.read(4), 'big')

        # Validate index
        if idx < 0 or idx >= num_labels:
            raise ValueError(f"Index {idx} out of range [0, {num_labels-1}]")

        f.seek(8 + idx)
        label = int.from_bytes(f.read(1), 'big')

    return label


def get_num_images(filename):
    """Get the total number of images in an IDX file."""
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        print(f"Magic number: {hex(magic)}")
        num_images = int.from_bytes(f.read(4), 'big')
    return num_images


def main():
    """Main prediction pipeline."""
    # Paths
    test_images_path = './mnist/MNIST_ORG/t10k-images.idx3-ubyte'
    test_labels_path = './mnist/MNIST_ORG/t10k-labels.idx1-ubyte'
    model_path = 'mnist_model.keras'

    # Get total number of test images
    total_images = get_num_images(test_images_path)

    # Parse command line argument
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
            if n < 0 or n >= total_images:
                print(f"Error: n must be between 0 and {total_images-1}")
                print("Using random image instead.")
                n = random.randint(0, total_images - 1)
        except ValueError:
            print(f"Error: Invalid argument '{sys.argv[1]}'. Must be an integer.")
            print("Using random image instead.")
            n = random.randint(0, total_images - 1)
    else:
        # No argument provided, use random
        n = random.randint(0, total_images - 1)
        print("No index provided. Using random image.")

    print(f"Selected image index: {n} (out of {total_images})")

    # Load the model
    print(f"\nLoading model from '{model_path}'...")
    try:
        model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run 'uv run train_model.py' first to train and save the model.")
        sys.exit(1)

    # Load the image and label
    print(f"\nLoading image {n}...")
    img = load_mnist_image(test_images_path, n)
    true_label = load_mnist_label(test_labels_path, n)

    # Preprocess the image for prediction
    img_normalized = img.astype('float32') / 255.0
    img_input = np.expand_dims(img_normalized, axis=-1)  # Add channel dimension
    img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension

    # Make prediction
    print("Running prediction...")
    prediction = model.predict(img_input, verbose=0)
    predicted_digit = np.argmax(prediction[0])
    confidence = prediction[0][predicted_digit] * 100

    # Display results
    print(f"\nTrue label: {true_label}")
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}%")

    # Show probabilities for all digits
    print("\nPrediction probabilities:")
    for digit in range(10):
        prob = prediction[0][digit] * 100
        bar = '█' * int(prob / 2)
        print(f"  {digit}: {prob:6.2f}% {bar}")

    # Visualize the image and prediction
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap='gray')

    # Color based on correctness
    color = 'green' if predicted_digit == true_label else 'red'
    status = '✓ Correct' if predicted_digit == true_label else '✗ Incorrect'

    plt.title(f'Image #{n}\n'
              f'True label: {true_label} | Predicted: {predicted_digit} ({confidence:.1f}%)\n'
              f'{status}',
              fontsize=14, color=color, weight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"\n{status}!")


if __name__ == "__main__":
    main()
