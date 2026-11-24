# Digit Recognition

MNIST digit recognition project using TensorFlow and Keras. The Jupyter notebook
explores the MNIST dataset, and the training script builds a CNN model achieving
**above 99% test accuracy**.

## What is here

- [character_recognition.ipynb](character_recognition.ipynb): MNIST exploration and visualization.
- `mnist/`: Raw MNIST IDX files (train/test images and labels).

## Requirements

- Python >= 3.11, < 3.14
- Dependencies: `numpy`, `matplotlib`, `tensorflow`

If you are using `uv`, it will read dependencies from [pyproject.toml](pyproject.toml).

## Quick start

### Explore the data

Open the notebook and run the cells in order (VS Code or Jupyter Lab):

```bash
jupyter lab
```

The notebook demonstrates:

- Loading individual MNIST images from IDX files
- Displaying sample digits with their labels
- Understanding the 28×28 grayscale format
