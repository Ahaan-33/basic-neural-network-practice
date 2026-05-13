# Basic CNN Shape Classifier

A beginner-friendly deep learning project built using PyTorch to classify simple synthetic images of circles and squares.

This project was created as a hands-on introduction to:

* convolutional neural networks (CNNs)
* PyTorch syntax
* training and validation workflows
* tensor operations
* Git/GitHub workflows
* local ML development using VS Code

---

# Project Overview

The model is trained on a synthetically generated dataset containing:

* circles
* squares

Images are:

* grayscale
* 16×16 pixels

The CNN learns to classify shapes using:

* convolutional layers
* max pooling
* fully connected layers
* backpropagation with Adam optimization

---

# Features

* Synthetic dataset generation
* PyTorch `ImageFolder` dataset loading
* Train/validation split
* CNN architecture implemented from scratch
* Training and validation loops
* Loss curve visualization
* Model saving using `torch.save`
* Git/GitHub version control workflow

---

# Project Structure

```text
basic-neural-network-practice/
│
├── data/
│   ├── circle/
│   └── square/
│
├── models/
│   └── shape_classifier.pth
│
├── src/
│   ├── model.py
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# CNN Architecture

The model consists of:

```text
Input Image (1×16×16)
↓
Conv2D (1 → 16)
↓
ReLU
↓
MaxPool
↓
Conv2D (16 → 32)
↓
ReLU
↓
MaxPool
↓
Flatten
↓
Fully Connected Layer
↓
Output Layer (2 classes)
```

Classes:

* `0 → circle`
* `1 → square`

---

# Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd basic-neural-network-practice
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running Training

Run:

```bash
python src/train.py
```

This will:

* load the dataset
* train the CNN
* evaluate validation accuracy
* display a training loss curve
* save the trained model

Saved model:

```text
models/shape_classifier.pth
```

---

# Example Output

```text
Epoch [1/10] Loss: 0.6421
Epoch [2/10] Loss: 0.3124
...
Validation Accuracy: 0.9750
Model saved successfully!
```

---

# Concepts Learned

This project was designed to build intuition for:

* tensors and tensor shapes
* convolution operations
* feature maps
* max pooling
* flattening
* logits
* forward propagation
* backpropagation
* loss functions
* optimizers
* train vs evaluation mode
* dataset pipelines
* validation workflows

---

# Future Improvements

Potential next steps:

* prediction/inference script
* confusion matrix visualization
* data augmentation
* noisy/shifted shapes
* MNIST digit classification
* GPU acceleration
* visualization of learned filters
* model checkpointing

---

# Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* VS Code
* Git + GitHub

---

# Purpose

This project was intentionally built from scratch as a learning exercise to understand how neural networks work internally before moving on to larger datasets and more advanced architectures.
