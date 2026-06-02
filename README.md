# Data-Scarce Training: Traditional ML vs Deep Learning for Digit and Face Recognition

## Overview

This project investigates the performance of **Traditional Machine Learning** and **Deep Learning** under extremely limited training data conditions.

According to the project requirements:

* Training Set: 10%
* Validation Set: 5%
* Testing Set: 85%

Additionally:

* No pretrained models are allowed.
* No transfer learning is allowed.
* All models must be trained from scratch.

The goal is to evaluate the generalization capability of different recognition approaches when only a small portion of data is available for training.

---

## Project Structure

```text
project/
│
├── datasets/
│   ├── digit_dataset.py
│   └── face_dataset.py
│
├── traditional/
│   ├── features.py
│   ├── svm_model.py
│   └── face_svm_model.py
│
├── deep_learning/
│   ├── cnn_model.py
│   ├── face_cnn_model.py
│   ├── train_cnn.py
│   └── train_face_cnn.py
│
├── utils/
│   ├── metrics.py
│   ├── split_dataset.py
│   └── seed.py
│
├── outputs/
│   ├── models/
│   └── reports/
│
├── data/
│   ├── dataset/
│   └── archive/
│
├── main.py
└── README.md
```

---

## Datasets

### 1. Digit Recognition Dataset

Directory Structure:

```text
dataset/
├── 0/
│   └── 0/
│       ├── image1.png
│       ├── image2.png
│       └── ...
├── 1/
├── 2/
...
├── 9/
```

Characteristics:

* 10 classes (0~9)
* Transparent PNG images
* Training from scratch
* Balanced class distribution

---

### 2. Face Recognition Dataset

Directory Structure:

```text
archive/
├── n000002/
│   ├── 0001_01.jpg
│   ├── 0001_02.jpg
│   └── ...
├── n000003/
├── n000004/
...
```

Characteristics:

* 50 identities
* Face images
* Folder names represent identity classes
* Labels are automatically remapped to consecutive integers

---

## Dataset Split

The dataset is split according to the project requirements:

| Dataset Split | Ratio |
| ------------- | ----- |
| Training      | 10%   |
| Validation    | 5%    |
| Testing       | 85%   |

The same split strategy is used for both Digit and Face datasets.

---

## Traditional Machine Learning

### Feature Extraction

#### Digit Recognition

Feature:

```text
HOG (Histogram of Oriented Gradients)
```

Classifier:

```text
Linear SVM
```

---

#### Face Recognition

Feature:

```text
HOG (Histogram of Oriented Gradients)
```

Classifier:

```text
Linear SVM
```

---

## Deep Learning

### Digit Recognition

CNN Architecture:

```text
Input (28x28)

Conv(32)
↓
MaxPool

Conv(64)
↓
MaxPool

Conv(128)

Fully Connected

10 Classes
```

---

### Face Recognition

CNN Architecture:

```text
Input (96x96)

Conv(32)
↓
MaxPool

Conv(64)
↓
MaxPool

Conv(128)
↓
MaxPool

Conv(256)
↓
MaxPool

AdaptiveAvgPool

Fully Connected

50 Classes
```

All networks are trained from scratch.

No pretrained weights are used.

---

## Installation

### Create Environment

```bash
pip install -r requirements.txt
```

### Required Packages

```text
numpy
opencv-python
matplotlib
scikit-learn
scikit-image
torch
torchvision
rich
joblib
```

---

## Usage

### Digit Recognition

Traditional ML

```bash
python main.py --task digit --method traditional
python main.py -t digit -m traditional
```

Deep Learning (default 20 epochs)

```bash
python main.py --task digit --method dl
python main.py -t digit -m dl
python main.py -t digit -m dl --epochs 50
```

Compare Both Methods

```bash
python main.py --task digit --method compare
python main.py -t digit -m compare
```

---

### Face Recognition

Traditional ML

```bash
python main.py --task face --method traditional
python main.py -t face -m traditional
```

Deep Learning (default 20 epochs)

```bash
python main.py --task face --method dl
python main.py -t face -m dl
python main.py -t face -m dl --epochs 50
```

Compare Both Methods

```bash
python main.py --task face --method compare
python main.py -t face -m compare
```

---

## Output Files

Generated files are stored in:

```text
outputs/
├── models/
│   ├── digit_hog_svm.pkl
│   ├── digit_cnn_best.pth
│   ├── face_hog_svm.pkl
│   └── face_cnn_best.pth
│
└── reports/
    ├── classification_report.json
    ├── confusion_matrix.png
    └── ...
```

---

## Evaluation Metrics

The following metrics are used:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## Experimental Goal

This project aims to compare:

```text
Traditional Feature Engineering
            vs
Deep Learning Representation Learning
```

under severe data scarcity conditions.

The study investigates:

1. How well handcrafted features perform with limited training data.
2. Whether CNNs can generalize better than traditional methods.
3. The difference between simple visual tasks (digits) and complex visual tasks (faces).

---

## Author

Yu, Ze-Xun

Department of Computer Science

Final Project - Pattern Recognition
