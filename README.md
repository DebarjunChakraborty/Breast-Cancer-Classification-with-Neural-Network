# Breast Cancer Classification using Neural Networks

This project implements a neural network-based classifier to distinguish between benign and malignant breast cancer cases using a public dataset.

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Preprocessing](#preprocessing)
5. [Model Architecture](#model-architecture)
6. [Evaluation Metrics](#evaluation-metrics)
7. [SDG Alignment](#sdg-alignment)
8. [Requirements](#requirements)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Results](#results)
12. [Contributing](#contributing)
13. [Acknowledgements](#acknowledgements)

---

## Overview

Early and accurate diagnosis of breast cancer significantly increases survival rates. This project uses a neural network model to classify breast cancer cases based on diagnostic features. The model is trained and evaluated using standardised machine learning workflows.

## Problem Statement

Breast cancer remains one of the leading causes of cancer-related deaths among women worldwide. The challenge is to classify cancer diagnoses accurately using predictive modelling to support early intervention.

## Dataset

The dataset used is derived from the Breast Cancer Wisconsin Diagnostic Dataset, containing features computed from digitised images of a fine needle aspirate (FNA) of a breast mass.

- **Target Labels**:  
  - `0` → Malignant  
  - `1` → Benign

## Preprocessing

- Feature scaling using standardisation (zero mean, unit variance)
- Splitting the dataset into training and testing sets
- Encoding labels for binary classification

## Model Architecture

The neural network is built using a feedforward design and includes:

- Input layer corresponding to the number of features
- One or more hidden layers with ReLU activation
- Output layer with sigmoid activation for binary classification

## Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

## SDG Alignment

| Goal | Relevance |
|------|-----------|
| **SDG 3: Good Health and Well-being** | Enhances early detection and diagnosis through AI, aiding in reducing mortality and promoting well-being. |
| **SDG 9: Industry, Innovation and Infrastructure** | Utilises innovative neural network algorithms and contributes to healthcare infrastructure through data-driven insights. |

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- tensorflow / keras
- matplotlib
- seaborn

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/breast-cancer-classification-nn.git
cd breast-cancer-classification-nn
pip install -r requirements.txt
```

## Usage

To run the model and replicate results:

```bash
jupyter notebook BreastCancerClassificationNN.ipynb
```

Follow the notebook steps for data loading, preprocessing, model training, and evaluation.

## Results

The model demonstrates high classification performance on the test set, achieving:

- Accuracy: >95%
- Low false positives and false negatives

## Contributing

Feel free to fork the repository and submit pull requests. Please ensure contributions are well-documented and tested.

## Acknowledgements

- Dataset provided by UCI Machine Learning Repository
- Thanks to all open-source contributors

