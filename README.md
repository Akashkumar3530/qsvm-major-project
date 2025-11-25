# Quantum Support Vector Machine for Image Classification

This project implements a Quantum Support Vector Machine (QSVM) and a Classical SVM
for image classification using the handwritten digits dataset (8x8 grayscale images).

## Features

- Classical SVM baseline with RBF kernel
- Quantum SVM using:
  - ZZFeatureMap for feature encoding
  - FidelityQuantumKernel built from state fidelities
  - Qiskit Machine Learning (QSVC)
- PCA-based dimensionality reduction → controls number of qubits
- Accuracy and confusion matrices for both models

## How to Run

```bash
# 1. Create and activate virtual environment (first time)
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run QSVM image classification
python3 qsvm_image_classification.py --pca_components 4
