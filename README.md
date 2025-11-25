# Quantum Support Vector Machine for Image Classification

This project implements a **Quantum Support Vector Machine (QSVM)** for image classification
and compares it with a classical Support Vector Machine (SVM).

The model uses the **8x8 handwritten digits dataset** (64-dimensional images) and
performs:

- PCA for dimensionality reduction (controls number of qubits)
- Classical SVM with RBF kernel (baseline)
- QSVM using:
  - `ZZFeatureMap` to encode image features into quantum states
  - `FidelityQuantumKernel` based on state fidelities
  - `QSVC` from Qiskit Machine Learning

## Project Objectives

- Study the limitations of classical SVMs on high-dimensional image data.
- Design a QSVM architecture that encodes image features into quantum states.
- Compare accuracy and performance between classical SVM and QSVM.
- Analyze how the number of PCA components (qubits) affects QSVM performance.

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Akashkumar3530/qsvm-major-project.git
cd qsvm-major-project

# 2. Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run QSVM image classification (default 4 PCA components / qubits)
python3 qsvm_image_classification.py --pca_components 4
