"""
Quantum Support Vector Machine for Image Classification
=======================================================

This script implements image classification using:

1. Classical SVM (baseline)
2. Quantum Support Vector Machine (QSVM) with a quantum kernel

Dataset:
- Handwritten digits dataset (8x8 grayscale images) from sklearn.

Flow:
- Load images (high-dimensional 64-feature space).
- Flatten and standardize features.
- Apply PCA to reduce dimension -> number of qubits.
- Train & evaluate Classical SVM.
- Train & evaluate QSVM that encodes image features into quantum states
  using a ZZFeatureMap and FidelityQuantumKernel.

This implementation matches the abstract:
"Quantum Support Vector Machine for Image Classification".
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.svm import SVC

# Qiskit / Quantum imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


# =========================
# Utility functions
# =========================
def plot_confusion_matrix(y_true, y_pred, title: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()

    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")

    plt.tight_layout()
    plt.show()


# =========================
# Data loading & preprocessing
# =========================
def load_image_data(test_size=0.2, random_state=42):
    """
    Load 8x8 grayscale handwritten digits images.
    Each image -> 64-dimensional feature vector.
    """
    digits = load_digits()
    X = digits.data  # shape (n_samples, 64)
    y = digits.target

    print("=== Image Dataset: Handwritten Digits ===")
    print(f"Number of samples   : {X.shape[0]}")
    print(f"Original feature dim: {X.shape[1]} (flattened 8x8 images)")
    print(f"Number of classes   : {len(np.unique(y))}")

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def preprocess_with_pca(X_train, X_test, n_components: int):
    """
    Standardize features and apply PCA to reduce to n_components
    (this becomes the number of qubits in QSVM).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if n_components > X_train_scaled.shape[1]:
        raise ValueError(
            f"n_components ({n_components}) cannot be > feature dimension ({X_train_scaled.shape[1]})."
        )

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"PCA reduced feature dimension to: {n_components}")
    print(f"(This equals the number of qubits in the quantum circuit.)")

    return X_train_pca, X_test_pca


# =========================
# Classical SVM
# =========================
def train_and_evaluate_classical_svm(X_train, X_test, y_train, y_test):
    print("\n=== Training Classical SVM (RBF kernel) ===")
    clf = SVC(kernel="rbf", gamma="scale", C=1.0)
    clf.fit(X_train, y_train)

    print("=== Evaluating Classical SVM ===")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Classical SVM Accuracy: {acc:.4f}")
    print("\nClassification Report (Classical SVM):")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred, title="Classical SVM Confusion Matrix")

    return clf, acc


# =========================
# QSVM (Quantum SVM)
# =========================
def build_qsvm(feature_dimension: int) -> QSVC:
    """
    Build a QSVM classifier using a quantum kernel based on state fidelities.
    - feature_dimension = number of PCA components = number of qubits.
    """
    # Quantum feature map: encodes classical image features into quantum states
    feature_map = ZZFeatureMap(
        feature_dimension=feature_dimension,
        reps=2,
        entanglement="full",
    )

    # StatevectorSampler is a V2 primitive that simulates quantum states
    sampler = StatevectorSampler()

    # ComputeUncompute estimates state fidelities between quantum states
    fidelity = ComputeUncompute(sampler=sampler)

    # Quantum kernel based on state fidelities (similarity between quantum states)
    quantum_kernel = FidelityQuantumKernel(
        fidelity=fidelity,
        feature_map=feature_map,
    )

    # QSVM classifier using the quantum kernel
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    return qsvc


def train_and_evaluate_qsvm(X_train, X_test, y_train, y_test):
    feature_dim = X_train.shape[1]
    print(f"\nFeature dimension after PCA (qubits) = {feature_dim}")

    if feature_dim > 10:
        print(
            "Warning: More than 10 qubits may be slow to simulate. "
            "Consider using fewer PCA components."
        )

    qsvc = build_qsvm(feature_dimension=feature_dim)

    print("\n=== Training Quantum SVM (QSVM) ===")
    qsvc.fit(X_train, y_train)

    print("=== Evaluating QSVM ===")
    y_pred = qsvc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"QSVM Accuracy: {acc:.4f}")
    print("\nClassification Report (QSVM):")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred, title="QSVM Confusion Matrix")

    return qsvc, acc


# =========================
# Main CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Quantum Support Vector Machine for Image Classification"
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=4,
        help="Number of PCA components (also number of qubits). Default = 4.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test size fraction. Default = 0.2",
    )

    args = parser.parse_args()

    # 1. Load image data
    X_train, X_test, y_train, y_test = load_image_data(test_size=args.test_size)

    # 2. PCA
    X_train_pca, X_test_pca = preprocess_with_pca(
        X_train, X_test, n_components=args.pca_components
    )

    # 3. Classical SVM
    classical_model, classical_acc = train_and_evaluate_classical_svm(
        X_train_pca, X_test_pca, y_train, y_test
    )

    # 4. QSVM
    qsvm_model, qsvm_acc = train_and_evaluate_qsvm(
        X_train_pca, X_test_pca, y_train, y_test
    )

    # 5. Summary
    print("\n================= SUMMARY COMPARISON =================")
    print(f"PCA Components (qubits): {args.pca_components}")
    print(f"Classical SVM Accuracy : {classical_acc:.4f}")
    print(f"QSVM Accuracy          : {qsvm_acc:.4f}")
    print("======================================================\n")


if __name__ == "__main__":
    main()
