"""
Quantum Support Vector Machine (QSVM) – Generalized Major Project
"""

from typing import Optional
import argparse
import numpy as np

# sklearn
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.svm import SVC

# plotting
import matplotlib.pyplot as plt
import pandas as pd

# qiskit imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
    plt.show()


def load_digits_data():
    data = load_digits()
    return data.data, data.target


def load_cancer_data():
    data = load_breast_cancer()
    return data.data, data.target


def load_custom_csv(csv_path, target_column):
    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"{target_column} NOT found in CSV columns")
    return df.drop(columns=[target_column]).values, df[target_column].values


def preprocess(X, y, n_components=4):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    print(f"PCA Components (qubits) = {n_components}")
    return X_train, X_test, y_train, y_test


def train_classical(X_train, X_test, y_train, y_test):
    clf = SVC(kernel="rbf")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\n=== Classical SVM Results ===")
    print("Accuracy =", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, "Classical SVM Confusion Matrix")


def train_qsvm(X_train, X_test, y_train, y_test):
    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2)
    fidelity = ComputeUncompute(sampler=StatevectorSampler())
    qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    qsvc = QSVC(quantum_kernel=qkernel)
    qsvc.fit(X_train, y_train)
    y_pred = qsvc.predict(X_test)
    print("\n=== Quantum SVM Results ===")
    print("Accuracy =", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, "QSVM Confusion Matrix")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["digits", "cancer", "custom"])
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--pca", type=int, default=4)

    args = parser.parse_args()

    if args.dataset == "digits":
        X, y = load_digits_data()
    elif args.dataset == "cancer":
        X, y = load_cancer_data()
    else:
        X, y = load_custom_csv(args.csv_path, args.target)

    X_train, X_test, y_train, y_test = preprocess(X, y, args.pca)
    train_classical(X_train, X_test, y_train, y_test)
    train_qsvm(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
