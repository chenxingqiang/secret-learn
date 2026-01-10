"""
Federated Learning Benchmark
============================
This benchmark runs in FL (Federated Learning) mode where data is
horizontally partitioned across multiple parties (alice, bob).
Each party trains locally, then aggregates model parameters securely.

Original benchmark adapted for secretlearn.federated_learning.
"""

import argparse
import os.path as op

import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = "mnist_tsne_output"

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot benchmark results for t-SNE")
    parser.add_argument(
        "--labels",
        type=str,
        default=op.join(LOG_DIR, "mnist_original_labels_10000.npy"),
        help="1D integer numpy array for labels",

    parser.add_argument(
        "--embedding",
        type=str,
        default=op.join(LOG_DIR, "mnist_secretlearn_TSNE_10000.npy"),
        help="2D float numpy array for embedded data",

    args = parser.parse_args()

    X = np.load(args.embedding)
    y = np.load(args.labels)

    for i in np.unique(y):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], alpha=0.2, label=int(i))
    plt.legend(loc="best")
    plt.show()
