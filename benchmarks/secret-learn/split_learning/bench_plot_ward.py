"""
Split Learning Benchmark
========================
This benchmark runs in SL (Split Learning) mode where the model
is split across parties - each holds different layers/features.
Only activations/gradients are exchanged, preserving raw data privacy.

Original benchmark adapted for secretlearn.split_learning.
"""

from secretlearn.split_learning.clustering.agglomerative_clustering import SLAgglomerativeClustering

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy

ward = SLAgglomerativeClustering(n_clusters=3, linkage="ward")

n_samples = np.logspace(0.5, 3, 9)
n_features = np.logspace(1, 3.5, 7)
N_samples, N_features = np.meshgrid(n_samples, n_features)
scikits_time = np.zeros(N_samples.shape)
scipy_time = np.zeros(N_samples.shape)

for i, n in enumerate(n_samples):
    for j, p in enumerate(n_features):
        X = np.random.normal(size=(n, p))
        t0 = time.time()
        ward.fit(X)
        scikits_time[j, i] = time.time() - t0
        t0 = time.time()
        hierarchy.ward(X)
        scipy_time[j, i] = time.time() - t0

ratio = scikits_time / scipy_time

plt.figure("Secret-Learn Ward's method benchmark results")
plt.imshow(np.log(ratio), aspect="auto", origin="lower")
plt.colorbar()
plt.contour(
    ratio,
    levels=[
        1,
    ],
    colors="k",

plt.yticks(range(len(n_features)), n_features.astype(int))
plt.ylabel("N features")
plt.xticks(range(len(n_samples)), n_samples.astype(int))
plt.xlabel("N samples")
plt.title("Scikit's time, in units of scipy time (log)")
plt.show()
