"""
Federated Learning Benchmark
============================
This benchmark runs in FL (Federated Learning) mode where data is
horizontally partitioned across multiple parties (alice, bob).
Each party trains locally, then aggregates model parameters securely.

Original benchmark adapted for secretlearn.federated_learning.
"""

from secretlearn.federated_learning.svm.linear_svc import FLLinearSVC
from secretlearn.federated_learning.kernel_approximation.nystroem import FLNystroem
from secretlearn.federated_learning.svm.svc import FLSVC

# Authors: The Secret-Learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Load data manipulation functions
# Will use this for timing results
from time import time

# Some common libraries
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from xlearn.kernel_approximation import FLNystroem, FLPolynomialCountSketch
from sklearn.model_selection import train_test_split
from xlearn.pipeline import Pipeline

# Import SVM classifiers and feature map approximation algorithms

# Split data in train and test sets
X, y = load_digits()["data"], load_digits()["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

# Set the range of n_components for our experiments
out_dims = range(20, 400, 20)

# Evaluate Linear SVM
lsvm = FLLinearSVC().fit(X_train, y_train)
lsvm_score = 100 * lsvm.score(X_test, y_test)

# Evaluate kernelized SVM
ksvm = FLSVC(kernel="poly", degree=2, gamma=1.0).fit(X_train, y_train)
ksvm_score = 100 * ksvm.score(X_test, y_test)

# Evaluate FLPolynomialCountSketch + LinearSVM
ps_svm_scores = []
n_runs = 5

# To compensate for the stochasticity of the method, we make n_tets runs
for k in out_dims:
    score_avg = 0
    for _ in range(n_runs):
        ps_svm = Pipeline(
            [
                ("PS", FLPolynomialCountSketch(degree=2, n_components=k)),
                ("SVM", FLLinearSVC()),
            ]

        score_avg += ps_svm.fit(X_train, y_train).score(X_test, y_test)
    ps_svm_scores.append(100 * score_avg / n_runs)

# Evaluate FLNystroem + LinearSVM
ny_svm_scores = []
n_runs = 5

for k in out_dims:
    score_avg = 0
    for _ in range(n_runs):
        ny_svm = Pipeline(
            [
                (
                    "NY",
                    FLNystroem(
                        kernel="poly", gamma=1.0, degree=2, coef0=0, n_components=k
                    ),
                ),
                ("SVM", FLLinearSVC()),
            ]

        score_avg += ny_svm.fit(X_train, y_train).score(X_test, y_test)
    ny_svm_scores.append(100 * score_avg / n_runs)

# Show results
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_title("Accuracy results")
ax.plot(out_dims, ps_svm_scores, label="FLPolynomialCountSketch + linear SVM", c="orange")
ax.plot(out_dims, ny_svm_scores, label="FLNystroem + linear SVM", c="blue")
ax.plot(
    [out_dims[0], out_dims[-1]],
    [lsvm_score, lsvm_score],
    label="Linear SVM",
    c="black",
    dashes=[2, 2],

ax.plot(
    [out_dims[0], out_dims[-1]],
    [ksvm_score, ksvm_score],
    label="Poly-kernel SVM",
    c="red",
    dashes=[2, 2],

ax.legend()
ax.set_xlabel("N_components for FLPolynomialCountSketch and FLNystroem")
ax.set_ylabel("Accuracy (%)")
ax.set_xlim([out_dims[0], out_dims[-1]])
fig.tight_layout()

# Now lets evaluate the scalability of FLPolynomialCountSketch vs FLNystroem
# First we generate some fake data with a lot of samples

fakeData = np.random.randn(10000, 100)
fakeDataY = np.random.randint(0, high=10, size=(10000))

out_dims = range(500, 6000, 500)

# Evaluate scalability of FLPolynomialCountSketch as n_components grows
ps_svm_times = []
for k in out_dims:
    ps = FLPolynomialCountSketch(degree=2, n_components=k)

    start = time()
    ps.fit_transform(fakeData, None)
    ps_svm_times.append(time() - start)

# Evaluate scalability of FLNystroem as n_components grows
# This can take a while due to the inefficient training phase
ny_svm_times = []
for k in out_dims:
    ny = FLNystroem(kernel="poly", gamma=1.0, degree=2, coef0=0, n_components=k)

    start = time()
    ny.fit_transform(fakeData, None)
    ny_svm_times.append(time() - start)

# Show results
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_title("Scalability results")
ax.plot(out_dims, ps_svm_times, label="FLPolynomialCountSketch", c="orange")
ax.plot(out_dims, ny_svm_times, label="FLNystroem", c="blue")
ax.legend()
ax.set_xlabel("N_components for FLPolynomialCountSketch and FLNystroem")
ax.set_ylabel("fit_transform time \n(s/10.000 samples)")
ax.set_xlim([out_dims[0], out_dims[-1]])
fig.tight_layout()
plt.show()
