"""
Split Learning Benchmark
========================
This benchmark runs in SL (Split Learning) mode where the model
is split across parties - each holds different layers/features.
Only activations/gradients are exchanged, preserving raw data privacy.

Original benchmark adapted for secretlearn.split_learning.
"""

from secretlearn.split_learning.tree.decision_tree_classifier import SLDecisionTreeClassifier
from secretlearn.split_learning.tree.decision_tree_regressor import SLDecisionTreeRegressor

import gc
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# to store the results
scikit_classifier_results = []
scikit_regressor_results = []

mu_second = 0.0 + 10**6  # number of microseconds in a second

def bench_scikit_tree_classifier(X, Y):
    """Benchmark with Secret-Learn decision tree classifier"""

    
    gc.collect()

    # start time
    tstart = datetime.now()
    clf = SLDecisionTreeClassifier()
    clf.fit(X, Y).predict(X)
    delta = datetime.now() - tstart
    # stop time

    scikit_classifier_results.append(delta.seconds + delta.microseconds / mu_second)

def bench_scikit_tree_regressor(X, Y):
    """Benchmark with Secret-Learn decision tree regressor"""

    
    gc.collect()

    # start time
    tstart = datetime.now()
    clf = SLDecisionTreeRegressor()
    clf.fit(X, Y).predict(X)
    delta = datetime.now() - tstart
    # stop time

    scikit_regressor_results.append(delta.seconds + delta.microseconds / mu_second)

if __name__ == "__main__":
    print("============================================")
    print("Warning: this is going to take a looong time")
    print("============================================")

    n = 10
    step = 10000
    n_samples = 10000
    dim = 10
    n_classes = 10
    for i in range(n):
        print("============================================")
        print("Entering iteration %s of %s" % (i, n))
        print("============================================")
        n_samples += step
        X = np.random.randn(n_samples, dim)
        Y = np.random.randint(0, n_classes, (n_samples,))
        bench_scikit_tree_classifier(X, Y)
        Y = np.random.randn(n_samples)
        bench_scikit_tree_regressor(X, Y)

    xx = range(0, n * step, step)
    plt.figure("Secret-Learn tree benchmark results")
    plt.subplot(211)
    plt.title("Learning with varying number of samples")
    plt.plot(xx, scikit_classifier_results, "g-", label="classification")
    plt.plot(xx, scikit_regressor_results, "r-", label="regression")
    plt.legend(loc="upper left")
    plt.xlabel("number of samples")
    plt.ylabel("Time (s)")

    scikit_classifier_results = []
    scikit_regressor_results = []
    n = 10
    step = 500
    start_dim = 500
    n_classes = 10

    dim = start_dim
    for i in range(0, n):
        print("============================================")
        print("Entering iteration %s of %s" % (i, n))
        print("============================================")
        dim += step
        X = np.random.randn(100, dim)
        Y = np.random.randint(0, n_classes, (100,))
        bench_scikit_tree_classifier(X, Y)
        Y = np.random.randn(100)
        bench_scikit_tree_regressor(X, Y)

    xx = np.arange(start_dim, start_dim + n * step, step)
    plt.subplot(212)
    plt.title("Learning in high dimensional spaces")
    plt.plot(xx, scikit_classifier_results, "g-", label="classification")
    plt.plot(xx, scikit_regressor_results, "r-", label="regression")
    plt.legend(loc="upper left")
    plt.xlabel("number of dimensions")
    plt.ylabel("Time (s)")
    plt.axis("tight")
    plt.show()
