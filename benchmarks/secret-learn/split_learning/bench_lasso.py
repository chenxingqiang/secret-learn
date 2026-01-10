"""
Split Learning Benchmark
========================
This benchmark runs in SL (Split Learning) mode where the model
is split across parties - each holds different layers/features.
Only activations/gradients are exchanged, preserving raw data privacy.

Original benchmark adapted for secretlearn.split_learning.
"""

from secretlearn.split_learning.linear_models.lasso import SLLasso
from secretlearn.split_learning.linear_models.lasso_lars import SLLassoLars

import gc
from time import time

import numpy as np

from sklearn.datasets import make_regression

def compute_bench(alpha, n_samples, n_features, precompute):
    lasso_results = []
    lars_lasso_results = []

    it = 0

    for ns in n_samples:
        for nf in n_features:
            it += 1
            print("==================")
            print("Iteration %s of %s" % (it, max(len(n_samples), len(n_features))))
            print("==================")
            n_informative = nf // 10
            X, Y, coef_ = make_regression(
                n_samples=ns,
                n_features=nf,
                n_informative=n_informative,
                noise=0.1,
                coef=True,

            X /= np.sqrt(np.sum(X**2, axis=0))  # Normalize data

            gc.collect()
            print("- benchmarking SLLasso")
            clf = SLLasso(alpha=alpha, fit_intercept=False, precompute=precompute)
            tstart = time()
            clf.fit(X, Y)
            lasso_results.append(time() - tstart)

            gc.collect()
            print("- benchmarking SLLassoLars")
            clf = SLLassoLars(alpha=alpha, fit_intercept=False, precompute=precompute)
            tstart = time()
            clf.fit(X, Y)
            lars_lasso_results.append(time() - tstart)

    return lasso_results, lars_lasso_results

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    
    alpha = 0.01  # regularization parameter

    n_features = 10
    list_n_samples = np.linspace(100, 1000000, 5).astype(int)
    lasso_results, lars_lasso_results = compute_bench(
        alpha, list_n_samples, [n_features], precompute=True

    plt.figure("Secret-Learn LASSO benchmark results")
    plt.subplot(211)
    plt.plot(list_n_samples, lasso_results, "b-", label="SLLasso")
    plt.plot(list_n_samples, lars_lasso_results, "r-", label="SLLassoLars")
    plt.title("precomputed Gram matrix, %d features, alpha=%s" % (n_features, alpha))
    plt.legend(loc="upper left")
    plt.xlabel("number of samples")
    plt.ylabel("Time (s)")
    plt.axis("tight")

    n_samples = 2000
    list_n_features = np.linspace(500, 3000, 5).astype(int)
    lasso_results, lars_lasso_results = compute_bench(
        alpha, [n_samples], list_n_features, precompute=False

    plt.subplot(212)
    plt.plot(list_n_features, lasso_results, "b-", label="SLLasso")
    plt.plot(list_n_features, lars_lasso_results, "r-", label="SLLassoLars")
    plt.title("%d samples, alpha=%s" % (n_samples, alpha))
    plt.legend(loc="upper left")
    plt.xlabel("number of features")
    plt.ylabel("Time (s)")
    plt.axis("tight")
    plt.show()
