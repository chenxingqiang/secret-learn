"""
Split Learning Benchmark
========================
This benchmark runs in SL (Split Learning) mode where the model
is split across parties - each holds different layers/features.
Only activations/gradients are exchanged, preserving raw data privacy.

Original benchmark adapted for secretlearn.split_learning.
"""

from secretlearn.split_learning.linear_models.sgd_regressor import SLSGDRegressor

import numpy as np
from scipy.sparse import csr_matrix

from sklearn.metrics import r2_score

np.random.seed(42)

def sparsity_ratio(X):
    return np.count_nonzero(X) / float(n_samples * n_features)

n_samples, n_features = 5000, 300
X = np.random.randn(n_samples, n_features)
inds = np.arange(n_samples)
np.random.shuffle(inds)
X[inds[int(n_features / 1.2) :]] = 0  # sparsify input
print("input data sparsity: %f" % sparsity_ratio(X))
coef = 3 * np.random.randn(n_features)
inds = np.arange(n_features)
np.random.shuffle(inds)
coef[inds[n_features // 2 :]] = 0  # sparsify coef
print("true coef sparsity: %f" % sparsity_ratio(coef))
y = np.dot(X, coef)

# add noise
y += 0.01 * np.random.normal((n_samples,))

# Split data in train set and test set
n_samples = X.shape[0]
X_train, y_train = X[: n_samples // 2], y[: n_samples // 2]
X_test, y_test = X[n_samples // 2 :], y[n_samples // 2 :]
print("test data sparsity: %f" % sparsity_ratio(X_test))

###############################################################################
clf = SLSGDRegressor(penalty="l1", alpha=0.2, max_iter=2000, tol=None)
clf.fit(X_train, y_train)
print("model sparsity: %f" % sparsity_ratio(clf.coef_))

def benchmark_dense_predict():
    for _ in range(300):
        clf.predict(X_test)

def benchmark_sparse_predict():
    X_test_sparse = csr_matrix(X_test)
    for _ in range(300):
        clf.predict(X_test_sparse)

def score(y_test, y_pred, case):
    r2 = r2_score(y_test, y_pred)
    print("r^2 on test data (%s) : %f" % (case, r2))

score(y_test, clf.predict(X_test), "dense model")
benchmark_dense_predict()
clf.sparsify()
score(y_test, clf.predict(X_test), "sparse model")
benchmark_sparse_predict()
