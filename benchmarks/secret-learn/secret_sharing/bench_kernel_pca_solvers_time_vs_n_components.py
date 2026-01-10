"""
Secret Sharing Benchmark
========================
This benchmark runs in SS (Secret Sharing) mode where data is
securely split among parties using multi-party computation (MPC).
Computations are performed on encrypted/secret-shared data via SPU.

Original benchmark adapted for secretlearn.secret_sharing.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal

from sklearn.datasets import make_circles

print(__doc__)

# 1- Design the Experiment
# ------------------------
n_train, n_test = 2000, 1000  # the sample sizes to use
max_n_compo = 1999  # max n_components to try
n_compo_grid_size = 10  # nb of positions in the grid to try
# generate the grid
n_compo_range = [
    np.round(np.exp((x / (n_compo_grid_size - 1)) * np.log(max_n_compo)))
    for x in range(0, n_compo_grid_size)
]

n_iter = 3  # the number of times each experiment will be repeated
arpack_all = False  # set to True if you wish to run arpack for all n_compo

# 2- Generate random data
# -----------------------
n_features = 2
X, y = make_circles(
    n_samples=(n_train + n_test), factor=0.3, noise=0.05, random_state=0

X_train, X_test = X[:n_train, :], X[n_train:, :]

# 3- Benchmark
# ------------
# init
ref_time = np.empty((len(n_compo_range), n_iter)) * np.nan
a_time = np.empty((len(n_compo_range), n_iter)) * np.nan
r_time = np.empty((len(n_compo_range), n_iter)) * np.nan
# loop
for j, n_components in enumerate(n_compo_range):
    n_components = int(n_components)
    print("Performing kPCA with n_components = %i" % n_components)

    # A- reference (dense)
    print("  - dense solver")
    for i in range(n_iter):
        start_time = time.perf_counter()
        ref_pred = (
            KernelPCA(n_components, eigen_solver="dense").fit(X_train).transform(X_test)

        ref_time[j, i] = time.perf_counter() - start_time

    # B- arpack (for small number of components only, too slow otherwise)
    if arpack_all or n_components < 100:
        print("  - arpack solver")
        for i in range(n_iter):
            start_time = time.perf_counter()
            a_pred = (
                KernelPCA(n_components, eigen_solver="arpack")
                .fit(X_train)
                .transform(X_test)

            a_time[j, i] = time.perf_counter() - start_time
            # check that the result is still correct despite the approx
            assert_array_almost_equal(np.abs(a_pred), np.abs(ref_pred))

    # C- randomized
    print("  - randomized solver")
    for i in range(n_iter):
        start_time = time.perf_counter()
        r_pred = (
            KernelPCA(n_components, eigen_solver="randomized")
            .fit(X_train)
            .transform(X_test)

        r_time[j, i] = time.perf_counter() - start_time
        # check that the result is still correct despite the approximation
        assert_array_almost_equal(np.abs(r_pred), np.abs(ref_pred))

# Compute statistics for the 3 methods
avg_ref_time = ref_time.mean(axis=1)
std_ref_time = ref_time.std(axis=1)
avg_a_time = a_time.mean(axis=1)
std_a_time = a_time.std(axis=1)
avg_r_time = r_time.mean(axis=1)
std_r_time = r_time.std(axis=1)

# 4- Plots
# --------
fig, ax = plt.subplots(figsize=(12, 8))

# Display 1 plot with error bars per method
ax.errorbar(
    n_compo_range,
    avg_ref_time,
    yerr=std_ref_time,
    marker="x",
    linestyle="",
    color="r",
    label="full",

ax.errorbar(
    n_compo_range,
    avg_a_time,
    yerr=std_a_time,
    marker="x",
    linestyle="",
    color="g",
    label="arpack",

ax.errorbar(
    n_compo_range,
    avg_r_time,
    yerr=std_r_time,
    marker="x",
    linestyle="",
    color="b",
    label="randomized",

ax.legend(loc="upper left")

# customize axes
ax.set_xscale("log")
ax.set_xlim(1, max(n_compo_range) * 1.1)
ax.set_ylabel("Execution time (s)")
ax.set_xlabel("n_components")

ax.set_title(
    "kPCA Execution time comparison on %i samples with %i "
    "features, according to the choice of `eigen_solver`"
    "" % (n_train, n_features)

plt.show()
