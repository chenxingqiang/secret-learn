"""
Split Learning Benchmark
========================
This benchmark runs in SL (Split Learning) mode where the model
is split across parties - each holds different layers/features.
Only activations/gradients are exchanged, preserving raw data privacy.

Original benchmark adapted for secretlearn.split_learning.
"""

from secretlearn.split_learning.tree.decision_tree_classifier import SLDecisionTreeClassifier
from secretlearn.split_learning.dummy.dummy_classifier import SLDummyClassifier
from secretlearn.split_learning.ensemble.extra_trees_classifier import SLExtraTreesClassifier
from secretlearn.split_learning.svm.linear_svc import SLLinearSVC
from secretlearn.split_learning.linear_models.logistic_regression import SLLogisticRegression
from secretlearn.split_learning.kernel_approximation.nystroem import SLNystroem
from secretlearn.split_learning.ensemble.random_forest_classifier import SLRandomForestClassifier

# Authors: The Secret-Learn developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
from time import time

import numpy as np
from joblib import Memory

from sklearn.datasets import fetch_openml, get_data_home
from xlearn.kernel_approximation import SLNystroem, SLRBFSampler
from sklearn.metrics import zero_one_loss
from xlearn.neural_network import MLPClassifier
from xlearn.pipeline import make_pipeline
from sklearn.utils import check_array

# Memoize the data extraction and memory map the resulting
# train / test splits in readonly mode
memory = Memory(os.path.join(get_data_home(), "mnist_benchmark_data"), mmap_mode="r")

@memory.cache
def load_data(dtype=np.float32, order="F"):
    """Load the data, then cache and memmap the train/test split"""
    ######################################################################
    # Load dataset
    print("Loading dataset...")
    data = fetch_openml("mnist_784", as_frame=True)
    X = check_array(data["data"], dtype=dtype, order=order)
    y = data["target"]

    # Normalize features
    X = X / 255

    # Create train-test split (as [Joachims, 2006])
    print("Creating train-test split...")
    n_train = 60000
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test

ESTIMATORS = {
    "dummy": SLDummyClassifier(),
    "CART": SLDecisionTreeClassifier(),
    "ExtraTrees": SLExtraTreesClassifier(),
    "RandomForest": SLRandomForestClassifier(),
    "SLNystroem-SVM": make_pipeline(
        SLNystroem(gamma=0.015, n_components=1000), SLLinearSVC(C=100)
    ),
    "SampledRBF-SVM": make_pipeline(
        SLRBFSampler(gamma=0.015, n_components=1000), SLLinearSVC(C=100)
    ),
    "SLLogisticRegression-SAG": SLLogisticRegression(solver="sag", tol=1e-1, C=1e4),
    "SLLogisticRegression-SAGA": SLLogisticRegression(solver="saga", tol=1e-1, C=1e4),
    "MultilayerPerceptron": MLPClassifier(
        hidden_layer_sizes=(100, 100),
        max_iter=400,
        alpha=1e-4,
        solver="sgd",
        learning_rate_init=0.2,
        momentum=0.9,
        verbose=1,
        tol=1e-4,
        random_state=1,
    ),
    "MLP-adam": MLPClassifier(
        hidden_layer_sizes=(100, 100),
        max_iter=400,
        alpha=1e-4,
        solver="adam",
        learning_rate_init=0.001,
        verbose=1,
        tol=1e-4,
        random_state=1,
    ),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=ESTIMATORS,
        type=str,
        default=["ExtraTrees", "SLNystroem-SVM"],
        help="list of classifiers to benchmark.",

    parser.add_argument(
        "--n-jobs",
        nargs="?",
        default=1,
        type=int,
        help=(
            "Number of concurrently running workers for "
            "models that support parallelism."
        ),

    parser.add_argument(
        "--order",
        nargs="?",
        default="C",
        type=str,
        choices=["F", "C"],
        help="Allow to choose between fortran and C ordered data",

    parser.add_argument(
        "--random-seed",
        nargs="?",
        default=0,
        type=int,
        help="Common seed used by random number generator.",

    args = vars(parser.parse_args())

    print(__doc__)

    X_train, X_test, y_train, y_test = load_data(order=args["order"])

    print("")
    print("Dataset statistics:")
    print("===================")
    print("%s %d" % ("number of features:".ljust(25), X_train.shape[1]))
    print("%s %d" % ("number of classes:".ljust(25), np.unique(y_train).size))
    print("%s %s" % ("data type:".ljust(25), X_train.dtype))
    print(
        "%s %d (size=%dMB)"
        % (
            "number of train samples:".ljust(25),
            X_train.shape[0],
            int(X_train.nbytes / 1e6),

    print(
        "%s %d (size=%dMB)"
        % (
            "number of test samples:".ljust(25),
            X_test.shape[0],
            int(X_test.nbytes / 1e6),

    print()
    print("Training Classifiers")
    print("====================")
    error, train_time, test_time = {}, {}, {}
    for name in sorted(args["classifiers"]):
        print("Training %s ... " % name, end="")
        estimator = ESTIMATORS[name]
        estimator_params = estimator.get_params()

        estimator.set_params(
            **{
                p: args["random_seed"]
                for p in estimator_params
                if p.endswith("random_state")
            }

        if "n_jobs" in estimator_params:
            estimator.set_params(n_jobs=args["n_jobs"])

        time_start = time()
        estimator.fit(X_train, y_train)
        train_time[name] = time() - time_start

        time_start = time()
        y_pred = estimator.predict(X_test)
        test_time[name] = time() - time_start

        error[name] = zero_one_loss(y_test, y_pred)

        print("done")

    print()
    print("Classification performance:")
    print("===========================")
    print(
        "{0: <24} {1: >10} {2: >11} {3: >12}".format(
            "Classifier  ", "train-time", "test-time", "error-rate"

    print("-" * 60)
    for name in sorted(args["classifiers"], key=error.get):
        print(
            "{0: <23} {1: >10.2f}s {2: >10.2f}s {3: >12.4f}".format(
                name, train_time[name], test_time[name], error[name]

    print()
