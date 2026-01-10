"""
Split Learning Benchmark
========================
This benchmark runs in SL (Split Learning) mode where the model
is split across parties - each holds different layers/features.
Only activations/gradients are exchanged, preserving raw data privacy.

Original benchmark adapted for secretlearn.split_learning.
"""

from secretlearn.split_learning.ensemble.adaboost_classifier import SLAdaBoostClassifier
from secretlearn.split_learning.dummy.dummy_classifier import SLDummyClassifier
from secretlearn.split_learning.ensemble.extra_trees_classifier import SLExtraTreesClassifier
from secretlearn.split_learning.linear_models.logistic_regression import SLLogisticRegression
from secretlearn.split_learning.naive_bayes.multinomial_nb import SLMultinomialNB
from secretlearn.split_learning.ensemble.random_forest_classifier import SLRandomForestClassifier

import argparse
from time import time

import numpy as np

from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array

ESTIMATORS = {
    "dummy": SLDummyClassifier(),
    "random_forest": SLRandomForestClassifier(max_features="sqrt", min_samples_split=10),
    "extra_trees": SLExtraTreesClassifier(max_features="sqrt", min_samples_split=10),
    "logistic_regression": SLLogisticRegression(),
    "naive_bayes": SLMultinomialNB(),
    "adaboost": SLAdaBoostClassifier(n_estimators=10),
}

###############################################################################
# Data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--estimators", nargs="+", required=True, choices=ESTIMATORS

    args = vars(parser.parse_args())

    data_train = fetch_20newsgroups_vectorized(subset="train")
    data_test = fetch_20newsgroups_vectorized(subset="test")
    X_train = check_array(data_train.data, dtype=np.float32, accept_sparse="csc")
    X_test = check_array(data_test.data, dtype=np.float32, accept_sparse="csr")
    y_train = data_train.target
    y_test = data_test.target

    print("20 newsgroups")
    print("=============")
    print(f"X_train.shape = {X_train.shape}")
    print(f"X_train.format = {X_train.format}")
    print(f"X_train.dtype = {X_train.dtype}")
    print(f"X_train density = {X_train.nnz / np.prod(X_train.shape)}")
    print(f"y_train {y_train.shape}")
    print(f"X_test {X_test.shape}")
    print(f"X_test.format = {X_test.format}")
    print(f"X_test.dtype = {X_test.dtype}")
    print(f"y_test {y_test.shape}")
    print()
    print("Classifier Training")
    print("===================")
    accuracy, train_time, test_time = {}, {}, {}
    for name in sorted(args["estimators"]):
        clf = ESTIMATORS[name]
        try:
            clf.set_params(random_state=0)
        except (TypeError, ValueError):
            pass

        print("Training %s ... " % name, end="")
        t0 = time()
        clf.fit(X_train, y_train)
        train_time[name] = time() - t0
        t0 = time()
        y_pred = clf.predict(X_test)
        test_time[name] = time() - t0
        accuracy[name] = accuracy_score(y_test, y_pred)
        print("done")

    print()
    print("Classification performance:")
    print("===========================")
    print()
    print("%s %s %s %s" % ("Classifier  ", "train-time", "test-time", "Accuracy"))
    print("-" * 44)
    for name in sorted(accuracy, key=accuracy.get):
        print(
            "%s %s %s %s"
            % (
                name.ljust(16),
                ("%.4fs" % train_time[name]).center(10),
                ("%.4fs" % test_time[name]).center(10),
                ("%.4f" % accuracy[name]).center(10),

    print()
