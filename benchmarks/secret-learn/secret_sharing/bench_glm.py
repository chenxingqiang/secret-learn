"""
Secret Sharing Benchmark
========================
This benchmark runs in SS (Secret Sharing) mode where data is
securely split among parties using multi-party computation (MPC).
Computations are performed on encrypted/secret-shared data via SPU.

Original benchmark adapted for secretlearn.secret_sharing.
"""

from secretlearn.secret_sharing.linear_models.lasso_lars import SSLassoLars
from secretlearn.secret_sharing.linear_models.linear_regression import SSLinearRegression
from secretlearn.secret_sharing.linear_models.ridge import SSRidge

from datetime import datetime

import numpy as np

from secretlearn import linear_model

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n_iter = 40

    time_ridge = np.empty(n_iter)
    time_ols = np.empty(n_iter)
    time_lasso = np.empty(n_iter)

    dimensions = 500 * np.arange(1, n_iter + 1)

    for i in range(n_iter):
        print("Iteration %s of %s" % (i, n_iter))

        n_samples, n_features = 10 * i + 3, 10 * i + 3

        X = np.random.randn(n_samples, n_features)
        Y = np.random.randn(n_samples)

        start = datetime.now()
        ridge = linear_model.SSRidge(alpha=1.0)
        ridge.fit(X, Y)
        time_ridge[i] = (datetime.now() - start).total_seconds()

        start = datetime.now()
        ols = linear_model.SSLinearRegression()
        ols.fit(X, Y)
        time_ols[i] = (datetime.now() - start).total_seconds()

        start = datetime.now()
        lasso = linear_model.SSLassoLars()
        lasso.fit(X, Y)
        time_lasso[i] = (datetime.now() - start).total_seconds()

    plt.figure("Secret-Learn GLM benchmark results")
    plt.xlabel("Dimensions")
    plt.ylabel("Time (s)")
    plt.plot(dimensions, time_ridge, color="r")
    plt.plot(dimensions, time_ols, color="g")
    plt.plot(dimensions, time_lasso, color="b")

    plt.legend(["SSRidge", "OLS", "SSLassoLars"], loc="upper left")
    plt.axis("tight")
    plt.show()
