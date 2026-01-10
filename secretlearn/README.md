# Secret-Learn

**Privacy-Preserving Machine Learning with JAX Acceleration**

573 sklearn-compatible implementations for SecretFlow.

## Installation

```bash
pip install secret-learn
```

## Quick Start

```python
import secretflow as sf
from secretlearn.federated_learning.linear_models.linear_regression import FLLinearRegression
from secretlearn.secret_sharing.clustering.kmeans import SSKMeans
from secretlearn.split_learning.ensemble.random_forest_classifier import SLRandomForestClassifier

# Initialize
sf.init(['alice', 'bob'])
alice, bob = sf.PYU('alice'), sf.PYU('bob')

# FL Mode - Horizontal partitioning
model = FLLinearRegression(devices={'alice': alice, 'bob': bob})
model.fit(fed_X, fed_y)

# SS Mode - MPC encryption via SPU
model = SSKMeans(spu=spu, n_clusters=3)
model.fit(fed_X)

# SL Mode - Vertical partitioning
model = SLRandomForestClassifier(devices={'alice': alice, 'bob': bob})
model.fit(fed_X, fed_y)
```

## Three Privacy Modes

| Mode | Directory | Use Case | Data |
|------|-----------|----------|------|
| **FL** | `federated_learning/` | Multiple orgs, same schema | Horizontal |
| **SS** | `secret_sharing/` | Max security, MPC via SPU | Encrypted |
| **SL** | `split_learning/` | Feature split, large models | Vertical |

## 191 Algorithms × 3 Modes = 573 Implementations

| Category | Count | Examples |
|----------|-------|----------|
| Linear Models | 39 | LinearRegression, Lasso, Ridge, LogisticRegression |
| Ensemble | 18 | RandomForest, GradientBoosting, AdaBoost |
| Preprocessing | 19 | StandardScaler, MinMaxScaler, OneHotEncoder |
| Clustering | 14 | KMeans, DBSCAN, AgglomerativeClustering |
| Decomposition | 14 | PCA, SVD, NMF, FastICA |
| Feature Selection | 12 | SelectKBest, RFE, VarianceThreshold |
| Neighbors | 11 | KNeighborsClassifier, LocalOutlierFactor |
| Covariance | 8 | EmpiricalCovariance, GraphicalLasso |
| SVM | 7 | SVC, SVR, LinearSVC, OneClassSVM |
| Others | 49 | NaiveBayes, Tree, Manifold, etc. |

## Import Pattern

```python
# Pattern: secretlearn.{mode}.{category}.{algorithm}
from secretlearn.federated_learning.clustering.kmeans import FLKMeans
from secretlearn.secret_sharing.clustering.kmeans import SSKMeans
from secretlearn.split_learning.clustering.kmeans import SLKMeans
```

## Key Features

- **100% sklearn API** - Drop-in replacement
- **JAX Acceleration** - 5x+ speedup
- **SecretFlow Integration** - SPU/HEU/TEE support
- **Secure Aggregation** - Privacy-preserving parameter sharing

## Links

- [GitHub](https://github.com/chenxingqiang/secret-learn)
- [PyPI](https://pypi.org/project/secret-learn/)
- [SecretFlow](https://github.com/secretflow/secretflow)

---

**Version**: 0.3.0 | **License**: BSD-3-Clause
