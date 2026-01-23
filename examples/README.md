# Secret-Learn Examples

This directory contains **579 example files** demonstrating privacy-preserving machine learning across three modes.

## Directory Structure

```
examples/
├── federated_learning/    # 193 FL examples
├── secret_sharing/        # 193 SS examples
└── split_learning/        # 193 SL examples
```

## Privacy-Preserving Modes

| Mode | Prefix | Description | Use Case |
|------|--------|-------------|----------|
| **Federated Learning** | `FL` | Horizontal data partitioning, local training with secure aggregation | Multiple organizations with similar data schemas |
| **Secret Sharing** | `SS` | Multi-party computation via SPU, encrypted computation | Maximum security, small-medium datasets |
| **Split Learning** | `SL` | Model layer splitting, only activations exchanged | Vertical data partitioning, large models |

## Quick Start

### 1. Federated Learning (FL)

```python
import secretflow as sf
import numpy as np
from secretflow.data import FedNdarray, PartitionWay

# Initialize SecretFlow
sf.init(['alice', 'bob'])
alice, bob = sf.PYU('alice'), sf.PYU('bob')

# Create federated data (horizontal partitioning)
X = np.random.randn(100, 10)
y = np.random.randn(100)

fed_X = FedNdarray(
    partitions={
        alice: alice(lambda x: x)(X[:50]),
        bob: bob(lambda x: x)(X[50:]),
    },
    partition_way=PartitionWay.HORIZONTAL
)

fed_y = FedNdarray(
    partitions={
        alice: alice(lambda x: x)(y[:50]),
        bob: bob(lambda x: x)(y[50:]),
    },
    partition_way=PartitionWay.HORIZONTAL
)

# Use FL algorithms
from secretlearn.federated_learning.linear_models.linear_regression import FLLinearRegression

model = FLLinearRegression(devices={'alice': alice, 'bob': bob})
model.fit(fed_X, fed_y)
predictions = model.predict(fed_X)
```

### 2. Secret Sharing (SS)

```python
import secretflow as sf
import numpy as np

# Initialize with SPU
sf.init(['alice', 'bob'])
alice, bob = sf.PYU('alice'), sf.PYU('bob')

spu = sf.SPU(
    sf.utils.testing.cluster_def(['alice', 'bob']),
    link_desc={'recv_timeout_ms': 3600000}
)

# Use SS algorithms - computations happen on encrypted data
from secretlearn.secret_sharing.decomposition.pca import SSPCA

model = SSPCA(spu=spu, devices={'alice': alice, 'bob': bob}, n_components=5)
model.fit(fed_X)  # Unsupervised - no y needed
transformed = model.transform(fed_X)
```

### 3. Split Learning (SL)

```python
import secretflow as sf
import numpy as np
from secretflow.data import FedNdarray, PartitionWay

# Initialize SecretFlow
sf.init(['alice', 'bob'])
alice, bob = sf.PYU('alice'), sf.PYU('bob')

# Create vertically partitioned data
X = np.random.randn(100, 10)
fed_X = FedNdarray(
    partitions={
        alice: alice(lambda x: x)(X[:, :5]),   # Alice has first 5 features
        bob: bob(lambda x: x)(X[:, 5:]),       # Bob has last 5 features
    },
    partition_way=PartitionWay.VERTICAL
)

# Use SL algorithms - model is split across parties
from secretlearn.split_learning.clustering.kmeans import SLKMeans

model = SLKMeans(devices={'alice': alice, 'bob': bob}, n_clusters=3)
model.fit(fed_X)
labels = model.predict(fed_X)
```

## Algorithm Categories

Each mode supports **191 algorithms** across these categories:

| Category | Examples | Count |
|----------|----------|-------|
| Linear Models | LinearRegression, LogisticRegression, Lasso, Ridge, SGD | 39 |
| Ensemble | RandomForest, GradientBoosting, AdaBoost, ExtraTrees | 18 |
| Clustering | KMeans, DBSCAN, AgglomerativeClustering | 14 |
| Decomposition | PCA, SVD, NMF, ICA | 14 |
| Tree | DecisionTree, ExtraTree | 6 |
| SVM | SVC, SVR, LinearSVC, OneClassSVM | 7 |
| Neighbors | KNeighbors, RadiusNeighbors, NearestNeighbors | 10 |
| Naive Bayes | GaussianNB, MultinomialNB, BernoulliNB | 5 |
| Neural Network | MLPClassifier, MLPRegressor | 2 |
| Others | Preprocessing, Feature Selection, Manifold, etc. | 76 |

## Running Examples

```bash
# Run a specific example
python examples/federated_learning/linear_models/linear_regression.py

# Run all FL examples (in background)
python scripts/test_all_fl_examples.py

# Run all SS examples
python scripts/test_all_ss_examples.py

# Run all SL examples
python scripts/test_all_sl_examples.py
```

## Naming Convention

All algorithms follow a consistent naming pattern:

```python
# Pattern: {Mode}{AlgorithmName}
# FL Mode
from secretlearn.federated_learning.{category}.{algorithm} import FL{Algorithm}

# SS Mode
from secretlearn.secret_sharing.{category}.{algorithm} import SS{Algorithm}

# SL Mode
from secretlearn.split_learning.{category}.{algorithm} import SL{Algorithm}
```

**Examples:**
```python
# KMeans across all modes
from secretlearn.federated_learning.clustering.kmeans import FLKMeans
from secretlearn.secret_sharing.clustering.kmeans import SSKMeans
from secretlearn.split_learning.clustering.kmeans import SLKMeans

# PCA across all modes
from secretlearn.federated_learning.decomposition.pca import FLPCA
from secretlearn.secret_sharing.decomposition.pca import SSPCA
from secretlearn.split_learning.decomposition.pca import SLPCA
```

## Supervised vs Unsupervised

```python
# Supervised algorithms (require y)
model.fit(X, y)

# Unsupervised algorithms (no y needed)
model.fit(X)
```

**Unsupervised algorithms include:** KMeans, PCA, DBSCAN, IsolationForest, LocalOutlierFactor, NMF, etc.

## Documentation

For detailed API documentation, see:
- [Architecture Overview](../doc/ARCHITECTURE.md)
- [Main README](../README.md)
- Individual module docstrings in `secretlearn/`
