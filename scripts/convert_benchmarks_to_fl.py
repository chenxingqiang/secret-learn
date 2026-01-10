#!/usr/bin/env python3
"""
Convert benchmarks to Federated Learning mode.
Transforms xlearn/sklearn imports to secretlearn.federated_learning.
"""
import os
import re
from pathlib import Path

# Algorithm mapping: sklearn class -> (module_path, FL class)
ALGORITHM_MAP = {
    # Linear Models
    'LogisticRegression': 'FLLogisticRegression',
    'Lasso': 'FLLasso',
    'LassoLars': 'FLLassoLars',
    'LassoCV': 'FLLassoCV',
    'Ridge': 'FLRidge',
    'RidgeCV': 'FLRidgeCV',
    'LinearRegression': 'FLLinearRegression',
    'SGDClassifier': 'FLSGDClassifier',
    'SGDRegressor': 'FLSGDRegressor',
    'ElasticNet': 'FLElasticNet',
    'Lars': 'FLLars',
    'OrthogonalMatchingPursuit': 'FLOrthogonalMatchingPursuit',
    'Perceptron': 'FLPerceptron',
    'PassiveAggressiveClassifier': 'FLPassiveAggressiveClassifier',
    'TweedieRegressor': 'FLTweedieRegressor',
    'PoissonRegressor': 'FLPoissonRegressor',
    'GammaRegressor': 'FLGammaRegressor',
    
    # Ensemble
    'RandomForestClassifier': 'FLRandomForestClassifier',
    'RandomForestRegressor': 'FLRandomForestRegressor',
    'ExtraTreesClassifier': 'FLExtraTreesClassifier',
    'ExtraTreesRegressor': 'FLExtraTreesRegressor',
    'AdaBoostClassifier': 'FLAdaBoostClassifier',
    'AdaBoostRegressor': 'FLAdaBoostRegressor',
    'GradientBoostingClassifier': 'FLGradientBoostingClassifier',
    'GradientBoostingRegressor': 'FLGradientBoostingRegressor',
    'HistGradientBoostingClassifier': 'FLHistGradientBoostingClassifier',
    'HistGradientBoostingRegressor': 'FLHistGradientBoostingRegressor',
    'BaggingClassifier': 'FLBaggingClassifier',
    
    # Decomposition
    'PCA': 'FLPCA',
    'IncrementalPCA': 'FLIncrementalPCA',
    'TruncatedSVD': 'FLTruncatedSVD',
    'NMF': 'FLNMF',
    'MiniBatchNMF': 'FLMiniBatchNMF',
    'FastICA': 'FLFastICA',
    'SparseCoder': 'FLSparseCoder',
    'DictionaryLearning': 'FLDictionaryLearning',
    
    # Clustering
    'KMeans': 'FLKMeans',
    'MiniBatchKMeans': 'FLMiniBatchKMeans',
    'AgglomerativeClustering': 'FLAgglomerativeClustering',
    
    # Tree
    'DecisionTreeClassifier': 'FLDecisionTreeClassifier',
    'DecisionTreeRegressor': 'FLDecisionTreeRegressor',
    'ExtraTreeClassifier': 'FLExtraTreeClassifier',
    'ExtraTreeRegressor': 'FLExtraTreeRegressor',
    
    # Naive Bayes
    'MultinomialNB': 'FLMultinomialNB',
    'GaussianNB': 'FLGaussianNB',
    'BernoulliNB': 'FLBernoulliNB',
    
    # Neighbors
    'KNeighborsClassifier': 'FLKNeighborsClassifier',
    'KNeighborsRegressor': 'FLKNeighborsRegressor',
    'NearestNeighbors': 'FLNearestNeighbors',
    'LocalOutlierFactor': 'FLLocalOutlierFactor',
    'RadiusNeighborsClassifier': 'FLRadiusNeighborsClassifier',
    
    # SVM
    'SVC': 'FLSVC',
    'SVR': 'FLSVR',
    'OneClassSVM': 'FLOneClassSVM',
    'LinearSVC': 'FLLinearSVC',
    'LinearSVR': 'FLLinearSVR',
    'NuSVC': 'FLNuSVC',
    
    # Anomaly Detection
    'IsolationForest': 'FLIsolationForest',
    
    # Preprocessing
    'StandardScaler': 'FLStandardScaler',
    'MinMaxScaler': 'FLMinMaxScaler',
    'PolynomialFeatures': 'FLPolynomialFeatures',
    
    # Manifold
    'TSNE': 'FLTSNE',
    
    # Dummy
    'DummyClassifier': 'FLDummyClassifier',
    'DummyRegressor': 'FLDummyRegressor',
    
    # Random Projection
    'GaussianRandomProjection': 'FLGaussianRandomProjection',
    'SparseRandomProjection': 'FLSparseRandomProjection',
    
    # Kernel Approximation
    'Nystroem': 'FLNystroem',
    'RBFSampler': 'FLRBFSampler',
    'AdditiveChi2Sampler': 'FLAdditiveChi2Sampler',
    'PolynomialCountSketch': 'FLPolynomialCountSketch',
    
    # Isotonic
    'IsotonicRegression': 'FLIsotonicRegression',
}

FL_HEADER = '''"""
Federated Learning Benchmark
============================
This benchmark runs in FL (Federated Learning) mode where data is
horizontally partitioned across multiple parties (alice, bob).
Each party trains locally, then aggregates model parameters securely.

Original benchmark adapted for secretlearn.federated_learning.
"""

'''

# Module mapping for imports
MODULE_MAP = {
    'FLLogisticRegression': 'secretlearn.federated_learning.linear_models.logistic_regression',
    'FLLasso': 'secretlearn.federated_learning.linear_models.lasso',
    'FLLassoLars': 'secretlearn.federated_learning.linear_models.lasso_lars',
    'FLRidge': 'secretlearn.federated_learning.linear_models.ridge',
    'FLLinearRegression': 'secretlearn.federated_learning.linear_models.linear_regression',
    'FLSGDClassifier': 'secretlearn.federated_learning.linear_models.sgd_classifier',
    'FLSGDRegressor': 'secretlearn.federated_learning.linear_models.sgd_regressor',
    'FLElasticNet': 'secretlearn.federated_learning.linear_models.elastic_net',
    'FLLars': 'secretlearn.federated_learning.linear_models.lars',
    'FLPerceptron': 'secretlearn.federated_learning.linear_models.perceptron',
    'FLPassiveAggressiveClassifier': 'secretlearn.federated_learning.linear_models.passive_aggressive_classifier',
    'FLTweedieRegressor': 'secretlearn.federated_learning.linear_models.tweedie_regressor',
    'FLPoissonRegressor': 'secretlearn.federated_learning.linear_models.poisson_regressor',
    'FLGammaRegressor': 'secretlearn.federated_learning.linear_models.gamma_regressor',
    'FLRandomForestClassifier': 'secretlearn.federated_learning.ensemble.random_forest_classifier',
    'FLRandomForestRegressor': 'secretlearn.federated_learning.ensemble.random_forest_regressor',
    'FLExtraTreesClassifier': 'secretlearn.federated_learning.ensemble.extra_trees_classifier',
    'FLExtraTreesRegressor': 'secretlearn.federated_learning.ensemble.extra_trees_regressor',
    'FLAdaBoostClassifier': 'secretlearn.federated_learning.ensemble.adaboost_classifier',
    'FLGradientBoostingClassifier': 'secretlearn.federated_learning.ensemble.gradient_boosting_classifier',
    'FLGradientBoostingRegressor': 'secretlearn.federated_learning.ensemble.gradient_boosting_regressor',
    'FLHistGradientBoostingClassifier': 'secretlearn.federated_learning.ensemble.histgradient_boosting_classifier',
    'FLHistGradientBoostingRegressor': 'secretlearn.federated_learning.ensemble.histgradient_boosting_regressor',
    'FLPCA': 'secretlearn.federated_learning.decomposition.pca',
    'FLIncrementalPCA': 'secretlearn.federated_learning.decomposition.incremental_pca',
    'FLTruncatedSVD': 'secretlearn.federated_learning.decomposition.truncated_svd',
    'FLNMF': 'secretlearn.federated_learning.decomposition.nmf',
    'FLMiniBatchNMF': 'secretlearn.federated_learning.decomposition.mini_batch_nmf',
    'FLFastICA': 'secretlearn.federated_learning.decomposition.fast_ica',
    'FLKMeans': 'secretlearn.federated_learning.clustering.kmeans',
    'FLMiniBatchKMeans': 'secretlearn.federated_learning.clustering.mini_batch_kmeans',
    'FLAgglomerativeClustering': 'secretlearn.federated_learning.clustering.agglomerative_clustering',
    'FLDecisionTreeClassifier': 'secretlearn.federated_learning.tree.decision_tree_classifier',
    'FLDecisionTreeRegressor': 'secretlearn.federated_learning.tree.decision_tree_regressor',
    'FLMultinomialNB': 'secretlearn.federated_learning.naive_bayes.multinomial_nb',
    'FLGaussianNB': 'secretlearn.federated_learning.naive_bayes.gaussian_nb',
    'FLKNeighborsClassifier': 'secretlearn.federated_learning.neighbors.k_neighbors_classifier',
    'FLKNeighborsRegressor': 'secretlearn.federated_learning.neighbors.k_neighbors_regressor',
    'FLNearestNeighbors': 'secretlearn.federated_learning.neighbors.nearest_neighbors',
    'FLLocalOutlierFactor': 'secretlearn.federated_learning.neighbors.local_outlier_factor',
    'FLSVC': 'secretlearn.federated_learning.svm.svc',
    'FLSVR': 'secretlearn.federated_learning.svm.svr',
    'FLOneClassSVM': 'secretlearn.federated_learning.svm.one_class_svm',
    'FLLinearSVC': 'secretlearn.federated_learning.svm.linear_svc',
    'FLLinearSVR': 'secretlearn.federated_learning.svm.linear_svr',
    'FLNuSVC': 'secretlearn.federated_learning.svm.nusvc',
    'FLIsolationForest': 'secretlearn.federated_learning.anomaly_detection.isolation_forest',
    'FLStandardScaler': 'secretlearn.federated_learning.preprocessing.standard_scaler',
    'FLTSNE': 'secretlearn.federated_learning.manifold.tsne',
    'FLDummyClassifier': 'secretlearn.federated_learning.dummy.dummy_classifier',
    'FLDummyRegressor': 'secretlearn.federated_learning.dummy.dummy_regressor',
    'FLGaussianRandomProjection': 'secretlearn.federated_learning.random_projection.gaussian_random_projection',
    'FLSparseRandomProjection': 'secretlearn.federated_learning.random_projection.sparse_random_projection',
    'FLNystroem': 'secretlearn.federated_learning.kernel_approximation.nystroem',
    'FLIsotonicRegression': 'secretlearn.federated_learning.isotonic.isotonic_regression',
}


def convert_file(filepath: Path) -> bool:
    """Convert a single benchmark file to FL mode."""
    content = filepath.read_text()
    original = content
    
    # Remove existing docstring
    content = re.sub(r'^"""[\s\S]*?"""\s*', '', content, count=1)
    content = re.sub(r"^'''[\s\S]*?'''\s*", '', content, count=1)
    
    # Remove old xlearn/sklearn multi-line imports completely
    content = re.sub(
        r'from (?:xlearn|sklearn)\.[a-z_.]+ import \([^)]+\)\s*',
        '',
        content,
        flags=re.DOTALL
    )
    
    # Remove single-line xlearn/sklearn imports for algorithms
    content = re.sub(
        r'from (?:xlearn|sklearn)\.(?:linear_model|ensemble|decomposition|cluster|tree|naive_bayes|neighbors|svm|dummy)[^\n]+\n',
        '',
        content
    )
    
    # Keep utility imports but redirect to sklearn
    content = re.sub(r'from xlearn\.utils', 'from sklearn.utils', content)
    content = re.sub(r'from xlearn\.datasets', 'from sklearn.datasets', content)
    content = re.sub(r'from xlearn\.metrics', 'from sklearn.metrics', content)
    content = re.sub(r'from xlearn\.model_selection', 'from sklearn.model_selection', content)
    content = re.sub(r'from xlearn\.preprocessing', 'from sklearn.preprocessing', content)
    
    # Collect FL classes used
    fl_classes_used = set()
    
    # Replace class names and track usage
    for sklearn_class, fl_class in ALGORITHM_MAP.items():
        if re.search(rf'\b{sklearn_class}\b', content):
            fl_classes_used.add(fl_class)
        content = re.sub(rf'(?<!FL)\b{sklearn_class}\b(?!\s*=)', fl_class, content)
    
    # Clean up orphaned import lines
    content = re.sub(r'^\s*[A-Z][a-zA-Z]+,\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*\)\s*$', '', content, flags=re.MULTILINE)
    
    # Remove multiple consecutive blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Generate FL imports
    fl_imports = []
    for fl_class in sorted(fl_classes_used):
        if fl_class in MODULE_MAP:
            fl_imports.append(f'from {MODULE_MAP[fl_class]} import {fl_class}')
    
    # Insert imports after the header
    import_block = '\n'.join(fl_imports) + '\n\n' if fl_imports else ''
    
    # Add FL header with imports
    content = FL_HEADER + import_block + content.lstrip()
    
    if content != original:
        filepath.write_text(content)
        return True
    return False


def main():
    benchmark_dir = Path(__file__).parent.parent / 'benchmarks' / 'secret-learn' / 'federated_learning'
    
    if not benchmark_dir.exists():
        print(f"Directory not found: {benchmark_dir}")
        return
    
    print("=" * 70)
    print(" Converting Benchmarks to Federated Learning Mode")
    print("=" * 70)
    
    files = list(benchmark_dir.glob('*.py'))
    converted = 0
    
    for filepath in sorted(files):
        print(f"Processing: {filepath.name}... ", end='')
        if convert_file(filepath):
            print("✓ converted")
            converted += 1
        else:
            print("- no changes")
    
    print()
    print(f"Converted: {converted}/{len(files)} files")
    print("=" * 70)


if __name__ == "__main__":
    main()
