#!/usr/bin/env python3
"""
Convert benchmarks to Federated Learning mode.
Transforms xlearn/sklearn imports to secretlearn.split_learning.
"""
import os
import re
from pathlib import Path

# Algorithm mapping: sklearn class -> (module_path, SL class)
ALGORITHM_MAP = {
    # Linear Models
    'LogisticRegression': 'SLLogisticRegression',
    'Lasso': 'SLLasso',
    'LassoLars': 'SLLassoLars',
    'LassoCV': 'SLLassoCV',
    'Ridge': 'SLRidge',
    'RidgeCV': 'SLRidgeCV',
    'LinearRegression': 'SLLinearRegression',
    'SGDClassifier': 'SLSGDClassifier',
    'SGDRegressor': 'SLSGDRegressor',
    'ElasticNet': 'SLElasticNet',
    'Lars': 'SLLars',
    'OrthogonalMatchingPursuit': 'SLOrthogonalMatchingPursuit',
    'Perceptron': 'SLPerceptron',
    'PassiveAggressiveClassifier': 'SLPassiveAggressiveClassifier',
    'TweedieRegressor': 'SLTweedieRegressor',
    'PoissonRegressor': 'SLPoissonRegressor',
    'GammaRegressor': 'SLGammaRegressor',
    
    # Ensemble
    'RandomForestClassifier': 'SLRandomForestClassifier',
    'RandomForestRegressor': 'SLRandomForestRegressor',
    'ExtraTreesClassifier': 'SLExtraTreesClassifier',
    'ExtraTreesRegressor': 'SLExtraTreesRegressor',
    'AdaBoostClassifier': 'SLAdaBoostClassifier',
    'AdaBoostRegressor': 'SLAdaBoostRegressor',
    'GradientBoostingClassifier': 'SLGradientBoostingClassifier',
    'GradientBoostingRegressor': 'SLGradientBoostingRegressor',
    'HistGradientBoostingClassifier': 'SLHistGradientBoostingClassifier',
    'HistGradientBoostingRegressor': 'SLHistGradientBoostingRegressor',
    'BaggingClassifier': 'SLBaggingClassifier',
    
    # Decomposition
    'PCA': 'SLPCA',
    'IncrementalPCA': 'SLIncrementalPCA',
    'TruncatedSVD': 'SLTruncatedSVD',
    'NMF': 'SLNMF',
    'MiniBatchNMF': 'SLMiniBatchNMF',
    'FastICA': 'SLFastICA',
    'SparseCoder': 'SLSparseCoder',
    'DictionaryLearning': 'SLDictionaryLearning',
    
    # Clustering
    'KMeans': 'SLKMeans',
    'MiniBatchKMeans': 'SLMiniBatchKMeans',
    'AgglomerativeClustering': 'SLAgglomerativeClustering',
    
    # Tree
    'DecisionTreeClassifier': 'SLDecisionTreeClassifier',
    'DecisionTreeRegressor': 'SLDecisionTreeRegressor',
    'ExtraTreeClassifier': 'SLExtraTreeClassifier',
    'ExtraTreeRegressor': 'SLExtraTreeRegressor',
    
    # Naive Bayes
    'MultinomialNB': 'SLMultinomialNB',
    'GaussianNB': 'SLGaussianNB',
    'BernoulliNB': 'SLBernoulliNB',
    
    # Neighbors
    'KNeighborsClassifier': 'SLKNeighborsClassifier',
    'KNeighborsRegressor': 'SLKNeighborsRegressor',
    'NearestNeighbors': 'SLNearestNeighbors',
    'LocalOutlierFactor': 'SLLocalOutlierFactor',
    'RadiusNeighborsClassifier': 'SLRadiusNeighborsClassifier',
    
    # SVM
    'SVC': 'SLSVC',
    'SVR': 'SLSVR',
    'OneClassSVM': 'SLOneClassSVM',
    'LinearSVC': 'SLLinearSVC',
    'LinearSVR': 'SLLinearSVR',
    'NuSVC': 'SLNuSVC',
    
    # Anomaly Detection
    'IsolationForest': 'SLIsolationForest',
    
    # Preprocessing
    'StandardScaler': 'SLStandardScaler',
    'MinMaxScaler': 'SLMinMaxScaler',
    'PolynomialFeatures': 'SLPolynomialFeatures',
    
    # Manifold
    'TSNE': 'SLTSNE',
    
    # Dummy
    'DummyClassifier': 'SLDummyClassifier',
    'DummyRegressor': 'SLDummyRegressor',
    
    # Random Projection
    'GaussianRandomProjection': 'SLGaussianRandomProjection',
    'SparseRandomProjection': 'SLSparseRandomProjection',
    
    # Kernel Approximation
    'Nystroem': 'SLNystroem',
    'RBFSampler': 'SLRBFSampler',
    'AdditiveChi2Sampler': 'SLAdditiveChi2Sampler',
    'PolynomialCountSketch': 'SLPolynomialCountSketch',
    
    # Isotonic
    'IsotonicRegression': 'SLIsotonicRegression',
}

SL_HEADER = '''"""
Split Learning Benchmark
========================
This benchmark runs in SL (Split Learning) mode where the model
is split across parties - each holds different layers/features.
Only activations/gradients are exchanged, preserving raw data privacy.

Original benchmark adapted for secretlearn.split_learning.
"""

'''

# Module mapping for imports
MODULE_MAP = {
    'SLLogisticRegression': 'secretlearn.split_learning.linear_models.logistic_regression',
    'SLLasso': 'secretlearn.split_learning.linear_models.lasso',
    'SLLassoLars': 'secretlearn.split_learning.linear_models.lasso_lars',
    'SLRidge': 'secretlearn.split_learning.linear_models.ridge',
    'SLLinearRegression': 'secretlearn.split_learning.linear_models.linear_regression',
    'SLSGDClassifier': 'secretlearn.split_learning.linear_models.sgd_classifier',
    'SLSGDRegressor': 'secretlearn.split_learning.linear_models.sgd_regressor',
    'SLElasticNet': 'secretlearn.split_learning.linear_models.elastic_net',
    'SLLars': 'secretlearn.split_learning.linear_models.lars',
    'SLPerceptron': 'secretlearn.split_learning.linear_models.perceptron',
    'SLPassiveAggressiveClassifier': 'secretlearn.split_learning.linear_models.passive_aggressive_classifier',
    'SLTweedieRegressor': 'secretlearn.split_learning.linear_models.tweedie_regressor',
    'SLPoissonRegressor': 'secretlearn.split_learning.linear_models.poisson_regressor',
    'SLGammaRegressor': 'secretlearn.split_learning.linear_models.gamma_regressor',
    'SLRandomForestClassifier': 'secretlearn.split_learning.ensemble.random_forest_classifier',
    'SLRandomForestRegressor': 'secretlearn.split_learning.ensemble.random_forest_regressor',
    'SLExtraTreesClassifier': 'secretlearn.split_learning.ensemble.extra_trees_classifier',
    'SLExtraTreesRegressor': 'secretlearn.split_learning.ensemble.extra_trees_regressor',
    'SLAdaBoostClassifier': 'secretlearn.split_learning.ensemble.adaboost_classifier',
    'SLGradientBoostingClassifier': 'secretlearn.split_learning.ensemble.gradient_boosting_classifier',
    'SLGradientBoostingRegressor': 'secretlearn.split_learning.ensemble.gradient_boosting_regressor',
    'SLHistGradientBoostingClassifier': 'secretlearn.split_learning.ensemble.histgradient_boosting_classifier',
    'SLHistGradientBoostingRegressor': 'secretlearn.split_learning.ensemble.histgradient_boosting_regressor',
    'SLPCA': 'secretlearn.split_learning.decomposition.pca',
    'SLIncrementalPCA': 'secretlearn.split_learning.decomposition.incremental_pca',
    'SLTruncatedSVD': 'secretlearn.split_learning.decomposition.truncated_svd',
    'SLNMF': 'secretlearn.split_learning.decomposition.nmf',
    'SLMiniBatchNMF': 'secretlearn.split_learning.decomposition.mini_batch_nmf',
    'SLFastICA': 'secretlearn.split_learning.decomposition.fast_ica',
    'SLKMeans': 'secretlearn.split_learning.clustering.kmeans',
    'SLMiniBatchKMeans': 'secretlearn.split_learning.clustering.mini_batch_kmeans',
    'SLAgglomerativeClustering': 'secretlearn.split_learning.clustering.agglomerative_clustering',
    'SLDecisionTreeClassifier': 'secretlearn.split_learning.tree.decision_tree_classifier',
    'SLDecisionTreeRegressor': 'secretlearn.split_learning.tree.decision_tree_regressor',
    'SLMultinomialNB': 'secretlearn.split_learning.naive_bayes.multinomial_nb',
    'SLGaussianNB': 'secretlearn.split_learning.naive_bayes.gaussian_nb',
    'SLKNeighborsClassifier': 'secretlearn.split_learning.neighbors.k_neighbors_classifier',
    'SLKNeighborsRegressor': 'secretlearn.split_learning.neighbors.k_neighbors_regressor',
    'SLNearestNeighbors': 'secretlearn.split_learning.neighbors.nearest_neighbors',
    'SLLocalOutlierFactor': 'secretlearn.split_learning.neighbors.local_outlier_factor',
    'SLSVC': 'secretlearn.split_learning.svm.svc',
    'SLSVR': 'secretlearn.split_learning.svm.svr',
    'SLOneClassSVM': 'secretlearn.split_learning.svm.one_class_svm',
    'SLLinearSVC': 'secretlearn.split_learning.svm.linear_svc',
    'SLLinearSVR': 'secretlearn.split_learning.svm.linear_svr',
    'SLNuSVC': 'secretlearn.split_learning.svm.nusvc',
    'SLIsolationForest': 'secretlearn.split_learning.anomaly_detection.isolation_forest',
    'SLStandardScaler': 'secretlearn.split_learning.preprocessing.standard_scaler',
    'SLTSNE': 'secretlearn.split_learning.manifold.tsne',
    'SLDummyClassifier': 'secretlearn.split_learning.dummy.dummy_classifier',
    'SLDummyRegressor': 'secretlearn.split_learning.dummy.dummy_regressor',
    'SLGaussianRandomProjection': 'secretlearn.split_learning.random_projection.gaussian_random_projection',
    'SLSparseRandomProjection': 'secretlearn.split_learning.random_projection.sparse_random_projection',
    'SLNystroem': 'secretlearn.split_learning.kernel_approximation.nystroem',
    'SLIsotonicRegression': 'secretlearn.split_learning.isotonic.isotonic_regression',
}


def convert_file(filepath: Path) -> bool:
    """Convert a single benchmark file to SL mode."""
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
    
    # Collect SL classes used
    fl_classes_used = set()
    
    # Replace class names and track usage
    for sklearn_class, fl_class in ALGORITHM_MAP.items():
        if re.search(rf'\b{sklearn_class}\b', content):
            fl_classes_used.add(fl_class)
        content = re.sub(rf'(?<!SL)\b{sklearn_class}\b(?!\s*=)', fl_class, content)
    
    # Clean up orphaned import lines
    content = re.sub(r'^\s*[A-Z][a-zA-Z]+,\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*\)\s*$', '', content, flags=re.MULTILINE)
    
    # Remove multiple consecutive blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Generate SL imports
    fl_imports = []
    for fl_class in sorted(fl_classes_used):
        if fl_class in MODULE_MAP:
            fl_imports.append(f'from {MODULE_MAP[fl_class]} import {fl_class}')
    
    # Insert imports after the header
    import_block = '\n'.join(fl_imports) + '\n\n' if fl_imports else ''
    
    # Add SL header with imports
    content = SL_HEADER + import_block + content.lstrip()
    
    if content != original:
        filepath.write_text(content)
        return True
    return False


def main():
    benchmark_dir = Path(__file__).parent.parent / 'benchmarks' / 'secret-learn' / 'split_learning'
    
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
