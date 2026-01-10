#!/usr/bin/env python3
"""
Convert benchmarks to Federated Learning mode.
Transforms xlearn/sklearn imports to secretlearn.secret_sharing.
"""
import os
import re
from pathlib import Path

# Algorithm mapping: sklearn class -> (module_path, SS class)
ALGORITHM_MAP = {
    # Linear Models
    'LogisticRegression': 'SSLogisticRegression',
    'Lasso': 'SSLasso',
    'LassoLars': 'SSLassoLars',
    'LassoCV': 'SSLassoCV',
    'Ridge': 'SSRidge',
    'RidgeCV': 'SSRidgeCV',
    'LinearRegression': 'SSLinearRegression',
    'SGDClassifier': 'SSSGDClassifier',
    'SGDRegressor': 'SSSGDRegressor',
    'ElasticNet': 'SSElasticNet',
    'Lars': 'SSLars',
    'OrthogonalMatchingPursuit': 'SSOrthogonalMatchingPursuit',
    'Perceptron': 'SSPerceptron',
    'PassiveAggressiveClassifier': 'SSPassiveAggressiveClassifier',
    'TweedieRegressor': 'SSTweedieRegressor',
    'PoissonRegressor': 'SSPoissonRegressor',
    'GammaRegressor': 'SSGammaRegressor',
    
    # Ensemble
    'RandomForestClassifier': 'SSRandomForestClassifier',
    'RandomForestRegressor': 'SSRandomForestRegressor',
    'ExtraTreesClassifier': 'SSExtraTreesClassifier',
    'ExtraTreesRegressor': 'SSExtraTreesRegressor',
    'AdaBoostClassifier': 'SSAdaBoostClassifier',
    'AdaBoostRegressor': 'SSAdaBoostRegressor',
    'GradientBoostingClassifier': 'SSGradientBoostingClassifier',
    'GradientBoostingRegressor': 'SSGradientBoostingRegressor',
    'HistGradientBoostingClassifier': 'SSHistGradientBoostingClassifier',
    'HistGradientBoostingRegressor': 'SSHistGradientBoostingRegressor',
    'BaggingClassifier': 'SSBaggingClassifier',
    
    # Decomposition
    'PCA': 'SSPCA',
    'IncrementalPCA': 'SSIncrementalPCA',
    'TruncatedSVD': 'SSTruncatedSVD',
    'NMF': 'SSNMF',
    'MiniBatchNMF': 'SSMiniBatchNMF',
    'FastICA': 'SSFastICA',
    'SparseCoder': 'SSSparseCoder',
    'DictionaryLearning': 'SSDictionaryLearning',
    
    # Clustering
    'KMeans': 'SSKMeans',
    'MiniBatchKMeans': 'SSMiniBatchKMeans',
    'AgglomerativeClustering': 'SSAgglomerativeClustering',
    
    # Tree
    'DecisionTreeClassifier': 'SSDecisionTreeClassifier',
    'DecisionTreeRegressor': 'SSDecisionTreeRegressor',
    'ExtraTreeClassifier': 'SSExtraTreeClassifier',
    'ExtraTreeRegressor': 'SSExtraTreeRegressor',
    
    # Naive Bayes
    'MultinomialNB': 'SSMultinomialNB',
    'GaussianNB': 'SSGaussianNB',
    'BernoulliNB': 'SSBernoulliNB',
    
    # Neighbors
    'KNeighborsClassifier': 'SSKNeighborsClassifier',
    'KNeighborsRegressor': 'SSKNeighborsRegressor',
    'NearestNeighbors': 'SSNearestNeighbors',
    'LocalOutlierFactor': 'SSLocalOutlierFactor',
    'RadiusNeighborsClassifier': 'SSRadiusNeighborsClassifier',
    
    # SVM
    'SVC': 'SSSVC',
    'SVR': 'SSSVR',
    'OneClassSVM': 'SSOneClassSVM',
    'LinearSVC': 'SSLinearSVC',
    'LinearSVR': 'SSLinearSVR',
    'NuSVC': 'SSNuSVC',
    
    # Anomaly Detection
    'IsolationForest': 'SSIsolationForest',
    
    # Preprocessing
    'StandardScaler': 'SSStandardScaler',
    'MinMaxScaler': 'SSMinMaxScaler',
    'PolynomialFeatures': 'SSPolynomialFeatures',
    
    # Manifold
    'TSNE': 'SSTSNE',
    
    # Dummy
    'DummyClassifier': 'SSDummyClassifier',
    'DummyRegressor': 'SSDummyRegressor',
    
    # Random Projection
    'GaussianRandomProjection': 'SSGaussianRandomProjection',
    'SparseRandomProjection': 'SSSparseRandomProjection',
    
    # Kernel Approximation
    'Nystroem': 'SSNystroem',
    'RBFSampler': 'SSRBFSampler',
    'AdditiveChi2Sampler': 'SSAdditiveChi2Sampler',
    'PolynomialCountSketch': 'SSPolynomialCountSketch',
    
    # Isotonic
    'IsotonicRegression': 'SSIsotonicRegression',
}

SS_HEADER = '''"""
Secret Sharing Benchmark
========================
This benchmark runs in SS (Secret Sharing) mode where data is
securely split among parties using multi-party computation (MPC).
Computations are performed on encrypted/secret-shared data via SPU.

Original benchmark adapted for secretlearn.secret_sharing.
"""

'''

# Module mapping for imports
MODULE_MAP = {
    'SSLogisticRegression': 'secretlearn.secret_sharing.linear_models.logistic_regression',
    'SSLasso': 'secretlearn.secret_sharing.linear_models.lasso',
    'SSLassoLars': 'secretlearn.secret_sharing.linear_models.lasso_lars',
    'SSRidge': 'secretlearn.secret_sharing.linear_models.ridge',
    'SSLinearRegression': 'secretlearn.secret_sharing.linear_models.linear_regression',
    'SSSGDClassifier': 'secretlearn.secret_sharing.linear_models.sgd_classifier',
    'SSSGDRegressor': 'secretlearn.secret_sharing.linear_models.sgd_regressor',
    'SSElasticNet': 'secretlearn.secret_sharing.linear_models.elastic_net',
    'SSLars': 'secretlearn.secret_sharing.linear_models.lars',
    'SSPerceptron': 'secretlearn.secret_sharing.linear_models.perceptron',
    'SSPassiveAggressiveClassifier': 'secretlearn.secret_sharing.linear_models.passive_aggressive_classifier',
    'SSTweedieRegressor': 'secretlearn.secret_sharing.linear_models.tweedie_regressor',
    'SSPoissonRegressor': 'secretlearn.secret_sharing.linear_models.poisson_regressor',
    'SSGammaRegressor': 'secretlearn.secret_sharing.linear_models.gamma_regressor',
    'SSRandomForestClassifier': 'secretlearn.secret_sharing.ensemble.random_forest_classifier',
    'SSRandomForestRegressor': 'secretlearn.secret_sharing.ensemble.random_forest_regressor',
    'SSExtraTreesClassifier': 'secretlearn.secret_sharing.ensemble.extra_trees_classifier',
    'SSExtraTreesRegressor': 'secretlearn.secret_sharing.ensemble.extra_trees_regressor',
    'SSAdaBoostClassifier': 'secretlearn.secret_sharing.ensemble.adaboost_classifier',
    'SSGradientBoostingClassifier': 'secretlearn.secret_sharing.ensemble.gradient_boosting_classifier',
    'SSGradientBoostingRegressor': 'secretlearn.secret_sharing.ensemble.gradient_boosting_regressor',
    'SSHistGradientBoostingClassifier': 'secretlearn.secret_sharing.ensemble.histgradient_boosting_classifier',
    'SSHistGradientBoostingRegressor': 'secretlearn.secret_sharing.ensemble.histgradient_boosting_regressor',
    'SSPCA': 'secretlearn.secret_sharing.decomposition.pca',
    'SSIncrementalPCA': 'secretlearn.secret_sharing.decomposition.incremental_pca',
    'SSTruncatedSVD': 'secretlearn.secret_sharing.decomposition.truncated_svd',
    'SSNMF': 'secretlearn.secret_sharing.decomposition.nmf',
    'SSMiniBatchNMF': 'secretlearn.secret_sharing.decomposition.mini_batch_nmf',
    'SSFastICA': 'secretlearn.secret_sharing.decomposition.fast_ica',
    'SSKMeans': 'secretlearn.secret_sharing.clustering.kmeans',
    'SSMiniBatchKMeans': 'secretlearn.secret_sharing.clustering.mini_batch_kmeans',
    'SSAgglomerativeClustering': 'secretlearn.secret_sharing.clustering.agglomerative_clustering',
    'SSDecisionTreeClassifier': 'secretlearn.secret_sharing.tree.decision_tree_classifier',
    'SSDecisionTreeRegressor': 'secretlearn.secret_sharing.tree.decision_tree_regressor',
    'SSMultinomialNB': 'secretlearn.secret_sharing.naive_bayes.multinomial_nb',
    'SSGaussianNB': 'secretlearn.secret_sharing.naive_bayes.gaussian_nb',
    'SSKNeighborsClassifier': 'secretlearn.secret_sharing.neighbors.k_neighbors_classifier',
    'SSKNeighborsRegressor': 'secretlearn.secret_sharing.neighbors.k_neighbors_regressor',
    'SSNearestNeighbors': 'secretlearn.secret_sharing.neighbors.nearest_neighbors',
    'SSLocalOutlierFactor': 'secretlearn.secret_sharing.neighbors.local_outlier_factor',
    'SSSVC': 'secretlearn.secret_sharing.svm.svc',
    'SSSVR': 'secretlearn.secret_sharing.svm.svr',
    'SSOneClassSVM': 'secretlearn.secret_sharing.svm.one_class_svm',
    'SSLinearSVC': 'secretlearn.secret_sharing.svm.linear_svc',
    'SSLinearSVR': 'secretlearn.secret_sharing.svm.linear_svr',
    'SSNuSVC': 'secretlearn.secret_sharing.svm.nusvc',
    'SSIsolationForest': 'secretlearn.secret_sharing.anomaly_detection.isolation_forest',
    'SSStandardScaler': 'secretlearn.secret_sharing.preprocessing.standard_scaler',
    'SSTSNE': 'secretlearn.secret_sharing.manifold.tsne',
    'SSDummyClassifier': 'secretlearn.secret_sharing.dummy.dummy_classifier',
    'SSDummyRegressor': 'secretlearn.secret_sharing.dummy.dummy_regressor',
    'SSGaussianRandomProjection': 'secretlearn.secret_sharing.random_projection.gaussian_random_projection',
    'SSSparseRandomProjection': 'secretlearn.secret_sharing.random_projection.sparse_random_projection',
    'SSNystroem': 'secretlearn.secret_sharing.kernel_approximation.nystroem',
    'SSIsotonicRegression': 'secretlearn.secret_sharing.isotonic.isotonic_regression',
}


def convert_file(filepath: Path) -> bool:
    """Convert a single benchmark file to SS mode."""
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
    
    # Collect SS classes used
    fl_classes_used = set()
    
    # Replace class names and track usage
    for sklearn_class, fl_class in ALGORITHM_MAP.items():
        if re.search(rf'\b{sklearn_class}\b', content):
            fl_classes_used.add(fl_class)
        content = re.sub(rf'(?<!SS)\b{sklearn_class}\b(?!\s*=)', fl_class, content)
    
    # Clean up orphaned import lines
    content = re.sub(r'^\s*[A-Z][a-zA-Z]+,\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*\)\s*$', '', content, flags=re.MULTILINE)
    
    # Remove multiple consecutive blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Generate SS imports
    fl_imports = []
    for fl_class in sorted(fl_classes_used):
        if fl_class in MODULE_MAP:
            fl_imports.append(f'from {MODULE_MAP[fl_class]} import {fl_class}')
    
    # Insert imports after the header
    import_block = '\n'.join(fl_imports) + '\n\n' if fl_imports else ''
    
    # Add SS header with imports
    content = SS_HEADER + import_block + content.lstrip()
    
    if content != original:
        filepath.write_text(content)
        return True
    return False


def main():
    benchmark_dir = Path(__file__).parent.parent / 'benchmarks' / 'secret-learn' / 'secret_sharing'
    
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
