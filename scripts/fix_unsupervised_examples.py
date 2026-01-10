#!/usr/bin/env python3
"""
Fix unsupervised algorithm examples to not pass y parameter to fit().

PCA, ICA, NMF, KMeans, etc. are unsupervised and only need X.
"""

from pathlib import Path

# Unsupervised algorithms that don't need y
UNSUPERVISED_ALGORITHMS = [
    'pca', 'kernel_pca', 'incremental_pca', 'sparse_pca', 'mini_batch_sparse_pca',
    'truncated_svd', 'fast_ica', 'factor_analysis', 'nmf', 'mini_batch_nmf',
    'dictionary_learning', 'mini_batch_dictionarylearning', 'sparse_coder',
    'latent_dirichlet_allocation',
    'kmeans', 'mini_batch_kmeans', 'bisecting_kmeans', 'dbscan', 'hdbscan',
    'optics', 'affinity_propagation', 'mean_shift', 'spectral_clustering',
    'agglomerative_clustering', 'birch', 'feature_agglomeration',
    'isolation_forest', 'local_outlier_factor', 'one_class_svm', 'sgd_one_class_svm',
    'elliptic_envelope',
    'tsne', 'mds', 'isomap', 'locally_linear_embedding', 'spectral_embedding',
    'gaussian_mixture', 'bayesian_gaussian_mixture',
    'binarizer', 'normalizer', 'standard_scaler', 'min_max_scaler', 'max_abs_scaler',
    'robust_scaler', 'quantile_transformer', 'power_transformer',
    'label_encoder', 'ordinal_encoder', 'one_hot_encoder',
    'k_bins_discretizer', 'splinetransformer',
    'kernel_centerer', 'polynomial_features', 'function_transformer',
    'additive_chi2_sampler', 'nystroem', 'rbf_sampler', 'skewed_chi2_sampler',
    'polynomial_count_sketch',
    'gaussian_random_projection', 'sparse_random_projection', 'base_random_projection',
    'feature_hasher', 'dict_vectorizer',
    'empirical_covariance', 'shrunkcovariance', 'ledoit_wolf', 'oas',
    'min_cov_det', 'graphical_lasso', 'graphical_lasso_cv',
    'kernel_density', 'nearest_neighbors', 'k_neighbors_transformer',
    'radius_neighbors_transformer', 'neighborhood_components_analysis',
]


def fix_file(filepath: Path) -> bool:
    """Fix fit() call to not pass y for unsupervised algorithms."""
    try:
        content = filepath.read_text()
    except:
        return False
    
    # Replace model.fit(fed_X, fed_y) with model.fit(fed_X)
    old_call = "model.fit(fed_X, fed_y)"
    new_call = "model.fit(fed_X)"
    
    if old_call in content:
        new_content = content.replace(old_call, new_call)
        filepath.write_text(new_content)
        return True
    return False


def main():
    base = Path(__file__).parent.parent
    
    fixed = 0
    for mode in ['SL', 'FL']:
        examples_dir = base / "examples" / mode
        for alg in UNSUPERVISED_ALGORITHMS:
            filepath = examples_dir / f"{alg}.py"
            if filepath.exists() and fix_file(filepath):
                print(f"  Fixed {mode}/{alg}.py")
                fixed += 1
    
    print(f"\nTotal: {fixed} files fixed")


if __name__ == "__main__":
    main()
