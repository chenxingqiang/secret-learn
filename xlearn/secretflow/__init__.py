# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
SecretFlow Integration Layer for JAX-sklearn

Provides seamless integration between secret-learn and SecretFlow,
enabling privacy-preserving machine learning with JAX acceleration.

Usage:
    # Direct algorithm import (recommended)
    from xlearn.secretflow.FL.clustering import FLKMeans
    from xlearn.secretflow.FL.linear_models import FLLinearRegression
    
    # Or import from top level
    from xlearn.secretflow import FL, SL, SS
    model = FL.clustering.FLKMeans(devices={...})
    
Integration Modes:
- FL (Federated Learning): Data stays in local PYUs, HEU secure aggregation
- SL (Split Learning): Model split across parties, collaborative training
- SS (Simple Sealed): Data aggregated to SPU, full MPC protection
"""

import sys
from pathlib import Path
from types import ModuleType


def _create_mode_module(mode_name: str):
    """Create a module that exposes algorithms by category"""
    
    class ModeModule:
        """Dynamic module for algorithm access"""
        
        def __init__(self, mode):
            self._mode = mode
            self._mode_upper = mode.upper()
        
        def __getattr__(self, category_name):
            """Access category (e.g., FL.clustering)"""
            if category_name.startswith('_'):
                raise AttributeError(f"No attribute '{category_name}'")
            
            # Create category module
            return _create_category_module(self._mode_upper, category_name)
        
        def __dir__(self):
            """List available categories"""
            return [
                'anomaly_detection', 'calibration', 'clustering', 'covariance',
                'cross_decomposition', 'decomposition', 'discriminant_analysis',
                'dummy', 'ensemble', 'feature_selection', 'gaussian_process',
                'isotonic', 'kernel_ridge', 'linear_models', 'manifold',
                'multiclass', 'multioutput', 'naive_bayes', 'neighbors',
                'neural_network', 'preprocessing', 'semi_supervised', 'svm', 'tree'
            ]
    
    return ModeModule(mode_name)


def _create_category_module(mode: str, category: str):
    """Create a module that exposes algorithms in a category"""
    
    class CategoryModule:
        """Dynamic module for algorithm access"""
        
        def __init__(self, mode_name, category_name):
            self._mode = mode_name
            self._category = category_name
        
        def __getattr__(self, algo_name):
            """Import algorithm class dynamically"""
            if algo_name.startswith('_'):
                raise AttributeError(f"No attribute '{algo_name}'")
            
            # Construct import path
            try:
                from importlib import import_module
                
                # Convert class name to module name (e.g., FLKMeans -> kmeans)
                if algo_name.startswith(self._mode):
                    module_algo = algo_name[len(self._mode):].lower()
                else:
                    module_algo = algo_name.lower()
                
                # Import from generated directory
                module_path = f"xlearn.secretflow.generated.{self._mode}.{self._category}.{module_algo}"
                module = import_module(module_path)
                
                return getattr(module, algo_name)
                
            except (ImportError, AttributeError) as e:
                raise AttributeError(
                    f"Algorithm '{algo_name}' not found in {self._mode}.{self._category}. "
                    f"Error: {e}"
                )
        
        def __dir__(self):
            """List available algorithms"""
            from pathlib import Path
            base_dir = Path(__file__).parent / "generated" / self._mode / self._category
            if base_dir.exists():
                return [
                    f.stem for f in base_dir.glob("*.py")
                    if f.stem != '__init__'
                ]
            return []
    
    return CategoryModule(mode, category)


# Create mode modules
FL = _create_mode_module('fl')
SL = _create_mode_module('sl')
SS = _create_mode_module('ss')


__all__ = [
    'FL',  # Federated Learning
    'SL',  # Split Learning
    'SS',  # Simple Sealed
]


# Module-level docstring for help()
__doc__ += """

Quick Examples:
--------------

FL Mode (Federated Learning):
    >>> from xlearn.secretflow.FL.clustering import FLKMeans
    >>> model = FLKMeans(
    >>>     devices={'alice': alice, 'bob': bob},
    >>>     heu=heu,
    >>>     n_clusters=3
    >>> )
    >>> model.fit(fed_X)  # Unsupervised
    
    >>> from xlearn.secretflow.FL.linear_models import FLLinearRegression
    >>> model = FLLinearRegression(
    >>>     devices={'alice': alice, 'bob': bob},
    >>>     heu=heu
    >>> )
    >>> model.fit(fed_X, fed_y)  # Supervised

SL Mode (Split Learning):
    >>> from xlearn.secretflow.SL.neural_network import SLMLPClassifier
    >>> model = SLMLPClassifier(
    >>>     devices={'alice': alice, 'bob': bob}
    >>> )
    >>> model.fit(fed_X, fed_y, epochs=10)

SS Mode (Simple Sealed):
    >>> from xlearn.secretflow.SS.decomposition import SSPCA
    >>> model = SSPCA(spu=spu, n_components=10)
    >>> model.fit(fed_X)

Alternative Access:
    >>> from xlearn.secretflow import FL, SL, SS
    >>> 
    >>> # Access by category
    >>> kmeans_class = FL.clustering.FLKMeans
    >>> pca_class = SS.decomposition.SSPCA
    
Available Algorithms:
    - 116 unique algorithms
    - 348 total implementations (116 × 3 modes)
    - 24 algorithm categories
    - 100% sklearn API compatible
"""
