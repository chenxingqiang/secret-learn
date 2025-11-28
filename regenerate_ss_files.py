#!/usr/bin/env python3
"""
Regenerate SS mode files with correct template
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, '/Users/xingqiangchen/jax-sklearn')
from secretlearn.algorithm_classifier import classify_algorithm

def snake_to_camel(snake_str):
    """Convert snake_case to CamelCase"""
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)

def get_sklearn_module_from_category(category):
    """Get sklearn module name from category directory name"""
    mapping = {
        'linear_models': 'linear_model',
        'clustering': 'cluster',
        'naive_bayes': 'naive_bayes',
        'neural_network': 'neural_network',
    }
    return mapping.get(category, category)

def generate_ss_unsupervised_template(algo_name, module_name):
    """Generate SS template for unsupervised algorithms"""
    return f'''# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Secret Sharing adapter for {algo_name}

{algo_name} is an UNSUPERVISED algorithm.
Data aggregated to SPU with full MPC protection.

Mode: Secret Sharing (SS)
"""

import logging
from typing import Union

try:
    from xlearn.{module_name} import {algo_name}
    USING_XLEARN = True
except ImportError:
    from sklearn.{module_name} import {algo_name}
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import SPU
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class SS{algo_name}:
    """Secret Sharing {algo_name} (Unsupervised)"""
    
    def __init__(self, spu: SPU, **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")
        
        self.spu = spu
        self.kwargs = kwargs
        self.model = None
        self._is_fitted = False
        
        if USING_XLEARN:
            logging.info(f"[SS] SS{algo_name} with JAX acceleration")
    
    def fit(self, x: Union[FedNdarray, VDataFrame]):
        """Fit (unsupervised - no y needed)"""
        if isinstance(x, VDataFrame):
            x = x.values
        
        logging.info(f"[SS] SS{algo_name} training in SPU")
        
        def _spu_fit(X, **kwargs):
            model = {algo_name}(**kwargs)
            model.fit(X)
            return model
        
        X_spu = x.to(self.spu)
        self.model = self.spu(_spu_fit)(X_spu, **self.kwargs)
        self._is_fitted = True
        return self
    
    def predict(self, x: Union[FedNdarray, VDataFrame]):
        """Predict cluster labels or anomalies"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        if isinstance(x, VDataFrame):
            x = x.values
        
        X_spu = x.to(self.spu)
        return self.spu(lambda m, X: m.predict(X))(self.model, X_spu)
    
    def transform(self, x: Union[FedNdarray, VDataFrame]):
        """Transform data (if supported)"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        if isinstance(x, VDataFrame):
            x = x.values
        
        X_spu = x.to(self.spu)
        
        def _transform(m, X):
            if hasattr(m, 'transform'):
                return m.transform(X)
            raise AttributeError("Model does not support transform")
        
        return self.spu(_transform)(self.model, X_spu)
'''

def generate_ss_supervised_template(algo_name, module_name):
    """Generate SS template for supervised algorithms"""
    return f'''# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Secret Sharing adapter for {algo_name}

{algo_name} is a SUPERVISED algorithm.
Data aggregated to SPU with full MPC protection.

Mode: Secret Sharing (SS)
"""

import logging
from typing import Union

try:
    from xlearn.{module_name} import {algo_name}
    USING_XLEARN = True
except ImportError:
    from sklearn.{module_name} import {algo_name}
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import SPU
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class SS{algo_name}:
    """Secret Sharing {algo_name} (Supervised)"""
    
    def __init__(self, spu: SPU, **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")
        
        self.spu = spu
        self.kwargs = kwargs
        self.model = None
        self._is_fitted = False
        
        if USING_XLEARN:
            logging.info(f"[SS] SS{algo_name} with JAX acceleration")
    
    def fit(self, x: Union[FedNdarray, VDataFrame], y: Union[FedNdarray, VDataFrame]):
        """Fit (supervised - labels required)"""
        if isinstance(x, VDataFrame):
            x = x.values
        if isinstance(y, VDataFrame):
            y = y.values
        
        logging.info(f"[SS] SS{algo_name} training in SPU")
        
        def _spu_fit(X, y, **kwargs):
            model = {algo_name}(**kwargs)
            model.fit(X, y)
            return model
        
        X_spu = x.to(self.spu)
        y_spu = y.to(self.spu)
        self.model = self.spu(_spu_fit)(X_spu, y_spu, **self.kwargs)
        self._is_fitted = True
        return self
    
    def predict(self, x: Union[FedNdarray, VDataFrame]):
        """Predict using model in SPU"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        if isinstance(x, VDataFrame):
            x = x.values
        
        X_spu = x.to(self.spu)
        return self.spu(lambda m, X: m.predict(X))(self.model, X_spu)
'''

def generate_ss_iterative_template(algo_name, module_name):
    """Generate SS template for iterative algorithms"""
    return f'''# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Secret Sharing adapter for {algo_name}

{algo_name} is an ITERATIVE SUPERVISED algorithm.
Data aggregated to SPU with full MPC protection.

Mode: Secret Sharing (SS)
"""

import logging
from typing import Union

try:
    from xlearn.{module_name} import {algo_name}
    USING_XLEARN = True
except ImportError:
    from sklearn.{module_name} import {algo_name}
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import SPU
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class SS{algo_name}:
    """Secret Sharing {algo_name} (Supervised, Iterative)"""
    
    def __init__(self, spu: SPU, **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")
        
        self.spu = spu
        self.kwargs = kwargs
        self.model = None
        self._is_fitted = False
        
        if USING_XLEARN:
            logging.info(f"[SS] SS{algo_name} with JAX acceleration")
    
    def fit(self, x: Union[FedNdarray, VDataFrame], y: Union[FedNdarray, VDataFrame], epochs: int = 10):
        """Fit (supervised, iterative)"""
        if isinstance(x, VDataFrame):
            x = x.values
        if isinstance(y, VDataFrame):
            y = y.values
        
        logging.info(f"[SS] SS{algo_name} training in SPU ({{epochs}} epochs)")
        
        def _spu_fit_iterative(X, y, epochs, **kwargs):
            import numpy as np
            model = {algo_name}(**kwargs)
            for epoch in range(epochs):
                if hasattr(model, 'partial_fit'):
                    if not hasattr(model, 'classes_'):
                        classes = np.unique(y)
                        model.partial_fit(X, y, classes=classes)
                    else:
                        model.partial_fit(X, y)
                else:
                    model.fit(X, y)
            return model
        
        X_spu = x.to(self.spu)
        y_spu = y.to(self.spu)
        self.model = self.spu(_spu_fit_iterative)(X_spu, y_spu, epochs, **self.kwargs)
        self._is_fitted = True
        return self
    
    def predict(self, x: Union[FedNdarray, VDataFrame]):
        """Predict using model in SPU"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        if isinstance(x, VDataFrame):
            x = x.values
        
        X_spu = x.to(self.spu)
        return self.spu(lambda m, X: m.predict(X))(self.model, X_spu)
'''

def regenerate_ss_file(filepath):
    """Regenerate a single SS file with correct template"""
    # Extract algorithm name from filename
    filename = filepath.stem
    algo_name = snake_to_camel(filename)
    
    # Get category from parent directory
    category = filepath.parent.name
    sklearn_module = get_sklearn_module_from_category(category)
    
    # Classify the algorithm
    try:
        char = classify_algorithm(algo_name)
        is_unsupervised = char.get('is_unsupervised', False)
        use_epochs = char.get('use_epochs', False)
    except:
        # Default to supervised non-iterative
        is_unsupervised = False
        use_epochs = False
    
    # Generate appropriate template
    if is_unsupervised:
        new_content = generate_ss_unsupervised_template(algo_name, sklearn_module)
    elif use_epochs:
        new_content = generate_ss_iterative_template(algo_name, sklearn_module)
    else:
        new_content = generate_ss_supervised_template(algo_name, sklearn_module)
    
    # Write the file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True

def find_problematic_ss_files():
    """Find SS files that need regeneration"""
    base_path = Path('/Users/xingqiangchen/jax-sklearn/secretlearn/SS')
    problematic = []
    
    for py_file in base_path.rglob('*.py'):
        if py_file.name == '__init__.py':
            continue
        
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for FL-mode code patterns
        if 'for party_name, device in devices.items():' in content:
            problematic.append(py_file)
        elif 'self.local_models' in content:
            problematic.append(py_file)
    
    return problematic

def main():
    print("="*90)
    print("Regenerating SS Mode Files with Correct Template")
    print("="*90)
    print()
    
    # Find problematic files
    problematic = find_problematic_ss_files()
    
    print(f"Found {len(problematic)} files to regenerate")
    print()
    
    if not problematic:
        print("✅ All SS files are correct!")
        return
    
    # Show some examples
    print("Files to regenerate:")
    for i, f in enumerate(problematic[:10], 1):
        print(f"  {i}. {f.parent.name}/{f.name}")
    if len(problematic) > 10:
        print(f"  ... and {len(problematic) - 10} more")
    print()
    
    response = input(f"Regenerate {len(problematic)} files? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Operation cancelled")
        return
    
    # Regenerate each file
    print("\nRegenerating...")
    success_count = 0
    error_count = 0
    
    for py_file in problematic:
        try:
            regenerate_ss_file(py_file)
            success_count += 1
            if success_count <= 20:
                print(f"✅ {py_file.parent.name}/{py_file.name}")
        except Exception as e:
            error_count += 1
            print(f"❌ {py_file.parent.name}/{py_file.name}: {str(e)}")
    
    if success_count > 20:
        print(f"... and {success_count - 20} more files regenerated")
    
    print()
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    
    # Verify
    print("\nVerifying...")
    verify_count = 0
    for py_file in Path('/Users/xingqiangchen/jax-sklearn/secretlearn/SS').rglob('*.py'):
        if py_file.name == '__init__.py':
            continue
        try:
            with open(py_file, 'r') as f:
                compile(f.read(), py_file.name, 'exec')
            verify_count += 1
        except SyntaxError as e:
            print(f"❌ Still has error: {py_file.name}")
    
    print(f"✅ {verify_count}/191 SS files have valid syntax")
    print()

if __name__ == '__main__':
    main()

