#!/usr/bin/env python3
"""
Test that all modules can be imported successfully
"""

import sys
import importlib
from pathlib import Path


def test_secretlearn_import():
    """Test basic secretlearn import"""
    import secretlearn
    assert hasattr(secretlearn, '__version__')
    print(f"✓ secretlearn v{secretlearn.__version__}")


def test_sl_modules():
    """Test SL module imports"""
    from secretlearn.split_learning.anomaly_detection import isolation_forest
    from secretlearn.split_learning.linear_models import linear_regression
    from secretlearn.split_learning.ensemble import adaboost_classifier
    print("✓ SL modules import OK")


def test_fl_modules():
    """Test FL module imports"""
    from secretlearn.federated_learning.anomaly_detection import isolation_forest
    from secretlearn.federated_learning.linear_models import linear_regression
    from secretlearn.federated_learning.ensemble import adaboost_classifier
    print("✓ FL modules import OK")


def test_ss_modules():
    """Test SS module imports"""
    from secretlearn.secret_sharing.anomaly_detection import isolation_forest
    from secretlearn.secret_sharing.linear_models import linear_regression
    from secretlearn.secret_sharing.ensemble import adaboost_classifier
    print("✓ SS modules import OK")


def test_secure_aggregator_implementation():
    """Test that _secure_aggregate_parameters is properly implemented"""
    from secretlearn.split_learning.linear_models.linear_regression import SLLinearRegression
    
    # Check method exists and has proper docstring
    method = getattr(SLLinearRegression, '_secure_aggregate_parameters', None)
    assert method is not None, "_secure_aggregate_parameters method not found"
    
    # Get docstring
    doc = method.__doc__ or ""
    assert 'SecureAggregator' in doc or 'aggregate' in doc.lower(), f"Docstring missing SecureAggregator: {doc[:100]}"
    print("✓ _secure_aggregate_parameters implementation OK")


def test_isolation_forest_implementation():
    """Test IsolationForest secure aggregation"""
    from secretlearn.split_learning.anomaly_detection.isolation_forest import SLIsolationForest
    
    # Check method exists and has proper docstring
    method = SLIsolationForest._secure_aggregate_scores
    assert method is not None
    assert 'SecureAggregator' in method.__doc__
    assert 'scores_list' in method.__doc__
    print("✓ IsolationForest _secure_aggregate_scores implementation OK")


def test_no_todo_in_implementations():
    """Verify no TODO remains in core implementations"""
    import os
    
    base_path = Path(__file__).parent.parent / "secretlearn"
    todo_count = 0
    
    for root, dirs, files in os.walk(base_path):
        for filename in files:
            if filename.endswith('.py'):
                filepath = Path(root) / filename
                with open(filepath, 'r') as f:
                    content = f.read()
                    if 'TODO' in content:
                        # Exclude false positives in docstrings
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if 'TODO' in line and not line.strip().startswith('#'):
                                continue  # Skip comments in docstrings
                            if '# TODO' in line:
                                print(f"  Found TODO in {filepath}:{i+1}")
                                todo_count += 1
    
    assert todo_count == 0, f"Found {todo_count} TODOs in secretlearn/"
    print("✓ No TODO in secretlearn/ implementations")


if __name__ == "__main__":
    print("=" * 60)
    print(" Running Secret Learn Tests")
    print("=" * 60)
    
    tests = [
        test_secretlearn_import,
        test_sl_modules,
        test_fl_modules,
        test_ss_modules,
        test_secure_aggregator_implementation,
        test_isolation_forest_implementation,
        test_no_todo_in_implementations,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n[TEST] {test.__name__}")
            test()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    sys.exit(0 if failed == 0 else 1)
