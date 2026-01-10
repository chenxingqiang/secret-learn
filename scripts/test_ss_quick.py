#!/usr/bin/env python3
"""
Quick SS examples test with small dataset
"""

import subprocess
import sys
import time
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent
TIMEOUT = 180

# Test a subset of SS examples
EXAMPLES = [
    'linear_regression.py',
    'adaboost_classifier.py', 
    'pca.py',
    'kmeans.py',
    'isolation_forest.py',
    'ridge.py',
    'lasso.py',
    'logistic_regression.py',
]


def run_example(name: str) -> tuple:
    """Run a single example"""
    filepath = BASE_PATH / "examples" / "secret_sharing" / name
    if not filepath.exists():
        return False, 0, f"File not found"
    
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = str(BASE_PATH)
    
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(filepath)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env=env,
            cwd=str(BASE_PATH)
        )
        duration = time.time() - start
        
        if result.returncode == 0:
            return True, duration, None
        else:
            error = result.stderr.strip().split('\n')[-1][:100] if result.stderr else "Unknown"
            return False, duration, error
            
    except subprocess.TimeoutExpired:
        return False, TIMEOUT, "TIMEOUT"
    except Exception as e:
        return False, time.time() - start, str(e)[:100]


def main():
    print("=" * 70)
    print(" SS Quick Test (8 examples)")
    print("=" * 70)
    
    passed = 0
    
    for name in EXAMPLES:
        print(f"\nTesting SS/{name}...", end=" ", flush=True)
        success, duration, error = run_example(name)
        
        if success:
            print(f"✓ PASS ({duration:.1f}s)")
            passed += 1
        else:
            print(f"✗ FAIL - {error}")
    
    print("\n" + "=" * 70)
    print(f" Results: {passed}/{len(EXAMPLES)} passed")
    print("=" * 70)
    
    return 0 if passed == len(EXAMPLES) else 1


if __name__ == "__main__":
    sys.exit(main())
