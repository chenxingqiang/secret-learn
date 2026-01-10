#!/usr/bin/env python3
"""
Run a batch of example files to verify they work

Note: SS examples require SPU setup and may timeout in single-machine mode
"""

import sys
import subprocess
import os
from pathlib import Path


EXAMPLES_TO_TEST = {
    'SL': [
        'linear_regression.py',
        'isolation_forest.py',
        'adaboost_classifier.py',
        'kmeans.py',
        'pca.py',
    ],
    'FL': [
        'linear_regression.py',
        'isolation_forest.py',
        'adaboost_classifier.py',
        'kmeans.py',
        'pca.py',
    ],
}

TIMEOUT = 120  # seconds


def run_example(mode: str, filename: str) -> tuple:
    """Run a single example and return (success, message)"""
    base_path = Path(__file__).parent.parent
    example_path = base_path / "examples" / mode / filename
    
    if not example_path.exists():
        return False, f"File not found: {example_path}"
    
    env = os.environ.copy()
    env['PYTHONPATH'] = str(base_path)
    
    try:
        result = subprocess.run(
            [sys.executable, str(example_path)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env=env,
            cwd=str(base_path)
        )
        
        if result.returncode == 0:
            # Check for success indicators
            if 'completed' in result.stdout.lower() or 'successfully' in result.stdout.lower():
                return True, "OK"
            return True, "Completed"
        else:
            # Extract error
            error = result.stderr.strip().split('\n')[-1] if result.stderr else "Unknown error"
            return False, error[:100]
            
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)[:100]


def main():
    print("=" * 70)
    print(" Running Example Tests")
    print("=" * 70)
    
    total = 0
    passed = 0
    failed = []
    
    for mode, examples in EXAMPLES_TO_TEST.items():
        print(f"\n--- {mode} Examples ---")
        
        for example in examples:
            total += 1
            success, msg = run_example(mode, example)
            
            status = "✓" if success else "✗"
            print(f"  {status} {mode}/{example}: {msg}")
            
            if success:
                passed += 1
            else:
                failed.append(f"{mode}/{example}: {msg}")
    
    print("\n" + "=" * 70)
    print(f" Results: {passed}/{total} passed")
    
    if failed:
        print("\n Failed examples:")
        for f in failed:
            print(f"   • {f}")
    
    print("=" * 70)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
