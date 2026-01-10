#!/usr/bin/env python3
"""
Test ALL SL examples locally (single-machine simulation)
Results saved to logs/sl_test_results.txt
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import time

BASE_PATH = Path(__file__).parent.parent
EXAMPLES_PATH = BASE_PATH / "examples" / "split_learning"
LOG_DIR = BASE_PATH / "logs"
TIMEOUT = 180  # 3 minutes per example


def run_example(filepath: Path) -> tuple:
    """Run a single example, return (success, duration, error)"""
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
            error = result.stderr.strip().split('\n')[-1] if result.stderr else "Unknown"
            return False, duration, error[:200]
            
    except subprocess.TimeoutExpired:
        return False, TIMEOUT, "TIMEOUT"
    except Exception as e:
        return False, time.time() - start, str(e)[:200]


def main():
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / "sl_test_results.txt"
    
    # Get all SL examples (exclude distributed folder)
    examples = sorted([
        f for f in EXAMPLES_PATH.glob("*.py") 
        if f.name != '__init__.py'
    ])
    total = len(examples)
    
    print(f"Testing {total} SL examples...")
    print(f"Results will be saved to: {log_file}")
    
    with open(log_file, 'w') as f:
        f.write(f"SL Examples Test Results (Local Mode)\n")
        f.write(f"Started: {datetime.now()}\n")
        f.write(f"Total examples: {total}\n")
        f.write("=" * 70 + "\n\n")
        
        passed = 0
        failed = []
        
        for i, filepath in enumerate(examples, 1):
            name = filepath.name
            print(f"[{i}/{total}] Testing {name}...", end=" ", flush=True)
            
            success, duration, error = run_example(filepath)
            
            if success:
                status = "✓ PASS"
                passed += 1
                print(f"{status} ({duration:.1f}s)")
                f.write(f"✓ {name}: PASS ({duration:.1f}s)\n")
            else:
                status = "✗ FAIL"
                print(f"{status} - {error}")
                f.write(f"✗ {name}: FAIL - {error}\n")
                failed.append((name, error))
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Results: {passed}/{total} passed\n")
        f.write(f"Finished: {datetime.now()}\n")
        
        if failed:
            f.write(f"\nFailed examples ({len(failed)}):\n")
            for name, err in failed:
                f.write(f"  - {name}: {err}\n")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} passed")
    print(f"Log saved to: {log_file}")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
