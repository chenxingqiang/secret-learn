#!/usr/bin/env python3
"""Test all SS examples in Docker on remote server."""
import os
import sys
import time
import subprocess
import glob

def main():
    # Get all SS example files
    ss_dir = "/root/sl_test/SS"
    examples = []
    
    # Find all .py files except __init__.py and test files
    for root, dirs, files in os.walk(ss_dir):
        for f in files:
            if f.endswith('.py') and not f.startswith('__') and not f.startswith('test_'):
                examples.append(os.path.join(root, f))
    
    examples.sort()
    total = len(examples)
    print("=" * 70)
    print(f" SS Examples Test - Total: {total} files")
    print("=" * 70)
    
    passed = 0
    failed = 0
    failed_list = []
    
    for i, example in enumerate(examples, 1):
        name = os.path.relpath(example, ss_dir)
        print(f"\n[{i}/{total}] Testing {name}...", flush=True)
        
        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, example],
                capture_output=True,
                text=True,
                timeout=120,  # 2 min timeout per example
                cwd=os.path.dirname(example)
            )
            elapsed = time.time() - start
            
            if result.returncode == 0:
                print(f"  ✓ PASS ({elapsed:.1f}s)")
                passed += 1
            else:
                print(f"  ✗ FAIL ({elapsed:.1f}s)")
                # Show first line of error
                err = result.stderr.strip().split('\n')[-1][:80] if result.stderr else "Unknown"
                print(f"    Error: {err}")
                failed += 1
                failed_list.append((name, err))
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ TIMEOUT (>120s)")
            failed += 1
            failed_list.append((name, "Timeout"))
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)[:60]}")
            failed += 1
            failed_list.append((name, str(e)[:60]))
    
    # Summary
    print("\n" + "=" * 70)
    print(f" RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 70)
    
    if failed_list:
        print("\nFailed examples:")
        for name, err in failed_list[:20]:  # Show first 20
            print(f"  - {name}: {err}")
        if len(failed_list) > 20:
            print(f"  ... and {len(failed_list) - 20} more")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
