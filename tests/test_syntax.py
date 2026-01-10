#!/usr/bin/env python3
"""
Test syntax validation for all Python files
"""

import sys
import ast
import os
from pathlib import Path


def check_syntax(filepath: Path) -> bool:
    """Check if a Python file has valid syntax"""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error in {filepath}:{e.lineno}: {e.msg}")
        return False


def test_all_syntax():
    """Test all Python files for syntax errors"""
    base_path = Path(__file__).parent.parent / "secretlearn"
    
    total = 0
    errors = 0
    
    for root, dirs, files in os.walk(base_path):
        for filename in files:
            if filename.endswith('.py'):
                filepath = Path(root) / filename
                total += 1
                if not check_syntax(filepath):
                    errors += 1
    
    print(f"\nChecked {total} files, {errors} syntax errors")
    assert errors == 0, f"Found {errors} syntax errors"
    print("✓ All files have valid syntax")


def test_examples_syntax():
    """Test all example files for syntax errors"""
    base_path = Path(__file__).parent.parent / "examples"
    
    total = 0
    errors = 0
    
    for root, dirs, files in os.walk(base_path):
        for filename in files:
            if filename.endswith('.py'):
                filepath = Path(root) / filename
                total += 1
                if not check_syntax(filepath):
                    errors += 1
    
    print(f"\nChecked {total} example files, {errors} syntax errors")
    assert errors == 0, f"Found {errors} syntax errors in examples"
    print("✓ All example files have valid syntax")


if __name__ == "__main__":
    print("=" * 60)
    print(" Syntax Validation Tests")
    print("=" * 60)
    
    tests = [
        test_all_syntax,
        test_examples_syntax,
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
