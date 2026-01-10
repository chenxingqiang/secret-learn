#!/usr/bin/env python3
"""
Validate all example files

Tests:
1. Syntax validation (can parse as Python)
2. Import validation (import statement is correct)
3. Main function exists
"""

import sys
import ast
import os
import re
from pathlib import Path
from typing import List, Tuple


def check_syntax(filepath: Path) -> Tuple[bool, str]:
    """Check Python syntax"""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def check_import_statement(filepath: Path) -> Tuple[bool, str]:
    """Check that import statement is syntactically correct"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check for broken import (space in class name)
    if re.search(r'from secretlearn\.\w+\.\w+\.\w+ import \w+ \w+', content):
        return False, "Broken import: space in class name"
    
    # Check for correct secretlearn import
    if 'from secretlearn.' not in content:
        return False, "Missing secretlearn import"
    
    return True, ""


def check_main_function(filepath: Path) -> Tuple[bool, str]:
    """Check that main() function exists"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    if 'def main():' not in content and 'def main(' not in content:
        return False, "Missing main() function"
    
    if 'if __name__ == "__main__":' not in content:
        return False, "Missing __main__ block"
    
    return True, ""


def check_class_name_consistency(filepath: Path) -> Tuple[bool, str]:
    """Check that imported class matches usage"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract imported class name
    import_match = re.search(r'from secretlearn\.\w+\.\w+\.\w+ import (\w+)', content)
    if not import_match:
        return True, ""  # Skip if no import found
    
    class_name = import_match.group(1)
    
    # Check if class is instantiated
    # Pattern: ClassName( or = ClassName(
    if f'{class_name}(' not in content and f'= {class_name}' not in content:
        return False, f"Imported {class_name} but not used"
    
    return True, ""


def validate_example(filepath: Path) -> List[str]:
    """Run all validations on an example file"""
    errors = []
    
    # Skip __init__.py
    if filepath.name == '__init__.py':
        return []
    
    # 1. Syntax check
    ok, msg = check_syntax(filepath)
    if not ok:
        errors.append(f"Syntax: {msg}")
    
    # 2. Import check
    ok, msg = check_import_statement(filepath)
    if not ok:
        errors.append(f"Import: {msg}")
    
    # 3. Main function check
    ok, msg = check_main_function(filepath)
    if not ok:
        errors.append(f"Structure: {msg}")
    
    # 4. Class name consistency
    ok, msg = check_class_name_consistency(filepath)
    if not ok:
        errors.append(f"Usage: {msg}")
    
    return errors


def test_examples_directory(dir_name: str) -> Tuple[int, int, List[str]]:
    """Test all examples in a directory"""
    base_path = Path(__file__).parent.parent / "examples" / dir_name
    
    if not base_path.exists():
        return 0, 0, [f"Directory {dir_name} not found"]
    
    total = 0
    failed = 0
    all_errors = []
    
    for filepath in sorted(base_path.glob("*.py")):
        if filepath.name == '__init__.py':
            continue
        
        total += 1
        errors = validate_example(filepath)
        
        if errors:
            failed += 1
            for err in errors:
                all_errors.append(f"{dir_name}/{filepath.name}: {err}")
    
    return total, failed, all_errors


def main():
    print("=" * 70)
    print(" Example Files Validation")
    print("=" * 70)
    
    total_files = 0
    total_failed = 0
    all_errors = []
    
    for dir_name in ['SL', 'FL', 'SS']:
        print(f"\n--- Testing {dir_name} examples ---")
        count, failed, errors = test_examples_directory(dir_name)
        total_files += count
        total_failed += failed
        all_errors.extend(errors)
        
        if failed == 0:
            print(f"✓ {count} files validated successfully")
        else:
            print(f"✗ {failed}/{count} files have errors")
    
    if all_errors:
        print("\n" + "=" * 70)
        print(" Errors Found:")
        print("=" * 70)
        for err in all_errors[:50]:  # Limit output
            print(f"  • {err}")
        if len(all_errors) > 50:
            print(f"  ... and {len(all_errors) - 50} more errors")
    
    print("\n" + "=" * 70)
    print(f" Results: {total_files - total_failed}/{total_files} passed")
    print("=" * 70)
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
