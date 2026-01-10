#!/usr/bin/env python3
"""
Rename FL/SL/SS directories to full lowercase names.

FL -> federated_learning
SL -> split_learning  
SS -> secret_sharing
"""
import os
import re
import shutil
from pathlib import Path

# Mapping: old -> new
RENAME_MAP = {
    'FL': 'federated_learning',
    'SL': 'split_learning',
    'SS': 'secret_sharing',
}

# Class prefix mapping
CLASS_PREFIX_MAP = {
    'FL': 'FL',  # Keep class prefixes short for usability
    'SL': 'SL',
    'SS': 'SS',
}

def rename_directories(base_path: Path):
    """Rename FL/SL/SS directories to full names."""
    dirs_to_rename = [
        base_path / 'secretlearn',
        base_path / 'examples',
        base_path / 'benchmarks' / 'secret-learn',
    ]
    
    for parent_dir in dirs_to_rename:
        if not parent_dir.exists():
            print(f"  Skip: {parent_dir} (not found)")
            continue
            
        for old_name, new_name in RENAME_MAP.items():
            old_path = parent_dir / old_name
            new_path = parent_dir / new_name
            
            if old_path.exists():
                if new_path.exists():
                    print(f"  Skip: {old_path} -> {new_path} (target exists)")
                else:
                    print(f"  Rename: {old_path} -> {new_path}")
                    shutil.move(str(old_path), str(new_path))

def update_imports(base_path: Path):
    """Update all import statements to use new directory names."""
    patterns = [
        # from secretlearn.federated_learning.xxx import -> from secretlearn.federated_learning.xxx import
        (r'from secretlearn\.FL\.', 'from secretlearn.federated_learning.'),
        (r'from secretlearn\.SL\.', 'from secretlearn.split_learning.'),
        (r'from secretlearn\.SS\.', 'from secretlearn.secret_sharing.'),
        # import secretlearn.federated_learning -> import secretlearn.federated_learning
        (r'import secretlearn\.FL', 'import secretlearn.federated_learning'),
        (r'import secretlearn\.SL', 'import secretlearn.split_learning'),
        (r'import secretlearn\.SS', 'import secretlearn.secret_sharing'),
        # Path references in strings
        (r"secretlearn/federated_learning/", "secretlearn/federated_learning/"),
        (r"secretlearn/split_learning/", "secretlearn/split_learning/"),
        (r"secretlearn/secret_sharing/", "secretlearn/secret_sharing/"),
        (r"examples/federated_learning/", "examples/federated_learning/"),
        (r"examples/split_learning/", "examples/split_learning/"),
        (r"examples/secret_sharing/", "examples/secret_sharing/"),
    ]
    
    # Find all Python files
    py_files = list(base_path.rglob('*.py'))
    py_files += list(base_path.rglob('*.md'))
    py_files += list(base_path.rglob('*.sh'))
    py_files += list(base_path.rglob('*.txt'))
    
    # Exclude certain directories
    exclude_dirs = {'.git', '__pycache__', 'dist', '.egg-info', 'venv', 'env'}
    py_files = [f for f in py_files if not any(d in f.parts for d in exclude_dirs)]
    
    updated_count = 0
    for filepath in py_files:
        try:
            content = filepath.read_text(encoding='utf-8')
            original = content
            
            for pattern, replacement in patterns:
                content = re.sub(pattern, replacement, content)
            
            if content != original:
                filepath.write_text(content, encoding='utf-8')
                updated_count += 1
                print(f"  Updated: {filepath}")
        except Exception as e:
            print(f"  Error: {filepath}: {e}")
    
    return updated_count

def update_init_files(base_path: Path):
    """Update __init__.py files with new module names."""
    init_file = base_path / 'secretlearn' / '__init__.py'
    
    if init_file.exists():
        content = init_file.read_text()
        
        # Update any references
        content = content.replace("from .FL", "from .federated_learning")
        content = content.replace("from .SL", "from .split_learning")
        content = content.replace("from .SS", "from .secret_sharing")
        
        init_file.write_text(content)
        print(f"  Updated: {init_file}")

def main():
    base_path = Path(__file__).parent.parent
    
    print("=" * 70)
    print(" Directory Renaming: FL/SL/SS -> Full Names")
    print("=" * 70)
    
    print("\n[1/4] Renaming directories...")
    rename_directories(base_path)
    
    print("\n[2/4] Updating import statements...")
    count = update_imports(base_path)
    print(f"  Total files updated: {count}")
    
    print("\n[3/4] Updating __init__.py files...")
    update_init_files(base_path)
    
    print("\n[4/4] Verification...")
    # Check new directories exist
    for subdir in ['secretlearn', 'examples']:
        for new_name in RENAME_MAP.values():
            path = base_path / subdir / new_name
            if path.exists():
                file_count = len(list(path.rglob('*.py')))
                print(f"  ✓ {subdir}/{new_name}: {file_count} files")
            else:
                print(f"  ✗ {subdir}/{new_name}: NOT FOUND")
    
    print("\n" + "=" * 70)
    print(" Renaming complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run tests: python -m pytest tests/")
    print("  2. Verify imports: python -c 'from secretlearn.federated_learning import *'")

if __name__ == "__main__":
    main()
