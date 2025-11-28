#!/usr/bin/env python3
"""
Fix duplicate 'spu' parameter in SS mode files
"""

import re
from pathlib import Path

def fix_ss_file(filepath):
    """Fix duplicate spu parameter in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix 1: Remove duplicate spu parameter in __init__
        # Pattern: spu : SPU,\n        spu: Optional[SPU] = None,
        content = re.sub(
            r'(\s+)(spu\s*:\s*SPU),\s*\n\s+spu:\s*Optional\[SPU\]\s*=\s*None,',
            r'\1spu: SPU,',
            content
        )
        
        # Fix 2: Remove duplicate self.spu assignment
        content = re.sub(
            r'(self\.spu\s*=\s*spu)\s*\n\s*self\.spu\s*=\s*spu',
            r'\1',
            content
        )
        
        # Fix 3: Fix class name (FL -> SS)
        content = re.sub(
            r'class FL(\w+):',
            r'class SS\1:',
            content
        )
        
        # Fix 4: Fix docstring class references
        content = re.sub(
            r'Federated Learning (\w+)',
            r'Secret Sharing \1',
            content,
            count=2  # Only replace first few occurrences in header
        )
        
        # Fix 5: Remove devices iteration (SS mode doesn't use devices)
        # Look for: for party_name, device in devices.items():
        if 'for party_name, device in devices.items():' in content:
            # This is a more complex fix - mark for manual review
            pass
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Fixed"
        
        return False, "No changes needed"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    print("="*90)
    print("Fixing SS Mode Duplicate Parameter Issues")
    print("="*90)
    print()
    
    base_path = Path('/Users/xingqiangchen/jax-sklearn/secretlearn/SS')
    
    fixed_count = 0
    error_count = 0
    
    for py_file in base_path.rglob('*.py'):
        if py_file.name == '__init__.py':
            continue
        
        success, message = fix_ss_file(py_file)
        
        if success:
            fixed_count += 1
            if fixed_count <= 20:
                print(f"✅ {py_file.relative_to(base_path)}")
        elif "Error" in message:
            error_count += 1
            print(f"❌ {py_file.relative_to(base_path)}: {message}")
    
    if fixed_count > 20:
        print(f"... and {fixed_count - 20} more files")
    
    print()
    print(f"Fixed: {fixed_count} files")
    print(f"Errors: {error_count} files")
    print()

if __name__ == '__main__':
    main()

