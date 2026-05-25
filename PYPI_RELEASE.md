# PyPI Release Guide for secret-learn v0.3.3

## Changes in v0.3.3

- Restructured package: moved from `secretlearn/` to `src/secretlearn/`
- Updated `pyproject.toml` to reference the new `src/` directory structure
- Maintained backward compatibility - package name remains `secret-learn`
- Import statements remain unchanged: `from secretlearn import ...`

## Pre-Release Checklist

- [x] Directory structure changed from `./secretlearn/` to `./src/secretlearn/`
- [x] Updated `pyproject.toml` with correct package finding configuration
- [x] Version bumped from 0.3.2 to 0.3.3
- [x] README.md updated with new version badge
- [x] Package builds successfully (wheel and sdist created)
- [x] Wheel contains all required code files
- [x] Local import test passes

## Build Commands

The package has already been built. The distribution files are in `dist/`:
- `dist/secret_learn-0.3.3-py3-none-any.whl` (wheel)
- `dist/secret_learn-0.3.3.tar.gz` (source distribution)

To rebuild if needed:
```bash
# Clean previous builds
rm -rf dist/ build/ src/secret_learn.egg-info/

# Install/upgrade build tools
pip install --upgrade build twine

# Build the package
python3 -m build
```

## Verify Package Contents

```bash
# Check wheel contents
unzip -l dist/secret_learn-0.3.3-py3-none-any.whl | less

# Check source distribution
tar -tzf dist/secret_learn-0.3.3.tar.gz | less
```

## Test Installation Locally

```bash
# Test in a virtual environment
python3 -m venv test_env
source test_env/bin/activate
pip install dist/secret_learn-0.3.3-py3-none-any.whl

# Test import
python3 -c "import secretlearn; print(secretlearn.__version__)"

# Cleanup
deactivate
rm -rf test_env
```

## Upload to PyPI

### Option 1: Upload to Test PyPI first (Recommended)

```bash
# Upload to Test PyPI
python3 -m twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ secret-learn==0.3.3
```

### Option 2: Upload directly to PyPI

```bash
# Upload to PyPI
python3 -m twine upload dist/*
```

You will be prompted for your PyPI credentials:
- Username: `__token__`
- Password: Your PyPI API token (starts with `pypi-`)

## Post-Release

1. Verify the package on PyPI: https://pypi.org/project/secret-learn/
2. Test installation from PyPI:
   ```bash
   pip install secret-learn==0.3.3
   ```
3. Create a Git tag:
   ```bash
   git tag -a v0.3.3 -m "Release version 0.3.3 - Restructured to src layout"
   git push origin v0.3.3
   ```

## Important Notes

1. **Package imports remain unchanged**: Users still use `from secretlearn import ...`
2. **The src layout**: This is a best practice for Python packages to avoid accidental imports from the project directory during development
3. **Version number**: 0.3.3 (incremented from 0.3.2)
4. **Breaking changes**: None - this is a structural change only

## Troubleshooting

If upload fails:
- Ensure you have the correct PyPI credentials
- Check that version 0.3.3 doesn't already exist on PyPI
- Verify your API token has upload permissions

If installation fails:
- Check PyPI project page for any issues
- Try installing with `--no-cache-dir` flag
- Verify all dependencies are available
