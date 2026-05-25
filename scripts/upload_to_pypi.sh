#!/usr/bin/env bash
# PyPI Upload Script for secret-learn
# This script automates the upload process to PyPI

set -e  # Exit on error

echo "========================================="
echo "PyPI Upload Script for secret-learn v0.3.3"
echo "========================================="
echo ""

# Check if dist directory exists and has files
if [ ! -d "dist" ] || [ -z "$(ls -A dist)" ]; then
    echo "Error: dist/ directory is empty or doesn't exist"
    echo "Run 'python3 -m build' first to build the package"
    exit 1
fi

# List files to be uploaded
echo "Files to be uploaded:"
ls -lh dist/
echo ""

# Ask user which PyPI to upload to
echo "Select upload destination:"
echo "1) Test PyPI (recommended for testing)"
echo "2) Production PyPI"
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo ""
        echo "Uploading to Test PyPI..."
        python3 -m twine upload --repository testpypi dist/*
        echo ""
        echo "✓ Upload complete!"
        echo ""
        echo "Test installation with:"
        echo "pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ secret-learn==0.3.3"
        ;;
    2)
        echo ""
        read -p "Are you sure you want to upload to PRODUCTION PyPI? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            echo "Uploading to Production PyPI..."
            python3 -m twine upload dist/*
            echo ""
            echo "✓ Upload complete!"
            echo ""
            echo "Test installation with:"
            echo "pip install secret-learn==0.3.3"
            echo ""
            echo "Don't forget to:"
            echo "1. Create a git tag: git tag -a v0.3.3 -m 'Release v0.3.3'"
            echo "2. Push the tag: git push origin v0.3.3"
        else
            echo "Upload cancelled."
            exit 0
        fi
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Done!"
