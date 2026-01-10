#!/bin/bash
# Setup script for remote server (Bob)
# Run this on the remote server: 47.80.8.23

set -e

echo "========================================"
echo " Setting up Bob (Remote Server)"
echo "========================================"

# Check Python version
echo -e "\n[1/5] Checking Python..."
python3 --version
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Python 3.10+ required"
    exit 1
fi
echo "  ✓ Python version OK"

# Create virtual environment
echo -e "\n[2/5] Creating virtual environment..."
if [ ! -d "venv_sl" ]; then
    python3 -m venv venv_sl
    echo "  ✓ Virtual environment created"
else
    echo "  ✓ Virtual environment exists"
fi
source venv_sl/bin/activate

# Install dependencies
echo -e "\n[3/5] Installing dependencies..."
pip install --upgrade pip
pip install secret-learn secretflow numpy

# Open firewall port
echo -e "\n[4/5] Configuring firewall..."
if command -v ufw &> /dev/null; then
    sudo ufw allow 9498/tcp
    echo "  ✓ Port 9498 opened (ufw)"
elif command -v firewall-cmd &> /dev/null; then
    sudo firewall-cmd --add-port=9498/tcp --permanent
    sudo firewall-cmd --reload
    echo "  ✓ Port 9498 opened (firewalld)"
else
    echo "  ⚠ Please manually open port 9498"
fi

# Test installation
echo -e "\n[5/5] Testing installation..."
python3 -c "import secretlearn; print(f'secret-learn version: {secretlearn.__version__}')"
python3 -c "import secretflow; print('secretflow: OK')"

echo -e "\n========================================"
echo " Setup Complete!"
echo "========================================"
echo ""
echo "To start Bob server:"
echo "  source venv_sl/bin/activate"
echo "  python bob_remote.py --alice_addr <ALICE_IP>:9497"
