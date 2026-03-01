#!/bin/bash
# Lucifer GPU Sim — Ubuntu Setup Script
# Run: chmod +x setup_ubuntu.sh && ./setup_ubuntu.sh

set -e

echo "=== Lucifer GPU Simulator — Ubuntu Setup ==="

# 1. Clone repo
echo ""
echo "[1/5] Cloning repository..."
cd ~
if [ -d "Lucifer" ]; then
    echo "  ~/Lucifer already exists, pulling latest..."
    cd Lucifer && git pull
else
    git clone https://github.com/iMzTee/Lucifer.git
    cd Lucifer
fi

# 2. Check NVIDIA driver + CUDA
echo ""
echo "[2/5] Checking GPU..."
nvidia-smi || { echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first:"; echo "  sudo apt install nvidia-driver-550"; exit 1; }

# 3. Install Python + pip
echo ""
echo "[3/5] Checking Python..."
python3 --version || { echo "Installing Python..."; sudo apt update && sudo apt install -y python3 python3-pip python3-venv; }

# 4. Create venv + install deps
echo ""
echo "[4/5] Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 5. Verify
echo ""
echo "[5/5] Verifying installation..."
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB')
else:
    print('ERROR: CUDA not available!')
    exit(1)

from gpu_sim.environment import GPUEnvironment
from gpu_sim.rewards import GPURewards
from gpu_sim.observations import build_obs_batch
env = GPUEnvironment(100, 'cuda', stage=0)
print('gpu_sim imports OK')
print()
print('=== Ready! Run with: ===')
print('  source venv/bin/activate')
print('  python -u lucifer2_gpu.py')
"

echo ""
echo "=== Setup complete! ==="
echo "To start training:"
echo "  cd ~/Lucifer"
echo "  source venv/bin/activate"
echo "  python -u lucifer2_gpu.py"
