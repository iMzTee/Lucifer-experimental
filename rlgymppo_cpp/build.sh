#!/bin/bash
# build.sh — Build RLGymPPO_CPP with Lucifer bot on Ubuntu
#
# Prerequisites:
#   - Ubuntu 22.04+ with CUDA 12.x installed
#   - cmake, build-essential, python3-dev
#   - RTX 2060 (or any CUDA GPU)
#
# Usage: bash build.sh
# Then:  cd ~/RLGymPPO_CPP/build && ./RLGymPPO_CPP

set -e

echo "=== Lucifer Bot — RLGymPPO_CPP Build Script ==="
echo ""

# ─── Step 1: Install build dependencies ───
echo "[1/7] Installing build dependencies..."
sudo apt install -y cmake build-essential python3-dev python3-pip unzip wget
pip3 install wandb 2>/dev/null || true

# ─── Step 2: Clone RLGymPPO_CPP ───
REPO_DIR="$HOME/RLGymPPO_CPP"
if [ ! -d "$REPO_DIR" ]; then
    echo "[2/7] Cloning RLGymPPO_CPP..."
    cd ~
    git clone https://github.com/ZealanL/RLGymPPO_CPP --recurse-submodules
else
    echo "[2/7] RLGymPPO_CPP already cloned, updating..."
    cd "$REPO_DIR"
    git pull
    git submodule update --init --recursive
fi

# ─── Step 3: Download LibTorch ───
LIBTORCH_DIR="$REPO_DIR/RLGymPPO_CPP/RLGymPPO_CPP/libtorch"
if [ ! -d "$LIBTORCH_DIR" ]; then
    echo "[3/7] Downloading LibTorch (CUDA 12.4, cxx11 ABI)..."
    cd "$REPO_DIR/RLGymPPO_CPP/RLGymPPO_CPP"

    # Check pytorch.org for the latest compatible version
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu124.zip"
    wget -q --show-progress "$LIBTORCH_URL" -O libtorch.zip
    unzip -q libtorch.zip
    rm libtorch.zip
    echo "    LibTorch extracted to $LIBTORCH_DIR"
else
    echo "[3/7] LibTorch already present."
fi

# ─── Step 4: Copy collision meshes ───
LUCIFER_DIR="$HOME/Lucifer"
MESHES_DEST="$REPO_DIR/collision_meshes"
if [ ! -d "$MESHES_DEST" ]; then
    echo "[4/7] Copying collision meshes..."
    if [ -d "$LUCIFER_DIR/collision_meshes" ]; then
        cp -r "$LUCIFER_DIR/collision_meshes" "$MESHES_DEST"
    else
        echo "    WARNING: collision_meshes not found at $LUCIFER_DIR/collision_meshes"
        echo "    You'll need to copy them manually before running."
    fi
else
    echo "[4/7] Collision meshes already present."
fi

# ─── Step 5: Copy Lucifer source files ───
echo "[5/7] Copying Lucifer source files..."
LUCIFER_CPP_SRC="$LUCIFER_DIR/rlgymppo_cpp"
# All files go to root (alongside examplemain.cpp)
CPP_DEST="$REPO_DIR"

if [ -d "$LUCIFER_CPP_SRC" ]; then
    cp "$LUCIFER_CPP_SRC/LuciferObs.h" "$CPP_DEST/"
    cp "$LUCIFER_CPP_SRC/LuciferRewards.h" "$CPP_DEST/"
    cp "$LUCIFER_CPP_SRC/LuciferStateSetter.h" "$CPP_DEST/"
    cp "$LUCIFER_CPP_SRC/LuciferCurriculum.h" "$CPP_DEST/"
    cp "$LUCIFER_CPP_SRC/LuciferMain.cpp" "$CPP_DEST/"
    echo "    Copied 5 Lucifer files to $CPP_DEST"
else
    echo "    ERROR: Lucifer C++ sources not found at $LUCIFER_CPP_SRC"
    echo "    Run this script from the Lucifer repo directory."
    exit 1
fi

# ─── Step 6: Patch CMakeLists.txt to use LuciferMain.cpp ───
echo "[6/7] Patching CMakeLists.txt..."
CMAKELISTS="$REPO_DIR/CMakeLists.txt"

# Replace examplemain.cpp with LuciferMain.cpp in the add_executable line
if grep -q "examplemain.cpp" "$CMAKELISTS"; then
    sed -i 's|examplemain\.cpp|LuciferMain.cpp|g' "$CMAKELISTS"
    echo "    Replaced examplemain.cpp with LuciferMain.cpp in CMakeLists.txt"
elif grep -q "LuciferMain.cpp" "$CMAKELISTS"; then
    echo "    CMakeLists.txt already patched."
else
    echo "    WARNING: Could not find examplemain.cpp in CMakeLists.txt"
    echo "    You may need to manually update the add_executable() call."
fi

# ─── Step 7: Build ───
echo "[7/7] Building..."
BUILD_DIR="$REPO_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build . -j$(nproc)

echo ""
echo "=== Build complete! ==="
echo ""
# Copy collision meshes to build dir so they're in the working directory
if [ -d "$MESHES_DEST" ] && [ ! -d "$BUILD_DIR/collision_meshes" ]; then
    ln -s "$MESHES_DEST" "$BUILD_DIR/collision_meshes" 2>/dev/null || \
    cp -r "$MESHES_DEST" "$BUILD_DIR/"
fi

echo "To run training:"
echo "  cd $BUILD_DIR"
echo "  ./RLGymPPO_CPP"
echo ""
echo "Checkpoints saved to: Checkpoints_2v2_cpp/"
echo "Logs: wandb project 'lucifer-2v2'"
