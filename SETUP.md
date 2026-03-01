# LuciferBot — Setup Guide

GPU-accelerated Rocket League bot training. Runs 40,000 parallel game environments entirely on GPU using PyTorch.

## Requirements

- **OS**: Ubuntu 22.04+ (or WSL2 with Ubuntu)
- **GPU**: NVIDIA GPU with 6+ GB VRAM (tested on RTX 2060)
- **Driver**: NVIDIA driver 550+ with CUDA 12.4 support
- **Python**: 3.10 - 3.12
- **RAM**: 4+ GB free

## Quick Setup

If you just want to get running, the setup script handles everything:

```bash
git clone https://github.com/iMzTee/Lucifer.git
cd Lucifer
chmod +x setup_ubuntu.sh
./setup_ubuntu.sh
source venv/bin/activate
python -u luciferbot.py
```

If that works, you're done. The rest of this guide explains each step in detail.

---

## Step-by-Step Setup

### 1. Install NVIDIA drivers

Check if drivers are already installed:

```bash
nvidia-smi
```

If not installed:

```bash
sudo apt update
sudo apt install -y nvidia-driver-550
sudo reboot
```

After reboot, verify:

```bash
nvidia-smi
```

You should see your GPU listed with driver version 550+.

### 2. Install Python

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git
```

Verify:

```bash
python3 --version
```

Should show Python 3.10, 3.11, or 3.12.

### 3. Clone the repository

```bash
cd ~
git clone https://github.com/iMzTee/Lucifer.git
cd Lucifer
```

### 4. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Your prompt should now show `(venv)` at the beginning.

### 5. Install PyTorch with CUDA

```bash
pip install --upgrade pip
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

This installs PyTorch 2.6.0 with CUDA 12.4 support. The download is ~2.5 GB.

### 6. Install remaining dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `numpy==1.26.4` — array operations
- `psutil==7.2.2` — memory monitoring
- `tensorboard==2.20.0` — training metrics logging

### 7. Verify the installation

```bash
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB')
from gpu_sim.environment import GPUEnvironment
env = GPUEnvironment(100, 'cuda', stage=0)
print('gpu_sim OK')
print('Ready!')
"
```

You should see your GPU name, VRAM, and "Ready!" with no errors.

### 8. Start training

```bash
python -u luciferbot.py
```

The `-u` flag disables output buffering so you see logs in real time.

On first run with no checkpoint, training starts from scratch. If checkpoints exist in `Checkpoints_LuciferBot/`, it loads the latest one automatically.

Press `Ctrl+C` to stop. It saves a checkpoint before exiting.

---

## Configuration

N_ENVS is automatically set per stage in `gpu_sim/constants.py` STAGE_CONFIG.
Only `TS_PER_ITERATION` is configurable in `luciferbot.py`:

```python
TS_PER_ITERATION = 200000   # Timesteps collected per training iteration
```

N_ENVS adjusts automatically per stage to keep VRAM usage stable (~2.5GB):

| Stage | Format | N_ENVS | Agents | Description |
|---|---|---|---|---|
| 0 | 1v0 | 160,000 | 1 | Solo mechanics |
| 1 | 1v1 | 80,000 | 2 | 1v1 mechanics |
| 2 | 1v1 | 80,000 | 2 | 1v1 game sense |
| 3 | 2v2 | 40,000 | 4 | 2v2 teamwork |

To adjust for your GPU, edit `STAGE_CONFIG` in `gpu_sim/constants.py`.

### PPO hyperparameters

These are in the `GPULearner.__init__` method:

| Parameter | Value | Description |
|---|---|---|
| `batch_size` | 50,000 | Samples per PPO update batch |
| `mini_batch_size` | 50,000 | Mini-batch size (same = no mini-batching) |
| `n_epochs` | 2 | PPO epochs per iteration |
| `clip_range` | 0.2 | PPO clipping range |
| `ent_coef` | 0.01 | Entropy bonus coefficient |
| `policy_lr` | 2e-4 | Policy learning rate (auto-decays) |
| `critic_lr` | 2e-4 | Critic learning rate |
| `policy_layer_sizes` | (2048, 2048, 1024, 1024) | Policy network layers |
| `critic_layer_sizes` | (2048, 2048, 1024, 1024) | Critic network layers |

### Curriculum stages

Training progresses through 4 stages automatically (1v0 → 1v1 → 2v2):

| Stage | Name | Format | Tick Skip | Timeout | Description |
|---|---|---|---|---|---|
| 0 | Solo Mechanics | 1v0 | 2 | 1800 | Speed flips, wave dashes, wall driving, aerials |
| 1 | 1v1 Mechanics | 1v1 | 2 | 2400 | Air dribbles, flip resets, wall plays, shooting |
| 2 | 1v1 Game Sense | 1v1 | 2 | 3600 | Advanced plays, saves, game reading |
| 3 | 2v2 Teamwork | 2v2 | 2 | 4800 | Rotation, passing, team coordination |

All stages use tick_skip=2 (60Hz decisions) for smooth, precise mechanics.
Stage advancement happens when entropy drops below 0.7 for 3 consecutive iterations.

### Render mode

Watch the bot play in real-time (1 environment, ~100MB VRAM):

```bash
python render_bot.py [--stage N] [--checkpoint PATH]
```

Controls: R=reset, Q=quit, 1-4=switch stage, Space=pause.

---

## File Structure

```
Lucifer/
  luciferbot.py              # Main training script
  render_bot.py              # 2D top-down visualizer (pygame)
  requirements.txt           # Pinned Python dependencies
  setup_ubuntu.sh            # One-command setup script
  .gitignore                 # Git ignore patterns
  curriculum_state.json      # Auto-generated: tracks curriculum progress
  Checkpoints_LuciferBot/    # Auto-generated: model checkpoints
  gpu_sim/                   # GPU simulation package
    __init__.py
    arena.py                 # Arena boundary + multi-surface detection
    collector.py             # Experience collection (variable n_agents)
    collision.py             # Ball-car, car-car collision detection
    constants.py             # Physics constants, stage config, agent layouts
    environment.py           # Batched GPU physics + mechanic scenarios
    game_state.py            # TensorState: all game state as GPU tensors
    observations.py          # 127-element obs with zero-padding
    physics.py               # Car physics, wall driving, surface-relative
    policy.py                # Custom 5-bin MultiDiscrete policy
    rewards.py               # 17 reward signals + events + team spirit
    utils.py                 # Quaternion math utilities
  collision_meshes/          # Arena collision mesh data
    soccar/mesh_*.cmf        # 16 mesh files for arena geometry
```

---

## Monitoring Training

While training runs, each iteration prints:

```
=======================================================
  LuciferBot — Stage 0 (Foundations) — Iter 100
=======================================================
  Total Steps       : 32,000,000
  Steps Collected   : 320,000
  Iter Time         : 4.3s  (collect: 3.9s  train: 2.7s  wait: 0.0s)
  Global SPS        : 74,000
  Mean Reward       : 1.40
  Entropy           : 3.50
  Clip Fraction     : 0.30
  Value Loss        : 5.00
  Stage Iter Count  : 100
```

| Metric | What it means |
|---|---|
| **Global SPS** | Steps per second — higher is faster training |
| **Mean Reward** | Average reward — should increase over time |
| **Entropy** | Policy randomness — decreases as policy specializes |
| **Clip Fraction** | PPO clipping — above 0.25 for 5 iters triggers LR decay |
| **Value Loss** | Critic prediction error — should decrease over time |

---

## Troubleshooting

**CUDA out of memory**: Reduce `N_ENVS` in `luciferbot.py`. Try 10,000 first.

**nvidia-smi not found**: Install NVIDIA drivers: `sudo apt install nvidia-driver-550`

**CUDA not available in Python**: Make sure you installed the CUDA-enabled PyTorch, not the CPU version. Reinstall with the `--index-url` flag from step 5.

**Training is slow (low SPS)**: Check that no other GPU-heavy processes are running (`nvidia-smi`). Close browsers, games, etc.

**Checkpoint not loading**: Checkpoints are loaded from `Checkpoints_LuciferBot/`, `Checkpoints_2v2_gpu/`, or `Checkpoints_2v2/` in that order. Make sure the folder contains a subfolder with `VARS.json` inside it.
