# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LuciferBot-experimental is the experimental branch of LuciferBot — a GPU-accelerated Rocket League bot trainer. It runs up to 160,000 parallel game environments entirely on GPU using PyTorch. The key experimental feature is an optional **C++ PPO extension** (`lucifer_ppo_cpp`) that replaces the Python PPO learner for faster training.

This repo shares the same `gpu_sim/` simulation package as the main Lucifer repo. Physics model now matches main Lucifer (speed-dependent throttle torque + steering curves from RocketSim, OBB-sphere ball-car collision, per-axis angular damping, coasting deceleration, powerslide steering). Flip mechanics remain simpler (no direction-dependent scaling or Z-damp).

## Commands

```bash
# Setup (first time)
chmod +x setup_ubuntu.sh && ./setup_ubuntu.sh
source venv/bin/activate

# Build C++ PPO extension (optional — falls back to Python PPO if not built)
pip install pybind11
python setup.py build_ext --inplace

# Train
python -u luciferbot.py          # -u for unbuffered output

# Render (2D top-down pygame view)
python render_bot.py [--stage N] [--checkpoint PATH]

# Verify install
python3 -c "from gpu_sim.environment import GPUEnvironment; env = GPUEnvironment(100, 'cuda', stage=0); print('OK')"
```

No test suite exists. Validation is done by running training and checking metrics (SPS, reward, entropy).

## Architecture

### Training Pipeline (`luciferbot.py`)

`GPULearner` orchestrates the main loop with **pipeline parallelism**: collection and training overlap via a background thread.

```
GPULearner.learn() loop:
  1. GPUCollector.collect_timesteps() — policy inference + env stepping on GPU
  2. Wait for previous training thread to finish
  3. Compute GAE (vectorized, on GPU)
  4. Sync frozen policy copy (collector uses frozen copy, trainer updates live copy)
  5. Launch PPOLearner.learn() in background thread
```

### C++ PPO Extension (`csrc/`)

Optional drop-in replacement for the Python PPO learner, built via `setup.py` using PyTorch's `CppExtension`. Auto-detected at import time (`import lucifer_ppo_cpp`).

**Files:**
- `ppo_module.cpp` — pybind11 bindings exposing `PPOLearner` to Python with zero-copy tensor passing (GIL released during `learn()`)
- `ppo_learner.h/cpp` — C++ PPO with AMP autocast, gradient clipping, GPU-resident metric accumulators (single CPU transfer at end)
- `networks.h/cpp` — C++ reimplementations of `MultiDiscreteFF5Bin` and `ValueEstimator` (mirrors Python exactly: same architecture, same Categorical distribution logic with -inf padding for duets)
- `grad_scaler.h/cpp` — Custom AMP GradScaler using `at::_amp_foreach_non_finite_check_and_unscale_` (fused CUDA kernel). Per-optimizer tracking, GPU-resident fp32 scale tensor for correct type promotion

**`CppPPOWrapper`** (in `luciferbot.py`) bridges C++ and Python: C++ owns the networks and does training, Python `MultiDiscreteFF5Bin`/`ValueEstimator` copies are synced via `_sync_policy_from_cpp()` / `_sync_value_net_from_cpp()` for frozen policy collection and GAE value predictions. Checkpoints are saved/loaded through Python `torch.save`/`torch.load` for format compatibility.

### Core Simulation (`gpu_sim/`)

All modules operate on `TensorState` — a flat struct of GPU tensors with shape `(E, ...)` for envs and `(E, A, ...)` for per-agent data, where E=n_envs and A=n_agents (1, 2, or 4).

**Data flow per step:**
```
controls → physics.apply_car_controls() → physics.integrate_positions()
         → arena.arena_collide_ball() + arena.arena_collide_cars()
         → collision.ball_car_collision() + collision.car_car_collision()
         → physics.update_rotation_vectors() + physics.update_demoed_cars()
         → rewards.compute() → observations.build_obs_batch()
```

**Key modules:**
- `game_state.py` — `TensorState`: all state as contiguous GPU tensors. Quaternion convention: (w,x,y,z).
- `environment.py` — `GPUEnvironment`: orchestrates step/reset, mechanic scenarios per stage, goal detection, boost pickup. Supports 1v0/1v1/2v2 via `get_agent_layout()`.
- `physics.py` — Car dynamics: speed-dependent throttle torque curves, piecewise-linear steer angle curves (+ powerslide), per-axis angular damping with input scaling, coasting deceleration, separate ground/air boost accel. Jump/flip simpler than main repo (no direction-dependent scaling).
- `arena.py` — Wall/floor/ceiling collision. Simpler surface detection (floor-only `car_on_ground`; no multi-surface wall/ceiling driving).
- `collision.py` — OBB-sphere ball-car collision with mass-based impulse, pairwise car-car demos/bumps. No-op for 1v0.
- `observations.py` — Builds `(E*A, 127)` obs tensor. Format: ball(9) + prev_action(8) + pads(34) + self(19) + ally(19) + enemy0(19) + enemy1(19). Absent slots zero-padded. Orange team gets inverted X/Y.
- `rewards.py` — Continuous + event reward channels with per-stage weights. Potential-based shaping, team spirit for 2v2.
- `policy.py` — `MultiDiscreteFF5Bin`: 5-bin discretization. Action space: `[5,5,5,5,5,2,2,2]` = 31 logits. Maps `{0,1,2,3,4}` → `{-1,-0.5,0,0.5,1}`.
- `ppo.py` — Python PPO with AMP mixed-precision (used when C++ extension unavailable). Checkpoints: `PPO_POLICY.pt`, `PPO_VALUE_NET.pt`, `*_OPTIMIZER.pt`, `VARS.json`.
- `collector.py` — `GPUCollector`: experience collection loop. Welford running stats for obs normalization. Re-initializes on stage change if n_agents changes.
- `constants.py` — Physics constants, arena dimensions, boost pads, `STAGE_CONFIG`, `get_agent_layout()`.
- `utils.py` — Quaternion math (multiply, integrate, euler conversion). All batched tensor ops.

### Curriculum System

4 stages with automatic advancement when entropy drops below 0.7 for 3 consecutive iterations:

| Stage | Format | n_envs  | tick_skip | timeout | Description |
|-------|--------|---------|-----------|---------|-------------|
| 0     | 1v0    | 160,000 | 2         | 1800    | Solo Mechanics |
| 1     | 1v1    | 80,000  | 2         | 2400    | 1v1 Mechanics |
| 2     | 1v1    | 80,000  | 2         | 3600    | 1v1 Game Sense |
| 3     | 2v2    | 40,000  | 2         | 4800    | 2v2 Teamwork |

`CurriculumTracker` manages stage advancement and LR decay (triggered by sustained high clip fraction). State persisted in `curriculum_state.json`.

### Checkpoints

Saved every 50 epochs to `Checkpoints_LuciferBot/`, keeping only the 3 most recent. Stage transitions get separate backups in `stage_backups/`. Ctrl+C triggers a manual save before exit. Checkpoint format is shared between Python and C++ PPO backends.

## Important Conventions

- Tensor shapes follow `(E, ...)` for global state, `(E, A, ...)` for per-agent state. Never transpose this convention.
- The frozen policy copy pattern (collector reads frozen, trainer updates live) is critical for pipeline parallelism — don't remove the `deepcopy` + `load_state_dict` sync.
- The C++ extension must produce identical network architectures and checkpoint formats as the Python version. The `CppPPOWrapper._sync_*_from_cpp()` methods copy weights from C++ to Python after each training step.
- The C++ `GradScaler` stores `scale_` as a GPU-resident fp32 tensor (not a Python float) — this is required for correct fp16→fp32 type promotion during `loss * scale_`.
- All stages use tick_skip=2 (60Hz decisions).
- Observation normalization uses Welford running stats on GPU. Stats are preserved across stage changes.
