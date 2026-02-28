import os
import time
import json
import gc
import types
import torch
import numpy as np
import multiprocessing
import warnings
import sys
import psutil
import threading
from copy import deepcopy
import threading
import copy
from torch.utils.tensorboard import SummaryWriter

# --- FILE LOGGING ---
# Tee stdout/stderr to lucifer_training.log so metrics survive os.execv restarts.
# Only active in the main process — workers use multiprocessing spawn and must not
# compete over the log file (Windows file locking would silently drop output).
class _TeeOutput:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try: s.write(data)
            except: pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except: pass
    def fileno(self):
        return self.streams[0].fileno()

if multiprocessing.current_process().name == "MainProcess":
    _log_file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lucifer_training.log"), "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeOutput(sys.__stdout__, _log_file)
    sys.stderr = _TeeOutput(sys.__stderr__, _log_file)

# --- CORE IMPORTS ---
import rlgym_sim
from rlgym_sim.utils.reward_functions.common_rewards import EventReward
from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym_sim.utils.action_parsers.discrete_act import DiscreteAction
from rlgym_sim.utils.state_setters import StateSetter, DefaultState

import rlgym_ppo.ppo
import rlgym_ppo.util as rlgym_util
from vectorized_collector import VectorizedCollector
from vectorized_env import STAGE_CONFIG

# --- CURRICULUM STATE ---
# Persists training state (stage, hyperparams, rolling metrics) across restarts.
# When a stage advance or hyperparam change is needed, the process saves this file,
# saves a checkpoint, then calls os.execv to restart itself cleanly.
CURRICULUM_STATE_FILE = "curriculum_state.json"
CURRICULUM_STAGE_NAMES = ["Foundations", "Game Play", "Mechanics", "Mastery"]

DEFAULT_CURRICULUM_STATE = {
    "stage": 0,
    "policy_lr": 2e-4,
    "critic_lr": 2e-4,
    "ent_coef": 0.01,
    "clip_high_count": 0,
    "stage_iter_count": 0,
    "entropy_low_count": 0,
}

def load_curriculum_state():
    if os.path.exists(CURRICULUM_STATE_FILE):
        state = {}
        with open(CURRICULUM_STATE_FILE) as f:
            state = json.load(f)
        for k, v in DEFAULT_CURRICULUM_STATE.items():
            if k not in state:
                state[k] = v
        return state
    return dict(DEFAULT_CURRICULUM_STATE)

def save_curriculum_state(state):
    with open(CURRICULUM_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def kill_zombies():
    """Kills only direct child worker processes spawned by this script."""
    print("\n[*] SURGICAL PURGE OF WORKER PROCESSES...")
    current_pid = os.getpid()
    try:
        parent = psutil.Process(current_pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        gone, still_alive = psutil.wait_procs(children, timeout=3)
        if still_alive:
            print(f"[!] {len(still_alive)} worker(s) didn't die — forcing...")
            for p in still_alive:
                try: p.kill()
                except: pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    print("[*] WORKERS PURGED.")


def kill_stale_training():
    """Kill any existing Python processes running lucifer2.py (from previous runs).
    Prevents memory exhaustion from dormant/orphaned workers."""
    current_pid = os.getpid()
    killed = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.pid == current_pid:
                continue
            cmdline = proc.info.get('cmdline') or []
            cmdline_str = ' '.join(cmdline).lower()
            if 'lucifer2.py' in cmdline_str and 'python' in (proc.info.get('name') or '').lower():
                print(f"[!] Killing stale training process PID={proc.pid}: {' '.join(cmdline[:3])}")
                proc.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    if killed:
        psutil.wait_procs([p for p in psutil.process_iter() if False], timeout=2)  # brief pause
        print(f"[*] Killed {killed} stale training process(es).")
    else:
        print("[*] No stale training processes found.")


# ---------------------------------------------------------------------------
# MEMORY MONITORING
# ---------------------------------------------------------------------------

def log_memory_usage(label):
    """Print current RAM and VRAM usage for diagnostics."""
    ram = psutil.Process().memory_info().rss / 1024**2
    msg = f"[MEM] {label}: RAM={ram:.0f}MB"
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        resv = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        msg += f"  VRAM={alloc:.0f}MB alloc / {resv:.0f}MB reserved / {peak:.0f}MB peak"
    print(msg)


def compute_max_envs(ts_per_iteration, target_ram_fraction=0.75):
    """Compute max number of envs that fit in available RAM.

    Estimates per-arena overhead at ~1.22 MB and reserves ~900 MB for
    Python/torch/collision meshes.  Caps at 5000 to stay sane.
    """
    total_ram = psutil.virtual_memory().total / 1024**2  # MB
    available = total_ram * target_ram_fraction - 900  # subtract fixed overhead
    per_env_mb = 1.22
    max_envs = int(available / per_env_mb)
    max_envs = max(10, min(max_envs, 2500))
    print(f"[*] Auto-scale: {total_ram:.0f}MB total RAM, "
          f"{available:.0f}MB usable → max {max_envs} envs")
    return max_envs


# ---------------------------------------------------------------------------
# AMP MONKEY-PATCH FOR PPOLearner.learn()
# ---------------------------------------------------------------------------

def _amp_learn(self, exp):
    """AMP-enabled replacement for PPOLearner.learn().

    Wraps forward passes in torch.cuda.amp.autocast and uses GradScaler
    for mixed-precision training.  Identical report dict to the original.
    """
    # Lazy-init scaler (persists across calls)
    if not hasattr(self, '_grad_scaler'):
        self._grad_scaler = torch.amp.GradScaler('cuda')
    scaler = self._grad_scaler

    n_iterations = 0
    n_minibatch_iterations = 0
    mean_entropy = 0
    mean_divergence = 0
    mean_val_loss = 0
    clip_fractions = []

    policy_before = torch.nn.utils.parameters_to_vector(
        self.policy.parameters()).cpu()
    critic_before = torch.nn.utils.parameters_to_vector(
        self.value_net.parameters()).cpu()

    t1 = time.time()
    for epoch in range(self.n_epochs):
        batches = exp.get_all_batches_shuffled(self.batch_size)
        for batch in batches:
            (batch_acts, batch_old_probs, batch_obs,
             batch_target_values, batch_advantages) = batch
            batch_acts = batch_acts.view(self.batch_size, -1)
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            for minibatch_slice in range(0, self.batch_size, self.mini_batch_size):
                start = minibatch_slice
                stop = start + self.mini_batch_size

                acts = batch_acts[start:stop].to(self.device)
                obs = batch_obs[start:stop].to(self.device)
                advantages = batch_advantages[start:stop].to(self.device)
                old_probs = batch_old_probs[start:stop].to(self.device)
                target_values = batch_target_values[start:stop].to(self.device)

                # ── AMP forward passes ──
                with torch.amp.autocast('cuda'):
                    vals = self.value_net(obs).view_as(target_values)
                    log_probs, entropy = self.policy.get_backprop_data(obs, acts)
                    log_probs = log_probs.view_as(old_probs)

                    ratio = torch.exp(log_probs - old_probs)
                    clipped = torch.clamp(
                        ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)

                    policy_loss = -torch.min(
                        ratio * advantages, clipped * advantages).mean()
                    minibatch_ratio = self.mini_batch_size / self.batch_size
                    value_loss = self.value_loss_fn(vals, target_values) * minibatch_ratio
                    ppo_loss = (policy_loss - entropy * self.ent_coef) * minibatch_ratio

                # KL & clip fraction (no grad, no autocast)
                with torch.no_grad():
                    log_ratio = log_probs.float() - old_probs.float()
                    kl = (torch.exp(log_ratio) - 1) - log_ratio
                    kl = kl.mean().detach().cpu().item()
                    clip_fraction = (
                        torch.mean((torch.abs(ratio.float() - 1) > self.clip_range).float())
                        .cpu().item())
                    clip_fractions.append(clip_fraction)

                # ── AMP backward + scaler ──
                scaler.scale(ppo_loss).backward()
                scaler.scale(value_loss).backward()

                mean_val_loss += (value_loss / minibatch_ratio).cpu().detach().item()
                mean_divergence += kl
                mean_entropy += entropy.cpu().detach().item()
                n_minibatch_iterations += 1

            # Unscale before clip_grad_norm_
            scaler.unscale_(self.policy_optimizer)
            scaler.unscale_(self.value_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.value_net.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), max_norm=0.5)

            scaler.step(self.policy_optimizer)
            scaler.step(self.value_optimizer)
            scaler.update()

            n_iterations += 1

    if n_iterations == 0:
        n_iterations = 1
    if n_minibatch_iterations == 0:
        n_minibatch_iterations = 1

    mean_entropy /= n_minibatch_iterations
    mean_divergence /= n_minibatch_iterations
    mean_val_loss /= n_minibatch_iterations
    mean_clip = np.mean(clip_fractions) if clip_fractions else 0

    policy_after = torch.nn.utils.parameters_to_vector(
        self.policy.parameters()).cpu()
    critic_after = torch.nn.utils.parameters_to_vector(
        self.value_net.parameters()).cpu()
    policy_update_magnitude = (policy_before - policy_after).norm().item()
    critic_update_magnitude = (critic_before - critic_after).norm().item()

    self.cumulative_model_updates += n_iterations

    report = {
        "PPO Batch Consumption Time": (time.time() - t1) / n_iterations,
        "Cumulative Model Updates": self.cumulative_model_updates,
        "Policy Entropy": mean_entropy,
        "Mean KL Divergence": mean_divergence,
        "Value Function Loss": mean_val_loss,
        "SB3 Clip Fraction": mean_clip,
        "Policy Update Magnitude": policy_update_magnitude,
        "Value Function Update Magnitude": critic_update_magnitude,
    }
    self.policy_optimizer.zero_grad()
    self.value_optimizer.zero_grad()

    return report


# ---------------------------------------------------------------------------
# MIXED STATE SETTER — replaces AerialStateSetter
# ---------------------------------------------------------------------------

class MixedStateSetter(StateSetter):
    """Configurable scenario mix: kickoff, random ground, aerial, ceiling.

    Each reset randomly picks a scenario based on configured probabilities.
    Probabilities are normalized internally so they always sum to 1.
    """
    _KICKOFF_POS = np.array([
        [-2048, -2560, 17], [2048, -2560, 17],
        [-256,  -3840, 17], [256,  -3840, 17], [0, -4608, 17],
    ], dtype=np.float32)

    def __init__(self, aerial=0.3, random_ground=0.3, ceiling=0.0, kickoff=0.4):
        total = aerial + random_ground + ceiling + kickoff
        self.p_kickoff = kickoff / total
        self.p_ground = random_ground / total
        self.p_aerial = aerial / total
        # ceiling = remainder

    def reset(self, state_wrapper):
        r = np.random.random()
        if r < self.p_kickoff:
            self._kickoff(state_wrapper)
        elif r < self.p_kickoff + self.p_ground:
            self._random_ground(state_wrapper)
        elif r < self.p_kickoff + self.p_ground + self.p_aerial:
            self._aerial(state_wrapper)
        else:
            self._ceiling(state_wrapper)

    def _kickoff(self, sw):
        """Standard kickoff: ball at center, cars at kickoff positions."""
        sw.ball.set_pos(0, 0, 93)
        sw.ball.set_lin_vel(0, 0, 0)
        sw.ball.set_ang_vel(0, 0, 0)
        n_blue = len(sw.blue_cars())
        n_orange = len(sw.orange_cars())
        idxs_b = np.random.choice(len(self._KICKOFF_POS), n_blue, replace=False)
        idxs_o = np.random.choice(len(self._KICKOFF_POS), n_orange, replace=False)
        for i, car in enumerate(sw.blue_cars()):
            p = self._KICKOFF_POS[idxs_b[i]]
            car.set_pos(p[0], p[1], p[2])
            car.set_rot(0, np.pi / 2, 0)
            car.set_lin_vel(0, 0, 0)
            car.boost = 0.33
        for i, car in enumerate(sw.orange_cars()):
            p = self._KICKOFF_POS[idxs_o[i]]
            car.set_pos(-p[0], -p[1], p[2])
            car.set_rot(0, -np.pi / 2, 0)
            car.set_lin_vel(0, 0, 0)
            car.boost = 0.33

    def _random_ground(self, sw):
        """Random ground positions with ball in play."""
        sw.ball.set_pos(
            np.random.uniform(-3000, 3000),
            np.random.uniform(-4000, 4000), 93)
        sw.ball.set_lin_vel(
            np.random.uniform(-1500, 1500),
            np.random.uniform(-1500, 1500), 0)
        sw.ball.set_ang_vel(0, 0, 0)
        for car in sw.blue_cars():
            car.set_pos(
                np.random.uniform(-3500, 3500),
                np.random.uniform(-4500, 0), 17)
            car.set_rot(0, np.random.uniform(-np.pi, np.pi), 0)
            car.set_lin_vel(0, 0, 0)
            car.boost = np.random.uniform(0.0, 1.0)
        for car in sw.orange_cars():
            car.set_pos(
                np.random.uniform(-3500, 3500),
                np.random.uniform(0, 4500), 17)
            car.set_rot(0, np.random.uniform(-np.pi, np.pi), 0)
            car.set_lin_vel(0, 0, 0)
            car.boost = np.random.uniform(0.0, 1.0)

    def _aerial(self, sw):
        """Ball in the air, cars on ground with high boost."""
        z = np.random.uniform(400, 1600)
        sw.ball.set_pos(
            np.random.uniform(-2500, 2500),
            np.random.uniform(-2500, 2500), z)
        sw.ball.set_lin_vel(
            np.random.uniform(-400, 400),
            np.random.uniform(-400, 400),
            np.random.uniform(-200, 200))
        sw.ball.set_ang_vel(0, 0, 0)
        for car in sw.blue_cars():
            car.set_pos(
                np.random.uniform(-3000, 3000),
                np.random.uniform(-4000, 0), 17)
            car.set_rot(0, np.pi / 2, 0)
            car.set_lin_vel(0, 0, 0)
            car.boost = np.random.uniform(0.7, 1.0)
        for car in sw.orange_cars():
            car.set_pos(
                np.random.uniform(-3000, 3000),
                np.random.uniform(0, 4000), 17)
            car.set_rot(0, -np.pi / 2, 0)
            car.set_lin_vel(0, 0, 0)
            car.boost = np.random.uniform(0.7, 1.0)

    def _ceiling(self, sw):
        """Ball high, one car per team near ceiling for ceiling-shot training."""
        z = np.random.uniform(1000, 1800)
        sw.ball.set_pos(
            np.random.uniform(-2000, 2000),
            np.random.uniform(-2000, 2000), z)
        sw.ball.set_lin_vel(
            np.random.uniform(-300, 300),
            np.random.uniform(-300, 300),
            np.random.uniform(-100, 200))
        sw.ball.set_ang_vel(0, 0, 0)

        blue_cars = list(sw.blue_cars())
        orange_cars = list(sw.orange_cars())

        # One blue car near ceiling
        if blue_cars:
            blue_cars[0].set_pos(
                np.random.uniform(-2000, 2000),
                np.random.uniform(-3000, 0), 1900)
            blue_cars[0].set_rot(np.pi, np.pi / 2, 0)  # upside down
            blue_cars[0].set_lin_vel(0, 0, 0)
            blue_cars[0].boost = np.random.uniform(0.5, 1.0)
        if len(blue_cars) > 1:
            blue_cars[1].set_pos(
                np.random.uniform(-3000, 3000),
                np.random.uniform(-4000, -2000), 17)
            blue_cars[1].set_rot(0, np.pi / 2, 0)
            blue_cars[1].set_lin_vel(0, 0, 0)
            blue_cars[1].boost = np.random.uniform(0.3, 0.8)

        # One orange car near ceiling
        if orange_cars:
            orange_cars[0].set_pos(
                np.random.uniform(-2000, 2000),
                np.random.uniform(0, 3000), 1900)
            orange_cars[0].set_rot(np.pi, -np.pi / 2, 0)
            orange_cars[0].set_lin_vel(0, 0, 0)
            orange_cars[0].boost = np.random.uniform(0.5, 1.0)
        if len(orange_cars) > 1:
            orange_cars[1].set_pos(
                np.random.uniform(-3000, 3000),
                np.random.uniform(2000, 4000), 17)
            orange_cars[1].set_rot(0, -np.pi / 2, 0)
            orange_cars[1].set_lin_vel(0, 0, 0)
            orange_cars[1].boost = np.random.uniform(0.3, 0.8)


# ---------------------------------------------------------------------------
# ENV FACTORY — 4 stages, rewards computed vectorized externally
# ---------------------------------------------------------------------------

def env_factory():
    _original_excepthook = sys.excepthook
    def _quiet_excepthook(exc_type, exc_val, exc_tb):
        if exc_type is KeyboardInterrupt:
            return
        _original_excepthook(exc_type, exc_val, exc_tb)
    sys.excepthook = _quiet_excepthook

    stage = load_curriculum_state()["stage"]
    cfg = STAGE_CONFIG.get(stage, STAGE_CONFIG[3])
    tick_skip = cfg["tick_skip"]
    timeout = cfg["timeout"]

    # State setter per stage
    if stage == 0:
        state_setter = DefaultState()
    elif stage == 1:
        state_setter = MixedStateSetter(aerial=0.2, random_ground=0.3, kickoff=0.5)
    elif stage == 2:
        state_setter = MixedStateSetter(aerial=0.4, random_ground=0.2, ceiling=0.1, kickoff=0.3)
    else:  # Stage 3+
        state_setter = MixedStateSetter(aerial=0.35, random_ground=0.25, ceiling=0.1, kickoff=0.3)

    # Dummy reward — actual rewards computed vectorized in VectorizedRewards
    reward_fn = EventReward(goal=1.0)

    return rlgym_sim.make(
        tick_skip=tick_skip, team_size=2, spawn_opponents=True,
        terminal_conditions=[GoalScoredCondition(), TimeoutCondition(timeout)],
        reward_fn=reward_fn,
        action_parser=DiscreteAction(),
        state_setter=state_setter)


# ---------------------------------------------------------------------------
# CURRICULUM TRACKER — 4 stages, entropy-based advancement
# ---------------------------------------------------------------------------

class CurriculumTracker:
    """
    Tracks training metrics and manages:
      1. Stage advancement — purely entropy-based: advance only when entropy
         drops below floor for ENTROPY_CONSECUTIVE iterations.
      2. Clip fraction management — reduce policy_lr if updates are too large.

    No step-based fallback — the bot must actually learn before advancing.
    4 stages: Foundations → Game Play → Mechanics → Mastery
    """

    # Entropy floors for advancement: when entropy drops below, advance
    ENTROPY_ADVANCE     = [1.5, 1.2, 1.0]        # Stage 0→1, 1→2, 2→3
    ENTROPY_CONSECUTIVE = 10                      # consecutive iters below floor

    CLIP_HIGH_THRESH    = 0.25
    CLIP_HIGH_ITERS     = 5

    def __init__(self, state):
        self.stage             = state["stage"]
        self.policy_lr         = state["policy_lr"]
        self.critic_lr         = state["critic_lr"]
        self.ent_coef          = state["ent_coef"]
        self.clip_high_count   = state.get("clip_high_count", 0)
        self.stage_iter_count  = state.get("stage_iter_count", 0)
        self.entropy_low_count = state.get("entropy_low_count", 0)
        self.restart_requested = False
        self.restart_reason    = ""

    def to_state_dict(self):
        return {
            "stage":             self.stage,
            "policy_lr":         self.policy_lr,
            "critic_lr":         self.critic_lr,
            "ent_coef":          self.ent_coef,
            "clip_high_count":   self.clip_high_count,
            "stage_iter_count":  self.stage_iter_count,
            "entropy_low_count": self.entropy_low_count,
        }

    def update(self, mean_reward, entropy, clip_fraction, value_loss):
        """Call after each training iteration. Returns True if restart needed."""
        self.stage_iter_count += 1

        # 1. CLIP FRACTION — reduce policy_lr if consistently too high
        if clip_fraction > self.CLIP_HIGH_THRESH:
            self.clip_high_count += 1
        else:
            self.clip_high_count = 0

        if self.clip_high_count >= self.CLIP_HIGH_ITERS:
            new_lr = round(self.policy_lr * 0.80, 7)
            if new_lr >= 1e-4:
                self.policy_lr = new_lr
                self.clip_high_count = 0
                self.restart_reason = (
                    f"Clip fraction >{self.CLIP_HIGH_THRESH} for "
                    f"{self.CLIP_HIGH_ITERS} iters — policy_lr → {self.policy_lr:.2e}")
                self.restart_requested = True
                return True

        # 2. STAGE ADVANCEMENT — purely entropy-based (no fallback)
        if self.stage < 3:
            floor = self.ENTROPY_ADVANCE[self.stage]
            if entropy < floor:
                self.entropy_low_count += 1
            else:
                self.entropy_low_count = 0

            if self.entropy_low_count >= self.ENTROPY_CONSECUTIVE:
                old_stage = self.stage
                self.stage += 1
                self.stage_iter_count = 0
                self.entropy_low_count = 0
                self.restart_reason = (
                    f"STAGE ADVANCE {old_stage} → {self.stage} "
                    f"({CURRICULUM_STAGE_NAMES[self.stage]}): "
                    f"Entropy {entropy:.2f} < {floor} for {self.ENTROPY_CONSECUTIVE} iters")
                self.restart_requested = True
                return True

        return False


# ---------------------------------------------------------------------------
# HISTORICAL POLICY POOL — Self-play against past checkpoints (Phase 4)
# ---------------------------------------------------------------------------

class HistoricalPolicyPool:
    """Stores policy snapshots for self-play diversity.

    Active from Stage 2+. Every `save_interval` epochs, saves the current
    policy state_dict. Every `swap_interval` iterations, temporarily loads
    a historical snapshot for collection (10% historical, 90% current).

    Not yet activated — will be enabled once Stage 0/1 are confirmed working.
    """

    def __init__(self, save_dir="historical_policies", max_snapshots=20,
                 save_interval=50, swap_interval=10):
        self.save_dir = save_dir
        self.max_snapshots = max_snapshots
        self.save_interval = save_interval
        self.swap_interval = swap_interval
        self.snapshots = []  # list of file paths, newest last
        os.makedirs(save_dir, exist_ok=True)

    def maybe_save(self, epoch, policy):
        """Save policy snapshot if interval reached."""
        if epoch % self.save_interval != 0:
            return
        path = os.path.join(self.save_dir, f"policy_epoch_{epoch}.pt")
        torch.save(policy.state_dict(), path)
        self.snapshots.append(path)
        # Prune old snapshots
        while len(self.snapshots) > self.max_snapshots:
            old = self.snapshots.pop(0)
            if os.path.exists(old):
                os.remove(old)

    def sample_snapshot(self):
        """Return a random snapshot path (recency-weighted). None if empty."""
        if not self.snapshots:
            return None
        n = len(self.snapshots)
        # Recency-weighted: newer snapshots more likely
        weights = np.arange(1, n + 1, dtype=np.float64)
        weights /= weights.sum()
        idx = np.random.choice(n, p=weights)
        return self.snapshots[idx]


# ---------------------------------------------------------------------------
# LEARNER
# ---------------------------------------------------------------------------

class Learner:
    def __init__(self, n_proc=40, ts_per_iteration=100000):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoints_folder = "Checkpoints_2v2"
        self.epoch = 0

        # Load curriculum state — this drives both env rewards (via env_factory)
        # and hyperparameters applied here.
        curriculum_state = load_curriculum_state()
        self.curriculum = CurriculumTracker(curriculum_state)

        self.ts_per_iteration = ts_per_iteration

        stage_name = CURRICULUM_STAGE_NAMES[self.curriculum.stage]
        print(f"[*] Curriculum Stage {self.curriculum.stage}: {stage_name}")
        print(f"[*] Steps per iteration: {ts_per_iteration:,}")
        print(f"[*] Hyperparams — policy_lr={self.curriculum.policy_lr:.2e}  "
              f"critic_lr={self.curriculum.critic_lr:.2e}  ent_coef={self.curriculum.ent_coef}")

        cfg = STAGE_CONFIG.get(self.curriculum.stage, STAGE_CONFIG[0])
        print(f"[*] Stage config — tick_skip={cfg['tick_skip']}  timeout={cfg['timeout']}")

        self.agent = VectorizedCollector(
            build_env_fn=env_factory, n_envs=n_proc, standardize_obs=True)
        shapes = self.agent.shapes

        self.ppo_learner = rlgym_ppo.ppo.PPOLearner(
            obs_space_size=np.prod(shapes[0]),
            act_space_size=shapes[1],
            device=self.device,
            batch_size=50000,
            mini_batch_size=50000,
            n_epochs=2,
            policy_type=shapes[2],
            policy_layer_sizes=(2048, 2048, 1024, 1024),
            critic_layer_sizes=(2048, 2048, 1024, 1024),
            continuous_var_range=0.1,
            policy_lr=self.curriculum.policy_lr,
            critic_lr=self.curriculum.critic_lr,
            clip_range=0.2,  # wider for fresh policy (was 0.1)
            ent_coef=self.curriculum.ent_coef,
        )
        # AMP monkey-patch: replace learn() with mixed-precision version
        if self.device == "cuda":
            self.ppo_learner.learn = types.MethodType(_amp_learn, self.ppo_learner)
            print("[*] AMP mixed-precision training enabled (monkey-patched PPOLearner.learn)")

        # Pipeline parallelism: frozen policy copy for collection on CPU
        self._frozen_policy = deepcopy(self.ppo_learner.policy)
        self._frozen_policy.cpu()
        self._frozen_policy.device = "cpu"
        self._frozen_policy.eval()
        self.agent.policy = self._frozen_policy

        # Background training state
        self._training_thread = None
        self._training_report = None
        self._training_meta = None
        self._training_exception = None

        # Historical policy pool (Phase 4 — activated from Stage 2+)
        self._policy_pool = HistoricalPolicyPool()

        # Per-parameter gradient clipping hooks — hard-clamp to [-1, 1], zero NaN/Inf.
        # When AMP is active, GradScaler produces intentionally large gradients that
        # look like Inf before unscale_(). The scaler handles this correctly via
        # unscale_ + skip-on-inf. So we disable the hooks when AMP is in use —
        # clip_grad_norm_ after unscale_ (in _amp_learn) is sufficient.
        if self.device != "cuda":
            grad_clip_val = 1.0
            def make_clip_hook(name):
                def clip_hook(grad):
                    if grad is not None and not torch.isfinite(grad).all():
                        print(f"[!] NaN/Inf gradient in {name} — zeroing.")
                        return torch.zeros_like(grad)
                    return torch.clamp(grad, -grad_clip_val, grad_clip_val)
                return clip_hook

            for name, param in self.ppo_learner.policy.named_parameters():
                param.register_hook(make_clip_hook(f"policy/{name}"))
            for name, param in self.ppo_learner.value_net.named_parameters():
                param.register_hook(make_clip_hook(f"critic/{name}"))
            print("[*] Gradient clipping hooks registered on policy and critic.")
        else:
            print("[*] AMP active — gradient clipping handled by GradScaler + clip_grad_norm_.")

        self.load_latest_checkpoint()

        # Sync frozen policy (CPU) with loaded checkpoint weights
        self._frozen_policy.load_state_dict(
            {k: v.cpu() for k, v in self.ppo_learner.policy.state_dict().items()})

    def _launch_training(self, buffer, meta):
        """Launch PPO training in a background thread."""
        ppo = self.ppo_learner

        def _train():
            try:
                t1 = time.time()
                report = ppo.learn(buffer)
                buffer.clear()
                gc.collect()
                torch.cuda.empty_cache()
                t_train = time.time() - t1

                # NaN weight check
                nan_found = False
                for name, param in ppo.policy.named_parameters():
                    if not torch.isfinite(param).all():
                        print(f"[!!!] NaN/Inf in POLICY weight: {name}")
                        nan_found = True
                for name, param in ppo.value_net.named_parameters():
                    if not torch.isfinite(param).all():
                        print(f"[!!!] NaN/Inf in CRITIC weight: {name}")
                        nan_found = True
                if nan_found:
                    print("[!!!] NaN detected. Network unrecoverable — delete checkpoints and restart.")

                meta['t_train'] = t_train
                meta['nan_found'] = nan_found
                self._training_report = report
                self._training_meta = meta
            except Exception as e:
                self._training_exception = e

        self._training_thread = threading.Thread(target=_train, daemon=True)
        self._training_thread.start()

    def _wait_for_training(self):
        """Wait for background training to complete.
        Returns (report, metadata) or (None, None) if no training was running."""
        if self._training_thread is None:
            return None, None
        self._training_thread.join()
        self._training_thread = None
        if self._training_exception is not None:
            exc = self._training_exception
            self._training_exception = None
            raise exc
        report = self._training_report
        meta = self._training_meta
        self._training_report = None
        self._training_meta = None
        return report, meta

    def _finalize_prev_iteration(self, report, meta):
        """Display metrics, run curriculum check, save periodic checkpoints."""
        epoch = meta['epoch']
        steps_gained = meta['steps_gained']
        t_coll = meta['t_coll']
        t_train = meta['t_train']
        mean_reward = meta['mean_reward']
        cumul_steps = meta['cumul_steps']
        wall_time = meta['wall_time']
        stage = self.curriculum.stage
        stage_name = CURRICULUM_STAGE_NAMES[stage]

        entropy = report.get('Policy Entropy', 0)
        clip = report.get('SB3 Clip Fraction', 0)
        value_loss = report.get('Value Function Loss', 0)

        print("\n\n" + "=" * 50)
        print(f"  LUCIFER v5.0 — Stage {stage} ({stage_name}) — Iter {epoch}")
        print("=" * 50)
        print(f"  Total Steps       : {cumul_steps:,}")
        print(f"  Steps Collected   : {steps_gained:,}")
        print(f"  Iter Time         : {wall_time:.2f}s  (collect: {t_coll:.1f}s  train: {t_train:.1f}s  [pipelined])")
        sps = int(steps_gained / wall_time) if wall_time > 0 else 0
        print(f"  Global SPS        : {sps}")
        print("")
        print(f"  Mean Reward       : {mean_reward:.6f}")
        print(f"  Entropy           : {entropy:.4f}")
        print(f"  Mean KL Div       : {report.get('Mean KL Divergence', 0):.6f}")
        print(f"  Clip Fraction     : {clip:.4f}")
        print(f"  Value Loss        : {value_loss:.4f}")
        print(f"  Policy Upd. Mag.  : {report.get('Policy Update Magnitude', 0):.6f}")
        print("")
        print(f"  Stage Iter Count  : {self.curriculum.stage_iter_count}")
        print(f"  Entropy Low Count : {self.curriculum.entropy_low_count}")
        print(f"  Cumul. Updates    : {int(report.get('Cumulative Model Updates', 0))}")
        print("=" * 50)
        log_memory_usage("post-training")
        print("")

        # Curriculum update
        if self.curriculum.update(mean_reward, entropy, clip, value_loss):
            save_curriculum_state(self.curriculum.to_state_dict())
            print(f"\n[CURRICULUM] {self.curriculum.restart_reason}")
            print(f"[CURRICULUM] Saving stage transition checkpoint...")
            self.save_stage_checkpoint(self.curriculum.stage, epoch)
            self._restart()

        # Historical policy pool save (Phase 4)
        if self.curriculum.stage >= 2:
            self._policy_pool.maybe_save(epoch, self.ppo_learner.policy)

        # Periodic checkpoint (training is done, safe to save)
        if epoch % 50 == 0:
            self.save_checkpoint(f"lucifer_epoch_{epoch}")

    def _restart(self):
        """Replace this process with a fresh copy (curriculum state already saved by caller)."""
        for method in ('stop', 'terminate', 'shutdown', 'close'):
            fn = getattr(self.agent, method, None)
            if fn:
                try: fn()
                except: pass
                break
        kill_zombies()
        sys.stdout.flush()
        print("[CURRICULUM] RESTARTING...\n")
        os.execv(sys.executable, [sys.executable, "-u"] + sys.argv)

    def learn(self):
        iter_start = time.time()
        stage_name = CURRICULUM_STAGE_NAMES[self.curriculum.stage]
        print(f"\n[*] Iteration {self.epoch + 1} [Stage {self.curriculum.stage}: {stage_name}]: Collecting steps...")

        # ── 1. Collect timesteps (overlapped with previous training) ──
        # Previous training may still be running in background — that's the point.
        # Collection uses frozen policy copy, training updates original policy.
        t0 = time.time()
        exp, metrics, steps_gained, _ = self.agent.collect_timesteps(self.ts_per_iteration)
        t_coll = time.time() - t0
        print(f"[*] Collected {steps_gained:,} steps in {t_coll:.1f}s")
        log_memory_usage("post-collection")
        sys.stdout.flush()

        # ── 2. Wait for previous training to finish ──
        # By now, training (6.5s) has had ~10s of collection time to complete.
        # In steady state this wait is near-zero.
        t_wait_start = time.time()
        prev_report, prev_meta = self._wait_for_training()
        t_wait = time.time() - t_wait_start
        if t_wait > 0.1:
            print(f"[*] Waited {t_wait:.1f}s for previous training")

        # ── 3. Finalize previous iteration (display metrics, curriculum, checkpoint) ──
        if prev_meta is not None:
            prev_meta['wall_time'] = iter_start - prev_meta['iter_start']
            self._finalize_prev_iteration(prev_report, prev_meta)

        # ── 4. Sync frozen policy (CPU) with trained policy ──
        self._frozen_policy.load_state_dict(
            {k: v.cpu() for k, v in self.ppo_learner.policy.state_dict().items()})
        gc.collect()
        torch.cuda.empty_cache()

        # ── 5. Compute GAE using updated value net ──
        states, actions, log_probs, rewards, next_states, dones, truncated = exp
        val_all = np.vstack([states, next_states[-1:]])
        # Batch value prediction to avoid GPU OOM on small VRAM cards
        val_chunks = []
        VAL_CHUNK = 20000
        for ci in range(0, len(val_all), VAL_CHUNK):
            chunk = torch.as_tensor(val_all[ci:ci+VAL_CHUNK],
                                     dtype=torch.float32, device=self.device)
            with torch.no_grad():
                val_chunks.append(self.ppo_learner.value_net(chunk).cpu().flatten().numpy())
            del chunk
        val_preds = np.concatenate(val_chunks)
        del val_chunks, val_all
        gc.collect()
        torch.cuda.empty_cache()

        rewards_arr = np.array(rewards)
        rewards_std = rewards_arr.std()
        if rewards_std > 1e-6:
            rewards_norm = (rewards_arr - rewards_arr.mean()) / (rewards_std + 1e-8)
            rewards_norm = np.clip(rewards_norm, -10.0, 10.0)
        else:
            rewards_norm = rewards_arr - rewards_arr.mean()

        vt, adv, _ = rlgym_util.torch_functions.compute_gae(
            rewards_norm, dones, truncated, np.nan_to_num(val_preds))

        # ── 6. Submit to fresh experience buffer (CPU to save VRAM) ──
        buffer = rlgym_ppo.ppo.ExperienceBuffer(
            max_size=self.ts_per_iteration + 20000, device="cpu", seed=123)
        buffer.submit_experience(
            states, actions, log_probs, rewards_norm, next_states, dones, truncated, vt, adv)
        del states, actions, log_probs, rewards_norm, next_states, dones, truncated, vt, adv
        del rewards_arr, rewards_std, val_preds, exp
        gc.collect()

        buf_size = buffer.rewards.shape[0] if len(buffer.rewards.shape) > 0 else 0
        print(f"[*] Buffer size: {buf_size:,} samples (batch=50k, mini=50k, epochs=2 → 8 passes)")
        log_memory_usage("pre-training")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        sys.stdout.flush()

        # ── 7. Launch training in background thread ──
        self.epoch += 1
        meta = {
            'epoch': self.epoch,
            'steps_gained': steps_gained,
            't_coll': t_coll,
            'mean_reward': float(np.mean(rewards)),
            'cumul_steps': self.agent.cumulative_timesteps,
            'iter_start': iter_start,
        }
        self._launch_training(buffer, meta)

        log_memory_usage("training-launched")

    def save_stage_checkpoint(self, stage, epoch):
        """Save a stage transition checkpoint that is never pruned."""
        stage_dir = os.path.join(self.checkpoints_folder, "stage_backups")
        os.makedirs(stage_dir, exist_ok=True)
        label = f"stage_{stage}_epoch_{epoch}"
        path = os.path.join(stage_dir, label)
        os.makedirs(path, exist_ok=True)
        self.ppo_learner.save_to(path)
        with open(os.path.join(path, "VARS.json"), "w") as f:
            json.dump({"steps": self.agent.cumulative_timesteps, "epoch": self.epoch, "stage": stage}, f)
        print(f"[*] Stage transition checkpoint saved: stage_backups/{label}")

    def save_checkpoint(self, label):
        path = os.path.join(self.checkpoints_folder, label)
        os.makedirs(path, exist_ok=True)
        self.ppo_learner.save_to(path)
        with open(os.path.join(path, "VARS.json"), "w") as f:
            json.dump({"steps": self.agent.cumulative_timesteps, "epoch": self.epoch}, f)

        import shutil
        all_ckpts = sorted([
            d for d in os.listdir(self.checkpoints_folder)
            if os.path.isdir(os.path.join(self.checkpoints_folder, d)) and d != "stage_backups"
        ], key=lambda d: os.path.getmtime(os.path.join(self.checkpoints_folder, d)))
        while len(all_ckpts) > 3:
            oldest = all_ckpts.pop(0)
            oldest_path = os.path.join(self.checkpoints_folder, oldest)
            shutil.rmtree(oldest_path)
            print(f"[*] Deleted old checkpoint: {oldest}")

    def load_latest_checkpoint(self):
        if not os.path.exists(self.checkpoints_folder): return
        potential = [d for d in os.listdir(self.checkpoints_folder)
                     if os.path.isdir(os.path.join(self.checkpoints_folder, d)) and d != "stage_backups"]
        if potential:
            latest = sorted(potential, key=lambda d: os.path.getmtime(
                os.path.join(self.checkpoints_folder, d)))[-1]
            path = os.path.join(self.checkpoints_folder, latest)
            if os.path.exists(os.path.join(path, "VARS.json")):
                self.ppo_learner.load_from(path)
                with open(os.path.join(path, "VARS.json"), "r") as f:
                    v = json.load(f)
                    self.agent.cumulative_timesteps, self.epoch = v["steps"], v["epoch"]
                print(f"[*] Loaded checkpoint at epoch {self.epoch} ({self.agent.cumulative_timesteps:,} steps)")


if __name__ == "__main__":
    kill_stale_training()
    try:
        n_proc = compute_max_envs(ts_per_iteration=200000)
        learner = Learner(n_proc=n_proc, ts_per_iteration=200000)
        while True:
            learner.learn()
    except KeyboardInterrupt:
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull_fd, 2)
        os.close(_devnull_fd)
        print("\n[!] MANUAL STOP.")
        if 'learner' in locals():
            # Wait for background training to finish before saving
            try:
                prev_report, prev_meta = learner._wait_for_training()
                if prev_meta is not None:
                    prev_meta['wall_time'] = time.time() - prev_meta['iter_start']
                    learner._finalize_prev_iteration(prev_report, prev_meta)
            except Exception:
                pass
            label = f"lucifer_manual_epoch_{learner.epoch}"
            print(f"[*] Saving checkpoint: {label}")
            learner.save_checkpoint(label)
            save_curriculum_state(learner.curriculum.to_state_dict())
            print(f"[*] Checkpoint + curriculum state saved.")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        if 'learner' in locals():
            # Wait for any background training before cleanup
            try:
                learner._wait_for_training()
            except Exception:
                pass
            for method in ('stop', 'terminate', 'shutdown', 'close'):
                fn = getattr(learner.agent, method, None)
                if fn:
                    try: fn()
                    except: pass
                    break
        kill_zombies()
        sys.stdout.flush()
        print("[*] BYE.\n")
        os._exit(0)
