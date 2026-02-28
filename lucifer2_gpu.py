"""lucifer2_gpu.py — GPU-accelerated training script using gpu_sim.

Minimal changes from lucifer2.py:
- GPUCollector replaces VectorizedCollector
- No rlgym_sim import (GPU sim replaces it)
- No env_factory (GPU sim handles resets internally)
- Adjustable n_envs (10,000-50,000 instead of 1,000)
- Everything else identical: CurriculumTracker, PPO, AMP, pipeline parallelism
"""

import os
import time
import json
import gc
import types
import torch
import numpy as np
import sys
import psutil
import threading
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

import rlgym_ppo.ppo
import rlgym_ppo.util as rlgym_util
from gpu_sim.collector import GPUCollector
from gpu_sim.constants import STAGE_CONFIG

# ─── Curriculum state persistence ───
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


# ─── Memory monitoring ───
def log_memory_usage(label):
    ram = psutil.Process().memory_info().rss / 1024**2
    msg = f"[MEM] {label}: RAM={ram:.0f}MB"
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        resv = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        msg += f"  VRAM={alloc:.0f}MB alloc / {resv:.0f}MB reserved / {peak:.0f}MB peak"
    print(msg)


# ─── AMP monkey-patch (identical to lucifer2.py) ───
def _amp_learn(self, exp):
    if not hasattr(self, '_grad_scaler'):
        self._grad_scaler = torch.amp.GradScaler('cuda')
    scaler = self._grad_scaler

    n_iterations = 0
    n_minibatch_iterations = 0
    mean_entropy = 0
    mean_divergence = 0
    mean_val_loss = 0
    clip_fractions = []

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

                with torch.amp.autocast('cuda'):
                    vals = self.value_net(obs).view_as(target_values)
                    log_probs, entropy = self.policy.get_backprop_data(obs, acts)
                    log_probs = log_probs.view_as(old_probs)
                    ratio = torch.exp(log_probs - old_probs)
                    clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
                    minibatch_ratio = self.mini_batch_size / self.batch_size
                    value_loss = self.value_loss_fn(vals, target_values) * minibatch_ratio
                    ppo_loss = (policy_loss - entropy * self.ent_coef) * minibatch_ratio

                with torch.no_grad():
                    log_ratio = log_probs.float() - old_probs.float()
                    kl = (torch.exp(log_ratio) - 1) - log_ratio
                    kl = kl.mean().detach().cpu().item()
                    clip_fraction = torch.mean(
                        (torch.abs(ratio.float() - 1) > self.clip_range).float()).cpu().item()
                    clip_fractions.append(clip_fraction)

                scaler.scale(ppo_loss).backward()
                scaler.scale(value_loss).backward()
                mean_val_loss += (value_loss / minibatch_ratio).cpu().detach().item()
                mean_divergence += kl
                mean_entropy += entropy.cpu().detach().item()
                n_minibatch_iterations += 1

            scaler.unscale_(self.policy_optimizer)
            scaler.unscale_(self.value_optimizer)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            scaler.step(self.policy_optimizer)
            scaler.step(self.value_optimizer)
            scaler.update()
            n_iterations += 1

    if n_iterations == 0: n_iterations = 1
    if n_minibatch_iterations == 0: n_minibatch_iterations = 1

    mean_entropy /= n_minibatch_iterations
    mean_divergence /= n_minibatch_iterations
    mean_val_loss /= n_minibatch_iterations
    mean_clip = np.mean(clip_fractions) if clip_fractions else 0

    self.cumulative_model_updates += n_iterations

    return {
        "PPO Batch Consumption Time": (time.time() - t1) / n_iterations,
        "Cumulative Model Updates": self.cumulative_model_updates,
        "Policy Entropy": mean_entropy,
        "Mean KL Divergence": mean_divergence,
        "Value Function Loss": mean_val_loss,
        "SB3 Clip Fraction": mean_clip,
    }


# ─── CurriculumTracker (identical to lucifer2.py) ───
class CurriculumTracker:
    ENTROPY_ADVANCE     = [1.5, 1.3, 1.0]
    ENTROPY_CONSECUTIVE = 10
    MIN_STAGE_ITERS     = [500, 2000, 2500]
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
            "stage": self.stage, "policy_lr": self.policy_lr,
            "critic_lr": self.critic_lr, "ent_coef": self.ent_coef,
            "clip_high_count": self.clip_high_count,
            "stage_iter_count": self.stage_iter_count,
            "entropy_low_count": self.entropy_low_count,
        }

    def update(self, mean_reward, entropy, clip_fraction, value_loss):
        self.stage_iter_count += 1

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

        if self.stage < 3:
            floor = self.ENTROPY_ADVANCE[self.stage]
            min_iters = self.MIN_STAGE_ITERS[self.stage]
            if entropy < floor:
                self.entropy_low_count += 1
            else:
                self.entropy_low_count = 0
            if (self.entropy_low_count >= self.ENTROPY_CONSECUTIVE
                    and self.stage_iter_count >= min_iters):
                old_stage = self.stage
                self.stage += 1
                self.stage_iter_count = 0
                self.entropy_low_count = 0
                self.restart_reason = (
                    f"STAGE ADVANCE {old_stage} → {self.stage} "
                    f"({CURRICULUM_STAGE_NAMES[self.stage]})")
                self.restart_requested = True
                return True

            if self.stage_iter_count % 100 == 0:
                print(f"[CURRICULUM] Stage {self.stage}: iter {self.stage_iter_count}/{min_iters}, "
                      f"entropy {entropy:.2f} (floor {floor}), "
                      f"low_count {self.entropy_low_count}/{self.ENTROPY_CONSECUTIVE}")
        return False


# ─── GPU Learner ───
class GPULearner:
    def __init__(self, n_envs=10000, ts_per_iteration=200000):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        assert self.device == "cuda", "GPU sim requires CUDA!"
        self.checkpoints_folder = "Checkpoints_2v2_gpu"
        self.epoch = 0

        curriculum_state = load_curriculum_state()
        self.curriculum = CurriculumTracker(curriculum_state)
        self.ts_per_iteration = ts_per_iteration

        stage_name = CURRICULUM_STAGE_NAMES[self.curriculum.stage]
        print(f"\n{'='*60}")
        print(f"  LUCIFER GPU SIM — Stage {self.curriculum.stage}: {stage_name}")
        print(f"  {n_envs:,} parallel environments on GPU")
        print(f"  Steps per iteration: {ts_per_iteration:,}")
        print(f"{'='*60}")

        # GPU Collector (replaces VectorizedCollector)
        self.agent = GPUCollector(
            n_envs=n_envs, device=self.device,
            standardize_obs=True, stage=self.curriculum.stage)

        # PPO Learner (same architecture as lucifer2.py)
        self.ppo_learner = rlgym_ppo.ppo.PPOLearner(
            obs_space_size=127,
            act_space_size=8,
            device=self.device,
            batch_size=50000,
            mini_batch_size=50000,
            n_epochs=2,
            policy_type=1,  # MultiDiscrete
            policy_layer_sizes=(2048, 2048, 1024, 1024),
            critic_layer_sizes=(2048, 2048, 1024, 1024),
            continuous_var_range=0.1,
            policy_lr=self.curriculum.policy_lr,
            critic_lr=self.curriculum.critic_lr,
            clip_range=0.2,
            ent_coef=self.curriculum.ent_coef,
        )

        # AMP monkey-patch
        self.ppo_learner.learn = types.MethodType(_amp_learn, self.ppo_learner)
        print("[*] AMP mixed-precision training enabled")

        # Background training state
        self._training_thread = None
        self._training_report = None
        self._training_meta = None
        self._training_exception = None

        # Load checkpoint BEFORE torch.compile (compiled models can't load_state_dict)
        self.load_latest_checkpoint()

        # torch.compile disabled — CUDAGraphs conflicts with tensor reuse in collection loop

        # Pipeline parallelism: frozen policy copy for collection
        # In GPU sim, frozen copy stays on GPU (same device)
        self._frozen_policy = deepcopy(self.ppo_learner.policy)
        self._frozen_policy.eval()
        self.agent.policy = self._frozen_policy

        log_memory_usage("init-complete")

    def _launch_training(self, buffer, meta):
        ppo = self.ppo_learner
        def _train():
            try:
                t1 = time.time()
                report = ppo.learn(buffer)
                buffer.clear()
                meta['t_train'] = time.time() - t1
                meta['nan_found'] = False
                for name, param in ppo.policy.named_parameters():
                    if not torch.isfinite(param).all():
                        print(f"[!!!] NaN/Inf in POLICY: {name}")
                        meta['nan_found'] = True
                for name, param in ppo.value_net.named_parameters():
                    if not torch.isfinite(param).all():
                        print(f"[!!!] NaN/Inf in CRITIC: {name}")
                        meta['nan_found'] = True
                self._training_report = report
                self._training_meta = meta
            except Exception as e:
                self._training_exception = e
        self._training_thread = threading.Thread(target=_train, daemon=True)
        self._training_thread.start()

    def _wait_for_training(self):
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

        sps = int(steps_gained / wall_time) if wall_time > 0 else 0

        print("\n\n" + "=" * 55)
        print(f"  LUCIFER GPU v5.0 — Stage {stage} ({stage_name}) — Iter {epoch}")
        print("=" * 55)
        print(f"  Total Steps       : {cumul_steps:,}")
        print(f"  Steps Collected   : {steps_gained:,}")
        t_wait = meta.get('t_wait', 0)
        print(f"  Iter Time         : {wall_time:.2f}s  (collect: {t_coll:.1f}s  train: {t_train:.1f}s  wait: {t_wait:.1f}s)")
        print(f"  Global SPS        : {sps:,}")
        print(f"  Mean Reward       : {mean_reward:.6f}")
        print(f"  Entropy           : {entropy:.4f}")
        print(f"  Clip Fraction     : {clip:.4f}")
        print(f"  Value Loss        : {value_loss:.4f}")
        print(f"  Stage Iter Count  : {self.curriculum.stage_iter_count}")
        print("=" * 55)
        log_memory_usage("post-training")

        if self.curriculum.update(mean_reward, entropy, clip, value_loss):
            save_curriculum_state(self.curriculum.to_state_dict())
            print(f"\n[CURRICULUM] {self.curriculum.restart_reason}")
            self.save_stage_checkpoint(self.curriculum.stage, epoch)
            # Update collector stage (no restart needed — GPU sim is lightweight)
            self.agent.set_stage(self.curriculum.stage)
            self.curriculum.restart_requested = False

        if epoch % 50 == 0:
            self.save_checkpoint(f"lucifer_gpu_epoch_{epoch}")

    def learn(self):
        iter_start = time.time()
        stage_name = CURRICULUM_STAGE_NAMES[self.curriculum.stage]
        print(f"\n[*] Iter {self.epoch + 1} [Stage {self.curriculum.stage}: {stage_name}]: Collecting...")

        # 1. Collect timesteps
        t0 = time.time()
        exp, _, steps_gained, _ = self.agent.collect_timesteps(self.ts_per_iteration)
        t_coll = time.time() - t0
        print(f"[*] Collected {steps_gained:,} steps in {t_coll:.1f}s "
              f"({int(steps_gained/t_coll):,} SPS)")

        # 2. Wait for previous training
        t_wait_start = time.time()
        prev_report, prev_meta = self._wait_for_training()
        t_wait = time.time() - t_wait_start
        if t_wait > 0.1:
            print(f"[*] Waited {t_wait:.1f}s for previous training")

        # 3. Finalize previous iteration
        if prev_meta is not None:
            prev_meta['wall_time'] = iter_start - prev_meta['iter_start']
            self._finalize_prev_iteration(prev_report, prev_meta)

        # 4. Sync frozen policy
        t_sync = time.time()
        self._frozen_policy.load_state_dict(self.ppo_learner.policy.state_dict())

        # 5. Compute GAE — chunked value predictions (50k chunks = 4 round-trips)
        t_gae = time.time()
        states, actions, log_probs, rewards, next_states, dones, truncated = exp
        n_samples = len(states) + 1  # states + 1 bootstrap value
        val_preds = np.empty(n_samples, dtype=np.float32)
        VAL_CHUNK = 50000
        with torch.no_grad():
            # Value predictions for all states
            for ci in range(0, len(states), VAL_CHUNK):
                end = min(ci + VAL_CHUNK, len(states))
                chunk_gpu = torch.as_tensor(states[ci:end],
                                             dtype=torch.float32, device=self.device)
                val_preds[ci:end] = self.ppo_learner.value_net(chunk_gpu).cpu().flatten().numpy()
            # Bootstrap value (single sample)
            boot_gpu = torch.as_tensor(next_states[-1:],
                                        dtype=torch.float32, device=self.device)
            val_preds[-1] = self.ppo_learner.value_net(boot_gpu).cpu().item()

        rewards_arr = np.array(rewards)
        rewards_std = rewards_arr.std()
        if rewards_std > 1e-6:
            rewards_norm = (rewards_arr - rewards_arr.mean()) / (rewards_std + 1e-8)
            rewards_norm = np.clip(rewards_norm, -10.0, 10.0)
        else:
            rewards_norm = rewards_arr - rewards_arr.mean()

        vt, adv, _ = rlgym_util.torch_functions.compute_gae(
            rewards_norm, dones, truncated, np.nan_to_num(val_preds))

        # 6. Experience buffer
        t_buf = time.time()
        buffer = rlgym_ppo.ppo.ExperienceBuffer(
            max_size=self.ts_per_iteration + 20000, device="cpu", seed=123)
        buffer.submit_experience(
            states, actions, log_probs, rewards_norm, next_states, dones, truncated, vt, adv)
        mean_reward = float(rewards_arr.mean())
        del states, actions, log_probs, rewards_norm, next_states, dones, truncated, vt, adv
        del rewards_arr, rewards_std, val_preds, exp
        t_buf_end = time.time()

        # Single cleanup pass (not 3x per iter)
        gc.collect()
        torch.cuda.empty_cache()

        t_overhead = time.time()
        print(f"[*] Overhead: sync={t_gae-t_sync:.2f}s  GAE={t_buf-t_gae:.2f}s  "
              f"buf={t_buf_end-t_buf:.2f}s  gc={t_overhead-t_buf_end:.2f}s")

        # 7. Launch training in background
        self.epoch += 1
        meta = {
            'epoch': self.epoch,
            'steps_gained': steps_gained,
            't_coll': t_coll,
            't_wait': t_wait,
            'mean_reward': mean_reward,
            'cumul_steps': self.agent.cumulative_timesteps,
            'iter_start': iter_start,
        }
        self._launch_training(buffer, meta)

    def save_stage_checkpoint(self, stage, epoch):
        stage_dir = os.path.join(self.checkpoints_folder, "stage_backups")
        os.makedirs(stage_dir, exist_ok=True)
        label = f"stage_{stage}_epoch_{epoch}"
        path = os.path.join(stage_dir, label)
        os.makedirs(path, exist_ok=True)
        self.ppo_learner.save_to(path)
        with open(os.path.join(path, "VARS.json"), "w") as f:
            json.dump({"steps": self.agent.cumulative_timesteps, "epoch": self.epoch, "stage": stage}, f)

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
            shutil.rmtree(os.path.join(self.checkpoints_folder, oldest))

    def load_latest_checkpoint(self):
        """Load from GPU checkpoints first, then fall back to CPU checkpoints."""
        for folder in [self.checkpoints_folder, "Checkpoints_2v2"]:
            if not os.path.exists(folder):
                continue
            potential = [d for d in os.listdir(folder)
                         if os.path.isdir(os.path.join(folder, d)) and d != "stage_backups"]
            if potential:
                latest = sorted(potential, key=lambda d: os.path.getmtime(
                    os.path.join(folder, d)))[-1]
                path = os.path.join(folder, latest)
                if os.path.exists(os.path.join(path, "VARS.json")):
                    self.ppo_learner.load_from(path)
                    with open(os.path.join(path, "VARS.json")) as f:
                        v = json.load(f)
                        self.agent.cumulative_timesteps = v["steps"]
                        self.epoch = v["epoch"]
                    print(f"[*] Loaded checkpoint: {path} "
                          f"(epoch {self.epoch}, {self.agent.cumulative_timesteps:,} steps)")
                    return
        print("[*] No checkpoint found — starting fresh")


if __name__ == "__main__":
    # ── Configuration ──
    # Start conservative: 10,000 envs on RTX 2060 (6GB VRAM)
    # Increase to 25,000 or 50,000 once VRAM usage is confirmed
    N_ENVS = 10000
    TS_PER_ITERATION = 200000

    print(f"\n[*] LUCIFER GPU SIMULATOR")
    print(f"[*] Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"[*] VRAM: {vram_total:.0f} MB")
    print(f"[*] Envs: {N_ENVS:,}")
    print(f"[*] Steps/iter: {TS_PER_ITERATION:,}")

    try:
        learner = GPULearner(n_envs=N_ENVS, ts_per_iteration=TS_PER_ITERATION)
        while True:
            learner.learn()
    except KeyboardInterrupt:
        print("\n[!] MANUAL STOP.")
        if 'learner' in locals():
            try:
                prev_report, prev_meta = learner._wait_for_training()
                if prev_meta is not None:
                    prev_meta['wall_time'] = time.time() - prev_meta['iter_start']
                    learner._finalize_prev_iteration(prev_report, prev_meta)
            except Exception:
                pass
            label = f"lucifer_gpu_manual_epoch_{learner.epoch}"
            learner.save_checkpoint(label)
            save_curriculum_state(learner.curriculum.to_state_dict())
            print(f"[*] Checkpoint saved: {label}")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        if 'learner' in locals():
            try: learner._wait_for_training()
            except: pass
        sys.stdout.flush()
        print("[*] BYE.\n")
