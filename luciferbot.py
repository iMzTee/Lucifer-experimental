"""LuciferBot — GPU-accelerated Rocket League bot training.

GPU-resident pipeline: physics sim, reward computation, observation building,
policy inference, GAE, and PPO training all stay on GPU end-to-end.
"""

import os
import time
import json
import gc
import torch
import numpy as np
import sys
import psutil
import threading
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from gpu_sim.collector import GPUCollector
from gpu_sim.constants import STAGE_CONFIG
from gpu_sim.ppo import PPOLearner, ExperienceBuffer
from gpu_sim.vis_sender import VisSender


# ─── Curriculum state persistence ───
CURRICULUM_STATE_FILE = "curriculum_state.json"
CURRICULUM_STAGE_NAMES = ["Ground Basics", "Ground Advanced", "Air Mechanics", "1v1 Basics", "1v1 Advanced", "2v2 Teamwork"]

DEFAULT_CURRICULUM_STATE = {
    "stage": 0,
    "policy_lr": 3e-4,
    "critic_lr": 3e-4,
    "ent_coef": 0.01,
    "clip_high_count": 0,
    "val_loss_high_count": 0,
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


# ─── CurriculumTracker ───
class CurriculumTracker:
    # Adjusted for 5-bin action space (max entropy ~10.1 vs old ~7.6)
    ENTROPY_ADVANCE     = [0.7, 0.7, 0.7, 0.7, 0.7]
    ENTROPY_CONSECUTIVE = 3
    MIN_STAGE_ITERS     = [500, 500, 1000, 2000, 2500]
    CLIP_HIGH_THRESH    = 0.25
    CLIP_HIGH_ITERS     = 5
    VAL_LOSS_HIGH_THRESH = 10.0
    VAL_LOSS_HIGH_ITERS  = 5

    def __init__(self, state):
        self.stage              = state["stage"]
        self.policy_lr          = state["policy_lr"]
        self.critic_lr          = state["critic_lr"]
        self.ent_coef           = state["ent_coef"]
        self.clip_high_count    = state.get("clip_high_count", 0)
        self.val_loss_high_count = state.get("val_loss_high_count", 0)
        self.stage_iter_count   = state.get("stage_iter_count", 0)
        self.entropy_low_count  = state.get("entropy_low_count", 0)
        self.restart_requested  = False
        self.restart_reason     = ""

    def to_state_dict(self):
        return {
            "stage": self.stage, "policy_lr": self.policy_lr,
            "critic_lr": self.critic_lr, "ent_coef": self.ent_coef,
            "clip_high_count": self.clip_high_count,
            "val_loss_high_count": self.val_loss_high_count,
            "stage_iter_count": self.stage_iter_count,
            "entropy_low_count": self.entropy_low_count,
        }

    def update(self, mean_reward, entropy, clip_fraction, value_loss):
        self.stage_iter_count += 1

        # ── Policy LR decay: clip fraction too high ──
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

        # ── Critic LR decay: value loss too high ──
        if value_loss > self.VAL_LOSS_HIGH_THRESH:
            self.val_loss_high_count += 1
        else:
            self.val_loss_high_count = 0

        if self.val_loss_high_count >= self.VAL_LOSS_HIGH_ITERS:
            new_lr = round(self.critic_lr * 0.80, 7)
            if new_lr >= 1e-4:
                self.critic_lr = new_lr
                self.val_loss_high_count = 0
                self.restart_reason = (
                    f"Value loss >{self.VAL_LOSS_HIGH_THRESH} for "
                    f"{self.VAL_LOSS_HIGH_ITERS} iters — critic_lr → {self.critic_lr:.2e}")
                self.restart_requested = True
                return True

        # ── Stage advance check ──
        if self.stage < 5:
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
                self.clip_high_count = 0
                self.val_loss_high_count = 0
                self.policy_lr = 3e-4
                self.critic_lr = 3e-4
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
    def __init__(self, ts_per_iteration=200000):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        assert self.device == "cuda", "GPU sim requires CUDA!"
        self.checkpoints_folder = "Checkpoints_LuciferBot"
        self.epoch = 0

        curriculum_state = load_curriculum_state()
        self.curriculum = CurriculumTracker(curriculum_state)
        self.ts_per_iteration = ts_per_iteration

        # Read n_envs from stage config
        stage = self.curriculum.stage
        cfg = STAGE_CONFIG.get(stage, STAGE_CONFIG[0])
        n_envs = cfg.get("n_envs", 40000)
        n_agents = cfg.get("n_agents", 4)

        stage_name = CURRICULUM_STAGE_NAMES[stage]
        print(f"\n{'='*60}")
        print(f"  LuciferBot — Stage {stage}: {stage_name}")
        print(f"  {n_envs:,} envs x {n_agents} agents on GPU")
        print(f"  Steps per iteration: {ts_per_iteration:,}")
        print(f"{'='*60}")

        # Visualization sender (zero overhead when disabled)
        vis_enabled = os.environ.get("VIS", "0") == "1"
        vis_sender = VisSender(env_idx=0, enabled=vis_enabled) if vis_enabled else None
        if vis_enabled:
            print("[*] Visualization enabled — sending env 0 to RocketSimVis (UDP 127.0.0.1:9273)")
            print("[*] Start viewer: cd rocketsimvis && python main.py")

        # GPU Collector
        self.agent = GPUCollector(
            n_envs=n_envs, device=self.device,
            standardize_obs=True, stage=stage, vis_sender=vis_sender)

        # PPO Learner
        self.ppo_learner = PPOLearner(
            obs_space_size=127,
            act_space_size=8,
            device=self.device,
            batch_size=50000,
            mini_batch_size=50000,
            n_epochs=2,
            policy_layer_sizes=(1024, 1024, 1024, 512),
            critic_layer_sizes=(1024, 1024, 1024, 512),
            policy_lr=self.curriculum.policy_lr,
            critic_lr=self.curriculum.critic_lr,
            clip_range=0.2,
            ent_coef=self.curriculum.ent_coef,
        )
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
        print(f"  LuciferBot — Stage {stage} ({stage_name}) — Iter {epoch}")
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
            # Update collector stage (handles n_agents/n_envs re-init)
            self.agent.set_stage(self.curriculum.stage)
            # Re-sync frozen policy after stage change
            self._frozen_policy.load_state_dict(self.ppo_learner.policy.state_dict())
            self.agent.policy = self._frozen_policy
            self.curriculum.restart_requested = False

        if epoch % 50 == 0:
            self.save_checkpoint(f"luciferbot_epoch_{epoch}")

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

        # 5. GPU GAE — value predictions + advantage estimation on GPU
        t_gae = time.time()
        states, actions, log_probs, rewards, next_states, dones, truncated = exp
        n_steps = states.shape[0]
        n_agents = self.agent.total_agents
        n_rounds = n_steps // n_agents
        VAL_CHUNK = 50000

        with torch.no_grad(), torch.amp.autocast('cuda'):
            # Value predictions for all states (stay on GPU)
            val_states = torch.empty(n_steps, device='cuda')
            for ci in range(0, n_steps, VAL_CHUNK):
                end = min(ci + VAL_CHUNK, n_steps)
                val_states[ci:end] = self.ppo_learner.value_net(states[ci:end]).flatten()
            # Bootstrap values for last round's next states
            boot_next = next_states[-n_agents:]
            boot_vals = torch.empty(n_agents, device='cuda')
            for ci in range(0, n_agents, VAL_CHUNK):
                end = min(ci + VAL_CHUNK, n_agents)
                boot_vals[ci:end] = self.ppo_learner.value_net(boot_next[ci:end]).flatten()

        # All values as float32: (n_rounds+1, n_agents)
        values = torch.cat([val_states, boot_vals]).float().reshape(n_rounds + 1, n_agents)
        values = torch.nan_to_num(values)
        del val_states, boot_vals

        # Reward normalization on GPU
        mean_reward = float(rewards.mean().item())
        rewards_std = rewards.std()
        if rewards_std > 1e-6:
            rewards_norm = ((rewards - rewards.mean()) / (rewards_std + 1e-8)).clamp(-10.0, 10.0)
        else:
            rewards_norm = rewards - rewards.mean()

        # Reshape experience for vectorized GAE
        r = rewards_norm.reshape(n_rounds, n_agents)
        d = dones.reshape(n_rounds, n_agents)
        tr = truncated.reshape(n_rounds, n_agents)

        # Vectorized GAE: n_rounds iterations instead of 200k
        gamma, lam = 0.99, 0.95
        gae = torch.zeros(n_agents, device='cuda')
        advantages = torch.empty(n_rounds, n_agents, device='cuda')
        for t in reversed(range(n_rounds)):
            not_done = 1.0 - d[t]
            continuation = not_done * (1.0 - tr[t])
            delta = r[t] + gamma * values[t + 1] * not_done - values[t]
            gae = delta + gamma * lam * continuation * gae
            advantages[t] = gae

        vt = (values[:n_rounds] + advantages).reshape(-1)
        adv = advantages.reshape(-1)
        del values, r, d, tr, gae, advantages

        # 6. GPU experience buffer (no CPU transfer)
        t_buf = time.time()
        buffer = ExperienceBuffer(
            max_size=self.ts_per_iteration + 20000, device="cuda", seed=123)
        buffer.submit_experience(
            states, actions, log_probs, rewards_norm, next_states, dones, truncated, vt, adv)
        del states, actions, log_probs, rewards, rewards_norm, next_states, dones, truncated, vt, adv, exp
        t_buf_end = time.time()

        # GC only every 10 iterations
        if self.epoch % 10 == 0:
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
        for folder in [self.checkpoints_folder, "Checkpoints_2v2_gpu", "Checkpoints_2v2"]:
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
    # n_envs is read from STAGE_CONFIG per stage:
    #   Stage 0 (1v0): 160k envs x 1 agent  — Ground Basics
    #   Stage 1 (1v0): 160k envs x 1 agent  — Ground Advanced
    #   Stage 2 (1v0): 160k envs x 1 agent  — Air Mechanics
    #   Stage 3 (1v1): 80k envs x 2 agents  — 1v1 Basics
    #   Stage 4 (1v1): 80k envs x 2 agents  — 1v1 Advanced
    #   Stage 5 (2v2): 40k envs x 4 agents  — 2v2 Teamwork
    TS_PER_ITERATION = 200000

    print(f"\n[*] LuciferBot")
    print(f"[*] Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"[*] VRAM: {vram_total:.0f} MB")
    print(f"[*] Steps/iter: {TS_PER_ITERATION:,}")

    try:
        learner = GPULearner(ts_per_iteration=TS_PER_ITERATION)
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
            label = f"luciferbot_manual_epoch_{learner.epoch}"
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
