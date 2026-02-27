"""Quick profiler for vectorized collector phases."""
import os, sys, time
import numpy as np
import torch

# Silence rlgym_sim warnings
import warnings
warnings.filterwarnings("ignore")

# Import everything needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lucifer2 import env_factory, load_curriculum_state
from vectorized_collector import VectorizedCollector
import rlgym_ppo.ppo
import RocketSim as rsim

N_ENVS = 40
N_STEPS = 10000  # small run for profiling
N_ROUNDS = N_STEPS // (N_ENVS * 4)  # rounds to collect

print(f"Profiling {N_ROUNDS} rounds ({N_STEPS} steps) with {N_ENVS} envs...")

# Create collector
collector = VectorizedCollector(build_env_fn=env_factory, n_envs=N_ENVS)
shapes = collector.shapes

# Create a minimal policy
learner = rlgym_ppo.ppo.PPOLearner(
    obs_space_size=np.prod(shapes[0]),
    act_space_size=shapes[1],
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=5000, mini_batch_size=5000, n_epochs=1,
    policy_type=shapes[2],
    policy_layer_sizes=(2048, 2048, 1024, 1024),
    critic_layer_sizes=(2048, 2048, 1024, 1024),
    continuous_var_range=0.1, policy_lr=1e-4, critic_lr=4e-4,
    clip_range=0.2, ent_coef=0.01,
)
collector.policy = learner.policy

# Phase timers
t_flatten = 0; t_normalize = 0; t_inference = 0; t_parse = 0
t_multi1 = 0; t_build_gs = 0; t_multi2 = 0
t_obs = 0; t_rewards = 0; t_done_check = 0; t_record = 0; t_reset = 0

n_ag = collector.n_agents_per_env
n_resets = 0

print("Running profiled collection...")
t_total = time.perf_counter()

for round_num in range(N_ROUNDS):
    # Phase 1a: Flatten obs
    t = time.perf_counter()
    obs_flat = np.empty((collector.total_agents, collector.obs_size), dtype=np.float32)
    idx = 0
    for env_obs in collector._current_obs:
        for agent_obs in env_obs:
            obs_flat[idx] = agent_obs
            idx += 1
    t_flatten += time.perf_counter() - t

    # Phase 1b: Normalize
    t = time.perf_counter()
    obs_norm = collector._normalize_obs_batch(obs_flat)
    t_normalize += time.perf_counter() - t

    # Phase 1c: GPU inference
    t = time.perf_counter()
    with torch.no_grad():
        actions_t, log_probs_t = collector.policy.get_action(obs_norm)
    actions_np = actions_t.cpu().numpy() if actions_t.is_cuda else actions_t.numpy()
    log_probs_np = log_probs_t.cpu().numpy() if log_probs_t.is_cuda else log_probs_t.numpy()
    t_inference += time.perf_counter() - t

    # Phase 2-3: Parse actions + set controls
    t = time.perf_counter()
    agent_idx = 0
    for i in range(N_ENVS):
        env_actions = actions_np[agent_idx:agent_idx + n_ag]
        formatted = collector._matches[i].format_actions(
            collector._matches[i].parse_actions(env_actions, collector.envs[i]._prev_state))
        collector._games[i]._set_controls(formatted)
        agent_idx += n_ag
    t_parse += time.perf_counter() - t

    # Phase 4: multi_step first tick
    t = time.perf_counter()
    rsim.Arena.multi_step(collector._arenas, 1)
    t_multi1 += time.perf_counter() - t

    # Phase 5: Build gamestates
    t = time.perf_counter()
    game_states = []
    for game in collector._games:
        game_states.append(game._build_gamestate())
    t_build_gs += time.perf_counter() - t

    # Phase 6: multi_step remaining ticks
    t = time.perf_counter()
    if collector.tick_skip > 1:
        rsim.Arena.multi_step(collector._arenas, collector.tick_skip - 1)
    t_multi2 += time.perf_counter() - t

    # Phase 7: Obs + rewards + done
    done_envs = []
    for i in range(N_ENVS):
        state = game_states[i]
        collector.envs[i]._prev_state = state

        t = time.perf_counter()
        obs_list = collector._matches[i].build_observations(state)
        t_obs += time.perf_counter() - t

        t = time.perf_counter()
        done = collector._matches[i].is_done(state)
        t_done_check += time.perf_counter() - t

        t = time.perf_counter()
        rewards = collector._matches[i].get_rewards(state, done)
        t_rewards += time.perf_counter() - t

        if not isinstance(obs_list, list): obs_list = [obs_list]
        if not isinstance(rewards, list): rewards = [rewards]

        if done:
            done_envs.append(i)
            collector._current_obs[i] = None
        else:
            collector._current_obs[i] = obs_list

    # Phase 8: Resets
    t = time.perf_counter()
    for i in done_envs:
        obs_reset = collector.envs[i].reset()
        obs_reset = obs_reset if isinstance(obs_reset, list) else [obs_reset]
        collector._current_obs[i] = obs_reset
        n_resets += 1
    t_reset += time.perf_counter() - t

t_total = time.perf_counter() - t_total

print(f"\n{'='*55}")
print(f"  PROFILE: {N_ROUNDS} rounds, {N_ROUNDS * collector.total_agents} steps")
print(f"  Total time: {t_total:.3f}s")
print(f"  Resets: {n_resets}")
print(f"{'='*55}")
print(f"  {'Phase':<30} {'Time':>8} {'%':>6}")
print(f"  {'-'*46}")
print(f"  {'1a. Flatten obs':<30} {t_flatten:>7.3f}s {100*t_flatten/t_total:>5.1f}%")
print(f"  {'1b. Normalize obs':<30} {t_normalize:>7.3f}s {100*t_normalize/t_total:>5.1f}%")
print(f"  {'1c. GPU inference':<30} {t_inference:>7.3f}s {100*t_inference/t_total:>5.1f}%")
print(f"  {'2-3. Parse + set controls':<30} {t_parse:>7.3f}s {100*t_parse/t_total:>5.1f}%")
print(f"  {'4. multi_step(1)':<30} {t_multi1:>7.3f}s {100*t_multi1/t_total:>5.1f}%")
print(f"  {'5. Build gamestates':<30} {t_build_gs:>7.3f}s {100*t_build_gs/t_total:>5.1f}%")
print(f"  {'6. multi_step(tick_skip-1)':<30} {t_multi2:>7.3f}s {100*t_multi2/t_total:>5.1f}%")
print(f"  {'7a. build_observations':<30} {t_obs:>7.3f}s {100*t_obs/t_total:>5.1f}%")
print(f"  {'7b. is_done':<30} {t_done_check:>7.3f}s {100*t_done_check/t_total:>5.1f}%")
print(f"  {'7c. get_rewards':<30} {t_rewards:>7.3f}s {100*t_rewards/t_total:>5.1f}%")
print(f"  {'8. Resets':<30} {t_reset:>7.3f}s {100*t_reset/t_total:>5.1f}%")

t_python = t_flatten + t_normalize + t_parse + t_build_gs + t_obs + t_done_check + t_rewards + t_record + t_reset
t_gpu = t_inference
t_cpp = t_multi1 + t_multi2
print(f"  {'-'*46}")
print(f"  {'TOTAL Python-side':<30} {t_python:>7.3f}s {100*t_python/t_total:>5.1f}%")
print(f"  {'TOTAL GPU inference':<30} {t_gpu:>7.3f}s {100*t_gpu/t_total:>5.1f}%")
print(f"  {'TOTAL C++ physics':<30} {t_cpp:>7.3f}s {100*t_cpp/t_total:>5.1f}%")
print(f"{'='*55}")

# Extrapolate to 100k steps
scale = 100000 / (N_ROUNDS * collector.total_agents)
print(f"\n  Projected for 100k steps: {t_total * scale:.1f}s  (SPS: {int(100000 / (t_total * scale))})")
