"""Profile optimized vectorized collector with raw extraction."""
import os, sys, time
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lucifer2 import env_factory, load_curriculum_state
from vectorized_collector import VectorizedCollector
from vectorized_env import build_obs_batch, EVENT_WEIGHTS
import rlgym_ppo.ppo
import RocketSim as rsim

N_ENVS = 40
N_STEPS = 10000
N_ROUNDS = N_STEPS // (N_ENVS * 4)

print(f"Profiling {N_ROUNDS} rounds ({N_STEPS} steps) with {N_ENVS} envs...")

collector = VectorizedCollector(build_env_fn=env_factory, n_envs=N_ENVS)
shapes = collector.shapes

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
n_ag = collector.n_agents_per_env
n_total = collector.total_agents
bs = collector._batch_state
vec_rewards = collector._vec_rewards
vec_terminals = collector._vec_terminals
prev_actions = collector._prev_actions
stage = collector._stage
event_weights = collector._event_weights

# Phase timers
t_normalize = 0; t_inference = 0; t_parse = 0
t_multi1 = 0; t_raw_get = 0; t_goal_check = 0; t_multi2 = 0
t_extract_raw = 0; t_terminals = 0; t_rewards = 0; t_obs_build = 0
t_norm_next = 0; t_record = 0; t_reset = 0
n_resets = 0

print("Running profiled collection...")
t_total = time.perf_counter()

for round_num in range(N_ROUNDS):
    # Phase 1a: Normalize obs
    t = time.perf_counter()
    obs_norm = collector._normalize_obs_batch(collector._current_obs)
    t_normalize += time.perf_counter() - t

    # Phase 1b: GPU inference
    t = time.perf_counter()
    with torch.no_grad():
        actions_t, log_probs_t = collector.policy.get_action(obs_norm)
    actions_np = actions_t.cpu().numpy() if actions_t.is_cuda else actions_t.numpy()
    log_probs_np = log_probs_t.cpu().numpy() if log_probs_t.is_cuda else log_probs_t.numpy()
    t_inference += time.perf_counter() - t

    # Phase 2-3: Parse + set controls
    t = time.perf_counter()
    agent_idx = 0
    for i in range(N_ENVS):
        env_actions = actions_np[agent_idx:agent_idx + n_ag]
        parsed = collector._matches[i].parse_actions(env_actions, collector.envs[i]._prev_state)
        formatted = collector._matches[i].format_actions(parsed)
        collector._games[i]._set_controls(formatted)
        prev_actions[i] = collector._matches[i]._prev_actions[:n_ag]
        agent_idx += n_ag
    t_parse += time.perf_counter() - t

    # Phase 4: multi_step first tick
    t = time.perf_counter()
    rsim.Arena.multi_step(collector._arenas, 1)
    t_multi1 += time.perf_counter() - t

    # Phase 5a: Get raw gym state
    t = time.perf_counter()
    raw_states = [arena.get_gym_state() for arena in collector._arenas]
    t_raw_get += time.perf_counter() - t

    # Phase 5b: Goal check + ball reset
    t = time.perf_counter()
    for i, raw in enumerate(raw_states):
        bsc = int(raw[0][2])
        osc = int(raw[0][3])
        if bsc != collector._game_scores[i][0] or osc != collector._game_scores[i][1]:
            collector._game_scores[i][0] = bsc
            collector._game_scores[i][1] = osc
            collector._games[i].blue_score = bsc
            collector._games[i].orange_score = osc
            collector._arenas[i].ball.set_state(rsim.BallState())
    t_goal_check += time.perf_counter() - t

    # Phase 6: multi_step remaining ticks
    t = time.perf_counter()
    if collector.tick_skip > 1:
        rsim.Arena.multi_step(collector._arenas, collector.tick_skip - 1)
    t_multi2 += time.perf_counter() - t

    # Phase 7a: Extract raw state
    t = time.perf_counter()
    bs.extract_raw(raw_states, collector._car_orders, collector._car_objects)
    t_extract_raw += time.perf_counter() - t

    # Phase 7b: Terminal check
    t = time.perf_counter()
    dones_batch = vec_terminals.check(bs)
    t_terminals += time.perf_counter() - t

    # Phase 7c: Rewards
    t = time.perf_counter()
    rewards_batch = vec_rewards.compute(bs, stage, event_weights)
    t_rewards += time.perf_counter() - t

    # Phase 7d: Obs building
    t = time.perf_counter()
    next_obs = build_obs_batch(bs, prev_actions)
    t_obs_build += time.perf_counter() - t

    # Phase 7e: Normalize next obs
    t = time.perf_counter()
    next_norm = collector._normalize_obs_batch(next_obs)
    t_norm_next += time.perf_counter() - t

    # Phase 8: Reset done envs + update current obs
    t = time.perf_counter()
    collector._current_obs[:] = next_obs
    done_indices = np.where(dones_batch)[0]
    for i in done_indices:
        obs_reset = collector.envs[i].reset()
        obs_reset = obs_reset if isinstance(obs_reset, list) else [obs_reset]
        base = i * n_ag
        for j, o in enumerate(obs_reset):
            collector._current_obs[base + j] = o
        prev_actions[i] = 0
        collector._game_scores[i][0] = collector._games[i].blue_score
        collector._game_scores[i][1] = collector._games[i].orange_score
        bs.extract_single(i, collector.envs[i]._prev_state)
        vec_terminals.reset_env(i, bs)
        vec_rewards.reset_env(i, bs)
        n_resets += 1
    t_reset += time.perf_counter() - t

t_total = time.perf_counter() - t_total

print(f"\n{'='*60}")
print(f"  PROFILE: {N_ROUNDS} rounds, {N_ROUNDS * n_total} steps")
print(f"  Total time: {t_total:.3f}s")
print(f"  Resets: {n_resets}")
print(f"{'='*60}")
print(f"  {'Phase':<35} {'Time':>8} {'%':>6}")
print(f"  {'-'*51}")
print(f"  {'1a. Normalize obs':<35} {t_normalize:>7.3f}s {100*t_normalize/t_total:>5.1f}%")
print(f"  {'1b. GPU inference':<35} {t_inference:>7.3f}s {100*t_inference/t_total:>5.1f}%")
print(f"  {'2-3. Parse + set controls':<35} {t_parse:>7.3f}s {100*t_parse/t_total:>5.1f}%")
print(f"  {'4. multi_step(1)':<35} {t_multi1:>7.3f}s {100*t_multi1/t_total:>5.1f}%")
print(f"  {'5a. get_gym_state (raw)':<35} {t_raw_get:>7.3f}s {100*t_raw_get/t_total:>5.1f}%")
print(f"  {'5b. Goal check + ball reset':<35} {t_goal_check:>7.3f}s {100*t_goal_check/t_total:>5.1f}%")
print(f"  {'6. multi_step(tick_skip-1)':<35} {t_multi2:>7.3f}s {100*t_multi2/t_total:>5.1f}%")
print(f"  {'7a. Extract raw → BatchState':<35} {t_extract_raw:>7.3f}s {100*t_extract_raw/t_total:>5.1f}%")
print(f"  {'7b. Terminal check':<35} {t_terminals:>7.3f}s {100*t_terminals/t_total:>5.1f}%")
print(f"  {'7c. Vectorized rewards':<35} {t_rewards:>7.3f}s {100*t_rewards/t_total:>5.1f}%")
print(f"  {'7d. Vectorized obs build':<35} {t_obs_build:>7.3f}s {100*t_obs_build/t_total:>5.1f}%")
print(f"  {'7e. Normalize next obs':<35} {t_norm_next:>7.3f}s {100*t_norm_next/t_total:>5.1f}%")
print(f"  {'8. Resets + update obs':<35} {t_reset:>7.3f}s {100*t_reset/t_total:>5.1f}%")

t_python = t_normalize + t_parse + t_extract_raw + t_goal_check + t_terminals + t_rewards + t_obs_build + t_norm_next + t_reset
t_gpu = t_inference
t_cpp = t_multi1 + t_multi2 + t_raw_get
print(f"  {'-'*51}")
print(f"  {'TOTAL Python-side':<35} {t_python:>7.3f}s {100*t_python/t_total:>5.1f}%")
print(f"  {'TOTAL GPU inference':<35} {t_gpu:>7.3f}s {100*t_gpu/t_total:>5.1f}%")
print(f"  {'TOTAL C++ (physics+raw state)':<35} {t_cpp:>7.3f}s {100*t_cpp/t_total:>5.1f}%")
print(f"{'='*60}")

scale = 100000 / (N_ROUNDS * n_total)
print(f"\n  Projected for 100k steps: {t_total * scale:.1f}s  (Collection SPS: {int(100000 / (t_total * scale))})")
print(f"  With ~6.5s training: {t_total * scale + 6.5:.1f}s  (Total SPS: {int(100000 / (t_total * scale + 6.5))})")
