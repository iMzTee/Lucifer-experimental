"""Test different env counts for SPS scaling."""
import os, sys, time, warnings
import numpy as np
import torch
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lucifer2 import env_factory
from vectorized_collector import VectorizedCollector
from vectorized_env import build_obs_batch
import rlgym_ppo.ppo
import RocketSim as rsim

for N_ENVS in [60, 80]:
    print(f'\n=== Testing with {N_ENVS} envs ===')
    collector = VectorizedCollector(build_env_fn=env_factory, n_envs=N_ENVS)
    shapes = collector.shapes
    learner = rlgym_ppo.ppo.PPOLearner(
        obs_space_size=np.prod(shapes[0]), act_space_size=shapes[1],
        device='cuda' if torch.cuda.is_available() else 'cpu',
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
    N_ROUNDS = max(5, 10000 // n_total)

    t0 = time.perf_counter()
    for _ in range(N_ROUNDS):
        obs_norm = collector._normalize_obs_batch(collector._current_obs)
        with torch.no_grad():
            actions_t, log_probs_t = collector.policy.get_action(obs_norm)
        actions_np = actions_t.cpu().numpy() if actions_t.is_cuda else actions_t.numpy()

        agent_idx = 0
        for i in range(N_ENVS):
            env_actions = actions_np[agent_idx:agent_idx + n_ag]
            parsed = collector._matches[i].parse_actions(env_actions, collector.envs[i]._prev_state)
            formatted = collector._matches[i].format_actions(parsed)
            collector._games[i]._set_controls(formatted)
            collector._prev_actions[i] = collector._matches[i]._prev_actions[:n_ag]
            agent_idx += n_ag

        rsim.Arena.multi_step(collector._arenas, 1)
        raw_states = [arena.get_gym_state() for arena in collector._arenas]
        for i, raw in enumerate(raw_states):
            bsc = int(raw[0][2]); osc = int(raw[0][3])
            if bsc != collector._game_scores[i][0] or osc != collector._game_scores[i][1]:
                collector._game_scores[i] = [bsc, osc]
                collector._games[i].blue_score = bsc
                collector._games[i].orange_score = osc
                collector._arenas[i].ball.set_state(rsim.BallState())
        if collector.tick_skip > 1:
            rsim.Arena.multi_step(collector._arenas, collector.tick_skip - 1)
        collector._batch_state.extract_raw(raw_states, collector._car_orders, collector._car_objects)
        dones = collector._vec_terminals.check(collector._batch_state)
        rewards = collector._vec_rewards.compute(collector._batch_state, collector._stage, collector._event_weights)
        next_obs = build_obs_batch(collector._batch_state, collector._prev_actions)
        collector._current_obs[:] = next_obs

        for i in np.where(dones)[0]:
            obs_reset = collector.envs[i].reset()
            obs_reset = obs_reset if isinstance(obs_reset, list) else [obs_reset]
            base = i * n_ag
            for j, o in enumerate(obs_reset):
                collector._current_obs[base + j] = o
            collector._prev_actions[i] = 0
            collector._game_scores[i] = [collector._games[i].blue_score, collector._games[i].orange_score]
            collector._batch_state.extract_single(i, collector.envs[i]._prev_state)
            collector._vec_terminals.reset_env(i, collector._batch_state)
            collector._vec_rewards.reset_env(i, collector._batch_state)

    elapsed = time.perf_counter() - t0
    steps = N_ROUNDS * n_total
    scale = 100000 / steps
    projected = elapsed * scale
    print(f'  {N_ROUNDS} rounds, {steps} steps in {elapsed:.3f}s')
    print(f'  Projected 100k: {projected:.1f}s (Collection SPS: {int(100000/projected)})')
    print(f'  With 6.5s train: Total SPS: {int(100000/(projected+6.5))}')
