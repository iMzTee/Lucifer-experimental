"""observations.py — GPU observation builder producing (E*A, 127) tensor.

Supports variable n_agents with zero-padding for absent player slots.
Format: ball(9) + prev_action(8) + pads(34) + self(19) + ally(19) + enemy0(19) + enemy1(19) = 127

Absent slots (ally in 1v0/1v1, enemy1 in 1v0/1v1, enemy0 in 1v0) are zeros.
"""

import torch
from .constants import (
    POS_COEF, LIN_VEL_COEF, ANG_VEL_COEF,
    N_BOOST_PADS,
)


def build_obs_batch(state, prev_actions):
    """Build observations for all envs/agents in batched PyTorch.

    Args:
        state: TensorState (with layout)
        prev_actions: (E, A, 8) previous actions per agent

    Returns: (E*A, 127) observation tensor on GPU.
    """
    E = state.n_envs
    A = state.n_agents
    device = state.device
    layout = state.layout

    # Zero-init: absent slots stay zero
    obs = torch.zeros(E, A, 127, device=device)

    # ── Inversion for orange team ──
    inv = torch.tensor([-1.0, -1.0, 1.0], device=device)

    # Normal ball
    ball_pos_n = state.ball_pos * POS_COEF
    ball_vel_n = state.ball_vel * LIN_VEL_COEF
    ball_ang_n = state.ball_ang_vel * ANG_VEL_COEF

    # Inverted ball (orange view)
    ball_pos_i = state.ball_pos * inv * POS_COEF
    ball_vel_i = state.ball_vel * inv * LIN_VEL_COEF
    ball_ang_i = state.ball_ang_vel * inv * ANG_VEL_COEF

    # ── Ball section (0-8) ──
    for i in range(A):
        if layout["is_orange"][i]:
            obs[:, i, 0:3] = ball_pos_i
            obs[:, i, 3:6] = ball_vel_i
            obs[:, i, 6:9] = ball_ang_i
        else:
            obs[:, i, 0:3] = ball_pos_n
            obs[:, i, 3:6] = ball_vel_n
            obs[:, i, 6:9] = ball_ang_n

    # ── Previous actions (9-16) ──
    obs[:, :, 9:17] = prev_actions

    # ── Boost pads (17-50) ──
    pad_avail = (state.boost_pad_timers <= 0).float()  # (E, 34)
    inv_pad_avail = pad_avail.flip(dims=[1])

    for i in range(A):
        if layout["is_orange"][i]:
            obs[:, i, 17:51] = inv_pad_avail
        else:
            obs[:, i, 17:51] = pad_avail

    # ── Player blocks helper ──
    def fill_block(agent_idx, block_start, source_car_idx, invert):
        """Fill a 19-element player block at obs[:, agent_idx, block_start:block_start+19]."""
        si = source_car_idx
        if invert:
            obs[:, agent_idx, block_start:block_start+3] = state.car_pos[:, si] * inv * POS_COEF
            obs[:, agent_idx, block_start+3:block_start+6] = state.car_fwd[:, si] * inv
            obs[:, agent_idx, block_start+6:block_start+9] = state.car_up[:, si] * inv
            obs[:, agent_idx, block_start+9:block_start+12] = state.car_vel[:, si] * inv * LIN_VEL_COEF
            obs[:, agent_idx, block_start+12:block_start+15] = state.car_ang_vel[:, si] * inv * ANG_VEL_COEF
        else:
            obs[:, agent_idx, block_start:block_start+3] = state.car_pos[:, si] * POS_COEF
            obs[:, agent_idx, block_start+3:block_start+6] = state.car_fwd[:, si]
            obs[:, agent_idx, block_start+6:block_start+9] = state.car_up[:, si]
            obs[:, agent_idx, block_start+9:block_start+12] = state.car_vel[:, si] * LIN_VEL_COEF
            obs[:, agent_idx, block_start+12:block_start+15] = state.car_ang_vel[:, si] * ANG_VEL_COEF
        obs[:, agent_idx, block_start+15] = state.car_boost[:, si]
        obs[:, agent_idx, block_start+16] = state.car_on_ground[:, si]
        obs[:, agent_idx, block_start+17] = state.car_has_flip[:, si]
        obs[:, agent_idx, block_start+18] = state.car_is_demoed[:, si]

    # ── Self (51-69), Ally (70-88), Enemy0 (89-107), Enemy1 (108-126) ──
    for i in range(A):
        is_orange = layout["is_orange"][i]

        # Self block (always filled)
        fill_block(i, 51, i, is_orange)

        # Ally block (only in 2v2)
        ally = layout["ally_idx"][i]
        if ally >= 0:
            fill_block(i, 70, ally, is_orange)
        # else: stays zero

        # Enemy0 block (in 1v1 and 2v2)
        e0 = layout["enemy0_idx"][i]
        if e0 >= 0:
            fill_block(i, 89, e0, is_orange)
        # else: stays zero

        # Enemy1 block (only in 2v2)
        e1 = layout["enemy1_idx"][i]
        if e1 >= 0:
            fill_block(i, 108, e1, is_orange)
        # else: stays zero

    return obs.reshape(E * A, 127)
