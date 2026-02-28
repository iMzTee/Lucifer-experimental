"""observations.py — GPU observation builder producing (E*4, 127) tensor.

Direct port of vectorized_env.py build_obs_batch from numpy to PyTorch.
Includes inverted views for orange team (negate x and y).
"""

import torch
from .constants import (
    POS_COEF, LIN_VEL_COEF, ANG_VEL_COEF,
    ALLY_IDX, ENEMY0_IDX, ENEMY1_IDX,
    N_BOOST_PADS,
)


def build_obs_batch(state, prev_actions):
    """Build observations for all envs/agents in batched PyTorch.

    Matches DefaultObs format: ball(9) + prev_action(8) + pads(34) +
    self(19) + ally(19) + enemy0(19) + enemy1(19) = 127

    Args:
        state: TensorState
        prev_actions: (E, 4, 8) previous actions per agent

    Returns: (E*4, 127) observation tensor on GPU.
    """
    E = state.n_envs
    device = state.device
    obs = torch.empty(E, 4, 127, device=device)

    # ── Inversion for orange team ──
    # Orange agents see the field mirrored: negate x, y (keep z)
    inv = torch.tensor([-1.0, -1.0, 1.0], device=device)

    # Normal ball
    ball_pos_n = state.ball_pos * POS_COEF          # (E, 3)
    ball_vel_n = state.ball_vel * LIN_VEL_COEF
    ball_ang_n = state.ball_ang_vel * ANG_VEL_COEF

    # Inverted ball (orange view)
    ball_pos_i = state.ball_pos * inv * POS_COEF
    ball_vel_i = state.ball_vel * inv * LIN_VEL_COEF
    ball_ang_i = state.ball_ang_vel * inv * ANG_VEL_COEF

    # ── Ball section (0-8) ──
    obs[:, :2, 0:3] = ball_pos_n.unsqueeze(1)
    obs[:, :2, 3:6] = ball_vel_n.unsqueeze(1)
    obs[:, :2, 6:9] = ball_ang_n.unsqueeze(1)
    obs[:, 2:, 0:3] = ball_pos_i.unsqueeze(1)
    obs[:, 2:, 3:6] = ball_vel_i.unsqueeze(1)
    obs[:, 2:, 6:9] = ball_ang_i.unsqueeze(1)

    # ── Previous actions (9-16) ──
    obs[:, :, 9:17] = prev_actions

    # ── Boost pads (17-50) ──
    # Normal pads for blue, inverted (reversed order) for orange
    pad_avail = (state.boost_pad_timers <= 0).float()  # (E, 34) — 1 if available
    inv_pad_avail = pad_avail.flip(dims=[1])           # reversed order for orange

    obs[:, :2, 17:51] = pad_avail.unsqueeze(1)
    obs[:, 2:, 17:51] = inv_pad_avail.unsqueeze(1)

    # ── Player blocks: helper function ──
    def fill_block(target, pos, fwd, up, vel, ang, boost, ground, flip, demoed):
        """Fill a 19-element player block. target: (E, N, 19)."""
        target[:, :, 0:3] = pos * POS_COEF
        target[:, :, 3:6] = fwd
        target[:, :, 6:9] = up
        target[:, :, 9:12] = vel * LIN_VEL_COEF
        target[:, :, 12:15] = ang * ANG_VEL_COEF
        target[:, :, 15] = boost
        target[:, :, 16] = ground
        target[:, :, 17] = flip
        target[:, :, 18] = demoed

    # Normal and inverted player state
    pp = state.car_pos          # (E, 4, 3)
    pf = state.car_fwd
    pu = state.car_up
    pv = state.car_vel
    pa = state.car_ang_vel
    pb = state.car_boost        # (E, 4)
    pg = state.car_on_ground
    pfl = state.car_has_flip
    pd = state.car_is_demoed

    # Inverted positions/velocities for orange
    ip = pp * inv
    if_ = pf * inv
    iu = pu * inv
    iv = pv * inv
    ia = pa * inv

    # ── Self (51-69) ──
    # Blue agents (0,1): see normal
    fill_block(obs[:, :2, 51:70], pp[:, :2], pf[:, :2], pu[:, :2],
               pv[:, :2], pa[:, :2], pb[:, :2], pg[:, :2], pfl[:, :2], pd[:, :2])
    # Orange agents (2,3): see inverted
    fill_block(obs[:, 2:, 51:70], ip[:, 2:], if_[:, 2:], iu[:, 2:],
               iv[:, 2:], ia[:, 2:], pb[:, 2:], pg[:, 2:], pfl[:, 2:], pd[:, 2:])

    # ── Ally (70-88): ALLY_IDX = [1, 0, 3, 2] ──
    fill_block(obs[:, :2, 70:89],
               pp[:, ALLY_IDX[:2]], pf[:, ALLY_IDX[:2]], pu[:, ALLY_IDX[:2]],
               pv[:, ALLY_IDX[:2]], pa[:, ALLY_IDX[:2]],
               pb[:, ALLY_IDX[:2]], pg[:, ALLY_IDX[:2]], pfl[:, ALLY_IDX[:2]], pd[:, ALLY_IDX[:2]])
    fill_block(obs[:, 2:, 70:89],
               ip[:, ALLY_IDX[2:]], if_[:, ALLY_IDX[2:]], iu[:, ALLY_IDX[2:]],
               iv[:, ALLY_IDX[2:]], ia[:, ALLY_IDX[2:]],
               pb[:, ALLY_IDX[2:]], pg[:, ALLY_IDX[2:]], pfl[:, ALLY_IDX[2:]], pd[:, ALLY_IDX[2:]])

    # ── Enemy 0 (89-107): ENEMY0_IDX = [2, 2, 0, 0] ──
    fill_block(obs[:, :2, 89:108],
               pp[:, ENEMY0_IDX[:2]], pf[:, ENEMY0_IDX[:2]], pu[:, ENEMY0_IDX[:2]],
               pv[:, ENEMY0_IDX[:2]], pa[:, ENEMY0_IDX[:2]],
               pb[:, ENEMY0_IDX[:2]], pg[:, ENEMY0_IDX[:2]], pfl[:, ENEMY0_IDX[:2]], pd[:, ENEMY0_IDX[:2]])
    fill_block(obs[:, 2:, 89:108],
               ip[:, ENEMY0_IDX[2:]], if_[:, ENEMY0_IDX[2:]], iu[:, ENEMY0_IDX[2:]],
               iv[:, ENEMY0_IDX[2:]], ia[:, ENEMY0_IDX[2:]],
               pb[:, ENEMY0_IDX[2:]], pg[:, ENEMY0_IDX[2:]], pfl[:, ENEMY0_IDX[2:]], pd[:, ENEMY0_IDX[2:]])

    # ── Enemy 1 (108-126): ENEMY1_IDX = [3, 3, 1, 1] ──
    fill_block(obs[:, :2, 108:127],
               pp[:, ENEMY1_IDX[:2]], pf[:, ENEMY1_IDX[:2]], pu[:, ENEMY1_IDX[:2]],
               pv[:, ENEMY1_IDX[:2]], pa[:, ENEMY1_IDX[:2]],
               pb[:, ENEMY1_IDX[:2]], pg[:, ENEMY1_IDX[:2]], pfl[:, ENEMY1_IDX[:2]], pd[:, ENEMY1_IDX[:2]])
    fill_block(obs[:, 2:, 108:127],
               ip[:, ENEMY1_IDX[2:]], if_[:, ENEMY1_IDX[2:]], iu[:, ENEMY1_IDX[2:]],
               iv[:, ENEMY1_IDX[2:]], ia[:, ENEMY1_IDX[2:]],
               pb[:, ENEMY1_IDX[2:]], pg[:, ENEMY1_IDX[2:]], pfl[:, ENEMY1_IDX[2:]], pd[:, ENEMY1_IDX[2:]])

    return obs.reshape(E * 4, 127)
