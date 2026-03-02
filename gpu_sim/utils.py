"""utils.py — Quaternion math and rotation helpers as batched tensor operations.

All functions operate on arbitrary batch dimensions. Convention: quat = (w, x, y, z).
Uses the rlgym_sim negated quaternion convention where needed.
"""

import torch
import math


def quat_normalize(q):
    """Normalize quaternion(s). q: (..., 4) → (..., 4)."""
    return q / (q.norm(dim=-1, keepdim=True) + 1e-8)


def quat_multiply(q1, q2):
    """Hamilton product of two quaternions. q1, q2: (..., 4) → (..., 4).

    Convention: (w, x, y, z).
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def quat_conjugate(q):
    """Quaternion conjugate (inverse for unit quaternions). q: (..., 4) → (..., 4)."""
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)


def quat_rotate_vector(q, v):
    """Rotate vector(s) by quaternion(s). q: (..., 4), v: (..., 3) → (..., 3).

    Uses q * v_quat * q_conj where v_quat = (0, vx, vy, vz).
    """
    v_quat = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    q_conj = quat_conjugate(q)
    rotated = quat_multiply(quat_multiply(q, v_quat), q_conj)
    return rotated[..., 1:]


def quat_from_axis_angle(axis, angle):
    """Create quaternion from axis-angle. axis: (..., 3), angle: (...) → (..., 4).

    axis should be normalized.
    """
    half = angle.unsqueeze(-1) * 0.5
    sin_half = torch.sin(half)
    cos_half = torch.cos(half)
    return torch.cat([cos_half, axis * sin_half], dim=-1)


def quat_integrate(q, ang_vel, dt):
    """Integrate quaternion by angular velocity over dt.

    q: (..., 4), ang_vel: (..., 3), dt: float → (..., 4).
    Uses first-order approximation: q_new = q + 0.5 * dt * omega_quat * q.
    """
    # Convert angular velocity to quaternion derivative
    omega_quat = torch.cat([
        torch.zeros_like(ang_vel[..., :1]),
        ang_vel
    ], dim=-1)

    # dq = 0.5 * omega_quat * q
    dq = 0.5 * quat_multiply(omega_quat, q)
    q_new = q + dq * dt
    return quat_normalize(q_new)


def quat_to_fwd_up(q):
    """Extract forward (+x) and up (+z) vectors from quaternion(s).

    Uses rlgym_sim convention: negate all components, then compute rotation matrix.
    q: (..., 4) → (fwd: (..., 3), up: (..., 3))
    """
    # rlgym_sim negates quaternion before computing rotation matrix
    w = -q[..., 0]
    x = -q[..., 1]
    y = -q[..., 2]
    z = -q[..., 3]

    norm_sq = w*w + x*x + y*y + z*z
    s = torch.where(norm_sq > 0, 1.0 / norm_sq, torch.zeros_like(norm_sq))

    # Forward = first column of rotation matrix
    fwd_x = 1.0 - 2.0 * s * (y*y + z*z)
    fwd_y = 2.0 * s * (x*y + z*w)
    fwd_z = 2.0 * s * (x*z - y*w)
    fwd = torch.stack([fwd_x, fwd_y, fwd_z], dim=-1)

    # Up = third column of rotation matrix
    up_x = 2.0 * s * (x*z + y*w)
    up_y = 2.0 * s * (y*z - x*w)
    up_z = 1.0 - 2.0 * s * (x*x + y*y)
    up = torch.stack([up_x, up_y, up_z], dim=-1)

    return fwd, up


def quat_from_euler(pitch, yaw, roll):
    """Create quaternion from Euler angles (pitch, yaw, roll).

    Uses ZYX convention matching rlgym_sim.
    pitch, yaw, roll: (...) → (..., 4)
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=-1)


def safe_normalize(v, dim=-1, eps=1e-6):
    """Normalize vectors, returning zero for zero-length vectors."""
    n = v.norm(dim=dim, keepdim=True)
    return v / (n + eps)


def piecewise_linear(x, bp_x, bp_y):
    """Vectorized piecewise linear interpolation.

    x: arbitrary-shape tensor of query values
    bp_x: (N,) sorted breakpoints (1D tensor on same device)
    bp_y: (N,) corresponding values
    Returns: tensor same shape as x with interpolated values.
    """
    idx = torch.searchsorted(bp_x, x.contiguous())
    idx = idx.clamp(1, len(bp_x) - 1)
    x0, x1 = bp_x[idx - 1], bp_x[idx]
    y0, y1 = bp_y[idx - 1], bp_y[idx]
    t = ((x - x0) / (x1 - x0)).clamp(0, 1)
    return y0 + t * (y1 - y0)
