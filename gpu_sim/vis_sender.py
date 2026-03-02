"""vis_sender.py — Lightweight UDP sender for RocketSimVis.

Sends game state snapshots over UDP to localhost:9273 each time send()
is called during collection. Cycles through random envs every few
seconds. Zero overhead when disabled.
"""

import socket
import json
import random
import time
import torch


class VisSender:
    """Sends game state to RocketSimVis over UDP.

    Each send() call transmits the current env's state as a single packet.
    The viewer interpolates between consecutive packets on its own.

    Args:
        n_envs: Total number of environments (for cycling).
        enabled: If False, all methods are no-ops (zero overhead).
        switch_interval: Seconds between switching to a new env.
        port: UDP port for RocketSimVis (default 9273).
    """

    def __init__(self, n_envs=1, enabled=False, switch_interval=10.0, port=9273):
        self.n_envs = n_envs
        self.enabled = enabled
        self.port = port
        self.switch_interval = switch_interval
        self.env_idx = 0
        self._sock = None
        self._last_switch = time.time()

        if enabled:
            self.env_idx = random.randint(0, max(0, n_envs - 1))
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._addr = ("127.0.0.1", port)
            print(f"[VIS] Watching env {self.env_idx}, switching every {switch_interval}s")

    def _switch_env(self):
        """Switch to a new random env."""
        self.env_idx = random.randint(0, max(0, self.n_envs - 1))
        self._last_switch = time.time()
        print(f"[VIS] Switched to env {self.env_idx}")

    def _send_udp(self, packet):
        """Send a packet over UDP (no-throw)."""
        try:
            self._sock.sendto(json.dumps(packet).encode("utf-8"), self._addr)
        except OSError:
            pass

    def send(self, state):
        """Snapshot the current env's state and send immediately.

        Args:
            state: TensorState from GPUEnvironment.
        """
        if not self.enabled:
            return

        # Switch env on interval
        now = time.time()
        if now - self._last_switch >= self.switch_interval:
            self._switch_env()

        i = self.env_idx
        n_agents = state.n_agents

        # ── Single GPU→CPU transfer: batch all reads into one .cpu() call ──
        ball_data = torch.stack([
            state.ball_pos[i], state.ball_vel[i], state.ball_ang_vel[i]
        ]).cpu()  # (3, 3)

        car_vecs = torch.stack([
            state.car_pos[i], state.car_vel[i], state.car_ang_vel[i],
            state.car_fwd[i], state.car_up[i],
        ])  # (5, A, 3)
        car_scalars = torch.stack([
            state.car_boost[i], state.car_on_ground[i],
            state.car_is_demoed[i], state.car_has_flip[i],
        ])  # (4, A)
        car_vecs_cpu = car_vecs.cpu()
        car_scalars_cpu = car_scalars.cpu()
        car_teams_cpu = state.car_team[i].cpu()
        boost_pad_states = (state.boost_pad_timers[i] == 0).cpu().tolist()

        # ── Unpack on CPU (no GPU sync from here) ──
        ball_phys = {
            "pos": ball_data[0].tolist(),
            "vel": ball_data[1].tolist(),
            "ang_vel": ball_data[2].tolist(),
        }

        cars = []
        for a in range(n_agents):
            car = {
                "team_num": int(car_teams_cpu[a].item()),
                "phys": {
                    "pos": car_vecs_cpu[0, a].tolist(),
                    "vel": car_vecs_cpu[1, a].tolist(),
                    "ang_vel": car_vecs_cpu[2, a].tolist(),
                    "forward": car_vecs_cpu[3, a].tolist(),
                    "up": car_vecs_cpu[4, a].tolist(),
                },
                "boost_amount": float(car_scalars_cpu[0, a].item()) * 100,
                "on_ground": bool(car_scalars_cpu[1, a].item() > 0.5),
                "is_demoed": bool(car_scalars_cpu[2, a].item() > 0.5),
                "has_flip": bool(car_scalars_cpu[3, a].item() > 0.5),
            }
            cars.append(car)

        packet = {
            "ball_phys": ball_phys,
            "cars": cars,
            "boost_pad_states": boost_pad_states,
        }

        self._send_udp(packet)

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None
