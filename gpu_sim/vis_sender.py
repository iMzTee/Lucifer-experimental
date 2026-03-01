"""vis_sender.py — Lightweight UDP sender for RocketSimVis.

Sends game state over UDP to localhost:9273. During collection bursts,
sends real positions+velocities so RocketSimVis interpolates smoothly.
When no new physics arrives (training phase), holds the last position
with zero velocity to prevent drift.

Cycles through random envs every few seconds. Zero overhead when disabled.
"""

import socket
import json
import random
import threading
import time
import torch


class VisSender:
    """Sends game state to RocketSimVis over UDP.

    During collection, raw physics states stream at full speed. Between
    collections, the background thread resends the last state with zero
    velocity so objects freeze in place instead of drifting.

    Args:
        n_envs: Total number of environments (for cycling).
        enabled: If False, all methods are no-ops (zero overhead).
        switch_interval: Seconds between switching to a new env.
        port: UDP port for RocketSimVis (default 9273).
    """

    STALE_TIMEOUT = 0.2  # seconds before switching to zero-velocity hold

    def __init__(self, n_envs=1, enabled=False, switch_interval=5.0, port=9273):
        self.n_envs = n_envs
        self.enabled = enabled
        self.port = port
        self.switch_interval = switch_interval
        self.env_idx = 0
        self._sock = None
        self._last_packet = None   # latest raw packet (with velocity)
        self._last_send_time = 0   # when send() was last called
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._last_switch = time.time()

        if enabled:
            self.env_idx = random.randint(0, max(0, n_envs - 1))
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._addr = ("127.0.0.1", port)
            self._thread = threading.Thread(target=self._send_loop, daemon=True)
            self._thread.start()
            print(f"[VIS] Watching env {self.env_idx}, switching every {switch_interval}s")

    def _switch_env(self):
        """Switch to a new random env."""
        self.env_idx = random.randint(0, max(0, self.n_envs - 1))
        self._last_switch = time.time()
        print(f"[VIS] Switched to env {self.env_idx}")

    @staticmethod
    def _zero_vel(packet):
        """Copy packet with all velocities zeroed."""
        result = {
            "ball_phys": {
                "pos": packet["ball_phys"]["pos"],
                "vel": [0, 0, 0],
                "ang_vel": [0, 0, 0],
            },
            "cars": [],
            "boost_pad_states": packet["boost_pad_states"],
        }
        for c in packet["cars"]:
            result["cars"].append({
                "team_num": c["team_num"],
                "phys": {
                    "pos": c["phys"]["pos"],
                    "vel": [0, 0, 0],
                    "ang_vel": [0, 0, 0],
                    "forward": c["phys"]["forward"],
                    "up": c["phys"]["up"],
                },
                "boost_amount": c["boost_amount"],
                "on_ground": c["on_ground"],
                "is_demoed": c["is_demoed"],
                "has_flip": c["has_flip"],
            })
        return result

    def _send_loop(self):
        """Background: relay fresh frames or hold position at 30fps."""
        while not self._stop.is_set():
            now = time.time()

            # Switch env on interval
            if now - self._last_switch >= self.switch_interval:
                with self._lock:
                    self._switch_env()

            with self._lock:
                packet = self._last_packet
                age = now - self._last_send_time

            if packet is not None:
                # Fresh data → send with real velocity (RocketSimVis interpolates)
                # Stale data → send with zero velocity (hold in place)
                if age > self.STALE_TIMEOUT:
                    packet = self._zero_vel(packet)
                try:
                    self._sock.sendto(
                        json.dumps(packet).encode("utf-8"), self._addr)
                except OSError:
                    pass

            self._stop.wait(1.0 / 30)

    def send(self, state):
        """Snapshot the current env's state.

        Args:
            state: TensorState from GPUEnvironment.
        """
        if not self.enabled:
            return

        i = self.env_idx
        n_agents = state.n_agents

        # ── Single GPU→CPU transfer: batch all reads into one .cpu() call ──
        # Ball: 3 vectors (pos, vel, ang_vel) = 9 floats
        # Per car: 5 vectors (pos, vel, ang_vel, fwd, up) = 15 floats + 4 scalars
        # Stack everything, transfer once, unpack on CPU
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

        with self._lock:
            self._last_packet = packet
            self._last_send_time = time.time()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
