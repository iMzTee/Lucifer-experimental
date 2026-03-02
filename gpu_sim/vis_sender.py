"""vis_sender.py — Lightweight UDP sender for RocketSimVis.

Sends game state over UDP to localhost:9273. During collection, send()
transmits directly for real-time updates. Between collections, the
background thread extrapolates positions using last known velocity so
movement continues smoothly. After STALE_TIMEOUT, objects freeze.

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

    During collection, send() fires immediately for real-time vis.
    Between collections, the background thread extrapolates positions
    using last known XY velocity for smooth continued motion.

    Args:
        n_envs: Total number of environments (for cycling).
        enabled: If False, all methods are no-ops (zero overhead).
        switch_interval: Seconds between switching to a new env.
        port: UDP port for RocketSimVis (default 9273).
    """

    STALE_TIMEOUT = 0.5  # seconds before switching to zero-velocity hold
    SPEED_SCALE = 2.0    # playback speed multiplier for viewer

    def __init__(self, n_envs=1, enabled=False, switch_interval=10.0, port=9273):
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

    def _send_udp(self, packet):
        """Send a packet over UDP (no-throw)."""
        try:
            self._sock.sendto(json.dumps(packet).encode("utf-8"), self._addr)
        except OSError:
            pass

    @staticmethod
    def _zero_vel(packet):
        """Copy packet with all velocities zeroed (freeze in place)."""
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

    @staticmethod
    def _extrapolate(packet, dt):
        """Extrapolate positions forward by dt using XY velocity.

        Car vel_z is already zeroed in send(), so only XY moves.
        Ball Z is also kept fixed to avoid gravity drift artifacts.
        """
        ball = packet["ball_phys"]
        result = {
            "ball_phys": {
                "pos": [
                    ball["pos"][0] + ball["vel"][0] * dt,
                    ball["pos"][1] + ball["vel"][1] * dt,
                    ball["pos"][2],  # no Z extrapolation
                ],
                "vel": ball["vel"],
                "ang_vel": ball["ang_vel"],
            },
            "cars": [],
            "boost_pad_states": packet["boost_pad_states"],
        }
        for c in packet["cars"]:
            pos = c["phys"]["pos"]
            vel = c["phys"]["vel"]
            result["cars"].append({
                "team_num": c["team_num"],
                "phys": {
                    "pos": [pos[0] + vel[0] * dt, pos[1] + vel[1] * dt, pos[2]],
                    "vel": vel,
                    "ang_vel": c["phys"]["ang_vel"],
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
        """Background: extrapolate between collections, freeze when stale."""
        while not self._stop.is_set():
            now = time.time()

            # Switch env on interval
            if now - self._last_switch >= self.switch_interval:
                with self._lock:
                    self._switch_env()

            with self._lock:
                packet = self._last_packet
                age = now - self._last_send_time

            if packet is not None and age > 0.05:
                # Resend last known state without extrapolation to avoid jitter
                if age > self.STALE_TIMEOUT:
                    self._send_udp(self._zero_vel(packet))
                else:
                    self._send_udp(packet)

            self._stop.wait(1.0 / 240)

    def send(self, state):
        """Snapshot the current env's state and send immediately.

        Args:
            state: TensorState from GPUEnvironment.
        """
        if not self.enabled:
            return

        i = self.env_idx
        n_agents = state.n_agents

        # ── Single GPU→CPU transfer: batch all reads into one .cpu() call ──
        ball_data = torch.stack([
            state.ball_pos[i], state.ball_vel[i], state.ball_ang_vel[i]
        ]).cpu()  # (3, 3)

        car_vel_xy = state.car_vel[i].clone()
        car_vel_xy[:, 2] = 0  # zero vel_z: prevents upward extrapolation drift
        car_vecs = torch.stack([
            state.car_pos[i], car_vel_xy, state.car_ang_vel[i],
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

        # Scale velocities for playback speed
        if self.SPEED_SCALE != 1.0:
            s = self.SPEED_SCALE
            packet["ball_phys"]["vel"] = [v * s for v in ball_phys["vel"]]
            packet["ball_phys"]["ang_vel"] = [v * s for v in ball_phys["ang_vel"]]
            for c in packet["cars"]:
                c["phys"]["vel"] = [v * s for v in c["phys"]["vel"]]
                c["phys"]["ang_vel"] = [v * s for v in c["phys"]["ang_vel"]]

        with self._lock:
            self._last_packet = packet
            self._last_send_time = time.time()

        # Send immediately — don't wait for background loop
        self._send_udp(packet)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
