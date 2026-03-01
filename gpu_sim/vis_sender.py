"""vis_sender.py — Lightweight UDP sender for RocketSimVis.

Extracts a single env from TensorState, builds the JSON packet
matching RocketSimVis protocol, and sends over UDP to localhost:9273.

Buffers snapshots during fast physics bursts and replays them in slow
motion via a background thread. Cycles through random envs every few
seconds to show different scenarios. Zero overhead when disabled.
"""

import socket
import json
import random
import threading
import time
from collections import deque


def _lerp(a, b, t):
    """Linearly interpolate between two lists of floats."""
    return [a[i] + (b[i] - a[i]) * t for i in range(len(a))]


def _lerp_packet(p1, p2, t):
    """Interpolate positions between two packets, zero velocities."""
    result = {
        "ball_phys": {
            "pos": _lerp(p1["ball_phys"]["pos"], p2["ball_phys"]["pos"], t),
            "vel": [0, 0, 0],
            "ang_vel": [0, 0, 0],
        },
        "cars": [],
        "boost_pad_states": p2["boost_pad_states"],
    }
    for c1, c2 in zip(p1["cars"], p2["cars"]):
        result["cars"].append({
            "team_num": c2["team_num"],
            "phys": {
                "pos": _lerp(c1["phys"]["pos"], c2["phys"]["pos"], t),
                "vel": [0, 0, 0],
                "ang_vel": [0, 0, 0],
                "forward": _lerp(c1["phys"]["forward"], c2["phys"]["forward"], t),
                "up": _lerp(c1["phys"]["up"], c2["phys"]["up"], t),
            },
            "boost_amount": c1["boost_amount"] + (c2["boost_amount"] - c1["boost_amount"]) * t,
            "on_ground": c2["on_ground"],
            "is_demoed": c2["is_demoed"],
            "has_flip": c2["has_flip"],
        })
    return result


def _zero_vel_packet(p):
    """Return packet with all velocities zeroed (prevents extrapolation)."""
    result = {
        "ball_phys": {
            "pos": p["ball_phys"]["pos"],
            "vel": [0, 0, 0],
            "ang_vel": [0, 0, 0],
        },
        "cars": [],
        "boost_pad_states": p["boost_pad_states"],
    }
    for c in p["cars"]:
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


class VisSender:
    """Sends game state to RocketSimVis over UDP.

    Buffers physics snapshots and replays with interpolation. Cycles
    to a random env every `switch_interval` seconds so you can watch
    different scenarios.

    Args:
        n_envs: Total number of environments (for cycling).
        enabled: If False, all methods are no-ops (zero overhead).
        switch_interval: Seconds between switching to a new env.
        transition_secs: Seconds to interpolate between each pair of frames.
        port: UDP port for RocketSimVis (default 9273).
    """

    def __init__(self, n_envs=1, enabled=False, switch_interval=5.0,
                 transition_secs=2.0, port=9273):
        self.n_envs = n_envs
        self.enabled = enabled
        self.port = port
        self.switch_interval = switch_interval
        self.transition_secs = transition_secs
        self.env_idx = 0
        self._sock = None
        self._queue = deque()
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
        """Switch to a new random env, clear the queue."""
        self.env_idx = random.randint(0, max(0, self.n_envs - 1))
        self._last_switch = time.time()
        self._queue.clear()
        print(f"[VIS] Switched to env {self.env_idx}")

    def _send_loop(self):
        """Background: interpolate between queued frames at 30fps."""
        prev = None
        target = None
        interp_start = 0

        while not self._stop.is_set():
            now = time.time()

            # Check if it's time to switch env
            if now - self._last_switch >= self.switch_interval:
                with self._lock:
                    self._switch_env()
                prev = None
                target = None

            # Grab new frames from queue
            with self._lock:
                while self._queue:
                    frame = self._queue.popleft()
                    if prev is None:
                        prev = frame
                    else:
                        if target is not None:
                            prev = target
                        target = frame
                        interp_start = now

            # Decide what to send
            if prev is not None and target is not None:
                elapsed = now - interp_start
                t = min(elapsed / self.transition_secs, 1.0)
                packet = _lerp_packet(prev, target, t)

                if t >= 1.0:
                    prev = target
                    target = None
            elif prev is not None:
                packet = _zero_vel_packet(prev)
            else:
                self._stop.wait(1.0 / 30)
                continue

            try:
                self._sock.sendto(json.dumps(packet).encode("utf-8"), self._addr)
            except OSError:
                pass

            self._stop.wait(1.0 / 30)

    def send(self, state):
        """Snapshot the current env's state into the replay queue.

        Args:
            state: TensorState from GPUEnvironment.
        """
        if not self.enabled:
            return

        i = self.env_idx
        n_agents = state.n_agents

        ball_phys = {
            "pos": state.ball_pos[i].cpu().tolist(),
            "vel": state.ball_vel[i].cpu().tolist(),
            "ang_vel": state.ball_ang_vel[i].cpu().tolist(),
        }

        cars = []
        for a in range(n_agents):
            car = {
                "team_num": int(state.car_team[i, a].item()),
                "phys": {
                    "pos": state.car_pos[i, a].cpu().tolist(),
                    "vel": state.car_vel[i, a].cpu().tolist(),
                    "ang_vel": state.car_ang_vel[i, a].cpu().tolist(),
                    "forward": state.car_fwd[i, a].cpu().tolist(),
                    "up": state.car_up[i, a].cpu().tolist(),
                },
                "boost_amount": float(state.car_boost[i, a].item()) * 100,
                "on_ground": bool(state.car_on_ground[i, a].item() > 0.5),
                "is_demoed": bool(state.car_is_demoed[i, a].item() > 0.5),
                "has_flip": bool(state.car_has_flip[i, a].item() > 0.5),
            }
            cars.append(car)

        boost_pad_states = (state.boost_pad_timers[i] == 0).cpu().tolist()

        packet = {
            "ball_phys": ball_phys,
            "cars": cars,
            "boost_pad_states": boost_pad_states,
        }

        with self._lock:
            self._queue.append(packet)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
