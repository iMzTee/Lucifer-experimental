"""vis_sender.py — Lightweight UDP sender for RocketSimVis.

Extracts a single env from TensorState and sends game state over UDP
to localhost:9273. Uses client-side extrapolation to show real-time
movement: the background thread advances positions using velocity and
gravity at game speed between sparse physics updates.

Cycles through random envs every few seconds. Zero overhead when disabled.
"""

import socket
import json
import random
import threading
import time

GRAVITY_Z = -650.0  # uu/s², matches constants.py
FLOOR_Z = 17.0      # car resting height
BALL_FLOOR_Z = 93.75 # ball resting height (BALL_RADIUS + 1)


class VisSender:
    """Sends game state to RocketSimVis over UDP.

    Between sparse physics snapshots, the background thread extrapolates
    positions using velocity + gravity at game speed (30fps), giving
    smooth real-time movement. When a new physics snapshot arrives, it
    corrects the prediction.

    Args:
        n_envs: Total number of environments (for cycling).
        enabled: If False, all methods are no-ops (zero overhead).
        switch_interval: Seconds between switching to a new env.
        port: UDP port for RocketSimVis (default 9273).
    """

    def __init__(self, n_envs=1, enabled=False, switch_interval=5.0, port=9273):
        self.n_envs = n_envs
        self.enabled = enabled
        self.port = port
        self.switch_interval = switch_interval
        self.env_idx = 0
        self._sock = None
        self._state = None  # mutable state for extrapolation
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
        """Switch to a new random env, clear state."""
        self.env_idx = random.randint(0, max(0, self.n_envs - 1))
        self._last_switch = time.time()
        self._state = None
        print(f"[VIS] Switched to env {self.env_idx}")

    def _extrapolate(self, dt):
        """Advance positions by dt seconds using velocity + gravity."""
        s = self._state
        if s is None:
            return

        # Ball
        bp = s["ball_phys"]
        bp["pos"][0] += bp["vel"][0] * dt
        bp["pos"][1] += bp["vel"][1] * dt
        bp["pos"][2] += bp["vel"][2] * dt
        bp["vel"][2] += GRAVITY_Z * dt
        # Floor clamp
        if bp["pos"][2] < BALL_FLOOR_Z:
            bp["pos"][2] = BALL_FLOOR_Z
            bp["vel"][2] = max(0, bp["vel"][2])

        # Cars
        for c in s["cars"]:
            p = c["phys"]
            p["pos"][0] += p["vel"][0] * dt
            p["pos"][1] += p["vel"][1] * dt
            p["pos"][2] += p["vel"][2] * dt
            if not c["on_ground"]:
                p["vel"][2] += GRAVITY_Z * dt
            # Floor clamp
            if p["pos"][2] < FLOOR_Z:
                p["pos"][2] = FLOOR_Z
                p["vel"][2] = max(0, p["vel"][2])

    def _build_packet(self):
        """Build a zero-velocity packet from current extrapolated state."""
        s = self._state
        packet = {
            "ball_phys": {
                "pos": list(s["ball_phys"]["pos"]),
                "vel": [0, 0, 0],
                "ang_vel": [0, 0, 0],
            },
            "cars": [],
            "boost_pad_states": s["boost_pad_states"],
        }
        for c in s["cars"]:
            packet["cars"].append({
                "team_num": c["team_num"],
                "phys": {
                    "pos": list(c["phys"]["pos"]),
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
        return packet

    def _send_loop(self):
        """Background: extrapolate at game speed, send at 30fps."""
        dt = 1.0 / 30
        while not self._stop.is_set():
            now = time.time()

            # Check if it's time to switch env
            if now - self._last_switch >= self.switch_interval:
                with self._lock:
                    self._switch_env()

            with self._lock:
                if self._state is not None:
                    self._extrapolate(dt)
                    packet = self._build_packet()
                else:
                    packet = None

            if packet is not None:
                try:
                    self._sock.sendto(
                        json.dumps(packet).encode("utf-8"), self._addr)
                except OSError:
                    pass

            self._stop.wait(dt)

    def send(self, state):
        """Snapshot the current env's state, correcting extrapolation.

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

        snapshot = {
            "ball_phys": ball_phys,
            "cars": cars,
            "boost_pad_states": boost_pad_states,
        }

        with self._lock:
            self._state = snapshot

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
