"""vis_sender.py — Lightweight UDP sender for RocketSimVis.

Extracts a single env from TensorState, builds the JSON packet
matching RocketSimVis protocol, and sends over UDP to localhost:9273.

Uses a background thread to re-send the last known state at ~30fps,
preventing RocketSimVis from extrapolating during training pauses.

Zero overhead when disabled (no socket, no CPU transfers).
"""

import socket
import json
import threading
import time


class VisSender:
    """Sends game state to RocketSimVis over UDP.

    A background thread continuously re-sends the last snapshot at ~30fps
    so the viewer doesn't drift objects using stale velocities between
    physics bursts.

    Args:
        env_idx: Which environment index to visualize (default 0).
        enabled: If False, all methods are no-ops (zero overhead).
        port: UDP port for RocketSimVis (default 9273).
    """

    def __init__(self, env_idx=0, enabled=False, port=9273):
        self.env_idx = env_idx
        self.enabled = enabled
        self.port = port
        self._sock = None
        self._last_packet = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

        if enabled:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._addr = ("127.0.0.1", port)
            self._thread = threading.Thread(target=self._send_loop, daemon=True)
            self._thread.start()

    def _send_loop(self):
        """Background thread: re-send last known state at ~30fps."""
        while not self._stop.is_set():
            with self._lock:
                packet_bytes = self._last_packet
            if packet_bytes is not None:
                try:
                    self._sock.sendto(packet_bytes, self._addr)
                except OSError:
                    pass
            self._stop.wait(1.0 / 30)

    def send(self, state):
        """Snapshot a single env's state for the background sender.

        Args:
            state: TensorState from GPUEnvironment.
        """
        if not self.enabled:
            return

        i = self.env_idx
        n_agents = state.n_agents

        # Ball physics
        ball_phys = {
            "pos": state.ball_pos[i].cpu().tolist(),
            "vel": state.ball_vel[i].cpu().tolist(),
            "ang_vel": state.ball_ang_vel[i].cpu().tolist(),
        }

        # Cars
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

        # Boost pad states: timer == 0 means active/available
        boost_pad_states = (state.boost_pad_timers[i] == 0).cpu().tolist()

        packet = {
            "ball_phys": ball_phys,
            "cars": cars,
            "boost_pad_states": boost_pad_states,
        }

        packet_bytes = json.dumps(packet).encode("utf-8")
        with self._lock:
            self._last_packet = packet_bytes

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
