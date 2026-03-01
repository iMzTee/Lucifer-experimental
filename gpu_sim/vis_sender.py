"""vis_sender.py — Lightweight UDP sender for RocketSimVis.

Extracts a single env from TensorState, builds the JSON packet
matching RocketSimVis protocol, and sends over UDP to localhost:9273.

Zero overhead when disabled (no socket, no CPU transfers).
"""

import socket
import json


class VisSender:
    """Sends game state to RocketSimVis over UDP.

    Args:
        env_idx: Which environment index to visualize (default 0).
        enabled: If False, all methods are no-ops (zero overhead).
        send_interval: Send every N-th call to avoid flooding the renderer.
        port: UDP port for RocketSimVis (default 9273).
    """

    def __init__(self, env_idx=0, enabled=False, send_interval=4, port=9273):
        self.env_idx = env_idx
        self.enabled = enabled
        self.send_interval = max(1, send_interval)
        self.port = port
        self._call_count = 0
        self._sock = None

        if enabled:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._addr = ("127.0.0.1", port)

    def send(self, state):
        """Send a single env's state to RocketSimVis.

        Args:
            state: TensorState from GPUEnvironment.
        """
        if not self.enabled:
            return

        self._call_count += 1
        if self._call_count % self.send_interval != 0:
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

        try:
            self._sock.sendto(json.dumps(packet).encode("utf-8"), self._addr)
        except OSError:
            pass  # silently drop if socket fails

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None
