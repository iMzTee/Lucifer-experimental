"""render_bot.py — 2D top-down visualizer for LuciferBot.

Loads a checkpoint, runs 1 environment with deterministic policy inference,
and renders a pygame 2D top-down view at 60 FPS.

Usage:
    python render_bot.py [--stage N] [--checkpoint PATH]

Controls:
    R       - Reset environment
    Q/Esc   - Quit
    1-4     - Switch stage
    Space   - Pause/Resume
"""

import argparse
import os
import sys
import math
import torch

from gpu_sim.ppo import PPOLearner
from gpu_sim.environment import GPUEnvironment
from gpu_sim.observations import build_obs_batch
from gpu_sim.constants import STAGE_CONFIG, ARENA_HALF_X, ARENA_HALF_Y, ARENA_HEIGHT, GOAL_HALF_WIDTH

try:
    import pygame
except ImportError:
    print("pygame not installed. Install with: pip install pygame")
    sys.exit(1)


# ── Constants ──
WINDOW_W, WINDOW_H = 800, 600
FIELD_MARGIN = 40
BG_COLOR = (30, 30, 30)
FIELD_COLOR = (40, 60, 40)
LINE_COLOR = (80, 100, 80)
BALL_COLOR = (255, 200, 50)
BLUE_COLOR = (50, 120, 255)
ORANGE_COLOR = (255, 120, 50)
BOOST_PAD_COLOR = (200, 200, 100)
GOAL_COLOR = (180, 180, 180)
TEXT_COLOR = (220, 220, 220)


def world_to_screen(x, y):
    """Convert world coords (x, y) to screen pixel coords."""
    # X: -4096 to 4096 → FIELD_MARGIN to WINDOW_W-FIELD_MARGIN
    # Y: -5120 to 5120 → WINDOW_H-FIELD_MARGIN to FIELD_MARGIN (flipped)
    field_w = WINDOW_W - 2 * FIELD_MARGIN
    field_h = WINDOW_H - 2 * FIELD_MARGIN
    sx = FIELD_MARGIN + (x + ARENA_HALF_X) / (2 * ARENA_HALF_X) * field_w
    sy = WINDOW_H - FIELD_MARGIN - (y + ARENA_HALF_Y) / (2 * ARENA_HALF_Y) * field_h
    return int(sx), int(sy)


def world_radius_to_screen(r):
    """Convert world radius to screen pixels."""
    field_w = WINDOW_W - 2 * FIELD_MARGIN
    return max(2, int(r / (2 * ARENA_HALF_X) * field_w))


class Renderer:
    def __init__(self, stage=0, checkpoint_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.stage = stage
        self.paused = False

        # Create single-env environment
        self.env = GPUEnvironment(1, self.device, stage=stage)
        self.n_agents = self.env.n_agents

        # Previous actions
        self._prev_actions = torch.zeros(1, self.n_agents, 8, device=self.device)

        # Load policy
        self.policy = None
        self._load_policy(checkpoint_path)

        # Obs normalization (simple running stats)
        self._obs_mean = torch.zeros(127, device=self.device)
        self._obs_std = torch.ones(127, device=self.device)

        # Initialize
        self.env.reset_all()

        # Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption(f"LuciferBot — Stage {stage}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self.font_large = pygame.font.SysFont("monospace", 18, bold=True)

    def _load_policy(self, checkpoint_path=None):
        """Load policy from checkpoint."""
        if checkpoint_path is None:
            # Find latest checkpoint
            for folder in ["Checkpoints_LuciferBot", "Checkpoints_2v2_gpu"]:
                if not os.path.exists(folder):
                    continue
                potential = [d for d in os.listdir(folder)
                             if os.path.isdir(os.path.join(folder, d)) and d != "stage_backups"]
                if potential:
                    latest = sorted(potential, key=lambda d: os.path.getmtime(
                        os.path.join(folder, d)))[-1]
                    checkpoint_path = os.path.join(folder, latest)
                    break

        if checkpoint_path is None:
            print("[!] No checkpoint found — running with random policy")
            return

        # Create PPO learner just to load the policy
        ppo = PPOLearner(
            obs_space_size=127,
            act_space_size=8,
            device=self.device,
            batch_size=1000,
            mini_batch_size=1000,
            n_epochs=1,
            policy_layer_sizes=(2048, 2048, 1024, 1024),
            critic_layer_sizes=(2048, 2048, 1024, 1024),
            policy_lr=2e-4,
            critic_lr=2e-4,
        )
        ppo.load_from(checkpoint_path)
        self.policy = ppo.policy
        self.policy.eval()
        print(f"[*] Loaded checkpoint: {checkpoint_path}")

    def _step(self):
        """Run one decision step."""
        if self.paused:
            return

        # Build obs
        obs = build_obs_batch(self.env.state, self._prev_actions)

        # Normalize (simple)
        obs_norm = (obs - self._obs_mean) / self._obs_std
        obs_norm = obs_norm.clamp(-5.0, 5.0)

        # Policy inference
        if self.policy is not None:
            with torch.no_grad():
                actions, _ = self.policy.get_action(obs_norm, deterministic=True)
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)
        else:
            # Random actions
            actions = torch.zeros(self.n_agents, 8, device=self.device)

        # Parse to controls
        controls = torch.zeros(1, self.n_agents, 8, device=self.device)
        actions_r = actions.reshape(1, self.n_agents, 8)
        controls[:, :, :5] = actions_r[:, :, :5] * 0.5 - 1.0
        controls[:, :, 5:] = actions_r[:, :, 5:]
        self._prev_actions.copy_(controls)

        # Step
        terminals = self.env.step(controls)

        if terminals.any():
            self.env.reset_done_envs(terminals)
            self._prev_actions[:] = 0.0

    def _draw(self):
        """Draw the scene."""
        self.screen.fill(BG_COLOR)
        s = self.env.state

        # ── Field outline ──
        field_rect = pygame.Rect(FIELD_MARGIN, FIELD_MARGIN,
                                  WINDOW_W - 2 * FIELD_MARGIN,
                                  WINDOW_H - 2 * FIELD_MARGIN)
        pygame.draw.rect(self.screen, FIELD_COLOR, field_rect)
        pygame.draw.rect(self.screen, LINE_COLOR, field_rect, 2)

        # Center line
        cl_y = world_to_screen(0, 0)[1]
        pygame.draw.line(self.screen, LINE_COLOR,
                         (FIELD_MARGIN, cl_y), (WINDOW_W - FIELD_MARGIN, cl_y), 1)

        # Center circle
        cx, cy = world_to_screen(0, 0)
        cr = world_radius_to_screen(1000)
        pygame.draw.circle(self.screen, LINE_COLOR, (cx, cy), cr, 1)

        # ── Goals ──
        for goal_y, color in [(ARENA_HALF_Y, ORANGE_COLOR), (-ARENA_HALF_Y, BLUE_COLOR)]:
            gl = world_to_screen(-GOAL_HALF_WIDTH, goal_y)
            gr = world_to_screen(GOAL_HALF_WIDTH, goal_y)
            pygame.draw.line(self.screen, color, gl, gr, 3)

        # ── Ball ──
        bx, by = s.ball_pos[0, 0].item(), s.ball_pos[0, 1].item()
        sx, sy = world_to_screen(bx, by)
        br = world_radius_to_screen(92.75)
        pygame.draw.circle(self.screen, BALL_COLOR, (sx, sy), max(br, 4))

        # Ball velocity line
        bvx, bvy = s.ball_vel[0, 0].item(), s.ball_vel[0, 1].item()
        vel_scale = 0.02
        ex, ey = world_to_screen(bx + bvx * vel_scale, by + bvy * vel_scale)
        pygame.draw.line(self.screen, (255, 255, 100), (sx, sy), (ex, ey), 1)

        # ── Cars ──
        for i in range(self.n_agents):
            if s.car_is_demoed[0, i].item() > 0.5:
                continue

            team = s.car_team[0, i].item()
            color = BLUE_COLOR if team == 0 else ORANGE_COLOR

            cx_w = s.car_pos[0, i, 0].item()
            cy_w = s.car_pos[0, i, 1].item()
            csx, csy = world_to_screen(cx_w, cy_w)

            # Forward direction
            fx = s.car_fwd[0, i, 0].item()
            fy = s.car_fwd[0, i, 1].item()

            # Draw as triangle
            car_len = world_radius_to_screen(150)
            car_wid = world_radius_to_screen(80)

            # Triangle points: nose, left rear, right rear
            angle = math.atan2(-fy, fx)  # screen Y is flipped
            cos_a, sin_a = math.cos(angle), math.sin(angle)

            nose = (csx + int(car_len * cos_a), csy + int(car_len * sin_a))
            left = (csx + int(-car_len * 0.5 * cos_a + car_wid * sin_a),
                    csy + int(-car_len * 0.5 * sin_a - car_wid * cos_a))
            right = (csx + int(-car_len * 0.5 * cos_a - car_wid * sin_a),
                     csy + int(-car_len * 0.5 * sin_a + car_wid * cos_a))

            pygame.draw.polygon(self.screen, color, [nose, left, right])
            pygame.draw.polygon(self.screen, (255, 255, 255), [nose, left, right], 1)

            # Boost indicator
            boost_pct = int(s.car_boost[0, i].item() * 100)
            boost_text = self.font.render(f"{boost_pct}%", True, TEXT_COLOR)
            self.screen.blit(boost_text, (csx - 15, csy + 12))

        # ── HUD ──
        # Score
        blue_score = s.blue_score[0].item()
        orange_score = s.orange_score[0].item()
        score_text = self.font_large.render(f"{blue_score} - {orange_score}", True, TEXT_COLOR)
        self.screen.blit(score_text, (WINDOW_W // 2 - 30, 5))

        # Stage info
        stage_names = ["Solo Mechanics", "1v1 Mechanics", "1v1 Game Sense", "2v2 Teamwork"]
        stage_text = self.font.render(
            f"Stage {self.stage}: {stage_names[self.stage]}  |  "
            f"Step {s.step_count[0].item()}/{self.env.timeout}", True, TEXT_COLOR)
        self.screen.blit(stage_text, (10, WINDOW_H - 22))

        # Car 0 speed
        if self.n_agents > 0:
            speed = s.car_vel[0, 0].norm().item()
            z = s.car_pos[0, 0, 2].item()
            info = self.font.render(f"Speed: {speed:.0f}  Z: {z:.0f}", True, TEXT_COLOR)
            self.screen.blit(info, (10, 5))

        # Pause indicator
        if self.paused:
            pause_text = self.font_large.render("PAUSED", True, (255, 100, 100))
            self.screen.blit(pause_text, (WINDOW_W // 2 - 35, WINDOW_H // 2))

        pygame.display.flip()

    def run(self):
        """Main render loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        self.env.reset_all()
                        self._prev_actions[:] = 0.0
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                        new_stage = event.key - pygame.K_1
                        if new_stage != self.stage:
                            self.stage = new_stage
                            cfg = STAGE_CONFIG.get(new_stage, STAGE_CONFIG[0])
                            new_n_agents = cfg.get("n_agents", 4)
                            self.env = GPUEnvironment(1, self.device, stage=new_stage)
                            self.n_agents = new_n_agents
                            self._prev_actions = torch.zeros(1, new_n_agents, 8, device=self.device)
                            self.env.reset_all()
                            pygame.display.set_caption(f"LuciferBot — Stage {new_stage}")

            self._step()
            self._draw()
            self.clock.tick(60)  # 60 FPS

        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LuciferBot 2D Visualizer")
    parser.add_argument("--stage", type=int, default=0, help="Curriculum stage (0-3)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    args = parser.parse_args()

    renderer = Renderer(stage=args.stage, checkpoint_path=args.checkpoint)
    renderer.run()
