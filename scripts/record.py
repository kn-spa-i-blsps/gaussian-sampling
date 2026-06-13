"""
Interactive viewer for a Gaussian Splat scene rendered with gsplat.

Controls:
  W / A / S / D   - move forward / left / back / right
  Q / E           - move down / up
  Arrow keys      - look around (Left/Right = yaw, Up/Down = pitch, full 360 range)
  SPACE           - toggle frame recording (saves PNGs to ./recording/)
  ESC             - quit

Dependencies: pygame, pillow, torch, gsplat, numpy
    pip install pygame pillow

Run:
    python splat_viewer.py
"""

import os
import math
import numpy as np
import torch
import pygame
import time
from PIL import Image

from sampling.sampling_engine import SamplingEngine

PLY_PATH = "splat.ply"
WIDTH, HEIGHT = 800, 600
FOV_DEG = 60.0

MOVE_SPEED = 2.0          # world units per second
TURN_SPEED = 1.5          # radians per second

RECORD_DIR = "recording"

WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def build_K(width, height, fov_deg):
  f = 0.5 * width / math.tan(math.radians(fov_deg) / 2.0)
  return np.array(
    [
      [f, 0, width / 2.0],
      [0, f, height / 2.0],
      [0, 0, 1.0],
    ],
    dtype=np.float32,
  )


def axis_angle_rotation(axis, angle):
  """Rodrigues' rotation formula: rotation matrix for `angle` radians about `axis`."""
  axis = axis / np.linalg.norm(axis)
  x, y, z = axis
  c, s = math.cos(angle), math.sin(angle)
  C = 1.0 - c
  return np.array(
    [
      [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
      [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
      [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ],
    dtype=np.float32,
  )


def initial_rotation():
  """Initial camera rotation matrix (columns = right, down, forward in world space),
  matching the original yaw=0, pitch=0 orientation (OpenCV convention)."""
  forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
  right = np.cross(forward, WORLD_UP)
  right /= np.linalg.norm(right)
  up = np.cross(right, forward)
  return np.stack([right, -up, forward], axis=1).astype(np.float32)


def build_c2w(position, R):
  c2w = np.eye(4, dtype=np.float32)
  c2w[:3, :3] = R
  c2w[:3, 3] = position
  return c2w


def main():
  pygame.init()
  screen = pygame.display.set_mode((WIDTH, HEIGHT))
  pygame.display.set_caption("Gaussian Splat Viewer")
  clock = pygame.time.Clock()

  engine = SamplingEngine(PLY_PATH)
  K = build_K(WIDTH, HEIGHT, FOV_DEG)

  position = np.array([0.0, 0.0, -3.0], dtype=np.float32)
  R = initial_rotation()  # camera-to-world rotation; columns = [right, down, forward]

  recording = False
  frame_idx = 0
  os.makedirs(RECORD_DIR, exist_ok=True)

  running = True
  while running:
    dt = clock.tick(60) / 1000.0  # seconds since last frame

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          running = False
        elif event.key == pygame.K_SPACE:
          recording = not recording
          print("Recording:", "ON" if recording else "OFF")

    keys = pygame.key.get_pressed()

    # --- look (arrow keys), full 360 range, no gimbal lock ---
    # Yaw: rotate the whole camera frame about the world up axis (pre-multiply).
    if keys[pygame.K_LEFT]:
      R = axis_angle_rotation(WORLD_UP, -TURN_SPEED * dt) @ R
    if keys[pygame.K_RIGHT]:
      R = axis_angle_rotation(WORLD_UP, TURN_SPEED * dt) @ R

    # Pitch: rotate about the camera's own current right axis (post-multiply).
    # If up/down feel inverted, flip the sign on these two lines.
    local_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if keys[pygame.K_UP]:
      R = R @ axis_angle_rotation(local_right, -TURN_SPEED * dt)
    if keys[pygame.K_DOWN]:
      R = R @ axis_angle_rotation(local_right, TURN_SPEED * dt)

    # --- movement (WASD + Q/E) ---
    right = R[:, 0]
    forward = R[:, 2]
    # R[:,1] is the camera's "down" axis (OpenCV convention), so "up" is its negation.
    cam_up = -R[:, 1]

    move = np.zeros(3, dtype=np.float32)
    if keys[pygame.K_w]:
      move += forward
    if keys[pygame.K_s]:
      move -= forward
    if keys[pygame.K_d]:
      move += right
    if keys[pygame.K_a]:
      move -= right
    if keys[pygame.K_e]:
      move += WORLD_UP
    if keys[pygame.K_q]:
      move -= WORLD_UP

    norm = np.linalg.norm(move)
    if norm > 0:
      position += (move / norm) * MOVE_SPEED * dt

    c2w = build_c2w(position, R)

    # --- render ---
    with torch.no_grad():
      image = engine.sample(c2w, K, WIDTH, HEIGHT)

    image_np = (image.cpu().numpy() * 255).astype(np.uint8)  # (H, W, 3)

    # pygame surfarray wants (W, H, 3)
    surf = pygame.surfarray.make_surface(image_np.transpose(1, 0, 2))
    screen.blit(surf, (0, 0))

    if recording:
      pygame.draw.circle(screen, (255, 0, 0), (20, 20), 8)
      Image.fromarray(image_np).save(
        os.path.join(RECORD_DIR, f"frame_{frame_idx:05d}.png")
      )
      frame_idx += 1

    pygame.display.flip()

  pygame.quit()


if __name__ == "__main__":
  main()