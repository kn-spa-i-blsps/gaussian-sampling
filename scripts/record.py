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
import argparse
from PIL import Image

from sampling.sampling_engine import SamplingEngine

import threading
import queue

save_queue = queue.Queue()

# Black background for unmarked frames
unmarked_bg_color = torch.tensor([0.0, 0.0, 0.0], device="cuda", dtype=torch.float32).unsqueeze(0)

def save_worker():
  while True:
    item = save_queue.get()
    if item is None:        # sentinel → shut down
        break
    path, arr = item
    Image.fromarray(arr).save(path)
    save_queue.task_done()



WIDTH, HEIGHT = 800, 600
FOV_DEG = 60.0

TURN_SPEED = 1.5          # radians per second

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
  parser = argparse.ArgumentParser()

  save_thread = threading.Thread(target=save_worker, daemon=True)
  save_thread.start()

  parser.add_argument("splat_path", type=str)
  parser.add_argument("--move_speed", type=float, default=2.0)
  parser.add_argument("--invert_y", type=bool, default=False)
  parser.add_argument("--record_fps", type=float, default=10.0)
  parser.add_argument("--record_dir", type=str, default="recording")
  args = parser.parse_args()

  record_tick = 1.0 / args.record_fps
  last_record_time = time.time()

  pygame.init()
  screen = pygame.display.set_mode((WIDTH, HEIGHT))
  pygame.display.set_caption("Gaussian Splat Viewer")
  clock = pygame.time.Clock()

  engine = SamplingEngine(args.splat_path)
  K = build_K(WIDTH, HEIGHT, FOV_DEG)

  position = np.array([0.0, 0.0, -3.0], dtype=np.float32)
  R = initial_rotation()  # camera-to-world rotation; columns = [right, down, forward]

  recording = False
  frame_idx = 0
  os.makedirs(args.record_dir, exist_ok=True)
  os.makedirs(os.path.join(args.record_dir, "marked"), exist_ok=True)
  os.makedirs(os.path.join(args.record_dir, "unmarked"), exist_ok=True)

  running = True
  while running:
    dt = clock.tick(60) / 1000.0

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

    # Left and right rotation
    if keys[pygame.K_LEFT]:
      R = axis_angle_rotation(WORLD_UP, -TURN_SPEED * dt) @ R
    if keys[pygame.K_RIGHT]:
      R = axis_angle_rotation(WORLD_UP, TURN_SPEED * dt) @ R

    # Up and down rotation (pitch)
    local_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if keys[pygame.K_UP]:
      R = R @ axis_angle_rotation(local_right, (-1 if args.invert_y else 1) * -TURN_SPEED * dt)
    if keys[pygame.K_DOWN]:
      R = R @ axis_angle_rotation(local_right, (-1 if args.invert_y else 1) * TURN_SPEED * dt)

    # --- movement (WASD + Q/E) ---
    right = R[:, 0]
    forward = R[:, 2]

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
      position += (move / norm) * args.move_speed * dt

    c2w = build_c2w(position, R)

    with torch.no_grad():
      image = engine.sample(c2w, K, WIDTH, HEIGHT)

    image_np = (image.cpu().numpy() * 255).astype(np.uint8)

    surf = pygame.surfarray.make_surface(image_np.transpose(1, 0, 2))
    screen.blit(surf, (0, 0))

    if recording:
      pygame.draw.circle(screen, (255, 0, 0), (20, 20), 8)
      if time.time() - last_record_time >= record_tick:
        last_record_time = time.time()
        path = os.path.join(args.record_dir, "marked", f"frame_{frame_idx:05d}.png")
        save_queue.put((path, image_np.copy()))

        with torch.no_grad():
          image = engine.sample(c2w, K, WIDTH, HEIGHT, bg_color=unmarked_bg_color)
        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        path = os.path.join(args.record_dir, "unmarked", f"frame_{frame_idx:05d}.png")
        save_queue.put((path, image_np.copy()))
        frame_idx += 1

    pygame.display.flip()

  if recording:
    print("Recording: OFF")

  pygame.quit()
  save_queue.put(None)
  print("Waiting for save queue to finish...")
  save_thread.join()
  print(f"Saved {frame_idx} frames to {args.record_dir}")


if __name__ == "__main__":
  main()