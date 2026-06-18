"""
Interactive viewer for a Gaussian Splat scene rendered with gsplat.

Controls:
  W / A / S / D   - move forward / left / back / right
  Q / E           - move down / up
  Arrow keys      - look around (Left/Right = yaw, Up/Down = pitch)
  SPACE           - toggle frame recording
  ESC             - quit
"""

import os
import math
import numpy as np
import torch
import time
import argparse
from PIL import Image

from sampling.sampling_engine import SamplingEngine

import threading
import queue

save_queue = queue.Queue()

# Black background for unmarked frames
unmarked_bg_color = torch.tensor([0.0, 0.0, 0.0], device="cuda", dtype=torch.float32).unsqueeze(0)

# Uses a 'None' sentinel value for shutdown
def save_worker():
  while True:
    item = save_queue.get()
    if item is None:
      break
    path, arr = item
    Image.fromarray(arr).save(path)
    save_queue.task_done()

# This can be configured later, should be kept in a file or something
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
      [ x * x * C + c,       x * y * C - z * s,  x * z * C + y * s ],
      [ y * x * C + z * s,   y * y * C + c,      y * z * C - x * s ],
      [ z * x * C - y * s,   z * y * C + x * s,  z * z * C + c     ],
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


def main(args):
  # Importing pygame initializes SDL so we don't want to do it before parsing args
  import pygame

  for i in range(args.workers):
    save_thread = threading.Thread(target=save_worker, daemon=True)
    save_thread.start()

  record_tick = 1.0 / args.record_fps
  last_record_time = time.time()

  pygame.init()
  screen = pygame.display.set_mode((args.width, args.height))
  pygame.display.set_caption("Gaussian Splat Viewer")
  clock = pygame.time.Clock()

  engine = SamplingEngine(args.splat_path)
  K = build_K(args.width, args.height, args.fov)

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
      R = axis_angle_rotation(WORLD_UP, -args.turn_speed * dt) @ R
    if keys[pygame.K_RIGHT]:
      R = axis_angle_rotation(WORLD_UP, args.turn_speed * dt) @ R

    # Up and down rotation (pitch)
    local_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if keys[pygame.K_UP]:
      R = R @ axis_angle_rotation(local_right, (-1 if args.invert_y else 1) * -args.turn_speed * dt)
    if keys[pygame.K_DOWN]:
      R = R @ axis_angle_rotation(local_right, (-1 if args.invert_y else 1) * args.turn_speed * dt)

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
      image = engine.sample(c2w, K, args.width, args.height)

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
          image = engine.sample(c2w, K, args.width, args.height, bg_color=unmarked_bg_color)
        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        path = os.path.join(args.record_dir, "unmarked", f"frame_{frame_idx:05d}.png")
        save_queue.put((path, image_np.copy()))
        frame_idx += 1

    pygame.display.flip()

  if recording:
    print("Recording: OFF")
  pygame.quit()

  for i in range(args.workers):
    save_queue.put(None)
  if frame_idx > 0:
    print("Waiting for save queue to finish...")
  save_thread.join()
  if frame_idx > 0:
    print(f"Saved {frame_idx} frames to {args.record_dir}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("splat_path", type=str,
                      help="Path to the .ply file.")
  parser.add_argument("--move_speed", type=float, default=2.0,
                      help="Camera movement speed.")
  parser.add_argument("--turn_speed", type=float, default=1.5,
                      help="Camera turn speed (radians per second).")
  parser.add_argument("--invert_y", type=bool, default=False,
                      help="Invert Y-axis rotation.")
  parser.add_argument("--record_fps", type=float, default=10.0,
                      help="Frames per second for recording.")
  parser.add_argument("--record_dir", type=str, default="recording",
                      help="Directory to save recorded frames.")
  parser.add_argument("-W", "--width", type=int, default=800,
                      help="Recording window width in pixels.")
  parser.add_argument("-H", "--height", type=int, default=600,
                      help="Recording window height in pixels.")
  parser.add_argument("--fov", type=float, default=60.0,
                      help="Camera field of view in degrees.")
  parser.add_argument("--workers", type=int, default=4,
                      help="Number of worker threads for recording.")
  args = parser.parse_args()

  if args.width <= 0 or args.height <= 0:
    print(f"Error, invalid window dimensions: {args.width} x {args.height}")
    exit(1)

  main(args)