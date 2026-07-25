"""
Interactive viewer for a Gaussian Splat scene rendered with gsplat.

Controls:
  W / A / S / D        - move forward / left / back / right
  Q / E                - move down / up (camera-local, no world "up" needed)
  Left mouse + drag     - look around (arcball/trackball; supports free roll)
  SPACE                - toggle frame recording
  ESC                  - quit
"""

import argparse
import math
import os
import time

import numpy as np

from gaussiansampler.utils.colors import parse_color

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


def screen_to_arcball_vector(mx, my, width, height, invert_y=False):
    """Map a window-space mouse position to a point on the arcball hemisphere,
    expressed in camera-local coordinates (x=right, y=down, z=forward)."""
    radius = 0.5 * min(width, height)
    x = (mx - width / 2.0) / radius
    y = (my - height / 2.0) / radius
    if invert_y:
        y = -y
    d2 = x * x + y * y
    if d2 <= 0.5:
        z = math.sqrt(max(0.0, 1.0 - d2))
    else:
        z = 0.5 / math.sqrt(d2)
    # z here points toward the viewer; camera-local forward is the opposite direction.
    norm = math.sqrt(x * x + y * y + z * z)
    return np.array([x, y, -z], dtype=np.float32) / norm


def rotation_between(v_from, v_to, sensitivity=1.0):
    """Rotation matrix taking v_from to v_to, scaled by sensitivity."""
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    axis = np.cross(v_from, v_to)
    axis_len = np.linalg.norm(axis)
    dot = np.clip(np.dot(v_from, v_to), -1.0, 1.0)
    angle = math.acos(dot) * sensitivity
    if axis_len < 1e-8 or angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    return axis_angle_rotation(axis / axis_len, angle)


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


def save_worker(save_queue):
    from PIL import Image

    # Uses a 'None' sentinel value for shutdown.
    while True:
        item = save_queue.get()
        if item is None:
            break
        path, arr = item
        Image.fromarray(arr).save(path)
        save_queue.task_done()


def recorder_main(args):
    # Imports happen here for extra snappiness when passing bad arguments
    import queue
    import threading

    import pygame
    import torch

    from gaussiansampler.sampling.sampling_engine import SamplingEngine

    save_queue = queue.Queue()

    mark_color = (
        torch.tensor(args.mark_color, device="cuda", dtype=torch.float32).unsqueeze(0)
        / 255.0
    )
    # Black background for unmarked frames.
    unmarked_bg_color = torch.tensor(
        [0.0, 0.0, 0.0], device="cuda", dtype=torch.float32
    ).unsqueeze(0)

    workers = []
    for i in range(args.workers):
        save_thread = threading.Thread(
            target=save_worker, args=(save_queue,), daemon=True
        )
        save_thread.start()
        workers.append(save_thread)

    record_tick = 1.0 / args.record_fps
    last_record_time = time.time()

    pygame.init()
    window_w = max(1, round(args.width * args.window_scale))
    window_h = max(1, round(args.height * args.window_scale))
    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption("Gaussian Splat Viewer")
    clock = pygame.time.Clock()

    engine = SamplingEngine(args.splat_path)
    global WORLD_UP
    approx_up = engine.approximate_world_up()
    if approx_up is None:
        print(
            "Warning: could not determine world up vector from splat file, using default [0, 1, 0]"
        )
        WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        WORLD_UP = -approx_up
        print(f"World up vector: {WORLD_UP}")
    K = build_K(args.width, args.height, args.fov)

    position = np.array([0.0, 0.0, -3.0], dtype=np.float32)
    R = initial_rotation()  # camera-to-world rotation; columns = [right, down, forward]

    dragging = False
    last_arcball_vec = None

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
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    mx, my = event.pos
                    last_arcball_vec = screen_to_arcball_vector(
                        mx, my, window_w, window_h, args.invert_y
                    )
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
                    last_arcball_vec = None
            elif event.type == pygame.MOUSEMOTION and dragging:
                mx, my = event.pos
                current_vec = screen_to_arcball_vector(
                    mx, my, window_w, window_h, args.invert_y
                )
                delta_R = rotation_between(
                    last_arcball_vec, current_vec, args.look_sensitivity
                )
                R = R @ delta_R
                last_arcball_vec = current_vec

        keys = pygame.key.get_pressed()

        right = R[:, 0]
        forward = R[:, 2]
        local_up = -R[:, 1]

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
            move += local_up
        if keys[pygame.K_q]:
            move -= local_up

        norm = np.linalg.norm(move)
        if norm > 0:
            position += (move / norm) * args.move_speed * dt

        c2w = build_c2w(position, R)

        with torch.no_grad():
            image = engine.sample(c2w, K, args.width, args.height, bg_color=mark_color)

        image_np = (image.cpu().numpy() * 255).astype(np.uint8)

        surf = pygame.surfarray.make_surface(image_np.transpose(1, 0, 2))
        if surf.get_size() != (window_w, window_h):
            surf = pygame.transform.scale(surf, (window_w, window_h))
        screen.blit(surf, (0, 0))

        if recording:
            pygame.draw.circle(screen, (255, 0, 0), (20, 20), 8)
            if time.time() - last_record_time >= record_tick:
                last_record_time = time.time()
                path = os.path.join(
                    args.record_dir, "marked", f"frame_{frame_idx:05d}.png"
                )
                save_queue.put((path, image_np.copy()))

                with torch.no_grad():
                    image = engine.sample(
                        c2w, K, args.width, args.height, bg_color=unmarked_bg_color
                    )
                image_np = (image.cpu().numpy() * 255).astype(np.uint8)
                path = os.path.join(
                    args.record_dir, "unmarked", f"frame_{frame_idx:05d}.png"
                )
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
    for thread in workers:
        thread.join()
    if frame_idx > 0:
        print(f"Saved {frame_idx} frames to {args.record_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("splat_path", type=str, help="Path to the .ply file.")
    parser.add_argument(
        "--move_speed", type=float, default=2.0, help="Camera movement speed."
    )
    parser.add_argument(
        "--invert_y",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Invert Y-axis rotation.",
    )
    parser.add_argument(
        "--look_sensitivity",
        type=float,
        default=1.0,
        help="Multiplier applied to arcball rotation angle while dragging.",
    )
    parser.add_argument(
        "--record_fps",
        type=float,
        default=10.0,
        help="Frames per second for recording.",
    )
    parser.add_argument(
        "--record_dir",
        type=str,
        default="recording",
        help="Directory to save recorded frames.",
    )
    parser.add_argument(
        "-W", "--width", type=int, default=800, help="Recording window width in pixels."
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=600,
        help="Recording window height in pixels.",
    )
    parser.add_argument(
        "--fov", type=float, default=60.0, help="Camera field of view in degrees."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker threads for recording."
    )
    parser.add_argument(
        "--mark_color",
        type=parse_color,
        default=(255, 0, 255),
        help="Color for marking recorded frames (RGB).",
    )
    parser.add_argument(
        "--window_scale",
        type=float,
        default=1.0,
        help="Scale factor for the display window. The actual window will be (width*scale) x (height*scale), with the rendered output stretched to fit. Recording is still saved at the original (width, height) resolution.",
    )
    args = parser.parse_args()
    import sys

    if args.width <= 0 or args.height <= 0:
        print(f"Error: invalid window dimensions: {args.width} x {args.height}")
        sys.exit(1)

    if args.window_scale <= 0:
        print(f"Error: invalid window scale: {args.window_scale}")
        sys.exit(1)

    if not os.path.isfile(args.splat_path):
        print(f"Error: splat file does not exist: {args.splat_path}")
        sys.exit(1)

    recorder_main(args)
