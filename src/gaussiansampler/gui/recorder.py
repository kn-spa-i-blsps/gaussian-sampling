"""
Interactive viewer for a Gaussian Splat scene rendered with gsplat.

Controls:
  W / A / S / D        - move forward / left / back / right
  Q / E                - move down / up (camera-local, no world "up" needed)
  Left mouse + drag     - look around (arcball/trackball; supports free roll)
  SPACE                - toggle frame recording
  ESC                  - quit

This version uses PyQt5 (QMainWindow + QToolBar) instead of pygame, so a
toolbar with additional options/controls can be added later without
restructuring the render loop or event handling.
"""

import math
import os
import time

import numpy as np


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
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
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


# The record accepts a renderer and an arguments object
# For more information about renderers refer to GsplatRenderer from the project as an example
def run(renderer, args):
    # Imports happen here for extra snappiness when passing bad arguments
    import queue
    import threading

    import torch
    from PySide6 import QtCore, QtGui, QtWidgets

    class RenderWidget(QtWidgets.QWidget):
        """Owns the render loop, camera state, input handling, and recording."""

        def __init__(self, renderer, args, save_queue, parent=None):
            super().__init__(parent)
            self.renderer = renderer
            self.args = args
            self.save_queue = save_queue

            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.setMinimumSize(1, 1)

            self.mark_color = (
                torch.tensor(
                    args.mark_color, device="cuda", dtype=torch.float32
                ).unsqueeze(0)
                / 255.0
            )
            self.unmarked_bg_color = torch.tensor(
                [0.0, 0.0, 0.0], device="cuda", dtype=torch.float32
            ).unsqueeze(0)

            self.K = build_K(args.width, args.height, args.fov)
            self.position = np.array([0.0, 0.0, -3.0], dtype=np.float32)
            self.R = initial_rotation()

            self.pressed_keys = set()
            self.dragging = False
            self.last_arcball_vec = None

            self.recording = False
            self.frame_idx = 0
            os.makedirs(args.record_dir, exist_ok=True)
            os.makedirs(os.path.join(args.record_dir, "marked"), exist_ok=True)
            os.makedirs(os.path.join(args.record_dir, "unmarked"), exist_ok=True)

            self.record_tick = 1.0 / args.record_fps
            self.last_record_time = time.time()

            self._current_qimage = None
            self._last_tick_time = time.time()

            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.on_tick)
            self.timer.start(int(1000 / 60))

        # -- input events --------------------------------------------------
        def keyPressEvent(self, event):
            if event.isAutoRepeat():
                return
            if event.key() == QtCore.Qt.Key_Escape:
                self.window().close()
                return
            if event.key() == QtCore.Qt.Key_Space:
                self.recording = not self.recording
                print("Recording:", "ON" if self.recording else "OFF")
                return
            self.pressed_keys.add(event.key())

        def keyReleaseEvent(self, event):
            if event.isAutoRepeat():
                return
            self.pressed_keys.discard(event.key())

        def mousePressEvent(self, event):
            if event.button() == QtCore.Qt.LeftButton:
                self.dragging = True
                pos = event.pos()
                self.last_arcball_vec = screen_to_arcball_vector(
                    pos.x(), pos.y(), self.width(), self.height(), self.args.invert_y
                )

        def mouseReleaseEvent(self, event):
            if event.button() == QtCore.Qt.LeftButton:
                self.dragging = False
                self.last_arcball_vec = None

        def mouseMoveEvent(self, event):
            if self.dragging:
                pos = event.pos()
                current_vec = screen_to_arcball_vector(
                    pos.x(), pos.y(), self.width(), self.height(), self.args.invert_y
                )
                delta_R = rotation_between(
                    self.last_arcball_vec, current_vec, self.args.look_sensitivity
                )
                self.R = self.R @ delta_R
                self.last_arcball_vec = current_vec

        # -- render loop ------------------------------------------------
        def on_tick(self):
            now = time.time()
            dt = now - self._last_tick_time
            self._last_tick_time = now

            right = self.R[:, 0]
            forward = self.R[:, 2]
            local_up = -self.R[:, 1]

            move = np.zeros(3, dtype=np.float32)
            if QtCore.Qt.Key_W in self.pressed_keys:
                move += forward
            if QtCore.Qt.Key_S in self.pressed_keys:
                move -= forward
            if QtCore.Qt.Key_D in self.pressed_keys:
                move += right
            if QtCore.Qt.Key_A in self.pressed_keys:
                move -= right
            if QtCore.Qt.Key_E in self.pressed_keys:
                move += local_up
            if QtCore.Qt.Key_Q in self.pressed_keys:
                move -= local_up

            norm = np.linalg.norm(move)
            if norm > 0:
                self.position += (move / norm) * self.args.move_speed * dt

            c2w = build_c2w(self.position, self.R)

            with torch.no_grad():
                image = self.renderer.render(
                    c2w,
                    self.K,
                    self.args.width,
                    self.args.height,
                    bg_color=self.mark_color,
                )
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
            self._current_qimage = self._to_qimage(image_np)

            if self.recording and now - self.last_record_time >= self.record_tick:
                self.last_record_time = now
                path = os.path.join(
                    self.args.record_dir,
                    "marked",
                    f"frame_{self.frame_idx:05d}.png",
                )
                self.save_queue.put((path, image_np.copy()))

                with torch.no_grad():
                    image = self.renderer.render(
                        c2w,
                        self.K,
                        self.args.width,
                        self.args.height,
                        bg_color=self.unmarked_bg_color,
                    )
                unmarked_np = (image.cpu().numpy() * 255).astype(np.uint8)
                path = os.path.join(
                    self.args.record_dir,
                    "unmarked",
                    f"frame_{self.frame_idx:05d}.png",
                )
                self.save_queue.put((path, unmarked_np.copy()))
                self.frame_idx += 1

            self.update()

        @staticmethod
        def _to_qimage(image_np):
            h, w, _ = image_np.shape
            contiguous = np.ascontiguousarray(image_np)
            qimg = QtGui.QImage(
                contiguous.data, w, h, 3 * w, QtGui.QImage.Format_RGB888
            )
            # Copy so the QImage owns its buffer (contiguous is a local var).
            return qimg.copy()

        # -- painting -----------------------------------------------------
        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            if self._current_qimage is not None:
                target = self.rect()
                painter.drawImage(target, self._current_qimage)
            if self.recording:
                painter.setBrush(QtGui.QColor(255, 0, 0))
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawEllipse(QtCore.QPoint(20, 20), 8, 8)
            painter.end()

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self, renderer, args, save_queue):
            super().__init__()
            self.setWindowTitle("Gaussian Splat Viewer")

            # Empty toolbar for now; add QActions/widgets here later.
            self.toolbar = QtWidgets.QToolBar("Options", self)
            self.toolbar.setMovable(False)
            self.addToolBar(self.toolbar)

            self.render_widget = RenderWidget(renderer, args, save_queue, self)
            self.setCentralWidget(self.render_widget)

            window_w = max(1, round(args.width * args.window_scale))
            window_h = max(1, round(args.height * args.window_scale))
            self.resize(window_w, window_h)

        def closeEvent(self, event):
            self.render_widget.timer.stop()
            super().closeEvent(event)

    save_queue = queue.Queue()

    workers = []
    for i in range(args.workers):
        save_thread = threading.Thread(
            target=save_worker, args=(save_queue,), daemon=True
        )
        save_thread.start()
        workers.append(save_thread)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(renderer, args, save_queue)
    window.render_widget.setFocus()
    window.show()
    app.exec()

    frame_idx = window.render_widget.frame_idx
    recording = window.render_widget.recording
    if recording:
        print("Recording: OFF")

    for i in range(args.workers):
        save_queue.put(None)
    if frame_idx > 0:
        print("Waiting for save queue to finish...")
    for thread in workers:
        thread.join()
    if frame_idx > 0:
        print(f"Saved {frame_idx} frames to {args.record_dir}")
