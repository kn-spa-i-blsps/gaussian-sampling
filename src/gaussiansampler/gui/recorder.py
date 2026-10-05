"""
Interactive viewer for a Gaussian Splat scene rendered with gsplat.

Controls:
  W / A / S / D        - move forward / left / back / right
  Q / E                - move down / up (camera-local, no world "up" needed)
  Left mouse + drag     - look around (arcball/trackball; supports free roll)
  Mouse wheel          - lens zoom in / out
  Middle mouse click   - reset zoom to 1.0x
  SPACE                - toggle frame recording
  ESC                  - quit

The top bar has two menu buttons ("Output" and "Rendering") that open forms
for adjusting recording/output and rendering settings live. Opening either
menu pauses rendering and recording until it is closed.
"""

import math
import os
import time
from types import SimpleNamespace

import numpy as np

from gaussiansampler.utils.colors import parse_color


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


def color_to_hex(color):
    """Convert an (R, G, B) tuple/sequence to a '#RRGGBB' string for display."""
    r, g, b = int(color[0]), int(color[1]), int(color[2])
    return f"#{r:02X}{g:02X}{b:02X}"


# The record accepts a renderer and an arguments object
# For more information about renderers refer to GsplatRenderer from the project as an example
def run(renderer, args, extension=None):
    """Run the shared recorder GUI.

    ``extension`` may customize camera initialization and frame rendering, and
    may contribute one additional settings menu. Its render and decoration
    hooks receive a snapshot of render state on the render worker thread.
    Ordinary recorder behavior remains here so extensions automatically
    receive future features.
    """
    # Imports happen here for extra snappiness when passing bad arguments
    import queue
    import threading

    import torch
    from PySide6 import QtCore, QtGui, QtWidgets

    class OutputMenu(QtWidgets.QWidget):
        """A dropdown-style form anchored under the 'Output' toolbar button."""

        def __init__(self, render_widget, parent=None):
            super().__init__(parent, QtCore.Qt.Popup)
            self.render_widget = render_widget

            layout = QtWidgets.QFormLayout(self)

            marked_row = QtWidgets.QHBoxLayout()
            self.marked_path_edit = QtWidgets.QLineEdit()
            self.marked_browse_btn = QtWidgets.QPushButton("...")
            self.marked_browse_btn.setFixedWidth(28)
            marked_row.addWidget(self.marked_path_edit)
            marked_row.addWidget(self.marked_browse_btn)
            layout.addRow("Marked path:", marked_row)

            unmarked_row = QtWidgets.QHBoxLayout()
            self.unmarked_path_edit = QtWidgets.QLineEdit()
            self.unmarked_browse_btn = QtWidgets.QPushButton("...")
            self.unmarked_browse_btn.setFixedWidth(28)
            unmarked_row.addWidget(self.unmarked_path_edit)
            unmarked_row.addWidget(self.unmarked_browse_btn)
            layout.addRow("Unmarked path:", unmarked_row)

            self.enable_marked_checkbox = QtWidgets.QCheckBox()
            layout.addRow("Enable marked:", self.enable_marked_checkbox)

            self.prefix_edit = QtWidgets.QLineEdit()
            layout.addRow("Prefix:", self.prefix_edit)

            self.frame_idx_edit = QtWidgets.QLineEdit()
            validator = QtGui.QIntValidator(0, 2_147_483_647, self)
            self.frame_idx_edit.setValidator(validator)
            layout.addRow("Next frame index:", self.frame_idx_edit)

            self.setLayout(layout)

            self.marked_browse_btn.clicked.connect(self._browse_marked)
            self.unmarked_browse_btn.clicked.connect(self._browse_unmarked)
            self.marked_path_edit.editingFinished.connect(self._apply_marked_path)
            self.unmarked_path_edit.editingFinished.connect(self._apply_unmarked_path)
            self.prefix_edit.returnPressed.connect(self._apply_prefix)
            self.frame_idx_edit.returnPressed.connect(self._apply_frame_idx)
            self.enable_marked_checkbox.stateChanged.connect(self._apply_enable_marked)

        def refresh_from_state(self):
            """Populate all fields with the render widget's current state."""
            rw = self.render_widget
            self.marked_path_edit.setText(rw.args.marked_dir)
            self.unmarked_path_edit.setText(rw.args.unmarked_dir)
            self.enable_marked_checkbox.blockSignals(True)
            self.enable_marked_checkbox.setChecked(rw.enable_marked)
            self.enable_marked_checkbox.blockSignals(False)
            self.prefix_edit.setText(rw.file_prefix)
            self.frame_idx_edit.setText(str(rw.frame_idx))

        def _browse_marked(self):
            start_dir = self.render_widget.args.marked_dir or os.getcwd()
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select marked output directory", start_dir
            )
            if path:
                self.marked_path_edit.setText(path)
                self._apply_marked_path()

        def _browse_unmarked(self):
            start_dir = self.render_widget.args.unmarked_dir or os.getcwd()
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select unmarked output directory", start_dir
            )
            if path:
                self.unmarked_path_edit.setText(path)
                self._apply_unmarked_path()

        def _apply_marked_path(self):
            path = self.marked_path_edit.text().strip()
            if path:
                self.render_widget.set_marked_dir(path)

        def _apply_unmarked_path(self):
            path = self.unmarked_path_edit.text().strip()
            if path:
                self.render_widget.set_unmarked_dir(path)

        def _apply_prefix(self):
            prefix = self.prefix_edit.text().strip()
            if prefix:
                self.render_widget.file_prefix = prefix

        def _apply_frame_idx(self):
            text = self.frame_idx_edit.text().strip()
            if text.isdigit():
                self.render_widget.frame_idx = int(text)

        def _apply_enable_marked(self, _state):
            self.render_widget.enable_marked = self.enable_marked_checkbox.isChecked()

    class ControlsMenu(QtWidgets.QWidget):
        """A dropdown-style form for camera movement controls."""

        def __init__(self, render_widget, parent=None):
            super().__init__(parent, QtCore.Qt.Popup)
            self.render_widget = render_widget

            layout = QtWidgets.QVBoxLayout(self)

            numeric_form = QtWidgets.QFormLayout()
            self.move_speed_edit = QtWidgets.QLineEdit()
            self.move_speed_edit.setValidator(
                QtGui.QDoubleValidator(0.0001, 1_000_000.0, 6, self)
            )
            numeric_form.addRow("Movement speed:", self.move_speed_edit)

            self.look_sensitivity_edit = QtWidgets.QLineEdit()
            self.look_sensitivity_edit.setValidator(
                QtGui.QDoubleValidator(0.0001, 1_000_000.0, 6, self)
            )
            numeric_form.addRow("Rotation sensitivity:", self.look_sensitivity_edit)
            layout.addLayout(numeric_form)

            keyboard_controls = QtWidgets.QGroupBox("Keyboard controls")
            controls_form = QtWidgets.QFormLayout(keyboard_controls)
            controls_form.addRow("W / A / S / D:", QtWidgets.QLabel("Move"))
            controls_form.addRow("Q / E:", QtWidgets.QLabel("Move down / up"))
            controls_form.addRow("Left mouse + drag:", QtWidgets.QLabel("Look around"))
            controls_form.addRow("Mouse wheel:", QtWidgets.QLabel("Lens zoom"))
            controls_form.addRow("Middle click:", QtWidgets.QLabel("Reset zoom"))
            controls_form.addRow("Space:", QtWidgets.QLabel("Toggle recording"))
            controls_form.addRow("X:", QtWidgets.QLabel("Lock/unlock rotation"))
            controls_form.addRow("Esc:", QtWidgets.QLabel("Quit"))
            layout.addWidget(keyboard_controls)

            self.setLayout(layout)

            self.move_speed_edit.returnPressed.connect(self._apply_values)
            self.move_speed_edit.editingFinished.connect(self._apply_values)
            self.look_sensitivity_edit.returnPressed.connect(self._apply_values)
            self.look_sensitivity_edit.editingFinished.connect(self._apply_values)

        def refresh_from_state(self):
            """Populate fields with current values without applying them."""
            args = self.render_widget.args
            self.move_speed_edit.setText(str(args.move_speed))
            self.look_sensitivity_edit.setText(str(args.look_sensitivity))

        def _apply_values(self):
            try:
                move_speed = float(self.move_speed_edit.text().strip())
                look_sensitivity = float(self.look_sensitivity_edit.text().strip())
            except ValueError:
                return
            if move_speed <= 0 or look_sensitivity <= 0:
                return

            args = self.render_widget.args
            args.move_speed = move_speed
            args.look_sensitivity = look_sensitivity

    class RenderingMenu(QtWidgets.QWidget):
        def __init__(self, render_widget, parent=None):
            super().__init__(parent, QtCore.Qt.Popup)
            self.render_widget = render_widget
            self._color_dialog_open = False

            outer = QtWidgets.QVBoxLayout(self)

            top_form = QtWidgets.QFormLayout()
            self.record_fps_edit = QtWidgets.QLineEdit()
            self.record_fps_edit.setValidator(
                QtGui.QDoubleValidator(0.0001, 1_000_000.0, 6, self)
            )
            top_form.addRow("Record FPS:", self.record_fps_edit)

            self.render_fps_edit = QtWidgets.QLineEdit()
            self.render_fps_edit.setValidator(
                QtGui.QDoubleValidator(0.0001, 1_000_000.0, 6, self)
            )
            top_form.addRow("Render FPS:", self.render_fps_edit)
            outer.addLayout(top_form)

            subform_box = QtWidgets.QGroupBox("Resolution")
            subform = QtWidgets.QFormLayout(subform_box)

            self.width_edit = QtWidgets.QLineEdit()
            self.width_edit.setValidator(QtGui.QIntValidator(1, 100_000, self))
            subform.addRow("Width:", self.width_edit)

            self.height_edit = QtWidgets.QLineEdit()
            self.height_edit.setValidator(QtGui.QIntValidator(1, 100_000, self))
            subform.addRow("Height:", self.height_edit)

            self.scale_edit = QtWidgets.QLineEdit()
            self.scale_edit.setValidator(
                QtGui.QDoubleValidator(0.0001, 1000.0, 6, self)
            )
            subform.addRow("Scale:", self.scale_edit)

            self.resolution_submit_btn = QtWidgets.QPushButton("Apply")
            subform.addRow(self.resolution_submit_btn)

            subform_box.setLayout(subform)
            outer.addWidget(subform_box)

            color_form = QtWidgets.QFormLayout()
            marked_color_row, self.marked_color_edit, self.marked_color_picker = (
                self._make_color_row()
            )
            color_form.addRow("Marked color:", marked_color_row)
            unmarked_color_row, self.unmarked_color_edit, self.unmarked_color_picker = (
                self._make_color_row()
            )
            color_form.addRow("Unmarked color:", unmarked_color_row)
            outer.addLayout(color_form)

            self.setLayout(outer)

            # Apply on Enter or when focus leaves the field.  Requiring Enter
            # alone made it easy to record at the previous FPS accidentally.
            self.record_fps_edit.editingFinished.connect(self._apply_record_fps)
            self.render_fps_edit.editingFinished.connect(self._apply_render_fps)
            self.resolution_submit_btn.clicked.connect(self._apply_resolution)
            self.marked_color_edit.returnPressed.connect(self._apply_marked_color)
            self.unmarked_color_edit.returnPressed.connect(self._apply_unmarked_color)
            self.marked_color_picker.clicked.connect(self._pick_marked_color)
            self.unmarked_color_picker.clicked.connect(self._pick_unmarked_color)

        def _make_color_row(self):
            row = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText("#RRGGBB")
            picker = QtWidgets.QPushButton("Pick…")
            picker.setToolTip("Choose a color")
            layout.addWidget(edit)
            layout.addWidget(picker)
            return row, edit, picker

        def _pick_color(self, edit, apply):
            try:
                initial = QtGui.QColor(edit.text().strip())
                if not initial.isValid():
                    initial = QtGui.QColor("#FF00FF")
            except (TypeError, ValueError):
                initial = QtGui.QColor("#FF00FF")

            self._color_dialog_open = True
            self.render_widget.set_paused(True)
            parent_window = self.window()
            dialog = QtWidgets.QColorDialog(initial, parent_window)
            dialog.setWindowTitle("Choose color")
            dialog.setOption(QtWidgets.QColorDialog.ShowAlphaChannel, True)
            dialog.setWindowModality(QtCore.Qt.ApplicationModal)

            dialog.adjustSize()
            if parent_window is not None:
                center = parent_window.frameGeometry().center()
                dialog.move(center - dialog.frameGeometry().center())

            try:
                if dialog.exec() == QtWidgets.QDialog.Accepted:
                    color = dialog.currentColor()
                    edit.setText(color.name().upper())
                    apply()
            finally:
                self._color_dialog_open = False
                # The popup is normally hidden when the dialog opens. If it
                # remains visible, keep the viewer paused until it is closed.
                if not self.isVisible():
                    self.render_widget.set_paused(False)

        def _pick_marked_color(self):
            self._pick_color(self.marked_color_edit, self._apply_marked_color)

        def _pick_unmarked_color(self):
            self._pick_color(self.unmarked_color_edit, self._apply_unmarked_color)

        def refresh_from_state(self):
            rw = self.render_widget
            self.record_fps_edit.setText(str(rw.args.record_fps))
            self.render_fps_edit.setText(str(rw.args.render_fps))
            self.width_edit.setText(str(rw.args.width))
            self.height_edit.setText(str(rw.args.height))
            self.scale_edit.setText(str(rw.args.window_scale))
            self.marked_color_edit.setText(color_to_hex(rw.args.mark_color))
            self.unmarked_color_edit.setText(color_to_hex(rw.args.unmarked_color))

        def _apply_record_fps(self):
            text = self.record_fps_edit.text().strip()
            try:
                fps = float(text)
            except ValueError:
                return
            if fps > 0:
                self.render_widget.set_record_fps(fps)

        def _apply_render_fps(self):
            text = self.render_fps_edit.text().strip()
            try:
                fps = float(text)
            except ValueError:
                return
            if fps > 0:
                self.render_widget.set_render_fps(fps)

        def _apply_resolution(self):
            w_text = self.width_edit.text().strip()
            h_text = self.height_edit.text().strip()
            s_text = self.scale_edit.text().strip()
            if not (w_text and h_text and s_text):
                return
            try:
                width = int(w_text)
                height = int(h_text)
                scale = float(s_text)
            except ValueError:
                return
            if width <= 0 or height <= 0 or scale <= 0:
                return
            self.render_widget.set_resolution(width, height, scale)

        def _apply_marked_color(self):
            text = self.marked_color_edit.text().strip()
            try:
                color = parse_color(text)
            except ValueError:
                return
            self.render_widget.set_mark_color(color)

        def _apply_unmarked_color(self):
            text = self.unmarked_color_edit.text().strip()
            try:
                color = parse_color(text)
            except ValueError:
                return
            self.render_widget.set_unmarked_color(color)

    class RenderWorker(QtCore.QObject):
        render_requested = QtCore.Signal(object)
        frame_ready = QtCore.Signal(object)

        def __init__(self, renderer, save_queue):
            super().__init__()
            self.renderer = renderer
            self.save_queue = save_queue
            self.render_requested.connect(
                self.render, QtCore.Qt.ConnectionType.QueuedConnection
            )

        @QtCore.Slot(object)
        def render(self, request):
            try:
                c2w = request["c2w"]
                context = request["context"]
                with torch.no_grad():
                    if extension is not None:
                        image = extension.render(context, c2w, request["mark_color"])
                    else:
                        image = self.renderer.render(
                            c2w,
                            request["K"],
                            request["width"],
                            request["height"],
                            bg_color=request["mark_color"],
                        )
                image_np = (image.detach().cpu().numpy() * 255).astype(np.uint8)
                display_np = image_np.copy()
                if extension is not None:
                    extension.decorate_image(context, display_np, c2w)

                if request["unmarked_path"] is not None:
                    with torch.no_grad():
                        if extension is not None:
                            unmarked = extension.render(
                                context, c2w, request["unmarked_color"]
                            )
                        else:
                            unmarked = self.renderer.render(
                                c2w,
                                request["K"],
                                request["width"],
                                request["height"],
                                bg_color=request["unmarked_color"],
                            )
                    unmarked_np = (unmarked.detach().cpu().numpy() * 255).astype(
                        np.uint8
                    )
                    if request["marked_path"] is not None:
                        self.save_queue.put((request["marked_path"], image_np))
                    self.save_queue.put((request["unmarked_path"], unmarked_np))

                self.frame_ready.emit((display_np, None))
            except (RuntimeError, ValueError, TypeError) as error:
                message = f"{type(error).__name__}: {error}"
                self.frame_ready.emit((None, message))

    class RenderWidget(QtWidgets.QWidget):
        """Owns camera state and UI input; renders asynchronously."""

        def __init__(self, renderer, args, save_queue, parent=None):
            super().__init__(parent)
            self.renderer = renderer
            self.args = args
            self.save_queue = save_queue
            self.locked_R = None

            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.setMinimumSize(1, 1)

            # Output-related state (mirrors args.record_dir / args.mark_color
            # but kept mutable and independent so the menus can update them
            # live without needing to reach back through argparse).
            self.args.marked_dir = os.path.join(args.record_dir, "marked")
            self.args.unmarked_dir = os.path.join(args.record_dir, "unmarked")
            self.args.unmarked_color = (0, 0, 0)
            self.enable_marked = True
            self.file_prefix = "image"

            self.mark_color = self._make_color_tensor(args.mark_color, torch)
            self.unmarked_bg_color = self._make_color_tensor(
                self.args.unmarked_color, torch
            )

            self.zoom_factor = 1.0
            self.target_zoom_factor = 1.0
            self.pending_wheel_steps = 0.0
            self.K = build_K(args.width, args.height, args.fov)
            self.position = np.array([0.0, 0.0, -3.0], dtype=np.float32)
            self.R = initial_rotation()

            if extension is not None:
                extension.initialize(self)

            self.pressed_keys = set()
            self.dragging = False
            self.last_arcball_vec = None

            self.recording = False
            self.paused = False
            self.frame_idx = 0
            os.makedirs(self.args.marked_dir, exist_ok=True)
            os.makedirs(self.args.unmarked_dir, exist_ok=True)

            self.record_tick = 1.0 / args.record_fps
            self.last_record_time = time.time()

            self._current_qimage = None
            self._last_tick_time = time.time()

            self.render_pending = False
            self.render_thread = QtCore.QThread(self)
            self.render_worker = RenderWorker(renderer, save_queue)
            self.render_worker.moveToThread(self.render_thread)
            self.render_worker.frame_ready.connect(self._on_render_finished)
            self.render_thread.finished.connect(self.render_worker.deleteLater)
            self.render_thread.start()

            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.on_tick)
            self.set_render_fps(args.render_fps)

        @staticmethod
        def _make_color_tensor(color, torch):
            return (
                torch.tensor(color, device="cuda", dtype=torch.float32).unsqueeze(0)
                / 255.0
            )

        # -- live setters used by the menus ---------------------------------
        def set_marked_dir(self, path):
            self.args.marked_dir = path
            os.makedirs(self.args.marked_dir, exist_ok=True)

        def set_unmarked_dir(self, path):
            self.args.unmarked_dir = path
            os.makedirs(self.args.unmarked_dir, exist_ok=True)

        def set_record_fps(self, fps):
            self.args.record_fps = fps
            self.record_tick = 1.0 / fps

        def set_render_fps(self, fps):
            self.args.render_fps = fps
            # QTimer accepts a millisecond interval.  A minimum of one avoids
            # turning very high target rates into a zero-interval busy timer.
            self.timer.start(max(1, round(1000 / fps)))

        def set_mark_color(self, color):
            self.args.mark_color = color
            self.mark_color = self._make_color_tensor(color, torch)

        def set_unmarked_color(self, color):
            self.args.unmarked_color = color
            self.unmarked_bg_color = self._make_color_tensor(color, torch)

        def set_resolution(self, width, height, scale):
            self.args.width = width
            self.args.height = height
            self.args.window_scale = scale
            self._update_intrinsics()
            # Mouse coordinates are interpreted in the displayed render
            # rectangle.  Changing its size while a drag is active would
            # leave last_arcball_vec in the coordinate system of the old
            # rectangle.
            self.reset_input_state()
            window = self.window()
            if window is not None:
                window.apply_window_size()

        def _update_intrinsics(self):
            self.K = build_K(self.args.width, self.args.height, self.args.fov)
            self.K[0, 0] *= self.zoom_factor
            self.K[1, 1] *= self.zoom_factor

        def set_zoom(self, zoom_factor, *, immediate=False):
            self.target_zoom_factor = min(20.0, max(0.1, zoom_factor))
            if immediate:
                self.pending_wheel_steps = 0.0
                self.zoom_factor = self.target_zoom_factor
                self._update_intrinsics()
            self.update()

        def _advance_zoom(self, dt):
            if self.zoom_factor == self.target_zoom_factor:
                return

            remaining = math.log(self.target_zoom_factor / self.zoom_factor)
            fraction = 1.0 - math.exp(-min(dt, 0.1) / 0.12)
            self.zoom_factor *= math.exp(remaining * fraction)
            if abs(math.log(self.target_zoom_factor / self.zoom_factor)) < 0.001:
                self.zoom_factor = self.target_zoom_factor
            self._update_intrinsics()

        # -- pause control ----------------------------------------------------
        def _on_render_finished(self, result):
            self.render_pending = False
            image_np, error = result
            if error is not None:
                print("Render failed:")
                print(error)
                return
            self._current_qimage = self._to_qimage(image_np)
            self.update()

        def shutdown(self):
            self.timer.stop()
            self.render_thread.quit()
            self.render_thread.wait()

        def set_paused(self, paused):
            """Pausing stops the render/recording tick but keeps the last
            rendered frame on screen and leaves input state untouched."""
            self.paused = paused
            if paused:
                self._last_tick_time = time.time()
            else:
                self._last_tick_time = time.time()
                self.last_record_time = time.time()

        # -- input events --------------------------------------------------
        def reset_input_state(self):
            self.pressed_keys.clear()
            self.dragging = False
            self.last_arcball_vec = None

        def keyPressEvent(self, event):
            if self.paused:
                return
            if event.isAutoRepeat():
                return
            if event.key() == QtCore.Qt.Key_Escape:
                self.window().close()
                return
            if event.key() == QtCore.Qt.Key_X:
                if self.locked_R is None:
                    self.locked_R = self.R
                else:
                    self.locked_R = None
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
            if self.paused:
                return
            if event.button() == QtCore.Qt.MiddleButton:
                self.set_zoom(1.0, immediate=True)
                event.accept()
                return
            if event.button() == QtCore.Qt.LeftButton:
                self.dragging = True
                pos = event.pos()
                render_w, render_h, offset_x, offset_y = self._render_rect()
                self.last_arcball_vec = screen_to_arcball_vector(
                    pos.x() - offset_x,
                    pos.y() - offset_y,
                    render_w,
                    render_h,
                    self.args.invert_y,
                )

        def mouseReleaseEvent(self, event):
            if event.button() == QtCore.Qt.LeftButton:
                self.dragging = False
                self.last_arcball_vec = None

        def wheelEvent(self, event):
            if self.paused:
                return
            delta = event.angleDelta().y()
            if delta:
                self.pending_wheel_steps += delta / 120.0
            else:
                pixel_delta = event.pixelDelta().y()
                if not pixel_delta:
                    return
                self.pending_wheel_steps += pixel_delta / 40.0
            event.accept()

        def mouseMoveEvent(self, event):
            if self.paused:
                return
            if self.dragging:
                pos = event.pos()
                render_w, render_h, offset_x, offset_y = self._render_rect()
                current_vec = screen_to_arcball_vector(
                    pos.x() - offset_x,
                    pos.y() - offset_y,
                    render_w,
                    render_h,
                    self.args.invert_y,
                )
                delta_R = rotation_between(
                    self.last_arcball_vec, current_vec, self.args.look_sensitivity
                )
                self.R = self.R @ delta_R
                self.last_arcball_vec = current_vec

        # -- layout helpers --------------------------------------------------
        def _render_rect(self):
            """Return the displayed render rectangle centered in this widget.

            Rendering happens at the recording resolution, but the image is
            displayed at the configured window scale.  Paint and arcball
            input must use this same rectangle so a mouse position refers to
            the same point in the image that the user sees.
            """
            render_w = max(1, round(self.args.width * self.args.window_scale))
            render_h = max(1, round(self.args.height * self.args.window_scale))
            offset_x = (self.width() - render_w) / 2.0
            offset_y = (self.height() - render_h) / 2.0
            return render_w, render_h, offset_x, offset_y

        # -- render loop ------------------------------------------------
        def on_tick(self):
            if self.paused:
                return

            now = time.time()
            dt = now - self._last_tick_time
            self._last_tick_time = now
            if self.pending_wheel_steps:
                zoom_step = 1.1**self.pending_wheel_steps
                self.target_zoom_factor = min(
                    20.0, max(0.1, self.target_zoom_factor * zoom_step)
                )
                self.pending_wheel_steps = 0.0
            self._advance_zoom(dt)

            curR = self.R
            if self.locked_R is not None:
                curR = self.locked_R

            right = curR[:, 0]
            forward = curR[:, 2]
            local_up = -curR[:, 1]

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

            if not self.render_pending:
                c2w = build_c2w(self.position, self.R)
                K = self.K.copy()
                width = self.args.width
                height = self.args.height
                record_frame = (
                    self.recording and now - self.last_record_time >= self.record_tick
                )
                frame_idx = self.frame_idx
                prefix = self.file_prefix
                marked_path = None
                unmarked_path = None
                if record_frame:
                    self.last_record_time = now
                    if self.enable_marked:
                        marked_path = os.path.join(
                            self.args.marked_dir, f"{prefix}{frame_idx:05d}.png"
                        )
                    unmarked_path = os.path.join(
                        self.args.unmarked_dir, f"{prefix}{frame_idx:05d}.png"
                    )
                    self.frame_idx += 1

                context = SimpleNamespace(
                    renderer=self.renderer,
                    args=SimpleNamespace(**vars(self.args)),
                    K=K.copy(),
                    render_mode=getattr(self, "render_mode", 0),
                    show_cameras=getattr(self, "show_cameras", False),
                )
                request = {
                    "c2w": c2w,
                    "K": K,
                    "width": width,
                    "height": height,
                    "context": context,
                    "mark_color": self.mark_color,
                    "unmarked_color": self.unmarked_bg_color,
                    "marked_path": marked_path,
                    "unmarked_path": unmarked_path,
                }
                self.render_pending = True
                self.render_worker.render_requested.emit(request)

            self.update()

        @staticmethod
        def _to_qimage(image_np):
            h, w, _ = image_np.shape
            contiguous = np.ascontiguousarray(image_np)
            qimg = QtGui.QImage(
                contiguous.data, w, h, 3 * w, QtGui.QImage.Format_RGB888
            )
            return qimg.copy()

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.fillRect(self.rect(), QtGui.QColor(30, 30, 30))
            if self._current_qimage is not None:
                render_w, render_h, offset_x, offset_y = self._render_rect()
                target = QtCore.QRectF(offset_x, offset_y, render_w, render_h)
                painter.drawImage(target, self._current_qimage)
            if self.recording:
                fps_text = f"{self.args.record_fps:g}"
                badge_diameter = max(
                    24, painter.fontMetrics().horizontalAdvance(fps_text) + 12
                )
                badge = QtCore.QRectF(8, 8, badge_diameter, badge_diameter)
                painter.setBrush(QtGui.QColor(255, 0, 0))
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawEllipse(badge)
                painter.setPen(QtGui.QColor(255, 255, 255))
                painter.drawText(badge, QtCore.Qt.AlignCenter, fps_text)
            margin = 12
            padding_x = 8
            padding_y = 4
            zoom_text = f"{self.zoom_factor:.1f}x"
            zoom_size = painter.fontMetrics().size(0, zoom_text)
            zoom_badge = QtCore.QRectF(
                self.width() - zoom_size.width() - 2 * padding_x - margin,
                self.height() - zoom_size.height() - 2 * padding_y - margin,
                zoom_size.width() + 2 * padding_x,
                zoom_size.height() + 2 * padding_y,
            )
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(0, 0, 0, 170))
            painter.drawRoundedRect(zoom_badge, 4, 4)
            painter.setPen(QtGui.QColor(255, 255, 255))
            painter.drawText(zoom_badge, QtCore.Qt.AlignCenter, zoom_text)
            if self.locked_R is not None:
                margin = 12
                padding_x = 8
                padding_y = 4
                text = "Rotation locked"
                font = painter.font()
                font.setBold(True)
                painter.setFont(font)
                text_size = painter.fontMetrics().size(0, text)
                badge = QtCore.QRectF(
                    self.width() - text_size.width() - 2 * padding_x - margin,
                    margin,
                    text_size.width() + 2 * padding_x,
                    text_size.height() + 2 * padding_y,
                )
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(0, 0, 0, 170))
                painter.drawRoundedRect(badge, 4, 4)
                painter.setPen(QtGui.QColor(255, 255, 255))
                painter.drawText(
                    badge,
                    QtCore.Qt.AlignCenter,
                    text,
                )
            painter.end()

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self, renderer, args, save_queue):
            super().__init__()
            title = getattr(extension, "window_title", "Gaussian Splat Viewer")
            self.setWindowTitle(title)

            self.render_widget = RenderWidget(renderer, args, save_queue, self)
            self.setCentralWidget(self.render_widget)

            self.output_menu = OutputMenu(self.render_widget, self)
            self.rendering_menu = RenderingMenu(self.render_widget, self)
            self.controls_menu = ControlsMenu(self.render_widget, self)
            self.extension_menu = None
            if extension is not None:
                self.extension_menu = extension.create_menu(
                    self.render_widget, self, QtCore, QtGui, QtWidgets
                )

            self.toolbar = QtWidgets.QToolBar("Options", self)
            self.toolbar.setMovable(False)
            self.addToolBar(self.toolbar)

            self.output_btn = QtWidgets.QToolButton(self.toolbar)
            self.output_btn.setText("Output")
            self.output_btn.clicked.connect(
                lambda: self._toggle_menu(self.output_menu, self.output_btn)
            )
            self.toolbar.addWidget(self.output_btn)

            self.rendering_btn = QtWidgets.QToolButton(self.toolbar)
            self.rendering_btn.setText("Rendering")
            self.rendering_btn.clicked.connect(
                lambda: self._toggle_menu(self.rendering_menu, self.rendering_btn)
            )
            self.toolbar.addWidget(self.rendering_btn)

            self.controls_btn = QtWidgets.QToolButton(self.toolbar)
            self.controls_btn.setText("Controls")
            self.controls_btn.clicked.connect(
                lambda: self._toggle_menu(self.controls_menu, self.controls_btn)
            )
            self.toolbar.addWidget(self.controls_btn)

            if self.extension_menu is not None:
                self.extension_btn = QtWidgets.QToolButton(self.toolbar)
                self.extension_btn.setText(extension.menu_label)
                self.extension_btn.clicked.connect(
                    lambda: self._toggle_menu(self.extension_menu, self.extension_btn)
                )
                self.toolbar.addWidget(self.extension_btn)

            self.apply_window_size()

        def apply_window_size(self):
            args = self.render_widget.args
            window_w = max(1, round(args.width * args.window_scale))
            window_h = max(1, round(args.height * args.window_scale))
            self.resize(window_w, window_h)

        def _toggle_menu(self, menu, button):
            if menu.isVisible():
                menu.hide()
                return
            self.render_widget.reset_input_state()
            menu.refresh_from_state()
            self.render_widget.set_paused(True)
            self._connect_menu_close_once(menu)
            pos = button.mapToGlobal(QtCore.QPoint(0, button.height()))
            menu.move(pos)
            menu.show()
            menu.setFocus()

        def _connect_menu_close_once(self, menu):
            # QtCore.Qt.Popup widgets auto-hide on outside click / escape;
            # hook hideEvent to resume rendering exactly once per show.
            if getattr(menu, "_resume_hooked", False):
                return
            menu._resume_hooked = True
            original_hide_event = menu.hideEvent

            def hide_event(event):
                if not getattr(menu, "_color_dialog_open", False):
                    self.render_widget.set_paused(False)
                original_hide_event(event)

            menu.hideEvent = hide_event

        def closeEvent(self, event):
            self.render_widget.shutdown()
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
