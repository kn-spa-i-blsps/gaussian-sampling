"""Water-Splatting additions for the shared recorder GUI."""

import numpy as np
import torch


class WaterRecorderExtension:
    """Inject water rendering controls without forking the recorder."""

    window_title = "Underwater Gaussian Splat Viewer"
    menu_label = "Water"

    def initialize(self, render_widget):
        c2w, center, radius = render_widget.renderer.get_centered_c2w()
        render_widget.position = c2w[:3, 3].astype(np.float32)
        render_widget.R = c2w[:3, :3].astype(np.float32)
        render_widget.center_target = center
        render_widget.orbit_radius = radius
        render_widget.render_mode = 0
        render_widget.show_cameras = False

    def render(self, render_widget, c2w, bg_color):
        renderer = render_widget.renderer
        args = render_widget.args
        mode = render_widget.render_mode

        if mode == 0:
            return renderer.render(
                c2w,
                render_widget.K,
                args.width,
                args.height,
                bg_color=bg_color,
            )

        device = renderer.splats["means"].device
        c2w_t = torch.as_tensor(c2w, dtype=torch.float32, device=device)
        K_t = torch.as_tensor(render_widget.K, dtype=torch.float32, device=device)

        if mode == 1:
            medium_rgb, _, _ = renderer.water_renderer.get_medium_parameters(
                c2w_t, K_t, args.width, args.height
            )
            return medium_rgb.clamp(0.0, 1.0)

        black = torch.zeros(3, device=device)
        white = torch.ones(3, device=device)
        image_black, depth_expected = renderer.render_with_depth(
            c2w, render_widget.K, args.width, args.height, bg_color=black
        )
        image_white = renderer.render(
            c2w, render_widget.K, args.width, args.height, bg_color=white
        )

        alpha = 1.0 - (image_white - image_black).mean(dim=-1, keepdim=True).clamp(0, 1)
        true_depth = depth_expected / alpha.clamp(min=1e-5)
        true_depth = torch.where(alpha < 1e-3, 10.0, true_depth)

        medium_rgb, sigma_back, sigma_att = (
            renderer.water_renderer.get_medium_parameters(
                c2w_t, K_t, args.width, args.height
            )
        )
        transmission_att = torch.exp(-sigma_att * true_depth)
        transmission_back = torch.exp(-sigma_back * true_depth)
        object_color = image_black * transmission_att
        medium_color = medium_rgb * (1.0 - alpha * transmission_back)
        return (object_color + medium_color).clamp(0.0, 1.0)

    def decorate_image(self, render_widget, image, c2w):
        """Draw optional COLMAP camera markers on the display-only image."""
        cameras = render_widget.renderer.dataset_cameras
        if not render_widget.show_cameras or not len(cameras):
            return

        try:
            points_world = np.column_stack((cameras, np.ones(len(cameras))))
            points_camera = (np.linalg.inv(c2w) @ points_world.T).T
            points_camera = points_camera[points_camera[:, 2] > 1e-4]
            if not len(points_camera):
                return

            fx, fy = render_widget.K[0, 0], render_widget.K[1, 1]
            cx, cy = render_widget.K[0, 2], render_widget.K[1, 2]
            u = points_camera[:, 0] * fx / points_camera[:, 2] + cx
            v = points_camera[:, 1] * fy / points_camera[:, 2] + cy
            for x, y in zip(u, v):
                ix, iy = int(x), int(y)
                if 4 <= ix < image.shape[1] - 4 and 4 <= iy < image.shape[0] - 4:
                    image[iy - 4 : iy + 4, ix - 4 : ix + 4] = 0
        except (ValueError, np.linalg.LinAlgError):
            # Camera markers are optional and should never stop recording.
            return

    def create_menu(self, render_widget, parent, QtCore, QtGui, QtWidgets):
        class WaterMenu(QtWidgets.QWidget):
            def __init__(self):
                super().__init__(parent, QtCore.Qt.Popup)
                self.render_widget = render_widget
                layout = QtWidgets.QFormLayout(self)

                self.mode_combo = QtWidgets.QComboBox()
                self.mode_combo.addItems(
                    ["Raw splat", "Water medium", "Full Water-Splatting"]
                )
                self.mode_combo.currentIndexChanged.connect(self._set_mode)
                layout.addRow("Render mode:", self.mode_combo)

                self.cameras_checkbox = QtWidgets.QCheckBox()
                self.cameras_checkbox.stateChanged.connect(self._set_show_cameras)
                layout.addRow("Show COLMAP cameras:", self.cameras_checkbox)

            def _set_mode(self, index):
                if index and self.render_widget.renderer.water_renderer is None:
                    print(
                        "Water rendering requires a .ckpt file next to the splat file."
                    )
                    index = 0
                self.render_widget.render_mode = index
                self.mode_combo.blockSignals(True)
                self.mode_combo.setCurrentIndex(index)
                self.mode_combo.blockSignals(False)
                self.render_widget.update()

            def _set_show_cameras(self, _state):
                self.render_widget.show_cameras = self.cameras_checkbox.isChecked()
                self.render_widget.update()

            def refresh_from_state(self):
                self.mode_combo.blockSignals(True)
                self.mode_combo.setCurrentIndex(self.render_widget.render_mode)
                self.mode_combo.blockSignals(False)
                self.cameras_checkbox.blockSignals(True)
                self.cameras_checkbox.setChecked(self.render_widget.show_cameras)
                self.cameras_checkbox.blockSignals(False)

        return WaterMenu()
