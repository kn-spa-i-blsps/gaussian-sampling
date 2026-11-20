import glob
import json
import os

import numpy as np
import torch

from gaussiansampler.sampling.gsplat_renderer import GsplatRenderer
from gaussiansampler.sampling.water_medium_renderer import WaterMediumRenderer


class WaterGsplatRenderer(GsplatRenderer):
    """Regular splat renderer enriched with Water-Splatting environment data."""

    def __init__(self, ply_path: str):
        # Load the optional environment before the base class warms up.
        self.load_environment(ply_path)
        super().__init__(ply_path)

    def load_environment(self, ply_path):
        """Load COLMAP cameras and the Water-Splatting medium checkpoint."""
        base_dir = os.path.dirname(ply_path)
        self.water_renderer = None

        # Loading cameras
        self.dataset_cameras = []
        try:
            dp_json_path = os.path.join(base_dir, "dataparser_transforms.json")
            images_bin_path = os.path.join(
                base_dir, "colmap", "sparse", "0", "images.bin"
            )
            images_txt_path = os.path.join(
                base_dir, "colmap", "sparse", "0", "images.txt"
            )

            def qvec2rotmat(qvec):
                return np.array(
                    [
                        [
                            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
                        ],
                        [
                            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
                        ],
                        [
                            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
                        ],
                    ]
                )

            if os.path.exists(dp_json_path):
                if os.path.exists(images_bin_path) or os.path.exists(images_txt_path):
                    with open(dp_json_path, "r") as f:
                        dp = json.load(f)
                    ns_transform = np.array(dp["transform"])
                    ns_scale = dp["scale"]

                    from nerfstudio.data.utils.colmap_parsing_utils import (
                        read_images_binary,
                        read_images_text,
                    )

                    if os.path.exists(images_bin_path):
                        colmap_images = read_images_binary(images_bin_path)
                    else:
                        colmap_images = read_images_text(images_txt_path)

                    for img in colmap_images.values():
                        R = qvec2rotmat(img.qvec)
                        t = img.tvec
                        w2c = np.eye(4)
                        w2c[:3, :3] = R
                        w2c[:3, 3] = t

                        c2w_colmap = np.linalg.inv(w2c)
                        c2w_transformed = ns_transform @ c2w_colmap
                        c2w_transformed[:3, 3] *= ns_scale

                        self.dataset_cameras.append(c2w_transformed[:3, 3])

                    print(
                        f"{len(self.dataset_cameras)} cameras were calibrated and loaded."
                    )
                else:
                    print(
                        "There are no files “images.bin” or “images.txt” in the colmap/sparse/0/ folder."
                    )
            else:
                print(
                    "There is no file dataparser_transforms.json in the folder with splat.ply."
                )
        # COLMAP metadata is optional; retain the raw-splat viewer if it is
        # malformed or cannot be read by the installed Nerfstudio version.
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"Error during camera transformation: {e}")

        self.dataset_cameras = np.array(self.dataset_cameras)

        # Water rendering
        ckpts_in_folder = glob.glob(os.path.join(base_dir, "*.ckpt"))

        if ckpts_in_folder:
            ckpt_path = ckpts_in_folder[0]
            print(
                f"TCNN weights were loaded from the file {os.path.basename(ckpt_path)}"
            )
            self.water_renderer = WaterMediumRenderer(
                config={}, ckpt_path=ckpt_path
            ).cuda()
        else:
            print(
                f"\nNo .ckpt file in the folder {base_dir}; water modes are unavailable."
            )

    def _medium_parameters(self, c2w, K, width, height):
        device = self.splats["means"].device
        c2w = torch.as_tensor(c2w, dtype=torch.float32, device=device)
        K = torch.as_tensor(K, dtype=torch.float32, device=device)
        return self.water_renderer.get_medium_parameters(c2w, K, width, height)

    def render_medium(self, c2w, K, width, height, bg_color=GsplatRenderer.bg_pink):
        """Render only the learned water medium; background color is unused."""
        medium_rgb, _, _ = self._medium_parameters(c2w, K, width, height)
        return medium_rgb.clamp(0.0, 1.0)

    def render_underwater(self, c2w, K, width, height, bg_color=GsplatRenderer.bg_pink):
        """Composite the splat with water attenuation and backscatter.

        The medium supplies the background, so bg_color is unused.
        """
        device = self.splats["means"].device
        black = torch.zeros(3, device=device)
        white = torch.ones(3, device=device)
        image_black, depth_expected = self.render_with_depth(
            c2w, K, width, height, bg_color=black
        )
        image_white = self.render(c2w, K, width, height, bg_color=white)

        alpha = 1.0 - (image_white - image_black).mean(dim=-1, keepdim=True).clamp(0, 1)
        true_depth = depth_expected / alpha.clamp(min=1e-5)
        true_depth = torch.where(alpha < 1e-3, 10.0, true_depth)

        medium_rgb, sigma_back, sigma_att = self._medium_parameters(
            c2w, K, width, height
        )
        transmission_att = torch.exp(-sigma_att * true_depth)
        transmission_back = torch.exp(-sigma_back * true_depth)
        object_color = image_black * transmission_att
        medium_color = medium_rgb * (1.0 - alpha * transmission_back)
        return (object_color + medium_color).clamp(0.0, 1.0)

    def get_centered_c2w(self):
        """
        Calculates the camera's starting position. Uses the first Colmap camera loaded.
        """
        means = self.splats["means"]

        center = torch.mean(means, dim=0).cpu().numpy()
        dists = torch.norm(means - torch.tensor(center, device=means.device), dim=-1)
        radius = torch.quantile(dists, 0.75).item() * 2.0

        if radius < 0.1:
            radius = 1.0

        if hasattr(self, "dataset_cameras") and len(self.dataset_cameras) > 0:
            cam_pos = np.array(self.dataset_cameras[0])
        else:
            cam_pos = center + np.array([0.0, -radius * 0.2, -radius])

        forward = center - cam_pos

        norm_forward = np.linalg.norm(forward)
        if norm_forward < 1e-6:
            forward = np.array([0.0, 0.0, 1.0])
        else:
            forward = forward / norm_forward

        up_approx = np.array([0.0, -1.0, 0.0])

        right = np.cross(forward, up_approx)
        norm_right = np.linalg.norm(right)
        if norm_right < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / norm_right

        up_true = np.cross(right, forward)
        up_true = up_true / np.linalg.norm(up_true)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up_true
        c2w[:3, 2] = forward
        c2w[:3, 3] = cam_pos

        return c2w, center, radius
