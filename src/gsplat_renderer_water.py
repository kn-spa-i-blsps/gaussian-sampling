import glob
import json
import math
import os

import numpy as np
import torch
from gsplat import rasterization
from plyfile import PlyData


def load_ply_to_splats(path, device="cuda"):
    plydata = PlyData.read(path)
    v = plydata["vertex"]

    means = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    opacities = v["opacity"][:, None]
    scales = np.stack([v[f"scale_{i}"] for i in range(3)], axis=-1)
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], axis=-1)

    sh0 = np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=-1)[:, None, :]

    rest_keys = sorted(
        [n for n in v.data.dtype.names if n.startswith("f_rest_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    if rest_keys:
        shN = np.stack([v[k] for k in rest_keys], axis=-1)
        n_coeffs = shN.shape[-1] // 3
        shN = shN.reshape(-1, 3, n_coeffs).transpose(0, 2, 1)
    else:
        shN = np.zeros((means.shape[0], 0, 3), dtype=np.float32)

    def to_t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).float().to(device)

    return {
        "means": to_t(means),
        "quats": to_t(quats),
        "scales": to_t(scales),
        "opacities": to_t(opacities).squeeze(-1),
        "sh0": to_t(sh0),
        "shN": to_t(shN),
    }


class GsplatRenderer:
    bg_pink = torch.tensor(
        [1.0, 0.0, 1.0], device="cuda", dtype=torch.float32
    ).unsqueeze(0)

    def __init__(self, ply_path: str):
        self.splats = load_ply_to_splats(ply_path)
        self.colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=-2)
        self.sh_degree = int(math.sqrt(self.colors.shape[-2]) - 1)
        self.quats_normalized = torch.nn.functional.normalize(
            self.splats["quats"], p=2, dim=-1
        )

        self.load_environment(ply_path)

        # This is needed to force JIT compilation of gsplat kernels
        # and warm up CUDA so that later rasterization runs smoothly.
        self._warmup()

    def load_environment(self, ply_path):
        """Loading the Colmapa camera and the environment Water-Splatting."""
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
        from renderer_water import WaterMediumRenderer

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

    def _warmup(self, width=64, height=64, iters=2):
        device = self.splats["means"].device
        c2w = torch.eye(4, device=device, dtype=torch.float32)
        K = torch.tensor(
            [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]],
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            for _ in range(iters):
                _ = self.render(c2w, K, width, height)
        torch.cuda.synchronize()

    def render(self, c2w, K, width, height, bg_color=None):
        """
        c2w: (4,4) camera-to-world matrix
        K:   (3,3) intrinsics matrix
        self.splats: dict with means, quats, scales, opacities, sh0, shN
        """
        device = self.splats["means"].device
        c2w = torch.as_tensor(c2w, dtype=torch.float32, device=device)
        K = torch.as_tensor(K, dtype=torch.float32, device=device)
        viewmat = c2w.inverse()

        if bg_color is None:
            bg_color = self.bg_pink
        else:
            bg_color = torch.as_tensor(bg_color, dtype=torch.float32, device=device)
            if bg_color.dim() == 1:
                bg_color = bg_color.unsqueeze(0)

        render_out, _, _ = rasterization(
            self.splats["means"],
            self.quats_normalized,
            self.splats["scales"].exp(),
            self.splats["opacities"].sigmoid(),
            self.colors,
            viewmat[None],
            K[None],
            width,
            height,
            packed=False,
            sh_degree=self.sh_degree,
            backgrounds=bg_color,
            render_mode="RGB+ED",
        )
        rgb = render_out[0, ..., :3].clamp(0, 1)
        depth = render_out[0, ..., 3:4]
        depth = torch.where(depth == 0, torch.tensor(100.0, device=device), depth)

        return rgb, depth

    def approximate_world_up(self):
        means = self.splats["means"]

        if means.shape[0] < 3:
            return None

        mean_center = torch.mean(means, dim=0)
        centered = means - mean_center
        cov = torch.mm(centered.T, centered) / means.shape[0]
        _, eigenvectors = torch.linalg.eigh(cov)
        normal = eigenvectors[:, 0].cpu().numpy().astype(np.float32)
        if normal[1] < 0:
            normal = -normal

        return normal

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
