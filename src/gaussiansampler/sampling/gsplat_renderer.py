import math

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

        # This is needed to force JIT compilation of gsplat kernels
        # and warm up CUDA so that later rasterization runs smoothly.
        self._warmup()

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

    def render(self, c2w, K, width, height, bg_color=bg_pink):
        """
        c2w: (4,4) camera-to-world matrix
        K:   (3,3) intrinsics matrix
        self.splats: dict with means, quats, scales, opacities, sh0, shN
                    (as returned by load_ply_to_splats)
        """
        device = self.splats["means"].device
        c2w = torch.as_tensor(c2w, dtype=torch.float32, device=device)
        K = torch.as_tensor(K, dtype=torch.float32, device=device)
        viewmat = c2w.inverse()

        render_colors, _, _ = rasterization(
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
        )
        return render_colors[0, ..., :3].clamp(0, 1)

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
