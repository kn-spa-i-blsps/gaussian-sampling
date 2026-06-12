import gsplat
import numpy as np
import math
import torch
from gsplat import rasterization
from plyfile import PlyData

bg_pink = torch.tensor([1.0, 0.0, 1.0], device="cuda", dtype=torch.float32).unsqueeze(0)

def load_ply_to_splats(path, device="cuda"):
  plydata = PlyData.read(path)
  v = plydata["vertex"]

  means = np.stack([v["x"], v["y"], v["z"]], axis=-1)
  opacities = v["opacity"][:, None]
  scales = np.stack([v[f"scale_{i}"] for i in range(3)], axis=-1)
  quats = np.stack([v[f"rot_{i}"] for i in range(4)], axis=-1)

  sh0 = np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=-1)[:, None, :]

  # higher-order SH coeffs, if present
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
    "scales": to_t(scales),       # log-space, as in the original PLY
    "opacities": to_t(opacities).squeeze(-1),  # logit-space
    "sh0": to_t(sh0),
    "shN": to_t(shN),
  }
class SamplingEngine:
  def __init__(self, ply_path: str):
    self.splats = load_ply_to_splats(ply_path)

  def sample(self, c2w, K, width, height, bg_color=bg_pink):
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

    colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=-2)
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

    render_colors, _, _ = rasterization(
      self.splats["means"],
      torch.nn.functional.normalize(self.splats["quats"], p=2, dim=-1),
      self.splats["scales"].exp(),
      self.splats["opacities"].sigmoid(),
      colors,
      viewmat[None],
      K[None],
      width,
      height,
      packed=False,
      sh_degree=sh_degree,
      backgrounds=bg_color,
    )
    return render_colors[0, ..., :3].clamp(0, 1)

