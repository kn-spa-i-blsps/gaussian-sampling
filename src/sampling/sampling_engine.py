import numpy as np
import math
import torch
from plyfile import PlyData


def _default_device():
  if torch.cuda.is_available():
    return "cuda"
  if torch.backends.mps.is_available():
    return "mps"
  return "cpu"


def _t2mx(t):
  import mlx.core as mx
  return mx.array(t.cpu().numpy())


def _mx2t(a, device):
  import numpy as np
  return torch.from_numpy(np.array(a)).to(device)


bg_pink_np = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)


def load_ply_to_splats(path, device=None):
  if device is None:
    device = _default_device()
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
    self.colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=-2)
    self.sh_degree = int(math.sqrt(self.colors.shape[-2]) - 1)
    self.quats_normalized = torch.nn.functional.normalize(self.splats["quats"], p=2, dim=-1)
    self._use_mlx = not torch.cuda.is_available()
    if self._use_mlx:
      self._init_mlx_cache()

    # Force JIT / warmup so later renders run without a compilation stall.
    self._warmup()

  def _warmup(self, width=512, height=512, iters=2):
    device = self.splats["means"].device
    c2w = torch.eye(4, device=device, dtype=torch.float32)
    K = torch.tensor(
      [[width, 0, width / 2],
        [0, width, height / 2],
        [0, 0, 1]],
      device=device, dtype=torch.float32,
    )
    with torch.no_grad():
      for _ in range(iters):
        _ = self.sample(c2w, K, width, height)
    if torch.cuda.is_available():
      torch.cuda.synchronize()

  def _sample_cuda(self, c2w, K, width, height, bg_color):
    from gsplat import rasterization
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

  def _init_mlx_cache(self):
    import mlx.core as mx
    self._mx = {
      "means":     _t2mx(self.splats["means"]),
      "quats":     _t2mx(self.quats_normalized),
      "scales":    _t2mx(self.splats["scales"].exp()),
      "opacities": _t2mx(self.splats["opacities"].sigmoid()),
      "colors":    _t2mx(self.colors),
    }
    mx.eval(*self._mx.values())

  def _sample_mlx(self, c2w, K, width, height, bg_color_np):
    import mlx.core as mx
    from gsplat_mlx import rasterization

    viewmat = c2w.inverse()

    render_colors, _, _ = rasterization(
      self._mx["means"],
      self._mx["quats"],
      self._mx["scales"],
      self._mx["opacities"],
      self._mx["colors"],
      _t2mx(viewmat[None]),
      _t2mx(K[None]),
      width,
      height,
      sh_degree=self.sh_degree,
      backgrounds=mx.array(bg_color_np),
      differentiable=False,
    )
    mx.eval(render_colors)
    device = self.splats["means"].device
    return _mx2t(render_colors[0, ..., :3], device).clamp(0, 1)

  def sample(self, c2w, K, width, height, bg_color=None):
    """
    c2w: (4,4) camera-to-world matrix
    K:   (3,3) intrinsics matrix
    bg_color: (1,3) torch tensor (CUDA path) or None to use default pink
    """
    device = self.splats["means"].device
    c2w = torch.as_tensor(c2w, dtype=torch.float32, device=device)
    K = torch.as_tensor(K, dtype=torch.float32, device=device)

    if self._use_mlx:
      bg_np = bg_color.cpu().numpy() if bg_color is not None else bg_pink_np
      return self._sample_mlx(c2w, K, width, height, bg_np)
    else:
      if bg_color is None:
        bg_color = torch.tensor([[1.0, 0.0, 1.0]], device=device, dtype=torch.float32)
      return self._sample_cuda(c2w, K, width, height, bg_color)
