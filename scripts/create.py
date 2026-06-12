import sampling.sampling_engine as sampling
import numpy as np
from PIL import Image


if __name__ == "__main__":
  engine = sampling.SamplingEngine("splat.ply")

  # Simple camera setup: looking at origin from z=5
  c2w = np.eye(4)
  c2w[:3, 3] = [0, 0, 5]  # camera position

  # Intrinsics: focal length 500, principal point at center
  width, height = 512, 512
  K = np.array([
    [500, 0, width / 2],
    [0, 500, height / 2],
    [0, 0, 1]
  ])

  # Sample and save
  image = engine.sample(c2w, K, width, height)
  image_np = (image.cpu().numpy() * 255).astype(np.uint8)
  Image.fromarray(image_np).save("sampled.png")
  print("Saved sampled.png")