import numpy as np
from PIL import Image

class MarkedScorer:
  MAX_DIST = np.sqrt(3 * 255**2)
  """
    mark_color:
      - Color given in RGB, used to mark holes in the image.
      - Should be consistent with what was used for gaussian splatting bg color.
      - The default mark_color is pink.
    alpha:
      - Controls the sensitivity of the scorer to colors that slightly differ
        from the mark color. With higher alphas the scorer scores colors similar to
        mark_color higher.
      - Should be in the range (0, inf).
    beta:
      - Controls the power mean, a higher beta increases the sensitivity to having any
        marked pixels in the image.
      - Should be in the range (0, inf), beta=1 is regular mean.
  """
  def __init__(self, alpha=0.001, beta=1.0, mark_color=np.array([255, 0, 255])):
    self._mark_color = mark_color
    self._alpha = alpha
    self._beta = beta

  def score_image(self, img):
    diff = img.astype(np.float32) - self._mark_color
    dist = np.linalg.norm(diff, axis=-1) / self.MAX_DIST
    pixel_scores = 1 - dist ** self._alpha
    return (pixel_scores ** self._beta).mean() ** (1 / self._beta)


  def score_files(self, img_paths):
    scores = []
    for path in img_paths:
      img = np.array(Image.open(path))
      scores.append((path, self.score_image(np.array(img))))
    return np.array(scores)