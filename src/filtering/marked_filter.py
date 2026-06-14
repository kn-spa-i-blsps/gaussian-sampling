import numpy as np

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
  """
  def __init__(self, alpha=1, mark_color=np.array([255, 0, 255])):
    self._mark_color = mark_color
    self._alpha = alpha

  def score_image(self, img):
    diff = img.astype(np.float32) - self._mark_color
    dist = np.linalg.norm(diff, axis=-1) / self.MAX_DIST
    return (dist ** self._alpha).mean()
