from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

bg_pink = np.array([255, 0, 255])


class MarkedScorer:
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
      - Should be in the range (0, inf), beta=1 is regular, arithmetic mean.
    """

    def __init__(self, alpha=0.001, beta=1.0, mark_color=bg_pink, max_workers=1):
        self._mark_color = mark_color
        self._alpha = alpha
        self._beta = beta
        self._max_workers = max_workers

    def score_image(self, img):
        MAX_DIST = np.sqrt(3 * 255**2)
        diff = img.astype(np.float32) - self._mark_color
        dist = np.linalg.norm(diff, axis=-1) / MAX_DIST
        pixel_scores = 1 - dist**self._alpha
        return 1 - (pixel_scores**self._beta).mean() ** (1 / self._beta)

    def score_file(self, img_path):
        img = np.array(Image.open(img_path))
        return self.score_image(np.asarray(img))

    def score_files(self, img_paths):
        workers = None if self._max_workers == -1 else self._max_workers
        with ProcessPoolExecutor(max_workers=workers) as executor:
            scores = np.array(list(executor.map(self.score_file, img_paths)))
        return scores

    """
    This is the API-compliant scoring function.
    Returns a list of (object, score) tuples sorted by score in ascending order.
  """

    def score(self, paths):
        scores = zip(paths, self.score_files(paths))
        return sorted(scores, key=lambda x: x[1])
