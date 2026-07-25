"""
Interactive binary-search image filter.
Controls:
  Y  - this image is good enough (keep this and everything above)
  N  - this image is not good enough (cut higher)
  ESC / Q - quit without outputting results
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pygame
from PIL import Image

from gaussiansampler.filtering.marked_scorer import MarkedScorer
from gaussiansampler.filtering.score_filter import InteractiveScoreFilter
from gaussiansampler.utils.colors import parse_color


class PygameInteractor:
    def __init__(self, window_size=(900, 700)):
        pygame.init()
        self._screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Image Filter — Y = good  |  N = bad  |  Q = quit")
        self._font = pygame.font.SysFont("monospace", 18)
        self._window_size = window_size
        self._quit = False

    def ask(self, scored_item):
        """
        scored_item: (path, score) tuple as returned by MarkedScorer.score()
        Returns True  → image is good enough (keep this and above)
                False → image is not good enough (cut here)
        Raises SystemExit if the user quits.
        """
        path, score = scored_item
        img_surface = self._load_image(path)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._shutdown()
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        self._shutdown()
                    elif event.key == pygame.K_y:
                        return True
                    elif event.key == pygame.K_n:
                        return False

            self._draw(img_surface, path, score)
            pygame.display.flip()
            pygame.time.Clock().tick(60)

    def _load_image(self, path):
        pil_img = Image.open(path).convert("RGB")
        w, h = self._window_size
        # Reserve the bottom strip for the label bar
        bar_h = 60
        max_h = h - bar_h
        # Uniformly scale the image to be as large as possible while fitting
        # inside the area above the bar, preserving aspect ratio.
        iw, ih = pil_img.size
        scale = min(w / iw, max_h / ih)
        new_w = max(1, round(iw * scale))
        new_h = max(1, round(ih * scale))
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        arr = np.array(pil_img)
        return pygame.surfarray.make_surface(arr.transpose(1, 0, 2))

    def _draw(self, img_surface, path, score):
        self._screen.fill((30, 30, 30))

        sw, sh = self._window_size

        # Center the image within the area above the bottom bar.
        iw, ih = img_surface.get_size()
        x = (sw - iw) // 2
        y = (sh - 60 - ih) // 2
        self._screen.blit(img_surface, (x, y))

        # Bottom label bar
        pygame.draw.rect(self._screen, (20, 20, 20), (0, sh - 60, sw, 60))
        name_text = self._font.render(
            f"{Path(path).name}   score: {score:.4f}", True, (200, 200, 200)
        )
        hint_text = self._font.render(
            "Y = good enough    N = not good enough    Q = quit", True, (120, 120, 120)
        )
        self._screen.blit(name_text, (10, sh - 55))
        self._screen.blit(hint_text, (10, sh - 30))

    def _shutdown(self):
        pygame.quit()
        print("Aborted.")
        sys.exit(0)

    def close(self):
        pygame.quit()


def filter_main(args):
    img_dir = Path(args.img_dir)
    if not img_dir.is_dir():
        print(f"Error: {img_dir} is not a valid directory.")
        sys.exit(1)

    files = sorted(img_dir.glob("*.png"))
    if not files:
        print("No images found in the directory.")
        sys.exit(0)

    print(f"Scoring {len(files)} images...")
    scorer = MarkedScorer(
        alpha=args.alpha,
        beta=args.beta,
        max_workers=args.workers,
        mark_color=args.mark_color,
    )

    interactor = PygameInteractor()

    score_filter = InteractiveScoreFilter(scorer, interactor)

    print("Scoring completed.")
    kept = score_filter.filter(files)

    interactor.close()

    if not kept:
        print("No images passed the filter.")
    else:
        print(f"\n{len(kept)} image(s) kept")
        if args.verbose:
            print("Kept images:")
            for p in sorted(kept):
                print(p)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive image filter using binary search."
    )
    parser.add_argument(
        "img_dir", help="Directory containing PNG images that you want to filter."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Alpha parameter for MarkedScorer, controls how sensitive "
        "the scoring is to colors that are almost the mark color. "
        "Higher alpha means that colors further from the mark color "
        "will still be considered bad.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="Beta parameter for MarkedScorer, controls how much "
        "the scoring is affected by the presence of a few bad pixels. "
        "Higher beta means that even a few bad pixels will lower the score significantly.",
    )
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument(
        "--mark_color",
        type=parse_color,
        default=np.array([255, 0, 255]),
        help="Color for marking recorded frames (RGB).",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose output.",
    )
    args = parser.parse_args()

    filter_main(args)
