from filtering.marked_scorer import MarkedScorer
from utils.colors import parse_color
from pathlib import Path
import argparse


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Score marked images in a directory.")
  parser.add_argument("dir",
                      help="Directory containing marked images.")
  parser.add_argument("--alpha", type=float, default=0.01,
                      help="Alpha parameter for the scorer.")
  parser.add_argument("--beta", type=float, default=2.0,
                      help="Beta parameter for the scorer.")
  parser.add_argument("--workers", type=int, default=-1,
                      help="Number of worker threads to use for scoring.")
  parser.add_argument("--mark_color", type=parse_color, default=(255, 0, 255),
                      help="Color for marking recorded frames (RGB).")

  args = parser.parse_args()

  scorer = MarkedScorer(alpha=args.alpha, beta=args.beta, max_workers=args.workers,
                       mark_color=args.mark_color)
  if not Path(args.dir).is_dir():
    print(f"Error: {args.dir} is not a valid directory.")
    exit(1)
  files = list(Path(args.dir).glob("*.png"))
  results = scorer.score(files)
  for file, score in results:
    print(f"{file.name}: {score:.6f}")