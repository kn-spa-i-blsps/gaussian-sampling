from filtering.marked_filter import MarkedScorer
from pathlib import Path
import argparse


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Score marked images in a directory.")
  parser.add_argument("dir", help="Directory containing marked images.")
  parser.add_argument("--alpha", type=float, default=0.001, help="Alpha parameter for the scorer.")
  parser.add_argument("--beta", type=float, default=1.0, help="Alpha parameter for the scorer.")

  args = parser.parse_args()

  scorer = MarkedScorer(alpha=args.alpha, beta=args.beta)
  files = list(Path(args.dir).glob("*.png"))
  results = scorer.score_files(files)
  sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
  for file, score in sorted_results:
    print(f"{file.name}: {score:.6f}")