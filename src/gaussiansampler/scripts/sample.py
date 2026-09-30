import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gaussiansampler.filtering.marked_scorer import MarkedScorer
from gaussiansampler.sampling.autosampler import AutoSampler
from gaussiansampler.sampling.gsplat_renderer import GsplatRenderer


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Warm up the automatic sampler, then save images from subsequent "
            "sampling steps."
        )
    )
    parser.add_argument("splat_path", type=Path, help="Path to the input .ply file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("samples"),
        help="Directory in which to save rendered images (default: samples).",
    )
    parser.add_argument(
        "--dry-run-steps",
        type=int,
        default=100,
        help="Number of sampler steps to run without saving images (default: 100).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
        help="Number of sampler steps to save images from (default: 50).",
    )
    parser.add_argument("-W", "--width", type=int, default=800)
    parser.add_argument("-H", "--height", type=int, default=600)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Marked-color sensitivity for the scorer (default: 0.01).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.99,
        help=(
            "Minimum scorer quality (higher means less mark-color coverage; "
            "default: 0.99)."
        ),
    )
    return parser


def build_intrinsics(width, height, fov_deg):
    focal_length = 0.5 * width / math.tan(math.radians(fov_deg) / 2)
    return np.array(
        [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def to_uint8_image(image):
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating) and image.max(initial=0) <= 1:
        return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.splat_path.is_file():
        raise SystemExit(f"Splat file does not exist: {args.splat_path}")
    if args.dry_run_steps < 0 or args.save_steps < 0:
        raise SystemExit("Step counts must be non-negative")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("Image dimensions must be positive")
    if not 0 < args.fov < 180:
        raise SystemExit("FOV must be between 0 and 180 degrees")
    if args.alpha <= 0:
        raise SystemExit("Alpha must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = GsplatRenderer(str(args.splat_path))
    scorer = MarkedScorer(alpha=args.alpha)
    sampler = AutoSampler(
        renderer,
        scorer,
        renderer.splats,
        width=args.width,
        height=args.height,
        fov_deg=args.fov,
        score_threshold=args.score_threshold,
    )
    intrinsics = build_intrinsics(args.width, args.height, args.fov)

    with torch.no_grad():
        for step in range(args.dry_run_steps):
            sampler.next_step()
            if (step + 1) % 100 == 0:
                print(f"Dry run: {step + 1}/{args.dry_run_steps} steps")

        scored_before = sampler.scored_count
        rejected_before = sampler.rejected_count
        image_count = 0
        for step in range(args.save_steps):
            poses = sampler.next_step()
            for pose in poses:
                image = renderer.render(pose, intrinsics, args.width, args.height)
                image_path = args.output_dir / f"sample_{image_count:05d}.png"
                Image.fromarray(to_uint8_image(image), mode="RGB").save(image_path)
                image_count += 1
            print(
                f"Sampling: {step + 1}/{args.save_steps} steps; "
                f"saved {image_count} images"
            )

    scored = sampler.scored_count - scored_before
    rejected = sampler.rejected_count - rejected_before
    rejected_percent = 100 * rejected / scored if scored else 0.0
    print(f"Saved {image_count} images to {args.output_dir}")
    print(f"Rejected by filter: {rejected}/{scored} ({rejected_percent:.1f}%)")


if __name__ == "__main__":
    main()
