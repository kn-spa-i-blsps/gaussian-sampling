import argparse
import os

from gaussiansampler.utils.colors import parse_color


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("splat_path", type=str, help="Path to the .ply file.")
    parser.add_argument(
        "--move_speed", type=float, default=2.0, help="Camera movement speed."
    )
    parser.add_argument(
        "--invert_y",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Invert Y-axis rotation.",
    )
    parser.add_argument(
        "--look_sensitivity",
        type=float,
        default=1.0,
        help="Multiplier applied to arcball rotation angle while dragging.",
    )
    parser.add_argument(
        "--record_fps",
        type=float,
        default=10.0,
        help="Frames per second for recording.",
    )
    parser.add_argument(
        "--render_fps",
        type=float,
        default=60.0,
        help="Target frames per second for the interactive renderer.",
    )
    parser.add_argument(
        "--record_dir",
        type=str,
        default="recording",
        help="Directory to save recorded frames.",
    )
    parser.add_argument(
        "-W", "--width", type=int, default=800, help="Recording window width in pixels."
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=600,
        help="Recording window height in pixels.",
    )
    parser.add_argument(
        "--fov", type=float, default=60.0, help="Camera field of view in degrees."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker threads for recording."
    )
    parser.add_argument(
        "--mark_color",
        type=parse_color,
        default=(255, 0, 255),
        help="Color for marking recorded frames (RGB).",
    )
    parser.add_argument(
        "--window_scale",
        type=float,
        default=1.0,
        help="Scale factor for the display window. The actual window will be (width*scale) x (height*scale), with the rendered output stretched to fit. Recording is still saved at the original (width, height) resolution.",
    )
    return parser


def main(argv=None, *, renderer_factory=None, recorder_extension=None):
    """Run the recorder, optionally replacing its renderer and GUI extension.

    Water-Splatting uses these two extension points so it inherits the complete
    ``gs-record`` CLI and GUI instead of maintaining a fork of either one.
    """
    args = build_parser().parse_args(argv)
    import sys

    if args.width <= 0 or args.height <= 0:
        print(f"Error: invalid window dimensions: {args.width} x {args.height}")
        sys.exit(1)

    if args.window_scale <= 0:
        print(f"Error: invalid window scale: {args.window_scale}")
        sys.exit(1)

    if args.render_fps <= 0:
        print(f"Error: render FPS must be positive: {args.render_fps}")
        sys.exit(1)

    if not os.path.isfile(args.splat_path):
        print(f"Error: splat file does not exist: {args.splat_path}")
        sys.exit(1)

    if renderer_factory is None:
        from gaussiansampler.sampling.gsplat_renderer import GsplatRenderer

        renderer_factory = GsplatRenderer

    from gaussiansampler.gui import recorder

    renderer = renderer_factory(args.splat_path)
    recorder.run(renderer, args, extension=recorder_extension)
