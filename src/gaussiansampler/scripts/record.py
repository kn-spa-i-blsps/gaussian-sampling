import argparse
import os
import shlex
from pathlib import Path

from gaussiansampler.utils.colors import parse_color

_RECORDER_CONFIG_TYPES = {
    "MOVE_SPEED": float,
    "INVERT_Y": "bool",
    "LOOK_SENSITIVITY": float,
    "RECORD_FPS": float,
    "RENDER_FPS": float,
    "RECORD_DIR": str,
    "WIDTH": int,
    "HEIGHT": int,
    "FOV": float,
    "WORKERS": int,
    "MARK_COLOR": parse_color,
    "ENABLE_MARKED": "bool",
    "WINDOW_SCALE": float,
}


def _parse_config_bool(value):
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("expected true/false, yes/no, on/off, or 1/0")


def load_recorder_config(path=None):
    """Read env-style defaults from recorder.conf in the current directory."""
    path = Path.cwd() / "recorder.conf" if path is None else Path(path)
    if not path.exists():
        return {}

    config = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        if "=" not in stripped:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")

        key, raw_value = stripped.split("=", 1)
        key = key.strip().upper()
        if key not in _RECORDER_CONFIG_TYPES:
            raise ValueError(f"{path}:{line_number}: unknown setting {key!r}")

        try:
            values = shlex.split(raw_value, comments=True)
            if len(values) != 1:
                raise ValueError("expected one value")
            value_type = _RECORDER_CONFIG_TYPES[key]
            config[key] = (
                _parse_config_bool(values[0])
                if value_type == "bool"
                else value_type(values[0])
            )
        except ValueError as error:
            raise ValueError(
                f"{path}:{line_number}: invalid value for {key}: {error}"
            ) from error

    return config


def build_parser():
    config = load_recorder_config()
    parser = argparse.ArgumentParser(
        epilog=(
            "Defaults may be set in recorder.conf in the current directory; "
            "explicit command-line options take precedence."
        )
    )
    parser.add_argument(
        "splat_path",
        type=str,
        help="Path to a .ply file or a directory containing splat.ply.",
    )
    parser.add_argument(
        "--move_speed",
        type=float,
        default=config.get("MOVE_SPEED", 2.0),
        help="Camera movement speed (config: MOVE_SPEED).",
    )
    parser.add_argument(
        "--invert_y",
        action=argparse.BooleanOptionalAction,
        default=config.get("INVERT_Y", False),
        help="Invert Y-axis rotation (config: INVERT_Y).",
    )
    parser.add_argument(
        "--look_sensitivity",
        type=float,
        default=config.get("LOOK_SENSITIVITY", 1.0),
        help="Multiplier applied to arcball rotation angle while dragging.",
    )
    parser.add_argument(
        "--record_fps",
        type=float,
        default=config.get("RECORD_FPS", 10.0),
        help="Frames per second for recording (config: RECORD_FPS).",
    )
    parser.add_argument(
        "--render_fps",
        type=float,
        default=config.get("RENDER_FPS", 60.0),
        help="Target frames per second for the interactive renderer (config: RENDER_FPS).",
    )
    parser.add_argument(
        "--record_dir",
        type=str,
        default=config.get("RECORD_DIR", "recording"),
        help="Directory to save recorded frames (config: RECORD_DIR).",
    )
    parser.add_argument(
        "-W",
        "--width",
        type=int,
        default=config.get("WIDTH", 800),
        help="Recording window width in pixels (config: WIDTH).",
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=config.get("HEIGHT", 600),
        help="Recording window height in pixels (config: HEIGHT).",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=config.get("FOV", 60.0),
        help="Camera field of view in degrees (config: FOV).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.get("WORKERS", 4),
        help="Number of worker threads for recording (config: WORKERS).",
    )
    parser.add_argument(
        "--mark_color",
        type=parse_color,
        default=config.get("MARK_COLOR", (255, 0, 255)),
        help="Color for marking recorded frames (RGB; config: MARK_COLOR).",
    )
    parser.add_argument(
        "--enable_marked",
        action=argparse.BooleanOptionalAction,
        default=config.get("ENABLE_MARKED", True),
        help="Save marked frames (config: ENABLE_MARKED).",
    )
    parser.add_argument(
        "--window_scale",
        type=float,
        default=config.get("WINDOW_SCALE", 1.0),
        help=(
            "Scale factor for the display window. The actual window will be "
            "(width*scale) x (height*scale), with the rendered output stretched "
            "to fit. Recording is still saved at the original (width, height) "
            "resolution (config: WINDOW_SCALE)."
        ),
    )
    return parser


def main(argv=None, *, renderer_factory=None, recorder_extension=None):
    """Run the recorder, optionally replacing its renderer and GUI extension.

    Water-Splatting uses these two extension points so it inherits the complete
    ``gs-record`` CLI and GUI instead of maintaining a fork of either one.
    """
    try:
        parser = build_parser()
    except ValueError as error:
        raise SystemExit(f"Invalid recorder.conf: {error}") from error
    args = parser.parse_args(argv)
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

    if os.path.isdir(args.splat_path):
        args.splat_path = os.path.join(args.splat_path, "splat.ply")

    if not os.path.isfile(args.splat_path):
        print(f"Error: splat file does not exist: {args.splat_path}")
        sys.exit(1)

    if renderer_factory is None:
        from gaussiansampler.sampling.gsplat_renderer import GsplatRenderer

        renderer_factory = GsplatRenderer

    from gaussiansampler.gui import recorder

    renderer = renderer_factory(args.splat_path)
    recorder.run(renderer, args, extension=recorder_extension)
