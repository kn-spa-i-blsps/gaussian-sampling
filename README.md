# Gaussian Sampling & Water-Splatting Viewer

View Gaussian splats, render new images, and filter them by background color.
The water viewer adds a learned underwater medium.

## Install

Run from the project directory:

```bash
uv venv
uv pip install -e .
```

For the water viewer, also install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn):

```bash
uv pip install 'setuptools>=72,<=73' wheel ninja
uv pip install --no-build-isolation -e '.[water]'
```

This needs Git, an NVIDIA GPU and driver, a CUDA toolkit with `nvcc` matching
PyTorch's CUDA version, and a C++ compiler. Rebuild tiny-cuda-nn after changing
PyTorch or CUDA.

## Use

Open a splat and record images with a free camera:

```bash
uv run gs-record /path/to/splat.ply
```

Open the water viewer:

```bash
uv run gs-record-water /path/to/scene/splat.ply
```

Put the medium's `.ckpt` file next to the `.ply`. The viewer loads the first
checkpoint it finds. Use **Mode** to switch between the splat, water, and combined
render. Optional camera markers come from `colmap/sparse/0/` next to the splat.

To set defaults for either recorder, copy [recorder.example.conf](recorder.example.conf)
to `recorder.conf` in the directory you run the command from. Edit it to set camera
controls, resolution, recording FPS, and output settings. Command-line options
override these defaults.

Score rendered images by how much background color they contain:

```bash
uv run gs-score /path/to/images
```

The default background color is magenta. Use `--mark_color` to change it.
Run any command with `--help` for options.

See [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md) for attribution and research citations.
