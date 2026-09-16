# Gaussian Sampling & Water-Splatting Viewer

A Python toolkit for Gaussian Splatting data augmentation and interactive
water-splatting rendering.

The project aims to streamline the process of data augumentation using gaussian splatting 
by automating sampling and image filtering.
## Installation
The best way is to install either through a conda or uv environment.

### Conda
Use this if you already have a conda environment for related projects with some of the heavier libraries installed, which might save you disk space, avoiding installing duplicate packages.
To install the package, run ```pip install -e .``` in the root directory of the project. \
If there are problems with 'backend that doesn't support PEP 660', consider updating pip with ```python -m install --upgrade pip```.

### uv
Best for keeping the project isolated from others, this is the easiest way.\
Just run the following commands in the root directory of the project:
```bash
uv venv
uv pip install -e .
```
And later run scripts like this:
```bash
uv run <script name>
```

## Scripts
The project includes the following scripts which are ready for practical use:

- `gs-record-water`: Opens the interactive Water-Splatting Viewer. It renders
  a `.ply` splat with a learned underwater medium, automatically loading the
  first `.ckpt` checkpoint and optional COLMAP camera data located alongside
  the splat.

  ```bash
  gs-record-water /path/to/scene/splat.ply
  ```

  Use the **Mode** menu to switch among the raw splat, the water medium alone,
  and the combined render. The combined modes require a `.ckpt` file in the
  same directory as the `.ply`; camera markers are loaded from
  `colmap/sparse/0/` when available.

- ```gs-record```: Allows exploring splats imported from .ply files
  and generating new images with a manual free camera.
- ```score.py```: Scores images based on the presence of a certain color.
  The intended for this is to render your splats with a very distinct background color
  which doesn't appear in the images themselves.
