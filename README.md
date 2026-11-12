# Gaussian Sampling & Water-Splatting Viewer

A Python toolkit for Gaussian Splatting data augmentation and interactive
water-splatting rendering.

See [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md) for required attribution and
research citation information.

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

### Water renderer: tiny-cuda-nn

The water medium renderer uses the PyTorch bindings of
[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn). Install the project
first using one of the methods above, then install the `water` extra in the
same environment. This builds a CUDA extension and requires Git, a CUDA-capable
NVIDIA GPU and driver, a CUDA toolkit with `nvcc`, and a compatible C++ build
toolchain. The CUDA toolkit must be compatible with the installed PyTorch
build. Check `nvcc --version` and `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`
before building.

With the conda environment activated:

```bash
python -m pip install --no-build-isolation -e '.[water]'
```

For the uv environment:

```bash
uv pip install --no-build-isolation -e '.[water]'
```

Build isolation is disabled because the extension's build needs the PyTorch
already installed in the target environment. The extra pins the tiny-cuda-nn
source revision found in the existing development environment; it does not
pin PyTorch, Nerfstudio, CUDA or their transitive dependencies. A matching
revision alone therefore does not guarantee an identical environment. Rebuild
tiny-cuda-nn after changing PyTorch or CUDA. To check the installation:

```bash
python -c "import torch, tinycudann; print('CUDA available:', torch.cuda.is_available()); print('tiny-cuda-nn:', tinycudann.__file__)"
```

For uv, run the check with `.venv/bin/python` (or activate `.venv` first).
An import check does not verify checkpoint compatibility or the rendered output.

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
