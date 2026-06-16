# gaussian-sampling
A Python-based library for data augumentation using gaussian splatting.

The project aims to streamline the process of data augumentation using gaussian splatting 
by automating sampling and image filtering.
## Installation
The best way is to install either through a conda or uv environment.

To install the package, run ```pip install -e .``` in the root directory of the project. \
If there are problems with 'backend that doesn's support PEP 660', consider updating pip with ```python -m install --upgrade pip```

## Scripts
The project includes the following scripts which are ready for practical use:
- `record.py`: Allows exploring splats imported from .ply files
  and generating new images with a manual free camera.
- `score.py`: Scores images based on the presence of a certain color. 
  The intended for this is to render your splats with a very distinct background color
  which doesn't appear in the images themselves. 