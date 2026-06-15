# gaussian-sampling
A Python-based library for data augumentation using gaussian splatting.

The project aims to streamline the process of data augumentation using gaussian splatting 
by automating sampling and image filtering.
## Scripts
The project includes the following scripts which are ready for practical use:
- `record.py`: Allows exploring splats imported from .ply files
  and generating new images with a manual free camera.
- `score.py`: Scores images based on the presence of a certain color. 
  The intended for this is to render your splats with a very distinct background color
  which doesn't appear in the images themselves. 