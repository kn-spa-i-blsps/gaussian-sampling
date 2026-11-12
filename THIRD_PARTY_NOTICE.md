# Third-party notices

## WaterSplatting

The optional underwater renderer adapts the neural medium-field rendering
approach and checkpoint interface from [WaterSplatting](https://github.com/water-splatting/water-splatting).
The Gaussian-splat loading, rasterization, viewer integration, and compositing
code in this repository are implemented here; the upstream WaterSplatting
repository is not bundled as a dependency.

WaterSplatting was created by Huapeng Li, Wenxuan Song, Tianao Xu, Alexandre
Elsig, and Jonas Kulhanek. The upstream code is licensed under the [Apache
License, Version 2.0](https://github.com/water-splatting/water-splatting/blob/main/LICENSE).
The applicable license text is included in this repository's [LICENSE](LICENSE).

For research using this renderer or the WaterSplatting method, cite:

```bibtex
@article{li2025watersplatting,
  title={{W}ater{S}platting: Fast Underwater {3D} Scene Reconstruction using Gaussian Splatting},
  author={Li, Huapeng and Song, Wenxuan and Xu, Tianao and Elsig, Alexandre and Kulhanek, Jonas},
  journal={3DV},
  year={2025}
}
```
