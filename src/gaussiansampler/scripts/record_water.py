"""Water-Splatting entry point built on the regular recorder."""


def main(argv=None):
    from gaussiansampler.gui.water_recorder import WaterRecorderExtension
    from gaussiansampler.scripts.record import main as record_main

    def create_water_renderer(splat_path):
        from gaussiansampler.sampling.water_gsplat_renderer import WaterGsplatRenderer

        return WaterGsplatRenderer(splat_path)

    return record_main(
        argv,
        renderer_factory=create_water_renderer,
        recorder_extension=WaterRecorderExtension(),
    )
