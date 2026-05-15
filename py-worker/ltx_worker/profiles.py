LOW_VRAM_PROFILES = {"colab_tiny", "colab_eco"}


def preferred_pipeline_names(model_key, profile):
    model_key = model_key or ""
    if model_key.startswith("sulphur_2_dev"):
        return [
            ("ltx_pipelines.ti2vid_one_stage", "TI2VidOneStagePipeline"),
        ]
    distilled = "distilled" in model_key or "distil" in model_key or profile in LOW_VRAM_PROFILES
    if distilled:
        return [
            ("ltx_pipelines.distilled_pipeline", "DistilledPipeline"),
            ("ltx_pipelines.distilled", "DistilledPipeline"),
            ("ltx_pipelines.ti2vid_one_stage", "TI2VidOneStagePipeline"),
        ]
    return [
        ("ltx_pipelines.ti2vid_two_stages", "TI2VidTwoStagesPipeline"),
        ("ltx_pipelines.ti2vid_two_stages_res2s", "TI2VidTwoStagesRes2sPipeline"),
        ("ltx_pipelines.ti2vid_one_stage", "TI2VidOneStagePipeline"),
    ]
