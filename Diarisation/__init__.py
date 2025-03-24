import sherpa_onnx
from utils.pre_start_init import paths

async def init_speaker_diarization(num_speakers: int = -1,
                                   cluster_threshold: float = 0.3):

    segmentation_model = str(paths.get("segmentation_model"))
    embedding_extractor_model = str(paths.get("embedding_extractor_model"))

    diarization_model_config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=segmentation_model,),
            provider= 'cuda',  # Похоже пока всегда cpu
            num_threads = 3, # Похоже пока всегда в 1 поток
            debug = False
            ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=embedding_extractor_model
            ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers,
            threshold= cluster_threshold
            ),
        min_duration_on=0.2,
        min_duration_off=0.4,
        )

    if not diarization_model_config.validate():
        raise RuntimeError(
            "Please check your config and make sure all required files exist"
        )

    return sherpa_onnx.OfflineSpeakerDiarization(diarization_model_config)