import config
from utils.pre_start_init import paths
from utils.do_logging import logger
from VoiceActivityDetector import vad

if not paths.get("speaker_model_path").exists():
    logger.info("Модель Для диаризации отсутствует. "
                "\nСкачайте её 'https://wenet.org.cn/downloads?models=wespeaker&version=voxblink2_samresnet100_ft.onnx' "
                "и поместите файл в папку ./models/Diar_model")
    raise FileExistsError
else:
    from .do_diarize import Diarizer

    diarizer = Diarizer(embedding_model_path=paths.get("speaker_model_path"),
                        vad=vad,
                        sample_rate=config.BASE_SAMPLE_RATE,
                        use_gpu=config.DIAR_WITH_GPU)