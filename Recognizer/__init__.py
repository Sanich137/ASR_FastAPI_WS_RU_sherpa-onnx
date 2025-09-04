from . import engine
from utils.globals import paths
from utils.do_logging import logger
import config
import sherpa_onnx


logger.info(f"{config.MODEL_NAME} chosen model.")

# Todo - тут нужно отработать для разных моделей
# if not model_settings.get("model").exists():
#     logger.error("Model files does`nt exist")
#     raise FileExistsError


def get_recognizer() -> sherpa_onnx.OfflineRecognizer:
    recognizer = None
    if config.MODEL_NAME== "Gigaam":
        recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(paths.get("gigaam_encoder_path")),
            tokens=str(paths.get("gigaam_tokens_path")),
            num_threads=config.NUM_THREADS,
            sample_rate=config.BASE_SAMPLE_RATE,
            decoding_method="greedy_search",
            provider=config.PROVIDER,
            feature_dim=64,
            debug=True if config.LOGGING_LEVEL == "DEBUG" else False,
        )
    elif config.MODEL_NAME=="Gigaam_rnnt":
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(paths.get("gigaam_rnnt_encoder_path")),
            decoder=str(paths.get("gigaam_rnnt_decoder_path")),
            joiner=str(paths.get("gigaam_rnnt_joiner_path")),
            tokens=str(paths.get("gigaam_rnnt_tokens_path")),
            feature_dim=64,
            num_threads=4,    #config.NUM_THREADS,
            dither=0.00003,
            sample_rate=config.BASE_SAMPLE_RATE,
            decoding_method="greedy_search",
            modeling_unit="cjkchar",  # по умолчанию "cjkchar" пишут что надо только для горячих слов
            # bpe_vocab=str(model_settings.get("bpe_vocab")), # Указан, но кажется не работает без hotwords
            lm_scale=0.2,  # по умолчанию 0.1
            provider=config.PROVIDER,
            model_type="nemo_transducer",
            debug=True if config.LOGGING_LEVEL == "DEBUG" else False,
        )

    logger.info(f"Model {config.MODEL_NAME} ready to start!")
    return recognizer