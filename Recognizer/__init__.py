from . import engine
from utils.pre_start_init import paths
from utils.do_logging import logger
import config
import sherpa_onnx

models_arguments = {
        # "Vosk5SmallStreaming": {
        #     "tokens": paths.get("vosk_small_streaming_tokens_path"),
        #     "encoder": paths.get("vosk_small_streaming_encoder_path"),
        #     "decoder": paths.get("vosk_small_streaming_decoder_path"),
        #     "joiner": paths.get("vosk_small_streaming_joiner_path"),
        #     "bpe_vocab": paths.get("vosk_small_streaming_bpe_vocab"),
        #     "num_threads": config.NUM_THREADS,
        #     "decoding_method": "greedy_search",
        #     "debug": True,  #  if config.LOGGING_LEVEL == "DEBUG" else False,
        #     "sample_rate": config.base_sample_rate,
        #     "feature_dim": 80,
        #     "provider": config.PROVIDER,
        #     "Base_Recognizer": sherpa_onnx.OnlineRecognizer
        #         },
        "Vosk5": {
            "tokens": paths.get("vosk_full_tokens_path"),
            "encoder": paths.get("vosk_full_encoder_path"),
            "decoder": paths.get("vosk_full_decoder_path"),
            "bpe_vocab": paths.get("vosk_full_bpe_vocab"),
            "joiner": paths.get("vosk_full_joiner_path"),
            "num_threads": config.NUM_THREADS,
            "decoding_method": "greedy_search",
            "debug": True if config.LOGGING_LEVEL == "DEBUG" else False,
            "sample_rate": config.BASE_SAMPLE_RATE,
            "feature_dim": 80,
            "provider": config.PROVIDER,
            "Base_Recognizer": sherpa_onnx.OfflineRecognizer
                },
        "Whisper": {
            "tokens": paths.get("whisper_tokens_path"),
            "encoder": paths.get("whisper_encoder_path"),
            "decoder": paths.get("whisper_decoder_path"),
            "num_threads": config.NUM_THREADS,
            "decoding_method": "greedy_search",
            "debug": True if config.LOGGING_LEVEL == "DEBUG" else False,
            "sample_rate": config.BASE_SAMPLE_RATE,
            "feature_dim": 80,
            "provider": config.PROVIDER,
            "Base_Recognizer": sherpa_onnx.OfflineRecognizer
                },
            }

model_settings = models_arguments.get(config.MODEL_NAME)

logger.debug(f"{config.MODEL_NAME} chosen model!")

# Todo - тут нужно отработать для разных моделей
# if not model_settings.get("model").exists():
#     logger.error("Model files does`nt exist")
#     raise FileExistsError


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
elif config.MODEL_NAME== "Whisper":
    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(model_settings.get("encoder")),
        decoder=str(model_settings.get("decoder")),
        tokens=str(model_settings.get("tokens")),
        language = "ru",
        task = "transcribe",
        num_threads=model_settings.get("num_threads", 1),
        decoding_method=model_settings.get("decoding_method", "greedy_search"),
        provider=model_settings.get("provider", "CPU"),
        tail_paddings = -100
    )
else:
    recognizer = model_settings.get("Base_Recognizer").from_transducer(
        encoder=str(model_settings.get("encoder")),
        decoder=str(model_settings.get("decoder")),
        joiner=str(model_settings.get("joiner")),
        tokens=str(model_settings.get("tokens")),
        num_threads=model_settings.get("num_threads", 1),
        sample_rate=model_settings.get("sample_rate", 16000),
        feature_dim=model_settings.get("feature_dim"),
        decoding_method=model_settings.get("decoding_method", "greedy_search"),
        provider=model_settings.get("provider", "CPU"),
        lm_scale=0.2,  # по умолчанию 0.1
        modeling_unit="bpe",  # по умолчанию "cjkchar" пишут что надо только для горячих слов
        bpe_vocab=str(model_settings.get("bpe_vocab")),
        debug=model_settings.get("debug", False),
    )

logger.debug(f"Model {config.MODEL_NAME} ready to start!")

# По неведомой причине, если импорт onnxruntime происходит до старта  recognizer - инициация модели не происходит.
# Поэтому импорт onnxruntime происходит в этой части кода.
import onnxruntime as ort