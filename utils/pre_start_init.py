# -*- coding: utf-8 -*-

import os
from utils.do_logging import logger
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketException
from utils.file_exists import assert_file_exists
import sherpa_onnx

import config

BASE_DIR = Path(__file__).resolve().parent.parent
paths = {
    "tokens_path": BASE_DIR / "models" / "vosk-model-small-streaming-ru" / "lang" / "tokens.txt",
    "encoder_path": BASE_DIR / "models" / "vosk-model-small-streaming-ru" / "am" / "encoder.onnx",
    "decoder_path": BASE_DIR / "models" / "vosk-model-small-streaming-ru" / "am" / "decoder.onnx",
    "joiner_path": BASE_DIR / "models" / "vosk-model-small-streaming-ru" / "am" / "joiner.onnx",
    "bpe_vocab": BASE_DIR / "models" / "vosk-model-small-streaming-ru" / "lang" / "bpe.model",
    "BASE_DIR": BASE_DIR,
    "test_file": BASE_DIR /'trash'/'2724.1726990043.1324706.wav',
    "trash_folder": BASE_DIR / 'trash',
}

models_arguments = {
        "vosk": {
            "tokens": paths.get("tokens_path"),
            "encoder": paths.get("encoder_path"),
            "decoder": paths.get("decoder_path"),
            "joiner": paths.get("joiner_path"),
            "num_threads": 4,
            "decoding_method": "greedy_search",
            "debug": False,
            "sample_rate": 16000,
            "feature_dim": 80,
            "bpe_vocab": paths.get("bpe_vocab"),
            "provider": "cpu",
                }
            }

model_config = models_arguments.get("vosk")

recognizer = None

if model_config.get("encoder"):
    assert_file_exists(model_config.get("encoder"))
    assert_file_exists(model_config.get("decoder"))
    assert_file_exists(model_config.get("joiner"))

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=str(model_config.get("encoder")),
        decoder=str(model_config.get("decoder")),
        joiner=str(model_config.get("joiner")),
        tokens=str(model_config.get("tokens")),
        num_threads=model_config.get("num_threads", 1),
        sample_rate=model_config.get("sample_rate", 8000),
        feature_dim=model_config.get("feature_dim"),
        decoding_method=model_config.get("decoding_method", "greedy_search"),
        provider=model_config.get("provider", "CPU"),
        lm_scale=0.2,  # по умолчанию 0.1
        modeling_unit="bpe",  # по умолчанию "cjkchar" пишут что надо только для горячих слов
        bpe_vocab=str(model_config.get("bpe_vocab")),
        debug=model_config.get("feature_dim", False),
    )
    logger.debug(f"Model {config.model_name} ready to start!")

else:
    logger.error("Please specify correct model")




@asynccontextmanager
async def lifespan(app: FastAPI):
    # on_start
    logger.debug("Приложение  FastAPI запущено")
    # await state_audio_classifier.infinity_worker()
    yield  # on_stop
    logger.debug("Приложение FastAPI завершено")

app = FastAPI(lifespan=lifespan,
              version="0.1",
              docs_url='/docs',
              root_path='/root',
              title='ASR-Vosk5-GPU on SHERPA-ONNX streaming model'
              )
