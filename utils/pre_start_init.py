# -*- coding: utf-8 -*-

from utils.do_logging import logger
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI

BASE_DIR = Path(__file__).resolve().parent.parent
paths = {
    "vosk_small_streaming_tokens_path": BASE_DIR / "models" / "vosk-model-small-streaming-ru" / "lang" / "tokens.txt",
    "vosk_small_streaming_encoder_path": BASE_DIR / "models" / "vosk-model-small-streaming-ru" / "am" / "encoder.onnx",
    "vosk_small_streaming_decoder_path": BASE_DIR / "models" / "vosk-model-small-streaming-ru" / "am" / "decoder.onnx",
    "vosk_small_streaming_joiner_path": BASE_DIR / "models" / "vosk-model-small-streaming-ru" / "am" / "joiner.onnx",
    "vosk_small_streaming_bpe_vocab": BASE_DIR / "models" / "vosk-model-small-streaming-ru" / "lang" / "bpe.model",

    "vosk_full_tokens_path": BASE_DIR / "models" / "vosk-model-ru" / "lang" / "tokens.txt",
    "vosk_full_encoder_path": BASE_DIR / "models" / "vosk-model-ru" / "am-onnx" / "encoder.onnx",
    "vosk_full_decoder_path": BASE_DIR / "models" / "vosk-model-ru" / "am-onnx" / "decoder.onnx",
    "vosk_full_joiner_path": BASE_DIR / "models" / "vosk-model-ru" / "am-onnx" / "joiner.onnx",
    "vosk_full_bpe_vocab": BASE_DIR / "models" / "vosk-model-ru" / "lang" / "bpe.model",

    "gigaam_tokens_path": BASE_DIR / "models" / "GigaAMv2_CTC_RU_ASR_for_sherpa_onnx" / "tokens.txt",
    "gigaam_encoder_path": BASE_DIR / "models" / "GigaAMv2_CTC_RU_ASR_for_sherpa_onnx" / "GigaAMv2_ctc_public.onnx",

    # "whisper_tokens_path": BASE_DIR / "models" / "sherpa-onnx-whisper-medium" / "medium-tokens.txt",
    # "whisper_encoder_path": BASE_DIR / "models" / "sherpa-onnx-whisper-medium" / "medium-encoder.onnx",
    # "whisper_decoder_path": BASE_DIR / "models" / "sherpa-onnx-whisper-medium" / "medium-decoder.onnx",
    # "whisper_encoder_path": BASE_DIR / "models" / "sherpa-onnx-whisper-small" / "small-encoder.int8.onnx",
    # "whisper_decoder_path": BASE_DIR / "models" / "sherpa-onnx-whisper-small" / "small-decoder.int8.onnx",
    # Todo - говорят производительность Whisper пофиксили. Протестить!

    "punctuation_model_path": BASE_DIR / "models" / "sbert_punc_case_ru_onnx",
    "vad_model_path": BASE_DIR / "models" / "VAD_silero_v5" / "silero_vad.onnx",

    # Diarisation model
    "segmentation_model": BASE_DIR / "models" / "Diar_model" / "model.onnx",
    "embedding_extractor_model": BASE_DIR / "models" / "Diar_model" / "wespeaker_en_voxceleb_resnet34_LM.onnx",
    # "embedding_extractor_model": BASE_DIR / "models" / "Diar_model" / "wespeaker_en_voxceleb_resnet152_LM.onnx",
    # "embedding_extractor_model": BASE_DIR / "models" / "Diar_model" / "voxblink2_samresnet100_meta.onnx",

    "BASE_DIR": BASE_DIR,
    "test_file": BASE_DIR /'trash'/'111.wav',
    "trash_folder": BASE_DIR / 'trash',
}

from collections import defaultdict

# Глобальные переменные
audio_overlap = defaultdict()
audio_buffer = defaultdict()
audio_to_asr = defaultdict()
audio_duration = defaultdict(float)
ws_collected_asr_res = defaultdict()
posted_and_downloaded_audio = defaultdict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # on_start
    logger.debug("Приложение  FastAPI запущено")

    yield  # on_stop
    logger.debug("Приложение FastAPI завершено")

app = FastAPI(lifespan=lifespan,
              version="1.0",
              docs_url='/docs',
              root_path='/root',
              title='ASR on SHERPA-ONNX'
              )