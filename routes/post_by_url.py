from pydub import AudioSegment
import config
import uuid
import asyncio
import aiofiles
import os
from utils.tokens_to_Result import process_asr_json, process_gigaam_asr
from utils.pre_start_init import (
    app,
    posted_and_downloaded_audio,
    audio_buffer,
    audio_overlap,
    audio_to_asr,
    audio_duration,
)
from utils.do_logging import logger
from utils.get_audio_file import getting_audiofile, open_default_audiofile
from utils.chunk_doing import find_last_speech_position
from utils.resamppling import resample_audiosegment
from models.fast_api_models import SyncASRRequest
from Recognizer.engine.stream_recognition import recognise_w_calculate_confidence, simple_recognise
from Recognizer.engine.sentensizer import do_sensitizing
from Recognizer.engine.echoe_clearing import remove_echo
from threading import Lock
import io

# Глобальный лок для потокобезопасности
audio_lock = Lock()

# Синхронные версии функций (заглушки, нужно переписать как синхронные)
def sync_simple_recognise(audio_data):  # Перепишите как синхронную
    return asyncio.run(simple_recognise(audio_data))

def sync_recognise_w_calculate_confidence(audio_data, num_trials):  # Перепишите как синхронную
    return asyncio.run(recognise_w_calculate_confidence(audio_data, num_trials))

def sync_resample_audiosegment(audio_data, target_sample_rate):  # Перепишите как синхронную
    return asyncio.run(resample_audiosegment(audio_data, target_sample_rate))

def sync_find_last_speech_position(post_id):  # Перепишите как синхронную
    return asyncio.run(find_last_speech_position(post_id))

def sync_process_gigaam_asr(asr_result, duration):  # Перепишите как синхронную
    return asyncio.run(process_gigaam_asr(asr_result, duration))

def sync_process_asr_json(asr_result, duration):  # Перепишите как синхронную
    return asyncio.run(process_asr_json(asr_result, duration))

def sync_remove_echo(raw_data):  # Перепишите как синхронную
    return asyncio.run(remove_echo(raw_data))

def sync_do_sensitizing(data, do_punctuation):  # Перепишите как синхронную
    return asyncio.run(do_sensitizing(data, do_punctuation))

def process_request(tmp_path, params):
    res = False
    error_description = str()

    result = {
        "success": res,
        "error_description": error_description,
        "raw_data": dict(),
        "sentenced_data": dict(),
    }

    post_id = uuid.uuid4()
    logger.debug(f'Принят новый "post_one_step_req" id = {post_id}')

    try:
        with audio_lock:
            posted_and_downloaded_audio[post_id] = AudioSegment.from_file(tmp_path)
    except Exception as e:
        logger.error(f"Error loading audio file: {e}")
        error_description = f"Error loading audio file: {e}"
        return result

    # Приводим фреймрейт к фреймрейту модели
    with audio_lock:
        if posted_and_downloaded_audio[post_id].frame_rate != config.BASE_SAMPLE_RATE:
            posted_and_downloaded_audio[post_id] = sync_resample_audiosegment(
                posted_and_downloaded_audio[post_id],
                config.BASE_SAMPLE_RATE
            )

    # Обрабатываем чанки с аудио по 15 секунд.
    for n_channel, mono_data in enumerate(posted_and_downloaded_audio[post_id].split_to_mono()):
        with audio_lock:
            audio_buffer[post_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
            audio_overlap[post_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
            audio_duration[post_id] = 0

        result['raw_data'].update({f"channel_{n_channel + 1}": list()})

        for overlap in mono_data[::config.MAX_OVERLAP_DURATION * 1000]:
            with audio_lock:
                audio_buffer[post_id] = overlap
                if overlap.duration_seconds == config.MAX_OVERLAP_DURATION:
                    sync_find_last_speech_position(post_id)
                else:
                    audio_to_asr[post_id] = audio_overlap[post_id] + overlap
                    audio_overlap[post_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)

            try:
                with audio_lock:
                    if config.MODEL_NAME == "Gigaam" or config.MODEL_NAME == "Whisper":
                        asr_result_wo_conf = sync_simple_recognise(audio_to_asr[post_id])
                        asr_result = sync_process_gigaam_asr(asr_result_wo_conf, audio_duration[post_id])
                        audio_duration[post_id] += audio_to_asr[post_id].duration_seconds
                        logger.debug(asr_result)
                    else:
                        asr_result_w_conf = sync_recognise_w_calculate_confidence(
                            audio_to_asr[post_id],
                            num_trials=config.RECOGNITION_ATTEMPTS
                        )
                        asr_result = sync_process_asr_json(asr_result_w_conf, audio_duration[post_id])
                        audio_duration[post_id] += audio_to_asr[post_id].duration_seconds
                        logger.debug(asr_result)
            except Exception as e:
                logger.error(f"Error ASR audio - {e}")
                error_description = f"Error ASR audio - {e}"
            else:
                result['raw_data'][f"channel_{n_channel + 1}"].append(asr_result)

        with audio_lock:
            del audio_overlap[post_id]
            del audio_buffer[post_id]
            del audio_to_asr[post_id]
            del audio_duration[post_id]

    if params.do_echo_clearing:
        try:
            result["raw_data"] = sync_remove_echo(result["raw_data"])
        except Exception as e:
            logger.error(f"Error echo clearing - {e}")
            error_description = f"Error echo clearing - {e}"
            res = False

    if params.do_dialogue:
        try:
            result["sentenced_data"] = sync_do_sensitizing(result["raw_data"], params.do_punctuation)
        except Exception as e:
            logger.error(f"do_sensitizing - {e}")
            error_description = f"do_sensitizing - {e}"
            res = False
        else:
            if not params.keep_raw:
                result["raw_data"].clear()
    else:
        result["sentenced_data"].clear()

    result['error_description'] = error_description
    result['success'] = res

    with audio_lock:
        del posted_and_downloaded_audio[post_id]

    return result

@app.post("/post_one_step_req")
async def post(params: SyncASRRequest):
    """
    На вход принимает HttpUrl - прямую ссылку на скачивание файла 'mp3', 'wav' или 'ogg'.\n
    Если на вход передаётся не моно, то ответ будет в несколько элементов списка для каждого канала.\n
    По умолчанию отдаёт сырой результат распознавания с разбивкой на части продолжительностью около 15 секунд\n

    :param: do_dialogue: - true, если нужно разбить речь на диалог\n
    :param: do_punctuation - true, если нужно расставить пунктуацию. Применяется к диалогу, общему тексту. В проекте.\n
    При проектировании таймаутов учитывайте скорость распознавания (около 100 секунд аудио распознаётся за 2-5 секунд
    распознавания одного канала)
    """
    # Получаем файл
    post_id = uuid.uuid4()
    if params.AudioFileUrl:
        res, error_description = await getting_audiofile(params.AudioFileUrl, post_id)
    else:
        res, error_description = await open_default_audiofile(post_id)

    if not res:
        logger.error(f'Ошибка получения файла - {error_description}, ссылка на файл - {params.AudioFileUrl}')
        return {
            "success": False,
            "error_description": error_description,
            "raw_data": dict(),
            "sentenced_data": dict(),
        }

    # Сохраняем файл на диск асинхронно
    async with aiofiles.tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
        with audio_lock:
            if isinstance(posted_and_downloaded_audio[post_id], str):
                with open(posted_and_downloaded_audio[post_id], 'rb') as f:
                    await tmp.write(f.read())
            elif isinstance(posted_and_downloaded_audio[post_id], io.BytesIO):
                await tmp.write(posted_and_downloaded_audio[post_id].getvalue())
            else:
                await tmp.write(posted_and_downloaded_audio[post_id])
        tmp_path = tmp.name

    try:
        # Запускаем обработку в потоке
        result = await asyncio.to_thread(process_request, tmp_path, params)
    finally:
        # Удаляем временный файл
        os.unlink(tmp_path)

    return result