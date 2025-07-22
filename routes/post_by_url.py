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
from Recognizer.engine.file_recognition import process_file
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
        result = await asyncio.to_thread(process_file, tmp_path, params)
    finally:
        # Удаляем временный файл
        os.unlink(tmp_path)

    return result