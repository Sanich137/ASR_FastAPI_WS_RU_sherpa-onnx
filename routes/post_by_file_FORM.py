from pydub import AudioSegment
import config
import asyncio
import uuid
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
from utils.chunk_doing import find_last_speech_position
from utils.resamppling import resample_audiosegment
from models.fast_api_models import PostFileRequest
from Recognizer.engine.sentensizer import do_sensitizing
from Recognizer.engine.echoe_clearing import remove_echo
from Recognizer.engine.file_recognition import process_file
from fastapi import Depends, File, Form, UploadFile
import aiofiles
import os
from threading import Lock


# Глобальный лок для потокобезопасности
audio_lock = Lock()

# Функция для извлечения параметров из FormData
def get_file_request(
    keep_raw: bool = Form(default=True, description="Сохранять сырые данные."),
    do_echo_clearing: bool = Form(default=False, description="Убирать межканальное эхо."),
    do_dialogue: bool = Form(default=False, description="Строить диалог."),
    do_punctuation: bool = Form(default=False, description="Восстанавливать пунктуацию."),
    do_diarization: bool = Form(default=False, description="Разделять по спикерам."),
    diar_vad_sensity: int = Form(default=3, description="Чувствительность VAD."),
    do_auto_speech_speed_correction: bool = Form(default=False, description="Корректировать скорость речи при распознавании."),
    speech_speed_correction_multiplier: float = Form(default=1, description="Базовый коэффициент скорости речи."),
    make_mono: bool = Form(default=False, description="Соединить несколько каналов в mono"),
) -> PostFileRequest:
    return PostFileRequest(
        keep_raw=keep_raw,
        do_echo_clearing=do_echo_clearing,
        do_dialogue=do_dialogue,
        do_punctuation=do_punctuation,
        do_diarization=do_diarization,
        diar_vad_sensity=diar_vad_sensity,
        make_mono=make_mono,
        do_auto_speech_speed_correction = do_auto_speech_speed_correction,
        speech_speed_correction_multiplier = speech_speed_correction_multiplier
    )


def sync_resample_audiosegment(audio_data, target_sample_rate):
    return asyncio.run(resample_audiosegment(audio_data, target_sample_rate))

def sync_find_last_speech_position(post_id):
    return asyncio.run(find_last_speech_position(post_id))

def sync_process_gigaam_asr(asr_result, duration, multiplier = 1):
    return asyncio.run(process_gigaam_asr(asr_result, duration, multiplier))

def sync_process_asr_json(asr_result, duration):
    return asyncio.run(process_asr_json(asr_result, duration))

def sync_remove_echo(raw_data):
    return asyncio.run(remove_echo(raw_data))

def sync_do_sensitizing(data, do_punctuation):
    return asyncio.run(do_sensitizing(data, do_punctuation=do_punctuation))



@app.post("/post_file")
async def async_receive_file(
    file: UploadFile = File(description="Аудиофайл для обработки"),
    params: PostFileRequest = Depends(get_file_request),
):
    res = True
    error_description = str()

    result = {
        "success": res,
        "error_description": error_description,
        "raw_data": dict(),
        "sentenced_data": dict(),
    }

    # Сохраняем файл на диск асинхронно
    try:
        async with aiofiles.tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
            await tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as e:
        error_description = f"Не удалось сохранить файл для распознавания: {file.filename}, размер файла: {file.size}, по причине: {e}"
        logger.error(error_description)
        result["success"] = False
        result["error_description"] = error_description
        return result
    else:
        logger.debug(f"Получен и сохранён файл {file.filename}")
        try:
            # Запускаем обработку в потоке
            result = await asyncio.to_thread(process_file, tmp_path, params)
        except Exception as e:
            error_description = f"Ошибка обработки в process_file - {e}"
            logger.error(error_description)
            result["success"] = False
            result['error_description'] = str(error_description)
        else:
            # структура ответа строится в process_file
            pass
        finally:
            # Удаляем временный файл
            os.unlink(tmp_path)
    finally:
        logger.info(result)
        return result
