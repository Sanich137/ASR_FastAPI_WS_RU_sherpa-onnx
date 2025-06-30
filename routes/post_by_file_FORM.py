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
from utils.slow_down_audio import do_slow_down_audio
from models.fast_api_models import PostFileRequest
from Recognizer.engine.stream_recognition import recognise_w_calculate_confidence, simple_recognise
from Recognizer.engine.sentensizer import do_sensitizing
from Recognizer.engine.echoe_clearing import remove_echo
from fastapi import Depends, File, Form, UploadFile
from pydantic import BaseModel
from Diarisation.diarazer import do_diarizing
import aiofiles
import os
from threading import Lock

# Глобальный лок для потокобезопасности
audio_lock = Lock()

# # Определение модели данных для параметров
# class PostFileRequest(BaseModel):
#     keep_raw: bool = True
#     do_echo_clearing: bool = False
#     do_dialogue: bool = False
#     do_punctuation: bool = False
#     do_diarization: bool = False
#     diar_vad_sensity: int = 4

# Функция для извлечения параметров из FormData
def get_file_request(
    keep_raw: bool = Form(default=True, description="Сохранять сырые данные."),
    do_echo_clearing: bool = Form(default=False, description="Убирать межканальное эхо."),
    do_dialogue: bool = Form(default=False, description="Строить диалог."),
    do_punctuation: bool = Form(default=False, description="Восстанавливать пунктуацию."),
    do_diarization: bool = Form(default=False, description="Разделять по спикерам."),
    diar_vad_sensity: int = Form(default=3, description="Чувствительность VAD."),
    do_speech_speed_correction: bool = Form(default=False, description="Корректировать скорость речи при распознавании."),
    speech_speed_correction_multiplier: float = Form(default=1, description="Базовый коэффициент скорости речи."),
) -> PostFileRequest:
    return PostFileRequest(
        keep_raw=keep_raw,
        do_echo_clearing=do_echo_clearing,
        do_dialogue=do_dialogue,
        do_punctuation=do_punctuation,
        do_diarization=do_diarization,
        diar_vad_sensity=diar_vad_sensity,
        do_speech_speed_correction = do_speech_speed_correction,
        speech_speed_correction_multiplier = speech_speed_correction_multiplier
    )


def sync_simple_recognise(audio_data):
    return asyncio.run(simple_recognise(audio_data))

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

def sync_do_diarizing(post_id, raw_data, diar_vad_sensity):
    return asyncio.run(do_diarizing(post_id, raw_data, diar_vad_sensity=diar_vad_sensity))

def sync_do_sensitizing(data, do_punctuation):
    return asyncio.run(do_sensitizing(data, do_punctuation=do_punctuation))



# Todo - перенести обработку в эту логику и обработку файлов по ссылке.
def process_file(tmp_path, params):
    res = False
    diarized = False
    error_description = str()

    result = {
        "success": res,
        "error_description": error_description,
        "raw_data": dict(),
        "sentenced_data": dict(),
    }

    post_id = uuid.uuid4()
    logger.debug(f'Принят новый "post_file" id = {post_id}')

    try:
        with audio_lock:
            posted_and_downloaded_audio[post_id] = AudioSegment.from_file(tmp_path)
    except Exception as e:
        error_description += f"Error loading audio file: {e}"
        logger.error(error_description)
        result["success"] = False
        result["error_description"] = error_description
        return result
    else:
        if posted_and_downloaded_audio[post_id].duration_seconds < 5:
            logger.debug("На вход передано аудио короче 5 секунд. Будет дополнено тишиной ещё 5 сек.")
            posted_and_downloaded_audio[post_id]+=AudioSegment.silent(5000,frame_rate=config.BASE_SAMPLE_RATE)

    if params.do_diarization and not config.CAN_DIAR:
        params.do_diarization = False
        error_description += "Diarization is not available\n"
        logger.error("Запрошена диаризация, но она не доступна.")

    # Приводим фреймрейт к фреймрейту модели
    try:
        with audio_lock:
            if posted_and_downloaded_audio[post_id].frame_rate != config.BASE_SAMPLE_RATE:
                posted_and_downloaded_audio[post_id] = sync_resample_audiosegment(
                    audio_data=posted_and_downloaded_audio[post_id],
                    target_sample_rate=config.BASE_SAMPLE_RATE,
                )
    except KeyError as e_key:
        error_description = f"Ошибка обращения по ключу {post_id} при изменения фреймрейта - {e_key}"
        logger.error(error_description)
        result["success"] = False
        result['error_description'] = str(error_description)
        return result

    except Exception as e:
        error_description = f"Ошибка изменения фреймрейта - {e}"
        logger.error(error_description)
        result["success"] = False
        result['error_description'] = str(error_description)
        return result
    else:
        # Обрабатываем чанки с аудио по N секунд
        for n_channel, mono_data in enumerate(posted_and_downloaded_audio[post_id].split_to_mono()):
            # Подготовительные действия
            try:
                with audio_lock:
                    audio_buffer[post_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
                    audio_overlap[post_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
                    audio_duration[post_id] = 0
            except Exception as e:
                    error_description = f"Ошибка изменения фреймрейта - {e}"
                    logger.error(error_description)
                    result["success"] = False
                    result['error_description'] = str(error_description)
                    return result

            result["raw_data"].update({f"channel_{n_channel + 1}": list()})

            # Основной процесс перебора чанков для распознавания
            for overlap in mono_data[::config.MAX_OVERLAP_DURATION * 1000]:
                with audio_lock:
                    if (audio_overlap[post_id].duration_seconds + overlap.duration_seconds) < 3:
                        overlap += AudioSegment.silent(3000, frame_rate=config.BASE_SAMPLE_RATE)
                    audio_buffer[post_id] = overlap
                    sync_find_last_speech_position(post_id)

                try:
                    with audio_lock:
                        audio_to_asr[post_id] = asyncio.run(do_slow_down_audio(audio_segment=audio_to_asr[post_id],
                                                                               slowdown_rate=params.speech_speed_correction_multiplier))

                        asr_result_wo_conf = sync_simple_recognise(audio_to_asr[post_id])



                except Exception as e:
                    logger.error(f"Error ASR audio - {e}")
                    error_description = f"Error ASR audio - {e}"
                else:
                    if config.MODEL_NAME == "Gigaam" or config.MODEL_NAME == "Whisper":
                        asr_result = sync_process_gigaam_asr(asr_result_wo_conf,
                                                             audio_duration[post_id],
                                                             params.speech_speed_correction_multiplier)
                    else:
                        asr_result = sync_process_asr_json(asr_result_wo_conf, audio_duration[post_id])

                    result["raw_data"][f"channel_{n_channel + 1}"].append(asr_result)

                    with audio_lock:
                        audio_duration[post_id] += audio_to_asr[post_id].duration_seconds
                    res = True
                    logger.debug(asr_result)

            with audio_lock:
                try:
                    del audio_overlap[post_id]
                    del audio_buffer[post_id]
                    del audio_to_asr[post_id]
                    del audio_duration[post_id]
                except Exception as e:
                    error_description = f"Ошибка при очистке данных - {e}"
                    logger.error(error_description)
                    result["success"] = False
                    result['error_description'] = str(error_description)

        if params.do_echo_clearing:
            try:
                result["raw_data"] = sync_remove_echo(result["raw_data"])
            except Exception as e:
                logger.error(f"Error echo clearing - {e}")
                error_description = f"Error echo clearing - {e}"
                res = False

        if params.do_diarization:
            try:
                result["diarized_data"] = sync_do_diarizing(
                    post_id, result["raw_data"], diar_vad_sensity=params.diar_vad_sensity
                )
            except Exception as e:
                logger.error(f"do_diarizing - {e}")
                error_description = f"do_diarizing - {e}"
                res = False
            else:
                diarized = True

        if params.do_dialogue:
            data_to_do_sensitizing = result["diarized_data"] if diarized else result["raw_data"]
            try:
                result["sentenced_data"] = sync_do_sensitizing(
                    data_to_do_sensitizing, do_punctuation=params.do_punctuation
                )
            except Exception as e:
                logger.error(f"do_sensitizing - {e}")
                error_description = f"do_sensitizing - {e}"
                res = False
            else:
                if not params.keep_raw:
                    result["raw_data"].clear()
        else:
            result["sentenced_data"].clear()

        result["error_description"] = error_description
        result["success"] = res

        with audio_lock:
            del posted_and_downloaded_audio[post_id]

        logger.debug(result)
        return result

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
