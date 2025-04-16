from pydub import AudioSegment
import io
import config

import uuid
from utils.tokens_to_Result import process_asr_json, process_gigaam_asr

from typing import Annotated
from utils.pre_start_init import (app,
                                  posted_and_downloaded_audio,
                                  audio_buffer,
                                  audio_overlap,
                                  audio_to_asr,
                                  audio_duration)

from fastapi import (UploadFile, File, Depends, Form)

from utils.do_logging import logger
from utils.chunk_doing import find_last_speech_position

from models.fast_api_models import PostFileRequestDiarize
from Diarisation.do_diarize import do_diarization
from Diarisation.diarazed_asr_matching import do_diarized_dialogue
from Recognizer.engine.stream_recognition import recognise_w_calculate_confidence, simple_recognise
from Recognizer.engine.sentensizer import do_sensitizing
from Recognizer.engine.echoe_clearing import remove_echo


# Функция для извлечения параметров из FormData
def get_file_request(
    keep_raw: Annotated[bool, Form()] = False,
    do_echo_clearing: Annotated[bool, Form()] = True,  # логика отличается. Тут подавление эха планируется засчёт
        # исключения одинаковых спикеров в разных каналах. # todo - Пробовать проверять спикеров при пересечении аудио.
    do_punctuation: Annotated[bool, Form()] = True,
    num_speakers: Annotated[int, Form()] = -1,
    cluster_threshold: Annotated[float, Form()] = 0.2

) -> PostFileRequestDiarize:
    """
    Модель для проверки запроса пользователя.
    :param keep_raw: Если False, то запрос вернёт только пост-обработанные данные do_punctuation и do_dialogue.
    :param do_echo_clearing: Проверяет наличие повторений между каналами.
    :param num_speakers: Предполагаемое количество спикеров в разговоре. -1 - значит мы не знаем сколько спикеров и определяем их параметром cluster_threshold.
    :param cluster_threshold: Значение от 0 до 1. Чем меньше, тем более чувствительное выделение спикеров (тем их больше)
    :param do_punctuation: Расставляет пунктуацию.
    """
    return PostFileRequestDiarize(
        keep_raw=keep_raw,
        do_echo_clearing=do_echo_clearing,
        do_punctuation=do_punctuation,
        num_speakers=num_speakers,
        cluster_threshold=cluster_threshold
    )


@app.post("/post_file_w_diar")
async def receive_file_for_diarization(
    file: Annotated[UploadFile, File(description="audio file")],
    params: Annotated[PostFileRequestDiarize, Depends(get_file_request)]
    ):
    """
    Модель для проверки запроса пользователя.
    \n:param keep_raw: Если False, то запрос вернёт только пост-обработанные данные do_punctuation и do_dialogue.
    \n:param do_echo_clearing: Проверяет наличие повторений между каналами.
    \n:param num_speakers: Предполагаемое количество спикеров в разговоре. -1 - значит мы не знаем сколько спикеров и определяем их параметром cluster_threshold.
    \n:param cluster_threshold: Значение от 0 до 1. Чем меньше, тем более чувствительное выделение спикеров (тем их больше)
    \n:param do_punctuation: Расставляет пунктуацию.
    """

    res = False
    error_description = str()
    result = {
        "success":res,
        "error_description": error_description,
        "raw_data": dict(),
        "diarized_timestamps": dict(),
        "sentenced_diarized_data": dict(),
        }

    post_id = uuid.uuid4()

    logger.debug(f'Принят новый "post_file"  id = {post_id}')

    if "audio" in file.content_type:
        posted_and_downloaded_audio[post_id] = AudioSegment.from_file(file.file)
    else:
        result["success"] = False
        result["error_description"] = "Not audio file received"

        return result

    # Производим Диаризацию.
    for n_channel, mono_data in enumerate(posted_and_downloaded_audio[post_id].split_to_mono()):
        result["diarized_timestamps"].update({f"channel_{n_channel + 1}": list()})
        result["diarized_timestamps"][f"channel_{n_channel + 1}"] = (
            await do_diarization(mono_data, num_speakers=-1, cluster_threshold= 0.2))

    # Обрабатываем чанки с аудио по 10 секунд.
    for n_channel, mono_data in enumerate(posted_and_downloaded_audio[post_id].split_to_mono()):

        audio_buffer[post_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
        audio_overlap[post_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
        audio_duration[post_id] = 0

        result['raw_data'].update({f"channel_{n_channel + 1}": list()})

        for overlap in mono_data[::config.MAX_OVERLAP_DURATION * 1000]:

            if (audio_overlap[post_id].duration_seconds + overlap.duration_seconds) < 3:
                # Исправил ошибку обработки последнего маленького чанка.
                overlap = overlap+AudioSegment.silent(3000, frame_rate=config.BASE_SAMPLE_RATE)

            audio_buffer[post_id] = overlap  # Кривизна вызвана особенностями реализации буфера в сокетах

            find_last_speech_position(post_id)

            # Обрабатываем основной массив данных
            try:
                asr_result_wo_conf = await simple_recognise(audio_to_asr[post_id])
            except Exception as e:
                logger.error(f"Error ASR audio - {e}")
                error_description = f"Error ASR audio - {e}"
            else:
                if config.MODEL_NAME == "Gigaam" or config.MODEL_NAME == "Whisper":
                    asr_result = await process_gigaam_asr(asr_result_wo_conf, audio_duration[post_id])
                else:
                    asr_result = await process_asr_json(asr_result_wo_conf, audio_duration[post_id])

                result['raw_data'][f"channel_{n_channel + 1}"].append(asr_result)

                audio_duration[post_id] += audio_to_asr[post_id].duration_seconds
                res = True
                logger.debug(asr_result)


        del audio_overlap[post_id]
        del audio_buffer[post_id]
        del audio_to_asr[post_id]
        del audio_duration[post_id]

    if params.do_echo_clearing:
        try:
            result["raw_data"] = await remove_echo(result["raw_data"])
        except Exception as e:
            logger.error(f"Error echo clearing - {e}")
            error_description = f"Error echo clearing - {e}"
            res = False

    # Сопоставляем временные метки диаризации с результатами распознавания.
    for name_channel in result.get("raw_data"):

        result["sentenced_diarized_data"].update({name_channel: list()})

        try:
            result["sentenced_diarized_data"] = await do_diarized_dialogue(result["raw_data"][name_channel],
                                                                     result["diarized_timestamps"][name_channel])
        except Exception as e:
            logger.error(f"await do_sensitizing - {e}")
            error_description = f"do_sensitizing - {e}"
            res = False
        else:
            if not params.keep_raw:
                result["raw_data"].clear()

    result['error_description'] = error_description
    result['success'] = res

    del posted_and_downloaded_audio[post_id]

    return result

