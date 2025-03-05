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

from models.fast_api_models import PostFileRequest

from Recognizer.engine.stream_recognition import recognise_w_calculate_confidence, simple_recognise
from Recognizer.engine.sentensizer import do_sensitizing
from Recognizer.engine.echoe_clearing import remove_echo


# Функция для извлечения параметров из FormData
def get_file_request(
    keep_raw: Annotated[bool, Form()] = True,
    do_echo_clearing: Annotated[bool, Form()] = False,
    do_dialogue: Annotated[bool, Form()] = False,
    do_punctuation: Annotated[bool, Form()] = False,
) -> PostFileRequest:
    return PostFileRequest(
        keep_raw=keep_raw,
        do_echo_clearing=do_echo_clearing,
        do_dialogue=do_dialogue,
        do_punctuation=do_punctuation,
    )


@app.post("/post_file")
async def receive_file(
    file: Annotated[UploadFile, File(description="Аудиофайл для обработки")],
    params: Annotated[PostFileRequest, Depends(get_file_request)]
    ):

    """
    :param file: Файл, который будет обработан.
    :param params:
    :return:

    """
    res = False
    error_description = str()
    result = {
        "success":res,
        "error_description": error_description,
        "raw_data": dict(),
        "sentenced_data": dict(),
        "punctuated_data": dict(),
    }
    post_id = uuid.uuid4()
    logger.debug(f'Принят новый "post_file"  id = {post_id}')

    if "audio" in file.content_type:
        posted_and_downloaded_audio[post_id] = AudioSegment.from_file(file.file)
    else:
        result["success"] = False
        result["error_description"] = "Not audio file received"

        return result

    # Приводим фреймрейт к фреймрейту модели
    if posted_and_downloaded_audio[post_id].frame_rate != config.BASE_SAMPLE_RATE:
        posted_and_downloaded_audio[post_id] = posted_and_downloaded_audio[post_id].set_frame_rate(
            config.BASE_SAMPLE_RATE)

    # Обрабатываем чанки с аудио по 15 секунд.
    for n_channel, mono_data in enumerate(posted_and_downloaded_audio[post_id].split_to_mono()):

        audio_buffer[post_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
        audio_overlap[post_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
        audio_duration[post_id] = 0

        result['raw_data'].update({f"channel_{n_channel + 1}": list()})

        for overlap in mono_data[::config.MAX_OVERLAP_DURATION * 1000]:

            audio_buffer[post_id] = overlap  # Кривизна вызвана особенностями реализации буфера в сокетах
            # Если кусок меньше заданного в конфиге, то это последние секунды аудио. И его нужно обработать полностью.
            if overlap.duration_seconds == config.MAX_OVERLAP_DURATION:
                find_last_speech_position(post_id)
            else:
                # По этому на распознавание подаём хвост от предыдущего + текущий кусок. В надежде, что суммарная
                # продолжительность не превысит максимальную? Самонадёянно, конечно.
                audio_to_asr[post_id] = audio_overlap[post_id] + overlap
                audio_overlap[post_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)

            # Обрабатываем основной массив данных
            try:
                if config.MODEL_NAME == "Gigaam" or config.MODEL_NAME == "Whisper":
                    asr_result_wo_conf = await simple_recognise(audio_to_asr[post_id])

                    asr_result = await process_gigaam_asr(asr_result_wo_conf, audio_duration[post_id])
                    audio_duration[post_id] += audio_to_asr[post_id].duration_seconds
                    logger.debug(asr_result)

                else:
                    asr_result_w_conf = recognise_w_calculate_confidence(audio_to_asr[post_id],
                                                                         num_trials=config.RECOGNITION_ATTEMPTS)
                    asr_result = await process_asr_json(asr_result_w_conf, audio_duration[post_id])
                    audio_duration[post_id] += audio_to_asr[post_id].duration_seconds
                    logger.debug(asr_result)
            except Exception as e:
                logger.error(f"Error ASR audio - {e}")
                error_description = f"Error ASR audio - {e}"
            else:
                result['raw_data'][f"channel_{n_channel + 1}"].append(asr_result)
                res = True

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

    if params.do_dialogue:
        try:
            result["sentenced_data"] = await do_sensitizing(result["raw_data"], params.do_punctuation)
        except Exception as e:
            logger.error(f"await do_sensitizing - {e}")
            error_description = f"do_sensitizing - {e}"
            res = False
        else:
            if not params.keep_raw:
                result["raw_data"].clear()
    else:
        result["sentenced_data"].clear()




    result['error_description'] = error_description
    result['success'] = res

    del posted_and_downloaded_audio[post_id]

    return result

