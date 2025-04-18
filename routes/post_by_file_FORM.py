from pydub import AudioSegment
import config

import uuid
from utils.tokens_to_Result import process_asr_json, process_gigaam_asr
from utils.pre_start_init import (app,
                                  posted_and_downloaded_audio,
                                  audio_buffer,
                                  audio_overlap,
                                  audio_to_asr,
                                  audio_duration)

from utils.do_logging import logger
from utils.chunk_doing import find_last_speech_position

from models.fast_api_models import PostFileRequest

from Recognizer.engine.stream_recognition import recognise_w_calculate_confidence, simple_recognise
from Recognizer.engine.sentensizer import do_sensitizing
from Recognizer.engine.echoe_clearing import remove_echo

from fastapi import Depends, File, Form, UploadFile
from pydantic import BaseModel

from Diarisation.diarazer import do_diarizing

# Определение модели данных для параметров
class PostFileRequest(BaseModel):
    keep_raw: bool = True  # Значение по умолчанию
    do_echo_clearing: bool = False  # Значение по умолчанию
    do_dialogue: bool = False  # Значение по умолчанию
    do_punctuation: bool = False  # Значение по умолчанию
    do_diarization: bool = False  # Значение по умолчанию
    diar_vad_sensity: int = 4 # Чувствительность VAD для диаризации

# Функция для извлечения параметров из FormData
def get_file_request(
    keep_raw: bool = Form(default=True, description="Если используются дополнительные параметры, то сохранять или нет в выводе сырые данные."),  # Значение по умолчанию
    do_echo_clearing: bool = Form(default=False, description="Пытаться ли убирать межканальное эхо на основе повторяющихся или близких слов."),  # Значение по умолчанию
    do_dialogue: bool = Form(default=False, description="Строить ли диалог/разбивать ли на фразы при выводе текста."),  # Значение по умолчанию
    do_punctuation: bool = Form(default=False, description="Восстанавливать ли пунктуацию."),  # Значение по умолчанию
    do_diarization: bool = Form(default=False, description="Разделять ли речь на спикеров по голосу. Затраты времени где-то  1 к 10 в зависимости от количества ядер CPU"),  # Значение по умолчанию
    diar_vad_sensity: int = Form(default=3, description="Чувствительность VAD для диаризации. Для шумного или быстрого диалога рекомендуется повысить до 4"),  # Значение по умолчанию
) -> PostFileRequest:
    return PostFileRequest(
        keep_raw=keep_raw,
        do_echo_clearing=do_echo_clearing,
        do_dialogue=do_dialogue,
        do_punctuation=do_punctuation,
        do_diarization=do_diarization,
        diar_vad_sensity = diar_vad_sensity
    )


@app.post("/post_file")
async def receive_file(
    file: UploadFile = File(description="Аудиофайл для обработки"),
    params: PostFileRequest = Depends(get_file_request)
):
    res = False
    diarized = False
    error_description = str()

    result = {
        "success":res,
        "error_description": error_description,
        "raw_data": dict(),
        "sentenced_data": dict(),
    }

    post_id = uuid.uuid4()
    logger.debug(f'Принят новый "post_file"  id = {post_id}')

    if "audio" in file.content_type:
        posted_and_downloaded_audio[post_id] = AudioSegment.from_file(file.file)
    else:
        res = False
        error_description += "Not audio file received"

        return result

    # Приводим Файл в моно, если получен параметр "диаризация"
    if params.do_diarization and config.CAN_DIAR:  # Todo - добавить в реквест выбор канала для диаризации. Совместить с удалением эха.
        if posted_and_downloaded_audio[post_id].channels > 1:
            posted_and_downloaded_audio[post_id] = posted_and_downloaded_audio[post_id].split_to_mono()[1] # [1]  # [0:60000]
    elif params.do_diarization and not config.CAN_DIAR:
        params.do_diarization = False
        error_description += "Diarization is not available\n"
        logger.error("Запрошена диаризация, но она не доступна.")


    # Приводим фреймрейт к фреймрейту модели
    if posted_and_downloaded_audio[post_id].frame_rate != config.BASE_SAMPLE_RATE:
        posted_and_downloaded_audio[post_id] = posted_and_downloaded_audio[post_id].set_frame_rate(
            config.BASE_SAMPLE_RATE)

    # Обрабатываем чанки с аудио по 15 секунд.
    for n_channel, mono_data in enumerate(posted_and_downloaded_audio[post_id].split_to_mono() if
                                          posted_and_downloaded_audio[post_id].channels > 1
                                          else [posted_and_downloaded_audio[post_id]]):

        audio_buffer[post_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
        audio_overlap[post_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
        audio_duration[post_id] = 0

        result['raw_data'].update({f"channel_{n_channel + 1}": list()})

        for overlap in mono_data[::config.MAX_OVERLAP_DURATION * 1000]:

            if (audio_overlap[post_id].duration_seconds + overlap.duration_seconds) < 3:
                overlap+=AudioSegment.silent(3000, frame_rate=config.BASE_SAMPLE_RATE)

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
        # Todo - Придумать как совместить удаление эха и диаризацию.
        try:
            result["raw_data"] = await remove_echo(result["raw_data"])
        except Exception as e:
            logger.error(f"Error echo clearing - {e}")
            error_description = f"Error echo clearing - {e}"
            res = False
    if params.do_diarization:
        try:
            result["diarized_data"] = await do_diarizing(post_id, result['raw_data'],
                                                         diar_vad_sensity = params.diar_vad_sensity)
        except Exception as e:
            logger.error(f"await do_diarizing - {e}")
            error_description = f"do_diarizing - {e}"
            res = False
        else:
            diarized = True


    if params.do_dialogue:
        data_to_do_sensitizing = result["diarized_data"] if diarized else result["raw_data"]

        try:
            result["sentenced_data"] = await do_sensitizing(data_to_do_sensitizing,
                                                            do_punctuation=params.do_punctuation)
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


    logger.debug(result)

    return result

