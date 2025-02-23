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
from utils.get_audio_file import getting_audiofile
from utils.chunk_doing import find_last_speech_position

from models.fast_api_models import SyncASRRequest

from Recognizer.engine.stream_recognition import recognise_w_calculate_confidence, simple_recognise


@app.post("/post_one_step_req")
async def post(p:SyncASRRequest):
    """
    На вход принимает HttpUrl - прямую ссылку на скачивание файла 'mp3', 'wav' или 'ogg'.\n
    Если на вход передаётся не моно, то ответ будет в несколько элементов списка для каждого канала.\n
    По умолчанию отдаёт сырой результат распознавания с разбивкой на части продолжительностью около 15 секунд\n
    :param: do_dialogue: - true, если нужно разбить речь на диалог - В проекте.\n
    :param: do_punctuation - true, если нужно расставить пунктуацию. Применяется к диалогу, общему тексту. В проекте.\n
    При проектировании таймаутов учитывайте скорость распознавания (около 100 секунд за 2 секунды одного канала)
    """

    result = {
        "success":False,
        "error_description": str(),
        "data": dict()
    }

    post_id = uuid.uuid4()
    logger.info(f'Принят новый "post_one_step_req"  id = {post_id}')
    audio_buffer[post_id] = AudioSegment.silent(100, frame_rate=config.base_sample_rate)
    audio_overlap[post_id] = AudioSegment.silent(100, frame_rate=config.base_sample_rate)
    audio_duration[post_id] = 0

    data = None
    res, error =  await getting_audiofile(p.AudioFileUrl, post_id)

    if res:
        posted_and_downloaded_audio[post_id] = AudioSegment.from_file(posted_and_downloaded_audio[post_id])

        # Приводим фреймрейт к фреймрейту модели
        if posted_and_downloaded_audio[post_id].frame_rate != config.base_sample_rate:
            posted_and_downloaded_audio[post_id] = posted_and_downloaded_audio[post_id].set_frame_rate(config.base_sample_rate)

        # Обрабатываем чанки с аудио по 15 секунд.
        for n_channel, mono_data in enumerate(posted_and_downloaded_audio[post_id].split_to_mono()):
            result['data'].update({f"channel_{n_channel + 1}": list()})
            for overlap in mono_data[::config.MAX_OVERLAP_DURATION*1000]:
                audio_buffer[post_id] = overlap  # Кривизна вызвана особенностями реализации буфера в сокетах
                find_last_speech_position(post_id)
                try:
                    if config.model_name == "Gigaam":
                        asr_result_wo_conf =simple_recognise(audio_to_asr[post_id])

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
                else:
                    result['data'][f"channel_{n_channel + 1}"].append(asr_result)

    else:
        logger.error(f'Ошибка получения файла - {error}, ссылка на файл - {p.AudioFileUrl}')
        result['error_description'] = error
        result['success'] = res

    del audio_overlap[post_id]
    del audio_buffer[post_id]
    del audio_to_asr[post_id]
    del audio_duration[post_id]
    del posted_and_downloaded_audio[post_id]


    return result
