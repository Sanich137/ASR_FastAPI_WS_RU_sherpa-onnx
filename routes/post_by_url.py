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
from utils.get_audio_file import getting_audiofile, open_default_audiofile
from utils.chunk_doing import find_last_speech_position

from models.fast_api_models import SyncASRRequest

from Recognizer.engine.stream_recognition import recognise_w_calculate_confidence, simple_recognise
from Recognizer.engine.sentensizer import do_sensitizing
from Recognizer.engine.echoe_clearing import remove_echo




@app.post("/post_one_step_req")
async def post(p:SyncASRRequest):
    """
    На вход принимает HttpUrl - прямую ссылку на скачивание файла 'mp3', 'wav' или 'ogg'.\n
    Если на вход передаётся не моно, то ответ будет в несколько элементов списка для каждого канала.\n
    По умолчанию отдаёт сырой результат распознавания с разбивкой на части продолжительностью около 15 секунд\n

    :param: do_dialogue: - true, если нужно разбить речь на диалог\n
    :param: do_punctuation - true, если нужно расставить пунктуацию. Применяется к диалогу, общему тексту. В проекте.\n
    При проектировании таймаутов учитывайте скорость распознавания (около 100 секунд аудио распознаётся за 2-5 секунд
    распознавания одного канала)
    """
    error = False
    result = {
        "success":error,
        "error_description": str(),
        "raw_data": dict(),
        "sentenced_data": dict(),
        "punctuated_data": dict(),
    }

    post_id = uuid.uuid4()
    logger.debug(f'Принят новый "post_one_step_req"  id = {post_id}')

    if p.AudioFileUrl:
        res, error =  await getting_audiofile(p.AudioFileUrl, post_id)
    else:
        res, error = await open_default_audiofile(post_id)

    if res:
        posted_and_downloaded_audio[post_id] = AudioSegment.from_file(posted_and_downloaded_audio[post_id])

        # Приводим фреймрейт к фреймрейту модели
        if posted_and_downloaded_audio[post_id].frame_rate != config.BASE_SAMPLE_RATE:
            posted_and_downloaded_audio[post_id] = posted_and_downloaded_audio[post_id].set_frame_rate(config.BASE_SAMPLE_RATE)

        # Обрабатываем чанки с аудио по 15 секунд.
        for n_channel, mono_data in enumerate(posted_and_downloaded_audio[post_id].split_to_mono()):

            audio_buffer[post_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
            audio_overlap[post_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
            audio_duration[post_id] = 0

            result['raw_data'].update({f"channel_{n_channel + 1}": list()})

            for overlap in mono_data[::config.MAX_OVERLAP_DURATION*1000]:

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
                else:
                    result['raw_data'][f"channel_{n_channel + 1}"].append(asr_result)

            del audio_overlap[post_id]
            del audio_buffer[post_id]
            del audio_to_asr[post_id]
            del audio_duration[post_id]

        if p.do_echo_clearing:
            try:
                result["raw_data"] = await remove_echo(result["raw_data"])
            except Exception as e:
                logger.error(f"Error echo clearing - {e}")

        if p.do_dialogue:
            try:
                result["sentenced_data"] = await do_sensitizing(result["raw_data"])
            except Exception as e:
                logger.error(f"await do_sensitizing - {e}")
                error = f"do_sensitizing - {e}"
            else:
                if not p.keep_raw:
                    result["raw_data"].clear()

    else:
        logger.error(f'Ошибка получения файла - {error}, ссылка на файл - {p.AudioFileUrl}')

    result['error_description'] = error
    result['success'] = res

    del posted_and_downloaded_audio[post_id]

    return result
