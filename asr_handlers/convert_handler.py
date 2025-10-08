from pydub import AudioSegment
from config import BASE_SAMPLE_RATE
from utils.resamppling import resample_audiosegment
from utils.do_logging import logger
from models.pipeline_model import ProcessingState


async def convert_handler(data: ProcessingState) -> ProcessingState:

    logger.info(f'Получено задание в convert_handler')
    data.results.success = False
    request_id = data.request_id
    params =  data.params
    tmp_path = data.stage_results.bytes_buffer


    try:
        if params.make_mono:
            posted_and_downloaded_audio = AudioSegment.from_file(tmp_path).set_channels(1)
        else:
            posted_and_downloaded_audio = AudioSegment.from_file(tmp_path)
    except Exception as e:
        error_description = f"Error loading audio file: {e}"
        logger.error(error_description)
        data.results.error_description = error_description
        return data

    # Проверка длины переданного на распознавание аудио
    try:
        if posted_and_downloaded_audio.duration_seconds < 5:
            logger.debug(f"На вход передано аудио короче 5 секунд. Будет дополнено тишиной ещё 5 сек.")
            posted_and_downloaded_audio += AudioSegment.silent(duration=5,
                                                                        frame_rate=BASE_SAMPLE_RATE)
    except Exception as e:
        error_description = f"Error len_fixing_file: {e}"
        logger.error(error_description)
        data.results.error_description = error_description
        return data

    # Приводим фреймрейт к фреймрейту модели
    try:
        if posted_and_downloaded_audio.frame_rate != BASE_SAMPLE_RATE:
            posted_and_downloaded_audio = await resample_audiosegment(
                audio_data=posted_and_downloaded_audio,
                target_sample_rate=BASE_SAMPLE_RATE)

    except KeyError as e_key:
        error_description = f"Ошибка обращения по ключу {request_id} при изменения фреймрейта - {e_key}"
        logger.error(error_description)
        data.results.error_description = str(error_description)

    except Exception as e:
        error_description = f"Ошибка изменения фреймрейта - {e}"
        logger.error(error_description)
        data.results.error_description = str(error_description)
    else:
        data.stage_results.posted_and_downloaded_audio = posted_and_downloaded_audio
        data.results.success = True

    logger.info(f'Получен ответ в convert_handler')

    return data
