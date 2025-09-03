from pydub import AudioSegment
from config import BASE_SAMPLE_RATE
from utils.resamppling import resample_audiosegment
from utils.globals import posted_and_downloaded_audio, bytes_buffer, request_parameters
from utils.do_logging import logger
from models.pipeline_model import ProcessingState


async def convert_handler(data: ProcessingState) -> ProcessingState:

    logger.info(f'Получено задание в convert_handler')

    request_id = data.request_id
    params =  data.params
    tmp_path = bytes_buffer.get(request_id)
    data.results.success = False

    try:
        if params.make_mono:
            posted_and_downloaded_audio[request_id] = AudioSegment.from_file(tmp_path).set_channels(1)
        else:
            posted_and_downloaded_audio[request_id] = AudioSegment.from_file(tmp_path)
    except Exception as e:
        error_description = f"Error loading audio file: {e}"
        logger.error(error_description)
        data.results.error_description = error_description

    # Проверка длины переданного на распознавание аудио
    try:
        if posted_and_downloaded_audio[request_id].duration_seconds < 5:
            logger.debug(f"На вход передано аудио короче 5 секунд. Будет дополнено тишиной ещё 5 сек.")
            posted_and_downloaded_audio[request_id] += AudioSegment.silent(duration=5,
                                                                        frame_rate=BASE_SAMPLE_RATE)
    except Exception as e:
        error_description = f"Error len_fixing_file: {e}"
        logger.error(error_description)
        data.results.error_description = error_description

    # Приводим фреймрейт к фреймрейту модели
    try:
        if posted_and_downloaded_audio[request_id].frame_rate != BASE_SAMPLE_RATE:
            posted_and_downloaded_audio[request_id] = await resample_audiosegment(
                audio_data=posted_and_downloaded_audio[request_id],
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
        data.results.success = True

    return data