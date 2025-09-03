from utils.do_logging import logger
from models.pipeline_model import ProcessingState
from utils.globals import posted_and_downloaded_audio
from Diarisation.diarazer import do_diarizing
from config import CAN_DIAR


async def diarize_handler(data: ProcessingState) -> ProcessingState:
    logger.info(f'Получено задание в echo_clearing_handler')  # post_asr_production
    params = data.params
    request_id = data.request_id
    data.results.success = False

    if params.do_diarization and not CAN_DIAR:
        error_description = "Запрошена диаризация, но она не доступна."
        logger.warn(error_description)
        params.do_diarization = False
        data.results.error_description = error_description
    # Проверяем возможность диаризации. Если здесь стерео-канал, то диаризацию выключаем.
    elif params.do_diarization and len(posted_and_downloaded_audio[request_id].split_to_mono()) != 1:
        error_description = f"Only mono diarization available."
        logger.warn(error_description)
        params.do_diarization = False

    if params.do_diarization:
        try:
            data.results.diarized_data = await do_diarizing(
                file_id=str(request_id), asr_raw_data=data.results.raw_data, diar_vad_sensity=params.diar_vad_sensity
            )
        except Exception as e:
            error_description = f"do_diarizing - {e}"
            logger.error(error_description)
            data.results.error_description = error_description
        else:
            data.results.success = False
    else:
        data.results.success = True

    logger.info(f'Возвращено задание из echo_clearing_handler')

    return data