from utils.do_logging import logger
from Recognizer.engine.echoe_clearing import remove_echo
from models.pipeline_model import ProcessingState


#  Todo Сюда перенести постобработку результатов ASR/

# post_asr_production

async def echo_clearing_handler(data: ProcessingState) -> ProcessingState:
    data.results.success = False
    logger.debug(f'Получено задание в echo_clearing_handler')  # post_asr_production

    try:
        data.results.raw_data = await remove_echo(data.results.raw_data)

    except Exception as e:
        error_description = f"Error echo clearing - {e}"
        logger.error(error_description)
        data.results.error_description = error_description
    else:
        data.results.success = True

    logger.debug(f'Возвращено задание из echo_clearing_handler')

    return data