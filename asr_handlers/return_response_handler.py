from utils.do_logging import logger
from models.pipeline_model import ProcessingState


async def return_response_handler(data: ProcessingState) -> ProcessingState:

    logger.debug(f'Получено задание в return_response_handler')
    return data
    # предусмотреть логику
