from io import BytesIO
from utils.do_logging import logger
from models.pipeline_model import ProcessingState


async def receive_handler(data: ProcessingState) -> ProcessingState:
    data.results.success = False
    logger.debug(f'Получено задание в receive_handler')
    file_content = data.stage_results.file_content

    try:
        buffer = BytesIO(file_content)
        buffer.seek(0)
        data.stage_results.bytes_buffer = buffer
    except Exception as e:
        error_description = e
        logger.error(f"Ошибка создания буфера: {str(error_description)}")
    else:
        data.results.success = True
    finally:
        logger.debug(f'Возвращен результат из receive_handler')
        return data
