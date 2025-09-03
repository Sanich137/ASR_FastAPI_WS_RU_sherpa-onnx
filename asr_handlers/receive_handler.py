from io import BytesIO
from utils.globals import bytes_buffer
from utils.do_logging import logger
from models.pipeline_model import ProcessingState



async def receive_handler(data: ProcessingState) -> ProcessingState:
    data.results.success = False
    logger.info(f'Получено задание в receive_handler')
    file_content = data.stage_results.get("file_content")

    try:
        buffer = BytesIO(file_content)
        buffer.seek(0)
        bytes_buffer[data.request_id] = buffer
    except Exception as e:
        error_description = e
        logger.error(f"Ошибка создания буфера: {str(error_description)}")
    else:
        data.results.success = True
    finally:
        logger.info(f'Возвращен результат из receive_handler')
        # Чистим промежуточные результаты.
        data.stage_results.clear()
        return data
