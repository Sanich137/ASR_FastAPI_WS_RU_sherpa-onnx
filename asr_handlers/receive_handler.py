import base64
from io import BytesIO
from utils.do_logging import logger
from models.pipeline_model import ProcessingState


async def receive_handler(data: ProcessingState) -> ProcessingState:
    data.results.success = False
    logger.debug(f'Получено задание в receive_handler')
    str_file_content = data.stage_results.file_content.encode("ascii")

    file_content = base64.decodebytes(str_file_content)

    try:
        buffer = BytesIO(file_content)
        buffer.seek(0)
        data.stage_results.bytes_buffer = base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        error_description = e
        logger.error(f"Ошибка создания буфера: {str(error_description)}")
    else:
        data.results.success = True
    finally:
        logger.debug(f'Возвращен результат из receive_handler')
        return data
