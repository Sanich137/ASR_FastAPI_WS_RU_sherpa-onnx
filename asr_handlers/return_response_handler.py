from utils.do_logging import logger
from models.pipeline_model import ProcessingState
from config import DELETE_LOCAL_FILE_AFTR_ASR
from utils.save_asr_to_md import save_to_file

async def return_response_handler(data: ProcessingState) -> ProcessingState:
    logger.debug(f'Получено задание в return_response_handler')


    # if data.params.save_file:
    #     asyncio.run(save_to_file(asr_data, file_path))
    #
    #     if DELETE_LOCAL_FILE_AFTR_ASR:
    #         try:
    #             file_path.unlink()
    #             logger.debug(f"Файл {data.request_id} удалён")
    #         except FileNotFoundError:
    #             logger.info(f"Не могу удалить файл {data.request_id} - не существует")
    #         except PermissionError:
    #             logger.error(f"Нет прав на удаление файла {data.request_id}")
    #         except Exception as e:
    #             logger.error(f"Ошибка при удалении файла: {e}")
    #         finally:
    #             logger.info(f"Работы с файлом {data.request_id} завершены")


    return data
