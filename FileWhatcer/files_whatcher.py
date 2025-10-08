import asyncio
import base64
import config
from utils.do_logging import logger
from models.fast_api_models import PostFileRequest, PostFileResponse
from models.pipeline_model import StageResults

def send_file_to_asr(event, file_path):
    from utils.pre_start_init import app
    from utils.pre_start_init import redis_manager  # Добавляем импорт redis_manager

    if not event.is_directory:
        file_params = PostFileRequest(
            speech_speed_correction_multiplier=1,
            do_diarization=config.CAN_DIAR,
            do_punctuation=config.CAN_PUNCTUATE,
            do_dialogue=True,
            do_echo_clearing=True,
            make_mono=config.MAKE_MONO,
            diar_vad_sensity=2,
            do_auto_speech_speed_correction=True,
            keep_raw=False,
        )

        logger.info(f"Получен новый файл")
        result = PostFileResponse(
            success=False,
            error_description=str(),
            raw_data=dict(),
            sentenced_data=dict(),
            diarized_data=dict()
        )

        with open(file_path, "rb") as f:
            buf = base64.b64encode(f.read()).decode('ascii')

        file_name = file_path.stem
        init_data = StageResults(
            file_content=buf,
        )

        try:
            # Создаем состояние через RedisManager
            request_id = asyncio.run(app.state.manager.create_state(
                result=result,
                init_data=init_data,
                params=file_params
            ))
            logger.debug(f"Создан запрос {request_id} для файла {file_name}")
        except Exception as e:
            error_description = str(e)
            logger.error(f"Не удалось создать состояние для запроса: {error_description}")
            result.error_description = error_description
            return result

        try:
            # Отправляем в Redis-очередь
            redis_manager.put("receive", request_id)
            logger.debug(f"Задача для {request_id} отправлена в очередь 'receive'")
        except Exception as e:
            error_description = str(e)
            logger.error(f"Не удалось отправить задачу в очередь: {error_description}")

    if config.DELETE_LOCAL_FILE_AFTR_ASR:
        try:
            file_path.unlink()
            logger.info(f"Файл {file_path} удалён")
        except FileNotFoundError:
            logger.info(f"Не могу удалить файл {file_path} - не существует")
        except PermissionError:
            logger.error(f"Нет прав на удаление файла {file_path}")
        except Exception as e:
            logger.error(f"Ошибка при удалении файла: {e}")
        finally:
            logger.info(f"Работы с файлом {file_path} завершены")

    return None