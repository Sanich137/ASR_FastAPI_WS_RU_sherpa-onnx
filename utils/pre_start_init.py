from utils.do_logging import logger
from asr_pipeline.state_manager import StateManager
from asr_pipeline.pipeline_config import PIPELINE_CONFIG
from asr_pipeline.routing import PipelineRouter
from asr_pipeline.worker import start_workers, stop_monitor, start_monitor
from contextlib import asynccontextmanager
from fastapi import FastAPI
from utils.globals import paths
from FileWhatcer import start_file_watcher
from config import DO_LOCAL_FILE_RECOGNITIONS
import threading

from db import RedisManager

try:
    redis_manager = RedisManager(
        host='127.0.0.1',
        port=6379,
        db=0,
        max_retries=5  # 5 попыток подключения
    )
    logger.info("Redis успешно инициализирован")
except ConnectionError as e:
    logger.critical(f"Не удалось инициализировать Redis: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Создаем роутер
        router = PipelineRouter(PIPELINE_CONFIG)

        # Создаем фабрику StateManager с использованием Redis
        def manager_factory():
            return StateManager(router, redis_manager)

        # Передаем RedisManager в start_workers
        start_workers(router= router, rm=redis_manager)

        # Сохраняем менеджер в состояние приложения
        app.state.manager = manager_factory()

        if DO_LOCAL_FILE_RECOGNITIONS:
            observer_thread = threading.Thread(
                target=lambda: start_file_watcher(file_path=str(paths.get("local_recognition_folder"))),
                daemon=True
            )
            observer_thread.start()
            logger.info("File watcher started")

        # Запуск мониторинга очередей
        start_monitor(redis_manager)

        yield  # Здесь приложение работает

    except Exception as e:
        logger.critical(f"Критическая ошибка при инициализации: {str(e)}")
        raise
    finally:
        stop_monitor.set()
        logger.debug("Приложение FastAPI остановлено")

app = FastAPI(
    lifespan=lifespan,
    version="1.0",
    docs_url='/docs',
    root_path='/root',
    title='ASR on SHERPA-ONNX'
)