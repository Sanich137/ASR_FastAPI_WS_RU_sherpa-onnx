# -*- coding: utf-8 -*-
import config
import logging
from logging.handlers import TimedRotatingFileHandler
from fastapi.logger import logger as fastapi_logger

logger = logging.getLogger(__name__)

# Формат лога — теперь начинается с времени
formatter = logging.Formatter(
    fmt=config.LOGGING_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S'  # Явно задаём формат даты
)

if config.IS_PROD:
    # Создаём обработчик с ротацией
    file_handler = TimedRotatingFileHandler(
        filename=config.FILENAME,
        when='midnight',
        interval=1,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(config.LOGGING_LEVEL)
    file_handler.setFormatter(formatter)

    # Добавляем обработчик к основному логгеру и fastapi_logger
    logger.addHandler(file_handler)
    fastapi_logger.addHandler(file_handler)

    # Устанавливаем уровень основного логгера
    logging.getLogger().setLevel(config.LOGGING_LEVEL)
else:
    # Для dev-среды — консольный вывод с тем же форматом
    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.LOGGING_LEVEL)
    console_handler.setFormatter(formatter)
    logging.basicConfig(
        level=config.LOGGING_LEVEL,
        format=config.LOGGING_FORMAT,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[console_handler]
    )