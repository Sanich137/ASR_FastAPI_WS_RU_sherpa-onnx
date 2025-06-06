# -*- coding: utf-8 -*-
import config
from fastapi.logger import logging

logger = logging.getLogger(__name__)

if config.IS_PROD:
    logging.basicConfig(
        filename=config.FILENAME,
        filemode=config.FILEMODE,
        level=config.LOGGING_LEVEL,
        format=config.LOGGING_FORMAT,
        encoding = "UTF-8"
        )
else:
    logging.basicConfig(
        level=config.LOGGING_LEVEL,
        format=config.LOGGING_FORMAT,
        encoding = "UTF-8"
        )

