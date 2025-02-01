# -*- coding: utf-8 -*-

import os
from fastapi.logger import logging
import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(
#    filename=os.getenv('FILENAME', f'GPU-transcriber-{datetime.datetime.now().date()}.log'),
#    filemode=os.getenv('FILEMODE', 'a'),
    level=os.getenv('LOGGING_LEVEL', 'DEBUG'),
    format=os.getenv('LOGGING_FORMAT', u'#%(levelname)-8s %(filename)s [LINE:%(lineno)d] [%(asctime)s]  %(message)s')
    )

logger.info(f"Using LOGGING_LEVEL '{os.getenv('LOGGING_LEVEL', 'DEBUG')}'" )
logger.debug(f"Using LOGGING_LEVEL '{os.getenv('LOGGING_LEVEL', 'DEBUG')}'")

# Todo - удалить после настройки логирования sherpa

# if os.getenv('LOGGING_LEVEL','INFO') == "DEBUG":
#     logger.debug("Vosk full logging mode enabled")
#     SetLogLevel(0)
# else:
#     logger.debug("Vosk logging only error mode enabled")
#     SetLogLevel(-1)
