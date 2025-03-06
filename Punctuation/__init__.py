from utils.pre_start_init import paths
from utils.do_logging import logger
from .punctuate import SbertPuncCaseOnnx

import faulthandler
faulthandler.enable()


try:
    sbertpunc = SbertPuncCaseOnnx(paths.get("punctuation_model_path"))
except Exception as e:
    logger.error(f"Error gatting punctuation model - {e}")
else:
    logger.info(f'Успешно загружена модель Пунктуации')