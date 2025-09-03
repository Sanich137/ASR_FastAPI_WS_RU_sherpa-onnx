from utils.globals import paths
from utils.do_logging import logger
from .punctuate import SbertPuncCaseOnnx
import config

try:
    sbertpunc = SbertPuncCaseOnnx(paths.get("punctuation_model_path"), use_gpu = config.PUNCTUATE_WITH_GPU)
except Exception as e:
    logger.error(f"Error getting punctuation model - {e}")
else:
    logger.info(f'Успешно загружена модель Пунктуации')