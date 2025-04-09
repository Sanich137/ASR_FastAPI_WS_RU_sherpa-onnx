import config
from .do_vad import SileroVAD
from utils.pre_start_init import paths
from utils.do_logging import logger
import requests

#
# try:
#     vad = SileroVAD("silero_v5__vad_orig.onnx")
# except Exception as e:
#     logger.error(f'Модель для vad не загружена. Работа не возможна.')
# else:
#     logger.info(f"загружена VAD модель ")

if not paths.get("vad_model_path").exists():
    logger.info("Модель silero_vad.onnx отсутствует. Предпринимаем попытку скачать её.")
    url = "https://github.com/snakers4/silero-vad/blob/v5.0/files/silero_vad.onnx?raw=true"
    output_file = str(paths.get("vad_model_path"))

    try:
        response = requests.get(url, stream=True)
    except Exception as e:
        logger.error(f"Ошибка выполнения запроса на скачивание модели - {e}")
    else:
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
            logger.info("Модель silero_vad.onnx успешно загружена")
        else:
            logger.error(f"Ошибка при скачивании файла. Статус: {response.status_code}")
else:
    logger.info("Будет использован VAD silero v5")