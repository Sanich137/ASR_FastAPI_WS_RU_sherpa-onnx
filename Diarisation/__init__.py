import config
from utils.pre_start_init import paths
from utils.do_logging import logger
from VoiceActivityDetector import vad
import requests


if config.CAN_DIAR:
    if not paths.get("diar_speaker_model_path").exists():
        logger.error(f"Модель для диаризации не найдена. Предпринимаются попытки скачать {config.DIAR_MODEL_NAME}.onnx")
        output_path = paths.get("diar_speaker_model_path")
        api_url = "https://modelscope.cn/api/v1/datasets/wenet/wespeaker_pretrained_models/oss/tree"

        try: # Предпринимаем попытки скачать и положить файл в нужное место.
            # 1. Получаем JSON с данными файлов
            response = requests.get(api_url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            # 2. Ищем нужный файл
            target_file = "voxblink2_samresnet100_ft.onnx"
            file_data = next((item for item in response.json()["Data"] if item["Key"] == target_file), None)

            if not file_data:
                raise Exception("Файл не найден в API") # Todo - указать тут список доступных моделей из response.json()["Data"]

            # 3. Скачиваем по прямой ссылке
            download_url = file_data["Url"]
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            logger.error(f"Файл модели не сохранён.Ошибка - {e})")

        else:
            logger.info(f"Модель успешно загружена : {output_path}")

    else:
        logger.info(f"Будет использован имеющийся файл {config.DIAR_MODEL_NAME}")


    if not paths.get("diar_speaker_model_path").exists():
        logger.error(f"Модель для Диаризации отсутствует. Диризация выключена и будет не доступна.")
        logger.error(f"Скачайте модель со страницы 'https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md' "
                     f"и поместите в по адресу: {str(paths.get("diar_speaker_model_path"))}")
        config.CAN_DIAR = False
    else:
        from .do_diarize import Diarizer

        diarizer = Diarizer(embedding_model_path=paths.get("diar_speaker_model_path"),
                            vad=vad,
                            max_phrase_gap=1,
                            batch_size=32,
                            cpu_workers=0,
                            use_gpu=False)

        logger.info(f"Успешно загружена модель Диаризации")
else:
    logger.info(f"Диаризация не включена и будет недоступна.")