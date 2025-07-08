import json
from datetime import datetime
from utils.do_logging import logger

async def save_to_md(json_data:json):
    # Todo - Перенести path в отдельный файл или разобраться с циклическим импортом.
    from utils.pre_start_init import paths

    filename = datetime.now().strftime("%Y%m%d_%H%M%S.md")
    file_path = paths.get("result_local_recognition_folder") / filename

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        try:
            if len(json_data.get("diarized_data", "").keys()) > 1:
                list_full_text_only = json_data.get('sentenced_data', {}).get('list_of_sentenced_recognitions', [])
                for sentences  in list_full_text_only:
                    speaker = str(sentences .get('speaker', 'N/A'))
                    text = str(sentences .get('text', '')).strip()
                    f.write(f"Спикер_{speaker}: {text}\n")
            else:
                texts = json_data.get('sentenced_data', {}).get('full_text_only', [])
                for text in texts if isinstance(texts, list) else []:
                    f.write(str(text).strip() + "\n")
        except Exception as e:
            logger.error(f"Ошибка сохранения файла: {e}", exc_info=True)
            return False
        else:
            logger.info(f"Результат распознавания сохранён в файл {filename}")