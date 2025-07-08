import logging
import asyncio

from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from pathlib import Path

import config
from models.fast_api_models import PostFileRequest
from utils.save_asr_to_md import save_to_md

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        # Todo - перенести paths в отдельный файл и импортировать его по необходимости
        from Recognizer.engine.file_recognition import process_file
        logging.info(f"Получено сообщение о новом файле {event.src_path}")
        if not event.is_directory:
            file_params = PostFileRequest()
            file_params.speech_speed_correction_multiplier = 1
            file_params.do_diarization = True
            file_params.do_punctuation = True
            file_params.do_dialogue = True
            file_params.do_echo_clearing = False
            file_params.make_mono = True
            file_params.diar_vad_sensity = 2
            file_params.do_auto_speech_speed_correction = True
            file_params.keep_raw = False

            file_path = Path(event.src_path)

            asr_data = process_file(
                tmp_path= file_path,
                params=file_params
            )

            if asr_data:
                asyncio.run(save_to_md(asr_data))

            if config.DELETE_LOCAL_FILE_AFTR_ASR:
                try:
                    file_path.unlink()
                    logging.info(f"Файл {file_path} удалён")
                except FileNotFoundError:
                    logging.info(f"Не могу удалить файл {file_path} - не существует")
                except PermissionError:
                    logging.error(f"Нет прав на удаление файла {file_path}")
                except Exception as e:
                    logging.error(f"Ошибка при удалении файла: {e}")
                finally:
                    logging.info(f"Работы с файлом {file_path} завершены")



def start_file_watcher(file_path: str) -> Observer:
    event_handler = NewFileHandler()
    # observer = Observer()
    observer = PollingObserver()
    observer.schedule(event_handler=event_handler, path=file_path, recursive=True)
    observer.start()
    return observer