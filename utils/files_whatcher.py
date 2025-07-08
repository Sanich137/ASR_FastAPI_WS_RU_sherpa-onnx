from utils.do_logging import logger
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            file_path = Path(event.src_path)
            print(f"Обнаружен новый файл: {file_path}")


def start_file_watcher(path: str) -> Observer:
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler=event_handler, path=path, recursive=True)
    observer.start()
    return observer