import os
import datetime

# server settings
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 49153))

# Model settings
MODEL_NAME = os.getenv('MODEL_NAME', "Gigaam")  ## Vosk5SmallStreaming  Vosk5 Gigaam Whisper
BASE_SAMPLE_RATE = int(os.getenv('BASE_SAMPLE_RATE', 16000))  # Стрим из астериска отдаёт только 8к
PROVIDER = os.getenv('PROVIDER',"CUDA")
NUM_THREADS = int(os.getenv('NUM_THREADS', 4))

# Logger settings
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL', 'DEBUG')
LOGGING_FORMAT = os.getenv('LOGGING_FORMAT', u'#%(levelname)-8s %(filename)s [LINE:%(lineno)d] [%(asctime)s]  %(message)s')
FILENAME = os.getenv('FILENAME', f'logs/ASR-{datetime.datetime.now().date()}.log')
FILEMODE = os.getenv('FILEMODE', 'a')
IS_PROD = int(os.getenv('IS_PROD', 1))

# Recognition_settings
MAX_OVERLAP_DURATION = 10  # Максимальная продолжительность буфера аудио (зависит от модели) приемлемый диапазон 10-15 сек.
RECOGNITION_ATTEMPTS = 1  # Пока не менять

print(f"Using '{LOGGING_LEVEL}' LOGGING_LEVEL")



