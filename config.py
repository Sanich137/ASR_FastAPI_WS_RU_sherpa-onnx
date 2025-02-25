import os

# server settings
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 49153))

# Model settings
MODEL_NAME = os.getenv('MODEL_NAME', "Gigaam")  ## Vosk5SmallStreaming  Vosk5 Gigaam Whisper
BASE_SAMPLE_RATE = os.getenv('BASE_SAMPLE_RATE', 8000)  # Стрим из астериска отдаёт только 8к
PROVIDER = os.getenv('PROVIDER',"CUDA")
NUM_THREADS = int(os.getenv('NUM_THREADS', 4))

# Logger settings
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL', 'DEBUG')

# Recognition_settings
MAX_OVERLAP_DURATION = 10  # Максимальная продолжительность буфера аудио (зависит от модели)
RECOGNITION_ATTEMPTS = 1  # Пока не менять
