import os

# server settings
host = os.getenv('HOST', '0.0.0.0')
port = int(os.getenv('PORT', 49153))

model_name = "Gigaam"  ## Vosk5SmallStreaming  Vosk5 Gigaam

base_sample_rate=8000  # Стрим из астериска отдаёт только 8к
MAX_OVERLAP_DURATION = 15  # Максимальная продолжительность буфера аудио (зависит от модели)
RECOGNITION_ATTEMPTS = 1
PROVIDER = "CUDA"