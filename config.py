import os

# server settings
host = os.getenv('HOST', '0.0.0.0')
port = int(os.getenv('PORT', 49153))

model_name = os.getenv('MODEL' "Gigaam")  ## Vosk5SmallStreaming  Vosk5 Gigaam Whisper
base_sample_rate=os.getenv('SAMPLE_RATE', 8000)  # Стрим из астериска отдаёт только 8к
PROVIDER = os.getenv('PROVIDER',"CUDA")




MAX_OVERLAP_DURATION = 12  # Максимальная продолжительность буфера аудио (зависит от модели)
RECOGNITION_ATTEMPTS = 1  # Пока не менять
