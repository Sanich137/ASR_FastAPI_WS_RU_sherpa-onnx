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
IS_PROD = True if int(os.getenv('IS_PROD', 1))==1 else False

# Recognition_settings
MAX_OVERLAP_DURATION = 15  # Максимальная продолжительность буфера аудио (зависит от модели) приемлемый диапазон 10-15 сек.
RECOGNITION_ATTEMPTS = 1  # Пока не менять

# Vad_settings
VAD_SENSITIVITY = int(os.getenv('VAD_SENSE', 3))  # 1 to 5 Higher - more words.
VAD_WITH_GPU = os.getenv('VAD_WITH_GPU', False)

# Punctuate_settings
PUNCTUATE_WITH_GPU = os.getenv('VAD_WITH_GPU', True)   # Если потребуется onnxruntime > 1.17.1, то изменить на False (ограничения sherpa-onnx)

# Diarisation_settings
CAN_DIAR = True if int(os.getenv('CAN_DIAR', 0)) == 1 else False
DIAR_MODEL_NAME = str(os.getenv('DIAR_MODEL_NAME', "voxblink2_samresnet100_ft")+".onnx")
DIAR_WITH_GPU = os.getenv('DIAR_WITH_GPU', False)
CPU_WORKERS = int(os.getenv('CPU_WORKERS', 0)) # Для значений меньше 1 будут использованы все доступные ядра -1. При больше 1  - указанное число ядер CPU. Работает только при GPU True

# Разных моделей для диаризации много.
# Если voxblink2_samresnet100_ft работает на вашей мощности не достаточно быстро, выберите модель меньшего размера:
# [('cnceleb_resnet34.onnx', 25), ('cnceleb_resnet34_LM.onnx', 25), ('voxblink2_samresnet100.onnx', 191),
# ('voxblink2_samresnet100_ft.onnx', 191), ('voxblink2_samresnet34.onnx', 96), ('voxblink2_samresnet34_ft.onnx', 96),
# ('voxceleb_CAM++.onnx', 27), ('voxceleb_CAM++_LM.onnx', 27), ('voxceleb_ECAPA1024.onnx', 56),
# ('voxceleb_ECAPA1024_LM.onnx', 56), ('voxceleb_ECAPA512.onnx', 23), ('voxceleb_ECAPA512_LM.onnx', 23),
# ('voxceleb_gemini_dfresnet114_LM.onnx', 24), ('voxceleb_resnet152_LM.onnx', 75), ('voxceleb_resnet221_LM.onnx', 90),
# ('voxceleb_resnet293_LM.onnx', 109), ('voxceleb_resnet34.onnx', 25), ('voxceleb_resnet34_LM.onnx', 25)]


print(f"Using '{LOGGING_LEVEL}' LOGGING_LEVEL")