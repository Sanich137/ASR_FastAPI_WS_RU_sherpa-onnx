# Сервер для распознавания речи на базе FastAPI

Этот проект представляет собой сервер для распознавания речи, использующий модели Vosk и GigaAM. Сервер реализован на базе FastAPI и поддерживает как потоковую передачу аудио, так и обработку аудиофайлов через POST-запросы.

## Основные особенности

- Поддержка двух моделей распознавания речи: **Vosk** и **GigaAM**.
- Отправка на распознавание **файла**, **ссылка на файл** или **файл аудио-потоком**. 
- Поддержка как CPU, так и GPU (с использованием CUDA).
- Реализация буфера аудио потока с возможностью настройки размера чанка.
- Интеграция с FastAPI для обработки входящих запросов и постобработки текста.
- При обработке можно строить "диалог" с пунктуацией. 
- При обработке файла или файла по ссылке можно удалять межканальное эхо. Для удаления эха обрабатывается текст.


## Установка и настройка

### Предварительные требования

- Убедитесь, что у вас установлены `python3`, `git`,`pip`, `git-lfs` и `ffmpeg`
- Для работы с GPU установите CUDA и cuDNN **обязательно!** по инструкции ниже.
- Если работа с GPU не требуется, "Установка CUDA и cuDNN (для GPU)" не выполняйте, в переменной окружения укажите `PROVIDER=CPU` 

#### Установка зависимостей

```bash
pip install -r requirements.txt
sudo apt install -y ffmpeg
sudo apt-get install git-lfs
sudo apt-get install libbz2-dev  # Если эти библиотеки не были установлены, придётся переустновить python после этих команд.
sudo apt-get install liblzma-dev  # Если эти библиотеки не были установлены, придётся переустновить python после этих команд.
``` 

#### Установка CUDA и cuDNN (для GPU)

```bash
sudo apt update && apt upgrade -y
sudo apt install gcc
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run &&
chmod +x cuda_12.4.0_550.54.14_linux.run
./cuda_12.4.0_550.54.14_linux.run \
  --silent \
  --toolkit \
  --installpath=/usr/local/cuda-12.4.0 \
  --no-opengl-libs \
  --no-drm \
  --no-man-page\
  --override
wget https://huggingface.co/csukuangfj/cudnn/resolve/main/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
tar xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz --strip-components=1 -C /usr/local/cuda-12.4.0
```
- Активация вновь установленного CUDA происходит с помощью [activate-cuda-12.4.sh](activate-cuda-12.4.sh) и действует до перезагрузки. 
```bash
source activate-cuda-12.4.sh
```
### Загрузка модели Vosk 0.52. До перехода sherpa-onnx на более новую версию onnxruntime использовать в этом проекте Vosk 0.54 невозможно.
```bash
git clone --filter=blob:none --no-checkout https://huggingface.co/alphacep/vosk-model-ru && \
cd vosk-model-ru && \
git checkout 0b81d4985ca88ccf8463cb222f9e284bb0ea06bb
```
### Загрузка модели GigaAM_CTC
```bash 
cd models && git clone https://huggingface.co/Alexanrd/GigaAMv2_CTC_RU_ASR_for_sherpa_onnx && cd ..
```

### Загрузка модели GigaAM_RNNT
```bash 
cd models && git clone https://huggingface.co/Alexanrd/GigaAMv2_RNNT_RU_ASR_for_sherpa_onnx && cd ..
```

### Загрузка модели для пунктуации. Обязательное условие.
```bash 
cd models && git clone https://huggingface.co/Alexanrd/sbert_punc_case_ru_onnx && cd ..
```



### Конфигурация
- Переменные в окружении:
```
# server
HOST="0.0.0.0"
PORT=49153

# ASR settings
MODEL_NAME=Gigaam      # доступные значения: Vosk5, Gigaam и Gigaam_rnnt 
BASE_SAMPLE_RATE=16000 # Частота дискретизации модели. К этой частоте будут приведены получаемые аудио. 
PROVIDER=CUDA          # доступные значения: CUDA и CPU 
NUM_THREADS=4          # Желательно не менее 2
MAX_OVERLAP_DURATION = 30 # Размер отправляемого на распознавание чанка. Для vosk < 18, для Gigaam_ctc 16-30, 
для Gigaam_rnnt можно от 10 до 30. 
BETWEEN_WORDS_PERCENTILE = 80 # Параметр определяет как мелко будет биться
# текст на предложения при подготовке диалога. Чем меньше значение, тем более короткие будут предложения.

# Logger settings
LOGGING_LEVEL="INFO"   # доступные значения: INFO, DEBUG
IS_PROD=1              # Влияет на логирование. Если 1, то логи пишем в файл. Если 0, то выводим в консоль.

# Vad settings
VAD_SENSITIVITY = 3 # Чувствительгность VAD при разделении аудио на чанки.
VAD_WITH_GPU = 0 # 1 - Использование GPU для работы VAD. Существенного прироста нет.

# Punctuate_settings
PUNCTUATE_WITH_GPU = 1 # Использование GPU для расстановки пунктуации.

# Diarisation_settings
CAN_DIAR = 0 # Включение и выключение возможности диаризации.
DIAR_MODEL_NAME = "voxceleb_resnet34_LM"  # Выбор модели для диаризации (скачает сам).
DIAR_WITH_GPU = 0 # 0 - использование для диаризации CPU, 1 - GPU
CPU_WORKERS = 0  # Количество CPU воркеров для диаризаии. 0 - решает onnxruntime (max).

# Настройки сервиса локального распознавания.
DO_LOCAL_FILE_RECOGNITIONS = 1 # Включить распознавание при помещении в папку файла.
DELETE_LOCAL_FILE_AFTR_ASR = 1 # Удалять файлы из папки после распознавания.
```

### Список доступных моделей для диаризации:
```text
В скобках указан размер модели.
Для работы на 8Gb GPU используйте модель размером не более 25 Мб, например, voxceleb_resnet34_LM. 
Лучшие результаты показывает voxblink2_samresnet100_ft, но в 8Gb карту вместе с ASR не поместится. 
Работать будет, но на GPU очень медленно.

('cnceleb_resnet34', 25), ('cnceleb_resnet34_LM', 25), ('voxblink2_samresnet100', 191), ('voxblink2_samresnet100_ft', 191),
('voxblink2_samresnet34', 96), ('voxblink2_samresnet34_ft', 96), ('voxceleb_CAM++', 27), ('voxceleb_CAM++_LM', 27),
('voxceleb_ECAPA1024', 56), ('voxceleb_ECAPA1024_LM', 56), ('voxceleb_ECAPA512', 23), ('voxceleb_ECAPA512_LM', 23),
('voxceleb_gemini_dfresnet114_LM', 24), ('voxceleb_resnet152_LM', 75), ('voxceleb_resnet221_LM', 90),
('voxceleb_resnet293_LM', 109), ('voxceleb_resnet34', 25), ('voxceleb_resnet34_LM', 25)

```


## Запуск и тестирование
#### Запуск сервера
```bash
cd /opt/ASR_FastAPI_WS_RU_sherpa-onnx
source venv/bin/activate
python3 main.py
```

#### Тестирование.

Для тестирования можно использовать [WS_Test.py](examples/streaming_client.py), перейти на страницу документации API:
```html
http://127.0.0.1:49153/docs#/
```
или на Demo страницу
```html
http://127.0.0.1:49153/demo
```

## Запуск как сервис в Ubuntu
- Для запуска приложения как сервиса в Ubuntu используйте пример файла [vosk_gpu.service](vosk_gpu.service). 
- Не забудьте изменить имя пользователя в файле и поместить его в /etc/systemd/system
- В случае запуска приложения как сервис, за корректное использование путей до CUDA будет отвечать файл [systemctl_environment](systemctl_environment)

``` bash
sudo cp vosk_gpu.service /etc/systemd/system/
```

### Запустите сервис:

```bash
sudo systemctl start vosk_gpu
```

### Проверьте работу сервиса:

```bash
journalctl -eu vosk_gpu -f
```

### Включите сервис в автозагрузку:

```bash
sudo systemctl enable vosk_gpu
```

# Лицензия и коммерческое использование

- Vosk: Модель доступна по лицензии [Apache 2.0](https://github.com/alphacep/vosk-api?tab=Apache-2.0-1-ov-file).
- GigaAM: Модель доступна по лицензии [MIT](https://github.com/salute-developers/GigaAM/blob/main/LICENSE).
- sbert_punc_case_ru: Модель доступна по лицензии [Apache 2.0](https://huggingface.co/kontur-ai/sbert_punc_case_ru).
- Silero: Модель доступна по лицензии [MIT](https://github.com/snakers4/silero-vad?tab=MIT-1-ov-file).

Внимание! Проверяйте допустимость коммерческого использования моделей перед их использованием в коммерческих проектах.

## Заключение
- Если у вас возникли вопросы или предложения по улучшению проекта, пожалуйста, 
свяжитесь со мной через [GitHub Issues](https://github.com/Sanich137/ASR_FastAPI_WS_RU_sherpa-onnx/issues) или [GitHub Discussions](https://github.com/Sanich137/ASR_FastAPI_WS_RU_sherpa-onnx/discussions)

## Работы
- 21 июля 2025 - реализован механизм распознавания переданных в папку файлов.
- 07 июля 2025 - реализован механизм улучшения распознавания при распознавании быстрой речи.
- 22 апреля 2025 - внедрена диаризация. Выбрать можно из [множества](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md) моделей wespeaker. (соблюдайте лицензии).
Использование опционально, возможна работа как на CPU, так и на GPU.
- 10 апреля 2025 - webrtcvad заменён на Silero v5, пунктуация производится силами GPU.
- 07 марта 2025 - Реализовано Разделение на предложения и пунктуацию при передаче задания в сокетах. 
- 06 марта 2025 - Реализована пунктуация в полной мере при работе с целыми файлами.(CPU). Для работы модель для пунктуации скачать обязательно.
- 03 марта 2025 - Реализована Тестовая страница для демонстрации функционала
- 28 февраля 2025 - Критическое обновление - переезд на CUDA 12.4 Для обновления пройдите весь ридми заново 

## Планируемы работы
 
- реализовать постобработку аудио для улучшения разборчивости аудио человеком.
- [возможно, деноиз от силеро](https://github.com/snakers4/silero-models?tab=readme-ov-file#models)
- уйти от VAD в Диаризации

