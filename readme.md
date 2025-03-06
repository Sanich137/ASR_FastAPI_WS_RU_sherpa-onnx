# Сервер для распознавания речи на базе FastAPI

Этот проект представляет собой сервер для распознавания речи, использующий модели Vosk и GigaAM. Сервер реализован на базе FastAPI и поддерживает как потоковую передачу аудио, так и обработку аудиофайлов через POST-запросы.

## Основные особенности

- Поддержка двух моделей распознавания речи: **Vosk** и **GigaAM**.
- Отправка на распознавание **файла**, **сслыки на файл** или **файла аудио-потоком**. 
- Поддержка как CPU, так и GPU (с использованием CUDA).
- Реализация буфера аудио потока с возможностью настройки размера чанка.
- Интеграция с FastAPI для обработки входящих запросов и постобработки текста.
- При обработке файлом (пока не потока) можно строить "диалог"
- При обработке файла (пока не потока) можно удалять межканальное эхо (текст, не аудио)


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
### Загрузка модели Vosk
```bash
cd models && git clone https://huggingface.co/alphacep/vosk-model-ru && cd ..
```
### Загрузка модели GigaAM
```bash 
cd models && git clone https://huggingface.co/Alexanrd/GigaAMv2_CTC_RU_ASR_for_sherpa_onnx && cd ..
```

### Загрузка модели для пунктуации. Обязательное условие.
```bash 
cd models && git clone https://huggingface.co/Alexanrd/sbert_punc_case_ru_onnx && cd ..
```



### Конфигурация
- Переменные в окружении:
```
LOGGING_LEVEL="INFO"   # доступные значения: INFO, DEBUG
NUM_THREADS=4          # Желательно не менее 2
HOST="0.0.0.0"
PORT=49153
MODEL_NAME=Gigaam      # доступные значения: Vosk5 и Gigaam. 
BASE_SAMPLE_RATE=8000  # для Gigaam лучше 8000, для Vosk5 - 16000 
PROVIDER=CUDA          # доступные значения: CUDA и CPU 
IS_PROD=1              # Влияет на логирование. Если 1, то логи пишем в файл. Если 0, то выводим в консоль.
```


## Запуск и тестирование
#### Запуск сервера
```bash
cd /opt/ASR_FastAPI_WS_RU_sherpa-onnx
source venv/bin/activate
python3 main.py
```

#### Тестирование.

Для тестирования можно использовать [WS_Test.py](WS_Test.py), перейти на страницу документации API:
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

- Внимание! Проверяйте допустимость коммерческого использования моделей перед их использованием в коммерческих проектах.

## Заключение
- Если у вас возникли вопросы или предложения по улучшению проекта, пожалуйста, свяжитесь со мной через [GitHub Issues](https://github.com/Sanich137/ASR_FastAPI_WS_RU_sherpa-onnx/issues).

## Работы
- 06 марта 2025 - Реализована пунктуация в полной мере при работе с целыми файлами.(CPU). Для работы модель для пунктуации скачать обязательно.
- 03 марта 2025 - Реализована Тестовая страница для демонстрации функционала
- 28 февраля 2025 - Критическое обновление - переезд на CUDA 12.4 Для обновления пройдите весь ридми заново 

## Планируемы работы
- реализовать Разделение на предложения и пунктуацию при передаче задания в сокетах.
- реализовать постобработку аудио для улучшения разборчивости аудио человеком.

