> Сервер для Распознавания на базе Vosk 5 версии.

- В работе можно будет использовать три версии - small-streaming, vosk-model-ru и gigaam. Сейчас используется vosk-model-ru.  
- Отличия в работе. 
  - Стриминговая версия с каждым новым чанком **будет** отдавать текст с накоплением. Говорят, оно работает быстрее.
  на самом деле Sherpa-onnx настолько быстр, что разницы быть не должно.
  - Полная версия в реализации интересна написанным буфером аудио потока. Технически с каждым новым чанком 
  мы отсекаем последние микрочанки с голосом и отправляем на распознавание всё остальное. И, если вдруг весь чанк 
  это голос, то копим чанки в один мегачанк, но не более чем 15 секунд т.к. большая модель за один стрим есть только 18. 
  - gigaam пока не запускал

  
> FastApi используется в т.ч. для проверки входящих запросов, возможно, для выполнения дополнительных инструкций,
таких как постобработка текста, история запросов и м.б. что-то ещё чего прикручу.

> Модели:
> - большая https://huggingface.co/alphacep/vosk-model-ru
> - малая   https://huggingface.co/alphacep/vosk-model-small-ru
> - гигаам 

> По умолчанию приложение работает на CPU. Как следствие с увеличением количества запросов время подготовки ответа растёт.
> 

> Для поддержки GPU на линукс:
>
> Ставим Cuda 11.8 и cudNN к ней [по инструкции:](https://k2-fsa.github.io/k2/installation/cuda-cudnn.html#cuda-11-8) 
> 1. sudo apt install gcc
> 2. wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
> 3. chmod +x cuda_11.8.0_520.61.05_linux.run
> 4. sudo ./cuda_11.8.0_520.61.05_linux.run \
>   --silent \
>  --toolkit \
>  --installpath=/usr/local/cuda-11.8.0 \
>  --no-opengl-libs \
>  --no-drm \
>  --no-man-page \
>  --override
>
> 5. wget https://huggingface.co/csukuangfj/cudnn/resolve/main/cudnn-linux-x86_64-8.9.1.23_cuda11-archive.tar.xz
> 
> 6. sudo tar xvf cudnn-linux-x86_64-8.9.1.23_cuda11-archive.tar.xz --strip-components=1 -C /usr/local/cuda-11.8.0

Если у Вас установлено несколько версий CUDA, 
то используем файл для переключения на cuda 11.8 [activate-cuda-11.8.sh](activate-cuda-11.8.sh) 
и переключаем на него:
> - source activate-cuda-11.8.sh


Установка:

1. Ставим основной пакет.
> - pip install -r requirements.txt
2. Не забываем поставить git-lfs и ffmpeg:
> - sudo apt-get install -y git-lfs 
> - sudo apt install -y ffmpeg
    
- Детали тут: https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=windows
    
- Если не поставить, то получите ошибку: RuntimeError: Failed to load model because protobuf parsing failed.

3. Затем:
>- cd models
>- git clone https://huggingface.co/alphacep/vosk-model-ru

В окружение добавляем переменные:

LOGGING_LEVEL = INFO (уровень логирования - по умолчанию DEBUG)
NUM_THREADS = 2 (количество потоков на распознавание. Нормально работает от двух и выше.)
HOST = "0.0.0.0.0" 
PORT = "49153" 

> Проверяем запуск на http://127.0.0.1:49153/docs#/ 

> Тестирование сокетов - [WS_Test.py](WS_Test.py)

