> Инструкция по установке и запуске контейнера.
1. Файл [Dockerfile](Dockerfile_GigaAM) создан для запуска на GPU Nvidia.
2. В образе уже будет установлена модель GigaAMv2

> Настройка оборудования.
1. Установите [Nvidia container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), 
чтобы драйвер видеокарты был доступен в контейнере. 

> Работа с образом.

4. Создаём образ:
```commandline
docker build -t asr /path/to/Dockerfile_GigaAM
```

3. Запускаем образ:
```commandline
docker run --runtime=nvidia -it --rm -p 8888:49153 asr
```

4. Сервис будет доступен по адресу http://127.0.0.1:8888/docs#

Работоспособность проверена на Win11 wsl2.
