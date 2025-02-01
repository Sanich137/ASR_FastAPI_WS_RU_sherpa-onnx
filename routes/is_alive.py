from utils.pre_start_init import app
import logging
import datetime
import os
import psutil


@app.get("/is_alive")
async def check_if_service_is_alive():
    """Перед направлением запроса, проверить наличие свободных мест. На практике, похоже, нормальная скорость
    обеспечивается при неболее чем 1 к 1 к количеству физических ядер процессора. """
    # Todo - запустить проверку сервиса - направить запрос и получить ответ. Сверить с ожидаемым ответом и, если ОК,
    #  Возвращать True
    logging.debug('GET_is_alive')
    logging.info(f'Сервер запущен')


    return {"error": False,
            "error_description": None,
            "data": "is running"}