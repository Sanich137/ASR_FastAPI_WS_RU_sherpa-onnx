from utils.pre_start_init import app
import logging
import datetime
import os
# import psutil


@app.get("/is_alive")
async def check_if_service_is_alive():

    logging.debug('GET_is_alive')
    logging.info(f'Сервер запущен')


    return {"error": False,
            "error_description": None,
            "data": "is running"}