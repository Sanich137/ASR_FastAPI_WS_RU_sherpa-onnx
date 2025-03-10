from utils.pre_start_init import app
import logging
import datetime
import os
# import psutil
from utils.pre_start_init import audio_to_asr


@app.get("/is_alive")
async def check_if_service_is_alive():

    logging.debug('GET_is_alive')
    logging.info(f'Сервер запущен')

    tasks_in_work = len(audio_to_asr)

    if tasks_in_work == 0:
        state = "idle"
    else:
        state = "in_work"

    return {"error": False,
            "error_description": None,
            "state": state,
            "tasks_in_work": tasks_in_work}