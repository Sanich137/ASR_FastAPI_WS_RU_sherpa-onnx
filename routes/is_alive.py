from utils.pre_start_init import app
import logging
import datetime
import os
import pynvml


def get_gpu_free_memory():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Первая видеокарта
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mb = mem_info.free / 1024**2
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_load = utilization.gpu
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except pynvml.NVMLError as e:
        return {"error": str(e)}
    finally:
        pynvml.nvmlShutdown()
    return free_mb, gpu_load,temperature


@app.get("/is_alive")
async def check_if_service_is_alive():

    logging.info('GET_is_alive')
    tasks_in_work = "Временно отключено"

    free_mb, gpu_load,temperature = get_gpu_free_memory()

    if tasks_in_work == 0:
        state = "idle"
    else:
        state = "in_work"

    return {"error": False,
            "error_description": None,
            "state": state,
            "tasks_in_work": tasks_in_work,
            "free_memory_mb": free_mb,
            "gpu_load_percent": gpu_load,
            "temperature_celsius": temperature
            }