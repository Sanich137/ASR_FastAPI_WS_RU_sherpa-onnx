import time
from typing import Callable, Awaitable, Dict, Any
import asyncio
from functools import partial

import Recognizer
from asr_pipeline.state_manager import StateManager
from asr_pipeline.pipeline_config import PIPELINE_CONFIG
from utils.do_logging import logger
import threading

# Создаём очереди из конфигурации
queues = {stage.name: asyncio.Queue() for stage in PIPELINE_CONFIG}

stop_monitor = threading.Event() # Флаг для остановки монитора

def monitor_queues():
    while not stop_monitor.is_set():
        output = []
        for stage in PIPELINE_CONFIG:
            # Используем stage.name как ключ
            queue_size = queues[stage.name].qsize()
            output.append(f"{stage.name}: {queue_size}")

        # Выводим состояние очередей (можно заменить на лог)
        print("\r" + " | ".join(output), end="")

        time.sleep(1)


def start_monitor():
    monitor_thread = threading.Thread(target=monitor_queues, daemon=True)
    monitor_thread.start()

async def worker(stage: str, handler: Callable, manager: StateManager):
    while True:
        request_id = await queues[stage].get()
        try:
            # Получаем текущее состояние по id
            state = await manager.get_state(request_id)

            handler_response = await handler(state)

            next_stage = await manager.update_state(handler_response, stage)

            if next_stage:
                await queues[next_stage].put(request_id)

        except Exception as e:
            logger.error(f"Error in {stage}: {str(e)}")
        finally:
            queues[stage].task_done()

def start_workers(manager: StateManager) -> None:
    """
    Запускает воркеры для всех этапов пайплайна.
    """
    for stage_config in PIPELINE_CONFIG:
        stage_name = stage_config.name
        handler_func = stage_config.handler
        num_workers = stage_config.num_workers
        extra_args = stage_config.extra_args

        # Создаём указанное количество воркеров для этапа
        for worker_idx in range(num_workers):
            # Для ASR создаём отдельный экземпляр распознавателя
            if stage_name == "asr":
                # Инициализация распознавателя
                extra_args["recognizer"] = Recognizer.get_recognizer()

            # Создаем частичную функцию с привязанными аргументами
            handler = partial(handler_func, **extra_args)

            worker_name = f"{stage_name}_worker_{worker_idx + 1}"

            # Создаём задачу с именем для отладки
            task = asyncio.create_task(
                worker(
                    stage=stage_name,
                    handler=handler,
                    manager=manager
                )
            )

            if not hasattr(manager, "worker_tasks"):
                manager.worker_tasks = []
            manager.worker_tasks.append(task)

            logger.debug(f"Started {worker_name} for stage '{stage_name}'")

    logger.info(f"Pipeline started with {len(PIPELINE_CONFIG)} stages")

    # Логируем детали по каждому этапу
    for stage_config in PIPELINE_CONFIG:
        logger.debug(
            f"Stage '{stage_config.name}': "
            f"{stage_config.num_workers} worker(s), "
            f"handler={stage_config.handler.__name__}"
        )