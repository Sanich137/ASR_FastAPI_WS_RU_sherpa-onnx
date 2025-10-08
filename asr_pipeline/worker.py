import atexit
import multiprocessing
import threading

from asr_pipeline.routing import PipelineRouter
import time
from typing import Callable, Awaitable, Dict, Any
import asyncio
from functools import partial

import Recognizer
from asr_pipeline.state_manager import StateManager
from asr_pipeline.pipeline_config import PIPELINE_CONFIG
from utils.do_logging import logger
from threading import Thread
from db import RedisManager

queue_lock = threading.Lock()

# # Создание очередей в Redis
# for stage in PIPELINE_CONFIG:
#     redis_manager.create_queue(stage.name)


stop_monitor = multiprocessing.Event() # Флаг для остановки монитора

def monitor_queues(redis_manager: RedisManager):
    while not stop_monitor.is_set():
        output = []
        for stage in PIPELINE_CONFIG:
            try:
                # Получаем размер очереди через Redis
                queue_name = f"queue:{stage.name}"
                queue_size = redis_manager.redis.llen(queue_name)
            except Exception as e:
                queue_size = "ERR"
                logger.error(f"Ошибка получения размера очереди {stage.name}: {e}")

            output.append(f"{stage.name}: {queue_size}")

        # Выводим состояние очередей
        print("\r" + " | ".join(output), end="")
        time.sleep(1)


def start_monitor(redis_manager: RedisManager):
    monitor_thread = threading.Thread(
        target=monitor_queues,
        args=(redis_manager,),
        daemon=True
    )
    monitor_thread.start()
    logger.info("Мониторинг очередей запущен")


async def async_worker(stage: str, handler: Callable, redis_manager: RedisManager, router: PipelineRouter):
    while True:
        try:
            # Получаем request_id из Redis-очереди
            request_id = await asyncio.get_event_loop().run_in_executor(
                None,
                redis_manager.get,
                stage
            )

            if not request_id:
                await asyncio.sleep(0.1)
                continue

            # Получаем состояние из Redis
            state = await asyncio.get_event_loop().run_in_executor(
                None,
                redis_manager.get_state,
                request_id
            )

            if not state:
                logger.error(f"Состояние не найдено для request_id: {request_id}")
                continue

            # Обрабатываем состояние
            handler_response = await handler(state)

            # Сохраняем обновленное состояние
            await asyncio.get_event_loop().run_in_executor(
                None,
                redis_manager.save_state,
                request_id,
                handler_response
            )

            # Обновляем состояние и получаем следующий этап
            next_stage = await redis_manager.update_state_and_get_next_stage(
                router,
                request_id,
                handler_response,
                stage
            )

            if next_stage:
                await redis_manager.put(next_stage, request_id)

        except Exception as e:
            logger.error(f"Ошибка в {stage}: {str(e)}")
            continue


def sync_worker(stage: str, handler_func: Callable, router: PipelineRouter, redis_host: str, redis_port: int,
                redis_db: int, extra_args: dict):
    # Создаем RedisManager внутри процесса
    redis_manager = RedisManager(
        host=redis_host,
        port=redis_port,
        db=redis_db
    )

    # Теперь передаем router в конструктор StateManager
    manager = StateManager(router, redis_manager)

    if stage == "asr":
        recognizer = Recognizer.get_recognizer()
        extra_args["recognizer"] = recognizer

    handler = partial(handler_func, **extra_args)

    while True:
        try:
            # Получаем request_id из Redis-очереди
            request_id = redis_manager.get(stage)
            if not request_id:
                time.sleep(0.1)
                continue

            # Получаем состояние из Redis
            state = redis_manager.get_state(request_id)
            if not state:
                logger.error(f"Состояние не найдено для request_id: {request_id}")
                continue

            # Обрабатываем состояние
            handler_response = handler(state)

            # Обновляем состояние через Redis
            next_stage = manager.update_state(handler_response, stage)

            if next_stage:
                redis_manager.put(next_stage, request_id)

        except Exception as e:
            logger.error(f"Ошибка в {stage}: {str(e)}")
            continue


def start_workers(router: PipelineRouter, rm: RedisManager):
    redis_manager = rm

    for stage_config in PIPELINE_CONFIG:
        stage_name = stage_config.name
        handler_func = stage_config.handler
        num_workers = stage_config.num_workers
        extra_args = stage_config.extra_args
        worker_type = stage_config.worker_type

        for worker_idx in range(num_workers):
            if worker_type == 'async':
                # Для асинхронных воркеров создаем задачу
                asyncio.create_task(
                    async_worker(
                        stage=stage_name,
                        handler=partial(handler_func, **extra_args),
                        redis_manager=redis_manager,
                        router=router
                    )
                )
            elif worker_type == 'process':
                # Передаем router и параметры подключения к Redis
                process = multiprocessing.Process(
                    target=sync_worker,
                    args=(
                        stage_name,
                        handler_func,
                        router,  # Добавляем router
                        redis_manager.host,
                        redis_manager.port,
                        redis_manager.db,
                        extra_args
                    ),
                    daemon=True
                )
                process.start()
            else:
                raise ValueError(f"Unknown worker type: {worker_type}")

            logger.debug(f"Started {stage_name}_worker_{worker_idx + 1}")

# def create_state_manager(router: PipelineRouter) -> StateManager:
#     return StateManager(router)

    #
    # # Добавляем обработку завершения работы
    # def worker_shutdown():
    #     if worker_type == 'process':
    #         for process in manager.worker_processes:
    #             process.terminate()
    #             process.join()
    #     else:
    #         for task in manager.worker_tasks:
    #             task.cancel()
    #
    # atexit.register(worker_shutdown)