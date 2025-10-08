from uuid import UUID, uuid4
from typing import Optional, Dict, Any
import json
import uuid
from redis import Redis
from typing import Any, Dict
from utils.do_logging import logger
from models.pipeline_model import ProcessingState
from asr_pipeline.routing import PipelineRouter
from models.fast_api_models import PostFileRequest

import time


class RedisManager:
    def __init__(self, host='localhost', port=6379, db=0, max_retries=5):
        self.host = host
        self.port = port
        self.db = db
        self.max_retries = max_retries
        self.prefix_queue = "queue:"
        self.prefix_state = "state:"

        self.redis = self._connect_with_retries()

    def _connect_with_retries(self):
        attempt = 0
        while attempt < self.max_retries:
            try:
                logger.info(f"Попытка подключения к Redis ({attempt + 1}/{self.max_retries})")
                redis_client = Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    decode_responses=True,
                    socket_timeout=30
                )
                if redis_client.ping():
                    logger.info("Успешное подключение к Redis")
                    return redis_client
            except Exception as e:
                logger.warning(f"Ошибка подключения к Redis: {str(e)}")
                attempt += 1
                time.sleep(2 ** attempt)  # Экспоненциальная задержка

        raise ConnectionError("Не удалось подключиться к Redis после нескольких попыток")

    def create_queue(self, queue_name: str):
        self.redis.lpush(f"{self.prefix_queue}{queue_name}", "")

    def put(self, queue_name: str, item: Any):
        try:
            # Проверяем, является ли item UUID
            if isinstance(item, uuid.UUID):
                # Преобразуем UUID в строку перед сериализацией
                item = str(item)

            self.redis.rpush(f"{self.prefix_queue}{queue_name}", json.dumps(item))
        except Exception as e:
            logger.error(f"Ошибка при добавлении в очередь: {str(e)}")

    def get(self, queue_name: str) -> Any:
        try:
            item = self.redis.blpop(f"{self.prefix_queue}{queue_name}", timeout=0)
            if item:
                # Преобразуем строку обратно в UUID при получении
                data = json.loads(item[1])
                if isinstance(data, str) and len(data) == 36 and '-' in data:
                    return uuid.UUID(data)
                return data
            return None
        except Exception as e:
            logger.error(f"Ошибка при получении из очереди: {str(e)}")
            return None

    def save_state(self, request_id: UUID, state: ProcessingState):
        try:
            # Используем метод to_json из ProcessingState
            self.redis.set(
                f"{self.prefix_state}{request_id}",
                state.to_json()
            )
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {str(e)}")

    def get_state(self, request_id: UUID) -> Optional[ProcessingState]:
        try:
            state_json = self.redis.get(f"{self.prefix_state}{request_id}")
            if state_json:
                # Используем метод from_json для десериализации
                return ProcessingState.from_json(state_json)
            return None
        except Exception as e:
            logger.error(f"Ошибка получения состояния: {str(e)}")
            return None

    async def update_state_and_get_next_stage(
        self,
        router: PipelineRouter,
        request_id: UUID,
        handler_response: dict,
        current_stage: str,
    ) -> Optional[str]:

        try:
            # Получаем текущее состояние
            current_state = self.get_state(request_id)

            if not current_state:
                raise ValueError("Состояние не найдено")

            # Обновляем данные состояния
            updated_state = {
                **current_state,
                **handler_response
            }

            # Сохраняем обновленное состояние
            self.save_state(request_id, updated_state)

            # Определяем следующий этап
            next_stage = router.get_next_stage(current_stage, current_state.params)
            return next_stage

        except Exception as e:
            logger.error(f"Ошибка обновления состояния: {str(e)}")
            return None

    def delete_state(self, request_id: uuid.UUID):
        try:
            self.redis.delete(f"{self.prefix_state}{request_id}")
        except Exception as e:
            logger.error(f"Ошибка удаления состояния: {str(e)}")