from typing import Dict, List, Optional
from uuid import UUID, uuid4
from models.pipeline_model import ProcessingState
from models.fast_api_models import PostFileRequest, SyncASRRequest
from asr_pipeline.routing import PipelineRouter

from utils.do_logging import logger
import time


class StateManager:
    """
    Менеджер состояний запросов ASR-пайплайна.

    Централизованно управляет:
    - Состоянием каждого запроса
    - Маршрутизацией между этапами
    - Жизненным циклом запросов
    """

    def __init__(self, router: Optional[PipelineRouter] = None):
        """
        Инициализация менеджера состояний.

        :param router: Роутер для определения следующих этапов
        """
        self.states: Dict[UUID, ProcessingState] = {}
        self.router = router or PipelineRouter()
        self.worker_tasks: List = []  # Для graceful shutdown в будущем

        logger.info("StateManager initialized with PipelineRouter")

    async def create_state(self, result, init_data: dict, params: PostFileRequest) -> UUID:
        """
        Создаёт новое состояние для запроса.

        :param init_data: Входные данные для начала пайплайна.
        :param result: Результат, который будем отдавать в самом конце выполнения всех очередей.
        :param params: Параметры обработки от пользователя.
        :return: Уникальный идентификатор запроса.
        """
        request_id = uuid4()
        state = ProcessingState(
            request_id=request_id,
            params=params,
            current_stage="receive",
            results=result,
            next_stage="convert",
            stage_results=init_data
        )
        self.states[request_id] = state

        logger.info(f"Created state for request {request_id}")
        return request_id

    async def update_state(self, handler_response: ProcessingState, stage: str) -> str:
        """
        Обновляет состояние запроса после выполнения этапа.

        :param request_id: Идентификатор запроса
        :param handler_response: Результат выполнения этапа
        :param stage: Название завершённого этапа
        :raises ValueError: Если запрос не найден
        """
        state = self.states.get(handler_response.request_id)
        if not state:
            logger.error(f"Attempt to update non-existent state: {handler_response.request_id}")
            raise ValueError(f"Invalid request ID: {handler_response.request_id}")
        else:
            self.states[handler_response.request_id] = handler_response

        # Определяем следующие этапы через роутер. Если предыдущий результат завершился не удачно, возвращаем ответ.
        if state.results.success:
            state.next_stage = self.router.get_next_stage(stage, params=handler_response.params)
        else:
            state.next_stage = "response"

        # Если это финальный этап, фиксируем время обработки
        if not state.next_stage:
            state.results["processing_time"] = time.time() - state.created_at
            logger.info(
                f"Request {handler_response.request_id} completed. "
                f"Total processing time: {state.results['processing_time']:.2f}s"
            )

        logger.info(f"Updated state for {handler_response.request_id}: {stage} → {state.next_stage}")
        return state.next_stage

    async def get_state(self, request_id: UUID) -> Optional[ProcessingState]:
        """
        Получает текущее состояние запроса.

        :param request_id: Идентификатор запроса
        :return: Состояние запроса или None если не найдено
        """
        state = self.states.get(request_id)
        if state:
            logger.debug(f"Retrieved state for {request_id}")
        else:
            logger.debug(f"State not found for {request_id}")
        return state

    async def cleanup_expired_states(self, max_age: float = 86400) -> int:
        """
        Очищает устаревшие состояния (например, старше 24 часов).

        :param max_age: Максимальный возраст состояния в секундах
        :return: Количество удалённых состояний
        """
        current_time = time.time()
        expired_ids = [
            req_id for req_id, state in self.states.items()
            if current_time - state.created_at > max_age
        ]

        for req_id in expired_ids:
            del self.states[req_id]
            logger.info(f"Cleaned up expired state: {req_id}")

        return len(expired_ids)

    def get_active_count(self) -> int:
        """
        Возвращает количество активных запросов.

        :return: Число активных состояний
        """
        return len(self.states)

    async def remove_state(self, request_id: UUID) -> bool:
        """
        Принудительно удаляет состояние запроса.

        :param request_id: Идентификатор запроса
        :return: True если удаление успешно
        """
        if request_id in self.states:
            del self.states[request_id]
            logger.debug(f"Removed state for {request_id}")
            return True
        return False