from models.pipeline_model import ProcessingState, StageResults
from asr_pipeline.routing import PipelineRouter
from models.fast_api_models import PostFileRequest
from uuid import UUID, uuid4
import asyncio
from db import RedisManager
import time


# class StateManager:
#     def __init__(self, router: Optional[PipelineRouter] = None, shared_states=None):
#         self.router = router or PipelineRouter()
#         self.shared_states = shared_states
#         self.worker_tasks: List = []
#
#     async def create_state(self, result, init_data: StageResults, params: PostFileRequest) -> UUID:
#         request_id = uuid4()
#         state = ProcessingState(
#             request_id=request_id,
#             params=params,
#             current_stage="receive",
#             results=result,
#             next_stage="convert",
#             stage_results=init_data,
#             created_at=time.time(),
#             processing_time=time.time()
#         )
#
#         # Сохраняем состояние в shared_states
#         self.shared_states[request_id] = state
#
#         logger.debug(f"Created state for request {request_id}")
#         return request_id
#
#     async def update_state(self, handler_response: ProcessingState, stage: str) -> str:
#         try:
#             # Получаем текущее состояние из shared_states
#             current_state = self.shared_states.get(handler_response.request_id)
#             if not current_state:
#                 raise ValueError(f"Invalid request ID: {handler_response.request_id}")
#
#             # Обновляем состояние
#             self.shared_states[handler_response.request_id] = handler_response
#
#             # Определяем следующий этап
#             if handler_response.results.success:
#                 next_stage = self.router.get_next_stage(stage, params=handler_response.params)
#             else:
#                 next_stage = "response"
#
#             # Обновляем информацию о следующем этапе
#             handler_response.next_stage = next_stage
#             self.shared_states[handler_response.request_id] = handler_response
#
#             # Если это финальный этап
#             if not next_stage:
#                 handler_response.current_stage = None
#                 handler_response.processing_time = time.time() - handler_response.created_at
#                 logger.info(
#                     f"Request {handler_response.request_id} completed. "
#                     f"Total processing time: {handler_response.processing_time:.2f} s"
#                 )
#
#             logger.debug(f"Updated state for {handler_response.request_id}: {stage} → {next_stage}")
#             return next_stage
#
#         except Exception as e:
#             logger.error(f"Error updating state: {str(e)}")
#             raise
#
#     async def get_state(self, request_id) -> Optional[ProcessingState]:
#         try:
#             return self.shared_states.get(request_id)
#         except Exception as e:
#             logger.error(f"Error getting state: {str(e)}")
#             return None
#
#     async def delete_state(self, request_id):
#         try:
#             if request_id in self.shared_states:
#                 del self.shared_states[request_id]
#         except Exception as e:
#             logger.error(f"Error deleting state: {str(e)}")

class StateManager:
    def __init__(self, router: PipelineRouter, redis_manager: RedisManager):
        self.router = router
        self.redis_manager = redis_manager

    async def create_state(self, result, init_data: StageResults, params: PostFileRequest) -> UUID:
        request_id = uuid4()
        state = ProcessingState(
            request_id=request_id,
            params=params,
            current_stage="receive",
            results=result,
            next_stage="convert",
            stage_results=init_data,
            created_at=time.time(),
            processing_time=time.time()
        )

        # Сохраняем состояние в Redis
        self.redis_manager.save_state(request_id, state)
        return request_id

    async def get_state(self, request_id: UUID) -> ProcessingState|None:
        state_dict = self.redis_manager.get_state(request_id)
        if state_dict:
            return ProcessingState(**state_dict)
        return None

    async def update_state(self, handler_response: ProcessingState, stage: str) -> str:
        # Сохраняем обновленное состояние
        self.redis_manager.save_state(handler_response.request_id, handler_response)

        # Определяем следующий этап
        if handler_response.results.success:
            next_stage = self.router.get_next_stage(stage, params=handler_response.params)
        else:
            next_stage = "response"

        return next_stage