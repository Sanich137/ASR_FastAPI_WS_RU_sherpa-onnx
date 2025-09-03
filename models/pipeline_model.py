from pydantic import BaseModel
from uuid import UUID, uuid4
from models.fast_api_models import PostFileRequest, PostFileResponse

import time


# ProcessingState для post запроса.
# Для url будет, видимо, другой или надо всё переписать на PostFileRequest
class ProcessingState(BaseModel):
    request_id: UUID
    params: PostFileRequest
    current_stage: str | None
    stage_results: dict | list
    next_stage: str | None
    results: PostFileResponse
    processing_time: float
    created_at: float