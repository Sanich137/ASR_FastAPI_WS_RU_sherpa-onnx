from pydantic import BaseModel
from uuid import UUID, uuid4
from models.fast_api_models import PostFileRequest, PostFileResponse

import time


# ProcessingState для post запроса.
# Для url будет, видимо, другой или надо всё переписать на PostFileRequest
class ProcessingState(BaseModel):
    request_id: UUID
    params: PostFileRequest
    current_stage: str
    stage_results: dict
    next_stage: str
    results: PostFileResponse
    created_at: float = time.time()