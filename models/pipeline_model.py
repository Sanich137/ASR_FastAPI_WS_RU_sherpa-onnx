from pydantic import BaseModel
from typing import Optional, Dict, Any
from io import BytesIO
from uuid import UUID, uuid4
from pydub import AudioSegment
from models.fast_api_models import PostFileRequest, PostFileResponse

class StageResults(BaseModel):
    bytes_buffer: Optional[BytesIO] = None
    file_content: Optional[bytes] = None
    posted_and_downloaded_audio: Optional[AudioSegment] = None
    audio_to_asr: Optional[list[list]] = None
    # По мере разработки тут будут появляться другие
    extra_data: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True  # Для поддержки bytes
        extra = 'allow'  # Разрешить добавление произвольных полей

# ProcessingState для post запроса.
# Для url будет, видимо, другой или надо всё переписать на PostFileRequest
class ProcessingState(BaseModel):
    request_id: UUID
    params: PostFileRequest
    current_stage: str | None
    stage_results: StageResults
    next_stage: str | None
    results: PostFileResponse
    processing_time: float
    created_at: float
