from pydantic import BaseModel
import base64
from typing import Optional, Dict, Any
from io import BytesIO
from uuid import UUID, uuid4
from pydub import AudioSegment
from models.fast_api_models import PostFileRequest, PostFileResponse

class StageResults(BaseModel):
    bytes_buffer: Optional[str] = None
    file_content: Optional[str] = None
    posted_and_downloaded_audio: Optional[str] = None
    audio_to_asr: Optional[str] = None
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
    current_stage: Optional[str] = None
    stage_results: StageResults
    next_stage: Optional[str] = None
    results: PostFileResponse
    processing_time: float
    created_at: float

    # Добавляем метод для преобразования в словарь
    def to_dict(self) -> dict:
        return self.dict()

    # Добавляем метод для создания объекта из словаря
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    # Добавляем метод для сериализации в JSON
    def to_json(self) -> str:
        return self.json()

    # Добавляем метод для десериализации из JSON
    @classmethod
    def from_json(cls, json_str: str):
        return cls.parse_raw(json_str)
