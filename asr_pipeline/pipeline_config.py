from typing import Callable, Union
import random
import asyncio
from asr_handlers.receive_handler import receive_handler
from asr_handlers.convert_handler import convert_handler
from asr_handlers.split_audio_handler import split_audio_handler
from asr_handlers.asr_recognize_handler import asr_recognize_handler
from asr_handlers.echo_clearing_handler import echo_clearing_handler
from asr_handlers.diarize_handler import diarize_handler
from asr_handlers.dialogue_handler import dialogue_handler
from asr_handlers.return_response_handler import return_response_handler


class PipelineStage:
    def __init__(
            self,
            name: str,
            handler: Callable,
            next_stage: list[str|None] = None,
            num_workers: int = 1,
            extra_args = None
            ):
        self.name = name
        self.handler = handler
        self.next_stage = next_stage
        # self.definition = str # todo попробовать тут хранить условия для получения задачи в очередь в связи с params.
        self.num_workers = num_workers
        self.extra_args = extra_args or {}  # Словарь с дополнительными аргументами



# Заглушки функций (оставлены без изменений)
# Пунктуация совмещена с посторением диалога. Посотмрим, может быть разделить
# async def punctuate_handler(data):
#     print(f"{"\n"}Starting punctuate: {data['raw']}")
#     await asyncio.sleep(random.uniform(0.1, 1))
#     print(f"{"\n"}Finished punctuate: {data['raw']}")
#     print(f"{"\n"}Finished punctuate: {data['raw']}")
#     return {"punctuated": ""}


# Все этапы в одном месте
# Но переопределяются в get_next_stages.
# Todo - сделать формирование списка next_stages до начала обработки и хранить для каждого запроса ?

PIPELINE_CONFIG = [
    PipelineStage(
        name="receive",
        handler=receive_handler,
        next_stage=["convert"],
        num_workers=5
        ),
    PipelineStage(
        name="convert",
        handler=convert_handler,
        next_stage=["split"],
        num_workers=5
        ),
    PipelineStage(
        name="split",
        handler=split_audio_handler,
        next_stage=["asr"],
        num_workers=5
        ),
    PipelineStage(
        name="asr",
        handler=asr_recognize_handler,
        next_stage=["echo_clearing"],
        num_workers=2,
        extra_args={"recognizer": None}
        ),
    PipelineStage(
        name="echo_clearing",
        handler=echo_clearing_handler,
        next_stage=["diarize"],
        num_workers=5
        ),
    PipelineStage(
        name="diarize",
        handler=diarize_handler,
        next_stage=["dialogue"],
        num_workers=5
        ),
    PipelineStage(
        name="dialogue",
        handler=dialogue_handler,
        next_stage=["response"],
        num_workers=5
        ),
    # Пунктуация совмещена с построением диалога. Посмотрим, может быть разделить
    # PipelineStage(
    #     name="punctuate",
    #     handler=punctuate_handler,
    #     next_stage=["response"],
    #     num_workers=1
    #     ),
    PipelineStage(
        name="response",
        handler=return_response_handler,
        next_stage=[None],
        num_workers=1,
        )
    ]

