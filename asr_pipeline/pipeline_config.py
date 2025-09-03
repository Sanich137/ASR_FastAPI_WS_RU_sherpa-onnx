from typing import Callable, List
import random
import asyncio
from asr_handlers.receive_handler import receive_handler
from asr_handlers.convert_handler import convert_handler


class PipelineStage:
    def __init__(
            self,
            name: str,
            handler: Callable,
            next_stage: List[str] = None,
            num_workers: int = 1
            ):
        self.name = name
        self.handler = handler
        self.next_stage = next_stage or List
        # self.definition = str
        self.num_workers = num_workers


# Заглушки функций (оставлены без изменений)

async def split_audio(data):
    print(f"{"\n"}Starting split_audio: {data['raw']}")
    await asyncio.sleep(random.uniform(0.1, 1))
    print(f"{"\n"}Finished split_audio: {data['raw']}")
    return {"parts": []}


async def asr_recognize_handler(data):
    print(f"{"\n"}Starting asr_recognize_handler: {data['raw']}")
    await asyncio.sleep(random.uniform(2, 5))
    print(f"{"\n"}Finished asr_recognize_handler: {data['raw']}")
    return {"transcripts": []}

async def echo_clearing_handler(data):
    print(f"{"\n"}Starting echo_clearing_handler: {data['raw']}")
    await asyncio.sleep(random.uniform(2, 5))
    print(f"{"\n"}Finished echo_clearing_handler: {data['raw']}")
    return {"transcripts": []}


async def diarize_handler(data):
    print(f"{"\n"}Starting diarize: {data['raw']}")
    await asyncio.sleep(random.uniform(1, 1))
    print(f"{"\n"}Finished diarize: {data['raw']}")
    return {"speakers": [], "mapped": []}


async def dialogue_handler(data):
    print(f"{"\n"}Starting diarize: {data['raw']}")
    await asyncio.sleep(random.uniform(1, 1))
    print(f"{"\n"}Finished diarize: {data['raw']}")
    return {"speakers": [], "mapped": []}


async def punctuate_handler(data):
    print(f"{"\n"}Starting punctuate: {data['raw']}")
    await asyncio.sleep(random.uniform(0.1, 1))
    print(f"{"\n"}Finished punctuate: {data['raw']}")
    print(f"{"\n"}Finished punctuate: {data['raw']}")
    return {"punctuated": ""}


async def return_response(data):
    print(f"{"\n"}Starting return_response: {data['raw']}")
    await asyncio.sleep(random.uniform(0.1, 1))
    print(f"{"\n"}Finished return_response: {data['raw']}")
    return data




# Все этапы в одном месте
# Но переопределяются в get_next_stages.
# Todo - сделать формирование списка next_stages до начала обработки и хранить для каждого запроса ?

PIPELINE_CONFIG = [
    PipelineStage(
        name="receive",
        handler=receive_handler,
        next_stage=["convert"],
        num_workers=1
        ),
    PipelineStage(
        name="convert",
        handler=convert_handler,
        next_stage=["split"],
        num_workers=1
        ),
    PipelineStage(
        name="split",
        handler=split_audio,
        next_stage=["asr"],
        num_workers=1
        ),
    PipelineStage(
        name="asr",
        handler=asr_recognize_handler,
        next_stage=["echo_clearing"],
        num_workers=1
        ),
    PipelineStage(
        name="echo_clearing",
        handler=echo_clearing_handler,
        next_stage=["diarize"],
        num_workers=1
        ),
    PipelineStage(
        name="diarize",
        handler=diarize_handler,
        next_stage=["dialogue"],
        num_workers=1
        ),
    PipelineStage(
        name="dialogue",
        handler=dialogue_handler,
        next_stage=["punctuate"],
        num_workers=1
        ),
    PipelineStage(
        name="punctuate",
        handler=punctuate_handler,
        next_stage=["response"],
        num_workers=1
        ),
    PipelineStage(
        name="response",
        handler=return_response,
        num_workers=1
        )
    ]

