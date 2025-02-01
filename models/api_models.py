from pydantic import BaseModel, HttpUrl
from typing import Union, Annotated
from fastapi import Query
import re

class TxtRequest(BaseModel):
    auth: str  # token
    question: str
    variants: Union[int, None] = 3


class AudioRequest(BaseModel):
    auth: str  # token
    AudioFileUrl: HttpUrl
    variants: Union[int, None] = 3


class AsyncAudioRequestNewTask(BaseModel):
    """
    Всегда получаем ссылку на аудио.\n
    По умолчанию сервис отдаёт только распознавание текста. Два списка, без пунктуации.\n
    do_classification: - поиск правовой категории обращения.\n
    classification_variants: - Количество предлагаемых вариантов классификации. По умолчанию 3.\n
    do_dialogue: - true, если нужно разбить речь на диалог - Строка, построчно(\\n), юрист/клиент.\n
    do_punctuation: - true, если нужно расставить пунктуацию. Применяется к диалогу.\n
    do_lawyer_text_description: - true, если нужно получить краткое описание коммуникации.\n
    do_analysis_of_sales_communication: - true, если нужен анализ соблюдения продавцом критериев продаж.\n
    do_quality_analysis_of_lawyer_communication: - true, если нужен анализ консультации на предмет соблюдения норм поведения и пр. в т.ч. клиента.\n
    """
    auth: str  # token
    AudioFileUrl: HttpUrl
    erp_file_id: Union[int, None] = 0  # В разработке
    do_punctuation: Union[bool, None] = False
    do_classification: Union[bool, None] = False
    classification_variants: Union[int, None] = 3
    do_dialogue: Union[bool, None] = False
    do_lawyer_text_description: Union[bool, None] = False
    do_analysis_of_sales_communication: Union[bool, None] = False
    do_quality_analysis_of_lawyer_communication: Union[bool, None] = False


class AsyncAudioRequestGetResult(BaseModel):
    """Если reuse = True, то результат запроса можно получить ещё раз. Если False, то запрос придётся делать вновь """
    auth: str  # token
    task_id: str
    reuse: Union[bool, None] = True  # оставлять ли сведения о запросе на будущее.


class RawRecTask(BaseModel):
    """
    Всегда получаем сырой результат распознавания \n
    По умолчанию сервис отдаёт только собранный из слов текст. Два списка, без пунктуации.\n
    do_classification: - поиск правовой категории обращения.\n
    classification_variants: - Количество предлагаемых вариантов классификации. По умолчанию 3.\n
    do_dialogue: - true, если нужно разбить речь на диалог - Строка, построчно(\\n), юрист/клиент.\n
    do_punctuation: - true, если нужно расставить пунктуацию. Применяется к диалогу, общему тексту.\n
    do_lawyer_text_description: - true, если нужно получить краткое описание коммуникации.\n
    do_analysis_of_sales_communication: - true, если нужен анализ соблюдения продавцом критериев продаж.\n
    do_quality_analysis_of_lawyer_communication: - true, если нужен анализ консультации на предмет соблюдения
    норм поведения и пр. в т.ч. клиента.\n
    """
    auth: str  # token
    RawRecStr: str
    do_punctuation: Union[bool, None] = False
    do_classification: Union[bool, None] = False
    classification_variants: Union[int, None] = 3
    do_dialogue: Union[bool, None] = False
    do_lawyer_text_description: Union[bool, None] = False
    do_analysis_of_sales_communication: Union[bool, None] = False
    do_quality_analysis_of_lawyer_communication: Union[bool, None] = False
    custom_prompt: Union[str, None] = None
