from pydantic import BaseModel, HttpUrl
from typing import Union, Annotated

class SyncASRRequest(BaseModel):
    """
    :parameter keep_raw: - Если False, то запрос вернёт только пост-обработанные данные do_punctuation и do_dialogue.
    :parameter do_echo_clearing - Проверяет наличие повторений во втором канале слов из первого с запозданием.
    Если нужно проверить наоборот - поменяй каналы местами
    """

    AudioFileUrl: HttpUrl
    do_echo_clearing: Union[bool, None] = True
    do_dialogue: Union[bool, None] = False
    do_punctuation: Union[bool, None] = False
    keep_raw: Union[bool, None] = True


class WebSocketModel(BaseModel):
    """OpenAPI не хочет описывать WS, а я не хочу изучать OPEN API. По этому описание тут.
    \n
    \n Подключение на порт: 49153
    \n На вход жду поток binary, buffer_size +- 6400, mono, wav.
    \n На вход я должен получить словарь {'text': '{ "config" : { "sample_rate" : any(int/float), "wait_null_answers": Bool}}'}
    \n (значение по ключу text - строка. мне удобнее получать json, но на тестовом стенде не завелось именно отправление json)
    \n Далее сообщения с данными {"bytes": binary} - словарь
    \n По окончании передачи {'text': '{ "eof" : 1}'}
    \n Ответ получать в формате:     {"silence": Bool,"data": str, "error": None/str}
    \n Пример ответа data: {
  "result" : [{
      "conf" : 1.000000,
      "end" : 3.120000,
      "start" : 2.340000,
      "word" : "здравствуйте"
    }, {
      "conf" : 1.000000,
      "end" : 3.870000,
      "start" : 3.600000,
      "word" : "вы"
    },
     ...
     {
      "conf" : 0.994019,
      "end" : 11.790000,
      "start" : 10.890000,
      "word" : "записываются"
    }],
  "text" : "здравствуйте вы ... записываются"
}
\n в дальнейшем в data появится ключ quality(float) c указанием на качество распознанного предложения.
    """
    pass