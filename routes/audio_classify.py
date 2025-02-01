from utils.do_logging import logging
# from Classifier.engine.classification import define_category
from utils.pre_start_init import auth_token, app
from models.api_models import AudioRequest
from Recognizer.engine.get_audio_file import getting_audiofile
# from Recognizer.engine.audio_recognition import offline_recognition
from Recognizer.engine.get_audio_file import del_audio_file

@app.post("/audio_classify")
async def audio_classify(audio_to_classify: AudioRequest):
    """
    !!! Разработка не поддерживается, уточните актуальный инструмент для распознавания аудио.!!!
    На вход ждём ссылку на двухканальный mp3 файл.
    В ответ отдаю список вариантов классификации.
    Максимальное количество вариантов указываем в параметре variants.
    По умолчанию будет три варианта классификации.
    Функция распознавания - синхронная, на период распознавания прерывает приём других запросов.
    """

    logging.debug('enter_audio_classify')
    auth_state = False
    have_file = False
    error_description = None
    error = True
    data = None

    if audio_to_classify.auth != auth_token:
        error_description = "wrong auth data"
    else:
        auth_state = True

    if auth_state:
        if await getting_audiofile(audio_to_classify.AudioFileUrl):
            have_file = True

    if have_file:
        try:
            question = offline_recognition(file_name=audio_to_classify.AudioFileUrl.path.split('/')[-1])
            #

            logging.debug(f'Вопрос на классификацию {question.get("recognised_text")[-1]}')
        except Exception as e:
            error_description = e
        else:
            error = False
            cat_ids, cat_names = define_category(question.get("recognised_text")[-1], audio_to_classify.variants)
            data = {
                'cat_ids': cat_ids,
                'cat_names': cat_names,
                "duration": question.get("duration"),
                "raw_recognition": question.get("raw_recognition"),
                "sentenced_recognition": question.get("sentenced_recognition"),
                "recognised_text": question.get("recognised_text")
                }
    # Удалить файл
    await del_audio_file(file_url=audio_to_classify.AudioFileUrl)

    response = {"error": error,
                "error_description": error_description,
                "data": data
                }

    logging.debug(f'response data - {data}')

    return response
