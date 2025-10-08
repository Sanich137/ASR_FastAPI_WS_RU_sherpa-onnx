import asyncio
from fastapi import HTTPException
from utils.pre_start_init import app
# from asr_pipeline.worker import queues

from utils.do_logging import logger
from models.fast_api_models import PostFileRequest, PostFileResponse
from models.pipeline_model import StageResults
from fastapi import Depends, File, Form, UploadFile


# Функция для извлечения параметров из FormData
def get_file_request(
    keep_raw: bool = Form(default=True, description="Сохранять сырые данные."),
    do_echo_clearing: bool = Form(default=False, description="Убирать межканальное эхо."),
    do_dialogue: bool = Form(default=False, description="Строить диалог."),
    do_punctuation: bool = Form(default=False, description="Восстанавливать пунктуацию."),
    do_diarization: bool = Form(default=False, description="Разделять по спикерам."),
    diar_vad_sensity: int = Form(default=3, description="Чувствительность VAD."),
    do_auto_speech_speed_correction: bool = Form(default=True, description="Корректировать скорость речи при распознавании."),
    speech_speed_correction_multiplier: float = Form(default=1, description="Базовый коэффициент скорости речи."),
    make_mono: bool = Form(default=False, description="Соединить несколько каналов в mono"),
) -> PostFileRequest:
    return PostFileRequest(
        keep_raw=keep_raw,
        do_echo_clearing=do_echo_clearing,
        do_dialogue=do_dialogue,
        do_punctuation=do_punctuation,
        do_diarization=do_diarization,
        diar_vad_sensity=diar_vad_sensity,
        make_mono=make_mono,
        do_auto_speech_speed_correction = do_auto_speech_speed_correction,
        speech_speed_correction_multiplier = speech_speed_correction_multiplier
    )


@app.post("/post_file")  # аменить /post_file_pipline
async def async_receive_file(
    file: UploadFile = File(..., description="Аудиофайл для обработки"),
    params: PostFileRequest = Depends(get_file_request),
    ):
    """
    Принимает аудиофайл и запускает ASR-пайплайн.
    """
    logger.info(f"Получен запрос /post_file")
    # Заранее готовим ответ структуру ответа на запрос
    result = PostFileResponse(
        success = False,
        error_description = str(),
        raw_data = dict(),
        sentenced_data = dict(),
        diarized_data = dict()
    )

    init_data = StageResults(
        file_content=await file.read(),
    )

    # Создаём состояние и получаем request_id
    try:
        # Получаем идентификатор запроса. В формат ProcessingState преобразуется при создании состояния.
        request_id = await app.state.manager.create_state(result=result,
                                                          init_data=init_data,
                                                          params=params)
        # Сохраняем параметры вызова

        logger.debug(f"Создан запрос {request_id} для файла {file.filename}")
    except Exception as e:
        error_description = str(e)
        logger.error(f"Не удалось создать состояние для запроса: {error_description}")
        result.error_description = error_description
        return result

    try:
        await queues["receive"].put(request_id)

        logger.debug(f"Задача для {request_id} отправлена в очередь 'receive'")
    except Exception as e:
        error_description = str(e)
        logger.error(f"Не удалось отправить задачу в очередь: {error_description}")

    else:
        result.success = True
        # Если тут отдать пустой return, то цикл очередей работает более правильно.
        return app.state.manager.states[request_id].results

    time_limit = 6000
    try:
        # Асинхронное ожидание с таймаутом
        async with asyncio.timeout(time_limit): # 100 минут
            while app.state.manager.states[request_id].current_stage is not None:
                await asyncio.sleep(0.2)  # Более эффективное ожидание

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Превышено время ожидания обработки в {time_limit} секунд."
        )
    except Exception as e:
        result.error_description = str(e)
        return result
    finally:
        # После завершения всех этапов
        final_result = app.state.manager.states[request_id].results
        final_result.success = True
        return final_result
