from utils.do_logging import logger
from models.pipeline_model import ProcessingState
from Recognizer.engine.sentensizer import do_sensitizing


async def dialogue_handler(data: ProcessingState) -> ProcessingState:
    logger.info(f'Получено задание в dialogue_handler')  # post_asr_production
    params = data.params
    data.results.success = False

    # Todo сравнить выводы после диаризации. Можно ли хранить ответы после диаризации в raw_data?

    diarized = True if len(data.results.diarized_data)>0 else False

    if params.do_dialogue:
        data_to_do_sensitizing = data.results.diarized_data if diarized else data.results.raw_data
        try:
            data.results.sentenced_data =  await do_sensitizing(
                                                            input_asr_json=data_to_do_sensitizing,
                                                            do_punctuation=params.do_punctuation
                                                                )
        except Exception as e:
            error_description = f"do_sensitizing - {e}"
            logger.error(error_description)
            data.results.error_description = error_description
        else:
            data.results.success = True
            if not params.keep_raw:
                data.results.raw_data.clear()
    else:
        data.results.sentenced_data.clear()

    logger.info(f'Возвращено задание из dialogue_handler')

    return data