from utils.do_logging import logger
from utils.globals import audio_duration, audio_to_asr, audio_overlap, audio_buffer
from models.pipeline_model import ProcessingState
from utils.tokens_to_Result import process_asr_json, process_gigaam_asr
from Recognizer.engine.stream_recognition import simple_recognise, recognise_w_speed_correction, simple_recognise_batch
from config import MODEL_NAME


async def asr_recognize_handler(data: ProcessingState) -> ProcessingState:
    request_id = data.request_id
    params = data.params
    logger.info(f'Получено задание в asr_recognize_handler')

    for n_channel, audio_asr_list in enumerate(audio_to_asr[request_id]):
        audio_duration[request_id] = float()
        for audio_asr in audio_asr_list:

            try:
                # Снижаем скорость аудио по необходимости
                if params.do_auto_speech_speed_correction or params.speech_speed_correction_multiplier != 1:
                    logger.debug("Будут использованы механизмы анализа скорости речи и замедления аудио")

                    asr_result_wo_conf, speed, multiplier = await recognise_w_speed_correction(audio_asr,
                                                                                             can_slow_down=True,
                                                                                             multiplier=params.speech_speed_correction_multiplier)
                    params.speech_speed_correction_multiplier = multiplier
                else:
                    # Производим распознавание
                    asr_result_wo_conf = await simple_recognise(audio_asr)

            except Exception as e:
                error_description = f"Error ASR audio - {e}"
                logger.error(error_description)
                data.results.error_description = error_description
            else:
                # Todo - перенести сбор в отдельный воркер.
                if MODEL_NAME == "Gigaam" or MODEL_NAME == "Gigaam_rnnt":  # Whisper
                    asr_result = await process_gigaam_asr(asr_result_wo_conf,
                                                                audio_duration[request_id],
                                                                params.speech_speed_correction_multiplier)
                else:
                    asr_result =  await process_asr_json(asr_result_wo_conf, audio_duration[request_id])

                data.results.raw_data[f"channel_{n_channel + 1}"].append(asr_result)

                audio_duration[request_id] += audio_asr.duration_seconds
                data.results.success = True
                logger.debug(asr_result)

    try:
        del audio_overlap[request_id]
        del audio_buffer[request_id]
        del audio_to_asr[request_id]
        del audio_duration[request_id]
    except Exception as e:
        error_description = f"Ошибка при очистке данных - {e}"
        logger.error(error_description)
        data.results.error_description = error_description
    else:
        data.results.success = True

    logger.info(f'Возвращено задание из asr_recognize_handler')
    return data

