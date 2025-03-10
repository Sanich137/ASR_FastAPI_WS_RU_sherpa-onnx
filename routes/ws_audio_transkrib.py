from pydub import AudioSegment

import ujson
import config
import uuid
from io import BytesIO
import subprocess


from utils.pre_start_init import app
from fastapi import WebSocket, WebSocketException
from utils.do_logging import logger
from utils.chunk_doing import find_last_speech_position
from utils.pre_start_init import audio_buffer, audio_overlap, audio_to_asr, audio_duration,ws_collected_asr_res
from utils.send_messages import send_messages
from utils.tokens_to_Result import process_asr_json, process_gigaam_asr

from Recognizer.engine.sentensizer import do_sensitizing
from Recognizer.engine.stream_recognition import recognise_w_calculate_confidence, simple_recognise


@app.websocket("/ws")
async def websocket(ws: WebSocket):
    wait_null_answers=True
    client_id = uuid.uuid4()
    logger.info(f'Принят новый сокет id = {client_id}')
    audio_buffer[client_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
    audio_overlap[client_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
    audio_duration[client_id] = 0
    ws_collected_asr_res[client_id] = {f"channel_{1}": list()}
    do_dialogue = False
    do_punctuation = False
    audio_format = 'raw'
    sample_rate = config.BASE_SAMPLE_RATE  # Если не получен фреймрейт в конфиге сокета, по попытается принять с конфигом модели.
    sentenced_data = None
    error_description = None

    await ws.accept()

    while True:
        try:
            message = await ws.receive()
        except Exception as wse:
            logger.error(f"receive WebSocketException - {wse}")
            return

        if isinstance(message, dict) and message.get('text'):
            try:
                if message.get('text') and 'config' in message.get('text'):
                    json_cfg = ujson.loads(message.get('text'))['config']
                    audio_format = json_cfg.get("audio_format", 'raw')

                    sample_rate = json_cfg.get('sample_rate')
                    wait_null_answers = json_cfg.get('wait_null_answers', wait_null_answers)

                    do_dialogue = json_cfg.get("do_dialogue", False)
                    do_punctuation = json_cfg.get("do_punctuation", False)

                    logger.info(f"\n Task received, config -  {message.get('text')}")
                    continue

                elif message.get('text') and 'eof' in message.get('text'):
                    logger.debug("EOF received\n")
                    break
                else:
                    logger.error(f"Can`t recognise  text part of  message {message.get('text')}")

            except Exception as e:
                logger.error(f'Error text message compiling. Message:{message} - error:{e}')
        elif isinstance(message, dict) and message.get('bytes'):
            try:
                # Получаем новый чанк с данными
                chunk = message.get('bytes')

                if audio_format == 'raw':
                    # Переводим чанк в объект Audiosegment
                    audiosegment_chunk = AudioSegment(
                        chunk,
                        frame_rate = sample_rate,  # Укажи частоту дискретизации
                        sample_width = 2,   # Ширина сэмпла (2 байта для int16)
                        channels = 1        # Количество каналов. По умолчанию - 1, Моно.
                        )
                else:
                    try:
                        buffer = BytesIO()
                        # Запускаем FFmpeg для конвертации
                        subprocess.run([
                            "ffmpeg", "-i", "input.webm", "-f", "wav", buffer
                        ])
                        audiosegment_chunk = AudioSegment.from_file(buffer)

                    except Exception as e:
                        logger.error(f"Ошибка принятия аудио - {e}")
                    else:
                        logger.info("Чанк принят и распознан")
                        audiosegment_chunk.export('chunk.wav', "wav")

                # Приводим фреймрейт к фреймрейту модели
                if audiosegment_chunk.frame_rate != config.BASE_SAMPLE_RATE:
                    audiosegment_chunk = audiosegment_chunk.set_frame_rate(config.BASE_SAMPLE_RATE)
                if audiosegment_chunk.channels != 1:
                    audiosegment_chunk = audiosegment_chunk.set_channels(1)

                # Копим буфер
                audio_buffer[client_id] += audiosegment_chunk

                # Накопили больше нормы
                if (audio_overlap[client_id]+audio_buffer[client_id]).duration_seconds >= config.MAX_OVERLAP_DURATION:

                    # Проверяем новый чанк перед объединением (там же режем хвост и добавляем его при необходимости)
                    find_last_speech_position(client_id)

                else:
                    continue
            except Exception as e:
                logger.error(f"AcceptWaveform error - {e}")
            else:
                try:
                    if config.MODEL_NAME == "Gigaam":
                        asr_first_result_wo_conf =await simple_recognise(audio_to_asr[client_id])
                        asr_result_words = await process_gigaam_asr(asr_first_result_wo_conf, audio_duration[client_id])

                        audio_duration[client_id] += audio_to_asr[client_id].duration_seconds
                        logger.debug(asr_result_words)

                        # Копим ответы для пунктуации
                        ws_collected_asr_res[client_id][f"channel_{1}"].append(asr_result_words)
                    else:
                        asr_result_w_conf = recognise_w_calculate_confidence(audio_to_asr[client_id],
                                                                             num_trials=config.RECOGNITION_ATTEMPTS)

                        asr_result_words = await process_asr_json(asr_result_w_conf, audio_duration[client_id])
                        audio_duration[client_id] += audio_to_asr[client_id].duration_seconds
                        logger.debug(asr_result_words)

                        # Копим ответы для пунктуации
                        ws_collected_asr_res[client_id][f"channel_{1}"].append(asr_result_words)

                except Exception as e:
                    logger.error(f"recognizer.get_result(stream()) error - {e}")
                else:
                    if len(asr_result_words.get("data").get("text")) == 0 or asr_result_words.get("data").get("text") == ' ':
                        if wait_null_answers:
                            if not await send_messages(ws, _silence = True, _data = None, _error = None):
                                logger.error(f"send_message not ok work canceled")
                                return
                            # await asyncio.sleep(0.01)
                        else:
                            logger.debug("sending silence partials skipped")
                            continue
                    else:
                        if not await send_messages(ws, _silence=False, _data=asr_result_words, _error=None):
                            logger.error(f"send_message not ok work canceled")
                            return
        else:
            error_description = f"Can`t parse message - {message}"
            logger.error(error_description)

            if not await send_messages(ws, _silence=False, _data=None, _error=error_description):
                logger.error(f"send_message not ok work canceled")
                return

    # Передаём на распознавание собранный не полный буфер
    # перевод в семплы для распознавания.
    audio_to_asr[client_id] = audio_overlap[client_id] + audio_buffer[client_id]
    logger.debug(f'итоговое сообщение - {audio_to_asr[client_id].duration_seconds} секунд')

    try:
        try:
            if audio_to_asr[client_id].duration_seconds < 2:
                audio_to_asr[client_id] = audio_to_asr[client_id] + AudioSegment.silent(1000, frame_rate=sample_rate)
        except Exception as e:
            logger.error(f"Error getting len of last chunk - {e}")
            last_result = None
            error_description = f"Error getting len of last chunk - {e}"
        else:
            if config.MODEL_NAME == "Gigaam":
                last_asr_result_w_conf = await simple_recognise(audio_to_asr[client_id])
                last_result = await process_gigaam_asr(last_asr_result_w_conf, audio_duration[client_id])
                logger.debug(f'Последний результат {last_result.get("data").get("text")}')

                ws_collected_asr_res[client_id][f"channel_{1}"].append(last_result)
                logger.debug(last_result)
            else:
                asr_result_w_conf = recognise_w_calculate_confidence(audio_to_asr[client_id],
                                                                     num_trials=config.RECOGNITION_ATTEMPTS)
                last_result = await process_asr_json(asr_result_w_conf, audio_duration[client_id])
                audio_duration[client_id] += audio_to_asr[client_id].duration_seconds
                ws_collected_asr_res[client_id][f"channel_{1}"].append(last_result)
                logger.debug(last_result)

    except Exception as e:
        logger.error(f"last_asr_result_w_conf error - {e}")

    else:
        if len(last_result.get("data").get("text")) == 0:
            is_silence = True
            last_result = None
        elif last_result.get("data").get("text") == ' ':
            is_silence = True
            last_result = None
        else:
            logger.debug(last_result)
            is_silence = False


        if do_dialogue:
            try:
                sentenced_data = await do_sensitizing(ws_collected_asr_res[client_id], do_punctuation)
            except Exception as e:
                logger.error(f"await do_sensitizing - {e}")
                error_description = f"do_sensitizing - {e}"

        if not await send_messages(ws, _silence=is_silence, _data=last_result, _error=error_description, _last_message=True,
                                   _sentenced_data=sentenced_data):
            logger.error(f"send_message not ok work canceled")
            return
    await ws.close()

    del audio_overlap[client_id]
    del audio_buffer[client_id]
    del audio_to_asr[client_id]
    del audio_duration[client_id]
    del ws_collected_asr_res[client_id]

    # try:
    #     print(f'/n \
    #     audio_overlap = {audio_overlap} \
    #     audio_buffer = {audio_buffer} \
    #     audio_to_asr = {audio_to_asr} \
    #     audio_duration = {audio_duration} \
    #     ws_collected_asr_res = {ws_collected_asr_res} \
    #         ')
    # except Exception as e:
    #     logger.error(f"Error printing globals - {e}")

