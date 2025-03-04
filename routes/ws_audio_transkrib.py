import io

from fastapi import FastAPI
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
from utils.pre_start_init import audio_buffer, audio_overlap, audio_to_asr, audio_duration
from utils.send_messages import send_messages
from utils.tokens_to_Result import process_asr_json, process_gigaam_asr
from utils.opus_to_raw import do_opus_to_raw_convertion
from Recognizer.engine.stream_recognition import recognise_w_calculate_confidence, simple_recognise


def bytes_to_seconds(audio_bytes: bytes) -> float:
    return len(audio_bytes) / (8000 * 2)


@app.websocket("/ws")
async def websocket(ws: WebSocket):
    wait_null_answers=True
    client_id = uuid.uuid4()
    logger.info(f'Принят новый сокет id = {client_id}')
    audio_buffer[client_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
    audio_overlap[client_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
    audio_duration[client_id] = 0


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
                    audio_cfg = json_cfg.get("audio_format", 'raw')
                    sample_width = json_cfg.get("sample_width", 2)
                    channels = json_cfg.get("channels", 1)
                    wait_null_answers = json_cfg.get('wait_null_answers', wait_null_answers)
                    sample_rate=json_cfg.get('sample_rate')
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

                if audio_cfg == 'raw':
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

                        asr_result_wo_conf =await simple_recognise(audio_to_asr[client_id])

                        result = await process_gigaam_asr(asr_result_wo_conf, audio_duration[client_id])
                        audio_duration[client_id] += audio_to_asr[client_id].duration_seconds
                        logger.debug(result)

                    else:
                        asr_result_w_conf = recognise_w_calculate_confidence(audio_to_asr[client_id],
                                                                             num_trials=config.RECOGNITION_ATTEMPTS)

                        result = await process_asr_json(asr_result_w_conf, audio_duration[client_id])
                        audio_duration[client_id] += audio_to_asr[client_id].duration_seconds
                        logger.debug(result)


                except Exception as e:
                    logger.error(f"recognizer.get_result(stream()) error - {e}")
                else:
                    if len(result.get("data").get("text")) == 0 or result.get("data").get("text") == ' ':
                        if wait_null_answers:
                            if not await send_messages(ws, _silence = True, _data = None, _error = None):
                                logger.error(f"send_message not ok work canceled")
                                return
                            # await asyncio.sleep(0.01)
                        else:
                            logger.debug("sending silence partials skipped")
                            continue
                    else:
                        if not await send_messages(ws, _silence=False, _data=result, _error=None):
                            logger.error(f"send_message not ok work canceled")
                            return
        else:
            error = f"Can`t parse message - {message}"
            logger.error(error)

            if not await send_messages(ws, _silence=False, _data=None, _error=error):
                logger.error(f"send_message not ok work canceled")
                return

    # Передаём на распознавание собранный не полный буфер
    # перевод в семплы для распознавания.
    audio_to_asr[client_id] = audio_overlap[client_id] + audio_buffer[client_id]
    logger.debug(f'итоговое сообщение - {audio_to_asr[client_id].duration_seconds} секунд')

    try:

        if config.MODEL_NAME == "Gigaam":

            last_asr_result_w_conf = await simple_recognise(audio_to_asr[client_id])
            last_result = await process_gigaam_asr(last_asr_result_w_conf, audio_duration[client_id])
            logger.debug(f'Последний результат {last_result.get("data").get("text")}')
        else:
            asr_result_w_conf = recognise_w_calculate_confidence(audio_to_asr[client_id],
                                                                 num_trials=config.RECOGNITION_ATTEMPTS)

            last_result = await process_asr_json(asr_result_w_conf, audio_duration[client_id])
            audio_duration[client_id] += audio_to_asr[client_id].duration_seconds
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

        if not await send_messages(ws, _silence=is_silence, _data=last_result, _error=None, _last_message=True):
            logger.error(f"send_message not ok work canceled")
            return
    await ws.close()

    del audio_overlap[client_id]
    del audio_buffer[client_id]
    del audio_to_asr[client_id]
    del audio_duration[client_id]
