from dbm import error
from pydub import AudioSegment

import ujson
import asyncio

from utils.pre_start_init import app, WebSocket, WebSocketException
from utils.pre_start_init import recognizer
from utils.do_logging import logger
from utils.bytes_to_samples_audio import get_np_array_samples_float32
from utils.chunk_doing import find_last_speech_position
from utils.pre_start_init import audio_buffer, audio_overlap, audio_to_asr

def bytes_to_seconds(audio_bytes: bytes) -> float:
    return len(audio_bytes) / (8000 * 2)


async def send_messages(_socket, _data=None, _silence=True, _error=None, log_comment=None, _last_message=False):
    ws = _socket
    is_ok = False
    if not _data:
        data = ""
    else:
        data = _data.text

    snd_mssg = {"silence": _silence,
                    "data": data,
                    "error": _error,
                    "last_message": _last_message
                    }
    try:
        await ws.send_json(snd_mssg)
    except Exception as e:
        logger.error(f"send_message on '{log_comment}', exception - {e}")
    else:
        logger.debug(snd_mssg)
        logger.info(snd_mssg)
        is_ok = True

    return is_ok


@app.websocket("/ws_buffer")
async def websocket(ws: WebSocket):
    sample_rate = 8000
    wait_null_answers=True

    await ws.accept()
    client_id = id(websocket)
    audio_buffer[client_id] = AudioSegment.silent(100, frame_rate=16000)
    audio_overlap[client_id] = AudioSegment.silent(100, frame_rate=16000)

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
                    wait_null_answers = json_cfg.get('wait_null_answers', wait_null_answers)
                    sample_rate=json_cfg.get('sample_rate', sample_rate)
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

                # Переводим чанк в объект Audiosegment
                audiosegment_chunk = AudioSegment(
                    chunk,
                    frame_rate = sample_rate,  # Укажи частоту дискретизации
                    sample_width = 2,   # Ширина сэмпла (2 байта для int16)
                    channels = 1        # Количество каналов. По умолчанию - 1, Моно.
                    )

                # Приводим фреймрейт к фреймрейту модели
                if sample_rate != 16000:
                    audiosegment_chunk = audiosegment_chunk.set_frame_rate(16000)

                # Копим буфер
                audio_buffer[client_id] += audiosegment_chunk

                # Накопили больше нормы
                if (audio_overlap[client_id]+audio_buffer[client_id]).duration_seconds > 15:

                    # Проверяем новый чанк перед объединением (там же режем хвост и добавляем его при необходимости)
                    find_last_speech_position(client_id)

                    stream = None
                    stream = recognizer.create_stream()

                    # перевод в семплы для распознавания.
                    samples = get_np_array_samples_float32(audio_to_asr[client_id].raw_data, 2)
                    stream.accept_waveform(sample_rate=audiosegment_chunk.frame_rate, waveform=samples)
                else:
                    continue
            except Exception as e:
                logger.error(f"AcceptWaveform error - {e}")
            else:
                recognizer.decode_stream(stream)
                try:
                    result = stream.result
                    logger.debug(result)
                except Exception as e:
                    logger.error(f"recognizer.get_result(stream()) error - {e}")
                else:
                    if len(result.text) == 0:
                        if wait_null_answers:
                            if not await send_messages(ws, _silence = True, _data = None, _error = None):
                                logger.error(f"send_message not ok work canceled")
                                return
                            await asyncio.sleep(0.01)
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

    try:
        last_stream = recognizer.create_stream()

        # перевод в семплы для распознавания.
        audio_to_asr[client_id] = audio_overlap[client_id] + audio_buffer[client_id]

        logger.debug(f'итоговое сообщение - {audio_to_asr[client_id].duration_seconds} секунд')
        samples = get_np_array_samples_float32(audio_to_asr[client_id].raw_data, 2)
        last_stream.accept_waveform(sample_rate=audio_to_asr[client_id].frame_rate, waveform=samples)
        recognizer.decode_stream(last_stream)
        last_result = last_stream.result
        logger.debug(f"Последний результат {last_result.text}")

    except Exception as e:
        logger.error(f"stream.result error - {e}")

    else:
        if len(last_result.text) == 0:
            is_silence = True
            result = None
        else:
            logger.debug(last_result)
            is_silence = False

        if not await send_messages(ws, _silence=is_silence, _data=last_result, _error=None, _last_message=True):
            logger.error(f"send_message not ok work canceled")
            return

    await ws.close()