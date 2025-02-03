from dbm import error

import ujson
import asyncio
import webrtcvad
import numpy as np
from utils.pre_start_init import app, WebSocket, WebSocketException
from utils.pre_start_init import recognizer
from utils.do_logging import logger
from utils.bytes_to_samples_audio import get_np_array
from utils.tokens_to_Result import process_asr_json


from collections import defaultdict

# from models.vosk_model import model

vad = webrtcvad.Vad(3)

# Глобальные переменные
client_overlap = defaultdict()
client_overlap_duration = defaultdict(float)
client_start_time = defaultdict(float)
MAX_OVERLAP_DURATION = 15.0

def find_last_speech_position(audio: bytes) -> tuple:
    """ Находит позицию последнего сегмента речи в аудио.
        Если не находит ни одного сегмента без речи, помечает его как полностью речь
    """
    speech_end = 0
    is_full = True
    audio = np.frombuffer(audio, dtype=np.int32)

    # Преобразование в int16
    if audio.dtype != np.int16:
        audio = (audio * 32767).astype(np.int16)

    # Преобразование в моно (если необходимо)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1).astype(np.int16)

    # Разделение на фрагменты
    frame_duration_ms = 30
    frame_length = int(8000 * frame_duration_ms / 1000)
    frames = [audio[i:i + frame_length] for i in range(0, len(audio), frame_length)]

    # Проверка каждого фрагмента
    for i, frame in enumerate(reversed(frames)):
        if not vad.is_speech(frame, 8000):
            speech_end = len(audio) - i*frame_length
            logger.debug(f"Найден не голос на speech_end = {speech_end}")
            break
    else:
        is_full = False

    return is_full, speech_end


def bytes_to_seconds(audio_bytes: bytes) -> float:
    return len(audio_bytes) / (8000 * 2)


async def send_messages(_socket, _data=None, _silence=True, _error=None, log_comment=None, _last_message=False):
    ws = _socket
    is_ok = False
    data = await process_asr_json(_data)

    snd_mssg = {"silence": _silence,
                    "data": data.get("data").get("text"),
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

    # Создаем поток для распознавания
    stream = recognizer.create_stream()
    wait_null_answers=True

    await ws.accept()
    client_id = id(websocket)

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
                    stream = recognizer.create_stream()
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
                # chunk = await get_np_array(message.get('bytes'))  #  Todo - не забвть о конвертации в ndarray
                chunk = message.get('bytes')

                # Проверяем новый чанк перед объединением
                is_chunk_full_speech, speech_end  = find_last_speech_position(chunk)

                logger.debug(f'is_chunk_full_speech = {is_chunk_full_speech}')
                logger.debug(f'speech_end = {speech_end}')

                logger.debug("accept_waveform")

                stream.accept_waveform(8000, data)



            except Exception as e:
                logger.error(f"AcceptWaveform error - {e}")
            else:
                if False:
                    pass
#                while recognizer.is_ready(stream):

                else:
                    recognizer.decode_stream(stream)
                    try:
                        result = stream.result_as_json_string
#                         result = ujson.loads(recognizer.get_result_as_json_string(stream))
                    except Exception as e:
                        logger.error(f"recognizer.get_result(stream()) error - {e}")
                    else:
                        if len(result.get("text")) == 0:
                            if wait_null_answers:
                                if not await send_messages(ws, _silence = True, _data = None, _error = None):
                                    logger.error(f"send_message not ok work canceled")
                                    return
                                await asyncio.sleep(1)
                            else:
                                logger.debug("sending silence partials skipped")
                                continue
                        # elif 'text' in result and len(ujson.decode(result).get('text')) == 0:
                        #     logger.debug("No text in result. Skipped")
                        #     continue

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

    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
        await asyncio.sleep(0.1)
    else:
        stream.input_finished()

        try:
            result = ujson.loads(recognizer.get_result_as_json_string(stream))
        except Exception as e:
            logger.error(f"recognizer.get_result(s)) error - {e}")
        else:
            if len(result.get("text")) == 0:
                is_silence = True
                result = None
            else:
                logger.debug(result)
                is_silence = False

            if not await send_messages(ws, _silence=is_silence, _data=result, _error=None, _last_message=True):
                logger.error(f"send_message not ok work canceled")
                return

    await ws.close()