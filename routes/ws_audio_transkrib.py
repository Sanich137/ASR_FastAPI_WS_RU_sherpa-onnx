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

vad = webrtcvad.Vad()
vad.set_mode(1)
# Глобальные переменные
client_overlap = defaultdict()
client_overlap_duration = defaultdict(float)
client_start_time = defaultdict(float)
MAX_OVERLAP_DURATION = 15.0

def find_last_speech_position(audio: bytes) -> tuple:
    """ Находит позицию последнего сегмента речи в аудио.
        Если не находит ни одного сегмента без речи, помечает его как полностью речь
    """
    sample_width = 2
    is_full = True
    audio = np.frombuffer(audio, dtype=np.int16)

    # Преобразование в int16
    if audio.dtype != np.int16:
        audio = (audio * 32767).astype(np.int16)

    # Преобразование в моно (если необходимо)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1).astype(np.int16)

    # audio = audio.astype(np.float32) / (2 ** (8 * sample_width - 1))

    speech_end = len(audio)
    # Разделение на фрагменты
    frame_duration_ms = 20
    frame_length = int(8000 * frame_duration_ms / 1000)
    frames = [audio[i:i + frame_length] for i in range(0, len(audio), frame_length)]

    # Проверка каждого фрагмента
    for i, frame in enumerate(reversed(frames)):
        try:
            if len(frame) < frame_length:
                continue  # Пропустить последний неполный фрагмент
            else:
                if not vad.is_speech(frame.tobytes(), 8000):
                    logger.debug(f"Найден не голос на speech_end = {speech_end}")
                    is_full = False
                    # break
                else:
                    logger.debug(f"Найден ГОЛОС на speech_end = {speech_end}")
            speech_end = len(audio) - i * frame_length  # Общая продолжительность аудио минус длинна Фрейма х количество
            # фреймов с голосом
        except Exception as e:
            logger.error(f"Ошибка VAD - {e}")

    return is_full, speech_end, audio[:speech_end]


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
                    sample_rate=json_cfg.get('sample_rate', 8000)
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
                stream = None
                stream = recognizer.create_stream()

                # Получаем новый чанк с данными
                # chunk = await get_np_array(message.get('bytes'))  #  Todo - не забвть о конвертации в ndarray
                chunk = message.get('bytes')
                # samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                samples = get_np_array(chunk, 2)
                # Проверяем новый чанк перед объединением
                # is_chunk_full_speech, speech_end, np_chunk_w_speech  = find_last_speech_position(chunk)

                # logger.debug(f'is_chunk_full_speech = {is_chunk_full_speech}')
                # logger.debug(f'speech_end = {speech_end}')

                logger.debug("accept_waveform")

                # stream.accept_waveform(sample_rate=8000, waveform=np_chunk_w_speech)
                stream.accept_waveform(sample_rate=sample_rate, waveform=samples)


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

    try:
        result = stream.result
    except Exception as e:
        logger.error(f"stream.result error - {e}")
    else:
        if len(result.text) == 0:
            is_silence = True
            result = None
        else:
            logger.debug(result)
            is_silence = False

        if not await send_messages(ws, _silence=is_silence, _data=result, _error=None, _last_message=True):
            logger.error(f"send_message not ok work canceled")
            return

    await ws.close()