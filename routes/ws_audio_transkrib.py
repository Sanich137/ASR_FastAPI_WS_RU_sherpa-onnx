from dbm import error

import ujson
import asyncio

from utils.pre_start_init import app, WebSocket, WebSocketException
from utils.pre_start_init import recognizer
from utils.do_logging import logger
from utils.bytes_to_samples_audio import get_np_array
from utils.tokens_to_Result import process_asr_json
from models.fast_api_models import WebSocketModel

# from models.vosk_model import model



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


@app.post("/ws")
async def post_not_websocket(ws:WebSocketModel):
    """Описание для вебсокета ниже в описании WebSocketModel """
    return f"Прочти инструкцию в Schemas - 'WebSocketModel'"


@app.websocket("/ws")
async def websocket(ws: WebSocket):
    sample_rate=8000
    # Создаем поток для распознавания
    stream = recognizer.create_stream()
    # online_recognizer = BatchRecognizer(model, sample_rate)
    wait_null_answers=True

    await ws.accept()

    while True:
        try:
            message = await ws.receive()
        except Exception as wse:
            logger.error(f"receive WebSocketException - {wse}")
            return

        # logger.debug(f'Raw message - {message}')

        if isinstance(message, dict) and message.get('text'):
            try:
                if message.get('text') and 'config' in message.get('text'):
                    json_cfg = ujson.loads(message.get('text'))['config']
                    wait_null_answers = json_cfg.get('wait_null_answers', wait_null_answers)
                    stream = recognizer.create_stream()
#                     online_recognizer = BatchRecognizer(model, sample_rate)
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
                data = await get_np_array(message.get('bytes'))
                stream.accept_waveform(8000, data)
                logger.debug("accept_waveform")
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