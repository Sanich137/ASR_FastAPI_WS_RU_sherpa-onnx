from utils.do_logging import logger



async def send_messages(_socket, _data=None, _silence=True, _error=None, log_comment=None, _last_message=False):
    ws = _socket
    is_ok = False
    if _last_message:
        logger.debug('Последнее сообщение')
    if not _data:
        data = ""
    else:
        data = _data.get('data')

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