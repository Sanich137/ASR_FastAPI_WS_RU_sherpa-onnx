import os
import logging
from typing import Any, Coroutine

import httpx
import io

from pydub import AudioSegment

from utils.pre_start_init import posted_and_downloaded_audio
from utils.pre_start_init import paths

async def getting_audiofile(file_url, post_id) -> [bool, str]:
    res = False
    error = str()
    file_ext = file_url.path.split('/')[-1].split('.')[-1]

    if file_ext in ['mp3', 'wav', 'ogg']:
        try:
            get_file_url = file_url.unicode_string()
        except Exception as e:
            logging.error(f"Error_url_parsing = {e}")
            # get_file_url = file_url.geturl()
        else:
            with httpx.Client() as sess:
                try:
                    response = sess.get(
                        url=get_file_url
                    )
                    file_data = response.content
                except Exception as e:
                    logging.error(f'Ошибка получения файла из ЕРП - {e}')
                    error = f"Getting file error - {e}"
                else:
                    posted_and_downloaded_audio[post_id] = io.BytesIO(file_data)
                    res = True
    else:
        error = "No audio file in request link"

    return res, error

# Todo - убрать корягу ниже
async def open_default_audiofile(post_id) -> tuple[bool, str]:
    res = False
    error = str()
    file = paths.get('test_file')
    file_ext = str(file).split('/')[-1].split('.')[-1]
    buffer = io.BytesIO()

    if file_ext in ['mp3', 'wav', 'ogg']:
        try:
            posted_and_downloaded_audio[post_id] = AudioSegment.from_file(file=file).export(buffer,  format="wav")
        except Exception as e:
            logging.error(f"Error_file_opening = {e}")
        else:
            res = True
    else:
        error = "No audio file in request link"

    return res, error
