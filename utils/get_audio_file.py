import os
import logging
import httpx
import io

from utils.pre_start_init import posted_and_downloaded_audio

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


