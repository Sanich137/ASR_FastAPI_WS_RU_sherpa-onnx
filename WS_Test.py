#!/usr/bin/env python3
import asyncio
from pathlib import Path

import websockets
from websockets.exceptions import ConnectionClosedOK
import ujson
from pydub import AudioSegment


async def ws_audio_to_text(uri, file_path, frame_rate=8000, wait_null_answers=False):
    recognised_raw_text = list()
    wait_null_answers = False
    wf = list() # wave_form - битовый поток аудио

    sound = AudioSegment.from_file(str(file_path))


    if sound.frame_rate != frame_rate:
        print(f'Текущий фреймрейт - {sound.frame_rate} будет исправлен на 8000')
        sound = sound.set_frame_rate(frame_rate)
        print(f'Фреймрейт после исправления - {sound.frame_rate}')
    else:
        print(f'Текущий фреймрейт - {sound.frame_rate} подходит для дальнейшей работы')

    separate_channels = sound.split_to_mono()

    for n_channel in range(len(separate_channels)):
        wf.append(separate_channels[n_channel].raw_data)
        recognised_raw_text.append(list())
        recognised_raw_text[n_channel].append(
                                    {'data': list()}
                                    )
        async with websockets.connect(uri) as websocket:
            # separate_channels[n_channel].set_frame_rate(8000)
            ws_config = {
                "sample_rate": separate_channels[n_channel].frame_rate,
                "wait_null_answers": wait_null_answers
                    }
            await websocket.send(ujson.dumps({'config': ws_config}, ensure_ascii=True))

            _from = 0
            _step = frame_rate*5  # sound.frame_rate * config.buffer_size_sec
            _to = _step
            pause_when = 20058003

            while _from < len(wf[n_channel]):

                data = wf[n_channel][_from: _to]
                _from = _to
                _to += _step
                if len(data) == 0:
                    break
                else:
                    if _to < pause_when:
                        await websocket.send(data)
                    else:
                        await asyncio.sleep(0.01)
                        print("Ждём 5 сек")
                        pause_when = pause_when+20058003

                    try:
                        ws_answer = ujson.loads(await asyncio.wait_for(websocket.recv(), 0.01))
                        # print(ws_answer.get("data").get("text"))
                        print(ws_answer)
                        # recognised_raw_text[n_channel].append(ws_answer)
                    except ConnectionClosedOK:
                        print("Connection closed from outer client")
                        break
                    except TimeoutError:
                        pass
                    else:
                        if ws_answer.get('last_message'):
                            await websocket.close()
                            break

                    ws_answer = None
                    # await asyncio.sleep(0.01)

            else:
                await websocket.send('{"eof" : 1}')

            while True:
                try:
                    ws_answer = ujson.loads(await asyncio.wait_for(websocket.recv(), 0.01))
                    try:
                        # print(ws_answer.get("data").get("text"))
                        print(ws_answer)
                    except Exception :
                        # print(ws_answer.get("data").get("text"))
                        print(ws_answer)
                    # recognised_raw_text[n_channel].append(ws_answer)
                except ConnectionClosedOK:
                    print("Connection closed from outer client")
                    break
                except TimeoutError:
                    pass
                else:
                    if ws_answer.get('last_message'):
                        await websocket.close()
                        break

                ws_answer = None
                await asyncio.sleep(0.01)

    return recognised_raw_text

path_to_file = Path(".\\trash\\1.wav")

asyncio.run(ws_audio_to_text('ws://127.0.0.1:49153/ws', path_to_file))
