#!/usr/bin/env python3

import asyncio
from asyncio import timeout
import websockets
from websockets.exceptions import ConnectionClosedOK
import sys
import wave
import ujson
from datetime import datetime


async def run_test(uri):
    started = datetime.now()
    async with websockets.connect(uri) as websocket:
        wait_null_answers = False
        # wf = wave.open("trash//2724.1726990043.1324706.wav", "rb")
        wf = wave.open("trash//q.wav", "rb")
        config = {
            "sample_rate": wf.getframerate(),
            "wait_null_answers": wait_null_answers
        }
        await websocket.send(ujson.dumps({'config': config}, ensure_ascii=True))
        buffer_size = 12800  # 0.8 seconds of audio, don't make it too small otherwise compute will be slow

        while True:
            data = wf.readframes(buffer_size)
            if len(data) == 0:
                break
            else:
                await websocket.send(data)
                await asyncio.sleep(1)
                print("Sent")

            try:
                print(await asyncio.wait_for(websocket.recv(), 0.01))
                # logger.info(f"Послушали")
            except TimeoutError:
                pass

        await websocket.send('{"eof" : 1}')

        while True:
            try:
                print(await websocket.recv())
            except ConnectionClosedOK:
                print("Connection closed from outer client")
                print((datetime.now() - started).total_seconds())

                break


#asyncio.run(run_test('ws://192.168.100.29:49152/ws'))
asyncio.run(run_test('ws://127.0.0.1:49153/ws_buffer'))
