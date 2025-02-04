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
        wf = wave.open("trash//orig.wav", "rb")
        frame_rate = wf.getframerate()
        config = {
            "sample_rate": frame_rate,
            "wait_null_answers": wait_null_answers
        }
        await websocket.send(ujson.dumps({'config': config}, ensure_ascii=True))
        buffer_size = frame_rate * 8  # 0.8 seconds of audio, don't make it too small otherwise compute will be slow

        while True:
            data = wf.readframes(buffer_size)
            if len(data) == 0:
                break
            else:
                await websocket.send(data)
                await asyncio.sleep(0.01)
                # print("--->")

            try:

                rsv = ujson.loads(await asyncio.wait_for(websocket.recv(), 0.01))
                print(rsv.get("data").get("text"))

            except TimeoutError:
                pass

        await websocket.send('{"eof" : 1}')

        while True:
            try:
                rsv = ujson.loads(await websocket.recv())
                try:
                    print(rsv.get("data").get("text"))
                except Exception:
                    print(rsv)
            except ConnectionClosedOK:
                print("Connection closed from outer client")
                print((datetime.now() - started).total_seconds())

                break


asyncio.run(run_test('ws://192.168.100.29:49152/ws'))
# asyncio.run(run_test('ws://127.0.0.1:49153/ws_buffer'))
