import time
from pathlib import Path

import numpy as np

from utils.do_logging import logger
from utils.pre_start_init import recognizer
from pydub import AudioSegment



def online_decode(audio, time_tail):
    sound = AudioSegment.from_raw(str(audio_path))

    stream = recognizer.create_stream()
    sample_rate = audio.frame_rate
    samples = audio.get_array_of_samples()

    samples_int16 = np.frombuffer(samples, dtype=np.int16)
    samples_float32 = samples_int16.astype(np.float32)
    samples_float32 = samples_float32 / 32768

    stream.accept_waveform(sample_rate, samples_float32)

    try:
        recognizer.decode_stream(stream)
    except Exception as e:
        logger.error(f"Decode error - {e}."
                     f"\n audio_duration = {audio.duration_seconds}"
                     f"\n len(audio) = {len(audio)}")
        response = dict()
        response['timestamps'] = ""
        response['tokens'] = ""
        response['text'] = ""
        response['error'] = e

    else:
        response = dict()
        response['timestamps'] = list()

        # Расставляем верные временные метки каждого слога
        for timestamp in stream.result.timestamps:
            response['timestamps'].append(timestamp+time_tail)

        response['tokens'] = stream.result.tokens
        response['text'] = stream.result.text+" "

        # logger.debug(stream.result.text)
    return response


if __name__ == "__main__":
    root_path = Path(__file__).resolve().parent

    debug = False

    paths_to_wav = [
                    #root_path / "vosk-model-ru" / "test.wav",
                    #root_path / "trash" / "mono.wav",
                    # root_path / "trash" / "q.wav",
                    # root_path / "trash" / "2.wav",
                    root_path / "trash" / "long.mp3",
                    # root_path / "trash" / "2222.1639387328.91599.mp3"
                ]


    arguments = {
        "tokens": tokens_path,
        "encoder": encoder_path,
        "decoder": decoder_path,
        "joiner": joiner_path,
        "num_threads": 8,
        "decoding_method": "greedy_search",
        "debug": debug,
        "sample_rate": 16000,
        "feature_dim": 80,
        "sound_files": paths_to_wav,
        "tokens_path": tokens_path,
        "bpe_vocab": bpe_vocab,
        "provider": "CUDA",
    }

    # offline_decode(arguments)
    #
    # #results = [s.result.text for s in streams]
    # end_time = time.time()
    # # results = []
    # print("Done!")
    #
    # elapsed_seconds = end_time - start_time
    # rtf = elapsed_seconds / total_duration
    # print(f"num_threads: {model_config.get("num_threads")}")
    # print(f"decoding_method: {model_config.get("decoding_method")}")
    # print(f"Wave duration: {total_duration:.3f} s")
    # print(f"Elapsed time: {elapsed_seconds:.3f} s")
    # print(
    #     f"Real time factor (RTF): {elapsed_seconds:.3f}/{total_duration:.3f} = {rtf:.3f}"
    # )

