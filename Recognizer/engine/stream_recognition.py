import logging

import ujson
from collections import defaultdict

import config
from Recognizer import recognizer
# from utils.pre_start_init import recognizer
from utils.bytes_to_samples_audio import get_np_array_samples_float32
from pydub import AudioSegment


async def recognise_w_calculate_confidence(audio_data,
                                     num_trials = 1,
                                     time_tolerance: float = 0.5) -> dict:
    """
    Вычисляет уверенность (`conf`) для каждого слова на основе многократного распознавания.

    :param audio_data: Аудиоданные в формате Audiosegment (puDub).
    :param num_trials: Количество попыток распознавания.
    :param time_tolerance: Допустимая погрешность временных меток (в секундах).
    :return: json c параметрами conf
    """

    # Результаты всех попыток распознавания
    result = dict()
    all_tokens = list()

    # Многократное распознавание
    for _ in range(num_trials):
        stream = None
        stream = recognizer.create_stream()

        # перевод в семплы для распознавания.
        samples = await get_np_array_samples_float32(audio_data.raw_data, 2)

        # передали аудиофрагмент на распознавание

        stream.accept_waveform(sample_rate=audio_data.frame_rate, waveform=samples)
        recognizer.decode_stream(stream)

        result = ujson.loads(str(stream.result))
        timestamps = result['timestamps']
        tokens = result['tokens']

        all_tokens.append(list(zip(tokens, timestamps)))

    # Словарь для хранения статистики по токенам
    token_stats = defaultdict(lambda: {"count": 0, "variants": defaultdict(int)})

    # Анализируем результаты всех прогонов
    for trial in all_tokens:
        for token, start in trial:
            # Группируем токены по времени начала с допуском
            time_key = round(start / time_tolerance) * time_tolerance
            token_stats[(token, time_key)]["count"] += 1
            token_stats[(token, time_key)]["variants"][token] += 1

    # Формируем итоговые токены с conf
    tokens = []
    timestamps = []
    probs = []

    for (token, time_key), stats in token_stats.items():
        # Наиболее часто встречающийся вариант
        most_common_token = max(stats["variants"].items(), key=lambda x: x[1])[0]

        # Уверенность (conf)
        conf = stats["variants"][most_common_token] / num_trials

        tokens.append(most_common_token)
        timestamps.append(time_key)
        probs.append(round(conf, 4))

    result["tokens"] = tokens
    result["timestamps"]=timestamps
    result["probs"] =probs

    # Возвращаем результат в формате JSON
    return result



async def simple_recognise(audio_data, ) -> dict:
    """
    Собираем токены в слова дополнительных вычислений не производит.

    :param audio_data: Аудиоданные в формате Audiosegment (puDub).
    :return: json =
        {
        "data": {
          "result": [

            {
              "conf": 1,
              "start": 32.22,
              "end": 32.46,
              "word": "сейчас"
            },
            {
              "conf": 1,
              "start": 33.26,
              "end": 33.74,
              "word": "попробуем"
            },
            {
              "conf": 1,
              "start": 37.46,
              "end": 37.82,
              "word": "свидания"
            }
          ],
          "text": "угу угу ладно сейчас попробуем все хорошо поняла вас спасибо да до свидания"
        }

    без дополнительных расчётов
    """

    # Приводим фреймрейт к фреймрейту модели
    if audio_data.frame_rate != config.BASE_SAMPLE_RATE:
        audio_data = audio_data.set_frame_rate(config.BASE_SAMPLE_RATE)


    stream = recognizer.create_stream()
    # перевод в семплы для распознавания.
    samples = await get_np_array_samples_float32(audio_data.raw_data, audio_data.sample_width)

    # передали аудиофрагмент на распознавание

    stream.accept_waveform(sample_rate=audio_data.frame_rate, waveform=samples)
    recognizer.decode_stream(stream)

    result = ujson.loads(str(stream.result))

    return result
