import ujson
import asyncio
from collections import defaultdict

import config
from Recognizer import recognizer
from utils.bytes_to_samples_audio import get_np_array_samples_float32
from utils.resamppling import resample_audiosegment


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
        audio_data = await resample_audiosegment(audio_data, config.BASE_SAMPLE_RATE)

    # перевод в семплы для распознавания.
    samples = await get_np_array_samples_float32(audio_data.raw_data, audio_data.sample_width)
    # Переносим декодирование в отдельный поток
    def decode_in_thread():
        stream = recognizer.create_stream()
        # передали аудиофрагмент на распознавание
        stream.accept_waveform(sample_rate=audio_data.frame_rate, waveform=samples)
        recognizer.decode_stream(stream)
        return str(stream.result)

    result_json = await asyncio.to_thread(decode_in_thread)

    # Парсим результат
    result = ujson.loads(result_json)

    return result

"{'emotion': '', 'event': '', 'lang': '', 'text': 'амлекс точка ру екатерина здравствуйте скажитепожалуйса как можно к вам обращить', 'timestamps': [2.48, 2.64, 2.84, 2.88, 2.96, 3.04, 3.12, 3.16, 3.24, 3.28, 3.36, 3.4, 3.44, 3.52, 3.56, 3.64, 3.76, 3.84, 3.88, 3.96, 4.0, 4.08, 4.12, 4.16, 4.24, 4.28, 4.36, 4.4, 4.48, 4.52, 4.56, 4.6, 4.68, 4.72, 4.76, 4.8, 4.88, 4.92, 15.76, 15.88, 15.92, 15.96, 16.04, 16.08, 16.12, 16.16, 16.24, 16.32, 16.36, 16.4, 16.44, 16.48, 16.52, 16.56, 16.64, 16.72, 16.76, 16.8, 16.84, 16.88, 16.92, 16.96, 17.0, 17.04, 17.08, 17.12, 17.16, 17.2, 17.24, 17.28, 17.32, 17.4, 17.44, 17.48, 17.52, 17.6, 17.64, 17.68, 17.72, 17.76], 'tokens': ['а', 'м', 'л', 'е', 'к', 'с', ' ', 'т', 'о', 'ч', 'к', 'а', ' ', 'р', 'у', ' ', 'е', 'к', 'а', 'т', 'е', 'р', 'и', 'н', 'а', ' ', 'з', 'д', 'р', 'а', 'в', 'с', 'т', 'в', 'у', 'й', 'т', 'е', ' ', 'с', 'к', 'а', 'ж', 'и', 'т', 'е', 'п', 'о', 'ж', 'а', 'л', 'у', 'й', 'с', 'а', ' ', 'к', 'а', 'к', ' ', 'м', 'о', 'ж', 'н', 'о', ' ', 'к', ' ', 'в', 'а', 'м', ' ', 'о', 'б', 'р', 'а', 'щ', 'и', 'т', 'ь'], 'words': []}"