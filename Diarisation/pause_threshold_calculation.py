import statistics
from typing import List, Dict, Any


def calculate_pause_threshold(data: List[Dict[str, Any]]) -> float:
    """
    Вычисляет оптимальное значение pause_threshold на основе временных меток сегментов.
    Args: data: Список словарей raw_asr["channel_1"]
    Returns: Оптимальное значение pause_threshold (в секундах).
    """

    # Извлечение пауз
    pauses = []

    # Формат raw_asr: список словарей внутри 'channel_1' с 'data' и 'result'
    for i in range(len(data) - 1):
        pause = data[i + 1]['data']['result'][0]['start'] - data[i]['data']['result'][-1]['end']
        if pause > 0:
            pauses.append(pause)
    # Добавляем паузы внутри сегментов (между словами)
    for segment in data:
        words = segment['data']['result']
        for i in range(len(words) - 1):
            pause = words[i + 1]['start'] - words[i]['end']
            if pause > 0:
                pauses.append(pause)

    if not pauses:
        return 0.2  # Значение по умолчанию, если пауз нет

    # Вычисление статистики
    median_pause = statistics.median(pauses)

    lower_quartile = statistics.quantiles(pauses, n=4)[0]  # 25-й процентиль

    upper_quartile = statistics.quantiles(pauses, n=20)[-2]  # n-й процентиль

    print(pauses)

    max_pause = max(pauses)

    std_pause = statistics.stdev(pauses) if len(pauses) > 1 else 0.1

    # Определение типа речи
    is_dialogue = len(data) > 10 and median_pause < 0.3 and std_pause < 0.2

    # Выбор порога
    if is_dialogue:
        # Для диалогов: порог ближе к верхнему квартилю с небольшой корректировкой вверх
        pause_threshold = min(upper_quartile + 0.05, (upper_quartile + max_pause) / 2)
    else:
        # Для монологов: порог чуть выше медианы
        pause_threshold = median_pause + 0.03

    # Ограничение порога
    pause_threshold = max(0.1, min(pause_threshold, 0.3))

    # Округление до двух знаков
    return round(pause_threshold, 2)