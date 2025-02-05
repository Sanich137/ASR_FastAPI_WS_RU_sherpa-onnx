import asyncio

import ujson

import math

# Функция для преобразования логарифмических вероятностей в обычные
def logprob_to_prob(logprob):
    return math.exp(logprob)  # Используем exp для преобразования ln(prob)



# Асинхронная функция для обработки JSON
async def process_asr_json(input_json):
    # Парсим JSON
    # data = ujson.loads(input_json)
    data = input_json

    # Формируем шаблон результата
    result = {"data": {"result": [], "text": ""}}

    # Собираем слова из токенов
    words = []
    current_word = {"tokens": [], "start": None, "end": None, "probs": []}

    if not data:
        return result

    for token, timestamp, prob in zip(data["tokens"], data["timestamps"], data["ys_probs"]):
        # Если токен начинается с пробела, это начало нового слова
        if token.startswith(" "):
            if current_word["tokens"]:
                words.append(current_word)
            current_word = {"tokens": [token.strip()], "start": timestamp, "end": timestamp, "probs": [prob]}
        else:
            current_word["tokens"].append(token)
            current_word["end"] = timestamp
            current_word["probs"].append(prob)

    # Добавляем последнее слово
    if current_word["tokens"]:
        words.append(current_word)


    for word in words:
        # Объединяем токены в слово
        word_text = "".join(word["tokens"]).strip()

        # Преобразуем лог-вероятности в обычные
        probs = [logprob_to_prob(p) for p in word["probs"]]

        # Рассчитываем среднюю вероятность
        avg_prob = sum(probs) / len(probs)

        # Нормализуем conf, чтобы уверенные предсказания были ближе к 1
        conf = 1 - math.exp(-avg_prob * 5)  # Масштабирующий коэффициент (5) можно настроить

        # Добавляем слово в результат
        result["data"]["result"].append({
            "conf": conf,
            "start": word["start"],
            "end": word["end"],
            "word": word_text
        })

        # Добавляем слово в итоговый текст
        result["data"]["text"] += word_text + " "

    # Убираем лишний пробел в конце текста
    result["data"]["text"] = result["data"]["text"].strip()

    print(result)
    return result


async def process_asr_as_object(input_result, time_shift = 0.0):
    # Парсим STR в JSON
    data = ujson.loads(input_result)

#    data = input_result

    # Формируем шаблон результата
    result = {"data": {"result": [], "text": ""}}

    # Собираем слова из токенов
    words = []
    current_word = {"tokens": [], "start": None, "end": None, "probs": []}

    if not data:
        return result

    for token, timestamp in zip(data["tokens"], data["timestamps"]):
        # Если токен начинается с пробела, это начало нового слова
        if token.startswith(" "):
            if current_word["tokens"]:
                words.append(current_word)
            current_word = {"tokens": [token.strip()], "start": float(timestamp)+time_shift,
                            "end": float(timestamp)+time_shift, }
        else:
            current_word["tokens"].append(token)
            current_word["end"] = float(timestamp)+time_shift

    # Добавляем последнее слово
    if current_word["tokens"]:
        words.append(current_word)


    for word in words:
        # Объединяем токены в слово
        word_text = "".join(word["tokens"]).strip()

        # # Преобразуем лог-вероятности в обычные
        # probs = [logprob_to_prob(p) for p in word["probs"]]
        #
        # # Рассчитываем среднюю вероятность
        # avg_prob = sum(probs) / len(probs)
        #
        # # Нормализуем conf, чтобы уверенные предсказания были ближе к 1
        # conf = 1 - math.exp(-avg_prob * 5)  # Масштабирующий коэффициент (5) можно настроить

        # Добавляем слово в результат
        result["data"]["result"].append({
            # "conf": conf,
            "start": word["start"],
            "end": word["end"],
            "word": word_text
        })

        # Добавляем слово в итоговый текст
        result["data"]["text"] += word_text + " "

    # Убираем лишний пробел в конце текста
    result["data"]["text"] = result["data"]["text"].strip()

    print(result)
    return result


if __name__ == "__main__":
    asr_str_json = {"lang": "",
                    "emotion": "",
                    "event": "",
                    "text": " кто юрист прикрепила телефонный разговор мы же с вами ещё беседуем по идее вы ну как бы разговор ещё не заканчивали поэтому где-то вот ну несколько минут тоже нужно подождать на это чтобы систем всё",
                    "timestamps": [0.00, 0.04, 0.16, 0.20, 0.32, 0.44, 0.60, 0.76, 0.80, 0.96, 1.08, 1.32, 1.72, 1.92,
                                   2.12, 2.24, 2.72, 2.88, 3.04, 3.16, 3.24, 3.44, 3.68, 3.80, 3.88, 4.00, 4.16, 4.32,
                                   4.52, 4.64, 4.76, 5.12, 6.08, 6.32, 6.52, 6.72, 6.88, 6.96, 7.16, 7.36, 7.56, 7.68,
                                   7.84, 7.96, 8.08, 8.20, 8.60, 9.28, 9.52, 9.56, 9.72, 10.28, 10.56, 11.04, 11.36,
                                   11.44, 11.64, 11.92, 12.04, 12.16, 12.32, 12.52, 12.68, 13.00, 13.32, 13.36, 13.44,
                                   13.52, 13.72],
                    "tokens": [" к", "то", " ", "ю", "ри", "ст", " при", "к", "ре", "пи", "ла", " телефон", "ный",
                               " раз", "го", "вор", " мы", " же", " с", " в", "ами", " ещё", " бе", "с", "е", "ду",
                               "ем", " по", " и", "де", "е", " вы", " ну", " как", " бы", " раз", "го", "вор", " ещё",
                               " не", " за", "ка", "н", "чи", "ва", "ли", " поэтому", " где", "-", "то", " вот", " ну",
                               " несколько", " минут", " то", "же", " нужно", " по", "до", "жд", "ать", " на", " это",
                               " чтобы", " с", "и", "ст", "ем", " всё"], "words": []}



    asyncio.run(process_asr_as_object(asr_str_json))