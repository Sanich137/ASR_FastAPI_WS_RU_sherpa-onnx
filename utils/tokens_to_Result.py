import asyncio

import ujson
import math

# Функция для преобразования логарифмических вероятностей в обычные
def logprob_to_prob(logprob):
    return math.exp(logprob)  # Используем exp для преобразования ln(prob)



# Асинхронная функция для обработки JSON
async def process_asr_json(input_json):
    # Парсим JSON
    #data = ujson.loads(input_json)
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


if __name__ == "__main__":
    asr_json = {"text": " алло ну давай да проверим вот я что-то там тебе поговорил этого достаточно раз два три",
                "tokens": [" а", "л", "ло", " ну", " да", "ва", "й", " да", " про", "вер", "им", " вот", " я", " что",
                           "-", "то", " там", " те", "бе", " по", "го", "вор", "ил", " этого", " до", "ста", "то",
                           "чно", " раз", " два", " три"],
                "timestamps": [2.24, 2.36, 2.48, 20.08, 20.64, 20.84, 20.96, 21.16, 21.44, 21.60, 21.76, 21.92, 22.08,
                               22.24, 22.44, 22.48, 22.56, 22.76, 22.84, 23.04, 23.24, 23.36, 23.56, 24.36, 24.60,
                               24.76, 24.88, 25.04, 29.24, 29.56, 29.84],
                "ys_probs": [-0.909737, -0.892509, -1.221103, -0.762347, -0.524013, -0.794737, -0.507589, -0.526858,
                             -0.217407, -0.284924, -0.557514, -0.656335, -0.863437, -0.454388, -0.599445, -0.074061,
                             -0.115918, -0.329356, -0.111481, -0.206194, -0.599760, -0.526217, -0.617012, -0.402843,
                             -0.261690, -0.122997, -0.596523, -0.608447, -0.674983, -0.692588, -0.426215],
                "lm_probs": [], "context_scores": [], "segment": 0, "words": [], "start_time": 0.00, "is_final": False}
    asyncio.run(process_asr_json(asr_json))