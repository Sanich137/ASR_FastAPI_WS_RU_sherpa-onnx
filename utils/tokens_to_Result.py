import asyncio
import math
#from utils.do_logging import logger
import logging as logger

# Функция для преобразования логарифмических вероятностей в обычные
def logprob_to_prob(logprob):
    return math.exp(logprob)  # Используем exp для преобразования ln(prob)

# Асинхронная функция для обработки JSON
async def process_asr_json(input_json, time_shift = 0.0):
    # Парсим JSON
    data = input_json

    # Формируем шаблон результата
    result = {"data": {"result": [], "text": ""}}

    # Собираем слова из токенов
    words = []
    current_word = {"tokens": [], "start": 0.0, "end": 0.0, "probs": []}

    if not data:
        return result

    for token, timestamp in zip(data["tokens"], data["timestamps"]):
        # Если токен начинается с пробела, это начало нового слова
        if token.startswith(" "):
            if current_word["tokens"]:
                words.append(current_word)
            current_word = {"tokens": [token.strip()], "start": timestamp, "end": timestamp}
        else:
            current_word["tokens"].append(token)
            current_word["end"] = timestamp

    # Добавляем последнее слово
    if current_word["tokens"]:
        words.append(current_word)


    for word in words:
        # Объединяем токены в слово
        word_text = "".join(word["tokens"]).strip()

        # Требуется в принимающем приложении. Для общей работы смысла не имеет.
        conf = 1

        # Добавляем слово в результат
        result["data"]["result"].append({
            "conf": round(conf, 2),
            "start": word.get("start", 0) +round(time_shift, 2),
            "end": word["end"]+round(time_shift, 2),
            "word": word_text
        })

        # Добавляем слово в итоговый текст
        result["data"]["text"] += word_text + " "

    # Убираем лишний пробел в конце текста
    result["data"]["text"] = result["data"]["text"].strip()

    logger.debug(f'Результат изменения формата ответа - {result}')

    return result


async def process_gigaam_asr(input_json, time_shift = 0.0):
    # Парсим JSON
    data = input_json

    # Формируем шаблон результата
    result = {"data": {"result": [], "text": ""}}

    # Собираем слова из токенов
    words = []
    current_word = ""
    start_time, end_time = 0.0, 0.0

    for i, token in enumerate(data['tokens']):
        if token != ' ':
            if current_word == "":
                start_time = round((data['timestamps'][i]+time_shift), 3)
            current_word += token
            end_time = round((data['timestamps'][i]+time_shift), 3)
        else:
            if current_word != "":
                words.append({'word': current_word, 'start': start_time, 'end': end_time})
                current_word = ""

    # Добавляем последнее слово, если оно есть
    if current_word != "":
        words.append({'word': current_word, 'start': start_time, 'end': end_time})

    # Формируем итоговый массив
    result['data'] = {
        'result': [{'conf': 1.0, 'start': word['start'], 'end': word['end'], 'word': word['word']} for word in words],
        'text': data['text']
    }
    return result


if __name__ == "__main__":
    asr_str_json = {'emotion': '', 'event': '', 'lang': '', 'text': 'амлекс точка ру екатерина здравствуйте скажитепожалуйса как можно к вам обращить', 'timestamps': [2.48, 2.64, 2.84, 2.88, 2.96, 3.04, 3.12, 3.16, 3.24, 3.28, 3.36, 3.4, 3.44, 3.52, 3.56, 3.64, 3.76, 3.84, 3.88, 3.96, 4.0, 4.08, 4.12, 4.16, 4.24, 4.28, 4.36, 4.4, 4.48, 4.52, 4.56, 4.6, 4.68, 4.72, 4.76, 4.8, 4.88, 4.92, 15.76, 15.88, 15.92, 15.96, 16.04, 16.08, 16.12, 16.16, 16.24, 16.32, 16.36, 16.4, 16.44, 16.48, 16.52, 16.56, 16.64, 16.72, 16.76, 16.8, 16.84, 16.88, 16.92, 16.96, 17.0, 17.04, 17.08, 17.12, 17.16, 17.2, 17.24, 17.28, 17.32, 17.4, 17.44, 17.48, 17.52, 17.6, 17.64, 17.68, 17.72, 17.76], 'tokens': ['а', 'м', 'л', 'е', 'к', 'с', ' ', 'т', 'о', 'ч', 'к', 'а', ' ', 'р', 'у', ' ', 'е', 'к', 'а', 'т', 'е', 'р', 'и', 'н', 'а', ' ', 'з', 'д', 'р', 'а', 'в', 'с', 'т', 'в', 'у', 'й', 'т', 'е', ' ', 'с', 'к', 'а', 'ж', 'и', 'т', 'е', 'п', 'о', 'ж', 'а', 'л', 'у', 'й', 'с', 'а', ' ', 'к', 'а', 'к', ' ', 'м', 'о', 'ж', 'н', 'о', ' ', 'к', ' ', 'в', 'а', 'м', ' ', 'о', 'б', 'р', 'а', 'щ', 'и', 'т', 'ь'], 'words': []}
    res = asyncio.run(process_gigaam_asr(asr_str_json))
    print(res)