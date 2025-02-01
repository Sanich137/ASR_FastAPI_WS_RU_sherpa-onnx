import logging

import ujson
import statistics
from utils.states_machine import State


#  Вычисляем скорость речи пользователя
def sensitized(is_async=False, task_id=None, raw_recognition=None):
    err_state = None
    if not raw_recognition:
        raw_recognition = State.response_data[task_id].get('raw_recognition')
    sentenced_recognition = []
    text_only = []

    if not raw_recognition:
        err_state = "Err_No_raw_recognition_data"
    else:
        for results in raw_recognition:
            words = list()
            word_pause = 1.5
            sentence_element = None
            one_text_only = str()
            err_state = None

            # Если использовать передачу аудио частями, тот тут будет список. Если отдавать всё аудио, то нет. По этому оставляем.
            for res in results:
                try:
                    json_res = ujson.loads(res)
                except Exception as e:
                    logging.debug(e)
                    json_res = res

                if "result" in json_res:
                    for w in json_res["result"]:
                        words.append(w)
                    one_text_only += (json_res.get("text", None)) + ' '
                else:
                    continue

            if not words:
                err_state = "Err_No_words"
            else:
                between_words_delta = []
                end_time = words[0].get('end')

                for word in words[1::]:
                    between_words_delta.append(word.get('end') - end_time)
                    end_time = word.get('end')
                # Предполагаем, что пауза в выражении там, где пробел больше чем в 1,5 раза разницы между слов
                words_mean = statistics.mean(between_words_delta) * word_pause

                sentence_element = []
                sentences = []
                start_time = 0
                end_time = 0
                for word in words:
                    if start_time == 0:
                        start_time = word.get('start')
                    if end_time == 0:
                        end_time = word.get('end')

                    if (word.get('start') - end_time) < words_mean:
                        sentences.append(word.get('word'))
                        end_time = word.get('end')

                    else:
                        sentence_element.append({
                            "start": start_time,
                            "text": ' '.join(str(word) for word in sentences)
                        })
                        sentences = list()
                        sentences.append(word.get('word'))
                        start_time = 0
                        end_time = 0
                        continue
                sentence_element.append({
                    "start": start_time,
                    "text": ' '.join(str(word) for word in sentences)
                })

            # Собрали с разбивкой по предложениям
            sentenced_recognition.append(sentence_element)

            # Собрали только текст
            text_only.append(one_text_only)

    speaker = ['Клиент: ', 'Юрист: ']

    for speaker_sentence_index in reversed(range(len(sentenced_recognition))):
        for sentence in sentenced_recognition[speaker_sentence_index]:
            sentence['text'] = speaker[speaker_sentence_index] + sentence['text']

    sentenced_recognition_joined = sentenced_recognition[0] + sentenced_recognition[1]
    new_sentenced_recognition = sorted(sentenced_recognition_joined, key=lambda d: d['start'])
    list_of_resentenced_recognition = str()
    for sentence in new_sentenced_recognition:
        list_of_resentenced_recognition += sentence.get('text') + '\n'

    if is_async:
        State.response_data[task_id]['state'] = 'text_successfully_sentenced'
        State.response_data[task_id]['sentenced_recognition'] = list_of_resentenced_recognition
        State.response_data[task_id]['recognised_text'] = text_only
        State.response_data[task_id]['error'] = err_state

    else:
        return {
            "sentenced_recognition": list_of_resentenced_recognition,
            "text_only": text_only,
            "state": err_state
        }
