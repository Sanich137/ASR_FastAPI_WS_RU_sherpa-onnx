from intervaltree import IntervalTree

async def do_diarized_dialogue(asr_data: list, diarized_data: list):
    """
        :param asr_data:  {"data":{"result":
                                            [
                                                {
                                                "start": float(),
                                                "end": float(),
                                                "word": str()
                                                }, ...
                                            ],
                                        "text": str()
                                },
                            ...
                            }
        :return:  {
                # "list_of_sentenced_recognitions": list() - список словарей с текстом, временем начала и номером спикера
                # "err_state": err_state - ошибка, если не удалось
            }
        """
    # собираем все слова после распознавания в один список
    word_list = list()
    word_list = sum([data.get("data").get("result") for data in asr_data], [])
    print(word_list)
    pass

    # Создание интервального дерева для диаризации
    diarization_tree = IntervalTree()

    for speaker_start, speaker_end, speaker_id in diarization_segments:
        diarization_tree.addi(speaker_start, speaker_end, speaker_id)

    # Сопоставление слов со спикерами
    for word in asr_words:
        overlapping = diarization_tree.overlap(word.start, word.end)
        if overlapping:
            best_match = max(overlapping, key=lambda x: x.end - x.begin)
            word.speaker = best_match.data
        else:
            # Расширение интервала на дельту (например, 0.2 сек)
            expanded_start = max(0, word.start - 0.2)
            expanded_end = word.end + 0.2
            overlapping = diarization_tree.overlap(expanded_start, expanded_end)
            if overlapping:
                best_match = max(overlapping, key=lambda x: x.end - x.begin)
                word.speaker = best_match.data
            else:
                word.speaker = "unknown"