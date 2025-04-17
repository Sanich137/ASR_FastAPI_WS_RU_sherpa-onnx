# from intervaltree import IntervalTree
from Diarisation import diarizer
from VoiceActivityDetector import vad
from Diarisation.do_diarize import load_and_preprocess_audio
from utils.pre_start_init import posted_and_downloaded_audio

async def do_diarizing(
        file_id:str = None,
        num_speakers:int = -1,
        filter_cutoff:int = 100,
        filter_order:int = 10,
        ):
    # Предобработка аудио.
    audio_frames = load_and_preprocess_audio(posted_and_downloaded_audio[file_id])

    # Непосредственно получение временных меток речи
    vad.set_mode(5)
    result = diarizer.diarize_and_merge(
        audio_frames,
        num_speakers=num_speakers,
        filter_cutoff=filter_cutoff,
        filter_order=filter_order,
    )

    for r in result:
        print(f"Спикер {r['speaker']}: {r['start']:.2f} - {r['end']:.2f} сек")

    # Построение структуры аналогично raw_data для дальнейшего построения диалога.




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
    # Создание интервального дерева для диаризации
    diarization_tree = IntervalTree()

    # собираем все слова после распознавания в один список
    words_list = sum([data.get("data").get("result") for data in asr_data], [])
    print(words_list)
    pass


    for speaker_start, speaker_end, speaker_id in diarized_data:
        diarization_tree.addi(speaker_start, speaker_end, speaker_id)

    # Сопоставление слов со спикерами
    for word in words_list:
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