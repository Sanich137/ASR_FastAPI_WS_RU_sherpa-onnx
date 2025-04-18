from Diarisation import diarizer
from Diarisation.do_diarize import load_and_preprocess_audio
from utils.pre_start_init import posted_and_downloaded_audio
from utils.do_logging import logger


def merge_asr_diarisation(asr_data, diarisation_data):
    def intersection(start1, end1, start2, end2):
        return max(0, min(end1, end2) - max(start1, start2))

    # Извлекаем все слова из asr_data
    words = []
    for block in asr_data["channel_1"]:
        for word_info in block["data"]["result"]:
            words.append({
                "start": word_info["start"],
                "end": word_info["end"],
                "word": word_info["word"],
                "conf": word_info["conf"]
            })

    # Присваиваем спикеров словам
    speaker_words = {}
    for word in words:
        assigned_speaker = "Unknown"
        for dia_block in diarisation_data:
            # Проверяем, попадает ли слово в интервал блока диаризации
            if dia_block["start"] <= word["start"] <= dia_block["end"] or \
                    dia_block["start"] <= word["end"] <= dia_block["end"] or \
                    intersection(word["start"], word["end"], dia_block["start"], dia_block["end"]) > 0:
                assigned_speaker = dia_block["speaker"]
                break
        if assigned_speaker not in speaker_words:
            speaker_words[assigned_speaker] = []
        speaker_words[assigned_speaker].append(word)

    # Группируем слова в блоки для каждого спикера
    speaker_blocks = {}
    for speaker, words in speaker_words.items():
        groups = group_words(words)
        speaker_blocks[speaker] = []
        for group in groups:
            text = " ".join([w["word"] for w in group])
            speaker_blocks[speaker].append({
                "data": {
                    "result": group,
                    "text": text
                }
            })

    # Обрабатываем блоки diarisation_data без слов
    for dia_block in diarisation_data:
        speaker = dia_block["speaker"]
        start = dia_block["start"]
        end = dia_block["end"]
        has_words = False
        for word in words:
            if intersection(start, end, word["start"], word["end"]) > 0 or \
                    (start <= word["start"] <= end) or (start <= word["end"] <= end):
                has_words = True
                break
        if not has_words:
            if speaker not in speaker_blocks:
                speaker_blocks[speaker] = []
            speaker_blocks[speaker].append({
                "data": {
                    "result": [{
                        "start": start,
                        "end": end,
                        "word": "...не разборчиво..."
                    }],
                    "text": "...не разборчиво..."
                }
            })

    # Сортируем блоки по времени начала
    for speaker in speaker_blocks:
        speaker_blocks[speaker].sort(key=lambda x: x["data"]["result"][0]["start"])

    return speaker_blocks


def group_words(words):
    if not words:
        return []
    words = sorted(words, key=lambda x: x["start"])
    groups = []
    current_group = [words[0]]
    for word in words[1:]:
        if word["start"] <= current_group[-1]["end"] + 0.5:  # Небольшой порог для объединения
            current_group.append(word)
        else:
            groups.append(current_group)
            current_group = [word]
    if current_group:
        groups.append(current_group)
    return groups


async def do_diarizing(
        file_id:str,
        asr_raw_data,
        num_speakers:int = -1,
        filter_cutoff:int = 100,
        filter_order:int = 10,
        ):
    # Предобработка аудио.
    audio_frames = load_and_preprocess_audio(posted_and_downloaded_audio[file_id])

    # Непосредственно получение временных меток речи
    diar_result = diarizer.diarize_and_merge(
        audio_frames,
        num_speakers=num_speakers,
        filter_cutoff=filter_cutoff,
        filter_order=filter_order,
    )

    for r in diar_result:
        logger.info(f"Спикер {r['speaker']}: {r['start']:.2f} - {r['end']:.2f} сек")


    # Построение структуры аналогично raw_data для дальнейшего построения диалога и вывод результата
    return merge_asr_diarisation(asr_raw_data, diar_result)




if __name__ == "__main__":
    asr_raw_data = {
      "channel_1": [
        {
          "data": {
            "result": [
              {
                "conf": 1.0,
                "start": 2.08,
                "end": 2.28,
                "word": "алло"
              }
            ],
            "text": "алло"
          }
        },
        {
          "data": {
            "result": [
              {
                "conf": 1.0,
                "start": 19.84,
                "end": 19.88,
                "word": "ну"
              },
              {
                "conf": 1.0,
                "start": 20.48,
                "end": 20.72,
                "word": "давай"
              },
              {
                "conf": 1.0,
                "start": 20.92,
                "end": 20.96,
                "word": "да"
              },
              {
                "conf": 1.0,
                "start": 21.2,
                "end": 21.56,
                "word": "проверим"
              },
              {
                "conf": 1.0,
                "start": 21.72,
                "end": 21.84,
                "word": "вот"
              },
              {
                "conf": 1.0,
                "start": 21.96,
                "end": 21.96,
                "word": "я"
              },
              {
                "conf": 1.0,
                "start": 22.08,
                "end": 22.16,
                "word": "что"
              },
              {
                "conf": 1.0,
                "start": 22.32,
                "end": 22.36,
                "word": "то"
              },
              {
                "conf": 1.0,
                "start": 22.48,
                "end": 22.56,
                "word": "там"
              },
              {
                "conf": 1.0,
                "start": 22.72,
                "end": 22.88,
                "word": "тебе"
              },
              {
                "conf": 1.0,
                "start": 23.0,
                "end": 23.48,
                "word": "поговорил"
              },
              {
                "conf": 1.0,
                "start": 24.12,
                "end": 24.32,
                "word": "этого"
              },
              {
                "conf": 1.0,
                "start": 24.56,
                "end": 25.04,
                "word": "достаточно"
              }
            ],
            "text": "ну давай да проверим вот я что то там тебе поговорил этого достаточно"
          }
        },
        {
          "data": {
            "result": [
              {
                "conf": 1.0,
                "start": 29.056,
                "end": 29.216,
                "word": "раз"
              },
              {
                "conf": 1.0,
                "start": 29.416,
                "end": 29.536,
                "word": "два"
              },
              {
                "conf": 1.0,
                "start": 29.696,
                "end": 29.816,
                "word": "три"
              }
            ],
            "text": "раз два три"
          }
        }
      ]
    }
    diar_result = [
      {
        "start": 1.25,
        "end": 2.48,
        "speaker": 0
      },
      {
        "start": 16.706,
        "end": 16.976,
        "speaker": 0
      },
      {
        "start": 19.618,
        "end": 25.137999999999998,
        "speaker": 0
      },
      {
        "start": 28.994,
        "end": 30.674,
        "speaker": 0
      }
    ]
    print(merge_asr_diarisation(asr_raw_data, diar_result))

