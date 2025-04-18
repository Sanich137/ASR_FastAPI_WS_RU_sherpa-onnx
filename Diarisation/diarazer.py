from Diarisation import diarizer
from Diarisation.do_diarize import load_and_preprocess_audio
from utils.pre_start_init import posted_and_downloaded_audio
from utils.do_logging import logger
from collections import defaultdict

from collections import defaultdict


def match_asr_with_diarization(asr_data, diarization_data, min_overlap_ratio=0.5,
                               max_pause=2.0, group_unmatched_words=True):
    """
    Финальная версия с сохранением структуры исходных данных

    Параметры:
        group_unmatched_words: если True, группирует несопоставленные слова с ближайшими сопоставленными
    """
    result = defaultdict(list)
    diarization_data_sorted = sorted(diarization_data, key=lambda x: x['start'])

    for channel, asr_segments in asr_data.items():
        for asr_segment in asr_segments:
            # Сохраняем все оригинальные поля из asr_segment
            segment_metadata = {k: v for k, v in asr_segment.items() if k != 'data'}
            words = asr_segment['data']['result']
            word_assignments = {}

            # Первый проход: сопоставление слов с интервалами диаризации
            for i, word in enumerate(words):
                word_start = word['start']
                word_end = word['end']
                word_duration = word_end - word_start
                best_speaker = None
                best_overlap = 0

                for diar_segment in diarization_data_sorted:
                    diar_start = diar_segment['start']
                    diar_end = diar_segment['end']

                    if word_duration == 0:
                        is_inside = diar_start <= word_start < diar_end
                        overlap_ratio = 1.0 if is_inside else 0.0
                        overlap = 1.0 if is_inside else 0.0
                    else:
                        overlap_start = max(word_start, diar_start)
                        overlap_end = min(word_end, diar_end)
                        overlap = max(0, overlap_end - overlap_start)
                        overlap_ratio = overlap / word_duration

                    if overlap_ratio >= min_overlap_ratio and overlap > best_overlap:
                        best_speaker = diar_segment['speaker']
                        best_overlap = overlap

                word_assignments[i] = best_speaker

            # Второй проход: группировка несопоставленных слов
            if group_unmatched_words:
                speech_segments = []
                current_speaker = None
                segment_start = None

                for i, word in enumerate(words):
                    speaker = word_assignments[i]

                    if speaker != current_speaker:
                        if current_speaker is not None:
                            speech_segments.append({
                                'speaker': current_speaker,
                                'start': segment_start,
                                'end': words[i - 1]['end']
                            })

                        current_speaker = speaker
                        segment_start = word['start']

                if current_speaker is not None:
                    speech_segments.append({
                        'speaker': current_speaker,
                        'start': segment_start,
                        'end': words[-1]['end']
                    })

                for i, word in enumerate(words):
                    if word_assignments[i] is None:
                        word_start = word['start']
                        word_end = word['end']

                        prev_segment = None
                        next_segment = None

                        for segment in speech_segments:
                            if segment['end'] <= word_start:
                                if prev_segment is None or segment['end'] > prev_segment['end']:
                                    prev_segment = segment
                            elif segment['start'] >= word_end:
                                if next_segment is None or segment['start'] < next_segment['start']:
                                    next_segment = segment

                        if prev_segment and next_segment:
                            dist_prev = word_start - prev_segment['end']
                            dist_next = next_segment['start'] - word_end
                            best_segment = prev_segment if dist_prev < dist_next else next_segment
                        elif prev_segment:
                            best_segment = prev_segment
                        elif next_segment:
                            best_segment = next_segment
                        else:
                            best_segment = None

                        if best_segment:
                            word_assignments[i] = best_segment['speaker']

            # Группируем слова по спикерам
            speaker_words = defaultdict(list)
            for i, word in enumerate(words):
                speaker = word_assignments[i] if word_assignments[i] is not None else 'unknown'
                speaker_words[speaker].append(word)

            # Формируем результирующие реплики с сохранением структуры
            for speaker, words_in_segment in speaker_words.items():
                if not words_in_segment:
                    continue

                words_sorted = sorted(words_in_segment, key=lambda x: x['start'])
                replicas = []
                current_replica = []

                for word in words_sorted:
                    if not current_replica:
                        current_replica.append(word)
                    else:
                        last_end = current_replica[-1]['end']
                        if word['start'] - last_end > max_pause:
                            replicas.append(current_replica)
                            current_replica = [word]
                        else:
                            current_replica.append(word)
                if current_replica:
                    replicas.append(current_replica)

                for replica in replicas:
                    # Сохраняем оригинальную структуру с обёрткой "data"
                    result[speaker].append({
                        **segment_metadata,  # Все дополнительные поля из asr_segment
                        'data': {
                            'result': replica,
                            'text': ' '.join(w['word'] for w in replica),
                            # Сохраняем все оригинальные поля из asr_segment['data']
                            **{k: v for k, v in asr_segment['data'].items() if k not in ['result', 'text']}
                        },
                        'start': replica[0]['start'],
                        'end': replica[-1]['end']
                    })

    # Сортировка результатов по времени начала
    for speaker in result:
        result[speaker].sort(key=lambda x: x['start'])

    return dict(result)



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
        filter_cutoff:int = 50,
        filter_order:int = 10,
        diar_vad_sensity: int = 3
        ):
    # Предобработка аудио.
    audio_frames = load_and_preprocess_audio(posted_and_downloaded_audio[file_id])

    # Непосредственно получение временных меток речи
    diar_result = diarizer.diarize_and_merge(
        audio_frames,
        num_speakers=num_speakers,
        filter_cutoff=filter_cutoff,
        filter_order=filter_order,
        vad_sensity=diar_vad_sensity
    )

    for r in diar_result:
        logger.debug(f"Спикер {r['speaker']}: {r['start']:.2f} - {r['end']:.2f} сек")


    # Построение структуры аналогично raw_data для дальнейшего построения диалога и вывод результата
#     return merge_asr_diarisation(asr_raw_data, diar_result)
    return match_asr_with_diarization(asr_raw_data, diar_result)




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

