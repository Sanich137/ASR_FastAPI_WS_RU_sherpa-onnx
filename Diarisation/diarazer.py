import config
from Diarisation.do_diarize import load_and_preprocess_audio
from utils.pre_start_init import posted_and_downloaded_audio
from utils.do_logging import logger


from collections import defaultdict

if config.CAN_DIAR:
    from Diarisation import diarizer

async def do_diarizing(
        file_id:str,
        asr_raw_data,
        num_speakers:int = -1,
        filter_cutoff:int = 30,
        filter_order:int = 10,
        ):
    # Предобработка аудио.
    # ВАЖНЫЙ момент. Здесь мы диаризируем только последний канал.
    samples_float32 = await load_and_preprocess_audio(audio=posted_and_downloaded_audio[file_id],
                                                   sample_rate=16000, filter_cutoff=filter_cutoff,
                                                   filter_order=filter_order)

    # Непосредственно получение временных меток речи
    diar_result = await diarizer.diarize(
        audio=samples_float32,
        asr_data=asr_raw_data["channel_1"],
        num_speakers=num_speakers,
        tau=10)

    for r in diar_result:
        logger.debug(f"Спикер {r['speaker']}: {r['start']:.2f} - {r['end']:.2f} сек")


    # Построение структуры аналогично raw_data для дальнейшего построения диалога и вывод результата
    m_asr = match_asr_with_diarization(asr_raw_data, diar_result)
    return m_asr

def match_asr_with_diarization(asr_data, diarization_data, min_overlap_ratio=0.5, max_pause=2.0, group_unmatched_words=True):
    result = defaultdict(list)
    diarization_data_sorted = sorted(diarization_data, key=lambda x: x['start'])

    for channel, asr_segments in asr_data.items():
        for asr_segment in asr_segments:
            segment_metadata = {k: v for k, v in asr_segment.items() if k != 'data'}
            words = asr_segment['data']['result']
            word_assignments = {}

            # Сопоставление слов с диаризацией
            for i, word in enumerate(words):
                word_start, word_end = word['start'], word['end']
                word_duration = word_end - word_start or 0.0001
                best_speaker, max_overlap = None, 0

                for diar_segment in diarization_data_sorted:
                    diar_start, diar_end = diar_segment['start'], diar_segment['end']
                    overlap_start = max(word_start, diar_start)
                    overlap_end = min(word_end, diar_end)
                    overlap = max(0, overlap_end - overlap_start)
                    overlap_ratio = overlap / word_duration

                    if word_duration <= 0.0001:
                        is_inside = diar_start <= word_start <= diar_end
                        overlap_ratio = 1.0 if is_inside else 0.0
                        overlap = 1.0 if is_inside else 0.0

                    if overlap_ratio >= min_overlap_ratio and overlap > max_overlap:
                        best_speaker = str(diar_segment['speaker'])  # Преобразуем спикера в строку
                        max_overlap = overlap

                word_assignments[i] = best_speaker

            # Группировка несопоставленных слов
            if group_unmatched_words:
                for i, speaker in enumerate(word_assignments.values()):
                    if speaker is None:
                        word_start, word_end = words[i]['start'], words[i]['end']
                        min_dist = float('inf')
                        best_speaker = None

                        for diar_segment in diarization_data_sorted:
                            dist_to_start = abs(word_start - diar_segment['start'])
                            dist_to_end = abs(word_start - diar_segment['end'])
                            dist = min(dist_to_start, dist_to_end)
                            if dist < min_dist and dist <= max_pause:
                                min_dist = dist
                                best_speaker = str(diar_segment['speaker'])  # Преобразуем в строку

                        word_assignments[i] = best_speaker if best_speaker is not None else "unknown"

            # Группировка слов по спикерам и формирование реплик
            speaker_words = defaultdict(list)
            for i, word in enumerate(words):
                speaker = word_assignments[i] if word_assignments[i] is not None else "unknown"
                speaker_words[speaker].append(word)

            for speaker, words_in_segment in speaker_words.items():
                if not words_in_segment:
                    continue

                words_sorted = sorted(words_in_segment, key=lambda x: x['start'])
                replicas = []
                current_replica = [words_sorted[0]]

                for word in words_sorted[1:]:
                    if word['start'] - current_replica[-1]['end'] <= max_pause:
                        current_replica.append(word)
                    else:
                        replicas.append(current_replica)
                        current_replica = [word]
                if current_replica:
                    replicas.append(current_replica)

                for replica in replicas:
                    replica_text = ' '.join(w['word'] for w in replica)
                    result[speaker].append({
                        **segment_metadata,
                        'data': {
                            'result': replica,
                            'text': replica_text,
                            **{k: v for k, v in asr_segment['data'].items() if k not in ['result', 'text']},
                            'replica_start': replica[0]['start'],
                            'replica_end': replica[-1]['end']
                        }
                    })

    for speaker in result:
        result[speaker].sort(key=lambda x: x['data']['replica_start'])

    return dict(result)


# if __name__ == "__main__":
#     asr_raw_data = {'channel_1': [{'data': {'result': [{'conf': 1.0, 'end': 3.0, 'start': 2.4, 'word': 'здравствуйте'}, {'conf': 1.0, 'end': 3.72, 'start': 3.68, 'word': 'вы'}, {'conf': 1.0, 'end': 4.28, 'start': 3.84, 'word': 'позвонили'}, {'conf': 1.0, 'end': 4.36, 'start': 4.36, 'word': 'в'}, {'conf': 1.0, 'end': 4.96, 'start': 4.52, 'word': 'амулекс'}, {'conf': 1.0, 'end': 5.36, 'start': 5.12, 'word': 'точка'}, {'conf': 1.0, 'end': 5.56, 'start': 5.48, 'word': 'ру'}, {'conf': 1.0, 'end': 6.2, 'start': 5.96, 'word': 'ваша'}, {'conf': 1.0, 'end': 6.68, 'start': 6.32, 'word': 'команда'}, {'conf': 1.0, 'end': 7.24, 'start': 6.8, 'word': 'юристов'}, {'conf': 1.0, 'end': 7.8, 'start': 7.8, 'word': 'в'}, {'conf': 1.0, 'end': 8.16, 'start': 7.92, 'word': 'целях'}, {'conf': 1.0, 'end': 8.76, 'start': 8.32, 'word': 'повышения'}, {'conf': 1.0, 'end': 9.24, 'start': 8.88, 'word': 'качества'}, {'conf': 1.0, 'end': 10.0, 'start': 9.4, 'word': 'обслуживания'}, {'conf': 1.0, 'end': 10.24, 'start': 10.12, 'word': 'все'}, {'conf': 1.0, 'end': 10.84, 'start': 10.4, 'word': 'разговоры'}, {'conf': 1.0, 'end': 11.68, 'start': 11.0, 'word': 'записываются'}], 'text': 'здравствуйте вы позвонили в амулекс точка ру ваша команда юристов в целях повышения качества обслуживания все разговоры записываются'}}, {'data': {'result': [{'conf': 1.0, 'end': 20.88, 'start': 20.36, 'word': 'амлекс'}, {'conf': 1.0, 'end': 21.2, 'start': 21.0, 'word': 'точка'}, {'conf': 1.0, 'end': 21.36, 'start': 21.32, 'word': 'ру'}, {'conf': 1.0, 'end': 22.0, 'start': 21.52, 'word': 'екатерина'}, {'conf': 1.0, 'end': 22.72, 'start': 22.12, 'word': 'здравствуйте'}], 'text': 'амлекс точка ру екатерина здравствуйте'}}, {'data': {'result': [{'conf': 1.0, 'end': 34.24, 'start': 33.72, 'word': 'аскажите'}, {'conf': 1.0, 'end': 34.8, 'start': 34.36, 'word': 'пожалуйста'}, {'conf': 1.0, 'end': 35.0, 'start': 34.92, 'word': 'как'}, {'conf': 1.0, 'end': 35.36, 'start': 35.16, 'word': 'можно'}, {'conf': 1.0, 'end': 35.44, 'start': 35.44, 'word': 'к'}, {'conf': 1.0, 'end': 35.68, 'start': 35.6, 'word': 'вам'}, {'conf': 1.0, 'end': 36.32, 'start': 35.84, 'word': 'обращаться'}, {'conf': 1.0, 'end': 38.84, 'start': 38.84, 'word': 'а'}, {'conf': 1.0, 'end': 39.48, 'start': 39.0, 'word': 'полностью'}, {'conf': 1.0, 'end': 39.88, 'start': 39.68, 'word': 'можно'}, {'conf': 1.0, 'end': 42.56, 'start': 42.2, 'word': 'хорошо'}, {'conf': 1.0, 'end': 42.96, 'start': 42.72, 'word': 'денис'}, {'conf': 1.0, 'end': 43.72, 'start': 43.08, 'word': 'владимирович'}], 'text': 'аскажите пожалуйста как можно к вам обращаться а полностью можно хорошо денис владимирович'}}, {'data': {'result': [{'conf': 1.0, 'end': 44.976, 'start': 44.656, 'word': 'сейчас'}, {'conf': 1.0, 'end': 45.456, 'start': 45.176, 'word': 'минуту'}, {'conf': 1.0, 'end': 46.056, 'start': 45.616, 'word': 'пожалуйста'}, {'conf': 1.0, 'end': 55.496, 'start': 54.976, 'word': 'оставайтесь'}, {'conf': 1.0, 'end': 55.656, 'start': 55.616, 'word': 'на'}, {'conf': 1.0, 'end': 56.096, 'start': 55.816, 'word': 'линии'}, {'conf': 1.0, 'end': 56.256, 'start': 56.256, 'word': 'я'}, {'conf': 1.0, 'end': 56.416, 'start': 56.336, 'word': 'вас'}, {'conf': 1.0, 'end': 56.896, 'start': 56.576, 'word': 'соединю'}, {'conf': 1.0, 'end': 57.056, 'start': 57.056, 'word': 'с'}, {'conf': 1.0, 'end': 57.696, 'start': 57.176, 'word': 'юридическим'}, {'conf': 1.0, 'end': 58.376, 'start': 57.816, 'word': 'департаментом'}, {'conf': 1.0, 'end': 58.616, 'start': 58.576, 'word': 'не'}, {'conf': 1.0, 'end': 59.416, 'start': 58.776, 'word': 'отключайтесь'}], 'text': 'сейчас минуту пожалуйста оставайтесь на линии я вас соединю с юридическим департаментом не отключайтесь'}}]}
#     diar_result = [{'end': 11.862, 'speaker': '0', 'start': 2.53}, {'end': 21.14, 'speaker': '1', 'start': 20.77}, {'end': 45.368, 'speaker': '1', 'start': 34.466}, {'end': 59.96, 'speaker': '1', 'start': 55.01}]
#     print(match_asr_with_diarization(
#         asr_raw_data,
#         diar_result,
#         min_overlap_ratio=0.5,
#         max_pause=1.5,
#         group_unmatched_words=True  # Включить группировку несопоставленных слов
#     ) )