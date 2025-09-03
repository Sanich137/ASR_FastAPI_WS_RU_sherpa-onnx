from utils.do_logging import logger
from utils.chunk_doing import find_last_speech_position
from utils.globals import posted_and_downloaded_audio, audio_buffer, audio_overlap, audio_duration, audio_to_asr
from models.pipeline_model import ProcessingState
from pydub import AudioSegment
from config import BASE_SAMPLE_RATE, MAX_OVERLAP_DURATION


async def split_audio_handler(data: ProcessingState) -> ProcessingState:
    data.results.success = False
    request_id = data.request_id
    splitted_chank = None
    list_of_audio = list()
    logger.info(f'Получено задание в split_audio_handler')
    audio_to_asr[request_id] = list()
    # Обрабатываем чанки с аудио по N секунд
    for n_channel, mono_data in enumerate(posted_and_downloaded_audio[request_id].split_to_mono()):
        # Подготовительные действия
        list_of_audio.append(list())
        try:
            audio_buffer[request_id] = AudioSegment.silent(1, frame_rate=BASE_SAMPLE_RATE)
            audio_overlap[request_id] = AudioSegment.silent(1, frame_rate=BASE_SAMPLE_RATE)
            audio_duration[request_id] = 0
        except Exception as e:
            error_description = f"Ошибка изменения фреймрейта - {e}"
            logger.error(error_description)
            data.results.error_description = str(error_description)
            return data

        # Формирование структуры для будущего ASR по количеству каналов
        data.results.raw_data.update({f"channel_{n_channel + 1}": list()})

        try:
            # Основной процесс перебора чанков для распознавания
            overlaps = list(mono_data[::MAX_OVERLAP_DURATION * 1000])  # Чанки аудио для распознавания
            total_chunks = len(overlaps)  # Количество чанков, для поиска последнего
            for idx, overlap in enumerate(overlaps):
                is_last_chunk = (idx == total_chunks - 1)  # Если чанк последний
                if (audio_overlap[request_id].duration_seconds + overlap.duration_seconds) < MAX_OVERLAP_DURATION:
                    silent_secs = MAX_OVERLAP_DURATION - (
                                audio_overlap[request_id].duration_seconds + overlap.duration_seconds)
                    overlap += AudioSegment.silent(silent_secs, frame_rate=BASE_SAMPLE_RATE)
                audio_buffer[request_id] = overlap

                # Последний чанк обрабатывается иначе.
                list_of_audio[n_channel].append(await find_last_speech_position(request_id, is_last_chunk))

        except Exception as e:
            error_description = f"При разделении аудио на чанки: {e}"
            logger.error(error_description)
            data.results.error_description = str(error_description)
            return data

        else:
            audio_to_asr[request_id].clear()
            data.results.success = True

    audio_to_asr[request_id] = list_of_audio

    logger.info(f'Возвращено задание из split_audio_handler')

    del mono_data
    del overlaps
    del list_of_audio

    return data
