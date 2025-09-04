from utils.do_logging import logger
from utils.chunk_doing import find_last_speech_position

from models.pipeline_model import ProcessingState
from pydub import AudioSegment
from config import BASE_SAMPLE_RATE, MAX_OVERLAP_DURATION


async def split_audio_handler(data: ProcessingState) -> ProcessingState:
    data.results.success = False

    posted_and_downloaded_audio = data.stage_results.posted_and_downloaded_audio

    list_of_audio_to_asr = list()
    logger.debug(f'Получено задание в split_audio_handler')

    # Обрабатываем чанки с аудио по N секунд
    for n_channel, mono_data in enumerate(posted_and_downloaded_audio.split_to_mono()):
        # Подготовительные действия
        list_of_audio_to_asr.append(list())
        try:
            audio_overlap = AudioSegment.silent(1, frame_rate=BASE_SAMPLE_RATE) # Это "хвост" аудио после VAD
            audio_duration = 0
        except Exception as e:
            error_description = f"Ошибка изменения фреймрейта - {e}"
            logger.error(error_description)
            data.results.error_description = str(error_description)
            return data

        # Формирование структуры для будущего ASR по количеству каналов
        data.results.raw_data.update({f"channel_{n_channel + 1}": list()})

        try:
            # Основной процесс перебора чанков для распознавания
            audio_chunks = list(mono_data[::MAX_OVERLAP_DURATION * 1000])  # Чанки аудио для распознавания
            total_chunks = len(audio_chunks)  # Количество чанков, для поиска последнего
            for idx, audio_chunk in enumerate(audio_chunks):
                is_last_chunk = (idx == total_chunks - 1)  # Если чанк последний
                if (audio_overlap.duration_seconds + audio_chunk.duration_seconds) < MAX_OVERLAP_DURATION:
                    silent_secs = MAX_OVERLAP_DURATION - (
                                audio_overlap.duration_seconds + audio_chunk.duration_seconds)
                    audio_chunk += AudioSegment.silent(silent_secs, frame_rate=BASE_SAMPLE_RATE)

                # Последний чанк обрабатывается иначе.
                audio_to_asr, audio_overlap = await find_last_speech_position(audio_chunk, audio_overlap, is_last_chunk)
                list_of_audio_to_asr[n_channel].append(audio_to_asr)

        except Exception as e:
            error_description = f"При разделении аудио на чанки: {e}"
            logger.error(error_description)
            data.results.error_description = str(error_description)
            return data

        else:
            del audio_to_asr
            data.results.success = True

    data.stage_results.audio_to_asr =  list_of_audio_to_asr

    logger.debug(f'Возвращено задание из split_audio_handler')

    del mono_data
    del audio_chunks
    del list_of_audio_to_asr

    return data
