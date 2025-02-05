import numpy as np
import webrtcvad
from utils.do_logging import logger
from utils.pre_start_init import (audio_overlap,
                                  audio_buffer,
                                  audio_to_asr,
                                  buffer_duration,
                                  audio_duration,
                                  MAX_OVERLAP_DURATION)

from pydub import AudioSegment

def find_last_speech_position(socket_id, sample_width = 2):
    """
        1. Берём собранное аудио, добавляем в начало overlap
        2. Конвертируем его в np.int16
        2. Находим позицию последнего сегмента тишины перед речью в аудио.
        3. Всё до этой позиции отправляем на распознавание
        4. Остаток, хвост, складываем отдельно как overlap
        5. Если не находит ни одного сегмента без речи, помечаем его как полностью речь и отдаём ра распознавание
    """
    vad = webrtcvad.Vad()
    vad.set_mode(1)
    frame_rate = audio_buffer[socket_id].frame_rate
    logger.debug(f"Получено из буфера на обработку аудио продолжительностью {audio_buffer[socket_id].duration_seconds} ")

    audio = np.frombuffer(audio_buffer[socket_id].raw_data, dtype=np.int16)

    # Преобразование в int16
    if audio.dtype != np.int16:
        audio = (audio * 32767).astype(np.int16)

    speech_end = len(audio)
    # Разделение на фрагменты
    frame_duration_ms = 20
    frame_length = int(frame_rate * frame_duration_ms / 1000)
    partial_frame_length = int()
    frames = [audio[i:i + frame_length] for i in range(0, len(audio), frame_length)]

    # Проверка каждого фрагмента на наличие голоса.
    for i, frame in enumerate(reversed(frames)):
        try:
            if len(frame) < frame_length:
                # Пропустить последний неполный фрагмент
                # Todo Похоже, его всегда нужно добавлять в audio_overlap
                partial_frame_length = len(frame)
                continue
            else:
                if not vad.is_speech(frame.tobytes(), sample_rate=frame_rate):
                    logger.debug(f"Найден не голос на speech_end = {speech_end-(i+1)*frame_length-partial_frame_length}")
                    is_full = False
                    break
                else:
                    # logger.debug(f"Найден ГОЛОС на speech_end = {speech_end-i*frame_length-partial_frame_length}")
                    continue
        except Exception as e:
            logger.error(f"Ошибка VAD - {e}")

    # speech_end - длина аудио фрагмента в каких единицах измерения?
    if not partial_frame_length:
        speech_end = len(audio) - (
                    i + 1) * frame_length  # Общая продолжительность аудио минус длинна Фрейма х количество
    else:
        speech_end = len(audio) - i * frame_length
        # фреймов с голосом

    separation_time = speech_end * 1000 / frame_rate

    audio_to_asr[socket_id] = audio_overlap[socket_id] + audio_buffer[socket_id][:separation_time]

    audio_overlap[socket_id] = audio_buffer[socket_id][separation_time:audio_buffer[socket_id].duration_seconds * 1000]

    logger.debug(f"Передано на ASR аудио продолжительностью {audio_to_asr[socket_id].duration_seconds} ")

    logger.debug(f"Передано в перекрытие аудио продолжительностью {audio_overlap[socket_id].duration_seconds} ")

    audio_buffer[socket_id] = AudioSegment.silent(100, frame_rate)

    return
