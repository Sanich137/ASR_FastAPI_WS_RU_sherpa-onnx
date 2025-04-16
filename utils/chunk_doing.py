from utils.do_logging import logger
from utils.bytes_to_samples_audio import get_np_array_samples_float32

from utils.pre_start_init import (audio_overlap,
                                  audio_buffer,
                                  audio_to_asr,
                                  )
from pydub import AudioSegment
from VoiceActivityDetector import vad

def find_last_speech_position(socket_id, sample_width = 2):
    """
        1. Берём собранное аудио, добавляем в начало overlap
        2. Конвертируем его в np.int16
        2. Находим позицию последнего сегмента тишины перед речью в аудио.
        3. Всё до этой позиции отправляем на распознавание
        4. Остаток, хвост, складываем отдельно как overlap
        5. Если не находит ни одного сегмента без речи, помечаем его как полностью речь и отдаём ра распознавание
    """
    frame_rate = audio_buffer[socket_id].frame_rate

    # 16000 - битрейт, требуемый силеро VAD.
    silero_bitrate = 16000
    if audio_buffer[socket_id].frame_rate != silero_bitrate:
        audio_for_vad = audio_buffer[socket_id].set_frame_rate(silero_bitrate)
    else:
        audio_for_vad = audio_buffer[socket_id]

    logger.debug(f"Получено из буфера на обработку аудио продолжительностью {audio_buffer[socket_id].duration_seconds} ")

    # Переводим в float32 для vad
    audio = get_np_array_samples_float32(audio_for_vad.raw_data)

    # Входные данные для деления фреймов
    speech_end = len(audio)
    min_silence_frames = 15
    frame_length = 512
    partial_frame_length = int()

    # Разделение на фрагменты
    frames = [audio[i:i + frame_length] for i in range(int(len(audio)//3), len(audio), frame_length)]

    # Проверка каждого фрагмента на наличие голоса.
    silence_frames = 0
    # Ниже переприсвоение стейтов - попытка сохранить стейт, если одновременно поступит несколько запросов.
    vad.reset_state()
    vad_state=vad.state
    for i, frame in enumerate(reversed(frames)):
        vad.state = vad_state
        try:
            if len(frame) < frame_length:
                # Пропустить последний неполный фрагмент
                partial_frame_length = len(frame)
                continue
            else:
                is_speech, vad_state = vad.is_speech(frame, sample_rate=frame_rate)
                if not is_speech:
                    # logger.debug(f"Найден не голос на speech_end = {speech_end-(i+1)*frame_length-partial_frame_length}")
                    silence_frames+=1
                    if silence_frames >= min_silence_frames:
                        break
                else:
                    silence_frames = 0
                    logger.debug(f"Найден ГОЛОС на speech_end = {speech_end-i*frame_length-partial_frame_length}")
                    continue
        except Exception as e:
            logger.error(f"Ошибка VAD - {e}"
                         f"\nframe_rate = {frame_rate}"
                         f"\nframe_length = {frame_length}")

    # speech_end - длина аудио фрагмента в каких единицах измерения?
    if not partial_frame_length:
        # Общая продолжительность аудио минус длинна Фрейма х количество
        speech_end = len(audio) - (i + 1) * frame_length
    else:
        speech_end = len(audio) - i * frame_length
        # фреймов с голосом

    separation_time = int(speech_end * 1000 / silero_bitrate)

    audio_to_asr[socket_id] = audio_overlap[socket_id] + audio_buffer[socket_id][:separation_time]

    audio_overlap[socket_id] = audio_buffer[socket_id][separation_time:]

    logger.debug(f"Передано на ASR аудио продолжительностью {audio_to_asr[socket_id].duration_seconds} ")

    logger.debug(f"Передано в перекрытие аудио продолжительностью {audio_overlap[socket_id].duration_seconds} ")

    audio_buffer[socket_id] = AudioSegment.silent(1, frame_rate)

    return
