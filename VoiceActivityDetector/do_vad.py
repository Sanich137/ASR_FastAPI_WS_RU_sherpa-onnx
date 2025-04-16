import numpy as np
import onnxruntime as ort
from pydub import AudioSegment
from pathlib import Path
# from utils.do_logging import logger
import logging as logger

class SileroVAD:
    def __init__(self, onnx_path: Path, sample_rate: int = 16000, use_gpu = False):

        if use_gpu:
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # Выключаем подробный лог
        session_options.inter_op_num_threads = 2
        session_options.intra_op_num_threads = 2

        self.session = ort.InferenceSession(path_or_bytes=onnx_path,
                                            sess_options=session_options,
                                            providers=providers
                                            )
        self.sample_rate = sample_rate

        # Инициализация состояния
        self.state = np.zeros((2, 1, 128), dtype=np.float32)  # [2, 1, 128]

        # Стандартный размер фрейма для Silero VAD v5
        self.frame_size = 512  # 31ms при 16000 Hz

        self.prob_level = 0.5

    def reset_state(self):
        """Сброс состояния к начальному"""
        self.state = np.zeros((2, 1, 128), dtype=np.float32)

    def set_mode(self, mode: int):
        """
        Higher level stets higher sensitivity
        :param mode: Int from 1 to 3.
        """
        if mode not in [1, 2, 3]:
            self.prob_level = 0.5
        elif mode == 1:
            self.prob_level = 0.85
        elif mode == 2:
            self.prob_level = 0.5
        elif mode == 3:
            self.prob_level = 0.15

    def is_speech(self, audio_frame: np.ndarray, sample_rate = None) -> bool:
        """Обработка аудио-фрейма (миничанка)
        Args:
            :param audio_frame: 1D numpy array размером 512 сэмплов
            :param sample_rate:
        Returns:
            bool: is_speech or not not speech
        """

        if len(audio_frame) != self.frame_size:
            raise ValueError(f"Ожидается фрейм размером {self.frame_size}, получен {len(audio_frame)}")

        inputs = {
            'input': audio_frame.reshape(1, -1).astype(np.float32),  # [1, 512]
            'state': self.state,
            'sr': np.array(self.sample_rate, dtype=np.int64)  # scalar int64
        }

        outputs = self.session.run(['output', 'stateN'], inputs)

        self.state = outputs[1]  # Обновляем состояние
        prob = float(outputs[0][0, 0])

        logger.debug(f"probs = {prob}")
        if prob >= self.prob_level:
            logger.debug("Найден голос")
            return True, self.state
        else:
            return False, self.state

    def get_speech_segments(self, audio_frames: np.ndarray, min_duration: float = 0.3, max_gap: float = 0.2) -> list[
        tuple[float, float, np.ndarray]]:
        """Получение сегментов речи с сглаживанием
        Args:
            audio_frames: np.ndarray [N, 512] - массив фреймов
            min_duration: минимальная длительность сегмента в секундах
            max_gap: максимальный разрыв между сегментами для объединения в секундах
        Returns:
            list[tuple[float, float, np.ndarray]]: список (start_time, end_time, audio_segment)
        """
        segments = []
        start_frame = None
        current_segment = []

        frame_duration = self.frame_size / self.sample_rate  # Длительность одного фрейма в секундах
        min_frames = int(min_duration / frame_duration)  # Минимальное число фреймов в сегменте
        max_gap_frames = int(max_gap / frame_duration)  # Максимальный разрыв в фреймах

        self.reset_state()

        for i, frame in enumerate(audio_frames):
            is_speech, self.state = self.is_speech(frame)

            if is_speech and start_frame is None:
                start_frame = i
                current_segment = [frame]
            elif is_speech and start_frame is not None:
                current_segment.append(frame)
            elif not is_speech and start_frame is not None:
                # Проверяем, не слишком ли короткий разрыв
                if i < len(audio_frames) - 1:
                    next_speech = False
                    for j in range(1, min(max_gap_frames + 1, len(audio_frames) - i)):
                        next_is_speech, _ = self.is_speech(audio_frames[i + j])
                        if next_is_speech:
                            next_speech = True
                            break
                    if next_speech:
                        current_segment.append(frame)  # Добавляем паузу в сегмент
                        continue

                # Завершаем сегмент
                end_frame = i - 1
                segment_frames = end_frame - start_frame + 1
                if segment_frames >= min_frames:  # Фильтруем короткие сегменты
                    start_time = start_frame * frame_duration
                    end_time = (end_frame + 1) * frame_duration
                    segment_audio = np.concatenate(current_segment, axis=0)
                    segments.append((start_time, end_time, segment_audio))
                start_frame = None
                current_segment = []

        # Завершаем последний сегмент
        if start_frame is not None:
            end_frame = len(audio_frames) - 1
            segment_frames = end_frame - start_frame + 1
            if segment_frames >= min_frames:
                start_time = start_frame * frame_duration
                end_time = (end_frame + 1) * frame_duration
                segment_audio = np.concatenate(current_segment, axis=0)
                segments.append((start_time, end_time, segment_audio))

        return segments


def load_and_preprocess_audio(file_path: str, target_frame_size: int = 512) -> np.ndarray:

    """Загрузка аудио и подготовка фреймов для обработки файла целиком"""

    audio = AudioSegment.from_file(file_path)

    # Конвертируем в моно 16kHz если нужно
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    if audio.channels > 1:
        audio = audio.split_to_mono()[0]

    # Нормализация в float32
    samples = np.frombuffer(audio.raw_data, dtype=np.int16)
    samples_float32 = samples.astype(np.float32) / 32768.0

    # Разбиваем на фреймы нужного размера
    num_frames = len(samples_float32) // target_frame_size
    frames = []
    for i in range(num_frames):
        start = i * target_frame_size
        end = start + target_frame_size
        frames.append(samples_float32[start:end])

    return np.array(frames)


if __name__ == "__main__":
    from datetime import datetime as dt

    # Инициализация VAD
    print("Инициализация VAD...")
    vad = SileroVAD(Path("../models/VAD_silero_v5/silero_vad.onnx"), use_gpu=False)
    vad.set_mode(3)
    # Загрузка и подготовка аудио
    audio_file = Path("C:/Users/kojevnikov/PycharmProjects/Sherpa_onnx_vosk_GPU/trash/q.Wav")
    audio_frames = load_and_preprocess_audio(audio_file)

    print(f"\nЗагружено {len(audio_frames)} фреймов по {vad.frame_size} сэмплов")

    time_start = dt.now()
    # Обработка каждого фрейма
    # # -- скорость тестирование
    # for i, frame in enumerate(audio_frames):
    #     vad.is_speech(frame)
    #     if i>10:
    #         break

    # Получение фрагментов аудио
    audio_segments = vad.get_speech_segments(audio_frames, min_duration=0.3, max_gap=0.4)

    for start, end, audio in audio_segments:
        print(f"Сегмент: {start:.2f}s - {end:.2f}s, длина аудио: {len(audio)} сэмплов")

    print(f"Время выполнения {(dt.now() - time_start).total_seconds()}")