import numpy as np
import onnxruntime as ort
from pydub import AudioSegment
from pathlib import Path
# from utils.do_logging import logger
import logging as logger

class SileroVAD:
    def __init__(self, onnx_path: Path, sample_rate: int = 16000, use_gpu=False):
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        session_options.inter_op_num_threads = 4
        session_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(path_or_bytes=onnx_path,
                                            sess_options=session_options,
                                            providers=providers)
        self.sample_rate = sample_rate
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.frame_size = 512
        self.prob_level = 0.5
        self.set_mode(3)

    def reset_state(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)

    def set_mode(self, mode: int):
        if mode not in [1, 2, 3, 4, 5]:
            self.prob_level = 0.5
        elif mode == 1:
            self.prob_level = 0.7
        elif mode == 2:
            self.prob_level = 0.6
        elif mode == 3:
            self.prob_level = 0.5
        elif mode == 4:
            self.prob_level = 0.4
        elif mode == 5:
            self.prob_level = 0.3

    def is_speech(self, audio_frame: np.ndarray, sample_rate = None) -> bool:
        """Обработка аудио-фрейма (миничанка)
        Args:
            :param audio_frame: 1D numpy array размером 512 сэмплов
            :param sample_rate:
        Returns:
            bool: is_speech or not not speech
        """

        if len(audio_frame) != self.frame_size:
            audio_frame = np.pad(audio_frame, (0, self.frame_size - len(audio_frame)), mode='constant')[:self.frame_size]

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

    def get_speech_segments(self, audio_frames: np.ndarray, min_duration: float, max_gap: float) -> list[tuple]:
        """Получение сегментов речи с сглаживанием
        Args:
            audio_frames: np.ndarray [N, 512] - массив фреймов
            min_duration: минимальная длительность сегмента в секундах
            max_gap: максимальный разрыв между сегментами для объединения в секундах
        Returns:
            list[tuple[float, float, np.ndarray]]: список (start_time, end_time, audio_segment)
        """
        if len(audio_frames.shape) == 2:
            audio_frames = audio_frames.flatten()

        audio_length_samples = len(audio_frames)
        window_size_samples = self.frame_size
        sample_rate = self.sample_rate

        # Параметры из get_speech_timestamps
        threshold = self.prob_level
        neg_threshold = threshold - 0.15
        min_speech_duration_ms = int(min_duration * 1000)
        min_silence_duration_ms = int(max_gap * 1000)
        speech_pad_ms = 30
        min_speech_samples = sample_rate * min_speech_duration_ms // 1000
        min_silence_samples = sample_rate * min_silence_duration_ms // 1000
        speech_pad_samples = sample_rate * speech_pad_ms // 1000

        # Получение вероятностей речи
        self.reset_state()
        speech_probs = []
        for current_start in range(0, audio_length_samples, window_size_samples):
            chunk = audio_frames[current_start:current_start + window_size_samples]
            if len(chunk) < window_size_samples:
                chunk = np.pad(chunk, (0, window_size_samples - len(chunk)), mode='constant')
            prob, new_state = self.is_speech(chunk)
            self.state = new_state
            speech_probs.append(prob)

        # Обработка вероятностей для выделения сегментов
        triggered = False
        speeches = []
        current_speech = {}
        temp_end = 0
        prev_end = 0
        next_start = 0

        for i, speech_prob in enumerate(speech_probs):
            current_sample = window_size_samples * i
            if speech_prob >= threshold and temp_end:
                temp_end = 0
                if next_start < prev_end:
                    next_start = current_sample

            if speech_prob >= threshold and not triggered:
                triggered = True
                current_speech['start'] = current_sample
                continue

            if speech_prob < neg_threshold and triggered:
                if not temp_end:
                    temp_end = current_sample
                if (current_sample - temp_end) > (sample_rate * 98 // 1000):
                    prev_end = temp_end
                if (current_sample - temp_end) < min_silence_samples:
                    continue
                else:
                    current_speech['end'] = temp_end
                    if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                        speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    continue

        if current_speech and (audio_length_samples - current_speech.get('start', 0)) > min_speech_samples:
            current_speech['end'] = audio_length_samples
            speeches.append(current_speech)

        # Применение speech_pad и корректировка границ
        for i, speech in enumerate(speeches):
            if i == 0:
                speech['start'] = max(0, speech['start'] - speech_pad_samples)
            if i != len(speeches) - 1:
                silence_duration = speeches[i + 1]['start'] - speech['end']
                if silence_duration < 2 * speech_pad_samples:
                    speech['end'] += silence_duration // 2
                    speeches[i + 1]['start'] = max(0, speeches[i + 1]['start'] - silence_duration // 2)
                else:
                    speech['end'] = min(audio_length_samples, speech['end'] + speech_pad_samples)
                    speeches[i + 1]['start'] = max(0, speeches[i + 1]['start'] - speech_pad_samples)
            else:
                speech['end'] = min(audio_length_samples, speech['end'] + speech_pad_samples)

        # Формирование результата
        segments = []
        for speech in speeches:
            start = speech['start'] / sample_rate
            end = speech['end'] / sample_rate
            audio_segment = audio_frames[int(start * sample_rate):int(end * sample_rate)]
            segments.append((start, end, audio_segment))

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