import numpy as np
import onnxruntime as ort
from pydub import AudioSegment
from pathlib import Path


class SileroVAD:
    def __init__(self, onnx_path: str, sr: int = 16000):
        self.session = ort.InferenceSession(onnx_path)
        self.sr = sr

        # Инициализация состояния согласно спецификации модели
        self.state = np.zeros((2, 1, 128), dtype=np.float32)  # [2, 1, 128]

        # Стандартный размер фрейма для Silero VAD v5
        self.frame_size = 512  # 31ms при 16000 Hz

    def reset_state(self):
        """Сброс состояния к начальному"""
        self.state = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """Обработка аудио-фрейма
        Args:
            audio_frame: 1D numpy array размером 512 сэмплов
        Returns:
            tuple: (speech_probability, new_state)
        """
        if len(audio_frame) != self.frame_size:
            raise ValueError(f"Ожидается фрейм размером {self.frame_size}, получен {len(audio_frame)}")

        inputs = {
            'input': audio_frame.reshape(1, -1).astype(np.float32),  # [1, 512]
            'state': self.state,
            'sr': np.array(self.sr, dtype=np.int64)
        }

        outputs = self.session.run(['output', 'stateN'], inputs)

        self.state = outputs[1]  # Обновляем состояние

        prob = float(outputs[0][0, 0])

        if prob >= 0.5:
            return True
        else:
            return False

    def __call__(self, audio_frame: np.ndarray) -> tuple:
        """Обработка аудио-фрейма
        Args:
            audio_frame: 1D numpy array размером 512 сэмплов
        Returns:
            tuple: (speech_probability, new_state)
        """
        if len(audio_frame) != self.frame_size:
            raise ValueError(f"Ожидается фрейм размером {self.frame_size}, получен {len(audio_frame)}")

        inputs = {
            'input': audio_frame.reshape(1, -1).astype(np.float32),  # [1, 512]
            'state': self.state,
            'sr': np.array(self.sr, dtype=np.int64)
        }

        outputs = self.session.run(['output', 'stateN'], inputs)

        self.state = outputs[1]  # Обновляем состояние

        return float(outputs[0][0, 0]), outputs[1]


def load_and_preprocess_audio(file_path: str, target_frame_size: int = 512) -> np.ndarray:
    """Загрузка аудио и подготовка фреймов"""
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
    # Инициализация VAD
    print("Инициализация VAD...")
    vad = SileroVAD("silero_v5__vad_orig.onnx")

    # Загрузка и подготовка аудио
    audio_file = Path("C:/Users/kojevnikov/PycharmProjects/Sherpa_onnx_vosk_GPU/trash/q.Wav")
    audio_frames = load_and_preprocess_audio(audio_file)

    print(f"\nЗагружено {len(audio_frames)} фреймов по {vad.frame_size} сэмплов")

    # Обработка каждого фрейма
    for i, frame in enumerate(audio_frames):
        prob, _ = vad(frame)
        if prob >= 0.5:
            print(f"Найден голос на {i * vad.frame_size / 16000}")
        else:
            print(f"Не голос на {i * vad.frame_size / 16000}")

        # print(f"Фрейм {i + 1}: Вероятность речи = {prob:.4f}")