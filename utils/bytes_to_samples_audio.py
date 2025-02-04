import numpy as np


def get_np_array(audio_bytes: bytes, sample_width: int = 2) -> np.ndarray:
    """
    Преобразует аудио в байтах в массив float32.
    :param audio_bytes: Аудиоданные в байтах.
    :param sample_width: Размер одного сэмпла в байтах (обычно 2 для 16-битного аудио).
    :return: Массив numpy с данными в формате float32.
    """
    # Определяем тип данных на основе sample_width
    dtype = np.int16 if sample_width == 2 else np.int32

    # Преобразуем байты в массив numpy
    samples = np.frombuffer(audio_bytes, dtype=dtype)

    # Нормализуем данные до диапазона [-1.0, 1.0]
    # samples_float32 = samples.astype(np.float32) / (2 ** (8 * sample_width - 1))
    # samples = np.frombuffer(samples, dtype=np.int16)
    samples_float32 = samples.astype(np.float32)
    samples_float32 = samples_float32 / 32768

    return samples_float32