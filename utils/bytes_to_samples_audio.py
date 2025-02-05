import numpy as np


def get_np_array_samples_float32(audio_bytes: bytes, sample_width: int = 2) -> np.ndarray:
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
    samples_float32 = samples.astype(np.float32)
    samples_float32 = samples_float32 / 32768

    return samples_float32

def get_np_array_samples_int16(audio_bytes: bytes, sample_width: int = 2) -> np.ndarray:
    """
    Преобразует аудио в байтах в массив float32.
    :param audio_bytes: Аудиоданные в байтах.
    :param sample_width: Размер одного сэмпла в байтах (обычно 2 для 16-битного аудио).
    :return: Массив numpy с данными в формате int16.
    """

    # Определяем тип данных на основе sample_width
    dtype = np.int16 if sample_width == 2 else np.int32

    # Преобразуем байты в массив numpy
    np_samples = np.frombuffer(audio_bytes, dtype=dtype)

   # Преобразование в int16
    if np_samples.dtype != np.int16:
        np_samples = (np_samples * 32767).astype(np.int16)

    return np_samples