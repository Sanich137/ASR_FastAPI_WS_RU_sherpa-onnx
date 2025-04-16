import onnxruntime as ort
import numpy as np
from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity
from librosa.feature import melspectrogram
from sklearn.cluster import AgglomerativeClustering
import librosa
from pathlib import Path
import noisereduce as nr
import hdbscan

# from utils.do_logging import logger  # Предполагаю, что у тебя есть логгер
import logging as logger


class SpeakerDiarizer:
    def __init__(self, vad_path: Path, speaker_model_path: Path, sample_rate: int = 16000, use_gpu: bool = True):
        self.vad = SileroVAD(vad_path, sample_rate=sample_rate, use_gpu=use_gpu)
        self.sample_rate = sample_rate
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.speaker_session = ort.InferenceSession(
            path_or_bytes=speaker_model_path,
            sess_options=session_options,
            providers=providers
        )
        self.reference_embeddings = {}

    def extract_embeddings(self, audio_segments: list[np.ndarray]) -> np.ndarray:
        min_length = 1024
        max_length = max(len(seg) for seg in audio_segments)
        max_length = max(max_length, min_length)

        padded_segments = [
            np.pad(seg, (0, max_length - len(seg)), mode='constant') if len(seg) < max_length else seg[:max_length]
            for seg in audio_segments]

        batch_specs = []
        for segment in padded_segments:
            # Шумоподавление
            reduced_noise = nr.reduce_noise(y=segment, sr=self.sample_rate)
            mel_spec = librosa.feature.melspectrogram(
                y=reduced_noise,
                sr=self.sample_rate,
                n_fft=400,
                hop_length=160,
                n_mels=80,
                fmax=8000,
                dtype=np.float64
            )
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max).T  # [T, 80]
            batch_specs.append(log_mel_spec)

        max_t = max(spec.shape[0] for spec in batch_specs)
        batch_specs = np.stack([np.pad(spec, ((0, max_t - spec.shape[0]), (0, 0)), mode='constant')
                                for spec in batch_specs], axis=0)

        inputs = {"feats": batch_specs.astype(np.float32)}
        embeddings = self.speaker_session.run(None, inputs)[0]
        return embeddings

    def diarize(self, audio_frames: np.ndarray, num_speakers: int = -1, cluster_threshold: float = 0.8) -> list[dict]:
        self.vad.set_mode(2)
        segments = self.vad.get_speech_segments(audio_frames, min_duration=0.1, max_gap=0.1)
        if not segments:
            print("Речь не обнаружена")
            return []

        print(f"Найдено сегментов до фильтрации: {len(segments)}")
        segments = [seg for seg in segments if seg[1] - seg[0] >= 0.15]  # Уменьшаем фильтр
        print(f"Найдено сегментов после фильтрации: {len(segments)}")
        if not segments:
            print("После фильтрации сегментов не осталось")
            return []

        segment_times = [(start, end) for start, end, _ in segments]
        segment_audios = [audio for _, _, audio in segments]
        embeddings = self.extract_embeddings(segment_audios)
        if len(embeddings) == 0:
            return []

        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarities = cosine_similarity(embeddings)
        print("Косинусные сходства между сегментами:")
        for i, sim in enumerate(similarities):
            print(f"Сегмент {i}: {sim}")

        distances = 1 - similarities
        distances = distances.astype(np.float64)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric="precomputed",
            cluster_selection_method="eom"
        )
        labels = clusterer.fit_predict(distances)

        # Постобработка выбросов
        for i in range(len(labels)):
            if labels[i] == -1:
                cluster_sims = {}
                for j in range(len(labels)):
                    if labels[j] != -1 and j != i:
                        cluster_sims[labels[j]] = cluster_sims.get(labels[j], []) + [similarities[i, j]]
                if cluster_sims:
                    best_cluster = max(cluster_sims, key=lambda k: np.mean(cluster_sims[k]))
                    if np.mean(cluster_sims[best_cluster]) > 0.5:
                        labels[i] = best_cluster

        # Объединение с учётом времени при num_speakers=-1
        if num_speakers == -1:
            unique_labels = set(labels) - {-1}
            current_num_clusters = len(unique_labels)
            while current_num_clusters > 1:
                cluster_sims = {}
                for l1 in unique_labels:
                    for l2 in unique_labels:
                        if l1 < l2:
                            sims = []
                            times_l1 = [t[0] for t, l in zip(segment_times, labels) if l == l1]
                            times_l2 = [t[0] for t, l in zip(segment_times, labels) if l == l2]
                            for i in range(len(labels)):
                                for j in range(len(labels)):
                                    if labels[i] == l1 and labels[j] == l2:
                                        time_diff = abs(segment_times[i][0] - segment_times[j][0])
                                        if time_diff < 10:  # Только близкие по времени
                                            sims.append(similarities[i, j])
                            if sims:
                                cluster_sims[(l1, l2)] = np.mean(sims)
                if not cluster_sims:
                    break
                merged = False
                for (l1, l2), sim in cluster_sims.items():
                    if sim > 0.995:  # Уменьшаем порог
                        labels[labels == l2] = l1
                        unique_labels.remove(l2)
                        current_num_clusters -= 1
                        merged = True
                        break
                if not merged:
                    break

        # Учёт num_speakers
        if num_speakers > 0:
            unique_labels = set(labels) - {-1}
            current_num_clusters = len(unique_labels)
            if current_num_clusters > num_speakers:
                while current_num_clusters > num_speakers:
                    cluster_sims = {}
                    for l1 in unique_labels:
                        for l2 in unique_labels:
                            if l1 < l2:
                                sims = [similarities[i, j] for i in range(len(labels)) for j in range(len(labels))
                                        if labels[i] == l1 and labels[j] == l2]
                                if sims:
                                    cluster_sims[(l1, l2)] = np.mean(sims)
                    if not cluster_sims:
                        break
                    (l1, l2) = max(cluster_sims, key=cluster_sims.get)
                    labels[labels == l2] = l1
                    unique_labels.remove(l2)
                    current_num_clusters -= 1
            elif current_num_clusters < num_speakers:
                print(f"Найдено {current_num_clusters} кластеров, меньше чем {num_speakers}.")

        return [{"start": float(f"{start:.3f}"), "end": float(f"{end:.3f}"), "speaker": int(speaker_id)}
                for (start, end), speaker_id in zip(segment_times, labels)]

    def set_reference_speakers(self, reference_audios: dict[int, np.ndarray]):
        self.reference_embeddings.clear()
        audio_segments = list(reference_audios.values())
        embeddings = self.extract_embeddings(audio_segments)
        for speaker_id, embedding in zip(reference_audios.keys(), embeddings):
            self.reference_embeddings[speaker_id] = embedding / np.linalg.norm(embedding)
        logger.debug(f"Установлено {len(self.reference_embeddings)} эталонных спикеров")


    def merge_segments(self, diarized_segments: list[dict], max_phrase_gap: float = 0.2) -> list[dict]:
        if not diarized_segments:
            return []

        # Сортируем по времени начала
        diarized_segments = sorted(diarized_segments, key=lambda x: x["start"])

        merged = []
        current_phrase = None

        for segment in diarized_segments:
            if current_phrase is None:
                current_phrase = {"start": segment["start"], "end": segment["end"], "speaker": segment["speaker"]}
            elif (segment["start"] - current_phrase["end"] <= max_phrase_gap and
                  segment["speaker"] == current_phrase["speaker"]):
                current_phrase["end"] = segment["end"]
            else:
                merged.append(current_phrase)
                current_phrase = {"start": segment["start"], "end": segment["end"], "speaker": segment["speaker"]}

        if current_phrase is not None:
            merged.append(current_phrase)

        return merged


    def diarize_and_merge(self, audio_frames: np.ndarray, num_speakers: int = -1, cluster_threshold: float = 0.8) -> list[
        dict]:
        # Сначала диаризуем мелкие сегменты
        raw_result = self.diarize(audio_frames, num_speakers, cluster_threshold)
        # Затем объединяем в фразы
        merged_result = self.merge_segments(raw_result, max_phrase_gap=0.2)
        return merged_result



    def identify_speaker(self, audio_frames: np.ndarray, similarity_threshold: float = 0.7) -> tuple[int, float]:
        if not self.reference_embeddings:
            logger.debug("Эталонные спикеры не установлены")
            return -1, 0.0

        segments = self.vad.get_speech_segments(audio_frames, min_duration=0.3, max_gap=0.2)
        if not segments:
            logger.debug("Речь не обнаружена")
            return -1, 0.0

        segment_audios = [audio for _, _, audio in segments]
        embeddings = self.extract_embeddings(segment_audios)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        avg_embedding = np.mean(embeddings, axis=0)

        ref_embeddings = np.array(list(self.reference_embeddings.values()))
        similarities = cosine_similarity([avg_embedding], ref_embeddings)[0]

        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]

        if max_similarity >= similarity_threshold:
            speaker_id = list(self.reference_embeddings.keys())[max_similarity_idx]
            logger.debug(f"Спикер идентифицирован: ID={speaker_id}, сходство={max_similarity:.3f}")
            return speaker_id, max_similarity
        else:
            logger.debug(f"Спикер не распознан, максимальное сходство={max_similarity:.3f}")
            return -1, max_similarity

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

        self.set_mode(2)

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

    def get_speech_segments(self, audio_frames: np.ndarray, min_duration: float = 0.1, max_gap: float = 0.1) -> list[
        tuple]:
        sample_rate = self.sample_rate  # 16000
        frame_size = self.frame_size  # 512

        # Если audio_frames — это уже массив фреймов (из load_and_preprocess_audio), преобразуем обратно в 1D массив
        if len(audio_frames.shape) == 2:  # [num_frames, frame_size]
            audio_frames = audio_frames.flatten()

        # Проверяем длину и дополняем, если нужно
        num_samples = len(audio_frames)
        num_frames = num_samples // frame_size
        if num_samples % frame_size != 0:
            padding = frame_size - (num_samples % frame_size)
            audio_frames = np.pad(audio_frames, (0, padding), mode='constant')

        # Разбиваем на фреймы
        frames = audio_frames.reshape(-1, frame_size)

        # Сбрасываем состояние VAD перед обработкой
        self.reset_state()
        speech_probs = []

        # Обрабатываем каждый фрейм через is_speech
        for frame in frames:
            is_speech_flag, new_state = self.is_speech(frame)
            self.state = new_state  # Обновляем состояние
            speech_probs.append(1.0 if is_speech_flag else 0.0)

        # Собираем сегменты
        segments = []
        start = None
        for i, prob in enumerate(speech_probs):
            if prob > 0.5:  # Порог вероятности речи
                if start is None:
                    start = i * frame_size / sample_rate  # Начало сегмента
            else:
                if start is not None:
                    end = i * frame_size / sample_rate
                    if end - start >= min_duration:
                        audio_segment = audio_frames[int(start * sample_rate):int(end * sample_rate)]
                        segments.append((start, end, audio_segment))
                    start = None

        # Последний сегмент
        if start is not None:
            end = len(audio_frames) / sample_rate
            if end - start >= min_duration:
                audio_segment = audio_frames[int(start * sample_rate):int(end * sample_rate)]
                segments.append((start, end, audio_segment))

        # Объединяем близкие сегменты с учётом max_gap
        merged_segments = []
        if segments:
            current_start, current_end, current_audio = segments[0]
            for next_start, next_end, next_audio in segments[1:]:
                if next_start - current_end <= max_gap:
                    current_end = next_end
                    current_audio = audio_frames[int(current_start * sample_rate):int(current_end * sample_rate)]
                else:
                    merged_segments.append((current_start, current_end, current_audio))
                    current_start, current_end, current_audio = next_start, next_end, next_audio
            merged_segments.append((current_start, current_end, current_audio))

        return merged_segments


def load_and_preprocess_audio(file_path: str, target_frame_size: int = 512) -> np.ndarray:
    """Загрузка аудио и подготовка фреймов для обработки файла целиком"""

    # Загружаем аудио и обрезаем до 60 секунд
    audio = AudioSegment.from_file(file_path)[0:60000]

    # Конвертируем в моно 16kHz, если нужно
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    if audio.channels > 1:
        audio = audio.split_to_mono()[1]  # Берем правый канал (можно изменить на [0] для левого)

    # Нормализация в float64
    samples = np.frombuffer(audio.raw_data, dtype=np.int16)
    samples_float64 = samples.astype(np.float64) / 32768.0  # Преобразуем в float64

    # Разбиваем на фреймы нужного размера
    num_frames = len(samples_float64) // target_frame_size
    frames = []
    for i in range(num_frames):
        start = i * target_frame_size
        end = start + target_frame_size
        frames.append(samples_float64[start:end])

    return np.array(frames, dtype=np.float64)  # Явно указываем float64


# Пример использования
if __name__ == "__main__":
    # variant 1
    from datetime import datetime as dt

    vad_path = Path("../models/VAD_silero_v5/silero_vad.onnx")
    speaker_model_path = Path("../models/Diar_model/wespeaker_en_voxceleb_resnet293_LM_meta.onnx")

    diarizer = SpeakerDiarizer(vad_path, speaker_model_path, use_gpu=False)

    # Загрузка аудио
    audio_file = Path("C:/Users/kojevnikov/PycharmProjects/Vosk5_FastAPI_streaming/trash/long.mp3")
    # audio_file = Path("C:/Users/kojevnikov/PycharmProjects/Sherpa_onnx_vosk_GPU/trash/q.wav")
    audio_frames = load_and_preprocess_audio(str(audio_file))
    time_start = dt.now()

    # Диаризация
    result = diarizer.diarize_and_merge(audio_frames, num_speakers=4, cluster_threshold=0.8)

    print(f"Время выполнения: {(dt.now() - time_start).total_seconds()} сек")
    for r in result:
        print(r)