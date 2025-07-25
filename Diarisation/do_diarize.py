import json
import asyncio
import os
import numpy as np
from pydub import AudioSegment
from pathlib import Path
import onnxruntime as ort
from python_speech_features import fbank
from scipy.signal import butter, sosfilt
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from umap import UMAP
from hdbscan import HDBSCAN
from concurrent.futures import ThreadPoolExecutor
import time
from collections import Counter
from utils.do_logging import logger
from utils.pre_start_init import paths
from utils.resamppling import resample_audiosegment


def load_asr_data(asr_file: str) -> list[dict]:
    with open(asr_file, 'r') as f:
        data = json.load(f)
    return data['raw_data']['channel_1']


async def load_and_preprocess_audio(audio: AudioSegment, sample_rate: int = 16000, filter_cutoff: float = 30,
                                    filter_order: int = 10) -> np.ndarray:
    start_time = time.perf_counter()
    if audio.frame_rate != sample_rate:
        audio = await resample_audiosegment(audio, sample_rate)
    if audio.channels > 1:
        audio = audio.split_to_mono()[0]
    samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
    samples_float32 = samples.astype(np.float32) / 32768.0

    if samples_float32.size > 0:
        sos = butter(filter_order, filter_cutoff, btype='high', fs=sample_rate, output='sos')
        samples_float32 = sosfilt(sos, samples_float32)
    else:
        logger.warning("Получен пустой аудиофайл, фильтрация пропущена")

    load_time = time.perf_counter() - start_time
    logger.debug(f"В работу принято аудио продолжительностью {audio.duration_seconds} сек")
    logger.debug(f"Процедура: Загрузка, фильтрация и предобработка аудио - {load_time:.4f} сек")
    return samples_float32


class Diarizer:
    def __init__(self, embedding_model_path: str,
                 sample_rate: int = 16000,
                 use_gpu: bool = False,
                 batch_size: int = 32,
                 cpu_workers: int = 0,
                 window_size: float = 0.5,
                 window_step: float = 0.25):
        self.sample_rate = sample_rate
        self.winlen = 0.025
        self.winstep = 0.01
        self.num_mel_bins = 80
        self.nfft = 512
        self.window_size = window_size  # seconds
        self.window_step = window_step  # seconds

        if use_gpu:
            self.batch_size = batch_size
            self.max_cpu_workers = 1
        else:
            self.batch_size = 1
            self.max_cpu_workers = os.cpu_count() - 1 if cpu_workers < 1 else cpu_workers

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        so = ort.SessionOptions()
        so.log_severity_level = 4
        so.enable_profiling = False
        so.inter_op_num_threads = 0
        so.intra_op_num_threads = 0
        self.embedding_session = ort.InferenceSession(
            embedding_model_path,
            sess_options=so,
            providers=providers
        )
        logger.debug(f"Используемые для диаризации провайдеры {self.embedding_session.get_providers()}")

    async def extract_fbank(self, audio):
        if audio.size == 0:
            logger.warning("Пустой аудиосегмент в extract_fbank, возвращается пустой массив")
            return np.array([])
        feats, _ = fbank(audio, samplerate=self.sample_rate, winlen=self.winlen, winstep=self.winstep,
                         nfilt=self.num_mel_bins, nfft=self.nfft, winfunc=np.hamming)
        feats = np.log(np.maximum(feats, 1e-8)).astype(np.float32)
        return feats

    async def extract_embeddings(self, fbanks, subseg_cmn):
        if not fbanks:
            logger.warning("Нет fbank для извлечения эмбеддингов")
            return np.array([])
        fbanks_array = np.stack(fbanks)
        if subseg_cmn:
            fbanks_array = fbanks_array - np.mean(fbanks_array, axis=1, keepdims=True)

        async def process_batch(batch):
            return await asyncio.to_thread(
                self.embedding_session.run,
                output_names=['embs'],
                input_feed={'feats': batch}
            )

        embeddings = []
        with ThreadPoolExecutor(max_workers=self.max_cpu_workers) as executor:
            tasks = [
                process_batch(fbanks_array[i:i + self.batch_size])
                for i in range(0, fbanks_array.shape[0], self.batch_size)
            ]
            for task in await asyncio.gather(*tasks):
                embeddings.append(task[0].squeeze())

        embeddings = np.vstack(embeddings)
        logger.debug(f"Извлечено {len(embeddings)} эмбеддингов")
        return embeddings

    async def diarize(self, audio: np.ndarray, num_speakers: int) -> list[dict]:
        start_time = time.perf_counter()

        # Разбиение аудио на окна
        window_samples = int(self.window_size * self.sample_rate)
        step_samples = int(self.window_step * self.sample_rate)
        windows = []
        window_times = []
        audio_duration = len(audio) / self.sample_rate

        for start in np.arange(0, audio_duration - self.window_size + self.window_step, self.window_step):
            start_sample = int(start * self.sample_rate)
            end_sample = min(start_sample + window_samples, len(audio))
            window_audio = audio[start_sample:end_sample]
            if len(window_audio) < window_samples:
                window_audio = np.pad(window_audio, (0, window_samples - len(window_audio)), mode='constant')
            windows.append(window_audio)
            window_times.append((start, min(start + self.window_size, audio_duration)))

        # Извлечение fbank для каждого окна
        fbanks = []
        for window in windows:
            fbank_feats = await self.extract_fbank(window)
            if fbank_feats.size > 0:
                fbanks.append(fbank_feats)

        if not fbanks:
            logger.info("Не удалось извлечь валидные fbank")
            return []

        # Извлечение эмбеддингов
        embeddings = await self.extract_embeddings(fbanks, subseg_cmn=True)
        if len(embeddings) == 0:
            logger.info("Не удалось извлечь валидные эмбеддинги")
            return []

        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        # Кластеризация
        if len(embeddings) <= 2:
            labels = [0] * len(embeddings)
            logger.info("Меньше 3 эмбеддингов, все метки установлены в 0")
        else:
            n_neighbors = 15
            umap_embeddings = UMAP(
                n_components=min(32, len(embeddings) - 2),
                metric='cosine',
                n_neighbors=n_neighbors,
                min_dist=0.05,
                random_state=2023,
                n_jobs=1
            ).fit_transform(embeddings)

            if num_speakers >= 2:
                clustering = AgglomerativeClustering(n_clusters=num_speakers, metric='cosine', linkage='average')
                labels = clustering.fit_predict(umap_embeddings)
                logger.info(f"Применена AgglomerativeClustering с {num_speakers} кластерами")
            else:
                labels = HDBSCAN(min_cluster_size=3, min_samples=3).fit_predict(umap_embeddings)
                if np.all(labels == -1):
                    labels = np.zeros_like(labels)
                    logger.info("HDBSCAN пометил все как шум, все метки установлены в 0")
                else:
                    unique_labels = np.unique(labels)
                    logger.info(f"HDBSCAN нашел {len(unique_labels)} кластеров: {unique_labels}")

        # Формирование сегментов
        results = []
        current_speaker = labels[0] if len(labels) > 0 else 0
        current_start = window_times[0][0] if window_times else 0
        current_end = window_times[0][1] if window_times else 0

        for i in range(1, len(labels)):
            if labels[i] != current_speaker or window_times[i][0] > current_end + 0.1:
                if current_end - current_start >= 0.1:  # Минимальная длительность сегмента
                    results.append({"start": current_start, "end": current_end, "speaker": current_speaker})
                    logger.debug(f"Создан сегмент: {current_start:.2f}-{current_end:.2f}, спикер {current_speaker}")
                current_speaker = labels[i]
                current_start = window_times[i][0]
            current_end = window_times[i][1]

        if current_end - current_start >= 0.1:
            results.append({"start": current_start, "end": current_end, "speaker": current_speaker})
            logger.debug(f"Создан сегмент: {current_start:.2f}-{current_end:.2f}, спикер {current_speaker}")

        total_diarize_time = time.perf_counter() - start_time
        logger.info(f"Процедура: Общая диаризация - {total_diarize_time:.4f} сек")

        return results


def extract_speech_segments(asr_data: dict, pause_threshold: float = 0.25, min_duration: float = 0.2) -> list[
    tuple[float, float]]:
    segments = []
    current_start = None
    current_end = None
    words_buffer = []

    for result in asr_data:
        words = result['data']['result']
        words_buffer.extend(words)

    processed_words = []
    for i, word in enumerate(words_buffer):
        start = word['start']
        end = word['end']
        if start == end:
            if i > 0 and i < len(words_buffer) - 1:
                prev_word = words_buffer[i - 1]
                next_word = words_buffer[i + 1]
                prev_gap = start - prev_word['end']
                next_gap = next_word['start'] - end
                if next_gap <= prev_gap:
                    end = next_word['start']
                else:
                    start = prev_word['end']
            elif i > 0:
                start = words_buffer[i - 1]['end']
                end = start + min_duration
            elif i < len(words_buffer) - 1:
                end = words_buffer[i + 1]['start']
            else:
                end = start + min_duration
        processed_words.append({'start': start, 'end': max(start, end)})
        logger.debug(f"Обработано слово: {word['word']}, start={start:.2f}, end={end:.2f}")

    for i, word in enumerate(processed_words):
        start = word['start']
        end = word['end']
        if current_start is None:
            current_start = start
            current_end = end
        else:
            if start - current_end <= pause_threshold:
                current_end = end
            else:
                if current_end - current_start >= min_duration:
                    segments.append((current_start, current_end))
                    logger.debug(f"Создан сегмент: {current_start:.2f} - {current_end:.2f} сек")
                current_start = start
                current_end = end

    if current_start is not None and current_end - current_start >= min_duration:
        segments.append((current_start, current_end))
        logger.debug(f"Создан сегмент: {current_start:.2f} - {current_end:.2f} сек")

    logger.debug(f"Всего сегментов: {len(segments)}")
    return segments


async def main():
    num_speakers = -1
    max_phrase_gap = 1
    use_gpu_diar = False
    batch_size = 32
    max_cpu_workers = 0
    speaker_model_path = Path("../models/DIARISATION_model/voxblink2_samresnet100_ft.onnx")
    audio_path = "../trash/Luxury_Girl_о_разводе_и_отношениях_с_Дэвидом___подкаст__luxurygirl.wav"
    asr_file = "../trash/Luxury_Girl_asr.json"

    audio = AudioSegment.from_file(audio_path)
    samples_float32 = await load_and_preprocess_audio(audio)

    asr_data = load_asr_data(asr_file)
    segments_times = extract_speech_segments(asr_data, pause_threshold=0.25, min_duration=0.2)

    segments = []
    for start, end in segments_times:
        start_sample = int(start * 16000)
        end_sample = int(end * 16000)
        if end_sample > len(samples_float32):
            logger.warning(f"Конец сегмента {end:.3f} превышает длину аудио, обрезается")
            end_sample = len(samples_float32)
        if start_sample >= end_sample:
            logger.warning(f"Некорректный сегмент {start:.3f}-{end:.3f}, пропускается")
            continue
        audio_fragment = samples_float32[start_sample:end_sample]
        segments.append((audio_fragment, start, end))

    diarizer = Diarizer(
        embedding_model_path=str(speaker_model_path),
        batch_size=batch_size,
        cpu_workers=max_cpu_workers,
        use_gpu=use_gpu_diar,
    )

    result = await diarizer.diarize(
        audio = samples_float32,
        num_speakers=num_speakers)

    for r in result:
        print(f"Спикер {r['speaker']}: {r['start']:.2f} - {r['end']:.2f} сек")


if __name__ == "__main__":
    asyncio.run(main())