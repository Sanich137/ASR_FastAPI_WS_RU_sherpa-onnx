import json
import asyncio
import os
import numpy as np
from pydub import AudioSegment
from pathlib import Path
import onnxruntime as ort
from python_speech_features import fbank
from scipy.signal import butter, sosfilt
from sklearn.metrics.pairwise import cosine_distances
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
    with open(asr_file, 'r', encoding='utf-8') as f:
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
    logger.info(f"В работу принято аудио продолжительностью {audio.duration_seconds} сек")
    logger.info(f"Процедура: Загрузка, фильтрация и предобработка аудио - {load_time:.4f} сек")
    return samples_float32


class Diarizer:
    def __init__(self, embedding_model_path: str, sample_rate: int = 16000, use_gpu: bool = False, batch_size: int = 32,
                 cpu_workers: int = 0, window_size: float = 0.5, window_step: float = 0.04, asr_margin: float = 0.1):
        self.sample_rate = sample_rate
        self.winlen = 0.025
        self.winstep = 0.01
        self.num_mel_bins = 80
        self.nfft = 512
        self.window_size = window_size
        self.window_step = window_step
        self.asr_margin = asr_margin

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
        logger.info(f"Используемые для диаризации провайдеры {self.embedding_session.get_providers()}")

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

    async def diarize(self, audio: np.ndarray, num_speakers: int, asr_data: list[dict],
                      tau: float = 5.0) -> list[dict]:
        start_time = time.perf_counter()

        # Извлечение речевых регионов из ASR
        word_regions = []
        for segment in asr_data:
            for word in segment['data']['result']:
                start = max(0, word['start'] - self.asr_margin)
                end = word['end'] + self.asr_margin
                word_regions.append((start, end))
        logger.debug(f"Извлечено {len(word_regions)} регионов слов из ASR")

        # Объединение пересекающихся регионов
        if not word_regions:
            logger.info("Нет данных ASR, диаризация невозможна")
            return []
        word_regions.sort(key=lambda x: x[0])
        merged_regions = []
        current_start, current_end = word_regions[0]
        for start, end in word_regions[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged_regions.append((current_start, current_end))
                current_start, current_end = start, end
        merged_regions.append((current_start, current_end))
        logger.debug(f"Объединено в {len(merged_regions)} речевых регионов")

        # Сегментация с помощью pyannote-segmentation-3.0
        segments = []
        for region_start, region_end in merged_regions:
            region_segments = self.segment_audio(audio, region_start, region_end)
            segments.extend(region_segments)
        if not segments:
            logger.info("Не удалось получить сегменты от pyannote-segmentation")
            return []

        # Извлечение fbank и эмбеддингов для сегментов
        fbanks = []
        segment_times = []
        segment_centers = []
        segment_speakers = []
        for seg in segments:
            start_sample = int(seg["start"] * self.sample_rate)
            end_sample = int(seg["end"] * self.sample_rate)
            if end_sample > len(audio):
                end_sample = len(audio)
            if end_sample <= start_sample:
                continue
            segment_audio = audio[start_sample:end_sample]
            if len(segment_audio) < int(self.window_size * self.sample_rate):
                segment_audio = np.pad(segment_audio,
                                       (0, int(self.window_size * self.sample_rate) - len(segment_audio)),
                                       'constant')
            fbank_feats = await self.extract_fbank(segment_audio)
            if fbank_feats.size > 0:
                fbanks.append(fbank_feats)
                segment_times.append((seg["start"], seg["end"]))
                segment_centers.append((seg["start"] + seg["end"]) / 2)
                segment_speakers.append(seg["speaker"])
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
                min_dist=0.15,
                random_state=2023,
                n_jobs=1
            ).fit_transform(embeddings)

            if num_speakers >= 2:
                clustering = AgglomerativeClustering(n_clusters=num_speakers, metric='cosine', linkage='average')
                labels = clustering.fit_predict(umap_embeddings)
                logger.info(f"Применена AgglomerativeClustering с {num_speakers} кластерами")
            else:
                feature_dist = cosine_distances(umap_embeddings).astype(np.float64)
                time_dist = np.zeros_like(feature_dist, dtype=np.float64)
                tau = 0.1 * (max(c for c in segment_centers) - min(c for c in segment_centers))

                for i in range(len(segment_centers)):
                    for j in range(i + 1, len(segment_centers)):
                        dist = abs(segment_centers[i] - segment_centers[j]) / tau
                        time_dist[i, j] = time_dist[j, i] = dist

                time_dist = np.clip(time_dist, 0, 1)
                max_time_dist = np.max(time_dist)
                time_dist_norm = time_dist / max_time_dist if max_time_dist > 0 else time_dist
                feature_dist_norm = feature_dist / np.max(feature_dist)
                combined_dist = np.sqrt(feature_dist_norm ** 2 + (time_dist_norm / tau) ** 2)

                clustering = HDBSCAN(
                    metric='precomputed',
                    min_cluster_size=2,
                    min_samples=3,
                    cluster_selection_epsilon=2
                )
                labels = clustering.fit_predict(combined_dist)
                if np.all(labels == -1):
                    labels = np.zeros_like(labels)
                    logger.info("HDBSCAN пометил все как шум, все метки установлены в 0")
                else:
                    unique_labels = np.unique(labels[labels != -1])
                    logger.info(f"HDBSCAN нашел {len(unique_labels)} кластеров: {unique_labels}")

        # Формирование финальных сегментов
        # Группируем сегменты по спикерам
        speaker_segments = {}
        for (start, end), label in zip(segment_times, labels):
            speaker = label if label != -1 else 0  # Шум помечаем как спикер 0
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append((start, end))

        # Объединяем сегменты для каждого спикера
        results = []
        for speaker, segs in speaker_segments.items():
            segs.sort(key=lambda x: x[0])  # Сортировка по времени начала
            current_start, current_end = segs[0]
            for start, end in segs[1:]:
                if start <= current_end + 0.1:  # Учитываем небольшой зазор
                    current_end = max(current_end, end)
                else:
                    # Выравнивание с ASR границами
                    asr_starts = [w['start'] for s in asr_data for w in s['data']['result']]
                    asr_ends = [w['end'] for s in asr_data for w in s['data']['result']]
                    nearest_start = min(asr_starts, key=lambda x: abs(x - current_start))
                    nearest_end = min(asr_ends, key=lambda x: abs(x - current_end))
                    if abs(nearest_start - current_start) <= self.asr_margin:
                        current_start = nearest_start
                    if abs(nearest_end - current_end) <= self.asr_margin:
                        current_end = nearest_end
                    results.append({"start": current_start, "end": current_end, "speaker": speaker})
                    logger.debug(f"Создан сегмент: {current_start:.2f}-{current_end:.2f}, спикер {speaker}")
                    current_start, current_end = start, end
            # Добавляем последний сегмент
            asr_starts = [w['start'] for s in asr_data for w in s['data']['result']]
            asr_ends = [w['end'] for s in asr_data for w in s['data']['result']]
            nearest_start = min(asr_starts, key=lambda x: abs(x - current_start))
            nearest_end = min(asr_ends, key=lambda x: abs(x - current_end))
            if abs(nearest_start - current_start) <= self.asr_margin:
                current_start = nearest_start
            if abs(nearest_end - current_end) <= self.asr_margin:
                current_end = nearest_end
            results.append({"start": current_start, "end": current_end, "speaker": speaker})
            logger.debug(f"Создан сегмент: {current_start:.2f}-{current_end:.2f}, спикер {speaker}")

        # Сортировка по времени начала
        results.sort(key=lambda x: x["start"])

        total_diarize_time = time.perf_counter() - start_time
        logger.info(f"Процедура: Общая диаризация - {total_diarize_time:.4f} сек")
        return results

async def main():
    num_speakers = -1
    use_gpu_diar = True
    batch_size = 32
    max_cpu_workers = 0
    speaker_model_path = Path("../models/DIARISATION_model/voxblink2_samresnet100_ft.onnx")
    audio_path = "../trash/Luxury_Girl_о_разводе_и_отношениях_с_Дэвидом___подкаст__luxurygirl.wav"
    asr_file = "../trash/replit/Luxury_Girl_asr.json"

    audio = AudioSegment.from_file(audio_path)
    samples_float32 = await load_and_preprocess_audio(audio)

    asr_data = load_asr_data(asr_file)

    diarizer = Diarizer(
        embedding_model_path=str(speaker_model_path),
        batch_size=batch_size,
        cpu_workers=max_cpu_workers,
        use_gpu=use_gpu_diar,
    )

    result = await diarizer.diarize(
        audio=samples_float32,
        asr_data=asr_data,
        num_speakers=num_speakers,
        tau=1
    )

    for r in result:
        print(f"Спикер {r['speaker']}: {r['start']:.2f} - {r['end']:.2f} сек")


if __name__ == "__main__":
    asyncio.run(main())