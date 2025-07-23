import json
import asyncio
import os
import numpy as np
from pydub import AudioSegment
from pathlib import Path
import onnxruntime as ort
from python_speech_features import fbank
from scipy.signal import butter, sosfilt
from sklearn.metrics import silhouette_score, pairwise_distances
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


async def load_and_preprocess_audio(audio: AudioSegment, sample_rate: int = 16000, filter_cutoff: float = 30,
                                    filter_order: int = 10) -> np.ndarray:

    start_time = time.perf_counter()
    if audio.frame_rate != sample_rate:
        audio = await resample_audiosegment(audio, sample_rate)
    if audio.channels > 1:
        audio = audio.split_to_mono()[0]



    samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
    samples_float32 = samples.astype(np.float32) / 32768.0

    # if samples_float32.size > 0:
    #     sos = butter(filter_order, filter_cutoff, btype='high', fs=sample_rate, output='sos')
    #     samples_float32 = sosfilt(sos, samples_float32)
    # else:
    #     logger.warning("Получен пустой аудиофайл, фильтрация пропущена")

    load_time = time.perf_counter() - start_time
    logger.info(f"В работу принято аудио продолжительностью {audio.duration_seconds} сек")
    logger.info(
        f"Процедура: Загрузка, фильтрация и предобработка аудио (load_and_preprocess_audio) - {load_time:.4f} сек")
    return samples_float32


class Diarizer:
    def __init__(self, embedding_model_path: str,
                 sample_rate: int = 16000,
                 use_gpu: bool = False,
                 batch_size: int = 1,
                 cpu_workers: int = 0):
        self.sample_rate = sample_rate
        self.winlen = 0.025
        self.winstep = 0.01
        self.num_mel_bins = 80
        self.nfft = 512

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

    async def subsegment(self, fbank, seg_id, window_fs, period_fs, frame_shift):
        subsegs = []
        subseg_fbanks = []
        if fbank.size == 0:
            logger.warning(f"Пустой fbank для сегмента {seg_id}, пропускается")
            return subsegs, subseg_fbanks
        num_frames, feat_dim = fbank.shape

        if num_frames <= window_fs:
            subseg = f"{seg_id}-00000000-{num_frames:08d}"
            subseg_fbank = np.resize(fbank, (window_fs, feat_dim))
            subsegs.append(subseg)
            subseg_fbanks.append(subseg_fbank)
        else:
            max_subseg_begin = num_frames - window_fs + period_fs
            for subseg_begin in range(0, max_subseg_begin, period_fs):
                subseg_end = min(subseg_begin + window_fs, num_frames)
                subseg = f"{seg_id}-{subseg_begin:08d}-{subseg_end:08d}"
                subseg_fbank = np.resize(fbank[subseg_begin:subseg_end], (window_fs, feat_dim))
                subsegs.append(subseg)
                subseg_fbanks.append(subseg_fbank)
        logger.debug(f"Сегмент {seg_id}: создано {len(subsegs)} подсегментов")
        return subsegs, subseg_fbanks

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

    async def diarize(self, segments: list[tuple[np.ndarray, float, float]], num_speakers: int) -> list[dict]:
        start_time = time.perf_counter()

        subsegs = []
        subseg_fbanks = []
        subseg_indices = []
        current_index = 0

        for audio, start, end in segments:
            if audio.size == 0:
                logger.warning(f"Пустой аудиосегмент для интервала {start:.3f}-{end:.3f}, пропускается")
                continue
            fbank_feats = await self.extract_fbank(audio)
            if fbank_feats.size == 0:
                logger.warning(f"Не удалось извлечь fbank для интервала {start:.3f}-{end:.3f}, пропускается")
                continue
            seg_id = f"{start:.3f}-{end:.3f}"
            frame_shift = int(self.winstep * 1000)
            window_fs = int(0.5 * 1000) // frame_shift  # Уменьшено до 0.5 сек
            period_fs = int(0.25 * 1000) // frame_shift  # Уменьшено до 0.25 сек
            tmp_subsegs, tmp_subseg_fbanks = await self.subsegment(fbank_feats, seg_id, window_fs, period_fs,
                                                                   frame_shift)
            if not tmp_subsegs:
                logger.warning(f"Не удалось создать подсегменты для интервала {start:.3f}-{end:.3f}, пропускается")
                continue
            num_subsegs = len(tmp_subsegs)
            subseg_indices.append(list(range(current_index, current_index + num_subsegs)))
            current_index += num_subsegs
            subsegs.extend(tmp_subsegs)
            subseg_fbanks.extend(tmp_subseg_fbanks)

        if not subseg_fbanks:
            logger.info("Не удалось извлечь валидные подсегменты")
            return []

        embeddings = await self.extract_embeddings(subseg_fbanks, subseg_cmn=True)
        if len(embeddings) == 0:
            logger.info("Не удалось извлечь валидные эмбеддинги")
            return []

        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        # Логирование косинусных расстояний между эмбеддингами сегментов
        segment_embeddings = []
        for indices in subseg_indices:
            seg_embs = embeddings[indices]
            mean_emb = np.mean(seg_embs, axis=0)
            segment_embeddings.append(mean_emb)
        if segment_embeddings:
            distances = pairwise_distances(segment_embeddings, metric='cosine')
            for i, (start, end) in enumerate([(s[1], s[2]) for s in segments]):
                logger.debug(f"Средние косинусные расстояния для сегмента {start:.2f}-{end:.2f}: {distances[i]}")

        if len(embeddings) <= 2:
            labels = [0] * len(embeddings)
            logger.info("Меньше 3 эмбеддингов, все метки установлены в 0")
        else:
            n_neighbors = 15  # Увеличено
            umap_embeddings = UMAP(
                n_components=min(32, len(embeddings) - 2),
                metric='cosine',
                n_neighbors=n_neighbors,
                min_dist=0.05,  # Уменьшено
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
                    logger.error("HDBSCAN пометил все как шум, все метки установлены в 0")
                else:
                    unique_labels = np.unique(labels)
                    logger.debug(f"HDBSCAN нашел {len(unique_labels)} кластеров: {unique_labels}")

                    # Постобработка: объединение похожих кластеров
                    if len(unique_labels) > 2 and -1 in unique_labels:
                        unique_labels = unique_labels[unique_labels != -1]
                    if len(unique_labels) > 2:
                        cluster_embeddings = []
                        for label in unique_labels:
                            cluster_embs = embeddings[labels == label]
                            cluster_embeddings.append(np.mean(cluster_embs, axis=0))
                        cluster_distances = pairwise_distances(cluster_embeddings, metric='cosine')
                        clustering = AgglomerativeClustering(n_clusters=2, metric='cosine', linkage='average')
                        new_labels = clustering.fit_predict(cluster_embeddings)
                        label_mapping = {old_label: new_labels[i] for i, old_label in enumerate(unique_labels)}
                        labels = np.array([label_mapping.get(label, 0) if label != -1 else 0 for label in labels])
                        logger.debug(f"Постобработка: объединено в 2 кластера, новые метки: {np.unique(labels)}")

        segment_speaker = []
        for i, indices in enumerate(subseg_indices):
            segment_labels = [labels[j] for j in indices]
            most_common_label = Counter(segment_labels).most_common(1)[0][0]
            segment_speaker.append(most_common_label)
            logger.debug(
                f"Сегмент {segments[i][1]:.2f}-{segments[i][2]:.2f}: метки {segment_labels}, выбран спикер {most_common_label}")
            logger.info(f"Сегмент {segments[i][1]:.2f}-{segments[i][2]:.2f}: выбран спикер {most_common_label}")

        results = []
        for (audio, start, end), speaker in zip(segments, segment_speaker):
            results.append({"start": start, "end": end, "speaker": speaker})

        total_diarize_time = time.perf_counter() - start_time
        logger.debug(f"Процедура: Общая диаризация (diarize) - {total_diarize_time:.4f} сек")

        return results


async def main():
    num_speakers = -1
    use_gpu_diar = True
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
        segments=segments,
        num_speakers=num_speakers
    )

    for r in result:
        print(f"Спикер {r['speaker']}: {r['start']:.2f} - {r['end']:.2f} сек")


if __name__ == "__main__":
    asyncio.run(main())