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
from sklearn import __version__ as sklearn_version
from sklearn.cluster import AgglomerativeClustering
import hdbscan
from concurrent.futures import ThreadPoolExecutor
import time
from utils.do_logging import logger
from utils.pre_start_init import paths
from utils.resamppling import resample_audiosegment

def load_asr_data(asr_file: str) -> list[dict]:
    with open(asr_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['raw_data']['channel_1']

async def load_and_preprocess_audio(audio: AudioSegment, sample_rate: int = 16000, filter_cutoff: float = 50,
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
    def __init__(self, embedding_model_path: str, segmentation_model_path: str, sample_rate: int = 16000,
                 use_gpu: bool = False, batch_size: int = 1, cpu_workers: int = 0, max_phrase_gap: float = 0.5,
                 min_duration: float = 0.15, asr_margin: float = 0.1, chunk_duration: float = 20.0,
                 embedding_threshold: float = 0.3, min_chunk_step: float = 0.1):
        self.sample_rate = sample_rate
        self.winlen = 0.025
        self.winstep = 0.01
        self.num_mel_bins = 80
        self.nfft = 512
        self.max_phrase_gap = max_phrase_gap
        self.min_duration = min_duration
        self.asr_margin = asr_margin
        self.chunk_duration = chunk_duration
        self.embedding_threshold = embedding_threshold
        self.min_chunk_step = min_chunk_step

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
        so.enable_mem_pattern = False

        self.embedding_session = ort.InferenceSession(
            embedding_model_path,
            sess_options=so,
            providers=providers
        )
        self.segmentation_session = ort.InferenceSession(
            segmentation_model_path,
            sess_options=so,
            providers=providers
        )
        segmentation_inputs = [input.name for input in self.segmentation_session.get_inputs()]
        self.segmentation_input_name = 'x' if 'x' in segmentation_inputs else segmentation_inputs[0]
        logger.info(f"Используемые для диаризации провайдеры: {self.embedding_session.get_providers()}")
        logger.info(f"Используемые для сегментации провайдеры: {self.segmentation_session.get_providers()}")

    async def highpass_filter(self, signal: np.ndarray, cutoff: float, filter_order: int) -> np.ndarray:
        sos = butter(filter_order, cutoff, btype='high', fs=self.sample_rate, output='sos')
        return sosfilt(sos, signal)

    async def extract_fbank(self, audio):
        if audio.size == 0:
            logger.warning("Пустой аудиосегмент в extract_fbank, возвращается пустой массив")
            return np.array([])

        feats, _ = fbank(audio, samplerate=self.sample_rate, winlen=self.winlen, winstep=self.winstep,
                         nfilt=self.num_mel_bins, nfft=self.nfft, winfunc=np.hamming)

        # feats = np.log(np.maximum(feats, 1e-8)).astype(np.float32)

        return feats.astype(np.float32)

    async def subsegment(self, fbank, seg_id, window_fs, period_fs, frame_shift):
        subsegs = []
        subseg_fbanks = []
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
        if len(embeddings) == 0:
            logger.warning("Получены пустые эмбеддинги")
            return np.array([])

        norms = np.linalg.norm(embeddings, axis=1)
        valid_mask = (norms > 1e-8) & (~np.isnan(embeddings).any(axis=1))
        if not np.any(valid_mask):
            logger.warning("Все эмбеддинги некорректны (нулевые или содержат NaN)")
            return np.array([])
        embeddings = embeddings[valid_mask]
        logger.info(f"Извлечено {len(embeddings)} валидных эмбеддингов, средняя норма: {np.mean(norms[valid_mask]):.4f}")
        return embeddings

    def segment_audio(self, audio: np.ndarray, region_start: float, region_end: float) -> list[dict]:
        logger.info(f"Начало сегментации региона: {region_start:.2f}-{region_end:.2f} сек")
        region_duration = region_end - region_start
        if region_duration < self.min_duration:
            logger.warning(f"Регион слишком короткий ({region_duration:.3f} сек < {self.min_duration} сек), пропускается")
            return []

        start_sample = int(region_start * self.sample_rate)
        end_sample = int(region_end * self.sample_rate)
        if end_sample > len(audio):
            end_sample = len(audio)
        if end_sample <= start_sample:
            logger.warning(
                f"Некорректный регион: start={region_start}, end={region_end}, длина аудио={len(audio) / self.sample_rate}")
            return []
        segment_audio = audio[start_sample:end_sample].reshape(1, 1, -1).astype(np.float32)
        logger.info(f"Форма входного аудио для сегментации: {segment_audio.shape}")

        inputs = {self.segmentation_input_name: segment_audio}
        logits = self.segmentation_session.run(None, inputs)[0][0]
        probs = 1 / (1 + np.exp(-logits))
        logger.info(f"Форма логитов: {probs.shape}, макс. вероятности: {np.max(probs, axis=0)}")

        results = []
        num_frames, num_classes = probs.shape
        frame_duration = region_duration / num_frames if num_frames > 0 else 0.0

        for frame_idx in range(num_frames):
            time = region_start + (frame_idx / num_frames) * region_duration
            active_classes = np.where(probs[frame_idx] > 0.5)[0]
            active_speakers = set()

            if len(active_classes) == 0 or np.array_equal(active_classes, [0]):
                max_class = np.argmax(probs[frame_idx][1:]) + 1 if np.max(probs[frame_idx][1:]) > 0.1 else None
                if max_class and max_class in [1, 2, 3]:
                    active_speakers.add(f"temp_speaker_{max_class}")
            else:
                for cls in active_classes:
                    if cls == 0:
                        continue
                    if cls in [1, 2, 3]:
                        active_speakers.add(f"temp_speaker_{cls}")
                    elif cls in [4, 5, 6]:
                        overlap_speakers = {
                            4: ["temp_speaker_1", "temp_speaker_2"],
                            5: ["temp_speaker_1", "temp_speaker_3"],
                            6: ["temp_speaker_2", "temp_speaker_3"]
                        }[cls]
                        active_speakers.update(overlap_speakers)

            for speaker in active_speakers:
                results.append({
                    "start": time,
                    "end": time + frame_duration,
                    "speaker": speaker
                })

        if not results:
            logger.warning(f"Не найдено активных спикеров в регионе {region_start:.2f}-{region_end:.2f}")
            return []

        speaker_segments = {}
        for seg in results:
            speaker = seg["speaker"]
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append((seg["start"], seg["end"]))

        merged_results = []
        for speaker, segs in speaker_segments.items():
            segs.sort(key=lambda x: x[0])
            current_start, current_end = segs[0]
            for start, end in segs[1:]:
                if start <= current_end + 0.001:
                    current_end = max(current_end, end)
                else:
                    merged_results.append({"start": current_start, "end": current_end, "speaker": speaker})
                    current_start, current_end = start, end
            merged_results.append({"start": current_start, "end": current_end, "speaker": speaker})

        merged_results.sort(key=lambda x: x["start"])
        logger.info(f"Получено {len(merged_results)} сегментов в регионе {region_start:.2f}-{region_end:.2f}")
        return merged_results

    async def diarize(self, audio: np.ndarray, num_speakers: int, asr_data: list[dict], filter_cutoff: float = 50,
                      filter_order: int = 10) -> list[dict]:
        start_time = time.perf_counter()

        # Подготовка регионов на основе ASR
        word_regions = []
        for segment in asr_data:
            for word in segment['data']['result']:
                start = word['start']
                end = word['end']
                word_regions.append({"start": start, "end": end, "word": word})
        word_regions.sort(key=lambda x: x["start"])

        audio_duration = len(audio) / self.sample_rate
        chunk_start = 0.0
        all_segments = []
        frame_shift = int(self.winstep * 1000)
        window_fs = int(1.5 * 1000) // frame_shift
        period_fs = int(0.75 * 1000) // frame_shift

        while chunk_start < audio_duration:
            chunk_end = min(chunk_start + self.chunk_duration, audio_duration)
            logger.info(f"Обработка чанка: {chunk_start:.2f}-{chunk_end:.2f} сек")

            # Извлечение аудио для чанка
            start_sample = int(chunk_start * self.sample_rate)
            end_sample = int(chunk_end * self.sample_rate)
            if end_sample > len(audio):
                end_sample = len(audio)
            if end_sample <= start_sample:
                logger.warning(f"Некорректный чанк: start={chunk_start}, end={chunk_end}")
                chunk_start = chunk_end + self.min_chunk_step
                continue
            chunk_audio = audio[start_sample:end_sample]

            # Сегментация чанка
            region_segments = self.segment_audio(chunk_audio, 0.0, chunk_end - chunk_start)
            if not region_segments:
                logger.debug(f"Чанк {chunk_start:.2f}-{chunk_end:.2f} не содержит сегментов")
                chunk_start = chunk_end + self.min_chunk_step
                continue

            # Корректировка временных границ по ASR
            adjusted_segments = []
            asr_starts = [w["start"] for w in word_regions if chunk_start <= w["start"] <= chunk_end]
            asr_ends = [w["end"] for w in word_regions if chunk_start <= w["end"] <= chunk_end]
            for seg in region_segments:
                seg_start = seg["start"] + chunk_start
                seg_end = seg["end"] + chunk_start
                nearest_start = min(asr_starts, key=lambda x: abs(x - seg_start)) if asr_starts else seg_start
                nearest_end = min(asr_ends, key=lambda x: abs(x - seg_end)) if asr_ends else seg_end
                if abs(nearest_start - seg_start) <= self.asr_margin:
                    seg_start = nearest_start
                if abs(nearest_end - seg_end) <= self.asr_margin:
                    seg_end = nearest_end
                if seg_end - seg_start >= self.min_duration:
                    adjusted_segments.append({
                        "start": seg_start,
                        "end": seg_end,
                        "speaker": seg["speaker"],
                        "chunk_start": chunk_start
                    })

            if not adjusted_segments:
                logger.debug(f"Чанк {chunk_start:.2f}-{chunk_end:.2f} не содержит валидных сегментов после фильтрации по ASR и min_duration")
                chunk_start = chunk_end + self.min_chunk_step
                continue

            # Извлечение эмбеддингов для подсегментов
            chunk_segment_embeddings = {}
            for seg in adjusted_segments:
                start = seg["start"]
                end = seg["end"]
                temp_speaker = seg["speaker"]
                start_sample = int(start * self.sample_rate)
                end_sample = int(end * self.sample_rate)
                if end_sample > len(audio):
                    end_sample = len(audio)
                if end_sample <= start_sample:
                    logger.warning(f"Некорректный сегмент: start={start}, end={end}")
                    continue
                segment_audio = audio[start_sample:end_sample]
                segment_audio = await self.highpass_filter(segment_audio, cutoff=filter_cutoff, filter_order=filter_order)
                seg_id = f"{start:.3f}-{end:.3f}"
                fbank_feats = await self.extract_fbank(segment_audio)
                if fbank_feats.size == 0:
                    logger.warning(f"Пустые fbank для сегмента {seg_id}")
                    continue
                subseg_ids, subseg_fbanks = await self.subsegment(fbank_feats, seg_id, window_fs, period_fs, frame_shift)
                if not subseg_fbanks:
                    logger.warning(f"Пустые подсегменты для сегмента {seg_id}")
                    continue
                embeddings = await self.extract_embeddings(subseg_fbanks, subseg_cmn=True)
                if embeddings.size == 0:
                    logger.warning(f"Не удалось извлечь эмбеддинги для сегмента {seg_id}")
                    continue
                chunk_segment_embeddings[seg_id] = {
                    "embeddings": embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8),
                    "temp_speaker": temp_speaker,
                    "start": start,
                    "end": end,
                    "subseg_ids": subseg_ids
                }

            if not chunk_segment_embeddings:
                logger.debug(f"Чанк {chunk_start:.2f}-{chunk_end:.2f} не содержит валидных эмбеддингов")
                chunk_start = chunk_end + self.min_chunk_step
                continue

            all_segments.append(chunk_segment_embeddings)

            # Определение начала следующего чанка
            if adjusted_segments:
                last_segment = max(adjusted_segments, key=lambda x: x["end"])
                chunk_start = last_segment["end"]
                logger.info(f"Следующий чанк начинается с: {chunk_start:.2f} сек")
            else:
                chunk_start = chunk_end + self.min_chunk_step
                logger.info(f"Следующий чанк начинается с: {chunk_start:.2f} сек (без сегментов, шаг {self.min_chunk_step})")

            if chunk_start >= audio_duration:
                logger.info(f"Достигнут конец аудио: chunk_start={chunk_start:.2f}, audio_duration={audio_duration:.2f}")
                break

        if not all_segments:
            logger.warning("Не удалось получить валидные сегменты для всего аудио")
            # Возвращаем временные метки из ASR с меткой "1"
            fallback_segments = []
            for segment in asr_data:
                for word in segment['data']['result']:
                    if word['end'] - word['start'] >= self.min_duration:
                        fallback_segments.append({
                            "start": word['start'],
                            "end": word['end'],
                            "speaker": "1"
                        })
            logger.info(f"Возвращено {len(fallback_segments)} временных сегментов на основе ASR с меткой '1'")
            return fallback_segments

        # Глобальная кластеризация подсегментов
        logger.info("Начало глобальной кластеризации подсегментов")
        all_subsegs = []
        all_embeddings = []
        subseg_to_seg = []
        for chunk_idx, chunk_segments in enumerate(all_segments):
            for seg_id, seg_data in chunk_segments.items():
                embeddings = seg_data["embeddings"]
                subseg_ids = seg_data["subseg_ids"]
                for i, subseg_id in enumerate(subseg_ids):
                    all_subsegs.append({
                        "subseg_id": subseg_id,
                        "seg_id": seg_id,
                        "chunk_idx": chunk_idx,
                        "temp_speaker": seg_data["temp_speaker"],
                        "start": seg_data["start"],
                        "end": seg_data["end"]
                    })
                    all_embeddings.append(embeddings[i])
                    subseg_to_seg.append(seg_id)

        if not all_embeddings:
            logger.warning("Не найдено подсегментов для кластеризации")
            # Возвращаем временные метки из segment_audio с меткой "1"
            fallback_segments = []
            for chunk_segments in all_segments:
                for seg_id, seg_data in chunk_segments.items():
                    if seg_data["end"] - seg_data["start"] >= self.min_duration:
                        fallback_segments.append({
                            "start": seg_data["start"],
                            "end": seg_data["end"],
                            "speaker": "1"
                        })
            logger.info(f"Возвращено {len(fallback_segments)} временных сегментов на основе segment_audio с меткой '1'")
            return fallback_segments

        all_embeddings = np.vstack(all_embeddings)
        logger.info(f"Собрано {len(all_embeddings)} подсегментов для кластеризации")

        # Вычисление матрицы косинусных расстояний
        logger.info("Вычисление матрицы косинусных расстояний")
        distance_matrix = cosine_distances(all_embeddings)
        logger.info(f"Размер матрицы расстояний: {distance_matrix.shape}")

        # Кластеризация
        if num_speakers == -1:
            logger.info("Использование HDBSCAN для кластеризации (num_speakers=-1)")
            # Преобразование матрицы расстояний в float64 для HDBSCAN
            distance_matrix_hdbscan = distance_matrix.astype(np.float64)
            clusterer = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=2, min_samples=3)
            cluster_labels = clusterer.fit_predict(distance_matrix_hdbscan)
        else:
            logger.info(f"Использование AHC для кластеризации (num_speakers={num_speakers})")
            clusterer = AgglomerativeClustering(
                n_clusters=num_speakers,
                metric="precomputed",
                linkage="average"
            )
            cluster_labels = clusterer.fit_predict(distance_matrix)

        # Проверка числа кластеров
        unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
        logger.info(f"Получено {len(unique_labels)} кластеров (без учета шума)")
        if len(unique_labels) == 0:
            logger.warning("Не удалось выделить валидные кластеры")
            # Возвращаем временные метки из segment_audio с меткой "1"
            fallback_segments = []
            for chunk_segments in all_segments:
                for seg_id, seg_data in chunk_segments.items():
                    if seg_data["end"] - seg_data["start"] >= self.min_duration:
                        fallback_segments.append({
                            "start": seg_data["start"],
                            "end": seg_data["end"],
                            "speaker": "1"
                        })
            logger.info(f"Возвращено {len(fallback_segments)} временных сегментов на основе segment_audio с меткой '1'")
            return fallback_segments

        # Присвоение меток сегментам
        seg_to_labels = {}
        for i, (subseg_info, label) in enumerate(zip(all_subsegs, cluster_labels)):
            seg_id = subseg_info["seg_id"]
            if seg_id not in seg_to_labels:
                seg_to_labels[seg_id] = []
            seg_to_labels[seg_id].append(label)

        segments = []
        for seg_id, labels in seg_to_labels.items():
            chunk_idx = [i for i, chunk in enumerate(all_segments) if seg_id in chunk][0]
            seg_data = all_segments[chunk_idx][seg_id]
            # Исключаем шум (label=-1 для HDBSCAN)
            valid_labels = [label for label in labels if label >= 0]
            if not valid_labels:
                logger.warning(f"Сегмент {seg_id} содержит только шумовые подсегменты, используется метка '1'")
                segments.append({
                    "start": seg_data["start"],
                    "end": seg_data["end"],
                    "speaker": "1"
                })
                continue
            # Выбираем наиболее частую метку
            label_counts = np.bincount(valid_labels)
            dominant_label = np.argmax(label_counts)
            segments.append({
                "start": seg_data["start"],
                "end": seg_data["end"],
                "speaker": f"speaker_{dominant_label}"
            })
            logger.info(f"Сегмент {seg_id}: start={seg_data['start']:.2f}, end={seg_data['end']:.2f}, speaker=speaker_{dominant_label}, распределение меток: {label_counts.tolist()}")

        segments.sort(key=lambda x: x["start"])
        logger.info(f"Сегменты после кластеризации (всего {len(segments)}):")
        for seg in segments:
            logger.info(f"Сегмент: start={seg['start']:.2f}, end={seg['end']:.2f}, speaker={seg['speaker']}")

        total_diarize_time = time.perf_counter() - start_time
        logger.debug(f"Процедура: Общая диаризация - {total_diarize_time:.4f} сек")
        return segments

    async def merge_segments(self, diarized_segments: list[dict]) -> list[dict]:
        start_time = time.perf_counter()

        if not diarized_segments:
            return []

        # Сортировка сегментов по времени начала
        diarized_segments = sorted(diarized_segments, key=lambda x: x["start"])
        logger.info(f"Объединение сегментов (всего {len(diarized_segments)}):")

        # Объединение соседних сегментов одного спикера
        merged = []
        current_phrase = None
        for segment in diarized_segments:
            if current_phrase is None:
                current_phrase = {"start": segment["start"], "end": segment["end"], "speaker": segment["speaker"]}
            elif (segment["start"] - current_phrase["end"] <= self.max_phrase_gap and
                  segment["speaker"] == current_phrase["speaker"]):
                current_phrase["end"] = segment["end"]
            else:
                merged.append(current_phrase)
                current_phrase = {"start": segment["start"], "end": segment["end"], "speaker": segment["speaker"]}

        if current_phrase is not None:
            merged.append(current_phrase)

        # Устранение наложений
        final_segments = []
        merged = sorted(merged, key=lambda x: x["start"])
        i = 0
        while i < len(merged):
            current_seg = merged[i]
            overlaps = []
            j = i + 1
            while j < len(merged):
                next_seg = merged[j]
                if next_seg["start"] < current_seg["end"]:
                    overlaps.append((j, next_seg))
                else:
                    break
                j += 1

            if not overlaps:
                final_segments.append(current_seg)
                logger.info(f"Сегмент добавлен: start={current_seg['start']:.2f}, end={current_seg['end']:.2f}, speaker={current_seg['speaker']}")
                i += 1
                continue

            # Разрешение наложений
            for j, overlap_seg in overlaps:
                overlap_start = max(current_seg["start"], overlap_seg["start"])
                overlap_end = min(current_seg["end"], overlap_seg["end"])
                split_point = (overlap_start + overlap_end) / 2
                logger.info(f"Обнаружено наложение: {current_seg['start']:.2f}-{current_seg['end']:.2f} (speaker {current_seg['speaker']}) и {overlap_seg['start']:.2f}-{overlap_seg['end']:.2f} (speaker {overlap_seg['speaker']}), разделение на {split_point:.2f}")

                if current_seg["start"] < split_point:
                    final_segments.append({"start": current_seg["start"], "end": split_point, "speaker": current_seg["speaker"]})
                    logger.info(f"Сегмент после разделения: start={current_seg['start']:.2f}, end={split_point:.2f}, speaker={current_seg['speaker']}")
                if split_point < overlap_seg["end"]:
                    merged[j] = {"start": split_point, "end": overlap_seg["end"], "speaker": overlap_seg["speaker"]}
                    logger.info(f"Сегмент после разделения: start={split_point:.2f}, end={overlap_seg['end']:.2f}, speaker={overlap_seg['speaker']}")

                current_seg["end"] = split_point
            i += 1

        # Фильтрация по минимальной длительности
        final_segments = [seg for seg in final_segments if seg["end"] - seg["start"] >= self.min_duration]
        logger.info(f"После фильтрации по минимальной длительности: {len(final_segments)} сегментов")

        # Преобразование меток спикеров в числовые
        unique_speakers = sorted(set(seg["speaker"] for seg in final_segments))
        speaker_mapping = {speaker: str(idx) for idx, speaker in enumerate(unique_speakers)}
        logger.info(f"Карта спикеров: {speaker_mapping}")
        for seg in final_segments:
            seg["speaker"] = speaker_mapping[seg["speaker"]]

        merge_time = time.perf_counter() - start_time
        logger.debug(f"Процедура: Объединение сегментов - {merge_time:.4f} сек")
        return final_segments

    async def diarize_and_merge(self, audio: np.ndarray, num_speakers: int, asr_data: list[dict],
                                filter_cutoff: float = 50, filter_order: int = 10) -> list[dict]:
        start_time = time.perf_counter()

        raw_result = await self.diarize(audio, num_speakers, asr_data, filter_cutoff, filter_order)
        merged_result = await self.merge_segments(raw_result)

        total_time = time.perf_counter() - start_time
        logger.debug(f"Процедура: Полная диаризация и объединение - {total_time:.4f} сек")
        return merged_result

async def main():
    num_speakers = -1
    use_gpu_diar = False
    batch_size = 32
    max_cpu_workers = 0
    speaker_model_path = Path("../models/DIARISATION_model/voxceleb_resnet34_LM.onnx")
    segmentation_model_path = Path("../models/Segmentation/model.onnx")
    audio_path = "../trash/replit/Luxury_Girl_audio.wav"
    asr_file = "../trash/replit/Luxury_Girl_asr.json"

    audio = AudioSegment.from_file(audio_path)
    samples_float32 = await load_and_preprocess_audio(audio)

    asr_data = load_asr_data(asr_file)

    diarizer = Diarizer(
        embedding_model_path=str(speaker_model_path),
        segmentation_model_path=str(segmentation_model_path),
        batch_size=batch_size,
        cpu_workers=max_cpu_workers,
        use_gpu=use_gpu_diar,
    )

    result = await diarizer.diarize_and_merge(
        audio=samples_float32,
        asr_data=asr_data,
        num_speakers=num_speakers
    )

    for r in result:
        print(f"Спикер {r['speaker']}: {r['start']:.2f} - {r['end']:.2f} сек")

if __name__ == "__main__":
    asyncio.run(main())