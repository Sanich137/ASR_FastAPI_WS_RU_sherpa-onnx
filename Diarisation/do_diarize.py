from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
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
import time
from umap import UMAP
from hdbscan import HDBSCAN
from utils.do_logging import logger
from utils.pre_start_init import paths
from utils.resamppling import resample_audiosegment

# import logging as logger

# Здесь SileroVAD остаётся только для тестов Диаризации на бою используется класс из do_vad
class SileroVAD:
    def __init__(self, onnx_path: Path, use_gpu=False):
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        self.sample_rate = 16000
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.frame_size = 512
        self.prob_level = 0.5
        self.set_mode(3)
        # Параметры сегментации (VAD)
        self.min_duration = 0.15  # Минимальная длительность речевого сегмента (сек)
        self.max_vad_gap = 1  # Максимальный промежуток между сегментами для их объединения (сек)

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 4
        session_options.inter_op_num_threads = 0
        session_options.intra_op_num_threads = 0

        self.session = ort.InferenceSession(path_or_bytes=onnx_path,
                                            sess_options=session_options,
                                            providers=providers)

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
            self.prob_level = 0.3
        elif mode == 5:
            self.prob_level = 0.15

    def is_speech(self, audio_frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """Обработка аудио-фрейма (миничанка)
                Args:
                    :param audio_frame: 1D numpy array размером 512 сэмплов
                    :param sample_rate:
                Returns:
                    float: вероятность, что чанк это речь
                    state: ntreott состояние модели.
                """
        if len(audio_frame) != self.frame_size:
            audio_frame = np.pad(audio_frame, (0, self.frame_size - len(audio_frame)), mode='constant')[
                          :self.frame_size]

        inputs = {
            'input': audio_frame.reshape(1, -1).astype(np.float32),
            'state': self.state,
            'sr': np.array(16000, dtype=np.int64)
        }

        outputs = self.session.run(['output', 'stateN'], inputs)
        self.state = outputs[1]
        prob = float(outputs[0][0, 0])

        return prob, self.state

    def get_speech_segments(self, audio_frames: np.ndarray) -> list[tuple]:
        if len(audio_frames.shape) == 2:
            audio_frames = audio_frames.flatten()

        audio_length_samples = len(audio_frames)
        window_size_samples = self.frame_size
        sample_rate = self.sample_rate

        # Параметры из get_speech_timestamps
        threshold = self.prob_level
        neg_threshold = threshold - 0.15
        min_speech_duration_ms = int(self.min_duration * 1000)
        min_silence_duration_ms = int(self.max_vad_gap * 1000)
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


def process_batch_sync(batch):
    """Синхронная функция для обработки батча в отдельном процессе."""
    session = ort.InferenceSession(
        path_or_bytes=paths.get("diar_speaker_model_path"),  # Укажи актуальный путь к модели
        providers=['CPUExecutionProvider'],
        sess_options=ort.SessionOptions()
    )
    return session.run(output_names=['embs'], input_feed={'feats': batch})[0].squeeze()

class Diarizer:
    def __init__(self, embedding_model_path: str,
                 vad,
                 sample_rate: int = 16000,
                 use_gpu: bool = False,
                 max_phrase_gap: float = 0.5,
                 min_duration: float = 0.15,
                 batch_size: int = 1,
                 cpu_workers: int = 0):

        self.vad = vad
        self.sample_rate = sample_rate
        self.winlen = 0.025
        self.winstep = 0.01
        self.num_mel_bins = 80
        self.nfft = 512
        self.max_phrase_gap = max_phrase_gap
        self.min_duration = min_duration

        if use_gpu:
            self.batch_size = batch_size
            self.max_cpu_workers = 1
        else:
            self.batch_size = 1
            self.max_cpu_workers = os.cpu_count() - 1 if cpu_workers < 1 else cpu_workers

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 4
        session_options.enable_profiling = False
        session_options.inter_op_num_threads = 0
        session_options.intra_op_num_threads = 0
        session_options.enable_mem_pattern = False
        self.embedding_session = ort.InferenceSession(
            embedding_model_path,
            sess_options=session_options,
            providers=providers
        )
        self.table = {}
        logger.debug(f"Используемые для диаризации провайдеры {self.embedding_session.get_providers()}")

    async def highpass_filter(self, signal: np.ndarray, cutoff: float, filter_order: int) -> np.ndarray:
        sos = butter(filter_order, cutoff, btype='high', fs=self.sample_rate, output='sos')
        return sosfilt(sos, signal)

    async def extract_fbank(self, audio):
        feats, _ = fbank(audio, samplerate=self.sample_rate, winlen=self.winlen, winstep=self.winstep,
                         nfilt=self.num_mel_bins, nfft=self.nfft, winfunc=np.hamming)
        feats = np.log(np.maximum(feats, 1e-8)).astype(np.float32)
        return feats

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

        return np.vstack(embeddings)

    async def merge_subsegments(self, subsegs, labels):
        merged = []
        current_start, current_end, current_label = None, None, None
        for subseg, label in zip(subsegs, labels):
            seg_id, start_frame, end_frame = subseg.rsplit('-', 2)
            start = float(seg_id.split('-')[0]) + int(start_frame) * 0.01
            end = float(seg_id.split('-')[0]) + int(end_frame) * 0.01
            if current_label is None:
                current_start, current_end, current_label = start, end, label
            elif label == current_label and start - current_end <= self.max_phrase_gap:
                logger.debug(f"Объединяем подсегменты: {current_end:.2f} -> {start:.2f}, пауза: {start - current_end:.2f} сек")
                current_end = end
            else:
                merged.append({"start": current_start, "end": current_end, "speaker": current_label})
                current_start, current_end, current_label = start, end, label
        if current_start is not None:
            merged.append({"start": current_start, "end": current_end, "speaker": current_label})
        return merged

    async def diarize(self, audio_frames: np.ndarray,
                      num_speakers: int,
                      filter_cutoff: float,
                      filter_order: int,
                      vad_sensity: int) -> list[dict]:
        await self.vad.reset_state()
        self.vad.set_mode(vad_sensity)
        start_time = time.perf_counter()

        logger.debug("Начало сегментации...")
        seg_start = time.perf_counter()
        segments = await self.vad.get_speech_segments(audio_frames)
        seg_time = time.perf_counter() - seg_start
        logger.debug(f"Процедура: Сегментация (get_speech_segments) - {seg_time:.4f} сек")

        if not segments:
            logger.debug("Речь не обнаружена")
            return []

        segments = [seg for seg in segments if seg[1] - seg[0] >= self.min_duration]
        if not segments:
            logger.debug("После фильтрации сегментов не осталось")
            return []

        frame_shift = int(self.winstep * 1000)
        window_fs = int(1.5 * 1000) // frame_shift
        period_fs = int(0.75 * 1000) // frame_shift

        subsegs = []
        subseg_audios = []
        for start, end, audio in segments:
            audio = await self.highpass_filter(audio, cutoff=filter_cutoff, filter_order=filter_order)
            seg_id = f"{start:.3f}-{end:.3f}"
            fbank_feats = await self.extract_fbank(audio)
            tmp_subsegs, tmp_subseg_fbanks = await self.subsegment(fbank_feats, seg_id, window_fs, period_fs, frame_shift)
            subsegs.extend(tmp_subsegs)
            subseg_audios.extend(tmp_subseg_fbanks)

        logger.debug("Начало извлечения эмбеддингов...")
        emb_start = time.perf_counter()
        embeddings = await self.extract_embeddings(subseg_audios, subseg_cmn=True)
        emb_time = time.perf_counter() - emb_start
        logger.debug(f"Процедура: Извлечение эмбеддингов (extract_embeddings) - {emb_time:.4f} сек")

        if len(embeddings) == 0:
            logger.debug("Не удалось извлечь валидные эмбеддинги")
            return []

        logger.debug(f"Количество эмбеддингов: {len(embeddings)}")

        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        logger.debug("Начало кластеризации...")
        clust_start = time.perf_counter()
        if len(embeddings) <= 2:
            labels = [0] * len(embeddings)
        else:
            n_neighbors = min(5, len(embeddings) - 1)
            umap_embeddings = UMAP(
                n_components=min(32, len(embeddings) - 2),
                metric='cosine',
                n_neighbors=n_neighbors,
                min_dist=0.1,
                random_state=2023,
                n_jobs=1
            ).fit_transform(embeddings)

            if num_speakers >= 2:
                clustering = AgglomerativeClustering(n_clusters=num_speakers, metric='cosine', linkage='average')
                labels = clustering.fit_predict(umap_embeddings)
                logger.debug(f"Использовано {num_speakers} спикеров (Agglomerative Clustering)")
            else:
                labels = HDBSCAN(min_cluster_size=3).fit_predict(umap_embeddings)
                logger.debug(f"Всего labels: {len(labels)}")
                if np.all(labels == -1):
                    logger.debug("Все точки помечены как шум, предполагается один спикер")
                    labels = np.zeros_like(labels)
                else:
                    unique_labels = np.unique(labels)
                    if len(unique_labels) > 1:
                        score = silhouette_score(embeddings, labels, metric='cosine')
                        logger.debug(f"Силуэтный коэффициент: {score:.4f}")
                        if score < 0.1:
                            logger.debug("Низкий силуэтный коэффициент, предполагается один спикер")
                            labels = np.zeros_like(labels)
                    else:
                        logger.debug("Найден только один кластер, силуэтный коэффициент не вычисляется")

        clust_time = time.perf_counter() - clust_start
        logger.debug(f"Процедура: Кластеризация - {clust_time:.4f} сек")
        logger.debug(f"Число кластеров: {len(np.unique(labels))}")

        logger.debug("Начало объединения подсегментов...")
        merge_start = time.perf_counter()
        merged_segments = await self.merge_subsegments(subsegs, labels)
        merged_segments = [seg for seg in merged_segments if seg["end"] - seg["start"] >= self.min_duration]
        merge_time = time.perf_counter() - merge_start
        logger.debug(f"Процедура: Объединение подсегментов - {merge_time:.4f} сек")

        total_diarize_time = time.perf_counter() - start_time
        logger.debug(f"Процедура: Общая диаризация (diarize) - {total_diarize_time:.4f} сек")

        return merged_segments

    async def merge_segments(self, diarized_segments: list[dict]) -> list[dict]:
        start_time = time.perf_counter()

        if not diarized_segments:
            return []

        diarized_segments = sorted(diarized_segments, key=lambda x: x["start"])
        merged = []
        current_phrase = None

        for segment in diarized_segments:
            if current_phrase is None:
                current_phrase = {"start": segment["start"], "end": segment["end"], "speaker": str(segment["speaker"])}
            elif (segment["start"] - current_phrase["end"] <= self.max_phrase_gap and
                  segment["speaker"] == current_phrase["speaker"]):
                current_phrase["end"] = segment["end"]
            else:
                merged.append(current_phrase)
                current_phrase = {"start": segment["start"], "end": segment["end"], "speaker": str(segment["speaker"])}

        if current_phrase is not None:
            merged.append(current_phrase)

        merge_time = time.perf_counter() - start_time
        logger.debug(f"Процедура: Объединение сегментов (merge_segments) - {merge_time:.4f} сек")

        return merged

    async def diarize_and_merge(self, audio_frames: np.ndarray, num_speakers: int,
                                filter_cutoff: int = 50, filter_order: int = 10,
                                vad_sensity: int = 3) -> list[dict]:
        start_time = time.perf_counter()

        raw_result = await self.diarize(audio_frames, num_speakers, filter_cutoff, filter_order, vad_sensity)
        merged_result = await self.merge_segments(raw_result)

        total_time = time.perf_counter() - start_time
        logger.debug(f"Процедура: Полная диаризация и объединение (diarize_and_merge) - {total_time:.4f} сек")

        return merged_result

async def load_and_preprocess_audio(audio: AudioSegment, target_frame_size: int = 512, sample_rate: int = 16000) -> np.ndarray:
    """
    :param audio: AudioSegment аудио данные
    :param target_frame_size: int = 512 требования для работы SILERO VAD
    :param sample_rate: int = 16000 требования для работы SILERO VAD
    """

    start_time = time.perf_counter()

    if audio.frame_rate != sample_rate:
        audio = await resample_audiosegment(audio,sample_rate)
    if audio.channels > 1:
        audio = audio.split_to_mono()[1][0:60000]
    samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
    samples_float32 = samples.astype(np.float32) / 32768.0
    num_frames = len(samples_float32) // target_frame_size
    frames = samples_float32[:num_frames * target_frame_size].reshape(num_frames, target_frame_size)

    load_time = time.perf_counter() - start_time
    logger.debug(f"В работу принято аудио продолжительностью {audio.duration_seconds} сек")
    logger.debug(f"Процедура: Загрузка и предобработка аудио (load_and_preprocess_audio) - {load_time:.4f} сек")

    return frames


# if __name__ == "__main__":
#     # Todo - перевести тест в async
#
#     # Параметры диаризации и кластеризации
#     num_speakers = -1  # Количество спикеров (-1 для автоматического определения)
#
#     # Параметры извлечения ембеддингов
#     max_phrase_gap = 1  # расстояние между фразами для объединения в один кластер.
#     use_gpu_diar = False  # По возможности использовать графический процессор для вычислений
#     batch_size = 32  # Размер батча для извлечения эмбеддингов при работе GPU
#     max_cpu_workers = 0  # Количество потоков для извлечения эмбедингов при использовании CPU
#
#     # # Параметры сегментации (VAD)
#     vad_mode = 4  # Режим чувствительности VAD (1, 2, 3, 4, 5)
#     use_gpu_vad = False  # По возможности использовать графический процессор для вычислений
#
#     vad_model_path = Path("../models/VAD_silero_v5/silero_vad.onnx")
#     speaker_model_path =  Path("../models/Diar_model/voxblink2_samresnet100_ft.onnx")
#
#     audio_path = "../trash/Роман.mp3"
#
#     vad = SileroVAD(vad_model_path, use_gpu=use_gpu_vad)
#     vad.set_mode(vad_mode)
#
#
#     audio = AudioSegment.from_file(audio_path)
#     audio_frames = await load_and_preprocess_audio(audio)
#     #Todo - в load_and_preprocess_audio должен передаваться аудиосегмент.
#
#     diarizer = Diarizer(embedding_model_path=str(speaker_model_path),
#                         vad=vad,
#                         max_phrase_gap=max_phrase_gap,
#                         batch_size=batch_size,
#                         cpu_workers=max_cpu_workers,
#                         use_gpu=use_gpu_diar,
#                         )
#
#     result = await diarizer.diarize_and_merge(
#         audio_frames,
#         num_speakers=num_speakers,
#         filter_cutoff=50,
#         filter_order=10,
#         vad_sensity=vad_mode
#     )
#
#     for r in result:
#         print(f"Спикер {r['speaker']}: {r['start']:.2f} - {r['end']:.2f} сек")

    for r in result:
        print(f"Спикер {r['speaker']}: {r['start']:.2f} - {r['end']:.2f} сек")

if __name__ == "__main__":
    asyncio.run(main())