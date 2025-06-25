from concurrent.futures import ThreadPoolExecutor
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
import logging as logger

class Diarizer:
    def __init__(self, embedding_model_path: str,
                 sample_rate: int = 16000,
                 use_gpu: bool = False,
                 max_phrase_gap: float = 0.5,
                 min_duration: float = 0.15,
                 filter_cutoff: int = 100,
                 filter_order: int = 10,
                 batch_size: int = 1,
                 cpu_workers: int = 0):
        self.sample_rate = sample_rate
        self.winlen = 0.025
        self.winstep = 0.01
        self.num_mel_bins = 80
        self.nfft = 512
        self.filter_cutoff = filter_cutoff
        self.filter_order = filter_order
        self.max_phrase_gap = max_phrase_gap
        self.min_duration = min_duration

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

    async def diarize(self, audio_frames: np.ndarray, asr_results: list, num_speakers: int,
                      filter_cutoff: float, filter_order: int) -> list[dict]:
        start_time = time.perf_counter()

        logger.debug("Начало обработки ASR сегментов...")
        seg_start = time.perf_counter()
        segments = []

        for channel in asr_results.get(f"channel_{len(asr_results.keys())}", []):
            for result in channel["data"]["result"]:
                start = result["start"]
                end = result["end"]
                if end - start >= self.min_duration:
                    audio_segment = audio_frames[int(start * self.sample_rate):int(end * self.sample_rate)]
                    segments.append((start, end, audio_segment))
                else:
                    audio_segment = audio_frames[int((start-0.05) * self.sample_rate):int((end+0.05) * self.sample_rate)]
                    segments.append((start, end, audio_segment))

        seg_time = time.perf_counter() - seg_start
        logger.debug(f"Процедура: Обработка ASR сегментов - {seg_time:.4f} сек")

        if not segments:
            logger.debug("Сегменты не найдены")
            return []

        frame_shift = int(self.winstep * 1000)
        window_fs = int(1.5 * 1000) // frame_shift
        period_fs = int(0.75 * 1000) // frame_shift

        subsegs = []
        subseg_audios = []
        for start, end, audio in segments:
            audio = await self.highpass_filter(audio, cutoff=self.filter_cutoff, filter_order=self.filter_order)
            seg_id = f"{start:.3f}-{end:.3f}"
            fbank_feats = await self.extract_fbank(audio)
            tmp_subsegs, tmp_subseg_fbanks = await self.subsegment(fbank_feats, seg_id, window_fs, period_fs, frame_shift)
            subsegs.extend(tmp_subsegs)
            subseg_audios.extend(tmp_subseg_fbanks)

        logger.debug("Начало извлечения эмбеддингов...")
        emb_start = time.perf_counter()
        embeddings = await self.extract_embeddings(subseg_audios, subseg_cmn=True)
        emb_time = time.perf_counter() - emb_start
        logger.debug(f"Процедура: Извлечение эмбеддингов - {emb_time:.4f} сек")

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
                # random_state=2023,
                n_jobs=1
            ).fit_transform(embeddings)

            if num_speakers ==-1:
                clustering = AgglomerativeClustering(n_clusters=2, metric='cosine', linkage='average')
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
        logger.debug(f"Процедура: Общая диаризация - {total_diarize_time:.4f} сек")

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
        logger.debug(f"Процедура: Объединение сегментов - {merge_time:.4f} сек")

        return merged

    async def diarize_and_merge(self, audio_frames: np.ndarray, asr_results: list,
                                num_speakers: int, filter_cutoff: int = 50,
                                filter_order: int = 10) -> list[dict]:
        start_time = time.perf_counter()

        raw_result = await self.diarize(audio_frames, asr_results, num_speakers, filter_cutoff, filter_order)

        merged_result = await self.merge_segments(raw_result)

        total_time = time.perf_counter() - start_time
        logger.debug(f"Процедура: Полная диаризация и объединение - {total_time:.4f} сек")

        return merged_result

async def load_and_preprocess_audio(audio: AudioSegment, sample_rate: int = 16000) -> np.ndarray:
    start_time = time.perf_counter()

    if audio.frame_rate != sample_rate:
        audio = audio.set_frame_rate(sample_rate)
    if audio.channels > 1:
        audio = audio.split_to_mono()[0]
    samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
    samples_float32 = samples.astype(np.float32) / 32768.0

    load_time = time.perf_counter() - start_time
    logger.debug(f"В работу принято аудио продолжительностью {audio.duration_seconds} сек")
    logger.debug(f"Процедура: Загрузка и предобработка аудио - {load_time:.4f} сек")

    return samples_float32

async def main():
    num_speakers = 2
    max_phrase_gap = 0.1
    use_gpu_diar = False
    batch_size = 1
    max_cpu_workers = 0

    speaker_model_path = Path("../models/DIARISATION_model/voxceleb_gemini_dfresnet114_LM.onnx")
    audio_path = "../trash/secret.wav"


    audio = AudioSegment.from_file(audio_path)
    audio_frames = await load_and_preprocess_audio(audio)

    # Пример результата ASR
    asr_results = {
        "channel_1": [
            {
                "data": {
                    "result": [
                        {
                            "conf": 1,
                            "start": 1.28,
                            "end": 1.48,
                            "word": "алло"
                        },
                        {
                            "conf": 1,
                            "start": 2.2,
                            "end": 2.44,
                            "word": "алло"
                        },
                        {
                            "conf": 1,
                            "start": 2.64,
                            "end": 2.96,
                            "word": "наталья"
                        },
                        {
                            "conf": 1,
                            "start": 3.08,
                            "end": 3.6,
                            "word": "анатольевна"
                        },
                        {
                            "conf": 1,
                            "start": 3.72,
                            "end": 4,
                            "word": "добрый"
                        },
                        {
                            "conf": 1,
                            "start": 4.16,
                            "end": 4.36,
                            "word": "день"
                        },
                        {
                            "conf": 1,
                            "start": 5.16,
                            "end": 5.4,
                            "word": "добрый"
                        },
                        {
                            "conf": 1,
                            "start": 5.56,
                            "end": 5.72,
                            "word": "день"
                        },
                        {
                            "conf": 1,
                            "start": 6.36,
                            "end": 6.84,
                            "word": "екатерина"
                        },
                        {
                            "conf": 1,
                            "start": 6.96,
                            "end": 7.48,
                            "word": "руководитель"
                        },
                        {
                            "conf": 1,
                            "start": 7.6,
                            "end": 7.88,
                            "word": "группы"
                        },
                        {
                            "conf": 1,
                            "start": 8,
                            "end": 8.28,
                            "word": "отдела"
                        },
                        {
                            "conf": 1,
                            "start": 8.44,
                            "end": 9.12,
                            "word": "сопровождения"
                        },
                        {
                            "conf": 1,
                            "start": 9.76,
                            "end": 10.08,
                            "word": "наталья"
                        },
                        {
                            "conf": 1,
                            "start": 10.16,
                            "end": 10.72,
                            "word": "анатольевна"
                        },
                        {
                            "conf": 1,
                            "start": 10.88,
                            "end": 11.08,
                            "word": "звоню"
                        },
                        {
                            "conf": 1,
                            "start": 11.24,
                            "end": 11.24,
                            "word": "с"
                        },
                        {
                            "conf": 1,
                            "start": 11.36,
                            "end": 11.8,
                            "word": "весенними"
                        },
                        {
                            "conf": 1,
                            "start": 11.92,
                            "end": 12.4,
                            "word": "праздниками"
                        },
                        {
                            "conf": 1,
                            "start": 12.56,
                            "end": 12.64,
                            "word": "вас"
                        },
                        {
                            "conf": 1,
                            "start": 12.76,
                            "end": 12.92,
                            "word": "всех"
                        },
                        {
                            "conf": 1,
                            "start": 13.04,
                            "end": 13.52,
                            "word": "поздравить"
                        },
                        {
                            "conf": 1,
                            "start": 14.12,
                            "end": 14.48,
                            "word": "спасбо"
                        },
                        {
                            "conf": 1,
                            "start": 14.68,
                            "end": 15,
                            "word": "большое"
                        },
                        {
                            "conf": 1,
                            "start": 15.16,
                            "end": 15.56,
                            "word": "взаимно"
                        },
                        {
                            "conf": 1,
                            "start": 15.96,
                            "end": 16,
                            "word": "да"
                        },
                        {
                            "conf": 1,
                            "start": 16.24,
                            "end": 16.72,
                            "word": "спасибо"
                        },
                        {
                            "conf": 1,
                            "start": 17,
                            "end": 17.08,
                            "word": "как"
                        },
                        {
                            "conf": 1,
                            "start": 17.2,
                            "end": 17.2,
                            "word": "у"
                        },
                        {
                            "conf": 1,
                            "start": 17.32,
                            "end": 17.4,
                            "word": "вас"
                        },
                        {
                            "conf": 1,
                            "start": 17.52,
                            "end": 17.68,
                            "word": "дела"
                        },
                        {
                            "conf": 1,
                            "start": 17.88,
                            "end": 18,
                            "word": "как"
                        },
                        {
                            "conf": 1,
                            "start": 18.16,
                            "end": 18.72,
                            "word": "настроение"
                        },
                        {
                            "conf": 1,
                            "start": 18.92,
                            "end": 19.48,
                            "word": "весеннее"
                        },
                        {
                            "conf": 1,
                            "start": 20.48,
                            "end": 20.52,
                            "word": "ну"
                        },
                        {
                            "conf": 1,
                            "start": 20.76,
                            "end": 22.08,
                            "word": "сейасв"
                        },
                        {
                            "conf": 1,
                            "start": 24.36,
                            "end": 24.44,
                            "word": "да"
                        },
                        {
                            "conf": 1,
                            "start": 24.64,
                            "end": 24.68,
                            "word": "да"
                        },
                        {
                            "conf": 1,
                            "start": 24.84,
                            "end": 24.88,
                            "word": "да"
                        },
                        {
                            "conf": 1,
                            "start": 25,
                            "end": 25.64,
                            "word": "каждыа"
                        },
                        {
                            "conf": 1,
                            "start": 25.92,
                            "end": 26.24,
                            "word": "выходишь"
                        },
                        {
                            "conf": 1,
                            "start": 26.36,
                            "end": 26.4,
                            "word": "на"
                        },
                        {
                            "conf": 1,
                            "start": 26.52,
                            "end": 26.76,
                            "word": "работу"
                        },
                        {
                            "conf": 1,
                            "start": 26.88,
                            "end": 27.28,
                            "word": "солнышко"
                        },
                        {
                            "conf": 1,
                            "start": 27.4,
                            "end": 27.4,
                            "word": "и"
                        },
                        {
                            "conf": 1,
                            "start": 27.56,
                            "end": 27.64,
                            "word": "как"
                        },
                        {
                            "conf": 1,
                            "start": 27.72,
                            "end": 27.96,
                            "word": "будто"
                        },
                        {
                            "conf": 1,
                            "start": 28.08,
                            "end": 28.12,
                            "word": "бы"
                        },
                        {
                            "conf": 1,
                            "start": 28.24,
                            "end": 28.72,
                            "word": "настроение"
                        },
                        {
                            "conf": 1,
                            "start": 28.92,
                            "end": 29.12,
                            "word": "лучше"
                        }
                  ],
                  "text": "алло алло наталья анатольевна добрый день добрый день екатерина руководитель группы отдела сопровождения наталья анатольевна звоню с весенними праздниками вас всех поздравить спасбо большое взаимно да спасибо как у вас дела как настроение весеннее ну сейасв да да да каждыа выходишь на работу солнышко и как будто бы настроение лучше"
                }
            }
        ]
    }

    diarizer = Diarizer(
        embedding_model_path=str(speaker_model_path),
        max_phrase_gap=max_phrase_gap,
        batch_size=batch_size,
        cpu_workers=max_cpu_workers,
        use_gpu=use_gpu_diar,
    )

    result = await diarizer.diarize_and_merge(
        audio_frames,
        asr_results,
        num_speakers=num_speakers,
        filter_cutoff=50,
        filter_order=10
    )

    for r in result:
        print(f"Спикер {r['speaker']}: {r['start']:.2f} - {r['end']:.2f} сек")

if __name__ == "__main__":
    asyncio.run(main())