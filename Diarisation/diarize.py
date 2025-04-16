import numpy as np
from pydub import AudioSegment
from pathlib import Path
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity
from python_speech_features import fbank
from scipy.signal import butter, sosfilt
from datetime import datetime as dt
from sklearn.metrics import silhouette_score
import time
import umap
import hdbscan

class SileroVAD:
    def __init__(self, onnx_path: Path, sample_rate: int = 16000, use_gpu=False):
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        session_options.inter_op_num_threads = 2
        session_options.intra_op_num_threads = 2

        self.session = ort.InferenceSession(path_or_bytes=onnx_path,
                                            sess_options=session_options,
                                            providers=providers)
        self.sample_rate = sample_rate
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.frame_size = 512
        self.prob_level = 0.5
        self.set_mode(3)

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
            self.prob_level = 0.4
        elif mode == 5:
            self.prob_level = 0.3

    def is_speech(self, audio_frame: np.ndarray, sample_rate=None) -> bool:
        if len(audio_frame) != self.frame_size:
            raise ValueError(f"Ожидается фрейм размером {self.frame_size}, получен {len(audio_frame)}")

        inputs = {
            'input': audio_frame.reshape(1, -1).astype(np.float32),
            'state': self.state,
            'sr': np.array(self.sample_rate, dtype=np.int64)
        }

        outputs = self.session.run(['output', 'stateN'], inputs)
        self.state = outputs[1]
        prob = float(outputs[0][0, 0])

        return prob >= self.prob_level, self.state

    def get_speech_segments(self, audio_frames: np.ndarray, min_duration: float, max_gap: float) -> list[tuple]:
        sample_rate = self.sample_rate
        frame_size = self.frame_size

        if len(audio_frames.shape) == 2:
            audio_frames = audio_frames.flatten()

        num_samples = len(audio_frames)
        num_frames = num_samples // frame_size
        if num_samples % frame_size != 0:
            padding = frame_size - (num_samples % frame_size)
            audio_frames = np.pad(audio_frames, (0, padding), mode='constant')

        frames = audio_frames.reshape(-1, frame_size)
        self.reset_state()
        speech_probs = []

        for frame in frames:
            is_speech_flag, new_state = self.is_speech(frame)
            self.state = new_state
            speech_probs.append(1.0 if is_speech_flag else 0.0)

        segments = []
        start = None
        for i, prob in enumerate(speech_probs):
            if prob >= self.prob_level:
                if start is None:
                    start = i * frame_size / sample_rate
            else:
                if start is not None:
                    end = i * frame_size / sample_rate
                    if end - start >= min_duration:
                        audio_segment = audio_frames[int(start * sample_rate):int(end * sample_rate)]
                        segments.append((start, end, audio_segment))
                    start = None

        if start is not None:
            end = len(audio_frames) / sample_rate
            if end - start >= min_duration:
                audio_segment = audio_frames[int(start * sample_rate):int(end * sample_rate)]
                segments.append((start, end, audio_segment))

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


class Diarizer:
    def __init__(self, embedding_model_path: str, vad, sample_rate: int = 16000, use_gpu: bool = False):
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        so.enable_profiling = False
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.embedding_session = ort.InferenceSession(embedding_model_path,
                                                      sess_options=so,
                                                      providers=providers)
        self.vad = vad
        self.sample_rate = sample_rate
        print(self.embedding_session.get_providers())

    def highpass_filter(self, signal: np.ndarray, cutoff: float, filter_order: int) -> np.ndarray:
        sos = butter(filter_order, cutoff, btype='high', fs=self.sample_rate, output='sos')
        return sosfilt(sos, signal)

    def extract_fbank(self, audio, winlen, winstep, num_mel_bins, nfft):
        feats, _ = fbank(audio, samplerate=self.sample_rate, winlen=winlen, winstep=winstep,
                         nfilt=num_mel_bins, nfft=nfft, winfunc=np.hamming)
        feats = np.log(np.maximum(feats, 1e-8)).astype(np.float32)
        return feats

    def subsegment(self, fbank, seg_id, window_fs, period_fs, frame_shift):
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

    def extract_embeddings(self, fbanks, batch_size, subseg_cmn):
        fbanks_array = np.stack(fbanks)
        if subseg_cmn:
            fbanks_array = fbanks_array - np.mean(fbanks_array, axis=1, keepdims=True)
        embeddings = []
        for i in range(0, fbanks_array.shape[0], batch_size):
            batch_feats = fbanks_array[i:i + batch_size]
            batch_embs = self.embedding_session.run(output_names=['embs'], input_feed={'feats': batch_feats})[0].squeeze()
            embeddings.append(batch_embs)
        return np.vstack(embeddings)

    def merge_subsegments(self, subsegs, labels):
        merged = []
        current_start, current_end, current_label = None, None, None
        for subseg, label in zip(subsegs, labels):
            seg_id, start_frame, end_frame = subseg.rsplit('-', 2)
            start = float(seg_id.split('-')[0]) + int(start_frame) * 0.01
            end = float(seg_id.split('-')[0]) + int(end_frame) * 0.01
            if current_label == label:
                current_end = end
            else:
                if current_start is not None:
                    merged.append({"start": current_start, "end": current_end, "speaker": current_label})
                current_start, current_end, current_label = start, end, label
        if current_start is not None:
            merged.append({"start": current_start, "end": current_end, "speaker": current_label})
        return merged

    def diarize(self, audio_frames: np.ndarray,
                num_speakers: int,
                cluster_threshold: float,
                min_duration: float,
                max_gap: float,
                winlen: float,
                winstep: float,
                num_mel_bins: int,
                nfft: int,
                merge_cluster_threshold: float,
                filter_cutoff: float,
                filter_order: int,
                batch_size: int) -> list[dict]:
        start_time = time.perf_counter()

        # 1. Сегментация VAD
        print("Начало сегментации...")
        seg_start = time.perf_counter()
        segments = self.vad.get_speech_segments(audio_frames, min_duration, max_gap)
        seg_time = time.perf_counter() - seg_start
        print(f"Процедура: Сегментация (get_speech_segments) - {seg_time:.4f} сек")

        if not segments:
            print("Речь не обнаружена")
            return []

        segments = [seg for seg in segments if seg[1] - seg[0] >= min_duration]
        if not segments:
            print("После фильтрации сегментов не осталось")
            return []

        # 2. Разделение на подсегменты
        frame_shift = int(winstep * 1000)  # в мс
        window_fs = int(1.5 * 1000) // frame_shift  # окно 1.5 сек
        period_fs = int(0.75 * 1000) // frame_shift  # шаг 0.75 сек

        subsegs = []
        subseg_audios = []
        for start, end, audio in segments:
            audio = self.highpass_filter(audio, cutoff=filter_cutoff, filter_order=filter_order)
            seg_id = f"{start:.3f}-{end:.3f}"
            fbank_feats = self.extract_fbank(audio, winlen, winstep, num_mel_bins, nfft)
            tmp_subsegs, tmp_subseg_fbanks = self.subsegment(fbank_feats, seg_id, window_fs, period_fs, frame_shift)
            subsegs.extend(tmp_subsegs)
            subseg_audios.extend(tmp_subseg_fbanks)

        # 3. Извлечение эмбеддингов
        print("Начало извлечения эмбеддингов...")
        emb_start = time.perf_counter()
        embeddings = self.extract_embeddings(subseg_audios, batch_size, subseg_cmn=True)
        emb_time = time.perf_counter() - emb_start
        print(f"Процедура: Извлечение эмбеддингов (extract_embeddings) - {emb_time:.4f} сек")

        if len(embeddings) == 0:
            print("Не удалось извлечь валидные эмбеддинги")
            return []

        print(f"Количество эмбеддингов: {len(embeddings)}")

        # 4. Кластеризация с UMAP + HDBSCAN
        print("Начало кластеризации...")
        clust_start = time.perf_counter()
        if len(embeddings) <= 2:
            labels = [0] * len(embeddings)
        else:
            umap_embeddings = umap.UMAP(n_components=min(32, len(embeddings) - 2),
                                        metric='cosine',
                                        n_neighbors=16,
                                        min_dist=0.05,
                                        random_state=2023).fit_transform(embeddings)
            labels = hdbscan.HDBSCAN(min_cluster_size=4).fit_predict(umap_embeddings)

        if len(np.unique(labels)) > 1:
            score = silhouette_score(embeddings, labels, metric='cosine')
            print(f"Силуэтный коэффициент: {score:.4f}")
        else:
            print("Найден только один кластер, силуэтный коэффициент не вычисляется")
        clust_time = time.perf_counter() - clust_start
        print(f"Процедура: Кластеризация - {clust_time:.4f} сек")
        print(f"Число кластеров: {len(np.unique(labels))}")

        # 5. Объединение подсегментов
        print("Начало объединения подсегментов...")
        merge_start = time.perf_counter()
        merged_segments = self.merge_subsegments(subsegs, labels)
        merge_time = time.perf_counter() - merge_start
        print(f"Процедура: Объединение подсегментов - {merge_time:.4f} сек")

        total_diarize_time = time.perf_counter() - start_time
        print(f"Процедура: Общая диаризация (diarize) - {total_diarize_time:.4f} сек")

        return merged_segments

    def merge_segments(self, diarized_segments: list[dict], max_phrase_gap: float) -> list[dict]:
        start_time = time.perf_counter()

        if not diarized_segments:
            return []

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

        merge_time = time.perf_counter() - start_time
        print(f"Процедура: Объединение сегментов (merge_segments) - {merge_time:.4f} сек")

        return merged

    def diarize_and_merge(self, audio_frames: np.ndarray, num_speakers: int, cluster_threshold: float,
                          min_duration: float, max_gap: float, max_phrase_gap: float,
                          winlen: float, winstep: float, num_mel_bins: int, nfft: int,
                          merge_cluster_threshold: float, filter_cutoff: float, filter_order: int,
                          batch_size: int = 8) -> list[dict]:
        start_time = time.perf_counter()

        raw_result = self.diarize(audio_frames, num_speakers, cluster_threshold, min_duration, max_gap,
                                  winlen, winstep, num_mel_bins, nfft, merge_cluster_threshold,
                                  filter_cutoff, filter_order, batch_size)
        merged_result = self.merge_segments(raw_result, max_phrase_gap)

        total_time = time.perf_counter() - start_time
        print(f"Процедура: Полная диаризация и объединение (diarize_and_merge) - {total_time:.4f} сек")

        return merged_result


def load_and_preprocess_audio(file_path: str, target_frame_size: int, sample_rate: int) -> np.ndarray:
    start_time = time.perf_counter()

    audio = AudioSegment.from_file(file_path)
    if audio.frame_rate != sample_rate:
        audio = audio.set_frame_rate(sample_rate)
    if audio.channels > 1:
        audio = audio.split_to_mono()[1][0:180000]
    samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
    samples_float32 = samples.astype(np.float32) / 32768.0
    num_frames = len(samples_float32) // target_frame_size
    frames = samples_float32[:num_frames * target_frame_size].reshape(num_frames, target_frame_size)

    load_time = time.perf_counter() - start_time
    print(f"Процедура: Загрузка и предобработка аудио (load_and_preprocess_audio) - {load_time:.4f} сек")

    return frames


if __name__ == "__main__":
    # Параметры диаризации и кластеризации
    num_speakers = -1  # Количество спикеров (-1 для автоматического определения)
    cluster_threshold = 0.7  # Порог расстояния для кластеризации

    max_phrase_gap = 0.5  # Максимальный промежуток для объединения сегментов (сек)
    merge_cluster_threshold = 0.4  # Порог сходства для объединения кластеров в постобработке
    use_gpu_diar = False  # По возможности использовать графический процессор для вычислений
    batch_size = 16  # Размер батча для извлечения эмбеддингов

    # Параметры сегментации (VAD)
    min_duration = 0.15  # Минимальная длительность речевого сегмента (сек)
    max_gap = 0.25  # Максимальный промежуток между сегментами для их объединения (сек)
    vad_mode = 4  # Режим чувствительности VAD (1, 2, 3, 4, 5)
    use_gpu_vad = False  # По возможности использовать графический процессор для вычислений

    # Параметры высокопроходного фильтра
    filter_cutoff = 100.0  # Частота среза высокопроходного фильтра (Гц)
    filter_order = 10  # Порядок фильтра

    # Параметры обработки признаков (fbank)
    winlen = 0.025  # Длина окна для fbank (сек)
    winstep = 0.01  # Шаг окна для fbank (сек)
    num_mel_bins = 80  # Количество мел-фильтров
    nfft = 512  # Размер FFT

    # Параметры аудио
    sample_rate = 16000  # Частота дискретизации (Гц)
    target_frame_size = 512  # Размер фрейма для обработки аудио

    vad_model_path = Path("../models/VAD_silero_v5/silero_vad.onnx")
    speaker_model_path = Path("../models/Diar_model/voxblink2_samresnet100_ft.onnx")
    audio_path = "../trash/long.mp3"

    vad = SileroVAD(vad_model_path, sample_rate=sample_rate, use_gpu=use_gpu_vad)
    vad.set_mode(vad_mode)

    audio_frames = load_and_preprocess_audio(audio_path, target_frame_size=target_frame_size, sample_rate=sample_rate)

    diarizer = Diarizer(speaker_model_path, vad=vad, sample_rate=sample_rate, use_gpu=use_gpu_diar)

    result = diarizer.diarize_and_merge(
        audio_frames,
        num_speakers=num_speakers,
        cluster_threshold=cluster_threshold,
        min_duration=min_duration,
        max_gap=max_gap,
        max_phrase_gap=max_phrase_gap,
        winlen=winlen,
        winstep=winstep,
        num_mel_bins=num_mel_bins,
        nfft=nfft,
        merge_cluster_threshold=merge_cluster_threshold,
        filter_cutoff=filter_cutoff,
        filter_order=filter_order,
        batch_size=batch_size
    )

    for r in result:
        print(f"Спикер {r['speaker']}: {r['start']:.2f} - {r['end']:.2f} сек")