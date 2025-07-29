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
    def __init__(self, embedding_model_path: str, segmentation_model_path: str, sample_rate: int = 16000,
                 use_gpu: bool = False, batch_size: int = 32, cpu_workers: int = 0, window_size: float = 0.5,
                 window_step: float = 0.04, asr_margin: float = 0.1, segmentation_frame_step: float = 0.0175):
        self.sample_rate = sample_rate
        # Fbanks
        self.winlen = 0.025
        self.winstep = 0.01
        self.num_mel_bins = 80
        self.nfft = 512
        self.window_size = window_size
        self.window_step = window_step

        # asr
        self.asr_margin = asr_margin
        self.segmentation_frame_step = segmentation_frame_step

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

        # Инициализация модели эмбеддингов
        self.embedding_session = ort.InferenceSession(
            embedding_model_path,
            sess_options=so,
            providers=providers
        )
        # Инициализация модели сегментации
        self.segmentation_session = ort.InferenceSession(
            segmentation_model_path,
            sess_options=so,
            providers=providers
        )
        # Проверка входных имен модели
        segmentation_inputs = [input.name for input in self.segmentation_session.get_inputs()]
        logger.info(f"Входные имена модели сегментации: {segmentation_inputs}")
        self.segmentation_input_name = 'x' if 'x' in segmentation_inputs else segmentation_inputs[0]
        logger.info(f"Используемые для диаризации провайдеры: {self.embedding_session.get_providers()}")
        logger.info(f"Используемые для сегментации провайдеры: {self.segmentation_session.get_providers()}")

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

        # Находим максимальную длину fbank-массива
        max_len = max(f.shape[0] for f in fbanks) if fbanks else 0
        if max_len == 0:
            logger.warning("Все fbank-массивы пусты")
            return np.array([])

        # Дополняем или урезаем fbank-массивы до одинаковой длины
        padded_fbanks = []
        for fbank in fbanks:
            if fbank.shape[0] < max_len:
                # Дополняем нули до max_len
                pad_width = ((0, max_len - fbank.shape[0]), (0, 0))
                padded_fbank = np.pad(fbank, pad_width, mode='constant', constant_values=0)
            else:
                # Урезаем до max_len
                padded_fbank = fbank[:max_len]
            padded_fbanks.append(padded_fbank)

        fbanks_array = np.stack(padded_fbanks)
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

    def segment_audio(self, audio: np.ndarray, region_start: float, region_end: float) -> list[dict]:
        """Выполняет сегментацию с помощью pyannote-segmentation-3.0."""
        start_sample = int(region_start * self.sample_rate)
        end_sample = int(region_end * self.sample_rate)
        if end_sample > len(audio):
            end_sample = len(audio)
        if end_sample <= start_sample:
            logger.warning(
                f"Некорректный регион: start={region_start}, end={region_end}, длина аудио={len(audio) / self.sample_rate}")
            return []
        segment_audio = audio[start_sample:end_sample].reshape(1, 1, -1).astype(np.float32)

        logger.debug(f"Сегмент: start={region_start:.2f}s, end={region_end:.2f}s, форма аудио={segment_audio.shape}")

        # Инференс модели сегментации
        inputs = {self.segmentation_input_name: segment_audio}
        logits = self.segmentation_session.run(None, inputs)[0][0]  # [1, num_frames, num_classes] -> [num_frames, 7]
        # Применяем сигмоиду к логитам
        probs = 1 / (1 + np.exp(-logits))
        logger.debug(f"Логиты: форма={probs.shape}, макс. вероятности={np.max(probs, axis=0)}")

        # Постобработка вероятностей
        results = []
        num_frames, num_classes = probs.shape
        region_duration = region_end - region_start
        frame_duration = region_duration / num_frames if num_frames > 0 else 0.0

        for frame_idx in range(num_frames):
            time = region_start + (frame_idx / num_frames) * region_duration
            active_classes = np.where(probs[frame_idx] > 0.5)[0]  # Порог 0.5
            active_speakers = set()

            # Пропускаем, если только не речь или ничего
            if len(active_classes) == 0 or np.array_equal(active_classes, [0]):
                max_class = np.argmax(probs[frame_idx][1:]) + 1 if np.max(probs[frame_idx][1:]) > 0.1 else None
                if max_class and max_class in [1, 2, 3]:
                    active_speakers.add(f"temp_speaker_{max_class}")
            else:
                for cls in active_classes:
                    if cls == 0:  # Пропускаем неречевые сегменты
                        continue
                    if cls in [1, 2, 3]:  # Спикеры
                        active_speakers.add(f"temp_speaker_{cls}")
                    elif cls in [4, 5, 6]:  # Пересечения
                        overlap_speakers = {
                            4: ["temp_speaker_1", "temp_speaker_2"],
                            5: ["temp_speaker_1", "temp_speaker_3"],
                            6: ["temp_speaker_2", "temp_speaker_3"]
                        }[cls]
                        active_speakers.update(overlap_speakers)

            # Создаем сегменты для каждого активного спикера в текущем фрейме
            for speaker in active_speakers:
                results.append({
                    "start": time,
                    "end": time + frame_duration,
                    "speaker": speaker
                })

        # Объединение соседних сегментов одного спикера
        if not results:
            logger.warning("Не найдено сегментов в регионе")
            return []

        # Группируем по спикерам и объединяем сегменты
        speaker_segments = {}
        for seg in results:
            speaker = seg["speaker"]
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append((seg["start"], seg["end"]))

        merged_results = []
        for speaker, segs in speaker_segments.items():
            segs.sort(key=lambda x: x[0])  # Сортировка по времени начала
            current_start, current_end = segs[0]
            for start, end in segs[1:]:
                if start <= current_end + 0.001:
                    current_end = max(current_end, end)
                else:
                    merged_results.append({"start": current_start, "end": current_end, "speaker": speaker})
                    logger.info(f'"start": {current_start:.2f}, "end": {current_end:.2f}, "speaker": {speaker}')
                    current_start, current_end = start, end
            logger.info(f'Последний "start": {current_start:.2f}, "end": {current_end:.2f}, "speaker": {speaker}')
            merged_results.append({"start": current_start, "end": current_end, "speaker": speaker})

        merged_results.sort(key=lambda x: x["start"])  # Сортировка по времени начала
        logger.info(f"Получено {len(merged_results)} сегментов в регионе")

        return merged_results

    async def diarize(self, audio: np.ndarray, num_speakers: int, asr_data: list[dict]) -> list[dict]:
        start_time = time.perf_counter()

        # Извлечение речевых регионов из ASR, разбиение на непересекающиеся регионы до 10 секунд
        max_region_duration = 20.0  # Максимальная длительность региона
        sample_rate = self.sample_rate

        # Список регионов: каждый регион содержит аудиофрагмент и временные границы
        regions = []
        current_region = {"audio": None, "start": None, "end": None, "words": []}
        word_regions = []

        # Собираем все слова с их временными метками
        for segment in asr_data:
            for word in segment['data']['result']:
                start = word['start']
                end = word['end']
                word_regions.append({"start": start, "end": end, "word": word})

        word_regions.sort(key=lambda x: x["start"])  # Сортировка по времени начала

        i = 0  # Индекс текущего слова
        while i < len(word_regions):
            word = word_regions[i]
            word_duration = word["end"] - word["start"]
            # Проверяем длительность региона по абсолютным временным меткам
            if current_region["start"] is None:
                current_region["start"] = word["start"]
            region_duration = word["end"] - current_region["start"]
            if region_duration > max_region_duration and current_region["words"]:
                # Финализируем текущий регион
                current_region["end"] = current_region["words"][-1]["end"]
                start_sample = int(current_region["start"] * sample_rate)
                end_sample = int(current_region["end"] * sample_rate)
                if end_sample > len(audio):
                    end_sample = len(audio)
                if end_sample > start_sample:
                    current_region["audio"] = audio[start_sample:end_sample]
                    print(
                        f"Регион создан: {current_region['start']:.2f}-{current_region['end']:.2f} сек, слов: {len(current_region['words'])}")
                    regions.append(current_region)
                # Начинаем новый регион
                current_region = {"audio": None, "start": word["start"], "end": None, "words": []}
            current_region["words"].append(word)
            i += 1

        # Добавляем последний регион, если он не пуст
        if current_region["words"]:
            current_region["end"] = current_region["words"][-1]["end"]
            start_sample = int(current_region["start"] * sample_rate)
            end_sample = int(current_region["end"] * sample_rate)
            if end_sample > len(audio):
                end_sample = len(audio)
            if end_sample > start_sample:
                current_region["audio"] = audio[start_sample:end_sample]
                print(
                    f"Регион создан: {current_region['start']:.2f}-{current_region['end']:.2f} сек, слов: {len(current_region['words'])}")
                regions.append(current_region)

        logger.debug(f"Создано {len(regions)} речевых регионов")

        # Сегментация с помощью pyannote-segmentation-3.0
        segments = []
        for region_idx, region in enumerate(regions):
            if not region["audio"].size:
                continue
            region_start = 0.0  # Время начала региона в локальной шкале
            region_duration = region["end"] - region["start"]
            region_end = region_duration
            logger.debug(
                f"Регион {region_idx}: {region['start']:.2f}-{region['end']:.2f} с, длительность: {region_duration:.2f} с")
            region_segments = self.segment_audio(region["audio"], region_start, region_end)
            print(f"Сегменты региона {region_idx}: {region_segments}")

            # Сопоставляем результаты сегментации
            adjusted_segments = []
            asr_starts = [w["start"] for w in region["words"]]
            asr_ends = [w["end"] for w in region["words"]]
            for seg in region_segments:
                seg_start = seg["start"]
                seg_end = seg["end"]
                # Переводим относительные времена в абсолютные
                abs_start = region["start"] + seg_start
                abs_end = region["start"] + seg_end
                # Корректируем границы с учетом ASR
                nearest_start = min(asr_starts, key=lambda x: abs(x - abs_start)) if asr_starts else abs_start
                nearest_end = min(asr_ends, key=lambda x: abs(x - abs_end)) if asr_ends else abs_end
                if abs(nearest_start - abs_start) <= self.asr_margin:
                    abs_start = nearest_start
                if abs(nearest_end - abs_end) <= self.asr_margin:
                    abs_end = nearest_end
                adjusted_segments.append({
                    "start": abs_start,
                    "end": abs_end,
                    "speaker": seg["speaker"]
                })
            segments.extend(adjusted_segments)
        if not segments:
            logger.info("Не удалось получить сегменты от pyannote-segmentation")
            return []

        # Извлечение fbank и эмбеддингов для сегментов
        embeddings = []
        segment_times = []
        segment_centers = []
        segment_speakers = []
        window_size_sec = 1  # Размер окна 0.5 с
        window_step_sec = 0.5  # Шаг окна 0.25 с
        window_size_samples = int(window_size_sec * self.sample_rate)
        window_step_samples = int(window_step_sec * self.sample_rate)

        for seg in segments:
            start_sample = int(seg["start"] * self.sample_rate)
            end_sample = int(seg["end"] * self.sample_rate)
            if end_sample > len(audio):
                end_sample = len(audio)
            if end_sample <= start_sample:
                continue
            print(f"Диаризация. Старт:{seg['start']:.2f} стоп {seg['end']:.2f}")
            segment_audio = audio[start_sample:end_sample]
            segment_length_samples = len(segment_audio)

            # Извлечение подсегментов с окном
            window_fbanks = []
            current_start = 0
            while current_start + window_size_samples <= segment_length_samples:
                window_audio = segment_audio[current_start:current_start + window_size_samples]
                if len(window_audio) < int(self.window_size * self.sample_rate):
                    window_audio = np.pad(
                        window_audio,
                        (0, int(self.window_size * self.sample_rate) - len(window_audio)),
                        'constant'
                    )
                fbank_feats = await self.extract_fbank(window_audio)
                if fbank_feats.size > 0:
                    window_fbanks.append(fbank_feats)
                current_start += window_step_samples

            # Если остались последние сэмплы короче окна, обработаем их отдельно
            if current_start < segment_length_samples:
                window_audio = segment_audio[
                               -window_size_samples:] if segment_length_samples >= window_size_samples else segment_audio
                if len(window_audio) < int(self.window_size * self.sample_rate):
                    window_audio = np.pad(
                        window_audio,
                        (0, int(self.window_size * self.sample_rate) - len(window_audio)),
                        'constant'
                    )
                fbank_feats = await self.extract_fbank(window_audio)
                if fbank_feats.size > 0:
                    window_fbanks.append(fbank_feats)

            if not window_fbanks:
                logger.info(f"Не удалось извлечь валидные fbank для сегмента {seg['start']:.2f}-{seg['end']:.2f}")
                continue

            # Извлечение эмбеддингов для всех подсегментов
            window_embeddings = await self.extract_embeddings(window_fbanks, subseg_cmn=True)
            if len(window_embeddings) == 0:
                logger.info(
                    f"Не удалось извлечь валидные эмбеддинги для сегмента {seg['start']:.2f}-{seg['end']:.2f}")
                continue

            # Усреднение эмбеддингов подсегментов
            mean_embedding = np.mean(window_embeddings, axis=0)
            segment_times.append((seg["start"], seg["end"]))
            segment_centers.append((seg["start"] + seg["end"]) / 2)
            segment_speakers.append(seg["speaker"])
            embeddings.append(mean_embedding)

        if not embeddings:
            logger.info("Не удалось извлечь валидные эмбеддинги")
            return []

        embeddings = np.array(embeddings).astype(np.float64)  # Преобразуем в float64 для совместимости с HDBSCAN
        logger.info(f"Количество эмбеддингов: {len(embeddings)}")

        # Кластеризация
        if len(embeddings) <= 2:
            labels = [0] * len(embeddings)
            logger.info("Меньше 3 эмбеддингов, все метки установлены в 0")
        else:
            n_neighbors = min(15, len(embeddings) - 1)  # Динамически устанавливаем n_neighbors
            logger.debug(f"Используется n_neighbors={n_neighbors} для UMAP")
            umap_embeddings = UMAP(
                n_components=min(32, len(embeddings) - 2),
                metric='cosine',
                n_neighbors=n_neighbors,
                low_memory = False,
                min_dist=0.15,
                n_jobs=4
            ).fit_transform(embeddings)

            if num_speakers >= 2:
                clustering = AgglomerativeClustering(n_clusters=num_speakers, metric='cosine', linkage='average')
                labels = clustering.fit_predict(umap_embeddings)
                logger.info(f"Применена AgglomerativeClustering с {num_speakers} кластерами")
            else:
                clustering = HDBSCAN(
                    metric='precomputed',
                    min_cluster_size=2,
                    min_samples=1,
                    cluster_selection_epsilon=0.7,
                    # allow_single_cluster=True
                )
                labels = clustering.fit_predict(cosine_distances(umap_embeddings).astype(np.float64))
                if np.all(labels == -1):
                    labels = np.zeros_like(labels)
                    logger.info("HDBSCAN пометил все как шум, все метки установлены в 0")
                else:
                    unique_labels = np.unique(labels[labels != -1])
                    logger.info(f"HDBSCAN нашел {len(unique_labels)} кластеров: {unique_labels}")

        # Формирование финальных сегментов
        speaker_segments = {}
        for (start, end), label in zip(segment_times, labels):
            speaker = label if label != -1 else 0
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append((start, end))

        results = []
        for speaker, segs in speaker_segments.items():
            segs.sort(key=lambda x: x[0])
            current_start, current_end = segs[0]
            for start, end in segs[1:]:
                if start <= current_end + 0.1:
                    current_end = max(current_end, end)
                else:
                    results.append({"start": current_start, "end": current_end, "speaker": speaker})
                    logger.debug(f"Создан сегмент: {current_start:.2f}-{current_end:.2f}, спикер {speaker}")
                    current_start, current_end = start, end
            results.append({"start": current_start, "end": current_end, "speaker": speaker})
            logger.debug(f"Создан сегмент: {current_start:.2f}-{current_end:.2f}, спикер {speaker}")

        results.sort(key=lambda x: x["start"])
        total_diarize_time = time.perf_counter() - start_time
        logger.info(f"Процедура: Общая диаризация - {total_diarize_time:.4f} сек")
        return results


async def main():
    num_speakers = -1
    use_gpu_diar = False
    batch_size = 32
    max_cpu_workers = 0
    speaker_model_path = Path("../models/DIARISATION_model/voxceleb_resnet34_LM.onnx")
    segmentation_model_path = Path("../models/Segmentation/model.onnx")
    # audio_path = "../trash/replit/Amiran_audio.mp3"
    # asr_file = "../trash/replit/Amiran_asr.json"

    audio_path = "../trash/replit/Amiran_audio.mp3"
    asr_file = "../trash/replit/Amiran_asr.json"

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

    result = await diarizer.diarize(
        audio=samples_float32,
        asr_data=asr_data,
        num_speakers=num_speakers
    )

    for r in result:
        print(f"Спикер {r['speaker']}: {r['start']:.2f} - {r['end']:.2f} сек")


if __name__ == "__main__":
    asyncio.run(main())