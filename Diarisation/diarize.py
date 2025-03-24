import config
from Diarisation import init_speaker_diarization
from datetime import datetime as dt
from pydub import AudioSegment
from utils.do_logging import logger
import asyncio


def progress_callback(num_processed_chunk: int, num_total_chunks: int) -> int:
    progress = num_processed_chunk / num_total_chunks * 100
    logger.debug(f"Diarization_progress: {progress:.3f}%")

    return 0


async def do_diarizing(audio: AudioSegment,
                       num_speakers: int = -1,
                       cluster_threshold: float = 0.1):
    """
     :param audio: AudioSegment, моно. Если передать не моно, то в работу будет взят первый канал.
     :param num_speakers: Предполагаемое количество спикеров. Если не знаем, то -1 и тогда количество
     спикеров будет определяться автоматически с использованием параметра cluster_threshold.
     :param  cluster_threshold: Чувствительность к выделению нового спикера.
     Если не нужны далёкие и тихие голоса, то 0,2 и выше. Если нужны тихие звуки, то 0,7-1,5
     :returns Возвращает список словарей [{"start": float(), "end": float(), "speaker": int(),}]

    """
    # Инициация модели происходит внутри функции так как предполагается давать пользователю возможность задавать
    # параметры чувствительности и количества спикеров.
    model = await init_speaker_diarization(num_speakers=num_speakers, cluster_threshold=cluster_threshold)

    audio = audio.set_frame_rate(model.sample_rate)
    audio, sample_rate = audio.get_array_of_samples(), audio.frame_rate

    if sample_rate != model.sample_rate:
        raise RuntimeError(
            f"Expected samples rate: {model.sample_rate}, given: {sample_rate}"
        )

    show_progress = False if config.IS_PROD == 1 else True

    if show_progress:
        sd_results = model.process(audio, callback=progress_callback).sort_by_start_time()
    else:
        sd_results = model.process(audio).sort_by_start_time()
    result = list()
    for r in sd_results:
        result.append({
            "start": float(f"{r.start:.3f}"),
            "end": float(f"{r.end:.3f}"),
            "speaker": int(f"{r.speaker}"),
        })
        logger.debug(f"{r.start:.3f} -- {r.end:.3f} speaker_{r.speaker:02}")

    return result


async def test_func(a_data):
    sd_rez = await do_diarizing(a_data)
    logger.error(f"sd_rezults - {sd_rez}")


if __name__ == "__main__":

    time_start = dt.now()
    # wave_filename = path+"protocol.mp3"
    wave_filename = "C://Users//kojevnikov//PycharmProjects//Vosk5_FastAPI_streaming//trash//long.mp3"
    # wave_filename = path+"q.wav"

    audio_data = AudioSegment.from_file(wave_filename).split_to_mono()[0][0:60000]

    asyncio.run(test_func(audio_data))

    logger.debug(f"Затрачено {(dt.now() - time_start).total_seconds()} сек.")
