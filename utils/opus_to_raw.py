import subprocess

async def do_opus_to_raw_convertion(audio_data):
# Вариант 2: Потоковая обработка через пайп (без сохранения в файл)
    ffmpeg_process = (
        subprocess.Popen(
            ["ffmpeg", "-i", "pipe:0", "-f", "wav", "pipe:1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
    )

    ffmpeg_process.stdin.write(audio_data)
    ffmpeg_process.stdin.close()
    wav_data = ffmpeg_process.stdout.read()

    return wav_data
