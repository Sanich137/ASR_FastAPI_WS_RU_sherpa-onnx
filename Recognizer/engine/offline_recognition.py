import ujson
from collections import defaultdict

import config
from utils.pre_start_init import recognizer
from utils.bytes_to_samples_audio import get_np_array_samples_float32
from pydub import AudioSegment


