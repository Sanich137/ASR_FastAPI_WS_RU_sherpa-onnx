# -*- coding: utf-8 -*-
import asyncio
import datetime

import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort
from pathlib import Path
import pynvml

# Прогнозируемые знаки препинания
PUNK_MAPPING = {".": "PERIOD", ",": "COMMA", "?": "QUESTION"}

# Прогнозируемый регистр LOWER - нижний регистр, UPPER - верхний регистр для первого символа,
# UPPER_TOTAL - верхний регистр для всех символов
LABELS_CASE = ["LOWER", "UPPER", "UPPER_TOTAL"]
# Добавим в пунктуацию метку O означающий отсутствие пунктуации
LABELS_PUNC = ["O"] + list(PUNK_MAPPING.values())

# Сформируем метки на основе комбинаций регистра и пунктуации
LABELS_list = []
for case in LABELS_CASE:
    for punc in LABELS_PUNC:
        LABELS_list.append(f"{case}_{punc}")
LABELS = {label: i + 1 for i, label in enumerate(LABELS_list)}
LABELS["O"] = -100
INVERSE_LABELS = {i: label for label, i in LABELS.items()}

LABEL_TO_PUNC_LABEL = {
    label: label.split("_")[-1] for label in LABELS.keys() if label != "O"
}
LABEL_TO_CASE_LABEL = {
    label: "_".join(label.split("_")[:-1]) for label in LABELS.keys() if label != "O"
}


def token_to_label(token, label):
    if type(label) == int:
        label = INVERSE_LABELS[label]
    if label == "LOWER_O":
        return token
    if label == "LOWER_PERIOD":
        return token + "."
    if label == "LOWER_COMMA":
        return token + ","
    if label == "LOWER_QUESTION":
        return token + "?"
    if label == "UPPER_O":
        return token.capitalize()
    if label == "UPPER_PERIOD":
        return token.capitalize() + "."
    if label == "UPPER_COMMA":
        return token.capitalize() + ","
    if label == "UPPER_QUESTION":
        return token.capitalize() + "?"
    if label == "UPPER_TOTAL_O":
        return token.upper()
    if label == "UPPER_TOTAL_PERIOD":
        return token.upper() + "."
    if label == "UPPER_TOTAL_COMMA":
        return token.upper() + ","
    if label == "UPPER_TOTAL_QUESTION":
        return token.upper() + "?"
    if label == "O":
        return token


def decode_label(label, classes="all"):
    if classes == "punc":
        return LABEL_TO_PUNC_LABEL[INVERSE_LABELS[label]]
    if classes == "case":
        return LABEL_TO_CASE_LABEL[INVERSE_LABELS[label]]
    else:
        return INVERSE_LABELS[label]


class SbertPuncCaseOnnx:
    def __init__(self, onnx_model_path, use_gpu = False):
        self.tokenizer = AutoTokenizer.from_pretrained(onnx_model_path,
                                                       strip_accents=False,
                                                       )
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 4  # Выключаем подробный лог
        session_options.enable_profiling = False
        session_options.enable_mem_pattern = False
        session_options.enable_mem_reuse = False
        session_options.enable_cpu_mem_arena = False
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.inter_op_num_threads = 0
        session_options.intra_op_num_threads = 0


        if not use_gpu:
            providers = ['CPUExecutionProvider']
        else:
            providers = [('CUDAExecutionProvider', {
                            'device_id': 1,
                            # 'arena_extend_strategy': 'kNextPowerOfTwo', # DВыделяет память с запасом
                            'arena_extend_strategy': 'kSameAsRequested' ,# 20.657809 на 1000 итераций и 26 Мб съел
                            'gpu_mem_limit': int(1.7 * 1024 * 1024 * 1024), #  Потребляет где-то 1.9 Гб памяти ГПУ
                            'cudnn_conv_algo_search': 'EXHAUSTIVE',
                            'do_copy_in_default_stream': True,
                                }),
                         'CPUExecutionProvider']
            # providers = ort.get_available_providers() можно завести на тензор провайдер, на 2хххRTX работает медленно.
            # https://developer.nvidia.com/nvidia-tensorrt-8x-download#
            # https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html#download

        self.session = ort.InferenceSession(path_or_bytes=f"{onnx_model_path}/model.onnx",
                                            sess_options=session_options,
                                            providers=providers)   # CPUExecutionProvider

    async def punctuate(self, text):
        text = text.strip().lower()

        # Разобъем предложение на слова
        words = text.split()

        tokenizer_output = self.tokenizer(words, is_split_into_words=True)

        if len(tokenizer_output.input_ids) > 512:
            return " ".join(
                [
                    await self.punctuate(" ".join(text_part))
                    for text_part in np.array_split(words, 2)
                ]
            )

        # Подготовка входных данных для модели
        input_ids = np.array(tokenizer_output.input_ids, dtype=np.int64).reshape(1, -1)
        attention_mask = np.array(tokenizer_output.attention_mask, dtype=np.int64).reshape(1, -1)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)  # Добавляем token_type_ids

        # Выполнение модели
        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,  # Передаём token_type_ids
            },
        )
        predictions = np.argmax(outputs[0], axis=2)

        # decode punctuation and casing
        splitted_text = []
        word_ids = tokenizer_output.word_ids()
        for i, word in enumerate(words):
            label_pos = word_ids.index(i)
            label_id = predictions[0][label_pos]
            label = decode_label(label_id)
            splitted_text.append(token_to_label(word, label))
        capitalized_text = " ".join(splitted_text)

        # self.session.end_profiling()

        return capitalized_text

def gpu_stat(gpu_index):

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)  # Первая видеокарта
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mb = mem_info.free / 1024**2
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_load = utilization.gpu
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except pynvml.NVMLError as e:
        return {"error": str(e)}
    finally:
        pynvml.nvmlShutdown()
    return free_mb, gpu_load,temperature

if __name__ == '__main__':
    from datetime import datetime as dt
    time_start = dt.now()
    model_path = str(Path("../models/sbert_punc_case_ru_onnx"))
    print(f" ресурсы до выполнения {gpu_stat(1)}")
    sbertpunc = SbertPuncCaseOnnx(model_path, use_gpu=True)
    print(f"Время на инициализацию {(dt.now() - time_start).total_seconds()}")
    print(f" ресурсы после инициализации {gpu_stat(1)}")
    input_text = ["channel_1: два\nchannel_2: татьяна добрый день это компания вас беспокоит меня зовут ульяна\nchannel_2: звоню уточнить по поводу документов мы у вас в чате запрашивали список документов\nchannel_2: скажите пожалуйста когда сможете прислать чтобы юристы ознакомились с ними\nchannel_1: два сегодня а да ну хорошо а доброго\nchannel_2: сегодня пришлете хорошо тогда ждем от вас всего доброго\nchannel_2: до свидания\n",
                  "время на переинициализацию модели",
                  "ресурсы до выполнения",
                  "",
                  "",
                  "",
                  "",
                  "",
                  "",
                  ""]
    for _ in range(1000):
        import logging as logger
        # Instead of using argparse, directly define the input and model path:
        input_text = "channel_1: два\nchannel_2: татьяна добрый день это компания вас беспокоит меня зовут ульяна\nchannel_2: звоню уточнить по поводу документов мы у вас в чате запрашивали список документов\nchannel_2: скажите пожалуйста когда сможете прислать чтобы юристы ознакомились с ними\nchannel_1: два сегодня а да ну хорошо а доброго\nchannel_2: сегодня пришлете хорошо тогда ждем от вас всего доброго\nchannel_2: до свидания\n"
        # print(f"Source text:   {input_text}\n")
        punctuated = asyncio.run(
                    sbertpunc.punctuate(input_text)
                        )

        # print(punctuated)

    print(f"Время выполнения {(dt.now() - time_start).total_seconds()}")
    print(f" ресурсы после выполнения {gpu_stat(1)}")

    time_start = dt.now()
    sbertpunc.session.end_profiling()
    print(f"Время на переинициализацию модели {(dt.now() - time_start).total_seconds()}")
    # asyncio.run(asyncio.sleep(5))
    print(f" ресурсы после окончания {gpu_stat(1)}")