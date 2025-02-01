# from threading import Semaphore, Thread, Lock
# from multiprocessing import Process, Queue, Pool, Lock
#
# from datetime import datetime, timedelta
# import ujson  # работа с json-файлами и json-строками
# from pydub import AudioSegment
# from pathlib import Path
# import logging
#
# logging.basicConfig(format=u'%(filename)s [LINE:%(lineno)d] #%(levelname)-8s [%(asctime)s]  %(message)s',
#                     # level=logging.INFO,
#                     # filename=f'Services-from-{datetime.datetime.now().date()}.log',
#                     filemode='a',
#                     level=logging.DEBUG,  # Можно заменить на другой уровень логирования.
#                     )
#
#
# def get_order_filename(q):
#     if q.empty():
#         file_to_work = None
#         logging.debug(f'Файлы кончились в очереди')
#     else:
#         try:
#             file_to_work = q.get(block=True, timeout=1)
#         except Exception as e:
#             logging.error(f'Произошла ошибка получения данных их очереди - {e}')
#             file_to_work = None
#     return file_to_work
#
#
# def get_model(gpu_lock=Lock()):
#     from vosk import BatchModel, GpuInit, SetLogLevel
#     SetLogLevel(-10)
#     gpu_lock.acquire()
#
#     GpuInit()
#     # GpuThreadInit()
#     base_dir = Path(__file__).resolve()  # .parent.parent.parent / 'DB_folder'
#     model_path = str(base_dir.parent.parent / 'vosk_models_files' / 'vosk-model-ru-0.42')
#     model = BatchModel(str(model_path))
#     gpu_lock.release()
#
#     return model
#
#
# def recognition_main_proc(q):
#     from vosk import BatchRecognizer
#     model = get_model()
#
#     while True:
#         # time_rec_start = datetime.now()
#         base_dir = Path(__file__).resolve()  # .parent.parent.parent / 'DB_folder'
#         file_name = get_order_filename(q)
#
#         if not file_name:
#             print(f'Получили сообщение, что файлы кончились')
#             break
#
#         file_path = str(base_dir.parent.parent / 'Audio_archive_folder' /
#                         file_name)
#
#         try:
#             sound = AudioSegment.from_file(str(file_path))
#         except Exception as e:
#             print(e)
#         else:
#             logging.debug(f'Файл принят в работу')
#             logging.info(f'Общая продолжительность аудиофайла {sound.duration_seconds} сек.')
#
#             # Фреймрейт на входе
#             logging.debug(f'Исходный фреймрейт аудио - {sound.frame_rate}')
#             # Меняем фреймрейт на 16кГц
#             sound = sound.set_frame_rate(16000)
#             logging.debug(f'Изменённый фреймрейт аудио - {sound.frame_rate}')
#
#             # Разбиваем звук по каналам
#             channels = sound.channels
#             separate_channels = sound.split_to_mono()
#
#             json_raw_data = list()
#             full_text = list()
#             json_text_data = list()
#             recognized_text = list()
#
#             try:
#                 raw_data_full = dict()
#
#                 treads = [Thread(target=recognition_channel_tread,
#                                     args=(BatchRecognizer,
#                                           model,
#                                           raw_data_full,
#                                           separate_channels,
#                                           channel),
#                                        ) for channel in range(channels)]
#                 for thread in treads:
#                     thread.start()
#                 for thread in treads:
#                     thread.join()
#
#
#
#
#             except Exception as e:
#                 logging.error(f'Возникла ошибка "{e}"')
#
#             else:
#                 logging.debug(f' Full - {raw_data_full}')
#                 json_to_save = {
#                     "json_text_data": json_text_data,
#                     "recognized_text": recognized_text,
#                     "duration": sound.duration_seconds,
#                     "work_time": (datetime.now() - time_rec_start).total_seconds(),
#                 }
#
#                 with open(base_dir.parent.parent / 'Audio_archive_folder' / f"{file_name}.json",
#                           "w", encoding='utf-8') as outfile:
#                     ujson.dump(json_to_save, outfile, ensure_ascii=False)
#
#
#
#                 logging.debug(f'На обработку файла затрачено - {(datetime.now() - time_rec_start).total_seconds()} сек.')
#
#     return
#
#
# def recognition_channel_tread(BatchRecognizer, model, raw_data_full, separate_channels, channel):
#
#     offline_recognizer = BatchRecognizer(model, separate_channels[channel].frame_rate, )
#     print(f"Приняли в работу канал № {channel+1}")
#     # Основная функция - c разбивкой на сэмплы (возможно, уменьшает нагрузку на ОЗУ)
#     _from = 0
#     _step = 4096
#     _to = _step
#
#     while _from < len(separate_channels[channel].raw_data):
#         offline_recognizer.AcceptWaveform(separate_channels[channel].raw_data[_from:_to])
#         _from = _to
#         _to += _step
#
#     offline_recognizer.FinishStream()
#     # Ждём ответа от GPU
#     model.Wait()
#     logging.debug(f"Обработали аудио канал № {channel + 1} за {datetime.now() - time_rec_start} сек.")
#     json_raw_data = list()
#     while True:
#         try:
#             raw_data = ujson.loads(offline_recognizer.Result())
#
#             # raw_data = ujson.loads(offline_recognizer.Result())
#         except Exception as e:
#             logging.debug(f'{e}')
#             break
#         else:
#             # logging.debug(f'Получили результат из строки в json {raw_data}')
#             json_raw_data.append(raw_data)
#
#     raw_data_full[channel] = json_raw_data
#     logging.debug(f"Результат распознавания текста - {raw_data_full}...")
#
#
# if __name__ == '__main__':
#     time_rec_start = datetime.now()
#     queue = Queue()
#     files_list = ["0001.1705950000.536284.mp3", "0004.1578538803.21.mp3", "5959.1578387598.15408.mp3",
#                   "0410.1686060236.704695.mp3"]
#
#     for f_name in files_list:
#         queue.put(f_name)
#
#     processes = [Process(target=recognition_main_proc, args=(queue,),
#                          ) for i in range(gpustat.gpu_count())]
#
#     for p in processes:
#         p.start()
#     for p in processes:
#         p.join()
#     for p in processes:
#         p.kill()
#
#     logging.info(f'На обработку {len(files_list)} файлов затрачено - {(datetime.now() - time_rec_start).total_seconds()} сек.')
