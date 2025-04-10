# Немного рассуждений для запомнинания.
Итак. Sherpa-onnx компилируется на onnxruntime 1.17.1
Эта версия просит CUDA 11.8 и CudNN 8.6.
Если в проекте запускать onnxruntime, то для поддержки CUDA можно использовать только onnxruntime 1.17.1.
На этой версии работает punct, но не рабоатет silero v5. Вообще не уверен, что silerov5 работает на CUDA. (проверить ?)

**Как не работает silero:** onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: [ONNXRuntimeError] : 6 :
RUNTIME_EXCEPTION : Exception during initialization: D:\a\_work\1\s\onnxruntime\core/framework/feeds_fetches_manager.h:
44 onnxruntime::FeedsFetchesInfo::FeedsFetchesInfo [ONNXRuntimeError] : 1 : FAIL : Error mapping output names: Could not
find OrtValue with name 'If_0_else_branch__Inline_0__/Squeeze_output_0'
**в итоге силеро нe работало, так как я не правильно передавал название модели....** Т.е. силеро работает.
Но, целессообразность использования в проекте указана ниже.

Silero onnxruntime 1.17.1
CPU Загружено 95999 фреймов по 512 сэмплов Время выполнения 9.571041 и 8.867979 на onnxruntime 1.21.0
GPU Загружено 95999 фреймов по 512 сэмплов Время выполнения 42.192122 и 37.642282 на onnxruntime 1.21.0
Результат очевиден - Silero без батчинга - GPU на ветер.
