from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
import onnx

# 1. Оптимизация графа модели
def optimize_onnx_model(input_model_path, output_model_path):
    # Настройка параметров сессии
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Создание сессии с оптимизацией
    session = InferenceSession(input_model_path, options)

    # Загрузка модели с помощью ONNX
    model = onnx.load(input_model_path)

    # Применение оптимизаций ONNX Runtime
    optimized_model = onnx.shape_inference.infer_shapes(model)
    onnx.save(optimized_model, output_model_path)

# 2. Квантизация модели (динамическая)
def quantize_onnx_model(input_model_path, output_model_path):
    quantize_dynamic(
        input_model_path,
        output_model_path,
        weight_type=QuantType.QInt8,
    )

# 3. Использование GPU
def create_gpu_session(model_path):
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        sess_options=options
    )
    return session

# Оптимизация графа
optimize_onnx_model("/content/sbert_punc_case_ru_onnx/model.onnx", "model_optimized.onnx")
print("Модель оптимизирована и сохранена в model_optimized.onnx")

# Квантизация
quantize_onnx_model("model_optimized.onnx", "model_quantized.onnx")
print("Модель оптимизирована, квантизирована и сохранена в model_quantized.onnx")

# Использование GPU
class SbertPuncCaseOnnxOptimized(SbertPuncCaseOnnx):
    def __init__(self, onnx_model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(onnx_model_path)
        self.session = create_gpu_session(f"{onnx_model_path}/model_quantized.onnx")

        