from utils.do_logging import logger
import uvicorn
import config
from utils.pre_start_init import app
import routes, models
from fastapi.openapi.utils import get_openapi


def custom_openapi():
    openapi_schema = get_openapi(
        title="ASR Speech Recognition API",
        version="1.0.0",
        description="Real-time Russian ASR via WebSocket. Send raw audio chunks (16 kHz, mono).",
        routes=app.routes,
        contact={"email": "your@email.com"},
    )
    # Добавляем пример для WebSocket
    openapi_schema["paths"]["/ws"]["websocket"] = {
        "summary": "Stream audio for transcription",
        "requestBody": {
            "content": {
                "audio/*": {
                    "example": {"description": "Raw audio bytes (PCM, 16-bit)"}
                }
            }
        },
        "responses": {
            "200": {
                "description": "ASR result in JSON",
                "content": {
                    "application/json": {
                        "example": {"text": "привет мир", "confidence": 0.95}
                    }
                }
            }
        }
    }
    app.openapi_schema = openapi_schema
    return openapi_schema




try:
    if __name__ == '__main__':
        app.openapi = custom_openapi
        uvicorn.run(app, host=config.HOST, port=config.PORT)
except KeyboardInterrupt:
    logger.info('\nDone')
except Exception as e:
    logger.error(f'\nDone with error {e}')

