from utils.pre_start_init import app
from models.fast_api_models import WebSocketModel

@app.post("/ws")
async def post_not_websocket(ws:WebSocketModel):
    """Описание для вебсокета ниже в описании WebSocketModel """
    return f"Прочти инструкцию в Schemas - 'WebSocketModel'"
