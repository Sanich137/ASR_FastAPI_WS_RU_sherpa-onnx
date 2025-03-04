
from utils.pre_start_init import app
from fastapi import WebSocket, WebSocketException, Request
from utils.do_logging import logger
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

@app.get("/demo", response_class=HTMLResponse)
async def demo_page(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}  # Контекст для Jinja2 (если нужно)
    )
