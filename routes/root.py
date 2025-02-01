from utils.pre_start_init import app
import config

@app.get("/")
async def root():
    print("Зашли в root")

    return {"error": True,
            "data": "No_service_selected",
            # "available_services": ["Vosk_Recognizer"],
            "comment": f"try_addr: http://{config.host}:{config.port}/docs"}