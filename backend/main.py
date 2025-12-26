from __future__ import annotations
import os, io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .inference import predict_binary, DEFAULT_MODEL_PATH

app = FastAPI(title="Waste CNN Classifier — Organik vs Non-Organik")
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(FRONTEND_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
def health():
    return {"status": "ok", "model_path": DEFAULT_MODEL_PATH}

@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        return JSONResponse(
            status_code=415,
            content={"error": "Unsupported file type. Use JPG/PNG/WEBP."},
        )

    data = await file.read()
    if len(data) > 5 * 1024 * 1024:
        return JSONResponse(
            status_code=413,
            content={"error": "File too large. Max 5MB."},
        )

    try:
        img = Image.open(io.BytesIO(data))
        return predict_binary(img)
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to process image: {e}"},
        )
