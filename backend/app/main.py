from contextlib import asynccontextmanager
from io import BytesIO

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError

from .model_loader import load_model
from .inference import predict_image
from .gradcam import generate_gradcam
import threading
import os

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, classes, device = load_model()

    app.state.model = model
    app.state.classes = classes
    app.state.device = device

    yield


app = FastAPI(
    title="Inspectra AI",
    description="Industrial Surface Defect Classification API",
    version="1.0.0",
    lifespan=lifespan
)

gradcam_lock = threading.Lock()

cors_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

production_origin = os.getenv("FRONTEND_URL", "").rstrip("/")

if production_origin:
    cors_origins.append(production_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "application": "Inspectra AI",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "efficientnet_b0"
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/bmp",
        "image/x-ms-bmp",
        "application/octet-stream"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type."
        )

    contents = file.file.read(MAX_FILE_SIZE + 1)

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds 10 MB limit."
        )

    try:
        image = Image.open(BytesIO(contents))
        image.load()
        image = image.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image."
        )

    result = predict_image(
        image,
        app.state.model,
        app.state.classes,
        app.state.device
    )

    print("MODEL PREDICTION:", result)
    return result


@app.post("/explain")
def explain(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/bmp",
        "image/x-ms-bmp",
        "application/octet-stream"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format."
        )

    contents = file.file.read(MAX_FILE_SIZE + 1)

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds 10 MB limit."
        )

    try:
        image = Image.open(BytesIO(contents))
        image.load()
        image = image.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image."
        )

    with gradcam_lock:
        result = generate_gradcam(
            image,
            app.state.model,
            app.state.classes,
            app.state.device
        )

    return result
