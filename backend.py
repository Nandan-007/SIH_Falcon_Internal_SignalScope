from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "best_model.pth"
STATIC_DIR = PROJECT_DIR / "static"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="SignalScope API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL MODEL STATE
# ============================================================

device = None
model = None
classes = None
transform = None
model_loaded = False


# ============================================================
# CLASS MAPPING HELPER
# ============================================================

def get_class_mapping(loaded_classes):
    """
    Normalize the checkpoint's class mapping into:

        {0: "real", 1: "ai_generated"}

    while still supporting either a dict or a list/tuple.
    """
    if isinstance(loaded_classes, dict):
        mapping = {
            int(index): str(class_name)
            for index, class_name in loaded_classes.items()
        }

    elif isinstance(loaded_classes, (list, tuple)):
        mapping = {
            index: str(class_name)
            for index, class_name in enumerate(loaded_classes)
        }

    else:
        raise ValueError(
            f"Unsupported classes format: {type(loaded_classes).__name__}"
        )

    if len(mapping) != 2:
        raise ValueError(
            f"Expected exactly 2 classes, found: {mapping}"
        )

    if 0 not in mapping or 1 not in mapping:
        raise ValueError(
            f"Expected class indices 0 and 1, found: {mapping}"
        )

    return mapping


def get_real_and_ai_indices(class_mapping):
    """
    Find the output indices for real and AI-generated classes.
    """
    real_idx = next(
        (
            index
            for index, class_name in class_mapping.items()
            if "real" in class_name.lower()
            and "ai" not in class_name.lower()
        ),
        None,
    )

    if real_idx is None:
        raise ValueError(
            f"Could not identify the real class from: {class_mapping}"
        )

    ai_idx = next(
        index for index in class_mapping.keys()
        if index != real_idx
    )

    return real_idx, ai_idx


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    global device, model, classes, transform, model_loaded

    print("=" * 60)
    print("Loading ResNet18 model from ./models/best_model.pth...")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Use a relative path relative to the backend.py directory
    BASE_DIR = Path(__file__).resolve().parent
    weights_path = BASE_DIR / "models" / "best_model.pth"

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("Model path:", MODEL_PATH)

    if not MODEL_PATH.exists():
        print(
            f"ERROR: Model file not found: {MODEL_PATH}"
        )
        model_loaded = False
    else:
        try:
            checkpoint = torch.load(
                MODEL_PATH,
                map_location=device
            )

            classes = checkpoint["classes"]

            # Validate and normalize checkpoint class mapping.
            class_mapping = get_class_mapping(classes)

            model = models.resnet18(
                weights=None
            )

            model.fc = nn.Linear(
                model.fc.in_features,
                2
            )

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            model = model.to(device)
            model.eval()

            model_loaded = True

            print(
                "Model loaded successfully."
            )
            print(
                "Classes:",
                class_mapping
            )

        except Exception as exc:
            model_loaded = False
            print(
                f"ERROR loading model: {exc}"
            )

    # Exact preprocessing used for the trained model.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("=" * 60)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": (
            device.type
            if device is not None
            else "unknown"
        ),
        "gpu": (
            torch.cuda.get_device_name(0)
            if (
                device is not None
                and device.type == "cuda"
                and torch.cuda.is_available()
            )
            else None
        ),
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def model_info():
    class_mapping = {}

    if classes is not None:
        try:
            normalized = get_class_mapping(classes)
            class_mapping = {
                str(index): name
                for index, name in normalized.items()
            }
        except Exception:
            class_mapping = {}

    return {
        "architecture": "ResNet18",
        "classes": class_mapping,
        "input_size": [224, 224],
        "validation_accuracy": 96.07,
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded"
        )

    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file format. "
                "Use JPG, JPEG, PNG or WEBP."
            )
        )

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                "File too large. "
                "Maximum size is 10 MB."
            )
        )

    try:
        image = Image.open(
            BytesIO(file_bytes)
        ).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    if not model_loaded:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded"
        )

    if (
        model is None
        or transform is None
        or device is None
        or classes is None
    ):
        raise HTTPException(
            status_code=500,
            detail="Model inference components are unavailable"
        )

    try:
        # Normalize class representation.
        class_mapping = get_class_mapping(
            classes
        )

        real_idx, ai_idx = get_real_and_ai_indices(
            class_mapping
        )

        # Exact preprocessing used during model training.
        input_tensor = transform(
            image
        ).unsqueeze(0).to(device)

        # Real model inference.
        with torch.no_grad():
            output = model(
                input_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        predicted_index = prediction.item()

        real_probability = (
            probabilities[0][real_idx].item() * 100
        )

        ai_probability = (
            probabilities[0][ai_idx].item() * 100
        )

        confidence_percent = (
            confidence.item() * 100
        )

        if predicted_index == real_idx:
            predicted_class = "real"
            display_label = "Likely real"
        else:
            predicted_class = "ai_generated"
            display_label = "Likely AI-generated"

        result = {
            "success": True,
            "prediction": predicted_class,
            "display_label": display_label,
            "confidence": round(
                confidence_percent,
                2
            ),
            "probabilities": {
                "real": round(
                    real_probability,
                    2
                ),
                "ai_generated": round(
                    ai_probability,
                    2
                )
            },
            "model": {
                "architecture": "ResNet18",
                "device": device.type
            }
        }

        return JSONResponse(
            content=result
        )

    except HTTPException:
        raise

    except Exception as exc:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ============================================================
# FRONTEND
# ============================================================

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(
            directory=str(STATIC_DIR)
        ),
        name="static"
    )


@app.get("/")
def read_root():
    html_path = STATIC_DIR / "index.html"

    if html_path.exists():
        return HTMLResponse(
            content=html_path.read_text(
                encoding="utf-8"
            ),
            status_code=200
        )

    return HTMLResponse(
        content="<h1>index.html not found in static folder</h1>",
        status_code=404
    )


# ============================================================
# LOCAL DEVELOPMENT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
