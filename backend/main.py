import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import timm
from timm.data import create_transform, resolve_model_data_config

# Setup project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / "code"
MODEL_DIR = PROJECT_ROOT / "models" / "retrained"  # retrained checkpoints only; legacy weights were removed
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CODE_DIR))

# Import gatekeeper & model builder
import importlib.util
from train_comparative_models import build_comparative_model

app = FastAPI(
    title="AppleAlt API",
    description="Inference backend for 3-stage validation and 4-model comparative disease classification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
APPLE_CONFIDENCE_THRESHOLD = 0.60
CLASSES = ["Healthy", "Alternaria"]

MODEL_SPECS = {
    "mobilenetv2": {"display_name": "MobileNetV2", "filename": "mobilenetv2_best.pth"},
    "resnet18": {"display_name": "ResNet18", "filename": "resnet18_best.pth"},
    "vgg16": {"display_name": "VGG16", "filename": "vgg16_best.pth"},
    "vgg19": {"display_name": "VGG19", "filename": "vgg19_best.pth"},
}

DISEASE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Global model caches
_gatekeeper_module = None
_crop_model = None
_crop_transform = None
_crop_labels = None
_disease_models: Dict[str, torch.nn.Module] = {}
_model_load_lock = threading.Lock()   # guards lazy checkpoint loading / cache mutation across concurrent requests

# Model used when the `model` form field is omitted (recommended for deployment)
DEFAULT_MODEL = "mobilenetv2"

# Accepted values of the optional `model` form field on /api/predict
MODEL_OPTIONS = {
    "all": "run all four models on the same image and return each model's prediction separately",
    **{k: f"run only {v['display_name']}" for k, v in MODEL_SPECS.items()},
}


def get_gatekeeper():
    global _gatekeeper_module
    if _gatekeeper_module is None:
        script_path = CODE_DIR / "04_quick_inference.py"
        prototype_path = CODE_DIR / "leaf_prototype.pt"
        if not script_path.exists():
            raise FileNotFoundError(f"Gatekeeper script not found: {script_path}")
        if not prototype_path.exists():
            raise FileNotFoundError(f"Prototype file not found: {prototype_path}")
        
        spec = importlib.util.spec_from_file_location("apple_leaf_gatekeeper", script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _gatekeeper_module = module
    return _gatekeeper_module


def get_crop_detector():
    global _crop_model, _crop_transform, _crop_labels
    if _crop_model is None:
        model_id = "hf-hub:ixrbhii/leaf-crop-detector-36-convnext"
        model = timm.create_model(model_id, pretrained=True)
        model.to(DEVICE)
        model.eval()

        data_config = resolve_model_data_config(model)
        transform = create_transform(**data_config, is_training=False)
        labels = model.pretrained_cfg.get("label_names", [])
        if not labels:
            raise ValueError("Failed to retrieve label names from pretrained_cfg.")

        _crop_model = model
        _crop_transform = transform
        _crop_labels = labels
    return _crop_model, _crop_transform, _crop_labels


def get_disease_model(model_key: str):
    """Thread-safe, lazily populated cache: a checkpoint is read from disk only the first time it is requested."""
    with _model_load_lock:
        return _get_disease_model_locked(model_key)


def _get_disease_model_locked(model_key: str):
    key = model_key.lower().strip()
    if key not in MODEL_SPECS:
        # fuzzy match
        for k in MODEL_SPECS:
            if k in key:
                key = k
                break
    if key not in MODEL_SPECS:
        key = "mobilenetv2"

    if key not in _disease_models:
        spec = MODEL_SPECS[key]
        ckpt_path = MODEL_DIR / spec["filename"]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
        
        model = build_comparative_model(key, num_classes=2)
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
        
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        _disease_models[key] = model

    return key, _disease_models[key]


@app.on_event("startup")
def startup_event():
    # Warm up models in background
    try:
        get_gatekeeper()
        get_crop_detector()
        get_disease_model("mobilenetv2")
    except Exception as exc:
        print(f"Warning during model warmup: {exc}")


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "AppleAlt API",
        "version": "1.0.0",
        "frontend_url": "http://localhost:3000",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict",
            "metrics": "/api/metrics",
        },
    }


@app.get("/api/health")
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "available_models": list(MODEL_SPECS.keys()),
        "loaded_models": list(_disease_models.keys()),
        "model_options": list(MODEL_OPTIONS.keys()),
        "apple_confidence_threshold": APPLE_CONFIDENCE_THRESHOLD,
    }


@app.get("/api/metrics")
@app.get("/metrics")
def get_metrics():
    """
    Returns empirical metrics from results/retrained_evaluation/comparison.csv
    and training history from results/retrained/<model>/history.csv.
    Responds 404 until the retraining pipeline has produced these files.
    """
    comparison_file = RESULTS_DIR / "retrained_evaluation" / "comparison.csv"
    if not comparison_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation metrics not available yet: results/retrained_evaluation/comparison.csv not found (retraining pending).",
        )

    import csv
    models_comparison = []
    with open(comparison_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse confusion matrix
            try:
                cm = json.loads(row["confusion_matrix"])
            except Exception:
                cm = [[0, 0], [0, 0]]
            models_comparison.append({
                "model_name": row["model_name"],
                "display_name": MODEL_SPECS.get(row["model_name"], {}).get("display_name", row["model_name"]),
                "test_samples": int(row["test_samples"]),
                "test_accuracy": float(row["test_accuracy"]),
                "precision_healthy": float(row["precision_healthy"]),
                "recall_healthy": float(row["recall_healthy"]),
                "f1_healthy": float(row["f1_healthy"]),
                "support_healthy": int(row["support_healthy"]),
                "precision_alternaria": float(row["precision_alternaria"]),
                "recall_alternaria": float(row["recall_alternaria"]),
                "f1_alternaria": float(row["f1_alternaria"]),
                "support_alternaria": int(row["support_alternaria"]),
                "precision_macro": float(row["precision_macro"]),
                "recall_macro": float(row["recall_macro"]),
                "f1_macro": float(row["f1_macro"]),
                "total_params": int(row["total_params"]),
                "file_size_mb": float(row["file_size_mb"]),
                "latency_ms_per_image": float(row["latency_ms_per_image"]),
                "throughput_images_per_sec": float(row["throughput_images_per_sec"]),
                "confusion_matrix": cm,
            })

    # Read training histories
    training_histories = {}
    for m_key in MODEL_SPECS.keys():
        hist_path = RESULTS_DIR / "retrained" / m_key / "history.csv"
        if hist_path.exists():
            epochs = []
            with open(hist_path, mode="r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for erow in r:
                    epochs.append({
                        "epoch": int(erow["epoch"]),
                        "train_loss": float(erow["train_loss"]),
                        "val_loss": float(erow["val_loss"]),
                        "train_acc": float(erow["train_acc"]),
                        "val_acc": float(erow["val_acc"]),
                        "epoch_duration_sec": float(erow["epoch_duration_sec"]),
                    })
            training_histories[m_key] = epochs

    return {
        "models": models_comparison,
        "histories": training_histories,
        "recommended_model": "mobilenetv2",
    }


def resolve_model_mode(model: Optional[str]):
    """Maps the optional `model` form field to (mode, keys). Omitted = DEFAULT_MODEL only."""
    value = (model or "").strip().lower()
    if value == "":
        return "single", [DEFAULT_MODEL]
    if value == "all":
        return "all", list(MODEL_SPECS.keys())
    if value in MODEL_SPECS:
        return "single", [value]
    raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Valid values: {list(MODEL_OPTIONS.keys())}")


def _treatment(prediction: str, title: str) -> Dict[str, str]:
    if prediction == "Alternaria":
        return {"title": title, "pathogen": "Alternaria leaf spot, a fungal disease of apple (Alternaria alternata)",
                "symptoms": "Small, round brown spots on the leaf, sometimes with yellowish tissue around them; spots can merge as the disease advances.",
                "recommended_action": "Treat this as a screening result. Have the leaf examined by a qualified agricultural expert or your local horticulture extension service before choosing any treatment."}
    return {"title": title, "pathogen": "No signs of Alternaria leaf spot in this photograph",
            "symptoms": "Even green colour, normal veins, no spots seen.",
            "recommended_action": "Continue routine orchard care and keep checking leaves regularly. A single photograph cannot rule out disease elsewhere on the tree."}


def _run_single_model(key: str, disease_input: torch.Tensor) -> Dict[str, Any]:
    """One forward pass of one cached model; returns prediction, probabilities and single-image inference time."""
    _, m_model = get_disease_model(key)
    with torch.inference_mode():
        t0 = time.perf_counter()
        logits = m_model(disease_input)
        probs = torch.softmax(logits, dim=1)[0]
        ms = (time.perf_counter() - t0) * 1000.0
    idx = int(torch.argmax(probs).item())
    pred = CLASSES[idx]
    return {
        "model_key": key,
        "model_name": MODEL_SPECS[key]["display_name"],
        "prediction": pred,
        "label": "Healthy Leaf" if pred == "Healthy" else "Alternaria Leaf Spot",
        "confidence": round(float(probs[idx].item()), 4),          # predicted-class probability of THIS model
        "probabilities": {"Healthy": round(float(probs[0].item()), 4), "Alternaria": round(float(probs[1].item()), 4)},
        "inference_ms": round(ms, 1),                                # single image, forward pass only, server CPU
    }


def _stage3_selected(mode: str, keys: List[str], image: Image.Image, common: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 3 for the explicit modes: one selected model, or all four reported individually."""
    disease_input = DISEASE_TRANSFORM(image).unsqueeze(0).to(DEVICE)  # identical preprocessing for every model
    results = []
    for key in keys:
        try:
            results.append(_run_single_model(key, disease_input))
        except FileNotFoundError as exc:
            return JSONResponse(status_code=503, content={"status": "error", "stage": "Stage 3 (Model Loading)",
                                                          "message": f"Checkpoint for {MODEL_SPECS[key]['display_name']} is not available: {exc}"})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"status": "error", "stage": "Stage 3 (Disease Classification)",
                                                          "message": f"{MODEL_SPECS[key]['display_name']} failed: {type(exc).__name__}: {exc}"})

    confidence_note = "Model confidence is the model's predicted-class probability for this image, not a guarantee that the prediction is correct."
    if mode == "single":
        r = results[0]
        return {
            "status": "success", "stage": "Stage 3 (Complete)", "mode": "single",
            **common,
            "model_key": r["model_key"], "model_name": r["model_name"],
            "final_prediction": r["prediction"], "final_label": r["label"],
            "confidence": r["confidence"], "confidence_type": "model_confidence", "confidence_note": confidence_note,
            "probabilities": r["probabilities"], "inference_ms": r["inference_ms"],
            "explanation": f"Single-model prediction by {r['model_name']}: {r['label']} with model confidence {r['confidence'] * 100:.1f}%.",
            "treatment_guidance": _treatment(r["prediction"], r["label"]),
        }

    # Four independent results, reported as-is: no vote, no combined prediction.
    return {
        "status": "success", "stage": "Stage 3 (Complete)", "mode": "all",
        **common,
        "model_results": results,
        "results_note": "Each model analysed the image independently; the four results are reported separately.",
        "confidence_type": "model_confidence", "confidence_note": confidence_note,
    }


@app.get("/api/predict")
@app.get("/predict")
def predict_info():
    """Informational response for GET requests to the prediction endpoint."""
    return {
        "status": "active",
        "service": "AppleAlt Pathology Classifier",
        "method": "POST",
        "expected_payload": "multipart/form-data with 'file' (image file)",
        "default_model": DEFAULT_MODEL,
        "model_options": MODEL_OPTIONS,
        "web_interface": "http://localhost:3000/detection",
    }


@app.post("/api/predict")
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    debug: bool = Form(False),
):
    """
    Executes the 3-Stage Cascaded Diagnostic Pipeline:
      Stage 1: Foliar authenticity gatekeeper
      Stage 2: 36-Crop plant species identification
      Stage 3: Disease classification (Healthy Leaf vs Alternaria Leaf Spot)
    Optional form field `model`: a model key = that model only (omitted = MobileNetV2);
    'all' = all four models, each result reported separately and never combined.
    """
    mode, selected_keys = resolve_model_mode(model)

    # Read image
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
        image = ImageOps.exif_transpose(pil_img).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    # STAGE 1: GATEKEEPER
    try:
        gatekeeper = get_gatekeeper()
        is_leaf, similarity, details, _ = gatekeeper.evaluate_leaf_authenticity(image)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "stage": "Stage 1 (Gatekeeper)",
                "message": f"Internal error during leaf authenticity evaluation: {exc}",
            },
        )

    if isinstance(details, dict) and "error" in details:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "stage": "Stage 1 (Gatekeeper)",
                "message": details["error"],
            },
        )

    if not is_leaf:
        return {
            "status": "rejected_non_leaf",
            "stage": "Stage 1 (Leaf Authenticity)",
            "similarity": round(float(similarity), 4),
            "gatekeeper_threshold": 0.33,
            "message": "The uploaded image was rejected as non-foliar content (e.g. room, screen, scenery, or non-plant object). Please upload a clear leaf image.",
        }

    # STAGE 2: CROP SPECIES IDENTIFICATION
    try:
        crop_model, crop_transform, crop_labels = get_crop_detector()
        crop_input = crop_transform(image).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            crop_logits = crop_model(crop_input)
            crop_probs = torch.softmax(crop_logits, dim=1)[0]

        top_crop_idx = torch.argmax(crop_probs).item()
        predicted_crop = crop_labels[top_crop_idx]
        crop_confidence = float(crop_probs[top_crop_idx].item())

        apple_idx = crop_labels.index("Apple") if "Apple" in crop_labels else 0
        apple_confidence = float(crop_probs[apple_idx].item())

        top3_vals, top3_idxs = torch.topk(crop_probs, 3)
        top3_crops = [
            {"crop": crop_labels[idx.item()], "confidence": round(float(val.item()), 4)}
            for val, idx in zip(top3_vals, top3_idxs)
        ]

        if predicted_crop == "Apple" and crop_confidence >= APPLE_CONFIDENCE_THRESHOLD:
            crop_status = "confirmed"
            crop_notice = "Apple foliage verified with high confidence."
        elif predicted_crop == "Apple":
            crop_status = "likely_apple"
            crop_notice = f"Apple foliage detected with moderate confidence ({crop_confidence * 100:.1f}%). Alternaria lesion patterns may attenuate foliar embeddings. Proceeding to pathology diagnosis."
        else:
            crop_status = "ambiguous_foliage"
            crop_notice = f"Stage 2 crop classifier predicted {predicted_crop} ({crop_confidence * 100:.1f}%). Foliar necrosis, chlorotic halos, or target spots frequently alter species classification embeddings. Passing specimen directly to Apple disease classifier."

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "stage": "Stage 2 (Crop Identification)",
                "message": f"Crop detector inference failure: {exc}",
            },
        )

    common_fields = {
        "similarity": round(float(similarity), 4),
        "crop_status": crop_status, "crop_notice": crop_notice,
        "predicted_crop": predicted_crop, "crop_confidence": round(crop_confidence, 4),
        "apple_confidence": round(apple_confidence, 4),
    }
    if debug:
        common_fields["top3_crops"] = top3_crops

    # STAGE 3: one selected model (MobileNetV2 by default), or all four reported individually
    return _stage3_selected(mode, selected_keys, image, common_fields)
