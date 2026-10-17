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
import numpy as np
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
    title="Apple Leaf Alternaria Detection API",
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

# Accepted values of the optional `model` form field on /api/predict
MODEL_OPTIONS = {
    "ensemble": "legacy default: 4-model majority vote, one final prediction (used when `model` is omitted)",
    "all": "run all four models on the same image and return every model's prediction for comparison",
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
        "service": "Apple Leaf Alternaria Detection API",
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


# Tie-breaker model label reported in debug diagnostics. Legacy validation-accuracy
# rankings were removed with the old checkpoints; re-rank after the retrained evaluation.
TIE_BREAKER_MODEL = "mobilenetv2"


def resolve_model_mode(model: Optional[str]):
    """Maps the optional `model` form field to (mode, keys). Omitted / 'ensemble' keeps the legacy behaviour."""
    value = (model or "").strip().lower()
    if value in ("", "ensemble", "consensus", "vote"):
        return "ensemble", list(MODEL_SPECS.keys())
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

    votes = {"Healthy": sum(1 for r in results if r["prediction"] == "Healthy")}
    votes["Alternaria"] = len(results) - votes["Healthy"]
    if votes["Healthy"] == votes["Alternaria"]:
        majority, majority_label = None, "No majority (2-2 split)"
    else:
        majority = "Healthy" if votes["Healthy"] > votes["Alternaria"] else "Alternaria"
        majority_label = "Healthy Leaf" if majority == "Healthy" else "Alternaria Leaf Spot"
    disagreeing = [r["model_name"] for r in results if majority is not None and r["prediction"] != majority]
    return {
        "status": "success", "stage": "Stage 3 (Complete)", "mode": "all",
        **common,
        "model_results": results,
        "agreement": {
            "unanimous": len({r["prediction"] for r in results}) == 1,
            "votes": votes,
            "majority_prediction": majority,
            "majority_label": majority_label,
            "disagreeing_models": disagreeing,
            "note": "Majority of four independent model outputs shown for comparison only; it is not a validated clinical or scientific consensus.",
        },
        "confidence_type": "model_confidence", "confidence_note": confidence_note,
        "treatment_guidance": _treatment(majority, majority_label) if majority else None,
    }


@app.get("/api/predict")
@app.get("/predict")
def predict_info():
    """Informational response for GET requests to the prediction endpoint."""
    return {
        "status": "active",
        "service": "Apple Leaf Alternaria Detection Pathology Classifier",
        "method": "POST",
        "expected_payload": "multipart/form-data with 'file' (image file)",
        "voting_ensemble": "4-model majority voting (MobileNetV2, ResNet18, VGG16, VGG19)",
        "model_options": MODEL_OPTIONS,
        "tie_breaker": "Healthy Leaf (conservative rule: disease requires strict majority evidence)",
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
      Stage 3: 4-Model majority-voting disease classification (Healthy Leaf vs Alternaria Leaf Spot)
    Optional form field `model`: omitted/'ensemble' = legacy voting; 'all' = per-model comparison; a model key = that model only.
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

    # STAGE 3 (explicit modes): one selected model, or all four reported individually
    if mode in ("single", "all"):
        return _stage3_selected(mode, selected_keys, image, common_fields)

    # STAGE 3: DISEASE CLASSIFICATION (All 4 Comparative Architectures via Majority Voting)
    try:
        disease_input = DISEASE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

        votes: Dict[str, str] = {}
        probs_by_model: Dict[str, Dict[str, float]] = {}

        with torch.inference_mode():
            for m_key in MODEL_SPECS:
                _, m_model = get_disease_model(m_key)
                m_logits = m_model(disease_input)
                m_probs = torch.softmax(m_logits, dim=1)[0]

                m_pred_idx = int(torch.argmax(m_probs).item())
                m_pred_cls = CLASSES[m_pred_idx]
                m_prob_h = float(m_probs[0].item())
                m_prob_a = float(m_probs[1].item())

                votes[m_key] = m_pred_cls
                probs_by_model[m_key] = {
                    "Healthy": round(m_prob_h, 4),
                    "Alternaria": round(m_prob_a, 4),
                }

        # Multi-model majority voting logic:
        # Each model casts one vote based on its predicted class.
        n_alternaria = sum(1 for cls_name in votes.values() if cls_name == "Alternaria")
        n_healthy = 4 - n_alternaria

        is_tie = False
        tie_break_used = False

        if n_healthy >= 3:
            final_prediction = "Healthy"
            final_label = "Healthy Leaf"
            explanation = (
                f"Consensus diagnosis: {n_healthy} of 4 transfer learning models voted Healthy Leaf."
                if n_healthy < 4
                else "Unanimous consensus across all 4 transfer learning models (4/4 voted Healthy Leaf)."
            )
        elif n_alternaria >= 3:
            final_prediction = "Alternaria"
            final_label = "Alternaria Leaf Spot"
            explanation = (
                f"Consensus diagnosis: {n_alternaria} of 4 transfer learning models voted Alternaria Leaf Spot."
                if n_alternaria < 4
                else "Unanimous consensus across all 4 transfer learning models (4/4 voted Alternaria Leaf Spot)."
            )
        else:
            # 2–2 tie: When 2 models predict Healthy and 2 predict Alternaria,
            # conservative tie-breaking rule resolves to Healthy Leaf to prevent false-positive disease misdiagnosis.
            is_tie = True
            tie_break_used = True
            final_prediction = "Healthy"
            final_label = "Healthy Leaf"
            explanation = (
                "2–2 tie between models resolved as Healthy Leaf (conservative rule: disease requires strict majority evidence)."
            )

        # Defensible confidence calculation:
        # Mean posterior probability assigned to the winning class across the models in the winning coalition
        winning_coalition = [m for m, cls_name in votes.items() if cls_name == final_prediction]
        winning_probs = [probs_by_model[m][final_prediction] for m in winning_coalition]
        ensemble_confidence = float(np.mean(winning_probs))

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "stage": "Stage 3 (Disease Classification)",
                "message": f"Disease classifier failure: {exc}",
            },
        )

    response_data: Dict[str, Any] = {
        "status": "success",
        "stage": "Stage 3 (Complete)",
        "mode": "ensemble",
        "similarity": round(float(similarity), 4),
        "crop_status": crop_status,
        "crop_notice": crop_notice,
        "predicted_crop": predicted_crop,
        "crop_confidence": round(crop_confidence, 4),
        "apple_confidence": round(apple_confidence, 4),
        "final_prediction": final_prediction,
        "final_label": final_label,
        "confidence": round(ensemble_confidence, 4),
        "explanation": explanation,
        "treatment_guidance": {
            "title": final_label,
            "pathogen": "Alternaria leaf spot, a fungal disease of apple (Alternaria alternata)" if final_prediction == "Alternaria" else "No signs of Alternaria leaf spot in this photograph",
            "symptoms": "Small, round brown spots on the leaf, sometimes with yellowish tissue around them; spots can merge as the disease advances." if final_prediction == "Alternaria" else "Even green colour, normal veins, no spots seen.",
            "recommended_action": "Treat this as a screening result. Have the leaf examined by a qualified agricultural expert or your local horticulture extension service before choosing any treatment." if final_prediction == "Alternaria" else "Continue routine orchard care and keep checking leaves regularly. A single photograph cannot rule out disease elsewhere on the tree.",
        },
    }

    if debug:
        response_data["internal_diagnostics"] = {
            "votes": votes,
            "vote_counts": {"Healthy": n_healthy, "Alternaria": n_alternaria},
            "is_tie": is_tie,
            "tie_break_used": tie_break_used,
            "tie_breaker_model": TIE_BREAKER_MODEL,
            "winning_coalition": winning_coalition,
            "individual_probabilities": probs_by_model,
        }

    return response_data
