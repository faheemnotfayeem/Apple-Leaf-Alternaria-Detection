"""
Apple Leaf Alternaria Detection: Production Diagnostic Web Application
========================================================================

Three-Stage Diagnostic Pipeline:
  Stage 1: ImageNet MobileNetV2 Deep Feature Embedding Gatekeeper
           (Rejects non-foliar images: desks, rooms, scenery, furniture, screens)
  Stage 2: Hugging Face 36-Crop ConvNeXt-Small Species Classifier
           (hf-hub:ixrbhii/leaf-crop-detector-36-convnext)
           (Verifies leaf belongs to 'Apple'; rejects other recognized plant species)
  Stage 3: 4-Model Majority Voting Foliar Pathology Engine
           (MobileNetV2, ResNet18, VGG16, VGG19 -> Healthy Leaf vs Alternaria Leaf Spot)
"""

import os
import sys
import importlib.util
from pathlib import Path

import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms

import timm
from timm.data import create_transform, resolve_model_data_config

# =========================================================
# 1. PATHS AND SYSTEM CONFIGURATION
# =========================================================
ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
MODEL_DIR = ROOT / "models" / "retrained"  # retrained checkpoints only; legacy weights were removed

sys.path.insert(0, str(CODE_DIR))

# Verification Thresholds & Class Mappings
APPLE_CONFIDENCE_THRESHOLD = 0.60  # Minimum crop detector confidence required for Apple
CLASSES = ["Healthy", "Alternaria"]

MODEL_FILES = {
    "MobileNetV2": ("mobilenetv2", "mobilenetv2_best.pth"),
    "ResNet18": ("resnet18", "resnet18_best.pth"),
    "VGG16": ("vgg16", "vgg16_best.pth"),
    "VGG19": ("vgg19", "vgg19_best.pth"),
}

TREATMENT_GUIDE = {
    "Alternaria": {
        "title": "ALTERNARIA LEAF SPOT",
        "pathogen": "Fungal Infection (Pathogen: Alternaria alternata)",
        "symptoms": "Circular brown to dark-brown necrotic lesions with characteristic concentric target rings and yellow chlorotic halos.",
        "action": "Spray protective/curative fungicide: Mancozeb 75 WP (2.5 g/L) or Propineb 70 WP (2.0 g/L). Prune infected canopy shoots to improve canopy aeration."
    },
    "Healthy": {
        "title": "HEALTHY APPLE FOLIAGE",
        "pathogen": "Healthy Apple Foliage (Baseline Control)",
        "symptoms": "Uniform emerald-green lamina, normal venation structure, zero necrotic foliar lesions.",
        "action": "Maintain routine orchard sanitation, balanced N-P-K nutrient application, and regular drip irrigation."
    }
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing for Stage 3 Disease Classification (Matches Training & Validation)
DISEASE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

st.set_page_config(
    page_title="Apple Leaf Alternaria Detection",
    page_icon="🍃",
    layout="centered",
)

# =========================================================
# 2. CACHED RESOURCE LOADERS
# =========================================================

# A. Load Model Builder
try:
    from train_comparative_models import build_comparative_model
except Exception as exc:
    st.error("Could not import model architecture factory from 'code/train_comparative_models.py'.")
    st.exception(exc)
    st.stop()

# B. Load Stage 1 Leaf Authenticity Gatekeeper
@st.cache_resource
def load_gatekeeper():
    """
    Loads the Stage 1 deep feature gatekeeper module from code/04_quick_inference.py.
    """
    script_path = CODE_DIR / "04_quick_inference.py"
    prototype_path = CODE_DIR / "leaf_prototype.pt"

    if not script_path.exists():
        raise FileNotFoundError(f"Leaf validator script not found: {script_path}")
    if not prototype_path.exists():
        raise FileNotFoundError(f"Apple leaf prototype not found: {prototype_path}")

    spec = importlib.util.spec_from_file_location("apple_leaf_gatekeeper", script_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not initialize spec for leaf validator.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "evaluate_leaf_authenticity"):
        raise AttributeError("Module does not define evaluate_leaf_authenticity().")

    return module

# C. Load Stage 2 Hugging Face 36-Crop Detector (ConvNeXt-Small)
@st.cache_resource
def load_crop_detector():
    """
    Loads pretrained 36-crop ConvNeXt-small classifier from Hugging Face Hub using timm.
    Repository: hf-hub:ixrbhii/leaf-crop-detector-36-convnext
    """
    model_id = "hf-hub:ixrbhii/leaf-crop-detector-36-convnext"
    model = timm.create_model(model_id, pretrained=True)
    model.to(DEVICE)
    model.eval()

    # Resolve official preprocessing data configuration
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)

    # Extract verified class label names
    label_names = model.pretrained_cfg.get("label_names", [])
    if not label_names:
        raise ValueError("Could not retrieve class label names from model pretrained_cfg.")

    return model, transform, label_names

# D. Load Stage 3 Disease Classification Model
@st.cache_resource
def load_disease_model(display_name: str):
    """
    Loads one of the four retrained comparative disease models from models/retrained/<model_name>_best.pth.
    """
    matched_key = None
    for k in MODEL_FILES:
        if k.lower() in display_name.lower():
            matched_key = k
            break
    if matched_key is None:
        if display_name in MODEL_FILES:
            matched_key = display_name
        else:
            raise KeyError(f"Unknown disease model '{display_name}'. Expected one of {list(MODEL_FILES.keys())}")

    model_name, filename = MODEL_FILES[matched_key]
    checkpoint_path = MODEL_DIR / filename

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint for {display_name} not found at: {checkpoint_path}. "
            "Stage 3 inference is unavailable until the retraining pipeline writes new weights to models/retrained/."
        )

    model = build_comparative_model(model_name, num_classes=2)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model

# =========================================================
# 3. THREE-STAGE DIAGNOSTIC INFERENCE PIPELINE
# =========================================================

def run_diagnostic_pipeline(uploaded_image, gatekeeper, crop_detector_bundle):
    """
    Executes the cascaded 3-stage diagnostic pipeline:
      Stage 1: Gatekeeper foliar authenticity verification.
      Stage 2: 36-crop plant species identification.
      Stage 3: Two-class apple leaf pathology classification.
    """
    crop_model, crop_transform, crop_labels = crop_detector_bundle

    # Ensure orientation is normalized and mode is RGB
    image = ImageOps.exif_transpose(uploaded_image).convert("RGB")

    # ---------------------------------------------------------
    # STAGE 1: LEAF AUTHENTICITY GATEKEEPER
    # ---------------------------------------------------------
    try:
        is_leaf, similarity, gatekeeper_details, _ = gatekeeper.evaluate_leaf_authenticity(image)
    except Exception as exc:
        return {
            "status": "error",
            "stage": "Stage 1 (Gatekeeper)",
            "message": f"Internal error during leaf authenticity evaluation: {exc}",
            "details": {"exception": str(exc)},
        }

    # If the gatekeeper encountered a read error
    if isinstance(gatekeeper_details, dict) and "error" in gatekeeper_details:
        return {
            "status": "error",
            "stage": "Stage 1 (Gatekeeper)",
            "message": gatekeeper_details["error"],
            "details": gatekeeper_details,
        }

    if not is_leaf:
        return {
            "status": "rejected_non_leaf",
            "stage": "Stage 1 (Leaf Authenticity)",
            "similarity": float(similarity),
            "details": gatekeeper_details,
            "message": "The uploaded image was rejected as non-foliar content (e.g., room, scenery, screen, or non-plant object). Please upload a clear leaf image.",
        }

    # ---------------------------------------------------------
    # STAGE 2: CROP SPECIES IDENTIFICATION
    # ---------------------------------------------------------
    try:
        crop_input = crop_transform(image).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            crop_logits = crop_model(crop_input)
            crop_probs = torch.softmax(crop_logits, dim=1)[0]

        top_crop_idx = torch.argmax(crop_probs).item()
        predicted_crop = crop_labels[top_crop_idx]
        crop_confidence = float(crop_probs[top_crop_idx].item())

        apple_idx = crop_labels.index("Apple") if "Apple" in crop_labels else 0
        apple_confidence = float(crop_probs[apple_idx].item())

        # Top 3 predictions for transparent reporting
        top3_vals, top3_idxs = torch.topk(crop_probs, 3)
        top3_crops = [
            (crop_labels[idx.item()], float(val.item()))
            for val, idx in zip(top3_vals, top3_idxs)
        ]

    except Exception as exc:
        return {
            "status": "error",
            "stage": "Stage 2 (Crop Identification)",
            "message": f"Crop detector inference failure: {exc}",
            "details": {"exception": str(exc)},
        }

    # Determine Crop Verification Status (non-blocking for genuine foliage)
    if predicted_crop == "Apple" and crop_confidence >= APPLE_CONFIDENCE_THRESHOLD:
        crop_status = "confirmed"
        crop_notice = f"Apple foliage verified with high confidence ({crop_confidence * 100:.1f}%)."
    elif predicted_crop == "Apple":
        crop_status = "likely_apple"
        crop_notice = f"Apple foliage detected with moderate confidence ({crop_confidence * 100:.1f}%). Foliar lesion textures may affect species embeddings. Proceeding to pathology diagnosis."
    else:
        crop_status = "ambiguous_foliage"
        crop_notice = f"Stage 2 crop classifier predicted {predicted_crop} ({crop_confidence * 100:.1f}%). Foliar necrosis, chlorotic halos, or target spots frequently alter species classification embeddings. Passing specimen directly to Apple disease classifier."

    # ---------------------------------------------------------
    # STAGE 3: PATHOLOGY CLASSIFICATION (4-Model Majority Voting)
    # ---------------------------------------------------------
    try:
        disease_input = DISEASE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

        votes = {}
        probs_by_model = {}
        with torch.inference_mode():
            for m_disp_name in MODEL_FILES:
                m_model = load_disease_model(m_disp_name)
                m_logits = m_model(disease_input)
                m_probs = torch.softmax(m_logits, dim=1)[0]

                m_pred_idx = int(torch.argmax(m_probs).item())
                m_pred_cls = CLASSES[m_pred_idx]
                m_prob_h = float(m_probs[0].item())
                m_prob_a = float(m_probs[1].item())

                votes[m_disp_name] = m_pred_cls
                probs_by_model[m_disp_name] = {
                    "Healthy": m_prob_h,
                    "Alternaria": m_prob_a,
                }

        # Multi-model majority voting logic
        n_alternaria = sum(1 for res in votes.values() if res == "Alternaria")
        n_healthy = 4 - n_alternaria

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
            # 2–2 tie fallback: consider as Healthy Leaf
            final_prediction = "Healthy"
            final_label = "Healthy Leaf"
            explanation = (
                "2–2 tie between models resolved as Healthy Leaf (conservative rule: disease requires strict majority evidence)."
            )

        winning_coalition = [m for m, cls_name in votes.items() if cls_name == final_prediction]
        winning_probs = [probs_by_model[m][final_prediction] for m in winning_coalition]
        ensemble_confidence = float(np.mean(winning_probs))

    except Exception as exc:
        return {
            "status": "error",
            "stage": "Stage 3 (Disease Classification)",
            "message": f"Disease classifier failure: {exc}",
            "details": {"exception": str(exc)},
        }

    return {
        "status": "success",
        "crop_status": crop_status,
        "crop_notice": crop_notice,
        "predicted_crop": predicted_crop,
        "crop_confidence": crop_confidence,
        "apple_confidence": apple_confidence,
        "top3_crops": top3_crops,
        "final_prediction": final_prediction,
        "final_label": final_label,
        "confidence": ensemble_confidence,
        "explanation": explanation,
        "similarity": float(similarity),
        "details": gatekeeper_details,
    }

# =========================================================
# 4. STREAMLIT USER INTERFACE
# =========================================================

st.title("Apple Leaf Alternaria Detection")
st.markdown("Automated deep-learning pathology diagnostic system for foliar Alternaria detection.")

# Pipeline Architecture Callout
st.info(
    "🛡️ **Cascaded 3-Stage Vision Pipeline Active:**  \n"
    "1. **Stage 1 (Gatekeeper)**: MobileNetV2 embedding distance filter (blocks desks, rooms, scenery, screens).  \n"
    "2. **Stage 2 (Species Validator)**: Hugging Face 36-Crop ConvNeXt (`Apple` threshold: **60.0%**).  \n"
    "3. **Stage 3 (Pathology Consensus)**: 4-Model majority voting across MobileNetV2, ResNet18, VGG16, and VGG19."
)

st.caption(f"**Hardware Device:** `{DEVICE.type.upper()}` | **Species Acceptance Threshold:** `{APPLE_CONFIDENCE_THRESHOLD * 100:.0f}% Apple Confidence`")

uploaded_file = st.file_uploader(
    "Upload an apple leaf photograph for pathology diagnosis:",
    type=["jpg", "jpeg", "png", "webp"],
    help="Accepts JPG, JPEG, PNG, and WEBP apple leaf images.",
)

if uploaded_file is not None:
    try:
        preview = Image.open(uploaded_file)
        preview = ImageOps.exif_transpose(preview).convert("RGB")

        st.image(preview, caption="Uploaded Specimen", use_container_width=True)

        if st.button("🔍 Analyze Leaf", type="primary", use_container_width=True):
            with st.spinner("Executing cascaded 3-stage diagnostic pipeline..."):
                try:
                    gatekeeper = load_gatekeeper()
                    crop_bundle = load_crop_detector()
                    result = run_diagnostic_pipeline(
                        preview, gatekeeper, crop_bundle
                    )
                except Exception as exc:
                    st.error("Model initialization failure.")
                    st.exception(exc)
                    st.stop()

            # --- RENDER RESULTS BASED ON STATUS ---

            if result["status"] == "error":
                st.error(f"❌ **System Error in {result['stage']}:**")
                st.write(result["message"])
                with st.expander("Technical Error Details"):
                    st.json(result["details"])

            elif result["status"] == "rejected_non_leaf":
                st.error("🚫 **Stage 1 Rejection: Non-Foliar Content Detected**")
                st.warning(result["message"])
                st.metric("Leaf Manifold Similarity", f"{result['similarity']:.4f}", help="Threshold: ≥ 0.3300")
                with st.expander("Stage 1 Gatekeeper Details"):
                    st.json(result["details"])

            elif result["status"] == "success":
                # Render Species Verification Status
                if result.get("crop_status") == "confirmed":
                    st.success("✅ **Species Verification: Confirmed Apple Leaf Foliage**")
                else:
                    st.info(f"ℹ️ **Species Verification Telemetry:** {result.get('crop_notice')}")

                st.markdown("---")
                st.subheader("🔬 Stage 3: Conclusive Pathology Diagnosis")

                prediction = result["final_prediction"]
                final_label = result["final_label"]
                confidence = result["confidence"]
                treatment = TREATMENT_GUIDE.get(prediction, {})

                if prediction == "Healthy":
                    st.success(f"### 🎉 Diagnosis: {final_label.upper()}")
                else:
                    st.error(f"### ⚠️ Diagnosis: {final_label.upper()}")

                st.write(f"**Voting Outcome:** {result['explanation']}")
                st.metric("Consensus Diagnostic Confidence", f"{confidence * 100:.2f}%")

                # Agronomic Treatment Recommendations
                st.markdown("---")
                st.subheader("🌾 Agronomic Management Protocol")
                st.write(f"**Pathogen Status:** {treatment.get('pathogen', 'N/A')}")
                st.write(f"**Key Diagnostic Symptoms:** {treatment.get('symptoms', 'N/A')}")
                st.info(f"**Recommended Action:** {treatment.get('action', 'N/A')}")

                with st.expander("Technical Pipeline Diagnostics"):
                    st.write(f"**Leaf Manifold Similarity:** `{result['similarity']:.4f}`")
                    st.write(f"**Crop Detector Top-3 Classes:**")
                    for c_name, c_prob in result["top3_crops"]:
                        st.write(f"- {c_name}: {c_prob * 100:.2f}%")
                    st.write("**Gatekeeper Telemetry:**")
                    st.json(result["details"])

    except Exception as exc:
        st.error("Could not process the uploaded image.")
        st.exception(exc)
