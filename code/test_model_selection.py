"""
API tests for the model-selection modes of /api/predict (requires the backend on 127.0.0.1:8000).
  - omitted `model`  -> legacy 4-model voting response (unchanged contract, no per-model exposure)
  - model=<key>      -> only that model, model-confidence response
  - model=all        -> four independent results + agreement block
  - unknown model    -> HTTP 400
Usage: python code/test_model_selection.py [--base http://127.0.0.1:8000] [--image path]
"""
import argparse, sys, time
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KEYS = ["mobilenetv2", "resnet18", "vgg16", "vgg19"]
NAMES = {"mobilenetv2": "MobileNetV2", "resnet18": "ResNet18", "vgg16": "VGG16", "vgg19": "VGG19"}


def post(base, image, model=None):
    with open(image, "rb") as f:
        data = {"model": model} if model is not None else {}
        t = time.time()
        r = requests.post(f"{base}/api/predict", files={"file": (Path(image).name, f, "image/jpeg")}, data=data, timeout=600)
    return r, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--image", default=None, help="leaf image; defaults to the first Alternaria image of the test manifest")
    a = ap.parse_args()
    image = a.image
    if image is None:
        import pandas as pd
        df = pd.read_csv(PROJECT_ROOT / "results" / "retrained" / "test_manifest.csv")
        image = str(PROJECT_ROOT / df[(df.label == "Alternaria") & (df.eval_eligible)].iloc[0]["relative_path"])
    print("image:", image)
    fails = []

    def chk(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
        if not ok:
            fails.append(name)

    # 1. legacy default
    r, dt = post(a.base, image)
    d = r.json()
    chk("default: HTTP 200", r.status_code == 200, f"{dt:.1f}s")
    chk("default: mode ensemble, one final prediction", d.get("mode") == "ensemble" and d.get("final_label") in ("Healthy Leaf", "Alternaria Leaf Spot"))
    chk("default: per-model results NOT exposed", all(k not in d for k in ("model_results", "all_models", "probabilities", "internal_diagnostics")))

    # 2. single models
    singles = {}
    for k in KEYS:
        r, dt = post(a.base, image, k)
        d = r.json()
        chk(f"{k}: HTTP 200", r.status_code == 200, f"{dt:.1f}s")
        chk(f"{k}: result attributed to the right model", d.get("mode") == "single" and d.get("model_key") == k and d.get("model_name") == NAMES[k])
        chk(f"{k}: one final prediction with model confidence", d.get("final_label") in ("Healthy Leaf", "Alternaria Leaf Spot")
            and d.get("confidence_type") == "model_confidence" and 0 <= d.get("confidence", -1) <= 1)
        p = d.get("probabilities", {})
        chk(f"{k}: probabilities sum to 1 and argmax matches prediction", abs(p.get("Healthy", 0) + p.get("Alternaria", 0) - 1) < 1e-3
            and max(p, key=p.get) == d.get("final_prediction"))
        chk(f"{k}: inference time present", isinstance(d.get("inference_ms"), (int, float)) and d["inference_ms"] > 0, f"{d.get('inference_ms')} ms")
        singles[k] = d

    # 3. all models
    r, dt = post(a.base, image, "all")
    d = r.json()
    chk("all: HTTP 200", r.status_code == 200, f"{dt:.1f}s")
    res = d.get("model_results", [])
    chk("all: four results in canonical order", [x.get("model_key") for x in res] == KEYS)
    chk("all: each result matches the single-model call for the same image",
        all(x["prediction"] == singles[x["model_key"]]["final_prediction"] and abs(x["confidence"] - singles[x["model_key"]]["confidence"]) < 1e-3 for x in res))
    ag = d.get("agreement", {})
    chk("all: agreement block consistent", ag.get("votes", {}).get("Healthy", -1) + ag.get("votes", {}).get("Alternaria", -1) == 4
        and ag.get("unanimous") == (len({x["prediction"] for x in res}) == 1) and "note" in ag)
    print("    votes:", ag.get("votes"), "| majority:", ag.get("majority_label"), "| disagreeing:", ag.get("disagreeing_models"))

    # 4. invalid model
    r, _ = post(a.base, image, "alexnet")
    chk("unknown model: HTTP 400 with explanation", r.status_code == 400 and "Valid values" in r.text)

    # 5. non-leaf still rejected in every mode
    desk = PROJECT_ROOT / "code" / "test_desk.jpg"
    if desk.exists():
        for m in (None, "mobilenetv2", "all"):
            r, _ = post(a.base, str(desk), m)
            chk(f"non-leaf rejected (model={m})", r.status_code == 200 and r.json().get("status") == "rejected_non_leaf")

    print("\nRESULT:", "ALL PASSED" if not fails else f"{len(fails)} FAILED -> {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
