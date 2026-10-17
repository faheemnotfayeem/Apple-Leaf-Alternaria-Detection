"""
Sync the website's displayed metrics from the verified evaluation files.
=======================================================================
Reads (never writes):
    results/retrained_evaluation/comparison.csv                 full test split
    results/retrained_evaluation/comparison_eval_eligible.csv   real-image (eval_eligible) subset
    results/retrained/<model>/history.csv, config.json          training curves, params, best val epoch
    results/retrained/split_summary.csv                         split sizes
Writes:
    frontend/src/data/projectData.ts

Run after every (re-)evaluation:  python code/sync_frontend_metrics.py
"""
import os, json, ast, datetime
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "results", "retrained_evaluation")
RES = os.path.join(ROOT, "results", "retrained")
OUT = os.path.join(ROOT, "frontend", "src", "data", "projectData.ts")

MODELS = ["mobilenetv2", "resnet18", "vgg16", "vgg19"]
NAMES = {"mobilenetv2": "MobileNetV2", "resnet18": "ResNet18", "vgg16": "VGG16", "vgg19": "VGG19"}
RECOMMENDED = "mobilenetv2"
ARCH = {
    "mobilenetv2": "Inverted residual blocks with linear bottlenecks; parameter-efficient mobile convolutional network.",
    "resnet18": "18-layer residual network with identity shortcut connections mitigating vanishing gradients.",
    "vgg16": "16-layer sequential network with small 3x3 receptive fields and heavy dense classification head.",
    "vgg19": "19-layer deep sequential network with additional convolutional depth in intermediate blocks.",
}

full = pd.read_csv(os.path.join(EVAL, "comparison.csv")).set_index("model_name")
real = pd.read_csv(os.path.join(EVAL, "comparison_eval_eligible.csv")).set_index("model_name")
split = pd.read_csv(os.path.join(RES, "split_summary.csv"))
test_manifest = pd.read_csv(os.path.join(RES, "test_manifest.csv"))
cfg = {m: json.load(open(os.path.join(RES, m, "config.json"), encoding="utf-8")) for m in MODELS}
hist = {m: pd.read_csv(os.path.join(RES, m, "history.csv")) for m in MODELS}


def srow(s, label):
    r = split[(split["split"] == s) & (split["label"] == label)].iloc[0]
    return r


def metric_entry(m, src, eval_set):
    r = src.loc[m]
    f = full.loc[m]
    cm = ast.literal_eval(r["confusion_matrix"])
    return {
        "id": m, "name": NAMES[m], "evalSet": eval_set,
        "testSamples": int(r["test_samples"]),
        "testAccuracy": float(r["test_accuracy"]),
        "precisionHealthy": float(r["precision_healthy"]), "recallHealthy": float(r["recall_healthy"]), "f1Healthy": float(r["f1_healthy"]),
        "supportHealthy": int(r["support_healthy"]),
        "precisionAlternaria": float(r["precision_alternaria"]), "recallAlternaria": float(r["recall_alternaria"]), "f1Alternaria": float(r["f1_alternaria"]),
        "supportAlternaria": int(r["support_alternaria"]),
        "precisionMacro": float(r["precision_macro"]), "recallMacro": float(r["recall_macro"]), "f1Macro": float(r["f1_macro"]),
        "totalParams": int(f["total_params"]), "fileSizeMb": float(f["file_size_mb"]),
        "latencyMs": float(f["latency_ms_per_image"]), "throughput": float(f["throughput_images_per_sec"]),
        "confusionMatrix": [[int(cm[0][0]), int(cm[0][1])], [int(cm[1][0]), int(cm[1][1])]],
        "bestValAccuracy": float(cfg[m]["best_val_accuracy"]), "bestEpoch": int(cfg[m]["best_epoch"]),
        "trainedOn": "local CPU" if cfg[m]["device"] == "cpu" else "Colab T4 GPU",
        "isRecommended": m == RECOMMENDED,
        "architectureDetails": ARCH[m],
    }


def ts(v, indent=2):
    return json.dumps(v, indent=indent, ensure_ascii=False)


real_test = test_manifest[test_manifest["eval_eligible"]]
eval_sets = {
    "full": {
        "id": "full", "label": "Full Test Set", "shortLabel": "Full test",
        "images": int(len(test_manifest)), "groups": int(test_manifest["group_id"].nunique()),
        "healthy": int((test_manifest["label"] == "Healthy").sum()), "alternaria": int((test_manifest["label"] == "Alternaria").sum()),
        "description": "Every image of the held-out test split, including flip/crop augmentations of web thumbnails and other re-exported variants "
                       "that were grouped with their source so no image family crosses splits.",
        "sourceCsv": "results/retrained_evaluation/comparison.csv",
    },
    "real": {
        "id": "real", "label": "Real-Image Subset", "shortLabel": "Real-image subset",
        "images": int(len(real_test)), "groups": int(real_test["group_id"].nunique()),
        "healthy": int((real_test["label"] == "Healthy").sum()), "alternaria": int((real_test["label"] == "Alternaria").sum()),
        "description": "Only original, unmodified photographs (eval_eligible = true): no augmentations, re-exports or synthetic images. "
                       "Intended as the more representative real-image evaluation.",
        "sourceCsv": "results/retrained_evaluation/comparison_eval_eligible.csv",
    },
}
metrics_by_set = {"full": [metric_entry(m, full, "full") for m in MODELS], "real": [metric_entry(m, real, "real") for m in MODELS]}
histories = {m: [{"epoch": int(r.epoch), "trainLoss": float(r.train_loss), "valLoss": float(r.val_loss),
                  "trainAcc": float(r.train_acc), "valAcc": float(r.val_acc)} for r in hist[m].itertuples()] for m in MODELS}
dataset_splits = {
    s: {"images": int(srow(s, "ALL")["images"]), "groups": int(srow(s, "ALL")["groups"]),
        "healthy": int(srow(s, "Healthy")["images"]), "alternaria": int(srow(s, "Alternaria")["images"]),
        "synthetic": int(srow(s, "ALL")["synthetic_images"]), "evalEligible": int(srow(s, "ALL")["eval_eligible_images"])}
    for s in ("train", "val", "test", "excluded")
}
dataset_splits["totalIncluded"] = sum(dataset_splits[s]["images"] for s in ("train", "val", "test"))
dataset_splits["totalImages"] = dataset_splits["totalIncluded"] + dataset_splits["excluded"]["images"]

pipeline_stages = [
    {"number": "1", "title": "Leaf Authenticity Validation", "shortName": "Gatekeeper",
     "technology": "Feature Space Embedding Cosine Similarity",
     "description": "Evaluates input against an empirical apple leaf feature manifold (MobileNetV2 feature extractor) with a cosine similarity threshold of >= 0.3300 to filter out non-foliar inputs such as rooms, scenery, screens, or arbitrary desktop objects."},
    {"number": "2", "title": "Crop Species Identification", "shortName": "Species Validator",
     "technology": "Hugging Face 36-Crop ConvNeXt-Small",
     "description": "Evaluates the specimen against 36 distinct agricultural crop categories. Accepts the sample only when the top prediction is 'Apple' with confidence >= 60.0%. Blocks other recognized crops and flags unrecognized species as a boundary limitation."},
    {"number": "3", "title": "Pathology Diagnosis", "shortName": "Disease Classifier",
     "technology": "Two-Class Transfer Learning Classifier",
     "description": "Diagnoses confirmed apple foliage into either Healthy (Index 0) or Alternaria Leaf Spot (Index 1). Operates across four evaluated backbones (MobileNetV2, ResNet18, VGG16, VGG19) with calibrated class probabilities."},
]

content = f"""// AUTO-GENERATED by code/sync_frontend_metrics.py on {datetime.date.today().isoformat()} from the verified evaluation
// files in results/retrained_evaluation/. Do not hand-edit metric values; re-run the script instead.

export type EvalSetId = "full" | "real";

export interface EvalSet {{
  id: EvalSetId;
  label: string;
  shortLabel: string;
  images: number;
  groups: number;      // independent duplicate/provenance groups (no group crosses train/val/test)
  healthy: number;
  alternaria: number;
  description: string;
  sourceCsv: string;
}}

export interface ModelMetric {{
  id: string;
  name: string;
  evalSet: EvalSetId;
  testSamples: number;
  testAccuracy: number;
  precisionHealthy: number;
  recallHealthy: number;
  f1Healthy: number;
  supportHealthy: number;
  precisionAlternaria: number;
  recallAlternaria: number;
  f1Alternaria: number;
  supportAlternaria: number;
  precisionMacro: number;
  recallMacro: number;
  f1Macro: number;
  totalParams: number;
  fileSizeMb: number;
  latencyMs: number;    // CPU, per image, measured with batch size 32 (amortised batch latency)
  throughput: number;   // images/sec under the same batch-32 CPU benchmark
  confusionMatrix: [[number, number], [number, number]]; // [[TN, FP], [FN, TP]]  (Healthy = negative class)
  bestValAccuracy: number; // validation accuracy of the selected checkpoint (model selection only, NOT a test metric)
  bestEpoch: number;
  trainedOn: string;
  isRecommended: boolean;
  architectureDetails: string;
}}

export interface EpochRecord {{
  epoch: number;
  trainLoss: number;
  valLoss: number;
  trainAcc: number;
  valAcc: number;
}}

export const EVALUATION_INFO = {{
  evaluatedOn: "local CPU (deployment hardware), single evaluation run",
  evaluatedAt: "{datetime.date.today().isoformat()}",
  epochsTrained: 5,
  latencyNote: "CPU inference latency per image, measured with batch size 32 (amortised batch latency, not single-image request latency).",
  recommendedModelId: "{RECOMMENDED}",
}} as const;

export const EVAL_SETS: Record<EvalSetId, EvalSet> = {ts(eval_sets)};

export const METRICS_BY_SET: Record<EvalSetId, ModelMetric[]> = {ts(metrics_by_set)};

/** Full-test-split metrics (kept under the historical name for existing imports). */
export const COMPARISON_METRICS: ModelMetric[] = METRICS_BY_SET.full;
/** Real-image (eval_eligible) subset metrics. */
export const REAL_SUBSET_METRICS: ModelMetric[] = METRICS_BY_SET.real;

export const EPOCH_HISTORIES: Record<string, EpochRecord[]> = {ts(histories)};

export const DATASET_SPLITS = {ts(dataset_splits)} as const;

export const PIPELINE_STAGES = {ts(pipeline_stages)};
"""
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write(content)
print(f"wrote {os.path.relpath(OUT, ROOT)}")
for s in ("full", "real"):
    print(s, {e["id"]: round(e["testAccuracy"] * 100, 2) for e in metrics_by_set[s]}, eval_sets[s]["images"], "images")
