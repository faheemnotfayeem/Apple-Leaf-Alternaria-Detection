"""
Apple Leaf Alternaria Detection: Comprehensive Held-Out Test Evaluation
========================================================================
Rigorous evaluation of four candidate transfer learning architectures:
  1. MobileNetV2
  2. ResNet18
  3. VGG16
  4. VGG19

Classes:
  0: Healthy Apple Leaf
  1: Alternaria Leaf Spot (Alternaria alternata)

Consumes exclusively:
  results/retrained/test_manifest.csv (held-out manifest produced by code/prepare_clean_splits.py;
  columns relative_path, label, class_index, image_id, group_id, eval_eligible, ...)

Metrics are reported for the full test split (comparison.csv) AND for the real, unmodified
`eval_eligible` subset (comparison_eval_eligible.csv). `--dry-run` checks manifest, image loading
and checkpoint presence without evaluating.

Reads checkpoints from:
  models/retrained/<model>_best.pth

Outputs saved to:
  results/retrained_evaluation/
    - comparison.csv
    - classification_report.csv
    - test_predictions.csv
    - confusion_matrices/confusion_matrix_<model>.png (for each model)
    - comparative_metrics_chart.png
    - final_evaluation_report.md
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------
CLASSES = ["Healthy", "Alternaria"]
CLASS_TO_IDX = {"Healthy": 0, "Alternaria": 1}
IDX_TO_CLASS = {0: "Healthy", 1: "Alternaria"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "retrained")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "retrained")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "retrained_evaluation")
CM_DIR = os.path.join(EVAL_DIR, "confusion_matrices")
TEST_MANIFEST = os.path.join(RESULTS_DIR, "test_manifest.csv")
REQUIRED_COLUMNS = {"relative_path", "label", "class_index", "image_id"}
DATA_DIR = os.path.join(PROJECT_ROOT, "DATASET")

# ---------------------------------------------------------
# Dataset Definition
# ---------------------------------------------------------
class HeldOutTestDataset(Dataset):
    """Reads images via the manifest's project-root-relative path; labels via CLASS_TO_IDX (identical to training)."""
    def __init__(self, df, project_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.project_root = project_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.project_root, row["relative_path"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Test image missing on disk: {img_path}")
        cls_name = row["label"]
        if cls_name not in CLASS_TO_IDX:
            raise ValueError(f"Invalid class label '{cls_name}' at row {idx}. Expected 'Healthy' or 'Alternaria'.")
        label = CLASS_TO_IDX[cls_name]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, row["image_id"], img_path


def load_test_manifest(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test manifest missing: {path}. Run code/prepare_clean_splits.py first.")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Test manifest lacks columns {sorted(missing)}")
    bad = sorted(set(df["label"]) - set(CLASSES))
    if bad:
        raise ValueError(f"Test manifest has unknown labels {bad}")
    if (df["label"].map(CLASS_TO_IDX) != df["class_index"]).any():
        raise ValueError(f"Test manifest class_index disagrees with CLASS_TO_IDX={CLASS_TO_IDX}")
    if "split" in df.columns and not (df["split"] == "test").all():
        raise ValueError("Test manifest contains rows whose split column is not 'test'")
    if "is_synthetic" in df.columns and df["is_synthetic"].any():
        print("WARNING: test manifest contains synthetic (CycleGAN) images")
    if "eval_eligible" not in df.columns:
        df["eval_eligible"] = True
    return df


# ---------------------------------------------------------
# Model Architecture Factory (shared with training / app / backend)
# ---------------------------------------------------------
sys.path.insert(0, SCRIPT_DIR)
from train_comparative_models import build_comparative_model, CLASS_TO_IDX as _TRAIN_CLASS_TO_IDX  # noqa: E402
assert _TRAIN_CLASS_TO_IDX == CLASS_TO_IDX, f"Class mapping mismatch: train={_TRAIN_CLASS_TO_IDX} eval={CLASS_TO_IDX}"


# ---------------------------------------------------------
# Evaluation Function for a Single Model
# ---------------------------------------------------------
def evaluate_single_model(model_name, checkpoint_path, test_loader, device):
    print(f"\n--- Evaluating Model: {model_name.upper()} ---")
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    model = build_comparative_model(model_name, num_classes=2, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # Verify output shape on test dummy
    with torch.no_grad():
        test_out = model(torch.randn(2, 3, 224, 224, device=device))
        assert test_out.shape == (2, 2), f"Expected output shape [2, 2], got {test_out.shape}"
    print(f"Checkpoint verified! Model output shape: {list(test_out.shape)}")

    total_params = sum(p.numel() for p in model.parameters())
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    y_true = []
    y_pred = []
    y_probs = []
    filenames = []

    t_start = time.time()
    with torch.no_grad():
        for images, targets, fnames, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_true.extend(targets.numpy())
            y_pred.extend(preds)
            y_probs.extend(probs)
            filenames.extend(fnames)

    total_infer_time = time.time() - t_start
    n_samples = len(y_true)
    latency_ms_per_img = (total_infer_time / n_samples) * 1000
    throughput_img_per_sec = n_samples / total_infer_time

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    prec_per_class, rec_per_class, f1_per_class, supp_per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Macro F1-Score: {f1_macro*100:.2f}% | Weighted F1-Score: {f1_weighted*100:.2f}%")
    print(f"Confusion Matrix [Healthy(0), Alternaria(1)]:\n{cm}")
    print(f"Inference Latency: {latency_ms_per_img:.2f} ms/image ({throughput_img_per_sec:.1f} imgs/sec)")

    metrics_dict = {
        "model_name": model_name,
        "test_samples": n_samples,
        "test_accuracy": acc,
        "precision_healthy": prec_per_class[0],
        "recall_healthy": rec_per_class[0],
        "f1_healthy": f1_per_class[0],
        "support_healthy": int(supp_per_class[0]),
        "precision_alternaria": prec_per_class[1],
        "recall_alternaria": rec_per_class[1],
        "f1_alternaria": f1_per_class[1],
        "support_alternaria": int(supp_per_class[1]),
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "total_params": total_params,
        "file_size_mb": round(file_size_mb, 2),
        "total_inference_time_sec": round(total_infer_time, 2),
        "latency_ms_per_image": round(latency_ms_per_img, 2),
        "throughput_images_per_sec": round(throughput_img_per_sec, 2),
        "confusion_matrix": cm.tolist()
    }

    pred_records = []
    for fn, yt, yp, pr in zip(filenames, y_true, y_pred, y_probs):
        pred_records.append({
            "image_id": fn,
            "True_Class": IDX_TO_CLASS[yt],
            "True_Index": int(yt),
            f"{model_name}_pred_class": IDX_TO_CLASS[yp],
            f"{model_name}_pred_idx": int(yp),
            f"{model_name}_prob_healthy": round(float(pr[0]), 5),
            f"{model_name}_prob_alternaria": round(float(pr[1]), 5),
            f"{model_name}_correct": bool(yt == yp)
        })

    return metrics_dict, pd.DataFrame(pred_records), cm

# ---------------------------------------------------------
# Plotting Utilities
# ---------------------------------------------------------
def plot_confusion_matrix(cm, model_name, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.8)

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            total_actual = cm[i].sum()
            pct = (count / total_actual) * 100 if total_actual > 0 else 0
            text_color = "white" if count > cm.max() / 2 else "black"
            ax.text(
                j, i, f"{count:,}\n({pct:.1f}%)",
                ha="center", va="center", color=text_color,
                fontsize=13, fontweight="bold"
            )

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Healthy (0)", "Alternaria (1)"], fontsize=11)
    ax.set_yticklabels(["Healthy (0)", "Alternaria (1)"], fontsize=11)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Actual Class (Ground Truth)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(f"Held-Out Test Confusion Matrix\n{model_name.upper()} (N = {cm.sum():,})", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_comparative_charts(summary_df, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    models = [m.upper() for m in summary_df["model_name"]]
    x = np.arange(len(models))
    width = 0.25

    # 1. Performance Metrics
    ax1.bar(x - width, summary_df["test_accuracy"] * 100, width, label="Accuracy (%)", color="#2ca02c")
    ax1.bar(x, summary_df["f1_macro"] * 100, width, label="Macro F1 (%)", color="#1f77b4")
    ax1.bar(x + width, summary_df["f1_weighted"] * 100, width, label="Weighted F1 (%)", color="#ff7f0e")
    
    ax1.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Held-Out Test Performance Comparison", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax1.set_ylim([90, 102])
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9, loc="lower right")

    # 2. Latency & Parameter Counts
    ax2_twin = ax2.twinx()
    p1 = ax2.bar(x - 0.15, summary_df["latency_ms_per_image"], 0.3, label="Latency (ms/img)", color="#d62728")
    p2 = ax2_twin.bar(x + 0.15, summary_df["total_params"] / 1e6, 0.3, label="Params (Millions)", color="#9467bd")

    ax2.set_ylabel("CPU Latency (ms/img)", fontsize=11, fontweight="bold", color="#d62728")
    ax2_twin.set_ylabel("Model Parameters (M)", fontsize=11, fontweight="bold", color="#9467bd")
    ax2.set_title("Efficiency & Computational Complexity", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

# ---------------------------------------------------------
# Markdown Report Generation
# ---------------------------------------------------------
def generate_markdown_report(summary_df, detailed_reports, save_path, test_df=None, subset_df=None):
    md = []
    md.append("# Comparative Held-Out Test Evaluation Report")
    md.append("## Apple Leaf Alternaria Detection System")
    md.append("**Academic Context**: Final Year B.Tech Project in CSE (Batch 2022–2026), GCET Safapora, Ganderbal, Kashmir")
    md.append(f"**Evaluation Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("**Pathology Classes**: Exactly two classes: `0: Healthy Apple Leaf`, `1: Alternaria Leaf Spot` (*Alternaria alternata*)")
    md.append("\n---\n")

    md.append("### 1. Held-Out Evaluation Protocol & Zero Leakage Compliance")
    md.append("- **Held-Out Manifest**: `results/retrained/test_manifest.csv`")
    if test_df is not None:
        n = len(test_df); nh = int((test_df["label"] == "Healthy").sum()); na = int((test_df["label"] == "Alternaria").sum())
        ng = test_df["group_id"].nunique() if "group_id" in test_df.columns else "n/a"
        ne = int(test_df["eval_eligible"].sum()) if "eval_eligible" in test_df.columns else n
        ns = int(test_df["is_synthetic"].sum()) if "is_synthetic" in test_df.columns else 0
        md.append(f"- **Total Test Images**: **{n:,}** images across {ng} duplicate/provenance groups.")
        md.append(f"- **Class Distribution**: {nh:,} Healthy ({nh/n*100:.2f}%) | {na:,} Alternaria ({na/n*100:.2f}%).")
        md.append(f"- **Real, unmodified (`eval_eligible`) images**: {ne:,}; synthetic images: {ns}.")
    md.append("- **Isolation Assurance**: the test manifest was produced by a group-aware split (no duplicate, re-export, augmentation or CycleGAN group crosses splits) and is never loaded during training or model selection.")
    md.append("- **Preprocessing**: Identical to validation preprocessing across all architectures:")
    md.append("  `Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`")
    md.append("\n---\n")

    md.append("### 2. Comparative Performance Summary")
    md.append("\n| Model | Parameters | Model Size | Accuracy | Macro F1 | Weighted F1 | Healthy F1 (Class 0) | Alternaria F1 (Class 1) | CPU Latency |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for _, row in summary_df.iterrows():
        m_name = row["model_name"].upper()
        params = f"{row['total_params']/1e6:.2f}M"
        size = f"{row['file_size_mb']:.1f} MB"
        acc = f"{row['test_accuracy']*100:.2f}%"
        macro_f1 = f"{row['f1_macro']*100:.2f}%"
        weighted_f1 = f"{row['f1_weighted']*100:.2f}%"
        h_f1 = f"{row['f1_healthy']*100:.2f}%"
        a_f1 = f"{row['f1_alternaria']*100:.2f}%"
        lat = f"{row['latency_ms_per_image']:.1f} ms"
        md.append(f"| **{m_name}** | {params} | {size} | **{acc}** | **{macro_f1}** | **{weighted_f1}** | {h_f1} | {a_f1} | {lat} |")

    md.append("\n---\n")

    md.append("### 3. Confusion Matrix Breakdown")
    for _, row in summary_df.iterrows():
        m_name = row["model_name"]
        cm = np.array(row["confusion_matrix"])
        md.append(f"\n#### {m_name.upper()}")
        md.append(f"- **True Healthy (TN)**: {cm[0,0]} / {cm[0].sum()} ({cm[0,0]/cm[0].sum()*100:.2f}%)")
        md.append(f"- **False Alternaria (FP)**: {cm[0,1]} / {cm[0].sum()} ({cm[0,1]/cm[0].sum()*100:.2f}%)")
        md.append(f"- **False Healthy (FN)**: {cm[1,0]} / {cm[1].sum()} ({cm[1,0]/cm[1].sum()*100:.2f}%)")
        md.append(f"- **True Alternaria (TP)**: {cm[1,1]} / {cm[1].sum()} ({cm[1,1]/cm[1].sum()*100:.2f}%)")
        md.append(f"- Confusion matrix visualization: `confusion_matrix_{m_name.lower()}.png`")

    md.append("\n---\n")

    md.append("### 4. Technical Analysis & Architectural Trade-offs")
    best_acc_row = summary_df.loc[summary_df["test_accuracy"].idxmax()]
    fastest_row = summary_df.loc[summary_df["latency_ms_per_image"].idxmin()]
    md.append(f"1. **Accuracy Leader**: **{best_acc_row['model_name'].upper()}** achieved the highest test accuracy of **{best_acc_row['test_accuracy']*100:.2f}%** and Macro F1 of **{best_acc_row['f1_macro']*100:.2f}%**.")
    md.append(f"2. **Inference Latency Leader**: **{fastest_row['model_name'].upper()}** delivered the fastest inference at **{fastest_row['latency_ms_per_image']:.2f} ms/image** ({fastest_row['throughput_images_per_sec']:.1f} frames/sec on CPU).")
    md.append("3. **Deployment Note**: model size and CPU latency above determine edge-deployment feasibility; interpret accuracy together with the real-image subset below.")
    if subset_df is not None and len(subset_df):
        md.append("\n---\n")
        md.append("### 4b. Real-Image Subset (`eval_eligible` only: original captures, no augmentation / re-export / synthetic)")
        md.append("\n| Model | Samples | Accuracy | Macro F1 | Healthy F1 | Alternaria F1 | Confusion [[TN,FP],[FN,TP]] |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
        for _, row in subset_df.iterrows():
            md.append(f"| **{row['model_name'].upper()}** | {row['test_samples']} | {row['test_accuracy']*100:.2f}% | {row['f1_macro']*100:.2f}% | {row['f1_healthy']*100:.2f}% | {row['f1_alternaria']*100:.2f}% | `{row['confusion_matrix']}` |")
    
    md.append("\n---\n")
    md.append("### 5. Generated Artifacts Registry")
    md.append("- `results/retrained_evaluation/comparison.csv`: High-level multi-model comparative metrics (full test split).")
    md.append("- `results/retrained_evaluation/comparison_eval_eligible.csv`: Same metrics on the real-image subset.")
    md.append("- `results/retrained_evaluation/classification_report_eval_eligible.csv`: Per-class precision/recall/F1 on the real-image subset.")
    md.append("- `results/retrained_evaluation/confusion_matrices/confusion_matrix_*_eval_eligible.png`: Confusion matrices on the real-image subset.")
    md.append("- `results/retrained_evaluation/classification_report.csv`: Detailed class-by-class precision, recall, and F1 metrics.")
    md.append("- `results/retrained_evaluation/test_predictions.csv`: Sample-level predictions and posterior probabilities.")
    md.append("- `results/retrained_evaluation/confusion_matrices/confusion_matrix_*.png`: High-resolution confusion matrices for all 4 architectures.")
    md.append("- `results/retrained_evaluation/comparative_metrics_chart.png`: Visual multi-metric comparative bar chart.")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

# ---------------------------------------------------------
# Real-image subset metrics
# ---------------------------------------------------------
def subset_metrics(combined_predictions, model_names, mask, subset_name):
    rows = []
    sub = combined_predictions[mask]
    for m in model_names:
        col = f"{m}_pred_idx"
        if col not in sub.columns:
            continue
        y_true = sub["True_Index"].astype(int).values
        y_pred = sub[col].astype(int).values
        acc = accuracy_score(y_true, y_pred)
        p, r, f, supp = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        rows.append({"model_name": m, "subset": subset_name, "test_samples": int(len(sub)), "test_accuracy": acc,
                     "precision_healthy": p[0], "recall_healthy": r[0], "f1_healthy": f[0], "support_healthy": int(supp[0]),
                     "precision_alternaria": p[1], "recall_alternaria": r[1], "f1_alternaria": f[1], "support_alternaria": int(supp[1]),
                     "precision_macro": pm, "recall_macro": rm, "f1_macro": fm, "confusion_matrix": cm.tolist()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Main Execution Entrypoint
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate all four transfer learning models on held-out test split.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to evaluate on ('cpu' or 'cuda')")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--models", nargs="+", default=["mobilenetv2", "resnet18", "vgg16", "vgg19"],
                        help="List of model architectures to evaluate (default: mobilenetv2 resnet18 vgg16 vgg19)")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest, image loading and checkpoint presence; evaluate nothing, write nothing")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"=== HELD-OUT TEST EVALUATION ENGINE ===")
    print(f"Device: {device}")
    
    # 1. Verify Manifest
    test_df = load_test_manifest(TEST_MANIFEST)
    print(f"Loaded held-out test manifest: {len(test_df)} samples ({int(test_df['eval_eligible'].sum())} eval_eligible)")
    print("Class distribution:\n", test_df["label"].value_counts())

    if args.dry_run:
        print("=== DRY RUN ===")
        missing = [p for p in test_df["relative_path"] if not os.path.exists(os.path.join(PROJECT_ROOT, p))]
        assert not missing, f"{len(missing)} test paths missing on disk, e.g. {missing[:3]}"
        t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        xb, yb, ids, _ = next(iter(DataLoader(HeldOutTestDataset(test_df, PROJECT_ROOT, t), batch_size=8, shuffle=False, num_workers=0)))
        assert xb.shape[1:] == (3, 224, 224) and set(yb.tolist()) <= {0, 1}
        print(f"batch {tuple(xb.shape)} labels {yb.tolist()} ids {list(ids)[:3]}...")
        for m in [x.lower() for x in args.models]:
            ck = os.path.join(MODELS_DIR, f"{m}_best.pth")
            print(f"{m}: checkpoint {'present' if os.path.exists(ck) else 'MISSING'} -> {os.path.relpath(ck, PROJECT_ROOT)}")
        print("DRY RUN OK: nothing was evaluated or written.")
        return

    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(CM_DIR, exist_ok=True)

    # Preprocessing
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = HeldOutTestDataset(test_df, PROJECT_ROOT, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Candidate Models to Evaluate
    candidate_models = [m.lower() for m in args.models]
    
    all_summary = []
    all_class_reports = []
    keep_cols = [c for c in ["image_id", "relative_path", "label", "group_id", "eval_eligible", "is_synthetic", "is_augmented", "source_family"] if c in test_df.columns]
    combined_predictions = test_df[keep_cols].copy().rename(columns={"label": "True_Class"})
    combined_predictions["True_Index"] = combined_predictions["True_Class"].map(CLASS_TO_IDX)

    for m_name in candidate_models:
        ckpt_path = os.path.join(MODELS_DIR, f"{m_name}_best.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing required model checkpoint: {ckpt_path}")

        metrics, preds_df, cm = evaluate_single_model(m_name, ckpt_path, test_loader, device)
        all_summary.append(metrics)

        # Plot individual confusion matrix
        cm_path = os.path.join(CM_DIR, f"confusion_matrix_{m_name}.png")
        plot_confusion_matrix(cm, m_name, cm_path)

        # Merge predictions
        cols_to_merge = [c for c in preds_df.columns if c.startswith(m_name)]
        preds_to_merge = preds_df[["image_id"] + cols_to_merge]
        combined_predictions = pd.merge(combined_predictions, preds_to_merge, on="image_id", how="left")

        # Class report entries
        all_class_reports.append({
            "model_name": m_name,
            "class": "Healthy",
            "precision": metrics["precision_healthy"],
            "recall": metrics["recall_healthy"],
            "f1_score": metrics["f1_healthy"],
            "support": metrics["support_healthy"]
        })
        all_class_reports.append({
            "model_name": m_name,
            "class": "Alternaria",
            "precision": metrics["precision_alternaria"],
            "recall": metrics["recall_alternaria"],
            "f1_score": metrics["f1_alternaria"],
            "support": metrics["support_alternaria"]
        })
        all_class_reports.append({
            "model_name": m_name,
            "class": "Macro Avg",
            "precision": metrics["precision_macro"],
            "recall": metrics["recall_macro"],
            "f1_score": metrics["f1_macro"],
            "support": metrics["test_samples"]
        })
        all_class_reports.append({
            "model_name": m_name,
            "class": "Weighted Avg",
            "precision": metrics["precision_weighted"],
            "recall": metrics["recall_weighted"],
            "f1_score": metrics["f1_weighted"],
            "support": metrics["test_samples"]
        })

    summary_df = pd.DataFrame(all_summary)
    class_report_df = pd.DataFrame(all_class_reports)

    # Save CSVs
    comparison_csv_path = os.path.join(EVAL_DIR, "comparison.csv")
    class_report_csv_path = os.path.join(EVAL_DIR, "classification_report.csv")
    preds_csv_path = os.path.join(EVAL_DIR, "test_predictions.csv")

    summary_df.to_csv(comparison_csv_path, index=False)
    class_report_df.to_csv(class_report_csv_path, index=False)
    combined_predictions.to_csv(preds_csv_path, index=False)

    # Real-image subset (original captures only): same artefacts as the full split
    subset_df = subset_metrics(combined_predictions, candidate_models, combined_predictions["eval_eligible"].astype(bool), "eval_eligible")
    subset_csv_path = os.path.join(EVAL_DIR, "comparison_eval_eligible.csv")
    subset_df.to_csv(subset_csv_path, index=False)
    subset_class_rows = []
    for _, r in subset_df.iterrows():
        for cls, key in (("Healthy", "healthy"), ("Alternaria", "alternaria")):
            subset_class_rows.append({"model_name": r["model_name"], "subset": "eval_eligible", "class": cls,
                                      "precision": r[f"precision_{key}"], "recall": r[f"recall_{key}"],
                                      "f1_score": r[f"f1_{key}"], "support": r[f"support_{key}"]})
        subset_class_rows.append({"model_name": r["model_name"], "subset": "eval_eligible", "class": "Macro Avg",
                                  "precision": r["precision_macro"], "recall": r["recall_macro"], "f1_score": r["f1_macro"],
                                  "support": r["test_samples"]})
        plot_confusion_matrix(np.array(r["confusion_matrix"]), f"{r['model_name']} (eval_eligible)",
                              os.path.join(CM_DIR, f"confusion_matrix_{r['model_name']}_eval_eligible.png"))
    subset_class_csv_path = os.path.join(EVAL_DIR, "classification_report_eval_eligible.csv")
    pd.DataFrame(subset_class_rows).to_csv(subset_class_csv_path, index=False)
    print(f"Saved real-image subset metrics: {subset_csv_path}")
    print(f"Saved real-image subset class report: {subset_class_csv_path}")
    print(f"\nSaved comparison summary: {comparison_csv_path}")
    print(f"Saved classification report: {class_report_csv_path}")
    print(f"Saved test predictions: {preds_csv_path}")

    # Plot Comparative Chart
    chart_path = os.path.join(EVAL_DIR, "comparative_metrics_chart.png")
    plot_comparative_charts(summary_df, chart_path)
    print(f"Saved comparative chart: {chart_path}")

    # Generate Markdown Report
    report_md_path = os.path.join(EVAL_DIR, "final_evaluation_report.md")
    generate_markdown_report(summary_df, class_report_df, report_md_path, test_df=test_df, subset_df=subset_df)
    print(f"Saved final evaluation report: {report_md_path}")

    print("\n=== EVALUATION PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
