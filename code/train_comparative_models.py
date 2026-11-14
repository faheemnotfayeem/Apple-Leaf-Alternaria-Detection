"""
AppleAlt: Comparative Transfer-Learning Framework
========================================================================
Supports 4 architectures:
  1. MobileNetV2
  2. ResNet18
  3. VGG16
  4. VGG19

Classes:
  0: Healthy Apple Leaf
  1: Alternaria Leaf Spot (Alternaria alternata)

Data source: DATASET/Alternaria leaf spot/ and DATASET/Healthy/ (never modified).
Data manifests (produced by code/03_perceptual_hash_audit.py + code/prepare_clean_splits.py):
  - results/retrained/train_manifest.csv
  - results/retrained/val_manifest.csv
  (results/retrained/test_manifest.csv is strictly reserved for code/evaluate_all_models.py)
  Required columns: relative_path (relative to the project root), label ("Healthy"/"Alternaria"),
  class_index (0/1). The manifest's class_index is asserted against CLASS_TO_IDX at load time.

Outputs:
  - models/retrained/<model>_best.pth
  - results/retrained/<model>/{history.csv, loss_curve.png, accuracy_curve.png, config.json, training.log}

Balanced cross-entropy weights for class imbalance are computed at runtime
from the training manifest: w_c = N / (2 * N_c).

Run `python code/train_comparative_models.py --dry-run` to validate manifests, data loading and
model construction without training or writing any file.

Resume (--resume): a rolling `models/retrained/<model>_last.pth` (model + optimizer + scheduler +
history + RNG state) is written after every epoch and is preferred for resuming; the legacy
fallback resumes from `<model>_best.pth`. A model whose history already holds `--epochs` epochs and
whose config.json exists is reported as complete and skipped. `_last.pth` is deleted on completion.
"""

import os
import sys
import json
import time
import shutil
import hashlib
import argparse
import random
import logging
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Constants and Defaults
# ---------------------------------------------------------
SEED = 42
CLASSES = ["Healthy", "Alternaria"]
CLASS_TO_IDX = {"Healthy": 0, "Alternaria": 1}
IDX_TO_CLASS = {0: "Healthy", 1: "Alternaria"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "DATASET")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "retrained")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "retrained")
CURVES_DIR = os.path.join(PROJECT_ROOT, "results", "retrained_evaluation", "curves")  # copies of the per-model curves

TRAIN_MANIFEST = os.path.join(RESULTS_DIR, "train_manifest.csv")
VAL_MANIFEST = os.path.join(RESULTS_DIR, "val_manifest.csv")
REQUIRED_COLUMNS = {"relative_path", "label", "class_index"}
CLASS_MAPPING_PATH = os.path.join(SCRIPT_DIR, "class_mapping.json")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------
# 2. Manifest loading, transforms and Dataset Definition
# ---------------------------------------------------------
def load_manifest(path: str, split_name: str) -> pd.DataFrame:
    """Loads a split manifest and enforces the canonical schema / class mapping."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{split_name} manifest missing: {path}. Run code/prepare_clean_splits.py first.")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{split_name} manifest {path} lacks columns {sorted(missing)}")
    bad_labels = sorted(set(df["label"]) - set(CLASSES))
    if bad_labels:
        raise ValueError(f"{split_name} manifest has unknown labels {bad_labels}; expected {CLASSES}")
    mismatch = df[df["label"].map(CLASS_TO_IDX) != df["class_index"]]
    if len(mismatch):
        raise ValueError(f"{split_name} manifest: {len(mismatch)} rows whose class_index disagrees with CLASS_TO_IDX={CLASS_TO_IDX}")
    if "split" in df.columns and not (df["split"] == split_name).all():
        raise ValueError(f"{split_name} manifest contains rows whose split column is not '{split_name}'")
    if df["relative_path"].duplicated().any():
        raise ValueError(f"{split_name} manifest has duplicated relative_path entries")
    return df


def build_transforms():
    """Train-time augmentation and deterministic eval preprocessing (224x224, ImageNet statistics)."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, eval_transform


class AppleLeafDataset(Dataset):
    """Reads images via the manifest's project-root-relative path; labels via CLASS_TO_IDX."""
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(PROJECT_ROOT, row["relative_path"])
        label = CLASS_TO_IDX[row["label"]]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ---------------------------------------------------------
# 3. Model Architecture Factory
# ---------------------------------------------------------
def build_comparative_model(model_name: str, num_classes: int = 2, pretrained: bool = True):
    """Architecture factory shared by training, evaluation, the Streamlit app and the FastAPI backend.
    pretrained=False skips the ImageNet download (used when a checkpoint will be loaded anyway)."""
    model_name = model_name.lower()
    
    if model_name == "mobilenetv2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        # Freeze lower layers, train top features
        for p in model.features[:-4].parameters():
            p.requires_grad = False
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_f, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
        
    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        # Freeze lower layers, train layer4 and fc
        for name, p in model.named_parameters():
            if not ("layer4" in name or "fc" in name):
                p.requires_grad = False
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_f, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
        
    elif model_name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        # Freeze conv1_1..conv5_2; train the last conv layer (conv5_3) and the whole classifier
        for p in model.features[:-4].parameters():
            p.requires_grad = False
        in_f = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_f, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
        
    elif model_name == "vgg19":
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        model = models.vgg19(weights=weights)
        # Freeze conv1_1..conv5_3; train the last conv layer (conv5_4) and the whole classifier
        for p in model.features[:-4].parameters():
            p.requires_grad = False
        in_f = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_f, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Choose from mobilenetv2, resnet18, vgg16, vgg19.")

    return model

# ---------------------------------------------------------
# 4. Training Engine
# ---------------------------------------------------------
def train_single_model(model_name: str, args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_dir = os.path.join(RESULTS_DIR, model_name.lower())
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name.lower()}_best.pth")
    last_path = os.path.join(MODELS_DIR, f"{model_name.lower()}_last.pth")
    history_path = os.path.join(out_dir, "history.csv")
    loss_curve_path = os.path.join(out_dir, "loss_curve.png")
    acc_curve_path = os.path.join(out_dir, "accuracy_curve.png")
    config_path = os.path.join(out_dir, "config.json")
    log_path = os.path.join(out_dir, "training.log")

    # Logging setup
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_path, mode="a" if args.resume else "w", encoding="utf-8")
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    logger.info(f"=== Starting Training Pipeline: {model_name.upper()} ===")
    logger.info(f"Execution Device: {device}")

    # Already complete? (idempotent re-runs with --resume)
    if args.resume and os.path.exists(config_path) and os.path.exists(history_path) and os.path.exists(checkpoint_path):
        try:
            prev_cfg = json.load(open(config_path, encoding="utf-8"))
            prev_hist = pd.read_csv(history_path)
        except Exception:
            prev_cfg, prev_hist = {}, pd.DataFrame()
        if prev_cfg.get("epochs_requested") == args.epochs and len(prev_hist) == args.epochs and prev_cfg.get("best_val_accuracy") is not None:
            logger.info(f"{model_name} already completed {args.epochs} epochs (best val acc {prev_cfg['best_val_accuracy']*100:.2f}%); skipping.")
            if os.path.exists(last_path):
                os.remove(last_path)
            return prev_cfg
    
    # Load and verify manifests (schema + class mapping enforced)
    train_df = load_manifest(TRAIN_MANIFEST, "train")
    val_df = load_manifest(VAL_MANIFEST, "val")
    if "is_synthetic" in val_df.columns and val_df["is_synthetic"].any():
        logger.warning("Validation manifest contains synthetic (CycleGAN) images; model selection will be influenced by them.")
    logger.info(f"Loaded Train Manifest: {len(train_df)} samples (Healthy: {(train_df['label']=='Healthy').sum()}, Alternaria: {(train_df['label']=='Alternaria').sum()})")
    logger.info(f"Loaded Val Manifest:   {len(val_df)} samples (Healthy: {(val_df['label']=='Healthy').sum()}, Alternaria: {(val_df['label']=='Alternaria').sum()})")

    # Balanced class weights
    n_total = len(train_df)
    n_healthy = (train_df["label"] == "Healthy").sum()
    n_alternaria = (train_df["label"] == "Alternaria").sum()
    w_healthy = n_total / (2.0 * n_healthy)
    w_alternaria = n_total / (2.0 * n_alternaria)
    class_weights = torch.tensor([w_healthy, w_alternaria], dtype=torch.float32, device=device)
    logger.info(f"Class imbalance handling formula: N / (2 * N_c)")
    logger.info(f"Class Weights [Healthy, Alternaria]: [{w_healthy:.6f}, {w_alternaria:.6f}]")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Transforms (224x224 RGB, ImageNet stats) shared with evaluation
    train_transform, val_transform = build_transforms()

    train_dataset = AppleLeafDataset(train_df, transform=train_transform)
    val_dataset = AppleLeafDataset(val_df, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    # Model instantiation
    model = build_comparative_model(model_name, num_classes=2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Architecture: {model_name} | Total Params: {total_params:,} | Trainable Params: {trainable_params:,}")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # Resume capability
    start_epoch = 1
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = None  # epoch at which <model>_best.pth was actually written (max val acc, tie-break lowest val loss)
    history = []

    if args.resume and os.path.exists(last_path):
        logger.info(f"Resuming training from rolling checkpoint: {last_path}")
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = int(ck["epoch"]) + 1
        best_val_acc = float(ck["best_val_acc"])
        best_val_loss = float(ck["best_val_loss"])
        best_epoch = ck.get("best_epoch")
        history = list(ck["history"])
        try:
            torch.set_rng_state(ck["rng_state_cpu"])
            if device.type == "cuda" and ck.get("rng_state_cuda") is not None:
                torch.cuda.set_rng_state_all(ck["rng_state_cuda"])
            random.setstate(ck["rng_state_py"]); np.random.set_state(ck["rng_state_np"])
        except Exception as exc:  # RNG restore is best-effort
            logger.warning(f"Could not restore RNG state: {exc}")
        logger.info(f"Resumed at epoch {start_epoch} (completed {len(history)} epochs, best_val_acc={best_val_acc:.4f})")
    elif args.resume and os.path.exists(checkpoint_path):
        logger.info(f"Resuming training from best checkpoint (no rolling checkpoint found): {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            best_epoch = int(checkpoint["epoch"])
        if "val_acc" in checkpoint:
            best_val_acc = checkpoint["val_acc"]
        if "val_loss" in checkpoint:
            best_val_loss = checkpoint["val_loss"]
        if os.path.exists(history_path):
            history = pd.read_csv(history_path).to_dict(orient="records")
        logger.info(f"Resumed at epoch {start_epoch} with previous best_val_acc={best_val_acc:.4f}")

    total_train_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        
        # --- Training Loop ---
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # --- Validation Loop ---
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss_sum += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        epoch_dur = time.time() - epoch_start
        curr_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(val_acc)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_acc": round(val_acc, 5),
            "learning_rate": curr_lr,
            "epoch_duration_sec": round(epoch_dur, 2)
        })

        is_best = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            # Save best checkpoint
            torch.save({
                "model_name": model_name,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "val_loss": best_val_loss,
                "classes": CLASSES,
                "class_to_idx": CLASS_TO_IDX,
                "criterion_selection": "Highest validation accuracy with tie-breaker on lowest validation loss"
            }, checkpoint_path)
            logger.info(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | {epoch_dur:.1f}s | [BEST SAVED]")
        else:
            logger.info(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | {epoch_dur:.1f}s")

        # History first (cheap, keeps the on-disk CSV current), then the rolling checkpoint for safe resume
        pd.DataFrame(history).to_csv(history_path, index=False)
        torch.save({
            "model_name": model_name, "epoch": epoch,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "best_val_acc": best_val_acc, "best_val_loss": best_val_loss, "best_epoch": best_epoch,
            "history": history, "rng_state_cpu": torch.get_rng_state(),
            "rng_state_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "rng_state_py": random.getstate(), "rng_state_np": np.random.get_state(),
        }, last_path)

    total_time = time.time() - total_train_start
    logger.info(f"Completed training in {total_time/60:.2f} minutes. Best Val Acc: {best_val_acc*100:.2f}%")

    # --- Save History and Curves ---
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(history_path, index=False)
    logger.info(f"Saved history CSV to: {history_path}")

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss", color="#1f77b4", marker="o", linewidth=2)
    plt.plot(hist_df["epoch"], hist_df["val_loss"], label="Validation Loss", color="#d62728", marker="s", linewidth=2)
    plt.title(f"{model_name.upper()} - Training & Validation Loss", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Weighted Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=200)
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["train_acc"] * 100, label="Train Accuracy", color="#2ca02c", marker="o", linewidth=2)
    plt.plot(hist_df["epoch"], hist_df["val_acc"] * 100, label="Validation Accuracy", color="#ff7f0e", marker="s", linewidth=2)
    plt.title(f"{model_name.upper()} - Training & Validation Accuracy", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(acc_curve_path, dpi=200)
    plt.close()

    # keep a copy of both curves in the shared curves folder
    os.makedirs(CURVES_DIR, exist_ok=True)
    shutil.copy2(loss_curve_path, os.path.join(CURVES_DIR, f"{model_name.lower()}_loss_curve.png"))
    shutil.copy2(acc_curve_path, os.path.join(CURVES_DIR, f"{model_name.lower()}_accuracy_curve.png"))

    # --- Save Training Metadata ---
    config_metadata = {
        "model_name": model_name,
        "device": str(device),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": len(hist_df),
        "batch_size": args.batch_size,
        "initial_lr": args.lr,
        "total_training_time_sec": round(total_time, 2),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "class_mapping": CLASS_TO_IDX,
        "class_weights": [round(w_healthy, 6), round(w_alternaria, 6)],
        "train_manifest": os.path.relpath(TRAIN_MANIFEST, PROJECT_ROOT),
        "val_manifest": os.path.relpath(VAL_MANIFEST, PROJECT_ROOT),
        "train_manifest_sha256": hashlib.sha256(open(TRAIN_MANIFEST, "rb").read()).hexdigest(),
        "val_manifest_sha256": hashlib.sha256(open(VAL_MANIFEST, "rb").read()).hexdigest(),
        "test_manifest_used_during_training": False,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "train_synthetic_samples": int(train_df["is_synthetic"].sum()) if "is_synthetic" in train_df.columns else None,
        "val_synthetic_samples": int(val_df["is_synthetic"].sum()) if "is_synthetic" in val_df.columns else None,
        "train_groups": int(train_df["group_id"].nunique()) if "group_id" in train_df.columns else None,
        "val_groups": int(val_df["group_id"].nunique()) if "group_id" in val_df.columns else None,
        "best_epoch": best_epoch,  # epoch of the saved best checkpoint (same tie-break rule as the checkpoint itself)
        "best_val_accuracy": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "checkpoint_path": checkpoint_path,
        "checkpoint_selection_criterion": "Highest validation accuracy with tie-breaker on lowest validation loss",
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_metadata, f, indent=2)

    # --- Verification of Checkpoint ---
    logger.info("Verifying checkpoint loading and output shape...")
    eval_model = build_comparative_model(model_name, num_classes=2).to(device)
    saved_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    eval_model.load_state_dict(saved_ckpt["model_state_dict"])
    eval_model.eval()
    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        dummy_out = eval_model(dummy_input)
    assert dummy_out.shape == (2, 2), f"Checkpoint verification failed: expected shape [2, 2], got {dummy_out.shape}"
    logger.info(f"Checkpoint verification passed! Output shape: {list(dummy_out.shape)}")
    if os.path.exists(last_path):
        os.remove(last_path)
        logger.info("Removed rolling checkpoint (training complete).")

    return config_metadata

def dry_run(models_to_check, batch_size=8):
    """Validates manifests, image loading and model construction end to end WITHOUT training,
    downloading pretrained weights, or writing any file."""
    print("=== DRY RUN: manifests, data loading, model construction ===")
    train_df = load_manifest(TRAIN_MANIFEST, "train")
    val_df = load_manifest(VAL_MANIFEST, "val")
    overlap = set(train_df["relative_path"]) & set(val_df["relative_path"])
    assert not overlap, f"{len(overlap)} images appear in both train and val manifests"
    if "group_id" in train_df.columns and "group_id" in val_df.columns:
        g_overlap = set(train_df["group_id"]) & set(val_df["group_id"])
        assert not g_overlap, f"{len(g_overlap)} duplicate groups cross train/val"
    for name, df in (("train", train_df), ("val", val_df)):
        syn = int(df["is_synthetic"].sum()) if "is_synthetic" in df.columns else "n/a"
        print(f"{name}: {len(df)} images | Healthy {(df['label']=='Healthy').sum()} | Alternaria {(df['label']=='Alternaria').sum()} | synthetic {syn}")
        missing = [p for p in df["relative_path"] if not os.path.exists(os.path.join(PROJECT_ROOT, p))]
        assert not missing, f"{len(missing)} {name} paths do not exist, e.g. {missing[:3]}"
    train_transform, val_transform = build_transforms()
    tl = DataLoader(AppleLeafDataset(train_df, train_transform), batch_size=batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(AppleLeafDataset(val_df, val_transform), batch_size=batch_size, shuffle=False, num_workers=0)
    xb, yb = next(iter(tl)); xv, yv = next(iter(vl))
    assert xb.shape[1:] == (3, 224, 224) and xv.shape[1:] == (3, 224, 224), "unexpected tensor shape"
    assert set(yb.tolist()) <= {0, 1} and set(yv.tolist()) <= {0, 1}, "labels outside {0,1}"
    print(f"train batch {tuple(xb.shape)} labels {yb.tolist()} | val batch {tuple(xv.shape)} labels {yv.tolist()}")
    for m in models_to_check:
        model = build_comparative_model(m, num_classes=2, pretrained=False).eval()
        with torch.no_grad():
            out = model(xv[:2])
        assert out.shape == (2, 2), f"{m}: expected output (2, 2), got {tuple(out.shape)}"
        print(f"{m}: constructed, forward pass OK, output {tuple(out.shape)}, params {sum(p.numel() for p in model.parameters()):,}")
    print("DRY RUN OK: nothing was trained or written.")


def main():
    parser = argparse.ArgumentParser(description="AppleAlt Comparative Training")
    parser.add_argument("--model", type=str, default="mobilenetv2", choices=["mobilenetv2", "resnet18", "vgg16", "vgg19", "all"],
                        help="Architecture to train")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default 0 for Windows CPU)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint if available")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifests, data loading and model construction; train nothing, write nothing")

    args = parser.parse_args()

    if args.dry_run:
        dry_run(["mobilenetv2", "resnet18", "vgg16", "vgg19"] if args.model == "all" else [args.model])
        return

    # Save shared class mapping
    with open(CLASS_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump({"classes": CLASSES, "class_to_idx": CLASS_TO_IDX, "idx_to_class": IDX_TO_CLASS}, f, indent=2)

    if args.model == "all":
        models_to_train = ["mobilenetv2", "resnet18", "vgg16", "vgg19"]
    else:
        models_to_train = [args.model]

    for m in models_to_train:
        print(f"\n=======================================================")
        print(f"       LAUNCHING TRAINING FOR: {m.upper()}")
        print(f"=======================================================")
        train_single_model(m, args)

if __name__ == "__main__":
    main()
