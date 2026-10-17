# Apple Leaf Alternaria Detection: An AI-Based Approach Using Deep Learning and Explainable AI

Automated data engineering, deep learning training, explainability (Grad-CAM), and diagnostic verification pipeline for the final-year B.Tech project in Computer Science & Engineering, Government College of Engineering & Technology (GCET), Safapora, Ganderbal, Kashmir (2022–2026).

---

## Target Classes & Diagnostic Scope

1. **`Healthy Apple Leaf`**: Baseline control leaf foliage with uniform emerald chlorophyll pigmentation, normal venation, and zero necrotic lesions.
2. **`Alternaria Leaf Spot`** (*Alternaria alternata*): Fungal necrotic circular lesions with characteristic concentric target rings and yellow chlorotic halos.

---

## Source Dataset (read-only)

The retraining pipeline consumes the two class folders below. They are the single source of truth and are never modified, moved, or written to by any script:

- `DATASET/Alternaria leaf spot/`
- `DATASET/Healthy/`

## Output Layout (retrained pipeline)

| Location | Contents |
| :--- | :--- |
| `models/retrained/` | New checkpoints (`<model>_best.pth`) written by the training script and loaded by the Streamlit app, FastAPI backend and `04_quick_inference.py`. |
| `results/retrained/` | Dataset audit report, deterministic split manifests, and per-model training history, curves and configs. |
| `results/retrained_evaluation/` | Held-out test evaluation: `comparison.csv`, `classification_report.csv`, `test_predictions.csv`, `confusion_matrices/`, `curves/`. |

All legacy checkpoints, split manifests, metrics, curves and evaluation reports from the previous (untrusted) pipeline have been removed.

---

## Pipeline Overview

| File | Purpose | Status |
| :--- | :--- | :--- |
| `04_quick_inference.py` | Cascaded diagnostic engine: Stage 1 Foundation Gatekeeper (`leaf_prototype.pt`, rejects non-foliar images) and Stage 2 MobileNetV2 two-class pathology engine with treatment guidelines and Grad-CAM. Imported by `app.py` and `backend/main.py` for the gatekeeper. | Active. Loads `models/retrained/mobilenetv2_best.pth`. |
| `train_comparative_models.py` | Four-architecture transfer-learning trainer (MobileNetV2, ResNet18, VGG16, VGG19) and the `build_comparative_model()` factory used by the app, backend and evaluator. Reads `results/retrained/{train,val}_manifest.csv` (`relative_path`, `label`, `class_index`), asserts the class mapping, writes `models/retrained/<model>_best.pth`. `--dry-run` validates everything without training; `--resume` continues from a per-epoch rolling checkpoint and skips completed models. | Active. MobileNetV2 trained locally (2026-10-16); ResNet18/VGG16/VGG19 are trained on Colab via `code/colab/`. |
| `evaluate_all_models.py` | Held-out test evaluation on `results/retrained/test_manifest.csv`: comparison tables for the full split and for the real-image `eval_eligible` subset, per-sample predictions, confusion matrices and a markdown report. `--dry-run` checks manifest and checkpoints. | Active. Writes to `results/retrained_evaluation/`. |
| `03_perceptual_hash_audit.py` | Read-only audit of `DATASET/`: SHA-256 exact duplicates, pHash+dHash near-duplicates under flips/rotations, filename provenance families, label-risk report. Writes `dataset_manifest.csv`, `duplicate_groups.csv`, `suspected_label_issues.csv`, `dataset_audit_report.md`. | Active. |
| `prepare_clean_splits.py` | Group-aware, stratified, seeded 70/15/15 split of the manifest plus 25 validation checks. Quarantines Grey-spot and Roboflow `malus` images (`excluded_manifest.csv`); `--synthetic-to-train` (project default) forces every CycleGAN group into training. Writes `train/val/test_manifest.csv`, `split_summary.csv`. `--validate-only` re-checks. | Active. |
| `generate_colab_notebook.py`, `colab_setup_guide.md` | Google Colab GPU path for the heavy VGG models. | Legacy; references old manifests and counts. Regenerate after the new splits exist. |
| `01_dataset_scanner.py`, `02_dataset_aggregator.py`, `catalog_diseases.py` | Provenance scripts for the original multi-source raw data collection (external `F:` drive). | Legacy; not used by the retraining pipeline. |
| `sync_frontend_metrics.py` | Regenerates `frontend/src/data/projectData.ts` from `results/retrained_evaluation/comparison*.csv`, the per-model histories/configs and `split_summary.csv`, so the website never carries hand-typed metrics. Run after every evaluation. | Active. |
| `test_model_selection.py` | API tests for the `model` form field of `/api/predict`: omitted = legacy 4-model vote (unchanged contract), `mobilenetv2`/`resnet18`/`vgg16`/`vgg19` = that model only, `all` = four independent results + agreement block, unknown = HTTP 400. Needs the backend running. | Active. |
| `test_majority_voting.py`, `test_pipeline_verification.py`, `test_backend_routes.py` | Tests for the 4-model voting rule, the 3-stage pipeline and the API routes. | Active once retrained checkpoints exist. |
| `class_mapping.json` | Shared class index mapping (`Healthy: 0`, `Alternaria: 1`). | Active. |
| `leaf_prototype.pt` | 1,280-D apple-leaf manifold prototype for the Stage 1 gatekeeper. Not a disease classifier. | Required by the website. |
| `test_desk.jpg`, `test_scenery.jpg` | Non-foliar samples for gatekeeper rejection tests. | Required by tests. |
| `CHECK_2_EVALUATION_GUIDE.md` | Evaluation defense guide and presentation walkthrough. | Documentation. |

---

## How to Run Offline Testing

```bash
# 1. Test an apple leaf image locally:
python code/04_quick_inference.py --image "path_to_leaf.jpg"

# 2. Test with Grad-CAM explainability heatmap overlay:
python code/04_quick_inference.py --image "path_to_leaf.jpg" --cam

# 3. Test non-leaf rejection (e.g. room / scenery):
python code/04_quick_inference.py --image "code/test_desk.jpg"
```

Stage 2 pathology inference is unavailable until `models/retrained/mobilenetv2_best.pth` exists.

---

## How to Retrain Locally

```bash
# 1. Audit the dataset and build deterministic, group-aware splits into results/retrained/:
python code/03_perceptual_hash_audit.py
python code/prepare_clean_splits.py --seed 42 --synthetic-to-train

# 1b. Verify the pipeline end to end without training anything:
python code/train_comparative_models.py --model all --dry-run
python code/evaluate_all_models.py --dry-run

# 2. Train comparative transfer learning models (writes models/retrained/<model>_best.pth):
python code/train_comparative_models.py --model mobilenetv2 --epochs 3 --batch-size 32
python code/train_comparative_models.py --model resnet18 --epochs 3 --batch-size 32

# 2b. ResNet18 / VGG16 / VGG19: see code/colab/README_COLAB.md (Colab T4), then copy the checkpoints back.

# 3. Evaluate all retrained checkpoints ONCE on the held-out split (writes results/retrained_evaluation/):
python code/evaluate_all_models.py
```
