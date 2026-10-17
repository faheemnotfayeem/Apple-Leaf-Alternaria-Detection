# Apple Leaf Alternaria Detection

A final-year B.Tech project that screens photographs of apple leaves for **Alternaria leaf spot**.
Upload a leaf photograph, the application checks that it shows an apple leaf, then one of four
transfer-learning models (MobileNetV2, ResNet18, VGG16, VGG19) reports whether the leaf looks
healthy or shows Alternaria leaf spot, with a confidence score. Results are a screening aid, not a
confirmed diagnosis.

## What is in this repository

| Path | Contents |
|---|---|
| `backend/` | FastAPI inference service (`/api/predict`, `/api/health`, `/api/metrics`) |
| `frontend/` | Next.js website: home, detection workspace, model comparison, performance analysis, about |
| `code/` | Dataset audit and split scripts, training and evaluation scripts, tests, Colab notebook |
| `results/retrained/` | Dataset manifests, group-aware train/val/test splits, per-model training history |
| `results/retrained_evaluation/` | Final held-out evaluation: comparison CSVs, confusion matrices, curves |

**Not included** (too large for GitHub): the image dataset (`DATASET/`) and the trained checkpoints
(`models/retrained/*.pth`). The website and backend expect them at those paths.

## Final test results

Evaluated once, on CPU, on 3,239 held-out images (1,299 of them original photographs).

| Model | Real-image accuracy | Full test accuracy | Real-image macro F1 | CPU latency (batch 32) |
|---|---:|---:|---:|---:|
| MobileNetV2 (recommended) | 99.54% | 98.67% | 99.48% | 11.6 ms |
| ResNet18 | 99.38% | 99.14% | 99.30% | 14.5 ms |
| VGG16 | 99.08% | 98.58% | 98.96% | 89.8 ms |
| VGG19 | 96.77% | 98.61% | 96.41% | 107.5 ms |

## Running locally

Backend (Python 3.10+, PyTorch, FastAPI):

```bash
pip install torch torchvision fastapi uvicorn pillow timm pandas scikit-learn
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Frontend (Node 18+):

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000. The frontend proxies `/api/*` to the backend on port 8000.

## Reproducing the pipeline

```bash
python code/prepare_clean_splits.py          # audit the dataset and build leakage-safe splits
python code/train_comparative_models.py      # train the four models (5 epochs, best-val checkpoint)
python code/evaluate_all_models.py           # single final evaluation on the held-out test split
python code/sync_frontend_metrics.py         # regenerate frontend/src/data/projectData.ts
```

## Disclaimer

Research prototype. Predictions are model outputs for screening and should be confirmed by a
qualified agricultural expert or local horticulture extension service before any treatment decision.
