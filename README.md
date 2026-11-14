# AppleAlt

**AI-Based Apple Leaf Alternaria Detection**

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
| `DATASET/` | Image dataset (Alternaria leaf spot / Healthy). **Not included**, see [Data sources](#data-sources) |
| `models/retrained/` | Trained checkpoints (`*.pth`). **Not included**, see [Model weights](#model-weights) |

## Model weights

The trained checkpoints are too large for a Git repository, so they are published as assets on the
[Releases](../../releases) page instead:

| File | Size |
|---|---:|
| `mobilenetv2_best.pth` (recommended, used by default) | ~23 MB |
| `resnet18_best.pth` | ~113 MB |
| `vgg16_best.pth` | ~1.5 GB |
| `vgg19_best.pth` | ~1.5 GB |

Download the ones you need into `models/retrained/`. MobileNetV2 alone is enough to run the website.
Without a checkpoint the backend still starts (it prints a warning), but predictions fail.

## Data sources

`DATASET/` (22,191 images) is **not included** in this repository. The images belong to the
public datasets below and remain under their own licenses, so download them from the original
sources into `DATASET/Alternaria leaf spot/` and `DATASET/Healthy/` if you want to retrain. The
original download links were not recorded, so each source was identified from its filename
convention and image counts (see `results/retrained/dataset_audit_report.md`) and then matched to
its official page.

| Source | Images in `DATASET/` | Link |
|---|---|---|
| Indigenous Dataset for Apple Leaf Disease Detection and Classification (`lpu_rdp_*`) | 2,549 Alternaria, 2,459 healthy | [Mendeley Data](https://data.mendeley.com/datasets/9m2dcb5mmr/3) · [paper](https://doi.org/10.1016/j.dib.2024.110165) |
| AppleLeaf9 (`Alternaria leaf spot (N)`, `Health (N)`) | 1,008 Alternaria (417 originals + Roboflow re-exports), 516 healthy | [GitHub](https://github.com/JasonYangCode/AppleLeaf9) · [paper](https://doi.org/10.3390/agronomy12112784) |
| PlantVillage, `Apple___healthy` class (`*___RS_HL *`) | 5,963 healthy (1,645 originals + flip/rotate copies) | [GitHub](https://github.com/spMohanty/PlantVillage-Dataset) · [paper](https://arxiv.org/abs/1511.08060) |
| Apple Tree Leaf Disease Segmentation Dataset, ATLDSD (`000NNN.jpg`, `IMG_*`) | 278 Alternaria, 409 healthy | [Science Data Bank](https://doi.org/10.57760/sciencedb.01627) |

Other images in the collection:

- **CycleGAN images** (`*_rec_A`, `*_rec_B`, 4,901 images) were generated from the images above with
  [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), and their
  `output_original_*` variants with [Augmentor](https://github.com/mdbloice/Augmentor). They are used for
  training only.
- **Web images** (`OIP.*`, `th-*`, `tinyurl.com_*`, 3,506 images including flip/crop copies) were
  collected individually from the web.

ATLDSD is identified by its class counts (exactly 278 Alternaria leaf spot and 409 healthy samples),
not by filename. The PlantVillage copy arrived already augmented, so it was probably downloaded from a
derived release rather than the original repository.

References:

- Yatoo, A. A., & Sharma, A. (2024). An indigenous dataset for the detection and classification of
  apple leaf diseases. *Data in Brief, 53*, 110165. https://doi.org/10.1016/j.dib.2024.110165
- Yatoo, A. (2024). *Indigenous Dataset for Apple Leaf Disease Detection and Classification* (Version 3)
  [Data set]. Mendeley Data. https://doi.org/10.17632/9m2dcb5mmr.3
- Yang, Q., Duan, S., & Wang, L. (2022). Efficient identification of apple leaf diseases in the wild
  using convolutional neural networks. *Agronomy, 12*(11), 2784. https://doi.org/10.3390/agronomy12112784
- Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable
  the development of mobile disease diagnostics. *arXiv:1511.08060*. https://arxiv.org/abs/1511.08060
- Feng, J., & Chao, X. (2022). *Apple Tree Leaf Disease Segmentation Dataset* (Version V1) [Data set].
  Science Data Bank. https://doi.org/10.57760/sciencedb.01627

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
pip install torch torchvision fastapi uvicorn python-multipart pillow timm pandas scikit-learn matplotlib
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The first start needs internet access: it downloads the ImageNet base weights for the models and the
leaf-crop detector from Hugging Face (`ixrbhii/leaf-crop-detector-36-convnext`). Both are cached, so
later starts work offline.

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

## License

The source code is released under the [MIT License](LICENSE). The datasets listed under
[Data sources](#data-sources) are not part of this repository and keep their original licenses.
