# Colab GPU migration for the retrained 4-model comparison

This folder supersedes the legacy `code/colab_setup_guide.md` / `code/generate_colab_notebook.py`
(those reference the old, removed splits).

## What stays identical to the local run
- Data: the finalized, group-aware manifests `results/retrained/{train,val,test}_manifest.csv`
  (SHA-256 of each is embedded in the notebook and verified after upload).
- Class mapping `Healthy = 0`, `Alternaria = 1` (asserted at load time by both scripts).
- Architectures, frozen layers, transforms, weighted cross-entropy, Adam lr 1e-4, batch 32, seed 42,
  5 epochs, best checkpoint by validation accuracy (tie-break lowest val loss).
- Evaluation: `code/evaluate_all_models.py` run once, after all four models finish, reporting the
  full test split and the `eval_eligible` real-image subset.

## Decision taken (2026-10-16)
MobileNetV2 was trained locally on the CPU (5 epochs) and its checkpoint and artefacts are kept as-is.
The local driver was stopped right after it, so **ResNet18 was not trained locally**. ResNet18, VGG16 and
VGG19 are trained on Colab (T4) with the same manifests, script, seed, preprocessing and five-epoch plan.
If the GPU runs out of memory the notebook retries that model with batch 16, then 8, and the batch size
used is recorded in that model's `config.json`. **Evaluation runs once, on the PC**, after the three
checkpoints are downloaded (`RUN_EVALUATION_ON_COLAB = False`), so all four models are evaluated
identically on the deployment CPU with the same evaluation script and test manifest.

## Files to upload (once) to Google Drive `MyDrive/apple_leaf_retrained/upload/`
| File | Contents | Size |
|---|---|---|
| `code/colab/upload/apple_leaf_project_files.zip` | `code/train_comparative_models.py` (resume-ready copy), `code/evaluate_all_models.py`, `code/class_mapping.json`, `results/retrained/{train,val,test}_manifest.csv`, `split_summary.csv` | ~2 MB |
| `apple_leaf_dataset.zip` (built in the session scratch folder, path given in the chat) | `DATASET/Alternaria leaf spot/*` and `DATASET/Healthy/*` image files only (the four stray CycleGAN `.pth` files are not shipped), stored without compression | ~1.0 GB |

Then open `code/colab/Apple_Leaf_Retrained_Training_Colab.ipynb` in Colab, select a **T4 GPU**
runtime, and run the cells top to bottom.

## What the notebook does
1. Checks the GPU, mounts Drive, unpacks both zips to fast local disk (`/content/apple_leaf`, same
   layout as the PC project so no path changes are needed).
2. Verifies the dataset: 10,265 + 11,926 images and the SHA-256 of **every** manifest image.
3. Restores any earlier outputs from `MyDrive/apple_leaf_retrained/outputs/` (resume) and starts a
   background sync to Drive every 2 minutes.
4. Trains ResNet18, VGG16, VGG19 sequentially with `--resume`; an audit hook proves the test manifest is
   never opened during training; GPU out-of-memory triggers a retry with a smaller batch size.
5. Verifies the three models' artefacts (5 epochs, finite metrics, best epoch, manifest hashes).
6. Skips evaluation (done on the PC) and lists the Drive folders to download.

## Changes required for Colab (already made in the bundled copy of the training script)
- **Resume:** a rolling `models/retrained/<model>_last.pth` (model + optimizer + scheduler + history + RNG
  state) is saved after every epoch and preferred by `--resume`; a model whose `config.json` and
  `history.csv` already show 5 epochs is skipped; `_last.pth` is deleted on completion. Tested locally in
  an isolated sandbox (crash after epoch 1, resume, idempotent re-run).
- **CUDA:** nothing else needed. Device is auto-detected, `pin_memory` follows the device,
  `--num-workers 2` is passed. GPU-saved checkpoints load on the CPU app because every loader uses
  `map_location`, and the checkpoint dict only holds tensors/str/int/dict, so `weights_only=True` works.
- **Dependencies:** torch, torchvision, pandas, scikit-learn, matplotlib, Pillow are preinstalled on Colab.
- **Drive space:** about 1.1 GB (dataset zip) + ~3.2 GB of best checkpoints (VGG files are ~1.5 GB each
  because they include the Adam state) + up to 1.5 GB for one rolling checkpoint at a time.
- **Latency columns** in `comparison.csv` will reflect the Colab GPU, not the deployment CPU.

## Expected durations on a T4 (rough)
MobileNetV2 and ResNet18 about 1 to 2 minutes per epoch; VGG16/VGG19 about 6 to 10 minutes per epoch.
Whole run roughly 1.5 to 2 hours including dataset extraction and verification.

## Bringing the results back and evaluating
1. Download from `MyDrive/apple_leaf_retrained/outputs/`:
   `models/retrained/{resnet18,vgg16,vgg19}_best.pth` and `results/retrained/{resnet18,vgg16,vgg19}/`
   into the PC project at the same relative paths. Do **not** overwrite `models/retrained/mobilenetv2_best.pth`
   or `results/retrained/mobilenetv2/`.
2. On the PC run, once: `python code/evaluate_all_models.py` (full test split + `eval_eligible` subset for all four models,
   outputs in `results/retrained_evaluation/`).
3. Do not update the website's displayed metrics until the evaluation outputs have been reviewed.
