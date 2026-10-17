> **SUPERSEDED (2026-10-16):** this guide refers to the old, removed split manifests and checkpoints. Use `code/colab/README_COLAB.md` and `code/colab/Apple_Leaf_Retrained_Training_Colab.ipynb` instead.

# Google Colab GPU Execution Guide: VGG16 & VGG19 Benchmark
## Four-Model Comparative Transfer Learning on GPU Hardware

### Overview
This guide provides exact step-by-step instructions to train **VGG16** and **VGG19** using **Google Colab's free GPU acceleration** (NVIDIA T4).
- **MobileNetV2** and **ResNet18** have already completed training locally on CPU.
- VGG16 (134.8M parameters) and VGG19 (140.1M parameters) are computationally heavy. On local CPU, each epoch takes ~25–35 minutes (~5+ hours total).
- On Google Colab's T4 GPU, each epoch completes in **~45–60 seconds** (~2.5–3.5 minutes per model for 3 epochs).

---

### Verification Checklist Before Running
1. **Existing Locked Splits**:
   - Training uses exclusively [`results/train_split.csv`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/results/train_split.csv) (6,418 images).
   - Validation uses exclusively [`results/val_split.csv`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/results/val_split.csv) (1,319 images).
   - Test split [`results/test_split.csv`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/results/test_split.csv) is **strictly reserved** and never loaded.
2. **Class Mapping**:
   - `0: Healthy`
   - `1: Alternaria`
3. **Preprocessing & ImageNet Normalization**:
   - Input dimensions: 224 × 224 RGB
   - Normalization: $\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$
   - Data Augmentation (Train): RandomResizedCrop(224, scale=(0.8, 1.0)), RandomHorizontalFlip, RandomRotation(15°), ColorJitter(0.1, 0.1)
   - Validation: Resize(256), CenterCrop(224)
4. **Class Imbalance Loss**:
   - Weighted Cross-Entropy Loss with exact balanced weights:
     - $w_0 (\text{Healthy}) = 2.068988$
     - $w_1 (\text{Alternaria}) = 0.659338$
5. **Persistence & Checkpoint Safety**:
   - Checkpoints and artifacts sync directly to Google Drive (`MyDrive/Apple_Leaf_Alternaria_Detection/`) in real time, preventing data loss if the session disconnects.
   - Checkpoint format includes `model_state_dict`, `optimizer_state_dict`, `val_acc`, `val_loss`, `classes`, `class_to_idx`, and `criterion_selection`, matching [`train_comparative_models.py`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/code/train_comparative_models.py).

---

### Step-by-Step Execution Instructions

#### Step 1: Open Notebook in Google Colab
1. In your browser, open [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and upload:
   [`code/Apple_Leaf_Comparative_Training_Colab.ipynb`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/code/Apple_Leaf_Comparative_Training_Colab.ipynb)

#### Step 2: Enable GPU Accelerator
1. In the top navigation bar, click **Runtime -> Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (or any available GPU).
3. Click **Save**.

#### Step 3: Run Cell 1 (GPU Verification)
1. Execute **Cell 1**.
2. Verify that it outputs:
   ```
   PyTorch Version: 2.x
   SUCCESS: CUDA GPU Active!
   GPU Device Name: Tesla T4 (or similar)
   Allocated VRAM: ~15 GB
   ```
   *(If it reports CPU, ensure you performed Step 2).*

#### Step 4: Run Cell 2 (Google Drive Persistence Mount)
1. Execute **Cell 2**.
2. Click **Connect to Google Drive** when prompted.
3. This creates `/content/drive/MyDrive/Apple_Leaf_Alternaria_Detection/` where all checkpoints and results will automatically be saved.

#### Step 5: Run Cell 3 (Data Synchronization)
Upload the images and split manifests:
- **Manifests**: Upload `train_split.csv` and `val_split.csv` from your local `results/` folder into `/content/apple_leaf_alternaria/results/`.
- **Dataset Images**: Upload your dataset images into `/content/apple_leaf_alternaria/alternaria for model/` (or upload as a zip and unzip directly in the notebook).

#### Step 6: Run Cell 4 (Load Architectures)
Execute **Cell 4** to initialize the model builders, dataset loader, and balanced class weights.

#### Step 7: Run Cell 5 & 6 (Train VGG16 and VGG19)
Execute **Cell 6** to start training:
- **VGG16** trains for 3 epochs (~2.5 minutes).
- **VGG19** trains for 3 epochs (~2.5 minutes).
- Each epoch logs training loss, validation loss, training accuracy, validation accuracy, duration, and saves the best model whenever validation accuracy improves.
- Outputs are automatically copied to Google Drive in real-time.

#### Step 8: Download Checkpoints Back to Local Workspace
1. Run **Cell 7** to download `vgg_models_and_results.zip` (or download the files directly from your Google Drive folder).
2. Copy the downloaded files into your local project folders:
   - Checkpoints:
     - Copy `vgg16_best.pth` -> [`models/vgg16_best.pth`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/models/)
     - Copy `vgg19_best.pth` -> [`models/vgg19_best.pth`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/models/)
   - Artifacts:
     - Copy `vgg16/` artifacts -> [`results/vgg16/`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/results/vgg16/)
     - Copy `vgg19/` artifacts -> [`results/vgg19/`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/results/vgg19/)

#### Step 9: Verify Loaded Checkpoints Locally
Run this command in PowerShell from the project root:
```powershell
python -c "import sys; sys.path.insert(0, 'code'); import torch; from train_comparative_models import build_comparative_model; m16 = build_comparative_model('vgg16'); m16.load_state_dict(torch.load('models/vgg16_best.pth')['model_state_dict']); print('VGG16 shape:', m16(torch.randn(2, 3, 224, 224)).shape); m19 = build_comparative_model('vgg19'); m19.load_state_dict(torch.load('models/vgg19_best.pth')['model_state_dict']); print('VGG19 shape:', m19(torch.randn(2, 3, 224, 224)).shape)"
```
Output:
```
VGG16 shape: torch.Size([2, 2])
VGG19 shape: torch.Size([2, 2])
```
