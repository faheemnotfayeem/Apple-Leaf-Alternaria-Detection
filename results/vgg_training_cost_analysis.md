# Training Cost Estimation & Architectural Benchmarks
## Four-Model Transfer Learning on CPU vs. GPU Acceleration
**Project**: AppleAlt: AI-Based Apple Leaf Alternaria Detection  
**Hardware Environment**: Windows 11 CPU-Only (Intel/AMD Multi-Core CPU, No CUDA) for local work; Google Colab NVIDIA T4 GPU for the heavy models  
**Final Dataset Scale** (`results/retrained/split_summary.csv`): 15,111 Training Images | 3,239 Validation Images | 3,239 Held-Out Test Images  
**Batch Size**: 32 | **Input Dimensions**: 224 × 224 × 3 RGB | **Epochs**: 5

> Updated 2026-10-22. Section 1 is the planning benchmark that was run on an earlier, smaller version of
> the dataset (6,418 / 1,319 / 1,352 images) before the retrained pipeline existed. Section 3 records what
> actually happened in the final runs; those numbers come from `results/retrained/<model>/config.json`.

---

### 1. Planning Benchmark (Local CPU, earlier dataset version)
Before launching full training, a smoke test timed identical forward-backward passes, weighted cross-entropy and optimizer steps for the four candidate architectures on the local CPU.

| Architecture | Total Parameters | Trainable Parameters (final setup) | CPU Train Time (ms/img, benchmark) | Benchmark 5-Epoch Estimate (6,418 imgs) | Scaled 5-Epoch Estimate (15,111 imgs) | Practical Feasibility on CPU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2** | **2.39 M** | 1.69 M | **17.7 ms** | ~9.5 min | ~22 min | **High** (Feasible locally) |
| **ResNet18** | **11.24 M** | 8.46 M | **16.2 ms*** | ~8.5 min* | ~20 min* | **High** (Feasible locally) |
| **VGG16** | **134.79 M** | 122.43 M | **324.4 ms** | ~2.9 hours | ~6.8 hours | **Prohibitive on CPU** |
| **VGG19** | **140.09 M** | 122.43 M | **226.3 ms** | ~2.0 hours | ~4.7 hours | **Prohibitive on CPU** |

\* The ResNet18 benchmark was taken when only its classifier head (0.07 M parameters) was trainable. The final setup also trains `layer4` (8.46 M trainable), so the backward pass, and therefore the real CPU time, is higher than this estimate.

Trainable parameters in the final setup: MobileNetV2 trains `features[15-18]` plus the new head; ResNet18 trains `layer4` plus the new head; VGG16/VGG19 train their last convolution layer plus the full fully connected classifier.

---

### 2. Analysis of the VGG Computational Bottleneck
1. **Architectural Complexity & Dense Weights**:
   - VGG architectures contain massive fully connected classifier stages (`25088 -> 4096 -> 4096 -> 128 -> 2`) totaling **134 to 140 million parameters** (~515–535 MB of weights per model). The saved checkpoints are about 1.45 GB each because they also store the Adam optimizer state for the 122.4 M trainable parameters.
   - In contrast, MobileNetV2 uses depthwise separable convolutions and inverted residuals (2.39 M parameters), while ResNet18 uses residual skip connections and global average pooling (11.24 M parameters).
2. **CPU Instruction Throughput**:
   - While MobileNetV2 and ResNet18 trained at ~16–18 ms/img in the benchmark, VGG16 required **324.4 ms per image**, roughly **18× slower**.
   - On the final 15,111-image training set, 5 epochs of VGG16 and VGG19 would occupy the CPU for an estimated **~11 hours** combined, risking thermal throttling and process interruption.
3. **GPU Hardware Acceleration**:
   - An NVIDIA T4 GPU (free on Google Colab) parallelizes the large matrix multiplications, so the VGG models train in minutes per epoch instead of hours.

---

### 3. Execution Strategy and Actual Results
1. **Local Training (CPU)**:
   - **MobileNetV2** was trained locally to completion: 5 epochs, **25.6 minutes** (~5 minutes per epoch), best validation accuracy 99.75% at epoch 5.
   - The local driver was stopped after MobileNetV2, so ResNet18 was not trained locally.
2. **Google Colab GPU (T4) for ResNet18, VGG16 and VGG19**:
   - Trained with the Colab notebook `code/colab/Apple_Leaf_Retrained_Training_Colab.ipynb` (see `code/colab/README_COLAB.md`), using the same training script, the same seed (42), preprocessing, weighted loss and the **same manifests** (`results/retrained/train_manifest.csv` and `results/retrained/val_manifest.csv`; their SHA-256 hashes in every model's `config.json` are identical).
   - Actual training times: **ResNet18 8.7 min** (~1.7 min/epoch), **VGG16 16.3 min** (~2.3–3.3 min/epoch), **VGG19 17.9 min** (~2.7–3.6 min/epoch).
   - Best validation accuracy: ResNet18 99.29% (epoch 4), VGG16 99.57% (epoch 2), VGG19 98.49% (epoch 3).
3. **Evaluation**:
   - All four checkpoints were evaluated once, on the PC CPU, with `code/evaluate_all_models.py` on the held-out test set. CPU inference latency (batch 32, including image loading): MobileNetV2 11.6 ms, ResNet18 14.5 ms, VGG16 89.8 ms, VGG19 107.5 ms per image. Full results: `results/retrained_evaluation/final_evaluation_report.md`.
