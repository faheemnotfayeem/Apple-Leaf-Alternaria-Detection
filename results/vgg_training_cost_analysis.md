# Training Cost Estimation & Architectural Benchmarks
## Four-Model Transfer Learning on CPU vs. GPU Acceleration
**Project**: Apple Leaf Alternaria Detection  
**Hardware Environment**: Windows 11 CPU-Only (`torch 2.11.0+cpu`, Intel/AMD Multi-Core CPU, No CUDA)  
**Dataset Scale**: 6,418 Training Images | 1,319 Validation Images | 1,352 Held-Out Test Images  
**Batch Size**: 32 | **Input Dimensions**: 224 × 224 × 3 RGB  

---

### 1. Empirical Latency & Cost Benchmark (Local CPU)
Prior to launching full training, a rigorous smoke test and latency benchmark was conducted on the local CPU hardware using identical forward-backward passes, weighted cross-entropy loss calculation, and optimization steps across all four candidate architectures.

| Architecture | Total Parameters | Trainable Parameters | CPU Latency (ms/img) | Estimated 1-Epoch Train Time | Estimated 5-Epoch Run Time | Practical Feasibility on CPU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2** | **2.39 M** | 1.69 M | **17.7 ms** | **~1.9 min** (114 s) | **~9.5 min** | **High** (Feasible locally) |
| **ResNet18** | **11.24 M** | 0.07 M | **16.2 ms** | **~1.7 min** (104 s) | **~8.5 min** | **High** (Feasible locally) |
| **VGG16** | **134.79 M** | 122.43 M | **324.4 ms** | **~34.7 min** (2,082 s) | **~2.9 hours** | **Prohibitive on CPU** |
| **VGG19** | **140.09 M** | 122.43 M | **226.3 ms** | **~24.2 min** (1,452 s) | **~2.0 hours** | **Prohibitive on CPU** |

---

### 2. Analysis of the VGG Computational Bottleneck
1. **Architectural Complexity & Dense Weights**:
   - VGG architectures contain massive fully connected classifier stages (`4096 -> 4096 -> 128 -> 2`) totaling over **134 to 140 million parameters** (~530–550 MB per model checkpoint).
   - In contrast, MobileNetV2 uses depthwise separable convolutions and inverted residuals (2.39M parameters), while ResNet18 uses residual skip connections and global average pooling (11.24M parameters).
2. **CPU Instruction Throughput**:
   - While MobileNetV2 and ResNet18 process images at ~16–18 ms/img on CPU, VGG16 requires **324.4 ms per image**—nearly **18× slower**.
   - Training VGG16 and VGG19 for 5 epochs each would tie up CPU resources for **~5 hours continuously**, risking thermal throttling, process interruption, and indefinite background execution.
3. **GPU Hardware Acceleration Factor**:
   - On a standard NVIDIA T4 Tensor Core GPU (available free on Google Colab), parallelized tensor cores accelerate large matrix multiplications by **25×–40×**.
   - Expected VGG16/VGG19 training time on Google Colab GPU is only **~40–60 seconds per epoch** (~3–5 minutes total per model).

---

### 3. Execution Strategy
In compliance with project specifications:
1. **Local Training**:
   - Execute **MobileNetV2** locally to completion.
   - Execute **ResNet18** locally to completion.
2. **Google Colab GPU Offloading for VGG Architectures**:
   - Provide a dedicated, ready-to-run Google Colab GPU training script and notebook (`code/train_colab_gpu.py` / `code/Apple_Leaf_Comparative_Training_Colab.ipynb`).
   - Use the **exact same deterministic manifests** (`results/train_split.csv` and `results/val_split.csv`) and weighted loss to ensure zero distribution divergence.
   - Provide exact, step-by-step instructions for the user to execute VGG16 and VGG19 on Colab and download the verified checkpoints directly into `models/`.
