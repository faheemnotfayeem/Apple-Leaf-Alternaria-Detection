# Comparative Held-Out Test Evaluation Report
## Apple Leaf Alternaria Detection System
**Academic Context**: Final Year B.Tech Project in CSE (Batch 2022–2026), GCET Safapora, Ganderbal, Kashmir
**Evaluation Date**: 2026-10-17 14:18:30
**Pathology Classes**: Exactly two classes: `0: Healthy Apple Leaf`, `1: Alternaria Leaf Spot` (*Alternaria alternata*)

---

### 1. Held-Out Evaluation Protocol & Zero Leakage Compliance
- **Held-Out Manifest**: `results/retrained/test_manifest.csv`
- **Total Test Images**: **3,239** images across 1091 duplicate/provenance groups.
- **Class Distribution**: 1,789 Healthy (55.23%) | 1,450 Alternaria (44.77%).
- **Real, unmodified (`eval_eligible`) images**: 1,299; synthetic images: 0.
- **Isolation Assurance**: the test manifest was produced by a group-aware split (no duplicate, re-export, augmentation or CycleGAN group crosses splits) and is never loaded during training or model selection.
- **Preprocessing**: Identical to validation preprocessing across all architectures:
  `Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

---

### 2. Comparative Performance Summary

| Model | Parameters | Model Size | Accuracy | Macro F1 | Weighted F1 | Healthy F1 (Class 0) | Alternaria F1 (Class 1) | CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MOBILENETV2** | 2.39M | 22.3 MB | **98.67%** | **98.65%** | **98.67%** | 98.81% | 98.50% | 11.6 ms |
| **RESNET18** | 11.24M | 107.5 MB | **99.14%** | **99.13%** | **99.14%** | 99.22% | 99.03% | 14.5 ms |
| **VGG16** | 134.79M | 1448.3 MB | **98.58%** | **98.56%** | **98.58%** | 98.72% | 98.40% | 89.8 ms |
| **VGG19** | 140.09M | 1468.5 MB | **98.61%** | **98.60%** | **98.61%** | 98.73% | 98.47% | 107.5 ms |

---

### 3. Confusion Matrix Breakdown

#### MOBILENETV2
- **True Healthy (TN)**: 1785 / 1789 (99.78%)
- **False Alternaria (FP)**: 4 / 1789 (0.22%)
- **False Healthy (FN)**: 39 / 1450 (2.69%)
- **True Alternaria (TP)**: 1411 / 1450 (97.31%)
- Confusion matrix visualization: `confusion_matrix_mobilenetv2.png`

#### RESNET18
- **True Healthy (TN)**: 1782 / 1789 (99.61%)
- **False Alternaria (FP)**: 7 / 1789 (0.39%)
- **False Healthy (FN)**: 21 / 1450 (1.45%)
- **True Alternaria (TP)**: 1429 / 1450 (98.55%)
- Confusion matrix visualization: `confusion_matrix_resnet18.png`

#### VGG16
- **True Healthy (TN)**: 1778 / 1789 (99.39%)
- **False Alternaria (FP)**: 11 / 1789 (0.61%)
- **False Healthy (FN)**: 35 / 1450 (2.41%)
- **True Alternaria (TP)**: 1415 / 1450 (97.59%)
- Confusion matrix visualization: `confusion_matrix_vgg16.png`

#### VGG19
- **True Healthy (TN)**: 1746 / 1789 (97.60%)
- **False Alternaria (FP)**: 43 / 1789 (2.40%)
- **False Healthy (FN)**: 2 / 1450 (0.14%)
- **True Alternaria (TP)**: 1448 / 1450 (99.86%)
- Confusion matrix visualization: `confusion_matrix_vgg19.png`

---

### 4. Technical Analysis & Architectural Trade-offs
1. **Accuracy Leader**: **RESNET18** achieved the highest test accuracy of **99.14%** and Macro F1 of **99.13%**.
2. **Inference Latency Leader**: **MOBILENETV2** delivered the fastest inference at **11.60 ms/image** (86.2 frames/sec on CPU).
3. **Deployment Note**: model size and CPU latency above determine edge-deployment feasibility; interpret accuracy together with the real-image subset below.

---

### 4b. Real-Image Subset (`eval_eligible` only: original captures, no augmentation / re-export / synthetic)

| Model | Samples | Accuracy | Macro F1 | Healthy F1 | Alternaria F1 | Confusion [[TN,FP],[FN,TP]] |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **MOBILENETV2** | 1299 | 99.54% | 99.48% | 99.66% | 99.29% | `[[871, 4], [2, 422]]` |
| **RESNET18** | 1299 | 99.38% | 99.30% | 99.54% | 99.06% | `[[868, 7], [1, 423]]` |
| **VGG16** | 1299 | 99.08% | 98.96% | 99.31% | 98.60% | `[[864, 11], [1, 423]]` |
| **VGG19** | 1299 | 96.77% | 96.41% | 97.54% | 95.28% | `[[833, 42], [0, 424]]` |

---

### 5. Generated Artifacts Registry
- `results/retrained_evaluation/comparison.csv`: High-level multi-model comparative metrics (full test split).
- `results/retrained_evaluation/comparison_eval_eligible.csv`: Same metrics on the real-image subset.
- `results/retrained_evaluation/classification_report_eval_eligible.csv`: Per-class precision/recall/F1 on the real-image subset.
- `results/retrained_evaluation/confusion_matrices/confusion_matrix_*_eval_eligible.png`: Confusion matrices on the real-image subset.
- `results/retrained_evaluation/classification_report.csv`: Detailed class-by-class precision, recall, and F1 metrics.
- `results/retrained_evaluation/test_predictions.csv`: Sample-level predictions and posterior probabilities.
- `results/retrained_evaluation/confusion_matrices/confusion_matrix_*.png`: High-resolution confusion matrices for all 4 architectures.
- `results/retrained_evaluation/comparative_metrics_chart.png`: Visual multi-metric comparative bar chart.