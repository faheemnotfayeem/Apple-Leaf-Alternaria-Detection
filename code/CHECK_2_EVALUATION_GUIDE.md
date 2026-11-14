# Check 2 (Mid-Term Evaluation) Defense & Presentation Guide
### Project: AppleAlt: AI-Based Apple Leaf Alternaria Detection
**Department of Computer Science & Engineering, GCET Safapora (2022–2026)**

> Updated 2026-10-22 to match the final retrained pipeline. All numbers below come from
> `results/retrained/` and `results/retrained_evaluation/`.

---

## 1. Project Identity & Problem Statement
* **Exact Title**: *AI-Driven Apple Leaf Alternaria Detection with Explainable Deep Learning*
* **Core Problem in Kashmir Orchards**:
  * In Kashmir apple orchards, **Alternaria Leaf Spot** (*Alternaria alternata*) is a widespread fungal disease leading to premature leaf drop and loss in fruit yield and tree vigor.
  * In early stages, small necrotic spots can be overlooked or misidentified. If left unmanaged, lesions coalesce into extensive necrotic zones with concentric rings and yellow halos.
  * Timely diagnosis lets growers start sanitation and a protective fungicide programme (as advised by the local horticulture extension service) before widespread canopy infection occurs.
  * Our computer vision system checks that the photo shows a leaf, rejects non-foliar/background images, and classifies apple leaves as **Healthy** or **Alternaria Leaf Spot** with a softmax confidence score and **Grad-CAM visual heatmaps**.

---

## 2. What We Have Built for Check 2 (Progress Summary)

| Component | Technical Detail | Where to Show the Teacher |
| :--- | :--- | :--- |
| **1. Data Engineering & Integrity** | Two class folders in `DATASET/` (11,926 Healthy, 10,265 Alternaria = 22,191 images), every image opened, verified and decoded (0 unreadable). Sources identified from filename patterns (PlantVillage, Indigenous LPU, AL9EE, CycleGAN, web thumbnails, phone photos). | [`03_perceptual_hash_audit.py`](03_perceptual_hash_audit.py) & [`dataset_audit_report.md`](../results/retrained/dataset_audit_report.md) |
| **2. Duplicate Audit (SHA-256 + pHash + dHash)** | SHA-256 for exact copies (1,996 redundant copies in 1,069 groups); 64-bit pHash and dHash under all 8 flips/rotations for near-duplicates (pHash ≤ 6 and dHash ≤ 5, or pHash ≤ 2); filename provenance families; union-find grouping into **7,797 groups**. | [`03_perceptual_hash_audit.py`](03_perceptual_hash_audit.py) & [`duplicate_groups.csv`](../results/retrained/duplicate_groups.csv) |
| **3. Leakage-Safe Split** | Group-aware, stratified, seeded (42) 70/15/15 split: 15,111 train / 3,239 val / 3,239 test; 602 doubtful images excluded; all CycleGAN groups forced into train; 25 validation checks pass. | [`prepare_clean_splits.py`](prepare_clean_splits.py) & [`split_summary.csv`](../results/retrained/split_summary.csv) |
| **4. Transfer Learning Pipeline** | Four ImageNet-pretrained CNNs (MobileNetV2, ResNet18, VGG16, VGG19) fine-tuned with Adam (lr 1e-4, weight decay 1e-4), ReduceLROnPlateau, class-weighted cross-entropy, batch 32, 5 epochs, best checkpoint by validation accuracy. | [`train_comparative_models.py`](train_comparative_models.py) & [`colab/Apple_Leaf_Retrained_Training_Colab.ipynb`](colab/Apple_Leaf_Retrained_Training_Colab.ipynb) |
| **5. Held-Out Evaluation** | Test set used once: MobileNetV2 98.67% (full test) / 99.54% (1,299 real originals); ResNet18 99.14% / 99.38%. | [`evaluate_all_models.py`](evaluate_all_models.py) & [`final_evaluation_report.md`](../results/retrained_evaluation/final_evaluation_report.md) |
| **6. Explainability (Grad-CAM)** | Gradient-weighted Class Activation Mapping on MobileNetV2 `features[-1]` highlights lesion sites vs background tissue. | [`04_quick_inference.py`](04_quick_inference.py) (`--cam`) |
| **7. Cascaded Gatekeeper Inference** | Stage 1 leaf gatekeeper (cosine similarity to `leaf_prototype.pt` + HSV foliage ratio; rejects desks, rooms, landscapes) followed by the MobileNetV2 two-class classifier. | [`04_quick_inference.py`](04_quick_inference.py) |

---

## 3. Tough Questions Teachers Ask & Your Exact Answers

### Q1: *"Why did you choose MobileNetV2 instead of ResNet or VGG?"*
> **Answer**:  
> *"We trained and compared all of them under identical settings. VGG16 and VGG19 have about 135–140 million parameters and take 90–108 ms per image on our CPU; ResNet18 has 11.2 million. Our MobileNetV2 has only **2.39 million parameters** (the original 1,000-class MobileNetV2 has about 3.5 million; we replaced its classifier with a small 2-class head). It uses inverted residual blocks with depthwise separable convolutions, reaches **99.54% accuracy on the real-image test subset**, and runs in **11.6 ms per image on CPU**, which suits future deployment on phones and low-cost hardware."*

### Q2: *"How do you know your model isn't just memorizing background artifacts (soil, hands, sky)?"*
> **Answer**:  
> *"We implemented **Grad-CAM (Gradient-Weighted Class Activation Mapping)**. It weights the feature maps of the final convolutional layer (`model.features[-1]`) by the average gradient of the predicted class and projects the result back onto the image. In our heatmaps the red zones concentrate on the brown necrotic lesions and chlorotic halos of Alternaria, showing the network attends to disease symptoms rather than background artifacts."*

### Q3: *"What is perceptual hashing and why did you use it?"*
> **Answer**:  
> *"In public agricultural datasets, many images are copies of the same physical leaf: byte-identical duplicates, flips, rotations, crops and re-exports. A naive random split would place one version in training and another in testing, creating **data leakage** and inflated accuracy. We used SHA-256 for exact copies and two 64-bit perceptual hashes, pHash (DCT-based) and dHash (gradient-based), compared with Hamming distance under all 8 flips/rotations, plus filename provenance. Union-find merged all links into 7,797 groups, and each group is assigned to exactly one split."*

### Q4: *"Where is the web interface?"*
> **Answer**:  
> *"Check 2 established the machine learning foundation: data audit, leak-proof partitioning, transfer learning and explainability. The validated models are served by a **FastAPI backend** (`backend/main.py`) and a **Next.js website** (`frontend/`), with a Streamlit version in `app.py`."*

---

## 4. Check 2 Live Demonstration Script (Step-by-Step)

1. **Step 1**: Show the dataset audit report:
   open [`results/retrained/dataset_audit_report.md`](../results/retrained/dataset_audit_report.md) to show the 22,191 audited images, 1,996 exact-duplicate copies and 7,797 duplicate/provenance groups.
2. **Step 2**: Show the split and its checks:
   ```bash
   python code/prepare_clean_splits.py --validate-only
   ```
3. **Step 3**: Open [`results/retrained_evaluation/final_evaluation_report.md`](../results/retrained_evaluation/final_evaluation_report.md) and the confusion matrices in `results/retrained_evaluation/confusion_matrices/`.
4. **Step 4**: Run the local test engine on a sample leaf photo with Grad-CAM:
   ```bash
   python code/04_quick_inference.py --image "DATASET/Alternaria leaf spot/lpu_rdp_alt670.jpeg" --cam
   ```
   Show how it passes the Stage 1 leaf gatekeeper, identifies Alternaria Leaf Spot, prints the class probabilities, and saves the `_gradcam.jpg` overlay next to the image.
5. **Step 5**: Run the test engine on a non-leaf image to demonstrate rejection:
   ```bash
   python code/04_quick_inference.py --image "code/test_desk.jpg"
   ```
   Show that its similarity score (about 0.14) is below the 0.3300 threshold and the image is rejected.
