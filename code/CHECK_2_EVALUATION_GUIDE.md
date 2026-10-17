# Check 2 (Mid-Term Evaluation) Defense & Presentation Guide
### Project: Apple Leaf Alternaria Detection: An AI-Based Approach Using Deep Learning and Explainable AI
**Department of Computer Science & Engineering, GCET Safapora (2022–2026)**

---

## 1. Project Identity & Problem Statement
* **Exact Title**: *AI-Driven Apple Leaf Alternaria Detection with Explainable Deep Learning*
* **Core Problem in Kashmir Orchards**:
  * In Kashmir apple orchards, **Alternaria Leaf Spot** (*Alternaria alternata*) is a widespread fungal disease leading to premature leaf drop and severe loss in fruit yield and tree vigor.
  * In early stages, small necrotic spots can be overlooked or misidentified. If left unmanaged, lesions rapidly coalesce into extensive necrotic zones with concentric rings and yellow halos.
  * Timely diagnosis enables farmers to apply protective/curative fungicides (such as Mancozeb 75 WP at 2.5 g/L or Propineb 70 WP at 2.0 g/L) before widespread canopy infection occurs.
  * Our computer vision system automatically validates leaf photos, rejects non-foliar/background noise, and classifies apple leaves into **Healthy** foliage and **Alternaria Leaf Spot** with calibrated probability scores and **Grad-CAM visual heatmaps**.

---

## 2. What We Have Built for Check 2 (Progress Summary)

| Component | Technical Detail | Where to Show the Teacher |
| :--- | :--- | :--- |
| **1. Data Engineering & Integrity** | Aggregated multi-source repositories (`AL9EE`, `archive`, `Indigenous`, `coco`), standardized formats to 3-channel RGB, verified zero byte corruptions across 9,324 audited leaf images. | [`01_dataset_scanner.py`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/code/01_dataset_scanner.py) & [`02_dataset_aggregator.py`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/code/02_dataset_aggregator.py) |
| **2. Perceptual Hashing (dHash)** | Computed 64-bit difference perceptual hashes (dHash) to group 7,481 unique physical leaf groups, strictly preventing near-duplicate/augmented leakage across Train, Validation, and Test splits. | [`03_perceptual_hash_audit.py`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/code/03_perceptual_hash_audit.py) & [`dataset_audit_report.csv`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/alternaria%20for%20model/dataset_audit_report.csv) |
| **3. Transfer Learning Pipeline** | Fine-tuned **MobileNetV2** (pre-trained on ImageNet) with AdamW optimizer, cosine annealing schedule, and label smoothing for two-class discrimination (`Healthy` vs `Alternaria`). | [`train_alternaria_model.py`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/code/train_alternaria_model.py) & [`Apple_Leaf_Alternaria_Detection_Colab.ipynb`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/code/Apple_Leaf_Alternaria_Detection_Colab.ipynb) |
| **4. Explainability (Grad-CAM)** | Implemented Gradient-Weighted Class Activation Mapping to visually highlight lesion sites vs background tissue, ensuring transparency. | Colab Notebook & [`04_quick_inference.py`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/code/04_quick_inference.py) |
| **5. Cascaded Gatekeeper Inference** | Two-stage diagnostic engine: Stage 1 Foundation Gatekeeper (rejects desks, rooms, landscapes) and Stage 2 Pathology Classifier with SKUAST agronomic recommendations. | [`04_quick_inference.py`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/code/04_quick_inference.py) |

---

## 3. Tough Questions Teachers Ask & Your Exact Answers

### Q1: *"Why did you choose MobileNetV2 instead of ResNet-50 or VGG-16?"*
> **Answer**:  
> *"ResNet-50 has 25.6 million parameters and VGG-16 has 138 million parameters, making them slow and computationally heavy. MobileNetV2 utilizes inverted residual blocks with depthwise separable convolutions, achieving high diagnostic accuracy with only **3.5 million parameters**. This allows inference latency under 25 milliseconds, which is critical for future deployment on edge mobile devices used by farmers directly in orchards without requiring high-end GPUs or high-speed connectivity."*

### Q2: *"How do you know your model isn't just memorizing background artifacts (soil, hands, sky)?"*
> **Answer**:  
> *"We implemented **Grad-CAM (Gradient-Weighted Class Activation Mapping)**. During testing, Grad-CAM maps the gradients flowing into the final convolutional layer (`model.features[-1]`) back onto the original image. As shown in our heatmaps, the peak activation intensity (red zones) concentrates specifically on the brown necrotic lesions and chlorotic halos of Alternaria, proving the network attends to pathological symptoms rather than background artifacts."*

### Q3: *"What is perceptual hashing (dHash) and why did you use it?"*
> **Answer**:  
> *"In public agricultural datasets, many images are pre-augmented copies (flipped, cropped, or brightness-adjusted) of the exact same physical leaf. A naive random split would place one version in the training set and another in the test set, creating severe **data leakage** and artificially inflated accuracy. We used 64-bit difference perceptual hashing (dHash) to assign identical physical leaves to distinct `Visual_Group_IDs`, ensuring that copies of the same leaf never cross between train, validation, and test splits."*

### Q4: *"Where is the web interface? Why aren't you showing a website today?"*
> **Answer**:  
> *"For Check 2, our mandate was to establish the core mathematical and machine learning foundation: robust data engineering, leak-proof dataset partitioning, CNN transfer learning, and diagnostic explainability. With the inference engine and model benchmarks validated today, **Check 3 introduces the full FastAPI backend and the Next.js interactive web/mobile interface for end-user deployment.**"*

---

## 4. Check 2 Live Demonstration Script (Step-by-Step)

1. **Step 1**: Show the terminal running the dataset scanner:
   ```bash
   python code/01_dataset_scanner.py
   ```
2. **Step 2**: Open [`dataset_audit_report.csv`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/alternaria%20for%20model/dataset_audit_report.csv) in Excel or VS Code to show the 9,324 audited entries with 64-bit pHash fingerprints and 7,481 unique physical leaf groups.
3. **Step 3**: Open the Google Colab Notebook [`Apple_Leaf_Alternaria_Detection_Colab.ipynb`](file:///c:/Users/haris/OneDrive/Desktop/Apple%20Leaf%20Alternaria%20Detection/code/Apple_Leaf_Alternaria_Detection_Colab.ipynb) on your browser to show the training loop, Confusion Matrix, and Grad-CAM heatmap cells.
4. **Step 4**: Run the local test engine in terminal on a sample leaf photo:
   ```bash
   python code/04_quick_inference.py --image "alternaria for model/AL9EE__Alternaria_leaf_spot_(1).jpg"
   ```
   Show how it passes Stage 1 Leaf Gatekeeper, identifies Alternaria Leaf Spot, and outputs the exact Kashmiri orchard agronomic recommendation (Mancozeb 75 WP or Propineb 70 WP).
5. **Step 5**: Run the test engine on a non-leaf image to demonstrate robust rejection:
   ```bash
   python code/04_quick_inference.py --image "code/test_desk.jpg"
   ```
   Show that it detects a similarity score below 0.3300 and rejects non-foliar scenes immediately.
