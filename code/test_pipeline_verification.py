import sys
from pathlib import Path
import pandas as pd
from PIL import Image

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import app

def test_full_pipeline():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE PIPELINE TEST SUITE")
    print("=" * 70)

    # 1. Load resources directly via app helper functions
    print("\n1. Initializing models...")
    gatekeeper = app.load_gatekeeper()
    crop_detector_bundle = app.load_crop_detector()
    model, transform, labels = crop_detector_bundle
    print(f"[OK] Gatekeeper loaded.")
    print(f"[OK] 36-Crop Detector loaded ({len(labels)} classes).")
    print(f"  Class 0: {labels[0]}")
    
    # Check if Pomegranate is in 36 classes
    pom_in_classes = "Pomegranate" in labels
    print(f"  Pomegranate in crop classes: {pom_in_classes}")

    # 2. Select test images
    test_cases = []

    # A. Scenery
    scenery_path = PROJECT_ROOT / "code" / "test_scenery.jpg"
    if scenery_path.exists():
        test_cases.append(("Scenery Test Image", scenery_path, "Non-foliar Scenery"))
    else:
        print("Scenery test image missing!")

    # B. Desk / Screenshot
    desk_path = PROJECT_ROOT / "code" / "test_desk.jpg"
    if desk_path.exists():
        test_cases.append(("Screenshot/Desk Test Image", desk_path, "Non-foliar Screenshot/Desk"))
    else:
        print("Desk test image missing!")

    # C. Known Healthy Apple Leaf (first image in the source dataset folder; read-only)
    healthy_path = sorted((PROJECT_ROOT / "DATASET" / "Healthy").glob("*.jpg"))[0]
    test_cases.append(("Known Healthy Apple Leaf", healthy_path, "Apple (Healthy)"))

    # D. Known Alternaria Apple Leaf (first image in the source dataset folder; read-only)
    alternaria_path = sorted((PROJECT_ROOT / "DATASET" / "Alternaria leaf spot").glob("*.jpg"))[0]
    test_cases.append(("Known Alternaria Apple Leaf", alternaria_path, "Apple (Alternaria)"))

    # E. Search for Pomegranate image
    pomegranate_matches = list(PROJECT_ROOT.glob("**/*pomegranate*")) + list(PROJECT_ROOT.glob("**/*Pomegranate*"))
    if pomegranate_matches:
        test_cases.append(("Pomegranate Leaf", pomegranate_matches[0], "Pomegranate Leaf"))
    else:
        print("Note: No pomegranate image exists in the project repository.")

    # F. Non-Apple Leaf (e.g. Tomato Leaf from available files)
    tomato_leaves = list(Path("C:/Users/haris/Downloads").glob("**/Tomato*/**/*.jpg")) + list(Path("C:/Users/haris/Downloads").glob("**/Tomato*/**/*.JPG"))
    if tomato_leaves:
        test_cases.append(("Tomato Leaf (Non-Apple Crop)", tomato_leaves[0], "Tomato Leaf"))
    
    print(f"\n2. Testing {len(test_cases)} images across 4 disease classification models...")
    print("-" * 70)

    results_table = []

    for name, img_path, desc in test_cases:
        print(f"\nEvaluating: {name} ({img_path.name})")
        img = Image.open(img_path)
        
        # Test with MobileNetV2
        res_mb = app.run_diagnostic_pipeline(img, gatekeeper, crop_detector_bundle)
        status = res_mb["status"]
        sim = res_mb.get("similarity", 0.0)
        
        row = {
            "Specimen": name,
            "Description": desc,
            "Status": status,
            "Stage 1 Sim": f"{sim:.4f}",
            "Stage 2 Pred": res_mb.get("predicted_crop", "N/A"),
            "Stage 2 Conf": f"{res_mb.get('crop_confidence', 0.0)*100:.1f}%" if "crop_confidence" in res_mb else "N/A",
            "Stage 3 Pred": res_mb.get("final_prediction", "Blocked"),
            "Stage 3 Conf": f"{res_mb.get('confidence', 0.0)*100:.1f}%" if "confidence" in res_mb else "Blocked",
        }
        results_table.append(row)
        print(f"  -> Result: {status}")
        if status == "success":
            print(f"     Crop: {row['Stage 2 Pred']} ({row['Stage 2 Conf']}) | Disease: {row['Stage 3 Pred']} ({row['Stage 3 Conf']})")
        elif status == "rejected_non_leaf":
            print(f"     Rejected at Stage 1: Leaf similarity {row['Stage 1 Sim']} < threshold")
        elif status == "rejected_non_apple":
            print(f"     Rejected at Stage 2: Detected {row['Stage 2 Pred']} ({row['Stage 2 Conf']})")
        elif status == "rejected_low_confidence":
            print(f"     Rejected at Stage 2: Low confidence Apple ({row['Stage 2 Conf']})")

    # 4-model majority-voting consensus on the two known specimens
    for title, path in (("Known Healthy Apple Leaf", healthy_path), ("Known Alternaria Apple Leaf", alternaria_path)):
        print(f"\n3. 4-model consensus on {title}:")
        res = app.run_diagnostic_pipeline(Image.open(path), gatekeeper, crop_detector_bundle)
        if res["status"] == "success":
            print(f"  - Consensus: {res['final_label']} ({res['confidence']*100:.2f}%) | {res['explanation']}")
        else:
            print(f"  - Blocked at {res.get('stage')}: {res.get('message')}")

    print("\n" + "=" * 70)
    print("SUMMARY RESULTS TABLE:")
    print("=" * 70)
    df_res = pd.DataFrame(results_table)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    test_full_pipeline()
