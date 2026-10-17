"""
Phase 1: Multi-Source Dataset Scanner & Discovery Module
========================================================
Project: Multi-Crop Plant Disease Detection
Author: Team GCET CSE (2022-2026)

Description:
Traverses the raw dataset repository ('Data rough'), identifies target crop 
disease folders across distinct public and indigenous sources, and computes
exact file counts, class distributions, and disk footprint.
"""

import os
import shutil

RAW_DATA_DIR = r"f:\project images data\Data rough"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

def scan_raw_datasets():
    print("=" * 65)
    print("      MULTI-CROP PLANT DISEASE DETECTION: DATASET SCANNER     ")
    print("=" * 65)
    print(f"Scanning raw repository: {RAW_DATA_DIR}\n")

    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Directory '{RAW_DATA_DIR}' does not exist.")
        return

    total_images = 0
    alternaria_images = 0
    source_stats = {}

    for root, dirs, files in os.walk(RAW_DATA_DIR):
        rel_path = os.path.relpath(root, RAW_DATA_DIR)
        parts = rel_path.split(os.sep)
        source_name = parts[0] if parts and parts[0] != '.' else "Root"

        # Check if the path belongs to Alternaria
        lower_path = root.lower()
        is_alternaria_dir = "alternaria" in lower_path or "altenaria" in lower_path

        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                total_images += 1
                if source_name not in source_stats:
                    source_stats[source_name] = {"total": 0, "alternaria": 0, "bytes": 0}

                file_path = os.path.join(root, f)
                file_size = os.path.getsize(file_path)
                source_stats[source_name]["total"] += 1
                source_stats[source_name]["bytes"] += file_size

                # Target class detection: Alternaria
                is_alt_file = is_alternaria_dir or ("alternaria" in f.lower() or "altenaria" in f.lower())
                # Exclude non-alternaria folders in the parent 'alternaria' directory
                if source_name == 'alternaria' and 'alternaria leaf spot' not in lower_path:
                    is_alt_file = False

                if is_alt_file:
                    alternaria_images += 1
                    source_stats[source_name]["alternaria"] += 1

    print(f"{'Source Dataset':<45} | {'Total':<8} | {'Alternaria':<10} | {'Size (MB)':<10}")
    print("-" * 80)
    for src, stat in sorted(source_stats.items()):
        size_mb = stat['bytes'] / (1024 * 1024)
        print(f"{src:<45} | {stat['total']:<8} | {stat['alternaria']:<10} | {size_mb:>8.2f} MB")
    print("-" * 80)
    print(f"Grand Total Images in Repository: {total_images}")
    print(f"Total Alternaria Images Identified: {alternaria_images}\n")

    # Storage Check
    total_d, used_d, free_d = shutil.disk_usage(RAW_DATA_DIR)
    print(f"Disk Status: {free_d / (1024**3):.2f} GB available on drive.")
    print("=" * 65)

if __name__ == "__main__":
    scan_raw_datasets()
