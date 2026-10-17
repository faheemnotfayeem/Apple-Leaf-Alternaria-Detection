"""
Phase 1: Collision-Free Multi-Source Image Aggregator
====================================================
Project: Multi-Crop Plant Disease Detection
Author: Team GCET CSE (2022-2026)

Description:
Consolidates target disease leaf images (Alternaria) from diverse nested 
datasets into a unified, flat training folder. Every image receives a prefix
encoding its provenance (source name + original split/context) to eliminate
filename collisions without data loss.
"""

import os
import shutil
import time

SOURCE_ROOT = r"f:\project images data\Data rough"
DEST_ROOT = r"f:\project images data\alternaria for model"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

# Mapping source subdirectories to standardized provenance prefixes
SOURCE_MAPPINGS = [
    {
        "name": "AL9EE",
        "path": os.path.join(SOURCE_ROOT, "AL9EE", "AL9EE", "Alternaria leaf spot"),
        "prefix": "AL9EE__"
    },
    {
        "name": "Indigenous_Train",
        "path": os.path.join(SOURCE_ROOT, "Indigenous Dataset for Apple Leaf Disease Detection and Classification", "Indigenous Dataset for Apple Leaf Disease Detection and Classification", "Train_Set", "Alternaria"),
        "prefix": "Indigenous_Train__"
    },
    {
        "name": "Indigenous_Test",
        "path": os.path.join(SOURCE_ROOT, "Indigenous Dataset for Apple Leaf Disease Detection and Classification", "Indigenous Dataset for Apple Leaf Disease Detection and Classification", "Test_Set", "Alternaria"),
        "prefix": "Indigenous_Test__"
    },
    {
        "name": "archive_Train",
        "path": os.path.join(SOURCE_ROOT, "archive", "AppleLeafDisease", "Train", "Altenaria_Leaf_Spot"),
        "prefix": "archive_Train__"
    },
    {
        "name": "archive_Test",
        "path": os.path.join(SOURCE_ROOT, "archive", "AppleLeafDisease", "Test", "Altenaria_Leaf_Spot"),
        "prefix": "archive_Test__"
    },
    {
        "name": "archive_All",
        "path": os.path.join(SOURCE_ROOT, "archive", "AppleLeafDisease", "All", "Altenaria_Leaf_Spot"),
        "prefix": "archive_All__"
    },
    {
        "name": "archive_Aug",
        "path": os.path.join(SOURCE_ROOT, "archive", "AppleLeafDisease", "Aug_Set", "Altenaria_Leaf_Spot"),
        "prefix": "archive_Aug__"
    },
    {
        "name": "alternaria_source",
        "path": os.path.join(SOURCE_ROOT, "alternaria", "Alternaria leaf spot"),
        "prefix": "alternaria_source__"
    }
]

def aggregate_dataset():
    os.makedirs(DEST_ROOT, exist_ok=True)
    print("=" * 65)
    print("       DATASET AGGREGATION & SOURCE PREFIXING PIPELINE        ")
    print("=" * 65)
    print(f"Destination: {DEST_ROOT}\n")

    total_copied = 0
    start_time = time.time()

    # 1. Process standard mapped folders
    for mapping in SOURCE_MAPPINGS:
        folder = mapping["path"]
        prefix = mapping["prefix"]
        tag = mapping["name"]

        if not os.path.exists(folder):
            print(f"Skipping missing directory: {folder}")
            continue

        count = 0
        for fname in os.listdir(folder):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                src_file = os.path.join(folder, fname)
                clean_name = fname.replace(" ", "_")
                dst_fname = f"{prefix}{clean_name}"
                dst_file = os.path.join(DEST_ROOT, dst_fname)

                # Collision safety check
                counter = 1
                base_name = dst_fname
                while os.path.exists(dst_file):
                    r_name, r_ext = os.path.splitext(base_name)
                    dst_file = os.path.join(DEST_ROOT, f"{r_name}_{counter}{r_ext}")
                    counter += 1

                shutil.copy2(src_file, dst_file)
                count += 1
                total_copied += 1

        print(f"[+] Sourced from {tag:<22} -> Copied {count:>5} files")

    # 2. Process apple_disease.coco dataset splits
    coco_dir = os.path.join(SOURCE_ROOT, "apple_disease.coco")
    for split in ['train', 'test', 'valid']:
        split_path = os.path.join(coco_dir, split)
        if os.path.exists(split_path):
            count = 0
            for fname in os.listdir(split_path):
                ext = os.path.splitext(fname)[1].lower()
                if ext in IMAGE_EXTENSIONS and ('alternaria' in fname.lower() or 'altenaria' in fname.lower()):
                    src_file = os.path.join(split_path, fname)
                    clean_name = fname.replace(" ", "_")
                    dst_fname = f"coco_{split}__{clean_name}"
                    dst_file = os.path.join(DEST_ROOT, dst_fname)

                    counter = 1
                    base_name = dst_fname
                    while os.path.exists(dst_file):
                        r_name, r_ext = os.path.splitext(base_name)
                        dst_file = os.path.join(DEST_ROOT, f"{r_name}_{counter}{r_ext}")
                        counter += 1

                    shutil.copy2(src_file, dst_file)
                    count += 1
                    total_copied += 1

            print(f"[+] Sourced from coco_{split:<17} -> Copied {count:>5} files")

    elapsed = time.time() - start_time
    print("-" * 65)
    print(f"Aggregation Completed in {elapsed:.2f} seconds.")
    print(f"Total Images in Consolidated Target Folder: {len(os.listdir(DEST_ROOT))}")
    print("=" * 65)

if __name__ == "__main__":
    aggregate_dataset()
