# Dataset Audit Report (retrained pipeline)

Generated: 2026-10-16 22:58  |  Source: `DATASET/` (read-only)  |  Script: `code/03_perceptual_hash_audit.py`

Class mapping (explicit, never alphabetical): `DATASET/Healthy` -> **Healthy = 0**, `DATASET/Alternaria leaf spot` -> **Alternaria = 1**.

## 1. Folder contents

| Folder | Total entries | Supported images | Non-image entries | Unreadable images |
|---|---:|---:|---:|---:|
| `Alternaria leaf spot` | 10269 | 10265 | 4 | 0 |
| `Healthy` | 11926 | 11926 | 0 | 0 |
| **Total** | 22195 | 22191 | 4 | 0 |

Non-image entries (ignored, not deleted):

- `Alternaria leaf spot/latest_net_D_A.pth` - unsupported extension '.pth', 11.1 MB
- `Alternaria leaf spot/latest_net_D_B.pth` - unsupported extension '.pth', 11.1 MB
- `Alternaria leaf spot/latest_net_G_A.pth` - unsupported extension '.pth', 45.5 MB
- `Alternaria leaf spot/latest_net_G_B.pth` - unsupported extension '.pth', 45.5 MB

All supported images opened, verified and decoded successfully (0 unreadable).

## 2. Formats, colour modes and dimensions

**Alternaria leaf spot** (10265 images)

- Extensions: `.jpg` 7646, `.jpeg` 2619
- Decoded formats: JPEG 7662, PNG 2603
- Colour modes: RGB 10265
- Distinct dimensions: 133; most common: 256x256 (5149), 512x512 (927), 416x416 (600), 183x275 (60), 167x162 (58), 101x142 (58)
- Width range 44-3074, height range 52-1846

**Healthy** (11926 images)

- Extensions: `.jpg` 6888, `.png` 2579, `.jpeg` 2459
- Decoded formats: JPEG 8938, PNG 2988
- Colour modes: RGB 11926
- Distinct dimensions: 2; most common: 256x256 (11410), 2048x1365 (516)
- Width range 256-2048, height range 256-1365

## 3. Provenance families (from filename conventions)

| Class | Source family | Images | Distinct family keys | Kinds |
|---|---|---:|---:|---|
| Alternaria | `AL9EE_alternaria` | 1008 | 417 | reexport 590, original 417, copy 1 |
| Alternaria | `AL9EE_greyspot` | 2 | 2 | reexport 2 |
| Alternaria | `CycleGAN_rec_B_alternaria` | 2322 | 476 | synthetic_augmented 1851, synthetic 471 |
| Alternaria | `Indigenous_LPU_alternaria` | 2549 | 2549 | original 2549 |
| Alternaria | `Numeric_unknown_alternaria` | 278 | 278 | original 278 |
| Alternaria | `Roboflow_malus` | 600 | 102 | reexport 600 |
| Alternaria | `Web_thumbnail` | 85 | 57 | original 56, copy 29 |
| Alternaria | `Web_thumbnail_augmented` | 3421 | 45 | augmented 3421 |
| Healthy | `AL9EE_field_healthy` | 516 | 516 | original 516 |
| Healthy | `CycleGAN_rec_A_healthy` | 2579 | 1546 | synthetic 1546, synthetic_augmented 1033 |
| Healthy | `Indigenous_LPU_healthy` | 2459 | 2459 | original 2459 |
| Healthy | `Phone_field_healthy` | 409 | 57 | original 409 |
| Healthy | `PlantVillage_healthy` | 5963 | 1645 | augmented 4318, original 1645 |

Image kinds: `original` = unmodified capture or earliest known version; `copy` = Windows '(n)' duplicate; `reexport` = Roboflow re-export of a known base image; `augmented` = flip/rotate/crop variant; `synthetic` = CycleGAN generator output (`_rec_A` / `_rec_B`); `synthetic_augmented` = Augmentor output of a synthetic image.

- Synthetic (CycleGAN) images: **4901** (22.1%). Augmented / re-exported images: **11815** (53.2%).
- `eval_eligible` (original, neither synthetic nor augmented, first occurrence of its byte content): **8325** images (Healthy 5022, Alternaria 3303).

## 4. Exact duplicates (SHA-256)

- Distinct file hashes: 20195 for 22191 images.
- Byte-identical groups: **1069**, redundant copies: **1996**.
  - Healthy: 113 groups, 117 redundant copies.
  - Alternaria: 956 groups, 1879 redundant copies.
- Byte-identical images present in BOTH classes: **0**.

## 5. Near-duplicate and provenance groups (union of exact + perceptual + filename families)

- Parameters: near-duplicate if, under some flip/rotation, pHash <= 6 AND dHash <= 5 (or pHash <= 2 alone); rec_B index alignment accepted at pHash <= 12; phone bursts within 10s. Calibration: true same-leaf pairs have dHash <= 5, different lab-photographed leaves that collide on pHash have dHash >= 6.
- Total groups: **7797** (5124 singletons, 2673 multi-image groups covering 17067 images).
- Pair links recorded: exact_sha256 1996, recB_index_alignment 332, near_dup_flip_rot 28411, near_dup 9781
- Near-duplicate link distances: pHash 0:24284, 2:8529, 4:3666, 6:1713 | dHash 0:19235, 1:9100, 2:4372, 3:2449, 4:1577, 5:1106, 6:167, 7:111, 8:31, 9:24, 10:8, 11:2, 12:3, 13:3, 14:1, 15:1, 16:1, 17:1
- Group size distribution: 1:5124, 2:437, 3:35, 4:637, 5:760, 6:282, 7:138, 8:100, 9:81, 10:51, 11:31, 12:20, 13:20, 14:11, 15:6, 16:4, 17:5, 18:3, 19:1, 20:2, 21:1, 22:1, 23:1, 29:1, 33:1 ...
- Largest groups: g3224 (291 imgs, web_tinyurl.com_3h4d86rn _middle_4_only); g3227 (232 imgs, web_tinyurl.com_hmc9up2d _2_4diiferent images); g632 (174 imgs, web_images-27); g512 (122 imgs, als_387); g614 (117 imgs, web_oip.kjidh458eo8mnnbjmd8dmwhafb)
- Groups spanning both classes: **0** (see `suspected_label_issues.csv`).

Which duplicates can be excluded safely: byte-identical copies (same SHA-256) carry no information and can be dropped at training time without loss; Roboflow re-exports and Windows copies that are byte-identical fall in that category. Flip/rotate/crop augmentations and CycleGAN outputs are NOT removed by this pipeline; they are kept in their group so the whole family lands in one split, and they are marked `eval_eligible = False` so evaluation can be reported on real, unmodified images.

## 6. Suspected label / provenance issues

- `foreign_disease_filename` (high): 2 rows, 2 groups
- `synthetic_alternaria_resembles_healthy` (medium): 271 rows, 75 groups
- `unverifiable_source_label` (medium): 600 rows, 101 groups

Key observations:

- The Alternaria folder holds CycleGAN outputs named `Health (N)_rec_B.jpg` plus Augmentor variants (`output_original_Health (N)_rec_B.png_<uuid>.jpg`). Visual inspection shows these are reconstructions of the AL9EE `Alternaria leaf spot (N)` leaves, with lesions partly removed, so their Alternaria label is weak. They are grouped with their AL9EE source image whenever the perceptual hash confirms it.
- The Healthy folder holds `Health (N)_rec_A.png` CycleGAN reconstructions (and Augmentor variants) of the `Health (N).jpg` field photos; 1,030 of them have no original in this dataset.
- `Web_thumbnail_augmented`: 45 tiny web images (about 140x128 px, ' - Copy' variants merged) each expanded into 57 flip/crop variants per file. Treated as 45 provenance families.
- PlantVillage healthy leaves appear as original + `_0/_1/_2/_flipTB` rotations/flips (up to 5 files per leaf). Treated as one group per UUID.
- Indigenous (`lpu_rdp_*`) files named `name (2).jpeg` are NOT copies: their perceptual hashes differ from the same-named file (distances 20-40 bits, i.e. unrelated images), so they are kept as separate originals and linked only when hashes agree.
- Four CycleGAN weight files (`latest_net_{G,D}_{A,B}.pth`) sit inside the Alternaria folder; they are ignored and left untouched.

## 7. Outputs

- `results/retrained/dataset_manifest.csv` - one row per image (split column filled by `prepare_clean_splits.py`).
- `results/retrained/duplicate_groups.csv` - reviewable list of all multi-image groups with recommendations.
- `results/retrained/suspected_label_issues.csv` - cross-class groups and other label risks.

## 8. Dataset decisions, split assignment and validation

Generated: 2026-10-16 23:06 by `code/prepare_clean_splits.py` (seed=42, ratios=[0.7, 0.15, 0.15], synthetic_to_train=True, exclusions=on)

### 8.1 Decisions applied

1. **Synthetic images**: every group containing a CycleGAN image (`_rec_A`, `_rec_B`, their Augmentor variants) is forced into TRAIN together with its real source image(s) and all linked re-exports. Groups forced: **894**. No synthetic image is in validation or test.
2. **Grey spot files**: excluded from all splits (filename names a different disease; no evidence of Alternaria). Files untouched on disk.
3. **Roboflow `malus` images**: excluded from the primary experiment (label unverifiable). Files untouched on disk.
4. **Ambiguous `rec_B` images** (271 flagged as perceptually close to a Healthy image): all are synthetic, so rule 1 places them and their groups in TRAIN only.
5. **Partially excluded groups**: 0 (remaining real members of such groups are assigned normally).

### 8.2 Excluded (quarantined) images

| Reason | Images | Groups | Class |
|---|---:|---:|---|
| Roboflow 'malus' export; Alternaria label cannot be verified from existing evidence | 600 | 101 | Alternaria |
| filename identifies a different disease ('Grey spot'); no evidence it is Alternaria | 2 | 2 | Alternaria |

- `DATASET/Alternaria leaf spot/Grey spot (232)_jpg.rf.787VYZMvOTt55fRpchml.jpg` (Grey spot)
- `DATASET/Alternaria leaf spot/Grey spot (261)_jpg.rf.y01vknUbIaDcKgl4K2jm.jpg` (Grey spot)
- plus 600 `malus_*_jpg.rf.*.jpg` files; full list in `excluded_manifest.csv`.

### 8.3 Split summary

Group-aware greedy stratified fill: every duplicate/provenance group lands in exactly one split; validation and test target 15 % of all included images per class; train takes the forced synthetic groups plus the remainder.

| Split | Class | Images | % of included class | % of split | Target | Groups | Eval-eligible images | Eval-eligible groups | Synthetic | Augmented |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | Healthy | 8348 | 70.0 | 55.24 | 8348.2 | 3197 | 3289 | 3194 | 2579 | 3513 |
| train | Alternaria | 6763 | 69.99 | 44.76 | 6764.1 | 2315 | 2457 | 2311 | 2322 | 3810 |
| train | ALL | 15111 | 69.99 | 100.0 | 15112.3 | 5512 | 5746 | 5505 | 4901 | 7323 |
| val | Healthy | 1789 | 15.0 | 55.23 | 1788.9 | 686 | 858 | 686 | 0 | 928 |
| val | Alternaria | 1450 | 15.01 | 44.77 | 1449.5 | 405 | 422 | 405 | 0 | 1026 |
| val | ALL | 3239 | 15.0 | 100.0 | 3238.3 | 1091 | 1280 | 1091 | 0 | 1954 |
| test | Healthy | 1789 | 15.0 | 55.23 | 1788.9 | 685 | 875 | 685 | 0 | 910 |
| test | Alternaria | 1450 | 15.01 | 44.77 | 1449.5 | 406 | 424 | 406 | 0 | 1026 |
| test | ALL | 3239 | 15.0 | 100.0 | 3238.3 | 1091 | 1299 | 1091 | 0 | 1936 |
| excluded | Healthy | 0 |  | 0.0 |  | 0 | 0 | 0 | 0 | 0 |
| excluded | Alternaria | 602 |  | 100.0 |  | 103 | 0 | 0 | 0 | 602 |
| excluded | ALL | 602 |  | 100.0 |  | 103 | 0 | 0 | 0 | 602 |

### 8.4 Validation checks

- [x] every manifest row is train/val/test/excluded (other=0)
- [x] train+val+test+excluded rows == manifest rows (21589+602 vs 22191)
- [x] train+val+test+excluded paths == manifest paths
- [x] no image path appears in more than one split
- [x] no image_id appears in more than one split
- [x] no excluded image appears in any split (excluded=602)
- [x] every excluded image has a recorded reason
- [x] train: both classes present ({'Healthy': 8348, 'Alternaria': 6763})
- [x] val: both classes present ({'Healthy': 1789, 'Alternaria': 1450})
- [x] test: both classes present ({'Healthy': 1789, 'Alternaria': 1450})
- [x] val: contains no synthetic (CycleGAN) images (synthetic=0)
- [x] val: real eval-eligible images from both classes ({'Healthy': 858, 'Alternaria': 422})
- [x] test: contains no synthetic (CycleGAN) images (synthetic=0)
- [x] test: real eval-eligible images from both classes ({'Healthy': 875, 'Alternaria': 424})
- [x] no exact-duplicate hash (sha256) crosses splits (violations=0)
- [x] no duplicate/provenance group crosses splits (violations=0)
- [x] no filename provenance family crosses splits (violations=0)
- [x] every group containing a CycleGAN image lies entirely in train (violations=0)
- [x] label matches class directory mapping
- [x] class_index matches label (Healthy=0, Alternaria=1)
- [x] relative_path directory agrees with class_dir
- [x] relative paths all under DATASET/
- [x] no non-image files included (bad=0)
- [x] per-class counts (splits + excluded) reconcile with supported images on disk (disk={'Alternaria': 10265, 'Healthy': 11926} manifest={'Healthy': 11926, 'Alternaria': 10265})
- [x] every split image exists on disk (missing=0)

### 8.5 Remaining label concerns (not excluded)

- CycleGAN `rec_B` images carry a weak Alternaria label; they are in TRAIN only. Consider a training ablation without `is_synthetic` rows.
- 57 `rec_B` images whose index does not match their AL9EE source stay as separate synthetic groups in TRAIN.
- Web-thumbnail families (45 bases, 57 variants each) and PlantVillage rotations are legitimate but heavily redundant; they are grouped, not removed.
