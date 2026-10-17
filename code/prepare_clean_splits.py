"""
Group-Aware, Stratified, Deterministic Train / Validation / Test Splitting
==========================================================================
Project: Apple Leaf Alternaria Detection: An AI-Based Approach Using Deep Learning and Explainable AI

Input : results/retrained/dataset_manifest.csv   (built by code/03_perceptual_hash_audit.py)
Output: results/retrained/dataset_manifest.csv   (split column filled: train / val / test / excluded)
        results/retrained/train_manifest.csv
        results/retrained/val_manifest.csv
        results/retrained/test_manifest.csv
        results/retrained/excluded_manifest.csv  (quarantine: images kept out of every split, with reason)
        results/retrained/split_summary.csv
        results/retrained/suspected_label_issues.csv  (a `decision` column is added)
        results/retrained/dataset_audit_report.md     (section 8 replaced)

Rules
-----
* A duplicate / provenance group (group_id) is assigned to exactly ONE split, so exact
  duplicates, re-exports, augmentations, CycleGAN derivatives and same-leaf captures never
  cross between train, validation and test.
* Exclusions (quarantine) are applied per image BEFORE assignment and recorded with a reason.
  Default exclusions (project decision of 2026-10-16):
      - `AL9EE_greyspot`  : filename names a different disease ("Grey spot").
      - `Roboflow_malus`  : disease label cannot be verified from any existing evidence.
  Remaining members of a partially-excluded group are still assigned normally (documented).
* `--synthetic-to-train`: every group that contains a CycleGAN image (and therefore also its
  real source image and all reconstructions / augmentations of it) is placed in TRAIN. No
  synthetic image can then appear in validation or test.
* Stratification is done at group level with a greedy largest-first fill that tracks the
  per-class deficit of every split. Validation and test targets are 15 % (default) of all
  INCLUDED images of each class; train receives the forced groups plus the remainder.
* Fully deterministic for a given --seed (default 42).
* Nothing is written to DATASET/; it is only read for existence / decode checks.

Usage
-----
    python code/prepare_clean_splits.py --synthetic-to-train      # project default
    python code/prepare_clean_splits.py --no-exclusions            # keep everything in play
    python code/prepare_clean_splits.py --validate-only            # re-run the checks only
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "DATASET")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "retrained")
MANIFEST = os.path.join(RESULTS_DIR, "dataset_manifest.csv")
REPORT = os.path.join(RESULTS_DIR, "dataset_audit_report.md")
ISSUES = os.path.join(RESULTS_DIR, "suspected_label_issues.csv")

CLASS_DIRS = {"Alternaria leaf spot": ("Alternaria", 1), "Healthy": ("Healthy", 0)}
CLASS_TO_IDX = {"Healthy": 0, "Alternaria": 1}
CLASSES = ["Healthy", "Alternaria"]
SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# source_family -> exclusion reason (applied unless --no-exclusions)
DEFAULT_EXCLUSIONS = {
    "AL9EE_greyspot": "filename identifies a different disease ('Grey spot'); no evidence it is Alternaria",
    "Roboflow_malus": "Roboflow 'malus' export; Alternaria label cannot be verified from existing evidence",
}


# ---------------------------------------------------------------------------
# Exclusions
# ---------------------------------------------------------------------------
def apply_exclusions(manifest, enabled=True):
    manifest = manifest.copy()
    manifest["exclusion_reason"] = ""
    if enabled:
        for fam, reason in DEFAULT_EXCLUSIONS.items():
            manifest.loc[manifest["source_family"] == fam, "exclusion_reason"] = reason
    manifest["excluded"] = manifest["exclusion_reason"] != ""
    # groups that are only partially excluded (remaining members are still assigned normally)
    g = manifest.groupby("group_id")["excluded"].agg(["any", "all"])
    partial = g[g["any"] & ~g["all"]].index.tolist()
    return manifest, partial


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------
def assign_splits(manifest, ratios, seed, synthetic_to_train):
    rng = np.random.default_rng(seed)
    inc = manifest[~manifest["excluded"]]
    g = inc.groupby("group_id")
    groups = pd.DataFrame({
        "size": g.size(),
        "n_healthy": g["label"].apply(lambda s: int((s == "Healthy").sum())),
        "n_alternaria": g["label"].apply(lambda s: int((s == "Alternaria").sum())),
        "has_synthetic": g["is_synthetic"].any(),
    })
    groups["rand"] = rng.random(len(groups))
    groups = groups.sort_values(["size", "rand"], ascending=[False, True])  # largest first, deterministic

    totals = {c: int((inc["label"] == c).sum()) for c in CLASSES}
    target = {"val": {c: ratios[1] * totals[c] for c in CLASSES},
              "test": {c: ratios[2] * totals[c] for c in CLASSES}}
    target["train"] = {c: totals[c] - target["val"][c] - target["test"][c] for c in CLASSES}
    current = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    assignment, forced = {}, 0

    # pass 1: forced groups
    if synthetic_to_train:
        for gid, row in groups[groups["has_synthetic"]].iterrows():
            assignment[gid] = "train"
            forced += 1
            current["train"]["Healthy"] += row["n_healthy"]
            current["train"]["Alternaria"] += row["n_alternaria"]

    # pass 2: greedy deficit fill for the rest
    for gid, row in groups.iterrows():
        if gid in assignment:
            continue
        n = {"Healthy": row["n_healthy"], "Alternaria": row["n_alternaria"]}
        best, choice = None, None
        for s in SPLITS:
            score = sum(n[c] * (target[s][c] - current[s][c]) / max(target[s][c], 1e-9) for c in CLASSES)
            key = (round(score, 9), sum(target[s][c] - current[s][c] for c in CLASSES))
            if best is None or key > best:
                best, choice = key, s
        assignment[gid] = choice
        for c in CLASSES:
            current[choice][c] += n[c]

    manifest = manifest.copy()
    manifest["split"] = manifest["group_id"].map(assignment)
    manifest.loc[manifest["excluded"], "split"] = "excluded"
    return manifest, target, forced


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def write_outputs(manifest, target):
    order = {"train": 0, "val": 1, "test": 2, "excluded": 3}
    manifest = manifest.sort_values(["split", "class_index", "relative_path"],
                                    key=lambda s: s.map(order) if s.name == "split" else s)
    manifest.to_csv(MANIFEST, index=False)
    for s in SPLITS:
        manifest[manifest["split"] == s].drop(columns=["excluded", "exclusion_reason"]).to_csv(
            os.path.join(RESULTS_DIR, f"{s}_manifest.csv"), index=False)
    manifest[manifest["split"] == "excluded"].to_csv(os.path.join(RESULTS_DIR, "excluded_manifest.csv"), index=False)

    rows = []
    included = manifest[manifest["split"] != "excluded"]
    for s in SPLITS + ["excluded"]:
        sub = manifest[manifest["split"] == s]
        for c in CLASSES + ["ALL"]:
            sc = sub if c == "ALL" else sub[sub["label"] == c]
            tot_c = len(included) if c == "ALL" else int((included["label"] == c).sum())
            ee = sc[sc["eval_eligible"]]
            rows.append({
                "split": s, "label": c, "class_index": "" if c == "ALL" else CLASS_TO_IDX[c],
                "images": len(sc),
                "pct_of_included_class": round(len(sc) / max(tot_c, 1) * 100, 2) if s != "excluded" else "",
                "pct_of_split": round(len(sc) / max(len(sub), 1) * 100, 2),
                "target_images": (round(target[s][c], 1) if c != "ALL" else round(sum(target[s].values()), 1)) if s in target else "",
                "groups": sc["group_id"].nunique(),
                "eval_eligible_images": len(ee),
                "eval_eligible_groups": ee["group_id"].nunique(),
                "synthetic_images": int(sc["is_synthetic"].sum()),
                "augmented_images": int(sc["is_augmented"].sum()),
                "exact_duplicate_images": int(sc["is_exact_duplicate"].sum()),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS_DIR, "split_summary.csv"), index=False)

    # record decisions next to the label issues
    if os.path.exists(ISSUES):
        issues = pd.read_csv(ISSUES)
        split_of = manifest.set_index("image_id")["split"]
        reason_of = manifest.set_index("image_id")["exclusion_reason"]
        issues["decision"] = issues["image_id"].map(
            lambda i: f"excluded: {reason_of.get(i, '')}" if split_of.get(i) == "excluded"
            else f"assigned to {split_of.get(i)} (group-level; synthetic groups forced to train)")
        issues.to_csv(ISSUES, index=False)
    return summary


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(decode_images=True):
    results = []

    def check(name, ok, detail=""):
        results.append({"check": name, "passed": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))

    print("\nValidating splits...")
    full = pd.read_csv(MANIFEST, dtype={"split": str})
    parts = {s: pd.read_csv(os.path.join(RESULTS_DIR, f"{s}_manifest.csv")) for s in SPLITS}
    excluded = pd.read_csv(os.path.join(RESULTS_DIR, "excluded_manifest.csv"))
    union = pd.concat(parts.values(), ignore_index=True)

    check("every manifest row is train/val/test/excluded", full["split"].isin(SPLITS + ["excluded"]).all(),
          f"other={int((~full['split'].isin(SPLITS + ['excluded'])).sum())}")
    check("train+val+test+excluded rows == manifest rows", len(union) + len(excluded) == len(full),
          f"{len(union)}+{len(excluded)} vs {len(full)}")
    check("train+val+test+excluded paths == manifest paths",
          set(union["relative_path"]) | set(excluded["relative_path"]) == set(full["relative_path"]))
    check("no image path appears in more than one split", union["relative_path"].duplicated().sum() == 0)
    check("no image_id appears in more than one split", union["image_id"].duplicated().sum() == 0)
    check("no excluded image appears in any split", not set(excluded["relative_path"]) & set(union["relative_path"]),
          f"excluded={len(excluded)}")
    check("every excluded image has a recorded reason", (excluded["exclusion_reason"].fillna("") != "").all() if len(excluded) else True)
    for s in SPLITS:
        counts = parts[s]["label"].value_counts().to_dict()
        check(f"{s}: both classes present", all(counts.get(c, 0) > 0 for c in CLASSES), str(counts))
    for s in ("val", "test"):
        syn = int(parts[s]["is_synthetic"].sum())
        check(f"{s}: contains no synthetic (CycleGAN) images", syn == 0, f"synthetic={syn}")
        ee = parts[s][parts[s]["eval_eligible"]]["label"].value_counts().to_dict()
        check(f"{s}: real eval-eligible images from both classes", all(ee.get(c, 0) > 0 for c in CLASSES), str(ee))
    sha_splits = union.groupby("sha256")["split"].nunique()
    check("no exact-duplicate hash (sha256) crosses splits", int((sha_splits > 1).sum()) == 0, f"violations={int((sha_splits > 1).sum())}")
    grp_splits = union.groupby("group_id")["split"].nunique()
    check("no duplicate/provenance group crosses splits", int((grp_splits > 1).sum()) == 0, f"violations={int((grp_splits > 1).sum())}")
    fam_splits = union.groupby("family_key")["split"].nunique()
    check("no filename provenance family crosses splits", int((fam_splits > 1).sum()) == 0, f"violations={int((fam_splits > 1).sum())}")
    # synthetic groups wholly in train (when any synthetic image exists in the splits)
    syn_groups = set(union[union["is_synthetic"]]["group_id"])
    leak = union[union["group_id"].isin(syn_groups) & (union["split"] != "train")]
    check("every group containing a CycleGAN image lies entirely in train", len(leak) == 0, f"violations={len(leak)}")
    expected_label = union["class_dir"].map({k: v[0] for k, v in CLASS_DIRS.items()})
    expected_idx = union["class_dir"].map({k: v[1] for k, v in CLASS_DIRS.items()})
    path_dir = union["relative_path"].str.split("/").str[1]
    check("label matches class directory mapping", (union["label"] == expected_label).all())
    check("class_index matches label (Healthy=0, Alternaria=1)",
          (union["class_index"] == expected_idx).all() and (union["label"].map(CLASS_TO_IDX) == union["class_index"]).all())
    check("relative_path directory agrees with class_dir", (path_dir == union["class_dir"]).all())
    check("relative paths all under DATASET/", union["relative_path"].str.startswith("DATASET/").all())
    exts = union["relative_path"].apply(lambda p: os.path.splitext(p)[1].lower())
    check("no non-image files included", exts.isin(IMAGE_EXTENSIONS).all(), f"bad={int((~exts.isin(IMAGE_EXTENSIONS)).sum())}")
    disk = {}
    for class_dir, (label, _) in CLASS_DIRS.items():
        d = os.path.join(DATASET_DIR, class_dir)
        disk[label] = sum(1 for n in os.listdir(d) if os.path.isfile(os.path.join(d, n)) and os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS)
    man = pd.concat([union, excluded])["label"].value_counts().to_dict()
    check("per-class counts (splits + excluded) reconcile with supported images on disk",
          all(disk[c] == man.get(c, 0) for c in CLASSES), f"disk={disk} manifest={man}")
    missing, unreadable = [], []
    if decode_images:
        print(f"  decoding {len(union)} split images for existence/readability ...", flush=True)
    for i, p in enumerate(union["relative_path"]):
        ap = os.path.join(PROJECT_ROOT, p)
        if not os.path.isfile(ap):
            missing.append(p); continue
        if decode_images:
            try:
                with Image.open(ap) as im:
                    im.verify()
                with Image.open(ap) as im:
                    im.load()
            except Exception as exc:
                unreadable.append(f"{p}: {exc}")
            if (i + 1) % 5000 == 0:
                print(f"    {i + 1}/{len(union)}", flush=True)
    check("every split image exists on disk", len(missing) == 0, f"missing={len(missing)}")
    if decode_images:
        check("every split image decodes with Pillow", len(unreadable) == 0, f"unreadable={len(unreadable)}")

    all_ok = all(r["passed"] for r in results)
    print(f"\nVALIDATION {'PASSED' if all_ok else 'FAILED'} ({sum(r['passed'] for r in results)}/{len(results)} checks)")
    return all_ok, results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report_section(summary, results, args, forced, partial_groups, manifest):
    excluded = manifest[manifest["split"] == "excluded"]
    L = ["", "## 8. Dataset decisions, split assignment and validation", "",
         f"Generated: {datetime.now():%Y-%m-%d %H:%M} by `code/prepare_clean_splits.py` "
         f"(seed={args.seed}, ratios={args.ratios}, synthetic_to_train={args.synthetic_to_train}, exclusions={'off' if args.no_exclusions else 'on'})", "",
         "### 8.1 Decisions applied", ""]
    L.append("1. **Synthetic images**: every group containing a CycleGAN image (`_rec_A`, `_rec_B`, their Augmentor variants) is forced into TRAIN "
             f"together with its real source image(s) and all linked re-exports. Groups forced: **{forced}**. No synthetic image is in validation or test.")
    L.append("2. **Grey spot files**: excluded from all splits (filename names a different disease; no evidence of Alternaria). Files untouched on disk.")
    L.append("3. **Roboflow `malus` images**: excluded from the primary experiment (label unverifiable). Files untouched on disk.")
    L.append("4. **Ambiguous `rec_B` images** (271 flagged as perceptually close to a Healthy image): all are synthetic, so rule 1 places them and their groups in TRAIN only.")
    L.append(f"5. **Partially excluded groups**: {len(partial_groups)} (remaining real members of such groups are assigned normally).")
    L.append("")
    L += ["### 8.2 Excluded (quarantined) images", ""]
    L.append("| Reason | Images | Groups | Class |")
    L.append("|---|---:|---:|---|")
    for (reason, label), sub in excluded.groupby(["exclusion_reason", "label"]):
        L.append(f"| {reason} | {len(sub)} | {sub['group_id'].nunique()} | {label} |")
    L.append("")
    if len(excluded) <= 10:
        for p in excluded["relative_path"]:
            L.append(f"- `{p}`")
    else:
        for p in excluded[excluded["source_family"] == "AL9EE_greyspot"]["relative_path"]:
            L.append(f"- `{p}` (Grey spot)")
        L.append(f"- plus {int((excluded['source_family'] == 'Roboflow_malus').sum())} `malus_*_jpg.rf.*.jpg` files; full list in `excluded_manifest.csv`.")
    L.append("")
    L += ["### 8.3 Split summary", ""]
    L.append("Group-aware greedy stratified fill: every duplicate/provenance group lands in exactly one split; validation and test target "
             "15 % of all included images per class; train takes the forced synthetic groups plus the remainder.")
    L.append("")
    L.append("| Split | Class | Images | % of included class | % of split | Target | Groups | Eval-eligible images | Eval-eligible groups | Synthetic | Augmented |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary.to_dict("records"):
        L.append(f"| {r['split']} | {r['label']} | {r['images']} | {r['pct_of_included_class']} | {r['pct_of_split']} | {r['target_images']} | "
                 f"{r['groups']} | {r['eval_eligible_images']} | {r['eval_eligible_groups']} | {r['synthetic_images']} | {r['augmented_images']} |")
    L += ["", "### 8.4 Validation checks", ""]
    for r in results:
        L.append(f"- [{'x' if r['passed'] else ' '}] {r['check']}" + (f" ({r['detail']})" if r["detail"] else ""))
    L.append("")
    L += ["### 8.5 Remaining label concerns (not excluded)", ""]
    L.append("- CycleGAN `rec_B` images carry a weak Alternaria label; they are in TRAIN only. Consider a training ablation without `is_synthetic` rows.")
    L.append("- 57 `rec_B` images whose index does not match their AL9EE source stay as separate synthetic groups in TRAIN.")
    L.append("- Web-thumbnail families (45 bases, 57 variants each) and PlantVillage rotations are legitimate but heavily redundant; they are grouped, not removed.")
    L.append("")

    existing = ""
    if os.path.exists(REPORT):
        with open(REPORT, encoding="utf-8") as f:
            existing = f.read()
    for marker in ("\n## 8. Dataset decisions, split assignment and validation", "\n## 8. Split assignment and validation"):
        if marker in existing:
            existing = existing[:existing.index(marker)].rstrip("\n") + "\n"
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(existing + "\n".join(L))


def main():
    ap = argparse.ArgumentParser(description="Group-aware stratified split of the audited dataset manifest.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--synthetic-to-train", action="store_true", help="force every group that contains a CycleGAN image into the training split")
    ap.add_argument("--no-exclusions", action="store_true", help="do not quarantine Grey spot / Roboflow malus images")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--no-decode", action="store_true", help="skip the Pillow decode of every image during validation")
    args = ap.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit("ratios must sum to 1")
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"Manifest not found: {MANIFEST}. Run code/03_perceptual_hash_audit.py first.")

    if not args.validate_only:
        manifest = pd.read_csv(MANIFEST)
        manifest = manifest.drop(columns=[c for c in ("excluded", "exclusion_reason") if c in manifest.columns])
        manifest["split"] = ""
        print(f"Loaded manifest: {len(manifest)} images, {manifest['group_id'].nunique()} groups")
        manifest, partial = apply_exclusions(manifest, enabled=not args.no_exclusions)
        print(f"Excluded images: {int(manifest['excluded'].sum())} | partially excluded groups: {len(partial)}")
        manifest, target, forced = assign_splits(manifest, args.ratios, args.seed, args.synthetic_to_train)
        print(f"Groups forced to train (contain synthetic images): {forced}")
        summary = write_outputs(manifest, target)
        print("\nSplit summary:")
        print(summary.to_string(index=False))

    ok, results = validate(decode_images=not args.no_decode)
    if not args.validate_only:
        write_report_section(summary, results, args, forced, partial, manifest)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
