"""
Dataset Audit, Duplicate Grouping & Manifest Builder (read-only on DATASET/)
===========================================================================
Project: Apple Leaf Alternaria Detection: An AI-Based Approach Using Deep Learning and Explainable AI

Scans the two source folders

    DATASET/Alternaria leaf spot/   ->  canonical label "Alternaria" (index 1)
    DATASET/Healthy/                ->  canonical label "Healthy"    (index 0)

and produces, under results/retrained/:

    dataset_manifest.csv        one row per supported image (split column empty until
                                prepare_clean_splits.py assigns it)
    duplicate_groups.csv        every image that belongs to a multi-image duplicate /
                                provenance group, with the reason and an exclusion recommendation
    suspected_label_issues.csv  cross-class groups, foreign-disease filenames, synthetic images
                                that are perceptually closer to the other class, etc.
    dataset_audit_report.md     human-readable audit

Nothing inside DATASET/ is ever written, moved, renamed or deleted. Non-image files
(e.g. CycleGAN *.pth weights) are listed in the report and ignored.

Duplicate / relatedness detection (all three are combined with union-find):
    1. Exact duplicates      identical SHA-256 of the file bytes.
    2. Near duplicates       for some dihedral transform (identity, flips, 90-degree rotations) of the
                             other image BOTH 64-bit pHash (DCT) distance <= PHASH_T AND 64-bit dHash
                             distance <= DHASH_T; or pHash distance <= PHASH_T_ALONE on its own.
                             Thresholds were calibrated on this dataset: true same-leaf pairs
                             (CycleGAN reconstruction vs. its source) have dHash <= 5, while different
                             lab-photographed leaves that collide on pHash have dHash >= 6.
    3. Provenance families   filename conventions that identify the same physical leaf / source
                             image (PlantVillage UUID, Roboflow "<base>_jpg.rf.<hash>", Augmentor
                             "output_original_<name>_<uuid>", CycleGAN "<name>_rec_A/_rec_B",
                             "<base>._<k>" augmentation runs, 10-second phone-camera bursts, and the
                             index alignment "Health (N)_rec_B" <-> "Alternaria leaf spot (N)" when the
                             perceptual hash confirms it). Indigenous "name (2).jpeg" files are NOT
                             treated as copies: their content differs from the same-named file.

Usage:
    python code/03_perceptual_hash_audit.py                 # full run
    python code/03_perceptual_hash_audit.py --hash-cache C  # reuse / write a hash cache CSV
"""

import os
import re
import sys
import csv
import json
import time
import hashlib
import argparse
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dctn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "DATASET")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "retrained")

# Explicit directory -> canonical label mapping. Never derived from alphabetical order.
CLASS_DIRS = {
    "Alternaria leaf spot": ("Alternaria", 1),
    "Healthy": ("Healthy", 0),
}
CLASS_TO_IDX = {"Healthy": 0, "Alternaria": 1}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

PHASH_T = 6        # pHash Hamming distance (of 64 bits) for a near-duplicate ...
DHASH_T = 5        # ... which must be corroborated by a dHash distance this small (same transform)
PHASH_T_ALONE = 2  # a pHash distance this small is accepted without dHash corroboration
RECB_INDEX_T = 12  # pHash distance to accept "Health (N)_rec_B" as derived from "Alternaria leaf spot (N)"
BURST_SECONDS = 10 # phone photos taken within this window are grouped as one capture session

MANIFEST_COLUMNS = [
    "image_id", "relative_path", "class_dir", "label", "class_index", "original_filename",
    "extension", "format", "mode", "width", "height", "file_size_bytes",
    "sha256", "phash", "dhash", "group_id", "group_size", "group_classes",
    "source_family", "family_key", "image_kind", "is_synthetic", "is_augmented",
    "is_exact_duplicate", "eval_eligible", "split",
]


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def sha256_of_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _bits_to_u64(bits):
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    return int(np.packbits(bits).view(">u8")[0])


def phash_u64(gray32):
    """pHash on a 32x32 float grayscale array (same recipe as the imagehash library)."""
    d = dctn(gray32, norm=None)[:8, :8]
    med = np.median(d)
    return _bits_to_u64(d > med)


def dhash_u64(img_gray):
    small = np.asarray(img_gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    return _bits_to_u64(small[:, 1:] > small[:, :-1])


def dhash_from_array(gray_arr):
    small = np.asarray(Image.fromarray(gray_arr).resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    return _bits_to_u64(small[:, 1:] > small[:, :-1])


def dihedral_variants(arr):
    """The 8 flips/rotations of a square array."""
    out = []
    for k in range(4):
        r = np.rot90(arr, k)
        out.append(r)
        out.append(np.fliplr(r))
    return out


def compute_image_record(path):
    """Opens one image, verifies it, and returns (meta, phash_variants[8], dhash_variants[8]) or raises.
    Variant 0 is the untransformed image; the dHash of variant 0 is computed from the full-resolution
    grayscale image, the other 7 from a 64x64 intermediate (sufficient for a 9x8 gradient hash)."""
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        fmt, mode, size = im.format, im.mode, im.size
        im = ImageOps.exif_transpose(im)
        gray = im.convert("L")
        g32 = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
        g64 = np.asarray(gray.resize((64, 64), Image.Resampling.LANCZOS), dtype=np.uint8)
        ph = [phash_u64(v) for v in dihedral_variants(g32)]
        dh = [dhash_from_array(np.ascontiguousarray(v)) for v in dihedral_variants(g64)]
        dh[0] = dhash_u64(gray)
    return {"format": fmt, "mode": mode, "width": size[0], "height": size[1]}, ph, dh


def hamming_min_block(q, variants):
    """q: (n,) uint64 canonical hashes of a query block; variants: (8, N) uint64.
    Returns (n, N) uint8 minimum Hamming distance over the 8 transforms."""
    best = None
    for v in range(variants.shape[0]):
        d = np.bitwise_count(np.bitwise_xor(q[:, None], variants[v][None, :])).astype(np.uint8)
        best = d if best is None else np.minimum(best, d)
    return best


def near_dup_block(qp, qd, pvar, dvar):
    """Joint pHash/dHash test for one query block against all images under all 8 transforms.
    qp, qd: (n,) uint64 canonical hashes; pvar, dvar: (8, N) uint64 variant hashes.
    Returns (match (n,N) bool, best_p (n,N) uint8, best_d (n,N) uint8, best_v (n,N) int8)."""
    match = None
    for v in range(pvar.shape[0]):
        dp = np.bitwise_count(np.bitwise_xor(qp[:, None], pvar[v][None, :])).astype(np.uint8)
        dd = np.bitwise_count(np.bitwise_xor(qd[:, None], dvar[v][None, :])).astype(np.uint8)
        ok = ((dp <= PHASH_T) & (dd <= DHASH_T)) | (dp <= PHASH_T_ALONE)
        if match is None:
            match, best_p, best_d, best_v = ok, dp, dd, np.zeros(dp.shape, dtype=np.int8)
        else:
            better = ok & (~match | (dp < best_p))
            best_p = np.where(better, dp, best_p); best_d = np.where(better, dd, best_d); best_v = np.where(better, v, best_v)
            match |= ok
    return match, best_p, best_d, best_v


# ---------------------------------------------------------------------------
# Provenance parsing (filename conventions observed in this dataset)
# ---------------------------------------------------------------------------
RX = {
    "pv": re.compile(r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})___RS_HL (\d+)(.*)\.(jpg|jpeg|png)$", re.I),
    "lpu_alt": re.compile(r"^lpu_rdp_alt(\d+)\s*(\(\d+\))*(\s*\(\d+\))*\.jpe?g$", re.I),
    "lpu_h": re.compile(r"^lpu_rdp_h(\d+)\s*(\(\d+\))*\.jpe?g$", re.I),
    "health_orig": re.compile(r"^Health \((\d+)\)\.jpg$"),
    "health_recA": re.compile(r"^Health \((\d+)\)_rec_A\.png$"),
    "health_recA_aug": re.compile(r"^output_original_Health \((\d+)\)_rec_A\.png_[0-9a-f-]{36}\.png$"),
    "recB": re.compile(r"^Health \((\d+)\)_rec_B\.jpg$"),
    "recB_aug": re.compile(r"^output_original_Health \((\d+)\)_rec_B\.png_[0-9a-f-]{36}\.jpg$"),
    "als_orig": re.compile(r"^Alternaria leaf spot \((\d+)\)\.jpg$"),
    "als_copy": re.compile(r"^Alternaria leaf spot \((\d+)\)\(\d+\)\.jpg$"),
    "als_rf": re.compile(r"^Alternaria leaf spot \((\d+)\)_jpg\.rf\.[A-Za-z0-9]+(\(\d+\))?\.jpg$"),
    "grey_rf": re.compile(r"^Grey spot \((\d+)\)_jpg\.rf\.[A-Za-z0-9]+\.jpg$"),
    "malus_rf": re.compile(r"^malus_(\d+)_jpg\.rf\.[A-Za-z0-9]+\.jpg$"),
    "numeric": re.compile(r"^(\d{6})\.jpg$"),
    "phone": re.compile(r"^IMG_(\d{8})_(\d{6})\.jpg$"),
    # web-thumbnail augmentation runs: "<base>[ - Copy[ (k)]]._<n>.jpg" or "<base>[ - Copy[ (k)]]_<n>.jpg"
    "web_aug": re.compile(r"^(?P<base>.+?)(?P<copy> - Copy(?: \(\d+\))?)?\.?_(?P<aug>\d{1,2})\.jpg$"),
    "web_orig": re.compile(r"^(?P<base>.+?)(?P<copy> - Copy(?: \(\d+\))?)?\.(?P<ext>jpe?g)$", re.I),
}


def parse_provenance(class_dir, fname):
    """Returns dict(source_family, family_key, image_kind, is_synthetic, is_augmented, note)."""
    def R(source, key, kind, syn=False, aug=False, note=""):
        return {"source_family": source, "family_key": key, "image_kind": kind,
                "is_synthetic": syn, "is_augmented": aug, "note": note}

    if class_dir == "Healthy":
        m = RX["pv"].match(fname)
        if m:
            suffix = m.group(3)
            return R("PlantVillage_healthy", f"pv_{m.group(1)}", "original" if suffix == "" else "augmented",
                     aug=(suffix != ""), note=f"PlantVillage leaf {m.group(2)} variant '{suffix}'")
        m = RX["lpu_h"].match(fname)
        if m:
            if fname.lower() == f"lpu_rdp_h{m.group(1)}.jpeg":
                return R("Indigenous_LPU_healthy", f"lpu_h_{m.group(1)}", "original")
            return R("Indigenous_LPU_healthy", f"lpu_h_{m.group(1)}__{fname.lower()}", "original",
                     note="filename collision '(n)' with a different image; linked only if hashes match")
        m = RX["health_orig"].match(fname)
        if m:
            return R("AL9EE_field_healthy", f"health_{m.group(1)}", "original")
        m = RX["health_recA"].match(fname)
        if m:
            return R("CycleGAN_rec_A_healthy", f"health_{m.group(1)}", "synthetic", syn=True,
                     note="CycleGAN reconstruction (rec_A) of a healthy leaf")
        m = RX["health_recA_aug"].match(fname)
        if m:
            return R("CycleGAN_rec_A_healthy", f"health_{m.group(1)}", "synthetic_augmented", syn=True, aug=True,
                     note="Augmentor output of a CycleGAN rec_A image")
        m = RX["phone"].match(fname)
        if m:
            return R("Phone_field_healthy", f"phone_{m.group(1)}_{m.group(2)}", "original",
                     note="burst grouping applied later")
        return R("Unknown_healthy", f"file_{fname}", "original", note="unrecognised filename pattern")

    # Alternaria leaf spot directory
    m = RX["lpu_alt"].match(fname)
    if m:
        if fname.lower() == f"lpu_rdp_alt{m.group(1)}.jpeg":
            return R("Indigenous_LPU_alternaria", f"lpu_alt_{m.group(1)}", "original")
        return R("Indigenous_LPU_alternaria", f"lpu_alt_{m.group(1)}__{fname.lower()}", "original",
                 note="filename collision '(n)' with a different image; linked only if hashes match")
    m = RX["recB"].match(fname)
    if m:
        return R("CycleGAN_rec_B_alternaria", f"recB_{m.group(1)}", "synthetic", syn=True,
                 note="CycleGAN reconstruction (rec_B); index N aligns with 'Alternaria leaf spot (N)'")
    m = RX["recB_aug"].match(fname)
    if m:
        return R("CycleGAN_rec_B_alternaria", f"recB_{m.group(1)}", "synthetic_augmented", syn=True, aug=True,
                 note="Augmentor output of a CycleGAN rec_B image")
    m = RX["als_orig"].match(fname)
    if m:
        return R("AL9EE_alternaria", f"als_{m.group(1)}", "original")
    m = RX["als_copy"].match(fname)
    if m:
        return R("AL9EE_alternaria", f"als_{m.group(1)}", "copy", note="Windows '(n)' copy")
    m = RX["als_rf"].match(fname)
    if m:
        return R("AL9EE_alternaria", f"als_{m.group(1)}", "reexport", aug=True, note="Roboflow export of AL9EE image")
    m = RX["grey_rf"].match(fname)
    if m:
        return R("AL9EE_greyspot", f"grey_{m.group(1)}", "reexport", aug=True,
                 note="filename says 'Grey spot' (a different apple disease)")
    m = RX["malus_rf"].match(fname)
    if m:
        return R("Roboflow_malus", f"malus_{m.group(1)}", "reexport", aug=True,
                 note="Roboflow 'malus' export; disease label not verifiable from filename")
    m = RX["numeric"].match(fname)
    if m:
        return R("Numeric_unknown_alternaria", f"num_{m.group(1)}", "original", note="6-digit numeric filename, source unknown")
    m = RX["web_aug"].match(fname)
    if m:
        base = m.group("base").strip().lower()
        return R("Web_thumbnail_augmented", f"web_{base}", "augmented", aug=True,
                 note=f"augmentation run #{m.group('aug')}" + (" of a Windows copy" if m.group("copy") else ""))
    m = RX["web_orig"].match(fname)
    if m:
        base = m.group("base").strip().lower()
        kind = "copy" if m.group("copy") else "original"
        return R("Web_thumbnail", f"web_{base}", kind)
    return R("Unknown_alternaria", f"file_{fname}", "original", note="unrecognised filename pattern")


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.reasons = defaultdict(set)

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b, reason):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra
        self.reasons[a].add(reason)
        self.reasons[b].add(reason)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def scan_files():
    rows, non_images = [], []
    for class_dir, (label, idx) in CLASS_DIRS.items():
        d = os.path.join(DATASET_DIR, class_dir)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Class folder missing: {d}")
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isdir(p):
                non_images.append({"class_dir": class_dir, "filename": name, "reason": "sub-directory"})
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                non_images.append({"class_dir": class_dir, "filename": name, "reason": f"unsupported extension '{ext}'",
                                   "size_bytes": os.path.getsize(p)})
                continue
            rows.append({"class_dir": class_dir, "label": label, "class_index": idx, "original_filename": name,
                         "relative_path": os.path.join("DATASET", class_dir, name).replace("\\", "/"),
                         "extension": ext, "file_size_bytes": os.path.getsize(p)})
    return rows, non_images


def hash_all(rows, cache_path=None):
    cached = {}
    if cache_path and os.path.exists(cache_path):
        cdf = pd.read_csv(cache_path, dtype=str)
        for r in cdf.to_dict("records"):
            cached[r["relative_path"]] = r
        print(f"Loaded hash cache with {len(cached)} entries from {cache_path}")

    unreadable = []
    out = []
    t0 = time.time()
    for i, r in enumerate(rows):
        abs_path = os.path.join(PROJECT_ROOT, r["relative_path"])
        c = cached.get(r["relative_path"])
        if c and c.get("sha256") and c.get("dhash_variants") and str(c.get("file_size_bytes")) == str(r["file_size_bytes"]):
            rec = dict(r)
            rec.update({"format": c["format"], "mode": c["mode"], "width": int(c["width"]), "height": int(c["height"]),
                        "sha256": c["sha256"],
                        "phash_variants": [int(x, 16) for x in c["phash_variants"].split("|")],
                        "dhash_variants": [int(x, 16) for x in c["dhash_variants"].split("|")]})
            out.append(rec)
        else:
            rec = dict(r)
            try:
                if r["file_size_bytes"] == 0:
                    raise ValueError("zero-byte file")
                meta, ph, dh = compute_image_record(abs_path)
                rec.update(meta)
                rec["sha256"] = sha256_of_file(abs_path)
                rec["phash_variants"] = ph
                rec["dhash_variants"] = dh
                out.append(rec)
            except Exception as exc:
                unreadable.append({"relative_path": r["relative_path"], "error": f"{type(exc).__name__}: {exc}"})
        if (i + 1) % 2000 == 0:
            print(f"  hashed {i + 1}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)

    if cache_path:
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["relative_path", "file_size_bytes", "format", "mode", "width", "height", "sha256", "phash_variants", "dhash_variants"])
            for rec in out:
                w.writerow([rec["relative_path"], rec["file_size_bytes"], rec["format"], rec["mode"], rec["width"], rec["height"],
                            rec["sha256"], "|".join(f"{x:016x}" for x in rec["phash_variants"]), "|".join(f"{x:016x}" for x in rec["dhash_variants"])])
        print(f"Hash cache written to {cache_path}")
    return out, unreadable


def build_groups(recs):
    n = len(recs)
    dsu = DSU(n)
    pair_log = []  # (i, j, reason, dist)

    # 1. exact duplicates
    by_sha = defaultdict(list)
    for i, r in enumerate(recs):
        by_sha[r["sha256"]].append(i)
    for sha, idxs in by_sha.items():
        for j in idxs[1:]:
            dsu.union(idxs[0], j, "exact_sha256")
            pair_log.append((idxs[0], j, "exact_sha256", 0))

    # 2. provenance families
    for i, r in enumerate(recs):
        r.update(parse_provenance(r["class_dir"], r["original_filename"]))
    # phone bursts: consecutive shots within BURST_SECONDS share a family key
    phone = sorted([i for i, r in enumerate(recs) if r["source_family"] == "Phone_field_healthy"],
                   key=lambda i: recs[i]["original_filename"])
    burst_id, last_ts = 0, None
    for i in phone:
        m = RX["phone"].match(recs[i]["original_filename"])
        ts = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        if last_ts is None or (ts - last_ts).total_seconds() > BURST_SECONDS:
            burst_id += 1
        last_ts = ts
        recs[i]["family_key"] = f"phone_burst_{burst_id:04d}"
    by_family = defaultdict(list)
    for i, r in enumerate(recs):
        by_family[r["family_key"]].append(i)
    for key, idxs in by_family.items():
        for j in idxs[1:]:
            dsu.union(idxs[0], j, "provenance_family")

    # CycleGAN "Health (N)_rec_B" <-> AL9EE "Alternaria leaf spot (N)": index alignment confirmed by pHash
    als_by_n = {}
    for i, r in enumerate(recs):
        if r["source_family"] == "AL9EE_alternaria" and r["image_kind"] == "original":
            als_by_n[int(RX["als_orig"].match(r["original_filename"]).group(1))] = i
    recb_aligned = recb_unaligned = 0
    for i, r in enumerate(recs):
        if r["source_family"] == "CycleGAN_rec_B_alternaria" and r["image_kind"] == "synthetic":
            n_idx = int(RX["recB"].match(r["original_filename"]).group(1))
            j = als_by_n.get(n_idx)
            if j is None:
                continue
            dmin = min(bin(recs[i]["phash_variants"][0] ^ v).count("1") for v in recs[j]["phash_variants"])
            if dmin <= RECB_INDEX_T:
                dsu.union(j, i, "recB_index_alignment")
                pair_log.append((j, i, "recB_index_alignment", dmin))
                recb_aligned += 1
            else:
                recs[i]["note"] = (recs[i].get("note", "") + f"; index N={n_idx} does NOT match AL9EE image (pHash {dmin})").strip("; ")
                recb_unaligned += 1
    print(f"  rec_B index alignment: {recb_aligned} confirmed, {recb_unaligned} rejected by pHash")

    # 3. perceptual near-duplicates: joint pHash + dHash test under the 8 dihedral transforms
    canon_p = np.array([r["phash_variants"][0] for r in recs], dtype=np.uint64)
    canon_d = np.array([r["dhash_variants"][0] for r in recs], dtype=np.uint64)
    pvar = np.array([r["phash_variants"] for r in recs], dtype=np.uint64).T  # (8, n)
    dvar = np.array([r["dhash_variants"] for r in recs], dtype=np.uint64).T  # (8, n)
    BLOCK = 512
    near_pairs = 0
    t0 = time.time()
    for s in range(0, n, BLOCK):
        match, bp, bd, bv = near_dup_block(canon_p[s:s + BLOCK], canon_d[s:s + BLOCK], pvar, dvar)
        bi, bj = np.nonzero(match)
        for a, b in zip(bi, bj):
            i, j = s + int(a), int(b)
            if j <= i or recs[i]["sha256"] == recs[j]["sha256"]:
                continue
            reason = "near_dup" if bv[a, b] == 0 else "near_dup_flip_rot"
            dsu.union(i, j, reason)
            pair_log.append((i, j, reason, int(bp[a, b]) * 100 + int(bd[a, b])))  # dist encodes pHash*100 + dHash
            near_pairs += 1
        if (s // BLOCK) % 10 == 0:
            print(f"  near-dup scan {min(s + BLOCK, n)}/{n} ({time.time() - t0:.0f}s, pairs so far {near_pairs})", flush=True)

    roots = [dsu.find(i) for i in range(n)]
    # deterministic group ids: order by the smallest relative_path in each component
    members = defaultdict(list)
    for i, r in enumerate(roots):
        members[r].append(i)
    ordered = sorted(members.values(), key=lambda idxs: min(recs[i]["relative_path"] for i in idxs))
    gid_of = {}
    for g, idxs in enumerate(ordered, start=1):
        for i in idxs:
            gid_of[i] = g
    return gid_of, members, dsu, pair_log


def main():
    ap = argparse.ArgumentParser(description="Audit DATASET/, build manifest, duplicate groups and label-issue report.")
    ap.add_argument("--hash-cache", default=None, help="CSV path to reuse/write per-image hashes (speeds up re-runs)")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 78)
    print("   DATASET AUDIT & MANIFEST BUILD (read-only on DATASET/)")
    print("=" * 78)
    rows, non_images = scan_files()
    print(f"Supported image files found: {len(rows)} | non-image entries: {len(non_images)}")

    recs, unreadable = hash_all(rows, args.hash_cache)
    print(f"Hashed {len(recs)} images; unreadable: {len(unreadable)}")

    gid_of, members, dsu, pair_log = build_groups(recs)

    # ---- assemble manifest ----
    for i, r in enumerate(recs):
        r["group_id"] = gid_of[i]
    group_rows = defaultdict(list)
    for i, r in enumerate(recs):
        group_rows[r["group_id"]].append(i)
    sha_counts = Counter(r["sha256"] for r in recs)
    for gid, idxs in group_rows.items():
        classes = sorted({recs[i]["label"] for i in idxs})
        for i in idxs:
            recs[i]["group_size"] = len(idxs)
            recs[i]["group_classes"] = "+".join(classes)
    first_of_sha = {}
    for i, r in enumerate(recs):  # recs are in sorted relative_path order per class folder
        first_of_sha.setdefault(r["sha256"], i)
    for i, r in enumerate(recs):
        r["image_id"] = f"IMG{i + 1:06d}"
        r["phash"] = f"{r['phash_variants'][0]:016x}"
        r["dhash"] = f"{r['dhash_variants'][0]:016x}"
        r["is_exact_duplicate"] = sha_counts[r["sha256"]] > 1
        # real, unmodified image AND the first occurrence of its byte content
        r["eval_eligible"] = (not r["is_synthetic"]) and (not r["is_augmented"]) and r["image_kind"] in ("original", "copy") \
            and first_of_sha[r["sha256"]] == i
        r["split"] = ""
    manifest = pd.DataFrame(recs)[MANIFEST_COLUMNS]
    manifest_path = os.path.join(RESULTS_DIR, "dataset_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    # ---- duplicate_groups.csv ----
    dup_rows = []
    for gid, idxs in sorted(group_rows.items()):
        if len(idxs) < 2:
            continue
        idxs_sorted = sorted(idxs, key=lambda i: (0 if recs[i]["image_kind"] == "original" else 1, recs[i]["relative_path"]))
        rep = idxs_sorted[0]
        seen_sha = set()
        for i in idxs_sorted:
            r = recs[i]
            reasons = sorted(dsu.reasons.get(i, {"provenance_family"}))
            if r["sha256"] in seen_sha:
                rec_action = "safe_to_exclude: byte-identical to another file in the group"
            elif i == rep:
                rec_action = "keep: group representative"
            elif r["image_kind"] == "copy":
                rec_action = "safe_to_exclude: Windows copy (verify hash first)" if r["is_exact_duplicate"] else "keep: filename copy but bytes differ"
            elif r["is_synthetic"]:
                rec_action = "review: synthetic (CycleGAN) image; keep grouped; exclude from val/test metrics"
            elif r["is_augmented"]:
                rec_action = "keep grouped: augmentation / re-export of the representative; not eval-eligible"
            else:
                rec_action = "keep grouped: perceptual near-duplicate"
            seen_sha.add(r["sha256"])
            dup_rows.append({"group_id": gid, "group_size": len(idxs), "group_classes": r["group_classes"],
                             "image_id": r["image_id"], "label": r["label"], "relative_path": r["relative_path"],
                             "sha256": r["sha256"], "phash": r["phash"], "image_kind": r["image_kind"],
                             "source_family": r["source_family"], "family_key": r["family_key"],
                             "link_reasons": "|".join(reasons), "is_representative": i == rep, "recommendation": rec_action})
    dup_df = pd.DataFrame(dup_rows)
    dup_df.to_csv(os.path.join(RESULTS_DIR, "duplicate_groups.csv"), index=False)

    # ---- suspected_label_issues.csv ----
    issues = []
    for gid, idxs in sorted(group_rows.items()):
        classes = {recs[i]["label"] for i in idxs}
        if len(classes) > 1:
            kinds = Counter(recs[i]["image_kind"] for i in idxs)
            fams = Counter(recs[i]["source_family"] for i in idxs)
            for i in idxs:
                r = recs[i]
                issues.append({"issue_type": "cross_class_group", "severity": "high", "group_id": gid, "image_id": r["image_id"],
                               "label": r["label"], "relative_path": r["relative_path"], "image_kind": r["image_kind"],
                               "source_family": r["source_family"],
                               "detail": f"group spans {sorted(classes)}; kinds={dict(kinds)}; families={dict(fams)}; "
                                         f"link reasons={sorted(dsu.reasons.get(i, set()))}",
                               "recommended_action": "whole group kept in ONE split by the group-aware splitter; manually verify labels before training"})
    for i, r in enumerate(recs):
        if r["source_family"] == "AL9EE_greyspot":
            issues.append({"issue_type": "foreign_disease_filename", "severity": "high", "group_id": r["group_id"], "image_id": r["image_id"],
                           "label": r["label"], "relative_path": r["relative_path"], "image_kind": r["image_kind"], "source_family": r["source_family"],
                           "detail": "filename says 'Grey spot' but file sits in the Alternaria folder",
                           "recommended_action": "exclude from training/eval unless an expert confirms Alternaria"})
        if r["source_family"] == "Roboflow_malus":
            issues.append({"issue_type": "unverifiable_source_label", "severity": "medium", "group_id": r["group_id"], "image_id": r["image_id"],
                           "label": r["label"], "relative_path": r["relative_path"], "image_kind": r["image_kind"], "source_family": r["source_family"],
                           "detail": "Roboflow 'malus' export; original class unknown", "recommended_action": "spot-check visually"})
        if r["source_family"].startswith("Unknown_"):
            issues.append({"issue_type": "unrecognised_filename", "severity": "low", "group_id": r["group_id"], "image_id": r["image_id"],
                           "label": r["label"], "relative_path": r["relative_path"], "image_kind": r["image_kind"], "source_family": r["source_family"],
                           "detail": "filename does not match any known provenance pattern", "recommended_action": "inspect"})
    # synthetic rec_B images whose nearest perceptual neighbour of the OTHER class is close
    recB_idx = [i for i, r in enumerate(recs) if r["source_family"] == "CycleGAN_rec_B_alternaria"]
    healthy_idx = np.array([i for i, r in enumerate(recs) if r["label"] == "Healthy"])
    if recB_idx and len(healthy_idx):
        h_variants = np.array([recs[i]["phash_variants"] for i in healthy_idx], dtype=np.uint64).T
        q = np.array([recs[i]["phash_variants"][0] for i in recB_idx], dtype=np.uint64)
        close_cnt = 0
        for s in range(0, len(q), 512):
            dmin = hamming_min_block(q[s:s + 512], h_variants)
            best = dmin.min(axis=1)
            arg = dmin.argmin(axis=1)
            for a in range(len(best)):
                if best[a] <= 10:
                    i = recB_idx[s + a]; j = int(healthy_idx[arg[a]])
                    close_cnt += 1
                    issues.append({"issue_type": "synthetic_alternaria_resembles_healthy", "severity": "medium", "group_id": recs[i]["group_id"],
                                   "image_id": recs[i]["image_id"], "label": recs[i]["label"], "relative_path": recs[i]["relative_path"],
                                   "image_kind": recs[i]["image_kind"], "source_family": recs[i]["source_family"],
                                   "detail": f"pHash distance {int(best[a])} to Healthy image {recs[j]['relative_path']}",
                                   "recommended_action": "synthetic image; exclude from val/test and consider excluding from training"})
    issues_df = pd.DataFrame(issues, columns=["issue_type", "severity", "group_id", "image_id", "label", "relative_path", "image_kind",
                                              "source_family", "detail", "recommended_action"])
    issues_df.to_csv(os.path.join(RESULTS_DIR, "suspected_label_issues.csv"), index=False)

    # ---- audit report ----
    write_report(manifest, recs, non_images, unreadable, group_rows, dup_df, issues_df, pair_log)
    print(f"\nManifest: {manifest_path}  ({len(manifest)} rows)")
    print(f"Groups: {len(group_rows)} total, {sum(1 for g in group_rows.values() if len(g) > 1)} multi-image")
    print(f"Label issues logged: {len(issues_df)}")


def write_report(manifest, recs, non_images, unreadable, group_rows, dup_df, issues_df, pair_log):
    L = []
    L.append("# Dataset Audit Report (retrained pipeline)\n")
    L.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M}  |  Source: `DATASET/` (read-only)  |  Script: `code/03_perceptual_hash_audit.py`\n")
    L.append("Class mapping (explicit, never alphabetical): `DATASET/Healthy` -> **Healthy = 0**, `DATASET/Alternaria leaf spot` -> **Alternaria = 1**.\n")

    L.append("## 1. Folder contents\n")
    L.append("| Folder | Total entries | Supported images | Non-image entries | Unreadable images |")
    L.append("|---|---:|---:|---:|---:|")
    for class_dir in CLASS_DIRS:
        d = os.path.join(DATASET_DIR, class_dir)
        total = len(os.listdir(d))
        n_img = int((manifest["class_dir"] == class_dir).sum())
        n_non = sum(1 for x in non_images if x["class_dir"] == class_dir)
        n_bad = sum(1 for x in unreadable if f"/{class_dir}/" in x["relative_path"])
        L.append(f"| `{class_dir}` | {total} | {n_img} | {n_non} | {n_bad} |")
    L.append(f"| **Total** | {sum(len(os.listdir(os.path.join(DATASET_DIR, c))) for c in CLASS_DIRS)} | {len(manifest)} | {len(non_images)} | {len(unreadable)} |\n")
    if non_images:
        L.append("Non-image entries (ignored, not deleted):\n")
        for x in non_images:
            L.append(f"- `{x['class_dir']}/{x['filename']}` - {x['reason']}" + (f", {x['size_bytes']/1e6:.1f} MB" if 'size_bytes' in x else ""))
        L.append("")
    if unreadable:
        L.append("Unreadable images (excluded from the manifest):\n")
        for x in unreadable:
            L.append(f"- `{x['relative_path']}`: {x['error']}")
        L.append("")
    else:
        L.append("All supported images opened, verified and decoded successfully (0 unreadable).\n")

    L.append("## 2. Formats, colour modes and dimensions\n")
    for class_dir in CLASS_DIRS:
        sub = manifest[manifest["class_dir"] == class_dir]
        L.append(f"**{class_dir}** ({len(sub)} images)\n")
        L.append("- Extensions: " + ", ".join(f"`{k}` {v}" for k, v in sub["extension"].value_counts().items()))
        L.append("- Decoded formats: " + ", ".join(f"{k} {v}" for k, v in sub["format"].value_counts().items()))
        L.append("- Colour modes: " + ", ".join(f"{k} {v}" for k, v in sub["mode"].value_counts().items()))
        dims = (sub["width"].astype(str) + "x" + sub["height"].astype(str)).value_counts()
        L.append(f"- Distinct dimensions: {len(dims)}; most common: " + ", ".join(f"{k} ({v})" for k, v in dims.head(6).items()))
        L.append(f"- Width range {sub['width'].min()}-{sub['width'].max()}, height range {sub['height'].min()}-{sub['height'].max()}\n")

    L.append("## 3. Provenance families (from filename conventions)\n")
    L.append("| Class | Source family | Images | Distinct family keys | Kinds |")
    L.append("|---|---|---:|---:|---|")
    for (lab, fam), sub in manifest.groupby(["label", "source_family"]):
        kinds = ", ".join(f"{k} {v}" for k, v in sub["image_kind"].value_counts().items())
        L.append(f"| {lab} | `{fam}` | {len(sub)} | {sub['family_key'].nunique()} | {kinds} |")
    L.append("")
    L.append("Image kinds: `original` = unmodified capture or earliest known version; `copy` = Windows '(n)' duplicate; "
             "`reexport` = Roboflow re-export of a known base image; `augmented` = flip/rotate/crop variant; "
             "`synthetic` = CycleGAN generator output (`_rec_A` / `_rec_B`); `synthetic_augmented` = Augmentor output of a synthetic image.\n")
    syn = manifest["is_synthetic"].sum(); aug = manifest["is_augmented"].sum()
    L.append(f"- Synthetic (CycleGAN) images: **{syn}** ({syn/len(manifest)*100:.1f}%). Augmented / re-exported images: **{aug}** ({aug/len(manifest)*100:.1f}%).")
    L.append(f"- `eval_eligible` (original, neither synthetic nor augmented, first occurrence of its byte content): **{manifest['eval_eligible'].sum()}** images "
             f"(Healthy {int(manifest[(manifest.label=='Healthy')]['eval_eligible'].sum())}, Alternaria {int(manifest[(manifest.label=='Alternaria')]['eval_eligible'].sum())}).\n")

    L.append("## 4. Exact duplicates (SHA-256)\n")
    sha_groups = manifest.groupby("sha256").size()
    multi = sha_groups[sha_groups > 1]
    L.append(f"- Distinct file hashes: {len(sha_groups)} for {len(manifest)} images.")
    L.append(f"- Byte-identical groups: **{len(multi)}**, redundant copies: **{int(multi.sum() - len(multi))}**.")
    for lab in ("Healthy", "Alternaria"):
        sub = manifest[manifest.label == lab].groupby("sha256").size(); m = sub[sub > 1]
        L.append(f"  - {lab}: {len(m)} groups, {int(m.sum() - len(m))} redundant copies.")
    cross = manifest.groupby("sha256")["label"].nunique(); cross = cross[cross > 1]
    L.append(f"- Byte-identical images present in BOTH classes: **{len(cross)}**.\n")

    L.append("## 5. Near-duplicate and provenance groups (union of exact + perceptual + filename families)\n")
    sizes = Counter(len(v) for v in group_rows.values())
    L.append(f"- Parameters: near-duplicate if, under some flip/rotation, pHash <= {PHASH_T} AND dHash <= {DHASH_T} (or pHash <= {PHASH_T_ALONE} alone); "
             f"rec_B index alignment accepted at pHash <= {RECB_INDEX_T}; phone bursts within {BURST_SECONDS}s. "
             f"Calibration: true same-leaf pairs have dHash <= 5, different lab-photographed leaves that collide on pHash have dHash >= 6.")
    L.append(f"- Total groups: **{len(group_rows)}** ({sum(1 for v in group_rows.values() if len(v) == 1)} singletons, "
             f"{sum(1 for v in group_rows.values() if len(v) > 1)} multi-image groups covering {sum(len(v) for v in group_rows.values() if len(v) > 1)} images).")
    reasons = Counter(p[2] for p in pair_log)
    L.append("- Pair links recorded: " + ", ".join(f"{k} {v}" for k, v in reasons.items()))
    nd = [p[3] for p in pair_log if p[2].startswith("near_dup")]
    if nd:
        L.append("- Near-duplicate link distances: pHash " + ", ".join(f"{k}:{v}" for k, v in sorted(Counter(x // 100 for x in nd).items()))
                 + " | dHash " + ", ".join(f"{k}:{v}" for k, v in sorted(Counter(x % 100 for x in nd).items())))
    L.append("- Group size distribution: " + ", ".join(f"{k}:{v}" for k, v in sorted(sizes.items())[:25]) + (" ..." if len(sizes) > 25 else ""))
    biggest = sorted(group_rows.values(), key=len, reverse=True)[:5]
    L.append("- Largest groups: " + "; ".join(f"g{recs[v[0]]['group_id']} ({len(v)} imgs, {recs[v[0]]['family_key']})" for v in biggest))
    gl = manifest.groupby("group_id")["label"].nunique()
    L.append(f"- Groups spanning both classes: **{int((gl > 1).sum())}** (see `suspected_label_issues.csv`).\n")
    L.append("Which duplicates can be excluded safely: byte-identical copies (same SHA-256) carry no information and can be dropped at training time "
             "without loss; Roboflow re-exports and Windows copies that are byte-identical fall in that category. Flip/rotate/crop augmentations and "
             "CycleGAN outputs are NOT removed by this pipeline; they are kept in their group so the whole family lands in one split, and they are marked "
             "`eval_eligible = False` so evaluation can be reported on real, unmodified images.\n")

    L.append("## 6. Suspected label / provenance issues\n")
    if len(issues_df):
        for t, sub in issues_df.groupby("issue_type"):
            L.append(f"- `{t}` ({sub['severity'].iloc[0]}): {len(sub)} rows, {sub['group_id'].nunique()} groups")
    else:
        L.append("- none")
    L.append("\nKey observations:\n")
    L.append("- The Alternaria folder holds CycleGAN outputs named `Health (N)_rec_B.jpg` plus Augmentor variants (`output_original_Health (N)_rec_B.png_<uuid>.jpg`). "
             "Visual inspection shows these are reconstructions of the AL9EE `Alternaria leaf spot (N)` leaves, with lesions partly removed, so their Alternaria label is weak. "
             "They are grouped with their AL9EE source image whenever the perceptual hash confirms it.")
    L.append("- The Healthy folder holds `Health (N)_rec_A.png` CycleGAN reconstructions (and Augmentor variants) of the `Health (N).jpg` field photos; 1,030 of them have no original in this dataset.")
    L.append("- `Web_thumbnail_augmented`: 45 tiny web images (about 140x128 px, ' - Copy' variants merged) each expanded into 57 flip/crop variants per file. Treated as 45 provenance families.")
    L.append("- PlantVillage healthy leaves appear as original + `_0/_1/_2/_flipTB` rotations/flips (up to 5 files per leaf). Treated as one group per UUID.")
    L.append("- Indigenous (`lpu_rdp_*`) files named `name (2).jpeg` are NOT copies: their perceptual hashes differ from the same-named file "
             "(distances 20-40 bits, i.e. unrelated images), so they are kept as separate originals and linked only when hashes agree.")
    L.append("- Four CycleGAN weight files (`latest_net_{G,D}_{A,B}.pth`) sit inside the Alternaria folder; they are ignored and left untouched.\n")
    L.append("## 7. Outputs\n")
    L.append("- `results/retrained/dataset_manifest.csv` - one row per image (split column filled by `prepare_clean_splits.py`).")
    L.append("- `results/retrained/duplicate_groups.csv` - reviewable list of all multi-image groups with recommendations.")
    L.append("- `results/retrained/suspected_label_issues.csv` - cross-class groups and other label risks.")
    with open(os.path.join(RESULTS_DIR, "dataset_audit_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
