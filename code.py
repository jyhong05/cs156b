#!/usr/bin/env python3
"""Week 1 rudimentary CheXpert exploration script.

Scope is intentionally limited to Week 1:
- verify CSV/data paths
- inspect label distributions (positive/negative/uncertain/missing)
- sample-check image path existence
- play with simple loss functions (BCE, weighted BCE, focal) on synthetic logits
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter
from pathlib import Path

TRAIN_IMAGE_BASE_DIR = Path("/resnick/groups/CS156b/from_central/data/train")
TRAIN_CSV = Path("/resnick/groups/CS156b/from_central/data/train/student_labels/train2023.csv")
TEST_IDS_CSV = Path("/groups/CS156b/data/student_labels/test_ids.csv")

TARGET_COLUMNS = [
    "Fracture",
    "Lung Opacity",
    "Pneumonia",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Pleural Other",
    "Support Devices",
    "Pleural Effusion",
    "No Finding",
]

ALIASES = {
    "Enlarged Cardiom.": "Enlarged Cardiomediastinum",
}


def canonical_label(col_name: str) -> str:
    return ALIASES.get(col_name, col_name)


def resolve_image_path(path_value: str) -> Path:
    rel = Path(path_value)
    if rel.is_absolute():
        return rel

    # Some CSV path styles include the train folder prefix, others are relative to train.
    candidate_a = TRAIN_IMAGE_BASE_DIR / rel
    candidate_b = TRAIN_IMAGE_BASE_DIR.parent / rel

    if candidate_a.exists() or not candidate_b.exists():
        return candidate_a
    return candidate_b


def read_csv_header_and_sample(path: Path, preview_rows: int = 5) -> tuple[list[str], list[dict[str, str]], int]:
    rows: list[dict[str, str]] = []
    total_rows = 0

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for row in reader:
            total_rows += 1
            if len(rows) < preview_rows:
                rows.append(row)

    return header, rows, total_rows


def parse_label(raw: str | None) -> str:
    if raw is None:
        return "nan"

    value = raw.strip()
    if value == "":
        return "nan"

    try:
        x = float(value)
    except ValueError:
        return "other"

    if math.isnan(x):
        return "nan"
    if x == 1.0:
        return "pos"
    if x == 0.0:
        return "neg"
    if x == -1.0:
        return "uncertain"
    return "other"


def summarize_labels(train_csv: Path) -> tuple[dict[str, Counter], int, Counter]:
    counts = {label: Counter() for label in TARGET_COLUMNS}
    view_counter = Counter()
    total_rows = 0

    with train_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []

        # Build map from canonical target label -> actual column name in CSV.
        colmap: dict[str, str] = {}
        for c in columns:
            canon = canonical_label(c)
            if canon in TARGET_COLUMNS and canon not in colmap:
                colmap[canon] = c

        for row in reader:
            total_rows += 1

            view = row.get("Frontal/Lateral", "")
            if view:
                view_counter[view] += 1

            for label in TARGET_COLUMNS:
                actual_col = colmap.get(label)
                parsed = parse_label(row.get(actual_col) if actual_col else None)
                counts[label][parsed] += 1

    return counts, total_rows, view_counter


def apply_policy(parsed: str, uncertain_policy: str, nan_policy: str) -> tuple[float, float]:
    # Returns (mapped_label, include_mask)
    if parsed == "pos":
        return 1.0, 1.0
    if parsed == "neg":
        return 0.0, 1.0
    if parsed == "uncertain":
        if uncertain_policy == "one":
            return 1.0, 1.0
        if uncertain_policy == "zero":
            return 0.0, 1.0
        return 0.0, 0.0
    if parsed == "nan":
        if nan_policy == "zero":
            return 0.0, 1.0
        if nan_policy == "uncertain":
            return apply_policy("uncertain", uncertain_policy, nan_policy)
        return 0.0, 0.0
    return 0.0, 0.0


def summarize_policy(counts: dict[str, Counter], uncertain_policy: str, nan_policy: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for label in TARGET_COLUMNS:
        c = counts[label]
        mapped_pos = 0.0
        mapped_neg = 0.0
        valid = 0.0

        for parsed_value, n in c.items():
            mapped, include = apply_policy(parsed_value, uncertain_policy, nan_policy)
            valid += include * n
            mapped_pos += mapped * include * n
            mapped_neg += (1.0 - mapped) * include * n

        pos_rate = mapped_pos / valid if valid > 0 else 0.0
        pos_weight = mapped_neg / max(mapped_pos, 1e-6)

        rows.append(
            {
                "label": label,
                "valid_fraction": f"{(valid / max(sum(c.values()), 1)):.6f}",
                "positive_rate": f"{pos_rate:.6f}",
                "pos_weight_for_bce": f"{pos_weight:.6f}",
            }
        )

    return rows


def sample_image_existence(train_csv: Path, sample_n: int) -> tuple[int, int]:
    checked = 0
    exists = 0

    with train_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path_val = row.get("Path")
            if not path_val:
                continue
            checked += 1
            if resolve_image_path(path_val).exists():
                exists += 1
            if checked >= sample_n:
                break

    return exists, checked


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def loss_playground(policy_rows: list[dict[str, str]], samples: int = 400) -> dict[str, float]:
    random.seed(42)
    rates = [max(0.01, min(0.99, float(r["positive_rate"]))) for r in policy_rows]
    weights = [max(1.0, min(200.0, float(r["pos_weight_for_bce"]))) for r in policy_rows]

    bce_sum = 0.0
    wbce_sum = 0.0
    focal_sum = 0.0
    n = 0

    for _ in range(samples):
        for j in range(len(rates)):
            y = 1.0 if random.random() < rates[j] else 0.0
            logit = random.uniform(-3.0, 3.0)
            p = min(max(sigmoid(logit), 1e-6), 1.0 - 1e-6)

            bce = -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
            w = weights[j] if y == 1.0 else 1.0
            wbce = w * bce

            pt = p if y == 1.0 else (1.0 - p)
            gamma = 2.0
            focal = -((1.0 - pt) ** gamma) * math.log(max(pt, 1e-6))

            bce_sum += bce
            wbce_sum += wbce
            focal_sum += focal
            n += 1

    return {
        "avg_bce": bce_sum / max(n, 1),
        "avg_weighted_bce": wbce_sum / max(n, 1),
        "avg_focal_gamma2": focal_sum / max(n, 1),
    }


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 1 rudimentary CheXpert data exploration")
    parser.add_argument("--train-csv", default=str(TRAIN_CSV))
    parser.add_argument("--test-csv", default=str(TEST_IDS_CSV))
    parser.add_argument("--sample-images", type=int, default=200)
    parser.add_argument("--uncertain-policy", choices=["ignore", "zero", "one"], default="ignore")
    parser.add_argument("--nan-policy", choices=["ignore", "zero", "uncertain"], default="ignore")
    args = parser.parse_args()

    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)

    print("=== Week 1 Rudimentary Pipeline ===")
    print("train_csv:", train_csv)
    print("test_csv:", test_csv)
    print("train_image_base_dir:", TRAIN_IMAGE_BASE_DIR)

    if not train_csv.exists() or not test_csv.exists():
        print("ERROR: train/test CSV path missing.")
        print("Run this inside HPC where those paths exist.")
        return

    train_header, train_preview, train_rows = read_csv_header_and_sample(train_csv)
    test_header, test_preview, test_rows = read_csv_header_and_sample(test_csv)

    print("train rows:", train_rows)
    print("test rows:", test_rows)
    print("train columns:", len(train_header))
    print("test columns:", len(test_header))

    print("\ntrain preview (first 2 rows):")
    for r in train_preview[:2]:
        print(r)

    print("\ntest preview (first 2 rows):")
    for r in test_preview[:2]:
        print(r)

    label_counts, _, view_counter = summarize_labels(train_csv)

    raw_rows: list[dict[str, str]] = []
    for label in TARGET_COLUMNS:
        c = label_counts[label]
        raw_rows.append(
            {
                "label": label,
                "n_pos": str(c["pos"]),
                "n_neg": str(c["neg"]),
                "n_uncertain": str(c["uncertain"]),
                "n_nan": str(c["nan"]),
                "n_other": str(c["other"]),
            }
        )

    policy_rows = summarize_policy(
        label_counts,
        uncertain_policy=args.uncertain_policy,
        nan_policy=args.nan_policy,
    )

    exists, checked = sample_image_existence(train_csv, sample_n=args.sample_images)
    losses = loss_playground(policy_rows)

    write_csv(
        Path("week1_raw_label_stats.csv"),
        raw_rows,
        fieldnames=["label", "n_pos", "n_neg", "n_uncertain", "n_nan", "n_other"],
    )
    write_csv(
        Path("week1_policy_stats.csv"),
        policy_rows,
        fieldnames=["label", "valid_fraction", "positive_rate", "pos_weight_for_bce"],
    )

    with Path("week1_report.txt").open("w", encoding="utf-8") as f:
        f.write("Week 1 Rudimentary Report\n")
        f.write("========================\n")
        f.write(f"train_csv={train_csv}\n")
        f.write(f"test_csv={test_csv}\n")
        f.write(f"train_rows={train_rows}\n")
        f.write(f"test_rows={test_rows}\n")
        f.write(f"frontal_lateral_counts={dict(view_counter)}\n")
        f.write(f"sample_image_exists={exists}/{checked}\n")
        f.write(f"policy_uncertain={args.uncertain_policy}\n")
        f.write(f"policy_nan={args.nan_policy}\n")
        f.write(f"loss_playground={losses}\n")

    print("\nFrontal/Lateral counts:", dict(view_counter))
    print(f"sample image existence: {exists}/{checked}")
    print("loss playground:", losses)
    print("\nSaved files:")
    print("- week1_raw_label_stats.csv")
    print("- week1_policy_stats.csv")
    print("- week1_report.txt")


if __name__ == "__main__":
    main()


