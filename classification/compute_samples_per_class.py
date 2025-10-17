#!/usr/bin/env python3
"""
compute_samples_per_class.py

Given:
  --dataset_name         (Hugging Face dataset path)
  --downscale_factor     (int >= 1)
  --max_repetitions      (int >= 1)

This script loads the dataset's train split, infers/uses a target label column,
computes class counts, and outputs a comma-separated "samples per class" string
suitable for the --sample_per_class argument in your trainer.

Logic:
  1. First downscale all class counts by downscale_factor
  2. Then repeat each downscaled count up to max_repetitions times while staying under the cap

It will also print useful diagnostics.
"""

import argparse
from datasets import load_dataset
from collections import Counter
import numpy as np
import sys

# Known label ranges for common columns in your project
LABEL_MIN_MAX_MAP = {
    "fw_edu_score": (0, 5),
    "dclm_score": (0, 5),
    "ocr_quality_score": (0, 3),
    "fw_edu_score_2": (0, 5),
}

def pick_target_column(example_columns, preferred=None):
    # If user provided a preferred and it's present, use it.
    if preferred and preferred in example_columns:
        return preferred
    # Otherwise pick the first known column that exists
    for col in LABEL_MIN_MAX_MAP.keys():
        if col in example_columns:
            return col
    # Fall back to common names
    for col in ["label", "labels", "target", "y"]:
        if col in example_columns:
            return col
    raise ValueError(
        f"Could not infer target column from columns: {sorted(example_columns)}. "
        f"Pass --target_column explicitly."
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW-Dev/fine-pdfs-classification-1-chunk-tb-teacher-1M-eng_Latn-Qwen_Qwen3-235B-A22B-Instruct-2507")
    parser.add_argument("--subset", type=str, default="ces_Latn")
    parser.add_argument("--downscale_factor", type=int, required=True,
                        help="Divide all class counts by this factor first.")
    parser.add_argument("--max_repetitions", type=int, required=True,
                        help="Per-class cap: at most this many times each original example can be repeated (on average).")
    parser.add_argument("--target_column", type=str, default=None,
                        help="(Optional) Label column. If omitted, will be inferred.")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use (default: train)")
    args = parser.parse_args()

    if args.downscale_factor < 1 or args.max_repetitions < 1:
        raise ValueError("--downscale_factor and --max_repetitions must be >= 1")

    ds = load_dataset(args.dataset_name, split=args.split, name=args.subset)
    # Infer target column
    target_col = pick_target_column(ds.column_names, args.target_column)
    print(f"[info] Using target column: {target_col}", file=sys.stderr)

    # Try to determine label bounds (min/max)
    if target_col in LABEL_MIN_MAX_MAP:
        min_label, max_label = LABEL_MIN_MAX_MAP[target_col]
    else:
        # Infer from data
        # Filter out bad rows first
        def ok(x):
            return (x.get(target_col) is not None and
                    x.get("text") is not None and str(x.get("text")).strip() != "")
        ds = ds.filter(ok)
        vals = [int(v) for v in ds[target_col] if v is not None]
        if not vals:
            raise ValueError(f"No valid labels found in column '{target_col}'.")
        min_label, max_label = min(vals), max(vals)
        print(f"[info] Inferred label range: [{min_label}, {max_label}]", file=sys.stderr)

    # Clean and clip labels; remove rows without text or label
    def valid_and_clip(example):
        lbl = example.get(target_col, None)
        txt = example.get("text", None)
        if lbl is None or txt is None or str(txt).strip() == "":
            return False
        try:
            v = int(lbl)
        except Exception:
            return False
        return (v >= min_label) and (v <= max_label)

    ds = ds.filter(valid_and_clip)
    labels = [int(v) for v in ds[target_col]]
    if not labels:
        raise ValueError("No usable examples after filtering.")

    # Build counts for classes in [min_label..max_label]
    counts = Counter(labels)
    # Ensure every class key exists (maybe zero)
    for c in range(min_label, max_label + 1):
        counts.setdefault(c, 0)

    class_with_min_count = min(counts, key=counts.get)
    target_samples_for_min_class = counts[class_with_min_count] * args.max_repetitions

    samples_per_class = []
    max_allowed_samples = target_samples_for_min_class * args.downscale_factor
    for c in range(min_label, max_label + 1):
        if counts[c] == 0:
            samples_per_class.append(0)
        else:
            samples_per_class.append(max(min(counts[c], max_allowed_samples), target_samples_for_min_class))

    print(f"[info] computed samples_per_class = {samples_per_class}", file=sys.stderr)
    print(f"Resulting samples: {','.join(str(s) for s in samples_per_class)}")
    print(f"Total steps: {sum(samples_per_class)/128}")



if __name__ == "__main__":
    main()
