"""Lab 02 baseline for Road Damage Detection (Team Surface 3).

Smallest complete path: load the documented RDD2022 split, check it, draw a
fixed-seed sample, fine-tune a pretrained YOLOv8n, evaluate it once on the
held-out test sample, and write reports/baseline_metrics.json.

Run from the repository root:
    python src/baseline_pipeline.py
"""

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
CLASS_NAMES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Other Corruption",
    "Pothole",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Short descriptions written into the JSON by the script (team-supplied text).
EXPERIMENT = "rdd_yolov8n_baseline_v1"
MODEL_DESC = "YOLOv8n pretrained on COCO, fine-tuned with default Ultralytics settings"
LIMITATION = (
    "Image-level random split from the provided RDD2022 copy: images from the "
    "same drive or road may appear in both training and test, so the score may "
    "be optimistic; potholes are the rarest class (~9-10% of objects) and "
    "are detected worst."
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=ROOT / "data" / "RDD_SPLIT", type=Path)
    p.add_argument("--sample-dir", default=ROOT / "data" / "RDD_SAMPLE", type=Path)
    p.add_argument("--out", default=ROOT / "reports" / "baseline_metrics.json", type=Path)
    p.add_argument("--train-n", type=int, default=5000)
    p.add_argument("--val-n", type=int, default=2000)
    p.add_argument("--test-n", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None,
                   help="mps (Apple GPU), 0 (CUDA GPU) or cpu; default: auto")
    return p.parse_args()


# ---------------------------------------------------------------- checks
def paired_files(split_dir: Path):
    """Return sorted (image, label) pairs and counts of unmatched files."""
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    for d in (img_dir, lbl_dir):
        if not d.is_dir():
            sys.exit(f"Missing folder: {d}\nSee data/README.md for where to place the data.")
    images = {f.stem: f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS}
    labels = {f.stem: f for f in lbl_dir.iterdir() if f.suffix == ".txt"}
    common = sorted(images.keys() & labels.keys())
    pairs = [(images[k], labels[k]) for k in common]
    return pairs, len(images.keys() - labels.keys()), len(labels.keys() - images.keys())


def check_label_file(path: Path):
    """Validate YOLO format; return list of class ids. Raise on bad lines."""
    classes = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{n} expected 5 values, got {len(parts)}")
        cls = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        if not 0 <= cls < len(CLASS_NAMES):
            raise ValueError(f"{path}:{n} class id {cls} out of range")
        if not all(0.0 <= v <= 1.0 for v in coords):
            raise ValueError(f"{path}:{n} box values must be normalised to [0, 1]")
        classes.append(cls)
    return classes


def base_name(stem: str) -> str:
    """'India_003323 (1)' -> 'India_003323' (to spot duplicated copies)."""
    return re.sub(r"\s*\(\d+\)$", "", stem)


# ---------------------------------------------------------------- sample
def build_sample(pairs, n, seed, dest: Path):
    rng = random.Random(seed)
    chosen = rng.sample(pairs, min(n, len(pairs)))
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "images").mkdir(parents=True)
    (dest / "labels").mkdir(parents=True)
    counts, bad = Counter(), 0
    kept = []
    for img, lbl in chosen:
        try:
            counts.update(check_label_file(lbl))
        except ValueError as e:
            print(f"  skipped bad label: {e}")
            bad += 1
            continue
        shutil.copy2(img, dest / "images" / img.name)
        shutil.copy2(lbl, dest / "labels" / lbl.name)
        kept.append(img.stem)
    return kept, counts, bad


def write_data_yaml(sample_dir: Path) -> Path:
    yaml_path = sample_dir / "data.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        f"path: {sample_dir.resolve()}\n"
        "train: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n{names}\n"
    )
    return yaml_path


def pick_device():
    """Use CUDA if present, then Apple MPS, otherwise CPU."""
    import torch
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    print(f"Data: {args.data_dir}")

    # 1. Load and check the documented split
    sizes = {"train": args.train_n, "val": args.val_n, "test": args.test_n}
    sample_info, checks = {}, {}
    for split in SPLITS:
        pairs, img_only, lbl_only = paired_files(args.data_dir / split)
        if not pairs:
            sys.exit(f"No image/label pairs found in {args.data_dir / split}")
        checks[split] = {
            "paired": len(pairs),
            "images_without_label": img_only,
            "labels_without_image": lbl_only,
        }
        print(f"{split}: {len(pairs)} pairs, {img_only} images w/o label, "
              f"{lbl_only} labels w/o image")

        # 2. Fixed-seed sample (sorted input, so the same on every machine)
        kept, counts, bad = build_sample(
            pairs, sizes[split], args.seed, args.sample_dir / split
        )
        sample_info[split] = {
            "images": len(kept),
            "objects": sum(counts.values()),
            "per_class": {CLASS_NAMES[c]: counts[c] for c in range(len(CLASS_NAMES))},
            "bad_labels_skipped": bad,
            "stems": kept,
        }

    # 3. Leakage check: same base image name in training and test samples
    train_bases = {base_name(s) for s in sample_info["train"]["stems"]}
    test_bases = {base_name(s) for s in sample_info["test"]["stems"]}
    overlap = len(train_bases & test_bases)
    print(f"Train/test base-name overlap in sample: {overlap}")
    for split in SPLITS:
        sample_info[split].pop("stems")

    data_yaml = write_data_yaml(args.sample_dir)

    # 4. Fit the baseline (pretrained YOLOv8n, default settings)
    from ultralytics import YOLO  # imported here so the checks run without it

    device = args.device or pick_device()
    print(f"Device: {device}")
    runs_dir = ROOT / "reports" / "runs" / "detect"

    model = YOLO("yolov8n.pt")
    train_kwargs = dict(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, seed=args.seed, deterministic=True,
        project=str(runs_dir), name="baseline_YOLOv8n", exist_ok=True,
        device=device,
    )
    model.train(**train_kwargs)

    # 5. Evaluate once on the held-out test sample
    best = runs_dir / "baseline_YOLOv8n" / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(
        data=str(data_yaml), split="test", imgsz=args.imgsz,
        project=str(runs_dir), name="baseline_YOLOv8n_test", exist_ok=True,
        device=device,
    )
    per_class_map50 = {
        CLASS_NAMES[int(c)]: round(float(metrics.box.ap50[i]), 4)
        for i, c in enumerate(metrics.box.ap_class_index)
    }

    # 6. Write the result record
    record = {
        "experiment": EXPERIMENT,
        "data": (f"RDD2022 sample, seed {args.seed}: {sample_info['train']['images']} "
                 f"train / {sample_info['val']['images']} val / "
                 f"{sample_info['test']['images']} test images"),
        "evaluation": (f"{sample_info['test']['images']} held-out test images "
                       f"({sample_info['test']['objects']} objects), used only once "
                       "after training"),
        "model": f"{MODEL_DESC}; {args.epochs} epochs, {args.imgsz} px, device {device}",
        "metric": {"name": "mAP50", "value": round(float(metrics.box.map50), 4)},
        "secondary_metrics": {
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
            "mAP50-95": round(float(metrics.box.map), 4),
            "mAP50_per_class": per_class_map50,
        },
        "data_checks": {
            "full_split": checks,
            "sample": sample_info,
            "train_test_base_name_overlap": overlap,
        },
        "limitation": LIMITATION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))
    print(f"\nmAP50 = {record['metric']['value']}  ->  written to {args.out}")


if __name__ == "__main__":
    main()
