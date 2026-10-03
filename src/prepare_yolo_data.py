"""Validate, de-duplicate and split a single-class YOLO person-detection dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_label(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return True, "negative_image"  # valid background image: no person is annotated
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) != 5 or int(values[0]) != 0:
                return False, "invalid_class_or_column_count"
            if not all(0 <= float(value) <= 1 for value in values[1:]):
                return False, "box_outside_normalised_range"
    except (UnicodeDecodeError, ValueError):
        return False, "unreadable_label"
    return True, "valid"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Raw directory containing images/ and labels/ folders")
    parser.add_argument("--output", default="data/yolo/processed")
    parser.add_argument("--limit", type=int, default=500, help="Maximum approved images; use 0 for every valid image")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    # Support both a flat `images/labels` export and Roboflow's `train|valid|test/images` export.
    pairs: list[tuple[Path, Path]] = []
    if (source / "images").exists():
        images_root, labels_root = source / "images", source / "labels"
        pairs = [(image, labels_root / image.relative_to(images_root).with_suffix(".txt"))
                 for image in images_root.rglob("*") if image.suffix.lower() in IMAGE_SUFFIXES]
    else:
        for source_split in ("train", "valid", "test"):
            images_root, labels_root = source / source_split / "images", source / source_split / "labels"
            if images_root.exists():
                pairs.extend((image, labels_root / image.relative_to(images_root).with_suffix(".txt"))
                             for image in images_root.rglob("*") if image.suffix.lower() in IMAGE_SUFFIXES)
    if not pairs:
        raise FileNotFoundError("Expected images/labels folders or Roboflow train/valid/test folders.")
    records, seen = [], set()
    for image, label in sorted(pairs):
        try:
            with Image.open(image) as im: im.verify()
        except (UnidentifiedImageError, OSError):
            records.append((image.name, "excluded", "corrupt_image")); continue
        image_hash = digest(image)
        if image_hash in seen:
            records.append((image.name, "excluded", "duplicate_image")); continue
        ok, reason = validate_label(label)
        if not ok:
            records.append((image.name, "excluded", reason)); continue
        seen.add(image_hash); records.append((image.name, "approved", reason, image, label))
    approved = [record for record in records if record[1] == "approved"]
    random.Random(args.seed).shuffle(approved)
    if args.limit:
        approved = approved[:args.limit]
    if len(approved) < 100:
        raise ValueError(f"Only {len(approved)} valid images found; at least 100 are required.")
    n = len(approved); train_end, val_end = round(n * .70), round(n * .85)
    split_items = {"train": approved[:train_end], "val": approved[train_end:val_end], "test": approved[val_end:]}
    for split, items in split_items.items():
        for image_dir in (output / "images" / split, output / "labels" / split): image_dir.mkdir(parents=True, exist_ok=True)
        for _, _, _, image, label in items:
            # Prefix prevents accidental clashes when source splits use the same image name.
            filename = f"{digest(image)[:12]}_{image.name}"
            shutil.copy2(image, output / "images" / split / filename)
            if label.exists(): shutil.copy2(label, output / "labels" / split / f"{Path(filename).stem}.txt")
    with (output / "dataset_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["image", "decision", "reason"])
        for record in records: writer.writerow(record[:3])
    absolute_output = output.resolve().as_posix()
    (output / "data.yaml").write_text(f"path: {absolute_output}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: person\n", encoding="utf-8")
    print({"approved": n, "train": len(split_items["train"]), "val": len(split_items["val"]), "test": len(split_items["test"]), "audit": str(output / "dataset_audit.csv")})


if __name__ == "__main__":
    main()
