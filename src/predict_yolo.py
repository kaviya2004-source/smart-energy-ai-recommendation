"""Generate reproducible person-detection outputs using the trained YOLOv8 best model."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def default_model() -> Path:
    copied = ROOT / "models" / "yolo_person_detector.pt"
    if copied.exists():
        return copied
    candidates = list((ROOT / "runs").rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError("YOLO best.pt not found. Train the YOLO model first.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/yolo/processed/images/test", help="Image file or directory")
    parser.add_argument("--model", default=None)
    parser.add_argument("--confidence", type=float, default=.25)
    args = parser.parse_args()
    output = ROOT / "data/yolo/reports/inference"
    model = YOLO(args.model or default_model())
    results = model.predict(source=args.source, conf=args.confidence, save=True, project=str(output), name="predictions", exist_ok=True)
    events = []
    for result in results:
        person_count = int(len(result.boxes)) if result.boxes is not None else 0
        events.append({"image": Path(result.path).name, "person_present": int(person_count > 0), "person_count": person_count, "confidence_threshold": args.confidence})
    csv_path = output / "person_detection_events.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "person_present", "person_count", "confidence_threshold"])
        writer.writeheader(); writer.writerows(events)
    print(f"Saved {len(events)} person-detection events to {csv_path}")


if __name__ == "__main__":
    main()
