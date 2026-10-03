"""Train and evaluate YOLOv8 after the validated person dataset is prepared."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def latest_best_model() -> Path:
    # Ultralytics may prefix a relative project path with runs/detect on Windows.
    candidates = list(Path("data/yolo/reports").rglob("best.pt")) + list(Path("runs").rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError("No trained best.pt found. Run training first.")
    return max(candidates, key=lambda item: item.stat().st_mtime)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/yolo/processed/data.yaml")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate the newest saved best.pt without re-training")
    args = parser.parse_args()
    if args.evaluate_only:
        best = latest_best_model()
    else:
        model = YOLO("yolov8n.pt")
        model.train(data=args.data, epochs=args.epochs, imgsz=640, batch=8,
                    project=str(Path("data/yolo/reports").resolve()), name="training", seed=42, patience=15)
        best = latest_best_model()
    model = YOLO(best)
    test_metrics = model.val(data=args.data, split="test", project=str(Path("data/yolo/reports").resolve()), name="test_metrics")
    metric_values = {key: float(value) for key, value in test_metrics.results_dict.items()}
    (Path("data/yolo/reports") / "test_metrics_summary.json").write_text(json.dumps(metric_values, indent=2), encoding="utf-8")
    print(f"Best model: {best}")

if __name__ == "__main__":
    main()
