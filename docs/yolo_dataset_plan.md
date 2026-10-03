# YOLOv8 Person Detection Plan

## Recommended source

Use the **CCTV Indoor Person Detection** dataset from Roboflow Universe. It is closest to this project because it is indoor CCTV imagery, contains 562 labelled images, and already provides Darknet/YOLO-style annotations. Before downloading, record its licence, source URL, version, download date, and citation in the report.

Keep the downloaded export unmodified under `data/yolo/raw/<dataset-name>/`. Do not manually overwrite it.

## Reproducible preparation

```powershell
python src/prepare_yolo_data.py --source "data/yolo/raw/cctv_indoor_person" --limit 500
```

The script verifies image readability, duplicate image hashes, one-class YOLO labels, normalised boxes, and image-label pairs. It retains 500 valid images and creates a reproducible 350 / 75 / 75 train / validation / test split.

## Training and test protocol

```powershell
python src/train_yolo.py --epochs 80
```

Use YOLOv8n transfer learning because the dataset is small. Select the epoch using validation performance; report test precision, recall, mAP50 and mAP50-95 only once. Keep all generated curves, confusion matrix and example detections in `data/yolo/reports/`.

## Integration rule

The final dashboard displays both models independently. A unified recommendation is allowed only if camera/home ID and time window match; otherwise it is an operational demonstration, not a joined statistical dataset.
