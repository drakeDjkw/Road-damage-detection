# Road Damage Detection — Team Surface 3

Yikwa Sandrakh, Wenda Waipo, Agata Melani
Artificial Intelligence Implementation Technologies · Baikal School of BRICS, INRTU

The project detects and classifies road surface defects in images from
vehicle cameras. It finds five classes: longitudinal crack, transverse crack,
alligator crack, other corruption and pothole. The project brief is in
[`docs/project_brief.md`](docs/project_brief.md).

## Lab 02 baseline

| | |
|---|---|
| Data | RDD2022 sample (seed 42): 5,000 train / 2,000 val / 2,000 test images |
| Baseline | YOLOv8n pretrained on COCO, fine-tuned with default settings (640 px, 50 epochs) |
| Primary metric | mAP50 on the held-out 2,000-image test sample |
| Limitation | Image-level random split, so frames from the same drive may leak between train and test; potholes are rare and detected worst |

## Environment

- Python 3.14 (tested with Python 3.14.0, Ultralytics 8.4.177 and
  torch 2.14.1)
- The script picks the hardware automatically:
  - an NVIDIA GPU (CUDA) if one is available;
  - otherwise the GPU of an Apple Silicon Mac (MPS);
  - otherwise the CPU.
- The reference run took about 3 hours on an Apple M4 (MPS).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Follow [`data/README.md`](data/README.md) to place the dataset in
`data/RDD_SPLIT/`. The data is not stored in this repository.

## Run the baseline

From the repository root:

```bash
python src/baseline_pipeline.py
```

This command creates **`reports/baseline_metrics.json`**. The file holds the
mAP50 value, the secondary metrics, the data-check counts and the limitation.
Training runs and weights are written to `reports/runs/detect/`, which is not
committed.

Options:

- Add `--epochs 2` for a quick check that the whole pipeline runs. Its score
  is not the reported baseline.
- Add `--device cpu` to force the CPU.

## Repository layout

```
data/README.md                          data source and placement
docs/project_brief.md                   project brief
docs/split_strategy.md                  evaluation decision
src/baseline_pipeline.py                runnable baseline
reports/baseline_metrics.json           result written by the pipeline
notebook/road_damage_detection.ipynb    EDA and the three-model comparison
```

## Reference result from the notebook

| Model   | Test mAP50 | Precision | Recall |
|---------|-----------|-----------|--------|
| YOLOv8n | 0.453     | 0.516     | 0.455  |
| YOLOv8s | 0.477     | 0.552     | 0.466  |
| YOLO11s | 0.461     | 0.522     | 0.472  |

The script draws its sample in sorted file order, so that the sample is the
same on every machine. As a result it picks slightly different images from
the notebook, and its YOLOv8n score will be close to 0.453 but not exactly
equal. The value that counts for the baseline is the one this script writes
to the JSON file.
