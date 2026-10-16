# Road Damage Detection — Team Surface 3

Yikwa Sandrakh, Wenda Waipo, Agata Melani
Artificial Intelligence Implementation Technologies · Baikal School of BRICS, INRTU

The project detects and classifies road surface defects in images from
vehicle cameras. It finds five classes: longitudinal crack, transverse crack,
alligator crack, other corruption and pothole.

## Lab 02 baseline

| | |
|---|---|
| Data | RDD2022 sample (seed 42): 10,000 train / 2,000 val / 2,000 test images |
| Baseline | YOLOv8n pretrained on COCO, fine-tuned with default settings (640 px, 50 epochs) |
| Primary metric | mAP50 on the held-out 2,000-image test sample |
| Limitation | Image-level random split, so frames from the same drive may leak between train and test; potholes are rare |

## Environment

- Python 3.12 (tested with Python 3.12.13, Ultralytics 8.4.92 and
  torch 2.10.0+cu128)
- An NVIDIA GPU is strongly recommended. The reference run used a Tesla T4
  and took about 2 hours. Training on a CPU will work but is very slow.

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
The training runs are written to `runs/`.

Options:

- Add `--device cpu` to force CPU training.
- Add `--epochs 5` for a quick check that the whole pipeline runs. Its score
  is not the reported baseline.

## Repository layout

```
data/README.md              data source and placement
docs/split_strategy.md      evaluation decision
src/baseline_pipeline.py    runnable baseline
reports/baseline_metrics.json   result written by the pipeline
notebooks/road_damage_detection.ipynb   EDA and the three-model comparison
```

## Reference result

The earlier notebook run used the same configuration and reached
**mAP50 = 0.512** on the test sample (precision 0.585, recall 0.501). That
run's sample was drawn in a different file order, so a rerun of this script
will give a close value but not exactly the same one. The value that counts
for the baseline is the one this script writes to the JSON file.
