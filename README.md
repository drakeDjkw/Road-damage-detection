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
git clone https://github.com/drakeDjkw/Road-damage-detection.git
cd Road-damage-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place the dataset in data/RDD_SPLIT/ as data/README.md describes
python src/baseline_pipeline.py --epochs 25
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

- Add `--epochs 5` for a quick check that the whole pipeline runs. Its score
  is not the reported baseline.
- Add `--device cpu` to force the CPU.

## Lab 04 inference demo

The demo sends one road image through the Lab 02 baseline model and records
two attempts: a normal one and a simulated "inference unavailable" one.

It needs the trained baseline weights at
`reports/runs/detect/baseline_YOLOv8n/weights/best.pt`. That file is
created by `python src/baseline_pipeline.py` and is not stored in Git.
Optional: `--section-id` sets the road-section ID (default `demo-section-01`).

From the repository root:

```bash
python src/inference_demo.py --input-ref data/RDD_SPLIT/test/images/Japan_012962.jpg
```

This creates **`reports/inference_demo.jsonl`** with two lines:

| Line | status | simulated | action |
|---|---|---|---|
| 1 | `ok` or `review` | `false` | add detections to the section summary for the planner / send frame to the planner's review queue |
| 2 | `unavailable` | `true` | mark frame as not checked and retry in the next batch run |

**Decision rule (decision-v1, provisional):** boxes with confidence ≥ 0.50
are confirmed. Boxes between 0.25 and 0.50 send the frame to review. A
frame that fails the input check is rejected with a reason and also goes to
review, so it is never counted as damage-free road.

The path is drawn in
[`docs/inference_architecture.drawio`](docs/inference_architecture.drawio);
open it at <https://app.diagrams.net>.

## Lab 05 input validation

Before prediction, every request is checked against
[`starter/config/input_contract.json`](starter/config/input_contract.json) by
[`starter/src/input_validation.py`](starter/src/input_validation.py).

| Result | What happens | `action` |
|---|---|---|
| pass | the Lab 04 prediction runs | `predict` |
| warning | no prediction; a person reviews the frame | `route_for_review` |
| error | no prediction; the input is rejected | `reject_input` |

If several rules fail, the most severe result wins: error > warning > pass.

| Rule | Type | Severity | Why |
|---|---|---|---|
| `required_fields_present` | structural | error | the request must name the file, format, size and source |
| `format_supported` | structural | error | the baseline only reads .jpg/.jpeg/.png |
| `dimensions_valid` | structural | error | width and height must be positive whole numbers |
| `section_id_present` | project-specific | error | Lab 03: a frame without a road-section ID is rejected |
| `source_approved` | project-specific | warning | Lab 03: cameras outside the six RDD2022 sources are out of scope |
| `size_in_evaluated_range` | project-specific | warning | the training sample's smallest side was 512 px; smaller frames were never evaluated |
| `preprocessing_identity_matches` | runtime | error | the model was evaluated with `letterbox-640-v1` (set in `src/inference_demo.py`, not by the request) |

These are request-level checks. They test the values supplied in the request,
not the file itself: the validator does not open, find or decode the image.

From the repository root:

```bash
python src/inference_demo.py
```

The Lab 05 tests run from `starter/`:

```bash
cd starter
python -m unittest tests/test_input_validation.py
```

The first command runs three requests and writes
**`reports/validation_demo.jsonl`**: one pass (prediction runs), one warning
(`China_Drone` source) and one error (`gif` format and no section ID, plus an
unknown source). Each line shows every check with its observed value and
status, the combined status, the action and `prediction_called`. All of these
are linked by one `request_id`.

The passing request needs the baseline weights, as in Lab 04. The tests use a
stub instead of the model and need no weights or data.

## Lab 06 cloud service

`starter/` holds a FastAPI service that puts the Lab 02 model behind an HTTP
API on a Yandex Cloud VM (Ubuntu 24.04, 2 vCPU, 4 GB RAM, CPU inference).

Each upload passes two layers of checks before the model runs:

| Layer | Contract | Code | What it checks |
| --- | --- | --- | --- |
| File | `config/upload_contract.json` | `image_validation.py` | content type, size ≤ 10 MB, the bytes decode as JPEG/PNG, colour mode, at least 64×64 px |
| Project (Lab 05) | `config/input_contract.json` | `src/input_validation.py` | section ID, approved camera source, smallest side ≥ 512 px, preprocessing `letterbox-640-v1` |

The Lab 05 validator now receives the format, width and height that the
server decoded from the file, not values the client claims.

| Result | HTTP | `action` | Model called |
| --- | --- | --- | --- |
| pass | 200 | `predict` | yes, then decision-v1 sorts the boxes (≥ 0.50 confirmed, 0.25–0.50 → review queue) |
| warning | 200 | `route_for_review` | no |
| error | 400 | `reject_input` | no |

Every request appends one line to `starter/reports/yolo_predictions.jsonl`
with the request ID, section ID, decoded image properties, every check,
`status`, `action`, `prediction_called` and `prediction`.

### Start the service on the VM

The weights are not in Git. Copy them from the computer that trained the model:

```bash
# on your own computer, from the repository root
ssh <username>@<public-ip> "mkdir -p ~/Road-damage-detection/starter/models"
scp reports/runs/detect/baseline_YOLOv8n/weights/best.pt \
    <username>@<public-ip>:~/Road-damage-detection/starter/models/best.pt
```

Then on the VM:

```bash
git clone https://github.com/drakeDjkw/Road-damage-detection.git   # or: git pull
cd Road-damage-detection/starter
sudo apt update && sudo apt install -y python3-venv libgl1 tmux
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir -r requirements.txt
cp .env.example .env
python3 -m unittest tests/test_app.py tests/test_input_validation.py
tmux new -s api
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Run uvicorn from inside `starter/`. Detach from tmux with `Ctrl+B`, then `D`;
the service keeps running if SSH disconnects. Port 8000 must be open in the
VM's security group.

### Demo requests

Open `http://<public-ip>:8000/docs` → `POST /predict` → **Try it out**:

| Request | `request_id` | `section_id` | `source` | file | Expected |
| --- | --- | --- | --- | --- | --- |
| valid | `req-cloud-001` | `demo-section-01` | `Japan` | `Japan_012962.jpg` | 200, `predict`, detections |
| rejected | `req-cloud-002` | `demo-section-01` | `Japan` | any `.txt` | 400, `reject_input`, `prediction_called: false` |

Or from another computer:

```bash
curl -F request_id=req-cloud-001 -F section_id=demo-section-01 -F source=Japan \
     -F file=@Japan_012962.jpg http://<public-ip>:8000/predict
```

Then on the VM: `tail -n 2 reports/yolo_predictions.jsonl`.

### Evidence

- [`reports/cloud_demo.jsonl`](reports/cloud_demo.jsonl): trace of one valid request (`req-cloud-001` → `predict`) and one rejected request (`req-cloud-002` → `reject_input`).
- [`docs/lab06_browser_health.png`](docs/lab06_browser_health.png): `GET /health` opened from another computer, with the trained model loaded.
- [`docs/lab06_browser_predict.png`](docs/lab06_browser_predict.png): `POST /predict` sent from another computer through `/docs` (`req-browser-001`, HTTP 200).

These detections come from the Lab 02 baseline. They show that the
deployment path works; they are not new evaluation results.

## Repository layout

```
data/README.md                          data source and placement
docs/project_brief.md                   project brief
docs/split_strategy.md                  evaluation decision
docs/project_decisions.md               Lab 03 acceptance decisions
docs/inference_architecture.drawio      Lab 04 inference path diagram
starter/src/baseline_pipeline.py        runnable baseline
starter/src/inference_demo.py           Lab 04 inference demo + Lab 05 validation demo
starter/src/input_validation.py         Lab 05 input validator
starter/config/input_contract.json      Lab 05 input contract (project rules)
starter/config/upload_contract.json     Lab 06 upload contract (file checks)
starter/image_validation.py             Lab 06 file checks
starter/app.py                          Lab 06 FastAPI service
starter/tests/test_input_validation.py  Lab 05 tests
starter/tests/test_app.py               Lab 06 tests
reports/baseline_metrics.json           result written by the pipeline
reports/inference_demo.jsonl            records written by the demo
reports/validation_demo.jsonl           Lab 05 validation trace
notebook/road_damage_detection.ipynb    EDA and the three-model comparison
```

## Baseline result

Created by `python src/baseline_pipeline.py` (Apple M4, MPS, 50 epochs):

| Metric | Value |
|---|---|
| **mAP50 (primary)** | **0.457** |
| Precision | 0.557 |
| Recall | 0.449 |
| mAP50-95 | 0.220 |

Per class mAP50: other corruption 0.614, alligator crack 0.507,
longitudinal crack 0.452, transverse crack 0.397, pothole 0.316.

**Limitation:** the split is random at the image level, so frames from the
same drive may appear in both training and test, which can make the score
optimistic. Potholes are the rarest class and are detected worst.