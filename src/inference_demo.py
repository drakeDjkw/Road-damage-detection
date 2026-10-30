"""Lab 04: connect one project input to a traceable response.

Road Damage Detection (Team Surface 3). One road image goes through the Lab 02
YOLOv8n baseline; the response and a simulated "unavailable" attempt are saved
to reports/inference_demo.jsonl.

Run from the repository root:
    python src/inference_demo.py --input-ref data/RDD_SPLIT/test/images/Japan_012962.jpg
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]

MODEL_VERSION = "baseline-v1"   # YOLOv8n from Lab 02 (test mAP50 0.457)
RULE_VERSION = "decision-v1"    # confidence bands below

# Actions decided in Lab 03 (docs/project_decisions.md).
NORMAL_ACTION = "add detections to the section summary for the planner"
REVIEW_ACTION = "send frame to the planner's review queue"
UNAVAILABLE_ACTION = "mark frame as not checked and retry in the next batch run"

# Same settings as the Lab 02 baseline (src/baseline_pipeline.py).
WEIGHTS = ROOT / "reports" / "runs" / "detect" / "baseline_YOLOv8n" / "weights" / "best.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
IMGSZ = 640

# decision-v1 (provisional: Lab 03 left the cutoff values open).
MIN_CONF = 0.25      # boxes below this are ignored (Ultralytics predict default)
CONFIRM_CONF = 0.50  # boxes at or above this are confirmed; 0.25-0.50 -> review
MIN_SIDE_PX = 64     # smaller images cannot show the road surface

SECTION_ID = "demo-section-01"  # set from --section-id in main()

_model = None


def check_input(path: Path) -> str | None:
    """First input check. Returns None if the image is usable, else the reason."""
    if not path.is_file():
        return "file not found"
    if path.suffix.lower() not in IMAGE_EXTS:   # same rule as the baseline
        return f"unsupported file type {path.suffix}"
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
    except Exception:
        return "image cannot be decoded"
    if min(width, height) < MIN_SIDE_PX:
        return f"image too small ({width}x{height})"
    return None


def load_model():
    """Load the Lab 02 baseline weights once."""
    global _model
    if _model is None:
        if not WEIGHTS.is_file():
            raise FileNotFoundError(f"baseline weights not found: {WEIGHTS}")
        from ultralytics import YOLO
        _model = YOLO(str(WEIGHTS))
    return _model


def project_predict(input_ref: str) -> dict[str, Any]:
    """Return one prediction by reusing the Lab 02 baseline model.

    Shape: {"result": {...}, "needs_review": bool}
    """
    path = Path(input_ref)
    if not path.is_absolute():
        path = ROOT / path

    reason = check_input(path)
    if reason:
        # Lab 03: an invalid frame is rejected with a reason and never counted
        # as damage-free road, so a person must look at it.
        return {"result": {"section_id": SECTION_ID, "rejected": reason},
                "needs_review": True}

    model = load_model()
    # Preprocessing (letterbox resize to 640 px, normalisation) happens inside
    # predict(), the same step the baseline's val() uses before computing mAP50.
    pred = model.predict(source=str(path), imgsz=IMGSZ, conf=MIN_CONF, verbose=False)[0]

    detections = []
    for cls_id, conf, box in zip(pred.boxes.cls.tolist(),
                                 pred.boxes.conf.tolist(),
                                 pred.boxes.xyxy.tolist()):
        detections.append({
            "class": model.names[int(cls_id)],
            "confidence": round(float(conf), 3),
            "box_xyxy": [round(float(v), 1) for v in box],
            "confirmed": float(conf) >= CONFIRM_CONF,
        })

    n_review = sum(not d["confirmed"] for d in detections)
    result = {
        "section_id": SECTION_ID,
        "n_detections": len(detections),
        "n_confirmed": len(detections) - n_review,
        "n_review": n_review,
        "detections": detections,
    }
    # decision-v1: any box in the 0.25-0.50 band sends the frame to review.
    return {"result": result, "needs_review": n_review > 0}


def run_attempt(input_ref: str, *, simulate_unavailable: bool) -> dict[str, Any]:
    started = time.perf_counter()

    if simulate_unavailable:
        status = "unavailable"
        result = None
        action = UNAVAILABLE_ACTION
    else:
        try:
            prediction = project_predict(input_ref)
        except Exception as error:   # real failure, e.g. weights missing
            print(f"prediction failed: {error}")
            status, result, action = "unavailable", None, UNAVAILABLE_ACTION
        else:
            result = prediction["result"]
            needs_review = bool(prediction.get("needs_review", False))
            status = "review" if needs_review else "ok"
            action = REVIEW_ACTION if needs_review else NORMAL_ACTION

    return {
        "request_id": str(uuid4()),
        "input_ref": input_ref,
        "status": status,
        "result": result,
        "model_version": MODEL_VERSION,
        "rule_version": RULE_VERSION,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "action": action,
        "simulated": simulate_unavailable,
    }


def write_evidence(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    global SECTION_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ref", required=True,
                        help="Path to one road image, relative to the repository root")
    parser.add_argument("--section-id", default=SECTION_ID,
                        help="Road-section ID the frame belongs to")
    parser.add_argument("--output", default="reports/inference_demo.jsonl")
    args = parser.parse_args()
    SECTION_ID = args.section_id

    # Load the model before timing, so latency_ms measures one prediction
    # rather than the one-off model load.
    try:
        load_model()
    except Exception as error:
        print(f"model not loaded: {error}")

    records = [
        run_attempt(args.input_ref, simulate_unavailable=False),
        run_attempt(args.input_ref, simulate_unavailable=True),
    ]
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    write_evidence(records, output)

    for record in records:
        print(json.dumps(record, ensure_ascii=False))
    print(f"Saved {len(records)} records to {output}")


if __name__ == "__main__":
    main()
