"""Lab 06: road damage inference API for the cloud VM.

Every upload goes through two layers of checks before the model runs:

1. File checks (config/upload_contract.json, image_validation.py):
   the bytes must be a JPEG or PNG that actually decodes.
2. Project checks (config/input_contract.json, src/input_validation.py):
   the Lab 05 rules - road-section ID, approved camera source,
   evaluated size range and preprocessing identity.

The most severe result wins: error > warning > pass.

    pass    -> run the YOLOv8n road damage model     (action: predict)
    warning -> no prediction, a person reviews it    (action: route_for_review)
    error   -> no prediction, the input is rejected  (action: reject_input)

Detections from a passing request are then sorted by decision-v1 (Lab 04):
boxes with confidence >= 0.50 are confirmed; any box between 0.25 and 0.50
sends the frame to the review queue.

Every request, including rejected ones, leaves one JSONL record.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from image_validation import load_contract, validate_image_upload
from src.input_validation import validate_request


BASE_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)
load_dotenv(BASE_DIR / ".env")

ROAD_DAMAGE_CLASSES = [
    "longitudinal crack",
    "transverse crack",
    "alligator crack",
    "other corruption",
    "pothole",
]
SEVERITY = {"pass": 0, "warning": 1, "error": 2}
ACTION = {"pass": "predict", "warning": "route_for_review", "error": "reject_input"}
HTTP_STATUS = {"pass": 200, "warning": 200, "error": 400}
CONFIRM_THRESHOLD = 0.50  # decision-v1 (Lab 04)


def resolve_model_path(model_name: str) -> str:
    """Resolve a relative weights path against this folder, not the shell's cwd."""
    candidate = Path(model_name)
    if not candidate.is_absolute() and (BASE_DIR / candidate).exists():
        return str((BASE_DIR / candidate).resolve())
    return model_name


class YoloPredictor:
    """Load the trained road damage model on the first valid request."""

    def __init__(self, model_name: str, confidence: float, image_size: int) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self.image_size = image_size
        self._model: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def __call__(self, image: Image.Image) -> dict[str, Any]:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(resolve_model_path(self.model_name))

        started = perf_counter()
        result = self._model.predict(
            source=image,
            conf=self.confidence,
            imgsz=self.image_size,
            device="cpu",
            verbose=False,
        )[0]
        latency_ms = round((perf_counter() - started) * 1000, 1)

        detections: list[dict[str, Any]] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": round(float(box.conf.item()), 4),
                        "box_xyxy": [round(float(value), 1) for value in box.xyxy[0]],
                    }
                )

        return {
            "model": Path(self.model_name).name,
            "confidence_threshold": self.confidence,
            "latency_ms": latency_ms,
            "object_count": len(detections),
            "detections": detections,
        }


def route_upload(
    file_checks: list[dict[str, Any]],
    image_metadata: dict[str, Any],
    filename: str | None,
    section_id: str,
    source: str,
    project_contract: dict[str, Any],
    preprocessing_id: str,
) -> tuple[list[dict[str, Any]], str, str]:
    """Combine the file checks with the Lab 05 project checks.

    The Lab 05 validator receives the values the server decoded from the
    file (format, width, height), not values the client claims.
    """
    decoded_format = image_metadata.get("format")
    request = {
        "input_ref": filename or None,
        "format": decoded_format.lower() if decoded_format else None,
        "width": image_metadata.get("width"),
        "height": image_metadata.get("height"),
        "source": source.strip() or None,
        "section_id": section_id.strip() or None,
    }
    checks = file_checks + validate_request(request, project_contract, preprocessing_id)
    status = max((c["status"] for c in checks), key=SEVERITY.__getitem__, default="pass")
    return checks, status, ACTION[status]


def apply_decision(prediction: dict[str, Any]) -> dict[str, Any]:
    """decision-v1: confirmed boxes vs. boxes that need a person to look."""
    detections = prediction.get("detections", [])
    confirmed = [d for d in detections if d["confidence"] >= CONFIRM_THRESHOLD]
    uncertain = [d for d in detections if d["confidence"] < CONFIRM_THRESHOLD]
    if uncertain:
        frame_status, frame_action = "review", "send_frame_to_review_queue"
    else:
        frame_status, frame_action = "ok", "add_detections_to_section_summary"
    return {
        "rule": "decision-v1",
        "confirm_threshold": CONFIRM_THRESHOLD,
        "confirmed_count": len(confirmed),
        "uncertain_count": len(uncertain),
        "frame_status": frame_status,
        "frame_action": frame_action,
    }


def append_jsonl(record: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_app(
    evidence_path: Path | None = None,
    predictor: Callable[[Image.Image], dict[str, Any]] | None = None,
) -> FastAPI:
    environment = os.getenv("ENVIRONMENT", "classroom")
    service_version = os.getenv("SERVICE_VERSION", "road-damage-api-v1")
    model_name = os.getenv("MODEL_NAME", "models/best.pt")
    confidence = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
    image_size = int(os.getenv("IMAGE_SIZE", "640"))
    # Ultralytics letterboxes to imgsz; any size other than 640 fails the
    # Lab 05 preprocessing_identity_matches rule on purpose.
    preprocessing_id = f"letterbox-{image_size}-v1"
    upload_contract = load_contract(BASE_DIR / "config" / "upload_contract.json")
    project_contract = load_contract(BASE_DIR / "config" / "input_contract.json")
    trace_path = evidence_path or BASE_DIR / os.getenv(
        "EVIDENCE_PATH", "reports/yolo_predictions.jsonl"
    )
    model_predict = predictor or YoloPredictor(model_name, confidence, image_size)

    application = FastAPI(
        title="Road Damage Detection API",
        version=service_version,
        description=(
            "Upload a JPEG or PNG road image with its road-section ID and camera source. "
            "The service checks the decoded file and the Lab 05 project rules, runs the "
            "5-class road damage YOLOv8n model on CPU only when every check passes, "
            "and records the decision as one JSONL line."
        ),
    )

    @application.get("/")
    def root() -> dict[str, str]:
        return {
            "service": "Road Damage Detection API",
            "docs": "/docs",
            "health": "/health",
        }

    @application.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "environment": environment,
            "service_version": service_version,
            "model": Path(model_name).name,
            "model_loaded": getattr(model_predict, "loaded", False),
            "classes": ROAD_DAMAGE_CLASSES,
            "upload_contract": upload_contract["contract_version"],
            "project_contract": project_contract["contract_version"],
            "preprocessing_id": preprocessing_id,
        }

    @application.post("/predict", response_model=None)
    async def predict(
        file: UploadFile = File(...),
        request_id: str = Form(default=""),
        section_id: str = Form(default="", description="Road-section ID, e.g. demo-section-01"),
        source: str = Form(default="", description="Camera source, e.g. Japan"),
    ) -> Any:
        resolved_request_id = request_id.strip() or f"req-{uuid4().hex[:8]}"
        maximum = int(upload_contract["maximum_upload_bytes"])
        data = await file.read(maximum + 1)
        file_checks, image, image_metadata = validate_image_upload(
            data, file.content_type, upload_contract
        )
        checks, status, action = route_upload(
            file_checks,
            image_metadata,
            file.filename,
            section_id,
            source,
            project_contract,
            preprocessing_id,
        )

        base_record: dict[str, Any] = {
            "request_id": resolved_request_id,
            "section_id": section_id.strip() or None,
            "source": source.strip() or None,
            "filename": file.filename,
            "upload_contract": upload_contract["contract_version"],
            "project_contract": project_contract["contract_version"],
            "preprocessing_id": preprocessing_id,
            "image": image_metadata,
            "checks": checks,
            "failed_rules": [c["rule"] for c in checks if c["status"] != "pass"],
            "status": status,
            "action": action,
        }

        if status != "pass" or image is None:
            record = {**base_record, "prediction_called": False, "prediction": None}
            append_jsonl(record, trace_path)
            return JSONResponse(status_code=HTTP_STATUS[status], content=record)

        try:
            prediction = model_predict(image)
        except Exception as error:
            LOGGER.exception("Inference failed for request %s", resolved_request_id)
            record = {
                **base_record,
                "status": "unavailable",
                "action": "mark_frame_not_checked_and_retry",
                "prediction_called": True,
                "prediction": None,
                "error_type": type(error).__name__,
            }
            append_jsonl(record, trace_path)
            return JSONResponse(status_code=503, content=record)

        record = {
            **base_record,
            "prediction_called": True,
            "prediction": prediction,
            "decision": apply_decision(prediction),
        }
        append_jsonl(record, trace_path)
        return record

    return application


app = create_app()
