"""Lab 05: executable input assurance for the road damage inference demo.

Checks a request against config/input_contract.json before prediction and
decides what happens next:

    pass    -> call the prediction function            (action: predict)
    warning -> do not predict, a person reviews it      (action: route_for_review)
    error   -> do not predict, the input is rejected    (action: reject_input)

When several rules fail, the most severe result wins: error > warning > pass.

These are request-level checks. They look at the values supplied in the
request (input_ref, format, width, height, source, section_id) and at the
runtime preprocessing identity. They do not open the file, prove it exists or
decode it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config" / "input_contract.json"

SEVERITY = {"pass": 0, "warning": 1, "error": 2}
ACTION = {"pass": "predict", "warning": "route_for_review", "error": "reject_input"}


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _check(rule: str, observed: Any, ok: bool, contract: dict[str, Any]) -> dict[str, Any]:
    """One check result in the agreed shape."""
    status = "pass" if ok else contract["rules"][rule]["severity"]
    return {"rule": rule, "observed": observed, "status": status, "action": ACTION[status]}


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_request(request: dict[str, Any], contract: dict[str, Any],
                     runtime_preprocessing_id: str) -> list[dict[str, Any]]:
    """Run every contract rule on one request. Returns one result per rule."""
    checks = []

    # Structural: required fields are present and non-empty.
    missing = [f for f in contract["required_fields"]
               if request.get(f) in (None, "")]
    checks.append(_check("required_fields_present",
                         {"missing": missing}, not missing, contract))

    # Structural: format is one the baseline reads.
    fmt = str(request.get("format", "")).lower().lstrip(".")
    checks.append(_check("format_supported", fmt or None,
                         fmt in contract["allowed_formats"], contract))

    # Structural: width and height describe a real image.
    width, height = request.get("width"), request.get("height")
    dims_ok = _is_positive_int(width) and _is_positive_int(height)
    checks.append(_check("dimensions_valid", {"width": width, "height": height},
                         dims_ok, contract))

    # Project-specific (Lab 03): every frame must belong to a road section.
    section_id = request.get("section_id")
    checks.append(_check("section_id_present", section_id,
                         isinstance(section_id, str) and section_id.strip() != "",
                         contract))

    # Project-specific warning (Lab 03): camera source must be one of the
    # RDD2022 sources the model was trained and evaluated on.
    source = request.get("source")
    checks.append(_check("source_approved", source,
                         source in contract["approved_sources"], contract))

    # Warning: frames smaller than the evaluated range go to review.
    # Only evaluated when the dimensions themselves are valid.
    if dims_ok:
        min_side = min(width, height)
        checks.append(_check("size_in_evaluated_range", {"min_side_px": min_side},
                             min_side >= contract["evaluated_min_side_px"], contract))

    # The runtime preprocessing must be the one the model was evaluated with.
    checks.append(_check("preprocessing_identity_matches",
                         {"runtime": runtime_preprocessing_id,
                          "expected": contract["expected_preprocessing_id"]},
                         runtime_preprocessing_id == contract["expected_preprocessing_id"],
                         contract))
    return checks


def combine(checks: list[dict[str, Any]]) -> tuple[str, str]:
    """Most severe status wins: error > warning > pass."""
    status = max((c["status"] for c in checks), key=SEVERITY.__getitem__, default="pass")
    return status, ACTION[status]


def route_request(request: dict[str, Any],
                  predict_fn: Callable[[dict[str, Any]], dict[str, Any]],
                  contract: dict[str, Any],
                  runtime_preprocessing_id: str) -> dict[str, Any]:
    """Validate one request, call prediction only on pass, return one trace record."""
    request_id = str(uuid4())
    started = time.perf_counter()

    checks = validate_request(request, contract, runtime_preprocessing_id)
    status, action = combine(checks)

    prediction_called = False
    prediction = None
    if status == "pass":
        prediction_called = True
        prediction = predict_fn(request)

    return {
        "request_id": request_id,
        "input_ref": request.get("input_ref"),
        "contract_version": contract["contract_version"],
        "preprocessing_id": runtime_preprocessing_id,
        "checks": checks,
        "failed_rules": [c["rule"] for c in checks if c["status"] != "pass"],
        "validation_status": status,
        "action": action,
        "prediction_called": prediction_called,
        "prediction": prediction,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }
