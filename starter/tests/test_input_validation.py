"""Lab 05 tests for src/input_validation.py.

Run from the repository root:
    python3 -m unittest tests/test_input_validation.py

They use a recording stub instead of the model, so they only prove whether
prediction was called, not what it would return.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from input_validation import combine, load_contract, route_request, validate_request  # noqa: E402

PREPROCESSING_ID = "letterbox-640-v1"   # same value as src/inference_demo.py

GOOD_REQUEST = {
    "input_ref": "data/RDD_SPLIT/test/images/Japan_012962.jpg",
    "format": "jpg", "width": 600, "height": 600,
    "source": "Japan", "section_id": "JP-R12-km034",
}


class RecordingPredictor:
    """Stands in for project_predict(); remembers whether it was called."""

    def __init__(self):
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        return {"status": "ok", "result": "stub", "action": "stub"}


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_contract()

    def test_error_overrides_warning(self):
        # GIF (error) + missing section ID (error) + unknown source (warning)
        request = {**GOOD_REQUEST, "format": "gif", "section_id": "",
                   "source": "dashcam-unknown"}
        checks = validate_request(request, self.contract, PREPROCESSING_ID)
        statuses = {c["rule"]: c["status"] for c in checks}
        self.assertEqual(statuses["source_approved"], "warning")
        self.assertEqual(statuses["format_supported"], "error")

        predictor = RecordingPredictor()
        record = route_request(request, predictor, self.contract, PREPROCESSING_ID)
        self.assertEqual(record["validation_status"], "error")
        self.assertEqual(record["action"], "reject_input")
        self.assertFalse(record["prediction_called"])
        self.assertEqual(predictor.calls, 0)

    def test_warning_does_not_call_prediction(self):
        request = {**GOOD_REQUEST, "source": "China_Drone"}
        predictor = RecordingPredictor()
        record = route_request(request, predictor, self.contract, PREPROCESSING_ID)
        self.assertEqual(record["validation_status"], "warning")
        self.assertEqual(record["action"], "route_for_review")
        self.assertFalse(record["prediction_called"])
        self.assertIsNone(record["prediction"])
        self.assertEqual(predictor.calls, 0)

    def test_pass_calls_prediction(self):
        predictor = RecordingPredictor()
        record = route_request(GOOD_REQUEST, predictor, self.contract, PREPROCESSING_ID)
        self.assertEqual(record["validation_status"], "pass")
        self.assertEqual(record["action"], "predict")
        self.assertTrue(record["prediction_called"])
        self.assertEqual(predictor.calls, 1)

    def test_small_frame_is_warning(self):
        request = {**GOOD_REQUEST, "width": 320, "height": 240}
        checks = validate_request(request, self.contract, PREPROCESSING_ID)
        self.assertEqual(combine(checks), ("warning", "route_for_review"))

    def test_wrong_preprocessing_is_error(self):
        checks = validate_request(GOOD_REQUEST, self.contract, "resize-416-v1")
        self.assertEqual(combine(checks), ("error", "reject_input"))


if __name__ == "__main__":
    unittest.main()
