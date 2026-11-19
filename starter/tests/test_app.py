"""Lab 06 tests for the road damage API.

Run from starter/:
    python3 -m unittest tests/test_app.py

A stub replaces the YOLO model, so these tests need no weights and no
download. They prove which requests reach prediction, not what it returns.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import ROAD_DAMAGE_CLASSES, create_app  # noqa: E402


def stub_prediction(confidence: float) -> dict:
    return {
        "model": "best.pt",
        "confidence_threshold": 0.25,
        "latency_ms": 1.0,
        "object_count": 1,
        "detections": [
            {
                "class_id": 4,
                "class_name": "pothole",
                "confidence": confidence,
                "box_xyxy": [1.0, 2.0, 30.0, 40.0],
            }
        ],
    }


class RoadDamageApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.trace_path = Path(self.temp_dir.name) / "trace.jsonl"
        self.predict_calls = 0
        self.stub_confidence = 0.9

        def fake_predict(_: Image.Image) -> dict:
            self.predict_calls += 1
            return stub_prediction(self.stub_confidence)

        self.client = TestClient(create_app(self.trace_path, fake_predict))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def image_bytes(width: int = 640, height: int = 640) -> bytes:
        buffer = BytesIO()
        Image.new("RGB", (width, height), color=(40, 80, 120)).save(buffer, format="JPEG")
        return buffer.getvalue()

    def request(
        self,
        request_id: str = "req-test",
        data: bytes | None = None,
        filename: str = "Japan_012962.jpg",
        content_type: str = "image/jpeg",
        section_id: str = "demo-section-01",
        source: str = "Japan",
    ):
        payload = self.image_bytes() if data is None else data
        return self.client.post(
            "/predict",
            data={"request_id": request_id, "section_id": section_id, "source": source},
            files={"file": (filename, payload, content_type)},
        )

    def trace(self) -> list[dict]:
        return [json.loads(line) for line in self.trace_path.read_text().splitlines()]

    def test_health_names_service_and_model(self) -> None:
        body = self.client.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["classes"], ROAD_DAMAGE_CLASSES)
        self.assertEqual(body["preprocessing_id"], "letterbox-640-v1")

    def test_pass_calls_prediction(self) -> None:
        response = self.request()
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "pass")
        self.assertEqual(body["action"], "predict")
        self.assertTrue(body["prediction_called"])
        self.assertIn(body["prediction"]["detections"][0]["class_name"], ROAD_DAMAGE_CLASSES)
        self.assertEqual(body["decision"]["frame_status"], "ok")
        self.assertEqual(self.predict_calls, 1)

    def test_uncertain_box_sends_frame_to_review(self) -> None:
        self.stub_confidence = 0.30
        body = self.request().json()
        self.assertEqual(body["decision"]["frame_status"], "review")
        self.assertEqual(body["decision"]["uncertain_count"], 1)

    def test_text_file_stops_before_prediction(self) -> None:
        response = self.request(
            request_id="req-invalid",
            data=b"not an image",
            filename="notes.txt",
            content_type="text/plain",
        )
        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["status"], "error")
        self.assertEqual(body["action"], "reject_input")
        self.assertFalse(body["prediction_called"])
        self.assertIn("image_decodes", body["failed_rules"])
        self.assertEqual(self.predict_calls, 0)

    def test_missing_section_id_is_rejected(self) -> None:
        response = self.request(section_id="")
        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["action"], "reject_input")
        self.assertIn("section_id_present", body["failed_rules"])
        self.assertEqual(self.predict_calls, 0)

    def test_unapproved_source_goes_to_review(self) -> None:
        body = self.request(source="China_Drone").json()
        self.assertEqual(body["status"], "warning")
        self.assertEqual(body["action"], "route_for_review")
        self.assertFalse(body["prediction_called"])
        self.assertEqual(self.predict_calls, 0)

    def test_small_image_goes_to_review(self) -> None:
        body = self.request(data=self.image_bytes(320, 240)).json()
        self.assertEqual(body["action"], "route_for_review")
        self.assertIn("size_in_evaluated_range", body["failed_rules"])
        self.assertEqual(self.predict_calls, 0)

    def test_each_request_leaves_one_trace_record(self) -> None:
        self.request(request_id="req-pass")
        self.request(request_id="req-bad", data=b"x", content_type="text/plain")
        records = self.trace()
        self.assertEqual([r["request_id"] for r in records], ["req-pass", "req-bad"])
        self.assertTrue(records[0]["prediction_called"])
        self.assertFalse(records[1]["prediction_called"])
        self.assertIsNone(records[1]["prediction"])


if __name__ == "__main__":
    unittest.main()
