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

from app import create_app

ROAD_DAMAGE_CLASSES = [
    "longitudinal crack",
    "transverse crack",
    "alligator crack",
    "other corruption",
    "pothole",
]


class CloudDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.trace_path = Path(self.temp_dir.name) / "trace.jsonl"
        self.predict_calls = 0

        def fake_predict(_: Image.Image) -> dict:
            self.predict_calls += 1
            return {
                "model": "best.pt",
                "confidence_threshold": 0.25,
                "latency_ms": 1.0,
                "object_count": 1,
                "detections": [
                    {
                        "class_id": 4,
                        "class_name": "pothole",
                        "confidence": 0.9,
                        "box_xyxy": [1.0, 2.0, 30.0, 40.0],
                    }
                ],
            }

        self.client = TestClient(create_app(self.trace_path, fake_predict))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def image_bytes(width: int = 640, height: int = 640) -> bytes:
        buffer = BytesIO()
        Image.new("RGB", (width, height), color=(40, 80, 120)).save(
            buffer,
            format="JPEG",
        )
        return buffer.getvalue()

    def request(
        self,
        request_id: str = "req-test",
        data: bytes | None = None,
        filename: str = "Japan_012962.jpg",
        content_type: str = "image/jpeg",
    ):
        payload = self.image_bytes() if data is None else data
        return self.client.post(
            "/predict",
            data={"request_id": request_id},
            files={"file": (filename, payload, content_type)},
        )

    def test_pass_calls_prediction(self) -> None:
        response = self.request()
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "pass")
        self.assertTrue(body["prediction_called"])
        self.assertEqual(body["prediction"]["object_count"], 1)
        detection = body["prediction"]["detections"][0]
        self.assertIn(detection["class_name"], ROAD_DAMAGE_CLASSES)
        self.assertEqual(self.predict_calls, 1)

    def test_invalid_upload_stops_before_prediction(self) -> None:
        response = self.request(
            request_id="req-invalid",
            data=b"not an image",
            filename="notes.txt",
            content_type="text/plain",
        )
        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["status"], "error")
        self.assertFalse(body["prediction_called"])
        self.assertEqual(self.predict_calls, 0)

    def test_each_request_leaves_one_trace_record(self) -> None:
        self.request(request_id="req-trace")
        records = [json.loads(line) for line in self.trace_path.read_text().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["request_id"], "req-trace")
        self.assertEqual(records[0]["image"]["width"], 640)


if __name__ == "__main__":
    unittest.main()
