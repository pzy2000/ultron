# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_client_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "skills" / "ultron-1.0.0" / "scripts" / "ultron_client.py"
    spec = importlib.util.spec_from_file_location("ultron_client_router_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestUltronClientRouterActions(unittest.TestCase):
    def test_router_complete_payload(self):
        mod = _load_client_module()
        calls = []

        def fake_api_request(method, endpoint, data=None):
            calls.append((method, endpoint, data))
            return {"success": True, "output": "ok"}

        payload = {
            "action": "router_complete",
            "messages": [{"role": "user", "content": "hi"}],
            "router_info": {"route": "simple"},
            "max_output_tokens": 12,
            "temperature": 0.1,
        }
        with patch.object(mod, "api_request", side_effect=fake_api_request):
            with patch.object(sys, "argv", ["ultron_client.py", json.dumps(payload)]):
                with patch("sys.stdout", new=io.StringIO()):
                    mod.main()

        self.assertEqual(calls[0][0], "POST")
        self.assertEqual(calls[0][1], "/router/complete")
        self.assertEqual(calls[0][2]["mode"], "direct")
        self.assertEqual(calls[0][2]["messages"], payload["messages"])

    def test_router_experience_payload(self):
        mod = _load_client_module()
        calls = []

        def fake_api_request(method, endpoint, data=None):
            calls.append((method, endpoint, data))
            return {"success": True, "output": "lesson"}

        payload = {
            "action": "router_experience",
            "messages": [{"role": "user", "content": "extract"}],
            "router_info": {"task": "debug"},
            "trajectory_ref": {"segment_id": "seg-1"},
        }
        with patch.object(mod, "api_request", side_effect=fake_api_request):
            with patch.object(sys, "argv", ["ultron_client.py", json.dumps(payload)]):
                with patch("sys.stdout", new=io.StringIO()):
                    mod.main()

        self.assertEqual(calls[0][0], "POST")
        self.assertEqual(calls[0][1], "/router/complete")
        self.assertEqual(calls[0][2]["mode"], "trajectory_experience")
        self.assertEqual(calls[0][2]["trajectory_ref"], {"segment_id": "seg-1"})


if __name__ == "__main__":
    unittest.main()
