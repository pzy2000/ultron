# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import tempfile
import unittest

from ultron.config import UltronConfig
from ultron.core.database import Database
from ultron.services.router import RouterService


@unittest.skipUnless(
    os.environ.get("ULTRON_ROUTER_E2E") == "1",
    "Set ULTRON_ROUTER_E2E=1 to run against a local OpenAI-compatible router model.",
)
class TestRouterE2E(unittest.TestCase):
    def test_direct_router_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = UltronConfig(data_dir=tmp)
            cfg.router_enabled = True
            cfg.router_base_url = os.environ.get(
                "ULTRON_ROUTER_BASE_URL", "http://127.0.0.1:8000/v1"
            )
            cfg.router_model = os.environ.get(
                "ULTRON_ROUTER_MODEL", "Qwen/Qwen3-1.7B"
            )
            cfg.router_api_key = os.environ.get("ULTRON_ROUTER_API_KEY", "")
            svc = RouterService(Database(os.path.join(tmp, "test.db")), cfg)
            result = svc.complete(
                mode="direct",
                messages=[{"role": "user", "content": "Reply with only: pong"}],
                max_output_tokens=8,
                temperature=0,
            )
        self.assertTrue(result["success"], result)
        self.assertIn("pong", result["output"].lower())


if __name__ == "__main__":
    unittest.main()
