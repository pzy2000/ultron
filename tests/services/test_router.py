# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from ultron.config import UltronConfig
from ultron.core.database import Database
from ultron.services.router import RouterService


class TestRouterService(unittest.TestCase):
    def _svc(self, enabled=True):
        tmp = tempfile.TemporaryDirectory()
        db = Database(os.path.join(tmp.name, "test.db"))
        cfg = UltronConfig(data_dir=tmp.name)
        cfg.router_enabled = enabled
        cfg.router_model = "Qwen/Qwen3-1.7B"
        cfg.router_base_url = "http://127.0.0.1:8000/v1"
        cfg.router_api_key = ""
        svc = RouterService(db=db, config=cfg)
        self.addCleanup(tmp.cleanup)
        return svc, db, tmp.name

    def test_disabled_returns_structured_error(self):
        svc, _db, _tmp = self._svc(enabled=False)
        result = svc.complete(
            mode="direct",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertFalse(result["success"])
        self.assertIn("disabled", result["error"])
        self.assertFalse(result["used_trajectory"])

    def test_direct_mode_calls_model(self):
        svc, _db, _tmp = self._svc(enabled=True)
        with patch.object(svc, "_call_model", return_value="small answer") as call:
            result = svc.complete(
                mode="direct",
                messages=[{"role": "user", "content": "2+2?"}],
                max_output_tokens=16,
                temperature=0.1,
            )
        self.assertTrue(result["success"])
        self.assertEqual(result["output"], "small answer")
        self.assertFalse(result["used_trajectory"])
        call.assert_called_once()

    def test_trajectory_mode_reads_segment_and_calls_model(self):
        svc, db, tmp = self._svc(enabled=True)
        session_file = os.path.join(tmp, "session.jsonl")
        with open(session_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"role": "user", "content": "bug"}) + "\n")
            f.write(json.dumps({"role": "assistant", "content": "fix"}) + "\n")
        db.save_task_segment(
            segment_id="seg-1",
            agent_id="agent-1",
            session_file=session_file,
            segment_index=0,
            start_line=1,
            end_line=2,
            fingerprint="fp",
            topic="debug",
        )
        with patch.object(svc, "_call_model", return_value="reuse this fix") as call:
            result = svc.complete(
                mode="trajectory_experience",
                messages=[{"role": "user", "content": "extract"}],
                router_info={"task": "debug"},
                trajectory_ref={"segment_id": "seg-1"},
            )
        self.assertTrue(result["success"])
        self.assertTrue(result["used_trajectory"])
        self.assertEqual(result["output"], "reuse this fix")
        sent_messages = call.call_args[0][0]
        self.assertIn("Trajectory segment", sent_messages[1]["content"])
        self.assertIn("bug", sent_messages[1]["content"])

    def test_trajectory_mode_accepts_agent_session_index_ref(self):
        svc, db, tmp = self._svc(enabled=True)
        session_file = os.path.join(tmp, "session.jsonl")
        with open(session_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"role": "user", "content": "question"}) + "\n")
            f.write(json.dumps({"role": "assistant", "content": "answer"}) + "\n")
        db.save_task_segment(
            segment_id="seg-2",
            agent_id="agent-2",
            session_file=session_file,
            segment_index=3,
            start_line=1,
            end_line=2,
            fingerprint="fp-2",
        )
        with patch.object(svc, "_call_model", return_value="experience"):
            result = svc.complete(
                mode="trajectory_experience",
                messages=[{"role": "user", "content": "extract"}],
                trajectory_ref={
                    "agent_id": "agent-2",
                    "session_file": session_file,
                    "segment_index": 3,
                },
            )
        self.assertTrue(result["success"])
        self.assertTrue(result["used_trajectory"])

    def test_missing_trajectory_ref_returns_error(self):
        svc, _db, _tmp = self._svc(enabled=True)
        result = svc.complete(
            mode="trajectory_experience",
            messages=[{"role": "user", "content": "extract"}],
            trajectory_ref={"segment_id": "missing"},
        )
        self.assertFalse(result["success"])
        self.assertIn("did not match", result["error"])

    @patch("ultron.services.router.HAS_OPENAI", True)
    @patch("ultron.services.router.OpenAI")
    def test_no_key_local_client_uses_dummy_key(self, mock_openai):
        svc, _db, _tmp = self._svc(enabled=True)
        svc._get_client()
        mock_openai.assert_called_once_with(
            api_key="ultron-router",
            base_url="http://127.0.0.1:8000/v1",
        )

    @patch("ultron.services.router.HAS_OPENAI", True)
    @patch("ultron.services.router.OpenAI")
    def test_call_model_extracts_text(self, mock_openai):
        svc, _db, _tmp = self._svc(enabled=True)
        msg = MagicMock()
        msg.content = "ok"
        choice = MagicMock(message=msg)
        response = MagicMock(choices=[choice])
        client = MagicMock()
        client.chat.completions.create.return_value = response
        mock_openai.return_value = client
        text = svc._call_model(
            [{"role": "user", "content": "hi"}],
            max_output_tokens=8,
            temperature=0.2,
        )
        self.assertEqual(text, "ok")
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "Qwen/Qwen3-1.7B")
        self.assertEqual(kwargs["max_tokens"], 8)
        self.assertEqual(kwargs["temperature"], 0.2)


if __name__ == "__main__":
    unittest.main()
