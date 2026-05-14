# Copyright (c) ModelScope Contributors. All rights reserved.
"""Callable model router for low-cost Ultron-owned inference."""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from ..config import UltronConfig, default_config
from ..core.database import Database
from ..core.llm_service import HAS_OPENAI, OpenAI
from .trajectory.session_reader import TrajectorySessionReader

logger = logging.getLogger(__name__)


class RouterService:
    """Run router requests against an Ultron-owned OpenAI-compatible model."""

    def __init__(
        self,
        db: Database,
        config: Optional[UltronConfig] = None,
        session_reader: Optional[TrajectorySessionReader] = None,
    ):
        self.db = db
        self.config = config or default_config
        self._reader = session_reader or TrajectorySessionReader()
        self._client = None
        self._client_key = ""
        self._client_base_url = ""
        self._settings_path = Path(self.config.data_dir) / "router_config.json"
        self._load_persisted_settings()

    def health(self) -> dict:
        return {
            "enabled": bool(getattr(self.config, "router_enabled", False)),
            "model": getattr(self.config, "router_model", ""),
            "base_url": getattr(self.config, "router_base_url", ""),
            "has_openai": HAS_OPENAI,
            "has_api_key": bool((getattr(self.config, "router_api_key", "") or "").strip()),
        }

    def get_settings(self) -> dict:
        """Return redacted runtime router settings."""
        return {
            "enabled": bool(getattr(self.config, "router_enabled", False)),
            "model": getattr(self.config, "router_model", ""),
            "base_url": getattr(self.config, "router_base_url", ""),
            "has_api_key": bool((getattr(self.config, "router_api_key", "") or "").strip()),
        }

    def update_settings(self, settings: dict[str, Any]) -> dict:
        """Persist runtime router settings. ``api_key`` is write-only."""
        if not isinstance(settings, dict):
            raise ValueError("settings must be an object")
        if "enabled" in settings:
            self.config.router_enabled = bool(settings.get("enabled"))
        if "model" in settings:
            model = str(settings.get("model", "")).strip()
            if not model:
                raise ValueError("model cannot be empty")
            self.config.router_model = model
        if "base_url" in settings:
            base_url = str(settings.get("base_url", "")).strip()
            if not base_url:
                raise ValueError("base_url cannot be empty")
            self.config.router_base_url = base_url
        if "api_key" in settings and settings.get("api_key") is not None:
            self.config.router_api_key = str(settings.get("api_key", "")).strip()
        self._client = None
        self._persist_settings()
        return self.get_settings()

    def complete(
        self,
        *,
        mode: str,
        messages: list[dict[str, Any]],
        router_info: Optional[dict[str, Any]] = None,
        trajectory_ref: Optional[dict[str, Any]] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        start = time.time()
        mode = (mode or "direct").strip()
        router_info = router_info or {}

        if not getattr(self.config, "router_enabled", False):
            return self._result(
                mode=mode,
                output="",
                start=start,
                success=False,
                error="Ultron router is disabled. Set ULTRON_ROUTER_ENABLED=1 on the server.",
            )

        if mode not in ("direct", "trajectory_experience"):
            return self._result(
                mode=mode,
                output="",
                start=start,
                success=False,
                error="Unsupported router mode. Use direct or trajectory_experience.",
            )

        clean_messages = self._normalize_messages(messages)
        if not clean_messages:
            return self._result(
                mode=mode,
                output="",
                start=start,
                success=False,
                error="messages must contain at least one role/content item.",
            )

        used_trajectory = False
        if mode == "trajectory_experience":
            resolved = self._build_trajectory_experience_messages(
                clean_messages,
                router_info=router_info,
                trajectory_ref=trajectory_ref or {},
            )
            if not resolved["success"]:
                return self._result(
                    mode=mode,
                    output="",
                    start=start,
                    success=False,
                    error=resolved["error"],
                )
            clean_messages = resolved["messages"]
            used_trajectory = True

        try:
            output = self._call_model(
                clean_messages,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning("Router model call failed: %s", e)
            return self._result(
                mode=mode,
                output="",
                start=start,
                success=False,
                used_trajectory=used_trajectory,
                error=str(e),
            )

        if output is None:
            return self._result(
                mode=mode,
                output="",
                start=start,
                success=False,
                used_trajectory=used_trajectory,
                error="Router model returned no text.",
            )
        return self._result(
            mode=mode,
            output=output,
            start=start,
            success=True,
            used_trajectory=used_trajectory,
        )

    def openai_chat_completions(self, request: dict[str, Any]) -> tuple[dict, int]:
        """Handle a non-streaming OpenAI-compatible chat completions request."""
        if not isinstance(request, dict):
            return {"error": {"message": "request body must be an object"}}, 400
        if request.get("stream") is True:
            return {
                "error": {
                    "message": "Ultron router OpenAI compatibility supports non-streaming requests only.",
                    "type": "invalid_request_error",
                    "param": "stream",
                    "code": "streaming_unsupported",
                }
            }, 400
        messages = request.get("messages", [])
        model = str(request.get("model", "") or "").strip()
        old_model = getattr(self.config, "router_model", "")
        if model:
            self.config.router_model = model
        try:
            result = self.complete(
                mode="direct",
                messages=messages,
                router_info={"source": "openai_chat_completions"},
                max_output_tokens=request.get("max_tokens"),
                temperature=request.get("temperature"),
            )
        finally:
            if model:
                self.config.router_model = old_model
        if not result.get("success"):
            return {
                "error": {
                    "message": result.get("error", "Router request failed"),
                    "type": "router_error",
                    "param": None,
                    "code": "router_failed",
                }
            }, 400
        now = int(time.time())
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": now,
            "model": model or result.get("model", getattr(self.config, "router_model", "")),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.get("output", ""),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }, 200

    def _result(
        self,
        *,
        mode: str,
        output: str,
        start: float,
        success: bool,
        used_trajectory: bool = False,
        error: str = "",
    ) -> dict:
        return {
            "success": bool(success),
            "mode": mode,
            "output": output or "",
            "model": getattr(self.config, "router_model", ""),
            "latency_ms": round((time.time() - start) * 1000, 1),
            "used_trajectory": bool(used_trajectory),
            "error": error or "",
        }

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        if not isinstance(messages, list):
            return out
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip()
            content = msg.get("content", "")
            if role not in ("system", "user", "assistant", "tool"):
                continue
            if content is None:
                content = ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            out.append({"role": role, "content": content})
        return out

    def _build_trajectory_experience_messages(
        self,
        messages: list[dict[str, str]],
        *,
        router_info: dict[str, Any],
        trajectory_ref: dict[str, Any],
    ) -> dict:
        segment = self._resolve_segment(trajectory_ref)
        if not segment:
            return {
                "success": False,
                "error": "trajectory_ref did not match a task segment.",
            }
        trajectory_messages = self._reader.read_segment_messages(segment)
        if not trajectory_messages:
            return {
                "success": False,
                "error": "trajectory segment exists but its session messages could not be read.",
            }

        prompt = (
            "Extract reusable experience from the task trajectory for the router request.\n"
            "Return a concise answer that the caller can use directly. Focus on facts, "
            "procedures, pitfalls, and decisions supported by the trajectory.\n\n"
            f"Router info:\n{json.dumps(router_info, ensure_ascii=False, indent=2)}\n\n"
            f"Trajectory segment:\n{json.dumps(trajectory_messages, ensure_ascii=False, indent=2)}\n\n"
            f"Current router messages:\n{json.dumps(messages, ensure_ascii=False, indent=2)}"
        )
        return {
            "success": True,
            "messages": [
                {
                    "role": "system",
                    "content": "You extract reusable experience from Ultron trajectories.",
                },
                {"role": "user", "content": prompt},
            ],
        }

    def _resolve_segment(self, ref: dict[str, Any]) -> Optional[dict]:
        if not isinstance(ref, dict):
            return None
        segment_id = str(ref.get("segment_id", "")).strip()
        if segment_id:
            return self.db.get_task_segment(segment_id)
        agent_id = str(ref.get("agent_id", "")).strip()
        session_file = str(ref.get("session_file", "")).strip()
        if not agent_id or not session_file or "segment_index" not in ref:
            return None
        try:
            segment_index = int(ref.get("segment_index"))
        except (TypeError, ValueError):
            return None
        return self.db.get_task_segment_by_ref(agent_id, session_file, segment_index)

    def _get_client(self):
        if not HAS_OPENAI or OpenAI is None:
            raise RuntimeError("openai package not installed, router unavailable")
        key = (getattr(self.config, "router_api_key", "") or "").strip() or "ultron-router"
        base_url = (getattr(self.config, "router_base_url", "") or "").strip()
        if not base_url:
            raise RuntimeError("ULTRON_ROUTER_BASE_URL is empty")
        if (
            self._client is None
            or self._client_key != key
            or self._client_base_url != base_url
        ):
            self._client = OpenAI(api_key=key, base_url=base_url)
            self._client_key = key
            self._client_base_url = base_url
        return self._client

    def _load_persisted_settings(self) -> None:
        if not self._settings_path.is_file():
            return
        try:
            raw = json.loads(self._settings_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
        except (OSError, ValueError):
            logger.warning("Could not read router settings from %s", self._settings_path)
            return
        if "enabled" in raw:
            self.config.router_enabled = bool(raw.get("enabled"))
        model = str(raw.get("model", "")).strip()
        if model:
            self.config.router_model = model
        base_url = str(raw.get("base_url", "")).strip()
        if base_url:
            self.config.router_base_url = base_url
        if "api_key" in raw and raw.get("api_key") is not None:
            self.config.router_api_key = str(raw.get("api_key", "")).strip()

    def _persist_settings(self) -> None:
        payload = {
            "enabled": bool(getattr(self.config, "router_enabled", False)),
            "model": getattr(self.config, "router_model", ""),
            "base_url": getattr(self.config, "router_base_url", ""),
            "api_key": getattr(self.config, "router_api_key", ""),
        }
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._settings_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _call_model(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: Optional[int],
        temperature: Optional[float],
    ) -> Optional[str]:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": getattr(self.config, "router_model", ""),
            "messages": messages,
        }
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max(1, int(max_output_tokens))
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        response = client.chat.completions.create(**kwargs)
        choices = getattr(response, "choices", None) or []
        if not choices:
            return None
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        return content if isinstance(content, str) else None
