# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Optional


class RouterMixin:
    def router_health(self) -> dict:
        """Return model-router availability and configuration summary."""
        return self.router_service.health()

    def router_complete(
        self,
        *,
        mode: str = "direct",
        messages: list[dict[str, Any]],
        router_info: Optional[dict[str, Any]] = None,
        trajectory_ref: Optional[dict[str, Any]] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """Call the Ultron-owned router model as a function."""
        return self.router_service.complete(
            mode=mode,
            messages=messages,
            router_info=router_info,
            trajectory_ref=trajectory_ref,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
