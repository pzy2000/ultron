# Copyright (c) ModelScope Contributors. All rights reserved.
from fastapi import APIRouter

from ultron import server_state
from ultron.api.schemas import RouterCompleteRequest

router = APIRouter(tags=["router"])


def _model_dict(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@router.get("/router/health")
async def router_health():
    u = server_state.ultron
    if u is None:
        raise RuntimeError("Server not initialized")
    return {"success": True, "data": u.router_health()}


@router.post("/router/complete")
async def router_complete(request: RouterCompleteRequest):
    u = server_state.ultron
    if u is None:
        raise RuntimeError("Server not initialized")
    result = u.router_complete(
        mode=request.mode,
        messages=[_model_dict(m) for m in request.messages],
        router_info=request.router_info,
        trajectory_ref=request.trajectory_ref,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
    )
    return result
