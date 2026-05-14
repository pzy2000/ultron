# Copyright (c) ModelScope Contributors. All rights reserved.
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ultron import server_state
from ultron.api.deps import get_current_user
from ultron.api.schemas import RouterCompleteRequest, RouterSettingsRequest

router = APIRouter(tags=["router"])


def _model_dict(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _model_update_dict(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_unset=True)
    return model.dict(exclude_unset=True)


@router.get("/router/health")
async def router_health():
    u = server_state.ultron
    if u is None:
        raise RuntimeError("Server not initialized")
    return {"success": True, "data": u.router_health()}


@router.get("/router/settings")
async def router_settings():
    u = server_state.ultron
    if u is None:
        raise RuntimeError("Server not initialized")
    return {"success": True, "data": u.router_service.get_settings()}


@router.post("/router/settings")
async def update_router_settings(
    request: RouterSettingsRequest,
    _user: dict = Depends(get_current_user),
):
    u = server_state.ultron
    if u is None:
        raise RuntimeError("Server not initialized")
    try:
        data = u.router_service.update_settings(_model_update_dict(request))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True, "data": data}


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


@router.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    u = server_state.ultron
    if u is None:
        raise RuntimeError("Server not initialized")
    body = await request.json()
    payload, status_code = u.router_service.openai_chat_completions(body)
    return JSONResponse(payload, status_code=status_code)
