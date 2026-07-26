from __future__ import annotations

from fastapi import Header, HTTPException, Request, status


def get_application(request: Request):
    return request.app.state.application


async def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    settings = request.app.state.application.settings
    if settings.require_api_key and (not settings.api_key or x_api_key != settings.api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
