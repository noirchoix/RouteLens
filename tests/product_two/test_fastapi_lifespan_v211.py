from __future__ import annotations

import asyncio
from pathlib import Path

from reacts.api.main import create_app
from reacts.settings import Settings


def test_create_app_uses_lifespan_for_shutdown(tmp_path: Path, monkeypatch) -> None:
    app = create_app(Settings(project_root=tmp_path))
    closed: list[bool] = []
    monkeypatch.setattr(app.state.application, "close", lambda: closed.append(True))

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            assert closed == []
        assert closed == [True]

    asyncio.run(exercise())
