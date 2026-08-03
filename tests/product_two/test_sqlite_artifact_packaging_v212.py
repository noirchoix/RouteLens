from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import reacts.artifacts.bundle as bundle


def _create_wal_registry(path: Path) -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE model_versions ("
            "model_id TEXT PRIMARY KEY, task TEXT, artifact_path TEXT, "
            "model_card_path TEXT, split_sha256 TEXT, "
            "runtime_load_required INTEGER, lifecycle_state TEXT)"
        )
        conn.execute(
            "INSERT INTO model_versions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "reaction_family:test",
                "reaction_family",
                "old/model.joblib",
                "old/model_card.json",
                "a" * 64,
                1,
                "active",
            ),
        )
        conn.commit()


def test_artifact_sqlite_helpers_close_connections_and_remove_wal_state(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.sqlite3"
    destination = tmp_path / "packaged.sqlite3"
    _create_wal_registry(source)

    real_connect = sqlite3.connect
    opened: list[TrackingConnection] = []

    class TrackingConnection(sqlite3.Connection):
        closed: bool

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.closed = False
            opened.append(self)

        def close(self) -> None:
            self.closed = True
            super().close()

    def tracking_connect(*args, **kwargs):
        kwargs["factory"] = TrackingConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(bundle.sqlite3, "connect", tracking_connect)

    bundle._copy_sqlite_snapshot(source, destination)
    assert opened and all(connection.closed for connection in opened)

    opened.clear()
    bundle._rebase_registry_database(
        destination,
        {"reaction_family:test": ("models/reaction_family/model.joblib", None)},
    )
    assert opened and all(connection.closed for connection in opened)
    assert not destination.with_name(destination.name + "-wal").exists()
    assert not destination.with_name(destination.name + "-shm").exists()

    opened.clear()
    rows = bundle._read_runtime_registry_database(destination)
    assert rows == [
        {
            "model_id": "reaction_family:test",
            "task": "reaction_family",
            "artifact_path": "models/reaction_family/model.joblib",
            "model_card_path": None,
            "split_sha256": "a" * 64,
        }
    ]
    assert opened and all(connection.closed for connection in opened)
