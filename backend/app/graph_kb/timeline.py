"""Graph timeline — save and load timestamped snapshots for evolution tracking."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)


def save_snapshot(user_id: str, event: str, stats: dict | None = None) -> Path:
    """Save a timestamped graph snapshot for timeline tracking."""
    timeline_dir = settings.data_dir / "users" / user_id / "timeline"
    timeline_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filepath = timeline_dir / f"{timestamp}_{event}.json"

    snapshot_data = {
        "timestamp": timestamp,
        "event": event,
        "stats": stats or {},
    }
    filepath.write_text(json.dumps(snapshot_data, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved timeline snapshot: %s (%s)", filepath.name, event)
    return filepath


def list_snapshots(user_id: str) -> list[dict[str, Any]]:
    """List all timeline snapshots for a user, newest first."""
    timeline_dir = settings.data_dir / "users" / user_id / "timeline"
    if not timeline_dir.exists():
        return []

    snapshots = []
    for f in sorted(timeline_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text("utf-8"))
            snapshots.append({
                "id": f.stem,
                "timestamp": data.get("timestamp", ""),
                "event": data.get("event", ""),
                "stats": data.get("stats", {}),
            })
        except Exception:
            pass
    return snapshots
