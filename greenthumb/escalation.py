"""Hand-off to human operators: creates tickets in a JSON Lines queue."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Expected pick-up time per priority, taken from policies/servizio_clienti_contatti.md
PICKUP_TIME_BY_PRIORITY = {
    "high": "entro 4 ore lavorative",
    "medium": "entro 1 giorno lavorativo",
    "low": "entro 2 giorni lavorativi",
}


class EscalationService:
    """Creates escalation tickets and appends them to a JSONL file (a stand-in for a ticketing system)."""

    def __init__(self, queue_path: Path) -> None:
        """Remember where tickets are written."""
        self._queue_path = queue_path
        self._lock = threading.Lock()

    def create_ticket(
        self,
        reason: str,
        summary: str,
        priority: str,
        session_id: str | None = None,
        order_id: str | None = None,
    ) -> dict[str, Any]:
        """Open a ticket for a human operator and return its public details."""
        ticket = {
            "ticket_id": f"ESC-{uuid.uuid4().hex[:8].upper()}",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reason": reason,
            "priority": priority,
            "summary": summary,
            "order_id": order_id,
            "session_id": session_id,
        }
        with self._lock:
            self._queue_path.parent.mkdir(parents=True, exist_ok=True)
            with self._queue_path.open("a", encoding="utf-8") as queue:
                queue.write(json.dumps(ticket, ensure_ascii=False) + "\n")
        return {
            "status": "ok",
            "ticket_id": ticket["ticket_id"],
            "priority": priority,
            "expected_pickup": PICKUP_TIME_BY_PRIORITY.get(priority, PICKUP_TIME_BY_PRIORITY["medium"]),
        }
