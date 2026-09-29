"""Read-only access to the order database (``data/orders.json``)."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

ORDER_ID_PATTERN = re.compile(r"^\d{4}$")

STATUS_LABELS = {
    "in_preparazione": "In preparazione",
    "spedito": "Spedito",
    "consegnato": "Consegnato",
    "reso_in_corso": "Reso in corso",
    "rimborsato": "Rimborsato",
    "annullato": "Annullato",
}


def normalize_order_id(raw_order_id: str) -> str:
    """Normalise user-typed ids such as ``"#1042"``, ``"ordine 1042"`` or ``" 1042 "``."""
    digits = re.sub(r"\D", "", str(raw_order_id))
    return digits


class OrderRepository:
    """In-memory repository over the JSON order file."""

    def __init__(self, orders_path: Path) -> None:
        """Load all orders from ``orders_path`` and index them by id."""
        payload = json.loads(orders_path.read_text(encoding="utf-8"))
        self._orders: dict[str, dict[str, Any]] = {order["order_id"]: order for order in payload["orders"]}

    def __len__(self) -> int:
        """Number of orders in the repository."""
        return len(self._orders)

    def get(self, order_id: str) -> dict[str, Any] | None:
        """Return the raw order record, or ``None`` if it does not exist."""
        return self._orders.get(normalize_order_id(order_id))

    def lookup(self, raw_order_id: str, today: date) -> dict[str, Any]:
        """Look up an order and return a tool-friendly result.

        The result always has a ``status`` key:

        * ``"ok"``: order found; ``order`` contains a customer-facing summary with
          ``days_until_delivery`` computed against ``today``;
        * ``"invalid_input"``: the id is not a 4-digit number;
        * ``"not_found"``: well-formed id with no matching order.
        """
        order_id = normalize_order_id(raw_order_id)
        if not ORDER_ID_PATTERN.match(order_id):
            return {
                "status": "invalid_input",
                "message": f"'{raw_order_id}' non è un numero d'ordine valido: deve essere di 4 cifre (es. 1042).",
            }
        order = self._orders.get(order_id)
        if order is None:
            return {
                "status": "not_found",
                "order_id": order_id,
                "message": f"Nessun ordine con numero {order_id}. Chiedi al cliente di verificare il numero.",
            }
        return {"status": "ok", "order": self._summarize(order, today)}

    @staticmethod
    def _summarize(order: dict[str, Any], today: date) -> dict[str, Any]:
        """Keep only what the assistant needs (no e-mail or other personal data)."""
        shipping = order["shipping"]
        days_until_delivery = None
        if shipping.get("estimated_delivery") and not shipping.get("delivered_at"):
            days_until_delivery = (date.fromisoformat(shipping["estimated_delivery"]) - today).days
        return {
            "order_id": order["order_id"],
            "status": order["status"],
            "status_label": STATUS_LABELS.get(order["status"], order["status"]),
            "created_at": order["created_at"],
            "items": [f"{item['quantity']}x {item['name']} ({item['sku']})" for item in order["items"]],
            "total_eur": order["total_eur"],
            "payment_method": order["payment_method"],
            "shipping_method": shipping["method"],
            "carrier": shipping.get("carrier"),
            "tracking_number": shipping.get("tracking_number"),
            "shipped_at": shipping.get("shipped_at"),
            "estimated_delivery": shipping.get("estimated_delivery"),
            "days_until_delivery": days_until_delivery,
            "delivery_overdue": days_until_delivery is not None and days_until_delivery < 0,
            "delivered_at": shipping.get("delivered_at"),
            "return": order.get("return"),
            "refund": order.get("refund"),
            "notes": order.get("notes"),
        }
