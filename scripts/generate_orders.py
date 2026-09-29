"""Generate the synthetic ``data/orders.json`` file.

Orders are defined as explicit specifications with dates expressed as day
offsets from a reference date (the simulated "today" of the demo, see
``GREENTHUMB_CURRENT_DATE``). This makes the dataset deterministic and keeps
relative facts (e.g. "order 1042 arrives tomorrow") consistent with the date
the agent believes it is.

Usage:
    python scripts/generate_orders.py [--reference-date 2026-06-16]
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "orders.json"
DEFAULT_REFERENCE_DATE = "2026-06-16"

CATALOG: dict[str, tuple[str, float]] = {
    "GT-BLB-101": ("Bulbi di tulipano Darwin Mix (20 bulbi)", 12.90),
    "GT-BLB-102": ("Bulbi di narciso 'Carlton' (15 bulbi)", 9.50),
    "GT-BLB-201": ("Tuberi di dalia decorativa 'Café au Lait' (3 tuberi)", 8.90),
    "GT-SEM-010": ("Semi di pomodoro Cuore di Bue", 2.80),
    "GT-SEM-021": ("Semi di basilico Genovese", 2.20),
    "GT-SEM-200": ("Miscuglio per prato rustico calpestabile 1 kg", 14.90),
    "GT-SUB-050": ("Terriccio universale biologico 50 L", 11.90),
    "GT-CON-005": ("Concime organico stallatico pellettato 5 kg", 9.90),
    "GT-IRR-020": ("Kit irrigazione a goccia per balcone con programmatore", 34.90),
    "GT-VAS-040": ("Vaso autoirrigante 40 cm", 24.90),
    "GT-ATT-011": ("Cesoie da potatura bypass Pro 21 cm", 29.90),
    "GT-PIA-305": ("Rosa rampicante 'New Dawn' in vaso 3 L", 19.90),
    "GT-PIA-120": ("Lavanda angustifolia 'Hidcote' vaso 14 cm", 5.90),
    "GT-PIA-410": ("Limone 'Quattro Stagioni' in vaso 20 cm", 39.90),
    "GT-PIA-220": ("Piantine di fragola rifiorente 'Mara des Bois' (set da 6)", 12.90),
    "GT-DIF-030": ("Olio di neem corroborante naturale 250 ml", 12.50),
    "GT-SER-004": ("Serra da balcone a 4 ripiani", 44.90),
}

SHIPPING_COSTS = {"standard": 5.90, "express": 9.90, "punto_ritiro": 3.90}
FREE_SHIPPING_THRESHOLD = 49.0

# Each spec: dates are day offsets relative to the reference date (0 = today).
ORDER_SPECS: list[dict[str, Any]] = [
    {"id": "1030", "customer": "Giulia Ferri", "city": "Torino", "created": -40, "items": {"GT-ATT-011": 1},
     "payment": "carta", "shipping": "standard", "carrier": "BRT", "shipped": -39, "delivered": -36,
     "status": "rimborsato",
     "return": {"requested": -30, "reason": "ripensamento", "received": -25},
     "refund": {"issued": -12, "deduction": 6.90, "note": "Trattenuti 6,90 EUR di costo del reso per ripensamento."}},
    {"id": "1031", "customer": "Marco Bellini", "city": "Bergamo", "created": -33, "items": {"GT-SUB-050": 4, "GT-CON-005": 1},
     "payment": "paypal", "shipping": "standard", "carrier": "GLS", "shipped": -32, "delivered": -29, "status": "consegnato"},
    {"id": "1032", "customer": "Sara Colombo", "city": "Monza", "created": -24, "items": {"GT-PIA-120": 6, "GT-PIA-305": 1},
     "payment": "carta", "shipping": "standard", "carrier": "BRT", "shipped": -22, "delivered": -20, "status": "consegnato"},
    {"id": "1033", "customer": "Luca Moretti", "city": "Verona", "created": -16, "items": {"GT-IRR-020": 1},
     "payment": "carta", "shipping": "standard", "carrier": "SDA", "shipped": -15, "delivered": -12,
     "status": "reso_in_corso",
     "return": {"requested": -3, "reason": "prodotto difettoso: il programmatore non si accende", "received": None},
     "notes": "Reso gratuito per difetto. In attesa di ritiro da parte del corriere."},
    {"id": "1034", "customer": "Chiara Galli", "city": "Parma", "created": -8, "items": {"GT-PIA-410": 1},
     "payment": "carta", "shipping": "standard", "status": "annullato", "cancelled": -7,
     "notes": "Annullato su richiesta del cliente prima della spedizione. Pre-autorizzazione rilasciata."},
    {"id": "1035", "customer": "Andrea Riva", "city": "Genova", "created": -4, "items": {"GT-SER-004": 1, "GT-SEM-021": 2},
     "payment": "klarna", "shipping": "standard", "carrier": "GLS", "shipped": -2, "eta": 2, "status": "spedito"},
    {"id": "1036", "customer": "Elena Rossi", "city": "Bologna", "created": -1, "items": {"GT-PIA-220": 1, "GT-SUB-050": 1},
     "payment": "paypal", "shipping": "standard", "status": "in_preparazione", "eta": 3,
     "notes": "Contiene piante vive: partenza prevista mercoledì (piante spedite solo lun-mer)."},
    {"id": "1037", "customer": "Davide Conti", "city": "Firenze", "created": -62, "items": {"GT-SEM-010": 3, "GT-SEM-021": 2},
     "payment": "carta", "shipping": "punto_ritiro", "carrier": "SDA", "shipped": -61, "delivered": -58, "status": "consegnato"},
    {"id": "1038", "customer": "Francesca Marino", "city": "Napoli", "created": -11, "items": {"GT-VAS-040": 2, "GT-SUB-050": 2},
     "payment": "carta", "shipping": "standard", "carrier": "BRT", "shipped": -9, "eta": -4, "status": "spedito",
     "notes": "Tracking fermo all'hub di Bologna dal giorno successivo alla spedizione. Consegna in ritardo rispetto alla stima."},
    {"id": "1039", "customer": "Paolo Greco", "city": "Bari", "created": -21, "items": {"GT-PIA-305": 2},
     "payment": "paypal", "shipping": "express", "carrier": "BRT", "shipped": -20, "delivered": -19,
     "status": "rimborsato",
     "return": {"requested": -19, "reason": "garanzia Arrivo Perfetto: rami spezzati e vaso rotto (foto inviate)", "received": None},
     "refund": {"issued": -17, "deduction": 0.0, "note": "Rimborso integrale su richiesta del cliente, pianta non da restituire."}},
    {"id": "1040", "customer": "Martina Fontana", "city": "Padova", "created": -6, "items": {"GT-BLB-101": 1, "GT-BLB-102": 1},
     "payment": "carta", "shipping": "standard", "carrier": "GLS", "shipped": -5, "delivered": -2, "status": "consegnato"},
    {"id": "1041", "customer": "Simone Barbieri", "city": "Trento", "created": -2, "items": {"GT-CON-005": 2, "GT-ATT-011": 1},
     "payment": "bonifico", "shipping": "standard", "status": "in_preparazione",
     "notes": "In attesa dell'accredito del bonifico. Senza pagamento entro 7 giorni dall'ordine l'ordine viene annullato."},
    {"id": "1042", "customer": "Laura Esposito", "city": "Milano", "created": -3, "items": {"GT-BLB-101": 2, "GT-BLB-102": 1},
     "payment": "carta", "shipping": "standard", "carrier": "BRT", "shipped": -1, "eta": 1, "status": "spedito"},
    {"id": "1043", "customer": "Roberto Villa", "city": "Brescia", "created": 0, "items": {"GT-BLB-201": 2, "GT-CON-005": 1},
     "payment": "applepay", "shipping": "standard", "status": "in_preparazione", "eta": 4},
    {"id": "1044", "customer": "Alessia Ricci", "city": "Roma", "created": -9, "items": {"GT-VAS-040": 2},
     "payment": "carta", "shipping": "standard", "carrier": "SDA", "shipped": -8, "delivered": -5, "status": "consegnato"},
    {"id": "1045", "customer": "Giorgio Lombardi", "city": "Modena", "created": -1, "items": {"GT-SEM-021": 1, "GT-DIF-030": 1},
     "payment": "paypal", "shipping": "express", "carrier": "GLS", "shipped": 0, "eta": 1, "status": "spedito"},
    {"id": "1046", "customer": "Valentina Serra", "city": "Cagliari", "created": -18, "items": {"GT-VAS-040": 1},
     "payment": "carta", "shipping": "standard", "carrier": "SDA", "shipped": -17, "delivered": -13,
     "status": "reso_in_corso",
     "return": {"requested": -2, "reason": "ripensamento: colore diverso da quello atteso", "received": None},
     "notes": "Etichetta di reso inviata. Ritiro previsto entro 3 giorni lavorativi."},
    {"id": "1047", "customer": "Federico Costa", "city": "Udine", "created": -19, "items": {"GT-SEM-200": 3},
     "payment": "carta", "shipping": "standard", "carrier": "GLS", "shipped": -18, "delivered": -15, "status": "consegnato"},
    {"id": "1048", "customer": "Ilaria Mancini", "city": "Perugia", "created": -2, "items": {"GT-IRR-020": 1, "GT-VAS-040": 1},
     "payment": "carta", "shipping": "punto_ritiro", "carrier": "SDA", "shipped": -1, "eta": 3, "status": "spedito"},
    {"id": "1049", "customer": "Stefano Leone", "city": "Palermo", "created": -12, "items": {"GT-BLB-201": 3},
     "payment": "paypal", "shipping": "standard", "status": "annullato", "cancelled": -10,
     "notes": "Annullato da GreenThumb per esaurimento scorte. Rimborso integrale emesso su PayPal."},
    {"id": "1050", "customer": "Beatrice Gatti", "city": "Lucca", "created": -8, "items": {"GT-PIA-410": 1, "GT-SUB-050": 1},
     "payment": "carta", "shipping": "standard", "carrier": "BRT", "shipped": -7, "delivered": -4, "status": "consegnato"},
    {"id": "1051", "customer": "Nicola Ferrara", "city": "Ancona", "created": -1, "items": {"GT-SUB-050": 6, "GT-CON-005": 2},
     "payment": "carta", "shipping": "standard", "status": "in_preparazione", "eta": 5,
     "notes": "Ordine voluminoso (oltre 100 kg): consegna su bancale al piano strada."},
    {"id": "1052", "customer": "Giada Pellegrini", "city": "Rimini", "created": -4, "items": {"GT-PIA-120": 3},
     "payment": "paypal", "shipping": "standard", "carrier": "GLS", "shipped": -3, "delivered": -1, "status": "consegnato"},
    {"id": "1053", "customer": "Tommaso Rinaldi", "city": "Treviso", "created": -70, "items": {"GT-ATT-011": 1},
     "payment": "carta", "shipping": "standard", "carrier": "BRT", "shipped": -69, "delivered": -66,
     "status": "rimborsato",
     "return": {"requested": -30, "reason": "garanzia: lama incrinata dopo un mese di utilizzo", "received": -24},
     "refund": {"issued": -14, "deduction": 0.0, "note": "Prodotto non sostituibile (esaurito): rimborso integrale in garanzia."}},
]


def _iso(reference: date, offset: int | None) -> str | None:
    """Convert a day offset into an ISO date string (``None`` stays ``None``)."""
    if offset is None:
        return None
    return (reference + timedelta(days=offset)).isoformat()


def _email_for(name: str) -> str:
    """Build a fictitious e-mail address from a customer name."""
    first, last = name.lower().split(" ", 1)
    return f"{first}.{last.replace(' ', '')}@example.com"


def build_order(spec: dict[str, Any], reference: date) -> dict[str, Any]:
    """Expand a compact order specification into the full order record."""
    items = []
    for sku, quantity in spec["items"].items():
        name, unit_price = CATALOG[sku]
        items.append({"sku": sku, "name": name, "quantity": quantity, "unit_price_eur": unit_price,
                      "line_total_eur": round(unit_price * quantity, 2)})
    subtotal = round(sum(item["line_total_eur"] for item in items), 2)
    shipping_cost = SHIPPING_COSTS[spec["shipping"]]
    if spec["shipping"] == "standard" and subtotal >= FREE_SHIPPING_THRESHOLD:
        shipping_cost = 0.0
    total = round(subtotal + shipping_cost, 2)

    history = [{"date": _iso(reference, spec["created"]), "status": "in_preparazione", "note": "Ordine confermato"}]
    if spec.get("cancelled") is not None:
        history.append({"date": _iso(reference, spec["cancelled"]), "status": "annullato", "note": spec.get("notes", "")})
    if spec.get("shipped") is not None:
        history.append({"date": _iso(reference, spec["shipped"]), "status": "spedito",
                        "note": f"Affidato al corriere {spec['carrier']}"})
    if spec.get("delivered") is not None:
        history.append({"date": _iso(reference, spec["delivered"]), "status": "consegnato", "note": "Consegna registrata"})

    order_return = None
    if "return" in spec:
        ret = spec["return"]
        order_return = {"requested_at": _iso(reference, ret["requested"]), "reason": ret["reason"],
                        "received_at": _iso(reference, ret["received"])}
        history.append({"date": order_return["requested_at"], "status": "reso_in_corso", "note": ret["reason"]})

    refund = None
    if "refund" in spec:
        ref = spec["refund"]
        refund = {"amount_eur": round(total - ref["deduction"], 2), "issued_at": _iso(reference, ref["issued"]),
                  "method": spec["payment"], "note": ref["note"]}
        history.append({"date": refund["issued_at"], "status": "rimborsato",
                        "note": f"Rimborso di {refund['amount_eur']:.2f} EUR"})

    estimated_delivery = _iso(reference, spec.get("eta"))
    if spec.get("delivered") is not None:
        estimated_delivery = _iso(reference, spec["delivered"])

    tracking = None
    if spec.get("carrier") and spec.get("shipped") is not None:
        tracking = f"{spec['carrier'][:3].upper()}{int(spec['id']) * 7919 % 10**9:09d}"

    return {
        "order_id": spec["id"],
        "customer_name": spec["customer"],
        "customer_email": _email_for(spec["customer"]),
        "created_at": _iso(reference, spec["created"]),
        "status": spec["status"],
        "items": items,
        "subtotal_eur": subtotal,
        "shipping_cost_eur": shipping_cost,
        "total_eur": total,
        "payment_method": spec["payment"],
        "shipping": {
            "method": spec["shipping"],
            "carrier": spec.get("carrier"),
            "tracking_number": tracking,
            "city": spec["city"],
            "shipped_at": _iso(reference, spec.get("shipped")),
            "estimated_delivery": estimated_delivery if spec["status"] not in ("annullato",) else None,
            "delivered_at": _iso(reference, spec.get("delivered")),
        },
        "return": order_return,
        "refund": refund,
        "notes": spec.get("notes"),
        "history": sorted(history, key=lambda event: event["date"]),
    }


def main() -> None:
    """Parse CLI arguments and write the orders file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-date", default=DEFAULT_REFERENCE_DATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reference = date.fromisoformat(args.reference_date)
    orders = [build_order(spec, reference) for spec in ORDER_SPECS]
    payload = {"reference_date": reference.isoformat(), "orders": orders}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(orders)} orders to {args.output}")


if __name__ == "__main__":
    main()
