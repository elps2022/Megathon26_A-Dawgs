"""Customer memory store that learns and persists across sessions.

Implements Section 10 optional extension: Customer Memory.
Defined in docs/ARCHITECTURE.md section 3.11.
"""

import json
from pathlib import Path
from typing import Any
from greenhome.schema import Property

DEFAULT_MEMORY_DIR = Path("data/memory")


class CustomerMemoryStore:
    """Manages persistent stateful customer profiles and living constraints."""

    def __init__(self, storage_dir: Path = DEFAULT_MEMORY_DIR):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, customer_id: str) -> Path:
        return self.storage_dir / f"{customer_id}.json"

    def get_memory(self, customer_id: str) -> dict[str, Any]:
        """Retrieve customer memory or return default profile."""
        path = self._file_path(customer_id)
        if not path.exists():
            return {
                "customer_id": customer_id,
                "tenure": "owner_occupier",
                "budget_max": None,
                "constraints": [],
                "completed_retrofits": [],
                "declined_retrofits": {},
                "session_timeline": [],
            }
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"customer_id": customer_id, "constraints": [], "session_timeline": []}

    def save_memory(self, customer_id: str, data: dict[str, Any]) -> None:
        """Persist customer memory to JSON file."""
        path = self._file_path(customer_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def record_constraint(self, customer_id: str, constraint: str, session_note: str = "") -> dict[str, Any]:
        """Record a newly stated constraint (e.g. 'tenant', 'budget_3000')."""
        mem = self.get_memory(customer_id)
        if constraint not in mem.get("constraints", []):
            mem.setdefault("constraints", []).append(constraint)

        if session_note:
            mem.setdefault("session_timeline", []).append({
                "type": "constraint_added",
                "detail": session_note or constraint,
            })
        self.save_memory(customer_id, mem)
        return mem

    def record_declined(self, customer_id: str, retrofit: str, reason: str = "") -> dict[str, Any]:
        """Record customer declining a recommendation."""
        mem = self.get_memory(customer_id)
        mem.setdefault("declined_retrofits", {})[retrofit] = reason or "Declined by customer"
        constraint_key = f"reject_{retrofit}"
        if constraint_key not in mem.setdefault("constraints", []):
            mem["constraints"].append(constraint_key)

        mem.setdefault("session_timeline", []).append({
            "type": "retrofit_declined",
            "retrofit": retrofit,
            "reason": reason,
        })
        self.save_memory(customer_id, mem)
        return mem

    def record_completed(self, customer_id: str, retrofit: str, date_str: str = "2026-10") -> dict[str, Any]:
        """Record a completed retrofit, updating the home state."""
        mem = self.get_memory(customer_id)
        if retrofit not in mem.setdefault("completed_retrofits", []):
            mem["completed_retrofits"].append(retrofit)

        mem.setdefault("session_timeline", []).append({
            "type": "retrofit_completed",
            "retrofit": retrofit,
            "date": date_str,
        })
        self.save_memory(customer_id, mem)
        return mem

    def apply_memory_to_property(self, p: Property) -> Property:
        """Update property characteristics based on recorded completed retrofits."""
        mem = self.get_memory(p.customer_id)
        completed = set(mem.get("completed_retrofits", []))
        clone = p.model_copy(deep=True)

        if "loft_insulation" in completed:
            clone.loft_insulation_mm = 300.0
        if "cavity_wall_insulation" in completed or "solid_wall_insulation" in completed:
            clone.wall_insulation = "full"
        if "double_glazing" in completed:
            clone.glazing = "double_new"
        if "smart_thermostat" in completed:
            clone.smart_controls = "smart_thermostat"
        if "heat_pump" in completed:
            clone.heating = "heat_pump"
        if "solar_pv" in completed:
            clone.solar_kwp = max(clone.solar_kwp, 3.5)

        return clone
