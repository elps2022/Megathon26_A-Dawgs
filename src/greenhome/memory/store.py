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

    def get_modified_customers(self) -> list[str]:
        """Return list of customer_ids that have stored modifications/memory."""
        modified = []
        for path in self.storage_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if (
                    data.get("completed_retrofits")
                    or data.get("declined_retrofits")
                    or (data.get("budget_max") is not None)
                    or data.get("constraints")
                ):
                    modified.append(path.stem)
            except Exception:
                pass
        return sorted(modified)

    def set_budget(self, customer_id: str, budget_gbp: float | None) -> dict[str, Any]:
        """Set customer's maximum out-of-pocket budget limit."""
        mem = self.get_memory(customer_id)
        mem["budget_max"] = budget_gbp
        mem.setdefault("session_timeline", []).append({
            "type": "budget_updated",
            "budget_max": budget_gbp,
        })
        self.save_memory(customer_id, mem)
        return mem

    def set_tenure(self, customer_id: str, tenure: str) -> dict[str, Any]:
        """Set customer's housing tenure (e.g. owner_occupier, tenant)."""
        mem = self.get_memory(customer_id)
        mem["tenure"] = tenure
        if tenure == "tenant" and "tenant" not in mem.setdefault("constraints", []):
            mem["constraints"].append("tenant")
        mem.setdefault("session_timeline", []).append({
            "type": "tenure_updated",
            "tenure": tenure,
        })
        self.save_memory(customer_id, mem)
        return mem

    def record_constraint(self, customer_id: str, constraint: str, session_note: str = "") -> dict[str, Any]:
        """Record a newly stated constraint (e.g. 'tenant', 'budget_3000')."""
        mem = self.get_memory(customer_id)
        if constraint.startswith("budget_"):
            try:
                b_val = float(constraint.split("_", 1)[1])
                mem["budget_max"] = b_val
            except (ValueError, IndexError):
                pass
        if constraint not in mem.get("constraints", []):
            mem.setdefault("constraints", []).append(constraint)

        if session_note:
            mem.setdefault("session_timeline", []).append({
                "type": "constraint_added",
                "detail": session_note or constraint,
            })
        self.save_memory(customer_id, mem)
        return mem

    def record_declined(self, customer_id: str, retrofit: str, reason: str = "", date_str: str = "12 Oct 2026") -> dict[str, Any]:
        """Record customer declining a recommendation (Architecture Spec 3.11)."""
        mem = self.get_memory(customer_id)
        mem.setdefault("declined_retrofits", {})[retrofit] = {
            "reason": reason or "Declined by customer preference",
            "date": date_str,
        }
        mem.setdefault("session_timeline", []).append({
            "type": "retrofit_declined",
            "retrofit": retrofit,
            "reason": reason or "Declined by customer preference",
            "date": date_str,
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

    def remove_completed(self, customer_id: str, retrofit: str) -> dict[str, Any]:
        """Remove a retrofit from completed list."""
        mem = self.get_memory(customer_id)
        if retrofit in mem.get("completed_retrofits", []):
            mem["completed_retrofits"].remove(retrofit)
        mem.setdefault("session_timeline", []).append({
            "type": "retrofit_uncompleted",
            "retrofit": retrofit,
        })
        self.save_memory(customer_id, mem)
        return mem

    def remove_declined(self, customer_id: str, retrofit: str) -> dict[str, Any]:
        """Remove a retrofit from declined list so it can be recommended again."""
        mem = self.get_memory(customer_id)
        if retrofit in mem.get("declined_retrofits", {}):
            del mem["declined_retrofits"][retrofit]
        mem.setdefault("session_timeline", []).append({
            "type": "retrofit_restored",
            "retrofit": retrofit,
        })
        self.save_memory(customer_id, mem)
        return mem
