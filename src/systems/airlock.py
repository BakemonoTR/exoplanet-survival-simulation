"""Bounded FIFO suitlock cycles on the mission clock (not a CFD model)."""

from dataclasses import dataclass, field


@dataclass
class AirlockController:
    cycle_ticks: int
    capacity: int = 2
    queue: list = field(default_factory=list)
    active: dict | None = None
    permits: dict = field(default_factory=dict)
    completed_cycles: int = 0
    return_reserves: dict = field(default_factory=dict)
    blocked_reason: str | None = None

    def advance(self, tick):
        """Advance completed pressure cycles from the mission clock.

        Cycle completion is a property of elapsed time, not of the original
        occupant asking for the chamber again.  Advancing once per simulation
        tick prevents an interrupted/self-care action from leaving a finished
        chamber occupied indefinitely.
        """
        self.permits = {
            key: valid_until
            for key, valid_until in self.permits.items()
            if valid_until >= tick
        }
        if self.active and tick >= self.active["ready_tick"]:
            completed_key = self.active["key"]
            self.completed_cycles += 1
            # Processing order may put the original occupant after another
            # requester.  Retain the permit through the following tick.
            self.permits[completed_key] = tick + 1
            self.active = None

    def charge_cycle(self, crew_ids, direction, stores):
        """Reserve return consumables from existing stores, never create them.

        Each departing person reserves a solo return, allowing a separated or
        rescued crew to enter. A shared return refunds its unused allocation.
        Reserved energy is part of the installed battery, not extra capacity.
        """
        ids = tuple(sorted(set(crew_ids)))
        self.blocked_reason = None
        if direction == "out":
            missing = [i for i in ids if i not in self.return_reserves]
            required_energy = 0.1 * (1 + len(missing))
            required_o2 = 0.03 * (1 + len(missing))
            reserved_energy = reserved_o2 = 0.0
        else:
            missing = []
            reserved_energy = sum(self.return_reserves.get(i, {}).get("energy", 0.0) for i in ids)
            reserved_o2 = sum(self.return_reserves.get(i, {}).get("o2", 0.0) for i in ids)
            required_energy = max(0.0, 0.1 - reserved_energy)
            required_o2 = max(0.0, 0.03 - reserved_o2)
        if stores.get("energy_stored_kwh", 0.0) + 1e-9 < required_energy:
            self.blocked_reason = "insufficient_airlock_energy"
            return False
        if stores.get("o2_reserve_kg", 0.0) + 1e-9 < required_o2:
            self.blocked_reason = "insufficient_airlock_oxygen"
            return False
        stores["energy_stored_kwh"] = max(0.0, stores.get("energy_stored_kwh", 0.0) - required_energy)
        stores["o2_reserve_kg"] = max(0.0, stores.get("o2_reserve_kg", 0.0) - required_o2)
        if direction == "out":
            for i in missing:
                self.return_reserves[i] = {"energy": 0.1, "o2": 0.03}
        else:
            for i in ids:
                self.return_reserves.pop(i, None)
            stores["energy_stored_kwh"] += max(0.0, reserved_energy - 0.1)
            stores["o2_reserve_kg"] += max(0.0, reserved_o2 - 0.03)
        stores["airlock_reserved_energy_kwh"] = sum(v["energy"] for v in self.return_reserves.values())
        stores["airlock_reserved_o2_kg"] = sum(v["o2"] for v in self.return_reserves.values())
        return True

    def cancel_outbound(self, crew_ids, stores):
        """Cancel a sheltered departure and release its unused return reserve."""
        ids = tuple(sorted(set(crew_ids)))
        if not ids:
            return False
        key = (ids, "out")
        changed = False
        before = len(self.queue)
        self.queue = [entry for entry in self.queue if entry.get("key") != key]
        changed = len(self.queue) != before
        if self.active and self.active.get("key") == key:
            self.active = None
            changed = True
        if key in self.permits:
            self.permits.pop(key, None)
            changed = True
        refunded_energy = 0.0
        refunded_o2 = 0.0
        for crew_id in ids:
            reserve = self.return_reserves.pop(crew_id, None)
            if reserve:
                refunded_energy += float(reserve.get("energy", 0.0))
                refunded_o2 += float(reserve.get("o2", 0.0))
                changed = True
        stores["energy_stored_kwh"] = (
            float(stores.get("energy_stored_kwh", 0.0)) + refunded_energy
        )
        stores["o2_reserve_kg"] = (
            float(stores.get("o2_reserve_kg", 0.0)) + refunded_o2
        )
        stores["airlock_reserved_energy_kwh"] = sum(
            float(value.get("energy", 0.0))
            for value in self.return_reserves.values()
        )
        stores["airlock_reserved_o2_kg"] = sum(
            float(value.get("o2", 0.0))
            for value in self.return_reserves.values()
        )
        return changed

    def request(self, crew_ids, direction, tick, charge=lambda: True):
        self.blocked_reason = None
        ids = tuple(sorted(set(crew_ids)))
        if not ids or len(ids) > self.capacity or direction not in {"in", "out"}:
            return False
        key = (ids, direction)
        expiry = max(6, self.cycle_ticks * 3)
        self.advance(tick)
        if key in self.permits:
            return True
        # A cancelled route cannot own the chamber or queue forever.
        self.queue = [q for q in self.queue if tick - q["last_seen"] <= expiry]
        if self.active and tick - self.active["last_seen"] > expiry:
            self.active = None
        if self.active and self.active["key"] == key:
            self.active["last_seen"] = tick
            return False
        entry = next((q for q in self.queue if q["key"] == key), None)
        if entry is None:
            entry = {"key": key, "last_seen": tick}
            # A new EVA with no available launch energy may not block an
            # inbound crew whose return allocation is already funded.
            # Preserve FIFO within each direction; never interrupt a cycle.
            if direction == "in":
                first_out = next((i for i, q in enumerate(self.queue) if q["key"][1] == "out"), len(self.queue))
                self.queue.insert(first_out, entry)
            else:
                self.queue.append(entry)
        entry["last_seen"] = tick
        if self.active is None and self.queue[0] is entry:
            if charge():
                self.queue.pop(0)
                self.active = {**entry, "ready_tick": tick + self.cycle_ticks,
                               "phase": "depressurizing" if direction == "out" else "repressurizing"}
            else:
                # A resource-blocked head cannot strand a funded return
                # behind it. Keep the request for retry without spending or
                # fabricating its missing consumables; active cycles stay FIFO.
                self.queue.append(self.queue.pop(0))
        return False

    def snapshot(self):
        return {
            "capacity": self.capacity, "cycle_ticks": self.cycle_ticks,
            "phase": self.active["phase"] if self.active else "idle",
            "occupants": list(self.active["key"][0]) if self.active else [],
            "ready_tick": self.active["ready_tick"] if self.active else None,
            "queue": [{"crew_ids": list(q["key"][0]), "direction": q["key"][1]} for q in self.queue],
            "completed_cycles": self.completed_cycles,
            "blocked_reason": self.blocked_reason,
            "return_reserves": {k: dict(v) for k, v in self.return_reserves.items()},
        }
