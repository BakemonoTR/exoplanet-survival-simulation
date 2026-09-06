"""Mission architecture and civilian-arrival acceptance rules.

This module deliberately contains no UI and no generative-model calls.  It is
the auditable contract between physical simulation time, the six-person
advance mission and the incoming civilian colony.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Optional


DEFAULT_PROFILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "mission_profile.json",
)


@dataclass(frozen=True)
class MissionClock:
    tick_minutes: float
    realtime_seconds_per_tick: float
    public_challenge_days: int
    planet_attempt_max_earth_days: Optional[int]
    civilian_support_window_days: int

    @property
    def ticks_per_earth_day(self) -> int:
        return int(round(24.0 * 60.0 / self.tick_minutes))

    @property
    def planet_attempt_max_ticks(self) -> Optional[int]:
        if self.planet_attempt_max_earth_days is None:
            return None
        return self.planet_attempt_max_earth_days * self.ticks_per_earth_day

    @property
    def support_window_ticks(self) -> int:
        return self.civilian_support_window_days * self.ticks_per_earth_day

    @property
    def public_challenge_ticks(self) -> int:
        return int(round(
            self.public_challenge_days * 24.0 * 60.0 * 60.0
            / self.realtime_seconds_per_tick
        ))

    @property
    def public_challenge_simulated_days(self) -> float:
        return (
            self.public_challenge_ticks * self.tick_minutes / 1440.0
        )

    @property
    def full_attempts_in_public_challenge(self) -> Optional[float]:
        attempt_ticks = self.planet_attempt_max_ticks
        if not attempt_ticks:
            return None
        return self.public_challenge_ticks / attempt_ticks

    def elapsed(self, tick: int) -> dict[str, float | int]:
        minutes = max(0, int(tick)) * self.tick_minutes
        return {
            "tick": max(0, int(tick)),
            "elapsed_earth_hours": round(minutes / 60.0, 3),
            "elapsed_earth_days": round(minutes / 1440.0, 4),
            "remaining_earth_days": (
                round(
                    max(
                        0.0,
                        self.planet_attempt_max_earth_days - minutes / 1440.0,
                    ),
                    4,
                )
                if self.planet_attempt_max_earth_days is not None else None
            ),
        }


class MissionProfile:
    """Validated read-only mission configuration."""

    def __init__(self, data: dict[str, Any]):
        self.data = data
        clock = data.get("simulation_clock", {})
        self.clock = MissionClock(
            tick_minutes=float(clock.get("tick_minutes", 5.0)),
            realtime_seconds_per_tick=float(
                clock.get("realtime_seconds_per_tick", 4.0)
            ),
            public_challenge_days=int(clock.get("public_challenge_days", 90)),
            planet_attempt_max_earth_days=(
                int(clock["planet_attempt_max_earth_days"])
                if clock.get("planet_attempt_max_earth_days") is not None
                else None
            ),
            civilian_support_window_days=int(
                clock.get("civilian_support_window_days", 30)
            ),
        )
        population = data.get("population", {})
        self.advance_crew = int(population.get("advance_crew", 6))
        self.arriving_civilians = int(population.get("arriving_civilians", 100))
        self.advance_crew_departure_planned = bool(
            population.get("advance_crew_departure_planned", False)
        )
        self.operational_support_population = int(population.get(
            "operational_support_population",
            self.arriving_civilians + (
                0 if self.advance_crew_departure_planned else self.advance_crew
            ),
        ))
        self.advance_crew_consumables = dict(
            data.get("advance_crew_consumables", {})
        )
        self.advance_crew_power = dict(data.get("advance_crew_power", {}))
        self.arrival_contract = dict(data.get("arrival_contract", {}))
        self.cargo_architecture = dict(data.get("cargo_architecture", {}))
        self.crew_lander_architecture = dict(
            data.get("crew_lander_architecture", {})
        )
        self.decision_authority = dict(data.get("decision_authority", {}))
        self.production_policy = dict(data.get("production_policy", {}))
        self.crew_surface_mobility = dict(
            data.get("crew_surface_mobility", {})
        )
        self.delivered_industry = dict(data.get("delivered_industry", {}))
        self.delivered_precision_stock = {
            str(material): dict(manifest)
            for material, manifest in dict(
                data.get("delivered_precision_stock", {})
            ).items()
            if isinstance(manifest, dict)
            and int(manifest.get("quantity", 0)) > 0
        }
        self.delivered_field_stock = {
            str(material): dict(manifest)
            for material, manifest in dict(
                data.get("delivered_field_stock", {})
            ).items()
            if isinstance(manifest, dict)
            and int(manifest.get("quantity", 0)) > 0
        }
        self.delivered_structure_kits = {
            str(recipe): int(count)
            for recipe, count in dict(
                data.get("delivered_structure_kits", {})
            ).items()
            if int(count) > 0
        }
        self.conditional_feedstocks = dict(
            data.get("conditional_feedstocks_if_planet_absent", {})
        )
        self.surface_fleet = dict(data.get("surface_fleet", {}))
        self.grid_cell_meters = float(
            data.get("surface_scale", {}).get("grid_cell_meters", 100.0)
        )
        self._validate()

    @classmethod
    def load(cls, path: str | None = None) -> "MissionProfile":
        with open(path or DEFAULT_PROFILE_PATH, "r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def _validate(self) -> None:
        if self.clock.tick_minutes <= 0:
            raise ValueError("mission tick_minutes must be positive")
        if self.clock.realtime_seconds_per_tick <= 0:
            raise ValueError("realtime seconds per tick must be positive")
        if self.advance_crew != 6:
            raise ValueError("ASAC-6 mission requires exactly six advance crew")
        if self.arriving_civilians != 100:
            raise ValueError("arrival contract requires exactly 100 civilians")
        expected_support_population = self.arriving_civilians + (
            0 if self.advance_crew_departure_planned else self.advance_crew
        )
        if self.operational_support_population != expected_support_population:
            raise ValueError(
                "operational support population must include every surface "
                "occupant at civilian arrival"
            )
        for key in ("potable_water_l", "oxygen_kg", "food_kcal"):
            if float(self.advance_crew_consumables.get(key, 0.0)) <= 0.0:
                raise ValueError(f"advance crew consumable {key} must be positive")
        for key in (
            "battery_capacity_kwh", "initial_charge_kwh",
            "integrated_lander_solar_peak_kw",
            "auxiliary_fuel_cell_power_kw", "auxiliary_reactant_energy_kwh",
        ):
            if float(self.advance_crew_power.get(key, 0.0)) <= 0.0:
                raise ValueError(f"advance crew power value {key} must be positive")
        reserve_fraction = float(self.advance_crew_power.get(
            "emergency_battery_reserve_fraction", 0.0
        ))
        if not 0.0 < reserve_fraction < 1.0:
            raise ValueError(
                "emergency_battery_reserve_fraction must be between zero and one"
            )
        if self.decision_authority.get("language_model") != "dialogue_only":
            raise ValueError("language model authority must remain dialogue_only")
        if self.decision_authority.get("language_model_may_mutate_state", True):
            raise ValueError("language model must not mutate simulation state")
        if self.grid_cell_meters <= 0:
            raise ValueError("grid cell scale must be positive")
        for material, manifest in self.delivered_precision_stock.items():
            if float(manifest.get("unit_mass_kg", 0.0)) <= 0.0:
                raise ValueError(
                    f"delivered precision stock {material} needs a positive unit mass"
                )
            if not manifest.get("serialized", False):
                raise ValueError(
                    f"flight-qualified precision stock {material} must be serialized"
                )
        for material, manifest in self.delivered_field_stock.items():
            if float(manifest.get("unit_mass_kg", 0.0)) <= 0.0:
                raise ValueError(
                    f"delivered field stock {material} needs a positive unit mass"
                )
        delivered_only = {
            str(material) for material in self.production_policy.get(
                "delivered_only_materials", []
            )
        }
        unknown_delivered_only = delivered_only.difference(
            self.delivered_precision_stock
        )
        if unknown_delivered_only:
            raise ValueError(
                "delivered-only production materials need serialized cargo: "
                + ", ".join(sorted(unknown_delivered_only))
            )
        payload_ceiling = float(self.cargo_architecture.get(
            "total_delivered_payload_ceiling_kg", 0.0
        ))
        precision_payload_mass = sum(
            int(manifest.get("quantity", 0))
            * float(manifest.get("unit_mass_kg", 0.0))
            for manifest in self.delivered_precision_stock.values()
        )
        if payload_ceiling <= 0.0:
            raise ValueError("cargo payload ceiling must be positive")
        if precision_payload_mass > payload_ceiling:
            raise ValueError(
                "serialized precision cargo exceeds the mission payload ceiling"
            )
        if self.cargo_architecture.get("readiness_credit_on_delivery", True):
            raise ValueError("delivered prefab cargo must not receive readiness credit")
        if (
            self.clock.planet_attempt_max_earth_days is not None
            and self.clock.planet_attempt_max_earth_days <= 0
        ):
            raise ValueError("planet attempt duration must be positive or unset")

    def mission_state(
        self,
        tick: int,
        colony_score: dict[str, Any],
        structures: dict[str, int],
        communications_operational: bool | None = None,
        landing_zone_operational: bool | None = None,
        support_soak_ticks: int = 0,
    ) -> dict[str, Any]:
        """Return backend-only civilian arrival and research-contract status."""
        categories = colony_score.get("categories", {}) or {}
        required_score = float(
            self.arrival_contract.get("required_colony_score_percent", 100.0)
        )
        balanced = bool(categories) and all(
            float(value) >= 1.0 for value in categories.values()
        )
        communications_ready = (
            structures.get("communications_array", 0) > 0
            if communications_operational is None
            else bool(communications_operational)
        )
        score_ready = float(colony_score.get("overall", 0.0)) >= required_score
        landing_ready = (
            structures.get("landing_zone", 0) > 0
            if landing_zone_operational is None
            else bool(landing_zone_operational)
        )
        required_soak_days = int(self.arrival_contract.get(
            "required_continuous_soak_days",
            self.clock.civilian_support_window_days,
        ))
        required_soak_ticks = required_soak_days * self.clock.ticks_per_earth_day
        completed_soak_ticks = max(0, int(support_soak_ticks))
        infrastructure_deadline_day = int(self.arrival_contract.get(
            "infrastructure_complete_by_day",
            max(0, (self.clock.planet_attempt_max_earth_days or 0)
                - required_soak_days),
        ))
        infrastructure_deadline_tick = (
            infrastructure_deadline_day * self.clock.ticks_per_earth_day
        )
        qualification_start_tick = (
            max(0, int(tick) - completed_soak_ticks + 1)
            if completed_soak_ticks > 0 else max(0, int(tick))
        )
        deadline_met = (
            qualification_start_tick <= infrastructure_deadline_tick
        )
        gates = {
            "capacity_score": score_ready,
            "balanced_categories": (
                balanced
                if self.arrival_contract.get("requires_balanced_categories", True)
                else True
            ),
            "communications": (
                communications_ready
                if self.arrival_contract.get("requires_communications_array", True)
                else True
            ),
            "operational_landing_zone": (
                landing_ready
                if self.arrival_contract.get(
                    "requires_operational_landing_zone", True
                ) else True
            ),
            "infrastructure_commissioned_by_deadline": deadline_met,
            # The engine increments this counter only while every other
            # arrival gate and every operational capacity remains valid.
            # Any outage resets the qualification run.
            "thirty_day_support_model": (
                completed_soak_ticks >= required_soak_ticks
                if self.arrival_contract.get(
                    "requires_thirty_day_support_model", True
                ) else True
            ),
        }
        return {
            "name": self.data.get("mission_name", "ASAC-6"),
            "advance_crew": self.advance_crew,
            "arriving_civilians": self.arriving_civilians,
            "operational_support_population": self.operational_support_population,
            "clock": self.clock.elapsed(tick),
            "planet_attempt_max_ticks": self.clock.planet_attempt_max_ticks,
            "support_window_days": self.clock.civilian_support_window_days,
            "support_soak": {
                "completed_ticks": completed_soak_ticks,
                "required_ticks": required_soak_ticks,
                "completed_days": round(
                    completed_soak_ticks / self.clock.ticks_per_earth_day, 3
                ),
                "required_days": required_soak_days,
                "qualification_start_tick": qualification_start_tick,
                "infrastructure_deadline_day": infrastructure_deadline_day,
                "infrastructure_deadline_tick": infrastructure_deadline_tick,
            },
            "cargo": {
                "payload_ceiling_kg": float(self.cargo_architecture.get(
                    "total_delivered_payload_ceiling_kg", 0.0
                )),
                "serialized_precision_payload_kg": round(sum(
                    int(manifest.get("quantity", 0))
                    * float(manifest.get("unit_mass_kg", 0.0))
                    for manifest in self.delivered_precision_stock.values()
                ), 3),
                "readiness_credit_on_delivery": False,
            },
            "decision_authority": dict(self.decision_authority),
            "arrival_gates": gates,
            "surface_acceptance_ready": all(gates.values()),
        }
