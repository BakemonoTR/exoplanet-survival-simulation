"""Closed deployment-payload ledger for the emergency surface mission."""

from __future__ import annotations

from typing import Any


class CargoLedgerError(ValueError):
    """Raised when declared landed hardware exceeds a physical payload cap."""


class CargoMassLedger:
    """Close the initial cargo boundary without granting readiness credit."""

    def __init__(self, profile, recipes: dict[str, dict[str, Any]]):
        self.profile = profile
        self.recipes = recipes
        self.lines: list[dict[str, Any]] = []
        self.crew_lines: list[dict[str, Any]] = []
        self.reservations: list[dict[str, Any]] = []
        self._build()

    @staticmethod
    def _positive(value: Any, label: str) -> float:
        result = float(value)
        if result < 0.0:
            raise CargoLedgerError(f"negative cargo mass for {label}")
        return result

    def _add(self, category: str, item: str, quantity: float, unit_mass: float):
        quantity = self._positive(quantity, item)
        unit_mass = self._positive(unit_mass, item)
        self.lines.append({
            "category": category, "item": item, "quantity": quantity,
            "unit_mass_kg": unit_mass, "mass_kg": quantity * unit_mass,
        })

    def _add_crew(
        self, category: str, item: str, quantity: float, unit_mass: float
    ):
        quantity = self._positive(quantity, item)
        unit_mass = self._positive(unit_mass, item)
        self.crew_lines.append({
            "category": category, "item": item, "quantity": quantity,
            "unit_mass_kg": unit_mass, "mass_kg": quantity * unit_mass,
        })

    def _build(self) -> None:
        for item, manifest in self.profile.delivered_precision_stock.items():
            self._add(
                "serialized_flight_hardware", item,
                int(manifest.get("quantity", 0)),
                manifest.get("unit_mass_kg", 0.0),
            )
        for item, manifest in self.profile.delivered_field_stock.items():
            self._add(
                "standard_field_assembly_stock", item,
                int(manifest.get("quantity", 0)),
                manifest.get("unit_mass_kg", 0.0),
            )
        packed_quantities = {
            material: int(manifest.get("quantity", 0))
            for material, manifest in {
                **self.profile.delivered_precision_stock,
                **self.profile.delivered_field_stock,
            }.items()
        }
        reserved_quantities: dict[str, int] = {}
        for recipe_name, count in self.profile.delivered_structure_kits.items():
            recipe = self.recipes.get(recipe_name, {})
            materials = {
                str(material): int(quantity) * int(count)
                for material, quantity in recipe.get("materials", {}).items()
                if int(quantity) > 0
            }
            if not materials:
                raise CargoLedgerError(
                    f"delivered structure kit {recipe_name} has no material BOM"
                )
            missing = {
                material: quantity
                for material, quantity in materials.items()
                if reserved_quantities.get(material, 0) + quantity
                > packed_quantities.get(material, 0)
            }
            if missing:
                raise CargoLedgerError(
                    f"delivered structure kit {recipe_name} exceeds packed "
                    f"field/precision stock: {missing}"
                )
            for material, quantity in materials.items():
                reserved_quantities[material] = (
                    reserved_quantities.get(material, 0) + quantity
                )
            self.reservations.append({
                "category": "reserved_structure_kit_allocation",
                "item": recipe_name,
                "quantity": int(count),
                "materials": materials,
                "additional_payload_mass_kg": 0.0,
            })

        industry = self.profile.delivered_industry
        for machine in ("stone_furnace", "forge", "cnc_fabricator"):
            self._add(
                "surface_industry", machine,
                industry.get(f"{machine}_count", 0),
                industry.get(f"{machine}_unit_mass_kg", 0.0),
            )
        for vehicle, spec in self.profile.surface_fleet.items():
            self._add(
                "surface_fleet", vehicle,
                spec.get("count", 0), spec.get("dry_mass_kg", 0.0),
            )

        unwrapped = sum(line["mass_kg"] for line in self.lines)
        packaging_fraction = self._positive(
            self.profile.cargo_architecture.get(
                "packaging_restraint_fraction", 0.0
            ), "packaging_restraint_fraction",
        )
        self._add(
            "packaging_and_restraints", "cargo_packaging", 1.0,
            unwrapped * packaging_fraction,
        )

        consumables = self.profile.advance_crew_consumables
        crew = self.profile.crew_lander_architecture
        self._add_crew(
            "survival_consumables", "potable_water",
            consumables.get("potable_water_l", 0.0), 1.0,
        )
        self._add_crew(
            "survival_consumables", "oxygen",
            consumables.get("oxygen_kg", 0.0), 1.0,
        )
        food_density = max(
            1.0,
            float(crew.get("packaged_food_energy_density_kcal_per_kg", 1.0)),
        )
        self._add_crew(
            "survival_consumables", "packaged_food", 1.0,
            float(consumables.get("food_kcal", 0.0)) / food_density,
        )
        self._add_crew(
            "crew_lander_systems", "integrated_life_support_power_and_tankage",
            1.0,
            crew.get("integrated_life_support_power_and_tankage_kg", 0.0),
        )
        self._add_crew(
            "crew_lander_systems", "crew_suits_tools_medical_and_personal",
            1.0,
            crew.get("crew_suits_tools_medical_and_personal_kg", 0.0),
        )

        if self.uncrewed_payload_mass_kg > self.uncrewed_payload_ceiling_kg:
            raise CargoLedgerError(
                "uncrewed cargo exceeds delivered payload ceiling by "
                f"{self.uncrewed_payload_mass_kg - self.uncrewed_payload_ceiling_kg:.1f} kg"
            )
        if self.crew_payload_mass_kg > self.crew_payload_ceiling_kg:
            raise CargoLedgerError(
                "crew lander payload exceeds ceiling by "
                f"{self.crew_payload_mass_kg - self.crew_payload_ceiling_kg:.1f} kg"
            )

    @property
    def uncrewed_payload_ceiling_kg(self) -> float:
        return float(self.profile.cargo_architecture.get(
            "total_delivered_payload_ceiling_kg", 0.0
        ))

    @property
    def crew_payload_ceiling_kg(self) -> float:
        return float(self.profile.crew_lander_architecture.get(
            "payload_ceiling_kg", 0.0
        ))

    @property
    def uncrewed_payload_mass_kg(self) -> float:
        return sum(float(line["mass_kg"]) for line in self.lines)

    @property
    def crew_payload_mass_kg(self) -> float:
        return sum(float(line["mass_kg"]) for line in self.crew_lines)

    def to_dict(self) -> dict[str, Any]:
        uncrewed = self.uncrewed_payload_mass_kg
        crew = self.crew_payload_mass_kg
        return {
            "closed": True,
            "readiness_credit": False,
            "uncrewed": {
                "lines": self.lines,
                "reservations": self.reservations,
                "payload_mass_kg": round(uncrewed, 3),
                "payload_ceiling_kg": self.uncrewed_payload_ceiling_kg,
                "reserve_kg": round(
                    self.uncrewed_payload_ceiling_kg - uncrewed, 3
                ),
            },
            "crew_lander": {
                "lines": self.crew_lines,
                "payload_mass_kg": round(crew, 3),
                "payload_ceiling_kg": self.crew_payload_ceiling_kg,
                "reserve_kg": round(self.crew_payload_ceiling_kg - crew, 3),
            },
        }
