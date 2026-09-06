"""Configuration contracts for physically auditable critical-system BOMs."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from urllib.parse import urlparse

from src.agents.agent import MATERIAL_DENSITY_KG


ROOT = Path(__file__).resolve().parent
MASS_TOLERANCE_KG = 0.05

# These are the material batches on the bootstrap paths for power, water,
# oxygen and food.  They must describe actual field operations rather than
# relying on prose such as "fabricated in the forge".
EARLY_CRITICAL_MATERIAL_RECIPES = (
    "reduced_iron_ingot",
    "metal_pipe",
    "glass_pane",
    "finned_heat_sink",
    "insulated_fabric",
    "machined_bolts_fasteners",
    "rope_cordage",
    "vacuum_gasket_seal",
    "composite_panel",
    "electronic_component",
    "hydroponic_rack",
    "photovoltaic_laminate",
    "power_cable_harness",
    "pressure_vessel_section",
    "pump_compressor_module",
    "structural_truss_section",
    "thermal_control_loop",
    "electrolysis_stack",
)

CRITICAL_SYSTEM_RECIPES = (
    "solar_panel",
    "life_support_distribution_grid",
    "potable_water_tank",
    "oxygen_buffer_tank",
    "water_collector",
    "isru_o2_unit",
    "greenhouse",
)

MASS_BUDGET_FIELDS = (
    "input_total_kg",
    "output_total_kg",
    "process_loss_kg",
    "delivered_core_mass_kg",
)

DELIVERED_WORKSHOP_TYPES = (
    "stone_furnace",
    "forge",
    "cnc_fabricator",
)


class BomRealismContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "config" / "recipes.json").open(
            "r", encoding="utf-8"
        ) as handle:
            cls.recipes = json.load(handle)["recipes"]
        with (ROOT / "config" / "mission_profile.json").open(
            "r", encoding="utf-8"
        ) as handle:
            cls.profile = json.load(handle)

    def _assert_finite_number(self, value, label, *, positive=False):
        self.assertIsInstance(value, (int, float), f"{label} must be numeric")
        self.assertNotIsInstance(value, bool, f"{label} must not be boolean")
        self.assertTrue(math.isfinite(value), f"{label} must be finite")
        if positive:
            self.assertGreater(value, 0, f"{label} must be positive")
        else:
            self.assertGreaterEqual(value, 0, f"{label} cannot be negative")

    def _assert_source_url(self, value, label):
        self.assertIsInstance(value, str, f"{label} must be a URL string")
        parsed = urlparse(value)
        self.assertIn(
            parsed.scheme,
            {"http", "https"},
            f"{label} must use an HTTP(S) source URL",
        )
        self.assertTrue(parsed.netloc, f"{label} must contain a source host")

    def _realism(self, recipe_name):
        realism = self.recipes[recipe_name].get("realism")
        self.assertIsInstance(
            realism,
            dict,
            f"{recipe_name} must provide structured realism metadata",
        )
        return realism

    def _core_inputs(self, recipe_name):
        realism = self._realism(recipe_name)
        inputs = realism.get("field_nonmanufacturable_inputs")
        self.assertIsInstance(
            inputs,
            list,
            f"{recipe_name}.realism.field_nonmanufacturable_inputs "
            "must be an explicit list (empty is allowed)",
        )
        self.assertEqual(
            len(inputs),
            len(set(inputs)),
            f"{recipe_name} repeats a field-nonmanufacturable input",
        )
        for material in inputs:
            self.assertIsInstance(material, str)
            self.assertTrue(material.strip())
            self.assertIn(
                material,
                self.recipes[recipe_name].get("materials", {}),
                f"{recipe_name} marks {material!r} as a delivered core but "
                "does not include it in the recipe BOM",
            )
        return inputs

    def test_early_critical_material_batches_have_explicit_operations_and_mass(self):
        for recipe_name in EARLY_CRITICAL_MATERIAL_RECIPES:
            with self.subTest(recipe=recipe_name):
                self.assertIn(recipe_name, self.recipes)
                recipe = self.recipes[recipe_name]
                self.assertEqual("material", recipe.get("output", {}).get("type"))

                operations = recipe.get("manufacturing_operations")
                self.assertIsInstance(
                    operations,
                    list,
                    f"{recipe_name} needs an explicit manufacturing_operations list",
                )
                self.assertTrue(
                    operations,
                    f"{recipe_name}.manufacturing_operations cannot be empty",
                )
                self.assertEqual(
                    len(operations),
                    len(set(operations)),
                    f"{recipe_name} repeats a manufacturing operation",
                )
                for operation in operations:
                    self.assertIsInstance(operation, str)
                    self.assertTrue(operation.strip())

                budget = recipe.get("mass_budget_kg")
                self.assertIsInstance(
                    budget,
                    dict,
                    f"{recipe_name} needs an explicit mass_budget_kg",
                )
                for field in MASS_BUDGET_FIELDS:
                    self.assertIn(field, budget, f"{recipe_name} is missing {field}")

                realism = self._realism(recipe_name)
                self._assert_source_url(
                    realism.get("source_url"),
                    f"{recipe_name}.realism.source_url",
                )
                self.assertIsInstance(realism.get("basis"), str)
                self.assertTrue(
                    realism["basis"].strip(),
                    f"{recipe_name}.realism.basis cannot be empty",
                )
                self._core_inputs(recipe_name)

    def test_mass_budgets_reconcile_with_bom_units_and_delivered_cores(self):
        precision_stock = self.profile.get("delivered_precision_stock", {})
        required_budgets = set(
            EARLY_CRITICAL_MATERIAL_RECIPES + CRITICAL_SYSTEM_RECIPES
        )
        budgeted_recipes = required_budgets | {
            recipe_name
            for recipe_name, recipe in self.recipes.items()
            if isinstance(recipe, dict) and "mass_budget_kg" in recipe
        }

        for recipe_name in sorted(budgeted_recipes):
            with self.subTest(recipe=recipe_name):
                recipe = self.recipes[recipe_name]
                budget = recipe.get("mass_budget_kg")
                self.assertIsInstance(
                    budget,
                    dict,
                    f"{recipe_name} must expose an auditable mass budget",
                )
                for field in MASS_BUDGET_FIELDS:
                    self.assertIn(field, budget, f"{recipe_name} is missing {field}")
                    self._assert_finite_number(
                        budget[field], f"{recipe_name}.mass_budget_kg.{field}"
                    )
                self.assertGreater(
                    budget["input_total_kg"]
                    + budget["delivered_core_mass_kg"],
                    0,
                    "a serialized-core integration batch may have no local feedstock",
                )
                self.assertGreater(budget["output_total_kg"], 0)

                delivered_cores = set(self._core_inputs(recipe_name))
                local_input_mass = 0.0
                delivered_core_mass = 0.0
                for material, quantity in recipe.get("materials", {}).items():
                    self._assert_finite_number(
                        quantity,
                        f"{recipe_name}.materials.{material}",
                        positive=True,
                    )
                    if material in delivered_cores:
                        stock_entry = precision_stock.get(material)
                        self.assertIsInstance(
                            stock_entry,
                            dict,
                            f"delivered core {material!r} is missing from "
                            "mission_profile.delivered_precision_stock",
                        )
                        self._assert_finite_number(
                            stock_entry.get("unit_mass_kg"),
                            f"delivered_precision_stock.{material}.unit_mass_kg",
                            positive=True,
                        )
                        delivered_core_mass += (
                            quantity * stock_entry.get("unit_mass_kg", 0.0)
                        )
                    else:
                        self.assertIn(
                            material,
                            MATERIAL_DENSITY_KG,
                            f"{material!r} needs a kg-per-unit value before it can "
                            f"appear in {recipe_name}'s mass budget",
                        )
                        local_input_mass += quantity * MATERIAL_DENSITY_KG[material]

                self.assertAlmostEqual(
                    local_input_mass,
                    budget["input_total_kg"],
                    delta=MASS_TOLERANCE_KG,
                    msg=f"{recipe_name} input_total_kg must exclude delivered cores",
                )
                self.assertAlmostEqual(
                    delivered_core_mass,
                    budget["delivered_core_mass_kg"],
                    delta=MASS_TOLERANCE_KG,
                    msg=f"{recipe_name} delivered core mass disagrees with mission stock",
                )

                available_mass = (
                    budget["input_total_kg"]
                    + budget["delivered_core_mass_kg"]
                )
                self.assertLessEqual(
                    budget["output_total_kg"],
                    available_mass + MASS_TOLERANCE_KG,
                    f"{recipe_name} creates mass: output exceeds local input plus "
                    "delivered cores",
                )
                self.assertAlmostEqual(
                    available_mass,
                    budget["output_total_kg"] + budget["process_loss_kg"],
                    delta=MASS_TOLERANCE_KG,
                    msg=f"{recipe_name} mass budget does not close",
                )

                output = recipe.get("output", {})
                if output.get("type") == "material":
                    material_id = output.get("material_id")
                    self.assertIn(
                        material_id,
                        MATERIAL_DENSITY_KG,
                        f"{recipe_name} output needs a physical kg-per-unit value",
                    )
                    expected_output_mass = (
                        output.get("quantity", 0)
                        * MATERIAL_DENSITY_KG[material_id]
                    )
                    self.assertAlmostEqual(
                        expected_output_mass,
                        budget["output_total_kg"],
                        delta=MASS_TOLERANCE_KG,
                        msg=f"{recipe_name} output mass disagrees with quantity × kg/unit",
                    )

    def test_nonmanufacturable_inputs_are_serialized_delivered_stock(self):
        precision_stock = self.profile.get("delivered_precision_stock")
        self.assertIsInstance(
            precision_stock,
            dict,
            "mission_profile must define delivered_precision_stock as a mapping",
        )

        referenced_cores = set()
        for recipe_name, recipe in self.recipes.items():
            if not isinstance(recipe, dict) or "realism" not in recipe:
                continue
            referenced_cores.update(self._core_inputs(recipe_name))
        self.assertTrue(
            referenced_cores,
            "critical BOMs must identify their field-nonmanufacturable cores",
        )

        for material in sorted(referenced_cores):
            with self.subTest(material=material):
                self.assertIn(
                    material,
                    precision_stock,
                    f"{material} is nonmanufacturable in the field and must be delivered",
                )
                entry = precision_stock[material]
                self.assertIsInstance(entry, dict)
                self.assertIsInstance(entry.get("quantity"), int)
                self.assertNotIsInstance(entry.get("quantity"), bool)
                self.assertGreater(entry["quantity"], 0)
                self._assert_finite_number(
                    entry.get("unit_mass_kg"),
                    f"delivered_precision_stock.{material}.unit_mass_kg",
                    positive=True,
                )
                self.assertIs(
                    entry.get("serialized"),
                    True,
                    f"{material} must remain serialized and auditable cargo",
                )
                self._assert_source_url(
                    entry.get("source_url"),
                    f"delivered_precision_stock.{material}.source_url",
                )

    def test_critical_system_metadata_and_daily_output_rates_are_consistent(self):
        tick_minutes = self.profile["simulation_clock"]["tick_minutes"]
        self._assert_finite_number(
            tick_minutes, "simulation_clock.tick_minutes", positive=True
        )
        ticks_per_day = 24.0 * 60.0 / tick_minutes
        rate_contracts = (
            (
                "solar_panel", "firm_service_capacity_kw",
                "energy_infrastructure", 1.0,
            ),
            (
                "water_collector",
                "water_collection_rate_liters_per_tick",
                "water_extraction",
                ticks_per_day,
            ),
            ("isru_o2_unit", "o2_production_kg_per_day", "oxygen_production", 1.0),
            (
                "greenhouse",
                "food_production_kcal_per_tick",
                "food_production",
                ticks_per_day,
            ),
        )

        for recipe_name, effect_key, contribution_key, multiplier in rate_contracts:
            with self.subTest(recipe=recipe_name):
                realism = self._realism(recipe_name)
                self._assert_source_url(
                    realism.get("source_url"),
                    f"{recipe_name}.realism.source_url",
                )
                self.assertIsInstance(realism.get("basis"), str)
                self.assertTrue(
                    realism["basis"].strip(),
                    f"{recipe_name}.realism.basis cannot be empty",
                )
                self._core_inputs(recipe_name)

                recipe = self.recipes[recipe_name]
                effect_rate = recipe["output"]["effects"].get(effect_key)
                contribution_rate = recipe.get("colony_contribution", {}).get(
                    contribution_key
                )
                self._assert_finite_number(
                    effect_rate, f"{recipe_name}.output.effects.{effect_key}", positive=True
                )
                self._assert_finite_number(
                    contribution_rate,
                    f"{recipe_name}.colony_contribution.{contribution_key}",
                    positive=True,
                )
                # Recipe JSON was authored against the legacy five-minute
                # clock; runtime normalization scales continuous per-tick
                # rates when mission tick resolution changes.
                runtime_effect_rate = (
                    effect_rate * tick_minutes / 5.0
                    if effect_key.endswith("_per_tick") else effect_rate
                )
                expected_daily_rate = runtime_effect_rate * multiplier
                self.assertAlmostEqual(
                    expected_daily_rate,
                    contribution_rate,
                    delta=max(0.01, expected_daily_rate * 1e-6),
                    msg=f"{recipe_name} effect rate and colony contribution use "
                    "different physical rates",
                )

    def test_o2_electrolysis_obeys_water_mass_stoichiometry(self):
        effects = self.recipes["isru_o2_unit"]["output"]["effects"]
        oxygen_kg_day = effects.get("o2_production_kg_per_day")
        water_kg_day = effects.get("water_consumption_kg_per_day")
        hydrogen_kg_day = effects.get("h2_byproduct_kg_per_day")

        self._assert_finite_number(
            oxygen_kg_day, "isru_o2_unit.output.effects.o2_production_kg_per_day",
            positive=True,
        )
        self._assert_finite_number(
            water_kg_day,
            "isru_o2_unit.output.effects.water_consumption_kg_per_day",
            positive=True,
        )
        self._assert_finite_number(
            hydrogen_kg_day,
            "isru_o2_unit.output.effects.h2_byproduct_kg_per_day",
            positive=True,
        )
        self.assertAlmostEqual(6.0, oxygen_kg_day, places=9)
        self.assertAlmostEqual(6.75, water_kg_day, places=9)
        self.assertAlmostEqual(0.75, hydrogen_kg_day, places=9)
        self.assertAlmostEqual(oxygen_kg_day * 1.125, water_kg_day, places=9)
        self.assertAlmostEqual(oxygen_kg_day * 0.125, hydrogen_kg_day, places=9)
        self.assertAlmostEqual(
            water_kg_day,
            oxygen_kg_day + hydrogen_kg_day,
            places=9,
        )

        ticks_per_day = (
            24.0 * 60.0 / self.profile["simulation_clock"]["tick_minutes"]
        )
        if "feedstock_consumption_per_tick" in effects:
            self.assertAlmostEqual(
                water_kg_day,
                effects["feedstock_consumption_per_tick"] * ticks_per_day,
                delta=1e-6,
                msg="legacy per-tick water consumption contradicts daily stoichiometry",
            )
        if "h2_byproduct_per_tick" in effects:
            self.assertAlmostEqual(
                hydrogen_kg_day,
                effects["h2_byproduct_per_tick"] * ticks_per_day,
                delta=1e-6,
                msg="legacy per-tick hydrogen output contradicts daily stoichiometry",
            )

    def test_manufacturing_operations_fit_delivered_workshops(self):
        industry = self.profile.get("delivered_industry")
        self.assertIsInstance(industry, dict)
        capabilities = industry.get("capabilities")
        self.assertIsInstance(
            capabilities,
            dict,
            "delivered_industry.capabilities must map each physical workshop type",
        )

        for machine_type in DELIVERED_WORKSHOP_TYPES:
            with self.subTest(machine=machine_type):
                count_key = f"{machine_type}_count"
                self.assertIn(count_key, industry)
                self.assertIsInstance(industry[count_key], int)
                self.assertNotIsInstance(industry[count_key], bool)
                self.assertEqual(
                    3 if machine_type == "cnc_fabricator" else 2,
                    industry[count_key],
                    f"the audited cargo manifest must cover every {machine_type} workshop",
                )
                self.assertIn(machine_type, capabilities)
                machine_operations = capabilities[machine_type]
                self.assertIsInstance(machine_operations, list)
                self.assertTrue(machine_operations)
                self.assertEqual(len(machine_operations), len(set(machine_operations)))
                for operation in machine_operations:
                    self.assertIsInstance(operation, str)
                    self.assertTrue(operation.strip())

        for recipe_name in EARLY_CRITICAL_MATERIAL_RECIPES:
            recipe = self.recipes[recipe_name]
            machine_type = recipe.get("requires_structure")
            if machine_type is None:
                continue
            with self.subTest(recipe=recipe_name, machine=machine_type):
                self.assertIn(
                    machine_type,
                    DELIVERED_WORKSHOP_TYPES,
                    f"{recipe_name} requires an undelivered workshop type",
                )
                self.assertGreater(industry[f"{machine_type}_count"], 0)
                unsupported = set(recipe.get("manufacturing_operations", ())) - set(
                    capabilities[machine_type]
                )
                self.assertFalse(
                    unsupported,
                    f"{recipe_name} assigns unsupported operations to {machine_type}: "
                    f"{sorted(unsupported)}",
                )

    def test_sulfur_is_optional_measured_feedstock_not_a_hidden_critical_gate(self):
        self.assertNotIn("chemical_battery", self.recipes)
        self.assertEqual(
            {}, self.profile.get("conditional_feedstocks_if_planet_absent", {})
        )

        def assert_no_sulfur_dependency(recipe_name, visited=None):
            visited = set() if visited is None else visited
            if recipe_name in visited:
                return
            visited.add(recipe_name)
            recipe = self.recipes[recipe_name]
            for material in recipe.get("materials", {}):
                self.assertNotEqual(
                    "sulfur", material,
                    f"critical recipe {recipe_name} silently depends on sulfur",
                )
                component = self.recipes.get(material)
                if isinstance(component, dict) and component.get(
                    "output", {}
                ).get("type") == "material":
                    assert_no_sulfur_dependency(material, visited)

        for recipe_name in CRITICAL_SYSTEM_RECIPES:
            assert_no_sulfur_dependency(recipe_name)

        for recipe in self.recipes.values():
            if isinstance(recipe, dict):
                effects = recipe.get("output", {}).get("effects", {})
                self.assertNotIn("fuel_consumption_sulfur_per_tick", effects)

        paver = self.recipes["sulfur_regolith_paver"]
        sulfur_kg = paver["materials"]["sulfur"] * MATERIAL_DENSITY_KG["sulfur"]
        regolith_kg = (
            paver["materials"]["regolith"] * MATERIAL_DENSITY_KG["regolith"]
        )
        self.assertAlmostEqual(57.5, sulfur_kg + regolith_kg, places=9)
        self.assertAlmostEqual(
            paver["output"]["effects"]["sulfur_mass_fraction"],
            sulfur_kg / (sulfur_kg + regolith_kg),
            delta=0.001,
        )
        capabilities = set(
            self.profile["delivered_industry"]["capabilities"]["cnc_fabricator"]
        )
        self.assertFalse(set(paver["manufacturing_operations"]) - capabilities)

    def test_hot_work_cell_is_not_a_global_prefab_assembly_gate(self):
        self.assertEqual(
            "forge", self.recipes["structural_truss_section"]["requires_structure"]
        )
        self.assertIsNone(self.recipes["habitat_module"]["requires_structure"])
        self.assertIsNone(
            self.recipes["power_distribution_grid"]["requires_structure"]
        )
        self.assertEqual(
            "cnc_fabricator",
            self.recipes["protective_suit"]["requires_structure"],
        )
        forge_ops = set(
            self.profile["delivered_industry"]["capabilities"]["forge"]
        )
        self.assertEqual({"hot_forming", "welding", "brazing"}, forge_ops)


if __name__ == "__main__":
    unittest.main()
