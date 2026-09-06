"""Focused regression tests for physical potable-reserve dispatch."""

import unittest
from pathlib import Path

from src.agents.agent import MATERIAL_DENSITY_KG, create_team_from_presets
from src.memory import vector_store
from src.orchestration.engine import SimulationEngine


ROOT = Path(__file__).resolve().parent
vector_store._use_tfidf_fallback = True


class WaterReserveDispatchTest(unittest.TestCase):
    def setUp(self):
        self.engine = SimulationEngine(
            str(ROOT / "config" / "planets" / "kepler-442b.json"),
            seed=42,
            db=False,
            max_ticks=5,
        )
        for agent in create_team_from_presets(
            str(ROOT / "config" / "agent_presets.json")
        ):
            self.engine.add_agent(agent)
        self.engine._init_agents()
        self.engine._initialized = True
        # Isolate the actual production/network physics from new crew orders.
        self.engine._process_agent_tick = lambda *_args, **_kwargs: {}
        self.engine.event_scheduler.events = []
        self.engine.get_day_night_phase = lambda: {
            "phase": "night", "solar_intensity": 0.0,
            "temp_modifier_c": 0.0,
        }
        original_cell_info = self.engine.world.get_cell_info
        self.temperature_c = 20.0
        self.engine.world.get_cell_info = lambda *args, **kwargs: {
            **original_cell_info(*args, **kwargs),
            "temperature_c": self.temperature_c,
        }
        self.engine._colony_resources["lander_auxiliary_energy_remaining_kwh"] = 0.0
        self.engine._colony_resources["energy_stored_kwh"] = 120.0
        self._add_structure("fluid-grid", "life_support_distribution_grid", 4, -4)
        self._add_structure("collector", "water_collector", 7, -4)
        self._add_structure("water-tank", "potable_water_tank", 6, -5)

    def _add_structure(self, name, kind, dx, dy):
        structure = {
            "id": name, "type": kind,
            "x": self.engine.lz_x + dx, "y": self.engine.lz_y + dy,
            "health": 1.0,
        }
        self.engine.placed_structures.append(structure)
        self.engine.structures_built[kind] = (
            self.engine.structures_built.get(kind, 0) + 1
        )
        return structure

    def _run(self):
        result = self.engine._run_tick()
        self.assertEqual([], result["agent_processing_errors"])
        return result["colony_production"]

    def _install_six_tanks(self):
        for index, (dx, dy) in enumerate(((6, -7), (4, -7), (2, -7), (8, -6), (8, -8))):
            self._add_structure(f"extra-tank-{index}", "potable_water_tank", dx, dy)
        self.assertEqual(
            90_000.0,
            self.engine._settlement_storage_capacity(
                "potable_water_tank", "potable_water_storage_capacity_liters"
            ),
        )

    def test_connected_tank_fills_beyond_crew_buffer(self):
        before = self.engine._colony_resources["water_reserve_l"]
        self.assertEqual(10_000.0, before)

        production = self._run()

        self.assertEqual(1, production["networked_water_collectors"])
        self.assertEqual(1, production["active_water_collectors"])
        self.assertEqual(25_000.0, self.engine._power_cycle_telemetry["water_collection_target_l"])
        self.assertGreater(production["water_production_l"], 0.0)
        self.assertGreater(self.engine._colony_resources["water_reserve_l"], before)
        self.assertGreater(self.engine._colony_capacity_overrides()["water"]["water_storage"], 0.0)

    def test_settlement_target_preserves_lander_exclusion(self):
        self._install_six_tanks()
        target = 10_000.0 + 81_540.0
        self.engine._colony_resources["water_reserve_l"] = target - 0.1

        production = self._run()

        self.assertEqual(target, self.engine._power_cycle_telemetry["water_collection_target_l"])
        self.assertEqual(1, production["active_water_collectors"])
        self.assertAlmostEqual(target, self.engine._colony_resources["water_reserve_l"], places=6)
        self.assertAlmostEqual(81_540.0, self.engine._colony_capacity_overrides()["water"]["water_storage"], places=6)

    def test_collectors_stop_at_readiness_target_even_with_extra_tank_space(self):
        self._install_six_tanks()
        self.engine._colony_resources["water_reserve_l"] = 91_540.0

        production = self._run()

        self.assertEqual(0, production["active_water_collectors"])
        self.assertEqual(0.0, production["water_production_l"])
        self.assertEqual(100_000.0, production["water_storage_capacity_l"])

    def test_disconnected_tank_does_not_enable_invisible_reserve(self):
        tank = next(s for s in self.engine.placed_structures if s["id"] == "water-tank")
        tank["x"] = self.engine.lz_x + 100

        production = self._run()

        self.assertEqual(0.0, production["settlement_water_storage_capacity_l"])
        self.assertEqual(180.0, self.engine._power_cycle_telemetry["water_collection_target_l"])
        self.assertEqual(0, production["active_water_collectors"])
        self.assertEqual(0.0, self.engine._colony_capacity_overrides()["water"]["water_storage"])

    def test_optional_fill_does_not_spend_emergency_battery_reserve(self):
        self.engine._colony_resources["energy_stored_kwh"] = 24.0

        production = self._run()

        self.assertEqual(1, production["networked_water_collectors"])
        self.assertEqual(0, production["active_water_collectors"])
        self.assertEqual(0.0, production["water_production_l"])
        self.assertEqual(0.0, production["auxiliary_production_kwh"])

    def test_low_crew_water_remains_priority_at_emergency_energy_reserve(self):
        self._add_structure("collector-2", "water_collector", 7, -6)
        self._add_structure("collector-3", "water_collector", 7, -8)
        self.engine._colony_resources["energy_stored_kwh"] = 24.0
        self.engine._colony_resources["water_reserve_l"] = 179.0

        production = self._run()

        self.assertEqual(3, production["networked_water_collectors"])
        # One skid covers the urgent shortfall; the other two may not use the
        # emergency battery just to accelerate optional settlement filling.
        self.assertEqual(1, production["active_water_collectors"])
        self.assertGreater(production["water_production_l"], 0.0)
        self.assertFalse(production["energy_deficit"])

    def test_incoming_solar_restores_emergency_floor_before_optional_fill(self):
        self.engine._colony_resources["water_reserve_l"] = 25_000.0
        self._run()
        base_load = self.engine._power_cycle_telemetry["load_kwh"]
        collector_cost = 5.0 * self.engine.SIM_HOURS_PER_TICK
        self.engine._colony_resources["water_reserve_l"] = 10_000.0
        self.engine._colony_resources["energy_stored_kwh"] = 24.0 - collector_cost
        solar_flux = self.engine.planet.data["surface"]["solar_flux_relative_to_earth"]
        self.engine._colony_resources["lander_integrated_solar_peak_kw"] = (
            (base_load + 1.5 * collector_cost)
            / (self.engine.SIM_HOURS_PER_TICK * solar_flux)
        )
        self.engine.get_day_night_phase = lambda: {
            "phase": "day", "solar_intensity": 1.0,
            "temp_modifier_c": 0.0,
        }

        production = self._run()

        self.assertGreater(production["solar_production_kwh"], base_load)
        self.assertEqual(0, production["active_water_collectors"])
        self.assertGreaterEqual(self.engine._colony_resources["energy_stored_kwh"], 24.0)

    def test_five_charged_solar_fields_restart_three_collectors(self):
        for index in range(5):
            self._add_structure(f"solar-{index}", "solar_panel", 15 + index, 5)
        self._add_structure("power-grid", "power_distribution_grid", 4, 4)
        self._add_structure("collector-2", "water_collector", 7, -6)
        self._add_structure("collector-3", "water_collector", 7, -8)
        self.engine._colony_resources["energy_stored_kwh"] = 60_120.0

        production = self._run()

        self.assertEqual(5, production["connected_solar_arrays"])
        self.assertEqual(3, production["active_water_collectors"])
        self.assertGreater(production["water_production_l"], 0.0)
        self.assertFalse(production["energy_deficit"])

    def test_optional_dispatch_counts_trace_heating_and_partial_fleet(self):
        self._add_structure("collector-2", "water_collector", 7, -6)
        self._add_structure("collector-3", "water_collector", 7, -8)
        self.temperature_c = -10.0
        # Measure mandatory loads with a full physical reservoir.
        self.engine._colony_resources["water_reserve_l"] = 25_000.0
        self._run()
        base_load = self.engine._power_cycle_telemetry["load_kwh"]
        cost_per_collector = 5.0 * self.engine.SIM_HOURS_PER_TICK + 0.25
        self.engine._colony_resources["water_reserve_l"] = 10_000.0
        self.engine._colony_resources["energy_stored_kwh"] = (
            24.0 + base_load + 2.1 * cost_per_collector
        )

        production = self._run()

        self.assertEqual(3, production["networked_water_collectors"])
        self.assertEqual(2, production["active_water_collectors"])
        self.assertGreaterEqual(self.engine._colony_resources["energy_stored_kwh"], 24.0)

    def test_extraction_conserves_water_and_stops_at_physical_capacity(self):
        before = 25_000.0 - 0.1
        self.engine._colony_resources["water_reserve_l"] = before

        self._run()

        cycle = self.engine._water_cycle_telemetry
        after = self.engine._colony_resources["water_reserve_l"]
        self.assertAlmostEqual(
            after - before,
            cycle["external_extraction_l"] + cycle["potable_recovered_l"]
            + cycle["greenhouse_recovery_l"] - cycle["process_consumption_l"],
            places=6,
        )
        self.assertAlmostEqual(25_000.0, after, places=6)

    def test_dry_atmosphere_still_requires_and_consumes_real_ice(self):
        self.engine.planet.atmosphere["composition"]["H2O"] = 0.0
        self.engine.central_depot_inventory["water_ice"] = 0.0
        dry = self._run()
        self.assertEqual(0.0, dry["water_production_l"])
        self.engine.central_depot_inventory["water_ice"] = 1.0
        self.engine._colony_resources["energy_stored_kwh"] = 120.0

        wet = self._run()

        self.assertGreater(wet["water_production_l"], 0.0)
        self.assertAlmostEqual(
            1.0 - self.engine.central_depot_inventory["water_ice"],
            self.engine._water_cycle_telemetry["external_extraction_l"]
            / MATERIAL_DENSITY_KG["water_ice"],
            places=6,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
