"""Thermal dispatch, cooling prerequisites and weather-safe route caching."""
import unittest

import test_shared_construction_realism as shared_fixture
import test_water_reserve_dispatch as water_fixture


class ThermalPlanningTest(unittest.TestCase):
    def setUp(self):
        fixture = shared_fixture.SharedConstructionRealismTest()
        fixture.setUp()
        self.engine = fixture.engine
        self.planner = self.engine.decision_engine

    def test_overheated_base_can_order_real_cooling_without_score_credit(self):
        self.planner._thermal_state = {
            "temperature_c": 40.0, "total_waste_heat_kw": 17.0,
            "lander_cooling_capacity_kw": 4.0, "net_heat_kw": 13.0,
        }
        order = self.planner._select_shared_capacity_order(
            self.planner._pooled_materials(), self.engine.structures_built, tick=1
        )
        self.assertEqual("radiator_panel", order["recipe"])
        self.assertEqual(2, order["required_count"])
        self.assertIsNone(self.planner.colony.get_structure_capacity_status("radiator_panel"))

    def test_sufficient_cooling_does_not_spawn_extra_radiators(self):
        self.planner._thermal_state = {
            "temperature_c": 40.0, "total_waste_heat_kw": 11.0,
            "lander_cooling_capacity_kw": 4.0, "net_heat_kw": -1.0,
        }
        structures = {**self.engine.structures_built, "radiator_panel": 1}
        order = self.planner._select_shared_capacity_order(
            self.planner._pooled_materials(), structures, tick=1
        )
        self.assertNotEqual("radiator_panel", order["recipe"])

    def test_route_weather_cache_refreshes_next_tick(self):
        calls = []
        def weather(x, y, tick):
            calls.append(tick)
            return {"traversal_cost": 1.0 + tick}
        self.engine.world.get_cell_info = weather
        self.engine.current_tick = 1
        self.assertEqual(2.0, self.engine._movement_route_cell(1, 2)["traversal_cost"])
        self.engine._movement_route_cell(1, 2)
        self.engine.current_tick = 2
        self.assertEqual(3.0, self.engine._movement_route_cell(1, 2)["traversal_cost"])
        self.assertEqual([1, 2], calls)


class ThermalDispatchTest(unittest.TestCase):
    def test_disconnected_process_skids_emit_no_operating_heat(self):
        fixture = water_fixture.WaterReserveDispatchTest()
        fixture.setUp()
        engine = fixture.engine
        for number in range(4):
            fixture._add_structure(f"remote-isru-{number}", "isru_o2_unit", 200 + number * 4, 200)
        fixture._run()
        state = engine._thermal_state
        self.assertEqual(0, state["active_structure_loads"]["isru_o2_unit"])
        self.assertEqual(1.5, state["steady_waste_heat_kw"])


class WaterRecoveryPlanningTest(unittest.TestCase):
    def test_powered_connected_water_shortage_orders_extraction(self):
        fixture = shared_fixture.SharedConstructionRealismTest()
        fixture.setUp()
        planner = fixture.engine.decision_engine
        planner._colony_resources = {
            "water_reserve_l": 180.0, "energy_stored_kwh": 100_000.0,
            "food_reserve_kcal": 5_000_000.0, "o2_reserve_kg": 2_000.0,
        }
        planner._power_cycle_telemetry = {
            "deficit": False, "installed_water_collectors": 5,
            "networked_water_collectors": 5,
        }
        planner._water_storage_capacity_l = 40_000.0
        structures = {**fixture.engine.structures_built, "solar_panel": 14,
                      "water_collector": 5, "water_purifier": 4,
                      "potable_water_tank": 2, "life_support_distribution_grid": 2}
        fixture._prefer_shared_strategy("solar_panel", structures)
        chosen = planner._select_shared_capacity_order(planner._pooled_materials(), structures, tick=1)
        self.assertEqual("water_collector", chosen["recipe"])


if __name__ == "__main__":
    unittest.main()
