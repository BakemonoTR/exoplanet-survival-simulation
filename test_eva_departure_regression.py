"""Short physical route, hydration and lander-compartment regressions."""
import unittest
from pathlib import Path

from src.agents.agent import create_team_from_presets
from src.memory import vector_store
from src.orchestration.engine import SimulationEngine

vector_store._use_tfidf_fallback = True
ROOT = Path(__file__).resolve().parent


class EvaDepartureRegressionTest(unittest.TestCase):
    def setUp(self):
        self.engine = SimulationEngine(str(ROOT / "config/planets/kepler-442b.json"), seed=42, db=False)
        self.engine.llm_client.call = lambda *a, **k: None
        self.crew = create_team_from_presets(str(ROOT / "config/agent_presets.json"))
        for crew in self.crew:
            self.engine.add_agent(crew)
        self.engine._init_agents()
        self.agent = self.crew[0]
        for crew in self.crew:
            crew.x, crew.y = self.engine._lander_airlock_position()
            crew._in_habitat = True
            crew.needs.energy = crew.needs.hunger = crew.needs.thirst = 100.0
            crew.needs.o2_supply = 100.0
            crew.needs.temperature_stress = 60.0
            crew._current_canister_remaining = 100.0
            crew.plss_co2_scrubber_pct = crew.plss_suit_battery_pct = 100.0
            crew.suit_condition = crew.suit_integrity = 1.0
            crew.inventory.items.pop("water_packs", None)
        self.engine.central_depot_inventory["water_packs"] = 0
        self.engine._colony_resources["water_reserve_l"] = 1000.0

    def test_empty_bags_are_filled_from_real_stock_before_exit(self):
        target = (self.engine.lz_x + 8, self.engine.lz_y + 5)
        self.assertTrue(self.engine._prepare_agent_for_eva(self.agent, target_position=target))
        self.assertEqual(2, self.agent.inventory.items["water_packs"])
        self.assertEqual(998.0, self.engine._colony_resources["water_reserve_l"])
        self.assertEqual(100.0, self.agent.needs.thirst)

    def test_low_hydration_stays_inside_without_free_drink(self):
        self.agent.needs.thirst = 30.0
        self.assertFalse(self.engine._prepare_agent_for_eva(self.agent))
        self.assertTrue(self.agent._in_habitat)
        self.assertEqual(30.0, self.agent.needs.thirst)
        self.assertEqual("preflight_hydration_required", self.agent.action.target["eva_denied"])

    def test_no_portable_water_means_no_departure(self):
        self.engine._colony_resources["water_reserve_l"] = 0.5
        self.assertFalse(self.engine._prepare_agent_for_eva(self.agent))
        self.assertEqual("portable_water_unavailable", self.agent.action.target["eva_denied"])
        self.assertEqual(0.5, self.engine._colony_resources["water_reserve_l"])

    def test_unsafe_round_trip_is_refused_even_when_base_has_water(self):
        target = (self.engine.lz_x + 90, self.engine.lz_y + 90)
        self.assertFalse(self.engine._prepare_agent_for_eva(self.agent, target_position=target))
        self.assertEqual("insufficient_round_trip_water", self.agent.action.target["eva_denied"])
        self.assertTrue(self.agent._in_habitat)

    def test_return_threshold_increases_with_routed_distance(self):
        self.agent._in_habitat = False
        self.agent.x, self.agent.y = self.engine.lz_x + 4, self.engine.lz_y + 4
        nearby = self.engine._eva_return_water_threshold(self.agent)
        self.agent.x += 10
        self.agent.y += 10
        self.assertGreater(self.engine._eva_return_water_threshold(self.agent), nearby)
        self.assertGreater(nearby, 35.0)

    def _tick(self, decision):
        self.engine.decision_engine.process_tick = lambda **k: decision
        self.engine.current_tick += 1
        return self.engine._process_agent_tick(self.agent, [], 0)

    def test_field_move_drinks_before_continuing_route(self):
        self.agent.x, self.agent.y = self.engine.lz_x + 8, self.engine.lz_y + 8
        self.agent._in_habitat = False
        self.agent.needs.thirst = 60.0
        self.agent.inventory.items["water_packs"] = 1
        self._tick({"action": "move", "target": {"x": self.agent.x + 1, "y": self.agent.y}})
        self.assertEqual("drink", self.agent.last_decision["action"])
        self.assertGreater(self.agent.needs.thirst, 85.0)
        self.assertFalse(self.agent.inventory.has_item("water_packs"))

    def test_empty_bag_aborts_before_fixed_fifteen_percent(self):
        self.agent.x, self.agent.y = self.engine.lz_x + 24, self.engine.lz_y + 24
        self.agent._in_habitat = False
        self.agent.needs.thirst = 60.0
        self._tick({"action": "move", "target": {"x": self.agent.x + 1, "y": self.agent.y}})
        self.assertTrue(self.agent.last_decision["target"].get("dehydration_return"))
        self.assertGreater(self.agent.needs.thirst, 15.0)

    def test_hydration_return_does_not_suppress_carried_oxygen_swap(self):
        self.agent.x, self.agent.y = self.engine.lz_x + 12, self.engine.lz_y + 12
        self.agent._in_habitat = False
        self.agent.needs.thirst = 50.0
        self.agent.needs.o2_supply = self.agent._current_canister_remaining = 30.0
        self.agent.inventory.items["oxygen_canisters"] = 1
        self._tick({"action": "refill_o2", "target": {"carried_spare": True}})
        self.assertEqual("refill_o2", self.agent.last_decision["action"])
        self.assertGreater(self.agent.needs.o2_supply, 95.0)
        self.assertFalse(self.agent.inventory.has_item("oxygen_canisters"))

    def test_oxygen_reserve_prevents_shelter_detour_then_hydration_recall(self):
        self.agent.x, self.agent.y = self.engine.lz_x + 26, self.engine.lz_y + 8
        self.agent._in_habitat = False
        self.agent.needs.thirst = 70.0
        self.agent.needs.o2_supply = self.agent._current_canister_remaining = 55.0
        self.agent.inventory.items.pop("oxygen_canisters", None)
        self._tick({"action": "move", "target": {
            "x": self.engine.lz_x + 5, "y": self.engine.lz_y - 7,
        }})
        target = self.agent.last_decision["target"]
        self.assertTrue(target.get("o2_return"))
        self.assertEqual(self.engine._lander_airlock_position(), (target["x"], target["y"]))

    def test_preflight_hydration_request_executes_real_drink(self):
        self.agent.needs.thirst = 60.0
        self.assertFalse(self.engine._prepare_agent_for_eva(self.agent))
        self._tick({"action": "plan_construction", "target": {"recipe": "solar_panel"}})
        self.assertEqual("drink", self.agent.last_decision["action"])
        self.assertGreater(self.agent.needs.thirst, 80.0)
        self.assertEqual(999.0, self.engine._colony_resources["water_reserve_l"])

    def test_indoor_compartments_are_distinct_and_inside_hull(self):
        bunks = {self.engine._indoor_activity_position(crew, "sleep") for crew in self.crew}
        self.assertEqual(6, len(bunks))
        positions = bunks | {self.engine._indoor_activity_position(self.agent, action)
                             for action in ("eat", "plan_construction")}
        self.assertEqual(8, len(positions))
        self.assertNotIn(self.engine._lander_airlock_position(), positions)
        self.assertTrue(all(self.engine._is_lander_footprint_cell(*p) for p in positions))

    def test_sleep_walks_to_berth_without_eva_or_teleport(self):
        decision = {"action": "sleep", "target": {"habitat": True, "ticks": 24}}
        origin = (self.agent.x, self.agent.y)
        for _ in range(8):
            before = (self.agent.x, self.agent.y)
            self._tick(decision)
            distance = abs(self.agent.x - before[0]) + abs(self.agent.y - before[1])
            self.assertLessEqual(distance, self.engine.EVA_WALK_SPEED_CELLS)
            self.assertTrue(self.agent._in_habitat)
            if self.agent.action.action_type == "sleep":
                break
        self.assertEqual("sleep", self.agent.action.action_type)
        self.assertNotEqual(origin, (self.agent.x, self.agent.y))
        self.assertEqual("bunk", self.agent.to_dict()["indoor_location"]["zone"])

    def test_medical_approach_cancels_pending_room_visit(self):
        self.engine._indoor_activity_decision(self.agent, "sleep", {"ticks": 24})
        target = {"x": self.engine.lz_x + 8, "y": self.engine.lz_y, "medical_response": True}
        action, actual = self.engine._indoor_activity_decision(self.agent, "move", target)
        self.assertEqual(("move", target), (action, actual))
        self.assertIsNone(self.agent._pending_indoor_activity)

    def test_rejected_water_preflight_creates_no_expedition(self):
        self.agent.needs.thirst = 70.0
        target = (self.engine.lz_x + 20, self.engine.lz_y)
        result = self.engine._start_expedition(self.agent, "water_ice", target)
        self.assertIsNone(result)
        self.assertFalse(getattr(self.agent, "_active_expedition", None))
        self.assertTrue(all(rover.state == "idle" for rover in self.engine.surface_fleet.crew_rovers))

    def test_rover_credits_drive_out_but_keeps_walk_back_water(self):
        agent = self.crew[4]  # Volkov, the failed run's dehydration victim
        target = (self.engine.lz_x + 70, self.engine.lz_y + 70)
        self.assertFalse(self.engine._prepare_eva_water(agent, target))
        self.assertTrue(self.engine._prepare_eva_water(agent, target, rover_outbound=True))
        self.assertEqual(2, agent.inventory.items["water_packs"])
        # Driving cannot authorize arbitrarily long unsupported walk-backs.
        far = (self.engine.lz_x + 120, self.engine.lz_y + 120)
        self.assertFalse(self.engine._prepare_eva_water(agent, far, rover_outbound=True))

    def test_partly_hydrated_long_route_requests_a_real_top_up(self):
        agent = self.crew[4]
        agent.needs.thirst = 80.0
        target = (self.engine.lz_x + 80, self.engine.lz_y)
        self.assertFalse(self.engine._prepare_eva_water(agent, target))
        self.assertTrue(agent._eva_hydration_pending)
        self.assertGreater(agent._eva_hydration_target, 80.0)
        agent.needs.thirst = 100.0
        self.assertTrue(self.engine._prepare_eva_water(agent, target))

    def test_mining_staging_avoids_unnecessary_diagonal_walk(self):
        target = self.engine._mining_staging_point(self.agent)
        self.assertEqual(self.engine.MIN_EXTRACTION_RADIUS_CELLS, self.engine._distance_from_lz(*target))
        route = self.engine._surface_vehicle_route(*self.engine._lander_airlock_position(), *target)
        self.assertLess(len(route) - 1, 2 * self.engine.MIN_EXTRACTION_RADIUS_CELLS)
        self.assertTrue(self.engine._prepare_eva_water(self.agent, target))

    def test_loaded_walker_returns_toward_real_airlock(self):
        self.agent.x, self.agent.y = self.engine.lz_x + 8, self.engine.lz_y + 8
        self.agent._in_habitat = False
        airlock = self.engine._lander_airlock_position()
        before = abs(self.agent.x - airlock[0]) + abs(self.agent.y - airlock[1])
        self.engine._send_loaded_agent_to_base(self.agent, 20.0, 20.0)
        after = abs(self.agent.x - airlock[0]) + abs(self.agent.y - airlock[1])
        self.assertLess(after, before)
        self.assertLessEqual(before - after, self.engine.EVA_WALK_SPEED_CELLS)


if __name__ == "__main__":
    unittest.main()
