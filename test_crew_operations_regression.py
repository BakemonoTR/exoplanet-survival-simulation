"""Regression coverage for the observed airlock/preflight/no-progress loops."""
import unittest
from unittest.mock import patch

import test_construction_rover_regression as fixture
from src.systems.airlock import AirlockController


class CrewOperationsRegressionTest(unittest.TestCase):
    def setUp(self):
        fixture.ConstructionRoverRegressionTest.setUp(self)

    def test_staggered_crew_recovers_and_reaches_remote_work(self):
        self.crew[0].needs.energy = 78.0
        self.crew[1].needs.energy = 56.0
        self.crew[1].needs.hunger = 53.0
        self.crew[1].suit_condition = 0.60
        self.site["required_work_hours"] = 8.0
        self.engine._start_construction_rover_trip(self.crew[0], self.site)
        reserved = list(self.engine._construction_sortie["crew_ids"])
        for _ in range(240):
            result = self.engine._run_tick()
            self.assertFalse(result["agent_processing_errors"])
            self.engine.current_tick += 1
            for crew in self.crew:
                self.assertFalse(not crew._in_habitat and crew.action.action_type in {"sleep", "service_suit"})
            if self.site.get("work_hours_completed", 0) > 0:
                break
        self.assertGreater(self.site.get("work_hours_completed", 0), 0,
                           [(c.name, c.needs.to_dict(), c.action.to_dict()) for c in self.crew])
        self.assertEqual(set(reserved), {c.id for c in self.crew if c._active_expedition})
        self.assertGreater(self.engine.airlock.completed_cycles, 0)

    def test_suit_service_has_no_early_or_outdoor_benefit(self):
        agent = self.crew[0]
        agent.suit_condition = 0.50
        agent.x, agent.y = self.engine.lz_x + 8, self.engine.lz_y + 8
        agent._in_habitat = False
        self.assertFalse(self.engine._start_suit_service(agent))
        self.assertEqual(0.50, agent.suit_condition)
        self.assertEqual("move", agent.action.action_type)
        agent._in_habitat = True
        agent.x, agent.y = self.engine.lz_x, self.engine.lz_y
        agent.x, agent.y = self.engine._indoor_activity_position(agent, "service_suit")
        agent.action.clear()
        self.assertTrue(self.engine._start_suit_service(agent))
        duration = agent.action.ticks_remaining
        self.assertEqual(6, duration)
        self.assertEqual(0.50, agent.suit_condition)
        with patch.object(self.planner, "process_tick", return_value={"action": "rest", "target": {}, "deterministic": True}):
            for _ in range(duration):
                self.engine.current_tick += 1
                self.engine._process_agent_tick(agent, [], 0)
        self.assertEqual(0.95, agent.suit_condition)

    def test_external_sleep_intent_returns_without_sleep(self):
        agent = self.crew[0]
        agent._in_habitat = False
        agent.x, agent.y = self.engine.lz_x + 8, self.engine.lz_y + 8
        with patch.object(self.planner, "process_tick", return_value={"action": "sleep", "target": {"field": True}}):
            self.engine._process_agent_tick(agent, [], 0)
        self.assertNotEqual("sleep", agent.action.action_type)
        self.assertEqual("shelter", agent.action.target.get("destination"))

    def test_enclosed_cnc_is_not_human_pressure_space_or_crew_quarters(self):
        agent = self.crew[0]
        cnc = next(
            structure for structure in self.engine.placed_structures
            if structure.get("type") == "cnc_fabricator"
        )
        agent.x, agent.y = int(cnc["x"]), int(cnc["y"])
        agent._in_habitat = False
        self.assertTrue(cnc.get("environmentally_sealed_machine"))
        self.assertFalse(cnc.get("pressurized"))
        self.assertIsNone(self.engine._pressurized_structure_at(agent.x, agent.y))
        self.assertFalse(self.engine._is_crew_quarters_location(agent))

        for requested_action in ("sleep", "wash", "service_suit"):
            with self.subTest(action=requested_action):
                agent.action.clear()
                agent.x, agent.y = int(cnc["x"]), int(cnc["y"])
                agent._in_habitat = False
                with patch.object(
                    self.planner,
                    "process_tick",
                    return_value={
                        "action": requested_action,
                        "target": {"habitat": True},
                        "deterministic": True,
                    },
                ):
                    self.engine._process_agent_tick(agent, [], 0)
                self.assertEqual("move", agent.action.action_type)
                self.assertEqual("shelter", agent.action.target.get("destination"))

        refuge = {
            "id": "test_greenhouse_refuge",
            "type": "greenhouse",
            "x": self.engine.lz_x + 12,
            "y": self.engine.lz_y + 12,
            "health": 1.0,
            "pressurized": True,
        }
        self.engine.placed_structures.append(refuge)
        agent.x, agent.y = refuge["x"], refuge["y"]
        agent._in_habitat = True
        self.assertTrue(self.engine._is_crew_quarters_location(agent))
        agent.needs.energy = 40.0
        agent.action.action_type = "sleep"
        agent.action.target = {"habitat": True, "storm_refuge": True}
        agent.action.ticks_remaining = 12
        energy_before = agent.needs.energy
        self.engine._process_agent_tick(agent, [], 0)
        self.assertGreater(agent.needs.energy, energy_before)
        self.assertEqual("sleep", agent.action.action_type)

    def test_rover_reserved_during_preparation_cannot_be_stolen(self):
        self.crew[1].action.action_type = "sleep"
        self.crew[1].action.ticks_remaining = 12
        self.engine._start_construction_rover_trip(self.crew[0], self.site)
        result = self.engine.surface_fleet.begin_crew_rover_trip(
            expedition_id="unrelated", crew_ids=["other-1", "other-2"],
            target_x=self.site["x"], target_y=self.site["y"],
        )
        self.assertFalse(result["reserved"])
        self.engine._release_construction_preparation()
        self.assertTrue(all(r.reservation_id is None for r in self.engine.surface_fleet.crew_rovers))

    def test_walk_home_uses_apron_and_waits_for_pressure_cycle(self):
        agent = self.crew[0]
        agent.x, agent.y = self.engine.lz_x, self.engine.lz_y - 4
        agent._in_habitat = False
        door = self.engine._lander_airlock_position()
        exterior = self.engine._lander_airlock_exterior_position()
        waiting_ticks = 0
        for _ in range(40):
            before = (agent.x, agent.y)
            dx, dy = self.engine._cardinal_step_toward(agent, *door)
            if (dx, dy) == (0, 0) and before == exterior:
                waiting_ticks += 1
            agent.x += dx
            agent.y += dy
            self.engine.current_tick += 1
            if self.engine._is_lander_footprint_cell(agent.x, agent.y):
                self.assertEqual(door, (agent.x, agent.y))
                self.assertEqual(exterior, before)
                break
        self.assertEqual(door, (agent.x, agent.y))
        self.assertGreaterEqual(waiting_ticks, 1)

    def test_overhaul_consumes_one_seal_and_waits_for_completion(self):
        agent = self.crew[0]
        agent.x, agent.y = self.engine._indoor_activity_position(agent, "service_suit")
        agent.suit_integrity = 0.40
        agent.inventory.materials["vacuum_gasket_seal"] = 1
        self.assertTrue(self.engine._start_suit_service(agent, pressure_overhaul=True))
        self.assertEqual(0, agent.inventory.materials["vacuum_gasket_seal"])
        self.assertEqual(0.40, agent.suit_integrity)
        # Resuming interrupted maintenance reuses its already committed part.
        self.assertTrue(self.engine._start_suit_service(agent, pressure_overhaul=True))
        self.assertEqual(0, agent.inventory.materials["vacuum_gasket_seal"])

    def test_full_active_o2_tank_still_provisions_missing_spare(self):
        agent, buddy = self.crew
        buddy.action.action_type = "sleep"
        self.engine._start_construction_rover_trip(agent, self.site)
        agent.inventory.items.pop("oxygen_canisters", None)
        self.engine.central_depot_inventory["oxygen_canisters"] = 2
        self.engine.central_depot_inventory["empty_oxygen_canisters"] = 1
        agent._current_canister_remaining = 100.0
        decision = self.engine._construction_preparation_decision(agent)
        self.assertEqual("refill_o2", decision["action"])
        before = self.engine._colony_resources["o2_reserve_kg"]
        self.engine._process_agent_tick(agent, [], 0)
        self.assertEqual(1, agent.inventory.items.get("oxygen_canisters", 0))
        self.assertAlmostEqual(before - agent.PLSS_CANISTER_O2_KG,
                               self.engine._colony_resources["o2_reserve_kg"])

    def test_local_worn_suit_receives_work_order_so_service_can_run(self):
        self.site["type"] = "life_support_distribution_grid"
        self.site["x"], self.site["y"] = self.engine.lz_x + 3, self.engine.lz_y
        for crew in self.crew:
            crew.suit_condition = 0.40
        decisions = [self.planner._shared_colony_work_decision(
            c, 1, self.engine.structures_built) for c in self.crew]
        self.assertTrue(any(d["action"] == "build" for d in decisions))
        # Assignment is not permission to leave with an unserviced suit.
        self.assertFalse(self.engine._prepare_agent_for_eva(self.crew[0]))
        self.assertTrue(self.crew[0]._in_habitat)


class AirlockQueueTest(unittest.TestCase):
    def test_fifo_pair_cycle_and_single_charge(self):
        gate = AirlockController(cycle_ticks=2)
        charges = []
        def charge():
            charges.append(1)
            return True
        self.assertFalse(gate.request(["a", "b"], "out", 0, charge))
        self.assertFalse(gate.request(["c"], "in", 0, charge))
        self.assertFalse(gate.request(["a", "b"], "out", 1, charge))
        self.assertTrue(gate.request(["a", "b"], "out", 2, charge))
        self.assertTrue(gate.request(["a", "b"], "out", 2, charge))
        self.assertEqual(1, len(charges))
        self.assertFalse(gate.request(["c"], "in", 2, charge))
        self.assertTrue(gate.request(["c"], "in", 4, charge))
        self.assertEqual(2, len(charges))
        self.assertEqual(2, gate.completed_cycles)

    def test_abandoned_request_expires_and_capacity_is_enforced(self):
        gate = AirlockController(cycle_ticks=1)
        self.assertFalse(gate.request(["a", "b", "c"], "out", 0))
        self.assertIsNone(gate.active)
        gate.request(["a"], "out", 0)
        self.assertFalse(gate.request(["b"], "in", 10))
        self.assertTrue(gate.request(["b"], "in", 11))


if __name__ == "__main__":
    unittest.main()
