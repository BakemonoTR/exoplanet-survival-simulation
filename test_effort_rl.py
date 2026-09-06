import os
import unittest

from src.agents.agent import create_team_from_presets
from src.orchestration.engine import SimulationEngine


ROOT = os.path.dirname(os.path.abspath(__file__))


class MovementEffortRLTest(unittest.TestCase):
    def setUp(self):
        self.engine = SimulationEngine(
            os.path.join(ROOT, "config", "planets", "kepler-442b.json"),
            seed=42,
            db=False,
        )
        self.agent = create_team_from_presets(
            os.path.join(ROOT, "config", "agent_presets.json")
        )[0]
        self.engine.add_agent(self.agent)
        self.engine._init_agents()
        self.agent._in_habitat = False
        self.agent.needs.energy = 95.0
        self.agent.needs.thirst = 95.0
        self.agent.needs.hunger = 95.0
        self.agent.needs.o2_supply = 100.0
        self.agent.needs.temperature_stress = 50.0
        self.agent.rl_epsilon_explore = 0.0

    def test_auxiliary_choice_does_not_overwrite_primary_transition(self):
        self.agent.last_state_key = "primary-state"
        self.agent.last_action_key = "build:greenhouse"
        self.agent.rl_episode_trace = [
            ("primary-state", "build:greenhouse")
        ]
        result = self.engine.decision_engine.select_auxiliary_action_via_rl(
            self.agent,
            "effort-test",
            [
                {"action": "pace", "target": {"destination": "steady"}},
                {"action": "pace", "target": {"destination": "surge"}},
            ],
        )

        self.assertIn(result["target"]["destination"], {"steady", "surge"})
        self.assertEqual("primary-state", self.agent.last_state_key)
        self.assertEqual("build:greenhouse", self.agent.last_action_key)
        self.assertEqual(
            [("primary-state", "build:greenhouse")],
            self.agent.rl_episode_trace,
        )

    def test_surge_has_a_real_metabolic_cost(self):
        self.agent.action.action_type = "move"
        self.agent.action.target = {"effort_mode": "steady"}
        steady = self.agent.get_activity_multiplier()
        self.agent.action.target = {"effort_mode": "surge"}
        surge = self.agent.get_activity_multiplier()

        self.assertEqual(1.5, steady)
        self.assertEqual(2.25, surge)

    def test_medical_envelope_masks_surge_even_with_high_q_value(self):
        self.agent.needs.energy = 64.0
        state = self.engine._movement_effort_state(self.agent, 8, 1.0)
        self.agent.q_table[state] = {
            "pace:steady": -10.0,
            "pace:surge": 100.0,
        }
        mode = self.engine._movement_effort_mode(
            self.agent,
            {"destination": "construction_site"},
            self.agent.x + 8,
            self.agent.y,
            2,
            1.0,
        )

        self.assertEqual("steady", mode)
        self.assertFalse(hasattr(self.agent, "_movement_effort_contract"))

    def test_completed_fast_route_updates_only_its_delayed_q_value(self):
        self.agent.last_state_key = "primary-state"
        self.agent.last_action_key = "gather:regolith"
        state = self.engine._movement_effort_state(self.agent, 8, 1.0)
        self.agent.q_table[state] = {
            "pace:steady": 0.0,
            "pace:surge": 1.1,
        }
        self.engine.current_tick = 100
        mode = self.engine._movement_effort_mode(
            self.agent,
            {"destination": "construction_site"},
            self.agent.x + 8,
            self.agent.y,
            2,
            1.0,
        )
        self.assertEqual("surge", mode)
        before = self.agent.q_table[state]["pace:surge"]

        self.engine.current_tick = 103
        reward = self.engine._settle_movement_effort(
            self.agent, completed=True, reason="test_arrival"
        )

        self.assertGreater(reward, 1.0)
        self.assertGreater(self.agent.q_table[state]["pace:surge"], before)
        self.assertEqual("primary-state", self.agent.last_state_key)
        self.assertEqual("gather:regolith", self.agent.last_action_key)
        self.assertGreater(
            self.engine._movement_effort_telemetry["estimated_ticks_saved"],
            0.0,
        )

    def test_existing_surge_stops_when_energy_crosses_safety_limit(self):
        target = {"destination": "construction_site"}
        x, y = self.agent.x + 8, self.agent.y
        state = self.engine._movement_effort_state(self.agent, 8, 1.0)
        self.agent.q_table[state] = {"pace:steady": 0.0, "pace:surge": 10.0}
        self.assertEqual("surge", self.engine._movement_effort_mode(
            self.agent, target, x, y, 2, 1.0
        ))
        self.agent.needs.energy = 70.0
        self.assertEqual("steady", self.engine._movement_effort_mode(
            self.agent, target, x, y, 2, 1.0
        ))
        self.assertEqual("safety_envelope_changed",
                         self.engine._movement_effort_telemetry["last_outcome"]["reason"])

    def test_route_projection_rejects_surge_that_would_break_reserve_midway(self):
        self.agent.needs.thirst = 68.0
        state = self.engine._movement_effort_state(self.agent, 12, 1.0)
        self.agent.q_table[state] = {
            "pace:steady": -10.0,
            "pace:surge": 100.0,
        }

        mode = self.engine._movement_effort_mode(
            self.agent,
            {"destination": "construction_site"},
            self.agent.x + 12,
            self.agent.y,
            2,
            1.0,
        )

        self.assertEqual("steady", mode)
        self.assertFalse(hasattr(self.agent, "_movement_effort_contract"))

    def test_route_reassignment_is_censored_without_corrupting_pace_q_value(self):
        state = self.engine._movement_effort_state(self.agent, 8, 1.0)
        self.agent.q_table[state] = {
            "pace:steady": 0.0,
            "pace:surge": 2.0,
        }
        self.engine.current_tick = 10
        mode = self.engine._movement_effort_mode(
            self.agent,
            {"destination": "construction_site"},
            self.agent.x + 8,
            self.agent.y,
            self.engine.EVA_WALK_SPEED_CELLS,
            1.0,
        )
        self.assertEqual("surge", mode)
        before = self.agent.q_table[state]["pace:surge"]

        self.engine.current_tick = 11
        reward = self.engine._settle_movement_effort(
            self.agent, completed=False, reason="route_changed"
        )

        self.assertEqual(0.0, reward)
        self.assertEqual(before, self.agent.q_table[state]["pace:surge"])
        self.assertEqual(
            1, self.engine._movement_effort_telemetry["censored_routes"]
        )
        self.assertEqual(
            0, self.engine._movement_effort_telemetry["aborted_routes"]
        )

    def test_profile_converts_nasa_calibrated_gait_to_grid_cells(self):
        self.assertEqual(4, self.engine.EVA_WALK_SPEED_CELLS)
        self.assertEqual(4, self.engine.EXPEDITION_MOVE_SPEED_CELLS)
        self.assertEqual(1.5, self.engine.MOVEMENT_SURGE_SPEED_MULTIPLIER)

    def test_return_approach_preserves_stride_and_stops_at_exterior(self):
        exterior = self.engine._lander_airlock_exterior_position()
        airlock = self.engine._lander_airlock_position()
        self.agent.x, self.agent.y = exterior[0] + 8, exterior[1]
        self.assertEqual((-2, 0), self.engine._cardinal_step_toward(
            self.agent, *airlock, step=2
        ))
        self.agent.x = exterior[0] + 1
        self.assertEqual((-1, 0), self.engine._cardinal_step_toward(
            self.agent, *airlock, step=2
        ))
        self.assertEqual(0, self.engine.airlock.completed_cycles)


class CapsuleStagingTest(unittest.TestCase):
    def test_shared_lander_stock_is_conserved_without_overloading_crew(self):
        engine = SimulationEngine(
            os.path.join(ROOT, "config", "planets", "kepler-442b.json"),
            seed=42,
            db=False,
        )
        crew = create_team_from_presets(
            os.path.join(ROOT, "config", "agent_presets.json")
        )
        for agent in crew:
            engine.add_agent(agent)
        engine._init_agents()

        for agent in crew:
            self.assertLessEqual(
                agent.operational_load_fraction(engine.planet.gravity_g),
                1.0,
                agent.id,
            )
        self.assertEqual(
            3,
            sum(
                agent.inventory.items.get("multitool_kit", 0)
                for agent in crew
            ),
        )
        self.assertEqual(78, engine.central_depot_inventory["ration_packs"])
        self.assertEqual(124, engine.central_depot_inventory["water_packs"])
        self.assertEqual(54, engine.central_depot_inventory["oxygen_canisters"])
        self.assertEqual(15, engine.central_depot_inventory["electronics_salvage"])
        self.assertEqual(10, engine.central_depot_inventory["medical_supplies"])
        self.assertTrue(all(
            "medical_supplies" not in agent.inventory.items for agent in crew
        ))


if __name__ == "__main__":
    unittest.main()
