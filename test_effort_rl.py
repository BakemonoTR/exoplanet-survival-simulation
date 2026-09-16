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

    def test_finished_machine_route_uses_shared_pace_policy(self):
        target = {
            'destination': 'manufacturing_machine',
            'resume_machine_cycle': True,
            'machine_id': 'shop-1',
        }
        self.engine._manufacturing_cycles['shop-1'] = {
            'completion_pending': True,
            'output_ready': True,
            'process_completed_tick': 0,
        }
        self.engine.current_tick = 20
        state = self.engine._movement_effort_state(
            self.agent, 8, 1.0, target
        )
        self.assertTrue(state.startswith('shop-effort-v1|'))
        self.engine.strategic_policy.operational_q_table[state] = {
            'shop_pace:steady': -5.0,
            'shop_pace:surge': 5.0,
        }
        self.engine.strategic_policy.evaluation_mode = True

        mode = self.engine._movement_effort_mode(
            self.agent, target, self.agent.x + 8, self.agent.y, 2, 1.0
        )

        self.assertEqual('surge', mode)
        self.assertEqual(
            'shared_operational',
            self.agent._movement_effort_contract['credit_policy'],
        )
        self.assertNotIn(state, self.agent.q_table)

    def test_finished_machine_pace_reward_updates_shared_table_only(self):
        state = ('shop-effort-v1|activity:machine_release|distance:medium|'
                 'energy:fresh|reserve:full|terrain:clear|wait:blocked|'
                 'backlog:one')
        self.engine.strategic_policy.operational_q_table[state] = {
            'shop_pace:steady': 1.0,
            'shop_pace:surge': 1.0,
        }
        tactical_before = dict(self.agent.q_table)
        self.engine.current_tick = 100
        self.agent._movement_effort_contract = {
            'signature': (8, 0, 'manufacturing_machine', '', '', ''),
            'mode': 'surge',
            'credit_policy': 'shared_operational',
            'state_key': state,
            'action_key': 'shop_pace:surge',
            'start_tick': 97,
            'expected_steady_ticks': 6,
            'expected_surge_ticks': 4,
            'start_energy': self.agent.needs.energy,
            'start_thirst': self.agent.needs.thirst,
            'start_o2': self.agent.needs.o2_supply,
        }

        reward = self.engine._settle_movement_effort(
            self.agent, completed=True, reason='arrived'
        )

        self.assertGreater(reward, 1.0)
        self.assertGreater(
            self.engine.strategic_policy.operational_q_table[state][
                'shop_pace:surge'
            ],
            1.0,
        )
        self.assertEqual(tactical_before, self.agent.q_table)
        self.assertEqual(
            1, self.engine._movement_effort_telemetry['shared_shop_outcomes']
        )

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

    def test_construction_surge_is_real_work_with_real_metabolic_cost(self):
        site = {
            "id": "test-site",
            "required_work_hours": 40.0,
            "work_hours_completed": 5.0,
            "utility_route_required": False,
        }
        self.agent.action.action_type = "build"
        self.agent.action.target = {"struct_id": site["id"]}
        state = self.engine._construction_effort_state(
            self.agent, site, active_robot_count=2
        )
        self.engine.strategic_policy.operational_q_table[state] = {
            "construction_pace:steady": -5.0,
            "construction_pace:surge": 5.0,
        }
        self.engine.strategic_policy.evaluation_mode = True
        baseline = self.engine._construction_productivity(
            self.agent, include_effort=False
        )

        mode = self.engine._construction_effort_mode(
            self.agent, site, active_robot_count=2
        )

        self.assertEqual("surge", mode)
        self.assertAlmostEqual(
            baseline * self.engine.CONSTRUCTION_SURGE_PRODUCTIVITY_MULTIPLIER,
            self.engine._construction_productivity(self.agent),
        )
        self.assertEqual(3.0, self.agent.get_activity_multiplier())
        self.assertEqual(
            "construction_pace:surge",
            self.agent._construction_effort_contract["action_key"],
        )

    def test_construction_medical_envelope_masks_learned_surge(self):
        site = {
            "id": "unsafe-site",
            "required_work_hours": 40.0,
            "work_hours_completed": 5.0,
        }
        self.agent.action.action_type = "build"
        self.agent.action.target = {"struct_id": site["id"]}
        self.agent.needs.energy = 60.0
        state = self.engine._construction_effort_state(
            self.agent, site, active_robot_count=2
        )
        self.engine.strategic_policy.operational_q_table[state] = {
            "construction_pace:steady": -5.0,
            "construction_pace:surge": 50.0,
        }
        self.engine.strategic_policy.evaluation_mode = True

        mode = self.engine._construction_effort_mode(
            self.agent, site, active_robot_count=2
        )

        self.assertEqual("steady", mode)
        self.assertFalse(
            hasattr(self.agent, "_construction_effort_contract")
        )
        self.assertEqual(2.0, self.agent.get_activity_multiplier())

    def test_construction_scrubber_return_reserve_masks_surge(self):
        site = {
            "id": "scrubber-site",
            "required_work_hours": 40.0,
            "work_hours_completed": 5.0,
        }
        self.agent.action.action_type = "build"
        self.agent.action.target = {"struct_id": site["id"]}
        self.agent.plss_co2_scrubber_pct = 20.0
        state = self.engine._construction_effort_state(
            self.agent, site, active_robot_count=2
        )
        self.engine.strategic_policy.operational_q_table[state] = {
            "construction_pace:steady": -5.0,
            "construction_pace:surge": 50.0,
        }
        self.engine.strategic_policy.evaluation_mode = True

        mode = self.engine._construction_effort_mode(
            self.agent, site, active_robot_count=2
        )

        self.assertEqual("steady", mode)
        self.assertFalse(
            hasattr(self.agent, "_construction_effort_contract")
        )

    def test_construction_shift_updates_shared_local_credit_only(self):
        site = {"id": "credit-site"}
        state = (
            "construction-effort-v1|phase:assembly|remaining:long|"
            "energy:fresh|reserve:full|robots:assisted"
        )
        self.engine.strategic_policy.operational_q_table[state] = {
            "construction_pace:steady": 0.0,
            "construction_pace:surge": 0.0,
        }
        self.agent.last_state_key = "primary-state"
        self.agent.last_action_key = "build:habitat_module"
        tactical_before = dict(self.agent.q_table)
        self.agent._construction_effort_contract = {
            "site_id": site["id"],
            "mode": "surge",
            "state_key": state,
            "action_key": "construction_pace:surge",
            "start_tick": 10,
            "active_ticks": 0,
            "productive_work_hours": 0.0,
            "baseline_work_hours": 0.0,
            "start_energy": self.agent.needs.energy,
            "start_hunger": self.agent.needs.hunger,
            "start_thirst": self.agent.needs.thirst,
            "start_o2": self.agent.needs.o2_supply,
            "start_scrubber": self.agent.plss_co2_scrubber_pct,
            "start_battery": self.agent.plss_suit_battery_pct,
        }
        for _ in range(3):
            self.engine._record_construction_effort(
                self.agent,
                site,
                productive_work_hours=0.2167,
                baseline_work_hours=0.1667,
            )
        self.agent.needs.energy -= 5.0
        self.agent.needs.hunger -= 4.0

        reward = self.engine._settle_construction_effort(
            self.agent, completed=True, reason="shift_complete"
        )

        self.assertGreater(reward, 0.0)
        self.assertGreater(
            self.engine.strategic_policy.operational_q_table[state][
                "construction_pace:surge"
            ],
            0.0,
        )
        self.assertEqual(tactical_before, self.agent.q_table)
        self.assertEqual("primary-state", self.agent.last_state_key)
        self.assertEqual("build:habitat_module", self.agent.last_action_key)
        self.assertEqual(
            1,
            self.engine._construction_effort_telemetry["shared_outcomes"],
        )

    def test_rover_return_reserve_uses_recorded_vehicle_route(self):
        self.agent.x = self.engine.lz_x + 20
        self.agent.y = self.engine.lz_y
        self.agent._in_habitat = False
        rover = self.engine.surface_fleet.crew_rovers[0]
        rover.x, rover.y = self.agent.x, self.agent.y
        rover.state = "in_use"
        rover.condition = 0.90
        route = [
            {"x": self.engine.lz_x + step, "y": self.engine.lz_y}
            for step in range(21)
        ]
        expedition_id = "test-rover-return"
        rover.mission = {"expedition_id": expedition_id, "route": route}
        self.agent._active_expedition = {
            "id": expedition_id,
            "kind": "construction_support",
            "transport": "crew_rover",
            "route": route,
            "route_index": 20,
            "move_speed_cells": 10,
        }

        self.assertEqual(3, self.engine._eva_return_transport_ticks(self.agent))
        self.assertGreater(
            self.engine._eva_walk_home_ticks(self.agent),
            self.engine._eva_return_transport_ticks(self.agent),
        )

    def test_distance_adjusted_scrubber_reserve_orders_return(self):
        self.agent.x = self.engine.lz_x + 20
        self.agent.y = self.engine.lz_y
        self.agent._in_habitat = False
        self.agent.plss_co2_scrubber_pct = 35.0
        self.agent.plss_suit_battery_pct = 100.0
        self.agent.suit_integrity = 1.0
        self.agent.suit_condition = 1.0
        self.engine.decision_engine.agents = [self.agent]
        self.engine.decision_engine.lz_x = self.engine.lz_x
        self.engine.decision_engine.lz_y = self.engine.lz_y

        decision = self.engine.decision_engine.process_tick(
            self.agent,
            tick=1,
            tick_events={},
            world_context={
                "effective_temperature_c": 18.0,
                "active_events": [],
            },
            nearby_agents=[],
        )

        self.assertEqual("move", decision["action"])
        self.assertIn("scrubber", decision["reasoning"].lower())

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
