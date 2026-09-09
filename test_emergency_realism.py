"""Regression tests for medical alarms, fatigue and safe extraction."""

import math
import unittest
from pathlib import Path

from src.agents.agent import AgentStatus, create_team_from_presets
from src.agents.decision import PlanStep, StrategicPlan
from src.memory import vector_store
from src.orchestration.engine import SimulationEngine
from src.systems.event_scheduler import WorldEvent


ROOT = Path(__file__).resolve().parent
vector_store._use_tfidf_fallback = True


class EmergencyRealismTest(unittest.TestCase):
    def setUp(self):
        self.engine = SimulationEngine(
            str(ROOT / "config" / "planets" / "kepler-442b.json"),
            seed=42,
            db=False,
        )
        self.engine.llm_client.call = lambda *_args, **_kwargs: None
        self.crew = create_team_from_presets(
            str(ROOT / "config" / "agent_presets.json")
        )[:2]
        for agent in self.crew:
            self.engine.add_agent(agent)
        self.engine._init_agents()
        for agent in self.crew:
            agent.needs.energy = 100.0
            agent.needs.hunger = 100.0
            agent.needs.thirst = 100.0
            agent.needs.o2_supply = 100.0
            agent.needs.temperature_stress = 60.0
            agent._current_canister_remaining = 100.0
            agent.plss_co2_scrubber_pct = 100.0
            agent.plss_suit_battery_pct = 100.0
            agent.suit_integrity = 1.0

    def _prefer_shared_strategy(self, recipe, structures):
        """Give a test policy one learned preference without changing masks."""
        planner = self.engine.decision_engine
        planner.colony.update_counts(structures)
        policy = planner.strategic_policy
        policy.commitment = None
        policy.initial_epsilon = 0.0
        policy.minimum_epsilon = 0.0
        state = planner.get_colony_strategy_state(structures)
        policy.q_table[state] = {
            policy.action_key(recipe): 100.0,
        }

    def _total_manifest_item(self, item):
        """Count finite stock regardless of its current lander custodian."""
        return int(self.engine.central_depot_inventory.get(item, 0)) + sum(
            int(crew.inventory.items.get(item, 0)) for crew in self.crew
        )

    def _total_food_kcal(self):
        """Count packaged meals and bulk habitat food in one energy ledger."""
        ration_kcal = float(
            self.crew[0].FOOD_TYPES["emergency_rations"]["kcal"]
        )
        packs = int(self.engine.central_depot_inventory.get("ration_packs", 0))
        for crew in self.crew:
            packs += int(crew.inventory.items.get("emergency_rations", 0))
            packs += int(crew.inventory.items.get("ration_pack", 0))
        return float(self.engine._colony_resources["food_reserve_kcal"]) + (
            packs * ration_kcal
        )

    def _safe_cardinal_target(self, distance, ordinal=0):
        """Return a test face outside both the campus and planned LZ corridor."""
        candidates = (
            (self.engine.lz_x, self.engine.lz_y - distance),
            (self.engine.lz_x, self.engine.lz_y + distance),
            (self.engine.lz_x - distance, self.engine.lz_y),
            (self.engine.lz_x + distance, self.engine.lz_y),
        )
        legal = [
            coord for coord in candidates
            if not self.engine._is_construction_protected_cell(*coord)
        ]
        return legal[ordinal]

    def test_incapacitated_crew_member_wakes_sleeping_rescuer(self):
        victim, rescuer = self.crew
        victim.status = AgentStatus.INCAPACITATED
        victim._in_habitat = False
        victim.x = self.engine.lz_x + 5
        victim.y = self.engine.lz_y

        rescuer.x = self.engine.lz_x
        rescuer.y = self.engine.lz_y
        rescuer._in_habitat = True
        rescuer.action.action_type = "sleep"
        rescuer.action.ticks_remaining = 20
        rescuer.action.target = {"ticks": 20, "habitat": True}
        rescuer.needs._consecutive_sleep_ticks = 4
        self.engine.current_tick = 10

        self.engine._process_agent_tick(rescuer, [], nearby_count=1)

        self.assertNotEqual("sleep", rescuer.action.action_type)
        self.assertEqual(10, rescuer._last_emergency_wake_tick)
        self.assertTrue(rescuer._team_emergency_alert["active"])
        self.assertIn(victim.id, rescuer._team_emergency_alert["victim_ids"])
        self.assertEqual(
            (victim.x, victim.y),
            (rescuer.action.target.get("x"), rescuer.action.target.get("y")),
        )

    def test_collapse_inside_base_also_receives_medical_response(self):
        victim, rescuer = self.crew
        for agent in (victim, rescuer):
            agent.x = self.engine.lz_x
            agent.y = self.engine.lz_y
            agent._in_habitat = True
        victim.status = AgentStatus.INCAPACITATED
        victim.needs.energy = 0.0
        rescuer.action.action_type = "sleep"
        rescuer.action.ticks_remaining = 20
        self.engine.current_tick = 11

        self.engine._process_agent_tick(rescuer, [], nearby_count=1)

        self.assertEqual(AgentStatus.INCAPACITATED, victim.status)
        self.assertEqual("medical_rest", victim.action.action_type)
        self.assertEqual(0.0, victim.needs.energy)
        self.assertEqual(11, rescuer._last_emergency_wake_tick)

        energy_at_admission = victim.needs.energy
        victim.tick_update(
            ambient_temp_c=21.0,
            gravity_multiplier=1.0,
            has_shelter=True,
            has_atmosphere=True,
            pressure_kpa=101.3,
        )
        self.assertGreater(victim.needs.energy, energy_at_admission)
        # Once pressure/oxygen are safe and monitored recovery has begun, an
        # exhaustion-only collapse is tracked as clinically critical rather
        # than as an unresponsive field casualty.
        self.assertEqual(AgentStatus.CRITICAL, victim.status)

    def test_fatigue_cannot_be_a_direct_cause_of_death(self):
        agent = self.crew[0]
        agent.needs.energy = 0.0
        agent.needs._energy_death_timer = agent.needs.EXHAUSTION_TICKS * 10

        self.assertIsNone(agent.needs.check_death())

    def test_fatigue_collapse_in_habitat_enters_monitored_rest(self):
        agent = self.crew[0]
        agent._in_habitat = True
        agent.needs.energy = 0.0
        agent.needs._energy_death_timer = 1
        agent.action.action_type = "idle"

        events = agent.tick_update(
            ambient_temp_c=21.0,
            gravity_multiplier=1.0,
            has_shelter=True,
            has_atmosphere=True,
            pressure_kpa=101.3,
        )

        self.assertFalse(events.get("died", False))
        self.assertTrue(events.get("fatigue_collapse"))
        self.assertEqual(AgentStatus.CRITICAL, agent.status)
        self.assertEqual("medical_rest", agent.action.action_type)
        before = agent.needs.energy
        agent.tick_update(
            ambient_temp_c=21.0,
            gravity_multiplier=1.0,
            has_shelter=True,
            has_atmosphere=True,
            pressure_kpa=101.3,
        )
        self.assertGreater(agent.needs.energy, before)

    def test_medical_admission_cannot_create_water_or_oxygen(self):
        victim, rescuer = self.crew
        for agent in (victim, rescuer):
            agent.x = self.engine.lz_x
            agent.y = self.engine.lz_y
            agent._in_habitat = True
            agent.inventory.items.pop("water_packs", None)
            agent.inventory.items.pop("oxygen_canisters", None)
        self.engine.central_depot_inventory["water_packs"] = 0
        self.engine.central_depot_inventory["oxygen_canisters"] = 0
        self.engine._colony_resources["water_reserve_l"] = 0.0
        self.engine._colony_resources["o2_reserve_kg"] = 0.0
        victim.status = AgentStatus.INCAPACITATED
        victim.needs.thirst = 0.0
        victim.needs.o2_supply = 0.0
        victim.needs.energy = 25.0
        rescuer.competency.medical = 10
        rescuer.action.action_type = "idle"
        self.engine.current_tick = 12

        self.engine._process_agent_tick(rescuer, [], nearby_count=1)

        self.assertEqual("medical_rest", victim.action.action_type)
        self.assertEqual(0.0, victim.needs.thirst)
        self.assertEqual(0.0, victim.needs.o2_supply)
        self.assertEqual(AgentStatus.INCAPACITATED, victim.status)

    def test_entering_empty_habitat_does_not_refill_oxygen(self):
        victim = self.crew[0]
        victim._in_habitat = False
        victim.needs.o2_supply = 3.0

        victim.enter_habitat()

        self.assertEqual(3.0, victim.needs.o2_supply)

    def test_medic_cannot_create_high_flow_oxygen_without_a_source(self):
        victim, medic = self.crew
        for agent in (victim, medic):
            agent.x = self.engine.lz_x
            agent.y = self.engine.lz_y
            agent._in_habitat = True
            agent.inventory.items.pop("oxygen_canisters", None)
        victim.needs.o2_supply = 20.0
        medic.competency.medical = 10
        self.engine._colony_resources["o2_reserve_kg"] = 0.0

        self.engine._process_agent_tick(medic, [], nearby_count=1)

        self.assertEqual(20.0, victim.needs.o2_supply)
        self.assertNotEqual("treat", medic.action.action_type)
        self.assertNotEqual("treat", medic.last_decision["action"])

    def test_non_sar_sleeper_keeps_continuous_sleep_during_alarm(self):
        victim, lead = self.crew
        sleeper = create_team_from_presets(
            str(ROOT / "config" / "agent_presets.json")
        )[2]
        self.engine.add_agent(sleeper)
        victim.status = AgentStatus.INCAPACITATED
        victim._in_habitat = False
        victim.x = self.engine.lz_x + 5
        victim.y = self.engine.lz_y
        lead.x = victim.x
        lead.y = victim.y
        lead.competency.medical = 10
        sleeper.x = self.engine.lz_x
        sleeper.y = self.engine.lz_y
        sleeper._in_habitat = True
        sleeper.competency.medical = 0
        sleeper.needs.energy = 50.0
        sleeper.action.action_type = "sleep"
        sleeper.action.ticks_remaining = 20
        sleeper.action.target = {"ticks": 20, "habitat": True}

        for tick in range(20, 24):
            self.engine.current_tick = tick
            self.engine._process_agent_tick(sleeper, [], nearby_count=1)

        self.assertEqual("sleep", sleeper.action.action_type)
        self.assertGreaterEqual(sleeper.needs._consecutive_sleep_ticks, 4)
        self.assertIsNone(getattr(sleeper, "_last_emergency_wake_tick", None))
        self.assertFalse(sleeper._team_emergency_alert["assigned_sar"])

    def test_low_suit_canister_is_replaced_before_eva(self):
        agent = self.crew[0]
        agent.suit_equipped = True
        agent._in_habitat = True
        agent._current_canister_remaining = 5.2
        agent.needs.o2_supply = 100.0
        agent.inventory.items["oxygen_canisters"] = 1

        result = agent.exit_habitat(has_atmosphere=False)

        self.assertTrue(result["exited"])
        self.assertEqual(100.0, agent._current_canister_remaining)
        self.assertEqual(100.0, agent.needs.o2_supply)
        self.assertFalse(agent.inventory.has_item("oxygen_canisters"))
        self.assertEqual(1, agent.inventory.items["empty_oxygen_canisters"])

    def test_habitat_o2_does_not_hide_depleted_plss_and_isru_refills_it(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.suit_equipped = True
        # Breathing in the habitat is nominal while the separate suit bottle
        # is nearly depleted: this was the airlock retry-loop state.
        agent.needs.o2_supply = 100.0
        agent._current_canister_remaining = 25.0
        agent.inventory.items.pop("oxygen_canisters", None)
        self.engine.central_depot_inventory["oxygen_canisters"] = 0
        self.engine.central_depot_inventory["empty_oxygen_canisters"] = 1
        self.engine._colony_resources["o2_reserve_kg"] = 10.0
        self.engine._colony_resources["energy_stored_kwh"] = 10.0
        station = {
            "id": "isru-test",
            "type": "isru_o2_unit",
            "x": self.engine.lz_x + 3,
            "y": self.engine.lz_y,
            "health": 1.0,
            "under_construction": False,
            "destroyed": False,
        }
        self.engine.placed_structures.append(station)
        self.engine.placed_structures.append({
            "id": "isru-test-fluid-grid",
            "type": "life_support_distribution_grid",
            "x": self.engine.lz_x + 2,
            "y": self.engine.lz_y,
            "health": 1.0,
            "under_construction": False,
            "destroyed": False,
        })
        # Isolate the outdoor compressor path. A normal pressurised campus has
        # an indoor lander-fed manifold and correctly uses that instead.
        self.engine.placed_structures = [
            structure for structure in self.engine.placed_structures
            if structure.get("type") != "eclss_lander_hub"
        ]
        self.engine.structures_built["isru_o2_unit"] = 1

        visited_station = False
        for _ in range(10):
            self.engine._process_agent_tick(agent, [], nearby_count=1)
            self.engine.current_tick += 1
            if max(abs(agent.x - station["x"]), abs(agent.y - station["y"])) <= 1:
                visited_station = True
            if agent.action.action_type == "refill_o2":
                break

        self.assertTrue(visited_station)
        self.assertEqual("refill_o2", agent.action.action_type)
        self.assertEqual("isru_o2_unit", agent.action.target["source"])
        self.assertEqual(100.0, agent._current_canister_remaining)
        self.assertEqual(1, agent.inventory.items["oxygen_canisters"])
        self.assertEqual(0, self.engine.central_depot_inventory["empty_oxygen_canisters"])
        self.assertLess(self.engine._colony_resources["o2_reserve_kg"], 10.0)

    def test_lander_eclss_refills_reusable_canister_before_isru_exists(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.suit_equipped = True
        agent.needs.o2_supply = 100.0
        agent._current_canister_remaining = 25.0
        agent.inventory.items.pop("oxygen_canisters", None)
        self.engine.central_depot_inventory["oxygen_canisters"] = 0
        self.engine.central_depot_inventory["empty_oxygen_canisters"] = 1
        self.engine._colony_resources["o2_reserve_kg"] = 10.0
        self.engine._colony_resources["energy_stored_kwh"] = 10.0

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("refill_o2", agent.action.action_type)
        self.assertEqual("eclss_lander_hub", agent.action.target["source"])
        self.assertEqual(100.0, agent._current_canister_remaining)
        self.assertEqual(1, agent.inventory.items["oxygen_canisters"])

    def test_pressurized_campus_manifold_prevents_low_plss_refill_loop(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x - 3
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.suit_equipped = True
        agent.needs.o2_supply = 100.0
        agent._current_canister_remaining = 16.0
        agent.inventory.items.pop("oxygen_canisters", None)
        self.engine.central_depot_inventory["oxygen_canisters"] = 0
        self.engine.central_depot_inventory["empty_oxygen_canisters"] = 1
        self.engine._colony_resources["o2_reserve_kg"] = 10.0
        self.engine._colony_resources["energy_stored_kwh"] = 10.0

        result = self.engine._execute_o2_refill_action(agent)

        self.assertTrue(result["refilled"])
        self.assertTrue(result["internal_manifold"])
        self.assertEqual("eclss_lander_hub", result["source"])
        self.assertEqual(100.0, agent._current_canister_remaining)
        self.assertEqual(1, agent.inventory.items["oxygen_canisters"])
        self.assertLess(self.engine._colony_resources["o2_reserve_kg"], 10.0)

    def test_full_plss_does_not_consume_a_full_carried_spare(self):
        agent = self.crew[0]
        agent._in_habitat = True
        agent._has_active_o2_canister = True
        agent._current_canister_remaining = 100.0
        agent.inventory.items["oxygen_canisters"] = 1
        agent.inventory.items.pop("empty_oxygen_canisters", None)
        loaded_before = agent._total_canisters_loaded

        result = self.engine._execute_o2_refill_action(agent)

        self.assertTrue(result["refilled"])
        self.assertTrue(result["spare_ready"])
        self.assertEqual("carried_spare_canister", result["source"])
        self.assertEqual(1, agent.inventory.items["oxygen_canisters"])
        self.assertNotIn("empty_oxygen_canisters", agent.inventory.items)
        self.assertEqual(loaded_before, agent._total_canisters_loaded)

    def test_eva_preflight_executes_internal_o2_service_before_retry(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x - 3
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.suit_equipped = True
        agent._has_active_o2_canister = True
        agent._current_canister_remaining = 16.0
        agent.inventory.items.pop("oxygen_canisters", None)
        self.engine.central_depot_inventory["oxygen_canisters"] = 0
        self.engine.central_depot_inventory["empty_oxygen_canisters"] = 1
        self.engine._colony_resources["o2_reserve_kg"] = 10.0
        self.engine._colony_resources["energy_stored_kwh"] = 10.0

        ready = self.engine._prepare_agent_for_eva(agent)

        self.assertFalse(ready)
        self.assertEqual("refill_o2", agent.action.action_type)
        self.assertEqual(100.0, agent._current_canister_remaining)
        self.assertEqual(1, agent.inventory.items["oxygen_canisters"])
        agent.action.clear()
        self.assertTrue(self.engine._prepare_agent_for_eva(agent))

    def test_bulk_o2_does_not_create_a_canister_away_from_isru_station(self):
        agent = self.crew[0]
        agent._in_habitat = True
        agent._current_canister_remaining = 100.0
        agent.inventory.items.pop("oxygen_canisters", None)
        self.engine.central_depot_inventory["oxygen_canisters"] = 0
        self.engine._colony_resources["o2_reserve_kg"] = 10.0
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "idle", "target": {}, "reasoning": "test"
        }

        self.engine._run_tick()

        self.assertFalse(agent.inventory.has_item("oxygen_canisters"))

    def test_eva_is_denied_when_no_safe_o2_supply_exists(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x + 1
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.suit_equipped = True
        agent._current_canister_remaining = 5.2
        agent.needs.o2_supply = 100.0
        agent.inventory.items.pop("oxygen_canisters", None)
        self.engine.central_depot_inventory["oxygen_canisters"] = 0
        self.engine._colony_resources["o2_reserve_kg"] = 0.0
        start = (agent.x, agent.y)
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "move",
            "target": {"x": self.engine.lz_x + 6, "y": self.engine.lz_y},
            "reasoning": "attempt EVA without life support",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual(start, (agent.x, agent.y))
        self.assertTrue(agent._in_habitat)
        self.assertEqual("idle", agent.action.action_type)
        self.assertEqual("no_o2_supply", agent.action.target["eva_denied"])

    def test_worn_suit_is_serviced_before_eva(self):
        agent = self.crew[0]
        agent.suit_integrity = 0.1
        agent._in_habitat = True
        agent.x, agent.y = self.engine._indoor_activity_position(agent, "service_suit")
        self.engine.central_depot_inventory["vacuum_gasket_seal"] = 1

        exited = self.engine._prepare_agent_for_eva(agent)

        self.assertFalse(exited)
        self.assertEqual(0.1, agent.suit_integrity)
        self.assertEqual("service_suit", agent.action.action_type)
        self.assertEqual(0, self.engine.central_depot_inventory["vacuum_gasket_seal"])

    def test_routine_suit_service_takes_time_without_consuming_pressure_seal(self):
        agent = self.crew[0]
        agent._in_habitat = True
        agent.x, agent.y = self.engine._indoor_activity_position(agent, "service_suit")
        agent.suit_integrity = 1.0
        agent.suit_condition = 0.40
        before_seals = self.engine.central_depot_inventory["vacuum_gasket_seal"]

        exited = self.engine._prepare_agent_for_eva(agent)

        self.assertFalse(exited)
        self.assertEqual("service_suit", agent.action.action_type)
        self.assertEqual(6, agent.action.ticks_remaining)
        self.assertEqual(0.40, agent.suit_condition)
        self.assertEqual(
            before_seals,
            self.engine.central_depot_inventory["vacuum_gasket_seal"],
        )

    def test_worn_suit_without_service_part_cannot_exit(self):
        agent = self.crew[0]
        agent.suit_integrity = 0.1
        agent._in_habitat = True
        agent.x, agent.y = self.engine._indoor_activity_position(agent, "service_suit")
        agent.inventory.materials.pop("vacuum_gasket_seal", None)
        agent.inventory.items.pop("vacuum_gasket_seal", None)
        self.engine.central_depot_inventory["vacuum_gasket_seal"] = 0

        exited = self.engine._prepare_agent_for_eva(agent)

        self.assertFalse(exited)
        self.assertTrue(agent._in_habitat)
        self.assertEqual("suit_integrity_unsafe", agent.action.target["eva_denied"])

    def test_habitat_crew_share_a_fabricated_suit_service_seal(self):
        agent, engineer = self.crew
        agent.suit_integrity = 0.40
        agent._in_habitat = True
        agent.x, agent.y = self.engine._indoor_activity_position(agent, "service_suit")
        engineer._in_habitat = True
        agent.inventory.materials.pop("vacuum_gasket_seal", None)
        agent.inventory.items.pop("vacuum_gasket_seal", None)
        self.engine.central_depot_inventory["vacuum_gasket_seal"] = 0
        engineer.inventory.add_material("vacuum_gasket_seal", 1)

        exited = self.engine._prepare_agent_for_eva(agent)

        self.assertFalse(exited)
        self.assertEqual(0.40, agent.suit_integrity)
        self.assertEqual("service_suit", agent.action.action_type)
        self.assertEqual(
            0, engineer.inventory.materials.get("vacuum_gasket_seal", 0)
        )

    def test_compromised_suit_triggers_return_before_pressure_loss(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x + 5
        agent.y = self.engine.lz_y
        agent._in_habitat = False
        agent.suit_integrity = 0.44
        agent.needs.o2_supply = 90.0
        self.engine.decision_engine.agents = [agent]
        self.engine.decision_engine.lz_x = self.engine.lz_x
        self.engine.decision_engine.lz_y = self.engine.lz_y

        decision = self.engine.decision_engine.process_tick(
            agent,
            tick=1,
            tick_events={},
            world_context={"effective_temperature_c": 18.0, "active_events": []},
            nearby_agents=[],
        )

        self.assertEqual("move", decision["action"])
        self.assertEqual(self.engine.lz_x, decision["target"]["x"])
        self.assertIn("PLSS", decision["reasoning"])

    def test_scheduler_stellar_flare_name_triggers_surface_return(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x + 5
        agent.y = self.engine.lz_y
        agent._in_habitat = False
        agent.suit_integrity = 1.0
        agent.needs.o2_supply = 100.0
        self.engine.decision_engine.agents = [agent]
        self.engine.decision_engine.lz_x = self.engine.lz_x
        self.engine.decision_engine.lz_y = self.engine.lz_y

        decision = self.engine.decision_engine.process_tick(
            agent,
            tick=1,
            tick_events={},
            world_context={
                "effective_temperature_c": 18.0,
                "active_events": [{"type": "stellar_flare", "severity": 0.5}],
            },
            nearby_agents=[],
        )

        self.assertEqual("move", decision["action"])
        self.assertEqual(self.engine.lz_x, decision["target"]["x"])
        self.assertIn("hazard", decision["reasoning"].lower())

    def test_fatigue_outranks_hygiene_when_water_is_depleted(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.needs.energy = 60.0
        agent.needs.hygiene = 5.0
        agent.inventory.items.pop("water_packs", None)
        self.engine.central_depot_inventory["water_packs"] = 0
        self.engine._colony_resources["water_reserve_l"] = 0.0
        self.engine.decision_engine.central_depot_inventory = self.engine.central_depot_inventory
        self.engine.decision_engine._colony_resources = self.engine._colony_resources

        decision = self.engine.decision_engine.process_tick(
            agent,
            tick=12,
            tick_events={},
            world_context={
                "effective_temperature_c": 21.0,
                "active_events": [],
                "colony_resources": dict(self.engine._colony_resources),
            },
            nearby_agents=[],
        )

        self.assertEqual("sleep", decision["action"])

    def test_action_duration_advances_only_once_per_tick(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.action.action_type = "sleep"
        agent.action.target = {"habitat": True}
        agent.action.ticks_remaining = 5

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual(4, agent.action.ticks_remaining)

    def test_long_build_is_interrupted_before_fatigue_collapse(self):
        agent = self.crew[0]
        agent._in_habitat = True
        # Isolate the fatigue interrupt while already at the recovery berth;
        # normal indoor orders now walk there before beginning sleep.
        agent.x, agent.y = self.engine._indoor_activity_position(agent, "sleep")
        agent.needs.energy = 65.0
        agent.action.action_type = "build"
        agent.action.target = {"structure": "habitat_module"}
        agent.action.ticks_remaining = 40

        events = self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("build", events["physiological_safety_interrupt"])
        self.assertEqual("sleep", agent.action.action_type)
        self.assertGreater(agent.needs.energy, 0.0)

    def test_long_machine_cycle_is_interrupted_at_plss_return_reserve(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x + 2
        agent.y = self.engine.lz_y
        agent._in_habitat = False
        agent.suit_equipped = True
        agent._needs_o2_support = True
        agent._current_canister_remaining = 39.0
        agent.needs.o2_supply = 39.0
        agent.action.action_type = "refine"
        agent.action.target = {
            "output": "metal_pipe",
            "completion_pending": True,
            "remaining_ticks": 40,
        }
        agent.action.ticks_remaining = 40

        events = self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("refine", events["physiological_safety_interrupt"])
        self.assertNotEqual("refine", agent.action.action_type)
        self.assertGreater(agent.needs.o2_supply, 0.0)
        self.assertTrue(hasattr(agent, "_paused_manufacturing"))

    def test_low_hygiene_is_not_a_clinical_critical_state(self):
        agent = self.crew[0]
        agent._in_habitat = True
        agent.needs.hygiene = 0.0

        events = agent.tick_update(
            ambient_temp_c=21.0,
            gravity_multiplier=1.0,
            has_shelter=True,
            has_atmosphere=True,
            pressure_kpa=101.3,
        )

        self.assertIn("hygiene_critical", events["warnings"])
        self.assertEqual(AgentStatus.ALIVE, agent.status)

    def test_cold_agent_returns_before_hypothermia(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x + 4
        agent.y = self.engine.lz_y
        agent._in_habitat = False
        agent.needs.temperature_stress = 45.0
        self.engine.decision_engine.agents = [agent]
        self.engine.decision_engine.lz_x = self.engine.lz_x
        self.engine.decision_engine.lz_y = self.engine.lz_y

        decision = self.engine.decision_engine.process_tick(
            agent,
            tick=1,
            tick_events={},
            world_context={"effective_temperature_c": -18.0, "active_events": []},
            nearby_agents=[],
        )

        self.assertEqual("move", decision["action"])
        self.assertEqual(self.engine.lz_x, decision["target"]["x"])
        self.assertIn("hypothermia", decision["reasoning"].lower())

    def test_fatigued_agent_returns_instead_of_sleeping_in_field(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x + 4
        agent.y = self.engine.lz_y
        agent._in_habitat = False
        agent.needs.energy = 49.0
        self.engine.decision_engine.agents = [agent]
        self.engine.decision_engine.lz_x = self.engine.lz_x
        self.engine.decision_engine.lz_y = self.engine.lz_y

        decision = self.engine.decision_engine.process_tick(
            agent,
            tick=1,
            tick_events={},
            world_context={"effective_temperature_c": 18.0, "active_events": []},
            nearby_agents=[],
        )

        self.assertEqual("move", decision["action"])
        self.assertTrue(decision["target"]["fatigue_return"])
        self.assertNotEqual("sleep", decision["action"])

    def test_o2_return_reserve_scales_with_distance_without_spare(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x + 20
        agent.y = self.engine.lz_y
        agent._in_habitat = False
        return_ticks = math.ceil(20 / self.engine.EVA_WALK_SPEED_CELLS)
        return_threshold = min(
            80.0,
            40.0 + return_ticks * agent.plss_o2_percent_per_tick(1.5),
        )
        agent.needs.o2_supply = return_threshold - 0.1
        agent.inventory.items.pop("oxygen_canisters", None)
        self.engine.decision_engine.agents = [agent]
        self.engine.decision_engine.lz_x = self.engine.lz_x
        self.engine.decision_engine.lz_y = self.engine.lz_y

        decision = self.engine.decision_engine.process_tick(
            agent,
            tick=1,
            tick_events={},
            world_context={"effective_temperature_c": 18.0, "active_events": []},
            nearby_agents=[],
        )

        self.assertEqual("move", decision["action"])
        self.assertEqual(self.engine.lz_x, decision["target"]["x"])

    def test_distant_agent_keeps_distance_scaled_energy_for_return(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x + 15
        agent.y = self.engine.lz_y
        agent._in_habitat = False
        return_ticks = math.ceil(15 / self.engine.EVA_WALK_SPEED_CELLS)
        agent.needs.energy = 55.0 + return_ticks * 1.25 - 0.1
        self.engine.decision_engine.agents = [agent]
        self.engine.decision_engine.lz_x = self.engine.lz_x
        self.engine.decision_engine.lz_y = self.engine.lz_y

        decision = self.engine.decision_engine.process_tick(
            agent,
            tick=1,
            tick_events={},
            world_context={"effective_temperature_c": 15.0, "active_events": []},
            nearby_agents=[],
        )

        self.assertEqual("move", decision["action"])
        self.assertTrue(decision["target"]["fatigue_return"])
        self.assertEqual(self.engine.lz_x, decision["target"]["x"])
        self.assertIn("return reserve", decision["reasoning"].lower())

    def test_distant_agent_swaps_carried_o2_before_critical_level(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x + 20
        agent.y = self.engine.lz_y
        agent._in_habitat = False
        return_ticks = math.ceil(20 / self.engine.EVA_WALK_SPEED_CELLS)
        return_threshold = min(
            80.0,
            40.0 + return_ticks * agent.plss_o2_percent_per_tick(1.5),
        )
        agent.needs.o2_supply = return_threshold - 0.1
        agent.inventory.items["oxygen_canisters"] = 1
        self.engine.decision_engine.agents = [agent]
        self.engine.decision_engine.lz_x = self.engine.lz_x
        self.engine.decision_engine.lz_y = self.engine.lz_y

        decision = self.engine.decision_engine.process_tick(
            agent,
            tick=1,
            tick_events={},
            world_context={"effective_temperature_c": 18.0, "active_events": []},
            nearby_agents=[],
        )

        self.assertEqual("refill_o2", decision["action"])
        self.assertTrue(decision["target"]["carried_spare"])

    def test_gathering_walks_to_zone_outside_base_buffer(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        start = (agent.x, agent.y)
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "gather",
            "target": {"resource": "regolith"},
            "reasoning": "test gather request",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        target = agent.action.target
        target_distance = self.engine._distance_from_lz(target["x"], target["y"])
        walked_distance = max(abs(agent.x - start[0]), abs(agent.y - start[1]))
        self.assertEqual(self.engine.MIN_EXTRACTION_RADIUS_CELLS, target_distance)
        self.assertLessEqual(
            walked_distance,
            self.engine.EVA_WALK_SPEED_CELLS,
            "Ajan fiziksel bir tickteki yürüme mesafesini aşmamalı",
        )
        self.assertEqual("move", agent.action.action_type)

    def test_production_excavation_cannot_start_in_future_building_campus(self):
        agent = self.crew[0]
        protected_site = self.engine._select_structure_site("solar_panel")
        agent.x, agent.y = protected_site
        agent._in_habitat = False
        agent._active_excavation = {
            "x": protected_site[0], "y": protected_site[1],
            "resource": "regolith",
        }
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "gather",
            "target": {"resource": "regolith"},
            "reasoning": "stale excavation inside future building ground",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("move", agent.action.action_type)
        self.assertIsNone(agent._active_excavation)
        self.assertFalse(
            self.engine._is_construction_protected_cell(
                agent.action.target["x"], agent.action.target["y"]
            )
        )
        self.assertEqual(
            0, self.engine.cell_excavation_depth.get(protected_site, 0)
        )

    def test_excavation_spoil_cannot_be_dumped_in_building_campus(self):
        excavation = self._safe_cardinal_target(
            self.engine.MIN_EXTRACTION_RADIUS_CELLS
        )

        drop = self.engine._spoil_drop_cell(*excavation, 10)

        self.assertFalse(self.engine._is_construction_protected_cell(*drop))

    def test_empty_deposit_order_is_cancelled_before_it_can_loop(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.inventory.materials.clear()
        emitted = []
        self.engine._on_event = emitted.append
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "deposit_materials",
            "target": {},
            "reasoning": "stale logistics order",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("idle", agent.last_decision["action"])
        self.assertEqual("rest", agent.action.action_type)
        self.assertTrue(agent.action.target["empty_deposit_rejected"])
        self.assertFalse(any(e.get("type") == "deposit_materials" for e in emitted))

    def test_engineer_offloads_heavy_payload_before_starting_another_eva(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.inventory.materials.clear()
        agent.inventory.add_material("basalt", 10)
        before = self.engine.central_depot_inventory.get("basalt", 0)

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("deposit_materials", agent.action.action_type)
        self.assertEqual(before + 10, self.engine.central_depot_inventory["basalt"])
        self.assertEqual({}, agent.inventory.materials)

    def test_heavy_payload_commits_to_storage_after_staging_at_habitat(self):
        agent = self.crew[0]
        habitat_x = self.engine.lz_x + 6
        habitat_y = self.engine.lz_y - 10
        agent.x = habitat_x
        agent.y = habitat_y - 1
        agent._in_habitat = False
        agent.action.action_type = "idle"
        agent.action.target = {}
        agent.inventory.materials.clear()
        agent.inventory.add_material("iron_ore", 6)
        depot_before = int(
            self.engine.central_depot_inventory.get("iron_ore", 0)
        )
        self.engine.placed_structures.append({
            "id": "nearby_habitat",
            "type": "habitat_module",
            "x": habitat_x,
            "y": habitat_y,
            "under_construction": False,
            "destroyed": False,
            "health": 1.0,
        })

        # Heavy field cargo first reaches the nearest pressure refuge.
        self.engine._process_agent_tick(agent, [], nearby_count=1)
        self.assertEqual((habitat_x, habitat_y), (agent.x, agent.y))

        # The next decision selects a real depot/crate access point.
        self.engine.current_tick += 1
        self.engine._process_agent_tick(agent, [], nearby_count=1)
        self.assertEqual("move", agent.action.action_type)
        self.assertEqual(
            "deposit_materials", agent.action.target.get("mission_action")
        )
        self.assertEqual(
            "material_storage", agent.action.target.get("destination")
        )
        storage_id = agent.action.target.get("storage_id")

        # Terrain and pressure transitions may include stationary or detour
        # ticks. Every visible haul leg must nevertheless retain the same
        # access point, and the finite payload must reach storage.
        for _ in range(12):
            if not agent.inventory.materials:
                break
            self.engine.current_tick += 1
            self.engine._process_agent_tick(agent, [], nearby_count=1)
            if agent.action.target.get("mission_action") == "deposit_materials":
                self.assertEqual(
                    storage_id, agent.action.target.get("storage_id")
                )
                self.assertEqual(
                    "material_storage",
                    agent.action.target.get("destination"),
                )

        self.assertEqual({}, agent.inventory.materials)
        self.assertEqual(
            depot_before + 6,
            self.engine.central_depot_inventory.get("iron_ore", 0),
        )

    def test_unsafe_heavy_hauler_returns_to_shelter_before_offloading(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x + 6
        agent.y = self.engine.lz_y - 18
        agent._in_habitat = False
        agent.needs.energy = 55.0
        agent.inventory.materials.clear()
        agent.inventory.add_material("iron_ore", 6)
        self.engine.placed_structures.append({
            "id": "recovery_habitat",
            "type": "habitat_module",
            "x": self.engine.lz_x + 6,
            "y": self.engine.lz_y - 10,
            "under_construction": False,
            "destroyed": False,
            "health": 1.0,
        })

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("move", agent.action.action_type)
        self.assertTrue(agent.action.target.get("fatigue_return"))
        self.assertEqual(
            (self.engine.lz_x + 6, self.engine.lz_y - 10),
            (agent.action.target.get("x"), agent.action.target.get("y")),
        )

    def test_fatigued_medic_keeps_return_route_instead_of_chasing_patient(self):
        patient, medic = self.crew
        patient.x = self.engine.lz_x + 12
        patient.y = self.engine.lz_y
        patient._in_habitat = False
        patient.injury_level = 0.45
        medic.x = self.engine.lz_x + 8
        medic.y = self.engine.lz_y
        medic._in_habitat = False
        medic.needs.energy = 60.0
        medic.action.action_type = "move"
        medic.action.target = {
            "x": self.engine.lz_x,
            "y": self.engine.lz_y,
            "destination": "shelter",
            "fatigue_return": True,
        }

        decision = self.engine.decision_engine.process_tick(
            medic,
            tick=42830,
            tick_events={},
            world_context={
                "effective_temperature_c": 20.0,
                "active_events": [],
                "colony_resources": self.engine._colony_resources,
            },
            nearby_agents=[{"agent_obj": patient}],
        )

        self.assertEqual("move", decision["action"])
        self.assertTrue(decision["target"].get("fatigue_return"))
        self.assertFalse(decision["target"].get("medical_response", False))

    def test_adjacent_remote_habitat_requires_physical_entry_before_sleep(self):
        agent = self.crew[0]
        habitat_x = self.engine.lz_x + 8
        habitat_y = self.engine.lz_y + 8
        self.engine.placed_structures.append({
            "id": "remote_sleep_habitat",
            "type": "habitat_module",
            "x": habitat_x,
            "y": habitat_y,
            "under_construction": False,
            "destroyed": False,
            "health": 1.0,
        })
        agent.x = habitat_x - 1
        agent.y = habitat_y
        agent._in_habitat = False
        agent.needs.energy = 55.0
        agent.action.action_type = "idle"
        agent.action.target = {}

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual((habitat_x, habitat_y), (agent.x, agent.y))
        self.assertTrue(agent._in_habitat)
        self.assertTrue(self.engine._is_pressurized_location(agent.x, agent.y))
        self.assertTrue(self.engine._is_crew_quarters_location(agent))
        self.assertFalse(
            agent.action.action_type == "sleep"
            and not self.engine._is_crew_quarters_location(agent)
        )

    def test_remote_chalcopyrite_ore_request_does_not_preempt_ready_starter_solar(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.needs.temperature_stress = 50.0
        self.engine.remote_resource_requests.add("chalcopyrite_ore")
        self.engine.central_depot_inventory.pop("basalt", None)

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        # A remote request cannot replace the policy's committed capacity
        # objective with a hard-coded research/build order.
        self.assertTrue(agent.last_decision["target"].get("shared_work_order"))
        objective = self.engine.strategic_policy.commitment.recipe
        self.assertIn(objective, agent.last_decision["reasoning"])
        self.assertNotIn("research_workbench", agent.last_decision["reasoning"])

    def test_delivered_solar_kit_is_protected_from_water_subassemblies(self):
        decision_engine = self.engine.decision_engine
        decision_engine.surface_requires_plss = True

        shared = decision_engine._select_shared_capacity_order(
            dict(self.engine.central_depot_inventory), {}
        )

        self.assertIn(
            shared["recipe"],
            {
                "solar_panel", "isru_o2_unit", "water_collector",
                "greenhouse", "habitat_module",
                "power_distribution_grid", "life_support_distribution_grid",
                "potable_water_tank", "oxygen_buffer_tank",
            },
        )
        self.assertTrue(shared["strategy_committed"])

    def test_anoxic_planet_builds_isru_before_greenhouse_after_power(self):
        decision_engine = self.engine.decision_engine
        decision_engine.surface_requires_plss = True
        structures = {"water_collector": 1, "solar_panel": 2}
        isru_recipe = decision_engine._recipes_cache["isru_o2_unit"]
        colony_mats = dict(isru_recipe["materials"])
        self._prefer_shared_strategy("isru_o2_unit", structures)

        decision = decision_engine._balanced_milestone_candidate(
            self.crew[0], colony_mats, structures
        )

        self.assertEqual("build", decision["action"])
        self.assertEqual("isru_o2_unit", decision["target"]["recipe"])

    def test_bootstrap_prioritizes_o2_before_redundant_second_solar_array(self):
        decision_engine = self.engine.decision_engine
        decision_engine.surface_requires_plss = True
        structures = {"water_collector": 1, "solar_panel": 1}
        isru_recipe = decision_engine._recipes_cache["isru_o2_unit"]
        colony_mats = dict(isru_recipe["materials"])
        self._prefer_shared_strategy("isru_o2_unit", structures)

        candidate = decision_engine._balanced_milestone_candidate(
            self.crew[0], colony_mats, structures
        )
        shared = decision_engine._select_shared_capacity_order(
            colony_mats, structures
        )

        self.assertEqual("isru_o2_unit", candidate["target"]["recipe"])
        self.assertEqual("isru_o2_unit", shared["recipe"])
        self.assertEqual("capacity:isru_o2_unit", shared["strategy_action"])

    def test_o2_bootstrap_secures_water_feedstock_before_isru_plant(self):
        decision_engine = self.engine.decision_engine
        decision_engine.surface_requires_plss = True
        structures = {"solar_panel": 1}
        water_recipe = decision_engine._recipes_cache["water_collector"]
        colony_mats = dict(water_recipe["materials"])
        self._prefer_shared_strategy("water_collector", structures)

        candidate = decision_engine._balanced_milestone_candidate(
            self.crew[0], colony_mats, structures
        )
        shared = decision_engine._select_shared_capacity_order(
            colony_mats, structures
        )

        self.assertEqual("water_collector", candidate["target"]["recipe"])
        self.assertEqual("water_collector", shared["recipe"])
        self.assertEqual("capacity:water_collector", shared["strategy_action"])

    def test_second_solar_follows_o2_and_water_bootstrap(self):
        decision_engine = self.engine.decision_engine
        decision_engine.surface_requires_plss = True
        structures = {
            "solar_panel": 1,
            "isru_o2_unit": 1,
            "water_collector": 1,
        }
        solar_recipe = decision_engine._recipes_cache["solar_panel"]
        colony_mats = dict(solar_recipe["materials"])
        self._prefer_shared_strategy("solar_panel", structures)

        shared = decision_engine._select_shared_capacity_order(
            colony_mats, structures
        )

        self.assertEqual("solar_panel", shared["recipe"])
        self.assertEqual("capacity:solar_panel", shared["strategy_action"])

    def test_capacity_plan_keeps_building_modules_after_first_copy(self):
        decision_engine = self.engine.decision_engine
        decision_engine.surface_requires_plss = True
        structures = {
            "solar_panel": 1,
            "isru_o2_unit": 1,
            "water_collector": 1,
            "greenhouse": 1,
            "habitat_module": 1,
        }
        self.engine.colony_score.update_counts(structures)
        solar_recipe = decision_engine._recipes_cache["solar_panel"]
        colony_mats = dict(solar_recipe["materials"])
        self._prefer_shared_strategy("solar_panel", structures)

        decision = decision_engine._balanced_milestone_candidate(
            self.crew[0], colony_mats, structures
        )

        self.assertEqual("build", decision["action"])
        self.assertEqual("solar_panel", decision["target"]["recipe"])
        # Kepler-442b receives 70% of Earth's flux and is modeled as rotating,
        # so firm planning uses 26 arrays rather than the 18-array nameplate
        # count (ceil(18 / 0.70)).
        self.assertEqual(26, decision["target"]["capacity_target_count"])

    def test_capacity_counts_are_derived_from_recipe_targets(self):
        expected = {
            "solar_panel": 26,
            "isru_o2_unit": 20,
            "water_collector": 14,
            "potable_water_tank": 6,
            "oxygen_buffer_tank": 2,
            "greenhouse": 6,
            "habitat_module": 9,
        }
        self.engine.colony_score.update_counts({})

        actual = {
            recipe: self.engine.colony_score.get_structure_capacity_status(
                recipe
            )["required_count"]
            for recipe in expected
        }

        self.assertEqual(expected, actual)

    def test_final_reserved_canister_can_commission_isru_o2_unit(self):
        # Isolate the final-cylinder transaction from two-seat transport;
        # construction rover mobilization has its own paired-crew regressions.
        self.engine.CREW_ROVER_MIN_DISTANCE_CELLS = 1000
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent._current_canister_remaining = 0.0
        agent.inventory.items.pop("oxygen_canisters", None)
        agent.inventory.materials.clear()
        # This test isolates the final-canister commissioning rule after the
        # delivered starter array has already been deployed; otherwise its
        # certified components are correctly unavailable to the ISRU build.
        self.engine.delivered_structure_kits.clear()
        recipe = self.engine._get_recipe("isru_o2_unit")
        for material, quantity in recipe["materials"].items():
            self.engine.central_depot_inventory[material] = quantity
        self.engine.central_depot_inventory["oxygen_canisters"] = 1
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "build",
            "target": {"recipe": "isru_o2_unit"},
            "reasoning": "commission local oxygen before portable reserve is exhausted",
        }

        for _ in range(30):
            self.engine._process_agent_tick(agent, [], nearby_count=1)
            if any(
                structure.get("type") == "isru_o2_unit"
                and structure.get("under_construction")
                and structure.get("materials_committed")
                for structure in self.engine.placed_structures
            ):
                break
            self.engine._dispatch_construction_cargo()
            self.engine._tick_surface_fleet()
            self.engine.current_tick += 1

        # Once cargo is committed, life-support construction now begins with
        # the real buried utility route rather than abstract assembly in place.
        self.assertEqual("move", agent.action.action_type)
        self.assertTrue(agent.action.target.get("utility_route_workfront"))
        self.assertEqual(0, self.engine.central_depot_inventory["oxygen_canisters"])
        self.assertTrue(any(
            structure.get("type") == "isru_o2_unit"
            and structure.get("under_construction")
            for structure in self.engine.placed_structures
        ))

    def test_failed_build_preflight_does_not_pull_depot_materials(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.inventory.materials.clear()
        for tool in ("multitool_kit", "hand_tools", "stone_hammer"):
            agent.inventory.items.pop(tool, None)
            agent.inventory.tool_durability.pop(tool, None)
        recipe = self.engine._get_recipe("isru_o2_unit")
        for material, quantity in recipe["materials"].items():
            self.engine.central_depot_inventory[material] = quantity
        before = dict(self.engine.central_depot_inventory)
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "build",
            "target": {"recipe": "isru_o2_unit"},
            "reasoning": "preflight transaction test",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("craft_blocked", agent.action.action_type)
        self.assertEqual(before, self.engine.central_depot_inventory)
        self.assertEqual({}, agent.inventory.materials)

    def test_tool_breaking_at_construction_start_is_removed_once(self):
        # Exercise one physical tool transaction, not rover buddy scheduling.
        self.engine.CREW_ROVER_MIN_DISTANCE_CELLS = 1000
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.needs.energy = 100.0
        agent._current_canister_remaining = 100.0
        self.engine.delivered_structure_kits.clear()
        for tool in ("hand_tools", "stone_hammer"):
            agent.inventory.items.pop(tool, None)
            agent.inventory.tool_durability.pop(tool, None)
        agent.inventory.items["multitool_kit"] = 1
        agent.inventory.tool_durability["multitool_kit"] = 3
        recipe = self.engine._get_recipe("isru_o2_unit")
        for material, quantity in recipe["materials"].items():
            self.engine.central_depot_inventory[material] = quantity
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "build",
            "target": {"recipe": "isru_o2_unit"},
            "reasoning": "construction tool lifecycle regression",
        }

        for _ in range(30):
            self.engine._process_agent_tick(agent, [], nearby_count=1)
            if agent.action.action_type == "build":
                break
            self.engine._dispatch_construction_cargo()
            self.engine._tick_surface_fleet()
            self.engine.current_tick += 1

        self.assertEqual("build", agent.action.action_type)
        self.assertNotIn("multitool_kit", agent.inventory.items)
        self.assertNotIn("multitool_kit", agent.inventory.tool_durability)
        self.assertEqual(2, agent.inventory.materials.get("scrap_metal", 0))

    def test_worn_out_tools_recover_through_configured_stone_hammer_recipe(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.needs.temperature_stress = 50.0
        agent.action.clear()
        agent.inventory.materials.clear()
        for crew_member in self.crew:
            for tool in ("multitool_kit", "hand_tools", "stone_hammer"):
                crew_member.inventory.items.pop(tool, None)
                crew_member.inventory.tool_durability.pop(tool, None)
        stone_recipe = self.engine._get_recipe("stone_hammer")
        for material, quantity in stone_recipe["materials"].items():
            self.engine.central_depot_inventory[material] = quantity

        self.engine.decision_engine.agents = self.engine.agents
        self.engine.decision_engine.central_depot_inventory = (
            self.engine.central_depot_inventory
        )
        decision = self.engine.decision_engine.process_tick(
            agent,
            tick=25,
            tick_events={},
            world_context={"effective_temperature_c": 21.0, "active_events": []},
            nearby_agents=[],
        )
        self.assertEqual("craft_item", decision["action"])
        self.assertEqual("stone_hammer", decision["target"]["recipe"])

        self.engine.decision_engine.process_tick = lambda **_kwargs: decision
        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("retrieve_materials", agent.action.action_type)
        self.assertFalse(agent.inventory.has_usable_tool())
        for tick in range(1, 3):
            self.engine.current_tick = tick
            self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertTrue(agent.inventory.has_usable_tool())
        self.assertTrue(agent.inventory.has_item("stone_hammer"))
        for material, quantity in stone_recipe["materials"].items():
            self.assertEqual(0, self.engine.central_depot_inventory[material])

    def test_remote_craft_materials_require_a_physical_storage_trip(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x + 8
        agent.y = self.engine.lz_y
        agent._in_habitat = False
        agent.needs.temperature_stress = 50.0
        agent.action.clear()
        agent.inventory.materials.clear()
        for tool in ("multitool_kit", "hand_tools", "stone_hammer"):
            agent.inventory.items.pop(tool, None)
            agent.inventory.tool_durability.pop(tool, None)
        recipe = self.engine._get_recipe("stone_hammer")
        for material, quantity in recipe["materials"].items():
            self.engine.central_depot_inventory[material] = quantity
        depot_before = dict(self.engine.central_depot_inventory)
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "craft_item",
            "target": {"recipe": "stone_hammer", "tool_recovery": True},
            "reasoning": "physical storage access regression",
            "deterministic": True,
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("move", agent.action.action_type)
        self.assertTrue(agent.action.target["material_pickup_route"])
        self.assertEqual(depot_before, self.engine.central_depot_inventory)
        self.assertEqual({}, agent.inventory.materials)
        self.assertFalse(agent.inventory.has_item("stone_hammer"))

    def test_chief_engineer_borrows_nearby_shared_tool_before_primitive_rebuild(self):
        engineer, donor = self.crew
        engineer.competency.engineering = 10
        for agent in (engineer, donor):
            agent.x = self.engine.lz_x
            agent.y = self.engine.lz_y
            agent._in_habitat = True
            agent.needs.temperature_stress = 50.0
            agent.action.clear()
        for tool in ("multitool_kit", "hand_tools", "stone_hammer"):
            engineer.inventory.items.pop(tool, None)
            engineer.inventory.tool_durability.pop(tool, None)
        donor.inventory.tool_durability["multitool_kit"] = 37
        self.engine.central_depot_inventory["basalt"] = 2
        self.engine.decision_engine.agents = self.engine.agents
        self.engine.decision_engine.central_depot_inventory = (
            self.engine.central_depot_inventory
        )

        decision = self.engine.decision_engine.process_tick(
            engineer,
            tick=26,
            tick_events={},
            world_context={
                "effective_temperature_c": 21.0,
                "active_events": [],
                "colony_resources": {"energy_stored_kwh": 10.0},
            },
            nearby_agents=[],
        )
        self.assertEqual("borrow_tool", decision["action"])

        self.engine.decision_engine.process_tick = lambda **_kwargs: decision
        self.engine._process_agent_tick(engineer, [], nearby_count=1)

        self.assertEqual(37, engineer.inventory.tool_durability["multitool_kit"])
        self.assertFalse(donor.inventory.has_item("multitool_kit"))

    def test_empty_colony_battery_does_not_loop_scanner_recharge(self):
        scanner_agent = self.crew[0]
        scanner_agent.x = self.engine.lz_x
        scanner_agent.y = self.engine.lz_y
        scanner_agent._in_habitat = True
        scanner_agent.needs.temperature_stress = 50.0
        scanner_agent.action.clear()
        scanner_agent.inventory.add_item("portable_scanner", 1)
        scanner_agent.inventory.tool_charge_pct["portable_scanner"] = 0.0
        self.engine.decision_engine.remote_resource_requests = {"chalcopyrite_ore"}
        self.engine.decision_engine.agents = self.engine.agents

        decision = self.engine.decision_engine.process_tick(
            scanner_agent,
            tick=27,
            tick_events={},
            world_context={
                "effective_temperature_c": 21.0,
                "active_events": [],
                "colony_resources": {"energy_stored_kwh": 0.0},
            },
            nearby_agents=[],
        )

        self.assertNotEqual("recharge_scanner", decision["action"])

    def test_dry_base_does_not_repeat_impossible_drink_order(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.needs.thirst = 25.0
        agent.inventory.items.pop("water_packs", None)
        self.engine.central_depot_inventory["water_packs"] = 0
        self.engine._colony_resources["water_reserve_l"] = 0.0

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertNotEqual("drink", agent.last_decision["action"])
        self.assertNotEqual("idle", agent.action.action_type)

    def test_dry_exhausted_crew_preserves_sleep_instead_of_collapsing(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.needs.thirst = 44.0
        agent.needs.energy = 40.0
        agent.inventory.items.pop("water_packs", None)
        self.engine.central_depot_inventory["water_packs"] = 0
        self.engine._colony_resources["water_reserve_l"] = 0.0
        agent.action.action_type = "sleep"
        agent.action.target = {"habitat": True}
        agent.action.ticks_remaining = 20
        agent.needs._consecutive_sleep_ticks = 10

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual(
            "sleep",
            agent.action.action_type,
            f"decision={agent.last_decision} resources={self.engine._colony_resources}",
        )
        self.assertNotEqual("drink", agent.action.action_type)
        self.assertNotEqual("idle", agent.action.action_type)

    def test_resource_survey_targets_confirmed_deposit_not_biome_potential(self):
        target = (self.engine.lz_x + 8, self.engine.lz_y)
        self.engine.world.get_biome = lambda *_args: {
            "resources": {"chalcopyrite_ore": 1.0}
        }

        self.assertIsNone(
            self.engine._find_resource_from_lz("chalcopyrite_ore", 6, 12)
        )
        self.engine.discovered_resources.setdefault("chalcopyrite_ore", set()).add(
            target
        )
        found = self.engine._find_resource_from_lz("chalcopyrite_ore", 6, 12)

        self.assertEqual(target, found)

    def test_local_exploration_uses_cardinal_steps_and_expands_in_stages(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._local_exploration_radius = 8
        agent.explored_cells = {
            (self.engine.lz_x + dx, self.engine.lz_y + dy)
            for dx in range(-8, 9)
            for dy in range(-8, 9)
        }

        dx, dy = self.engine._nearest_unexplored_step(agent)

        # The 4 x 4 lander's fixed south suitlock is two cells from this
        # interior datum. A ten-minute indoor transit may cover both cells,
        # but it must stop at that exact pressurized airlock.
        self.assertEqual(2, abs(dx) + abs(dy))
        self.assertEqual(
            self.engine._lander_airlock_position(),
            (agent.x + dx, agent.y + dy),
        )
        self.assertEqual(12, agent._local_exploration_radius)

    def test_six_local_scans_do_not_unlock_remote_survey_early(self):
        scanner_agent = next(
            agent for agent in self.crew
            if agent.inventory.has_item("portable_scanner")
        )
        for _ in range(6):
            station = self.engine._next_portable_scan_station(scanner_agent)
            self.assertIsNotNone(station)
            self.engine._portable_scan_centers.add(station)

        remaining_local_station = self.engine._next_portable_scan_station(None)

        self.assertIsNotNone(remaining_local_station)
        self.assertLessEqual(
            self.engine._distance_from_lz(*remaining_local_station),
            self.engine.LOCAL_EVA_RADIUS_CELLS,
        )

    def test_portable_scan_traverse_is_staged_off_axis_and_gap_free(self):
        offsets = self.engine._portable_scan_station_offsets()
        scan_radius = self.engine.PORTABLE_SCANNER_SCAN_RADIUS_CELLS

        self.assertEqual(len(offsets), len(set(offsets)))
        first_radius = self.engine.MIN_EXTRACTION_RADIUS_CELLS
        local_radii = list(range(
            first_radius,
            self.engine.LOCAL_EVA_RADIUS_CELLS + 1,
            scan_radius * 2 + 1,
        ))
        reconnaissance_radii = []
        for pair_start in range(0, len(local_radii), 2):
            radial_pair = local_radii[pair_start:pair_start + 2]
            for _sector in range(6):
                reconnaissance_radii.extend(radial_pair)
        self.assertEqual(
            reconnaissance_radii,
            [
                max(abs(dx), abs(dy))
                for dx, dy in offsets[:len(reconnaissance_radii)]
            ],
        )
        self.assertFalse(any(
            dx == 0 or dy == 0 or abs(dx) == abs(dy)
            for dx, dy in offsets[:len(reconnaissance_radii)]
        ))

        covered = {
            (station_x + footprint_x, station_y + footprint_y)
            for station_x, station_y in offsets
            for footprint_y in range(-scan_radius, scan_radius + 1)
            for footprint_x in range(-scan_radius, scan_radius + 1)
        }
        missing = {
            (x, y)
            for y in range(
                -self.engine.LOCAL_EVA_RADIUS_CELLS,
                self.engine.LOCAL_EVA_RADIUS_CELLS + 1,
            )
            for x in range(
                -self.engine.LOCAL_EVA_RADIUS_CELLS,
                self.engine.LOCAL_EVA_RADIUS_CELLS + 1,
            )
            if self.engine.MIN_EXTRACTION_RADIUS_CELLS
            <= max(abs(x), abs(y))
            <= self.engine.LOCAL_EVA_RADIUS_CELLS
            and (x, y) not in covered
        }
        self.assertEqual(set(), missing)

        # Regression for the former coverage seam near the Kepler-442b LZ.
        # This is purely a geometry assertion; station selection never reads
        # sulfur or any other biome/deposit table.
        seam = (-16, 18)
        self.assertTrue(any(
            max(abs(seam[0] - dx), abs(seam[1] - dy)) <= scan_radius
            for dx, dy in offsets[:12]
        ))

    def test_resource_specific_misses_escalate_after_adequate_local_sample(self):
        resource = "sulfur"
        self.engine._local_resource_miss_centers[resource] = {
            (self.engine.lz_x + 6 + index, self.engine.lz_y)
            for index in range(self.engine.LOCAL_RESOURCE_MISS_LIMIT)
        }

        self.assertTrue(
            self.engine._resource_local_campaign_exhausted(resource)
        )

        known_local = (self.engine.lz_x + 7, self.engine.lz_y + 1)
        self.engine.discovered_resources[resource] = {known_local}
        self.assertFalse(
            self.engine._resource_local_campaign_exhausted(resource)
        )

    def test_final_blind_test_pit_dispatches_local_portable_scanner(self):
        agent = next(
            crew for crew in self.crew
            if not crew.inventory.has_item("portable_scanner")
        )
        resource = "chalcopyrite_ore"
        station = self.engine._next_hand_prospect_station(agent, resource)
        self.assertIsNotNone(station)
        agent.x, agent.y = station
        agent._in_habitat = False
        agent.needs.energy = 100.0
        agent.needs.hunger = 100.0
        agent.needs.thirst = 100.0
        agent.needs.o2_supply = 100.0
        agent.needs.temperature_stress = 50.0
        self.engine._local_resource_miss_centers[resource] = {
            (self.engine.lz_x + 30 + index, self.engine.lz_y)
            for index in range(self.engine.LOCAL_RESOURCE_MISS_LIMIT - 1)
        }
        original_get_cell_info = self.engine.world.get_cell_info

        def barren_test_station(x, y, tick=0):
            if (x, y) == station:
                return {"base_resources": {"regolith": "moderate"}}
            return original_get_cell_info(x, y, tick)

        self.engine.world.get_cell_info = barren_test_station
        self.engine.cell_geology.pop(station, None)
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "gather",
            "target": {"resource": resource},
            "reasoning": "open the final blind test pit",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("prospect", agent.action.action_type)
        self.assertIn(resource, self.engine.remote_resource_requests)
        self.assertIsNotNone(self.engine._next_portable_scan_station(None))

    def test_detector_sample_budget_advances_to_regional_ring(self):
        scanner_agent = next(
            crew for crew in self.crew
            if crew.inventory.has_item("portable_scanner")
        )
        resource = "chalcopyrite_ore"
        scanner_agent.x = self.engine.lz_x
        scanner_agent.y = self.engine.lz_y
        scanner_agent._in_habitat = True
        scanner_agent.needs.energy = 100.0
        scanner_agent.needs.hunger = 100.0
        scanner_agent.needs.thirst = 100.0
        scanner_agent.needs.o2_supply = 100.0
        scanner_agent.needs.temperature_stress = 50.0
        misses = {
            (self.engine.lz_x + 30 + index, self.engine.lz_y)
            for index in range(self.engine.LOCAL_RESOURCE_MISS_LIMIT)
        }
        self.engine._local_resource_miss_centers[resource] = set(misses)
        self.engine._portable_resource_miss_centers[resource] = set(misses)
        self.engine.remote_resource_requests.add(resource)

        self.engine._process_agent_tick(scanner_agent, [], nearby_count=1)

        expedition = getattr(scanner_agent, "_active_expedition", None)
        self.assertIsInstance(expedition, dict)
        self.assertEqual("regional_survey", expedition["kind"])
        self.assertGreater(
            expedition["authorized_radius"],
            self.engine.LOCAL_EVA_RADIUS_CELLS,
        )

    def test_regional_buddy_spare_is_refilled_after_delivered_stock_runs_out(self):
        scanner_agent = next(
            crew for crew in self.crew
            if crew.inventory.has_item("portable_scanner")
        )
        scanner_agent.x = self.engine.lz_x
        scanner_agent.y = self.engine.lz_y
        scanner_agent.inventory.items["oxygen_canisters"] = 1
        for crew in self.crew:
            crew.x = self.engine.lz_x
            crew.y = self.engine.lz_y
            crew._in_habitat = True
            crew.needs.energy = 100.0
            crew.needs.hunger = 100.0
            crew.needs.thirst = 100.0
            crew.needs.o2_supply = 100.0
            crew.needs.temperature_stress = 50.0
            crew._current_canister_remaining = 100.0
            crew.plss_co2_scrubber_pct = 100.0
            crew.plss_suit_battery_pct = 100.0
            crew.suit_integrity = 1.0
            if crew.id != scanner_agent.id:
                crew.inventory.items.pop("oxygen_canisters", None)
        self.engine.central_depot_inventory["oxygen_canisters"] = (
            self.engine.MIN_CENTRAL_MAINTENANCE_O2_CANISTERS
        )
        self.engine.central_depot_inventory["empty_oxygen_canisters"] = 10
        reserve_floor = (
            self.engine.MIN_CENTRAL_MAINTENANCE_O2_CANISTERS
            * scanner_agent.PLSS_CANISTER_O2_KG
        )
        self.engine._colony_resources["o2_reserve_kg"] = (
            reserve_floor + scanner_agent.PLSS_CANISTER_O2_KG
        )

        self.assertIsNone(
            self.engine._find_expedition_buddy(scanner_agent, 30)
        )
        candidate = self.engine._find_expedition_buddy(
            scanner_agent, 30, require_spare_canister=False
        )
        self.assertIsNotNone(candidate)
        self.assertTrue(self.engine._provision_expedition_spare(candidate))

        self.assertEqual(1, candidate.inventory.items["oxygen_canisters"])
        self.assertEqual("refill_o2", candidate.action.action_type)
        self.assertTrue(candidate.action.target["expedition_provisioning"])
        for _ in range(3):
            candidate.action.tick()
        self.assertIsNotNone(
            self.engine._find_expedition_buddy(scanner_agent, 30)
        )

    def test_portable_scanner_uses_rechargeable_battery_at_physical_station(self):
        scanner_agent = next(
            agent for agent in self.crew
            if agent.inventory.has_item("portable_scanner")
        )
        scanner_agent.x = self.engine.lz_x
        scanner_agent.y = self.engine.lz_y
        scanner_agent._in_habitat = True
        scanner_agent.needs.temperature_stress = 50.0
        scanner_agent.needs.energy = 100.0
        target = self.engine._next_portable_scan_station(scanner_agent)
        self.engine.remote_resource_requests.add("chalcopyrite_ore")
        self.engine._resource_burial_depth = lambda *_args: 1
        original_get_biome = self.engine.world.get_biome

        def scanner_biome(x, y):
            biome = dict(original_get_biome(x, y))
            biome["resources"] = (
                {"chalcopyrite_ore": "trace"} if (x, y) == target else {}
            )
            return biome

        self.engine.world.get_biome = scanner_biome

        for _ in range(20):
            self.engine.current_tick += 1
            self.engine._process_agent_tick(scanner_agent, [], nearby_count=1)
            if target in self.engine.discovered_resources.get("chalcopyrite_ore", set()):
                break

        self.assertIn(target, self.engine.discovered_resources["chalcopyrite_ore"])
        self.assertEqual(
            92.0, scanner_agent.inventory.tool_charge_pct["portable_scanner"]
        )
        self.assertEqual(499, scanner_agent.inventory.tool_durability["portable_scanner"])
        self.assertNotIn("chalcopyrite_ore", self.engine.remote_resource_requests)

    def test_one_scan_measures_all_open_bom_resources_with_one_charge(self):
        scanner_agent = next(
            agent for agent in self.crew
            if agent.inventory.has_item("portable_scanner")
        )
        station = self.engine._next_portable_scan_station(scanner_agent)
        scanner_agent.x, scanner_agent.y = station
        scanner_agent._in_habitat = False
        scanner_agent.action.clear()
        scanner_agent.needs.temperature_stress = 50.0
        scanner_agent.inventory.tool_charge_pct["portable_scanner"] = 100.0
        scanner_agent.inventory.tool_durability["portable_scanner"] = 500
        self.engine._resource_burial_depth = lambda *_args: 1
        original_get_biome = self.engine.world.get_biome

        def scanner_biome(x, y):
            biome = dict(original_get_biome(x, y))
            # One detector pass covers the physical 3x3 footprint, but it
            # still sees only the immediately underlying layer in each cell.
            # Put the two objectives in neighbouring cells rather than
            # unrealistically seeing through two stacked ore layers at once.
            resources = {}
            if (x, y) == station:
                resources = {"chalcopyrite_ore": "trace"}
            elif (x, y) == (station[0] + 1, station[1]):
                resources = {"silica_sand": "trace"}
            biome["resources"] = resources
            return biome

        self.engine.world.get_biome = scanner_biome
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "survey_resources",
            "target": {
                "resource": "chalcopyrite_ore",
                "survey_resources": [
                    "chalcopyrite_ore", "silica_sand", "iron_ore"
                ],
                "x": station[0],
                "y": station[1],
                "survey_action": "portable_scanner",
            },
            "reasoning": "one physical multi-objective detector pass",
        }

        self.engine._process_agent_tick(scanner_agent, [], nearby_count=1)

        self.assertIn(station, self.engine.discovered_resources["chalcopyrite_ore"])
        self.assertIn(
            (station[0] + 1, station[1]),
            self.engine.discovered_resources["silica_sand"],
        )
        self.assertIn(
            station, self.engine._portable_resource_miss_centers["iron_ore"]
        )
        self.assertEqual(92.0, scanner_agent.inventory.tool_charge_pct[
            "portable_scanner"
        ])
        self.assertEqual(499, scanner_agent.inventory.tool_durability[
            "portable_scanner"
        ])
        self.assertEqual(
            ["chalcopyrite_ore", "silica_sand", "iron_ore"],
            scanner_agent.last_decision["target"]["survey_resources"],
        )

    def test_distant_detector_hit_creates_returning_recovery_contract(self):
        scanner_agent = next(
            agent for agent in self.crew
            if agent.inventory.has_item("portable_scanner")
        )
        target = self.engine._next_portable_scan_station(scanner_agent)
        self.assertGreaterEqual(
            self.engine._distance_from_lz(*target),
            self.engine.CREW_ROVER_MIN_DISTANCE_CELLS,
        )
        scanner_agent.x, scanner_agent.y = target
        scanner_agent._in_habitat = False
        scanner_agent.needs.temperature_stress = 50.0
        scanner_agent.action.clear()
        self.engine.remote_resource_requests.add("sulfur")
        self.engine._resource_burial_depth = lambda *_args: 1
        original_get_biome = self.engine.world.get_biome

        def scanner_biome(x, y):
            biome = dict(original_get_biome(x, y))
            biome["resources"] = (
                {"sulfur": "trace"} if (x, y) == target else {}
            )
            return biome

        self.engine.world.get_biome = scanner_biome
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "survey_resources",
            "target": {
                "resource": "sulfur",
                "x": target[0],
                "y": target[1],
                "survey_action": "portable_scanner",
            },
            "reasoning": "measure the sulfur anomaly",
        }

        self.engine._process_agent_tick(scanner_agent, [], nearby_count=1)

        self.assertIn(target, self.engine.discovered_resources["sulfur"])
        contract = scanner_agent._detected_resource_recovery
        self.assertEqual(target, (contract["x"], contract["y"]))
        self.assertEqual("sulfur", contract["resource"])
        self.assertIsNone(getattr(scanner_agent, "_active_excavation", None))
        self.assertEqual("move", scanner_agent.action.action_type)
        self.assertEqual("shelter", scanner_agent.action.target["destination"])
        self.assertTrue(scanner_agent.action.target["detector_recovery"])

    def test_detector_recovery_at_base_launches_two_person_rover(self):
        lead, buddy = self.crew
        for crew in self.crew:
            crew.x = self.engine.lz_x
            crew.y = self.engine.lz_y
            crew._in_habitat = True
            crew.needs.energy = 100.0
            crew.needs.hunger = 100.0
            crew.needs.thirst = 100.0
            crew.needs.o2_supply = 100.0
            crew.needs.temperature_stress = 50.0
            crew.plss_co2_scrubber_pct = 100.0
            crew.plss_suit_battery_pct = 100.0
            crew.suit_integrity = 1.0
            crew.inventory.items["oxygen_canisters"] = 1
            crew.action.clear()
        target = self._safe_cardinal_target(18)
        self.engine.discovered_resources.setdefault("sulfur", set()).add(target)
        lead._detected_resource_recovery = {
            "x": target[0],
            "y": target[1],
            "resource": "sulfur",
            "recovery_goal_units": 8,
        }

        self.engine._process_agent_tick(lead, [], nearby_count=1)

        expedition = lead._active_expedition
        self.assertIsInstance(expedition, dict)
        self.assertEqual("resource_recovery", expedition["kind"])
        self.assertEqual("crew_rover", expedition["transport"])
        self.assertEqual(buddy.id, expedition["buddy_id"])
        self.assertEqual(8, expedition["recovery_goal_units"])
        self.assertEqual("move", lead.action.action_type)
        self.assertTrue(lead.action.target["expedition"])
        self.assertEqual("crew_rover", lead.action.target["transport"])
        self.assertEqual(target, (
            lead.action.target["x"], lead.action.target["y"]
        ))
        self.assertIsNotNone(
            self.engine.surface_fleet.crew_rover_for_expedition(
                expedition["id"]
            )
        )

    def test_busy_buddy_is_reserved_instead_of_starving_rover_dispatch(self):
        lead, buddy = self.crew
        for crew in self.crew:
            crew.x, crew.y = self.engine.lz_x, self.engine.lz_y
            crew._in_habitat = True
            crew.needs.energy = 100.0
            crew.needs.hunger = 100.0
            crew.needs.thirst = 100.0
            crew.needs.o2_supply = 100.0
            crew.needs.temperature_stress = 50.0
            crew.plss_co2_scrubber_pct = 100.0
            crew.plss_suit_battery_pct = 100.0
            crew.suit_integrity = 1.0
            crew.inventory.items["oxygen_canisters"] = 1
            crew.action.clear()
        target = self._safe_cardinal_target(18)
        self.engine.discovered_resources.setdefault("sulfur", set()).add(target)
        lead._detected_resource_recovery = {
            "x": target[0], "y": target[1], "resource": "sulfur",
            "recovery_goal_units": 8,
        }
        buddy.action.action_type = "move"
        buddy.action.target = {
            "x": self.engine.lz_x + 1,
            "y": self.engine.lz_y,
            "destination": "ordinary_job",
        }
        buddy.action.ticks_remaining = 1

        self.engine._process_agent_tick(lead, [], nearby_count=1)

        self.assertIsNone(getattr(lead, "_active_expedition", None))
        self.assertEqual(
            lead.id, buddy._expedition_buddy_reservation["lead_id"]
        )
        self.assertTrue(lead.action.target["waiting_for_crew_readiness"])

        # On the buddy's own turn the persistent assignment wins over a new
        # ordinary route, but does not interrupt physiological self-care.
        decision = self.engine.decision_engine.process_tick(
            buddy,
            tick=self.engine.current_tick + 1,
            tick_events={},
            world_context={
                "active_events": [],
                "effective_temperature_c": 21.0,
                "colony_resources": self.engine._colony_resources,
            },
            nearby_agents=[],
        )
        self.assertEqual("stand_watch", decision["action"])
        self.assertTrue(decision["target"]["expedition_buddy_standby"])

        buddy.action.action_type = "stand_watch"
        buddy.action.target = dict(decision["target"])
        buddy.action.ticks_remaining = 1
        self.engine._process_agent_tick(lead, [], nearby_count=1)

        self.assertIsInstance(lead._active_expedition, dict)
        self.assertEqual("crew_rover", lead._active_expedition["transport"])
        self.assertEqual(buddy.id, lead._active_expedition["buddy_id"])

    def test_radius_25_detector_face_is_authorized_only_by_rover(self):
        lead, _buddy = self.crew
        for crew in self.crew:
            crew.x, crew.y = self.engine.lz_x, self.engine.lz_y
            crew._in_habitat = True
            crew.needs.energy = 100.0
            crew.needs.hunger = 100.0
            crew.needs.thirst = 100.0
            crew.needs.o2_supply = 100.0
            crew.needs.temperature_stress = 50.0
            crew.plss_co2_scrubber_pct = 100.0
            crew.plss_suit_battery_pct = 100.0
            crew.suit_integrity = 1.0
            crew.inventory.items["oxygen_canisters"] = 1
            crew.action.clear()
        target = self._safe_cardinal_target(25)
        self.assertFalse(self.engine._inside_eva_operating_area(*target))
        self.engine.discovered_resources.setdefault("sulfur", set()).add(target)
        lead._detected_resource_recovery = {
            "x": target[0], "y": target[1], "resource": "sulfur",
            "recovery_goal_units": 6,
        }

        self.engine._process_agent_tick(lead, [], nearby_count=1)

        expedition = lead._active_expedition
        self.assertEqual("crew_rover", expedition["transport"])
        self.assertEqual(target, (
            expedition["target_x"], expedition["target_y"]
        ))
        self.assertTrue(lead.action.target["expedition"])

    def test_recovery_preflight_hazard_does_not_reserve_rover(self):
        lead, _buddy = self.crew
        for crew in self.crew:
            crew.x, crew.y = self.engine.lz_x, self.engine.lz_y
            crew._in_habitat = True
            crew.needs.energy = 100.0
            crew.needs.hunger = 100.0
            crew.needs.thirst = 100.0
            crew.needs.o2_supply = 100.0
            crew.needs.temperature_stress = 50.0
            crew.plss_co2_scrubber_pct = 100.0
            crew.plss_suit_battery_pct = 100.0
            crew.suit_integrity = 1.0
            crew.inventory.items["oxygen_canisters"] = 1
            crew.action.clear()
        target = self._safe_cardinal_target(18)
        self.engine.discovered_resources.setdefault("sulfur", set()).add(target)
        lead._detected_resource_recovery = {
            "x": target[0], "y": target[1], "resource": "sulfur",
            "recovery_goal_units": 6,
        }
        self.engine._active_events = [{"type": "dust_storm"}]

        self.engine._process_agent_tick(lead, [], nearby_count=1)

        self.assertIsNone(getattr(lead, "_active_expedition", None))
        self.assertEqual("idle", self.engine.surface_fleet.crew_rovers[0].state)
        self.assertEqual(
            "active_surface_hazard", lead.action.target["eva_denied"]
        )

    def test_returning_rover_ignores_remote_habitat_and_drives_to_lz(self):
        lead, _buddy = self.crew
        for crew in self.crew:
            crew.x, crew.y = self.engine.lz_x, self.engine.lz_y
            crew._in_habitat = True
            crew.needs.energy = 100.0
            crew.needs.hunger = 100.0
            crew.needs.thirst = 100.0
            crew.needs.o2_supply = 100.0
            crew.needs.temperature_stress = 50.0
            crew.plss_co2_scrubber_pct = 100.0
            crew.plss_suit_battery_pct = 100.0
            crew.suit_integrity = 1.0
            crew.inventory.items["oxygen_canisters"] = 1
            crew.action.clear()
        target = (self.engine.lz_x + 18, self.engine.lz_y)
        self.engine.discovered_resources.setdefault("sulfur", set()).add(target)
        expedition = self.engine._start_expedition(
            lead, "sulfur", target, require_rover=True
        )
        self.assertIsNotNone(expedition)
        lead.x, lead.y = target
        lead._in_habitat = False
        expedition["status"] = "returning"
        self.engine.placed_structures.append({
            "id": "remote-habitat",
            "type": "habitat_module",
            "x": target[0] - 1,
            "y": target[1],
            "under_construction": False,
            "destroyed": False,
            "health": 1.0,
        })

        self.engine._process_agent_tick(lead, [], nearby_count=1)

        self.assertLess(
            self.engine._distance_from_lz(lead.x, lead.y), 18
        )
        self.assertGreater(
            self.engine._distance_from_lz(lead.x, lead.y), 1
        )
        self.assertIsInstance(lead._active_expedition, dict)
        self.assertEqual("in_use", self.engine.surface_fleet.crew_rovers[0].state)

    def test_returning_expedition_state_is_cleared_at_landing_hub(self):
        agent = self.crew[0]
        agent.x, agent.y = self.engine.lz_x, self.engine.lz_y
        agent._in_habitat = True
        agent.needs.energy = 82.0
        agent.needs.hunger = 100.0
        agent.needs.thirst = 100.0
        agent.needs.o2_supply = 100.0
        agent.needs.temperature_stress = 50.0
        agent._active_expedition = {
            "id": "stale-return",
            "status": "returning",
            "role": "buddy",
            "transport": "on_foot",
            "resource": "chalcopyrite_ore",
            "authorized_radius": 40,
        }
        agent.action.action_type = "move"
        agent.action.target = {
            "x": self.engine.lz_x,
            "y": self.engine.lz_y,
            "destination": "shelter",
            "expedition": True,
        }
        agent.action.ticks_remaining = 1
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "rest", "target": {}, "reasoning": "post-trip recovery"
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertIsNone(agent._active_expedition)
        self.assertIsNone(agent._pending_expedition)
        self.assertNotEqual("move", agent.action.action_type)

    def test_rover_recovery_returns_when_recipe_goal_is_loaded(self):
        lead, _buddy = self.crew
        for crew in self.crew:
            crew.x = self.engine.lz_x
            crew.y = self.engine.lz_y
            crew._in_habitat = True
            crew.needs.energy = 100.0
            crew.needs.hunger = 100.0
            crew.needs.thirst = 100.0
            crew.needs.o2_supply = 100.0
            crew.needs.temperature_stress = 50.0
            crew.plss_co2_scrubber_pct = 100.0
            crew.plss_suit_battery_pct = 100.0
            crew.suit_integrity = 1.0
            crew.inventory.items["oxygen_canisters"] = 1
            crew.action.clear()
        target = self._safe_cardinal_target(18)
        self.engine.discovered_resources.setdefault("sulfur", set()).add(target)
        expedition = self.engine._start_expedition(
            lead,
            "sulfur",
            target,
            require_rover=True,
            expedition_kind="resource_recovery",
            recovery_goal_units=2,
        )
        self.assertIsNotNone(expedition)
        lead._detected_resource_recovery = {
            "x": target[0], "y": target[1], "resource": "sulfur",
            "recovery_goal_units": 2,
        }
        lead.x, lead.y = target
        lead._in_habitat = False
        expedition["status"] = "working"
        lead.action.clear()
        self.engine.cell_geology[target] = {
            "layers": [{
                "material": "sulfur",
                "initial_quantity": 100,
                "remaining": 100,
            }],
            "current_index": 0,
        }
        self.engine.cell_resource_units[(*target, "sulfur")] = 100
        original_get_cell_info = self.engine.world.get_cell_info

        def sulfur_cell(x, y, tick=0):
            info = dict(original_get_cell_info(x, y, tick))
            if (x, y) == target:
                info["base_resources"] = {"sulfur": "trace"}
            return info

        self.engine.world.get_cell_info = sulfur_cell
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "gather",
            "target": {"resource": "sulfur", "expedition": True},
            "reasoning": "load only the scheduled sulfur requirement",
        }

        self.engine._process_agent_tick(lead, [], nearby_count=1)

        rover = self.engine.surface_fleet.crew_rover_for_expedition(
            expedition["id"]
        )
        self.assertGreaterEqual(rover.payload["sulfur"], 2)
        self.assertLess(rover.payload_mass_kg, 490.0)
        self.assertGreater(
            self.engine.cell_geology[target]["layers"][0]["remaining"], 0
        )
        self.assertEqual("returning", expedition["status"])
        self.assertEqual("move", lead.action.action_type)
        self.assertEqual("shelter", lead.action.target["destination"])
        self.assertTrue(
            self.engine._complete_crew_rover_expedition(expedition)
        )
        self.assertIsNone(lead._detected_resource_recovery)

    def test_detector_contract_strips_overburden_and_returns_sulfur_by_rover(self):
        lead, buddy = self.crew
        for crew in self.crew:
            crew.x, crew.y = self.engine.lz_x, self.engine.lz_y
            crew._in_habitat = True
            crew.needs.energy = 100.0
            crew.needs.hunger = 100.0
            crew.needs.thirst = 100.0
            crew.needs.o2_supply = 100.0
            crew.needs.temperature_stress = 50.0
            crew.plss_co2_scrubber_pct = 100.0
            crew.plss_suit_battery_pct = 100.0
            crew.suit_integrity = 1.0
            crew.inventory.items["oxygen_canisters"] = 1
            crew.action.clear()

        target = self._safe_cardinal_target(18)
        self.engine.discovered_resources.setdefault("regolith", set()).add(target)
        self.engine.discovered_resources.setdefault("sulfur", set()).add(target)
        self.engine.cell_geology[target] = {
            "layers": [
                {"material": "regolith", "initial_quantity": 4, "remaining": 4},
                {"material": "sulfur", "initial_quantity": 20, "remaining": 20},
            ],
            "current_index": 0,
        }
        self.engine.cell_resource_units[(*target, "regolith")] = 4
        self.engine.cell_resource_units[(*target, "sulfur")] = 20
        lead._detected_resource_recovery = {
            "x": target[0],
            "y": target[1],
            "resource": "sulfur",
            "recovery_goal_units": 2,
        }
        original_get_cell_info = self.engine.world.get_cell_info

        def sulfur_cell(x, y, tick=0):
            info = dict(original_get_cell_info(x, y, tick))
            if (x, y) == target:
                info["base_resources"] = {
                    "regolith": "moderate", "sulfur": "trace"
                }
            return info

        self.engine.world.get_cell_info = sulfur_cell

        def recovery_policy(*, agent, **_kwargs):
            expedition = getattr(agent, "_active_expedition", None)
            if isinstance(expedition, dict):
                if expedition.get("status") == "returning":
                    return {
                        "action": "move",
                        "target": {
                            "x": self.engine.lz_x,
                            "y": self.engine.lz_y,
                            "destination": "shelter",
                            "expedition": True,
                        },
                        "reasoning": "return rover and conserved payload",
                    }
                if expedition.get("status") == "working":
                    if expedition.get("role") == "buddy":
                        return {
                            "action": "stand_watch",
                            "target": {"field": True, "expedition": True},
                            "reasoning": "buddy watch",
                        }
                    return {
                        "action": "gather",
                        "target": {
                            "resource": "sulfur",
                            "expedition": True,
                            "recovery_goal_units": 2,
                        },
                        "reasoning": "strip then recover sulfur",
                    }
                return {
                    "action": "move",
                    "target": {
                        "x": target[0],
                        "y": target[1],
                        "resource": "sulfur",
                        "destination": "extraction_face",
                        "expedition": True,
                        "transport": "crew_rover",
                    },
                    "reasoning": "continue rover outbound leg",
                }
            if agent is lead and isinstance(
                getattr(lead, "_detected_resource_recovery", None), dict
            ):
                return {
                    "action": "gather",
                    "target": {
                        **lead._detected_resource_recovery,
                        "detector_recovery": True,
                    },
                    "reasoning": "launch confirmed sulfur recovery",
                }
            return {
                "action": "rest", "target": {}, "reasoning": "stand by"
            }

        self.engine.decision_engine.process_tick = recovery_policy

        for _ in range(80):
            self.engine.current_tick += 1
            self.engine._process_agent_tick(lead, [], nearby_count=1)
            self.engine._process_agent_tick(buddy, [], nearby_count=1)
            if (
                lead._detected_resource_recovery is None
                and self.engine.central_depot_inventory.get("sulfur", 0) >= 2
            ):
                break

        self.assertEqual(0, self.engine.cell_geology[target]["layers"][0]["remaining"])
        self.assertLess(
            self.engine.cell_geology[target]["layers"][1]["remaining"], 20
        )
        self.assertGreaterEqual(
            self.engine.central_depot_inventory.get("sulfur", 0), 2
        )
        self.assertIsNone(lead._detected_resource_recovery)
        self.assertEqual("charging", self.engine.surface_fleet.crew_rovers[0].state)

    def test_exposing_requested_layer_keeps_work_face_active(self):
        agent = self.crew[0]
        target = self._safe_cardinal_target(
            self.engine.MIN_EXTRACTION_RADIUS_CELLS
        )
        agent.x, agent.y = target
        agent._in_habitat = False
        agent.needs.energy = 100.0
        agent.needs.hunger = 100.0
        agent.needs.thirst = 100.0
        agent.needs.o2_supply = 100.0
        agent.needs.temperature_stress = 50.0
        agent._active_excavation = {
            "x": target[0], "y": target[1], "resource": "sulfur",
            "capacity_recipe": "water_collector",
        }
        self.engine.cell_geology[target] = {
            "layers": [
                {"material": "regolith", "initial_quantity": 2, "remaining": 2},
                {"material": "sulfur", "initial_quantity": 20, "remaining": 20},
            ],
            "current_index": 0,
        }
        self.engine.discovered_resources.setdefault("sulfur", set()).add(target)
        original_get_cell_info = self.engine.world.get_cell_info

        def sulfur_cell(x, y, tick=0):
            info = dict(original_get_cell_info(x, y, tick))
            if (x, y) == target:
                info["base_resources"] = {
                    "regolith": "moderate", "sulfur": "trace"
                }
            return info

        self.engine.world.get_cell_info = sulfur_cell
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "gather",
            "target": {
                "resource": "sulfur",
                "capacity_recipe": "water_collector",
            },
            "reasoning": "clear the final overburden",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual(
            "sulfur", self.engine._current_geology_layer(*target)["material"]
        )
        self.assertIsInstance(agent._active_excavation, dict)
        self.assertEqual("sulfur", agent._active_excavation["resource"])

    def test_scanner_recharges_from_real_colony_energy_only_at_base(self):
        scanner_agent = next(
            agent for agent in self.crew
            if agent.inventory.has_item("portable_scanner")
        )
        scanner_agent.x = self.engine.lz_x
        scanner_agent.y = self.engine.lz_y
        scanner_agent._in_habitat = True
        scanner_agent.inventory.tool_charge_pct["portable_scanner"] = 0.0
        self.engine._colony_resources["energy_stored_kwh"] = 1.0
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "recharge_scanner",
            "target": {"resource": "chalcopyrite_ore"},
            "reasoning": "battery below field reserve",
        }

        self.engine._process_agent_tick(scanner_agent, [], nearby_count=1)

        self.assertEqual("recharge_scanner", scanner_agent.action.action_type)
        self.assertAlmostEqual(
            25.0, scanner_agent.inventory.tool_charge_pct["portable_scanner"],
            places=4,
        )
        self.assertLess(self.engine._colony_resources["energy_stored_kwh"], 1.0)
        self.assertGreater(scanner_agent.action.target["energy_draw_kwh"], 0.0)

    def test_scanner_rescans_after_opening_shallow_test_pit(self):
        scanner_agent = next(
            agent for agent in self.crew
            if agent.inventory.has_item("portable_scanner")
        )
        scanner_agent.x = self.engine.lz_x
        scanner_agent.y = self.engine.lz_y
        scanner_agent._in_habitat = True
        scanner_agent.needs.energy = 100.0
        scanner_agent.needs.hunger = 100.0
        scanner_agent.needs.thirst = 100.0
        scanner_agent.needs.o2_supply = 100.0
        scanner_agent.needs.temperature_stress = 50.0
        target = self.engine._next_portable_scan_station(scanner_agent)
        self.engine.remote_resource_requests.add("chalcopyrite_ore")
        self.engine._resource_burial_depth = lambda _x, _y, resource: {
            "basalt": 1,
            "iron_ore": 2,
            "chalcopyrite_ore": 3,
        }.get(resource, 0)
        original_get_biome = self.engine.world.get_biome

        def layered_scanner_biome(x, y):
            biome = dict(original_get_biome(x, y))
            biome["resources"] = (
                {
                    "regolith": "moderate",
                    "basalt": "moderate",
                    "iron_ore": "moderate",
                    "chalcopyrite_ore": "trace",
                }
                if (x, y) == target else {}
            )
            return biome

        self.engine.world.get_biome = layered_scanner_biome

        # A conservative crew member may return to the lander for one real
        # fatigue/self-care cycle between the second pit cut and final scan.
        # The survey contract must survive that interruption and resume; do
        # not make this test depend on completing the EVA in one shift.
        for _ in range(180):
            self.engine._process_agent_tick(scanner_agent, [], nearby_count=1)
            self.engine.current_tick += 1
            if target in self.engine.discovered_resources.get("chalcopyrite_ore", set()):
                break

        self.assertIn(
            target,
            self.engine.discovered_resources.get("chalcopyrite_ore", set()),
        )
        self.assertEqual(2, self.engine.cell_excavation_depth[target])
        self.assertLess(
            scanner_agent.inventory.tool_charge_pct["portable_scanner"], 100.0
        )

    def test_mining_exposes_one_complete_layer_and_conserves_overburden(self):
        agent = self.crew[0]
        target = self._safe_cardinal_target(
            self.engine.MIN_EXTRACTION_RADIUS_CELLS
        )
        agent.x, agent.y = target
        agent._in_habitat = False
        agent.needs.energy = 100.0
        agent.needs.hunger = 100.0
        agent.needs.thirst = 100.0
        agent.needs.o2_supply = 100.0
        agent.needs.temperature_stress = 50.0
        original_get_cell_info = self.engine.world.get_cell_info

        def layered_cell(x, y, tick=0):
            info = dict(original_get_cell_info(x, y, tick))
            if (x, y) == target:
                info["base_resources"] = {
                    "regolith": "moderate",
                    "iron_ore": "rich",
                }
            return info

        self.engine.world.get_cell_info = layered_cell
        self.engine._get_initial_cell_resource_capacity = lambda *_args: 4
        self.engine.discovered_resources.setdefault("iron_ore", set()).add(target)
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "gather",
            "target": {"resource": "iron_ore"},
            "reasoning": "clear overburden to the detected iron layer",
        }

        for _ in range(4):
            agent.action.ticks_remaining = 0
            self.engine._process_agent_tick(agent, [], nearby_count=1)
            if self.engine._current_geology_layer(*target)["material"] == "iron_ore":
                break

        current = self.engine._current_geology_layer(*target)
        self.assertEqual("iron_ore", current["material"])
        spoil_coord = next(iter(self.engine.spoil_piles))
        self.assertNotEqual(target, spoil_coord)
        self.assertEqual(1, max(
            abs(spoil_coord[0] - target[0]),
            abs(spoil_coord[1] - target[1]),
        ))
        self.assertEqual(4, self.engine.spoil_piles[spoil_coord]["regolith"])
        self.assertEqual(0, agent.inventory.materials.get("iron_ore", 0))
        snapshot = self.engine._get_state_snapshot()
        visible_at_target = {
            resource
            for resource, coords in snapshot["discovered_resources"].items()
            if any((entry[0], entry[1]) == target for entry in coords)
        }
        self.assertEqual({"iron_ore"}, visible_at_target)
        self.assertEqual(4, snapshot["spoil_piles"][0]["total_units"])
        self.assertEqual(["regolith"], snapshot["spoil_piles"][0]["stack_order"])

        agent.action.ticks_remaining = 0
        self.engine._process_agent_tick(agent, [], nearby_count=1)
        self.assertGreater(agent.inventory.materials.get("iron_ore", 0), 0)

    def test_scan_shows_current_quantity_and_exactly_one_next_layer(self):
        target = (
            self.engine.lz_x + self.engine.MIN_EXTRACTION_RADIUS_CELLS,
            self.engine.lz_y,
        )
        original_get_cell_info = self.engine.world.get_cell_info

        def layered_cell(x, y, tick=0):
            info = dict(original_get_cell_info(x, y, tick))
            if (x, y) == target:
                info["base_resources"] = {
                    "regolith": "moderate",
                    "iron_ore": "rich",
                    "chalcopyrite_ore": "trace",
                }
            return info

        self.engine.world.get_cell_info = layered_cell
        self.engine._resource_burial_depth = lambda _x, _y, resource: {
            "iron_ore": 1,
            "chalcopyrite_ore": 2,
        }.get(resource, 0)
        self.engine._get_initial_cell_resource_capacity = lambda *_args: 7

        self.engine._reveal_detectable_resources(
            *target, detector_penetration_layers=1
        )
        snapshot = self.engine._get_state_snapshot()
        column = next(
            item for item in snapshot["cell_geology"]
            if (item["x"], item["y"]) == target
        )
        self.assertEqual("regolith", column["current_layer"]["material"])
        self.assertEqual(7, column["current_layer"]["remaining"])
        self.assertEqual("iron_ore", column["detected_next_layer"]["material"])
        self.assertEqual(1, column["detected_below_count"])
        self.assertNotIn(target, self.engine.discovered_resources.get("chalcopyrite_ore", set()))

        self.engine._advance_depleted_geology_layer(*target)
        snapshot = self.engine._get_state_snapshot()
        column = next(
            item for item in snapshot["cell_geology"]
            if (item["x"], item["y"]) == target
        )
        self.assertEqual("iron_ore", column["current_layer"]["material"])
        self.assertIsNone(column["detected_next_layer"])

        self.engine._reveal_detectable_resources(
            *target, detector_penetration_layers=1
        )
        snapshot = self.engine._get_state_snapshot()
        column = next(
            item for item in snapshot["cell_geology"]
            if (item["x"], item["y"]) == target
        )
        self.assertEqual("chalcopyrite_ore", column["detected_next_layer"]["material"])

    def test_neighbor_spoil_stack_keeps_established_order_when_quantity_changes(self):
        excavation = self._safe_cardinal_target(
            self.engine.MIN_EXTRACTION_RADIUS_CELLS
        )

        first = self.engine._add_to_spoil_pile(*excavation, "regolith", 5)
        second = self.engine._add_to_spoil_pile(*excavation, "iron_ore", 3)
        third = self.engine._add_to_spoil_pile(*excavation, "sulfur", 2)

        self.assertEqual(first, second)
        self.assertEqual(second, third)
        self.assertNotEqual(excavation, first)
        snapshot_pile = next(
            pile for pile in self.engine._get_state_snapshot()["spoil_piles"]
            if (pile["x"], pile["y"]) == first
        )
        self.assertEqual(
            ["regolith", "iron_ore", "sulfur"],
            snapshot_pile["stack_order"],
        )

        self.engine._add_to_spoil_pile(*excavation, "iron_ore", 1)
        snapshot_pile = next(
            pile for pile in self.engine._get_state_snapshot()["spoil_piles"]
            if (pile["x"], pile["y"]) == first
        )
        self.assertEqual(
            ["regolith", "iron_ore", "sulfur"],
            snapshot_pile["stack_order"],
        )
        self.assertEqual(4, snapshot_pile["materials"]["iron_ore"])

    def test_agent_at_requested_spoil_pile_collects_instead_of_stationary_move(self):
        agent = self.crew[0]
        excavation = self._safe_cardinal_target(
            self.engine.MIN_EXTRACTION_RADIUS_CELLS
        )
        pile_coord = self.engine._add_to_spoil_pile(
            *excavation, "graphite", 5
        )
        agent.x, agent.y = pile_coord
        agent._in_habitat = False
        agent.needs.energy = 100.0
        agent.needs.hunger = 100.0
        agent.needs.thirst = 100.0
        agent.needs.o2_supply = 100.0
        agent.needs.temperature_stress = 50.0
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "gather",
            "target": {"resource": "graphite"},
            "reasoning": "collect the designated surface spoil stack",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("gather", agent.action.action_type)
        self.assertGreater(agent.inventory.materials.get("graphite", 0), 0)

    def test_hand_prospecting_walks_to_one_reserved_test_station(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.needs.energy = 100.0
        agent.needs.hunger = 100.0
        agent.needs.thirst = 100.0
        agent.needs.o2_supply = 100.0
        agent.needs.temperature_stress = 50.0
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "gather",
            "target": {"resource": "chalcopyrite_ore"},
            "reasoning": "conduct a real geological test pit",
        }

        actions = []
        for _ in range(20):
            self.engine.current_tick += 1
            self.engine._process_agent_tick(agent, [], nearby_count=1)
            actions.append(agent.action.action_type)
            if agent.action.action_type == "prospect":
                break

        self.assertIn("prospect", actions)
        first_prospect = actions.index("prospect")
        self.assertGreaterEqual(first_prospect, 5)
        self.assertTrue(all(action == "move" for action in actions[:first_prospect]))
        self.assertEqual(1, len(self.engine._hand_prospect_centers))
        self.assertGreaterEqual(
            self.engine._distance_from_lz(agent.x, agent.y),
            self.engine.MIN_EXTRACTION_RADIUS_CELLS,
        )

    def test_absent_planet_resource_is_not_prospected_forever(self):
        agent = self.crew[0]
        self.assertNotIn("water_ice", self.engine.planetary_resources)
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "gather",
            "target": {"resource": "water_ice"},
            "reasoning": "invalid search requested by a stale plan",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("idle", agent.action.action_type)
        self.assertTrue(agent.action.target["resource_unavailable_on_planet"])
        self.assertNotIn("water_ice", self.engine.remote_resource_requests)

    def test_partial_overburden_creates_excavation_commitment(self):
        agent = self.crew[0]
        target = self._safe_cardinal_target(
            self.engine.MIN_EXTRACTION_RADIUS_CELLS
        )
        agent.x, agent.y = target
        agent._in_habitat = False
        agent.needs.energy = 100.0
        agent.needs.hunger = 100.0
        agent.needs.thirst = 100.0
        agent.needs.o2_supply = 100.0
        agent.needs.temperature_stress = 50.0
        original_get_cell_info = self.engine.world.get_cell_info

        def layered_cell(x, y, tick=0):
            info = dict(original_get_cell_info(x, y, tick))
            if (x, y) == target:
                info["base_resources"] = {
                    "regolith": "moderate",
                    "iron_ore": "rich",
                }
            return info

        self.engine.world.get_cell_info = layered_cell
        self.engine._get_initial_cell_resource_capacity = lambda *_args: 20
        self.engine.discovered_resources.setdefault("iron_ore", set()).add(target)
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "gather",
            "target": {"resource": "iron_ore"},
            "reasoning": "start a real production face",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual(
            {"x": target[0], "y": target[1], "resource": "iron_ore"},
            agent._active_excavation,
        )
        self.assertGreater(
            self.engine._current_geology_layer(*target)["remaining"], 0
        )

    def test_arrival_preserves_extraction_mission_and_starts_work(self):
        agent = self.crew[0]
        agent._in_habitat = False
        agent.x = self.engine.lz_x + self.engine.MIN_EXTRACTION_RADIUS_CELLS
        agent.y = self.engine.lz_y
        agent.needs.energy = 100.0
        agent.needs.hunger = 100.0
        agent.needs.thirst = 100.0
        agent.needs.o2_supply = 100.0
        agent.needs.temperature_stress = 50.0
        target = (agent.x + 1, agent.y)
        agent.action.action_type = "move"
        agent.action.target = {
            "x": target[0],
            "y": target[1],
            "resource": "iron_ore",
            "destination": "extraction_face",
            "mission_action": "gather",
        }
        agent.action.ticks_remaining = 0

        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "move",
            "target": dict(agent.action.target),
            "reasoning": "travel to assigned iron face",
        }
        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("arrived", agent.action.action_type)
        self.assertEqual("iron_ore", agent.action.target["resource"])
        self.assertEqual("gather", agent.action.target["mission_action"])

        self.engine.decision_engine.process_tick = (
            self.engine.decision_engine.__class__.process_tick.__get__(
                self.engine.decision_engine,
                self.engine.decision_engine.__class__,
            )
        )
        decision = self.engine.decision_engine._process_tick_internal(
            agent,
            self.engine.current_tick,
            {},
            {"colony_resources": dict(self.engine._colony_resources)},
            [],
        )
        self.assertEqual("gather", decision["action"])
        self.assertEqual("iron_ore", decision["target"]["resource"])

    def test_two_miners_reserve_different_known_faces(self):
        first, second = self.crew
        for agent in (first, second):
            agent.x = self.engine.lz_x
            agent.y = self.engine.lz_y
            agent._in_habitat = False
        faces = {
            self._safe_cardinal_target(
                self.engine.MIN_EXTRACTION_RADIUS_CELLS, ordinal=0
            ),
            self._safe_cardinal_target(
                self.engine.MIN_EXTRACTION_RADIUS_CELLS, ordinal=1
            ),
        }
        self.engine.discovered_resources["sulfur"] = set(faces)
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "gather",
            "target": {"resource": "sulfur"},
            "reasoning": "parallel sulfur mining",
        }

        self.engine._process_agent_tick(first, [], nearby_count=1)
        self.engine._process_agent_tick(second, [], nearby_count=1)

        first_face = (first.action.target["x"], first.action.target["y"])
        second_face = (second.action.target["x"], second.action.target["y"])
        self.assertIn(first_face, faces)
        self.assertIn(second_face, faces)
        self.assertNotEqual(first_face, second_face)

    def test_idle_outdoors_returns_home_instead_of_exploring(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x + 5
        agent.y = self.engine.lz_y
        agent._in_habitat = False
        agent.needs.energy = 100.0
        agent.needs.temperature_stress = 50.0
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "idle",
            "target": {},
            "reasoning": "no assigned field mission",
        }

        exterior = self.engine._lander_airlock_exterior_position()
        route_before = self.engine._surface_vehicle_route(agent.x, agent.y, *exterior)
        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("move", agent.action.action_type)
        self.assertEqual("shelter", agent.action.target["destination"])
        # The safe path goes around the hull. Distance to the base centre need
        # not decrease on each step, but remaining apron-route length must.
        self.assertLess(
            len(self.engine._surface_vehicle_route(agent.x, agent.y, *exterior)),
            len(route_before),
        )

    def test_unbounded_move_is_rejected_before_airlock_exit(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True

        decision = self.engine.decision_engine._finalize_decision(
            agent,
            {
                "action": "move",
                "target": {"dx": 1, "dy": 0},
                "reasoning": "walk somewhere",
            },
            tick=1,
        )

        self.assertEqual("rest", decision["action"])
        self.assertEqual(
            "missing_absolute_mission_target",
            decision["target"]["route_rejected"],
        )

    def test_fatigue_is_not_a_hidden_airlock_interlock(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x + 1
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.needs.energy = 70.0
        agent.needs.temperature_stress = 50.0
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "gather",
            "target": {"resource": "chalcopyrite_ore"},
            "reasoning": "routine field geology",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertFalse(
            isinstance(agent.action.target, dict)
            and agent.action.target.get("eva_denied")
            == "insufficient_mission_energy_reserve"
        )

    def test_routine_eva_does_not_reopen_airlock_during_surface_hazard(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x + 1
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        self.engine._active_events = [{"type": "dust_storm"}]
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "gather",
            "target": {"resource": "chalcopyrite_ore"},
            "reasoning": "routine geology should wait",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertTrue(agent._in_habitat)
        self.assertEqual("rest", agent.action.action_type)
        self.assertEqual("active_surface_hazard", agent.action.target["eva_denied"])

    def test_two_cell_resource_route_cannot_bypass_suit_preflight(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.suit_condition = 0.49
        deposit = (self.engine.lz_x + 6, self.engine.lz_y)
        self.engine.discovered_resources.setdefault("chalcopyrite_ore", set()).add(
            deposit
        )
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "gather",
            "target": {"resource": "chalcopyrite_ore"},
            "reasoning": "known deposit route",
        }
        start = (agent.x, agent.y)

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual(start, (agent.x, agent.y))
        self.assertTrue(agent._in_habitat)
        self.assertEqual("move", agent.action.action_type)
        self.assertEqual("service_suit", agent.action.target["indoor_activity"])
        self.assertEqual(0.49, agent.suit_condition)

    def test_unreachable_active_excavation_is_released_instead_of_idle_loop(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        unreachable = (
            self.engine.lz_x + self.engine.LOCAL_EVA_RADIUS_CELLS + 1,
            self.engine.lz_y,
        )
        agent._active_excavation = {
            "x": unreachable[0],
            "y": unreachable[1],
            "resource": "chalcopyrite_ore",
        }
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "move",
            "target": {
                "x": unreachable[0],
                "y": unreachable[1],
                "destination": "active_excavation_face",
                "resource": "chalcopyrite_ore",
            },
            "reasoning": "return to unfinished working face",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertIsNone(agent._active_excavation)
        self.assertIn("chalcopyrite_ore", self.engine.remote_resource_requests)
        self.assertEqual("idle", agent.action.action_type)
        self.assertEqual(
            list(unreachable),
            agent.action.target["cancelled_unreachable_target"],
        )

    def test_homeward_or_rejected_move_is_not_productive_shared_work(self):
        planner = self.engine.decision_engine
        planner.agents = self.crew
        worker = self.crew[0]
        recipe = "test_pressure_shell"
        resource = "chalcopyrite_ore"
        contract = {
            "action": "gather",
            "recipe": recipe,
            "stock_key": resource,
            "quantity": 1,
            "assigned_tick": 1,
            "review_tick": 1,
        }
        safety_routes = {
            "forced_return": {
                "destination": "shelter",
                "forced_return": True,
            },
            "fatigue_return": {
                "destination": "habitat",
                "fatigue_return": True,
            },
            "oxygen_return": {
                "destination": "shelter",
                "o2_return": True,
            },
            "dehydration_return": {
                "destination": "shelter",
                "dehydration_return": True,
            },
            "idle_return": {
                "destination": "shelter",
                "idle_return": True,
            },
            "cancelled_target": {
                "cancelled_unreachable_target": [9999, 9999],
            },
            "rejected_route": {
                "route_rejected": "outside_eva_radius",
            },
            "eva_denied": {
                "destination": "habitat",
                "eva_denied": "insufficient_round_trip_energy",
            },
        }

        for route_name, route_target in safety_routes.items():
            with self.subTest(route=route_name):
                worker._shared_work_contract = dict(contract)
                worker._hand_prospect_target = None
                worker._active_excavation = None
                worker._detected_resource_recovery = None
                worker._active_expedition = None
                worker.action.action_type = "move"
                worker.action.target = {
                    "resource": resource,
                    **route_target,
                }
                worker.action.ticks_remaining = 1

                self.assertFalse(
                    planner._shared_contract_has_active_wip(worker, contract)
                )
                self.assertNotIn(
                    resource,
                    planner._active_raw_campaign_claims(recipe, tick=100),
                )

    def test_rejected_plan_execution_does_not_consume_step_time(self):
        planner = self.engine.decision_engine
        planner.current_tick = 100
        agent = self.crew[0]

        for rejection in (
            "craft_blocked",
            "recipe_unavailable",
            "tool_transfer_blocked",
        ):
            with self.subTest(rejection=rejection):
                step = PlanStep(
                    action="build",
                    target={"recipe": "habitat_module"},
                    description="assemble a pressure-rated habitat",
                    estimated_ticks=20,
                    ticks_spent=3,
                )
                plan = StrategicPlan(
                    goal="increase pressurised capacity",
                    steps=[step],
                    created_tick=90,
                )
                planner._plans[agent.id] = plan
                agent.action.action_type = rejection
                agent.action.target = {
                    "recipe": "habitat_module",
                    "reason": "preflight_rejected",
                }
                agent.action.ticks_remaining = 1

                decision = planner._execute_plan_step(agent, plan)

                self.assertIsNone(decision)
                self.assertEqual(3, step.ticks_spent)
                self.assertNotIn(agent.id, planner._plans)
                retry_tick = planner._plan_retry_after_tick[agent.id]
                self.assertGreater(retry_tick, planner.current_tick)
                self.assertLessEqual(retry_tick, planner.current_tick + 6)

    def test_unknown_action_becomes_explicit_rejection_state(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "teleport_without_physics",
            "target": {"x": self.engine.lz_x + 20},
            "reasoning": "invalid stale order",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("invalid_action_rejected", agent.action.action_type)
        self.assertEqual(
            {
                "invalid_action_rejected": True,
                "reason": "unknown_action",
                "requested_action": "teleport_without_physics",
            },
            agent.action.target,
        )
        self.assertEqual(1, agent.action.ticks_remaining)

    def test_critical_field_dehydration_aborts_to_real_base_water(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x + 5
        agent.y = self.engine.lz_y
        agent._in_habitat = False
        agent.needs.thirst = 14.0
        agent.inventory.items.pop("water_packs", None)
        self.engine.central_depot_inventory["water_packs"] = 0
        self.engine._colony_resources["water_reserve_l"] = 20.0
        planner = self.engine.decision_engine
        planner.agents = self.crew
        planner.central_depot_inventory = self.engine.central_depot_inventory
        planner._colony_resources = self.engine._colony_resources
        planner.lz_x = self.engine.lz_x
        planner.lz_y = self.engine.lz_y

        decision = planner.process_tick(
            agent,
            tick=100,
            tick_events={"warnings": ["thirst_critical"]},
            world_context={
                "colony_resources": self.engine._colony_resources,
                "active_events": [],
                "effective_temperature_c": 20.0,
            },
            nearby_agents=[],
        )

        self.assertEqual("move", decision["action"])
        self.assertEqual("shelter", decision["target"]["destination"])
        self.assertTrue(decision["target"]["dehydration_return"])

    def test_airlock_crew_can_drink_from_central_water_reserve(self):
        agent = self.crew[0]
        agent.x, agent.y = self.engine._lander_airlock_position()
        agent._in_habitat = True
        agent.inventory.items.pop("water_packs", None)
        self.engine.central_depot_inventory["water_packs"] = 0
        self.engine._colony_resources["water_reserve_l"] = 5.0
        agent.needs.thirst = 40.0
        self.engine.decision_engine.agents = self.crew
        self.engine.decision_engine.central_depot_inventory = (
            self.engine.central_depot_inventory
        )
        self.engine.decision_engine._colony_resources = self.engine._colony_resources
        self.engine.decision_engine.lz_x = self.engine.lz_x
        self.engine.decision_engine.lz_y = self.engine.lz_y

        decision = self.engine.decision_engine.process_tick(
            agent,
            tick=1,
            tick_events={},
            world_context={
                "effective_temperature_c": 21.0,
                "active_events": [],
                "colony_resources": self.engine._colony_resources,
            },
            nearby_agents=[],
        )

        self.assertEqual("drink", decision["action"])
        self.assertTrue(agent._in_habitat)

    def test_routine_eva_is_denied_until_crew_has_useful_work_reserve(self):
        agent = self.crew[0]
        agent._in_habitat = True
        agent.x, agent.y = self.engine._indoor_activity_position(agent, "sleep")
        agent.needs.energy = 64.0

        self.assertFalse(self.engine._prepare_agent_for_eva(agent))
        self.assertEqual("sleep", agent.action.action_type)
        self.assertEqual(
            "insufficient_round_trip_energy",
            agent.action.target["eva_denied"],
        )

    def test_distance_scaled_eva_denial_schedules_sleep_not_retry_loop(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x + 1
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        target_x = self.engine.lz_x + 15
        # Stay just below this route's budget as the physical walking pace
        # changes; the invariant is recovery, not a historical fixed speed.
        agent.needs.energy = self.engine._routine_eva_start_energy_threshold(
            agent, target_x - agent.x, 15
        ) - 0.5
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "move",
            "target": {
                "x": target_x,
                "y": self.engine.lz_y,
                "destination": "work_site",
            },
            "reasoning": "validated distant field duty",
            "deterministic": True,
        }

        first = self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertNotIn("physiological_safety_interrupt", first)
        self.assertEqual("move", agent.action.action_type)
        self.assertEqual("sleep", agent.action.target["indoor_activity"])
        for _ in range(8):
            self.engine.current_tick += 1
            self.engine._process_agent_tick(agent, [], nearby_count=1)
            if agent.action.action_type == "sleep":
                break
        self.assertEqual("sleep", agent.action.action_type)
        self.assertEqual(
            "insufficient_round_trip_energy",
            agent.action.target["eva_denied"],
        )
        self.assertTrue(agent.action.target["preflight_recovery"])
        remaining = agent.action.ticks_remaining
        energy = agent.needs.energy

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("sleep", agent.action.action_type)
        self.assertEqual(remaining - 1, agent.action.ticks_remaining)
        self.assertGreater(agent.needs.energy, energy)

    def test_routine_eva_start_energy_has_hysteresis_above_return_floor(self):
        agent = self.crew[0]
        start_floor = self.engine.MIN_ROUTINE_EVA_START_ENERGY_PCT
        self.assertGreater(start_floor, 65.0)
        agent._in_habitat = True
        agent.x, agent.y = self.engine._indoor_activity_position(agent, "sleep")
        agent.suit_condition = 1.0
        agent.suit_integrity = 1.0
        agent._current_canister_remaining = 100.0
        agent.plss_co2_scrubber_pct = 100.0
        agent.plss_suit_battery_pct = 100.0
        agent.needs.energy = start_floor - 0.1

        self.assertFalse(self.engine._prepare_agent_for_eva(agent))
        self.assertTrue(agent._in_habitat)
        self.assertEqual("sleep", agent.action.action_type)
        self.assertEqual(
            start_floor,
            agent.action.target["required_energy_pct"],
        )

        agent.action.clear()
        agent.needs.energy = start_floor

        self.assertTrue(self.engine._prepare_agent_for_eva(agent))
        self.assertFalse(agent._in_habitat)

    def test_hydration_has_a_minimum_repeat_interval(self):
        agent = self.crew[0]
        agent._in_habitat = True
        agent.needs.thirst = 40.0
        agent._last_drink_tick = 10
        self.engine.decision_engine.agents = self.crew
        self.engine.decision_engine.central_depot_inventory = (
            self.engine.central_depot_inventory
        )
        self.engine.decision_engine._colony_resources = self.engine._colony_resources
        self.engine.decision_engine.lz_x = self.engine.lz_x
        self.engine.decision_engine.lz_y = self.engine.lz_y

        decision = self.engine.decision_engine.process_tick(
            agent,
            tick=12,
            tick_events={},
            world_context={
                "effective_temperature_c": 21.0,
                "active_events": [],
                "colony_resources": self.engine._colony_resources,
            },
            nearby_agents=[],
        )

        self.assertNotEqual("drink", decision["action"])

    def test_protected_maintenance_canisters_do_not_trigger_refill_loop(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.needs.o2_supply = 39.0
        agent.needs.energy = 20.0
        agent.needs.hunger = 90.0
        agent.needs.thirst = 90.0
        agent.inventory.items.pop("oxygen_canisters", None)
        self.engine.central_depot_inventory["oxygen_canisters"] = 2
        self.engine.decision_engine.agents = self.crew
        self.engine.decision_engine.central_depot_inventory = (
            self.engine.central_depot_inventory
        )
        self.engine.decision_engine.min_central_maintenance_o2_canisters = 2
        self.engine.decision_engine.lz_x = self.engine.lz_x
        self.engine.decision_engine.lz_y = self.engine.lz_y

        decision = self.engine.decision_engine.process_tick(
            agent,
            tick=1,
            tick_events={},
            world_context={
                "effective_temperature_c": 21.0,
                "active_events": [],
                "colony_resources": self.engine._colony_resources,
            },
            nearby_agents=[],
        )

        self.assertNotEqual("refill_o2", decision["action"])

    def test_bootstrap_recipe_and_manifested_field_stock_remain_auditable(self):
        research_materials = self.engine._get_recipe("research_workbench")["materials"]

        self.assertEqual(2, research_materials.get("electronic_component"))
        self.assertNotIn("electronics_salvage", research_materials)
        self.assertEqual(
            self.engine.delivered_field_stock["reduced_iron_ingot"]["quantity"],
            self.engine.central_depot_inventory["reduced_iron_ingot"],
        )

    def test_incapacitated_patient_receives_mass_balanced_rehydration(self):
        medic = max(self.crew, key=lambda crew: crew.competency.medical)
        patient = next(crew for crew in self.crew if crew is not medic)
        for crew in (medic, patient):
            crew.x, crew.y = self.engine.lz_x, self.engine.lz_y
            crew._in_habitat = True
        patient.status = AgentStatus.INCAPACITATED
        patient.needs.thirst = 0.0
        patient.needs._thirst_death_timer = 12
        water_before = self.engine._colony_resources["water_reserve_l"]
        fluid_before = self._total_manifest_item("sterile_iv_fluid_bags")
        access_before = self._total_manifest_item("iv_io_administration_sets")
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "treat",
            "target": {
                "patient": patient.name,
                "patient_id": patient.id,
                "protocol": "rehydration",
            },
            "reasoning": "administer measured habitat rehydration",
            "deterministic": True,
        }

        self.engine._process_agent_tick(medic, [], nearby_count=2)

        self.assertEqual(35.0, patient.needs.thirst)
        self.assertEqual(0, patient.needs._thirst_death_timer)
        self.assertEqual(water_before, self.engine._colony_resources["water_reserve_l"])
        self.assertEqual("iv_io_rehydration", medic.action.target["protocol"])
        self.assertEqual("iv_io", medic.action.target["hydration_route"])
        self.assertEqual(
            fluid_before - 1,
            self._total_manifest_item("sterile_iv_fluid_bags"),
        )
        self.assertEqual(
            access_before - 1,
            self._total_manifest_item("iv_io_administration_sets"),
        )

    def test_cross_trained_responder_can_stabilize_incapacitated_cmo(self):
        responder = next(
            crew for crew in self.crew if crew.competency.medical == 4
        )
        cmo = next(
            crew for crew in self.crew if crew.competency.medical >= 7
        )
        for crew in (responder, cmo):
            crew.x, crew.y = self.engine.lz_x, self.engine.lz_y
            crew._in_habitat = True
        cmo.status = AgentStatus.INCAPACITATED
        cmo.needs.thirst = 0.0
        cmo.needs._thirst_death_timer = 12
        fluid_before = self._total_manifest_item("sterile_iv_fluid_bags")
        access_before = self._total_manifest_item("iv_io_administration_sets")

        self.engine._process_agent_tick(responder, [], nearby_count=2)

        self.assertEqual("treat", responder.action.action_type)
        self.assertEqual("iv_io_rehydration", responder.action.target["protocol"])
        self.assertEqual("iv_io", responder.action.target["hydration_route"])
        self.assertTrue(responder.action.target["treatment_applied"])
        self.assertGreater(cmo.needs.thirst, 0.0)
        self.assertEqual(0, cmo.needs._thirst_death_timer)
        self.assertEqual(
            fluid_before - 1,
            self._total_manifest_item("sterile_iv_fluid_bags"),
        )
        self.assertEqual(
            access_before - 1,
            self._total_manifest_item("iv_io_administration_sets"),
        )

    def test_untrained_actor_cannot_execute_medical_protocol(self):
        responder, cmo = self.crew
        responder.competency.medical = 3
        for crew in (responder, cmo):
            crew.x, crew.y = self.engine.lz_x, self.engine.lz_y
            crew._in_habitat = True
        cmo.needs.thirst = 0.0
        water_before = self.engine._colony_resources["water_reserve_l"]
        supplies_before = self._total_manifest_item("medical_supplies")
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "treat",
            "target": {
                "patient": cmo.name,
                "patient_id": cmo.id,
                "protocol": "rehydration",
            },
            "reasoning": "invalid untrained treatment request",
            "deterministic": True,
        }

        self.engine._process_agent_tick(responder, [], nearby_count=2)

        self.assertEqual("idle", responder.action.action_type)
        self.assertEqual(0.0, cmo.needs.thirst)
        self.assertEqual(
            water_before,
            self.engine._colony_resources["water_reserve_l"],
        )
        self.assertEqual(
            supplies_before,
            self._total_manifest_item("medical_supplies"),
        )

    def test_incapacitated_patient_receives_mass_balanced_nutrition(self):
        medic = max(self.crew, key=lambda crew: crew.competency.medical)
        patient = next(crew for crew in self.crew if crew is not medic)
        for crew in (medic, patient):
            crew.x, crew.y = self.engine.lz_x, self.engine.lz_y
            crew._in_habitat = True
        patient.status = AgentStatus.INCAPACITATED
        patient.needs.hunger = 0.0
        patient.needs._hunger_death_timer = 20
        food_before = self._total_food_kcal()
        supplies_before = self._total_manifest_item("medical_supplies")
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "treat",
            "target": {
                "patient": patient.name,
                "patient_id": patient.id,
                "protocol": "nutrition_support",
            },
            "reasoning": "administer measured habitat nutrition",
            "deterministic": True,
        }

        self.engine._process_agent_tick(medic, [], nearby_count=2)

        self.assertGreater(patient.needs.hunger, 0.0)
        self.assertEqual(0, patient.needs._hunger_death_timer)
        self.assertEqual(
            food_before - 700.0,
            self._total_food_kcal(),
        )
        self.assertEqual(
            supplies_before,
            self._total_manifest_item("medical_supplies"),
        )

    def test_real_payload_is_transferred_once_to_central_depot(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.inventory.materials.clear()
        agent.inventory.add_material("regolith", 3)
        before = self.engine.central_depot_inventory.get("regolith", 0)
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "deposit_materials",
            "target": {},
            "reasoning": "valid logistics order",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual(before + 3, self.engine.central_depot_inventory["regolith"])
        self.assertEqual({}, agent.inventory.materials)
        self.assertEqual("deposit_materials", agent.action.action_type)

    def test_clean_solar_order_is_cancelled_when_array_is_not_dusty(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        self.engine.placed_structures.append({
            "id": "test_solar",
            "type": "solar_panel",
            "x": self.engine.lz_x + 6,
            "y": self.engine.lz_y,
            "dust_fouling_level": 0.05,
            "under_construction": False,
            "destroyed": False,
        })
        start = (agent.x, agent.y)
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "clean_solar_panels",
            "target": {"structure_id": "test_solar"},
            "reasoning": "stale cleaning order",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual(start, (agent.x, agent.y))
        self.assertEqual("rest", agent.action.action_type)
        self.assertTrue(agent.action.target["solar_cleaning_not_needed"])

    def test_solar_maintenance_can_use_protected_o2_reserve(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent._current_canister_remaining = 0.0
        agent.needs.o2_supply = 100.0
        agent.inventory.items.pop("oxygen_canisters", None)
        self.engine.central_depot_inventory["oxygen_canisters"] = 2
        # Enough metered gas to cycle the suitlock, not to fill a PLSS bottle.
        self.engine._colony_resources["o2_reserve_kg"] = 0.10
        self.engine.placed_structures.append({
            "id": "dirty_solar",
            "type": "solar_panel",
            "x": self.engine.lz_x + 6,
            "y": self.engine.lz_y,
            "dust_fouling_level": 1.00,
            "under_construction": False,
            "destroyed": False,
        })
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "clean_solar_panels",
            "target": {"structure_id": "dirty_solar"},
            "reasoning": "critical power maintenance",
        }

        for _ in range(8):
            self.engine.current_tick += 1
            self.engine._process_agent_tick(agent, [], nearby_count=1)
            if not agent._in_habitat:
                break

        self.assertFalse(agent._in_habitat)
        self.assertEqual(1, self.engine.central_depot_inventory["oxygen_canisters"])
        self.assertGreater(agent._current_canister_remaining, 0.0)
        self.assertEqual("move", agent.action.action_type)

    def test_dirty_solar_cleaning_preempts_unfinished_build_milestone(self):
        technician, other = self.crew
        technician.competency.engineering = 10
        other.competency.engineering = 0
        for agent in self.crew:
            agent.x = self.engine.lz_x
            agent.y = self.engine.lz_y
            agent._in_habitat = True
            agent.action.clear()
            agent.needs.energy = 100.0
            agent.needs.hunger = 100.0
            agent.needs.thirst = 100.0
            agent.needs.o2_supply = 100.0
            agent.needs.temperature_stress = 50.0
        panel = {
            "id": "scheduled_dirty_solar",
            "type": "solar_panel",
            "x": self.engine.lz_x + 6,
            "y": self.engine.lz_y,
            "dust_fouling_level": 1.0,
            "under_construction": False,
            "destroyed": False,
        }
        self.engine.placed_structures.append(panel)
        emitted = []
        self.engine._on_event = emitted.append

        for tick in range(1, 15):
            self.engine.current_tick = tick
            self.engine._process_agent_tick(technician, [], nearby_count=1)
            if panel["dust_fouling_level"] == 0.0:
                break

        self.assertEqual(0.0, panel["dust_fouling_level"])
        self.assertLessEqual(self.engine.current_tick, 8)
        self.assertTrue(any(
            event.get("type") == "clean_solar_panels" for event in emitted
        ))
        self.assertIn(
            technician.last_decision["action"],
            {"clean_solar_panels", "continue"},
        )

    def test_any_external_structure_repair_can_use_protected_o2_reserve(self):
        agent = self.crew[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent._current_canister_remaining = 0.0
        agent.inventory.items.pop("oxygen_canisters", None)
        self.engine.central_depot_inventory["oxygen_canisters"] = 2
        self.engine.central_depot_inventory["regolith"] = (
            self.engine._reserved_kit_materials().get("regolith", 0) + 3
        )
        self.engine._colony_resources["o2_reserve_kg"] = 0.10
        self.engine.structures_built["water_collector"] = 1
        self.engine.structure_health["water_collector"] = 0.30
        self.engine.placed_structures.append({
            "id": "damaged_water_collector",
            "type": "water_collector",
            "x": self.engine.lz_x + 6,
            "y": self.engine.lz_y,
            "health": 0.30,
            "under_construction": False,
            "destroyed": False,
        })
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "repair",
            "target": {"structure": "water_collector"},
            "reasoning": "critical exterior maintenance",
        }

        for _ in range(8):
            self.engine.current_tick += 1
            self.engine._process_agent_tick(agent, [], nearby_count=1)
            if not agent._in_habitat:
                break

        self.assertFalse(agent._in_habitat)
        self.assertEqual(1, self.engine.central_depot_inventory["oxygen_canisters"])
        self.assertEqual("move", agent.action.action_type)
        self.assertEqual(
            "external_structure_repair", agent.action.target["maintenance"]
        )
        self.assertEqual("repair", agent.action.target["maintenance_action"])

        for _ in range(8):
            self.engine.current_tick += 1
            self.engine._process_agent_tick(agent, [], nearby_count=1)
            if agent.action.action_type == "repair":
                break

        self.assertEqual("repair", agent.action.action_type)
        self.assertGreater(self.engine.structure_health["water_collector"], 0.30)

    def test_proxima_flare_uses_sv_rate_and_lander_shielding(self):
        """A dimensionless flare multiplier must not become Sv/tick."""
        engine = SimulationEngine(
            str(ROOT / "config" / "planets" / "proxima-centauri-b.json"),
            seed=42,
            db=False,
        )
        engine.llm_client.call = lambda *_args, **_kwargs: None
        agent = create_team_from_presets(
            str(ROOT / "config" / "agent_presets.json")
        )[0]
        engine.add_agent(agent)
        engine._init_agents()
        agent.x = engine.lz_x
        agent.y = engine.lz_y
        agent.suit_equipped = False
        before = agent.cumulative_radiation_sv
        flare = WorldEvent(
            event_type="stellar_flare",
            start_tick=0,
            duration_ticks=12,
            severity=0.5,
            radius_units=0,
            effects={"radiation_multiplier": 37.5},
        )

        engine._process_agent_tick(agent, [flare], nearby_count=0)

        received = agent.cumulative_radiation_sv - before
        # Terminator background and flare values are explicit hourly scenario
        # priors, converted once to the configured tick, then attenuated to
        # 2% in the lander's precursor storm vault.
        hourly_background = engine.planet.radiation_system[
            "baseline_dose_by_biome_sv_per_hour"
        ]["radiation_shadow_zone"]
        expected = (
            hourly_background * engine.SIM_HOURS_PER_TICK
            + 0.05 * engine.SIM_HOURS_PER_TICK * 0.5 * 0.6
        ) * 0.02
        self.assertAlmostEqual(expected, received, places=8)
        self.assertLess(received, 0.001)

    def test_commissioned_habitat_uses_integrated_storm_shielding(self):
        engine = SimulationEngine(
            str(ROOT / "config" / "planets" / "proxima-centauri-b.json"),
            seed=42,
            db=False,
        )
        engine.llm_client.call = lambda *_args, **_kwargs: None
        agent = create_team_from_presets(
            str(ROOT / "config" / "agent_presets.json")
        )[0]
        engine.add_agent(agent)
        engine._init_agents()
        agent.x = engine.lz_x + 5
        agent.y = engine.lz_y
        agent.suit_equipped = False
        engine.placed_structures.append({
            "id": "shielded-habitat-test",
            "type": "habitat_module",
            "x": agent.x,
            "y": agent.y,
            "health": 1.0,
            "under_construction": False,
            "destroyed": False,
        })
        cell = engine.world.get_cell_info(agent.x, agent.y, engine.current_tick)
        external = float(
            engine.planet.radiation_system[
                "baseline_dose_by_biome_sv_per_hour"
            ][cell["biome_id"]]
        ) * engine.SIM_HOURS_PER_TICK
        before = agent.cumulative_radiation_sv

        engine._process_agent_tick(agent, [], nearby_count=0)

        self.assertTrue(agent._in_habitat)
        self.assertAlmostEqual(
            external * 0.02,
            agent.cumulative_radiation_sv - before,
            places=8,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
