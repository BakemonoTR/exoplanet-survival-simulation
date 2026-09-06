"""Regression tests for physical food accounting and SAR self-care."""

import unittest
from pathlib import Path

from src.agents.agent import AgentStatus, create_team_from_presets
from src.memory import vector_store
from src.orchestration.engine import SimulationEngine


ROOT = Path(__file__).resolve().parent
vector_store._use_tfidf_fallback = True


class FoodMassBalanceTest(unittest.TestCase):
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
            agent.action.clear()
            agent.x = self.engine.lz_x
            agent.y = self.engine.lz_y
            agent._in_habitat = True
            agent.needs.energy = 100.0
            agent.needs.hunger = 100.0
            agent.needs.thirst = 100.0
            agent.needs.o2_supply = 100.0
            agent.needs.temperature_stress = 50.0
            agent._current_canister_remaining = 100.0
            agent.plss_co2_scrubber_pct = 100.0
            agent.plss_suit_battery_pct = 100.0
            agent.suit_integrity = 1.0

    @staticmethod
    def _no_physiology_tick(**_kwargs):
        return {
            "warnings": [],
            "died": False,
            "action_completed": False,
        }

    def test_bulk_food_is_not_continuously_drained_without_a_meal(self):
        before = self.engine._colony_resources["food_reserve_kcal"]
        for agent in self.crew:
            agent.action.action_type = "sleep"
            agent.action.target = {"ticks": 12, "habitat": True}
            agent.action.ticks_remaining = 12
            agent.inventory.items["emergency_rations"] = 2

        tick_events = self.engine._run_tick()

        self.assertEqual(
            before, self.engine._colony_resources["food_reserve_kcal"]
        )
        self.assertEqual(
            0.0,
            tick_events["colony_production"]["crew_food_consumed_kcal"],
        )

    def test_mature_greenhouse_accumulates_biomass_but_not_packaged_food(self):
        self.engine.get_day_night_phase = lambda: {
            "phase": "day", "solar_intensity": 1.0,
            "temp_modifier_c": 0.0,
        }
        self.engine.current_tick = self.engine._greenhouse_maturity_ticks()
        greenhouse = {
            "id": "test_greenhouse",
            "type": "greenhouse",
            "x": self.engine.lz_x + 4,
            "y": self.engine.lz_y + 4,
            "health": 1.0,
            "completed_tick": 0,
            "commissioned_tick": 0,
            "crop_growth_ticks": self.engine._greenhouse_maturity_ticks(),
            "last_crop_service_tick": self.engine.current_tick,
            "last_crop_harvest_tick": 0,
            "unharvested_food_kcal": 0.0,
            "pressurized": True,
        }
        self.engine.placed_structures.append(greenhouse)
        self.engine.structures_built["greenhouse"] = 1
        self.engine.placed_structures.append({
            "id": "food_test_life_grid",
            "type": "life_support_distribution_grid",
            "x": self.engine.lz_x + 2,
            "y": self.engine.lz_y + 2,
            "health": 1.0,
        })
        self.engine.structures_built["life_support_distribution_grid"] = 1
        for index in range(3):
            self.engine.placed_structures.append({
                "id": f"food_test_solar_{index}",
                "type": "solar_panel",
                "x": self.engine.lz_x + 6 + index * 2,
                "y": self.engine.lz_y,
                "health": 1.0,
                "dust_fouling_level": 0.0,
            })
        self.engine.placed_structures.append({
            "id": "food_test_grid",
            "type": "power_distribution_grid",
            "x": self.engine.lz_x + 2,
            "y": self.engine.lz_y,
            "health": 1.0,
        })
        self.engine.structures_built["solar_panel"] = 3
        self.engine.structures_built["power_distribution_grid"] = 1
        self.engine._colony_resources["food_reserve_kcal"] = 1_000.0
        self.engine._food_storage_capacity_kcal = 10_000.0
        for agent in self.crew:
            agent.action.action_type = "sleep"
            agent.action.target = {"ticks": 12, "habitat": True}
            agent.action.ticks_remaining = 12
            agent.inventory.items["emergency_rations"] = 2

        production = self.engine._run_tick()["colony_production"]

        self.assertEqual(1_000.0, self.engine._colony_resources["food_reserve_kcal"])
        self.assertEqual(0.0, production["food_production_kcal"])
        self.assertGreater(production["crop_biomass_generated_kcal"], 0.0)
        self.assertGreater(greenhouse["unharvested_food_kcal"], 0.0)

    def test_harvest_action_moves_only_ripe_calories_into_storage(self):
        agent = self.crew[0]
        agent.tick_update = self._no_physiology_tick
        agent.action.clear()
        greenhouse = {
            "id": "harvest_greenhouse",
            "type": "greenhouse",
            "x": self.engine.lz_x + 4,
            "y": self.engine.lz_y + 4,
            "health": 1.0,
            "completed_tick": 0,
            "pressurized": True,
            "unharvested_food_kcal": 5_000.0,
            "last_crop_harvest_tick": 0,
        }
        self.engine.placed_structures.append(greenhouse)
        self.engine.structures_built["greenhouse"] = 1
        agent.x = greenhouse["x"]
        agent.y = greenhouse["y"]
        self.engine.placed_structures.append({
            "id": "harvest_life_grid", "type": "life_support_distribution_grid",
            "x": self.engine.lz_x + 2, "y": self.engine.lz_y + 2, "health": 1.0,
        })
        agent._in_habitat = True
        self.engine._colony_resources["food_reserve_kcal"] = 1_000.0
        self.engine._food_storage_capacity_kcal = 10_000.0
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "farm",
            "target": {
                "greenhouse_id": greenhouse["id"],
                "x": greenhouse["x"],
                "y": greenhouse["y"],
                "destination": "greenhouse_workstation",
                "sub_action": "harvest",
            },
            "reasoning": "physical harvest and packing test",
            "deterministic": True,
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual(6_000.0, self.engine._colony_resources["food_reserve_kcal"])
        self.assertEqual(0.0, greenhouse["unharvested_food_kcal"])
        self.assertEqual(5_000.0, self.engine._greenhouse_harvested_kcal_this_tick)
        self.assertEqual("farm", agent.action.action_type)
        self.assertEqual(3, agent.action.ticks_remaining)

    def test_daily_crop_care_is_assigned_to_botany_lead(self):
        botanist = self.crew[0]
        botanist.action.clear()
        botanist.competency.engineering = 0
        botanist.competency.medical = 0
        botanist.competency.physics = 0
        botanist.competency.botany_bio = 10
        botanist.competency.leadership_social = 0
        greenhouse = {
            "id": "care_greenhouse",
            "type": "greenhouse",
            "x": self.engine.lz_x + 4,
            "y": self.engine.lz_y + 4,
            "health": 1.0,
            "completed_tick": 0,
            "last_crop_service_tick": 0,
            "unharvested_food_kcal": 0.0,
        }
        self.engine.decision_engine.agents = [botanist]
        self.engine.placed_structures.extend([
            greenhouse,
            {
                "id": "care_life_grid", "type": "life_support_distribution_grid",
                "x": self.engine.lz_x + 2, "y": self.engine.lz_y + 2, "health": 1.0,
            },
        ])
        self.engine.decision_engine.placed_structures = self.engine.placed_structures
        due_tick = 24 * 6

        decision = self.engine.decision_engine._greenhouse_duty_decision(
            botanist, due_tick
        )

        self.assertEqual("farm", decision["action"])
        self.assertEqual("crop_care", decision["target"]["sub_action"])
        self.assertEqual(greenhouse["id"], decision["target"]["greenhouse_id"])

    def test_one_ration_restores_only_its_body_scaled_700_kcal(self):
        agent = self.crew[0]
        agent.x, agent.y = self.engine._indoor_activity_position(agent, "eat")
        agent.tick_update = self._no_physiology_tick
        agent.needs.hunger = 30.0
        agent.inventory.items["emergency_rations"] = 1
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "eat",
            "target": {},
            "reasoning": "consume one physical ration",
        }
        expected = 30.0 + agent.hunger_points_for_kcal(700.0)

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertAlmostEqual(expected, agent.needs.hunger, places=6)
        self.assertEqual(700, agent.total_kcal_consumed)
        self.assertFalse(agent.inventory.has_item("emergency_rations"))

    def test_empty_food_sources_cannot_restore_hunger(self):
        agent = self.crew[0]
        agent.x, agent.y = self.engine._indoor_activity_position(agent, "eat")
        agent.tick_update = self._no_physiology_tick
        agent.needs.hunger = 30.0
        agent.inventory.items.pop("emergency_rations", None)
        agent.inventory.items.pop("ration_pack", None)
        self.engine.central_depot_inventory["ration_packs"] = 0
        self.engine._colony_resources["food_reserve_kcal"] = 0.0
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "eat",
            "target": {},
            "reasoning": "invalid empty meal request",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual(30.0, agent.needs.hunger)
        self.assertEqual(0, agent.total_kcal_consumed)
        self.assertEqual("idle", agent.action.action_type)
        self.assertTrue(agent.action.target["food_unavailable"])

    def test_empty_meal_does_not_send_crew_from_airlock_to_galley(self):
        agent = self.crew[0]
        agent.x, agent.y = self.engine._lander_airlock_position()
        agent.tick_update = self._no_physiology_tick
        agent.needs.hunger = 60.0
        agent.inventory.items.pop("emergency_rations", None)
        agent.inventory.items.pop("ration_pack", None)
        self.engine.central_depot_inventory["ration_packs"] = 0
        self.engine._colony_resources["food_reserve_kcal"] = 0.0
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "eat", "target": {},
            "reasoning": "invalid empty meal at the airlock",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("idle", agent.action.action_type)
        self.assertTrue(agent.action.target["food_unavailable"])
        self.assertEqual(self.engine._lander_airlock_position(), (agent.x, agent.y))
        self.assertEqual(60.0, agent.needs.hunger)
        self.assertEqual(0, agent.total_kcal_consumed)

    def test_stocked_exhausted_crew_eats_before_starting_another_sleep(self):
        agent = self.crew[0]
        agent.action.clear()
        agent.needs.energy = 10.0
        agent.needs.hunger = 10.0
        agent.needs.thirst = 100.0
        self.engine._colony_resources["food_reserve_kcal"] = 10_000.0
        self.engine.decision_engine.agents = self.crew
        self.engine.decision_engine.lz_x = self.engine.lz_x
        self.engine.decision_engine.lz_y = self.engine.lz_y

        decision = self.engine.decision_engine.process_tick(
            agent,
            tick=20,
            tick_events={},
            world_context={
                "active_events": [],
                "effective_temperature_c": 21.0,
                "colony_resources": self.engine._colony_resources,
            },
            nearby_agents=[],
        )

        self.assertEqual("eat", decision["action"])
        self.assertTrue(decision["target"]["pre_sleep_meal"])

    def test_sleep_wakes_before_one_tick_can_cross_into_starvation(self):
        agent = self.crew[0]
        agent.needs.hunger = 44.0
        agent.needs.thirst = 100.0
        agent.needs.energy = 30.0
        agent.action.action_type = "sleep"
        agent.action.target = {"ticks": 12, "habitat": True}
        agent.action.ticks_remaining = 12

        events = agent.tick_update(
            ambient_temp_c=21.0,
            gravity_multiplier=1.3,
            has_shelter=True,
            has_atmosphere=True,
            pressure_kpa=101.3,
        )

        self.assertIn(
            "sleep_interrupted_by_survival_threat", events["warnings"]
        )
        self.assertNotEqual("sleep", agent.action.action_type)
        self.assertGreater(agent.needs.hunger, 0.0)

    def test_motor_pre_duty_interlock_overrides_work_with_available_meal(self):
        agent = self.crew[0]
        agent.x, agent.y = self.engine._indoor_activity_position(agent, "eat")
        agent.tick_update = self._no_physiology_tick
        agent.needs.hunger = 10.0
        agent.needs.thirst = 100.0
        agent.inventory.items["emergency_rations"] = 1
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "build",
            "target": {"recipe": "solar_panel"},
            "reasoning": "stale work order",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertEqual("eat", agent.action.action_type)
        self.assertEqual(700, agent.total_kcal_consumed)
        self.assertGreater(agent.needs.hunger, 10.0)

    def test_partial_bulk_meal_uses_and_restores_the_same_kcal(self):
        agent = self.crew[0]
        agent.x, agent.y = self.engine._indoor_activity_position(agent, "eat")
        agent.tick_update = self._no_physiology_tick
        agent.needs.hunger = 30.0
        agent.inventory.items.pop("emergency_rations", None)
        agent.inventory.items.pop("ration_pack", None)
        self.engine.central_depot_inventory["ration_packs"] = 0
        self.engine._colony_resources["food_reserve_kcal"] = 150.0
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "eat",
            "target": {},
            "reasoning": "consume the remaining bulk food",
        }
        expected = 30.0 + agent.hunger_points_for_kcal(150.0)

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertAlmostEqual(expected, agent.needs.hunger, places=6)
        self.assertEqual(150, agent.total_kcal_consumed)
        self.assertEqual(0.0, self.engine._colony_resources["food_reserve_kcal"])

    def test_sar_lead_hydrates_from_carried_pack_before_dispatch(self):
        victim, rescuer = self.crew
        victim.status = AgentStatus.INCAPACITATED
        victim._in_habitat = False
        victim.x = self.engine.lz_x + 5
        victim.y = self.engine.lz_y
        rescuer._in_habitat = False
        rescuer.x = self.engine.lz_x + 2
        rescuer.y = self.engine.lz_y
        rescuer.needs.thirst = 40.0
        rescuer.inventory.items["water_packs"] = 1
        self.engine.decision_engine.agents = self.crew
        self.engine.decision_engine.lz_x = self.engine.lz_x
        self.engine.decision_engine.lz_y = self.engine.lz_y

        decision = self.engine.decision_engine.process_tick(
            rescuer,
            tick=20,
            tick_events={},
            world_context={
                "active_events": [],
                "effective_temperature_c": 21.0,
                "colony_resources": self.engine._colony_resources,
            },
            nearby_agents=[],
        )

        self.assertEqual("drink", decision["action"])
        self.assertTrue(decision["target"]["sar_self_care"])
        self.assertEqual(victim.id, decision["target"]["victim_id"])

    def test_unprovisioned_dehydrated_crew_member_is_not_sar_fit(self):
        victim, rescuer = self.crew
        victim.status = AgentStatus.INCAPACITATED
        rescuer._in_habitat = False
        rescuer.needs.thirst = 40.0
        rescuer.inventory.items.pop("water_packs", None)

        selected = self.engine.decision_engine.select_sar_rescuer_id(
            victim, self.crew
        )

        self.assertIsNone(selected)

    def test_larger_crew_member_has_larger_nominal_calorie_budget(self):
        crew = create_team_from_presets(
            str(ROOT / "config" / "agent_presets.json")
        )
        yuki = next(agent for agent in crew if agent.id == "agent_03")
        nikolai = next(agent for agent in crew if agent.id == "agent_05")

        self.assertGreater(
            nikolai.nominal_food_kcal_per_day(),
            yuki.nominal_food_kcal_per_day(),
        )

    def test_bulk_manifest_covers_thirty_nominal_crew_days(self):
        crew = create_team_from_presets(
            str(ROOT / "config" / "agent_presets.json")
        )[:5]
        thirty_day_need = 30.0 * sum(
            agent.nominal_food_kcal_per_day() for agent in crew
        )

        self.assertGreaterEqual(
            self.engine._colony_resources["food_reserve_kcal"],
            thirty_day_need * 1.4,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
