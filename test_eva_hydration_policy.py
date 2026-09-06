"""Regression tests for hydration as a learned risk, not an EVA lock."""

import unittest
from pathlib import Path

from src.agents.agent import create_team_from_presets
from src.memory import vector_store
from src.orchestration.engine import SimulationEngine


ROOT = Path(__file__).resolve().parent
vector_store._use_tfidf_fallback = True


class EvaHydrationPolicyTest(unittest.TestCase):
    def _engine_and_agent(self):
        engine = SimulationEngine(
            str(ROOT / "config" / "planets" / "kepler-442b.json"),
            seed=42,
            db=False,
        )
        engine.llm_client.call = lambda *_args, **_kwargs: None
        agent = create_team_from_presets(
            str(ROOT / "config" / "agent_presets.json")
        )[0]
        engine.add_agent(agent)
        engine._init_agents()
        return engine, agent

    def test_empty_drink_bag_does_not_force_a_hidden_eva_return(self):
        engine, agent = self._engine_and_agent()
        agent.x = engine.lz_x + 10
        agent.y = engine.lz_y
        agent._in_habitat = False
        agent.action.clear()
        agent.needs.energy = 100.0
        agent.needs.hunger = 100.0
        agent.needs.thirst = 50.0
        agent.needs.o2_supply = 100.0
        agent.needs.temperature_stress = 50.0
        agent._current_canister_remaining = 100.0
        agent.plss_co2_scrubber_pct = 100.0
        agent.plss_suit_battery_pct = 100.0
        agent.suit_integrity = 1.0
        agent.inventory.items.pop("water_packs", None)
        engine.central_depot_inventory["water_packs"] = 0
        engine._colony_resources["water_reserve_l"] = 0.0

        engine._process_agent_tick(agent, [], nearby_count=0)

        target = agent.last_decision.get("target", {})
        self.assertFalse(target.get("hydration_return", False))
        self.assertNotIn("ending EVA", agent.last_decision.get("reasoning", ""))

    def test_remote_expedition_has_medical_go_no_go_interlock(self):
        engine, agent = self._engine_and_agent()
        agent.x = engine.lz_x
        agent.y = engine.lz_y
        agent._in_habitat = True
        agent.needs.energy = 40.0
        agent.needs.hunger = 20.0
        agent.needs.thirst = 10.0
        agent.needs.o2_supply = 100.0
        agent._current_canister_remaining = 100.0
        agent.plss_co2_scrubber_pct = 100.0
        agent.plss_suit_battery_pct = 100.0
        agent.suit_integrity = 1.0
        agent.inventory.items["oxygen_canisters"] = 1

        self.assertFalse(engine._agent_ready_for_expedition(agent, 18))
        agent.needs.energy = 90.0
        agent.needs.hunger = 90.0
        agent.needs.thirst = 90.0
        self.assertTrue(engine._agent_ready_for_expedition(agent, 18))

    def test_eva_drink_enters_body_pool_not_magic_return_fraction(self):
        engine, agent = self._engine_and_agent()
        agent.x = engine.lz_x + 5
        agent.y = engine.lz_y
        agent._in_habitat = False
        agent.inventory.items["water_packs"] = 1
        agent.tick_update = lambda **_kwargs: {
            "warnings": [], "died": False, "action_completed": False,
        }
        engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "drink", "target": {"field": True},
            "reasoning": "drink one physical EVA pack",
        }

        engine._process_agent_tick(agent, [], nearby_count=0)

        self.assertAlmostEqual(1.0, agent._recoverable_body_water_l, places=6)
        self.assertEqual([], engine._water_recovery_queue)

        agent.x = engine.lz_x
        agent.y = engine.lz_y
        agent._in_habitat = True
        engine._run_tick()

        self.assertGreater(agent._recoverable_body_water_l, 0.0)
        self.assertLess(agent._recoverable_body_water_l, 1.0)
        self.assertTrue(any(
            batch.get("source") == "habitat_metabolic_wastewater"
            and 0.0 < batch.get("liters", 0.0) < 0.9
            for batch in engine._water_recovery_queue
        ))

    def test_partial_habitat_reserve_is_drinkable(self):
        engine, agent = self._engine_and_agent()
        agent.x = engine.lz_x
        agent.y = engine.lz_y
        agent._in_habitat = True
        agent.action.clear()
        agent.needs.energy = 100.0
        agent.needs.hunger = 100.0
        agent.needs.thirst = 20.0
        agent.needs.o2_supply = 100.0
        agent.needs.temperature_stress = 50.0
        agent.inventory.items.pop("water_packs", None)
        engine.central_depot_inventory["water_packs"] = 0
        engine._colony_resources["water_reserve_l"] = 0.5

        decision = engine.decision_engine.process_tick(
            agent,
            tick=1,
            tick_events={},
            world_context={
                "active_events": [],
                "effective_temperature_c": 21.0,
                "colony_resources": engine._colony_resources,
            },
            nearby_agents=[],
        )

        self.assertEqual("drink", decision["action"])

    def test_processor_drip_does_not_create_perpetual_drink_orders(self):
        engine, agent = self._engine_and_agent()
        agent.x = engine.lz_x
        agent.y = engine.lz_y
        agent._in_habitat = True
        agent.action.clear()
        agent.needs.energy = 40.0
        agent.needs.hunger = 100.0
        agent.needs.thirst = 20.0
        agent.needs.o2_supply = 100.0
        agent.needs.temperature_stress = 50.0
        agent.inventory.items.pop("water_packs", None)
        engine.central_depot_inventory["water_packs"] = 0
        engine._colony_resources["water_reserve_l"] = 0.10

        decision = engine.decision_engine.process_tick(
            agent,
            tick=1,
            tick_events={},
            world_context={
                "active_events": [],
                "effective_temperature_c": 21.0,
                "colony_resources": engine._colony_resources,
            },
            nearby_agents=[],
        )

        self.assertEqual("sleep", decision["action"])

    def test_dry_base_conservation_sleep_really_restores_energy(self):
        engine, agent = self._engine_and_agent()
        agent.x = engine.lz_x
        agent.y = engine.lz_y
        agent._in_habitat = True
        agent.action.clear()
        agent.needs.energy = 40.0
        agent.needs.hunger = 100.0
        agent.needs.thirst = 20.0
        agent.needs.o2_supply = 100.0
        agent.needs.temperature_stress = 50.0
        agent.inventory.items.pop("water_packs", None)
        engine.central_depot_inventory["water_packs"] = 0
        engine._colony_resources["water_reserve_l"] = 0.0

        engine._process_agent_tick(agent, [], nearby_count=0)
        self.assertEqual("sleep", agent.action.action_type)
        self.assertTrue(agent.action.target["dry_base_conservation_sleep"])
        energy_at_sleep_start = agent.needs.energy

        engine._process_agent_tick(agent, [], nearby_count=0)

        self.assertEqual("sleep", agent.action.action_type)
        self.assertGreater(agent.needs.energy, energy_at_sleep_start)

    def test_thirst_wakes_sleeper_before_syncope_when_water_is_carried(self):
        engine, agent = self._engine_and_agent()
        agent.x = engine.lz_x
        agent.y = engine.lz_y
        agent._in_habitat = True
        agent.action.action_type = "sleep"
        agent.action.target = {"ticks": 24, "habitat": True}
        agent.action.ticks_remaining = 12
        agent.needs._consecutive_sleep_ticks = 3
        agent.needs.thirst = 44.0
        agent.needs.energy = 50.0
        agent.inventory.items["water_packs"] = 1

        engine._process_agent_tick(agent, [], nearby_count=0)

        self.assertNotEqual("sleep", agent.action.action_type)
        self.assertNotEqual("incapacitated", agent.status.value)
        self.assertGreater(agent.needs.thirst, 44.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
