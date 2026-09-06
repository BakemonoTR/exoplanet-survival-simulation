"""Regression tests for anthropometric life-support consumption."""

import unittest
from pathlib import Path

from src.agents.agent import (
    ACTIVITY_MET_TABLE,
    Anthropometrics,
    create_team_from_presets,
)
from src.memory import vector_store
from src.orchestration.engine import SimulationEngine


ROOT = Path(__file__).resolve().parent
vector_store._use_tfidf_fallback = True


class PersonalizedLifeSupportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.crew = create_team_from_presets(
            str(ROOT / "config" / "agent_presets.json")
        )

    def test_reference_crew_member_matches_nasa_daily_targets(self):
        agent = self.crew[0]
        agent.anthropometrics = Anthropometrics(
            height_cm=175.0,
            mass_kg=80.0,
            foot_length_cm=26.5,
            shoe_size_eu=42,
        )

        self.assertAlmostEqual(
            0.72,
            agent.oxygen_consumption_kg_per_tick(1.0) * agent.TICKS_PER_DAY,
            places=6,
        )
        self.assertAlmostEqual(
            2.5,
            agent.water_requirement_l_per_tick(
                is_eva=False, activity_multiplier=1.0
            )
            * agent.TICKS_PER_DAY,
            places=6,
        )
        self.assertAlmostEqual(
            (2.5 / 24.0) + 0.24,
            agent.water_requirement_l_per_tick(
                is_eva=True, activity_multiplier=1.5
            )
            * agent.TICKS_PER_HOUR,
            places=6,
        )

    def test_personal_mass_and_bsa_change_consumption(self):
        yuki = next(a for a in self.crew if a.id == "agent_03")
        nikolai = next(a for a in self.crew if a.id == "agent_05")

        self.assertGreater(
            nikolai.oxygen_consumption_kg_per_tick(1.0),
            yuki.oxygen_consumption_kg_per_tick(1.0),
        )
        self.assertGreater(
            nikolai.water_requirement_l_per_tick(is_eva=False),
            yuki.water_requirement_l_per_tick(is_eva=False),
        )
        self.assertEqual(45, nikolai.anthropometrics.shoe_size_eu)
        self.assertEqual(37, yuki.anthropometrics.shoe_size_eu)

    def test_height_and_mass_both_affect_life_support_scaling(self):
        agent = self.crew[0]
        agent.anthropometrics = Anthropometrics(160.0, 70.0, 25.0, 40)
        shorter_o2 = agent.oxygen_consumption_kg_per_tick(1.0)
        shorter_water = agent.water_requirement_l_per_tick(False, 1.0)
        agent.anthropometrics = Anthropometrics(190.0, 70.0, 27.0, 42)

        self.assertGreater(agent.oxygen_consumption_kg_per_tick(1.0), shorter_o2)
        self.assertGreater(
            agent.water_requirement_l_per_tick(False, 1.0), shorter_water
        )

    def test_movement_and_heavy_work_raise_o2_and_water_demand(self):
        agent = self.crew[0]
        agent.anthropometrics = Anthropometrics(175.0, 80.0, 26.5, 42)

        idle_o2 = agent.oxygen_consumption_kg_per_tick(1.0)
        walking_o2 = agent.oxygen_consumption_kg_per_tick(1.5)
        mining_o2 = agent.oxygen_consumption_kg_per_tick(2.2)
        idle_water = agent.water_requirement_l_per_tick(False, 1.0)
        walking_water = agent.water_requirement_l_per_tick(False, 1.5)
        mining_water = agent.water_requirement_l_per_tick(False, 2.2)

        self.assertLess(idle_o2, walking_o2)
        self.assertLess(walking_o2, mining_o2)
        self.assertLess(idle_water, walking_water)
        self.assertLess(walking_water, mining_water)

    def test_production_repair_farming_and_carrying_are_not_idle_load(self):
        agent = self.crew[0]
        agent.anthropometrics = Anthropometrics(175.0, 80.0, 26.5, 42)
        physical_actions = (
            "refine",
            "repair",
            "repair_tool",
            "farm",
            "recycle_scrap",
            "deposit_materials",
            "rescue",
        )

        for action in physical_actions:
            with self.subTest(action=action):
                self.assertGreater(ACTIVITY_MET_TABLE[action], 1.0)
                agent.action.action_type = action
                self.assertGreater(
                    agent.oxygen_consumption_kg_per_tick(),
                    agent.oxygen_consumption_kg_per_tick(1.0),
                )
                self.assertGreater(
                    agent.water_requirement_l_per_tick(False),
                    agent.water_requirement_l_per_tick(False, 1.0),
                )

        self.assertGreater(
            ACTIVITY_MET_TABLE["carry_person"], ACTIVITY_MET_TABLE["move"]
        )

    def test_reference_plss_charge_lasts_eight_hours_at_walking_load(self):
        agent = self.crew[0]
        agent.anthropometrics = Anthropometrics(175.0, 80.0, 26.5, 42)

        consumed_pct = (
            agent.plss_o2_percent_per_tick(1.5)
            * 8
            * agent.TICKS_PER_HOUR
        )

        self.assertAlmostEqual(100.0, consumed_pct, places=6)

    def test_api_snapshot_exposes_body_and_personal_targets(self):
        engine = SimulationEngine(
            str(ROOT / "config" / "planets" / "kepler-442b.json"),
            seed=42,
            db=False,
        )
        engine.llm_client.call = lambda *_args, **_kwargs: None
        agent = self.crew[0]
        engine.add_agent(agent)
        engine._init_agents()

        state_agent = engine._get_state_snapshot()["agents"][0]

        self.assertEqual(168.0, state_agent["anthropometrics"]["height_cm"])
        self.assertEqual(39, state_agent["anthropometrics"]["shoe_size_eu"])
        self.assertIn("nominal_o2_kg_per_day", state_agent["life_support_profile"])
        self.assertIn("habitat_water_l_per_day", state_agent["life_support_profile"])

    def test_drinking_one_pack_records_exactly_one_liter(self):
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
        agent._in_habitat = True
        agent.needs.thirst = 60.0
        agent.inventory.items["water_packs"] = 1
        before_reserve = engine._colony_resources["water_reserve_l"]
        before_water_total = engine._water_mass_ledger()["tracked_total_l"]
        engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "drink",
            "target": {},
            "reasoning": "test hydration",
        }

        engine._process_agent_tick(agent, [], nearby_count=1)

        # This tick's metabolic loss occurs before the drink action.
        self.assertGreater(agent.needs.thirst, 94.0)
        self.assertLessEqual(agent.needs.thirst, 95.0)
        self.assertEqual(1.0, agent.total_water_consumed_l)
        self.assertFalse(agent.inventory.has_item("water_packs"))
        self.assertEqual(before_reserve, engine._colony_resources["water_reserve_l"])
        self.assertEqual(1.0, agent._recoverable_body_water_l)
        self.assertEqual([], engine._water_recovery_queue)
        self.assertEqual(
            before_water_total,
            engine._water_mass_ledger()["tracked_total_l"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
