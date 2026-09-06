"""Regression tests for physically supplied structure maintenance."""

import unittest
from pathlib import Path

from src.agents.agent import create_team_from_presets
from src.memory import vector_store
from src.orchestration.engine import SimulationEngine


ROOT = Path(__file__).resolve().parent
vector_store._use_tfidf_fallback = True


class MaintenanceMaterialTest(unittest.TestCase):
    def setUp(self):
        self.engine = SimulationEngine(
            str(ROOT / "config" / "planets" / "kepler-442b.json"),
            seed=42,
            db=False,
        )
        self.engine.llm_client.call = lambda *_args, **_kwargs: None
        self.agent = create_team_from_presets(
            str(ROOT / "config" / "agent_presets.json")
        )[0]
        self.engine.add_agent(self.agent)
        self.engine._init_agents()
        self.asset = {
            "id": "water_service_test",
            "type": "water_collector",
            "x": self.engine.lz_x + 4,
            "y": self.engine.lz_y,
            "health": 0.30,
            "under_construction": False,
            "destroyed": False,
        }
        self.engine.placed_structures.append(self.asset)
        self.engine.structures_built["water_collector"] = 1

    def test_repair_consumes_exact_configured_spares(self):
        self.engine.central_depot_inventory["regolith"] = 8

        repaired = self.engine.repair_structure(
            self.agent, "water_collector", self.asset["id"]
        )

        self.assertTrue(repaired)
        self.assertEqual(5, self.engine.central_depot_inventory["regolith"])
        self.assertGreater(self.asset["health"], 0.30)

    def test_repair_cannot_create_spares(self):
        self.engine.central_depot_inventory["regolith"] = 2

        repaired = self.engine.repair_structure(
            self.agent, "water_collector", self.asset["id"]
        )

        self.assertFalse(repaired)
        self.assertEqual(2, self.engine.central_depot_inventory["regolith"])
        self.assertEqual(0.30, self.asset["health"])

    def test_condition_inspection_consumes_no_spares_or_health(self):
        self.engine.current_tick = 400
        self.asset["health"] = 0.93
        self.asset["maintenance_due"] = True
        self.engine.central_depot_inventory["regolith"] = 2

        inspected = self.engine.repair_structure(
            self.agent,
            "water_collector",
            self.asset["id"],
            inspection_only=True,
        )

        self.assertTrue(inspected)
        self.assertEqual(2, self.engine.central_depot_inventory["regolith"])
        self.assertEqual(0.93, self.asset["health"])
        self.assertEqual(400, self.asset["last_maintenance_tick"])
        self.assertFalse(self.asset["maintenance_due"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
