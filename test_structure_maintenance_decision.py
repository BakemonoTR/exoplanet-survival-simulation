"""Regression tests for starvation-free, material-backed maintenance orders."""

import unittest
from pathlib import Path

from src.agents.agent import create_team_from_presets
from src.memory import vector_store
from src.orchestration.engine import SimulationEngine


ROOT = Path(__file__).resolve().parent
vector_store._use_tfidf_fallback = True


class StructureMaintenanceDecisionTest(unittest.TestCase):
    def setUp(self):
        self.engine = SimulationEngine(
            str(ROOT / "config" / "planets" / "kepler-442b.json"),
            seed=42,
            db=False,
        )
        self.engine.llm_client.call = lambda *_args, **_kwargs: None
        self.crew = create_team_from_presets(
            str(ROOT / "config" / "agent_presets.json")
        )[:5]
        for agent in self.crew:
            self.engine.add_agent(agent)
        self.engine._init_agents()

        self.planner = self.engine.decision_engine
        self.planner.agents = self.crew
        self.planner.central_depot_inventory = self.engine.central_depot_inventory
        self.planner.delivered_structure_kits = self.engine.delivered_structure_kits
        self.planner.placed_structures = self.engine.placed_structures
        self.planner.structures_built = self.engine.structures_built
        self.planner.lz_x = self.engine.lz_x
        self.planner.lz_y = self.engine.lz_y
        self.planner.surface_requires_plss = True
        self.planner._colony_resources = self.engine._colony_resources

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

    def _damage_forge_with_spares(self):
        forge = next(
            structure for structure in self.engine.placed_structures
            if structure["type"] == "forge"
        )
        forge["health"] = 0.50
        self.engine.central_depot_inventory["basalt"] = 5
        return forge

    def _decision(self, agent, tick=100):
        return self.planner.process_tick(
            agent,
            tick=tick,
            tick_events={},
            world_context={
                "effective_temperature_c": 21.0,
                "active_events": [],
                "colony_resources": self.engine._colony_resources,
            },
            nearby_agents=[],
        )

    def test_maintenance_preempts_pending_scanner_with_one_exact_owner(self):
        forge = self._damage_forge_with_spares()
        chief = self.crew[0]
        chief.inventory.items["portable_scanner"] = 1
        chief.inventory.tool_durability["portable_scanner"] = 500
        chief.inventory.tool_charge_pct["portable_scanner"] = 100.0
        self.planner.planetary_resources = {"sulfur"}
        self.planner.remote_resource_requests = {"sulfur"}

        decisions = {
            agent.id: self._decision(agent)
            for agent in self.crew
        }
        repairs = [
            (agent_id, decision)
            for agent_id, decision in decisions.items()
            if decision["action"] == "repair"
        ]

        self.assertEqual(1, len(repairs))
        self.assertEqual(chief.id, repairs[0][0])
        self.assertEqual(forge["id"], repairs[0][1]["target"]["structure_id"])
        self.assertEqual(
            {"reduced_iron_ingot": 3, "basalt": 5},
            repairs[0][1]["target"]["repair_materials"],
        )

    def test_scanner_locked_chief_does_not_block_available_technician(self):
        forge = self._damage_forge_with_spares()
        chief = self.crew[0]
        chief.action.action_type = "move"
        chief.action.ticks_remaining = 1
        chief.action.target = {
            "x": self.engine.lz_x + 6,
            "y": self.engine.lz_y,
            "destination": "survey_station",
            "survey_action": "portable_scanner",
            "resource": "sulfur",
        }

        decisions = {
            agent.id: self._decision(agent, tick=101)
            for agent in self.crew
        }
        repair_owners = [
            agent_id for agent_id, decision in decisions.items()
            if decision["action"] == "repair"
        ]

        # Volkov is the highest-engineering crew member who is actually free;
        # reserving the order to the scanner-locked chief would emit no repair.
        self.assertEqual([self.crew[4].id], repair_owners)
        self.assertEqual(
            forge["id"], decisions[repair_owners[0]]["target"]["structure_id"]
        )

    def test_reserved_starter_kit_parts_cannot_back_a_repair_order(self):
        self.engine.structures_built["isru_o2_unit"] = 1
        isru = {
            "id": "damaged_isru",
            "type": "isru_o2_unit",
            "x": self.engine.lz_x + 5,
            "y": self.engine.lz_y,
            "health": 0.50,
            "under_construction": False,
            "destroyed": False,
        }
        self.engine.placed_structures.append(isru)
        # Every reported electronic component belongs to the sealed starter
        # solar kit. Override the normal field-stock fixture so only the
        # copper-bearing spare stock is free in this reservation-boundary test.
        self.engine.central_depot_inventory["chalcopyrite_ore"] = 1
        self.engine.central_depot_inventory["electronic_component"] = (
            self.engine._get_recipe("solar_panel")["materials"][
                "electronic_component"
            ]
        )
        self.assertEqual(
            self.engine._get_recipe("solar_panel")["materials"][
                "electronic_component"
            ],
            self.engine.central_depot_inventory["electronic_component"],
        )

        decision = self.planner._structure_maintenance_decision(
            self.crew[0], self.engine.structures_built
        )

        self.assertIsNotNone(decision)
        self.assertNotEqual("repair", decision["action"])
        self.assertEqual(isru["id"], decision["target"]["maintenance_for"])
        self.assertEqual("isru_o2_unit", decision["target"]["maintenance_structure"])

    def test_due_healthy_asset_gets_inspection_without_spare_dependency(self):
        forge = next(
            structure for structure in self.engine.placed_structures
            if structure["type"] == "forge"
        )
        forge["health"] = 0.96
        forge["maintenance_due"] = True
        self.engine.central_depot_inventory["basalt"] = 0

        decision = self.planner._structure_maintenance_decision(
            self.crew[0], self.engine.structures_built
        )

        self.assertIsNotNone(decision)
        self.assertEqual("repair", decision["action"])
        self.assertTrue(decision["target"]["inspection_only"])
        self.assertEqual("inspection", decision["target"]["maintenance_action"])
        self.assertEqual({}, decision["target"]["repair_materials"])

    def test_inspection_route_resumes_through_executable_repair_action(self):
        forge = next(
            structure for structure in self.engine.placed_structures
            if structure["type"] == "forge"
        )
        agent = self.crew[0]
        agent.action.action_type = "move"
        agent.action.ticks_remaining = 1
        agent.action.target = {
            "x": forge["x"],
            "y": forge["y"],
            "structure": "forge",
            "structure_id": forge["id"],
            "maintenance": "external_structure_repair",
            "maintenance_action": "inspection",
            "inspection_only": True,
        }

        decision = self._decision(agent, tick=102)

        self.assertEqual("repair", decision["action"])
        self.assertEqual("inspection", decision["target"]["maintenance_action"])
        self.assertTrue(decision["target"]["inspection_only"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
