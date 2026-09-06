"""Focused regressions for utility-feasible greenhouse sites and crew work."""

import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.agent import create_team_from_presets
from src.memory import vector_store
from src.orchestration.engine import SimulationEngine


ROOT = Path(__file__).resolve().parent
vector_store._use_tfidf_fallback = True


class GreenhouseUtilityRegressionTest(unittest.TestCase):
    def setUp(self):
        self.engine = SimulationEngine(
            str(ROOT / "config" / "planets" / "kepler-442b.json"),
            seed=42, db=False,
        )
        self.engine.llm_client.call = lambda *_args, **_kwargs: None
        self.agent = create_team_from_presets(
            str(ROOT / "config" / "agent_presets.json")
        )[0]
        self.engine.add_agent(self.agent)
        self.engine._init_agents()
        self.agent.action.clear()
        self.agent.needs.energy = 100.0
        self.agent.needs.hunger = 100.0
        self.agent.needs.thirst = 100.0
        self.agent.needs.o2_supply = 100.0
        self.agent.needs.temperature_stress = 50.0
        self.agent.competency.engineering = 0
        self.agent.competency.medical = 0
        self.agent.competency.physics = 0
        self.agent.competency.leadership_social = 0
        self.agent.competency.botany_bio = 10
        self.engine.decision_engine.agents = [self.agent]
        self.engine.decision_engine.placed_structures = self.engine.placed_structures

    def add(self, kind, identifier, dx=None, dy=None):
        if dx is None:
            x, y = self.engine._select_structure_site(kind)
        else:
            x, y = self.engine.lz_x + dx, self.engine.lz_y + dy
        structure = {
            "id": identifier, "type": kind, "x": x, "y": y,
            "health": 1.0, "completed_tick": 0, "commissioned_tick": 0,
            "last_crop_service_tick": 0, "unharvested_food_kcal": 0.0,
        }
        self.engine.placed_structures.append(structure)
        self.engine.structures_built[kind] = (
            self.engine.structures_built.get(kind, 0) + 1
        )
        return structure

    def candidate(self, dx, dy, score=0):
        return (score, max(abs(dx), abs(dy)),
                self.engine.lz_y + dy, self.engine.lz_x + dx)

    def isolated_grid(self, *, line_length=None, ports=None):
        self.engine.placed_structures.clear()
        grid = self.add("life_support_distribution_grid", "grid", 0, 0)
        effects = self.engine._get_recipe("life_support_distribution_grid")["output"]["effects"]
        if line_length is not None:
            effects["installed_line_length_m"] = line_length
        if ports is not None:
            effects["max_connected_endpoints"] = ports
        self.engine._utility_network_cache_signature = None
        return grid

    def test_seed42_two_greenhouses_have_real_routes_with_unchanged_limits(self):
        grids = [self.add("life_support_distribution_grid", f"grid-{i}") for i in range(2)]
        self.assertEqual(
            [(4, 4), (5, -5)],
            [(s["x"] - self.engine.lz_x, s["y"] - self.engine.lz_y) for s in grids],
        )
        farms = [self.add("greenhouse", f"farm-{i}") for i in range(2)]
        network = self.engine._life_support_network_snapshot()
        self.assertEqual(12, network["service_radius_cells"])
        self.assertEqual(9000.0, network["maximum_length_m"])
        self.assertEqual(32, network["maximum_endpoints"])
        self.assertLessEqual(network["installed_length_m"], network["maximum_length_m"])
        for farm in farms:
            self.assertTrue(self.engine._life_support_connected(farm))
            self.assertLessEqual(min(max(abs(farm["x"] - g["x"]), abs(farm["y"] - g["y"])) for g in grids), 12)
            self.assertTrue(any(r["structure_id"] == farm["id"] and r["path"] for r in network["routes"]))

    def test_original_run233_parcels_are_rejected(self):
        self.add("life_support_distribution_grid", "grid-1")
        self.add("life_support_distribution_grid", "grid-2")
        for candidate in [self.candidate(-9, -8), self.candidate(-8, -11)]:
            self.assertIsNone(self.engine._select_utility_feasible_site("greenhouse", [candidate]))

    def test_greenhouse_requires_commissioned_grid_before_ground_is_broken(self):
        grid = self.add("life_support_distribution_grid", "unfinished-grid", 4, 4)
        grid["under_construction"] = True
        with self.assertRaisesRegex(RuntimeError, "utility-feasible"):
            self.engine._select_structure_site("greenhouse")

    def test_missing_grid_rejects_greenhouse_before_terrain_scan(self):
        with patch.object(
            self.engine.world,
            "get_cell_info",
            wraps=self.engine.world.get_cell_info,
        ) as cell_info:
            with self.assertRaisesRegex(RuntimeError, "utility-feasible"):
                self.engine._select_structure_site("greenhouse")

        self.assertEqual(
            0, cell_info.call_count,
            "a missing utility grid is a global precondition, not a parcel search",
        )

    def test_no_endpoint_overbooking_or_existing_consumer_displacement(self):
        self.isolated_grid(ports=1)
        self.add("water_collector", "existing-water", 2, 0)
        self.assertTrue(self.engine._life_support_connected(self.engine.placed_structures[-1]))
        self.assertIsNone(self.engine._select_utility_feasible_site("greenhouse", [self.candidate(0, 2)]))

    def test_radius_alone_does_not_bypass_corridor_budget(self):
        self.isolated_grid(line_length=100.0)
        self.assertIsNone(self.engine._select_utility_feasible_site("greenhouse", [self.candidate(2, 0)]))

    def test_funded_pending_consumer_reserves_port_without_activating_it(self):
        self.isolated_grid(ports=1)
        pending = self.add("greenhouse", "pending-farm", 2, 0)
        pending["under_construction"] = True
        pending["materials_committed"] = False
        before = self.engine._life_support_network_snapshot()
        candidate = self.candidate(0, 2)
        # A mere uncommitted plan is not an endpoint reservation.
        self.assertIsNotNone(self.engine._select_utility_feasible_site("greenhouse", [candidate]))
        self.engine._construction_cargo_reservations[pending["id"]] = {
            "total_mass_kg": 127602.4, "status": "reserved_at_depot",
        }
        self.assertIsNone(self.engine._select_utility_feasible_site("greenhouse", [candidate]))
        self.assertIs(before, self.engine._life_support_network_snapshot())
        self.assertNotIn(pending["id"], before["physically_connected_structure_ids"])
        self.assertTrue(pending["under_construction"])
        self.assertFalse(pending["materials_committed"])

    def test_preview_cannot_steal_an_existing_farms_line_budget(self):
        self.isolated_grid(line_length=600.0)
        existing = self.add("greenhouse", "existing-farm", 6, 0)
        before = self.engine._life_support_network_snapshot()
        self.assertTrue(self.engine._life_support_connected(existing))
        self.assertIsNone(self.engine._select_utility_feasible_site("greenhouse", [self.candidate(0, 2)]))
        self.assertIs(before, self.engine._life_support_network_snapshot())

    def test_later_process_skid_cannot_displace_commissioned_greenhouse(self):
        self.isolated_grid(line_length=600.0)
        existing = self.add("greenhouse", "existing-farm", 6, 0)
        existing["completed_tick"] = 10
        existing["commissioned_tick"] = 10
        self.assertTrue(self.engine._life_support_connected(existing))

        later = self.add("water_collector", "later-water", 0, 2)
        later["completed_tick"] = 20
        later["commissioned_tick"] = 20
        self.engine._utility_network_cache_signature = None
        network = self.engine._life_support_network_snapshot()

        endpoints = {
            endpoint["structure_id"]: endpoint
            for endpoint in network["endpoints"]
        }
        self.assertTrue(endpoints[existing["id"]]["connected"])
        self.assertEqual(
            "line_length_budget_exhausted", endpoints[later["id"]]["reason"]
        )

    def test_obstructed_route_is_not_accepted_inside_radius(self):
        self.isolated_grid()
        x, y = self.engine.lz_x + 3, self.engine.lz_y + 3
        for cell in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            self.engine.spoil_piles[cell] = {"mass": 1}
        self.assertIsNone(self.engine._select_utility_feasible_site("greenhouse", [self.candidate(3, 3)]))

    def test_candidate_scan_builds_existing_network_only_once(self):
        self.isolated_grid()
        x, y = self.engine.lz_x + 3, self.engine.lz_y + 3
        for cell in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            self.engine.spoil_piles[cell] = {"mass": 1}
        candidates = [self.candidate(3, 3), self.candidate(-3, -3, score=1)]

        with patch.object(
            self.engine,
            "_life_support_network_snapshot",
            wraps=self.engine._life_support_network_snapshot,
        ) as snapshot:
            selected = self.engine._select_utility_feasible_site(
                "greenhouse", candidates
            )

        self.assertEqual(
            (self.engine.lz_x - 3, self.engine.lz_y - 3), selected
        )
        self.assertEqual(
            1,
            snapshot.call_count,
            "candidate evaluation must extend one base network, not rebuild it",
        )

    def test_preview_does_not_mutate_live_structures_or_cached_network(self):
        self.isolated_grid()
        before = self.engine._life_support_network_snapshot()
        structures = list(self.engine.placed_structures)
        selected = self.engine._select_utility_feasible_site("greenhouse", [self.candidate(3, 3)])
        self.assertEqual((self.engine.lz_x + 3, self.engine.lz_y + 3), selected)
        self.assertEqual(structures, self.engine.placed_structures)
        self.assertIs(before, self.engine._life_support_network_snapshot())
        self.assertNotIn("__proposed_greenhouse_site__", before["physically_connected_structure_ids"])

    def test_physical_site_can_be_planned_while_bus_is_temporarily_unpowered(self):
        self.isolated_grid()
        self.engine._colony_resources["energy_stored_kwh"] = 0.0
        self.assertIsNotNone(self.engine._select_utility_feasible_site("greenhouse", [self.candidate(3, 3)]))

    def test_disconnected_farm_gets_no_crop_duty_but_connected_farm_does(self):
        farm = self.add("greenhouse", "farm", 4, 4)
        decision = self.engine.decision_engine
        self.assertIsNone(decision._greenhouse_duty_decision(self.agent, 144))
        self.add("life_support_distribution_grid", "grid", 2, 2)
        task = decision._greenhouse_duty_decision(self.agent, 144)
        self.assertEqual("farm", task["action"])
        self.assertEqual(farm["id"], task["target"]["greenhouse_id"])
        self.engine._colony_resources["energy_stored_kwh"] = 0.0
        self.assertIsNone(decision._greenhouse_duty_decision(self.agent, 144))

    def test_executor_rejects_stale_farm_action_after_grid_loss(self):
        grid = self.add("life_support_distribution_grid", "grid", 2, 2)
        farm = self.add("greenhouse", "farm", 4, 4)
        self.assertTrue(self.engine._life_support_connected(farm))
        grid["destroyed"] = True
        self.agent.tick_update = lambda **_kwargs: {
            "warnings": [], "died": False, "action_completed": False,
        }
        self.agent.x, self.agent.y = farm["x"], farm["y"]
        self.engine.current_tick = 144
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "farm", "target": {
                "greenhouse_id": farm["id"], "sub_action": "crop_care",
            }, "deterministic": True,
        }
        self.engine._process_agent_tick(self.agent, [], nearby_count=1)
        self.assertEqual("invalid_action_rejected", self.agent.action.action_type)
        self.assertEqual(0, farm["last_crop_service_tick"])


if __name__ == "__main__":
    unittest.main()
