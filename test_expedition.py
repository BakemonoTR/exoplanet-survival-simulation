"""Regression tests for staged, purposeful long-range exploration."""

import unittest
from pathlib import Path

from src.agents.agent import create_team_from_presets
from src.memory import vector_store
from src.orchestration.engine import SimulationEngine


ROOT = Path(__file__).resolve().parent
vector_store._use_tfidf_fallback = True


class ExpeditionPlanningTest(unittest.TestCase):
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
        self.engine.lz_x = self.engine.world.center
        self.engine.lz_y = self.engine.world.center
        self.agent.x = self.engine.lz_x
        self.agent.y = self.engine.lz_y
        self.agent.spawn_x = self.engine.lz_x
        self.agent.spawn_y = self.engine.lz_y
        self.agent.needs.energy = 100.0
        self.agent.needs.hunger = 100.0
        self.agent.needs.thirst = 100.0
        self.agent.needs.o2_supply = 100.0
        self.agent._current_canister_remaining = 100.0
        self.agent.plss_co2_scrubber_pct = 100.0
        self.agent.plss_suit_battery_pct = 100.0
        self.agent.suit_integrity = 1.0
        self.agent.inventory.items["oxygen_canisters"] = 2

    def _commission(self, structure_type: str):
        self.engine.structures_built[structure_type] = (
            self.engine.structures_built.get(structure_type, 0) + 1
        )
        self.engine.placed_structures.append({
            "id": f"test_{structure_type}",
            "type": structure_type,
            "x": self.engine.lz_x + len(self.engine.placed_structures) + 1,
            "y": self.engine.lz_y,
            "health": 1.0,
        })

    def _add_ready_rover_buddy(self):
        """A four-kilometre sortie needs a real second seat and physical PLSS."""
        buddy = create_team_from_presets(str(ROOT / "config/agent_presets.json"))[1]
        self.engine.add_agent(buddy)
        buddy.x, buddy.y = self.engine.lz_x, self.engine.lz_y
        buddy.spawn_x, buddy.spawn_y = buddy.x, buddy.y
        buddy._in_habitat = True
        buddy.needs.energy = buddy.needs.hunger = buddy.needs.thirst = 100.0
        buddy.needs.o2_supply = buddy._current_canister_remaining = 100.0
        buddy.plss_co2_scrubber_pct = buddy.plss_suit_battery_pct = 100.0
        buddy.suit_integrity = buddy.suit_condition = 1.0
        buddy.inventory.items["oxygen_canisters"] = 2
        buddy.action.clear()
        return buddy

    def test_infrastructure_unlocks_survey_and_communications_in_stages(self):
        self.assertEqual(24, self.engine._infrastructure_expedition_radius_cells())

        self.engine.structures_built["research_workbench"] = 1
        self.assertEqual(24, self.engine._infrastructure_expedition_radius_cells())

        self._commission("communication_relay")
        self.assertEqual(50, self.engine._infrastructure_expedition_radius_cells())

        self.engine.structures_built["laboratory_module"] = 1
        self.assertEqual(100, self.engine._infrastructure_expedition_radius_cells())

        self._commission("power_distribution_grid")
        self._commission("communications_array")
        self.assertEqual(200, self.engine._infrastructure_expedition_radius_cells())

    def test_regional_search_grid_does_not_prioritize_a_resource_biome(self):
        sulfur_station = self.engine._next_regional_scan_station(
            self.agent, "sulfur"
        )
        rare_station = self.engine._next_regional_scan_station(
            self.agent, "chalcopyrite_ore"
        )

        self.assertEqual(sulfur_station, rare_station)
        self.assertGreater(
            self.engine._distance_from_lz(*sulfur_station),
            self.engine.LOCAL_EVA_RADIUS_CELLS,
        )

    def test_unavailable_local_rover_releases_scanner_worker_until_retry(self):
        station = (self.engine.lz_x + 18, self.engine.lz_y)
        self.agent._in_habitat = True
        self.agent.inventory.items["portable_scanner"] = 1
        self.agent.inventory.tool_durability["portable_scanner"] = 500
        self.agent.inventory.tool_charge_pct["portable_scanner"] = 100.0
        self.engine.current_tick = 100
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "survey_resources",
            "target": {
                "resource": "silica_sand",
                "survey_resources": ["silica_sand"],
                "survey_action": "portable_scanner",
                "x": station[0],
                "y": station[1],
            },
            "reasoning": "local scanner regression",
            "deterministic": True,
        }
        self.engine._start_regional_survey_expedition = (
            lambda *_args, **_kwargs: None
        )

        self.engine._process_agent_tick(self.agent, [], nearby_count=0)

        self.assertEqual("rest", self.agent.action.action_type)
        self.assertTrue(self.agent.action.target["waiting_for_buddy_or_rover"])
        self.assertEqual(124, self.agent._regional_survey_retry_tick)
        self.assertEqual(124, self.agent.action.target["retry_tick"])

    def test_regional_samples_expand_across_alternating_direction_groups(self):
        distances = []
        for _ in range(8):
            station = self.engine._next_regional_scan_station(
                self.agent, "sulfur"
            )
            self.assertFalse(
                self.engine._is_construction_protected_cell(*station)
            )
            distances.append(self.engine._distance_from_lz(*station))
            self.engine._portable_scan_centers.add(station)

        # The reserved future landing corridor may remove one otherwise valid
        # ring station.  Sampling remains outward and sector-balanced rather
        # than driving a rover through planned blast/debris clearance.
        self.assertEqual([26, 26, 26, 26], distances[:4])
        self.assertEqual(sorted(distances), distances)
        self.assertTrue(all(30 <= distance <= 34 for distance in distances[4:]))

    def test_long_trip_requires_infrastructure_and_round_trip_plss_budget(self):
        self.assertFalse(self.engine._can_launch_expedition(self.agent, 40))

        self.engine.structures_built["research_workbench"] = 1
        self._commission("communication_relay")
        self.assertTrue(self.engine._can_launch_expedition(self.agent, 40))
        self.assertTrue(self.engine._can_launch_expedition(self.agent, 50))

        # A surveyed coordinate is not automatically walkable: without a
        # vehicle/forward shelter this distance exceeds the continuous EVA
        # round-trip budget even when deep survey and comms can see it.
        self.engine.structures_built["laboratory_module"] = 1
        self._commission("power_distribution_grid")
        self._commission("communications_array")
        self.assertFalse(self.engine._can_launch_expedition(self.agent, 90))

    def test_first_regional_trip_uses_a_ready_buddy_before_relay_exists(self):
        buddy = create_team_from_presets(
            str(ROOT / "config" / "agent_presets.json")
        )[1]
        self.engine.add_agent(buddy)
        buddy.x = self.engine.lz_x
        buddy.y = self.engine.lz_y
        buddy.spawn_x = self.engine.lz_x
        buddy.spawn_y = self.engine.lz_y
        buddy.needs.energy = 100.0
        buddy.needs.hunger = 100.0
        buddy.needs.thirst = 100.0
        buddy.needs.o2_supply = 100.0
        buddy._current_canister_remaining = 100.0
        buddy.plss_co2_scrubber_pct = 100.0
        buddy.plss_suit_battery_pct = 100.0
        buddy.suit_integrity = 1.0
        buddy.inventory.items["oxygen_canisters"] = 2
        self.engine.structures_built["research_workbench"] = 1

        self.assertTrue(self.engine._can_launch_expedition(self.agent, 40))
        expedition = self.engine._start_expedition(
            self.agent, "chalcopyrite_ore", (self.engine.lz_x + 40, self.engine.lz_y)
        )
        self.assertEqual(buddy.id, expedition["buddy_id"])
        self.assertEqual("buddy", buddy._active_expedition["role"])
        self.assertEqual("move", buddy.action.action_type)
        self.assertTrue(buddy.action.target["expedition"])

    def test_remote_resource_block_creates_science_infrastructure_milestone(self):
        self.engine.decision_engine.remote_resource_requests = {"chalcopyrite_ore"}
        recipe = self.engine.decision_engine._recipes_cache["research_workbench"]
        colony_mats = dict(recipe["materials"])
        decision = self.engine.decision_engine._exploration_infrastructure_candidate(
            colony_mats, self.engine.structures_built
        )
        self.assertEqual("build", decision["action"])
        self.assertEqual("research_workbench", decision["target"]["recipe"])

    def test_remote_deposit_becomes_a_named_authorized_expedition(self):
        buddy = self._add_ready_rover_buddy()
        self.engine.structures_built["research_workbench"] = 1
        self._commission("communication_relay")
        lz_x, lz_y = self.engine.lz_x, self.engine.lz_y

        target = (lz_x + 40, lz_y)
        self.assertIsNone(
            self.engine._find_resource_from_lz("chalcopyrite_ore", 1, 24)
        )
        self.assertIsNone(
            self.engine._find_resource_from_lz("chalcopyrite_ore", 25, 50)
        )
        self.engine.discovered_resources.setdefault("chalcopyrite_ore", set()).add(
            target
        )
        target = self.engine._find_resource_from_lz("chalcopyrite_ore", 25, 50)
        self.assertEqual((lz_x + 40, lz_y), target)

        expedition = self.engine._start_expedition(
            self.agent, "chalcopyrite_ore", target
        )
        self.assertEqual("crew_rover", expedition["transport"])
        self.assertEqual(buddy.id, expedition["buddy_id"])
        self.assertEqual("chalcopyrite_ore", expedition["resource"])
        self.assertEqual("outbound", expedition["status"])
        self.assertEqual(50, expedition["authorized_radius"])
        self.assertGreater(
            expedition["available_ticks_at_launch"],
            expedition["required_ticks_at_launch"],
        )
        self.assertEqual(50, self.engine._agent_eva_radius_cells(self.agent))

    def test_gatherer_launches_remote_trip_only_as_explicit_expedition(self):
        self.engine._init_agents()
        self.engine._initialized = True
        buddy = self._add_ready_rover_buddy()
        self.engine.structures_built["research_workbench"] = 1
        self._commission("communication_relay")
        self.agent.needs.energy = 100.0
        self.agent.needs.hunger = 100.0
        self.agent.needs.thirst = 100.0
        self.agent.needs.o2_supply = 100.0
        self.agent._current_canister_remaining = 100.0
        self.agent.plss_co2_scrubber_pct = 100.0
        self.agent.plss_suit_battery_pct = 100.0
        self.agent.suit_integrity = 1.0
        self.agent.inventory.items["oxygen_canisters"] = 2
        target = (self.engine.lz_x + 40, self.engine.lz_y)

        self.engine.discovered_resources.setdefault("chalcopyrite_ore", set()).add(
            target
        )
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "gather",
            "target": {"resource": "chalcopyrite_ore"},
            "reasoning": "mission dependency requires chalcopyrite ore",
            "deterministic": True,
        }

        # A named resource absent under the base triggers geological planning
        # before any excavation movement; the planned route is explicit.
        self.engine._process_agent_tick(self.agent, [], 0)

        expedition = getattr(self.agent, "_active_expedition", None)
        self.assertIsInstance(expedition, dict)
        self.assertEqual("crew_rover", expedition["transport"])
        self.assertEqual(buddy.id, expedition["buddy_id"])
        self.assertEqual("chalcopyrite_ore", expedition["resource"])
        self.assertEqual("outbound", expedition["status"])
        self.assertTrue(self.agent.action.target["expedition"])
        self.assertEqual(target, (
            self.agent.action.target["x"], self.agent.action.target["y"]
        ))
        self.assertLessEqual(
            self.engine._distance_from_lz(self.agent.x, self.agent.y),
            self.engine.MIN_EXTRACTION_RADIUS_CELLS + 2,
        )

    def test_expedition_aborts_early_when_return_reserve_or_energy_tightens(self):
        buddy = self._add_ready_rover_buddy()
        self.engine.structures_built["research_workbench"] = 1
        self._commission("communication_relay")
        target = (self.engine.lz_x + 40, self.engine.lz_y)
        self.engine._start_expedition(self.agent, "chalcopyrite_ore", target)
        self.assertEqual("crew_rover", self.agent._active_expedition["transport"])
        self.agent.x, self.agent.y = target
        self.agent._in_habitat = False
        self.agent.needs.energy = 54.0
        self.agent.action.action_type = "arrived"
        self.agent.action.ticks_remaining = 0

        decision_engine = self.engine.decision_engine
        decision_engine.agents = self.engine.agents
        decision_engine.lz_x = self.engine.lz_x
        decision_engine.lz_y = self.engine.lz_y
        decision_engine.max_eva_radius_cells = self.engine.LOCAL_EVA_RADIUS_CELLS
        decision_engine.expedition_move_speed_cells = self.engine.EXPEDITION_MOVE_SPEED_CELLS
        decision_engine.expedition_available_ticks = self.engine._expedition_available_ticks(self.agent)
        decision = decision_engine._process_tick_internal(
            self.agent,
            tick=20,
            tick_events={"warnings": []},
            world_context={"active_events": []},
            nearby_agents=[],
        )

        self.assertEqual("move", decision["action"])
        self.assertEqual("shelter", decision["target"]["destination"])
        self.assertTrue(decision["target"]["expedition"])
        self.assertEqual("returning", self.agent._active_expedition["status"])
        self.assertEqual("returning", buddy._active_expedition["status"])

    def test_buddy_expedition_reaches_remote_deposit_and_returns(self):
        buddy = create_team_from_presets(
            str(ROOT / "config" / "agent_presets.json")
        )[1]
        self.engine.add_agent(buddy)
        self.engine._init_agents()
        self.engine._initialized = True
        self.engine.structures_built["research_workbench"] = 1

        for crew in (self.agent, buddy):
            crew.needs.energy = 100.0
            crew.needs.hunger = 100.0
            crew.needs.thirst = 100.0
            crew.needs.o2_supply = 100.0
            crew._current_canister_remaining = 100.0
            crew.plss_co2_scrubber_pct = 100.0
            crew.plss_suit_battery_pct = 100.0
            crew.suit_integrity = 1.0
            crew.inventory.items["oxygen_canisters"] = 2

        target = (self.engine.lz_x + 40, self.engine.lz_y)
        original_get_cell_info = self.engine.world.get_cell_info

        def cell_with_remote_deposit(x, y, tick=0):
            info = dict(original_get_cell_info(x, y, tick))
            if (x, y) == target:
                info["base_resources"] = {
                    "regolith": "moderate",
                    "chalcopyrite_ore": "trace",
                }
            return info

        self.engine.world.get_cell_info = cell_with_remote_deposit
        # Isolate expedition transport from stratigraphy while preserving the
        # universal overburden rule: a prior physical survey/excavation has
        # exhausted the regolith and confirmed the now-exposed vein.
        self.engine.cell_geology[target] = {
            "layers": [
                {"material": "regolith", "initial_quantity": 5, "remaining": 0},
                {"material": "chalcopyrite_ore", "initial_quantity": 8, "remaining": 8},
            ],
            "current_index": 1,
        }
        self.engine.cell_resource_units[(*target, "regolith")] = 0
        self.engine.cell_resource_units[(*target, "chalcopyrite_ore")] = 8
        self.engine.depleted_cell_resources.add((*target, "regolith"))
        self.engine.discovered_resources.setdefault("chalcopyrite_ore", set()).add(
            target
        )
        self.engine.revealed_cell_resources.add(target)
        self.engine._start_expedition(self.agent, "chalcopyrite_ore", target)
        self.agent.action.action_type = "move"
        self.agent.action.target = {
            "x": target[0], "y": target[1],
            "resource": "chalcopyrite_ore", "expedition": True,
        }
        self.agent.action.ticks_remaining = 1

        max_distance = 0
        returned = False
        for _ in range(90):
            for crew in (self.agent, buddy):
                self.engine._process_agent_tick(crew, [], 1)
                max_distance = max(
                    max_distance,
                    self.engine._distance_from_lz(crew.x, crew.y),
                )
            self.engine.current_tick += 1
            if (
                max_distance > self.engine.LOCAL_EVA_RADIUS_CELLS
                and self.agent._active_expedition is None
                and buddy._active_expedition is None
                and self.agent._in_habitat
                and buddy._in_habitat
            ):
                returned = True
                break

        self.assertTrue(
            returned,
            (
                f"lead=({self.agent.x},{self.agent.y}) "
                f"lead_exp={self.agent._active_expedition} "
                f"lead_action={self.agent.action.action_type}:"
                f"{self.agent.action.target}; buddy=({buddy.x},{buddy.y}) "
                f"buddy_exp={buddy._active_expedition} "
                f"buddy_action={buddy.action.action_type}:{buddy.action.target}"
            ),
        )
        self.assertGreaterEqual(max_distance, 40)
        recovered_chalcopyrite_ore = (
            self.agent.inventory.materials.get("chalcopyrite_ore", 0)
            + self.engine.central_depot_inventory.get("chalcopyrite_ore", 0)
        )
        self.assertGreater(recovered_chalcopyrite_ore, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
