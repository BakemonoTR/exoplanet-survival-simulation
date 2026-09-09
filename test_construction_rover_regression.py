"""Focused two-seat construction transport and supervision regressions."""

import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.agent import create_team_from_presets
from src.memory import vector_store
from src.orchestration.engine import SimulationEngine


ROOT = Path(__file__).resolve().parent
vector_store._use_tfidf_fallback = True


class ConstructionRoverRegressionTest(unittest.TestCase):
    def setUp(self):
        self.engine = SimulationEngine(
            str(ROOT / "config/planets/kepler-442b.json"), seed=42, db=False
        )
        self.engine.llm_client.call = lambda *_args, **_kwargs: None
        self.crew = create_team_from_presets(str(ROOT / "config/agent_presets.json"))[:2]
        for crew in self.crew:
            self.engine.add_agent(crew)
        self.engine._init_agents()
        self.engine._initialized = True
        self.planner = self.engine.decision_engine
        self.planner.agents = self.crew
        self.planner.placed_structures = self.engine.placed_structures
        self.planner.structures_built = self.engine.structures_built
        self.planner.lz_x, self.planner.lz_y = self.engine.lz_x, self.engine.lz_y
        self.planner.expedition_available_ticks = 36
        self.engine.world.get_cell_info = lambda *_args: {
            "temperature_c": 20.0, "traversal_cost": 1.0, "traversable": True,
        }
        for crew in self.crew:
            crew.x, crew.y = self.engine._lander_airlock_position()
            crew._in_habitat = True
            crew.needs.energy = crew.needs.hunger = crew.needs.thirst = 100.0
            crew.needs.o2_supply = crew._current_canister_remaining = 100.0
            crew.needs.temperature_stress = 50.0
            crew.plss_co2_scrubber_pct = crew.plss_suit_battery_pct = 100.0
            crew.suit_integrity = crew.suit_condition = 1.0
            crew.inventory.items["oxygen_canisters"] = 2
            crew.inventory.items["water_packs"] = 2
            crew.inventory.materials.clear()
            crew.action.clear()
        self.site = {
            "id": "remote-solar", "type": "solar_panel",
            "x": self.engine.lz_x - 15, "y": self.engine.lz_y + 10,
            "under_construction": True, "destroyed": False,
            "materials_committed": True, "required_work_hours": 120.0,
            "work_hours_completed": 0.0, "progress": 0.0,
            "ticks_remaining": 720, "total_ticks": 720,
        }
        self.engine.placed_structures.append(self.site)
        self.planner.shared_work_order = {
            "recipe": "solar_panel", "stage": "construction",
            "site_id": self.site["id"],
            "construction_crew_ids": [crew.id for crew in self.crew],
            "assignments": {crew.id: "construction" for crew in self.crew},
        }

    def _launch(self):
        self.assertTrue(self.engine._start_construction_rover_trip(self.crew[0], self.site))
        state = getattr(self.crew[0], "_active_expedition", None)
        self.assertIsInstance(state, dict, self.crew[0].action.target)
        return state

    def _arrive(self):
        state = self._launch()
        for _ in range(10):
            self.engine.current_tick += 1
            self.engine._move_construction_rover_team(self.crew[0], self.crew[0].action.target)
            if state["status"] == "working":
                break
        self.assertEqual("working", state["status"])
        return state

    def test_distant_site_uses_two_seats_and_one_accounted_route(self):
        initial_energy = self.engine.surface_fleet.crew_rovers[0].battery_kwh
        state = self._arrive()
        rover = self.engine.surface_fleet.crew_rovers[0]
        distance_km = (len(state["route"]) - 1) * 0.1
        self.assertGreaterEqual(distance_km, 2.3)
        self.assertEqual(2, len(rover.mission["crew_ids"]))
        self.assertEqual("in_use", rover.state)
        self.assertEqual((self.site["x"], self.site["y"]), (self.crew[0].x, self.crew[0].y))
        self.assertEqual((self.crew[0].x, self.crew[0].y), (self.crew[1].x, self.crew[1].y))
        self.assertAlmostEqual(distance_km, rover.total_distance_km)
        self.assertAlmostEqual(initial_energy - distance_km * 1.2, rover.battery_kwh)
        before = rover.total_distance_km
        self.engine._move_construction_rover_team(self.crew[1], self.crew[1].action.target)
        self.assertEqual(before, rover.total_distance_km, "second seat cannot move the rover twice per tick")

    def test_unavailable_rover_or_buddy_never_starts_a_solo_walk(self):
        origin = (self.crew[0].x, self.crew[0].y)
        self.crew[1].action.action_type = "sleep"
        self.assertTrue(self.engine._start_construction_rover_trip(self.crew[0], self.site))
        self.assertIsNone(getattr(self.crew[0], "_active_expedition", None))
        self.assertEqual(origin, (self.crew[0].x, self.crew[0].y))
        self.crew[1].action.clear()
        self.engine.surface_fleet.crew_rovers[0].battery_kwh = 0.0
        self.assertTrue(self.engine._start_construction_rover_trip(self.crew[0], self.site))
        self.assertEqual("insufficient_return_energy", self.crew[0].action.target["reason"])
        self.assertEqual(origin, (self.crew[0].x, self.crew[0].y))

    def test_short_local_site_stays_on_foot(self):
        self.site["x"], self.site["y"] = self.engine.lz_x, self.engine.lz_y + 4
        self.assertFalse(self.engine._start_construction_rover_trip(self.crew[0], self.site))
        self.assertEqual("idle", self.engine.surface_fleet.crew_rovers[0].state)

    def test_pending_indoor_activity_does_not_get_hijacked_for_rover(self):
        self.crew[1]._pending_indoor_activity = {"action": "sleep", "target": {}}
        self.assertTrue(self.engine._start_construction_rover_trip(self.crew[0], self.site))
        self.assertIsNone(getattr(self.crew[0], "_active_expedition", None))
        self.assertEqual("sleep", self.crew[1]._pending_indoor_activity["action"])

    def test_buddy_safety_recall_returns_the_whole_vehicle(self):
        self._arrive()
        self.planner._recall_expedition_team(self.crew[1])
        self.engine.current_tick += 1
        self.engine._move_construction_rover_team(self.crew[0], {"x": self.site["x"], "y": self.site["y"]})
        for crew in self.crew:
            self.assertEqual("returning", crew._active_expedition["status"])
        self.assertEqual((self.crew[0].x, self.crew[0].y), (self.crew[1].x, self.crew[1].y))

    def test_utility_workfront_walk_does_not_drag_parked_rover(self):
        self._arrive()
        lead = self.crew[0]
        rover = self.engine.surface_fleet.crew_rovers[0]
        parked = (rover.x, rover.y)
        target = {
            "x": lead.x + 5,
            "y": lead.y,
            "destination": "construction_site",
            "construction_route": True,
            "utility_route_workfront": True,
            "expedition": True,
            "transport": "on_foot",
        }
        self.planner.process_tick = lambda **_kwargs: {
            "action": "move", "target": target, "deterministic": True,
        }

        self.engine.current_tick += 1
        self.engine._process_agent_tick(lead, [], 1)

        self.assertEqual(parked, (rover.x, rover.y))
        self.assertNotEqual(parked, (lead.x, lead.y))

    def test_split_returning_crew_walk_individually_to_physical_rover(self):
        state = self._arrive()
        lead, buddy = self.crew
        rover = self.engine.surface_fleet.crew_rovers[0]
        # Reproduce a saved/legacy mismatch: route metadata still ends at the
        # structure, while the physical rover is parked at the workfront.
        rover.x, rover.y = self.site["x"] + 5, self.site["y"]
        lead.x, lead.y = self.site["x"], self.site["y"]
        buddy.x, buddy.y = self.site["x"] + 3, self.site["y"]
        buddy.action.action_type = "sleep"
        buddy.action.target = {"habitat": True}
        buddy.action.ticks_remaining = 4
        for crew in self.crew:
            crew._in_habitat = False
            crew._active_expedition["status"] = "returning"
            crew._active_expedition["route_index"] = len(state["route"]) - 1

        self.engine.current_tick += 1
        self.engine._move_construction_rover_team(
            lead, {"destination": "shelter", "expedition": True}
        )

        self.assertEqual((self.site["x"] + 1, self.site["y"]), (lead.x, lead.y))
        self.assertEqual((self.site["x"] + 3, self.site["y"]), (buddy.x, buddy.y))
        self.assertEqual("sleep", buddy.action.action_type)
        self.engine._move_construction_rover_team(
            buddy,
            {"x": self.engine.lz_x, "y": self.engine.lz_y,
             "destination": "material_storage"},
        )
        self.assertEqual((self.site["x"] + 4, self.site["y"]), (buddy.x, buddy.y))
        self.assertEqual(
            "construction_rover_rendezvous", buddy.action.target["destination"]
        )

    def test_faulted_rover_falls_back_to_physical_walkback(self):
        self._arrive()
        origin = (self.crew[0].x, self.crew[0].y)
        self.engine.surface_fleet.crew_rovers[0].battery_kwh = 0.0
        self.engine.current_tick += 1
        self.engine._move_construction_rover_team(self.crew[0], {"destination": "shelter"})
        for crew in self.crew:
            self.assertEqual("on_foot", crew._active_expedition["transport"])
            self.assertTrue(crew._active_expedition["rover_fault"])
        for _ in range(3):
            self.engine.current_tick += 1
            for crew in self.crew:
                self.engine._process_agent_tick(crew, [], 1)
        for crew in self.crew:
            walked = abs(crew.x - origin[0]) + abs(crew.y - origin[1])
            self.assertGreater(walked, 0)
            self.assertLessEqual(walked, 3)
            self.assertFalse(crew._in_habitat)
        self.assertEqual("fault", self.engine.surface_fleet.crew_rovers[0].state)

    def test_short_engine_loop_completes_remote_work_and_returns(self):
        self.site["required_work_hours"] = 1.0
        self._launch()
        for _ in range(30):
            result = self.engine._run_tick()
            self.assertFalse(result["agent_processing_errors"])
            self.engine.current_tick += 1
            if not self.site["under_construction"] and all(
                getattr(crew, "_active_expedition", None) is None for crew in self.crew
            ):
                break
        self.assertFalse(self.site["under_construction"])
        self.assertEqual(1, self.engine.surface_fleet.crew_rovers[0].completed_jobs)
        self.assertTrue(all(crew._in_habitat for crew in self.crew))

    def test_build_dispatcher_launches_transport_before_remote_footstep(self):
        self.planner.process_tick = lambda **_kwargs: {
            "action": "build", "target": {
                "recipe": self.site["type"], "struct_id": self.site["id"],
            }, "deterministic": True,
        }
        origin = (self.crew[0].x, self.crew[0].y)
        self.engine._process_agent_tick(self.crew[0], [], 1)
        self.assertEqual(origin, (self.crew[0].x, self.crew[0].y))
        self.assertEqual("construction_support", self.crew[0]._active_expedition["kind"])
        self.assertEqual("in_use", self.engine.surface_fleet.crew_rovers[0].state)

    def test_planned_site_keeps_rover_destination_until_groundbreaking(self):
        self.engine.placed_structures.remove(self.site)
        self.engine.delivered_structure_kits.clear()
        planned = {**self.site, "planned": True, "id": "planned-remote-solar"}
        self.assertTrue(self.engine._start_construction_rover_trip(self.crew[0], planned))
        state = self.crew[0]._active_expedition
        for _ in range(10):
            self.engine.current_tick += 1
            self.engine._move_construction_rover_team(self.crew[0], self.crew[0].action.target)
            if state["status"] == "working":
                break
        decision = self.planner._process_tick_internal(self.crew[0], self.engine.current_tick, {})
        self.assertEqual("build", decision["action"])
        recipe = self.engine._get_recipe("solar_panel")
        self.engine.central_depot_inventory.update(recipe["materials"])
        self.planner.process_tick = lambda **_kwargs: decision
        with patch.object(self.crew[0], "can_craft", return_value={"can_craft": True}), \
             patch.object(self.engine, "_select_structure_site", side_effect=AssertionError("reserved destination must not change")):
            self.engine._process_agent_tick(self.crew[0], [], 1)
        actual = next(s for s in self.engine.placed_structures if s["type"] == "solar_panel")
        self.assertEqual((planned["x"], planned["y"]), (actual["x"], actual["y"]))
        for crew in self.crew:
            self.assertEqual(actual["id"], crew._active_expedition["site_id"])
            self.assertFalse(crew._active_expedition["planned_site"])

    def test_completed_topoff_does_not_rearm_stale_hydration_target(self):
        agent = self.crew[0]
        agent.needs.thirst = 98.5
        agent._eva_hydration_pending = False
        agent._eva_hydration_target = 99.0

        ready = self.engine._prepare_eva_water(
            agent,
            (self.engine.lz_x + 6, self.engine.lz_y + 7),
            rover_outbound=True,
        )

        self.assertTrue(ready, agent.action.target)
        self.assertFalse(agent._eva_hydration_pending)
        self.assertEqual(80.0, agent._eva_hydration_target)

    def test_underqualified_rover_buddy_waits_for_qualified_lead(self):
        self.engine.placed_structures.remove(self.site)
        planned = {
            **self.site,
            "planned": True,
            "id": "planned-isru-preflight",
            "type": "isru_o2_unit",
        }
        lead, buddy = self.crew
        self.assertGreaterEqual(lead.competency.engineering, 6)
        self.assertLess(buddy.competency.engineering, 6)
        reservation = {
            "site": planned,
            "crew_ids": [lead.id, buddy.id],
            "lead_id": lead.id,
            "created_tick": self.engine.current_tick,
            "energy_ready": {lead.id: True, buddy.id: True},
        }
        self.engine._construction_sortie = reservation
        for member in (lead, buddy):
            member._construction_preparation = reservation

        decision = self.engine._construction_preparation_decision(buddy)

        self.assertEqual("stand_watch", decision["action"])
        self.assertEqual("awaiting_construction_lead", decision["target"]["reason"])
        self.assertEqual(lead.id, decision["target"]["construction_lead_id"])

    def test_planned_sortie_clears_preflight_with_support_buddy(self):
        self.engine.placed_structures.remove(self.site)
        planned = {
            **self.site,
            "planned": True,
            "id": "planned-isru-bounded-preflight",
            "type": "isru_o2_unit",
        }
        lead, buddy = self.crew
        buddy.action.action_type = "sleep"
        buddy.action.target = {"habitat": True, "ticks": 2}
        buddy.action.ticks_remaining = 2
        buddy._pending_material_pickup = {
            "action": "craft_item",
            "target": {"recipe": "stone_hammer"},
        }
        self.assertTrue(self.engine._start_construction_rover_trip(lead, planned))
        self.assertIsNone(getattr(lead, "_active_expedition", None))
        self.assertIsNone(buddy._pending_material_pickup)

        blocked_actions = []
        for _ in range(40):
            result = self.engine._run_tick()
            self.assertFalse(result["agent_processing_errors"])
            blocked_actions.append(buddy.action.action_type)
            self.engine.current_tick += 1
            if isinstance(getattr(lead, "_active_expedition", None), dict):
                break

        self.assertIsInstance(getattr(lead, "_active_expedition", None), dict)
        self.assertEqual(
            lead._active_expedition["id"], buddy._active_expedition["id"]
        )
        self.assertNotIn("craft_blocked", blocked_actions)

    def test_completed_site_returns_both_crew_and_releases_rover(self):
        state = self._arrive()
        one_way_distance = self.engine.surface_fleet.crew_rovers[0].total_distance_km
        self.site["under_construction"] = False
        decision = self.planner._process_tick_internal(self.crew[0], self.engine.current_tick, {})
        self.assertEqual("shelter", decision["target"].get("destination"), decision)
        for _ in range(10):
            self.engine.current_tick += 1
            self.engine._move_construction_rover_team(self.crew[0], decision["target"])
            if getattr(self.crew[0], "_active_expedition", None) is None:
                break
        rover = self.engine.surface_fleet.crew_rovers[0]
        self.assertEqual("charging", rover.state)
        self.assertEqual(1, rover.completed_jobs)
        self.assertAlmostEqual(2 * one_way_distance, rover.total_distance_km)
        for crew in self.crew:
            self.assertTrue(crew._in_habitat)
            self.assertIsNone(crew._active_expedition)

    def test_rover_buddy_builds_and_robots_still_require_onsite_supervision(self):
        state = self._arrive()
        for crew in self.crew:
            decision = self.planner._process_tick_internal(crew, self.engine.current_tick, {})
            self.assertEqual("build", decision["action"], decision)
            self.assertEqual(self.site["id"], decision["target"]["struct_id"])
            crew.action.action_type = "build"
            crew.action.target = decision["target"]
            crew.action.ticks_remaining = 1
        with patch.object(self.engine, "_process_agent_tick", return_value=[]):
            self.engine._run_tick()
        self.assertEqual(2, self.site["active_builder_count"])
        self.assertGreater(self.site["work_hours_completed"], 0.0)
        fleet = self.engine.surface_fleet
        for _ in range(30):
            fleet.tick(available_grid_energy_kwh=0.0,
                       excavation_callback=lambda *_args: (0, 0.0, True),
                       unload_callback=lambda _payload: None)
        assisted = fleet.assembly_assist(site_id=self.site["id"], builder_count=2)
        self.assertGreater(assisted["work_hours"], 0.0)
        unsupervised = fleet.assembly_assist(site_id=self.site["id"], builder_count=0)
        self.assertEqual(0.0, unsupervised["work_hours"])
        self.assertTrue(any(robot.state == "waiting_supervision" for robot in fleet.assembly_robots))


if __name__ == "__main__":
    unittest.main()
