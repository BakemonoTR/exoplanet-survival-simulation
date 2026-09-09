"""Regression tests for the precursor mission architecture.

These tests intentionally exercise backend contracts only.  The mission
state and vehicle telemetry are not exposed in the frontend yet.
"""

import os
import unittest

from src.agents.agent import create_team_from_presets
from src.memory import vector_store
from src.orchestration.engine import SimulationEngine
from src.systems.event_scheduler import EventScheduler
from src.systems.mission_profile import MissionProfile
from src.systems.surface_fleet import SurfaceFleet
from src.world.generator import PlanetConfig


ROOT = os.path.dirname(os.path.abspath(__file__))
vector_store._use_tfidf_fallback = True


class MissionProfileTest(unittest.TestCase):
    def test_clock_population_and_decision_authority_are_explicit(self):
        profile = MissionProfile.load()

        self.assertEqual(6, profile.advance_crew)
        self.assertEqual(100, profile.arriving_civilians)
        self.assertEqual(10.0, profile.clock.tick_minutes)
        self.assertEqual(2.0, profile.clock.realtime_seconds_per_tick)
        self.assertEqual(144, profile.clock.ticks_per_earth_day)
        self.assertEqual(51_840, profile.clock.planet_attempt_max_ticks)
        self.assertEqual(3_888_000, profile.clock.public_challenge_ticks)
        self.assertEqual(
            27_000.0, profile.clock.public_challenge_simulated_days
        )
        self.assertEqual(
            75.0, profile.clock.full_attempts_in_public_challenge
        )
        self.assertEqual("dialogue_only", profile.decision_authority["language_model"])
        self.assertFalse(
            profile.decision_authority["language_model_may_mutate_state"]
        )
        manifest = profile.advance_crew_consumables
        self.assertEqual(360, manifest["design_days"])
        self.assertEqual(10000.0, manifest["potable_water_l"])
        self.assertEqual(1814.4, manifest["oxygen_kg"])
        self.assertEqual(5832000.0, manifest["food_kcal"])
        self.assertEqual(
            set(profile.delivered_precision_stock),
            set(profile.production_policy["delivered_only_materials"]),
        )

    def test_engine_uses_crew_manifest_without_counting_it_as_capacity(self):
        engine = SimulationEngine(
            os.path.join(ROOT, "config", "planets", "kepler-442b.json"),
            db=False,
        )
        self.assertEqual(10000.0, engine._colony_resources["water_reserve_l"])
        self.assertEqual(1814.4, engine._colony_resources["o2_reserve_kg"])
        self.assertEqual(5832000.0, engine._colony_resources["food_reserve_kcal"])
        self.assertEqual(
            18.0,
            engine._colony_resources["lander_integrated_solar_peak_kw"],
        )
        self.assertEqual(
            288.0,
            engine._colony_resources["lander_auxiliary_energy_remaining_kwh"],
        )
        self.assertEqual(1419, engine.delivered_field_stock[
            "structural_truss_section"
        ]["quantity"])
        self.assertEqual(
            1419,
            engine.central_depot_inventory["structural_truss_section"],
        )
        self.assertEqual(0.0, engine.colony_score.get_overall_score())

    def test_every_planet_marks_observation_inference_and_scenario_priors(self):
        for filename in (
            "kepler-442b.json", "proxima-centauri-b.json", "ross-128b.json",
            "teegardens-star-b.json", "trappist-1e.json",
        ):
            planet = PlanetConfig(os.path.join(
                ROOT, "config", "planets", filename
            ))
            epistemic = planet.data["epistemic_model"]
            self.assertEqual(
                "scenario-constrained synthetic surface realization",
                epistemic["classification"],
            )
            self.assertTrue(epistemic["observed_parameters"])
            self.assertTrue(epistemic["inferred_parameters"])
            self.assertTrue(epistemic["scenario_assumptions"])
            self.assertGreater(
                float(planet.radiation_system["baseline_dose_sv_per_hour"]),
                0.0,
            )
            self.assertEqual(
                "scenario_prior_not_surface_measurement",
                planet.radiation_system["model_status"],
            )


class PlanetaryEventCadenceTest(unittest.TestCase):
    def test_stellar_flares_do_not_apply_generic_mechanical_hull_damage(self):
        planet = PlanetConfig(os.path.join(
            ROOT, "config", "planets", "proxima-centauri-b.json"
        ))
        scheduler = EventScheduler(
            planet, seed=42, max_ticks=25_920, tick_minutes=10.0
        )
        flares = [
            event for event in scheduler.events
            if event.event_type == "stellar_flare"
        ]

        self.assertGreater(len(flares), 100)
        self.assertTrue(all(
            event.effects["structure_damage_per_tick"] == 0.0
            for event in flares
        ))

    def test_micrometeorite_clusters_are_sparse_local_and_bounded(self):
        planet = PlanetConfig(os.path.join(
            ROOT, "config", "planets", "proxima-centauri-b.json"
        ))
        scheduler = EventScheduler(
            planet, seed=42, max_ticks=25_920, tick_minutes=10.0
        )
        clusters = [
            event for event in scheduler.events
            if event.event_type == "micrometeorite_shower"
        ]

        self.assertGreaterEqual(len(clusters), 3)
        self.assertLessEqual(len(clusters), 10)
        self.assertTrue(all(
            event.epicenter is not None and event.radius_units > 0
            for event in clusters
        ))
        self.assertTrue(all(
            event.effects["structure_damage_per_tick"] * event.duration_ticks
            <= 0.021 + 1e-9
            for event in clusters
        ))

    def test_active_geology_does_not_mean_destructive_quakes_every_few_hours(self):
        planet = PlanetConfig(os.path.join(
            ROOT, "config", "planets", "kepler-442b.json"
        ))
        scheduler = EventScheduler(
            planet, seed=42, max_ticks=8_640, tick_minutes=10.0
        )
        quakes = [
            event for event in scheduler.events
            if event.event_type == "seismic_quake"
        ]

        self.assertGreater(len(quakes), 0)
        self.assertLessEqual(len(quakes), 9)
        self.assertTrue(all(
            event.effects["structure_damage_per_tick"] * event.duration_ticks
            <= 0.036 + 1e-9
            for event in quakes
        ))

    def test_event_calendar_is_invariant_to_tick_resolution(self):
        planet = PlanetConfig(os.path.join(
            ROOT, "config", "planets", "kepler-442b.json"
        ))
        at_five = EventScheduler(
            planet, seed=7, max_ticks=17_280, tick_minutes=5.0
        )
        at_ten = EventScheduler(
            planet, seed=7, max_ticks=8_640, tick_minutes=10.0
        )
        five_minutes = [
            event.start_tick * 5.0 for event in at_five.events
            if event.event_type == "seismic_quake"
        ]
        ten_minutes = [
            event.start_tick * 10.0 for event in at_ten.events
            if event.event_type == "seismic_quake"
        ]

        self.assertEqual(len(five_minutes), len(ten_minutes))
        self.assertTrue(all(
            abs(left - right) <= 10.0
            for left, right in zip(five_minutes, ten_minutes)
        ))


class SurfaceFleetTest(unittest.TestCase):
    def setUp(self):
        self.profile = MissionProfile.load()
        self.fleet = SurfaceFleet(self.profile)
        self.fleet.place_at_base(100, 100)

    def test_delivered_fleet_and_dispatch_masks(self):
        self.assertEqual(1, len(self.fleet.crew_rovers))
        self.assertEqual(4, len(self.fleet.excavators))
        self.assertEqual(2, len(self.fleet.cargo_transporters))

        unverified = self.fleet.dispatch_excavator(
            resource="regolith",
            target_x=115,
            target_y=100,
            current_tick=1,
            dispatched_by="agent-1",
            verified_exposed_face=False,
            protected_ground=False,
        )
        protected = self.fleet.dispatch_excavator(
            resource="regolith",
            target_x=115,
            target_y=100,
            current_tick=1,
            dispatched_by="agent-1",
            verified_exposed_face=True,
            protected_ground=True,
        )

        self.assertEqual("unverified_exposed_face", unverified["reason"])
        self.assertEqual("construction_ground_protected", protected["reason"])

    def test_excavator_target_mask_includes_round_trip_work_and_reserve(self):
        near = self.fleet.excavator_dispatch_feasibility(280, 100)
        far = self.fleet.excavator_dispatch_feasibility(290, 100)

        self.assertTrue(near["feasible"])
        self.assertFalse(far["feasible"])
        self.assertEqual("insufficient_return_energy", far["reason"])
        rejected = self.fleet.dispatch_excavator(
            resource="regolith",
            target_x=290,
            target_y=100,
            current_tick=1,
            dispatched_by="agent-1",
            verified_exposed_face=True,
            protected_ground=False,
        )
        self.assertFalse(rejected["dispatched"])
        self.assertEqual("insufficient_return_energy", rejected["reason"])

    def test_excavator_conserves_callback_material_until_unload(self):
        dispatched = self.fleet.dispatch_excavator(
            resource="regolith",
            target_x=101,
            target_y=100,
            current_tick=1,
            dispatched_by="agent-1",
            verified_exposed_face=True,
            protected_ground=False,
        )
        self.assertTrue(dispatched["dispatched"])

        remaining_units = 100
        unloaded = {}

        def excavate(job, requested_kg):
            nonlocal remaining_units
            units = min(remaining_units, max(1, int(requested_kg // 1.5)))
            remaining_units -= units
            return units, units * 1.5, remaining_units == 0

        def unload(payload):
            for resource, quantity in payload.items():
                unloaded[resource] = unloaded.get(resource, 0) + quantity

        for _ in range(200):
            result = self.fleet.tick(
                available_grid_energy_kwh=100.0,
                excavation_callback=excavate,
                unload_callback=unload,
            )
            if result["events"]:
                break

        self.assertGreater(unloaded.get("regolith", 0), 0)
        self.assertEqual(
            100,
            remaining_units + unloaded["regolith"],
        )
        self.assertLessEqual(
            unloaded["regolith"] * 1.5,
            self.profile.surface_fleet["excavator"]["payload_kg"] + 1.5,
        )

    def test_rover_requires_two_people_walkback_range_and_return_energy(self):
        one_person = self.fleet.begin_crew_rover_trip(
            expedition_id="one",
            crew_ids=["a"],
            target_x=120,
            target_y=100,
        )
        too_far = self.fleet.begin_crew_rover_trip(
            expedition_id="far",
            crew_ids=["a", "b"],
            target_x=180,
            target_y=100,
        )
        accepted = self.fleet.begin_crew_rover_trip(
            expedition_id="valid",
            crew_ids=["a", "b"],
            target_x=150,
            target_y=100,
        )

        self.assertEqual("two_person_rover_crew_required", one_person["reason"])
        self.assertEqual("outside_walkback_limit", too_far["reason"])
        self.assertTrue(accepted["reserved"])
        rover = self.fleet.crew_rovers[0]
        before = rover.battery_kwh
        self.assertTrue(
            self.fleet.record_crew_rover_movement(
                "valid", 100, 100, 108, 100
            )
        )
        self.assertLess(rover.battery_kwh, before)
        self.assertTrue(self.fleet.complete_crew_rover_trip("valid"))
        self.assertEqual("charging", rover.state)

    def test_rover_cargo_is_capacity_limited_and_explicitly_unloaded(self):
        accepted = self.fleet.begin_crew_rover_trip(
            expedition_id="haul",
            crew_ids=["a", "b"],
            target_x=118,
            target_y=100,
        )
        self.assertTrue(accepted["reserved"])

        loaded = self.fleet.load_crew_rover_payload(
            "haul", "sulfur", requested_units=400, unit_mass_kg=2.0
        )
        self.assertEqual(245, loaded["loaded_units"])
        self.assertTrue(loaded["payload_full"])
        self.assertEqual({"sulfur": 245}, self.fleet.unload_crew_rover_payload("haul"))
        self.assertEqual({}, self.fleet.crew_rovers[0].payload)
        self.assertEqual(0.0, self.fleet.crew_rovers[0].payload_mass_kg)

    def test_heavy_cargo_is_deck_limited_and_credited_only_after_unload(self):
        rejected = self.fleet.dispatch_cargo_transport(
            site_id="oversize",
            target_x=106,
            target_y=100,
            payload={"test_module": 1},
            payload_mass_kg=35_001.0,
            current_tick=1,
        )
        self.assertFalse(rejected["dispatched"])
        self.assertEqual("cargo_payload_limit_exceeded", rejected["reason"])

        accepted = self.fleet.dispatch_cargo_transport(
            site_id="site-a",
            target_x=106,
            target_y=100,
            payload={"structural_truss_section": 100},
            payload_mass_kg=4_100.0,
            current_tick=1,
        )
        self.assertTrue(accepted["dispatched"])
        delivered = []
        first = self.fleet.tick(
            available_grid_energy_kwh=100.0,
            excavation_callback=lambda *_args: (0, 0.0, False),
            unload_callback=lambda _payload: None,
        )
        self.assertFalse(any(
            event.get("type") == "cargo_delivery_complete"
            for event in first["events"]
        ))
        for _ in range(20):
            result = self.fleet.tick(
                available_grid_energy_kwh=100.0,
                excavation_callback=lambda *_args: (0, 0.0, False),
                unload_callback=lambda _payload: None,
            )
            delivered.extend(
                event for event in result["events"]
                if event.get("type") == "cargo_delivery_complete"
            )
            if delivered:
                break
        self.assertEqual({"structural_truss_section": 100}, delivered[0]["payload"])
        self.assertEqual("site-a", delivered[0]["site_id"])


class EngineMissionIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.engine = SimulationEngine(
            os.path.join(ROOT, "config", "planets", "kepler-442b.json"),
            db=False,
            max_ticks=5,
        )
        for agent in create_team_from_presets(
            os.path.join(ROOT, "config", "agent_presets.json")
        ):
            self.engine.add_agent(agent)
        self.engine._init_agents()
        self.engine._initialized = True

    def test_exhausted_flight_core_cannot_be_fabricated_from_regolith(self):
        pooled = dict(self.engine.central_depot_inventory)
        pooled.pop("preintegrated_cea_module_segment", None)

        feasibility = self.engine.decision_engine._bootstrap_order_feasibility(
            "greenhouse", pooled
        )

        self.assertFalse(feasibility["actionable"])
        self.assertEqual(
            4,
            feasibility["finite_cargo_deficits"][
                "preintegrated_cea_module_segment"
            ],
        )

    def test_construction_acceptance_is_seeded_and_rework_bounded(self):
        builders = self.engine.agents[:2]
        for builder in builders:
            builder.competency.engineering = 0
        robot_ids = [robot.id for robot in self.engine.surface_fleet.assembly_robots[:2]]
        for robot in self.engine.surface_fleet.assembly_robots[:2]:
            robot.condition = 0.50
        structure = {
            "id": "qa-low-skill-habitat",
            "type": "habitat_module",
            "acceptance_attempts": 0,
        }

        first = self.engine._construction_acceptance_record(
            structure, builders, robot_ids
        )
        repeated = self.engine._construction_acceptance_record(
            structure, builders, robot_ids
        )
        self.assertEqual(first, repeated)
        self.assertEqual("rework_required", first["status"])
        self.assertIn("pressure_test", first["checks"])

        structure["acceptance_attempts"] = 2
        third = self.engine._construction_acceptance_record(
            structure, builders, robot_ids
        )
        self.assertEqual(3, third["attempt"])
        self.assertEqual("passed", third["status"])

    def test_real_parallel_workshops_and_backend_mission_state(self):
        self.assertEqual(2, self.engine.structures_built["stone_furnace"])
        self.assertEqual(2, self.engine.structures_built["forge"])
        self.assertEqual(3, self.engine.structures_built["cnc_fabricator"])
        self.assertEqual(2, len([
            item for item in self.engine.placed_structures
            if item["type"] == "stone_furnace"
        ]))
        self.assertEqual(2, len([
            item for item in self.engine.placed_structures
            if item["type"] == "forge"
        ]))
        self.assertEqual(
            {
                "struct_cnc_fabricator_1",
                "struct_cnc_fabricator_2",
                "struct_cnc_fabricator_3",
            },
            {
                item["id"] for item in self.engine.placed_structures
                if item["type"] == "cnc_fabricator"
            },
        )

        snapshot = self.engine._get_state_snapshot()
        self.assertEqual(100, snapshot["mission"]["arriving_civilians"])
        self.assertTrue(snapshot["mission"]["cargo_mass_ledger"]["closed"])
        self.assertFalse(
            snapshot["mission"]["cargo_mass_ledger"]["readiness_credit"]
        )
        self.assertEqual(4, len(snapshot["surface_fleet"]["excavators"]))
        self.assertEqual(
            2, len(snapshot["surface_fleet"]["cargo_transporters"])
        )
        self.assertEqual(
            "dialogue_only",
            snapshot["mission"]["decision_authority"]["language_model"],
        )

    def test_low_flux_planet_requires_enough_real_grid_nodes(self):
        decision = self.engine.decision_engine
        solar = self.engine.colony_score.get_structure_capacity_status(
            "solar_panel"
        )
        self.assertEqual(26, solar["required_count"])
        self.assertEqual(
            2,
            decision._required_physical_structure_count(
                "power_distribution_grid"
            ),
        )
        counts = dict(self.engine.structures_built)
        counts["power_distribution_grid"] = 1
        self.assertEqual(
            0.5,
            decision._planning_fulfillment(
                "power_distribution_grid", None, counts
            ),
        )
        counts["power_distribution_grid"] = 2
        self.assertEqual(
            1.0,
            decision._planning_fulfillment(
                "power_distribution_grid", None, counts
            ),
        )

    def test_fluid_network_plans_finite_multi_vault_capacity(self):
        decision = self.engine.decision_engine
        self.assertEqual(
            5,
            decision._required_physical_structure_count(
                "life_support_distribution_grid"
            ),
        )
        counts = dict(self.engine.structures_built)
        counts["life_support_distribution_grid"] = 1
        self.assertAlmostEqual(
            0.2,
            decision._planning_fulfillment(
                "life_support_distribution_grid", None, counts
            ),
        )

    def test_disconnected_commissioned_endpoint_prioritizes_fourth_vault(self):
        decision = self.engine.decision_engine
        captured = {}

        class CapturePolicy:
            def choose(self, **kwargs):
                captured["candidates"] = list(kwargs["candidates"])
                return next(
                    candidate for candidate in captured["candidates"]
                    if candidate["recipe"] == "life_support_distribution_grid"
                )

        decision.strategic_policy = CapturePolicy()
        decision.life_support_network_snapshot = lambda: {
            "endpoints": [{
                "structure_id": "late-o2-unit",
                "physically_connected": False,
                "reason": "line_length_budget_exhausted",
            }]
        }
        counts = dict(self.engine.structures_built)
        counts["life_support_distribution_grid"] = 3
        order = decision._select_shared_capacity_order(
            decision._pooled_materials(), counts, tick=10_000
        )
        grid = next(
            candidate for candidate in captured["candidates"]
            if candidate["recipe"] == "life_support_distribution_grid"
        )
        self.assertEqual("life_support_distribution_grid", order["recipe"])
        self.assertTrue(grid["utility_recovery_priority"])

    def test_empty_tanks_reduce_readiness_without_triggering_overbuild(self):
        status = self.engine.colony_score.get_structure_capacity_status(
            "potable_water_tank"
        )
        counts = dict(self.engine.structures_built)
        counts["potable_water_tank"] = status["required_count"]
        self.assertEqual(0.0, status["fulfillment"])
        self.assertEqual(1.0, self.engine.decision_engine._planning_fulfillment(
            "potable_water_tank", status, counts
        ))

    def test_water_fill_deadline_accounts_for_crop_losses_and_remaining_days(self):
        from types import SimpleNamespace
        decision = self.engine.decision_engine
        previous_colony = decision.colony
        statuses = {
            "potable_water_tank": {"target_capacity": 81540.0, "current_capacity": 7700.0},
            "water_collector": {"current_capacity": 2200.0},
        }
        decision.colony = SimpleNamespace(get_structure_capacity_status=lambda name: statuses.get(name))
        counts = {"solar_panel": 20, "life_support_distribution_grid": 4,
                  "greenhouse": 5, "potable_water_tank": 4}
        try:
            forecast = decision._water_stock_deadline_status(counts, 300 * 144)
            self.assertTrue(forecast["at_risk"])
            self.assertLess(forecast["net_l_per_day"], 201.0)
            self.assertLess(forecast["projected_reserve_l"], 15000.0)
            statuses["water_collector"]["current_capacity"] = 2800.0
            self.assertFalse(decision._water_stock_deadline_status(counts, 100 * 144)["at_risk"])
            statuses["potable_water_tank"]["current_capacity"] = 81540.0
            self.assertFalse(decision._water_stock_deadline_status(counts, 330 * 144)["at_risk"])
        finally:
            decision.colony = previous_colony

    def test_pipe_length_exhaustion_reopens_grid_after_all_ports_are_funded(self):
        decision = self.engine.decision_engine
        counts = dict(self.engine.structures_built)
        counts["life_support_distribution_grid"] = 4
        decision.life_support_network_snapshot = lambda: {
            "nodes": [{"id": f"vault-{i}"} for i in range(4)],
            "endpoints": [{
                "structure_id": "last-habitat",
                "physically_connected": False,
                "reason": "line_length_budget_exhausted",
            }],
        }
        self.assertEqual(5, decision._required_physical_structure_count(
            "life_support_distribution_grid"
        ))
        self.assertAlmostEqual(0.8, decision._planning_fulfillment(
            "life_support_distribution_grid", None, counts
        ))
        _, requirements, remaining = decision._refresh_mission_forecast_recipe(counts)
        self.assertEqual(1, remaining["life_support_distribution_grid"])
        self.assertGreaterEqual(requirements["metal_pipe"], 1350)
        full_recipe, _, full_counts = decision._refresh_mission_forecast_recipe(
            self.engine.structures_built
        )
        self.assertEqual(5, full_counts["life_support_distribution_grid"])
        self.assertEqual({}, decision._finite_cargo_deficits(
            full_recipe, dict(self.engine.central_depot_inventory)
        ))

    def test_engine_and_world_share_one_solar_clock(self):
        cycle = self.engine.world.get_day_phase(0)["cycle_length_ticks"]
        for tick in (0, cycle // 4, cycle // 2, 3 * cycle // 4):
            self.engine.current_tick = tick
            engine_phase = self.engine.get_day_night_phase()
            cell = self.engine.world.get_cell_info(
                self.engine.lz_x, self.engine.lz_y, tick
            )
            self.assertAlmostEqual(
                float(cell["light"]["light_level"]),
                float(engine_phase["solar_intensity"]),
            )
            self.assertEqual(
                cell["light"]["phase_name"], engine_phase["phase"]
            )
            self.assertEqual(0.0, engine_phase["temp_modifier_c"])

    def test_maintenance_intervals_are_multi_day_not_hourly(self):
        self.assertEqual(
            4_320,
            self.engine._get_recipe("solar_panel")["maintenance"][
                "interval_ticks"
            ],
        )
        self.assertEqual(
            4_320,
            self.engine._get_recipe("habitat_module")["maintenance"][
                "interval_ticks"
            ],
        )
        self.assertEqual(
            1_008,
            self.engine._get_recipe("greenhouse")["maintenance"][
                "interval_ticks"
            ],
        )

    def test_greenhouse_maturity_requires_powered_irrigated_service_ticks(self):
        required_ticks = self.engine._greenhouse_maturity_ticks()
        self.assertEqual(4_032, required_ticks)
        greenhouse = {
            "id": "test_greenhouse",
            "type": "greenhouse",
            "x": self.engine.lz_x + 1,
            "y": self.engine.lz_y + 1,
            "health": 1.0,
            "completed_tick": 0,
            "crop_growth_ticks": required_ticks - 1,
            "last_crop_service_tick": required_ticks * 2,
        }
        self.engine.placed_structures.append(greenhouse)
        self.engine.structures_built["greenhouse"] = 1
        self.engine.current_tick = required_ticks * 2

        # Wall-clock age alone must never create a harvest or score capacity.
        self.assertEqual([], self.engine._mature_greenhouses())
        greenhouse["crop_growth_ticks"] = required_ticks
        self.assertEqual([greenhouse], self.engine._mature_greenhouses())

        self.engine._active_greenhouse_ids = set()
        self.assertEqual(
            0.0,
            self.engine._colony_capacity_overrides()["food"],
        )
        self.engine._active_greenhouse_ids = {"test_greenhouse"}
        self.assertGreater(
            self.engine._colony_capacity_overrides()["food"],
            0.0,
        )

    def test_robot_removes_real_exposed_units_and_unloads_them(self):
        x = self.engine.lz_x - self.engine.MIN_EXTRACTION_RADIUS_CELLS
        y = self.engine.lz_y
        geology = self.engine._get_cell_geology(x, y)
        layer = self.engine._current_geology_layer(x, y)
        self.assertEqual("regolith", layer["material"])
        before_access_trench = int(layer["remaining"])
        before_ground = int(layer["recoverable_remaining"])
        self.assertEqual(1_000_000, before_ground)
        before_depot = self.engine.central_depot_inventory.get("regolith", 0)
        dispatched = self.engine.surface_fleet.dispatch_excavator(
            resource="regolith",
            target_x=x,
            target_y=y,
            current_tick=1,
            dispatched_by=self.engine.agents[0].id,
            verified_exposed_face=True,
            protected_ground=False,
        )
        self.assertTrue(dispatched["dispatched"])

        for _ in range(300):
            events = self.engine._tick_surface_fleet()
            if any(e.get("type") == "excavator_job_complete" for e in events):
                break

        after_ground = int(
            geology["layers"][0]["recoverable_remaining"]
        )
        delivered = (
            self.engine.central_depot_inventory.get("regolith", 0) - before_depot
        )
        self.assertGreater(delivered, 0)
        self.assertGreater(delivered * 1.5, 900.0)
        self.assertEqual(before_ground, after_ground + delivered)
        self.assertEqual(
            before_access_trench,
            int(geology["layers"][0]["remaining"]),
        )

    def test_hard_rock_attachment_allows_but_does_not_magic_basalt(self):
        x = self.engine.lz_x - self.engine.MIN_EXTRACTION_RADIUS_CELLS
        y = self.engine.lz_y
        self.engine.cell_geology[(x, y)] = {
            "layers": [{
                "material": "basalt",
                "initial_quantity": 40,
                "remaining": 40,
                "recoverable_initial_quantity": 10_000,
                "recoverable_remaining": 10_000,
            }],
            "current_index": 0,
        }
        self.engine.discovered_resources.setdefault("basalt", set()).add((x, y))

        target = self.engine._robot_dispatch_target("basalt")
        self.assertIsNotNone(target)
        self.assertEqual("basalt", target["resource"])

        self.engine.surface_fleet.excavator_spec["hard_rock_capable"] = False
        self.engine._robot_dispatch_cache = {}
        self.assertIsNone(self.engine._robot_dispatch_target("basalt"))

    def test_bulk_order_can_dispatch_two_verified_faces_in_parallel(self):
        coordinates = [
            (
                self.engine.lz_x - self.engine.MIN_EXTRACTION_RADIUS_CELLS,
                self.engine.lz_y,
            ),
            (
                self.engine.lz_x
                - self.engine.MIN_EXTRACTION_RADIUS_CELLS - 1,
                self.engine.lz_y,
            ),
        ]
        self.engine._robot_dispatch_cache = {}
        self.engine.discovered_resources["regolith"] = set(coordinates)
        for coordinate in coordinates:
            self.engine.cell_geology[coordinate] = {
                "layers": [{
                    "material": "regolith",
                    "initial_quantity": 40,
                    "remaining": 40,
                    "recoverable_initial_quantity": 10_000,
                    "recoverable_remaining": 10_000,
                }],
                "current_index": 0,
            }

        first = self.engine._robot_dispatch_target("regolith")
        self.assertIsNotNone(first)
        result = self.engine.surface_fleet.dispatch_excavator(
            resource="regolith",
            objective_resource="regolith",
            target_x=first["x"],
            target_y=first["y"],
            current_tick=1,
            dispatched_by=self.engine.agents[0].id,
            verified_exposed_face=True,
            protected_ground=False,
        )
        self.assertTrue(result["dispatched"])

        second = self.engine._robot_dispatch_target("regolith")
        self.assertIsNotNone(second)
        self.assertNotEqual(
            (first["x"], first["y"]),
            (second["x"], second["y"]),
        )

    def test_detector_confirmed_buried_seam_keeps_one_robotic_recovery_face(self):
        """A robot strips real layers; it never teleports the buried sulfur."""
        x = self.engine.lz_x - self.engine.MIN_EXTRACTION_RADIUS_CELLS
        y = self.engine.lz_y
        layers = [
            {"material": "regolith", "initial_quantity": 4, "remaining": 4},
            {"material": "sulfur", "initial_quantity": 4, "remaining": 4},
        ]
        self.engine.cell_geology[(x, y)] = {
            "layers": layers,
            "current_index": 0,
        }
        for layer in layers:
            self.engine.cell_resource_units[(x, y, layer["material"])] = 4
        # Regolith is the visible face; sulfur is known only because a real
        # detector pass has already verified this coordinate.
        self.engine.discovered_resources.setdefault("regolith", set()).add((x, y))
        self.engine.discovered_resources.setdefault("sulfur", set()).add((x, y))
        self.engine._colony_resources["energy_stored_kwh"] = 100.0

        dispatched_layers = []
        for expected_face in ("regolith", "sulfur"):
            target = self.engine._robot_dispatch_target("sulfur")
            self.assertIsNotNone(target)
            self.assertEqual((x, y), (target["x"], target["y"]))
            self.assertEqual(expected_face, target["resource"])
            self.assertEqual("sulfur", target["requested_resource"])
            dispatched_layers.append(target["resource"])
            dispatched = self.engine.surface_fleet.dispatch_excavator(
                resource=target["resource"],
                objective_resource=target["requested_resource"],
                target_x=x,
                target_y=y,
                current_tick=self.engine.current_tick,
                dispatched_by=self.engine.agents[0].id,
                verified_exposed_face=target["verified_exposed_face"],
                protected_ground=target["protected_ground"],
            )
            self.assertTrue(dispatched["dispatched"])

            for _ in range(300):
                events = self.engine._tick_surface_fleet()
                if any(
                    event.get("type") == "excavator_job_complete"
                    and (event.get("job") or {}).get("objective_resource")
                    == "sulfur"
                    for event in events
                ):
                    break
            else:
                self.fail(f"robot did not finish the {expected_face} layer")

            if expected_face == "regolith":
                # A later detector hit that happens to be closer must not make
                # the scheduler abandon the face it already stripped.
                alternate = (self.engine.lz_x + 14, self.engine.lz_y)
                alternate_layers = [
                    {"material": "regolith", "initial_quantity": 4, "remaining": 4},
                    {"material": "sulfur", "initial_quantity": 4, "remaining": 4},
                ]
                self.engine.cell_geology[alternate] = {
                    "layers": alternate_layers,
                    "current_index": 0,
                }
                self.engine.discovered_resources.setdefault("regolith", set()).add(
                    alternate
                )
                self.engine.discovered_resources.setdefault("sulfur", set()).add(
                    alternate
                )

        self.assertEqual(["regolith", "sulfur"], dispatched_layers)
        self.assertEqual(4, self.engine.central_depot_inventory.get("sulfur", 0))
        self.assertEqual(2, self.engine.cell_geology[(x, y)]["current_index"])

    def test_llm_planning_is_disabled_and_rl_can_select_valid_robot_job(self):
        agent = self.engine.agents[0]
        agent._in_habitat = True
        agent.rl_epsilon_explore = 0.0
        self.engine.decision_engine.robot_dispatch_targets = {
            "regolith": {
                "resource": "regolith",
                "x": self.engine.lz_x + 15,
                "y": self.engine.lz_y,
                "verified_exposed_face": True,
                "protected_ground": False,
            }
        }
        decision = self.engine.decision_engine._consider_excavator_dispatch(
            agent,
            {"action": "gather", "target": {"resource": "regolith"}},
            {"colony_resources": self.engine._colony_resources},
            self.engine.structures_built,
        )

        self.assertFalse(self.engine.decision_engine.language_planning_enabled)
        self.assertEqual("dispatch_excavator", decision["action"])
        self.assertEqual(
            "gather",
            decision["target"]["human_fallback"]["action"],
        )

    def test_failed_robot_dispatch_releases_agent_to_human_gather(self):
        agent = self.engine.agents[0]
        self.engine._apply_failed_excavator_fallback(
            agent,
            {
                "resource": "regolith",
                "requested_resource": "sulfur",
                "human_fallback": {
                    "action": "gather",
                    "target": {
                        "resource": "sulfur",
                        "capacity_recipe": "water_collector",
                    },
                    "reasoning": "Continue the shared sulfur work order",
                },
            },
            "no_ready_excavator",
        )

        self.assertEqual("gather", agent.action.action_type)
        self.assertEqual(0, agent.action.ticks_remaining)
        self.assertEqual("sulfur", agent.action.target["resource"])
        self.assertEqual(
            "no_ready_excavator",
            agent.action.target["robot_dispatch_failed"],
        )
        self.assertEqual(
            "excavator_dispatch_fallback",
            agent.last_decision["source"],
        )

    def test_verified_one_kilometre_haul_uses_buddy_rover_and_unloads(self):
        lead = self.engine.agents[0]
        for crew in self.engine.agents[:2]:
            crew.x = self.engine.lz_x
            crew.y = self.engine.lz_y
            crew._in_habitat = True
            crew.needs.energy = 100.0
            crew.needs.hunger = 100.0
            crew.needs.thirst = 100.0
            crew.needs.o2_supply = 100.0
            crew.plss_co2_scrubber_pct = 100.0
            crew.plss_suit_battery_pct = 100.0
            crew.suit_integrity = 1.0
            crew.inventory.items["oxygen_canisters"] = 1
        target = (self.engine.lz_x + 18, self.engine.lz_y)
        self.engine.discovered_resources.setdefault("sulfur", set()).add(target)

        expedition = self.engine._start_expedition(lead, "sulfur", target)

        self.assertEqual("crew_rover", expedition["transport"])
        self.assertIsNotNone(expedition["buddy_id"])
        loaded = self.engine.surface_fleet.load_crew_rover_payload(
            expedition["id"], "sulfur", 34, 2.0
        )
        self.assertEqual(34, loaded["loaded_units"])
        before = self.engine.central_depot_inventory.get("sulfur", 0)
        self.assertTrue(self.engine._complete_crew_rover_expedition(expedition))
        self.assertEqual(
            before + 34,
            self.engine.central_depot_inventory.get("sulfur", 0),
        )

    def test_two_workshops_are_two_concurrent_machine_slots(self):
        first, second, third = self.engine.agents[:3]
        self.engine.decision_engine.placed_structures = self.engine.placed_structures
        first.action.action_type = "refine"
        first.action.target = {
            "completion_pending": True,
            "machine_type": "forge",
            "machine_id": "struct_cnc_forge_1",
        }
        self.assertFalse(
            self.engine.decision_engine._all_machine_slots_busy(
                "forge", self.engine.agents, third
            )
        )
        second.action.action_type = "refine"
        second.action.target = {
            "completion_pending": True,
            "machine_type": "forge",
            "machine_id": "struct_cnc_forge_2",
        }
        self.assertTrue(
            self.engine.decision_engine._all_machine_slots_busy(
                "forge", self.engine.agents, third
            )
        )

    def test_machine_route_reserves_the_selected_physical_slot(self):
        first, second, third = self.engine.agents[:3]
        self.engine.decision_engine.placed_structures = self.engine.placed_structures
        first.action.action_type = "move"
        first.action.target = {
            "destination": "manufacturing_machine",
            "machine_type": "forge",
            "machine_id": "struct_cnc_forge_1",
            "output": "metal_pipe",
        }
        second.action.action_type = "arrived"
        second.action.target = {
            "destination": "manufacturing_machine",
            "machine_type": "forge",
            "machine_id": "struct_cnc_forge_2",
            "output": "structural_truss_section",
        }

        self.assertTrue(
            self.engine.decision_engine._all_machine_slots_busy(
                "forge", self.engine.agents, third
            )
        )

    def test_machine_route_also_reserves_its_unconsumed_bom(self):
        planner = self.engine.decision_engine
        requester = self.engine.agents[0]
        route_owner = self.engine.agents[1]
        planner.placed_structures = self.engine.placed_structures
        planner.agents = self.engine.agents
        self.engine.delivered_structure_kits.clear()
        self.engine.central_depot_inventory.clear()
        self.engine.central_depot_inventory["reduced_iron_ingot"] = 2
        self.engine._colony_resources["energy_stored_kwh"] = 100.0
        planner.central_depot_inventory = self.engine.central_depot_inventory
        planner.delivered_structure_kits = self.engine.delivered_structure_kits
        planner._colony_resources = self.engine._colony_resources
        route_owner.action.action_type = "move"
        route_owner.action.target = {
            "destination": "manufacturing_machine",
            "machine_type": "cnc_fabricator",
            "machine_id": "struct_cnc_fabricator_1",
            "output": "metal_pipe",
        }
        candidate = {
            "action": "refine",
            "target": {"output": "metal_pipe"},
        }

        self.assertFalse(planner._refine_candidate_is_executable(
            candidate,
            requester,
            self.engine.agents,
            self.engine.structures_built,
            tick=10,
        ))

        route_owner.action.clear()
        self.assertTrue(planner._refine_candidate_is_executable(
            candidate,
            requester,
            self.engine.agents,
            self.engine.structures_built,
            tick=11,
        ))

    def test_shared_contract_reserves_planned_output_for_other_workers(self):
        first, second = self.engine.agents[:2]
        self.engine.decision_engine.agents = self.engine.agents
        first._shared_work_contract = {
            "recipe": "water_collector",
            "stock_key": "metal_pipe",
            "quantity_reserved": 12,
            "review_tick": 100,
        }

        reserved = self.engine.decision_engine._contract_reservations(
            "water_collector", second, tick=10
        )

        self.assertEqual({"metal_pipe": 12}, reserved)
        self.assertEqual(
            {},
            self.engine.decision_engine._contract_reservations(
                "water_collector", second, tick=101
            ),
        )

    def test_pressurized_footprint_step_does_not_toggle_eva_state(self):
        agent = self.engine.agents[0]
        agent.x = self.engine.lz_x
        agent.y = self.engine.lz_y
        agent._in_habitat = True
        agent.needs.energy = 100.0
        agent.needs.hunger = 100.0
        agent.needs.thirst = 100.0
        agent.needs.temperature_stress = 50.0
        deposit = (self.engine.lz_x + 6, self.engine.lz_y)
        self.engine.discovered_resources.setdefault("chalcopyrite_ore", set()).add(
            deposit
        )
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "gather",
            "target": {"resource": "chalcopyrite_ore"},
            "reasoning": "known deposit route",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)

        self.assertTrue(
            self.engine._is_pressurized_location(agent.x, agent.y),
            f"position=({agent.x},{agent.y}) action={agent.action.action_type}:"
            f"{agent.action.target}",
        )
        self.assertTrue(agent._in_habitat)

    def test_paused_operator_does_logistics_while_both_forges_are_busy(self):
        first, second, waiting = self.engine.agents[:3]
        decision_engine = self.engine.decision_engine
        decision_engine.placed_structures = self.engine.placed_structures
        decision_engine.structures_built = self.engine.structures_built
        decision_engine.agents = self.engine.agents
        decision_engine.central_depot_inventory = self.engine.central_depot_inventory
        decision_engine._colony_resources = {"energy_stored_kwh": 100.0}
        decision_engine.surface_requires_plss = True
        for index, worker in enumerate((first, second), start=1):
            worker.action.action_type = "refine"
            worker.action.target = {
                "completion_pending": True,
                "machine_type": "forge",
                "machine_id": f"struct_cnc_forge_{index}",
            }
        waiting._paused_manufacturing = {
            "output": "metal_pipe",
            "machine_type": "forge",
            "remaining_ticks": 20,
            "completion_pending": True,
        }
        waiting._in_habitat = True
        waiting.needs.energy = 100.0
        waiting.needs.hunger = 100.0
        waiting.needs.thirst = 100.0
        waiting.needs.temperature_stress = 50.0

        decision = decision_engine.process_tick(
            waiting,
            tick=1,
            tick_events={},
            world_context={"active_events": [], "colony_resources": {}},
            nearby_agents=[],
        )

        self.assertEqual("gather", decision["action"])
        self.assertEqual("metal_pipe", decision["target"]["paused_output"])
        self.assertTrue(decision["target"]["resume_when_machine_available"])

    def test_score_100_still_requires_orbital_communications(self):
        self.engine.colony_score._current = {
            category: float(config["target"])
            for category, config in self.engine.colony_score._targets.items()
        }
        self.engine.colony_score._communications_ready = False
        without_comms = self.engine.mission_profile.mission_state(
            1,
            self.engine.colony_score.to_dict(),
            self.engine.structures_built,
        )
        self.assertFalse(without_comms["surface_acceptance_ready"])
        self.assertEqual(90.0, self.engine.colony_score.get_overall_score())
        self.engine.structures_built["communications_array"] = 1
        self.engine.colony_score._communications_ready = True
        with_comms = self.engine.mission_profile.mission_state(
            51_839,
            self.engine.colony_score.to_dict(),
            {**self.engine.structures_built, "landing_zone": 1},
            communications_operational=True,
            landing_zone_operational=True,
            support_soak_ticks=4_320,
        )
        self.assertTrue(with_comms["surface_acceptance_ready"])

    def test_unpowered_or_disconnected_array_cannot_clear_arrival_gate(self):
        self.engine.colony_score._current = {
            category: float(config["target"])
            for category, config in self.engine.colony_score._targets.items()
        }
        self.engine.structures_built.update({
            "power_distribution_grid": 1,
            "communications_array": 1,
        })
        self.engine.placed_structures.extend([
            {
                "id": "test_grid", "type": "power_distribution_grid",
                "x": self.engine.lz_x + 2, "y": self.engine.lz_y,
                "health": 1.0,
            },
            {
                "id": "test_array", "type": "communications_array",
                "x": self.engine.lz_x + 3, "y": self.engine.lz_y,
                "health": 1.0,
            },
        ])
        self.engine._colony_resources["energy_stored_kwh"] = 0.0

        self.assertFalse(self.engine._communications_array_operational())
        self.assertEqual(
            self.engine.LOCAL_EVA_RADIUS_CELLS,
            self.engine._communication_radius_cells(),
        )
        state = self.engine.mission_profile.mission_state(
            1,
            self.engine.colony_score.to_dict(),
            self.engine.structures_built,
            communications_operational=False,
        )
        self.assertFalse(state["arrival_gates"]["communications"])

        self.engine._colony_resources["energy_stored_kwh"] = 1.0
        self.assertTrue(self.engine._communications_array_operational())
        self.assertEqual(200, self.engine._communication_radius_cells())
        next(
            item for item in self.engine.placed_structures
            if item.get("id") == "test_grid"
        )["destroyed"] = True
        self.assertFalse(self.engine._communications_array_operational())

    def test_grid_connects_extra_arrays_without_creating_energy(self):
        self.engine.get_day_night_phase = lambda: {
            "phase": "day", "solar_intensity": 1.0,
            "temp_modifier_c": 0.0,
        }
        for index in range(2):
            self.engine.placed_structures.append({
                "id": f"test_solar_{index}",
                "type": "solar_panel",
                "x": self.engine.lz_x + 5 + index * 2,
                "y": self.engine.lz_y,
                "health": 1.0,
                "dust_fouling_level": 0.0,
            })
        self.engine.structures_built["solar_panel"] = 2

        without_grid = self.engine._run_tick()["colony_production"]
        self.assertEqual(1, without_grid["connected_solar_arrays"])
        self.assertEqual(1, without_grid["unconnected_solar_arrays"])

        self.engine.placed_structures.append({
            "id": "test_grid", "type": "power_distribution_grid",
            "x": self.engine.lz_x + 2, "y": self.engine.lz_y,
            "health": 1.0,
        })
        self.engine.structures_built["power_distribution_grid"] = 1
        with_grid = self.engine._run_tick()["colony_production"]

        self.assertEqual(2, with_grid["connected_solar_arrays"])
        self.assertEqual(0, with_grid["unconnected_solar_arrays"])
        self.assertGreater(
            with_grid["gross_solar_production_kwh"],
            with_grid["solar_production_kwh"],
        )
        self.assertGreater(
            with_grid["solar_production_kwh"],
            without_grid["solar_production_kwh"],
        )

    def test_lander_solar_feeds_battery_without_counting_as_field_array(self):
        self.engine.get_day_night_phase = lambda: {
            "phase": "day", "solar_intensity": 1.0,
            "temp_modifier_c": 0.0,
        }
        production = self.engine._run_tick()["colony_production"]

        self.assertGreater(production["lander_solar_production_kwh"], 0.0)
        self.assertEqual(0.0, production["field_solar_production_kwh"])
        self.assertEqual(0, production["connected_solar_arrays"])
        self.assertEqual(0, self.engine.structures_built.get("solar_panel", 0))
        self.assertEqual(0.0, production["auxiliary_production_kwh"])
        self.assertGreater(production["energy_delta"], 0.0)
        self.assertLess(
            self.engine._colony_resources["energy_stored_kwh"], 100.0
        )

    def test_full_oxygen_tank_load_sheds_idle_isru_skids(self):
        self.engine.get_day_night_phase = lambda: {
            "phase": "day", "solar_intensity": 1.0,
            "temp_modifier_c": 0.0,
        }
        for index in range(5):
            self.engine.placed_structures.append({
                "id": f"test_isru_{index}",
                "type": "isru_o2_unit",
                "x": self.engine.lz_x + 3 + index,
                "y": self.engine.lz_y,
                "health": 1.0,
            })
        self.engine.structures_built["isru_o2_unit"] = 5
        self.engine._colony_resources["o2_reserve_kg"] = (
            self.engine._o2_storage_capacity_kg
        )

        production = self.engine._run_tick()["colony_production"]

        self.assertEqual(5, production["installed_isru_units"])
        self.assertEqual(0, production["active_isru_units"])
        self.assertFalse(production["energy_deficit"])

    def test_purifier_does_not_multiply_extracted_water(self):
        self.engine.get_day_night_phase = lambda: {
            "phase": "day", "solar_intensity": 1.0,
            "temp_modifier_c": 0.0,
        }
        # The delivered flight tank starts full, so demand-controlled OGS is
        # normally idle. Create real headroom to exercise its water ledger.
        self.engine._colony_resources["o2_reserve_kg"] = 0.0
        self.engine.placed_structures.append({
            "id": "test_purifier", "type": "water_purifier",
            "x": self.engine.lz_x + 2, "y": self.engine.lz_y,
            "health": 1.0,
        })
        self.engine.structures_built["water_purifier"] = 1

        production = self.engine._run_tick()["colony_production"]

        self.assertEqual(0.0, production["water_production_l"])
        expected_ogs_net_l = (
            4.2 / 144.0 * 1.125 * 0.50
        )
        self.assertAlmostEqual(
            expected_ogs_net_l,
            production["water_process_consumption_l"],
            places=4,
        )

    def test_purified_and_extracted_water_share_one_readiness_capacity(self):
        self.engine.colony_score.update_counts({
            "water_collector": 1,
            "water_purifier": 1,
            "potable_water_tank": 6,
        })

        snapshot = self.engine.colony_score.to_dict()
        self.assertEqual(200.0, snapshot["current_counts"]["water"])
        self.assertAlmostEqual(200.0 / 2718.0, snapshot["categories"]["water"])
        self.assertEqual(
            6,
            self.engine.colony_score.get_structure_capacity_status(
                "water_purifier"
            )["required_count"],
        )

    def test_settlement_tanks_and_fluid_assets_require_physical_utility_loop(self):
        collector = {
            "id": "networked_collector",
            "type": "water_collector",
            "x": self.engine.lz_x + 7,
            "y": self.engine.lz_y - 4,
            "health": 1.0,
        }
        water_tank = {
            "id": "networked_water_tank",
            "type": "potable_water_tank",
            "x": self.engine.lz_x + 6,
            "y": self.engine.lz_y - 5,
            "health": 1.0,
        }
        oxygen_tank = {
            "id": "networked_oxygen_tank",
            "type": "oxygen_buffer_tank",
            "x": self.engine.lz_x + 6,
            "y": self.engine.lz_y - 3,
            "health": 1.0,
        }
        self.engine.placed_structures.extend([
            collector, water_tank, oxygen_tank
        ])

        self.assertFalse(self.engine._life_support_connected(collector))
        self.assertEqual(
            0.0,
            self.engine._settlement_storage_capacity(
                "potable_water_tank",
                "potable_water_storage_capacity_liters",
            ),
        )

        utility = {
            "id": "fluid_grid",
            "type": "life_support_distribution_grid",
            "x": self.engine.lz_x + 4,
            "y": self.engine.lz_y - 4,
            "health": 1.0,
            "under_construction": True,
            "construction_phase": "sectional_pressure_test",
        }
        self.engine.placed_structures.append(utility)

        # Pipe inventory at an unfinished trench is not a fluid connection.
        self.assertFalse(self.engine._life_support_connected(collector))
        incomplete_network = self.engine._life_support_network_snapshot()
        self.assertFalse(incomplete_network["transfer_enabled"])
        self.assertEqual([], incomplete_network["segments"])

        utility["under_construction"] = False
        utility["construction_phase"] = "commissioned"
        utility["completed_tick"] = 200
        utility["commissioned_tick"] = 200

        self.assertTrue(self.engine._life_support_connected(collector))
        self.assertEqual(
            15_000.0,
            self.engine._settlement_storage_capacity(
                "potable_water_tank",
                "potable_water_storage_capacity_liters",
            ),
        )
        self.assertEqual(
            2_000.0,
            self.engine._settlement_storage_capacity(
                "oxygen_buffer_tank", "oxygen_storage_capacity_kg"
            ),
        )

        network = self.engine._life_support_network_snapshot()
        self.assertEqual("buried_radial_spine", network["topology"])
        self.assertTrue(network["transfer_enabled"])
        self.assertEqual(
            "accepted_grid_and_endpoint_commissioning_plus_bus_power",
            network["activation_gate"],
        )
        self.assertFalse(network["redundant_loop"])
        self.assertGreater(len(network["segments"]), 0)
        self.assertLessEqual(
            network["installed_length_m"], network["maximum_length_m"]
        )
        self.assertEqual(16, network["maximum_endpoints"])
        for route in network["routes"]:
            for first, second in zip(route["path"], route["path"][1:]):
                self.assertEqual(
                    1,
                    abs(first["x"] - second["x"])
                    + abs(first["y"] - second["y"]),
                )
        self.assertEqual(
            network,
            self.engine._get_state_snapshot()["utility_network"],
        )

        utility["destroyed"] = True
        self.assertFalse(self.engine._life_support_connected(collector))
        self.assertEqual(
            0.0,
            self.engine._settlement_storage_capacity(
                "oxygen_buffer_tank", "oxygen_storage_capacity_kg"
            ),
        )

    def test_storage_module_counts_derive_from_thirty_day_targets(self):
        self.assertEqual(
            6,
            self.engine.colony_score.get_structure_capacity_status(
                "potable_water_tank"
            )["required_count"],
        )
        self.assertEqual(
            2,
            self.engine.colony_score.get_structure_capacity_status(
                "oxygen_buffer_tank"
            )["required_count"],
        )

    def test_readiness_credits_actual_settlement_inventory_not_empty_tanks(self):
        self.engine.placed_structures.extend([
            {
                "id": "fluid-grid", "type": "life_support_distribution_grid",
                "x": self.engine.lz_x + 4, "y": self.engine.lz_y - 4,
                "health": 1.0,
            },
            {
                "id": "water-tank", "type": "potable_water_tank",
                "x": self.engine.lz_x + 6, "y": self.engine.lz_y - 4,
                "health": 1.0,
            },
            {
                "id": "oxygen-tank", "type": "oxygen_buffer_tank",
                "x": self.engine.lz_x + 5, "y": self.engine.lz_y - 6,
                "health": 1.0,
            },
        ])
        self.engine._colony_resources["water_reserve_l"] = (
            self.engine._lander_water_storage_capacity_l
        )
        self.engine._colony_resources["o2_reserve_kg"] = (
            self.engine._lander_o2_storage_capacity_kg
        )

        empty = self.engine._colony_capacity_overrides()
        self.assertEqual(0.0, empty["water"]["water_storage"])
        self.assertEqual(0.0, empty["o2"]["oxygen_storage"])

        self.engine._colony_resources["water_reserve_l"] += 1_234.0
        self.engine._colony_resources["o2_reserve_kg"] += 100.0
        partial = self.engine._colony_capacity_overrides()
        self.assertEqual(1_234.0, partial["water"]["water_storage"])
        self.assertEqual(100.0, partial["o2"]["oxygen_storage"])

    def test_communications_build_requires_grid_and_consumes_one_radio(self):
        builder = self.engine.agents[0]
        builder.x = self.engine.lz_x
        builder.y = self.engine.lz_y
        builder._in_habitat = True
        builder.competency.engineering = 10
        builder.competency.physics = 10
        builder.inventory.items["multitool_kit"] = 1
        builder.tick_update = lambda **_kwargs: {
            "warnings": [], "died": False, "action_completed": False,
        }
        self.engine._select_structure_site = lambda _recipe: (
            builder.x, builder.y
        )
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "build",
            "target": {"recipe": "communications_array"},
            "reasoning": "test",
        }
        before_beacon = self.engine.central_depot_inventory.get(
            "emergency_beacon_parts", 0
        )

        self.engine._process_agent_tick(builder, [], nearby_count=0)

        self.assertEqual("craft_blocked", builder.action.action_type)
        self.assertEqual(
            "missing_operational_structure", builder.action.target["reason"]
        )
        self.assertEqual(
            before_beacon,
            self.engine.central_depot_inventory.get(
                "emergency_beacon_parts", 0
            ),
        )

        self.engine.placed_structures.append({
            "id": "test_grid", "type": "power_distribution_grid",
            "x": self.engine.lz_x + 1, "y": self.engine.lz_y,
            "health": 1.0,
        })
        self.engine.structures_built["power_distribution_grid"] = 1
        recipe = self.engine._get_recipe("communications_array")
        for material, needed in recipe["materials"].items():
            if material == "emergency_beacon_parts":
                continue
            self.engine.central_depot_inventory[material] = (
                self.engine.central_depot_inventory.get(material, 0)
                + int(needed) + 100
            )
        builder.action.clear()

        self.engine._process_agent_tick(builder, [], nearby_count=0)

        self.assertEqual("stage_materials", builder.action.action_type)
        self.assertTrue(any(
            site.get("type") == "communications_array"
            and site.get("under_construction", False)
            for site in self.engine.placed_structures
        ))
        site = next(
            site for site in self.engine.placed_structures
            if site.get("type") == "communications_array"
            and site.get("under_construction", False)
        )
        self.assertFalse(site["materials_committed"])
        for tick in range(20):
            self.engine.current_tick = tick
            self.engine._dispatch_construction_cargo()
            self.engine._tick_surface_fleet()
            if site.get("materials_committed"):
                break
        self.assertTrue(site["materials_committed"])
        builder.action.clear()
        self.engine._process_agent_tick(builder, [], nearby_count=0)
        self.assertEqual("build", builder.action.action_type)
        self.assertEqual(
            0,
            self.engine.central_depot_inventory.get(
                "emergency_beacon_parts", 0
            ),
        )

    def test_optional_sulfur_is_not_backfilled_as_hidden_mission_cargo(self):
        self.assertIn("sulfur", self.engine.planetary_resources)
        self.assertNotIn("sulfur", self.engine.central_depot_inventory)

        proxima = SimulationEngine(
            os.path.join(
                ROOT, "config", "planets", "proxima-centauri-b.json"
            ),
            db=False,
            max_ticks=1,
        )
        self.assertNotIn("sulfur", proxima.planetary_resources)
        self.assertNotIn("sulfur", proxima.central_depot_inventory)
        self.assertEqual({}, proxima.delivered_contingency_feedstocks)


if __name__ == "__main__":
    unittest.main()
