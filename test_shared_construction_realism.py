"""Regression tests for coordinated, mass-balanced colony construction."""

import unittest
from pathlib import Path

from src.agents.agent import AgentStatus, MATERIAL_DENSITY_KG, create_team_from_presets
from src.memory import vector_store
from src.orchestration.engine import SimulationEngine


ROOT = Path(__file__).resolve().parent
vector_store._use_tfidf_fallback = True


class SharedConstructionRealismTest(unittest.TestCase):
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
        self.engine.decision_engine.agents = self.crew
        self.engine.decision_engine.central_depot_inventory = (
            self.engine.central_depot_inventory
        )
        self.engine.decision_engine.placed_structures = self.engine.placed_structures
        self.engine.decision_engine.structures_built = self.engine.structures_built
        self.engine.decision_engine.lz_x = self.engine.lz_x
        self.engine.decision_engine.lz_y = self.engine.lz_y
        self.engine.decision_engine.surface_requires_plss = True
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
        """Give the shared policy one explicit learned preference for a test."""
        planner = self.engine.decision_engine
        planner.colony.update_counts(structures)
        policy = planner.strategic_policy
        policy.commitment = None
        policy.initial_epsilon = 0.0
        policy.minimum_epsilon = 0.0
        state = planner.get_colony_strategy_state(structures)
        policy.q_table[state] = {policy.action_key(recipe): 100.0}

    def test_solar_bill_of_materials_is_an_array_not_loose_scrap(self):
        recipe = self.engine._get_recipe("solar_panel")
        mass_kg = sum(
            quantity * MATERIAL_DENSITY_KG[material]
            for material, quantity in recipe["materials"].items()
        )

        self.assertGreater(mass_kg, 45_000.0)
        self.assertLess(mass_kg, 50_000.0)
        self.assertEqual(120, recipe["construction"]["work_person_hours"])
        self.assertEqual(
            1800, recipe["materials"]["qualified_pv_blanket_segment"]
        )
        self.assertGreaterEqual(recipe["materials"]["machined_bolts_fasteners"], 20)
        self.assertEqual(4, recipe["materials"]["solar_tracking_drive_core"])
        self.assertEqual(4, recipe["materials"]["power_conditioning_unit_core"])

    def test_full_bom_projection_exposes_late_raw_feedstocks(self):
        deficits = self.engine.decision_engine._raw_bom_deficits(
            "water_collector", {}
        )

        # Carbon is the reducing feedstock in the local metal chain. Sulfur is
        # deliberately absent: it is not a chemically valid iron reductant.
        self.assertNotIn("sulfur", deficits)
        self.assertGreater(deficits["graphite"], 0)
        self.assertGreater(deficits["iron_ore"], 0)
        self.assertGreaterEqual(deficits["basalt"], 40)

    def test_full_bom_projection_reuses_surplus_batch_output(self):
        planner = self.engine.decision_engine
        original = planner._recipes_cache
        planner._recipes_cache = {
            "test_structure": {
                "materials": {"branch_a": 1, "branch_b": 1},
                "output": {"type": "structure"},
            },
            "branch_a": {
                "materials": {"shared_part": 3},
                "output": {"type": "material", "quantity": 1},
            },
            "branch_b": {
                "materials": {"shared_part": 1},
                "output": {"type": "material", "quantity": 1},
            },
            "shared_part": {
                "materials": {"iron_ore": 10},
                "output": {"type": "material", "quantity": 4},
            },
        }
        try:
            deficits = planner._raw_bom_deficits("test_structure", {})
        finally:
            planner._recipes_cache = original

        # One four-part batch supplies both sibling branches. Ordering two
        # batches here would violate mass balance and waste an expedition.
        self.assertEqual({"iron_ore": 10}, deficits)

    def test_remaining_mission_manifest_exposes_year_start_shop_deficits(self):
        planner = self.engine.decision_engine
        recipe_name, requirements, remaining = (
            planner._refresh_mission_forecast_recipe(
                self.engine.structures_built
            )
        )
        pooled = planner._materials_available_for_recipe(
            recipe_name,
            planner._pooled_materials(),
            self.engine.structures_built,
        )
        actions = planner._shared_dependency_actions(recipe_name, pooled)
        refine_outputs = {
            action.get("target", {}).get("output")
            for action in actions
            if action.get("action") == "refine"
        }

        self.assertEqual(26, remaining["solar_panel"])
        self.assertEqual(4, remaining["life_support_distribution_grid"])
        self.assertGreater(
            requirements["metal_pipe"], pooled.get("metal_pipe", 0)
        )
        self.assertGreater(
            requirements["vacuum_gasket_seal"],
            pooled.get("vacuum_gasket_seal", 0),
        )
        self.assertIn("metal_pipe", refine_outputs)
        self.assertIn("vacuum_gasket_seal", refine_outputs)
        self.assertEqual(
            {}, planner._finite_cargo_deficits(recipe_name, pooled)
        )

    def test_mission_manifest_subtracts_materials_committed_to_active_site(self):
        planner = self.engine.decision_engine
        _, before, before_counts = planner._refresh_mission_forecast_recipe({})
        self.engine.placed_structures.append({
            "id": "committed_water_site",
            "type": "water_collector",
            "under_construction": True,
            "destroyed": False,
            "materials_committed": True,
        })

        _, after, after_counts = planner._refresh_mission_forecast_recipe({})
        unit = planner._recipes_cache["water_collector"]["materials"]

        self.assertEqual(
            before_counts["water_collector"] - 1,
            after_counts["water_collector"],
        )
        for material, quantity in unit.items():
            self.assertEqual(
                before[material] - int(quantity), after[material], material
            )

    def test_support_queue_uses_downstream_critical_path_not_json_order(self):
        planner = self.engine.decision_engine
        original = planner._recipes_cache
        planner._recipes_cache = {
            "critical_structure": {
                "materials": {"short_part": 1, "long_part": 1},
                "base_duration_ticks": 10,
                "output": {"type": "structure"},
            },
            "short_part": {
                "materials": {"basalt": 1},
                "base_duration_ticks": 2,
                "output": {"type": "material", "quantity": 1},
            },
            "long_part": {
                "materials": {"long_subassembly": 1},
                "base_duration_ticks": 50,
                "output": {"type": "material", "quantity": 1},
            },
            "long_subassembly": {
                "materials": {"iron_ore": 1},
                "base_duration_ticks": 60,
                "output": {"type": "material", "quantity": 1},
            },
        }
        try:
            actions = planner._shared_dependency_actions(
                "critical_structure", {}
            )
        finally:
            planner._recipes_cache = original

        self.assertEqual("iron_ore", actions[0]["target"]["resource"])
        self.assertGreater(
            actions[0]["target"]["critical_path_ticks"],
            next(
                action["target"]["critical_path_ticks"]
                for action in actions
                if action["target"].get("resource") == "basalt"
            ),
        )

    def test_future_utility_shortage_precedes_optional_component_stock(self):
        planner = self.engine.decision_engine
        forecast = planner.MISSION_FORECAST_RECIPE
        planner._recipes_cache[forecast] = {
            "materials": {"electronic_component": 2, "metal_pipe": 4},
            "output": {"type": "planning_manifest"},
        }
        planner._recipes_cache["__remaining_utility_supply__"] = {
            "materials": {"metal_pipe": 4},
            "output": {"type": "planning_manifest"},
        }
        pool = {"reduced_iron_ingot": 20, "qualified_control_board_core": 2}
        actions = planner._shared_dependency_actions(forecast, pool)
        refines = [action for action in actions if action["action"] == "refine"]
        self.assertEqual("metal_pipe", refines[0]["target"]["output"])
        self.assertTrue(refines[0]["target"]["utility_supply_priority"])
        # Real delivered inventory or machine-owned planned WIP satisfies the
        # debt; no second pipe batch is requested once that quantity is covered.
        actions = planner._shared_dependency_actions(forecast, {**pool, "metal_pipe": 4})
        self.assertFalse(any(a["target"].get("utility_supply_priority") for a in actions))
        self.assertFalse(any(a["target"].get("output") == "metal_pipe" for a in actions))

    def test_staged_critical_batch_keeps_workshop_busy_during_raw_search(self):
        planner = self.engine.decision_engine
        operator = max(
            self.crew,
            key=lambda crew: int(crew.competency.engineering),
        )
        planner._colony_resources = {"energy_stored_kwh": 100.0}
        actions = [
            {
                "action": "refine",
                "target": {
                    "output": "electronic_component",
                    "critical_path_ticks": 144,
                },
            },
            {
                "action": "gather",
                "target": {
                    "resource": "iron_ore",
                    "critical_path_ticks": 100,
                },
            },
        ]

        ready = planner._executable_refine_support_actions(
            actions,
            operator,
            self.crew,
            {"qualified_control_board_core": 1},
        )

        self.assertEqual("electronic_component", ready[0]["target"]["output"])

        cnc_ids = [
            structure["id"]
            for structure in self.engine.placed_structures
            if structure["type"] == "cnc_fabricator"
        ]
        blockers = [
            crew for crew in self.crew if crew.id != operator.id
        ][:len(cnc_ids)]
        for blocker, machine_id in zip(blockers, cnc_ids):
            blocker.action.action_type = "refine"
            blocker.action.target = {
                "completion_pending": True,
                "machine_type": "cnc_fabricator",
                "machine_id": machine_id,
            }
        self.assertEqual(
            [],
            planner._executable_refine_support_actions(
                actions,
                operator,
                self.crew,
                {"qualified_control_board_core": 1},
            ),
        )

    def test_non_engineer_receives_shared_bill_of_material_task(self):
        non_engineer = self.crew[-1]
        x, y = self.engine._select_structure_site("solar_panel")
        self.engine.placed_structures.append({
            "id": "shared_solar_site", "type": "solar_panel",
            "x": x, "y": y, "under_construction": True,
            "destroyed": False, "progress": 0.0,
        })
        decision = self.engine.decision_engine._shared_colony_work_decision(
            non_engineer, tick=1, structures_built=self.engine.structures_built
        )

        self.assertIsNotNone(decision)
        self.assertTrue(decision["target"]["shared_work_order"])
        self.assertEqual("build", decision["action"])
        self.assertIn("solar_panel", decision["reasoning"])

    def test_construction_crew_is_not_reselected_while_walking_to_site(self):
        planner = self.engine.decision_engine
        x, y = self.engine._select_structure_site("water_collector")
        self.engine.placed_structures.append({
            "id": "stable_water_site", "type": "water_collector",
            "x": x, "y": y, "under_construction": True,
            "destroyed": False, "progress": 0.1,
        })

        for crew in self.crew:
            planner._shared_colony_work_decision(
                crew, tick=100, structures_built=self.engine.structures_built
            )
        original_ids = list(planner.shared_work_order["construction_crew_ids"])

        # Reverse the distance ranking. A position-based scheduler would now
        # swap crews and make the astronauts already en route turn around.
        for crew in self.crew:
            if crew.id in original_ids:
                crew.x, crew.y = self.engine.lz_x, self.engine.lz_y
            else:
                crew.x, crew.y = x, y
        for crew in self.crew:
            planner._shared_colony_work_decision(
                crew, tick=101, structures_built=self.engine.structures_built
            )

        self.assertEqual(
            original_ids,
            planner.shared_work_order["construction_crew_ids"],
        )
        self.assertTrue(all(
            planner.shared_work_order["assignments"][crew_id] == "construction"
            for crew_id in original_ids
        ))

        removed_id = original_ids[-1]
        next(crew for crew in self.crew if crew.id == removed_id).status = (
            AgentStatus.INCAPACITATED
        )
        for crew in self.crew:
            if crew.status == AgentStatus.INCAPACITATED:
                continue
            planner._shared_colony_work_decision(
                crew, tick=102, structures_built=self.engine.structures_built
            )
        replacement_ids = planner.shared_work_order["construction_crew_ids"]
        self.assertNotIn(removed_id, replacement_ids)
        self.assertEqual(len(original_ids), len(replacement_ids))

    def test_sleeping_site_owner_gets_a_temporary_shift_reliever(self):
        planner = self.engine.decision_engine
        # Local walk-up crews can change shifts individually. Remote rover
        # pairs deliberately retain their shared departure reservation.
        x, y = self.engine.lz_x + 3, self.engine.lz_y
        self.engine.placed_structures.append({
            "id": "shift_water_site", "type": "water_collector",
            "x": x, "y": y, "under_construction": True,
            "destroyed": False, "progress": 0.1,
        })
        for crew in self.crew:
            planner._shared_colony_work_decision(
                crew, tick=200, structures_built=self.engine.structures_built
            )
        primary_ids = list(
            planner.shared_work_order["construction_crew_ids"]
        )
        sleeping = next(crew for crew in self.crew if crew.id == primary_ids[0])
        sleeping.action.action_type = "sleep"
        sleeping.needs.energy = 50.0

        for crew in self.crew:
            planner._shared_colony_work_decision(
                crew, tick=201, structures_built=self.engine.structures_built
            )

        self.assertEqual(
            primary_ids,
            planner.shared_work_order["construction_crew_ids"],
        )
        active_ids = {
            crew_id for crew_id, role
            in planner.shared_work_order["assignments"].items()
            if role == "construction"
        }
        self.assertNotIn(sleeping.id, active_ids)
        self.assertEqual(3, len(active_ids))

    def _stage_isru_with_electronics_deficit(self, missing_components=3):
        """Stage every ISRU top-level part except a bounded electronics gap."""
        planner = self.engine.decision_engine
        planner.delivered_structure_kits = {}
        self.engine.central_depot_inventory.clear()
        for crew in self.crew:
            crew.inventory.materials.clear()
            crew._shared_work_contract = None
            crew.action.clear()
        requirements = dict(planner._recipes_cache["isru_o2_unit"]["materials"])
        requirements["electronic_component"] -= missing_components
        self.engine.central_depot_inventory.update(requirements)
        # Electronics are reconfigured from finite flight-qualified boards;
        # the crew cannot manufacture semiconductors from raw ore.
        batches = missing_components
        self.engine.central_depot_inventory[
            "qualified_control_board_core"
        ] = batches
        self.engine._colony_resources["energy_stored_kwh"] = 1000.0
        planner._colony_resources = self.engine._colony_resources

    def test_active_site_support_uses_idle_forge_for_next_milestone(self):
        planner = self.engine.decision_engine
        self._stage_isru_with_electronics_deficit(missing_components=3)
        x, y = self.engine._select_structure_site("water_collector")
        site = {
            "id": "parallel-water-site", "type": "water_collector",
            "x": x, "y": y, "under_construction": True,
            "destroyed": False, "progress": 0.2,
        }
        self.engine.placed_structures.append(site)
        self.engine.structures_built["solar_panel"] = 1

        decisions = {}
        for crew in self.crew:
            decisions[crew.id] = planner._shared_colony_work_decision(
                crew, tick=200, structures_built=self.engine.structures_built
            )

        support_worker = next(
            crew for crew in self.crew
            if decisions[crew.id]
            and decisions[crew.id]["action"] == "refine"
            and decisions[crew.id]["target"].get("output")
            == "electronic_component"
        )
        first_contract_id = support_worker._shared_work_contract["id"]
        self.assertEqual(
            "parallel-water-site",
            decisions[support_worker.id]["target"]["construction_support_for"],
        )
        self.assertEqual("water_collector", support_worker._shared_work_contract["recipe"])
        self.assertEqual("water_collector", planner.shared_work_order["recipe"])
        self.assertEqual("construction", planner.shared_work_order["stage"])
        self.assertEqual("parallel-water-site", planner.shared_work_order["site_id"])

        continued = planner._shared_colony_work_decision(
            support_worker, tick=201, structures_built=self.engine.structures_built
        )
        self.assertEqual("refine", continued["action"])
        self.assertEqual(first_contract_id, continued["target"]["work_contract_id"])
        self.assertEqual("water_collector", planner.shared_work_order["recipe"])
        self.assertEqual("construction", planner.shared_work_order["stage"])

    def test_two_forges_receive_distinct_single_batch_reservations(self):
        planner = self.engine.decision_engine
        self._stage_isru_with_electronics_deficit(missing_components=2)
        workers = [self.crew[2], self.crew[3]]
        assignments = {}
        decisions = []
        for worker in workers:
            decisions.append(planner._shared_supply_decision(
                agent=worker,
                tick=300,
                structures_built=self.engine.structures_built,
                living=self.crew,
                recipe_name="isru_o2_unit",
                order={"recipe": "isru_o2_unit", "required_count": 1},
                assignments=assignments,
            ))

        self.assertEqual(["refine", "refine"], [d["action"] for d in decisions])
        contracts = [worker._shared_work_contract for worker in workers]
        self.assertEqual(
            ["electronic_component", "electronic_component"],
            [contract["stock_key"] for contract in contracts],
        )
        self.assertEqual([1, 1], [contract["quantity_reserved"] for contract in contracts])
        self.assertEqual([3, 4], [contract["target_stock"] for contract in contracts])
        self.assertTrue(all(
            contract["shift_end_tick"] - contract["shift_start_tick"]
            == self.engine.MANUFACTURING_SHIFT_TICKS
            for contract in contracts
        ))

    def test_one_machine_gets_only_one_same_tick_refine_contract(self):
        planner = self.engine.decision_engine
        self._stage_isru_with_electronics_deficit(missing_components=2)
        cnc_machines = [
            structure for structure in self.engine.placed_structures
            if structure["type"] == "cnc_fabricator"
        ]
        self.assertGreaterEqual(len(cnc_machines), 2)
        retained_machine = cnc_machines[0]
        self.engine.placed_structures[:] = [
            structure for structure in self.engine.placed_structures
            if structure["type"] != "cnc_fabricator"
            or structure["id"] == retained_machine["id"]
        ]
        workers = [self.crew[2], self.crew[3]]
        assignments = {}
        depot_before = dict(self.engine.central_depot_inventory)
        energy_before = float(
            self.engine._colony_resources["energy_stored_kwh"]
        )

        decisions = [
            planner._shared_supply_decision(
                agent=worker,
                tick=300,
                structures_built=self.engine.structures_built,
                living=self.crew,
                recipe_name="isru_o2_unit",
                order={"recipe": "isru_o2_unit", "required_count": 1},
                assignments=assignments,
            )
            for worker in workers
        ]

        refine_decisions = [
            decision for decision in decisions
            if decision["action"] == "refine"
        ]
        refine_contracts = [
            worker._shared_work_contract for worker in workers
            if worker._shared_work_contract["action"] == "refine"
        ]
        self.assertEqual(1, len(refine_decisions))
        self.assertEqual(1, len(refine_contracts))
        self.assertEqual("cnc_fabricator", refine_contracts[0]["machine_type"])
        self.assertEqual(300, refine_contracts[0]["slot_claim_tick"])
        self.assertEqual(depot_before, self.engine.central_depot_inventory)
        self.assertEqual(
            energy_before,
            self.engine._colony_resources["energy_stored_kwh"],
        )

        # The anonymous planning claim expires after its own tick. A real
        # route, not the old contract, owns the physical slot thereafter.
        owner = next(
            worker for worker in workers
            if worker._shared_work_contract["action"] == "refine"
        )
        requester = next(worker for worker in workers if worker.id != owner.id)
        owner.action.clear()
        self.assertFalse(planner._all_machine_slots_busy(
            "cnc_fabricator", self.crew, requester, tick=301
        ))
        owner.action.action_type = "move"
        owner.action.target = {
            "destination": "manufacturing_machine",
            "machine_type": "cnc_fabricator",
            "machine_id": retained_machine["id"],
            "output": "electronic_component",
        }
        self.assertTrue(planner._all_machine_slots_busy(
            "cnc_fabricator", self.crew, requester, tick=301
        ))

    def test_persistent_contract_replans_when_dependency_machine_is_full(self):
        planner = self.engine.decision_engine
        planner.delivered_structure_kits = {}
        self.engine.central_depot_inventory.clear()
        for crew in self.crew:
            crew.inventory.materials.clear()
            crew._shared_work_contract = None
            crew.action.clear()

        habitat_bom = dict(planner._recipes_cache["habitat_module"]["materials"])
        habitat_bom["pressure_vessel_section"] -= 1
        self.engine.central_depot_inventory.update(habitat_bom)
        # The next pressure-section batch is blocked by a missing seal, while
        # the seal feedstock itself is physically staged and manufacturable.
        self.engine.central_depot_inventory["fluoroelastomer_seal_stock"] = 1
        self.engine._colony_resources["energy_stored_kwh"] = 1000.0
        planner._colony_resources = self.engine._colony_resources

        cnc_machines = [
            structure for structure in self.engine.placed_structures
            if structure["type"] == "cnc_fabricator"
        ]
        self.assertGreaterEqual(len(cnc_machines), 2)
        machine_owners = self.crew[:len(cnc_machines)]
        for owner, machine in zip(machine_owners, cnc_machines):
            owner.action.action_type = "move"
            owner.action.target = {
                "destination": "manufacturing_machine",
                "machine_type": "cnc_fabricator",
                "machine_id": machine["id"],
                "output": "metal_pipe",
            }

        worker = self.crew[len(cnc_machines)]
        worker._shared_work_contract = {
            "id": "blocked-pressure-section",
            "recipe": "habitat_module",
            "action": "refine",
            "stock_key": "pressure_vessel_section",
            "target": {"output": "pressure_vessel_section"},
            "target_stock": 12,
            "quantity_reserved": 1,
            "review_tick": 500,
            "shift_end_tick": 500,
        }

        decision = planner._shared_supply_decision(
            agent=worker,
            tick=300,
            structures_built=self.engine.structures_built,
            living=self.crew,
            recipe_name="habitat_module",
            order={"recipe": "habitat_module", "required_count": 9},
            assignments={},
        )

        self.assertFalse(
            decision["action"] == "refine"
            and decision.get("target", {}).get("output")
            == "vacuum_gasket_seal"
        )
        replacement = getattr(worker, "_shared_work_contract", None)
        self.assertTrue(
            replacement is None
            or replacement.get("id") != "blocked-pressure-section"
        )

    def test_running_machine_batch_survives_contract_review_timeout(self):
        planner = self.engine.decision_engine
        self._stage_isru_with_electronics_deficit(missing_components=3)
        worker, requester = self.crew[2], self.crew[3]
        worker._shared_work_contract = {
            "id": "live-batch", "recipe": "isru_o2_unit",
            "action": "refine", "stock_key": "electronic_component",
            "target": {"output": "electronic_component"},
            "target_stock": 4, "quantity_reserved": 1,
            "review_tick": 10,
        }
        worker.action.action_type = "refine"
        worker.action.target = {
            "output": "electronic_component", "completion_pending": True,
        }
        worker.action.ticks_remaining = 50

        reserved = planner._contract_reservations(
            "isru_o2_unit", requester, tick=500
        )

        self.assertIsNotNone(worker._shared_work_contract)
        self.assertEqual(1, reserved["electronic_component"])

    def test_shared_material_assignment_persists_to_a_bounded_stock_target(self):
        worker = self.crew[1]
        for agent in self.crew:
            agent.inventory.materials.clear()
        self.engine.central_depot_inventory.clear()
        planner = self.engine.decision_engine
        planner._select_shared_capacity_order = lambda *_args: {
            "recipe": "water_collector",
            "required_count": 1,
            "category": "water",
        }

        first = planner._shared_colony_work_decision(
            worker, tick=100, structures_built=self.engine.structures_built
        )
        contract = dict(worker._shared_work_contract)
        second = planner._shared_colony_work_decision(
            worker, tick=101, structures_built=self.engine.structures_built
        )

        self.assertIsNotNone(contract)
        self.assertEqual(first["action"], second["action"])
        self.assertEqual(
            first["target"].get("resource") or first["target"].get("output"),
            second["target"].get("resource") or second["target"].get("output"),
        )
        self.assertEqual(contract["id"], second["target"]["work_contract_id"])
        self.assertLessEqual(contract["target_stock"], 12)

    def test_ready_starter_solar_bypasses_blocked_water_head_of_line(self):
        planner = self.engine.decision_engine
        planner.planetary_resources = set(self.engine.planetary_resources)
        planner.remote_resource_requests = {"basalt"}
        planner.local_survey_exhausted_resources = {"basalt"}
        self._prefer_shared_strategy("solar_panel", {})

        order = planner._select_shared_capacity_order(
            planner._pooled_materials(), {}
        )
        recipe = planner._recipes_cache["solar_panel"]
        lead = max(
            (
                crew for crew in self.crew
                if crew.can_craft({**recipe, "materials": {}}).get(
                    "can_craft", False
                )
            ),
            key=lambda crew: (
                int(crew.competency.engineering),
                int(crew.genome.strength),
                crew.id,
            ),
        )
        decision = planner._shared_colony_work_decision(
            lead, tick=100, structures_built={}
        )

        self.assertEqual("solar_panel", order["recipe"])
        self.assertTrue(order["ready"])
        self.assertEqual("build", decision["action"])
        self.assertEqual("solar_panel", decision["target"]["recipe"])

    def test_rejected_site_is_replanned_until_campus_changes(self):
        planner = self.engine.decision_engine
        self._prefer_shared_strategy("solar_panel", {})
        planner.mark_capacity_site_unavailable("solar_panel")
        order = planner._select_shared_capacity_order(planner._pooled_materials(), {}, tick=1)
        self.assertIsNotNone(order)
        self.assertNotEqual("solar_panel", order["recipe"])

        self.engine.placed_structures.append({
            "id": "new-grid", "type": "power_distribution_grid",
            "x": self.engine.lz_x + 8, "y": self.engine.lz_y + 8,
        })
        self._prefer_shared_strategy("solar_panel", {})
        order = planner._select_shared_capacity_order(planner._pooled_materials(), {}, tick=2)
        self.assertEqual("solar_panel", order["recipe"])

    def test_delivered_solar_kit_is_visible_only_to_solar_bom(self):
        planner = self.engine.decision_engine
        solar_materials = dict(
            planner._recipes_cache["solar_panel"]["materials"]
        )
        planner.delivered_structure_kits = {
            "solar_panel": solar_materials,
        }
        pooled = planner._pooled_materials()

        solar_pool = planner._materials_available_for_recipe(
            "solar_panel", pooled, {}
        )
        water_pool = planner._materials_available_for_recipe(
            "water_collector", pooled, {}
        )

        self.assertEqual(
            pooled["qualified_pv_blanket_segment"],
            solar_pool["qualified_pv_blanket_segment"],
        )
        for material in (
            "qualified_pv_blanket_segment",
            "electronic_component",
            "machined_bolts_fasteners",
        ):
            self.assertEqual(
                pooled.get(material, 0) - solar_materials[material],
                water_pool.get(material, 0),
            )

    def test_blocked_water_allows_actionable_bootstrap_material_work(self):
        planner = self.engine.decision_engine
        # Isolate bootstrap feasibility from the separately-tested mandatory
        # radiation-shelter safety mask.
        planner.severe_radiation_environment = False
        planner.planetary_resources = set(self.engine.planetary_resources)
        planner.remote_resource_requests = {"basalt"}
        planner.local_survey_exhausted_resources = {"basalt"}
        self.engine.central_depot_inventory[
            "qualified_pv_blanket_segment"
        ] = (
            planner._recipes_cache["solar_panel"]["materials"]
            ["qualified_pv_blanket_segment"] - 1
        )
        self._prefer_shared_strategy("solar_panel", {})

        order = planner._select_shared_capacity_order(
            planner._pooled_materials(), {}
        )

        solar_check = planner._bootstrap_order_feasibility(
            "solar_panel", planner._pooled_materials()
        )
        # A missing flight-qualified PV blanket is a closed-cargo deficit,
        # not something astronauts can replace with basalt. The planner must
        # keep working on another physically actionable capacity objective.
        self.assertFalse(solar_check["actionable"])
        self.assertEqual(
            1,
            solar_check["finite_cargo_deficits"]
            ["qualified_pv_blanket_segment"],
        )
        self.assertNotEqual("solar_panel", order["recipe"])
        self.assertTrue(order["actionable"])

    def test_planet_present_remote_feedstock_stays_actionable_for_survey(self):
        planner = self.engine.decision_engine
        planner.planetary_resources = set(self.engine.planetary_resources)
        planner.planetary_resources.add("graphite")
        planner.remote_resource_requests = {"graphite"}
        planner.local_survey_exhausted_resources = {"graphite"}
        planner.robot_dispatch_targets = {}

        check = planner._bootstrap_order_feasibility(
            "water_collector", planner._pooled_materials()
        )

        self.assertTrue(check["actionable"])
        self.assertNotIn("graphite", check["blocked_resources"])

    def test_contract_is_replaced_when_stock_is_no_longer_in_current_bom(self):
        planner = self.engine.decision_engine
        planner._select_shared_capacity_order = lambda *_args: {
            "recipe": "water_collector",
            "required_count": 1,
            "category": "water",
        }
        for crew in self.crew:
            crew.inventory.materials.clear()
        self.engine.central_depot_inventory.clear()
        water_materials = planner._recipes_cache["water_collector"]["materials"]
        self.engine.central_depot_inventory.update({
            material: quantity
            for material, quantity in water_materials.items()
            if material != "basalt"
        })
        worker = self.crew[1]
        worker._shared_work_contract = {
            "id": "stale-chalcopyrite-contract",
            "recipe": "water_collector",
            "action": "gather",
            "target": {"resource": "chalcopyrite_ore", "quantity_needed": 12},
            "stock_key": "chalcopyrite_ore",
            "target_stock": 23,
            "quantity_reserved": 12,
            "review_tick": 500,
        }

        decision = planner._shared_colony_work_decision(
            worker, tick=100, structures_built={}
        )

        self.assertEqual("gather", decision["action"])
        self.assertEqual("basalt", decision["target"]["resource"])
        self.assertEqual("basalt", worker._shared_work_contract["stock_key"])
        self.assertNotEqual(
            "stale-chalcopyrite-contract",
            worker._shared_work_contract["id"],
        )

    def test_carried_bom_material_is_deposited_even_when_worker_is_at_base(self):
        worker = self.crew[1]
        worker.x = self.engine.lz_x
        worker.y = self.engine.lz_y
        worker.inventory.materials.clear()
        worker.inventory.add_material("iron_ore", 3)
        planner = self.engine.decision_engine
        planner._select_shared_capacity_order = lambda *_args: {
            "recipe": "water_collector",
            "required_count": 1,
            "category": "water",
        }

        decision = planner._shared_colony_work_decision(
            worker, tick=100, structures_built=self.engine.structures_built
        )

        self.assertEqual("deposit_materials", decision["action"])
        self.assertEqual("central_depot", decision["target"]["destination"])

    def test_empty_construction_site_does_not_build_itself(self):
        x, y = self.engine._select_structure_site("solar_panel")
        site = {
            "id": "test_site",
            "type": "solar_panel",
            "x": x,
            "y": y,
            "under_construction": True,
            "destroyed": False,
            "required_work_hours": 10.0,
            "work_hours_completed": 0.0,
            "progress": 0.0,
            "ticks_remaining": 120,
            "total_ticks": 120,
        }
        self.engine.placed_structures.append(site)
        for agent in self.crew:
            agent.x = self.engine.lz_x
            agent.y = self.engine.lz_y
            agent._in_habitat = True
            agent.action.action_type = "sleep"
            agent.action.target = {"ticks": 20, "habitat": True}
            agent.action.ticks_remaining = 20
            agent.needs._consecutive_sleep_ticks = 2

        self.engine._run_tick()

        self.assertEqual(0.0, site["work_hours_completed"])
        self.assertEqual(0.0, site["progress"])
        self.assertEqual(0, site["active_builder_count"])

    def test_bom_planning_does_not_create_a_physical_map_structure(self):
        planner = self.crew[0]
        planner.x = self.engine.lz_x
        planner.y = self.engine.lz_y
        planner._in_habitat = True
        structure_count = len(self.engine.placed_structures)
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "plan_construction",
            "target": {"recipe": "solar_panel", "materials_pending": True},
            "reasoning": "prepare the second solar-array BOM",
        }

        self.engine._process_agent_tick(planner, [], nearby_count=1)
        self.assertEqual(structure_count, len(self.engine.placed_structures))
        self.assertEqual("plan_construction", planner.action.action_type)
        self.assertTrue(planner.action.target["planning_only"])

    def test_site_productivity_is_capped_at_recommended_crew(self):
        x, y = self.engine._select_structure_site("solar_panel")
        site = {
            "id": "team_site",
            "type": "solar_panel",
            "x": x,
            "y": y,
            "under_construction": True,
            "destroyed": False,
            "required_work_hours": 10.0,
            "work_hours_completed": 0.0,
            "progress": 0.0,
            "ticks_remaining": 120,
            "total_ticks": 120,
        }
        self.engine.placed_structures.append(site)
        for index, agent in enumerate(self.crew):
            if index < 3:
                agent.x, agent.y = x, y
                agent._in_habitat = False
                agent.action.action_type = "build"
                agent.action.target = {"recipe": "solar_panel", "struct_id": "team_site"}
                agent.action.ticks_remaining = 120
            else:
                agent.action.action_type = "sleep"
                agent.action.target = {"ticks": 20, "habitat": True}
                agent.action.ticks_remaining = 20

        self.engine._run_tick()

        active_builders = [
            agent for agent in self.crew
            if agent.action.action_type == "build"
            and isinstance(agent.action.target, dict)
            and agent.action.target.get("struct_id") == "team_site"
        ]
        self.assertEqual(2, site["active_builder_count"])
        self.assertLessEqual(len(active_builders), 2)
        self.assertGreater(site["work_hours_completed"], 0.0)

    def test_third_construction_route_claim_is_rejected(self):
        x, y = self.engine._select_structure_site("solar_panel")
        site = {
            "id": "bounded_route_site",
            "type": "solar_panel",
            "x": x,
            "y": y,
            "under_construction": True,
            "destroyed": False,
            "materials_committed": True,
            "progress": 0.1,
        }
        self.engine.placed_structures.append(site)
        route = {
            "recipe": "solar_panel",
            "struct_id": site["id"],
            "x": x,
            "y": y,
            "destination": "construction_site",
        }
        first, second, third = self.crew[:3]
        for crew in (first, second, third):
            crew.action.action_type = "move"
            crew.action.target = dict(route)
            crew.action.ticks_remaining = 1
        self.engine.decision_engine.shared_work_order = {
            "site_id": site["id"],
            "assignments": {
                first.id: "construction",
                second.id: "construction",
            },
        }

        self.assertIsNone(
            self.engine._outbound_work_order_rejection(first, route)
        )
        self.assertEqual(
            "construction_crew_capacity_claimed",
            self.engine._outbound_work_order_rejection(third, route),
        )

    def test_solar_arrays_use_a_clear_planned_field_lattice(self):
        first_x, first_y = self.engine._select_structure_site("solar_panel")
        first_distance = max(
            abs(first_x - self.engine.lz_x), abs(first_y - self.engine.lz_y)
        )
        self.assertGreaterEqual(first_distance, 5)
        self.assertLessEqual(first_distance, 12)
        for structure in self.engine.placed_structures:
            if structure["type"] == "solar_panel":
                continue
            clearance = max(
                abs(first_x - structure["x"]), abs(first_y - structure["y"])
            )
            self.assertGreaterEqual(clearance, 5)
        self.engine.placed_structures.append({
            "id": "solar_test_1", "type": "solar_panel",
            "x": first_x, "y": first_y, "destroyed": False,
        })

        second_x, second_y = self.engine._select_structure_site("solar_panel")
        separation = max(abs(second_x - first_x), abs(second_y - first_y))
        self.assertEqual(2, separation)
        self.assertEqual(0, abs(second_x - first_x) % 2)
        self.assertEqual(0, abs(second_y - first_y) % 2)

    def test_delivered_industry_has_unique_physical_work_cells(self):
        occupied = {}
        for structure in self.engine.placed_structures:
            coordinate = (structure["x"], structure["y"])
            self.assertNotIn(
                coordinate,
                occupied,
                f"{structure['id']} overlaps {occupied.get(coordinate)}",
            )
            occupied[coordinate] = structure["id"]

    def test_repeated_process_assets_keep_a_service_lane(self):
        planned = []
        for index in range(5):
            x, y = self.engine._select_structure_site("isru_o2_unit")
            planned.append((x, y))
            self.engine.placed_structures.append({
                "id": f"o2-layout-{index}",
                "type": "isru_o2_unit",
                "x": x,
                "y": y,
                "health": 1.0,
            })
        for index, first in enumerate(planned):
            for second in planned[index + 1:]:
                self.assertGreaterEqual(
                    max(abs(first[0] - second[0]), abs(first[1] - second[1])),
                    2,
                )

    def test_surface_routes_are_cardinal_and_avoid_foundations(self):
        blocker = {
            "id": "route-blocker",
            "type": "storage_crate",
            "x": self.engine.lz_x + 3,
            "y": self.engine.lz_y,
            "health": 1.0,
        }
        self.engine.placed_structures.append(blocker)
        route = self.engine._surface_vehicle_route(
            self.engine.lz_x,
            self.engine.lz_y,
            self.engine.lz_x + 6,
            self.engine.lz_y,
        )
        self.assertTrue(route)
        self.assertNotIn((blocker["x"], blocker["y"]), route)
        self.assertTrue(all(
            abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
            for first, second in zip(route, route[1:])
        ))

    def test_all_planned_buildings_stay_inside_reserved_campus(self):
        self.engine.placed_structures.append({
            "id": "campus-grid", "type": "life_support_distribution_grid",
            "x": self.engine.lz_x + 4, "y": self.engine.lz_y + 4, "health": 1.0,
        })
        recipes = (
            "storage_crate", "habitat_module",
            "medical_station", "water_collector", "isru_o2_unit",
            "greenhouse", "solar_panel",
        )
        for recipe in recipes:
            x, y = self.engine._select_structure_site(recipe)
            distance = self.engine._distance_from_lz(x, y)
            self.assertLessEqual(
                distance,
                self.engine.CONSTRUCTION_CAMPUS_RADIUS_CELLS,
                recipe,
            )
            self.assertTrue(
                self.engine._is_construction_protected_cell(x, y), recipe
            )

    def test_campus_uses_separate_habitation_agriculture_and_utility_zones(self):
        self.engine.placed_structures.append({
            "id": "campus-grid", "type": "life_support_distribution_grid",
            "x": self.engine.lz_x + 4, "y": self.engine.lz_y + 4, "health": 1.0,
        })
        recipes = (
            "habitat_module", "greenhouse", "water_collector",
            "solar_panel", "communications_array",
        )
        sites = {}
        for recipe in recipes:
            profile = self.engine._structure_site_profile(recipe)
            preferred = self.engine._site_plan_offset(recipe, profile)
            site = self.engine._select_structure_site(recipe)
            sites[recipe] = site
            actual = (
                site[0] - self.engine.lz_x,
                site[1] - self.engine.lz_y,
            )
            # Seed rotation may change compass quadrants, but every structure
            # must remain in the intended transformed planning sector.
            self.assertGreater(
                actual[0] * preferred[0] + actual[1] * preferred[1],
                0,
                recipe,
            )
        self.assertEqual(len(recipes), len(set(sites.values())))
        self.assertGreaterEqual(
            max(self.engine._distance_from_lz(*site) for site in sites.values()),
            10,
        )

    def test_seeded_campus_plan_varies_between_attempts_but_replays(self):
        recipes = (
            "life_support_distribution_grid", "habitat_module",
            "water_collector", "greenhouse", "solar_panel",
            "communications_array",
        )

        def layout(seed):
            engine = SimulationEngine(
                str(ROOT / "config" / "planets" / "kepler-442b.json"),
                seed=seed,
                db=False,
            )
            engine._init_agents()
            result = []
            for index, recipe in enumerate(recipes):
                x, y = engine._select_structure_site(recipe)
                result.append((recipe, x - engine.lz_x, y - engine.lz_y))
                engine.placed_structures.append({
                    "id": f"layout-{index}", "type": recipe,
                    "x": x, "y": y, "health": 1.0,
                })
            return result

        first = layout(42)
        replay = layout(42)
        next_attempt = layout(43)
        self.assertEqual(first, replay)
        self.assertNotEqual(first, next_attempt)
        self.assertGreaterEqual(
            max(max(abs(x), abs(y)) for _, x, y in first), 10
        )

    def test_site_cache_does_not_rescan_for_unrelated_excavation(self):
        first = self.engine._select_structure_site("habitat_module")
        footprint = self.engine._structure_footprint_cells(
            "habitat_module", *first
        )
        disturbed = next(
            (self.engine.lz_x + dx, self.engine.lz_y + dy)
            for dx in range(-15, 16)
            for dy in range(-15, 16)
            if (self.engine.lz_x + dx, self.engine.lz_y + dy) not in footprint
        )
        self.engine.cell_excavation_depth[disturbed] = 1

        calls = 0
        original = self.engine.world.get_cell_info

        def tracked(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        self.engine.world.get_cell_info = tracked
        replay = self.engine._select_structure_site("habitat_module")

        self.assertEqual(first, replay)
        self.assertEqual(0, calls)

    def test_lander_hub_has_4x4_footprint_and_one_fixed_airlock(self):
        hub = next(
            structure for structure in self.engine.placed_structures
            if structure["type"] == "eclss_lander_hub"
        )
        footprint = self.engine._structure_footprint_cells(
            hub["type"], hub["x"], hub["y"], hub
        )
        airlock = self.engine._lander_airlock_position()
        exterior = self.engine._lander_airlock_exterior_position()

        self.assertEqual(16, len(footprint))
        self.assertEqual(
            {
                self.engine.lz_x - 1, self.engine.lz_x,
                self.engine.lz_x + 1, self.engine.lz_x + 2,
            },
            {x for x, _ in footprint},
        )
        self.assertEqual(
            (self.engine.lz_x, self.engine.lz_y + 2), airlock
        )
        self.assertIn(airlock, footprint)
        self.assertNotIn(exterior, footprint)

        crew = self.crew[0]
        crew.x, crew.y = self.engine.lz_x, self.engine.lz_y
        crew._in_habitat = True
        path = []
        for _ in range(8):
            dx, dy = self.engine._cardinal_step_toward(
                crew, self.engine.lz_x - 10, self.engine.lz_y - 10, 1
            )
            crew.x += dx
            crew.y += dy
            path.append((crew.x, crew.y))
            self.engine.current_tick += 1
            if not self.engine._is_lander_footprint_cell(crew.x, crew.y):
                break
        self.assertIn(airlock, path)
        self.assertEqual(exterior, path[-1])

    def test_frontend_draws_4x4_lander_without_external_header(self):
        frontend = (ROOT / "frontend" / "index.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("const baseSize = 4 * scale", frontend)
        self.assertIn("name: 'AIRLOCK'", frontend)
        self.assertNotIn("Top Base Header Badge", frontend)
        self.assertNotIn("NASA DRA 5.0 SURFACE HABITAT LANDER", frontend)

    def test_cardinal_rover_and_utility_spines_remain_unbuilt(self):
        recipes = (
            "life_support_distribution_grid",
            "water_collector", "potable_water_tank", "water_purifier",
            "isru_o2_unit", "habitat_module", "greenhouse",
            "communications_array",
        )
        for index, recipe in enumerate(recipes):
            x, y = self.engine._select_structure_site(recipe)
            self.assertNotEqual(self.engine.lz_x, x, recipe)
            self.assertNotEqual(self.engine.lz_y, y, recipe)
            profile = self.engine._structure_site_profile(recipe)
            self.engine.placed_structures.append({
                "id": f"corridor-layout-{index}", "type": recipe,
                "x": x, "y": y, "health": 1.0,
                "render_scale": profile["render_scale"],
            })

    def test_three_circuit_utility_bom_counts_actual_tube_length(self):
        recipe = self.engine._get_recipe(
            "life_support_distribution_grid"
        )
        effects = recipe["output"]["effects"]
        circuits = int(effects["isolated_fluid_circuits"])
        corridor_m = float(effects["installed_corridor_length_m"])
        tube_m = float(effects["installed_tube_length_m"])
        unit_length_m = float(effects["tube_length_per_metal_pipe_unit_m"])

        self.assertEqual(corridor_m * circuits, tube_m)
        self.assertEqual(
            recipe["materials"]["metal_pipe"] * unit_length_m,
            tube_m,
        )
        input_mass = sum(
            quantity * MATERIAL_DENSITY_KG[material]
            for material, quantity in recipe["materials"].items()
        )
        self.assertAlmostEqual(
            recipe["mass_budget_kg"]["input_total_kg"],
            input_mass,
        )

    def test_arrival_landing_ellipse_has_footprint_and_one_km_exclusion(self):
        landing_x, landing_y = self.engine._select_structure_site(
            "landing_zone"
        )
        distance = self.engine._distance_from_lz(landing_x, landing_y)
        self.assertGreaterEqual(distance, 18)
        self.assertLessEqual(distance, 24)
        footprint = self.engine._structure_footprint_cells(
            "landing_zone", landing_x, landing_y
        )
        self.assertEqual(15, len(footprint))
        for structure in self.engine.placed_structures:
            existing = self.engine._structure_footprint_cells(
                structure["type"], structure["x"], structure["y"], structure
            )
            gap = min(
                max(abs(lx - ex), abs(ly - ey))
                for lx, ly in footprint for ex, ey in existing
            )
            self.assertGreater(gap, 10, structure["type"])
        self.assertTrue(
            self.engine._is_construction_protected_cell(
                landing_x, landing_y
            )
        )

    def test_manufactured_output_appears_only_after_machine_cycle(self):
        # Exercise the supervised legacy profile here; the autonomous profile
        # has separate setup, full-duration processing and unload assertions.
        self.engine.AUTONOMOUS_MANUFACTURING_ENABLED = False
        agent = self.crew[0]
        cnc = next(
            structure for structure in self.engine.placed_structures
            if structure["type"] == "cnc_fabricator"
        )
        agent.x, agent.y = cnc["x"], cnc["y"]
        agent._in_habitat = False
        agent.inventory.materials.clear()
        self.engine.central_depot_inventory["reduced_iron_ingot"] = 1
        self.engine._colony_resources["energy_stored_kwh"] = 20.0
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "refine",
            "target": {"output": "machined_bolts_fasteners"},
            "reasoning": "timing regression",
        }

        self.engine._process_agent_tick(agent, [], nearby_count=1)
        self.assertEqual(0, agent.inventory.materials.get("machined_bolts_fasteners", 0))
        self.assertTrue(agent.action.target["completion_pending"])

        agent.action.ticks_remaining = 1
        self.engine._process_agent_tick(agent, [], nearby_count=1)
        self.assertEqual(20, agent.inventory.materials["machined_bolts_fasteners"])

    def test_frontend_distinguishes_cargo_staging_from_real_construction(self):
        frontend = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

        self.assertIn(
            "placedStructures.filter(s => !s.destroyed)",
            frontend,
        )
        self.assertIn("struct.under_construction === true", frontend)
        self.assertIn("targetId === struct.id", frontend)
        self.assertIn("`CARGO STAGING ${pctStr}`", frontend)
        self.assertIn("`CONSTRUCTION ${pctStr}`", frontend)
        self.assertNotIn("? 'MATERIALS'", frontend)
        self.assertNotIn("builderAgent ? 'BUILDING' : 'PAUSED'", frontend)
        self.assertIn(
            "isOutsideBase && struct.materials_committed !== false && !struct.under_construction",
            frontend,
        )
        self.assertIn("column.detected_next_layer", frontend)
        self.assertIn("layerOrder: Number(nextLayer.relative_order || 1)", frontend)


if __name__ == "__main__":
    unittest.main(verbosity=2)
