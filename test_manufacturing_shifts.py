"""Regression tests for physical manufacturing batches and operator shifts."""

import unittest
from pathlib import Path

from src.agents.agent import create_team_from_presets
from src.memory import vector_store
from src.orchestration.engine import SimulationEngine


ROOT = Path(__file__).resolve().parent
vector_store._use_tfidf_fallback = True


class ManufacturingShiftTest(unittest.TestCase):
    def setUp(self):
        self.engine = SimulationEngine(
            str(ROOT / "config" / "planets" / "kepler-442b.json"),
            seed=42,
            db=False,
        )
        self.engine.llm_client.call = lambda *_args, **_kwargs: None
        # This class locks the legacy continuously supervised mode so its
        # shift/handoff regressions remain meaningful alongside the default
        # closed-loop automation tests below.
        self.engine.AUTONOMOUS_MANUFACTURING_ENABLED = False
        self.crew = create_team_from_presets(
            str(ROOT / "config" / "agent_presets.json")
        )[:3]
        for agent in self.crew:
            self.engine.add_agent(agent)
        self.engine._init_agents()

        cnc_machines = [
            structure for structure in self.engine.placed_structures
            if structure["type"] == "cnc_fabricator"
        ]
        self.machine, self.other_machine = cnc_machines[:2]
        # One explicit machine makes slot ownership and material accounting
        # unambiguous in these tests.
        self.engine.placed_structures[:] = [self.machine]
        self.engine.delivered_structure_kits.clear()
        self.engine.central_depot_inventory.clear()
        self.engine.central_depot_inventory["reduced_iron_ingot"] = 2
        self.engine._colony_resources["energy_stored_kwh"] = 100.0
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "refine",
            "target": {"output": "metal_pipe"},
            "reasoning": "manufacturing shift regression",
        }

        for agent in self.crew:
            agent.x, agent.y = self.machine["x"], self.machine["y"]
            self._make_fit(agent)

    @staticmethod
    def _make_fit(agent):
        agent.needs.energy = 100.0
        agent.needs.hunger = 100.0
        agent.needs.thirst = 100.0
        agent.needs.o2_supply = 100.0
        agent.needs.temperature_stress = 60.0
        agent._current_canister_remaining = 100.0
        agent.plss_co2_scrubber_pct = 100.0
        agent.plss_suit_battery_pct = 100.0
        agent.suit_integrity = 1.0

    def _start_cycle(self):
        operator = self.crew[0]
        self.engine.current_tick = 0
        self.engine._process_agent_tick(operator, [], nearby_count=3)
        machine_id = str(self.machine["id"])
        self.assertEqual("refine", operator.action.action_type)
        self.assertIn(machine_id, self.engine._manufacturing_cycles)
        return operator, self.engine._manufacturing_cycles[machine_id]

    def test_physiological_interrupt_keeps_wip_on_the_same_machine(self):
        operator, cycle = self._start_cycle()
        energy_after_start = self.engine._colony_resources["energy_stored_kwh"]
        remaining_material = self.engine.central_depot_inventory.get(
            "reduced_iron_ingot", 0
        )

        self.engine.current_tick = 1
        operator.needs.thirst = 45.0
        events = self.engine._process_agent_tick(operator, [], nearby_count=3)

        self.assertEqual("refine", events["physiological_safety_interrupt"])
        self.assertFalse(cycle["active"])
        self.assertEqual(str(self.machine["id"]), cycle["machine_id"])
        self.assertTrue(cycle["completion_pending"])
        self.assertTrue(hasattr(operator, "_paused_manufacturing"))

        relief = max(
            self.crew[1:],
            key=lambda crew: int(crew.competency.engineering),
        )
        self.engine._process_agent_tick(relief, [], nearby_count=3)
        self.assertNotEqual("refine", relief.action.action_type)
        self.assertEqual(
            "all_physical_machines_occupied", relief.action.target["reason"]
        )
        self.assertEqual(
            energy_after_start,
            self.engine._colony_resources["energy_stored_kwh"],
        )
        self.assertEqual(
            remaining_material,
            self.engine.central_depot_inventory.get("reduced_iron_ingot", 0),
        )

    def test_shift_handoff_requires_capable_fit_relief_and_outputs_once(self):
        # Use a 12-hour acceptance-test batch so the NASA-derived 6.5-hour
        # operations shift genuinely requires a relief operator.
        self.engine.central_depot_inventory.clear()
        self.engine.central_depot_inventory["qualified_control_board_core"] = 1
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "refine",
            "target": {"output": "electronic_component"},
            "reasoning": "long manufacturing shift regression",
        }
        operator, cycle = self._start_cycle()
        energy_after_start = self.engine._colony_resources["energy_stored_kwh"]
        material_after_start = self.engine.central_depot_inventory.get(
            "qualified_control_board_core", 0
        )

        last_events = {}
        for tick in range(1, self.engine.MANUFACTURING_SHIFT_TICKS + 1):
            self.engine.current_tick = tick
            self._make_fit(operator)
            last_events = self.engine._process_agent_tick(
                operator, [], nearby_count=3
            )

        self.assertIn("manufacturing_shift_ended", last_events)
        self.assertFalse(cycle["active"])
        remaining_at_handoff = cycle["remaining_ticks"]
        self.assertGreater(remaining_at_handoff, 0)

        # Relief does not have to independently rediscover the exact WIP
        # recipe: the engine offers the due physical batch before planning.
        self.engine.decision_engine.process_tick = lambda agent, **_kwargs: {
            "action": "refine",
            "target": {
                "output": (
                    getattr(agent, "_paused_manufacturing", {})
                    .get("output", "machined_bolts_fasteners")
                )
            },
            "reasoning": "accept offered WIP, otherwise choose another batch",
        }
        relief = max(
            self.crew[1:],
            key=lambda crew: int(crew.competency.engineering),
        )
        original_engineering = relief.competency.engineering
        relief.competency.engineering = 0
        self.engine.current_tick += 1
        blocked = self.engine._process_agent_tick(relief, [], nearby_count=3)
        self.assertNotIn("manufacturing_handoff", blocked)
        self.assertEqual("craft_blocked", relief.action.action_type)

        relief.competency.engineering = original_engineering
        relief.needs.energy = 65.0
        self.engine.current_tick += 1
        self.engine._process_agent_tick(relief, [], nearby_count=3)
        self.assertNotEqual("refine", relief.action.action_type)

        self._make_fit(relief)
        self.engine.current_tick += 1
        handoff_events = self.engine._process_agent_tick(
            relief, [], nearby_count=3
        )
        self.assertIn("manufacturing_handoff", handoff_events)
        self.assertEqual(operator.id, handoff_events["manufacturing_handoff"]["from_operator_id"])
        self.assertEqual(relief.id, cycle["operator_id"])
        self.assertEqual(remaining_at_handoff, relief.action.ticks_remaining)
        self.assertEqual(
            energy_after_start,
            self.engine._colony_resources["energy_stored_kwh"],
        )
        self.assertEqual(
            material_after_start,
            self.engine.central_depot_inventory.get(
                "qualified_control_board_core", 0
            ),
        )

        relief.action.ticks_remaining = 1
        self.engine.current_tick += 1
        self._make_fit(relief)
        completion = self.engine._process_agent_tick(relief, [], nearby_count=3)
        expected_quantity = int(
            self.engine._get_recipe("electronic_component")["output"]["quantity"]
        )
        self.assertEqual(
            expected_quantity,
            completion["manufacturing_completed"]["quantity"],
        )
        self.assertNotIn(str(self.machine["id"]), self.engine._manufacturing_cycles)

        self.engine.current_tick += 1
        self._make_fit(relief)
        self.engine._process_agent_tick(relief, [], nearby_count=3)
        total_output = sum(
            crew.inventory.materials.get("electronic_component", 0)
            for crew in self.crew
        )
        self.assertEqual(expected_quantity, total_output)
        self.assertEqual(
            energy_after_start,
            self.engine._colony_resources["energy_stored_kwh"],
        )

    def test_previous_operator_cannot_clone_wip_onto_second_machine(self):
        operator, cycle = self._start_cycle()
        self.engine.placed_structures.append(self.other_machine)
        output_before = operator.inventory.materials.get("metal_pipe", 0)

        self.engine.current_tick = 1
        cycle["shift_end_tick"] = 1
        self._make_fit(operator)
        self.engine._process_agent_tick(operator, [], nearby_count=3)
        self.assertFalse(cycle["active"])

        # The old operator still carries a read-only view of the paused WIP.
        # A free second CNC must not turn that view into a material-free clone.
        self.engine.current_tick = 2
        self._make_fit(operator)
        blocked = self.engine._process_agent_tick(
            operator, [], nearby_count=3
        )

        self.assertNotEqual("refine", operator.action.action_type)
        self.assertEqual(
            "machine_cycle_awaiting_relief", operator.action.target["reason"]
        )
        self.assertEqual(1, len(self.engine._manufacturing_cycles))
        self.assertEqual(
            output_before,
            operator.inventory.materials.get("metal_pipe", 0),
        )

        # Even an already-running stale action copy cannot emit output without
        # a canonical cycle bound to that exact physical machine.
        operator.action.action_type = "refine"
        operator.action.target = {
            "output": "metal_pipe",
            "output_quantity": 2,
            "completion_pending": True,
            "machine_type": "cnc_fabricator",
            "machine_id": str(self.other_machine["id"]),
        }
        operator.action.ticks_remaining = 1
        self.engine.current_tick = 3
        self._make_fit(operator)
        rejected = self.engine._process_agent_tick(
            operator, [], nearby_count=3
        )

        self.assertIn("stale_manufacturing_completion_rejected", rejected)
        self.assertEqual(
            output_before,
            operator.inventory.materials.get("metal_pipe", 0),
        )

    def test_reclaiming_cycle_clears_expired_relief_offer(self):
        operator, cycle = self._start_cycle()
        relief = self.crew[1]
        self.engine._pause_manufacturing_cycle(
            operator,
            "physiological_safety_interrupt",
            cycle=cycle,
            remaining_ticks=cycle["remaining_ticks"],
        )
        relief._paused_manufacturing = self.engine._manufacturing_action_target(
            cycle, resumed=True, handoff=True
        )

        self.engine.current_tick = int(cycle["shift_end_tick"]) + 100
        self._make_fit(operator)
        self.engine._assign_manufacturing_cycle(operator, cycle)

        self.assertFalse(hasattr(relief, "_paused_manufacturing"))
        self.assertTrue(cycle["active"])
        self.assertEqual(operator.id, cycle["operator_id"])

    def test_relief_keeps_machine_route_until_paused_cycle_handoff(self):
        operator, cycle = self._start_cycle()
        self.engine.current_tick = 1
        cycle["shift_end_tick"] = 1
        self.engine._pause_manufacturing_cycle(
            operator,
            "operator_shift_complete",
            cycle=cycle,
            remaining_ticks=cycle["remaining_ticks"],
        )

        relief = self.crew[1]
        relief.x = int(self.machine["x"]) - 4
        relief.y = int(self.machine["y"])
        relief._in_habitat = False
        relief.action.clear()
        self._make_fit(relief)

        # First tick accepts the offered WIP and begins the physical route.
        self.engine.current_tick = 2
        self.engine._process_agent_tick(relief, [], nearby_count=3)
        self.assertEqual("move", relief.action.action_type)
        self.assertTrue(relief.action.target.get("resume_machine_cycle"))
        self.assertEqual(relief.id, cycle.get("route_operator_id"))

        # On subsequent route ticks the old operator's paused ownership must
        # not make this relief operator see its own reserved CNC as occupied.
        for tick in range(3, 10):
            self.engine.current_tick = tick
            self._make_fit(relief)
            events = self.engine._process_agent_tick(
                relief, [], nearby_count=3
            )
            self.assertNotEqual("gather", relief.action.action_type)
            if "manufacturing_handoff" in events:
                break
        else:
            self.fail("relief abandoned the reserved machine route")

        self.assertTrue(cycle["active"])
        self.assertEqual(relief.id, cycle["operator_id"])

    def test_field_contract_finishes_bounded_load_but_cannot_become_permanent(self):
        planner = self.engine.decision_engine
        worker = self.crew[1]
        worker._shared_work_contract = {
            "id": "old-iron-shift",
            "recipe": "isru_o2_unit",
            "action": "gather",
            "target": {"resource": "iron_ore"},
            "stock_key": "iron_ore",
            "target_stock": 12,
            "quantity_reserved": 12,
            "review_tick": 96,
            "shift_end_tick": 78,
        }
        worker._active_excavation = {
            "resource": "iron_ore",
            "x": worker.x + 4,
            "y": worker.y,
            "capacity_recipe": "isru_o2_unit",
        }

        # A parent subassembly has satisfied the immediate iron branch.  The
        # astronaut may still complete the small load promised for this field
        # shift instead of reacting robotically to an inventory edge.
        planner._bom_outstanding_requirements = lambda *_args, **_kwargs: {}
        during_shift = planner._shared_supply_decision(
            agent=worker,
            tick=20,
            structures_built=self.engine.structures_built,
            living=self.crew,
            recipe_name="isru_o2_unit",
            order={"recipe": "isru_o2_unit", "required_count": 10},
            assignments={},
        )
        self.assertEqual("gather", during_shift["action"])
        self.assertEqual(
            "old-iron-shift", worker._shared_work_contract["id"]
        )

        # The persistent geological face is not irreversible manufacturing
        # WIP.  At the shift boundary ownership ends and the crew can be moved
        # onto the live critical path; the world layer itself is not modified.
        planner._shared_dependency_actions = lambda *_args, **_kwargs: []
        after_shift = planner._shared_supply_decision(
            agent=worker,
            tick=79,
            structures_built=self.engine.structures_built,
            living=self.crew,
            recipe_name="isru_o2_unit",
            order={"recipe": "isru_o2_unit", "required_count": 10},
            assignments={},
            preserve_construction_order=True,
        )
        self.assertEqual("stand_watch", after_shift["action"])
        self.assertTrue(after_shift["target"]["materials_ready"])
        self.assertIsNone(worker._shared_work_contract)
        self.assertIsNone(worker._active_excavation)

    def test_started_machine_batch_survives_contract_shift_boundary(self):
        planner = self.engine.decision_engine
        operator = self.crew[0]
        operator._shared_work_contract = {
            "id": "pipe-batch",
            "recipe": "isru_o2_unit",
            "action": "refine",
            "target": {"output": "metal_pipe"},
            "stock_key": "metal_pipe",
            "target_stock": 8,
            "quantity_reserved": 2,
            "review_tick": 48,
            "shift_end_tick": 40,
        }
        operator.action.action_type = "refine"
        operator.action.target = {
            "output": "metal_pipe",
            "completion_pending": True,
        }
        planner._bom_outstanding_requirements = lambda *_args, **_kwargs: {}

        decision = planner._shared_supply_decision(
            agent=operator,
            tick=80,
            structures_built=self.engine.structures_built,
            living=self.crew,
            recipe_name="isru_o2_unit",
            order={"recipe": "isru_o2_unit", "required_count": 10},
            assignments={},
        )

        self.assertEqual("refine", decision["action"])
        self.assertEqual("pipe-batch", operator._shared_work_contract["id"])


class AutonomousManufacturingTest(unittest.TestCase):
    def setUp(self):
        self.engine = SimulationEngine(
            str(ROOT / "config" / "planets" / "kepler-442b.json"),
            seed=42,
            db=False,
        )
        self.engine.llm_client.call = lambda *_args, **_kwargs: None
        self.operator = create_team_from_presets(
            str(ROOT / "config" / "agent_presets.json")
        )[0]
        self.engine.add_agent(self.operator)
        self.engine._init_agents()
        self.machine = next(
            structure for structure in self.engine.placed_structures
            if structure["type"] == "cnc_fabricator"
        )
        self.engine.placed_structures[:] = [self.machine]
        self.engine.delivered_structure_kits.clear()
        self.engine.central_depot_inventory.clear()
        self.engine.central_depot_inventory["reduced_iron_ingot"] = 2
        self.engine._colony_resources["energy_stored_kwh"] = 100.0
        self.engine.decision_engine.process_tick = lambda **_kwargs: {
            "action": "refine",
            "target": {"output": "metal_pipe"},
            "reasoning": "autonomous manufacturing regression",
        }
        self.operator.x = int(self.machine["x"])
        self.operator.y = int(self.machine["y"])
        ManufacturingShiftTest._make_fit(self.operator)

    def test_operator_can_load_second_cell_and_unload_first_while_it_runs(self):
        from copy import deepcopy

        other_machine = deepcopy(self.machine)
        other_machine["id"] = "second_cnc"
        self.engine.placed_structures.append(other_machine)
        self.engine.central_depot_inventory["reduced_iron_ingot"] = 4
        self.engine.current_tick = 0
        self.engine._process_agent_tick(self.operator, [], nearby_count=1)
        first = self.engine._manufacturing_cycles[str(self.machine["id"])]

        self.operator.action.clear()
        self.engine.current_tick = 1
        self.assertIsNone(self.engine._offer_manufacturing_handoff(self.operator))
        self.assertFalse(hasattr(self.operator, "_paused_manufacturing"))
        planner = self.engine.decision_engine
        self.assertEqual(2, planner._contract_reservations(
            "isru_o2_unit", self.operator, 1
        )["metal_pipe"])
        self.assertEqual(0, planner._pooled_materials().get("metal_pipe", 0))

        self.engine._process_agent_tick(self.operator, [], nearby_count=1)
        second = self.engine._manufacturing_cycles["second_cnc"]
        self.assertTrue(first["active"])
        self.assertTrue(second["active"])
        self.assertEqual(4, planner._contract_reservations(
            "isru_o2_unit", self.operator, 1
        )["metal_pipe"])
        self.assertEqual(0, self.engine.central_depot_inventory["reduced_iron_ingot"])

        self.engine.current_tick = first["nominal_process_ticks"]
        self.engine._advance_autonomous_manufacturing_cycles()
        self.assertTrue(first["output_ready"])
        self.assertFalse(second["output_ready"])
        self.operator.action.clear()
        offered = self.engine._offer_manufacturing_handoff(self.operator)
        self.assertIs(first, offered)
        self.assertEqual(first["machine_id"], self.operator._paused_manufacturing["machine_id"])
        self.assertTrue(second["active"])
        self.assertEqual(0, planner._pooled_materials().get("metal_pipe", 0))

    def test_machine_runs_unattended_for_full_process_then_requires_unload(self):
        energy_before = self.engine._colony_resources["energy_stored_kwh"]
        self.engine.current_tick = 0
        self.engine._process_agent_tick(self.operator, [], nearby_count=1)
        machine_id = str(self.machine["id"])
        cycle = self.engine._manufacturing_cycles[machine_id]
        duration = int(cycle["nominal_process_ticks"])
        energy_after_start = self.engine._colony_resources["energy_stored_kwh"]

        self.assertTrue(self.engine.AUTONOMOUS_MANUFACTURING_ENABLED)
        self.assertEqual("operate_machine", self.operator.action.action_type)
        self.assertLess(energy_after_start, energy_before)
        self.assertEqual(0, self.engine.central_depot_inventory.get(
            "reduced_iron_ingot", 0
        ))
        self.assertIsNone(cycle["operator_id"])
        self.assertTrue(cycle["active"])
        self.assertFalse(cycle["output_ready"])
        self.assertEqual(0, self.operator.inventory.materials.get(
            "metal_pipe", 0
        ))

        # The initiator may sleep or do other work while the same machine WIP
        # advances; the watchdog must not demand a ghost continuous operator.
        self.operator.action.action_type = "sleep"
        self.operator.action.target = {"habitat": True}
        self.operator.action.ticks_remaining = duration + 10
        self.engine.current_tick = duration - 1
        self.assertEqual([], self.engine._advance_autonomous_manufacturing_cycles())
        self.assertEqual(1, cycle["remaining_ticks"])
        self.assertEqual([], self.engine._enforce_manufacturing_cycle_invariants())
        self.assertEqual(0, self.operator.inventory.materials.get(
            "metal_pipe", 0
        ))

        self.engine.current_tick = duration
        completed = self.engine._advance_autonomous_manufacturing_cycles()
        self.assertEqual(1, len(completed))
        self.assertTrue(cycle["output_ready"])
        self.assertFalse(cycle["active"])
        self.assertEqual(0, cycle["remaining_ticks"])
        refreshed_offer = self.operator._paused_manufacturing
        self.assertTrue(refreshed_offer["output_ready"])
        self.assertEqual(
            self.engine.MANUFACTURING_UNLOAD_TICKS,
            refreshed_offer["remaining_ticks"],
        )
        self.assertEqual(0, self.operator.inventory.materials.get(
            "metal_pipe", 0
        ))

        ManufacturingShiftTest._make_fit(self.operator)
        self.operator.action.clear()
        self.engine._assign_manufacturing_cycle(self.operator, cycle)
        self.assertEqual("refine", self.operator.action.action_type)
        self.assertEqual(
            self.engine.MANUFACTURING_UNLOAD_TICKS,
            self.operator.action.ticks_remaining,
        )

        self.engine.current_tick += 1
        events = self.engine._process_agent_tick(
            self.operator, [], nearby_count=1
        )
        self.assertEqual(
            2, events["manufacturing_completed"]["quantity"]
        )
        self.assertEqual(2, self.operator.inventory.materials["metal_pipe"])
        self.assertNotIn(machine_id, self.engine._manufacturing_cycles)
        self.assertEqual(
            energy_after_start,
            self.engine._colony_resources["energy_stored_kwh"],
        )
        self.assertEqual(
            1,
            self.engine._manufacturing_automation_telemetry[
                "cycles_unloaded"
            ],
        )

        # A stale post-completion action cannot mint the batch again.
        self.engine.current_tick += 1
        ManufacturingShiftTest._make_fit(self.operator)
        self.engine._process_agent_tick(self.operator, [], nearby_count=1)
        self.assertEqual(2, self.operator.inventory.materials["metal_pipe"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
