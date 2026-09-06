"""Counterexamples from the full-horizon failed mission (no free resources)."""
import copy
import random
import unittest
from unittest.mock import patch

import test_construction_rover_regression as fixture
from src.agents.agent import AgentStatus
from src.systems.airlock import AirlockController


class AirlockConservationTest(unittest.TestCase):
    def test_return_allocation_survives_general_bus_depletion(self):
        lock = AirlockController(1)
        stores = {'energy_stored_kwh': 1.0, 'o2_reserve_kg': 1.0}
        self.assertTrue(lock.charge_cycle(['a', 'b'], 'out', stores))
        self.assertAlmostEqual(.7, stores['energy_stored_kwh'])
        self.assertAlmostEqual(.2, stores['airlock_reserved_energy_kwh'])
        self.assertAlmostEqual(.91, stores['o2_reserve_kg'])
        stores['energy_stored_kwh'] = 0.0  # real intervening loads consumed it
        self.assertTrue(lock.charge_cycle(['a', 'b'], 'in', stores))
        self.assertAlmostEqual(.1, stores['energy_stored_kwh'])  # unused solo allocation
        self.assertAlmostEqual(.94, stores['o2_reserve_kg'])
        self.assertEqual({}, lock.return_reserves)

    def test_departure_does_not_spend_last_energy_without_return(self):
        lock = AirlockController(1)
        stores = {'energy_stored_kwh': .1, 'o2_reserve_kg': 1.0}
        before = dict(stores)
        self.assertFalse(lock.charge_cycle(['a'], 'out', stores))
        self.assertEqual(before, stores)
        self.assertEqual('insufficient_airlock_energy', lock.blocked_reason)

    def test_return_not_starved_by_unfunded_outbound_queue(self):
        lock = AirlockController(1)
        self.assertFalse(lock.request(['out'], 'out', 1, lambda: False))
        self.assertFalse(lock.request(['in'], 'in', 1, lambda: True))
        self.assertTrue(lock.request(['in'], 'in', 2, lambda: True))

    def test_funded_return_not_starved_by_unfunded_inbound_queue(self):
        lock = AirlockController(1)
        stores = {'energy_stored_kwh': 0.0, 'o2_reserve_kg': 0.0}
        lock.return_reserves['funded'] = {'energy': 0.1, 'o2': 0.03}
        for tick in range(1, 4):
            lock.request(['unfunded'], 'in', tick,
                         lambda: lock.charge_cycle(['unfunded'], 'in', stores))
            admitted = lock.request(['funded'], 'in', tick,
                         lambda: lock.charge_cycle(['funded'], 'in', stores))
            if admitted:
                break
        self.assertTrue(admitted)
        self.assertEqual(0.0, stores['energy_stored_kwh'])


class MissionFailureFixTest(unittest.TestCase):
    def setUp(self):
        fixture.ConstructionRoverRegressionTest.setUp(self)

    def test_commissioned_shielded_habitat_sleep_does_not_open_suit(self):
        crew = self.crew[0]
        crew.x, crew.y = self.engine.lz_x + 7, self.engine.lz_y - 5
        self.engine.placed_structures.append({
            'id': 'shielded-habitat', 'type': 'habitat_module',
            'x': crew.x, 'y': crew.y, 'health': 1.0,
            'under_construction': False, 'destroyed': False})
        crew.needs.energy = 90.0
        crew.action.action_type = 'sleep'
        crew.action.target = {'habitat': True}
        crew.action.ticks_remaining = 8
        with patch.object(self.planner, 'process_tick', return_value=None):
            self.engine._process_agent_tick(crew, [], 1)
        self.assertTrue(crew._in_habitat)
        self.assertTrue(self.engine._is_pressurized_location(crew.x, crew.y))
        self.assertEqual('sleep', crew.action.action_type)
        self.assertEqual(100.0, crew._current_canister_remaining)

    def test_emergency_return_consumes_carried_ration_before_starvation(self):
        crew = self.crew[0]
        crew.x, crew.y = self.engine.lz_x - 15, self.engine.lz_y + 3
        crew._in_habitat = False
        crew.needs.hunger = 10.0
        crew.inventory.items['emergency_rations'] = 2
        decision = {'action': 'move', 'target': {'x': self.engine.lz_x,
                    'y': self.engine.lz_y, 'forced_return': True}, 'deterministic': True}
        before = crew.total_kcal_consumed
        with patch.object(self.planner, 'process_tick', return_value=decision):
            self.engine._process_agent_tick(crew, [], 1)
        self.assertEqual('eat', crew.action.action_type)
        self.assertEqual(1, crew.inventory.items['emergency_rations'])
        self.assertEqual(before + 700, crew.total_kcal_consumed)
        self.assertGreater(crew.needs.hunger, 10.0)
        self.assertFalse(crew._in_habitat)

    def test_emergency_field_meal_cannot_draw_on_remote_bulk_food(self):
        crew = self.crew[0]
        crew.x, crew.y = self.engine.lz_x - 15, self.engine.lz_y + 3
        crew._in_habitat = False
        crew.needs.hunger = 10.0
        crew.inventory.items.pop('emergency_rations', None)
        crew.inventory.items.pop('ration_pack', None)
        self.engine._colony_resources['food_reserve_kcal'] = 10000.0
        decision = {'action': 'move', 'target': {'x': self.engine.lz_x,
                    'y': self.engine.lz_y, 'forced_return': True}, 'deterministic': True}
        before = crew.total_kcal_consumed
        with patch.object(self.planner, 'process_tick', return_value=decision):
            self.engine._process_agent_tick(crew, [], 1)
        self.assertEqual('move', crew.action.action_type)
        self.assertEqual(before, crew.total_kcal_consumed)
        self.assertEqual(10000.0, self.engine._colony_resources['food_reserve_kcal'])

    def test_maintenance_route_honors_distance_scaled_o2_return(self):
        crew = self.crew[0]
        crew.x, crew.y = self.engine.lz_x + 24, self.engine.lz_y
        crew._in_habitat = False
        crew.needs.energy = 100.0
        crew.needs.hunger = 100.0
        crew.needs.thirst = 100.0
        crew.needs.temperature_stress = 50.0
        crew.needs.o2_supply = 50.0
        crew._current_canister_remaining = 50.0
        crew.inventory.items.pop('oxygen_canisters', None)
        crew.action.action_type = 'move'
        crew.action.target = {
            'x': crew.x + 1, 'y': crew.y,
            'maintenance': True,
            'maintenance_action': 'inspection',
        }
        crew.action.ticks_remaining = 1
        crew._routine_eva_o2_return_threshold = (
            self.engine._routine_eva_o2_return_threshold(crew, 24)
        )
        decision = self.planner._process_tick_internal(
            crew, 100, {},
            {
                'active_events': [], 'temperature_c': 20.0,
                'effective_temperature_c': 20.0,
                'colony_resources': dict(self.engine._colony_resources),
            },
            [],
        )
        self.assertEqual('move', decision['action'])
        self.assertEqual(
            (self.engine.lz_x, self.engine.lz_y),
            (decision['target']['x'], decision['target']['y']),
        )
        self.assertIn('PLSS return reserve', decision['reasoning'])

    def test_working_rover_lead_rejects_stale_arrival_resource(self):
        lead, buddy = self.crew
        tx, ty = self.engine.lz_x + 15, self.engine.lz_y - 18
        for crew, role in ((lead, 'lead'), (buddy, 'buddy')):
            crew.x, crew.y = tx, ty
            crew._in_habitat = False
            crew._active_expedition = {
                'id': 'iron-sortie', 'kind': 'resource_recovery',
                'resource': 'iron_ore', 'target_x': tx, 'target_y': ty,
                'authorized_radius': 24, 'status': 'working', 'role': role,
                'lead_id': lead.id, 'buddy_id': buddy.id,
                'transport': 'crew_rover', 'move_speed_cells': 16,
            }
        lead.action.action_type = 'arrived'
        lead.action.target = {'x': tx, 'y': ty, 'resource': 'graphite',
                              'destination': 'extraction_face',
                              'mission_action': 'gather', 'expedition': True,
                              'transport': 'crew_rover'}
        decision = self.planner._process_tick_internal(
            lead, 100, {}, {'active_events': [], 'temperature_c': 20.0,
                            'effective_temperature_c': 20.0}, [])
        self.assertEqual('gather', decision['action'])
        self.assertEqual('iron_ore', decision['target']['resource'])
        self.assertTrue(decision['target']['expedition'])

    def test_rover_transit_does_not_enter_external_habitat_by_overlap(self):
        lead, buddy = self.crew
        destination = (self.engine.lz_x + 18, self.engine.lz_y + 10)
        expedition = self.engine._start_expedition(
            lead, 'iron_ore', destination, require_rover=True
        )
        self.assertIsNotNone(expedition)
        rover = self.engine.surface_fleet.crew_rover_for_expedition(
            expedition['id']
        )
        for _ in range(12):
            self.engine.current_tick += 1
            self.engine._move_resource_rover_team(
                lead, {'x': destination[0], 'y': destination[1],
                       'destination': 'extraction_face', 'expedition': True}
            )
            if expedition['status'] == 'working':
                break
        self.assertEqual('working', expedition['status'])
        self.assertEqual(destination, (rover.x, rover.y))
        self.engine.placed_structures.append({
            'id': 'route-overlap-habitat', 'type': 'habitat_module',
            'x': destination[0], 'y': destination[1],
            'under_construction': False, 'destroyed': False, 'health': 1.0,
        })
        decision = {
            'action': 'move', 'deterministic': True,
            'target': {'x': destination[0], 'y': destination[1],
                       'resource': 'iron_ore',
                       'destination': 'extraction_face', 'expedition': True,
                       'transport': 'crew_rover'},
        }
        self.engine.current_tick += 1
        with patch.object(self.planner, 'process_tick', return_value=decision):
            self.engine._process_agent_tick(lead, [], 1)
        self.assertFalse(lead._in_habitat)
        self.assertFalse(buddy._in_habitat)
        self.assertEqual((rover.x, rover.y), (lead.x, lead.y))
        self.assertEqual((lead.x, lead.y), (buddy.x, buddy.y))
        self.assertEqual('in_use', rover.state)

    def test_unoccupied_habitat_cold_standby_preserves_bootstrap_airlock_power(self):
        engine = self.engine
        engine.structures_built.update(habitat_module=1, oxygen_buffer_tank=2)
        engine.placed_structures.append({
            'id': 'empty-habitat', 'type': 'habitat_module',
            'x': engine.lz_x + 8, 'y': engine.lz_y + 8,
            'under_construction': False, 'destroyed': False, 'health': 1.0})
        engine._colony_resources['energy_stored_kwh'] = 0.0
        engine._colony_resources['lander_auxiliary_energy_remaining_kwh'] = 0.0
        engine.current_tick = 72  # near peak daylight for this deterministic planet
        engine.agents = []
        engine._run_tick()
        self.assertLess(engine._power_cycle_telemetry['load_kwh'], 1.0)
        self.assertGreater(engine._colony_resources['energy_stored_kwh'], 0.1)
        self.assertEqual(0, engine._power_cycle_telemetry['active_habitats'])

    def test_occupied_habitat_keeps_full_nameplate_load(self):
        engine = self.engine
        crew = self.crew[0]
        hx, hy = engine.lz_x + 8, engine.lz_y + 8
        engine.structures_built.update(habitat_module=1, oxygen_buffer_tank=2)
        engine.placed_structures.append({
            'id': 'occupied-habitat', 'type': 'habitat_module',
            'x': hx, 'y': hy, 'under_construction': False,
            'destroyed': False, 'health': 1.0})
        crew.x, crew.y, crew._in_habitat = hx, hy, True
        crew.action.action_type = 'sleep'
        crew.action.ticks_remaining = 8
        engine.agents = [crew]
        engine._colony_resources['energy_stored_kwh'] = 0.0
        engine._colony_resources['lander_auxiliary_energy_remaining_kwh'] = 0.0
        engine.current_tick = 72
        engine._run_tick()
        self.assertEqual(1, engine._power_cycle_telemetry['active_habitats'])
        self.assertAlmostEqual(19.0 / 6.0,
                               engine._power_cycle_telemetry['load_kwh'])

    def test_rescue_cannot_pick_up_patient_through_lander_wall(self):
        rescuer, patient = self.crew
        rescuer.x, rescuer.y = self.engine.lz_x + 1, self.engine.lz_y + 2
        patient.x, patient.y = self.engine._lander_airlock_exterior_position()
        patient._in_habitat = False
        patient.status = AgentStatus.CRITICAL
        before = (patient.x, patient.y)
        decision = {'action': 'rescue', 'target': {'victim_id': patient.id}, 'deterministic': True}
        with patch.object(self.planner, 'process_tick', return_value=decision):
            self.engine._process_agent_tick(rescuer, [], 1)
        self.assertEqual(before, (patient.x, patient.y))
        self.assertFalse(patient._in_habitat)

    def test_rescue_returns_both_occupants_in_one_funded_airlock_cycle(self):
        rescuer, patient = self.crew
        for crew in self.crew:
            crew.x, crew.y = self.engine._lander_airlock_exterior_position()
            crew._in_habitat = False
        patient.status = AgentStatus.INCAPACITATED
        patient.needs.energy = 0.0
        decision = {'action': 'rescue', 'target': {'victim_id': patient.id}, 'deterministic': True}
        with patch.object(self.planner, 'process_tick', return_value=decision):
            self.engine._process_agent_tick(rescuer, [], 1)
        self.assertEqual(set(c.id for c in self.crew), set(self.engine.airlock.snapshot()['occupants']))
        self.assertFalse(patient._in_habitat)
        rescuer.action.clear()
        self.engine.current_tick += self.engine.airlock.cycle_ticks
        with patch.object(self.planner, 'process_tick', return_value=decision):
            self.engine._process_agent_tick(rescuer, [], 1)
        self.assertTrue(patient._in_habitat)
        self.assertTrue(rescuer._in_habitat)
        self.assertEqual(self.engine._lander_airlock_position(), (patient.x, patient.y))
        self.assertEqual(1, self.engine.airlock.completed_cycles)
        self.assertIsNone(self.engine.airlock.active)

    def test_uncommissioned_or_failed_pressure_hull_cannot_supply_shelter(self):
        crew = self.crew[0]
        crew.x, crew.y = self.engine.lz_x + 7, self.engine.lz_y - 5
        vault = {'id': 'failed-hull', 'type': 'habitat_module',
                 'x': crew.x, 'y': crew.y, 'health': 1.0}
        self.engine.placed_structures.append(vault)
        for invalid in ({'under_construction': True}, {'destroyed': True}, {'health': 0.0}):
            vault.update(under_construction=False, destroyed=False, health=1.0)
            vault.update(invalid)
            with self.subTest(invalid=invalid):
                self.assertFalse(self.engine._is_pressurized_location(crew.x, crew.y))

    def test_full_oxygen_buffer_does_not_run_electrolyser(self):
        self.engine._run_tick()
        self.assertFalse(self.engine._power_cycle_telemetry['lander_ogs_active'])
        self.engine._colony_resources['o2_reserve_kg'] = 1.0
        self.engine.current_tick += 1
        self.engine._run_tick()
        self.assertTrue(self.engine._power_cycle_telemetry['lander_ogs_active'])

    def test_recovered_patient_can_drink_after_stale_coma_flag(self):
        patient = self.crew[0]
        patient.status = AgentStatus.CRITICAL
        patient.needs.thirst = 30.0
        patient.action.action_type = 'medical_rest'
        patient.action.target = {'habitat': True, 'conscious': False, 'oral_fluids_allowed': False}
        patient.action.ticks_remaining = 12
        before = patient.inventory.items.get('water_packs', 0)
        self.engine._process_agent_tick(patient, [], 0)
        self.assertTrue(patient.action.target.get('oral_fluids_allowed'))
        self.engine.current_tick += 1
        self.engine._process_agent_tick(patient, [], 0)
        self.assertGreater(patient.needs.thirst, 50)
        self.assertEqual(before-1, patient.inventory.items.get('water_packs', 0))

    def test_conscious_medical_rest_patient_eats_before_relapse(self):
        patient = self.crew[0]
        patient.status = AgentStatus.CRITICAL
        patient.needs.hunger = 5.0
        patient.needs.thirst = 70.0
        patient.needs.o2_supply = 80.0
        patient.needs.energy = 60.0
        patient.action.action_type = 'medical_rest'
        patient.action.target = {
            'habitat': True, 'conscious': True,
            'oral_fluids_allowed': True,
        }
        patient.action.ticks_remaining = 24
        before = patient.inventory.items.get('emergency_rations', 0)
        self.engine._process_agent_tick(patient, [], 0)
        self.assertGreater(patient.needs.hunger, 25.0)
        self.assertEqual(
            before - 1,
            patient.inventory.items.get('emergency_rations', 0),
        )
        self.assertEqual('medical_rest', patient.action.action_type)
        self.assertEqual(
            'oral', patient.action.target.get('last_nutrition_route')
        )

    def test_iv_uses_actual_depot_stock_once(self):
        responder, patient = self.crew
        responder.competency.medical = 7
        patient.status = AgentStatus.INCAPACITATED
        patient.needs.thirst = 0.0
        for crew in self.crew:
            crew.inventory.items.pop('sterile_iv_fluid_bags', None)
            crew.inventory.items.pop('iv_io_administration_sets', None)
        self.engine.central_depot_inventory.update(sterile_iv_fluid_bags=1, iv_io_administration_sets=1)
        decision = {'action': 'treat', 'target': {'patient_id': patient.id, 'patient': patient.name, 'protocol': 'iv_io_rehydration'}, 'deterministic': True}
        with patch.object(self.planner, 'process_tick', return_value=decision):
            self.engine._process_agent_tick(responder, [], 0)
        self.assertGreater(patient.needs.thirst, 0)
        self.assertEqual(0, self.engine.central_depot_inventory['sterile_iv_fluid_bags'])
        self.assertEqual(0, self.engine.central_depot_inventory['iv_io_administration_sets'])
        self.assertEqual('iv_io', responder.action.target.get('hydration_route'))

    def test_dual_vital_triage_treats_the_more_advanced_failure_clock(self):
        responder, patient = self.crew
        responder.competency.medical = 7
        responder.inventory.items['medical_supplies'] = 1
        responder.inventory.items['sterile_iv_fluid_bags'] = 1
        responder.inventory.items['iv_io_administration_sets'] = 1
        patient.status = AgentStatus.INCAPACITATED
        patient.needs.hunger = 0.0
        patient.needs.thirst = 0.0
        patient.needs.o2_supply = 100.0
        patient.needs.temperature_stress = 50.0
        patient.needs._hunger_death_timer = 100
        patient.needs._thirst_death_timer = 0
        patient.action.action_type = 'medical_rest'
        patient.action.target = {
            'habitat': True, 'conscious': False,
            'oral_fluids_allowed': False,
        }
        responder.x, responder.y = patient.x, patient.y
        responder._in_habitat = patient._in_habitat = True
        self.engine._colony_resources['food_reserve_kcal'] = 1400.0
        self.planner.medical_item_available = (
            lambda first_responder, casualty, item: bool(
                self.engine._medical_item_stocks(
                    first_responder, casualty, item
                )
            )
        )
        decision = self.planner._process_tick_internal(
            responder, 100, {},
            {
                'active_events': [], 'temperature_c': 20.0,
                'effective_temperature_c': 20.0,
                'colony_resources': dict(self.engine._colony_resources),
            },
            [{'agent_obj': patient}],
        )
        self.assertEqual('treat', decision['action'])
        self.assertEqual('nutrition_support', decision['target']['protocol'])

        food_before = self.engine._colony_resources['food_reserve_kcal']
        supply_before = responder.inventory.items['medical_supplies']
        ration_before = patient.inventory.items['emergency_rations']
        with patch.object(self.planner, 'process_tick', return_value=decision):
            self.engine._process_agent_tick(responder, [], 0)
        self.assertGreater(patient.needs.hunger, 0.0)
        self.assertEqual(0, patient.needs._hunger_death_timer)
        self.assertEqual(food_before,
                         self.engine._colony_resources['food_reserve_kcal'])
        self.assertEqual(ration_before - 1,
                         patient.inventory.items['emergency_rations'])
        self.assertEqual(supply_before,
                         responder.inventory.items['medical_supplies'])

    def test_conscious_patient_keeps_enteral_supply_for_true_incapacitation(self):
        responder, patient = self.crew
        responder.competency.medical = 7
        responder.x, responder.y = patient.x, patient.y
        responder._in_habitat = patient._in_habitat = True
        responder.inventory.items['medical_supplies'] = 1
        patient.status = AgentStatus.ALIVE
        patient.needs.hunger = 20.0
        patient.needs.thirst = 80.0
        patient.needs.o2_supply = 100.0
        patient.needs.energy = 100.0
        patient.needs.temperature_stress = 50.0
        patient.action.clear()
        self.engine._colony_resources['food_reserve_kcal'] = 1400.0
        self.planner.medical_item_available = (
            lambda first_responder, casualty, item: bool(
                self.engine._medical_item_stocks(
                    first_responder, casualty, item
                )
            )
        )
        context = {
            'active_events': [], 'temperature_c': 20.0,
            'effective_temperature_c': 20.0,
            'colony_resources': dict(self.engine._colony_resources),
        }
        decision = self.planner._process_tick_internal(
            responder, 100, {}, context, [{'agent_obj': patient}],
        )
        self.assertFalse(
            decision.get('action') == 'treat'
            and decision.get('target', {}).get('protocol')
            == 'nutrition_support'
        )
        self.assertEqual(1, responder.inventory.items['medical_supplies'])

        responder.inventory.items.pop('medical_supplies', None)
        patient.status = AgentStatus.INCAPACITATED
        patient.needs.hunger = 0.0
        patient.needs._hunger_death_timer = 100
        patient.action.action_type = 'medical_rest'
        patient.action.target = {
            'habitat': True, 'conscious': False,
            'oral_fluids_allowed': False,
        }
        patient.action.ticks_remaining = 24
        decision = self.planner._process_tick_internal(
            responder, 101, {}, context, [{'agent_obj': patient}],
        )
        self.assertEqual('treat', decision['action'])
        self.assertEqual('nutrition_support', decision['target']['protocol'])

    def test_dual_vital_triage_falls_back_to_available_iv(self):
        responder, patient = self.crew
        responder.competency.medical = 7
        patient.status = AgentStatus.INCAPACITATED
        patient.needs.hunger = patient.needs.thirst = 0.0
        patient.needs.o2_supply = 100.0
        patient.needs.temperature_stress = 50.0
        patient.needs._hunger_death_timer = 100
        patient.needs._thirst_death_timer = 0
        patient.action.action_type = 'medical_rest'
        patient.action.target = {
            'habitat': True, 'conscious': False,
            'oral_fluids_allowed': False,
        }
        responder.x, responder.y = patient.x, patient.y
        responder._in_habitat = patient._in_habitat = True
        for crew in self.crew:
            crew.inventory.items.pop('medical_supplies', None)
        patient.inventory.items.pop('emergency_rations', None)
        patient.inventory.items.pop('ration_pack', None)
        self.engine.central_depot_inventory['ration_packs'] = 0
        responder.inventory.items['sterile_iv_fluid_bags'] = 1
        responder.inventory.items['iv_io_administration_sets'] = 1
        self.engine._colony_resources['food_reserve_kcal'] = 0.0
        self.planner.medical_item_available = (
            lambda first_responder, casualty, item: bool(
                self.engine._medical_item_stocks(
                    first_responder, casualty, item
                )
            )
        )
        decision = self.planner._process_tick_internal(
            responder, 100, {},
            {
                'active_events': [], 'temperature_c': 20.0,
                'effective_temperature_c': 20.0,
                'colony_resources': dict(self.engine._colony_resources),
            },
            [{'agent_obj': patient}],
        )
        self.assertEqual('treat', decision['action'])
        self.assertEqual('iv_io_rehydration', decision['target']['protocol'])

    def test_outbound_rover_contract_survives_completed_self_care(self):
        lead, _buddy = self.crew
        self.planner.shared_work_order = {}
        destination = (self.engine.lz_x - 18, self.engine.lz_y - 5)
        expedition = self.engine._start_expedition(
            lead, 'silica_sand', destination, require_rover=True
        )
        self.assertIsNotNone(expedition)
        self.assertEqual('outbound', expedition['status'])
        lead.action.clear()  # A just-completed drink/meal erased the route.
        decision = self.planner._process_tick_internal(
            lead, 100, {},
            {
                'active_events': [], 'temperature_c': 20.0,
                'effective_temperature_c': 20.0,
                'colony_resources': dict(self.engine._colony_resources),
            },
            [],
        )
        self.assertEqual('move', decision['action'])
        self.assertEqual(destination,
                         (decision['target']['x'], decision['target']['y']))
        self.assertTrue(decision['target']['expedition'])
        self.assertEqual('crew_rover', decision['target']['transport'])

    def test_remote_o2_service_recalls_rover_before_passenger_walks(self):
        lead, buddy = self.crew
        self.assertTrue(
            self.engine._start_construction_rover_trip(lead, self.site)
        )
        for _ in range(10):
            self.engine.current_tick += 1
            self.engine._move_construction_rover_team(
                lead, lead.action.target
            )
            if lead._active_expedition['status'] == 'working':
                break
        self.assertEqual('working', lead._active_expedition['status'])
        rover = self.engine.surface_fleet.crew_rovers[0]
        buddy.inventory.items.pop('oxygen_canisters', None)
        buddy.needs.o2_supply = buddy._current_canister_remaining = 39.0
        station = {
            'id': 'nearby-isru', 'type': 'isru_o2_unit',
            'x': buddy.x - 2, 'y': buddy.y,
            'under_construction': False, 'destroyed': False, 'health': 1.0,
        }
        origin = (buddy.x, buddy.y)
        with patch.object(
            self.engine, '_nearest_o2_filling_station', return_value=station
        ):
            result = self.engine._execute_o2_refill_action(buddy)
        self.assertTrue(result.get('rover_returning'))
        self.assertEqual(origin, (buddy.x, buddy.y))
        self.assertEqual((rover.x, rover.y), (buddy.x, buddy.y))
        self.assertEqual('in_use', rover.state)
        self.assertEqual(
            {'returning'},
            {crew._active_expedition['status'] for crew in (lead, buddy)},
        )

    def test_global_rng_changes_do_not_change_policy_exploration(self):
        agent = self.crew[0]
        agent.rl_epsilon_explore = 1.0
        state = agent._policy_rng.getstate()
        candidates = [{'action': 'sleep', 'target': {}}, {'action': 'stand_watch', 'target': {}}]
        first = [self.planner.select_action_via_rl(agent, 'test', copy.deepcopy(candidates))['action'] for _ in range(20)]
        random.seed(97531)
        for _ in range(150):
            random.random()
        agent._policy_rng.setstate(state)
        second = [self.planner.select_action_via_rl(agent, 'test', copy.deepcopy(candidates))['action'] for _ in range(20)]
        self.assertEqual(first, second)

    def test_construction_spare_request_cannot_divert_return_from_airlock(self):
        agent, buddy = self.crew
        buddy.action.action_type = 'sleep'
        self.engine._start_construction_rover_trip(agent, self.site)
        agent.x, agent.y = self.engine._lander_airlock_exterior_position()
        agent._in_habitat = False
        agent._construction_o2_service = True
        agent.needs.energy = 38.0
        agent._current_canister_remaining = 60.0
        agent.inventory.items.pop('oxygen_canisters', None)
        decision = self.engine._construction_preparation_decision(agent)
        self.assertEqual('move', decision['action'])
        self.assertEqual('shelter', decision['target']['destination'])
        self.assertFalse(agent._construction_o2_service)

    def test_sar_alarm_does_not_cancel_required_preflight_sleep(self):
        victim, responder = self.crew
        victim.status = AgentStatus.INCAPACITATED
        victim._in_habitat = False
        victim.x, victim.y = self.engine.lz_x+5, self.engine.lz_y
        responder.needs.energy = 60.0
        responder.x, responder.y = self.engine._indoor_activity_position(responder, 'sleep')
        responder.action.action_type = 'sleep'
        responder.action.target = {'habitat': True, 'preflight_recovery': True, 'required_energy_pct': 85.0, 'ticks': 12}
        responder.action.ticks_remaining = 12
        for tick in range(4):
            self.engine.current_tick = tick
            self.engine._process_agent_tick(responder, [], 1)
        self.assertEqual('sleep', responder.action.action_type)
        self.assertFalse(responder.action.target.get('woken_by_team_emergency'))
        self.assertGreater(responder.needs._consecutive_sleep_ticks, 1)

    def test_sar_prefers_ready_responder_to_exhausted_expert(self):
        victim, expert = self.crew
        victim._in_habitat = False
        expert.needs.energy = 30.0
        expert.competency.medical = 10
        ready = copy.deepcopy(expert)
        ready.id = 'ready-responder'
        ready.needs.energy = 100.0
        ready.competency.medical = 4
        self.assertEqual(ready.id, self.planner.select_sar_rescuer_id(victim, [victim, expert, ready]))

    def test_sheltered_expedition_buddy_recovers_instead_of_standing_watch(self):
        buddy, lead = self.crew
        buddy.x, buddy.y = self.engine.lz_x + 12, self.engine.lz_y + 8
        buddy._in_habitat = True
        buddy.needs.energy = 10.0
        expedition_id = 'field-watch-regression'
        buddy._active_expedition = {
            'id': expedition_id, 'role': 'buddy', 'lead_id': lead.id,
            'status': 'working', 'kind': 'resource_recovery',
            'resource': 'iron_ore', 'transport': 'on_foot',
            'move_speed_cells': 2, 'authorized_radius': 30,
        }
        lead._active_expedition = {
            'id': expedition_id, 'role': 'lead', 'buddy_id': buddy.id,
            'status': 'working', 'kind': 'resource_recovery',
            'resource': 'iron_ore', 'transport': 'on_foot',
            'move_speed_cells': 2, 'authorized_radius': 30,
        }
        decision = self.planner.process_tick(
            buddy, 1, {},
            {'temperature_c': 20.0, 'effective_temperature_c': 20.0,
             'colony_resources': dict(self.engine._colony_resources)},
            [],
        )
        self.assertEqual('sleep', decision['action'])
        self.assertEqual('returning', buddy._active_expedition['status'])
        self.assertEqual('returning', lead._active_expedition['status'])

    def test_on_foot_expedition_return_uses_rough_terrain_pace(self):
        lead, buddy = self.crew
        lead.x, lead.y = self.engine.lz_x + 20, self.engine.lz_y
        lead._in_habitat = False
        expedition_id = 'rough-return-regression'
        lead._active_expedition = {
            'id': expedition_id, 'role': 'lead', 'buddy_id': buddy.id,
            'status': 'working', 'kind': 'resource_recovery',
            'resource': 'iron_ore', 'transport': 'on_foot',
            'move_speed_cells': 2, 'authorized_radius': 30,
        }
        buddy._active_expedition = {
            'id': expedition_id, 'role': 'buddy', 'lead_id': lead.id,
            'status': 'working', 'kind': 'resource_recovery',
            'resource': 'iron_ore', 'transport': 'on_foot',
            'move_speed_cells': 2, 'authorized_radius': 30,
        }
        self.planner.expedition_available_ticks = 25
        decision = self.planner.process_tick(
            lead, 1, {},
            {'temperature_c': 20.0, 'effective_temperature_c': 20.0,
             'colony_resources': dict(self.engine._colony_resources)},
            [],
        )
        self.assertEqual('move', decision['action'])
        self.assertTrue(decision['target']['forced_return'])
        self.assertEqual('returning', lead._active_expedition['status'])

    def test_resource_rover_returns_both_seats_at_vehicle_pace(self):
        lead, buddy = self.crew
        destination = (self.engine.lz_x + 18, self.engine.lz_y + 10)
        expedition = self.engine._start_expedition(
            lead, 'basalt', destination, require_rover=True
        )
        self.assertIsNotNone(expedition)
        rover = self.engine.surface_fleet.crew_rover_for_expedition(expedition['id'])
        for _ in range(12):
            self.engine.current_tick += 1
            self.assertTrue(self.engine._move_resource_rover_team(
                lead, {'x': destination[0], 'y': destination[1], 'expedition': True}
            ))
            recorded = rover.total_distance_km
            self.assertTrue(self.engine._move_resource_rover_team(buddy, buddy.action.target))
            self.assertEqual(recorded, rover.total_distance_km)
            if expedition['status'] == 'working':
                break
        self.assertEqual('working', expedition['status'])
        self.assertEqual(destination, (lead.x, lead.y))
        self.assertEqual((lead.x, lead.y), (buddy.x, buddy.y))
        before_return = (lead.x, lead.y)
        self.engine.current_tick += 1
        self.engine._move_resource_rover_team(lead, {'destination': 'shelter'})
        travelled = abs(lead.x-before_return[0]) + abs(lead.y-before_return[1])
        self.assertGreater(travelled, 1, 'a returning rover must not use the one-cell pedestrian motor')
        self.assertEqual((lead.x, lead.y), (buddy.x, buddy.y))
        for _ in range(12):
            if lead._active_expedition is None:
                break
            self.engine.current_tick += 1
            self.engine._move_resource_rover_team(lead, {'destination': 'shelter'})
        self.assertIsNone(lead._active_expedition)
        self.assertIsNone(buddy._active_expedition)
        self.assertTrue(lead._in_habitat and buddy._in_habitat)
        self.assertEqual(1, rover.completed_jobs)
        self.assertGreater(rover.total_distance_km, 3.0)

    def test_resource_rover_fault_downgrades_both_seats_to_walkback(self):
        lead, buddy = self.crew
        expedition = self.engine._start_expedition(
            lead, 'basalt', (self.engine.lz_x+18, self.engine.lz_y+10),
            require_rover=True,
        )
        self.assertIsNotNone(expedition)
        rover = self.engine.surface_fleet.crew_rover_for_expedition(expedition['id'])
        rover.state = 'fault'
        self.assertFalse(self.engine._move_resource_rover_team(lead, {'destination': 'shelter'}))
        for crew in self.crew:
            self.assertEqual('on_foot', crew._active_expedition['transport'])
            self.assertEqual(1, crew._active_expedition['move_speed_cells'])

    def test_rover_does_not_claim_indoor_recovery_moves(self):
        lead, buddy = self.crew
        self.engine._start_expedition(lead, 'basalt',
            (self.engine.lz_x + 18, self.engine.lz_y + 10), require_rover=True)
        target = {'x': self.engine.lz_x, 'y': self.engine.lz_y,
                  'destination': 'indoor_activity', 'indoor_activity': 'sleep'}
        for crew in (lead, buddy):
            self.assertFalse(self.engine._move_resource_rover_team(crew, target))
        self.assertIsNone(self.engine.airlock.active)

    def test_northbound_surface_walk_preserves_hull_boundary_and_return_reserve(self):
        crew = self.crew[0]
        crew.x, crew.y = self.engine._lander_airlock_exterior_position()
        crew._in_habitat = False
        reserve = {'energy': 0.1, 'o2': 0.03}
        self.engine.airlock.return_reserves[crew.id] = dict(reserve)
        target = (self.engine.lz_x + 2, self.engine.lz_y - 18)
        for _ in range(40):
            self.engine.current_tick += 1
            dx, dy = self.engine._cardinal_step_toward(crew, *target, 2)
            crew.x += dx
            crew.y += dy
            self.assertFalse(self.engine._is_lander_footprint_cell(crew.x, crew.y))
            if (crew.x, crew.y) == target:
                break
        self.assertEqual(target, (crew.x, crew.y))
        self.assertEqual(reserve, self.engine.airlock.return_reserves[crew.id])
        self.assertIsNone(self.engine.airlock.active)

    def test_rover_rendezvous_does_not_move_or_wake_recovering_buddy(self):
        lead, buddy = self.crew
        self.engine._start_expedition(lead, 'basalt',
            (self.engine.lz_x + 18, self.engine.lz_y + 10), require_rover=True)
        buddy.x, buddy.y = self.engine.lz_x, self.engine.lz_y
        buddy.action.action_type = 'sleep'
        buddy.action.target = {'habitat': True}
        buddy.action.ticks_remaining = 8
        before = (buddy.x, buddy.y, buddy.action.to_dict())
        self.engine._move_resource_rover_team(lead, {'expedition': True})
        self.assertEqual(before, (buddy.x, buddy.y, buddy.action.to_dict()))
        self.assertEqual('stand_watch', lead.action.action_type)
        self.assertEqual('buddy_completing_preflight', lead.action.target['reason'])
        self.assertIsNone(self.engine.airlock.active)

    def test_resumed_rover_survey_moves_both_seats_and_charges_vehicle(self):
        lead, buddy = self.crew
        destination = (self.engine.lz_x + 18, self.engine.lz_y + 10)
        expedition = self.engine._start_regional_survey_expedition(
            lead, 'basalt', destination, ['basalt'], survey_scope='local')
        self.assertIsNotNone(expedition)
        for _ in range(2):
            self.engine.current_tick += 1
            self.engine._move_resource_rover_team(lead, {'expedition': True})
        self.assertFalse(lead._in_habitat)
        rover = self.engine.surface_fleet.crew_rover_for_expedition(expedition['id'])
        before = rover.total_distance_km
        lead.inventory.tool_durability['portable_scanner'] = 100
        lead.inventory.tool_charge_pct['portable_scanner'] = 100.0
        lead.action.clear()
        decision = {'action': 'survey_resources', 'deterministic': True,
            'target': {'x': destination[0], 'y': destination[1],
                       'resource': 'basalt', 'survey_action': 'portable_scanner'}}
        self.engine.current_tick += 1
        with patch.object(self.planner, 'process_tick', return_value=decision):
            self.engine._process_agent_tick(lead, [], 1)
        self.assertEqual((lead.x, lead.y), (buddy.x, buddy.y))
        self.assertEqual((lead.x, lead.y), (rover.x, rover.y))
        self.assertGreater(rover.total_distance_km, before)

    def test_rover_never_teleports_a_separated_passenger(self):
        lead, buddy = self.crew
        expedition = self.engine._start_expedition(lead, 'basalt',
            (self.engine.lz_x + 18, self.engine.lz_y + 10), require_rover=True)
        for _ in range(3):
            self.engine.current_tick += 1
            self.engine._move_resource_rover_team(lead, {'expedition': True})
        buddy.x += 3
        before = (buddy.x, buddy.y)
        rover = self.engine.surface_fleet.crew_rover_for_expedition(expedition['id'])
        distance = rover.total_distance_km
        self.engine.current_tick += 1
        self.assertFalse(self.engine._move_resource_rover_team(lead, {'destination': 'shelter'}))
        self.assertEqual(before, (buddy.x, buddy.y))
        self.assertEqual(distance, rover.total_distance_km)
        self.assertEqual('on_foot', buddy._active_expedition['transport'])

    def test_construction_departure_waits_for_buddy_recovery_and_rendezvous(self):
        lead, buddy = self.crew
        self.engine._start_construction_rover_trip(lead, self.site)
        buddy.x, buddy.y = self.engine.lz_x, self.engine.lz_y
        buddy.action.action_type = 'sleep'
        buddy.action.ticks_remaining = 8
        before = (buddy.x, buddy.y)
        self.engine.current_tick += 1
        self.engine._move_construction_rover_team(lead, lead.action.target)
        self.assertEqual(before, (buddy.x, buddy.y))
        self.assertEqual('sleep', buddy.action.action_type)
        self.assertIsNone(self.engine.airlock.active)
        buddy.action.clear()
        for _ in range(8):
            self.engine.current_tick += 1
            self.engine._move_construction_rover_team(lead, lead.action.target)
            if not lead._in_habitat:
                break
        self.assertFalse(lead._in_habitat)
        self.assertEqual((lead.x, lead.y), (buddy.x, buddy.y))

    def test_scanner_recharge_returns_occupied_rover_as_a_pair(self):
        lead, buddy = self.crew
        expedition = self.engine._start_regional_survey_expedition(lead, 'basalt',
            (self.engine.lz_x + 18, self.engine.lz_y + 10), ['basalt'])
        for _ in range(3):
            self.engine.current_tick += 1
            self.engine._move_resource_rover_team(lead, {'expedition': True})
        lead.inventory.tool_charge_pct['portable_scanner'] = 0.0
        lead.action.clear()
        self.engine.current_tick += 1
        with patch.object(self.planner, 'process_tick', return_value={
            'action': 'recharge_scanner', 'target': {}, 'deterministic': True}):
            self.engine._process_agent_tick(lead, [], 1)
        rover = self.engine.surface_fleet.crew_rover_for_expedition(expedition['id'])
        self.assertEqual((lead.x, lead.y), (buddy.x, buddy.y))
        self.assertEqual((lead.x, lead.y), (rover.x, rover.y))
        self.assertEqual('returning', expedition['status'])

    def test_unlaunched_survey_releases_rover_after_preflight_deadline(self):
        lead, buddy = self.crew
        expedition = self.engine._start_regional_survey_expedition(lead, 'basalt',
            (self.engine.lz_x + 18, self.engine.lz_y + 10), ['basalt'])
        rover = self.engine.surface_fleet.crew_rover_for_expedition(expedition['id'])
        self.engine.current_tick += 48
        lead.action.clear()
        with patch.object(self.planner, 'process_tick', return_value={
            'action': 'sleep', 'target': {'habitat': True}, 'deterministic': True}):
            self.engine._process_agent_tick(lead, [], 1)
        self.assertIsNone(lead._active_expedition)
        self.assertIsNone(buddy._active_expedition)
        self.assertIsNone(rover.mission)
        self.assertEqual(0, rover.total_distance_km)


    def test_construction_can_reuse_partly_charged_rover_with_safe_return_energy(self):
        rover = self.engine.surface_fleet.crew_rovers[0]
        rover.state = 'charging'
        rover.battery_kwh = 26.64
        self.assertTrue(self.engine._start_construction_rover_trip(self.crew[0], self.site))
        self.assertIsInstance(self.crew[0]._active_expedition, dict)
        self.assertEqual('in_use', rover.state)
        self.assertAlmostEqual(26.64, rover.battery_kwh)

    def test_partly_charged_rover_still_requires_round_trip_and_emergency_reserve(self):
        fleet = self.engine.surface_fleet
        rover = fleet.crew_rovers[0]
        rover.state = 'charging'
        rover.battery_kwh = 9.0
        result = fleet.begin_crew_rover_trip(
            expedition_id='insufficient-battery', crew_ids=[c.id for c in self.crew],
            target_x=self.site['x'], target_y=self.site['y'])
        self.assertFalse(result['reserved'])
        self.assertEqual('insufficient_return_energy', result['reason'])
        self.assertEqual('charging', rover.state)
        self.assertEqual(9.0, rover.battery_kwh)
        self.assertIsNone(rover.mission)


    def test_outdoor_emergency_oxygen_survives_next_physical_tick(self):
        patient, medic = self.crew
        for crew in (patient, medic):
            crew.x, crew.y = self.engine.lz_x, self.engine.lz_y + 6
            crew._in_habitat = False
            crew.action.clear()
        patient.inventory.items.pop('oxygen_canisters', None)
        patient._has_active_o2_canister = True
        patient._current_canister_remaining = patient.needs.o2_supply = 10.0
        medic.inventory.items['oxygen_canisters'] = 1
        medic.competency.medical = 5
        empty_before = patient.inventory.items.get('empty_oxygen_canisters', 0)
        decision = {'action': 'treat', 'target': {'patient_id': patient.id,
                    'protocol': 'emergency_oxygen'}, 'deterministic': True}
        with patch.object(self.planner, 'process_tick', return_value=decision):
            self.engine._process_agent_tick(medic, [], 1)
        self.assertEqual(0, medic.inventory.items.get('oxygen_canisters', 0))
        self.assertEqual(0, patient.inventory.items.get('oxygen_canisters', 0))
        self.assertEqual(100.0, patient._current_canister_remaining)
        self.assertEqual(empty_before + 1, patient.inventory.items.get('empty_oxygen_canisters', 0))
        patient.tick_update(ambient_temp_c=20.0, has_shelter=False, has_atmosphere=False)
        self.assertGreater(patient.needs.o2_supply, 90.0)
        self.assertEqual(patient._current_canister_remaining, patient.needs.o2_supply)


    def test_cancelled_construction_buddy_releases_unlaunched_rover(self):
        lead, buddy = self.crew
        self.engine._start_construction_rover_trip(lead, self.site)
        rover = self.engine.surface_fleet.crew_rovers[0]
        positions = [(c.x, c.y) for c in self.crew]
        battery = rover.battery_kwh
        buddy._active_expedition = None
        self.assertTrue(self.engine._move_construction_rover_team(lead, lead.action.target))
        self.assertIsNone(lead._active_expedition)
        self.assertIsNone(rover.mission)
        self.assertEqual(positions, [(c.x, c.y) for c in self.crew])
        self.assertEqual(battery, rover.battery_kwh)

    def test_missing_construction_lead_in_field_preserves_positions_and_returns_on_foot(self):
        lead, buddy = self.crew
        self.engine._start_construction_rover_trip(lead, self.site)
        rover = self.engine.surface_fleet.crew_rovers[0]
        for c in self.crew:
            c.x, c.y = self.site['x'], self.site['y']
            c._in_habitat = False
        rover.x, rover.y = self.site['x'], self.site['y']
        lead._active_expedition = None
        positions = [(c.x, c.y) for c in self.crew]
        self.assertTrue(self.engine._move_construction_rover_team(buddy, buddy.action.target))
        self.assertEqual(positions, [(c.x, c.y) for c in self.crew])
        self.assertEqual('on_foot', buddy._active_expedition['transport'])
        self.assertEqual('returning', buddy._active_expedition['status'])
        self.assertEqual('fault', rover.state)

    def test_cancelled_survey_buddy_does_not_overwrite_new_assignment(self):
        lead, buddy = self.crew
        expedition = self.engine._start_regional_survey_expedition(lead, 'basalt',
            (self.engine.lz_x + 18, self.engine.lz_y + 10), ['basalt'])
        rover = self.engine.surface_fleet.crew_rover_for_expedition(expedition['id'])
        replacement = {'id': 'unrelated-new-assignment', 'status': 'outbound'}
        buddy._active_expedition = replacement
        self.assertTrue(self.engine._move_resource_rover_team(lead, {
            'x': expedition['target_x'], 'y': expedition['target_y'],
            'destination': 'resource_scan', 'expedition': True}))
        self.assertIsNone(lead._active_expedition)
        self.assertIs(buddy._active_expedition, replacement)
        self.assertEqual({'id': 'unrelated-new-assignment', 'status': 'outbound'}, replacement)
        self.assertIsNone(rover.mission)


    def test_rescue_approach_cannot_enter_hull_as_a_diagonal_shortcut(self):
        rescuer, patient = self.crew
        rescuer.x, rescuer.y = self.engine.lz_x - 2, self.engine.lz_y
        patient.x, patient.y = self.engine.lz_x, self.engine.lz_y + 5
        rescuer._in_habitat = patient._in_habitat = False
        decision = {'action': 'rescue', 'target': {'victim_id': patient.id}, 'deterministic': True}
        for _ in range(2):
            rescuer.action.clear()
            self.engine.current_tick += 1
            with patch.object(self.planner, 'process_tick', return_value=decision):
                self.engine._process_agent_tick(rescuer, [], 1)
            self.assertFalse(self.engine._is_lander_footprint_cell(rescuer.x, rescuer.y))
        self.assertEqual(0, self.engine.airlock.completed_cycles)

    def test_machine_approach_cannot_enter_hull_from_exterior_apron(self):
        worker = self.crew[0]
        worker.x, worker.y = self.engine._lander_airlock_exterior_position()
        worker._in_habitat = False
        worker.competency.engineering = 10
        machine = next(s for s in self.engine.placed_structures if s['type'] == 'cnc_fabricator')
        machine['x'], machine['y'] = self.engine.lz_x - 4, self.engine.lz_y - 4
        decision = {'action': 'refine', 'target': {'output': 'pressure_vessel_section',
                    'machine_id': machine['id']}, 'deterministic': True}
        for _ in range(2):
            worker.action.clear()
            self.engine.current_tick += 1
            with patch.object(self.planner, 'process_tick', return_value=decision):
                self.engine._process_agent_tick(worker, [], 1)
            self.assertFalse(self.engine._is_lander_footprint_cell(worker.x, worker.y))
        self.assertEqual(0, self.engine.airlock.completed_cycles)


class WaterQueueRegressionTest(unittest.TestCase):
    def test_power_loss_keeps_water_and_original_warmup_deadlines(self):
        engine = fixture.SimulationEngine.__new__(fixture.SimulationEngine)
        engine.current_tick = 10
        engine._water_recovery_queue = [
            {'ready_tick': 4, 'liters': 1.25, 'source': 'habitat_metabolic_wastewater'},
            {'ready_tick': 8, 'liters': 2.5, 'source': 'habitat_metabolic_wastewater'},
            {'ready_tick': 11, 'liters': 0.75, 'source': 'habitat_metabolic_wastewater'},
        ]
        self.assertEqual(0.0, engine._process_water_recovery_queue(online=False))
        self.assertEqual(2, len(engine._water_recovery_queue))
        self.assertEqual(4.5, engine._water_mass_ledger()['treatment_queue_l'])
        self.assertEqual(3.75, engine._process_water_recovery_queue(online=True))
        self.assertEqual(0.75, engine._water_mass_ledger()['treatment_queue_l'])
        engine.current_tick = 11
        self.assertEqual(0.75, engine._process_water_recovery_queue(online=True))
        self.assertEqual([], engine._water_recovery_queue)

    def test_long_outage_has_bounded_batches_and_conserves_captured_mass(self):
        engine = fixture.SimulationEngine.__new__(fixture.SimulationEngine)
        engine._water_recovery_queue = []
        for tick in range(10000):
            engine.current_tick = tick
            for _ in range(6):
                engine._water_recovery_queue.append({
                    'ready_tick': tick + 6, 'liters': 0.01,
                    'source': 'habitat_metabolic_wastewater'})
            self.assertEqual(0.0, engine._process_water_recovery_queue(online=False))
            self.assertLessEqual(len(engine._water_recovery_queue), 37)
        self.assertAlmostEqual(600.0, engine._water_mass_ledger()['treatment_queue_l'])
        engine.current_tick = 10006
        self.assertAlmostEqual(600.0, engine._process_water_recovery_queue(online=True), places=7)
        self.assertEqual([], engine._water_recovery_queue)


class RoutingCacheRegressionTest(unittest.TestCase):
    def test_route_queries_each_cell_once_and_refreshes_next_query(self):
        from types import SimpleNamespace
        engine = fixture.SimulationEngine.__new__(fixture.SimulationEngine)
        engine.current_tick = 0
        calls = []
        def cell(x, y, tick):
            calls.append((x, y, tick))
            return {'traversable': tick == 0 and (x, y) not in {(12, 10), (10, 12)},
                    'traversal_cost': 1.0}
        engine.world = SimpleNamespace(map_size=100, get_cell_info=cell)
        route = engine._surface_vehicle_route(10, 10, 14, 14)
        self.assertEqual((10, 10), route[0])
        self.assertEqual((14, 14), route[-1])
        self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1
                            for a, b in zip(route, route[1:])))
        self.assertEqual(len(calls), len(set(calls)))
        engine.current_tick = 1
        self.assertEqual([], engine._surface_vehicle_route(10, 10, 14, 14))
        self.assertTrue(any(tick == 1 for _, _, tick in calls))

    def test_scan_geometry_cache_cannot_be_mutated_or_reused_after_radius_change(self):
        engine = fixture.SimulationEngine.__new__(fixture.SimulationEngine)
        expected = engine._portable_scan_station_offsets()
        with patch.object(engine, '_square_ring_offset', wraps=engine._square_ring_offset) as build:
            result = engine._portable_scan_station_offsets()
            self.assertEqual(expected, result)
            result.clear()
            self.assertEqual(expected, engine._portable_scan_station_offsets())
            build.assert_not_called()
            engine.LOCAL_EVA_RADIUS_CELLS += 3
            self.assertNotEqual(expected, engine._portable_scan_station_offsets())
            self.assertGreater(build.call_count, 0)


if __name__ == '__main__':
    unittest.main()
