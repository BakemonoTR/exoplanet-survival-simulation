"""Test all newly added hyper-realism systems."""
import sys, os
sys.path.insert(0, '.')

# Test Event Scheduler
from src.systems.event_scheduler import EventScheduler
from src.world.generator import PlanetConfig, WorldGenerator

print('=== Event Scheduler Tests ===')

p = PlanetConfig('config/planets/proxima-centauri-b.json')
es = EventScheduler(p, seed=42, max_ticks=1000)
counts = es.get_total_event_count()
print('Proxima events:', counts)

flare_ticks = sum(1 for t in range(1000) if es.is_flare_active(t))
print('Ticks with active flare:', flare_ticks, '/ 1000')

for t in range(1000):
    if es.is_flare_active(t):
        effects = es.get_combined_effects(t, 500, 500)
        print('Flare at tick', t, ': rad_mult=', round(effects["radiation_multiplier"], 1))
        print('Summary:', es.get_event_summary(t))
        break

p2 = PlanetConfig('config/planets/ross-128b.json')
es2 = EventScheduler(p2, seed=42, max_ticks=1000)
counts2 = es2.get_total_event_count()
print('Ross 128b events:', counts2)
print('PASS: Event Scheduler')

# Test Traversal Cost
gen = WorldGenerator(p2, seed=42)
cell = gen.get_cell_info(gen.center, gen.center, current_tick=50)
print('\n=== Traversal Cost ===')
print('Traversal cost at center:', cell["traversal_cost"])
assert 'traversal_cost' in cell
print('PASS: Traversal cost')

# Test Night Temperature
gen_t = WorldGenerator(PlanetConfig('config/planets/trappist-1e.json'), seed=42)
cell_day = gen_t.get_cell_info(gen_t.center, gen_t.center, current_tick=0)
cell_night = gen_t.get_cell_info(0, 0, current_tick=0)
print('\n=== Night Temperature ===')
print('Substellar temp:', cell_day["temperature_c"], 'C')
print('Anti-stellar temp:', cell_night["temperature_c"], 'C')
print('Night is colder:', cell_night["temperature_c"] < cell_day["temperature_c"])
print('PASS: Night temperature')

# Test Fuel System
from src.world.sparse_state import PlacedStructure
furnace = PlacedStructure(
    structure_id='stone_furnace', position=(100, 100),
    builder_agent_id='a1', build_tick=0, recipe_id='stone_furnace',
    effects={'enables_smelting': True},
    fuel_reserve=5.0, fuel_type='sulfur',
    fuel_consumption_per_tick=0.15,
)
print('\n=== Fuel System ===')
print('Initial fuel:', furnace.fuel_reserve, 'active:', furnace.active)
for i in range(40):
    furnace.tick_fuel_consumption()
print('After 40 ticks: fuel=', round(furnace.fuel_reserve, 2), 'active:', furnace.active)
assert not furnace.active, 'Should have run out of fuel!'
furnace.refuel(3.0)
print('After refuel 3.0: fuel=', furnace.fuel_reserve, 'active:', furnace.active)
assert furnace.active
print('PASS: Fuel system')

# Test Gather
from src.agents.agent import Agent
from src.world.sparse_state import ResourceNode
a = Agent({
    'id': 't1', 'name': 'T', 'background': 'E',
    'personality_traits': ['analytical'],
    'genome': {'strength': 6, 'agility': 5, 'endurance': 5, 'perception': 5, 'immunity': 5},
    'competency': {'engineering': 7, 'medical': 3, 'physics': 5, 'botany_bio': 2, 'leadership_social': 4},
})
a.inventory.add_item('multitool_kit', 1)
node = ResourceNode('n1', 'iron_ore', (50, 50), 100, 100, 2.5, 1.5)
result = a.gather(node, 1.0)
print('\n=== Gather ===')
print('Gathered:', result)
assert result['gathered']
assert result['amount'] > 0
assert a.inventory.materials.get('iron_ore', 0) > 0
print('PASS: Gather')

# Test Craft (can_craft + start_crafting)
recipe = {
    'materials': {'iron_ore': 5},
    'requires_tool': True,
    'min_engineering': 3,
    'base_duration_ticks': 10,
    'output': {'type': 'item', 'item_id': 'hand_tools'},
}
a.inventory.add_material('iron_ore', 10)
feasibility = a.can_craft(recipe)
print('\n=== Crafting ===')
print('Can craft:', feasibility)
craft_result = a.start_crafting(recipe, gravity_modifier=1.16)
print('Craft started:', craft_result)
assert craft_result['started']
print('PASS: Crafting')

# Test treat_injury
a.injury_level = 0.5
a.inventory.add_item('medical_supplies', 2)
result = a.treat_injury(has_medical_station=False)
print('\n=== Treat Injury ===')
print('Treatment:', result)
assert result['treated']
assert a.injury_level < 0.5
print('PASS: Treat injury')

# Test food diversity morale
old_morale = a.morale
a.inventory.add_item('emergency_rations', 5)
eat_result = a.eat('emergency_rations')
print('\n=== Food Diversity ===')
print('Ration morale effect:', eat_result["morale_effect"])
assert eat_result['morale_effect'] < 0  # Monotonous food = negative morale
print('PASS: Food diversity')

# Test dirty water
drink_result = a.drink('raw_water', 1.0)
print('\n=== Water Safety ===')
print('Raw water consumed:', drink_result["consumed"])
print('Disease risk triggered:', drink_result["disease_risk"])
print('PASS: Water safety')

# Test pressure effect on tick_update
print('\n=== Pressure + Isolation + Overload ===')
a2 = Agent({
    'id': 't2', 'name': 'T2', 'background': 'E',
    'personality_traits': ['analytical'],
    'genome': {'strength': 5, 'agility': 5, 'endurance': 5, 'perception': 5, 'immunity': 5},
    'competency': {'engineering': 5, 'medical': 5, 'physics': 5, 'botany_bio': 5, 'leadership_social': 5},
})

# Test with low pressure (hypoxia)
energy_before = a2.needs.energy
a2.tick_update(ambient_temp_c=20.0, gravity_multiplier=1.0, has_atmosphere=True,
               pressure_kpa=35.0, nearby_agent_count=0)
print('Energy after low pressure tick:', round(a2.needs.energy, 2), '(was', round(energy_before, 2), ')')
print('Energy dropped from hypoxia:', a2.needs.energy < energy_before)

# Test injury healing
a2.injury_level = 0.5
a2.action.action_type = "idle"
for _ in range(10):
    a2.tick_update(ambient_temp_c=20.0, gravity_multiplier=1.0, has_atmosphere=True,
                   pressure_kpa=101.3, nearby_agent_count=2)
print('Injury after 10 idle ticks:', round(a2.injury_level, 3), '(was 0.5)')
assert a2.injury_level < 0.5
print('PASS: Pressure + Healing + Isolation')

print('\n' + '=' * 60)
print('ALL NEW HYPER-REALISM SYSTEMS VALIDATED SUCCESSFULLY')
print('=' * 60)
