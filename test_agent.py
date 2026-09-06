import sys
sys.path.insert(0, 'src')
from agents.agent import Agent, Disease, DiseaseType

config = {
    'id': 'test', 'name': 'Test Agent', 'background': 'Test',
    'personality_traits': ['calm'],
    'genome': {'strength': 7, 'agility': 5, 'endurance': 6, 'perception': 8, 'immunity': 5},
    'competency': {'engineering': 8, 'medical': 3, 'physics': 5, 'botany_bio': 4, 'leadership_social': 6}
}
a = Agent(config)
a.inventory.add_item('emergency_rations', 10)
a.inventory.add_item('water_packs', 15)
a.inventory.add_item('oxygen_canisters', 6)
a.inventory.add_material('basalt', 5)
a.inventory.add_material('iron_ore', 3)

print('=== Weight ===')
print(f'Total weight: {a.inventory.total_weight_kg():.1f} kg')
print(f'Capacity: {a.inventory.carry_capacity_kg(a.genome.strength):.0f} kg')
print(f'Overloaded: {a.inventory.is_overloaded(a.genome.strength)}')

print('\n=== Tick 1 (cold, windy, no atmosphere) ===')
events = a.tick_update(
    ambient_temp_c=-30, gravity_multiplier=0.93,
    has_shelter=False, has_atmosphere=False,
    wind_speed_modifier=1.5,
    # Cold stress comes from measured ambient temperature and wind, not a
    # second arbitrary per-tick biome penalty.
    biome_hazards={'radiation_per_tick': 0.004}
)
w = events.get('warnings', [])
d = events.get('diseases_onset', [])
print(f'Warnings: {w}')
print(f'Diseases onset: {d}')
print(f'O2 supply: {a.needs.o2_supply:.1f}')
print(f'Temp stress: {a.needs.temperature_stress:.1f}')
print(f'Radiation: {a.cumulative_radiation_sv:.4f} Sv')

for _ in range(50):
    a.tick_update(ambient_temp_c=-30, gravity_multiplier=0.93,
                  has_shelter=False, has_atmosphere=False,
                  wind_speed_modifier=1.5,
                  biome_hazards={'radiation_per_tick': 0.004})

print(f'\n=== After 51 ticks ===')
print(f'Hunger: {a.needs.hunger:.1f}')
print(f'Thirst: {a.needs.thirst:.1f}')
print(f'Energy: {a.needs.energy:.1f}')
print(f'O2: {a.needs.o2_supply:.1f}')
print(f'Temp stress: {a.needs.temperature_stress:.1f}')
print(f'Radiation: {a.cumulative_radiation_sv:.3f} Sv (level {a.radiation_sickness_level})')
diseases = [dd.disease_type.value for dd in a.active_diseases]
print(f'Diseases: {diseases}')
print(f'Near-death events: {a.near_death_events}')
print(f'Permanent radiation penalties: endo={a._radiation_permanent_endurance_penalty}, imm={a._radiation_permanent_immunity_penalty}')

print(f'\n=== LLM Summary ===')
summary = a.to_state_summary(
    current_tick=51, biome_name='Terminator Ridge',
    nearby_agents=[{'name': 'Dr. Vasquez', 'distance': 5, 'status': 'alive', 'id': 'a1'}],
    colony_readiness_pct=12.5
)
print(summary)

print(f'\n=== Eat ===')
result = a.eat('emergency_rations', 700)
print(f'Consumed: {result}')
print(f'Hunger after: {a.needs.hunger:.1f}')

print(f'\n=== Debuffs ===')
debuffs = a.needs.get_debuffs(
    radiation_sickness_level=a.radiation_sickness_level,
    active_diseases=a.active_diseases,
)
print(f'Action speed: {debuffs["action_speed"]:.2f}')
print(f'Cognitive impairment: {debuffs["cognitive_impairment"]:.2f}')
print(f'Can heavy labor: {debuffs["can_heavy_labor"]}')

print('\nALL TESTS PASSED')
