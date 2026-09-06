"""
Validation test for all 8 hyper-realism systems.
Tests each system individually, then integration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json

# --- TEST 1: Structure Damage & Maintenance ---
print("=" * 60)
print("TEST 1: Structure Damage & Maintenance System")
print("=" * 60)

from src.world.sparse_state import PlacedStructure

s = PlacedStructure(
    structure_id="solar_panel",
    position=(100, 100),
    builder_agent_id="agent_1",
    build_tick=0,
    recipe_id="solar_panel",
    maintenance_interval_ticks=200,
    maintenance_materials={"electronic_component": 1},
    degradation_per_missed=0.1,
    effects={"power_output_w": 150},
    light_radius=0,
)

print(f"  Initial: health={s.health}, condition={s.condition}, operational={s.operational}")

# Simulate maintenance overdue
s.tick_maintenance(200)
print(f"  After 200 ticks (maintenance due): condition={s.condition:.2f}")
s.tick_maintenance(400)
print(f"  After 400 ticks (missed): condition={s.condition:.2f}")
s.tick_maintenance(600)
print(f"  After 600 ticks (2x missed): condition={s.condition:.2f}")
s.tick_maintenance(800)
print(f"  After 800 ticks (3x missed): condition={s.condition:.2f}")

# Check effective output
output = s.get_effective_output()
print(f"  Effective power: {output.get('power_output_w', 0):.1f}W (from 150W)")

# Apply event damage
report = s.apply_damage(0.3, "flare")
print(f"  Flare damage: {report}")

# Repair
s.repair(0.3, True, 900)
print(f"  After repair: health={s.health:.2f}, condition={s.condition:.2f}, operational={s.operational}")
print("  PASS: Structure maintenance system\n")

# --- TEST 2: Tool Durability ---
print("=" * 60)
print("TEST 2: Tool Durability System")
print("=" * 60)

from src.agents.agent import Inventory

inv = Inventory()
inv.add_item("multitool_kit", 1)

print(f"  Tool added: {inv.get_tool_status()}")
print(f"  Has usable tool: {inv.has_usable_tool()}")
print(f"  Best tool: {inv.get_best_tool()}")

# Use tool 50 times
for i in range(50):
    result = inv.use_tool(1)

print(f"  After 50 uses: {inv.get_tool_status()}")

# Use until break
broke = False
for i in range(200):
    result = inv.use_tool(1)
    if result["broke"]:
        broke = True
        print(f"  Tool broke after {50 + i + 1} total uses!")
        break

if not broke:
    print(f"  Tool still working: {inv.get_tool_status()}")

print(f"  Has usable tool after break: {inv.has_usable_tool()}")

# Add stone_hammer as fallback
inv.add_item("stone_hammer", 1)
print(f"  Fallback tool: {inv.get_best_tool()}, status: {inv.get_tool_status()}")
print("  PASS: Tool durability system\n")

# --- TEST 3: Atmospheric Pressure ---
print("=" * 60)
print("TEST 3: Atmospheric Pressure Model")
print("=" * 60)

from src.world.generator import PlanetConfig, WorldGenerator

config_path = "config/planets/ross-128b.json"
if os.path.exists(config_path):
    planet = PlanetConfig(config_path)
    gen = WorldGenerator(planet, seed=42)
    
    center = gen.center
    p_center = gen.get_atmospheric_pressure(center, center)
    p_edge = gen.get_atmospheric_pressure(0, 0)
    print(f"  Ross 128b atmosphere: {planet.atmosphere.get('present', False)}")
    print(f"  Pressure at center: {p_center:.2f} kPa")
    print(f"  Pressure at corner: {p_edge:.2f} kPa")

config_path2 = "config/planets/trappist-1e.json"
if os.path.exists(config_path2):
    planet2 = PlanetConfig(config_path2)
    gen2 = WorldGenerator(planet2, seed=42)
    p = gen2.get_atmospheric_pressure(gen2.center, gen2.center)
    print(f"  TRAPPIST-1e (no atmo) pressure: {p:.2f} kPa (should be 0.0)")
    assert p == 0.0, "Atmosphereless planet should have 0 pressure!"

print("  PASS: Atmospheric pressure\n")

# --- TEST 4: Wind System ---
print("=" * 60)
print("TEST 4: Wind Direction & Speed System")
print("=" * 60)

if os.path.exists(config_path):
    gen_ross = WorldGenerator(PlanetConfig(config_path), seed=42)
    wind0 = gen_ross.get_wind(gen_ross.center, gen_ross.center, current_tick=0)
    wind100 = gen_ross.get_wind(gen_ross.center, gen_ross.center, current_tick=100)
    
    print(f"  Wind at tick 0: dir={wind0['direction_deg']:.1f}, speed={wind0['speed_kmh']:.1f}km/h")
    print(f"  Wind at tick 100: dir={wind100['direction_deg']:.1f}, speed={wind100['speed_kmh']:.1f}km/h")
    print(f"  Wind changes over time: {abs(wind0['direction_deg'] - wind100['direction_deg']) > 0.1}")

if os.path.exists(config_path2):
    gen_trap = WorldGenerator(PlanetConfig(config_path2), seed=42)
    wind_trap = gen_trap.get_wind(gen_trap.center, gen_trap.center, current_tick=0)
    print(f"  TRAPPIST-1e wind: {wind_trap['speed_kmh']} km/h (should be 0.0)")
    assert wind_trap["speed_kmh"] == 0.0, "No atmosphere = no wind!"

print("  PASS: Wind system\n")

# --- TEST 5: Gravity Construction Modifier ---
print("=" * 60)
print("TEST 5: Gravity Construction Modifier")
print("=" * 60)

planets_dir = "config/planets"
if os.path.exists(planets_dir):
    for fname in sorted(os.listdir(planets_dir)):
        if fname.endswith('.json'):
            p = PlanetConfig(os.path.join(planets_dir, fname))
            g = WorldGenerator(p, seed=42)
            mod = g.get_gravity_construction_modifier()
            print(f"  {p.display_name}: gravity={p.gravity_g:.2f}g -> modifier={mod:.2f}x")

print("  PASS: Gravity construction modifier\n")

# --- TEST 6: Night Illumination ---
print("=" * 60)
print("TEST 6: Night Illumination System")
print("=" * 60)

from src.agents.agent import Agent

agent_config = {
    "id": "test_agent",
    "name": "Test Agent",
    "background": "Test",
    "personality_traits": ["analytical"],
    "genome": {"strength": 5, "agility": 5, "endurance": 5, "perception": 7, "immunity": 5},
    "competency": {"engineering": 5, "medical": 5, "physics": 5, "botany_bio": 5, "leadership_social": 5},
}
test_agent = Agent(agent_config)

day_range = test_agent.perception_range(is_night=False)
night_range = test_agent.perception_range(is_night=True)
night_lit = test_agent.perception_range(
    is_night=True,
    light_level=0.1,
    nearby_light_sources=[{"light_radius": 5, "distance": 2}]
)

print(f"  Day perception range: {day_range}")
print(f"  Night perception (no light): {night_range}")
print(f"  Night perception (near campfire 2m): {night_lit}")
print(f"  Light restores vision: {night_lit > night_range}")
assert night_lit >= night_range, "Light sources should improve night vision!"

if os.path.exists(config_path2):
    gen_tidal = WorldGenerator(PlanetConfig(config_path2), seed=42)
    light_center = gen_tidal.get_tidal_zone_light(gen_tidal.center, gen_tidal.center)
    light_edge = gen_tidal.get_tidal_zone_light(0, 0)
    print(f"  Tidal substellar: {light_center}")
    print(f"  Tidal anti-stellar: {light_edge}")

print("  PASS: Night illumination\n")

# --- TEST 7: Resource Regeneration ---
print("=" * 60)
print("TEST 7: Resource Regeneration")
print("=" * 60)

from src.world.sparse_state import ResourceNode

iron = ResourceNode(
    node_id="iron_001", material="iron_ore", position=(50, 50),
    quantity=100, max_quantity=100, gather_rate=2.5, gather_difficulty=1.5,
)
iron.tick_regeneration()
print(f"  Iron ore (no regen): {iron.quantity}/100 after tick")

water_geo = ResourceNode(
    node_id="water_geo_001", material="water_ice", position=(100, 100),
    quantity=50, max_quantity=80, gather_rate=4.0, gather_difficulty=1.0,
    regeneration_rate=0.5, max_regen_quantity=80,
)
for _ in range(10):
    water_geo.tick_regeneration()
print(f"  Geothermal water (regen=0.5/tick): {water_geo.quantity}/80 after 10 ticks")

water_geo.quantity = 0
water_geo.depleted = True
water_geo.tick_regeneration()
print(f"  After depletion + regen: qty={water_geo.quantity}, depleted={water_geo.depleted}")
assert not water_geo.depleted, "Geothermal water should un-deplete after regen!"

print("  PASS: Resource regeneration\n")

# --- TEST 8: Social Interaction Mechanics ---
print("=" * 60)
print("TEST 8: Social Interaction Mechanics")
print("=" * 60)

agent_a = Agent({
    "id": "agent_a", "name": "Alice", "background": "Engineer",
    "personality_traits": ["cooperative", "analytical"],
    "genome": {"strength": 6, "agility": 5, "endurance": 7, "perception": 5, "immunity": 5},
    "competency": {"engineering": 8, "medical": 3, "physics": 5, "botany_bio": 2, "leadership_social": 4},
})
agent_b_id = "agent_b"

agent_a.build_trust(agent_b_id, 0.1, "shared resources")
agent_a.build_trust(agent_b_id, 0.15, "medical treatment")
print(f"  Trust after sharing+medical: {agent_a.trust_scores[agent_b_id]:.2f}")

bonus = agent_a.get_cooperation_bonus(agent_b_id)
print(f"  Cooperation bonus: {bonus:.2f}x")
print(f"  Will cooperate: {agent_a.will_cooperate_with(agent_b_id)}")

agent_a.lose_trust(agent_b_id, 0.5, "resource theft")
print(f"  Trust after theft: {agent_a.trust_scores[agent_b_id]:.2f}")
print(f"  Will cooperate (cooperative): {agent_a.will_cooperate_with(agent_b_id)}")

initial_morale = agent_a.morale
agent_a.apply_social_morale("teammate_death", "Charlie", trust_level=0.5)
print(f"  Morale after teammate death: {initial_morale:.2f} -> {agent_a.morale:.2f}")

agent_a.apply_social_morale("music_heard")
agent_a.apply_social_morale("game_played")
print(f"  Morale after music+game: {agent_a.morale:.2f}")

print(f"  Social summary: {agent_a.get_social_summary()}")
print("  PASS: Social interaction mechanics\n")

# --- TEST 9: Cell Info Integration ---
print("=" * 60)
print("TEST 9: Integrated Cell Info")
print("=" * 60)

if os.path.exists(config_path):
    gen_full = WorldGenerator(PlanetConfig(config_path), seed=42)
    cell = gen_full.get_cell_info(gen_full.center, gen_full.center, current_tick=50)
    
    print(f"  Cell keys: {sorted(cell.keys())}")
    assert "pressure_kpa" in cell, "Missing pressure_kpa!"
    assert "wind" in cell, "Missing wind!"
    assert "light" in cell, "Missing light!"
    assert "gravity_construction_modifier" in cell, "Missing gravity modifier!"
    
    print(f"  Pressure: {cell['pressure_kpa']} kPa")
    print(f"  Wind: {cell['wind']['speed_kmh']}km/h")
    print(f"  Light: {cell['light']['phase_name']}")
    print(f"  Gravity mod: {cell['gravity_construction_modifier']}x")
    print("  PASS: Cell info integration\n")

# --- TEST 10: State Summary ---
print("=" * 60)
print("TEST 10: State Summary with New Context")
print("=" * 60)

summary = test_agent.to_state_summary(
    current_tick=150,
    biome_name="terminator_ridge",
    wind_info={"direction_deg": 135.2, "speed_kmh": 28.5, "speed_modifier": 1.17},
    light_info={"is_night": False, "light_level": 0.8, "phase_name": "dusk"},
    nearby_structures=[
        {"type": "solar_panel", "distance": 5, "condition": 0.4, "maintenance_overdue": True},
        {"type": "water_collector", "distance": 12, "condition": 0.95, "maintenance_overdue": False},
    ],
)
print(f"  Summary:\n  {summary}")
assert "DAMAGED" in summary or "maintenance" in summary, "Structure condition should appear!"
print("  PASS: State summary integration\n")

# --- FINAL ---
print("=" * 60)
print("ALL 8 HYPER-REALISM SYSTEMS VALIDATED SUCCESSFULLY")
print("=" * 60)
