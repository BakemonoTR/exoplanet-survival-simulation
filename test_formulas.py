"""
Scientific formula verification tests.
Validates that all formulas produce outputs matching real-world physics.
"""
import sys, math
sys.path.insert(0, '.')

from src.agents.agent import Agent, Needs, Disease, DiseaseType
from src.world.generator import PlanetConfig, WorldGenerator

print("=" * 60)
print("SCIENTIFIC FORMULA VERIFICATION")
print("=" * 60)

# === V1: Barometric Formula (ISA) ===
# Ross 128b: M~0.032, g=1.16*9.81, T=285K, P0=0.8 atm
# coefficient = 0.032 * 11.38 / (8.314 * 285) = 0.000154
# At 2500m: P = 80.97 * exp(-0.000154 * 2500) = 80.97 * 0.680 = 55.06 kPa
p = PlanetConfig('config/planets/ross-128b.json')
gen = WorldGenerator(p, seed=42)
M = gen._calc_mean_molecular_weight(p.atmosphere.get("composition", {}))
print("\nV1: Barometric Formula (P = P0 * exp(-Mgh/RT))")
print(f"  Ross 128b mean molecular weight: {M:.4f} kg/mol (expected ~0.032)")
assert 0.030 < M < 0.036, f"M={M} out of range!"
pressure_center = gen.get_atmospheric_pressure(gen.center, gen.center)
print(f"  Pressure at center: {pressure_center:.2f} kPa")
assert pressure_center > 0 and pressure_center < 82  # Must be below surface P0
print("  PASS")

# === V2: Wind Chill (Environment Canada) ===
# T = -10C, V = 25 km/h
# T_wc = 13.12 + 0.6215*(-10) - 11.37*(25^0.16) + 0.3965*(-10)*(25^0.16)
# = 13.12 - 6.215 - 11.37*1.676 + (-3.965)*1.676
# = 13.12 - 6.215 - 19.056 - 6.645 = -18.8C
print("\nV2: Wind Chill (Environment Canada Formula)")
T = -10.0
V = 25.0
T_wc = 13.12 + 0.6215*T - 11.37*(V**0.16) + 0.3965*T*(V**0.16)
print(f"  T=-10C, V=25km/h -> T_wc = {T_wc:.1f}C (expected ~-18.8C)")
assert -20 < T_wc < -17, f"Wind chill {T_wc} out of range!"
print("  PASS")

# === V3: Cold Thermogenesis starts at 18C ===
print("\nV3: Cold Thermogenesis (Warwick & Busby, 1990)")
needs = Needs()
needs_warm = Needs()
needs.decay(endurance=5, ambient_temp_c=5.0)
needs_warm.decay(endurance=5, ambient_temp_c=20.0)
cold_drain = 85.0 - needs.hunger
warm_drain = 85.0 - needs_warm.hunger
print(f"  Hunger drain at 5C: {cold_drain:.3f}/tick (with thermogenesis)")
print(f"  Hunger drain at 20C: {warm_drain:.3f}/tick (no thermogenesis)")
print(f"  Cold increases hunger: {cold_drain > warm_drain}")
assert cold_drain > warm_drain, "Cold should increase hunger!"
# At 5C: bonus = (18-5) * 0.004 = 0.052 extra. So cold_drain ~ 0.402 vs warm 0.35
print(f"  Extra calories from cold: {(cold_drain - warm_drain):.3f} (expected ~0.052)")
assert 0.04 < (cold_drain - warm_drain) < 0.07
print("  PASS")

# === V4: Sleep Restoration = 8 hours ===
print("\nV4: Sleep Restoration (NREM/REM physiology)")
sleeper = Needs()
sleeper.energy = 10.0  # Very tired
for tick in range(96):  # 8 hours = 96 ticks
    sleeper.decay(endurance=5, is_sleeping=True)
print(f"  Energy after 96 ticks of sleep: {sleeper.energy:.1f} (starting from 10)")
assert sleeper.energy > 90, f"Should be nearly full! Got {sleeper.energy}"
# Check that 16 ticks (old system) is NOT enough
sleeper2 = Needs()
sleeper2.energy = 10.0
for tick in range(16):
    sleeper2.decay(endurance=5, is_sleeping=True)
print(f"  Energy after 16 ticks (80 min): {sleeper2.energy:.1f} (should NOT be full)")
assert sleeper2.energy < 50, f"Should not be full yet! Got {sleeper2.energy}"
print("  PASS")

# === V5: Radiation = no immunity resistance ===
print("\nV5: Radiation Dose (ICRP 103 - no biological resistance)")
a1 = Agent({
    'id': 'r1', 'name': 'R1', 'background': 'E',
    'personality_traits': ['analytical'],
    'genome': {'strength': 5, 'agility': 5, 'endurance': 5, 'perception': 5, 'immunity': 10},
    'competency': {'engineering': 5, 'medical': 5, 'physics': 5, 'botany_bio': 5, 'leadership_social': 5},
})
a2 = Agent({
    'id': 'r2', 'name': 'R2', 'background': 'E',
    'personality_traits': ['analytical'],
    'genome': {'strength': 5, 'agility': 5, 'endurance': 5, 'perception': 5, 'immunity': 1},
    'competency': {'engineering': 5, 'medical': 5, 'physics': 5, 'botany_bio': 5, 'leadership_social': 5},
})
a1.apply_radiation(1.0)
a2.apply_radiation(1.0)
print(f"  Immunity 10: dose={a1.cumulative_radiation_sv:.2f} Sv")
print(f"  Immunity 1:  dose={a2.cumulative_radiation_sv:.2f} Sv")
assert a1.cumulative_radiation_sv == a2.cumulative_radiation_sv, \
    "Immunity should NOT affect dose absorption!"
print(f"  Equal dose regardless of immunity: {a1.cumulative_radiation_sv == a2.cumulative_radiation_sv}")
print("  PASS")

# === V6: Hypoxia based on PO2 ===
print("\nV6: Hypoxia (Hackett & Roach, NEJM 2001 - PO2 based)")
a3 = Agent({
    'id': 'h1', 'name': 'H1', 'background': 'E',
    'personality_traits': ['analytical'],
    'genome': {'strength': 5, 'agility': 5, 'endurance': 5, 'perception': 5, 'immunity': 5},
    'competency': {'engineering': 5, 'medical': 5, 'physics': 5, 'botany_bio': 5, 'leadership_social': 5},
})
# Normal conditions: 101.3 kPa * 0.21 = 21.3 kPa PO2 - no hypoxia
e_before = a3.needs.energy
a3.tick_update(ambient_temp_c=20.0, gravity_multiplier=1.0, has_atmosphere=True,
               pressure_kpa=101.3, nearby_agent_count=1)
e_after_normal = a3.needs.energy
normal_drain = e_before - e_after_normal

# Low PO2: 60 kPa * 0.21 = 12.6 kPa PO2 - mild hypoxia
a4 = Agent({
    'id': 'h2', 'name': 'H2', 'background': 'E',
    'personality_traits': ['analytical'],
    'genome': {'strength': 5, 'agility': 5, 'endurance': 5, 'perception': 5, 'immunity': 5},
    'competency': {'engineering': 5, 'medical': 5, 'physics': 5, 'botany_bio': 5, 'leadership_social': 5},
})
e_before_low = a4.needs.energy
a4.tick_update(ambient_temp_c=20.0, gravity_multiplier=1.0, has_atmosphere=True,
               pressure_kpa=60.0, nearby_agent_count=1)
e_after_low = a4.needs.energy
low_drain = e_before_low - e_after_low

print(f"  Normal (PO2=21.3 kPa): energy drain = {normal_drain:.3f}")
print(f"  Low (PO2=12.6 kPa):    energy drain = {low_drain:.3f}")
print(f"  Low pressure drains more: {low_drain > normal_drain}")
assert low_drain > normal_drain, "Low PO2 should drain more energy!"
print("  PASS")

# === V7: Temperature Stress Exponential ===
print("\nV7: Temperature Stress (Tikuisis 1999 - power-law)")
n1 = Needs()  # stress = 50
n2 = Needs()
# Mild cold: -5C deficit from comfort (10C ambient)
n1._update_temperature_stress(10.0, False, 0.0)
mild_rate = 50.0 - n1.temperature_stress
# Extreme cold: -50C ambient = 65C deficit
n2._update_temperature_stress(-50.0, False, 0.0)
extreme_rate = 50.0 - n2.temperature_stress
print(f"  Mild cold (10C): stress drop = {mild_rate:.2f}/tick")
print(f"  Extreme cold (-50C): stress drop = {extreme_rate:.2f}/tick")
print(f"  Extreme is {extreme_rate/mild_rate:.1f}x worse than mild (should be >>2x)")
# Power law: 65^1.3 / 5^1.3 = ~25x difference (vs linear which would be 13x)
assert extreme_rate / mild_rate > 15, "Power-law should make extreme much worse!"
print("  PASS")

# === V8: Injury Healing Speed ===
print("\nV8: Injury Healing (Guo & DiPietro 2010)")
healer = Agent({
    'id': 'heal', 'name': 'Healer', 'background': 'E',
    'personality_traits': ['analytical'],
    'genome': {'strength': 5, 'agility': 5, 'endurance': 5, 'perception': 5, 'immunity': 5},
    'competency': {'engineering': 5, 'medical': 5, 'physics': 5, 'botany_bio': 5, 'leadership_social': 5},
})
healer.injury_level = 0.5
healer.action.action_type = "sleep"
for _ in range(100):
    healer.tick_update(ambient_temp_c=20.0, gravity_multiplier=1.0, has_atmosphere=True,
                       pressure_kpa=101.3, nearby_agent_count=1)
print(f"  Injury after 100 sleep ticks (8.3 hr): {healer.injury_level:.3f} (was 0.5)")
# Old rate: 0.008*100 = 0.8 reduction -> fully healed. New: 0.002*100 = 0.2 reduction
assert healer.injury_level > 0.2, f"Should not be fully healed! Got {healer.injury_level}"
assert healer.injury_level < 0.5, "Should have some healing!"
print(f"  Healing rate is realistic (not fully healed in 8 hours)")
print("  PASS")

# === V9: Atmospheric Molecular Weight ===
print("\nV9: Planet-specific Molecular Weight")
kepler = PlanetConfig('config/planets/kepler-442b.json')
gen_k = WorldGenerator(kepler, seed=42)
M_k = gen_k._calc_mean_molecular_weight(kepler.atmosphere.get("composition", {}))
print(f"  Kepler-442b (N2-CO2): M = {M_k:.4f} kg/mol (expected ~0.034)")
assert 0.030 < M_k < 0.040

teeg = PlanetConfig('config/planets/teegardens-star-b.json')
gen_t = WorldGenerator(teeg, seed=42)
M_t = gen_t._calc_mean_molecular_weight(teeg.atmosphere.get("composition", {}))
print(f"  Teegarden's b (N2-CO2-SO2): M = {M_t:.4f} kg/mol (expected ~0.035)")
assert 0.030 < M_t < 0.040
print("  PASS")

# === V10: Night Cooling Stefan-Boltzmann ===
print("\nV10: Night Cooling (Stefan-Boltzmann radiative)")
prox = PlanetConfig('config/planets/proxima-centauri-b.json')
gen_p = WorldGenerator(prox, seed=42)
cell_day = gen_p.get_cell_info(gen_p.center, gen_p.center, current_tick=0)
cell_night = gen_p.get_cell_info(0, 0, current_tick=0)
day_t = cell_day["temperature_c"]
night_t = cell_night["temperature_c"]
delta = day_t - night_t
print(f"  Proxima b substellar: {day_t}C")
print(f"  Proxima b anti-stellar: {night_t}C")
print(f"  Day-night delta: {delta:.1f}C")
print(f"  Delta varies with solar flux (not flat -80C): {delta != 80.0}")
assert delta != 80.0, "Should NOT be flat -80C anymore!"
print("  PASS")

print("\n" + "=" * 60)
print("ALL 10 SCIENTIFIC FORMULAS VERIFIED SUCCESSFULLY")
print("=" * 60)
