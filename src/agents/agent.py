"""
Agent internal state model.

Each agent is a fully stateful entity with:
- Genome (immutable physical attributes)
- Competency vector (immutable skill profile)
- Needs (decaying physiological requirements)
- Inventory (carried items and materials)
- Position and movement state
- Current task/plan state
- Radiation exposure (cumulative, planet-specific)
- Social relationships (trust scores with other agents)

This is NOT a game character. It is a scientifically modeled autonomous
entity with realistic physiological constraints based on NASA human
spaceflight standards (NASA-STD-3001).
"""

import json
import math
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import numpy as np


TACTICAL_POLICY_META_KEY = "__tactical_meta__"
TACTICAL_STATE_SCHEMA_VERSION = 4


class AgentStatus(str, Enum):
    ALIVE = "alive"
    CRITICAL = "critical"  # One or more needs in danger zone
    INCAPACITATED = "incapacitated"  # Cannot perform most actions
    DEAD = "dead"


class DeathCause(str, Enum):
    STARVATION = "starvation"
    DEHYDRATION = "dehydration"
    EXHAUSTION = "exhaustion"
    HYPOTHERMIA = "hypothermia"
    HYPERTHERMIA = "hyperthermia"
    RADIATION_POISONING = "radiation_poisoning"
    DISEASE = "disease"
    INJURY = "injury"
    SUFFOCATION = "suffocation"  # O₂ depletion on atmosphereless planets
    VACUUM_EXPOSURE = "vacuum_exposure"  # Ebullism + anoxia in vacuum without suit
    HYPERCAPNIA = "hypercapnia"  # CO2 scrubber exhaustion / poisoning in suit
    SUIT_DEPRESSURIZATION = "suit_depressurization"  # Unsealed suit puncture in vacuum


@dataclass
class Genome:
    """
    Immutable physical attributes assigned at agent creation.

    Each attribute is on a 1-10 scale (3-9 typical range).
    These affect action efficiency, not decision-making.

    Scientific basis: Individual physiological variation in
    strength, cardiovascular fitness, sensory acuity, and
    immune response documented in spaceflight medicine literature.
    """
    strength: int       # Carry capacity, construction speed, melee
    agility: int        # Movement speed, evasion, dexterity
    endurance: int      # Slows need decay rates, stamina
    perception: int     # Vision range, threat detection distance
    immunity: int       # Disease resistance, radiation tolerance

    @property
    def total(self) -> int:
        return self.strength + self.agility + self.endurance + self.perception + self.immunity

    def to_dict(self) -> dict:
        return {
            "strength": self.strength,
            "agility": self.agility,
            "endurance": self.endurance,
            "perception": self.perception,
            "immunity": self.immunity,
        }


@dataclass
class Competency:
    """
    Multidisciplinary skill profile. NOT an RPG XP system.

    Each domain is 0-10. This is a fixed CV-like profile that determines:
    - LLM system prompt (shapes reasoning style)
    - RAG retrieval access depth
    - Action quality modifiers (engineering → build quality)

    Does NOT change during simulation. "Learning" comes from
    memory/reflection, not stat increases.
    """
    engineering: int        # Construction, ISRU, materials science
    medical: int            # Treatment, diagnosis, hygiene management
    physics: int            # Hazard prediction, energy systems, planetary science
    botany_bio: int         # Food production, biological systems, ecology
    leadership_social: int  # Negotiation, coordination, conflict resolution

    def get_primary_domain(self) -> str:
        """Return the highest-scoring domain (for prompt generation)."""
        scores = self.to_dict()
        return max(scores, key=scores.get)

    def get_domains_above(self, threshold: int) -> list[str]:
        """Return all domains scoring at or above threshold."""
        return [k for k, v in self.to_dict().items() if v >= threshold]

    def to_dict(self) -> dict:
        return {
            "engineering": self.engineering,
            "medical": self.medical,
            "physics": self.physics,
            "botany_bio": self.botany_bio,
            "leadership_social": self.leadership_social,
        }


@dataclass(frozen=True)
class Anthropometrics:
    """Immutable crew body measurements used by life-support calculations."""

    height_cm: float
    mass_kg: float
    foot_length_cm: float
    shoe_size_eu: int

    @property
    def body_surface_area_m2(self) -> float:
        # Mosteller, NEJM 1987: BSA = sqrt(height_cm * mass_kg / 3600)
        return math.sqrt(self.height_cm * self.mass_kg / 3600.0)

    @property
    def bmi(self) -> float:
        height_m = self.height_cm / 100.0
        return self.mass_kg / (height_m * height_m)

    def to_dict(self) -> dict:
        return {
            "height_cm": round(self.height_cm, 1),
            "mass_kg": round(self.mass_kg, 1),
            "foot_length_cm": round(self.foot_length_cm, 1),
            "shoe_size_eu": self.shoe_size_eu,
            "body_surface_area_m2": round(self.body_surface_area_m2, 3),
            "bmi": round(self.bmi, 1),
        }


# Material density table (kg per unit) — based on real material densities
# Used for carry weight calculations
MATERIAL_DENSITY_KG = {
    # Raw Geological Minerals (kg per unit)
    "regolith": 1.5,         # Loose lunar/martian soil, ~1.5 g/cm³
    "iron_ore": 4.5,         # Hematite/magnetite ore chunk, ~4.5 kg/unit
    "silica_sand": 2.6,      # Vitrifiable SiO2-rich sand, ~2.6 kg/unit
    "water_ice": 0.9,        # Water ice core, 0.917 kg/L
    "sulfur": 2.0,           # Elemental sulfur crystals, 2.07 kg/unit
    "calcite": 2.7,          # CaCO₃ flux, 2.7 kg/unit
    "olivine": 3.3,          # (Mg,Fe)₂SiO₄ mineral, 3.3 kg/unit
    "graphite": 1.8,         # Carbon reductant concentrate, 1.8 kg/unit
    "basalt": 2.9,           # Dense volcanic anchor rock, 2.9 kg/unit
    "chalcopyrite_ore": 4.2, # CuFeS2-bearing ore lot; not usable as wire
    
    # Precision Refined Aerospace Components (Physical mass in kg)
    "reduced_iron_ingot": 5.0,         # Reduced/cast structural iron bar (5.0 kg)
    "metal_pipe": 3.5,                 # High-pressure hydraulic/pneumatic pipe conduit (3.5 kg)
    "machined_bolts_fasteners": 0.2,   # Set of M12 aerospace high-tensile fasteners (0.2 kg)
    "vacuum_gasket_seal": 0.25,        # Fluoroelastomer pressure O-ring seal (0.25 kg)
    "finned_heat_sink": 8.0,           # Folded radiative cooling matrix (8.0 kg)
    "composite_panel": 12.0,           # Carbon fiber reinforced polymer hull panel (12.0 kg)
    "glass_pane": 4.0,                 # Vitrified quartz vacuum window (4.0 kg)
    "electronic_component": 0.8,       # Avionics relay board & microchip housing (0.8 kg)
    "electrical_copper_stock": 7.5,    # Lot-tracked conductor/braid stock
    "sulfur_regolith_paver": 57.0,     # One inspected sulfur-bound aggregate batch
    "insulated_fabric": 1.5,           # Multi-layer basalt thermal insulation sheet (1.5 kg)
    "rope_cordage": 3.0,                # 50 m high-strength rigging coil
    "electronics_salvage": 1.0,         # Recoverable flight-electronics lot
    "structural_truss_section": 41.0,   # Anchored, cross-braced structural bay
    "power_cable_harness": 18.0,        # Power/data conductors, connectors and shielding
    "photovoltaic_laminate": 8.0,       # Qualified lightweight PV blanket segment
    "pump_compressor_module": 36.0,     # Motor, pump, valves, manifold and controls
    "thermal_control_loop": 55.0,       # Pumped loop, exchanger and radiator hardware
    "pressure_vessel_section": 64.0,    # Proof-tested pressure hull bay
    "hydroponic_rack": 48.0,            # Irrigation, tray, sensor and frame assembly
    "electrolysis_stack": 52.0,         # SOEC stack, manifold, insulation and controls
    # Serialized flight-qualified cores. These are inventory units, not
    # geological resources; their masses are mirrored by mission_profile.json.
    "qualified_pv_blanket_segment": 8.0,
    "fluoroelastomer_seal_stock": 1.0,
    "qualified_control_board_core": 0.8,
    "eva_suit_life_support_core": 8.5,
    "laboratory_instrument_core": 30.0,
    "pharmaceutical_reagent_pack": 0.8,
    "pump_motor_bearing_core": 14.0,
    "pem_membrane_electrode_core": 30.0,
    "crop_seed_nutrient_pack": 20.0,
    "solar_tracking_drive_core": 12.0,
    "power_conditioning_unit_core": 12.0,
    "emergency_beacon_parts": 5.0,
    "preintegrated_cea_module_segment": 31472.05,
    "preintegrated_habitat_core": 32000.0,
    "integrated_habitat_storm_vault_liner": 5000.0,
    "qualified_grid_storage_module": 30000.0,
    "landing_navigation_core": 500.0,
    "qualified_potable_water_tank_liner": 350.0,
    "qualified_lox_tank_core": 1800.0,
    "cargo_transporter_service_module": 25.0,
}


class DiseaseType(Enum):
    """Typed disease system — different causes produce different symptoms."""
    INFECTION = "infection"            # From low hygiene. Fever, fatigue.
    HYPOTHERMIC_SHOCK = "hypothermic_shock"  # From prolonged cold. Shivering, confusion.
    HEAT_STROKE = "heat_stroke"        # From prolonged heat. Delirium, organ stress.
    RADIATION_ARS = "radiation_ars"    # Acute Radiation Syndrome. Nausea, immunodeficiency.
    TOXIC_EXPOSURE = "toxic_exposure"  # From volcanic gases. Respiratory damage.


@dataclass
class Disease:
    """Active disease with type-specific effects."""
    disease_type: DiseaseType
    severity: float = 1.0          # 1.0=standard, >1.0=severe
    ticks_remaining: int = 40
    ticks_total: int = 40

    # Per-type symptom effects
    EFFECTS = {
        DiseaseType.INFECTION: {
            "hunger_drain_bonus": 0.3,  # Fever burns calories
            "energy_drain_bonus": 0.25,
            "action_speed_mod": 0.7,
            "duration_ticks": 40,  # ~3.3 sim hours
        },
        DiseaseType.HYPOTHERMIC_SHOCK: {
            "energy_drain_bonus": 0.5,  # Shivering exhausts
            "action_speed_mod": 0.5,
            "movement_speed_mod": 0.4,
            "perception_penalty": -3,  # Confusion
            "duration_ticks": 24,
        },
        DiseaseType.HEAT_STROKE: {
            "thirst_drain_bonus": 0.6,  # Severe dehydration
            "energy_drain_bonus": 0.4,
            "action_speed_mod": 0.3,
            "duration_ticks": 20,
        },
        DiseaseType.RADIATION_ARS: {
            "hunger_drain_bonus": 0.4,  # Nausea/vomiting
            "energy_drain_bonus": 0.3,
            "immunity_penalty": -3,  # Immunocompromised
            "action_speed_mod": 0.6,
            "duration_ticks": 60,  # Longer recovery
        },
        DiseaseType.TOXIC_EXPOSURE: {
            "energy_drain_bonus": 0.35,
            "action_speed_mod": 0.65,
            "duration_ticks": 30,
        },
    }

    def get_effects(self) -> dict:
        return self.EFFECTS.get(self.disease_type, {})


@dataclass
class Needs:
    """
    Physiological needs model grounded in real human physiology.

    All values are 0-100 (percentage of maximum capacity).
    At 0, the agent enters a death countdown timer.

    Metabolic basis (NASA-STD-3001, ISS crew health data):
    - hunger: Maps to ~2200 kcal/day at rest, ~3000 kcal/day heavy labor.
      100 = 2200 kcal stored energy. Decay rate ≈ 0.35/tick at rest
      → 100→0 in ~286 ticks (23.8 sim hours) matching ~24h without food
      before cognitive impairment onset.
    - thirst: NASA minimum 2.5 L/day, personalized by body surface area,
      with an additional 0.24 L/hour during EVA.
      Dehydration effects begin at 2% body mass water loss (thirst < 60).
    - energy: Maps to ~8h sleep per 24h cycle.
      Decay ≈ 0.30/tick → 100→0 in ~333 ticks (27.8 hours).
      Cognitive impairment at <30 (equivalent to >18h awake).
    - hygiene: Infection risk curve. No direct physiological mapping
      but tracks time since last wash/decontamination event.
    - temperature_stress: Core body temperature proxy.
      50 = 37°C (normal). 0 = hypothermia (<35°C). 100 = hyperthermia (>40°C).
    - o2_supply: Oxygen reserves (only relevant on atmosphereless planets).
      100 = full canister. Decay ≈ 0.58/tick → 100→0 in ~172 ticks (14.3 hours)
      matching ~840g O₂/day consumption at moderate activity.
    """
    hunger: float = 85.0
    thirst: float = 85.0
    energy: float = 70.0
    hygiene: float = 60.0
    temperature_stress: float = 50.0
    o2_supply: float = 100.0  # Only consumed on atmosphereless planets

    # Death countdown timers (consecutive ticks at need=0 before death)
    _hunger_death_timer: int = 0
    _thirst_death_timer: int = 0
    _energy_death_timer: int = 0
    _temp_death_timer: int = 0
    _o2_death_timer: int = 0

    # Sleep tracking (for continuous sleep requirement)
    _consecutive_sleep_ticks: int = 0
    _ticks_since_last_full_sleep: int = 0
    _sleep_debt: float = 0.0  # Accumulated sleep debt

    def __init__(self, hunger: float = 85.0, thirst: float = 85.0, energy: float = 70.0,
                 hygiene: float = 60.0, temperature_stress: float = 50.0, o2_supply: float = 100.0):
        self.hunger = hunger
        self.thirst = thirst
        self.energy = energy
        self.hygiene = hygiene
        self.temperature_stress = temperature_stress
        self.o2_supply = o2_supply
        self._hunger_death_timer = 0
        self._thirst_death_timer = 0
        self._energy_death_timer = 0
        self._temp_death_timer = 0
        self._o2_death_timer = 0
        self._consecutive_sleep_ticks = 0
        self._ticks_since_last_full_sleep = 0
        self._sleep_debt = 0.0
        self.configure_timebase(5.0)

    # Death thresholds (ticks at need=0 before death)
    # Based on real survival medicine:
    STARVATION_TICKS: int = 864    # 3 full days (72 hours) without food
    DEHYDRATION_TICKS: int = 288   # 24 hours (1 full planetary day) without water
    EXHAUSTION_TICKS: int = 144    # 12 hours of total physical collapse (0 energy)
    TEMP_DEATH_TICKS: int = 36     # 3 hours (36 ticks) of sustained severe hypothermia (<28°C core)
    SUFFOCATION_TICKS: int = 6     # 30 sim minutes without O₂ (anoxia/hypoxia)

    # Base decay rates per tick (5 sim minutes)
    BASE_HUNGER_DECAY: float = 0.35    # 100→0 in ~286 ticks (23.8h)
    # One 1 L drink normally restores the 35-point proactive hydration band
    # (65→100). 2.5 L/day therefore corresponds to 0.304 points per tick.
    BASE_THIRST_DECAY: float = (2.5 / 288.0) * 35.0
    BASE_ENERGY_DECAY: float = 0.30    # 100→0 in ~333 ticks (27.8h)
    BASE_HYGIENE_DECAY: float = 0.15   # 100→0 in ~667 ticks (55.6h)
    BASE_O2_DECAY: float = 0.58        # 100→0 in ~172 ticks (14.3h) = ~840g O₂/day

    # Continuous sleep requirement
    FULL_SLEEP_TICKS: int = 16         # 80 sim minutes for full rest cycle
    SLEEP_DEBT_PENALTY: float = 0.05   # Per tick of accumulated sleep debt → extra energy drain

    def configure_timebase(self, tick_minutes: float) -> None:
        """Preserve physiological durations when mission tick size changes."""
        tick_minutes = max(0.1, float(tick_minutes))
        scale = tick_minutes / 5.0
        self.tick_minutes = tick_minutes
        self.ticks_per_hour = 60.0 / tick_minutes
        self.STARVATION_TICKS = max(1, math.ceil(72 * self.ticks_per_hour))
        self.DEHYDRATION_TICKS = max(1, math.ceil(24 * self.ticks_per_hour))
        self.EXHAUSTION_TICKS = max(1, math.ceil(12 * self.ticks_per_hour))
        self.TEMP_DEATH_TICKS = max(1, math.ceil(3 * self.ticks_per_hour))
        self.SUFFOCATION_TICKS = max(1, math.ceil(30.0 / tick_minutes))
        self.BASE_HUNGER_DECAY = 0.35 * scale
        self.BASE_THIRST_DECAY = (2.5 / (1440.0 / tick_minutes)) * 35.0
        self.BASE_ENERGY_DECAY = 0.30 * scale
        self.BASE_HYGIENE_DECAY = 0.15 * scale
        self.BASE_O2_DECAY = 0.58 * scale
        self.FULL_SLEEP_TICKS = max(1, math.ceil(80.0 / tick_minutes))
        self.SLEEP_DEBT_PENALTY = 0.05 * scale

    def decay(self, endurance: int, gravity_multiplier: float = 1.0,
              activity_multiplier: float = 1.0,
              ambient_temp_c: float = 20.0,
              wind_speed_modifier: float = 1.0,
              has_shelter: bool = False,
              shelter_temp_reduction: float = 0.0,
              has_atmosphere: bool = True,
              is_sleeping: bool = False,
              sleep_quality: float = 1.0,
              morale: float = 1.0,
              active_diseases: list = None,
              is_canister_active: bool = False,
              water_requirement_l_per_tick: float | None = None) -> list[str]:
        """
        Apply per-tick physiological decay. Returns list of critical warnings.

        This models real metabolic processes:
        - Basal Metabolic Rate (BMR) drives hunger/thirst at rest
        - Activity multiplier models Metabolic Equivalent of Task (MET):
          1.0=rest/idle (1 MET), 1.5=walking (3.5 MET), 2.0=heavy labor (6 MET)
        - Gravity compounds physical exertion during active labor
        - Wind chill affects effective temperature (empirical formula)
        - Low morale accelerates all decay (stress response → elevated cortisol)
        """
        warnings = []
        if active_diseases is None:
            active_diseases = []
        tick_scale = getattr(self, "tick_minutes", 5.0) / 5.0

        # === MODIFIERS ===

        # Endurance modifier (genome): each point above 5 reduces metabolic
        # waste by 5% (higher cardiovascular efficiency)
        endurance_mod = 1.0 - 0.05 * (endurance - 5)
        endurance_mod = max(0.5, min(1.5, endurance_mod))

        # Morale modifier: demoralized agents have elevated stress hormones
        # → 10-30% faster need decay (cortisol-driven catabolism)
        morale_mod = 1.0 + (1.0 - morale) * 0.3

        # Disease modifier: active diseases accelerate specific needs
        disease_hunger_bonus = sum(
            d.get_effects().get("hunger_drain_bonus", 0) for d in active_diseases
        ) * tick_scale
        disease_thirst_bonus = sum(
            d.get_effects().get("thirst_drain_bonus", 0) for d in active_diseases
        ) * tick_scale
        disease_energy_bonus = sum(
            d.get_effects().get("energy_drain_bonus", 0) for d in active_diseases
        ) * tick_scale

        # === HUNGER (caloric expenditure) ===
        # Base: 0.35/tick ~ 2200 kcal/day at rest (Harris-Benedict BMR)
        # Heavy labor: 0.35 x 2.0 = 0.70/tick ~ 4400 kcal/day (realistic for EVA work)
        # Gravity affects physical work energy: W = F*d, F = m*g
        gravity_hunger_mod = 1.0 + (gravity_multiplier - 1.0) * 0.5 * max(0, activity_multiplier - 1.0)
        hunger_rate = (self.BASE_HUNGER_DECAY * endurance_mod * activity_multiplier
                       * gravity_hunger_mod * morale_mod + disease_hunger_bonus)
        # Cold thermogenesis: shivering begins at ~18C (clothed thermoneutral)
        # ~2.5% BMR increase per C below thermoneutral (Warwick & Busby, 1990)
        if ambient_temp_c < 18:
            cold_hunger_bonus = (
                (18 - ambient_temp_c) * 0.004 * tick_scale
            )
            hunger_rate += cold_hunger_bonus
        self.hunger = max(0.0, self.hunger - hunger_rate)

        # === THIRST (fluid loss) ===
        # Sweat rate increases with heat and exertion
        # Sawka et al., ACSM Position Stand (2007):
        # Sweat rate = f(temperature, activity). Key interaction:
        # Rest+30C: 0.3 L/h, Heavy+30C: 1.0 L/h, Heavy+40C: 2.0+ L/h
        heat_thirst_mod = 1.0
        if ambient_temp_c > 25:
            # Heat x activity interaction: sweating scales with both
            heat_thirst_mod += (ambient_temp_c - 25) * 0.02 * activity_multiplier
        elif ambient_temp_c < -10:
            # Cold air is dry — respiratory water loss (insensible loss)
            heat_thirst_mod += (-10 - ambient_temp_c) * 0.005
        if water_requirement_l_per_tick is None:
            thirst_rate = (
                self.BASE_THIRST_DECAY * endurance_mod * activity_multiplier
                * heat_thirst_mod * morale_mod + disease_thirst_bonus
            )
        else:
            # Physical demand is expressed in liters; 1 L replenishes the
            # normal 35-point proactive drinking band (65→100).
            thirst_rate = (
                max(0.0, water_requirement_l_per_tick) * 35.0
                * endurance_mod * heat_thirst_mod * morale_mod
                + disease_thirst_bonus
            )
        self.thirst = max(0.0, self.thirst - thirst_rate)

        # === ENERGY (fatigue accumulation) ===
        # Gravity increases work during physical activity, not during sleep/rest
        circadian_mod = 1.15 if not has_shelter else 1.0
        # Cap sleep debt to max 8.0 so it never creates infinite exhaustion death spirals
        self._sleep_debt = min(8.0, max(0.0, self._sleep_debt))
        sleep_debt_drain = self._sleep_debt * self.SLEEP_DEBT_PENALTY
        if is_sleeping:
            # === POLYSOMNOGRAPHIC SLEEP ARCHITECTURE (NASA SP-368 / Rechtschaffen & Kales) ===
            # Sleep progresses through progressive physiological stages:
            # Sleep staging is based on elapsed simulated minutes, independent
            # of wall-clock playback speed.
            self._consecutive_sleep_ticks += 1
            t = self._consecutive_sleep_ticks
            sleep_minutes = t * getattr(self, "tick_minutes", 5.0)

            if sleep_minutes <= 20:
                # Stage N1: Sleep Latency / Alpha-to-Theta transition (0-20 min)
                rest_rate = 0.60 * sleep_quality * tick_scale
            elif sleep_minutes <= 50:
                # Stage N2: Light-to-Moderate Stable Sleep (20-50 min)
                rest_rate = 1.40 * sleep_quality * tick_scale
            elif sleep_minutes <= 160:
                # Stage N3: Slow-Wave Deep Sleep / SWS (50-160 min)
                rest_rate = 2.40 * sleep_quality * tick_scale
            else:
                # Stage REM & Subsequent ~90-minute ultradian cycles (160+ min)
                rest_rate = 1.60 * sleep_quality * tick_scale

            self.energy = min(100.0, self.energy + rest_rate)
            # Reduce sleep debt proportionally to sleep depth
            self._sleep_debt = max(
                0.0,
                self._sleep_debt
                - (0.5 if sleep_minutes <= 50 else 1.2) * tick_scale,
            )
            if self.energy >= 90.0 or sleep_minutes >= 160:
                self._ticks_since_last_full_sleep = 0
                self._sleep_debt = 0.0
        else:
            self._consecutive_sleep_ticks = 0
            self._ticks_since_last_full_sleep += 1
            # Accumulate sleep debt after ~16 waking hours.
            if (
                self._ticks_since_last_full_sleep
                * getattr(self, "tick_minutes", 5.0) > 16 * 60
            ):
                self._sleep_debt = min(
                    8.0, self._sleep_debt + 0.05 * tick_scale
                )
            # Physical exertion under high gravity
            gravity_energy_mod = 1.0 + (gravity_multiplier - 1.0) * max(0.0, activity_multiplier - 1.0) * 0.5
            energy_rate = (self.BASE_ENERGY_DECAY * endurance_mod * gravity_energy_mod
                           * activity_multiplier * morale_mod * circadian_mod
                           + sleep_debt_drain + disease_energy_bonus)
            self.energy = max(0.0, self.energy - energy_rate)

        # === HYGIENE ===
        # Constant decay, slightly faster in dusty/dirty environments
        self.hygiene = max(0.0, self.hygiene - self.BASE_HYGIENE_DECAY * morale_mod)

        # === O₂ SUPPLY (atmosphereless planets only when not managed by suit canister) ===
        if not has_atmosphere and not is_canister_active:
            o2_rate = self.BASE_O2_DECAY * activity_multiplier
            self.o2_supply = max(0.0, self.o2_supply - o2_rate)
            if self.o2_supply <= 10:
                warnings.append("o2_critical")

        # === TEMPERATURE STRESS ===
        effective_wind = wind_speed_modifier if not has_shelter else 1.0
        self._update_temperature_stress(
            ambient_temp_c, has_shelter, shelter_temp_reduction,
            effective_wind, activity_multiplier
        )

        # === CRITICAL THRESHOLD WARNINGS ===
        if self.hunger <= 15:
            warnings.append("hunger_critical")
        if self.thirst <= 15:
            warnings.append("thirst_critical")
        if self.energy <= 15:
            warnings.append("energy_critical")
        if self.hygiene <= 10:
            warnings.append("hygiene_critical")
        if self.temperature_stress <= 10 or self.temperature_stress >= 90:
            warnings.append("temperature_critical")

        # === DEATH TIMER UPDATES ===
        if self.hunger <= 0:
            self._hunger_death_timer += 1
        else:
            self._hunger_death_timer = 0

        if self.thirst <= 0:
            self._thirst_death_timer += 1
        else:
            self._thirst_death_timer = 0

        if self.energy <= 0:
            self._energy_death_timer += 1
        else:
            self._energy_death_timer = 0

        if self.temperature_stress <= 0 or self.temperature_stress >= 100:
            self._temp_death_timer += 1
        else:
            self._temp_death_timer = 0

        if not has_atmosphere and self.o2_supply <= 0:
            self._o2_death_timer += 1
        else:
            self._o2_death_timer = 0

        return warnings

    def _update_temperature_stress(self, ambient_temp_c: float,
                                    has_shelter: bool,
                                    shelter_temp_reduction: float,
                                    wind_speed_modifier: float = 1.0,
                                    activity_multiplier: float = 1.0):
        """
        Thermoregulation model.

        Human core temperature: 37°C (98.6°F).
        Comfort zone: 15-25°C (with light clothing).
        Hypothermia onset: core temp <35°C (ambient sustained <5°C).
        Hyperthermia onset: core temp >40°C (ambient sustained >45°C).

        Modifiers:
        - Wind chill: empirical formula (Environment Canada model)
          T_wc = 13.12 + 0.6215*T - 11.37*V^0.16 + 0.3965*T*V^0.16
          where V = wind speed in km/h, T = ambient temp in °C
        - Activity heat: physical exertion generates body heat
          (each MET above 1.0 adds ~1.5°C effective warming)
        - Shelter: reduces wind and moderates temperature
        - Emergency blanket: +25% cold resistance (handled externally)
        """
        comfort_low = 15.0
        comfort_high = 25.0

        # === Wind chill calculation ===
        effective_temp = ambient_temp_c
        if ambient_temp_c < 10 and wind_speed_modifier > 1.0:
            # Simplified wind chill (Environment Canada formula adapted)
            # Higher wind_speed_modifier = stronger wind
            wind_kmh = (wind_speed_modifier - 1.0) * 20 + 5  # Convert modifier to ~km/h
            if wind_kmh > 4.8:  # Formula valid for wind > 4.8 km/h
                wind_chill = (13.12 + 0.6215 * ambient_temp_c
                              - 11.37 * (wind_kmh ** 0.16)
                              + 0.3965 * ambient_temp_c * (wind_kmh ** 0.16))
                effective_temp = min(effective_temp, wind_chill)

        # === Activity heat generation ===
        # Gagge & Gonzalez (1996), Environmental Ergonomics:
        # Metabolic heat = (MET-1) * 1.2 W/kg * 70 kg * 0.75 efficiency
        # Each MET above rest raises effective body temp by ~1.0-1.5C (90 min)
        # For per-tick: ~5C warming per MET accounts for heat accumulation
        if activity_multiplier > 1.0:
            body_heat = (activity_multiplier - 1.0) * 5.0  # ~5C per MET above rest
            effective_temp += body_heat

        # === EVA suit thermal protection ===
        # NASA Artemis xEMU / Apollo A7L Multi-Layer Insulation (MLI) & LCVG active heating
        # Provides thermal stress attenuation against extreme cold/heat when suit is equipped
        suit_temp_mod = getattr(self, '_suit_temp_reduction_active', 0.0)
        if suit_temp_mod > 0:
            if effective_temp < comfort_low:
                effective_temp = effective_temp + (comfort_low - effective_temp) * suit_temp_mod
            elif effective_temp > comfort_high:
                effective_temp = effective_temp - (effective_temp - comfort_high) * suit_temp_mod

        # === Shelter effect ===
        if has_shelter:
            if shelter_temp_reduction >= 0.9:
                # Fully sealed & heated base or habitat module (ECLSS 21°C)
                effective_temp = 21.0
            elif shelter_temp_reduction > 0:
                # Partial shelter (e.g. basic shelter, windbreak)
                effective_temp = effective_temp + (21.0 - effective_temp) * shelter_temp_reduction
            else:
                effective_temp = 21.0

        # === Stress calculation ===
        # Tikuisis et al. (1999), Wilderness & Environmental Medicine:
        # Hypothermia onset time with NASA insulated spacesuit ~ 4-6 hours in sub-zero ambient
        if effective_temp < comfort_low:
            deficit = comfort_low - effective_temp
            rate = (
                0.008 * (deficit ** 1.1)
                * (getattr(self, "tick_minutes", 5.0) / 5.0)
            )
            self.temperature_stress = max(0.0, self.temperature_stress - rate)
        elif effective_temp > comfort_high:
            excess = effective_temp - comfort_high
            rate = (
                0.008 * (excess ** 1.1)
                * (getattr(self, "tick_minutes", 5.0) / 5.0)
            )
            self.temperature_stress = min(100.0, self.temperature_stress + rate)
        else:
            # In comfort zone / heated habitat — gradual physiological core re-warming back to 37C (stress=50)
            # Clinical rate: ~1.5-2.0°C/hr (~1.0-1.2 stress/tick) to prevent lethal peripheral afterdrop shock
            rewarm_rate = (
                1.2 if (has_shelter and shelter_temp_reduction >= 0.9) else 0.6
            ) * (getattr(self, "tick_minutes", 5.0) / 5.0)
            if self.temperature_stress < 50:
                self.temperature_stress = min(50.0, self.temperature_stress + rewarm_rate)
            elif self.temperature_stress > 50:
                self.temperature_stress = max(50.0, self.temperature_stress - rewarm_rate)

    def check_death(self) -> Optional[DeathCause]:
        """Check if any death condition is met. Returns cause or None."""
        if self._o2_death_timer >= self.SUFFOCATION_TICKS:
            return DeathCause.SUFFOCATION
        if self._thirst_death_timer >= self.DEHYDRATION_TICKS:
            return DeathCause.DEHYDRATION
        if self._hunger_death_timer >= self.STARVATION_TICKS:
            return DeathCause.STARVATION
        # Sleep deprivation / exertional fatigue is not itself a medically
        # valid cause of death. Zero energy causes collapse below; mortality
        # can follow only from a real secondary insult such as anoxia,
        # exposure, dehydration, starvation or injury.
        if self._temp_death_timer >= self.TEMP_DEATH_TICKS:
            if self.temperature_stress <= 0:
                return DeathCause.HYPOTHERMIA
            else:
                return DeathCause.HYPERTHERMIA
        return None

    def get_debuffs(self, radiation_sickness_level: int = 0,
                    active_diseases: list = None) -> dict:
        """
        Calculate current debuffs based on need levels, radiation sickness,
        and active diseases.

        Returns modifiers that affect agent capabilities.

        Radiation sickness debuffs (based on real ARS progression):
        - Level 0: No effect
        - Level 1 (>1 Sv): Nausea, fatigue → -20% action speed, +25% hunger drain
        - Level 2 (>2 Sv): Chronic damage → PERMANENT endurance -2, immunity -2,
          -40% action speed, can't do heavy labor
        - Level 3 (>4 Sv): Terminal → incapacitated, death countdown
        """
        if active_diseases is None:
            active_diseases = []

        debuffs = {
            "movement_speed": 1.0,
            "can_craft": True,
            "can_gather": True,
            "can_heavy_labor": True,
            "action_speed": 1.0,
            "perception_modifier": 0,
            "disease_risk_per_tick": 0.0,
            "social_penalty": 0.0,
            "cognitive_impairment": 0.0,  # 0=clear, 1.0=severely impaired
        }

        # === HUNGER DEBUFFS ===
        # Based on starvation physiology: glycogen depletion → muscle catabolism
        if self.hunger <= 10:
            debuffs["movement_speed"] *= 0.3
            debuffs["can_craft"] = False
            debuffs["can_heavy_labor"] = False
            debuffs["action_speed"] *= 0.4
            debuffs["cognitive_impairment"] += 0.4
        elif self.hunger <= 20:
            debuffs["movement_speed"] *= 0.5
            debuffs["can_craft"] = False
            debuffs["action_speed"] *= 0.6
            debuffs["cognitive_impairment"] += 0.2
        elif self.hunger <= 40:
            debuffs["movement_speed"] *= 0.8
            debuffs["action_speed"] *= 0.8

        # === THIRST DEBUFFS ===
        # Dehydration: 2% body mass loss → cognitive decline, >5% → organ stress
        if self.thirst <= 10:
            debuffs["movement_speed"] *= 0.3
            debuffs["can_craft"] = False
            debuffs["can_heavy_labor"] = False
            debuffs["action_speed"] *= 0.3
            debuffs["cognitive_impairment"] += 0.5
        elif self.thirst <= 20:
            debuffs["movement_speed"] *= 0.5
            debuffs["can_craft"] = False
            debuffs["action_speed"] *= 0.5
            debuffs["cognitive_impairment"] += 0.3

        # === ENERGY DEBUFFS ===
        # Sleep deprivation: >24h → microsleeps, >48h → hallucinations
        if self.energy <= 10:
            debuffs["can_craft"] = False
            debuffs["can_gather"] = False
            debuffs["can_heavy_labor"] = False
            debuffs["movement_speed"] *= 0.2
            debuffs["action_speed"] *= 0.2
            debuffs["cognitive_impairment"] += 0.6
        elif self.energy <= 20:
            debuffs["can_craft"] = False
            debuffs["can_gather"] = False
            debuffs["movement_speed"] *= 0.3
            debuffs["action_speed"] *= 0.3
            debuffs["cognitive_impairment"] += 0.3
        elif self.energy <= 35:
            debuffs["action_speed"] *= 0.7
            debuffs["cognitive_impairment"] += 0.1

        # === HYGIENE DEBUFFS ===
        # Infection risk follows exponential curve with time since last wash
        if self.hygiene <= 10:
            debuffs["disease_risk_per_tick"] = 0.005  # ~50% chance per 100 ticks
            debuffs["social_penalty"] = 0.3
        elif self.hygiene <= 20:
            debuffs["disease_risk_per_tick"] = 0.002
            debuffs["social_penalty"] = 0.2
        elif self.hygiene <= 30:
            debuffs["disease_risk_per_tick"] = 0.001
            debuffs["social_penalty"] = 0.1

        # === O2 DEBUFFS ===
        # Hypoxia stages: confusion → unconsciousness → death
        if self.o2_supply <= 5:
            debuffs["can_craft"] = False
            debuffs["can_gather"] = False
            debuffs["can_heavy_labor"] = False
            debuffs["movement_speed"] *= 0.2
            debuffs["cognitive_impairment"] += 0.8
        elif self.o2_supply <= 15:
            debuffs["action_speed"] *= 0.5
            debuffs["cognitive_impairment"] += 0.4
        elif self.o2_supply <= 30:
            debuffs["action_speed"] *= 0.8
            debuffs["cognitive_impairment"] += 0.1

        # === RADIATION SICKNESS DEBUFFS ===
        # Real ARS progression (IAEA/UNSCEAR data)
        if radiation_sickness_level >= 3:
            # Terminal (>4 Sv): unable to function
            debuffs["can_craft"] = False
            debuffs["can_gather"] = False
            debuffs["can_heavy_labor"] = False
            debuffs["movement_speed"] *= 0.1
            debuffs["action_speed"] *= 0.1
            debuffs["cognitive_impairment"] += 0.9
        elif radiation_sickness_level >= 2:
            # Chronic (>2 Sv): permanent damage, severely reduced capacity
            debuffs["can_heavy_labor"] = False
            debuffs["movement_speed"] *= 0.5
            debuffs["action_speed"] *= 0.6
            debuffs["disease_risk_per_tick"] += 0.003  # Immunocompromised
            debuffs["cognitive_impairment"] += 0.3
        elif radiation_sickness_level >= 1:
            # Mild (>1 Sv): nausea, reduced performance
            debuffs["action_speed"] *= 0.8
            debuffs["disease_risk_per_tick"] += 0.001

        # === ACTIVE DISEASE DEBUFFS ===
        for disease in active_diseases:
            effects = disease.get_effects()
            if "action_speed_mod" in effects:
                debuffs["action_speed"] *= effects["action_speed_mod"]
            if "movement_speed_mod" in effects:
                debuffs["movement_speed"] *= effects["movement_speed_mod"]
            if "perception_penalty" in effects:
                debuffs["perception_modifier"] += effects["perception_penalty"]

        # === SLEEP DEBT COGNITIVE EFFECTS ===
        if self._sleep_debt > 5:
            debuffs["cognitive_impairment"] += min(0.5, self._sleep_debt * 0.05)

        # Clamp cognitive impairment
        debuffs["cognitive_impairment"] = min(1.0, debuffs["cognitive_impairment"])

        return debuffs

    def to_compact_string(self) -> str:
        """
        One-line summary for LLM context.
        Reports only notable/concerning values to keep token count low.
        """
        parts = []
        if self.hunger < 40:
            parts.append(f"hungry ({self.hunger:.0f}/100)")
        if self.thirst < 40:
            parts.append(f"thirsty ({self.thirst:.0f}/100)")
        if self.energy < 40:
            parts.append(f"tired ({self.energy:.0f}/100)")
        if self.hygiene < 30:
            parts.append(f"low hygiene ({self.hygiene:.0f}/100)")
        if self.temperature_stress < 20:
            parts.append(f"freezing ({self.temperature_stress:.0f}/100)")
        elif self.temperature_stress > 80:
            parts.append(f"overheating ({self.temperature_stress:.0f}/100)")
        if self.o2_supply < 40:
            parts.append(f"LOW O₂ ({self.o2_supply:.0f}/100)")
        if self._sleep_debt > 3:
            parts.append(f"sleep-deprived (debt: {self._sleep_debt:.0f})")

        if not parts:
            return "all needs stable"
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {
            "hunger": round(self.hunger, 1),
            "thirst": round(self.thirst, 1),
            "energy": round(self.energy, 1),
            "hygiene": round(self.hygiene, 1),
            "temperature_stress": round(self.temperature_stress, 1),
            "o2_supply": round(self.o2_supply, 1),
            "sleep_debt": round(self._sleep_debt, 1),
            "ticks_since_full_sleep": self._ticks_since_last_full_sleep,
        }


@dataclass
class Inventory:
    """
    Agent's carried items and materials with realistic weight modeling
    and tool durability tracking.
    """
    materials: dict[str, int] = field(default_factory=dict)
    items: dict[str, int] = field(default_factory=dict)

    # Tool durability tracking: item_id → remaining uses
    # Tools with 0 durability are broken and cannot be used
    tool_durability: dict[str, int] = field(default_factory=dict)

    # Rechargeable field instruments keep battery state separate from wear.
    # A scanner does not become scrap merely because its battery is empty.
    tool_charge_pct: dict[str, float] = field(default_factory=dict)

    # Base carry capacity in kg (modified by genome.strength and gravity)
    BASE_CARRY_CAPACITY_KG: float = 30.0

    # Default durability for tools (max uses before breaking)
    TOOL_MAX_DURABILITY = {
        "stone_hammer": 30,
        "multitool_kit": 200,
        "hand_tools": 100,
        "portable_scanner": 500,
    }

    RECHARGEABLE_TOOLS = {"portable_scanner"}

    # Item weights (kg per unit)
    ITEM_WEIGHTS_KG = {
        "emergency_rations": 0.3,       # ~300g per meal pack
        "water_packs": 1.0,            # 1 kg per liter
        "oxygen_canisters": 2.5,       # Pressurized canister
        "empty_oxygen_canisters": 2.14,  # Same vessel without its 0.36 kg O2 charge
        "emergency_blanket": 0.2,      # Mylar blanket
        "multitool_kit": 3.0,          # Full toolkit
        "electronics_salvage": 0.8,    # Circuit boards, components
        "medical_supplies": 0.5,       # Per treatment pack
        "sterile_iv_fluid_bags": 1.05, # 1 L crystalloid plus sterile bag
        "iv_io_administration_sets": 0.15,  # Tubing + IV/IO access kit
        "portable_scanner": 2.0,       # Handheld device
        "emergency_beacon_parts": 5.0, # Antenna + electronics
        "hand_tools": 4.0,             # Crafted iron tools
        "stone_hammer": 2.0,           # Crude basalt tool
        "protective_suit": 12.0,       # Full-body suit
        "bedroll": 3.5,                # Insulated sleeping surface
        "musical_instrument": 2.0,     # Percussion/wind instrument
        "game_set": 3.0,               # Carved basalt chess/go set
        "pharmaceutical_kit": 0.8,     # Medical supplies
    }

    # Refined/intermediate material weights (kg per unit)
    REFINED_MATERIAL_WEIGHTS = {
        "reduced_iron_ingot": 5.0,      # Reduced/cast iron bar
        "glass_pane": 2.0,             # Flat glass sheet
        "insulated_fabric": 0.8,       # Basalt fiber textile
        "rope_cordage": 0.5,           # Carbon fiber rope per unit
        "composite_panel": 5.0,        # Structural panel
        "electronic_component": 0.3,   # Circuit assembly
    }

    def add_material(self, material: str, quantity: int):
        self.materials[material] = self.materials.get(material, 0) + quantity

    def remove_material(self, material: str, quantity: int) -> bool:
        """Remove material. Returns False if insufficient."""
        if self.materials.get(material, 0) < quantity:
            return False
        self.materials[material] -= quantity
        if self.materials[material] <= 0:
            del self.materials[material]
        return True

    def has_materials(self, requirements: dict[str, int]) -> bool:
        """Check if inventory has all required materials."""
        for mat, qty in requirements.items():
            if self.materials.get(mat, 0) < qty:
                return False
        return True

    def add_item(self, item_id: str, quantity: int = 1):
        self.items[item_id] = self.items.get(item_id, 0) + quantity
        # Initialize durability for tools
        if item_id in self.TOOL_MAX_DURABILITY and item_id not in self.tool_durability:
            self.tool_durability[item_id] = self.TOOL_MAX_DURABILITY[item_id]
        if item_id in self.RECHARGEABLE_TOOLS and item_id not in self.tool_charge_pct:
            self.tool_charge_pct[item_id] = 100.0

    def remove_item(self, item_id: str, quantity: int = 1) -> bool:
        """Remove item from inventory. Returns False if insufficient."""
        if self.items.get(item_id, 0) < quantity:
            return False
        self.items[item_id] -= quantity
        if self.items[item_id] <= 0:
            del self.items[item_id]
            if item_id in self.tool_durability:
                del self.tool_durability[item_id]
            if item_id in self.tool_charge_pct:
                del self.tool_charge_pct[item_id]
        return True

    def has_item(self, item_id: str) -> bool:
        return self.items.get(item_id, 0) > 0

    def has_usable_tool(self) -> bool:
        """
        Check if agent has any usable tool (not broken).
        Priority: multitool_kit > hand_tools > stone_hammer
        """
        for tool in ("multitool_kit", "hand_tools", "stone_hammer"):
            if self.has_item(tool) and self.tool_durability.get(tool, 0) > 0:
                return True
        return False

    def get_best_tool(self) -> Optional[str]:
        """
        Get the best available tool (by effectiveness).
        Returns None if no usable tools available.
        """
        for tool in ("multitool_kit", "hand_tools", "stone_hammer"):
            if self.has_item(tool) and self.tool_durability.get(tool, 0) > 0:
                return tool
        return None

    def use_tool(self, uses: int = 1) -> dict:
        """
        Consume durability from the best available tool.
        Returns dict with tool used, remaining durability, and whether it broke.
        """
        tool = self.get_best_tool()
        if tool is None:
            return {"tool_used": None, "durability_remaining": 0, "broke": False}

        self.tool_durability[tool] = max(0, self.tool_durability.get(tool, 0) - uses)
        broke = self.tool_durability[tool] <= 0

        if broke:
            # Remove broken tool from inventory and salvage scrap metal
            self.items[tool] -= 1
            if self.items[tool] <= 0:
                del self.items[tool]
            del self.tool_durability[tool]
            
            # Salvage usable scrap materials
            if tool == "stone_hammer":
                self.materials["basalt_scrap"] = self.materials.get("basalt_scrap", 0) + 1
            else:
                self.materials["scrap_metal"] = self.materials.get("scrap_metal", 0) + 2

        return {
            "tool_used": tool,
            "durability_remaining": self.tool_durability.get(tool, 0),
            "broke": broke,
        }

    def repair_tool(self, tool_id: str, repair_amount: int = 50) -> bool:
        """
        Repair a tool (requires forge structure).
        Restores up to 50% of max durability per repair.
        Returns True if repair was performed.
        """
        if not self.has_item(tool_id) or tool_id not in self.TOOL_MAX_DURABILITY:
            return False
        max_dur = self.TOOL_MAX_DURABILITY[tool_id]
        current = self.tool_durability.get(tool_id, 0)
        self.tool_durability[tool_id] = min(max_dur, current + repair_amount)
        return True

    def total_weight_kg(self) -> float:
        """
        Total carried weight in kg using real material densities.
        Includes raw materials, refined materials, and items.
        """
        material_weight = sum(
            qty * MATERIAL_DENSITY_KG.get(mat, self.REFINED_MATERIAL_WEIGHTS.get(mat, 1.5))
            for mat, qty in self.materials.items()
        )
        item_weight = sum(
            qty * self.ITEM_WEIGHTS_KG.get(item, 1.0)
            for item, qty in self.items.items()
        )
        return material_weight + item_weight

    def carry_capacity_kg(self, strength: int, gravity_multiplier: float = 1.0) -> float:
        """
        Max carry capacity in kg, modified by strength and gravity.

        Strength 5 (baseline) = 30 kg capacity at 1.0g
        Each strength point adds/removes 5 kg capacity
        Higher gravity reduces capacity proportionally
        """
        base = self.BASE_CARRY_CAPACITY_KG + (strength - 5) * 5.0
        return base / gravity_multiplier

    def is_overloaded(self, strength: int, gravity_multiplier: float = 1.0) -> bool:
        return self.total_weight_kg() > self.carry_capacity_kg(strength, gravity_multiplier)

    def overload_fraction(self, strength: int, gravity_multiplier: float = 1.0) -> float:
        """How much over capacity (1.0 = at limit, 1.5 = 50% over)."""
        cap = self.carry_capacity_kg(strength, gravity_multiplier)
        return self.total_weight_kg() / cap if cap > 0 else 999.0

    def get_tool_status(self) -> str:
        """One-line tool status for LLM context."""
        parts = []
        for tool in ("multitool_kit", "hand_tools", "stone_hammer"):
            if self.has_item(tool):
                dur = self.tool_durability.get(tool, 0)
                max_dur = self.TOOL_MAX_DURABILITY.get(tool, 100)
                pct = int(dur / max_dur * 100) if max_dur > 0 else 0
                parts.append(f"{tool}: {pct}%")
        return ", ".join(parts) if parts else "no tools"

    def to_compact_string(self) -> str:
        """One-line summary for LLM context."""
        parts = []
        for mat, qty in sorted(self.materials.items()):
            parts.append(f"{qty} {mat}")
        for item, qty in sorted(self.items.items()):
            parts.append(f"{qty}x {item}")
        return ", ".join(parts) if parts else "empty"

    def to_dict(self) -> dict:
        return {
            "materials": dict(self.materials),
            "items": dict(self.items),
            "tool_durability": dict(self.tool_durability),
            "tool_charge_pct": {
                key: round(value, 1)
                for key, value in self.tool_charge_pct.items()
            },
        }


@dataclass
class ActionState:
    """Tracks the agent's current action and progress."""
    action_type: Optional[str] = None  # move, gather, build, eat, drink, sleep, wash, talk, trade, flee, explore
    target: Optional[dict] = None      # Action-specific target info
    ticks_remaining: int = 0           # Ticks until action completes
    ticks_elapsed: int = 0             # Ticks spent on current action
    interruptible: bool = True         # Can this action be interrupted by events?

    @property
    def is_active(self) -> bool:
        return self.action_type is not None and self.ticks_remaining > 0

    def tick(self) -> bool:
        """Advance action by one tick. Returns True if action just completed."""
        if not self.is_active:
            return False
        self.ticks_remaining -= 1
        self.ticks_elapsed += 1
        return self.ticks_remaining <= 0

    def clear(self):
        """Clear the current action."""
        self.action_type = None
        self.target = None
        self.ticks_remaining = 0
        self.ticks_elapsed = 0

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "ticks_remaining": self.ticks_remaining,
            "ticks_elapsed": self.ticks_elapsed,
        }


# Metabolic Equivalent of Task (MET) values for activity cost calculation
# NASA EVA metabolic rate data + exercise physiology references
# Maps action types to activity_multiplier for Needs.decay()
ACTIVITY_MET_TABLE = {
    "idle": 1.0,          # Sitting/standing, BMR only
    "sleep": 0.8,         # Sleep reduces metabolic rate by ~20%
    "medical_rest": 0.8,  # Monitored recovery in a habitat medical bunk
    "rest": 0.9,          # Awake recovery
    "arrived": 1.0,
    "eat": 1.0,           # Eating = idle
    "drink": 1.0,
    "wash": 1.2,          # Light activity
    "talk": 1.0,          # Social interaction
    "trade": 1.0,
    "explore": 1.5,       # Walking on rough terrain (~3.5 MET)
    "move": 1.5,          # Walking
    "gather": 1.8,        # Moderate physical labor (~4.5 MET)
    "build": 2.0,         # Heavy construction (~6 MET, NASA EVA metabolic rate)
    "craft": 1.5,         # Crafting = moderate
    "refine": 1.4,        # Operating furnace/forge production equipment
    "operate_machine": 1.3,  # Loading, program verification and safe startup
    "produce": 1.4,
    "production": 1.4,
    "manufacture": 1.5,
    "mine": 2.2,          # Heavy excavation (~7 MET)
    "prospect": 1.8,      # Hand sampling / shallow test pit
    "flee": 2.5,          # Running on alien terrain (~8 MET)
    "carry_heavy": 2.3,   # Hauling materials
    "medical": 1.2,       # Medical treatment (light physical)
    "treat": 1.2,
    "treat_injury": 1.2,
    "scan": 1.3,          # Using scanner while walking
    "repair": 1.8,        # Structural repair with powered/hand tools
    "repair_tool": 1.4,
    "maintenance": 1.6,
    "clean_solar_panels": 1.6,
    "seal_patch": 1.3,
    "recycle_scrap": 1.6,
    "farm": 1.6,          # Planting, watering and harvesting
    "deposit_materials": 1.7,
    "scavenge_corpse": 1.7,
    "rescue": 2.6,        # Stabilizing/carrying a crewmate in 1.31 g
    "carry_person": 2.6,
    "communicate": 1.0,
    "use_item": 1.1,
    "refill_o2": 1.1,
    "enter_habitat": 1.2,
    "exit_habitat": 1.2,
}


class Agent:
    """
    Full agent entity with all state.

    Created from agent_presets.json configuration.
    Scientifically modeled autonomous entity with realistic physiological
    constraints based on NASA-STD-3001 human spaceflight standards.
    """

    # Food quality table — affects morale when consumed
    FOOD_TYPES = {
        "emergency_rations": {"kcal": 700, "morale": -0.01, "label": "monotonous packaged meal"},
        "greenhouse_produce": {"kcal": 500, "morale": 0.03, "label": "fresh vegetables"},
        "foraged_lichen": {"kcal": 200, "morale": -0.02, "label": "bitter alien lichen"},
    }

    TICKS_PER_DAY = 288.0
    TICKS_PER_HOUR = 12.0
    SIM_MINUTES_PER_TICK = 5.0
    REFERENCE_MASS_KG = 80.0
    REFERENCE_HEIGHT_CM = 175.0
    NASA_O2_NOMINAL_KG_PER_DAY = 0.72
    NASA_WATER_L_PER_DAY = 2.5
    NASA_EVA_WATER_L_PER_HOUR = 0.24
    REFERENCE_NOMINAL_KCAL_PER_DAY = 2200.0
    # 0.36 kg lasts 8 h for an 80 kg astronaut walking at 1.5× nominal load.
    PLSS_CANISTER_O2_KG = 0.36

    # Tool effectiveness multipliers for gathering
    GATHER_TOOL_BONUS = {
        None: 0.3,              # Bare hands — very slow
        "stone_hammer": 0.8,    # Crude but functional
        "hand_tools": 1.0,      # Standard rate
        "multitool_kit": 1.2,   # Best efficiency
    }

    def __init__(self, agent_config: dict, agent_index: int = 0):
        self.id = agent_config["id"]
        self.name = agent_config["name"]
        self.background = agent_config["background"]
        self.personality_traits = agent_config.get("personality_traits", [])
        self._rng = np.random.default_rng(max(0, int(agent_index)))

        body = agent_config.get("anthropometrics", {})
        self.anthropometrics = Anthropometrics(
            height_cm=float(body.get("height_cm", self.REFERENCE_HEIGHT_CM)),
            mass_kg=float(body.get("mass_kg", self.REFERENCE_MASS_KG)),
            foot_length_cm=float(body.get("foot_length_cm", 26.5)),
            shoe_size_eu=int(body.get("shoe_size_eu", 42)),
        )

        # Immutable attributes
        g = agent_config["genome"]
        self.genome = Genome(
            strength=g["strength"],
            agility=g["agility"],
            endurance=g["endurance"],
            perception=g["perception"],
            immunity=g["immunity"],
        )

        c = agent_config["competency"]
        self.competency = Competency(
            engineering=c["engineering"],
            medical=c["medical"],
            physics=c["physics"],
            botany_bio=c["botany_bio"],
            leadership_social=c["leadership_social"],
        )

        # Mutable state
        self.needs = Needs()
        self.inventory = Inventory()
        # NASA EVA Starting Field Manifest: 2 emergency food rations (1400 kcal), 2 water packs (2L), 2 O2 canisters
        self.inventory.items["emergency_rations"] = 2
        self.inventory.items["water_packs"] = 2
        self.inventory.items["oxygen_canisters"] = 2
        self.action = ActionState()
        self.status = AgentStatus.ALIVE
        self.death_cause: Optional[DeathCause] = None
        self.death_tick: Optional[int] = None
        self.corpse_salvaged: bool = False
        self.last_decision: dict = {}
        
        # Reinforcement Learning (RL) Cognitive Policy Engine
        self.q_table: dict[str, dict[str, float]] = {}  # state_key -> {action_key: Q_value}
        self.last_state_key: Optional[str] = None
        self.last_action_key: Optional[str] = None
        # Set only when the RL policy actually selects a new transition. This
        # prevents one old action from being rewarded/penalized on every later
        # deterministic sleep, treatment, or emergency tick.
        self._rl_transition_pending: bool = False
        # Bounded episode history lets a terminal outcome credit or penalize
        # the planning decisions that led to it, instead of teaching only the
        # final action taken immediately before death/success.
        self.rl_episode_trace: list[tuple[str, str]] = []
        self.rl_planet_id: Optional[str] = None
        self.rl_policy_schema_reset: Optional[dict] = None
        self.total_accumulated_reward: float = 0.0
        self.rl_learning_rate: float = 0.20
        self.rl_discount_factor: float = 0.85
        self.rl_epsilon_explore: float = 0.12
        
        # Explored cells tracking
        self.explored_cells = set()

        # Position (set during world init)
        self.x: int = 0
        self.y: int = 0
        self.spawn_x: int = 0
        self.spawn_y: int = 0

        # Radiation tracking (per IAEA/UNSCEAR dose-response data)
        self.cumulative_radiation_sv: float = 0.0
        self.radiation_sickness_level: int = 0  # 0=none, 1=mild, 2=chronic, 3=terminal
        # Permanent stat penalties from chronic radiation (Level 2+)
        # These are IRREVERSIBLE — once applied, they persist even if
        # cumulative dose doesn't increase further
        self._radiation_permanent_endurance_penalty: int = 0
        self._radiation_permanent_immunity_penalty: int = 0
        self._radiation_penalties_applied: bool = False

        # Typed disease system (replaces simple boolean)
        self.active_diseases: list[Disease] = []

        # Strategic plan (current high-level goal)
        self.strategic_goal: Optional[dict] = None
        self.strategic_plan_steps: list[dict] = []
        # Short-lived colony work assignment. Survival may interrupt it, but
        # after recovery the astronaut resumes the same physical BOM task
        # instead of selecting a different material every decision tick.
        self._shared_work_contract: Optional[dict] = None

        # Social relationships (trust scores with other agents, -1.0 to 1.0)
        self.trust_scores: dict[str, float] = {}

        # Morale (affected by deaths, social interactions, isolation)
        # 1.0=baseline, <0.5=severely demoralized, >1.0=inspired (from leadership)
        self.morale: float = 1.0

        # Injury system
        self.injury_level: float = 0.0  # 0=healthy, 1.0=critical injury

        # === EVA SUIT SYSTEM (atmosphereless planet survival) ===
        # On atmosphereless worlds, agents MUST wear a pressurized EVA suit
        # to survive outside habitats. Without suit in vacuum:
        # - 0-15 sec: Useful consciousness time (NASA JSC data)
        # - 15-30 sec: Loss of consciousness (anoxia)
        # - 60-90 sec: Ebullism (body fluids boil at <6.3 kPa = Armstrong limit)
        # - >90 sec: Death
        # Reference: NASA SP-368, "Bioastronautics Data Book" (1973);
        #            Harding & Mills, "Aviation Medicine" (1983)
        self.suit_equipped: bool = False       # Whether wearing pressurized suit
        self.suit_integrity: float = 1.0       # 1.0=pristine, 0.0=breached
        self.suit_condition: float = 1.0       # Reversible dust/joint/service condition
        self.suit_durability_ticks: int = 800  # Total EVA ticks before suit fails
        self._suit_durability_minutes: float = 800 * 5.0
        self.suit_ticks_used: int = 0          # Cumulative EVA ticks in current suit
        self.suit_radiation_factor: float = 0.9  # Modest incidental attenuation; no GCR shield
        self.suit_temp_reduction: float = 0.75   # Temperature stress reduction
        self.suit_o2_efficiency: float = 1.2    # O2 canister lasts 20% longer

        # NASA xEMU PLSS (Portable Life Support System) Subsystems
        # Regenerable solid-amine bed. Habitat servicing still requires a
        # powered rack; it is not an infinitely replaced LiOH cartridge.
        self.plss_co2_scrubber_pct: float = 100.0
        self.plss_suit_battery_pct: float = 100.0  # 4 kWh suit heating & avionics battery
        self.has_micro_puncture: bool = False       # Micro-meteoroid / sharp rock puncture
        self.hypercapnia_level: float = 0.0         # 0.0 = nominal, 1.0 = lethal CO2 buildup

        # EVA tracking
        self._eva_ticks_continuous: int = 0    # Consecutive ticks outside habitat
        self._vacuum_exposure_ticks: int = 0   # Ticks in vacuum WITHOUT suit
        self._in_habitat: bool = False         # Whether inside pressurized structure
        self._active_expedition: Optional[dict] = None
        self._pending_expedition: Optional[dict] = None

        # O2 canister management
        # 1 canister = 0.36 kg O₂: 8 hours for the 80 kg reference astronaut
        # while walking at 1.5× nominal metabolic load.
        # Without canisters on atmosphereless planet = suffocation.
        self._current_canister_remaining: float = 0.0  # Remaining O2 in current canister (0-100)
        self._total_canisters_loaded: int = 0  # How many canisters have been used
        self._has_active_o2_canister: bool = False

        # Statistics (for post-simulation analysis)
        self.total_resources_gathered: dict[str, int] = {}
        self.total_structures_built: int = 0
        self.total_llm_calls: int = 0
        self.ticks_alive: int = 0
        self.total_distance_traveled: float = 0.0
        self.total_kcal_consumed: int = 0
        self.total_water_consumed_l: float = 0.0
        self.total_o2_canisters_used: int = 0
        self.diseases_contracted: int = 0
        self.near_death_events: int = 0
        self.total_social_interactions: int = 0
        self.configure_timebase(5.0)

    def configure_timebase(self, tick_minutes: float) -> None:
        """Configure per-agent physical rates for the mission clock."""
        tick_minutes = max(0.1, float(tick_minutes))
        self.SIM_MINUTES_PER_TICK = tick_minutes
        self.TICKS_PER_HOUR = 60.0 / tick_minutes
        self.TICKS_PER_DAY = 1440.0 / tick_minutes
        self.needs.configure_timebase(tick_minutes)
        durability_minutes = float(getattr(
            self, "_suit_durability_minutes", 800 * 5.0
        ))
        self.suit_durability_ticks = max(
            1, math.ceil(durability_minutes / tick_minutes)
        )

    def ensure_tactical_policy_schema(self) -> bool:
        """Make a persisted tactical table compatible with this observation.

        Q rows are stored by planet in the database. The embedded metadata is
        a second, cheap guard against reusing values after the state encoding
        changes or after an incorrectly routed cross-planet payload.
        """
        metadata = self.q_table.get(TACTICAL_POLICY_META_KEY)
        expected_planet = str(self.rl_planet_id or "")
        loaded_version = (
            int(metadata.get("state_schema_version", 0))
            if isinstance(metadata, dict) else 0
        )
        loaded_planet = (
            str(metadata.get("planet_id") or "")
            if isinstance(metadata, dict) else ""
        )
        incompatible = bool(self.q_table) and (
            loaded_version != TACTICAL_STATE_SCHEMA_VERSION
            or bool(
                expected_planet
                and loaded_planet
                and loaded_planet != expected_planet
            )
        )
        if incompatible:
            old_state_count = sum(
                1 for key in self.q_table
                if key != TACTICAL_POLICY_META_KEY
            )
            self.q_table.clear()
            self.rl_episode_trace.clear()
            self.last_state_key = None
            self.last_action_key = None
            self._rl_transition_pending = False
            self.rl_policy_schema_reset = {
                "loaded_version": loaded_version,
                "required_version": TACTICAL_STATE_SCHEMA_VERSION,
                "loaded_planet": loaded_planet or None,
                "required_planet": expected_planet or None,
                "discarded_states": old_state_count,
            }
        self.q_table[TACTICAL_POLICY_META_KEY] = {
            "state_schema_version": TACTICAL_STATE_SCHEMA_VERSION,
            "planet_id": expected_planet or loaded_planet or None,
        }
        return incompatible

    def _legacy_duration_ticks(self, legacy_ticks: float) -> int:
        """Preserve a duration authored against the old five-minute tick."""
        return max(1, math.ceil(
            float(legacy_ticks) * 5.0 / self.SIM_MINUTES_PER_TICK
        ))

    def _scaled_tick_probability(self, legacy_probability: float) -> float:
        """Convert a five-minute hazard probability to this tick duration."""
        probability = max(0.0, min(1.0, float(legacy_probability)))
        scale = self.SIM_MINUTES_PER_TICK / 5.0
        return 1.0 - (1.0 - probability) ** scale

    def _hourly_probability_for_tick(self, hourly_probability: float) -> float:
        """Convert a conditional EVA-hour probability to this physics tick.

        Biome accident priors are authored per hour spent exposed, not per
        arbitrary renderer/physics tick.  This keeps the cumulative risk
        invariant when the simulation tick duration changes.
        """
        probability = max(0.0, min(1.0, float(hourly_probability)))
        hours = max(0.0, self.SIM_MINUTES_PER_TICK / 60.0)
        return 1.0 - (1.0 - probability) ** hours

    # === SOCIAL INTERACTION MECHANICS ===

    def build_trust(self, other_agent_id: str, amount: float = 0.05,
                    reason: str = "cooperation"):
        """
        Increase trust toward another agent.

        Trust changes from:
        - Working together on same task (+0.05/tick)
        - Sharing resources (+0.1)
        - Medical treatment received (+0.15)
        - Successful joint builds (+0.08)
        - Social time in recreation area (+0.005/tick)

        Trust is asymmetric: A can trust B more than B trusts A.
        Range: -1.0 (enemy) to +1.0 (deep trust)
        """
        current = self.trust_scores.get(other_agent_id, 0.0)
        self.trust_scores[other_agent_id] = min(1.0, current + amount)
        self.total_social_interactions += 1

    def lose_trust(self, other_agent_id: str, amount: float = 0.1,
                   reason: str = "conflict"):
        """
        Decrease trust toward another agent.

        Trust decreases from:
        - Resource theft/hoarding (-0.15)
        - Refusing to help when asked (-0.1)
        - Failing a joint task (-0.05)
        - Conflict/argument (-0.1)
        - Abandoning injured teammate (-0.2)
        """
        current = self.trust_scores.get(other_agent_id, 0.0)
        self.trust_scores[other_agent_id] = max(-1.0, current - amount)

    def get_cooperation_bonus(self, other_agent_id: str) -> float:
        """
        Calculate cooperation efficiency bonus based on trust.

        High trust (>0.5): +15% efficiency on joint tasks
        Neutral (0.0): no bonus
        Distrust (<-0.3): -20% efficiency, may refuse cooperation

        Returns multiplier (1.0 = baseline, 1.15 = high trust, 0.8 = distrust)
        """
        trust = self.trust_scores.get(other_agent_id, 0.0)
        if trust > 0.5:
            return 1.0 + trust * 0.15  # Up to 1.15
        elif trust > 0:
            return 1.0 + trust * 0.1   # Up to 1.05
        elif trust > -0.3:
            return 1.0 + trust * 0.1   # Down to 0.97
        else:
            return 0.8                  # Severe distrust

    def will_cooperate_with(self, other_agent_id: str) -> bool:
        """
        Determine if agent is willing to cooperate with another.

        Based on trust level and personality traits.
        Even distrusted agents may cooperate in emergencies.
        """
        trust = self.trust_scores.get(other_agent_id, 0.0)
        if trust >= -0.3:
            return True
        # Deeply distrusted — refuse unless leader personality
        if "cooperative" in self.personality_traits or "pragmatic" in self.personality_traits:
            return True  # These personality types cooperate despite distrust
        return False

    def apply_social_morale(self, event_type: str, other_agent_name: str = "",
                            trust_level: float = 0.0):
        """
        Apply morale changes from social events.

        Events and their effects:
        - teammate_death: -0.15 (stronger if trusted: -0.25)
        - shared_meal: +0.05
        - argument: -0.05
        - successful_build: +0.03
        - isolation (no contact >50 ticks): -0.02/tick
        - music_heard: +0.03
        - received_medical_aid: +0.08
        - gave_medical_aid: +0.05 (prosocial boost)
        """
        MORALE_EFFECTS = {
            "teammate_death": -0.15,
            "shared_meal": 0.05,
            "argument": -0.05,
            "successful_build": 0.03,
            "isolation_tick": -0.02,
            "music_heard": 0.03,
            "received_medical_aid": 0.08,
            "gave_medical_aid": 0.05,
            "memorial_built": 0.06,
            "game_played": 0.04,
            "journal_written": 0.06,
        }

        base_effect = MORALE_EFFECTS.get(event_type, 0.0)

        # Trust amplifies emotional impact (both positive and negative)
        if trust_level > 0.3 and base_effect < 0:
            base_effect *= 1.5  # Losing a trusted ally hurts more
        elif trust_level > 0.3 and base_effect > 0:
            base_effect *= 1.3  # Positive interactions with trusted people feel better

        self.morale = max(0.1, min(1.5, self.morale + base_effect))

    def get_social_summary(self) -> str:
        """
        One-line social context for LLM decisions.
        """
        if not self.trust_scores:
            return "No social bonds yet."
        trusted = [aid for aid, t in self.trust_scores.items() if t > 0.3]
        distrusted = [aid for aid, t in self.trust_scores.items() if t < -0.3]
        parts = []
        if trusted:
            parts.append(f"Trusted: {len(trusted)} agents")
        if distrusted:
            parts.append(f"Distrusted: {len(distrusted)} agents")
        if not parts:
            parts.append("All relationships neutral")
        return ", ".join(parts)

    def set_position(self, x: int, y: int):
        """Set agent position."""
        self.x = x
        self.y = y

    def perception_range(self, is_night: bool = False,
                         light_level: float = 1.0,
                         nearby_light_sources: list = None) -> int:
        """
        View radius based on genome perception and conditions.

        Base 7 (15×15 grid), +1 per perception point above 5.
        Night reduces range by 50%, but nearby light sources restore it partially.
        Light sources (campfire, habitat_module) provide illumination
        within their light_radius, restoring up to 80% of daytime perception.
        Disease perception penalties stack.

        Args:
            is_night: whether it's currently dark
            light_level: ambient light from day/night cycle (0.0-1.0)
            nearby_light_sources: list of {light_radius, distance} dicts
        """
        base = 7 + max(0, self.genome.perception - 5)

        # Night / low light penalty
        if is_night or light_level < 0.3:
            night_penalty = 0.5  # Default: 50% reduction

            # Nearby light sources restore partial vision
            if nearby_light_sources:
                best_light = 0.0
                for light in nearby_light_sources:
                    # Light effectiveness drops with distance
                    dist = light.get("distance", 99)
                    radius = light.get("light_radius", 0)
                    if radius > 0 and dist <= radius:
                        effectiveness = 1.0 - (dist / radius)
                        best_light = max(best_light, effectiveness)
                # Best light source restores up to 80% of daytime vision
                night_penalty = 0.5 + best_light * 0.3  # 0.5 → 0.8

            base = max(3, int(base * night_penalty))

        # Disease perception penalty
        for disease in self.active_diseases:
            penalty = disease.get_effects().get("perception_penalty", 0)
            base = max(2, base + penalty)

        # Radiation chronic damage
        if self.radiation_sickness_level >= 2:
            base = max(3, base - 2)

        return base

    def get_activity_multiplier(self) -> float:
        """
        Get current Metabolic Equivalent of Task (MET) based on action type.

        NASA EVA data shows astronauts performing extravehicular activities
        burn 200-400 kcal/hour (4-8 MET equivalent).
        """
        if self.action.action_type:
            if (
                self.action.action_type == "rescue"
                and isinstance(self.action.target, dict)
                and self.action.target.get("carrying_victim")
            ):
                return ACTIVITY_MET_TABLE["carry_person"]
            activity = ACTIVITY_MET_TABLE.get(self.action.action_type, 1.0)
            target = (
                self.action.target
                if isinstance(self.action.target, dict) else {}
            )
            # A learned surge is real exertion, not free movement. Brisk
            # suited travel raises metabolic O2, water, food, fatigue and
            # thermal loads through the same physiology path as every other
            # action. The engine permits it only inside a conservative health
            # envelope and rewards the completed trip after measuring both
            # elapsed time and reserve loss.
            if (
                self.action.action_type == "move"
                and target.get("effort_mode") == "surge"
            ):
                activity *= max(
                    1.0,
                    min(2.0, float(target.get("effort_multiplier", 1.50))),
                )
            return activity
        return ACTIVITY_MET_TABLE["idle"]

    def oxygen_consumption_kg_per_tick(
        self, activity_multiplier: float | None = None
    ) -> float:
        """Personal O₂ demand scaled by BSA and current physical activity."""
        activity = (
            self.get_activity_multiplier()
            if activity_multiplier is None else max(0.5, float(activity_multiplier))
        )
        reference_bsa = math.sqrt(
            self.REFERENCE_HEIGHT_CM * self.REFERENCE_MASS_KG / 3600.0
        )
        body_scale = self.anthropometrics.body_surface_area_m2 / reference_bsa
        return (
            self.NASA_O2_NOMINAL_KG_PER_DAY
            / self.TICKS_PER_DAY
            * body_scale
            * activity
        )

    def plss_o2_percent_per_tick(
        self, activity_multiplier: float | None = None
    ) -> float:
        return (
            self.oxygen_consumption_kg_per_tick(activity_multiplier)
            / self.PLSS_CANISTER_O2_KG
            * 100.0
        )

    def water_requirement_l_per_tick(
        self,
        is_eva: bool | None = None,
        activity_multiplier: float | None = None,
    ) -> float:
        """Personal hydration demand from BSA, activity and EVA conditions."""
        reference_bsa = math.sqrt(
            self.REFERENCE_HEIGHT_CM * self.REFERENCE_MASS_KG / 3600.0
        )
        body_scale = self.anthropometrics.body_surface_area_m2 / reference_bsa
        demand = self.NASA_WATER_L_PER_DAY / self.TICKS_PER_DAY * body_scale
        activity = (
            self.get_activity_multiplier()
            if activity_multiplier is None else max(0.5, float(activity_multiplier))
        )
        eva = (not self._in_habitat) if is_eva is None else bool(is_eva)
        # NASA's 0.24 L/EVA-hour minimum is used as the walking-load
        # increment. Apply that increment to any physical work, including
        # movement and construction inside a pressurized habitat. An EVA has
        # at least walking-equivalent thermal/respiratory hydration load even
        # if its current action is briefly idle.
        activity_excess = max(0.0, activity - 1.0)
        if eva:
            activity_excess = max(0.5, activity_excess)
        if activity_excess > 0.0:
            demand += (
                self.NASA_EVA_WATER_L_PER_HOUR / self.TICKS_PER_HOUR
                * body_scale
                * (activity_excess / 0.5)
            )
        return demand

    def nominal_food_kcal_per_day(self) -> float:
        """Body-scaled reference energy budget represented by 100 hunger points."""
        reference_bsa = math.sqrt(
            self.REFERENCE_HEIGHT_CM * self.REFERENCE_MASS_KG / 3600.0
        )
        body_scale = self.anthropometrics.body_surface_area_m2 / reference_bsa
        return self.REFERENCE_NOMINAL_KCAL_PER_DAY * body_scale

    def hunger_points_for_kcal(self, kcal: float) -> float:
        """Convert physical food energy to this crew member's hunger scale."""
        return (
            max(0.0, float(kcal))
            / max(1.0, self.nominal_food_kcal_per_day())
            * 100.0
        )

    def get_life_support_profile(self) -> dict:
        """Return the agent's auditable, personalized daily resource targets."""
        return {
            "nominal_o2_kg_per_day": round(
                self.oxygen_consumption_kg_per_tick(1.0) * self.TICKS_PER_DAY,
                3,
            ),
            "habitat_water_l_per_day": round(
                self.water_requirement_l_per_tick(
                    is_eva=False, activity_multiplier=1.0
                ) * self.TICKS_PER_DAY,
                2,
            ),
            "walking_o2_kg_per_hour": round(
                self.oxygen_consumption_kg_per_tick(1.5)
                * self.TICKS_PER_HOUR,
                4,
            ),
            "walking_water_l_per_hour": round(
                self.water_requirement_l_per_tick(
                    is_eva=False, activity_multiplier=1.5
                ) * self.TICKS_PER_HOUR,
                3,
            ),
            "total_water_l_per_hour_during_eva": round(
                self.water_requirement_l_per_tick(
                    is_eva=True, activity_multiplier=1.5
                ) * self.TICKS_PER_HOUR,
                3,
            ),
            "nominal_food_kcal_per_day": round(
                self.nominal_food_kcal_per_day(), 0
            ),
            "plss_canister_o2_kg": self.PLSS_CANISTER_O2_KG,
        }

    def movement_speed(self, gravity_multiplier: float = 1.0) -> float:
        """
        Cells per tick movement speed.

        Modifiers:
        - Genome agility: ±10% per point from baseline
        - Gravity: inversely proportional (1.3g = 77% speed)
        - Overloaded: -50% if over carry capacity, scaled by overload fraction
        - Debuffs: from needs, diseases, radiation
        - Terrain: handled externally by the engine
        """
        base = 1.0 + 0.1 * (self.genome.agility - 5)
        debuffs = self.needs.get_debuffs(
            radiation_sickness_level=self.radiation_sickness_level,
            active_diseases=self.active_diseases,
        )
        speed = base * debuffs["movement_speed"]

        # Gravity effect on walking speed
        speed /= gravity_multiplier

        # Overloaded penalty (scaled — 50% over = 50% speed penalty)
        overload = self.operational_load_fraction(gravity_multiplier)
        if overload > 1.0:
            penalty = max(0.3, 1.0 - (overload - 1.0) * 0.5)
            speed *= penalty

        # Injury penalty
        if self.injury_level > 0:
            speed *= max(0.2, 1.0 - self.injury_level * 0.6)

        return max(0.1, speed)

    def operational_load_weight_kg(self) -> float:
        """Return mobile payload mass above the worn pressure-suit assembly."""
        weight = self.inventory.total_weight_kg()
        if self.suit_equipped and self.inventory.has_item("protective_suit"):
            # The strength-derived carry allowance is defined as payload on
            # top of the assigned suit/PLSS. Keep the suit in the physical
            # inventory and mass ledger, but do not charge it twice as cargo.
            weight -= self.inventory.ITEM_WEIGHTS_KG["protective_suit"]
        return max(0.0, weight)

    def operational_load_fraction(
        self, gravity_multiplier: float | None = None
    ) -> float:
        gravity = float(
            getattr(self, "gravity_g", 1.0)
            if gravity_multiplier is None else gravity_multiplier
        )
        capacity = self.inventory.carry_capacity_kg(
            self.genome.strength, gravity
        )
        return (
            self.operational_load_weight_kg() / capacity
            if capacity > 0.0 else 999.0
        )

    def apply_radiation(self, dose_sv: float):
        """
        Apply radiation dose. Cumulative — does not reset.

        ICRP Publication 103 / IAEA Safety Report Series No. 2:
        Radiation damages DNA via direct ionization — this is a PHYSICAL process
        independent of immune function. Immunity does NOT reduce dose absorbed.

        ARS thresholds (acute whole-body dose):
        - >1.0 Sv: Prodromal syndrome (nausea, fatigue, lymphopenia)
        - >2.0 Sv: Hematopoietic syndrome (permanent bone marrow damage)
        - >4.0 Sv: LD50/60 without medical treatment
        - >6.0 Sv: GI syndrome, nearly always fatal

        Immunity affects RECOVERY from ARS (opportunistic infection resistance)
        but NOT the dose absorbed. This is handled in disease progression.
        """
        # Direct dose accumulation — no biological resistance to ionizing radiation
        # Shelter/shielding reduction is applied BEFORE this method is called
        self.cumulative_radiation_sv += dose_sv

        # Update sickness level based on IAEA/UNSCEAR dose-response thresholds
        if self.cumulative_radiation_sv >= 4.0:
            self.radiation_sickness_level = 3  # Terminal (LD50/60)
        elif self.cumulative_radiation_sv >= 2.0:
            self.radiation_sickness_level = 2  # Hematopoietic syndrome
        elif self.cumulative_radiation_sv >= 1.0:
            self.radiation_sickness_level = 1  # Prodromal
        else:
            self.radiation_sickness_level = 0

    def check_radiation_death(self) -> bool:
        """Check if radiation exposure is lethal."""
        return self.cumulative_radiation_sv >= 4.0

    # === EVA SUIT MANAGEMENT ===

    def load_o2_canister(self) -> dict:
        """
        Load an O2 canister from inventory into the active suit supply.

        Each canister provides 0.36 kg O₂. For NASA's 80 kg reference
        astronaut this is 12 hours nominal or 8 hours while walking.
        """
        if not self.inventory.has_item("oxygen_canisters"):
            return {"loaded": False, "reason": "no_canisters"}

        self.inventory.items["oxygen_canisters"] -= 1
        if self.inventory.items["oxygen_canisters"] <= 0:
            del self.inventory.items["oxygen_canisters"]

        # The pressure vessel is reusable hardware, not disposable mass. A
        # swap returns the depleted/part-used cylinder to the astronaut so it
        # can be recharged at an operational oxygen plant.
        if (
            getattr(self, "_has_active_o2_canister", False)
            and self._current_canister_remaining < 99.9
        ):
            self.inventory.items["empty_oxygen_canisters"] = (
                self.inventory.items.get("empty_oxygen_canisters", 0) + 1
            )
        self._current_canister_remaining = 100.0
        self._has_active_o2_canister = True
        self.needs.o2_supply = 100.0
        self._total_canisters_loaded += 1
        self.total_o2_canisters_used += 1
        return {"loaded": True, "canisters_remaining": self.inventory.items.get("oxygen_canisters", 0)}

    def equip_suit(self, suit_effects: dict = None) -> dict:
        """
        Equip a protective suit for EVA operations.

        Must have protective_suit in inventory. Suit provides:
        - Pressurized environment (prevents vacuum death)
        - Modest incidental radiation attenuation (10%; not storm/GCR shielding)
        - Temperature stress reduction (30%)
        - O2 efficiency bonus (20% longer canister life)
        - Movement speed penalty (10% slower)
        """
        if not self.inventory.has_item("protective_suit"):
            return {"equipped": False, "reason": "no_suit_available"}

        if self.suit_equipped:
            return {"equipped": False, "reason": "already_wearing_suit"}

        self.suit_equipped = True
        # Apply suit properties (from crafted suit recipe or default)
        if suit_effects:
            self.suit_radiation_factor = suit_effects.get("radiation_reduction_factor", 0.9)
            self.suit_temp_reduction = suit_effects.get("temperature_stress_reduction", 0.3)
            self.suit_o2_efficiency = 1.0 + suit_effects.get("o2_efficiency_bonus", 0.2)
            self.suit_durability_ticks = int(
                suit_effects.get("durability_ticks", self.suit_durability_ticks)
            )
            self._suit_durability_minutes = (
                self.suit_durability_ticks * self.SIM_MINUTES_PER_TICK
            )
        # Reset wear tracking for new suit equip
        self.suit_ticks_used = 0
        self.suit_integrity = 1.0
        self.suit_condition = 1.0

        # Auto-load first O2 canister if available
        if self._current_canister_remaining <= 0:
            self.load_o2_canister()

        return {"equipped": True, "integrity": self.suit_integrity}

    def remove_suit(self) -> dict:
        """
        Remove EVA suit. Only safe inside a pressurized habitat.
        On atmosphereless planets outside habitat: removing suit = death.
        """
        if not self.suit_equipped:
            return {"removed": False, "reason": "no_suit_worn"}

        self.suit_equipped = False
        return {"removed": True, "suit_integrity": round(self.suit_integrity, 2),
                "ticks_used": self.suit_ticks_used}

    def enter_habitat(self) -> dict:
        """
        Enter a pressurized habitat structure.
        Stops EVA O2 consumption and allows suit removal.

        Entering a pressure vessel is not itself an oxygen source.  Blood/loop
        oxygen recovery is handled by the simulation life-support pass and is
        therefore conditional on the habitat actually having an O2 reserve.
        """
        self._in_habitat = True
        self._eva_ticks_continuous = 0
        self._vacuum_exposure_ticks = 0
        return {"entered": True}

    def exit_habitat(self, has_atmosphere: bool = True,
                     minimum_o2_reserve: float = 35.0,
                     minimum_suit_integrity: float = 0.35,
                     minimum_suit_condition: float = 0.35) -> dict:
        """
        Exit habitat to outdoor environment.
        On atmosphereless planets, requires suit + loaded O2 canister.
        """
        if not has_atmosphere:
            if not self.suit_equipped:
                return {"exited": False, "reason": "suit_required",
                        "warning": "Exiting habitat without suit on airless planet = death"}
            if self.suit_integrity < minimum_suit_integrity:
                return {
                    "exited": False,
                    "reason": "suit_integrity_unsafe",
                    "warning": (
                        f"EVA denied: suit integrity {self.suit_integrity:.0%} is "
                        f"below the {minimum_suit_integrity:.0%} pressure-safety limit"
                    ),
                }
            if self.suit_condition < minimum_suit_condition:
                return {
                    "exited": False,
                    "reason": "suit_service_due",
                    "warning": (
                        f"EVA denied: suit service condition {self.suit_condition:.0%} "
                        f"is below the {minimum_suit_condition:.0%} limit"
                    ),
                }
            if (
                self._current_canister_remaining < minimum_o2_reserve
                and not self.inventory.has_item("oxygen_canisters")
            ):
                return {"exited": False, "reason": "no_o2_supply",
                        "warning": (
                            f"EVA denied: suit O2 {self._current_canister_remaining:.1f}% "
                            f"is below {minimum_o2_reserve:.1f}% reserve and no spare is available"
                        )}
            # Replace a low canister before opening the outer airlock. The
            # habitat's breathable O2 and suit supply are separate reservoirs.
            if self._current_canister_remaining < minimum_o2_reserve:
                self.load_o2_canister()

        self._in_habitat = False
        if not has_atmosphere:
            self.needs.o2_supply = max(0.0, self._current_canister_remaining)
        return {"exited": True, "suit_equipped": self.suit_equipped,
                "o2_remaining": round(self._current_canister_remaining, 1)}

    def repair_suit(self, repair_amount: float = 0.3) -> dict:
        """
        Repair suit integrity using materials.
        Requires: insulated_fabric (2) + reduced_iron_ingot (1)
        """
        if not self.suit_equipped:
            return {"repaired": False, "reason": "no_suit_to_repair"}

        fabric = self.inventory.materials.get("insulated_fabric", 0)
        metal = self.inventory.materials.get("reduced_iron_ingot", 0)
        if fabric < 2 or metal < 1:
            return {"repaired": False, "reason": "insufficient_materials",
                    "need": {"insulated_fabric": max(0, 2 - fabric),
                             "reduced_iron_ingot": max(0, 1 - metal)}}

        self.inventory.materials["insulated_fabric"] -= 2
        self.inventory.materials["reduced_iron_ingot"] -= 1
        old_integrity = self.suit_integrity
        self.suit_integrity = min(1.0, self.suit_integrity + repair_amount)
        self.suit_condition = min(1.0, self.suit_condition + repair_amount)
        return {"repaired": True,
                "integrity_before": round(old_integrity, 2),
                "integrity_after": round(self.suit_integrity, 2)}

    def apply_seal_patch(self) -> dict:
        """
        Apply an emergency pressure seal patch to an EVA suit micro-puncture.
        Uses 1x vacuum_gasket_seal or emergency patch.
        """
        if not self.has_micro_puncture:
            return {"patched": False, "reason": "no_puncture"}
        
        has_seal = self.inventory.materials.get("vacuum_gasket_seal", 0) > 0 or self.inventory.has_item("vacuum_gasket_seal")
        if not has_seal:
            return {"patched": False, "reason": "missing_vacuum_gasket_seal"}
        if has_seal:
            if self.inventory.materials.get("vacuum_gasket_seal", 0) > 0:
                self.inventory.materials["vacuum_gasket_seal"] -= 1
            else:
                self.inventory.remove_item("vacuum_gasket_seal", 1)
        self.has_micro_puncture = False
        self.suit_integrity = min(1.0, self.suit_integrity + 0.15)
        return {"patched": True, "suit_integrity": round(self.suit_integrity, 2)}

    def recharge_plss(self):
        """Regenerate PLSS sorbent and battery on the habitat service rack."""
        scale = self.SIM_MINUTES_PER_TICK / 5.0
        self.plss_co2_scrubber_pct = min(
            100.0, self.plss_co2_scrubber_pct + 8.0 * scale
        )
        self.plss_suit_battery_pct = min(
            100.0, self.plss_suit_battery_pct + 6.0 * scale
        )
        self.hypercapnia_level = max(
            0.0, self.hypercapnia_level - 0.25 * scale
        )

    def get_eva_status(self) -> dict:
        """Get current EVA suit and O2 status for LLM context."""
        if not self.suit_equipped:
            return {"suit": "none", "eva_safe": self._in_habitat}

        return {
            "suit": "equipped",
            "integrity": f"{self.suit_integrity:.0%}",
            "service_condition": f"{self.suit_condition:.0%}",
            "o2_remaining": round(self._current_canister_remaining, 1),
            "canisters_in_inventory": self.inventory.items.get("oxygen_canisters", 0),
            "eva_hours": round(
                self._eva_ticks_continuous * self.SIM_MINUTES_PER_TICK / 60,
                1,
            ),
            "max_eva_hours_at_rest": round(
                self._current_canister_remaining
                / max(0.001, self.plss_o2_percent_per_tick(1.0))
                * self.SIM_MINUTES_PER_TICK / 60,
                1,
            ),
            "in_habitat": self._in_habitat,
        }

    def tick_update(self, ambient_temp_c: float, gravity_multiplier: float = 1.0,
                    has_shelter: bool = False, shelter_effects: dict = None,
                    has_atmosphere: bool = True,
                    wind_speed_modifier: float = 1.0,
                    biome_hazards: dict = None,
                    pressure_kpa: float = 101.3,
                    nearby_agent_count: int = 0,
                    sleep_quality: float = 1.0) -> dict:
        """
        Per-tick physiological update for the agent.
        Called every simulation tick by the engine.

        This is the central integration point — connects the metabolic model,
        disease system, radiation tracking, pressure effects, injury healing,
        isolation mechanics, and action state.

        Args:
            ambient_temp_c: Current temperature at agent's position
            gravity_multiplier: Planet gravity (e.g., 1.3 for Kepler-442b)
            has_shelter: Whether agent is inside a shelter structure
            shelter_effects: Dict with shelter properties
            has_atmosphere: Whether planet has breathable atmosphere
            wind_speed_modifier: Wind speed at position (1.0=calm)
            biome_hazards: Hazard modifiers from current biome
            pressure_kpa: Atmospheric pressure at position
            nearby_agent_count: Number of agents within social range
            sleep_quality: Sleep quality multiplier (0.5=outdoors, 1.0=habitat)

        Returns:
            Dict of events: warnings, death, action completion, etc.
        """
        if self.status == AgentStatus.DEAD or getattr(self.status, 'value', str(self.status)) == 'dead':
            self.status = AgentStatus.DEAD
            return {"status": "dead"}

        self.ticks_alive += 1
        physical_tick_scale = self.SIM_MINUTES_PER_TICK / 5.0
        if biome_hazards is None:
            biome_hazards = {}

        events = {
            "warnings": [],
            "action_completed": False,
            "died": False,
            "diseases_onset": [],
            "diseases_recovered": [],
            "radiation_sickness_changed": False,
            "near_death": False,
        }

        # === DETERMINE ACTIVITY LEVEL (MET) ===
        activity_multiplier = self.get_activity_multiplier()
        
        # === SLEEP INTERRUPTION / WAKING UP ===
        if self.action.action_type == "sleep":
            sleep_target = (
                self.action.target if isinstance(self.action.target, dict) else {}
            )
            dry_base_conservation_sleep = bool(
                sleep_target.get("dry_base_conservation_sleep", False)
            )
            # Hunger and thirst wake a sleeper before clinical syncope. A
            # ten-minute heavy-metabolism step can cross from a low reserve to
            # zero before the next decision, so use the same 45% proactive
            # self-care band as the work scheduler. A physically dry base is
            # the exception: waking cannot
            # create water, so sleep conserves energy until the processor has
            # accumulated a meaningful potable dose.
            if (
                self.needs.temperature_stress < 20.0
                or self.needs.hunger <= 45.0
                or (
                    self.needs.thirst <= 45.0
                    and not dry_base_conservation_sleep
                )
            ):
                self.action.clear()  # Waking up
                events["warnings"].append("sleep_interrupted_by_survival_threat")
            # Wake up if fully rested (energy >= 95.0) and slept at least 12 ticks
            elif (
                self.needs.energy >= 95.0
                and self.needs._consecutive_sleep_ticks
                >= self._legacy_duration_ticks(12)
            ):
                self.action.clear()  # Waking up fully rested
                
        # Medical convalescence uses the same restorative physiology as sleep,
        # but is not ended early by the normal "fully rested" wake-up rule.
        is_sleeping = self.action.action_type in ("sleep", "medical_rest")

        # EVA sleep restriction: sleeping in a pressurized EVA suit is possible
        # but extremely poor quality. The suit is rigid, pressurized at ~30 kPa,
        # and uncomfortable — deep NREM/REM sleep is nearly impossible.
        # Apollo crews slept in the LM, never in EVA suits on the surface.
        # We allow it (emergency/first day scenario) but at 15% effectiveness.
        if is_sleeping and self.suit_equipped and not self._in_habitat:
            sleep_quality = 0.15  # Severely degraded — barely a nap
            events["warnings"].append("poor_sleep_in_EVA_suit")

        # === VACUUM / UNBREATHABLE ATMOSPHERE LIFE SUPPORT CHECK ===
        # NASA SP-368 "Bioastronautics Data Book":
        # - Without pressurized suit in vacuum (<6.3 kPa = Armstrong limit):
        #   * 0-15 sec: Useful consciousness time
        #   * 15-30 sec: Loss of consciousness (cerebral anoxia)
        #   * 60-90 sec: Ebullism (body fluids boil), death
        # - 1 tick = 5 sim minutes = 300 seconds. Vacuum kills in <2 ticks.
        ambient_o2_fraction = float(biome_hazards.get("o2_fraction", 0.21))
        ambient_po2_kpa = max(0.0, pressure_kpa * ambient_o2_fraction)
        requires_plss = (
            not self._in_habitat
            and (not has_atmosphere or ambient_po2_kpa < 16.0)
        )
        is_vacuum = (not has_atmosphere) or pressure_kpa < 6.3

        if requires_plss:
            if not self.suit_equipped or self.suit_integrity <= 0:
                # UNPROTECTED VACUUM OR HYPOXIC/ANOXIC EXPOSURE
                self._vacuum_exposure_ticks += 1
                if not is_vacuum:
                    events["warnings"].append("unbreathable_atmosphere_unprotected")
                    self.needs.o2_supply = max(0.0, self.needs.o2_supply - 40.0)
                    self.needs.energy = max(0.0, self.needs.energy - 15.0)
                    if (
                        self._vacuum_exposure_ticks
                        * self.SIM_MINUTES_PER_TICK >= 10.0
                    ):
                        self.status = AgentStatus.INCAPACITATED
                        events["warnings"].append("CRITICAL: hypoxic loss of consciousness")
                    else:
                        events["warnings"].append("CRITICAL: insufficient oxygen partial pressure")
                else:
                    events["warnings"].append("vacuum_exposure_unprotected")

                if is_vacuum and self._vacuum_exposure_ticks >= 1:
                    # Ebullism + anoxia = death
                    # Real loss of survivability is much shorter than one
                    # mission tick, so the first completed unprotected tick
                    # is already fatal at either supported timebase.
                    self.status = AgentStatus.DEAD
                    self.death_cause = DeathCause.VACUUM_EXPOSURE
                    self.death_tick = self.ticks_alive
                    events["died"] = True
                    events["death_cause"] = "vacuum_exposure"
                    events["warnings"].append("DEATH: vacuum exposure — ebullism and anoxia")
                    return events
                elif is_vacuum:
                    # First tick: rapid consciousness loss, extreme stress
                    self.needs.energy = max(0.0, self.needs.energy - 30.0)
                    self.needs.o2_supply = max(0.0, self.needs.o2_supply - 50.0)
                    self.injury_level = min(1.0, self.injury_level + 0.5)
                    events["warnings"].append("CRITICAL: vacuum exposure without suit — consciousness fading")
            else:
                # Suited EVA — suit provides pressurized micro-environment
                self._vacuum_exposure_ticks = 0
                self._eva_ticks_continuous += 1
                self.suit_ticks_used += 1

                # Normal joint cycling, dust and abrasion reduce reversible
                # service condition. They do not silently create a pressure
                # leak; structural integrity changes only through damage.
                if self.suit_durability_ticks > 0:
                    wear_rate = 1.0 / self.suit_durability_ticks
                    wear_rate *= activity_multiplier
                    self.suit_condition = max(0.0, self.suit_condition - wear_rate)

                # Regenerable PLSS solid-amine sorbent capacity: the active
                # bed is depleted during EVA and restored only at the habitat
                # service rack, rather than by an imaginary fresh cartridge.
                tick_scale = self.SIM_MINUTES_PER_TICK / 5.0
                co2_drain = 0.35 * activity_multiplier * tick_scale
                self.plss_co2_scrubber_pct = max(0.0, self.plss_co2_scrubber_pct - co2_drain)
                if self.plss_co2_scrubber_pct <= 10.0:
                    events["warnings"].append("plss_co2_scrubber_critical")
                if self.plss_co2_scrubber_pct <= 0.0:
                    # Hypercapnia: CO2 buildup causes acute acidosis & suffocation
                    self.hypercapnia_level = min(
                        1.0, self.hypercapnia_level + 0.12 * tick_scale
                    )
                    events["warnings"].append("plss_hypercapnia_toxic_co2_onset")
                    if self.hypercapnia_level >= 1.0:
                        self.status = AgentStatus.DEAD
                        self.death_cause = DeathCause.HYPERCAPNIA
                        self.death_tick = self.ticks_alive
                        events["died"] = True
                        events["death_cause"] = "hypercapnia"
                        events["warnings"].append("DEATH: hypercapnia — lethal CO2 poisoning in suit")
                        return events
                else:
                    self.hypercapnia_level = max(
                        0.0, self.hypercapnia_level - 0.05 * tick_scale
                    )

                # 2. Suit Battery (4 kWh heating & life support electronics)
                battery_drain = (
                    0.25 if ambient_temp_c >= 0 else 0.40
                ) * tick_scale
                self.plss_suit_battery_pct = max(0.0, self.plss_suit_battery_pct - battery_drain)
                if self.plss_suit_battery_pct <= 10.0:
                    events["warnings"].append("plss_suit_battery_critical")

                # 3. O₂ canister consumption — individualized by body mass
                # and current activity against NASA's 80 kg reference load.
                o2_per_tick = self.plss_o2_percent_per_tick(activity_multiplier)

                # Micro-puncture leak check
                if self.has_micro_puncture:
                    o2_per_tick *= 2.5
                    self.suit_integrity = max(
                        0.0, self.suit_integrity - 0.015 * tick_scale
                    )
                    events["warnings"].append("suit_micro_puncture_active_leak")

                if self.suit_integrity <= 0.1:
                    events["warnings"].append("suit_integrity_critical")
                elif self.suit_integrity <= 0.3:
                    events["warnings"].append("suit_integrity_low")
                if self.suit_condition <= 0.25:
                    events["warnings"].append("suit_service_overdue")

                self._current_canister_remaining -= o2_per_tick
                self.needs.o2_supply = max(0.0, self._current_canister_remaining)

                if self._current_canister_remaining <= 0:
                    # Try to load next canister
                    if self.inventory.has_item("oxygen_canisters"):
                        self.load_o2_canister()
                        events["warnings"].append("o2_canister_auto_loaded")
                    else:
                        events["warnings"].append("o2_depleted_no_canisters")
                        # Without O2, suffocation begins (handled by needs.decay)

                # EVA fatigue: prolonged EVA is physically exhausting
                # NASA ISS EVA average: 6-7 hours max. Longer = dangerous fatigue.
                if (
                    self._eva_ticks_continuous * self.SIM_MINUTES_PER_TICK
                    > 7 * 60
                ):
                    events["warnings"].append("eva_time_excessive")
                    self.needs.energy = max(
                        0.0, self.needs.energy - 0.5 * tick_scale
                    )
        else:
            # Inside habitat or on atmospheric planet
            self._vacuum_exposure_ticks = 0
            self._eva_ticks_continuous = 0
            if getattr(self, '_in_habitat', False):
                self.recharge_plss()

        # === APPLY RADIATION (before needs decay — dose affects metabolism) ===
        radiation_dose = biome_hazards.get("radiation_per_tick", 0.0)
        if radiation_dose > 0:
            # Suit provides partial radiation shielding
            if self.suit_equipped and self.suit_integrity > 0.3:
                radiation_dose *= self.suit_radiation_factor
            # Shelter provides heavy shielding (regolith/basalt walls)
            if has_shelter:
                shelter_rad_factor = (shelter_effects or {}).get("radiation_reduction", 0.05)
                radiation_dose *= shelter_rad_factor

            old_level = self.radiation_sickness_level
            self.apply_radiation(radiation_dose)
            if self.radiation_sickness_level != old_level:
                events["radiation_sickness_changed"] = True
                events["warnings"].append(
                    f"radiation_level_{self.radiation_sickness_level}"
                )
            # Apply permanent stat damage at Level 2 (one-time)
            if (self.radiation_sickness_level >= 2
                    and not self._radiation_penalties_applied):
                self._radiation_permanent_endurance_penalty = 2
                self._radiation_permanent_immunity_penalty = 2
                self._radiation_penalties_applied = True
                events["warnings"].append("radiation_permanent_damage")

        # === EFFECTIVE GENOME (after radiation penalties) ===
        effective_endurance = max(
            1, self.genome.endurance - self._radiation_permanent_endurance_penalty
        )
        effective_immunity = max(
            1, self.genome.immunity - self._radiation_permanent_immunity_penalty
        )

        # === SHELTER EFFECTS ===
        shelter_temp_red = 0.0
        if shelter_effects:
            shelter_temp_red = shelter_effects.get("temperature_stress_reduction", 0.0)

        # === NEED DECAY (fully wired) ===
        # Set suit thermal protection for temperature stress calc
        self.needs._suit_temp_reduction_active = (
            self.suit_temp_reduction if self.suit_equipped and self.suit_integrity > 0.3 else 0.0
        )
        is_canister_active = requires_plss and self.suit_equipped
        warnings = self.needs.decay(
            endurance=effective_endurance,
            gravity_multiplier=gravity_multiplier,
            activity_multiplier=activity_multiplier,
            ambient_temp_c=ambient_temp_c,
            wind_speed_modifier=wind_speed_modifier,
            has_shelter=has_shelter,
            shelter_temp_reduction=shelter_temp_red,
            has_atmosphere=has_atmosphere,
            is_sleeping=is_sleeping,
            sleep_quality=sleep_quality,
            morale=self.morale,
            active_diseases=self.active_diseases,
            is_canister_active=is_canister_active,
            water_requirement_l_per_tick=self.water_requirement_l_per_tick(
                is_eva=requires_plss,
                activity_multiplier=activity_multiplier,
            ),
        )
        events["warnings"].extend(warnings)

        # Track near-death events
        if any(w.endswith("_critical") for w in warnings):
            self.near_death_events += 1
            events["near_death"] = True

        # === ATMOSPHERIC PRESSURE EFFECTS (PO2-based hypoxia) ===
        # Hackett & Roach, NEJM 2001 — hypoxia depends on O2 partial pressure:
        # PO2 = total_pressure * O2_fraction
        # >16 kPa: normal | 13-16: mild AMS | 10-13: moderate | <10: severe | <6: lethal
        # Bypass: habitat life support OR pressurized EVA suit with O2 supply
        _in_habitat_or_suit = (
            getattr(self, "_in_habitat", False) or
            (self.suit_equipped and self.suit_integrity > 0.1 and self.needs.o2_supply > 0)
        )
        if has_atmosphere and pressure_kpa > 0 and not _in_habitat_or_suit:
            # Get O2 fraction from biome hazards (set by engine from planet config)
            o2_fraction = biome_hazards.get("o2_fraction", 0.21)
            po2_kpa = pressure_kpa * o2_fraction

            if po2_kpa < 6.0:
                # Lethal hypoxia — loss of consciousness within minutes
                self.needs.energy = max(0.0, self.needs.energy - 2.0)
                events["warnings"].append("lethal_hypoxia")
            elif po2_kpa < 10.0:
                # Severe hypoxia (>5500m equivalent)
                severity = (10.0 - po2_kpa) / 4.0  # 0..1
                self.needs.energy = max(0.0, self.needs.energy - 0.3 * severity)
                events["warnings"].append("severe_hypoxia")
            elif po2_kpa < 16.0:
                # Mild-moderate hypoxia (2400-5500m equivalent)
                severity = (16.0 - po2_kpa) / 6.0  # 0..1
                self.needs.energy = max(0.0, self.needs.energy - 0.08 * severity)


        # === OVERLOAD ENERGY PENALTY ===
        # Carrying more than capacity burns extra calories
        overload = self.operational_load_fraction(gravity_multiplier)
        if overload > 1.0 and not is_sleeping:
            overload_energy_drain = (overload - 1.0) * 0.15
            self.needs.energy = max(0.0, self.needs.energy - overload_energy_drain)
            self.needs.hunger = max(
                0.0,
                self.needs.hunger
                - (overload - 1.0) * 0.1 * physical_tick_scale,
            )

        # === PASSIVE INJURY HEALING ===
        # Guo & DiPietro (2010), J Dental Research — wound healing phases:
        # Light injury (0.1-0.3): 3-7 days | Moderate (0.3-0.7): 2-6 weeks
        # Severe (>0.7): 6-12 weeks. Healing slows for severe injuries.
        if self.injury_level > 0:
            # Base healing rate
            if is_sleeping:
                heal_rate = 0.002   # legacy five-minute physical rate
            elif self.action.action_type in (None, "idle", "talk", "eat", "drink"):
                heal_rate = 0.0008  # legacy five-minute physical rate
            else:
                heal_rate = 0.0  # No healing during physical activity
            heal_rate *= physical_tick_scale
            # Severe injuries (>0.7) heal 50% slower (complex tissue repair)
            if self.injury_level > 0.7:
                heal_rate *= 0.5
            # Immunity bonus: better immune response speeds healing
            heal_rate *= 1.0 + 0.03 * (self.genome.immunity - 5)
            self.injury_level = max(0.0, self.injury_level - heal_rate)

        # === ISOLATION MORALE DECAY ===
        if nearby_agent_count > 0:
            self._ticks_without_social = 0
        else:
            self._ticks_without_social = getattr(self, '_ticks_without_social', 0) + 1
            if self._ticks_without_social > 50:
                # Isolation stress — morale decays
                self.apply_social_morale("isolation_tick")

        # === DISEASE TICK (typed disease system) ===
        recovered = []
        for disease in self.active_diseases[:]:
            disease.ticks_remaining -= 1
            if disease.ticks_remaining <= 0:
                recovered.append(disease)
                self.active_diseases.remove(disease)
        if recovered:
            events["diseases_recovered"] = [
                d.disease_type.value for d in recovered
            ]

        # === DISEASE ONSET CHECKS ===
        debuffs = self.needs.get_debuffs(
            radiation_sickness_level=self.radiation_sickness_level,
            active_diseases=self.active_diseases,
        )

        # Hygiene-based infection
        has_infection = any(
            d.disease_type == DiseaseType.INFECTION for d in self.active_diseases
        )
        if debuffs["disease_risk_per_tick"] > 0 and not has_infection:
            # EVA suit isolation: sealed suit dramatically reduces pathogen exposure
            effective_disease_risk = debuffs["disease_risk_per_tick"]
            if self.suit_equipped and not self._in_habitat:
                effective_disease_risk *= 0.1  # 90% reduction in sealed suit
            effective_disease_risk = self._scaled_tick_probability(
                effective_disease_risk
            )
            if self._rng.random() < effective_disease_risk:
                immunity_save = effective_immunity / 10.0
                if self._rng.random() > immunity_save:
                    infection = Disease(
                        disease_type=DiseaseType.INFECTION,
                        ticks_remaining=self._legacy_duration_ticks(
                            Disease.EFFECTS[DiseaseType.INFECTION]["duration_ticks"]
                        ),
                        ticks_total=self._legacy_duration_ticks(
                            Disease.EFFECTS[DiseaseType.INFECTION]["duration_ticks"]
                        ),
                    )
                    self.active_diseases.append(infection)
                    self.diseases_contracted += 1
                    events["diseases_onset"].append("infection")

        # Temperature-triggered diseases
        has_hypothermic = any(
            d.disease_type == DiseaseType.HYPOTHERMIC_SHOCK for d in self.active_diseases
        )
        has_heatstroke = any(
            d.disease_type == DiseaseType.HEAT_STROKE for d in self.active_diseases
        )

        # Hypothermic shock: prolonged cold exposure (temp_stress < 15 for extended time)
        if self.needs.temperature_stress < 15 and not has_hypothermic:
            cold_chance = self._scaled_tick_probability(
                (15 - self.needs.temperature_stress) * 0.002
            )
            if self._rng.random() < cold_chance:
                shock = Disease(
                    disease_type=DiseaseType.HYPOTHERMIC_SHOCK,
                    ticks_remaining=self._legacy_duration_ticks(
                        Disease.EFFECTS[DiseaseType.HYPOTHERMIC_SHOCK]["duration_ticks"]
                    ),
                    ticks_total=self._legacy_duration_ticks(
                        Disease.EFFECTS[DiseaseType.HYPOTHERMIC_SHOCK]["duration_ticks"]
                    ),
                )
                self.active_diseases.append(shock)
                self.diseases_contracted += 1
                events["diseases_onset"].append("hypothermic_shock")

        # Heat stroke: prolonged heat (temp_stress > 85)
        if self.needs.temperature_stress > 85 and not has_heatstroke:
            heat_chance = self._scaled_tick_probability(
                (self.needs.temperature_stress - 85) * 0.003
            )
            if self._rng.random() < heat_chance:
                stroke = Disease(
                    disease_type=DiseaseType.HEAT_STROKE,
                    ticks_remaining=self._legacy_duration_ticks(
                        Disease.EFFECTS[DiseaseType.HEAT_STROKE]["duration_ticks"]
                    ),
                    ticks_total=self._legacy_duration_ticks(
                        Disease.EFFECTS[DiseaseType.HEAT_STROKE]["duration_ticks"]
                    ),
                )
                self.active_diseases.append(stroke)
                self.diseases_contracted += 1
                events["diseases_onset"].append("heat_stroke")

        # Radiation ARS (triggered by radiation level, separate from cumulative death)
        has_ars = any(
            d.disease_type == DiseaseType.RADIATION_ARS for d in self.active_diseases
        )
        if self.radiation_sickness_level >= 1 and not has_ars:
            ars = Disease(
                disease_type=DiseaseType.RADIATION_ARS,
                severity=float(self.radiation_sickness_level),
                ticks_remaining=self._legacy_duration_ticks(
                    Disease.EFFECTS[DiseaseType.RADIATION_ARS]["duration_ticks"]
                ),
                ticks_total=self._legacy_duration_ticks(
                    Disease.EFFECTS[DiseaseType.RADIATION_ARS]["duration_ticks"]
                ),
            )
            self.active_diseases.append(ars)
            self.diseases_contracted += 1
            events["diseases_onset"].append("radiation_ars")

        # Toxic exposure from volcanic biomes. The biome prior is conditional
        # on one hour of unsheltered EVA; event probabilities have already
        # been converted to this tick by the scheduler.
        toxic_chance = biome_hazards.get(
            "toxic_exposure_probability_per_eva_hour", 0
        )
        event_toxic_chance = biome_hazards.get(
            "event_toxic_exposure_probability_per_tick", 0
        )
        has_toxic = any(
            d.disease_type == DiseaseType.TOXIC_EXPOSURE for d in self.active_diseases
        )
        if (toxic_chance > 0 or event_toxic_chance > 0) and not has_toxic and not has_shelter:
            combined_toxic_chance = 1.0 - (
                1.0 - self._hourly_probability_for_tick(toxic_chance)
            ) * (1.0 - max(0.0, min(1.0, event_toxic_chance)))
            if self._rng.random() < combined_toxic_chance:
                toxic = Disease(
                    disease_type=DiseaseType.TOXIC_EXPOSURE,
                    ticks_remaining=self._legacy_duration_ticks(
                        Disease.EFFECTS[DiseaseType.TOXIC_EXPOSURE]["duration_ticks"]
                    ),
                    ticks_total=self._legacy_duration_ticks(
                        Disease.EFFECTS[DiseaseType.TOXIC_EXPOSURE]["duration_ticks"]
                    ),
                )
                self.active_diseases.append(toxic)
                self.diseases_contracted += 1
                events["diseases_onset"].append("toxic_exposure")

        # === INJURY from biome and scheduled environmental hazards ===
        # Every configured physical hazard reaches physiology; descriptive
        # biome fields are not allowed to remain UI-only warnings.
        for hazard_key in (
            "crevasse_injury_probability_per_eva_hour",
            "collapse_injury_probability_per_eva_hour",
            "rockslide_injury_probability_per_eva_hour",
            "rockfall_injury_probability_per_eva_hour",
            "ground_instability_injury_probability_per_eva_hour",
            "scalding_injury_probability_per_eva_hour",
            "flooding_injury_probability_per_eva_hour",
            "seismic_injury_probability_per_eva_hour",
            "cryovolcanic_injury_probability_per_eva_hour",
            "environmental_injury_probability_per_tick",
        ):
            prob = biome_hazards.get(hazard_key, 0)
            if (
                not has_shelter
                and
                prob > 0
                and self._rng.random() < (
                    max(0.0, min(1.0, prob))
                    if hazard_key == "environmental_injury_probability_per_tick"
                    else self._hourly_probability_for_tick(prob)
                )
            ):
                injury_amount = 0.2 + self._rng.random() * 0.3  # 0.2-0.5
                self.injury_level = min(1.0, self.injury_level + injury_amount)
                events["warnings"].append(f"injury_{hazard_key.split('_')[0]}")
                if self.injury_level >= 1.0:
                    self._die(DeathCause.INJURY, self.ticks_alive)
                    events["died"] = True
                    events["death_cause"] = DeathCause.INJURY.value
                    return events

        biome_suit_damage_chance = self._hourly_probability_for_tick(
            biome_hazards.get(
                "micrometeorite_suit_damage_probability_per_eva_hour", 0.0
            )
        )
        event_suit_damage_chance = max(0.0, min(1.0, float(
            biome_hazards.get("suit_damage_probability_per_tick", 0.0)
        )))
        suit_damage_chance = 1.0 - (
            1.0 - biome_suit_damage_chance
        ) * (1.0 - event_suit_damage_chance)
        if (
            suit_damage_chance > 0.0
            and self.suit_equipped
            and not self._in_habitat
            and self._rng.random()
            < max(0.0, min(1.0, suit_damage_chance))
        ):
            puncture_damage = 0.05 + self._rng.random() * 0.15
            self.suit_integrity = max(
                0.0, self.suit_integrity - puncture_damage
            )
            self.has_micro_puncture = True
            events["warnings"].append("suit_micrometeoroid_damage")

        if (
            biome_hazards.get("uv_lethal_unprotected", False)
            and not has_shelter
            and not (self.suit_equipped and self.suit_integrity > 0.3)
        ):
            self.injury_level = min(1.0, self.injury_level + 0.25)
            events["warnings"].append("acute_uv_exposure")

        # === CHECK DEATH CONDITIONS ===
        death_cause = self.needs.check_death()
        if death_cause:
            self._die(death_cause, self.ticks_alive)
            events["died"] = True
            events["death_cause"] = death_cause.value
            return events

        # Check radiation death
        if self.check_radiation_death():
            self._die(DeathCause.RADIATION_POISONING, self.ticks_alive)
            events["died"] = True
            events["death_cause"] = DeathCause.RADIATION_POISONING.value
            return events

        # === UPDATE STATUS (only if alive) ===
        if self.status != AgentStatus.DEAD and getattr(self.status, 'value', str(self.status)) != 'dead':
            # Check loss of consciousness / incapacitation (clinical syncope, severe hypothermic coma, anoxic blackout)
            fatigue_collapse = (
                self.needs.energy <= 0.0
                or self.needs._energy_death_timer > 0
            )
            other_unconscious_cause = (
                self.needs.temperature_stress <= 0.0 or
                self.needs._temp_death_timer > 0 or
                self.needs.o2_supply <= 0.0 or
                self.needs._o2_death_timer > 0 or
                self.needs.thirst <= 0.0 or
                self.needs._thirst_death_timer > 0 or
                self.needs.hunger <= 0.0 or
                self.needs._hunger_death_timer > 0 or
                self.injury_level >= 0.80 or
                self.radiation_sickness_level >= 3
            )

            if fatigue_collapse and self._in_habitat and not other_unconscious_cause:
                # A fatigue collapse in a pressurized habitat is monitored
                # recovery, not a fatal coma and not a SAR transport problem.
                self.status = AgentStatus.CRITICAL
                self.action.action_type = "medical_rest"
                self.action.target = {
                    "habitat": True,
                    "fatigue_recovery": True,
                    "cause": "exertional_collapse",
                    # Exertional collapse in this branch explicitly excludes
                    # anoxia, shock and other unconscious causes. The patient
                    # remains able to take measured oral fluids in the bunk.
                    "conscious": True,
                    "oral_fluids_allowed": True,
                }
                self.action.ticks_remaining = max(
                    self.action.ticks_remaining,
                    self._legacy_duration_ticks(24),
                )
                events["fatigue_collapse"] = True
            elif other_unconscious_cause or fatigue_collapse:
                self.status = AgentStatus.INCAPACITATED
                self.action.action_type = "unconscious"
                self.action.target = {
                    "cause": (
                        "exertional_collapse"
                        if fatigue_collapse and not other_unconscious_cause
                        else "critical_syncope"
                    )
                }
                events["unconscious"] = True
            elif self.action.action_type == "medical_rest" and self.action.ticks_remaining > 0:
                self.status = AgentStatus.CRITICAL
                # This branch is reachable only after all physiological
                # unconsciousness conditions above have cleared. Do not keep
                # an old coma flag forever after successful stabilization.
                if isinstance(self.action.target, dict):
                    self.action.target["conscious"] = True
                    self.action.target["oral_fluids_allowed"] = True
                    self.action.target["awaiting_resources"] = False
            # "critical" is a clinical state, not a generic warning badge.
            # Hygiene/inconvenience warnings previously made an otherwise
            # stable crew appear medically critical indefinitely.
            elif (
                self.needs.temperature_stress < 25
                or self.needs.temperature_stress > 85
                or self.needs.o2_supply < 25
                or self.needs.hunger <= 15
                or self.needs.thirst <= 15
                or self.needs.energy <= 15
                or self.injury_level >= 0.60
                or self.radiation_sickness_level >= 2
            ):
                self.status = AgentStatus.CRITICAL
            else:
                self.status = AgentStatus.ALIVE

        # === ADVANCE CURRENT ACTION (only if conscious) ===
        if self.action.is_active and self.status != AgentStatus.INCAPACITATED:
            completed = self.action.tick()
            if completed:
                events["action_completed"] = True

        return events

    def _die(self, cause: DeathCause, tick: int):
        """Process agent death."""
        self.status = AgentStatus.DEAD
        self.death_cause = cause
        self.death_tick = tick
        self.corpse_salvaged = False
        self.action.clear()
        self.action.action_type = "dead"
        self.action.target = {}

    def salvage_corpse(self, receiver) -> dict:
        """
        Transfer all items and materials from this deceased agent to a living receiver agent or depot.
        Returns details of salvaged resources.
        """
        if getattr(self.status, 'value', str(self.status)) != 'dead':
            return {"salvaged": False, "reason": "agent_not_dead"}
        
        salvaged_items = {k: v for k, v in self.inventory.items.items() if v > 0}
        salvaged_materials = {k: v for k, v in self.inventory.materials.items() if v > 0}
        
        # Transfer items
        for item, qty in salvaged_items.items():
            if hasattr(receiver, "inventory"):
                receiver.inventory.add_item(item, qty)
            elif isinstance(receiver, dict):
                receiver[item] = receiver.get(item, 0) + qty
        self.inventory.items.clear()
        
        # Transfer materials
        for mat, qty in salvaged_materials.items():
            if hasattr(receiver, "inventory"):
                receiver.inventory.add_material(mat, qty)
            elif isinstance(receiver, dict):
                receiver[mat] = receiver.get(mat, 0) + qty
        self.inventory.materials.clear()
        
        self.corpse_salvaged = True
        return {
            "salvaged": True,
            "items": salvaged_items,
            "materials": salvaged_materials,
            "from_agent": self.name
        }

    def eat(self, source: str = "emergency_rations", kcal: int = None) -> dict:
        """
        Consume food to restore hunger.

        Food quality affects morale:
        - emergency_rations: 700 kcal, morale -0.01 (monotonous diet)
        - greenhouse_produce: 500 kcal, morale +0.03 (fresh food)
        - foraged_lichen: 200 kcal, morale -0.02 (unpleasant)

        Returns consumption details for logging/memory.
        """
        food_info = self.FOOD_TYPES.get(source, {"kcal": 500, "morale": 0, "label": source})
        # EVA restriction: cannot eat with helmet sealed in vacuum
        if self.suit_equipped and not self._in_habitat:
            return {"consumed": False, "source": source, "kcal": 0,
                    "morale_effect": 0, "reason": "cannot_eat_during_EVA"}
        if kcal is None:
            kcal = food_info["kcal"]

        result = {"consumed": False, "source": source, "kcal": 0, "morale_effect": 0}

        if source == "emergency_rations" and self.inventory.has_item("emergency_rations"):
            self.inventory.items["emergency_rations"] -= 1
            if self.inventory.items["emergency_rations"] <= 0:
                del self.inventory.items["emergency_rations"]
            hunger_restore = self.hunger_points_for_kcal(kcal)
            self.needs.hunger = min(100.0, self.needs.hunger + hunger_restore)
            self.morale = max(0.1, min(1.5, self.morale + food_info["morale"]))
            result.update({"consumed": True, "kcal": kcal, "morale_effect": food_info["morale"]})
            self.total_kcal_consumed += kcal
        elif source in ("greenhouse_produce", "foraged_lichen"):
            hunger_restore = self.hunger_points_for_kcal(kcal)
            self.needs.hunger = min(100.0, self.needs.hunger + hunger_restore)
            self.morale = max(0.1, min(1.5, self.morale + food_info["morale"]))
            result.update({"consumed": True, "kcal": kcal, "morale_effect": food_info["morale"]})
            self.total_kcal_consumed += kcal

        return result

    def drink(self, source: str = "water_packs", liters: float = 1.0) -> dict:
        """
        Consume water to restore thirst.

        Water safety:
        - water_packs: Always safe (sealed, pre-packaged)
        - clean_water: Safe (purified via water_purifier)
        - raw_water: UNSAFE — 8% disease risk per drink (untreated ice melt)

        Returns consumption details.
        """
        result = {"consumed": False, "source": source, "liters": 0, "disease_risk": False}

        # EVA restriction: cannot drink with helmet sealed in vacuum
        # (NASA suits have a small straw for in-suit water bag, but no external drink)
        if self.suit_equipped and not self._in_habitat:
            # Suit internal water bag allows limited drinking (0.5L max)
            if source == "suit_water" or liters <= 0.5:
                pass  # Allowed via suit straw
            else:
                result["reason"] = "cannot_drink_during_EVA"
                return result

        if source == "water_packs":
            if self.inventory.has_item("water_packs"):
                consume_units = min(liters, self.inventory.items.get("water_packs", 0))
                self.inventory.items["water_packs"] -= int(consume_units)
                if self.inventory.items.get("water_packs", 0) <= 0:
                    if "water_packs" in self.inventory.items:
                        del self.inventory.items["water_packs"]
                thirst_restore = (consume_units / 3.0) * 100.0
                self.needs.thirst = min(100.0, self.needs.thirst + thirst_restore)
                self.total_water_consumed_l += consume_units
                result.update({"consumed": True, "liters": consume_units})
        elif source == "clean_water":
            # From water_purifier structure — safe
            thirst_restore = (liters / 3.0) * 100.0
            self.needs.thirst = min(100.0, self.needs.thirst + thirst_restore)
            self.total_water_consumed_l += liters
            result.update({"consumed": True, "liters": liters})
        elif source == "raw_water":
            # Untreated ice melt — disease risk
            thirst_restore = (liters / 3.0) * 100.0
            self.needs.thirst = min(100.0, self.needs.thirst + thirst_restore)
            self.total_water_consumed_l += liters
            result.update({"consumed": True, "liters": liters})
            # 8% infection chance per drink of untreated water
            if self._rng.random() < 0.08:
                has_infection = any(
                    d.disease_type == DiseaseType.INFECTION for d in self.active_diseases
                )
                if not has_infection:
                    infection = Disease(
                        disease_type=DiseaseType.INFECTION,
                        ticks_remaining=self._legacy_duration_ticks(
                            Disease.EFFECTS[DiseaseType.INFECTION]["duration_ticks"]
                        ),
                        ticks_total=self._legacy_duration_ticks(
                            Disease.EFFECTS[DiseaseType.INFECTION]["duration_ticks"]
                        ),
                    )
                    self.active_diseases.append(infection)
                    self.diseases_contracted += 1
                    result["disease_risk"] = True
        elif source == "water_collector":
            thirst_restore = (liters / 3.0) * 100.0
            self.needs.thirst = min(100.0, self.needs.thirst + thirst_restore)
            self.total_water_consumed_l += liters
            result.update({"consumed": True, "liters": liters})

        return result

    def refill_o2(self, canisters: int = 1) -> dict:
        """
        Refill O₂ supply from oxygen canisters.

        Each physical canister is one full 0.36 kg PLSS charge.
        """
        result = {"consumed": False, "canisters": 0}
        if self.inventory.has_item("oxygen_canisters"):
            self.inventory.items["oxygen_canisters"] -= canisters
            if self.inventory.items["oxygen_canisters"] <= 0:
                del self.inventory.items["oxygen_canisters"]
            if (
                getattr(self, "_has_active_o2_canister", False)
                and self._current_canister_remaining < 99.9
            ):
                self.inventory.items["empty_oxygen_canisters"] = (
                    self.inventory.items.get("empty_oxygen_canisters", 0) + 1
                )
            o2_restore = canisters * 100.0
            self._current_canister_remaining = min(
                100.0, self._current_canister_remaining + o2_restore
            )
            self.needs.o2_supply = self._current_canister_remaining
            self._has_active_o2_canister = True
            result["consumed"] = True
            result["canisters"] = canisters
            self.total_o2_canisters_used += canisters
        return result

    def sleep_tick(self):
        """
        Process one tick of sleep.
        Energy restoration is handled in Needs.decay() when is_sleeping=True.
        """
        pass  # Actual logic is in Needs.decay() with is_sleeping flag

    def wash(self, water_cost_liters: float = 2.0) -> dict:
        """
        Restore hygiene. Requires water.

        Hygiene model:
        - Full wash: 2L water → hygiene restored to 90
        - Quick wash: 0.5L → hygiene +20
        - Medical station: enables full hygiene (100) with 1L water
        """
        result = {"washed": False, "water_used": 0, "hygiene_restored": 0}

        # EVA restriction: cannot wash in sealed suit
        if self.suit_equipped and not self._in_habitat:
            result["reason"] = "cannot_wash_during_EVA"
            return result

        # Check water availability
        water_available = self.inventory.items.get("water_packs", 0)
        if water_available >= water_cost_liters:
            self.inventory.items["water_packs"] -= int(water_cost_liters)
            if self.inventory.items.get("water_packs", 0) <= 0:
                if "water_packs" in self.inventory.items:
                    del self.inventory.items["water_packs"]

            if water_cost_liters >= 2.0:
                old = self.needs.hygiene
                self.needs.hygiene = min(90.0, self.needs.hygiene + 50.0)
                result["hygiene_restored"] = self.needs.hygiene - old
            else:
                old = self.needs.hygiene
                self.needs.hygiene = min(90.0, self.needs.hygiene + 20.0)
                result["hygiene_restored"] = self.needs.hygiene - old

            result["washed"] = True
            result["water_used"] = water_cost_liters

        return result

    # === RESOURCE GATHERING ===

    # Tool effectiveness multipliers for gathering
    GATHER_TOOL_BONUS = {
        None: 0.3,              # Bare hands — very slow
        "stone_hammer": 0.8,    # Crude but functional
        "hand_tools": 1.0,      # Standard rate
        "multitool_kit": 1.2,   # Best efficiency
    }

    def gather(self, resource_node, gravity_multiplier: float = 1.0) -> dict:
        """
        Extract resources from a ResourceNode.

        Gather rate formula:
        amount = node.gather_rate × tool_bonus × (1 + 0.05 × engineering) / (gravity × difficulty)

        Each gather action also:
        - Consumes 1 tool durability use
        - Depletes the resource node
        - Adds materials to inventory
        - Tracks statistics

        Args:
            resource_node: ResourceNode from sparse_state
            gravity_multiplier: Planet gravity

        Returns:
            Dict with amount gathered, tool used, node status
        """
        result = {
            "gathered": False, "material": resource_node.material,
            "amount": 0, "tool_used": None, "node_depleted": False,
        }

        if resource_node.depleted:
            return result

        # Determine tool bonus
        best_tool = self.inventory.get_best_tool()
        tool_bonus = self.GATHER_TOOL_BONUS.get(best_tool, 0.3)

        # Engineering skill bonus (+5% per point)
        engineering_bonus = 1.0 + 0.05 * self.competency.engineering

        # Calculate gather amount
        amount = (resource_node.gather_rate * tool_bonus * engineering_bonus
                  / (gravity_multiplier * resource_node.gather_difficulty))
        amount = max(1, int(amount))

        # Cap at available quantity
        actual = min(amount, resource_node.quantity)

        # Execute gather
        resource_node.quantity -= actual
        if resource_node.quantity <= 0:
            resource_node.quantity = 0
            resource_node.depleted = True
            result["node_depleted"] = True

        # Add to inventory
        self.inventory.add_material(resource_node.material, actual)

        # Use tool durability
        if best_tool:
            tool_result = self.inventory.use_tool(1)
            result["tool_used"] = tool_result["tool_used"]
            if tool_result["broke"]:
                result["tool_broke"] = True

        # Track stats
        mat = resource_node.material
        self.total_resources_gathered[mat] = self.total_resources_gathered.get(mat, 0) + actual

        result["gathered"] = True
        result["amount"] = actual
        return result

    # === CRAFTING / BUILDING ===

    def can_craft(self, recipe: dict, gravity_multiplier: float = 1.0) -> dict:
        """
        Check if agent can craft a recipe. Returns feasibility details.

        Checks:
        1. Material requirements met
        2. Tool requirements met (if required)
        3. Competency requirements met
        4. Structure prerequisites met (checked externally)
        """
        result = {
            "can_craft": True, "missing_materials": {},
            "missing_tool": False, "missing_competency": [],
        }

        # Material check
        materials_needed = recipe.get("materials", {})
        for mat, qty in materials_needed.items():
            available = self.inventory.materials.get(mat, 0)
            if available < qty:
                result["can_craft"] = False
                result["missing_materials"][mat] = qty - available

        # Tool check
        if recipe.get("requires_tool", False) and not self.inventory.has_usable_tool():
            result["can_craft"] = False
            result["missing_tool"] = True

        # Competency check
        min_eng = recipe.get("min_engineering", 0)
        if self.competency.engineering < min_eng:
            result["can_craft"] = False
            result["missing_competency"].append(f"engineering {self.competency.engineering}/{min_eng}")

        min_med = recipe.get("min_medical", 0)
        if min_med > 0 and self.competency.medical < min_med:
            result["can_craft"] = False
            result["missing_competency"].append(f"medical {self.competency.medical}/{min_med}")

        min_phys = recipe.get("min_physics", 0)
        if min_phys > 0 and self.competency.physics < min_phys:
            result["can_craft"] = False
            result["missing_competency"].append(f"physics {self.competency.physics}/{min_phys}")

        return result

    def start_crafting(self, recipe: dict, gravity_modifier: float = 1.0) -> dict:
        """
        Begin crafting a recipe. Consumes materials, starts action timer.

        Duration formula:
        ticks = base_duration × gravity_mod × (1 - 0.05 × engineering)

        Returns craft start details or failure reason.
        """
        feasibility = self.can_craft(recipe, gravity_modifier)
        if not feasibility["can_craft"]:
            return {"started": False, "reason": feasibility}

        # Atmosphere restriction: some recipes impossible in vacuum
        if recipe.get("requires_atmosphere") and not getattr(self, '_has_atmosphere_context', True):
            return {"started": False, "reason": "requires_atmosphere",
                    "detail": "This structure cannot be built on an atmosphereless planet"}

        # Consume materials
        for mat, qty in recipe.get("materials", {}).items():
            self.inventory.remove_material(mat, qty)

        # Consume tool durability
        if recipe.get("requires_tool", False):
            self.inventory.use_tool(2)  # Crafting uses 2 durability

        # Calculate duration
        base_dur = recipe.get("base_duration_ticks", 10)
        eng_reduction = 1.0 - 0.05 * self.competency.engineering
        eng_reduction = max(0.5, eng_reduction)  # Cap at 50% reduction
        duration = max(1, int(base_dur * gravity_modifier * eng_reduction))

        # Set action state
        self.action.action_type = "craft" if recipe.get("output", {}).get("type") != "structure" else "build"
        self.action.target = {
            "recipe_id": recipe.get("output", {}).get("structure_id",
                         recipe.get("output", {}).get("item_id",
                         recipe.get("output", {}).get("material_id", "unknown"))),
            "output_type": recipe.get("output", {}).get("type", "item"),
        }
        self.action.ticks_remaining = duration
        self.action.ticks_elapsed = 0

        return {
            "started": True,
            "recipe": self.action.target["recipe_id"],
            "duration_ticks": duration,
            "output_type": self.action.target["output_type"],
        }

    # === MEDICAL TREATMENT ===

    def treat_injury(self, has_medical_station: bool = False) -> dict:
        """
        Treat injuries using medical supplies.

        Treatment model:
        - Requires medical_supplies (1 unit consumed)
        - Base healing: -0.15 injury level
        - Medical station: doubles effectiveness (-0.30)
        - Medical competency bonus: +0.02 per point above 5
        - Without supplies: manual treatment -0.05 (requires medical ≥ 7)

        Returns treatment details.
        """
        result = {"treated": False, "injury_reduced": 0, "supplies_used": 0}

        # EVA restriction: cannot perform medical treatment in sealed suit
        # (need to access wounds, apply bandages — impossible in vacuum)
        if self.suit_equipped and not self._in_habitat:
            result["reason"] = "cannot_treat_during_EVA"
            return result

        if self.injury_level <= 0:
            return result

        medical_bonus = max(0, self.competency.medical - 5) * 0.02

        if self.inventory.has_item("medical_supplies"):
            self.inventory.items["medical_supplies"] -= 1
            if self.inventory.items["medical_supplies"] <= 0:
                del self.inventory.items["medical_supplies"]

            base_heal = 0.15
            if has_medical_station:
                base_heal = 0.30
            heal_amount = base_heal + medical_bonus

            old_injury = self.injury_level
            self.injury_level = max(0.0, self.injury_level - heal_amount)
            result.update({
                "treated": True,
                "injury_reduced": old_injury - self.injury_level,
                "supplies_used": 1,
            })
        elif self.competency.medical >= 7:
            # Improvised treatment without supplies
            heal_amount = 0.05 + medical_bonus
            old_injury = self.injury_level
            self.injury_level = max(0.0, self.injury_level - heal_amount)
            result.update({
                "treated": True,
                "injury_reduced": old_injury - self.injury_level,
                "supplies_used": 0,
            })

        return result

    def to_state_summary(self, current_tick: int = 0,
                         biome_name: str = "",
                         nearby_agents: list = None,
                         nearby_structures: list = None,
                         colony_readiness_pct: float = 0.0,
                         wind_info: dict = None,
                         light_info: dict = None) -> str:
        """
        Generate compact state summary for LLM context.

        This is the multi-line summary injected into the agent's decision
        prompt (design doc §4.5). Includes:
        - Physiological status (needs, diseases, injuries)
        - Environment (biome, wind, light, temperature stress, hazards)
        - Social context (nearby agents, trust levels)
        - Resources (inventory, weight, capacity, tool durability)
        - Mission progress (colony readiness)

        Token budget: ~200-300 tokens. Compact but complete.
        """
        if nearby_agents is None:
            nearby_agents = []
        if nearby_structures is None:
            nearby_structures = []

        parts = [f"Tick {current_tick}."]

        # Biome + environment context
        if biome_name:
            env_parts = [f"Location: {biome_name}"]
            if light_info:
                env_parts.append(f"{light_info.get('phase_name', 'unknown')} (light: {light_info.get('light_level', 1.0):.1f})")
            if wind_info and wind_info.get('speed_kmh', 0) > 5:
                env_parts.append(f"wind {wind_info['speed_kmh']:.0f}km/h")
            parts.append(f"{', '.join(env_parts)}.")

        # Needs (only notable ones)
        needs_str = self.needs.to_compact_string()
        if needs_str != "all needs stable":
            parts.append(f"Physiological: {needs_str}.")
        else:
            parts.append("All needs stable.")

        # Active diseases
        if self.active_diseases:
            disease_names = [d.disease_type.value for d in self.active_diseases]
            parts.append(f"AFFLICTED: {', '.join(disease_names)}.")

        # Injury
        if self.injury_level > 0.1:
            severity = "minor" if self.injury_level < 0.3 else (
                "moderate" if self.injury_level < 0.6 else "severe"
            )
            parts.append(f"Injured ({severity}, {self.injury_level:.0%}).")

        # Radiation (if relevant)
        if self.cumulative_radiation_sv > 0.1:
            level_desc = ["", "mild ARS", "chronic damage", "TERMINAL"][
                min(3, self.radiation_sickness_level)
            ]
            parts.append(
                f"Radiation: {self.cumulative_radiation_sv:.2f} Sv"
                f" ({level_desc}, lethal at 4.0)."
            )
            if self._radiation_penalties_applied:
                parts.append("Permanent: endurance -2, immunity -2 from radiation.")

        # EVA suit status (critical on atmosphereless planets)
        if self.suit_equipped:
            suit_status = f"EVA Suit: {self.suit_integrity:.0%} integrity"
            o2_hours = round(
                self._current_canister_remaining
                / max(0.001, self.plss_o2_percent_per_tick(1.0))
                * self.SIM_MINUTES_PER_TICK / 60,
                1,
            )
            canisters = self.inventory.items.get('oxygen_canisters', 0)
            suit_status += f", O2: {o2_hours}h remaining, {canisters} canisters spare"
            if (
                self._eva_ticks_continuous * self.SIM_MINUTES_PER_TICK
                > 6 * 60
            ):
                suit_status += " [LONG EVA - return to habitat soon]"
            parts.append(suit_status + ".")
        elif not getattr(self, '_in_habitat', True):
            parts.append("WARNING: No EVA suit equipped outside habitat!")

        # Morale
        if self.morale < 0.7:
            parts.append(f"Morale LOW ({self.morale:.0%}).")
        elif self.morale > 1.1:
            parts.append(f"Morale HIGH ({self.morale:.0%}).")

        # Competency highlight
        primary = self.competency.get_primary_domain()
        score = getattr(self.competency, primary)
        parts.append(f"Primary: {primary} ({score}/10).")

        # Inventory with weight + tool status
        inv_str = self.inventory.to_compact_string()
        weight = self.inventory.total_weight_kg()
        capacity = self.inventory.carry_capacity_kg(self.genome.strength)
        tool_status = self.inventory.get_tool_status()
        parts.append(f"Carrying ({weight:.1f}/{capacity:.0f} kg): {inv_str}.")
        if tool_status != "no tools":
            parts.append(f"Tools: {tool_status}.")
        else:
            parts.append("WARNING: No usable tools!")

        # Current action
        if self.action.is_active:
            parts.append(
                f"Doing: {self.action.action_type}"
                f" ({self.action.ticks_remaining} ticks left)."
            )
        else:
            parts.append("Idle \u2014 needs new task.")

        # Nearby agents with trust context
        if nearby_agents:
            agent_strs = []
            for a in nearby_agents[:5]:  # Cap at 5 for token budget
                dist = a.get("distance", "?")
                name = a.get("name", "unknown")
                status = a.get("status", "alive")
                trust = self.trust_scores.get(a.get("id", ""), 0.0)
                trust_label = "trusted" if trust > 0.3 else (
                    "distrusted" if trust < -0.3 else "neutral"
                )
                agent_strs.append(
                    f"{name} ({dist}m, {status}, {trust_label})"
                )
            parts.append(f"Nearby: {'; '.join(agent_strs)}.")

        # Nearby structures with condition
        if nearby_structures:
            struct_strs = []
            for s in nearby_structures[:5]:
                s_type = s.get('type', 'unknown')
                s_dist = s.get('distance', '?')
                s_condition = s.get('condition', 1.0)
                if s_condition < 0.5:
                    struct_strs.append(f"{s_type} ({s_dist}m, DAMAGED {s_condition:.0%})")
                elif s.get('maintenance_overdue', False):
                    struct_strs.append(f"{s_type} ({s_dist}m, needs maintenance)")
                else:
                    struct_strs.append(f"{s_type} ({s_dist}m)")
            parts.append(f"Structures: {', '.join(struct_strs)}.")

        # Colony readiness
        if colony_readiness_pct > 0:
            parts.append(f"Colony readiness: {colony_readiness_pct:.0f}%.")

        return " ".join(parts)

    def explore_surroundings(self, cx: int, cy: int, radius: int = 7):
        if not hasattr(self, "explored_cells") or self.explored_cells is None:
            self.explored_cells = set()
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                self.explored_cells.add((cx + dx, cy + dy))

    def to_dict(self) -> dict:
        """Full state serialization for persistence/API."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "position": {"x": self.x, "y": self.y},
            "indoor_location": getattr(self, "indoor_location", None) if self._in_habitat else None,
            "anthropometrics": self.anthropometrics.to_dict(),
            "life_support_profile": self.get_life_support_profile(),
            "genome": self.genome.to_dict(),
            "competency": self.competency.to_dict(),
            "needs": self.needs.to_dict(),
            "inventory": self.inventory.to_dict(),
            "action": self.action.to_dict(),
            "radiation_sv": round(self.cumulative_radiation_sv, 3),
            "radiation_sickness": self.radiation_sickness_level,
            "radiation_permanent_penalties": {
                "endurance": self._radiation_permanent_endurance_penalty,
                "immunity": self._radiation_permanent_immunity_penalty,
            },
            "active_diseases": [
                {
                    "type": d.disease_type.value,
                    "severity": d.severity,
                    "ticks_remaining": d.ticks_remaining,
                    "ticks_total": d.ticks_total,
                }
                for d in self.active_diseases
            ],
            "injury_level": round(self.injury_level, 2),
            "morale": round(self.morale, 2),
            "strategic_goal": self.strategic_goal,
            "trust_scores": self.trust_scores,
            "ticks_alive": self.ticks_alive,
            "death_cause": self.death_cause.value if self.death_cause else None,
            "plss": {
                "co2_scrubber_pct": round(getattr(self, "plss_co2_scrubber_pct", 100.0), 1),
                "suit_battery_pct": round(getattr(self, "plss_suit_battery_pct", 100.0), 1),
                "suit_integrity_pct": round(getattr(self, "suit_integrity", 1.0) * 100.0, 1),
                "suit_condition_pct": round(getattr(self, "suit_condition", 1.0) * 100.0, 1),
                "has_micro_puncture": getattr(self, "has_micro_puncture", False),
                "hypercapnia_level": round(getattr(self, "hypercapnia_level", 0.0), 2),
            },
            "expedition": (
                dict(self._active_expedition)
                if isinstance(getattr(self, "_active_expedition", None), dict)
                else None
            ),
            "statistics": {
                "total_resources_gathered": self.total_resources_gathered,
                "total_structures_built": self.total_structures_built,
                "total_llm_calls": self.total_llm_calls,
                "total_distance_traveled": round(self.total_distance_traveled, 1),
                "total_kcal_consumed": self.total_kcal_consumed,
                "total_water_consumed_l": round(self.total_water_consumed_l, 1),
                "total_o2_canisters_used": self.total_o2_canisters_used,
                "diseases_contracted": self.diseases_contracted,
                "near_death_events": self.near_death_events,
                "total_social_interactions": self.total_social_interactions,
            },
        }

    def is_in_sleep_lockout(self, has_emergency: bool = False) -> bool:
        """
        Check if agent is in the first 40 simulated minutes of sleep.
        Prevents premature wake-ups from minor need fluctuations unless a life hazard occurs.
        """
        if getattr(self.action, 'action_type', '') == 'sleep' and not has_emergency:
            return (
                getattr(self.needs, '_consecutive_sleep_ticks', 0)
                * self.SIM_MINUTES_PER_TICK < 40.0
            )
        return False

    def to_telemetry_dict(self) -> dict:
        """
        Safe JSON-serializable telemetry dictionary with all Enums and NumPy types converted.
        Used for real-time WebSocket broadcasting and Simulation Analytics.
        """
        d = self.to_dict()
        d["rl_reward_accumulated"] = round(getattr(self, "total_accumulated_reward", 0.0), 2)
        d["q_table_states"] = sum(
            1 for key in getattr(self, "q_table", {})
            if key != TACTICAL_POLICY_META_KEY
        )
        d["sleep_debt"] = round(getattr(self.needs, "_sleep_debt", 0.0), 2)
        d["consecutive_sleep_ticks"] = getattr(self.needs, "_consecutive_sleep_ticks", 0)
        d["plss"] = {
            "o2_canister_pct": round(max(0.0, getattr(self, "_current_canister_remaining", 0.0)), 1),
            "co2_scrubber_pct": round(getattr(self, "plss_co2_scrubber_pct", 100.0), 1),
            "suit_battery_pct": round(getattr(self, "plss_suit_battery_pct", 100.0), 1),
            "suit_integrity_pct": round(getattr(self, "suit_integrity", 1.0) * 100.0, 1),
            "suit_condition_pct": round(getattr(self, "suit_condition", 1.0) * 100.0, 1),
            "has_micro_puncture": getattr(self, "has_micro_puncture", False),
            "hypercapnia_level": round(getattr(self, "hypercapnia_level", 0.0), 2),
        }
        return d


def create_team_from_presets(presets_path: str = "config/agent_presets.json") -> list[Agent]:
    """Load all agents from the presets configuration file."""
    with open(presets_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    agents = []
    for i, agent_config in enumerate(data["agents"]):
        agent = Agent(agent_config, agent_index=i)

        # Set initial needs from config
        initial = data.get("starting_conditions", {}).get("initial_needs", {})
        if initial:
            agent.needs.hunger = initial.get("hunger", 85)
            agent.needs.thirst = initial.get("thirst", 85)
            agent.needs.energy = initial.get("energy", 70)
            agent.needs.hygiene = initial.get("hygiene", 60)
            agent.needs.temperature_stress = initial.get("temperature_stress", 50)

        # Initialize trust scores (neutral with all other agents)
        for other_config in data["agents"]:
            if other_config["id"] != agent_config["id"]:
                agent.trust_scores[other_config["id"]] = 0.0

        agents.append(agent)

    return agents


def distribute_capsule_inventory(
    agents: list[Agent],
    presets_path: str = "config/agent_presets.json",
    central_depot_inventory: Optional[dict[str, int]] = None,
):
    """Issue field kit and stage shared capsule stock at the lander.

    Every ``Agent`` already carries a two-meal, two-litre and two-cylinder
    emergency EVA loadout. When a physical central depot is supplied, only
    that loadout stays on the person and the remainder of the manifest is
    staged at the landing hub. The legacy no-depot call retains its historical
    even split for standalone callers.
    """
    if not agents:
        return
    with open(presets_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    capsule = data["capsule_inventory"]["items"]

    consumable_map = {
        "emergency_rations": "ration_packs",
        "water_packs": "water_packs",
        "oxygen_canisters": "oxygen_canisters",
    }
    if central_depot_inventory is None:
        for item_name in consumable_map:
            quantity = int(capsule[item_name]["quantity"])
            per_agent = quantity // len(agents)
            remainder = quantity % len(agents)
            for index, agent in enumerate(agents):
                issued = per_agent + (1 if index < remainder else 0)
                agent.inventory.add_item(item_name, issued)
    else:
        for item_name, depot_name in consumable_map.items():
            manifest_quantity = int(capsule[item_name]["quantity"])
            already_issued = sum(
                int(agent.inventory.items.get(item_name, 0))
                for agent in agents
            )
            staged = max(0, manifest_quantity - already_issued)
            central_depot_inventory[depot_name] = (
                int(central_depot_inventory.get(depot_name, 0)) + staged
            )

    # Emergency blankets — one per agent
    for agent in agents:
        agent.inventory.add_item("emergency_blanket", 1)

    # Multitools — three total, issued first to the crew most likely to operate
    # fabrication equipment or supervise field assembly. Strength breaks an
    # engineering tie; the item count still comes only from the manifest.
    agents_by_strength = sorted(
        agents,
        key=lambda a: (
            -int(getattr(a.competency, "engineering", 0)),
            -int(getattr(a.genome, "strength", 0)),
            a.id,
        ),
    )
    multitool_count = int(capsule["multitool_kit"]["quantity"])
    issued_multitools = min(multitool_count, len(agents))
    for i in range(issued_multitools):
        agents_by_strength[i].inventory.add_item("multitool_kit", 1)
    if central_depot_inventory is not None and multitool_count > issued_multitools:
        central_depot_inventory["multitool_kit"] = (
            int(central_depot_inventory.get("multitool_kit", 0))
            + multitool_count - issued_multitools
        )

    # Sensitive electronics and clinical stock stay in the protected lander
    # locker when that physical store exists. A no-depot compatibility caller
    # keeps the original named custodian behavior.
    eng_agent = max(agents, key=lambda a: a.competency.engineering)
    med_agent = max(agents, key=lambda a: a.competency.medical)
    electronics_quantity = int(capsule["electronics_salvage"]["quantity"])
    medical_quantity = int(capsule["medical_supplies"]["quantity"])
    if central_depot_inventory is None:
        eng_agent.inventory.add_item(
            "electronics_salvage", electronics_quantity
        )
        med_agent.inventory.add_item("medical_supplies", medical_quantity)
    else:
        central_depot_inventory["electronics_salvage"] = (
            int(central_depot_inventory.get("electronics_salvage", 0))
            + electronics_quantity
        )
        central_depot_inventory["medical_supplies"] = (
            int(central_depot_inventory.get("medical_supplies", 0))
            + medical_quantity
        )
    # Invasive rehydration is not made from generic bandages plus tank water.
    # The landed medical manifest carries sterile isotonic fluid and a separate
    # single-use vascular/IO administration set; both are finite consumables.
    for item_name in (
        "sterile_iv_fluid_bags",
        "iv_io_administration_sets",
    ):
        item_manifest = capsule.get(item_name, {})
        quantity = max(0, int(item_manifest.get("quantity", 0)))
        if quantity:
            if central_depot_inventory is None:
                med_agent.inventory.add_item(item_name, quantity)
            else:
                central_depot_inventory[item_name] = (
                    int(central_depot_inventory.get(item_name, 0)) + quantity
                )

    # Portable scanner — give to highest perception agent
    perc_agent = max(agents, key=lambda a: a.genome.perception)
    perc_agent.inventory.add_item("portable_scanner", 1)

    # The single qualified emergency-beacon radio remains in the central
    # depot.  SimulationEngine owns that canonical mission item; duplicating
    # a second copy in the leader's backpack made the communications-array
    # prerequisite physically meaningless.
