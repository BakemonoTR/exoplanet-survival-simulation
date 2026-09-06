"""
Simulation Engine — Main tick loop.

Orchestrates all systems:
1. Event scheduling (flares, quakes, storms)
2. Agent physiology (needs decay, radiation, EVA)
3. Deterministic and reinforcement-learning decision making
4. Action execution (move, gather, build, eat, sleep...)
5. Colony score tracking
6. State persistence

Design: wall-clock pacing and simulated time are explicit mission settings.
"""

import time
import json
import hashlib
import heapq
import logging
import os
import random
import math
import uuid
import threading
from typing import Optional, Callable

import numpy as np

from src.agents.agent import (
    Agent, AgentStatus, DeathCause, MATERIAL_DENSITY_KG, distribute_capsule_inventory
)
from src.agents.prompts import build_system_prompt
from src.agents.decision import DecisionEngine
from src.agents.strategic_rl import (
    ColonyStrategicPolicy,
)
from src.orchestration.llm_client import LocalNarrativeClient, LLMCallType
from src.orchestration.fallback import FallbackDecisionEngine
from src.memory.vector_store import MemoryManager
from src.systems.colony_score import ColonyScore
from src.systems.cargo_ledger import CargoLedgerError, CargoMassLedger
from src.systems.event_scheduler import EventScheduler
from src.systems.mission_profile import MissionProfile
from src.systems.surface_fleet import FleetJob, SurfaceFleet
from src.systems.airlock import AirlockController
from src.systems.timebase import (
    normalize_tick_based_config,
    scale_tick_count,
    ticks_for_minutes,
)
from src.world.generator import WorldGenerator, PlanetConfig

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    Main simulation loop that ties all systems together.
    
    Lifecycle:
    1. Initialize (load planet, generate world, create agents)
    2. Run tick loop until end condition
    3. Report results
    
    End conditions:
    - Colony readiness >= 80% (SUCCESS)
    - All agents dead (FAILURE)
    - Max ticks reached (TIMEOUT, default 1000 = ~3.5 sim days)
    """
    
    # Configuration
    STRATEGIC_EVAL_INTERVAL = 10   # Ticks between strategic evaluations
    REFLECTION_INTERVAL = 20       # Optional local narrative cadence
    CHECKPOINT_INTERVAL = 10       # Ticks between state saves
    # Backwards-compatible class value. Instance defaults come from the
    # validated mission profile (360 days / 51,840 ticks).
    MAX_TICKS_DEFAULT = 51840
    TARGET_TICK_SECONDS = 2.0      # Two physics ticks per four-second UI snapshot
    # Rendering and persistence consume state snapshots, not physics ticks.
    # At accelerated audit speeds publish at most ten live snapshots/second;
    # every simulation tick still executes in full below.
    LIVE_STATE_MIN_INTERVAL_SECONDS = 0.10
    SIM_MINUTES_PER_TICK = 10.0
    SIM_HOURS_PER_TICK = SIM_MINUTES_PER_TICK / 60.0
    # NASA's standard crew workday allocates 6.5 h to scheduled operations.
    # A two-hour duty block created more handovers than useful machine work;
    # 39 ten-minute ticks models one primary operations shift while vitals can
    # still interrupt it earlier.
    MANUFACTURING_SHIFT_TICKS = 39
    # A six-person precursor crew operates on foot around the landing hub.
    # Long-range deposits require later survey/transport infrastructure; they
    # must not turn an ordinary gather/explore decision into a multi-kilometre
    # unsupported EVA.
    LOCAL_EVA_RADIUS_CELLS = 24
    MAX_EVA_RADIUS_CELLS = LOCAL_EVA_RADIUS_CELLS  # backwards-compatible audit/config name
    # All normal modules and the planned solar lattice live inside this
    # surveyed campus. Foundations, buried utilities, access lanes and future
    # expansion remain intact even before a particular building is erected.
    CONSTRUCTION_CAMPUS_RADIUS_CELLS = 15
    CONSTRUCTION_GROUND_BUFFER_CELLS = 2
    # The lander datum is a grid-cell anchor, not the geometric centre of an
    # odd-sized sprite.  These asymmetric offsets describe a real 4 x 4
    # planning-cell footprint (400 m x 400 m on the present coarse civil
    # grid).  Every surface movement uses the same south-side suitlock.
    LANDER_FOOTPRINT_MIN_DX = -1
    LANDER_FOOTPRINT_MAX_DX = 2
    LANDER_FOOTPRINT_MIN_DY = -1
    LANDER_FOOTPRINT_MAX_DY = 2
    LANDER_AIRLOCK_DX = 0
    LANDER_AIRLOCK_DY = 2
    LANDER_AIRLOCK_EXTERIOR_DX = 0
    LANDER_AIRLOCK_EXTERIOR_DY = 3
    MIN_EXTRACTION_RADIUS_CELLS = (
        CONSTRUCTION_CAMPUS_RADIUS_CELLS
        + CONSTRUCTION_GROUND_BUFFER_CELLS + 1
    )
    MIN_EVA_EXIT_O2_PCT = 35.0
    MIN_ROUTINE_EVA_EXIT_O2_PCT = 55.0
    # Keep a real hysteresis band above the outdoor return threshold.  The
    # distance-aware MOVE check can raise this further for long routes.
    MIN_ROUTINE_EVA_START_ENERGY_PCT = 72.0
    MIN_EVA_EXIT_SUIT_INTEGRITY = 0.45
    MIN_EVA_EXIT_SUIT_CONDITION = 0.50
    # The hard limit above is for emergency work. Routine sorties need enough
    # joint/bearing service margin to finish a useful local task and return;
    # opening at exactly 50% caused a one-step EVA followed by an immediate
    # 49.8% retreat.
    MIN_ROUTINE_EVA_EXIT_SUIT_CONDITION = 0.65
    MIN_CENTRAL_MAINTENANCE_O2_CANISTERS = 2
    SOLAR_CLEANING_DUST_THRESHOLD = 0.20
    EXPEDITION_MOVE_SPEED_CELLS = 2
    EXPEDITION_WORK_RESERVE_TICKS = 6
    # A 100 m grid makes a ten-cell trip a one-kilometre haul.  Beyond this
    # point a two-person unpressurised rover sortie is safer and materially
    # more efficient than repeated solo walks, while short campus work stays
    # on foot.
    CREW_ROVER_MIN_DISTANCE_CELLS = 10
    MAX_CONTINUOUS_EVA_TICKS = 36
    # A conservative ISS-class precursor loop.  NASA's current brine
    # processor demonstration can reach 98%, but the delivered lander is not
    # silently credited with that advanced hardware.
    HABITAT_WATER_RECOVERY_FRACTION = 0.90
    ADVANCED_WATER_RECOVERY_FRACTION = 0.98
    WATER_PROCESSOR_WARMUP_MINUTES = 60.0
    # Standard Sabatier closes roughly half of the OGS water loop when methane
    # is vented.  A 75% credit would require additional methane-pyrolysis or
    # equivalent oxygen-recovery hardware that is not in the lander manifest.
    SABATIER_WATER_RETURN_FRACTION = 0.50
    # The first delivered PV wing has a pre-integrated lander input.  Further
    # arrays need an installed PMAD node before their outputs can be pooled.
    LANDER_DIRECT_SOLAR_INPUTS = 1
    POWER_GRID_DEFAULT_EFFICIENCY = 0.95
    POWER_GRID_DEFAULT_SOLAR_FEEDERS = 20
    PORTABLE_SCANNER_SCAN_RADIUS_CELLS = 1
    PORTABLE_SCANNER_SCAN_CHARGE_PCT = 8.0
    PORTABLE_SCANNER_CHARGE_KWH = 0.10
    PORTABLE_SCANNER_CHARGE_STEP_PCT = 25.0
    PORTABLE_SCANNER_MAX_TEST_PIT_LAYERS = 2
    # Twelve non-overlapping 3x3 scanner footprints (or twelve spaced test
    # pits) sample more than one hundred local cells. Continuing the same
    # campaign after twelve resource-specific misses is inferior to a planned
    # buddy traverse into the next geological zone.
    LOCAL_RESOURCE_MISS_LIMIT = 12
    ORBITAL_RECON_OPERATIONAL_RADIUS_CELLS = 50
    BOOTSTRAP_RESOURCE_RECIPES = (
        "solar_panel",
        "water_collector",
        "isru_o2_unit",
    )
    
    # Tactical LLM call triggers
    NEED_CRITICAL_THRESHOLD = 25.0
    
    def __init__(self, planet_config_path: str,
                 agent_configs: list[dict] = None,
                 seed: int = 42,
                 max_ticks: int = None,
                 tick_speed: float = None,
                 on_tick: Callable = None,
                 on_event: Callable = None,
                 db = None,
                 strategic_deadline_learning: bool = False):
        """
        Initialize simulation.
        
        Args:
            planet_config_path: Path to planet JSON config
            agent_configs: List of agent config dicts (or None for defaults)
            seed: Random seed for deterministic world generation
            max_ticks: Maximum simulation ticks
            tick_speed: Seconds per tick (None = mission profile value)
            on_tick: Callback(tick, state_dict) called each tick
            on_event: Callback(event_dict) called on notable events
            db: Database persistence instance
            strategic_deadline_learning: Opt-in strategy experiment with isolated persistence
        """
        # Load planet
        self.seed = int(seed)
        self.mission_profile = MissionProfile.load()
        self.planet = PlanetConfig(planet_config_path)
        self.world = WorldGenerator(
            self.planet,
            seed=seed,
            tick_minutes=self.mission_profile.clock.tick_minutes,
        )
        # The mission profile is the authoritative clock.  All rates below are
        # expressed through these instance values so changing tick resolution
        # cannot silently change physical consumption or work duration.
        self.SIM_MINUTES_PER_TICK = self.mission_profile.clock.tick_minutes
        self.SIM_HOURS_PER_TICK = self.SIM_MINUTES_PER_TICK / 60.0
        self.TARGET_TICK_SECONDS = (
            self.mission_profile.clock.realtime_seconds_per_tick
        )
        self.MANUFACTURING_SHIFT_TICKS = ticks_for_minutes(
            6.5 * 60.0, self.SIM_MINUTES_PER_TICK
        )
        self.EXPEDITION_WORK_RESERVE_TICKS = ticks_for_minutes(
            60.0, self.SIM_MINUTES_PER_TICK
        )
        self.EXPEDITION_LAUNCH_MARGIN_TICKS = ticks_for_minutes(
            50.0, self.SIM_MINUTES_PER_TICK
        )
        self.MAX_CONTINUOUS_EVA_TICKS = ticks_for_minutes(
            6.0 * 60.0, self.SIM_MINUTES_PER_TICK
        )
        # Convert the documented suited gait into this map/time resolution.
        # Gravity, terrain, visibility, load and physiology reduce this
        # nominal rate later in the movement kernel.  The configured 2.4 km/h
        # lies inside NASA's measured suited-walking range; it is deliberately
        # below the roughly 4 km/h near-maximum reported for a Mars-gravity
        # suit test.
        mobility = dict(self.mission_profile.crew_surface_mobility)
        routine_walk_kph = max(
            0.1, float(mobility.get("routine_suited_walk_kph", 2.4))
        )
        expedition_walk_kph = max(
            0.1,
            float(mobility.get(
                "planned_expedition_walk_kph", routine_walk_kph
            )),
        )

        def suited_cells_per_tick(speed_kph: float) -> int:
            return max(1, int(round(
                speed_kph * self.SIM_HOURS_PER_TICK * 1000.0
                / self.mission_profile.grid_cell_meters
            )))

        self.EVA_WALK_SPEED_CELLS = suited_cells_per_tick(routine_walk_kph)
        self.EXPEDITION_MOVE_SPEED_CELLS = suited_cells_per_tick(
            expedition_walk_kph
        )
        self.MOVEMENT_SURGE_SPEED_MULTIPLIER = max(
            1.0,
            min(2.0, float(mobility.get(
                "rl_surge_speed_multiplier", 1.5
            ))),
        )
        self.MOVEMENT_SURGE_METABOLIC_MULTIPLIER = max(
            1.0,
            min(2.0, float(mobility.get(
                "rl_surge_metabolic_multiplier", 1.5
            ))),
        )
        self.surface_fleet = SurfaceFleet(self.mission_profile)
        self.airlock = AirlockController(ticks_for_minutes(10.0, self.SIM_MINUTES_PER_TICK))
        # Mission planners know which resource classes exist somewhere on the
        # selected planet, but agents still have to survey to learn locations.
        # This prevents an endless search for a material the planet config
        # explicitly says does not exist (for example Kepler-442b water ice).
        self.planetary_resources: set[str] = {
            resource
            for biome in self.planet.biome_list
            for resource in (biome.get("resources", {}) or {})
        }
        self._db = db
        if self._db is None:
            try:
                from src.api.database import SimulationDB
                self._db = SimulationDB()
            except Exception:
                self._db = None
        
        # Create agents
        self.agents: list[Agent] = []
        if agent_configs:
            for cfg in agent_configs:
                crew = Agent(cfg)
                crew.configure_timebase(self.SIM_MINUTES_PER_TICK)
                self.agents.append(crew)
        # If no configs provided, they'll be added via add_agent()
        
        # Systems
        resolved_max_ticks = (
            int(max_ticks)
            if max_ticks is not None
            else int(
                self.mission_profile.clock.planet_attempt_max_ticks
                or self.MAX_TICKS_DEFAULT
            )
        )
        self.event_scheduler = EventScheduler(
            self.planet,
            seed=seed,
            max_ticks=resolved_max_ticks,
            tick_minutes=self.SIM_MINUTES_PER_TICK,
        )
        self.colony_score = ColonyScore()
        planning_solar_flux = max(0.0, float(self.planet.data.get(
            "surface", {}
        ).get("solar_flux_relative_to_earth", 1.0)))
        planning_firm_multiplier = min(
            1.0,
            planning_solar_flux * (2.0 if self.planet.tidally_locked else 1.0),
        )
        self.colony_score.set_structure_capacity_multiplier(
            "solar_panel", planning_firm_multiplier
        )
        self.llm_client = LocalNarrativeClient()
        self.memory = MemoryManager()
        self.decision_engine = DecisionEngine(
            llm_client=self.llm_client,
            memory_manager=self.memory,
            colony_score=self.colony_score,
        )
        self.decision_engine.greenhouse_utility_connected = (
            self._life_support_connected
        )
        self.decision_engine.life_support_network_snapshot = (
            self._life_support_network_snapshot
        )
        radiation_system = self.planet.data.get("radiation_system", {}) or {}
        baseline_doses = radiation_system.get("baseline_dose_per_tick", {}) or {}
        flare_profile = self.planet.data.get("flare_profile", {}) or {}
        self.decision_engine.severe_radiation_environment = bool(
            radiation_system.get("enabled", False)
            and (
                max(
                    [float(value) for value in baseline_doses.values()] or [0.0]
                ) >= 0.001
                or float(flare_profile.get("flare_events_per_80_days", 0.0))
                >= 20.0
            )
        )
        self.strategic_policy = ColonyStrategicPolicy(
            self.planet.id, seed=self.seed,
            deadline_learning=strategic_deadline_learning,
        )
        self._strategy_deadline_readiness = 0.0
        self.decision_engine.strategic_policy = self.strategic_policy
        self._strategic_policy_loaded = False
        self._terminal_reward_applied = False
        self._terminal_learning_finalized = False
        # Consecutive ticks for which every operational arrival gate remains
        # valid. Any outage resets this qualification run.
        self._support_soak_ticks = 0
        self.decision_engine.manufacturing_shift_ticks = (
            self.MANUFACTURING_SHIFT_TICKS
        )
        self.decision_engine.max_continuous_eva_ticks = (
            self.MAX_CONTINUOUS_EVA_TICKS
        )
        # Physics and validated RL actions own the state transition.  Language
        # generation is retained solely for non-authoritative crew dialogue.
        self.decision_engine.language_planning_enabled = False
        
        # State
        self.engine_id = str(uuid.uuid4())
        self.end_reason: Optional[str] = "running"
        self.current_tick: int = 0
        self.max_ticks = resolved_max_ticks
        self.tick_speed = tick_speed or self.TARGET_TICK_SECONDS
        self.running: bool = False
        self.paused: bool = False
        self._stop_event = threading.Event()
        # Physical Base Logistics: Central Storage Depot Silo at Landing Zone
        self.central_depot_inventory: dict[str, int] = {
            # Planned precursor reserve: 30 x 0.36 kg field cylinders. These
            # are life-support consumables, not free construction materials.
            "oxygen_canisters": 30,
            "ration_packs": 30,
            "water_packs": 40,
        }
        # Semiconductor junctions, PEM membranes, bearings, space-rated
        # polymers and qualified control boards cannot be made from loose ore
        # by a six-person field shop.  They are finite, mass-accounted cargo;
        # local industry still has to produce the bulk frames, pipes, anchors
        # and housings around them.
        self.delivered_precision_stock: dict[str, dict] = {}
        for material, manifest in self.mission_profile.delivered_precision_stock.items():
            quantity = max(0, int(manifest.get("quantity", 0)))
            if quantity <= 0:
                continue
            self.central_depot_inventory[material] = (
                self.central_depot_inventory.get(material, 0) + quantity
            )
            self.delivered_precision_stock[material] = dict(manifest)
        # Standardized trusses, pipes and tested subassemblies are finite
        # landed stock, but unlike serialized flight cores the field shop can
        # reproduce them from local feedstock.  They remove only the
        # implausible from-ore portion of the emergency critical path.
        self.delivered_field_stock: dict[str, dict] = {}
        for material, manifest in self.mission_profile.delivered_field_stock.items():
            quantity = max(0, int(manifest.get("quantity", 0)))
            if quantity <= 0:
                continue
            self.central_depot_inventory[material] = (
                self.central_depot_inventory.get(material, 0) + quantity
            )
            self.delivered_field_stock[material] = dict(manifest)
        self.delivered_contingency_feedstocks: dict[str, int] = {}
        for material, quantity in self.mission_profile.conditional_feedstocks.items():
            # Mineable feedstock is cargo only when orbital survey says the
            # selected planet lacks it entirely. Otherwise the crew must find
            # and extract it; no recipe is altered to make that easier.
            if material not in self.planetary_resources and int(quantity) > 0:
                self.central_depot_inventory[material] = int(quantity)
                self.delivered_contingency_feedstocks[material] = int(quantity)
        self.world.central_depot_inventory = self.central_depot_inventory

        # Structures placed in world
        self.structures_built: dict[str, int] = {}
        self.placed_structures: list[dict] = []
        # Construction cargo has three conserved locations: reserved at the
        # landing-zone depot, physically aboard a transporter, or delivered at
        # one named site.  A central-depot quantity is never directly consumed
        # by a remote build action.
        self.site_material_staging: dict[str, dict[str, int]] = {}
        self._construction_cargo_reservations: dict[str, dict] = {}
        # A batch belongs to its physical machine, not to whichever astronaut
        # happens to be operating it.  Materials and power are committed once
        # when this record is created; operator actions merely advance its
        # remaining supervised process time.
        self._manufacturing_cycles: dict[str, dict] = {}
        automation = dict(
            self.mission_profile.delivered_industry.get("automation", {})
        )
        self.AUTONOMOUS_MANUFACTURING_ENABLED = bool(
            automation.get("autonomous_batch_control", False)
        )
        self.MANUFACTURING_SETUP_TICKS = ticks_for_minutes(
            max(0.0, float(automation.get("human_setup_minutes", 10.0))),
            self.SIM_MINUTES_PER_TICK,
        )
        self.MANUFACTURING_UNLOAD_TICKS = ticks_for_minutes(
            max(
                0.0,
                float(automation.get(
                    "human_unload_inspection_minutes", 10.0
                )),
            ),
            self.SIM_MINUTES_PER_TICK,
        )
        self._manufacturing_automation_telemetry = {
            "enabled": self.AUTONOMOUS_MANUFACTURING_ENABLED,
            "human_setup_ticks_per_batch": self.MANUFACTURING_SETUP_TICKS,
            "human_unload_ticks_per_batch": self.MANUFACTURING_UNLOAD_TICKS,
            "cycles_started": 0,
            "process_ticks_elapsed": 0,
            "process_cycles_completed": 0,
            "cycles_unloaded": 0,
            "nominal_process_ticks_committed": 0,
        }
        # The planner needs the same canonical machine ledger as execution;
        # otherwise an inactive/relief-reserved WIP batch looks like a free
        # workshop and agents repeatedly submit an impossible REFINE order.
        self.decision_engine.manufacturing_cycles = self._manufacturing_cycles
        
        # Pre-cache recipes to eliminate per-tick file I/O
        self._recipes_cache: dict[str, dict] = {}
        recipes_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config', 'recipes.json'
        )
        try:
            if os.path.exists(recipes_path):
                with open(recipes_path, 'r', encoding='utf-8') as f:
                    self._recipes_cache = normalize_tick_based_config(
                        json.load(f).get("recipes", {}),
                        self.mission_profile.clock.tick_minutes,
                    )
        except Exception as e:
            logger.warning(f"Failed to load recipes cache in SimulationEngine: {e}")

        # Close the complete landed-mass boundary before any simulation state
        # can advance.  Cargo is audited separately for the uncrewed freighters
        # and the crew lander; being listed here never grants readiness credit.
        self.cargo_mass_ledger = CargoMassLedger(
            self.mission_profile, self._recipes_cache
        ).to_dict()

        # Delivered structure kits are reservation labels over parts already
        # present in the audited precision/field stock. They remain physically
        # stored in the depot, but are not a second copy of the same hardware.
        # Their flight-qualified subassemblies are reservation-controlled.
        # A pump-module order therefore cannot silently dismantle the starter
        # array before the crew deploys it.  The recipe remains the sole BOM
        # authority: changing a recipe automatically changes the packed kit.
        self.delivered_structure_kits: dict[str, dict[str, int]] = {}
        for recipe_name, kit_count in self.mission_profile.delivered_structure_kits.items():
            recipe = self._recipes_cache.get(recipe_name, {})
            kit_materials = {
                str(material): int(quantity) * int(kit_count)
                for material, quantity in recipe.get("materials", {}).items()
                if int(quantity) > 0
            }
            if not kit_materials:
                logger.warning(
                    "Delivered kit %s has no valid recipe BOM; ignoring it",
                    recipe_name,
                )
                continue
            missing = {
                material: quantity - int(
                    self.central_depot_inventory.get(material, 0)
                )
                for material, quantity in kit_materials.items()
                if int(self.central_depot_inventory.get(material, 0)) < quantity
            }
            if missing:
                raise CargoLedgerError(
                    f"Delivered kit {recipe_name} is not contained in packed "
                    f"field/precision stock: {missing}"
                )
            self.delivered_structure_kits[recipe_name] = kit_materials
        self.decision_engine.delivered_structure_kits = (
            self.delivered_structure_kits
        )
        
        # Colony resource reserves
        initial_res = getattr(self.planet, "initial_colony_resources", {})
        crew_consumables = self.mission_profile.advance_crew_consumables
        crew_power = self.mission_profile.advance_crew_power
        self._colony_resources: dict[str, float] = {
            "energy_stored_kwh": float(initial_res.get(
                "energy_stored_kwh", crew_power["initial_charge_kwh"]
            )),
            # Delivered lander power keeps the six-person precursor alive but
            # never counts as constructed 106-person settlement capacity.
            "lander_integrated_solar_peak_kw": float(initial_res.get(
                "lander_integrated_solar_peak_kw",
                crew_power["integrated_lander_solar_peak_kw"],
            )),
            "lander_auxiliary_power_kw": float(
                initial_res.get(
                    "lander_auxiliary_power_kw",
                    crew_power["auxiliary_fuel_cell_power_kw"],
                )
            ),
            "lander_auxiliary_energy_remaining_kwh": float(initial_res.get(
                "lander_auxiliary_energy_remaining_kwh",
                crew_power["auxiliary_reactant_energy_kwh"],
            )),
            # The 180-day experiment cannot be human-rated around an
            # unfinished ISRU plant. These are delivered crew consumables,
            # never construction feedstock or colony-readiness capacity.
            "o2_reserve_kg": float(initial_res.get(
                "o2_reserve_kg", crew_consumables["oxygen_kg"]
            )),
            "water_reserve_l": float(initial_res.get(
                "water_reserve_l", crew_consumables["potable_water_l"]
            )),
            "food_reserve_kcal": float(initial_res.get(
                "food_reserve_kcal", crew_consumables["food_kcal"]
            )),
        }
        # Rated dry/cold-food stowage must at least hold the delivered
        # manifest. Never silently discard packed food merely because an old
        # gameplay cap was lower than the mission load.
        self._food_storage_capacity_kcal = max(
            float(crew_consumables["food_kcal"]),
            self._colony_resources["food_reserve_kcal"]
        )
        self._lander_water_storage_capacity_l = max(
            float(crew_consumables.get("potable_water_l", 0.0)),
            self._colony_resources["water_reserve_l"],
        )
        self._water_storage_capacity_l = self._lander_water_storage_capacity_l
        self._lander_o2_storage_capacity_kg = float(
            crew_consumables.get(
                "oxygen_buffer_capacity_kg",
                self._colony_resources["o2_reserve_kg"],
            )
        )
        self._o2_storage_capacity_kg = self._lander_o2_storage_capacity_kg
        # Treated output is held here during the physical WPA warm-up/batch
        # cycle.  Consumed water first resides in each crew member's explicit
        # body-water pool; a drink is never credited straight back to storage.
        self._water_recovery_queue: list[dict[str, float | int]] = []
        self._water_cycle_telemetry: dict[str, float] = {}
        self._movement_effort_telemetry = {
            "eligible_routes": 0,
            "steady_routes": 0,
            "surge_routes": 0,
            "completed_routes": 0,
            "aborted_routes": 0,
            "aborted_by_reason": {},
            "censored_routes": 0,
            "censored_by_reason": {},
            "estimated_ticks_saved": 0.0,
            "total_route_reward": 0.0,
            "by_mode": {
                mode: {
                    "completed_routes": 0,
                    "aborted_routes": 0,
                    "censored_routes": 0,
                    "elapsed_ticks": 0,
                    "expected_steady_ticks": 0.0,
                    "estimated_ticks_saved": 0.0,
                    "energy_loss": 0.0,
                    "thirst_loss": 0.0,
                    "o2_loss": 0.0,
                    "total_reward": 0.0,
                }
                for mode in ("steady", "surge")
            },
        }
        
        # Structure health tracks real damage, not a game-style countdown.
        # Calendar maintenance is recorded per physical asset; missing an
        # inspection increases vulnerability to an actual hazard but cannot
        # make qualified hardware spontaneously collapse in calm weather.
        self.structure_health: dict[str, float] = {}
        
        # Day/Night cycle state
        dn = self.planet.day_night_cycle
        if dn and dn.get("enabled"):
            self._dn_enabled = True
            self._dn_cycle_length = scale_tick_count(
                dn.get("cycle_length_ticks", 576), self.SIM_MINUTES_PER_TICK
            )
            self._dn_day_ticks = (
                scale_tick_count(
                    dn["day_ticks"], self.SIM_MINUTES_PER_TICK
                )
                if dn.get("day_ticks") is not None
                else self._dn_cycle_length // 2
            )
            self._dn_night_ticks = max(
                1, self._dn_cycle_length - self._dn_day_ticks
            )
            self._dn_transition = scale_tick_count(
                dn.get("dawn_dusk_transition_ticks", 24),
                self.SIM_MINUTES_PER_TICK,
            )
        elif self.planet.tidally_locked:
            # Tidally locked planets have no day/night cycle
            self._dn_enabled = False
        else:
            # Fallback: use day_length_hours from surface config if available
            day_hours = self.planet.data.get("surface", {}).get("day_length_hours", None)
            if day_hours and day_hours > 0:
                self._dn_enabled = True
                ticks_per_hour = 60.0 / self.SIM_MINUTES_PER_TICK
                self._dn_cycle_length = int(day_hours * ticks_per_hour)
                self._dn_day_ticks = self._dn_cycle_length // 2
                self._dn_night_ticks = self._dn_cycle_length - self._dn_day_ticks
                self._dn_transition = max(
                    ticks_for_minutes(30, self.SIM_MINUTES_PER_TICK),
                    self._dn_cycle_length // 30,
                )
            else:
                self._dn_enabled = False
        
        # Discovered resources (resource_name -> set of (x, y) coordinates)
        self.discovered_resources: dict[str, set[tuple[int, int]]] = {}
        self.depleted_cell_resources: set[tuple[int, int, str]] = set()
        self.cell_resource_units: dict[tuple[int, int, str], int] = {}
        self.revealed_cell_resources: set[tuple[int, int]] = set()
        self.remote_resource_requests: set[str] = set()
        self._resource_survey_cache: dict[tuple[str, int, int], tuple[int, int] | None] = {}
        self._portable_scan_centers: set[tuple[int, int]] = set()
        self._portable_scan_attempts: dict[tuple[int, int], int] = {}
        self._portable_resource_miss_centers: dict[
            str, set[tuple[int, int]]
        ] = {}
        self._hand_prospect_centers: set[tuple[int, int]] = set()
        self._local_resource_miss_centers: dict[str, set[tuple[int, int]]] = {}
        self.cell_excavation_depth: dict[tuple[int, int], int] = {}
        # Sparse vertical geology. A cell is materialized only when surveyed
        # or excavated, so a 2000x2000 world does not allocate four million
        # layer stacks.
        self.cell_geology: dict[tuple[int, int], dict] = {}
        self.spoil_piles: dict[tuple[int, int], dict[str, int]] = {}
        
        # Callbacks
        self._on_tick = on_tick
        self._on_event = on_event
        self.decision_engine.on_event = self._on_event
        
        # Agent prompt caches (system prompts built once)
        self._system_prompts: dict[str, str] = {}
        
        # Tick timing
        self._tick_times: list[float] = []
        self._last_state_snapshot_tick: int | None = None
        
        logger.info(
            f"SimulationEngine initialized: planet={self.planet.name}, "
            f"seed={seed}, max_ticks={self.max_ticks}"
        )
    
    def scan_surroundings(self, cx: int, cy: int, radius: int = 7):
        """Scan cells around (cx, cy) and record base_resources to discovered_resources."""
        if not hasattr(self, "discovered_resources") or self.discovered_resources is None:
            self.discovered_resources = {}
        if not hasattr(self, "depleted_cell_resources"):
            self.depleted_cell_resources = set()
        if not hasattr(self, "revealed_cell_resources"):
            self.revealed_cell_resources = set()

        # Structure footprints are invariant during this one scan. Building
        # every footprint again for every one of the 225 visible cells made a
        # moving six-person crew perform tens of thousands of identical set
        # constructions per simulation tick.
        occupied_structure_cells: set[tuple[int, int]] = set()
        for structure in getattr(self, "placed_structures", []):
            occupied_structure_cells.update(self._structure_footprint_cells(
                str(structure.get("type", "structure")),
                int(structure.get("x", cx)),
                int(structure.get("y", cy)),
                structure,
            ))
        lander_footprint_cells = self._lander_footprint_cells()

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = cx + dx
                y = cy + dy
                if 0 <= x < self.world.map_size and 0 <= y < self.world.map_size:
                    # Solid foundations cover their complete declared parcel
                    # footprint; a 4 x 4 lander must not expose geology under
                    # its outer row merely because its datum is off-centre.
                    is_under_base = (x, y) in lander_footprint_cells
                    is_under_building = (x, y) in occupied_structure_cells
                    if is_under_base or is_under_building:
                        continue

                    # Discovery needs only the static geological unit. Calling
                    # full dynamic cell physics (wind, pressure, light and
                    # temperature) for all 225 visible cells made every agent
                    # tick needlessly recompute millions of noise samples.
                    base_res = self.world.get_biome(x, y).get("resources", {})
                    if base_res:
                        # 1. Surface Regolith is universally discoverable across visible terrain
                        if (x, y, "regolith") not in self.depleted_cell_resources:
                            if "regolith" not in self.discovered_resources:
                                self.discovered_resources["regolith"] = set()
                            self.discovered_resources["regolith"].add((x, y))

                        # 2. Merely seeing a disturbed cell must not disclose
                        # every deeper mineral. Exact resources are added by a
                        # detector pass or by excavating down to their layer.
                        for res in base_res.keys():
                            if res == "regolith" or (x, y, res) in self.depleted_cell_resources:
                                continue
                            if (x, y) in self.discovered_resources.get(res, set()):
                                self.discovered_resources.setdefault(res, set()).add((x, y))

    @staticmethod
    def _square_ring_offset(radius: int, perimeter_position: int) -> tuple[int, int]:
        """Map a clockwise square-perimeter distance to a grid offset."""
        radius = max(1, int(radius))
        perimeter = 8 * radius
        p = int(perimeter_position) % perimeter
        side = 2 * radius
        if p < side:
            return -radius + p, -radius
        if p < 2 * side:
            return radius, -radius + (p - side)
        if p < 3 * side:
            return radius - (p - 2 * side), radius
        return -radius, radius - (p - 3 * side)

    def _portable_scan_station_offsets(self) -> list[tuple[int, int]]:
        """Build a staged local traverse whose detector footprints tile the annulus.

        The handheld detector measures its current 100 m cell and the eight
        adjacent cells, so station centres are separated by at most three
        cells.  Square rings at the same three-cell spacing tile every cell
        between the protected construction campus and the local EVA limit.

        The first pass is reconnaissance rather than an exhaustive inner
        circumference: six offset angular sectors alternate between the two
        inner rings, followed by the two outer rings.  This gives an early,
        directionally balanced sample while the later clockwise infill keeps
        the plan gap-free.  Nothing here reads biome or deposit data.
        """
        geometry = (self.PORTABLE_SCANNER_SCAN_RADIUS_CELLS,
                    self.MIN_EXTRACTION_RADIUS_CELLS, self.LOCAL_EVA_RADIUS_CELLS)
        cached = getattr(self, "_portable_scan_offsets_cache", None)
        if cached is not None and cached[0] == geometry:
            return list(cached[1])
        footprint_width = self.PORTABLE_SCANNER_SCAN_RADIUS_CELLS * 2 + 1
        radii = list(range(
            self.MIN_EXTRACTION_RADIUS_CELLS,
            self.LOCAL_EVA_RADIUS_CELLS + 1,
            footprint_width,
        ))
        if not radii:
            return []

        dense_points: dict[int, list[tuple[int, int]]] = {}
        anchor_indices: dict[int, list[int]] = {}
        reconnaissance_sectors = 6
        # An off-axis phase prevents cardinal/diagonal spokes and puts every
        # traverse through the interior of a geological sector, not a corner.
        phase_fraction = 1.0 / 16.0
        for radius in radii:
            positions = list(range(0, 8 * radius, footprint_width))
            dense_points[radius] = [
                self._square_ring_offset(radius, position)
                for position in positions
            ]
            count = len(positions)
            anchors: list[int] = []
            for sector in range(reconnaissance_sectors):
                fraction = (phase_fraction + sector / reconnaissance_sectors) % 1.0
                index = int(round(fraction * count)) % count
                if index not in anchors:
                    anchors.append(index)
            anchor_indices[radius] = anchors

        ordered: list[tuple[int, int]] = []
        emitted: dict[int, set[int]] = {radius: set() for radius in radii}
        # Alternate adjacent radii within a sector.  The route therefore
        # advances as short radial transects joined clockwise around the LZ,
        # instead of repeatedly drawing long rays through the base.
        sector_order = [reconnaissance_sectors - 1, *range(reconnaissance_sectors - 1)]
        for pair_start in range(0, len(radii), 2):
            radial_pair = radii[pair_start:pair_start + 2]
            for sector in sector_order:
                for radius in radial_pair:
                    anchors = anchor_indices[radius]
                    if sector >= len(anchors):
                        continue
                    index = anchors[sector]
                    if index in emitted[radius]:
                        continue
                    emitted[radius].add(index)
                    ordered.append(dense_points[radius][index])

        # Complete each ring clockwise.  Together with the three-cell radial
        # spacing this makes the eventual survey coverage mathematically
        # complete without increasing the detector's physical sensing range.
        for radius in radii:
            points = dense_points[radius]
            start = int(round(phase_fraction * len(points))) % len(points)
            for step in range(len(points)):
                index = (start + step) % len(points)
                if index in emitted[radius]:
                    continue
                emitted[radius].add(index)
                ordered.append(points[index])
        self._portable_scan_offsets_cache = (geometry, tuple(ordered))
        return ordered

    def _next_portable_scan_station(self, agent: Agent | None = None) -> tuple[int, int] | None:
        """Return the next untested station in the physical local traverse."""
        lz_x = getattr(self, "lz_x", self.world.center)
        lz_y = getattr(self, "lz_y", self.world.center)
        used = getattr(self, "_portable_scan_centers", set())
        for dx, dy in self._portable_scan_station_offsets():
            station = (lz_x + dx, lz_y + dy)
            if station in used:
                continue
            if (
                0 <= station[0] < self.world.map_size
                and 0 <= station[1] < self.world.map_size
                and not self._is_construction_protected_cell(*station)
            ):
                return station
        return None

    def _next_regional_scan_station(
        self, agent: Agent | None = None, resource: str | None = None
    ) -> tuple[int, int] | None:
        """Choose the next unbiased station in a stratified expanding survey.

        A geological reconnaissance does not measure every square metre of a
        whole ring before moving one cell farther out. Cardinal and diagonal
        sectors alternate across gradually expanding rings, yielding a
        stratified low-discrepancy sample without favoring any resource biome.
        Exact deposits still require a physical detector reading.
        """
        lz_x = getattr(self, "lz_x", self.world.center)
        lz_y = getattr(self, "lz_y", self.world.center)
        used = getattr(self, "_portable_scan_centers", set())
        origin_x = agent.x if agent is not None else lz_x
        origin_y = agent.y if agent is not None else lz_y
        candidates: list[tuple[int, int, int, int]] = []
        max_radius = 50
        radii = range(self.LOCAL_EVA_RADIUS_CELLS + 2, max_radius + 1, 4)
        for ring_index, radial in enumerate(radii):
            # Four quadrant-balanced stations per ring, with a golden-ratio
            # offset that changes on every ring.  Every station remains on the
            # square ring (so range auditing is exact), but successive rings
            # do not repeat cardinal/diagonal rays and draw an artificial X.
            fraction = ((ring_index + 1) * 0.6180339887498949) % 1.0
            offset = max(1, min(radial - 1, int(round(fraction * radial))))
            ring_points = (
                (radial, offset),
                (-offset, radial),
                (-radial, -offset),
                (offset, -radial),
            )
            for dx, dy in ring_points:
                station = (lz_x + dx, lz_y + dy)
                if station in used:
                    continue
                if not (
                    0 <= station[0] < self.world.map_size
                    and 0 <= station[1] < self.world.map_size
                ):
                    continue
                if self._is_construction_protected_cell(*station):
                    continue
                travel = abs(station[0] - origin_x) + abs(station[1] - origin_y)
                candidates.append((ring_index, travel, station[0], station[1]))
        if candidates:
            _, _, sx, sy = min(candidates)
            return sx, sy
        return None

    def _bootstrap_planetary_raw_resources(self) -> set[str]:
        """Derive landing-site resource classes from the unchanged recipe graph."""
        raw_resources: set[str] = set()
        visiting: set[str] = set()

        def visit(material_or_recipe: str) -> None:
            if material_or_recipe in visiting:
                return
            recipe = self._recipes_cache.get(material_or_recipe)
            if not recipe:
                if material_or_recipe in self.planetary_resources:
                    raw_resources.add(material_or_recipe)
                return
            visiting.add(material_or_recipe)
            for material in recipe.get("materials", {}):
                visit(str(material))
            visiting.remove(material_or_recipe)

        for recipe_name in self.BOOTSTRAP_RESOURCE_RECIPES:
            visit(recipe_name)
        return raw_resources

    @staticmethod
    def _survey_resources_from_target(
        target: dict | None, expedition: dict | None = None
    ) -> list[str]:
        """Return ordered, unique objectives carried by one physical scan."""
        ordered: list[str] = []
        for source in (target or {}, expedition or {}):
            values = source.get("survey_resources", [])
            if not isinstance(values, (list, tuple)):
                values = []
            primary = source.get("resource")
            for value in ([primary] if primary else []) + list(values):
                value = str(value or "").strip()
                if value and value not in ordered:
                    ordered.append(value)
        return ordered

    def _start_regional_survey_expedition(
        self,
        agent: Agent,
        resource: str,
        station: tuple[int, int],
        survey_resources: list[str] | None = None,
        *,
        survey_scope: str = "regional",
    ) -> dict | None:
        """Launch a buddy/rover-supported scanner traverse without magic discovery."""
        target_distance = self._distance_from_lz(*station)
        if not self._agent_ready_for_expedition(agent, target_distance):
            return None
        buddy = self._find_expedition_buddy(agent, target_distance)
        if buddy is None:
            return None
        expedition = self._start_expedition(
            agent,
            resource,
            station,
            require_rover=(
                target_distance >= self.CREW_ROVER_MIN_DISTANCE_CELLS
            ),
        )
        if expedition is None:
            return None
        # _start_expedition normally receives a confirmed deposit. This is a
        # blind measurement station, so remove that provisional knowledge.
        self.discovered_resources.get(resource, set()).discard(station)
        for member in (agent, buddy):
            state = getattr(member, "_active_expedition", None)
            if isinstance(state, dict):
                state["kind"] = "regional_survey"
                state["survey_station_x"] = int(station[0])
                state["survey_station_y"] = int(station[1])
                state["authorized_radius"] = target_distance
                state["survey_scope"] = str(survey_scope)
                state["survey_resources"] = list(
                    survey_resources or [resource]
                )
        return expedition

    def _order_expedition_partner_home(self, expedition: dict | None):
        """Keep a buddy from being stranded when the lead reaches the airlock."""
        if not isinstance(expedition, dict) or expedition.get("role") != "lead":
            return
        buddy_id = expedition.get("buddy_id")
        buddy = next((crew for crew in self.agents if crew.id == buddy_id), None)
        buddy_state = getattr(buddy, "_active_expedition", None) if buddy else None
        if isinstance(buddy_state, dict):
            buddy_state["status"] = "returning"

    def _complete_crew_rover_expedition(self, expedition: dict | None) -> bool:
        """Unload conserved rover cargo at the hub, then release the rover."""
        if (
            not isinstance(expedition, dict)
            or expedition.get("transport") != "crew_rover"
            or expedition.get("role") != "lead"
        ):
            return False
        expedition_id = str(expedition.get("id", ""))
        if not expedition_id:
            return False
        payload = self.surface_fleet.unload_crew_rover_payload(expedition_id)
        for material, quantity in payload.items():
            if int(quantity) > 0:
                self.central_depot_inventory[material] = (
                    self.central_depot_inventory.get(material, 0)
                    + int(quantity)
                )
        lead = next((
            crew for crew in self.agents
            if crew.id == expedition.get("lead_id")
        ), None)
        recovery_resource = expedition.get("resource")
        recovery_contract = (
            getattr(lead, "_detected_resource_recovery", None)
            if lead is not None else None
        )
        if (
            isinstance(recovery_contract, dict)
            and recovery_resource
            and (
                int(recovery_contract.get("x", -1)),
                int(recovery_contract.get("y", -1)),
            ) == (
                int(expedition.get("target_x", -2)),
                int(expedition.get("target_y", -2)),
            )
        ):
            capacity_recipe = (
                recovery_contract.get("capacity_recipe")
                or expedition.get("capacity_recipe")
            )
            if capacity_recipe:
                remaining = self.decision_engine._raw_bom_deficits(
                    capacity_recipe,
                    self.decision_engine._pooled_materials(),
                ).get(recovery_resource, 0)
                recovery_complete = int(remaining) <= 0
                if not recovery_complete:
                    recovery_contract["recovery_goal_units"] = max(
                        1, int(remaining)
                    )
            else:
                recovery_complete = int(payload.get(
                    recovery_resource, 0
                )) >= max(
                    1,
                    int(expedition.get("recovery_goal_units", 1)),
                )
            if recovery_complete:
                lead._detected_resource_recovery = None
        completed = self.surface_fleet.complete_crew_rover_trip(expedition_id)
        if completed and payload and self._on_event:
            manifest = ", ".join(
                f"{quantity}x {material.replace('_', ' ')}"
                for material, quantity in sorted(payload.items())
            )
            self._on_event({
                "type": "rover_unload",
                "agent": expedition.get("lead_name", "Crew"),
                "cause": f"Crew rover returned and unloaded {manifest}",
                "tick": self.current_tick,
            })
        return completed

    def _continue_regional_survey(
        self,
        lead: Agent,
        expedition: dict,
        station: tuple[int, int],
    ) -> bool:
        """Continue a safe multi-station transect before returning to base."""
        if expedition.get("transport") == "crew_rover":
            # The first implementation reserves energy for one audited
            # out-and-back destination. A second station requires a fresh
            # dispatch calculation after return/recharge.
            return False
        buddy_id = expedition.get("buddy_id")
        buddy = next((crew for crew in self.agents if crew.id == buddy_id), None)
        buddy_state = getattr(buddy, "_active_expedition", None) if buddy else None
        if buddy is None or not isinstance(buddy_state, dict):
            return False

        target_radius = self._distance_from_lz(*station)
        for member in (lead, buddy):
            if (
                getattr(member.status, "value", str(member.status)) != "alive"
                or member.needs.energy < 70.0
                or member.needs.hunger < 55.0
                or member.needs.thirst < 60.0
                or member.needs.o2_supply < 55.0
            ):
                return False
            leg_cells = abs(member.x - station[0]) + abs(member.y - station[1])
            leg_ticks = math.ceil(leg_cells / self.EXPEDITION_MOVE_SPEED_CELLS)
            return_ticks = math.ceil(
                target_radius / self.EXPEDITION_MOVE_SPEED_CELLS
            )
            required = (
                leg_ticks + return_ticks + self.EXPEDITION_WORK_RESERVE_TICKS + 20
            )
            if self._expedition_available_ticks(member) < required:
                return False

        for member, state in ((lead, expedition), (buddy, buddy_state)):
            state["target_x"] = int(station[0])
            state["target_y"] = int(station[1])
            state["survey_station_x"] = int(station[0])
            state["survey_station_y"] = int(station[1])
            state["authorized_radius"] = max(
                int(state.get("authorized_radius", 0)), target_radius
            )
            state["status"] = "outbound"
            state["stations_visited"] = int(state.get("stations_visited", 1)) + 1
            member.action.action_type = "move"
            route_target = {
                "x": int(station[0]),
                "y": int(station[1]),
                "resource": expedition.get("resource"),
                "survey_resources": list(
                    expedition.get("survey_resources", [])
                ),
                "expedition": True,
            }
            if member.id == lead.id:
                route_target["survey_action"] = "portable_scanner"
            member.action.target = route_target
            member.action.ticks_remaining = 1
        return True

    def _resource_burial_depth(self, x: int, y: int, resource: str) -> int:
        """Stable shallow-geology layer for a procedural deposit (1..3)."""
        digest = hashlib.sha256(f"{x}:{y}:{resource}".encode("utf-8")).digest()
        if resource == "regolith":
            return 0
        # Conductive/metallic bodies can be detected slightly deeper than
        # non-conductive feedstocks, but every handheld result remains shallow.
        return 1 + digest[0] % 3

    def _get_cell_geology(self, x: int, y: int) -> dict:
        """Materialize a deterministic top-to-bottom layer stack for one cell."""
        key = (x, y)
        existing = self.cell_geology.get(key)
        if existing is not None:
            return existing

        # ``get_cell_info`` is the world's primary cell API and its
        # ``base_resources`` entry is the extraction authority.  Reading the
        # biome separately allowed fixtures (and future localized deposits) to
        # produce a layer identity from one source and a quantity from another.
        resources = self.world.get_cell_info(
            x, y, self.current_tick
        ).get("base_resources", {})
        # Loose surface regolith is the physical overburden on every exposed
        # rocky grid. Biome resource tables describe the units beneath it and
        # previously made some cells begin directly with ore, contradicting
        # both the renderer and the excavation model.
        ordered: list[str] = ["regolith"]
        subsurface = [resource for resource in resources if resource != "regolith"]
        subsurface.sort(key=lambda resource: (
            self._resource_burial_depth(x, y, resource),
            hashlib.sha256(f"layer:{x}:{y}:{resource}".encode("utf-8")).digest(),
        ))
        ordered.extend(subsurface)

        layers = []
        for material in ordered:
            quantity = self._get_initial_cell_resource_capacity(
                x, y, material, resources
            )
            recoverable_quantity = self._get_recoverable_cell_resource_capacity(
                x, y, material, resources
            )
            layers.append({
                "material": material,
                # ``remaining`` is the bounded trench/bore volume that must
                # be stripped to expose the next stratum.  The whole 100 m
                # grid cell is not removed to reach a detector-confirmed seam.
                "initial_quantity": quantity,
                "remaining": quantity,
                # Once this material is the requested production target, the
                # robot works laterally along the exposed face.  This separate
                # inventory carries the grid-area/depth mass balance.
                "recoverable_initial_quantity": recoverable_quantity,
                "recoverable_remaining": recoverable_quantity,
            })
            self.cell_resource_units[(x, y, material)] = recoverable_quantity

        geology = {"layers": layers, "current_index": 0}
        self.cell_geology[key] = geology
        self._expose_current_geology_layer(x, y)
        return geology

    def _current_geology_layer(self, x: int, y: int) -> dict | None:
        geology = self._get_cell_geology(x, y)
        index = int(geology.get("current_index", 0))
        layers = geology.get("layers", [])
        return layers[index] if 0 <= index < len(layers) else None

    def _expose_current_geology_layer(self, x: int, y: int) -> dict | None:
        geology = self.cell_geology.get((x, y))
        if geology is None:
            return None
        index = int(geology.get("current_index", 0))
        layers = geology.get("layers", [])
        if not (0 <= index < len(layers)):
            return None
        layer = layers[index]
        material = layer["material"]
        self.discovered_resources.setdefault(material, set()).add((x, y))
        self.revealed_cell_resources.add((x, y))
        return layer

    def _advance_depleted_geology_layer(self, x: int, y: int) -> dict | None:
        """Close the exhausted layer and expose exactly one layer below it."""
        geology = self._get_cell_geology(x, y)
        index = int(geology.get("current_index", 0))
        layers = geology.get("layers", [])
        if index < len(layers):
            exhausted = layers[index]
            material = exhausted["material"]
            self.depleted_cell_resources.add((x, y, material))
            coords = self.discovered_resources.get(material)
            if coords is not None:
                coords.discard((x, y))
                if not coords:
                    self.discovered_resources.pop(material, None)
        geology["current_index"] = index + 1
        return self._expose_current_geology_layer(x, y)

    def _robot_dispatch_target(self, requested_resource: str) -> dict | None:
        """Find the exposed face above a detector-confirmed recovery target.

        The requested material and the material exposed at the bucket wheel
        are deliberately separate.  A detector hit can therefore anchor a
        recovery face while successive loose overburden layers are removed,
        but it never lets the excavator skip a layer or prospect a new cell.
        Consolidated basalt is eligible only when the delivered excavator
        configuration includes its audited hard-rock attachment.
        """
        if self.surface_fleet.available_excavator() is None:
            return None
        hard_rock_capable = bool(
            self.surface_fleet.excavator_spec.get("hard_rock_capable", False)
        )

        def available_units(layer: dict | None, target: str) -> int:
            """Return production inventory for a seam, trench inventory otherwise."""
            if not layer:
                return 0
            if (
                str(layer.get("material", "")) == str(target)
                and "recoverable_remaining" in layer
            ):
                return max(0, int(layer.get("recoverable_remaining", 0)))
            return max(0, int(layer.get("remaining", 0)))

        occupied = {
            (vehicle.job.target_x, vehicle.job.target_y)
            for vehicle in self.surface_fleet.excavators
            if vehicle.job is not None
        }

        cached = getattr(self, "_robot_dispatch_cache", {}).get(
            requested_resource
        )
        if isinstance(cached, dict):
            cached_xy = (int(cached["x"]), int(cached["y"]))
            # An in-flight layer job owns this detector-confirmed face.  Do
            # not abandon it while it is travelling, excavating or returning.
            # Other idle vehicles may still open another independently
            # verified face; otherwise a six-unit fleet can never work one
            # bulk shielding order in parallel.
            if cached_xy not in occupied:
                geology = self.cell_geology.get(cached_xy)
                if geology is None:
                    current_material = "regolith"
                    remaining = 1
                    target_still_below = requested_resource == "regolith"
                else:
                    index = int(geology.get("current_index", 0))
                    layers = geology.get("layers", [])
                    layer = layers[index] if 0 <= index < len(layers) else None
                    current_material = str(layer.get("material", "")) if layer else ""
                    remaining = available_units(layer, requested_resource)
                    target_still_below = any(
                        str(candidate.get("material", "")) == requested_resource
                        and available_units(candidate, requested_resource) > 0
                        for candidate in layers[index:]
                    )
                if (
                    not self._is_construction_protected_cell(*cached_xy)
                    and remaining > 0
                    and target_still_below
                    and (current_material != "basalt" or hard_rock_capable)
                    and self.surface_fleet.excavator_dispatch_feasibility(
                        *cached_xy
                    ).get("feasible", False)
                ):
                    return {
                        **cached,
                        "resource": current_material,
                        "requested_resource": requested_resource,
                        "overburden": current_material != requested_resource,
                    }
                # The seam was exhausted, the site became protected, or no
                # valid physical layer remains. Only then discard this face.
                self._robot_dispatch_cache.pop(requested_resource, None)

        # Prefer a face where the requested material is already physically
        # exposed.  If a detector verified it immediately below the current
        # face, remove only that overburden first.
        coordinates = list(self.discovered_resources.get(requested_resource, set()))
        candidates: list[tuple[int, int, int, int, str]] = []
        for x, y in coordinates:
            if (x, y) in occupied or self._is_construction_protected_cell(x, y):
                continue
            # Target selection is read-only. Calling _current_geology_layer
            # here would materialize a new stack and mark its first layer as
            # discovered merely because the dispatcher inspected candidates.
            geology = self.cell_geology.get((x, y))
            if geology is None:
                if requested_resource != "regolith":
                    continue
                material = "regolith"
                remaining = 1
            else:
                index = int(geology.get("current_index", 0))
                layers = geology.get("layers", [])
                current = layers[index] if 0 <= index < len(layers) else None
                if current is None:
                    continue
                material = str(current.get("material", ""))
                remaining = available_units(current, requested_resource)
            if remaining <= 0:
                continue
            if geology is not None:
                target_still_below = any(
                    str(candidate.get("material", "")) == requested_resource
                    and available_units(candidate, requested_resource) > 0
                    for candidate in layers[index:]
                )
                if not target_still_below:
                    continue
            # Basalt requires the serialized ripper/crusher attachment; a
            # mission profile without it cannot silently mine hard rock.
            if material == "basalt" and not hard_rock_capable:
                continue
            if not self.surface_fleet.excavator_dispatch_feasibility(
                x, y
            ).get("feasible", False):
                continue
            distance = max(abs(x - self.lz_x), abs(y - self.lz_y))
            candidates.append((0 if material == requested_resource else 1,
                               distance, x, y, material))

        # Regolith itself is visually verified without a subsurface detector.
        if requested_resource == "regolith" and not candidates:
            for x, y in self.discovered_resources.get("regolith", set()):
                if (x, y) in occupied or self._is_construction_protected_cell(x, y):
                    continue
                geology = self.cell_geology.get((x, y))
                if geology is None:
                    is_exposed_regolith = True
                else:
                    index = int(geology.get("current_index", 0))
                    layers = geology.get("layers", [])
                    current = layers[index] if 0 <= index < len(layers) else None
                    is_exposed_regolith = bool(
                        current
                        and current.get("material") == "regolith"
                        and available_units(current, "regolith") > 0
                    )
                if is_exposed_regolith:
                    if not self.surface_fleet.excavator_dispatch_feasibility(
                        x, y
                    ).get("feasible", False):
                        continue
                    distance = max(abs(x - self.lz_x), abs(y - self.lz_y))
                    candidates.append((0, distance, x, y, "regolith"))

        if not candidates:
            return None
        _, _, x, y, exposed_resource = min(candidates)
        selected = {
            "resource": exposed_resource,
            "requested_resource": requested_resource,
            "overburden": exposed_resource != requested_resource,
            "x": int(x),
            "y": int(y),
            "verified_exposed_face": True,
            "protected_ground": False,
        }
        if not hasattr(self, "_robot_dispatch_cache"):
            self._robot_dispatch_cache = {}
        self._robot_dispatch_cache[requested_resource] = dict(selected)
        return selected

    def _apply_failed_excavator_fallback(
        self,
        agent: Agent,
        dispatch_target: dict,
        failure_reason: str,
    ) -> None:
        """Release a failed robot command to its validated crew work order."""
        fallback = dispatch_target.get("human_fallback", {})
        fallback_target = (
            fallback.get("target", {}) if isinstance(fallback, dict) else {}
        )
        if not isinstance(fallback_target, dict):
            fallback_target = {}
        if not fallback_target.get("resource"):
            fallback_target["resource"] = dispatch_target.get(
                "requested_resource",
                dispatch_target.get("resource", "regolith"),
            )
        fallback_target = {
            **fallback_target,
            "robot_dispatch_failed": str(failure_reason),
        }
        agent.action.action_type = "gather"
        agent.action.target = fallback_target
        # Zero means the ordinary decision/execution path immediately owns
        # the next tick; there is no fabricated one-tick GATHER completion.
        agent.action.ticks_remaining = 0
        agent.last_decision = {
            "action": "gather",
            "target": dict(fallback_target),
            "reasoning": (
                fallback.get("reasoning")
                if isinstance(fallback, dict) and fallback.get("reasoning")
                else "Excavator unavailable; continuing the human work order"
            ),
            "tick": self.current_tick,
            "deterministic": True,
            "fallback": True,
            "source": "excavator_dispatch_fallback",
        }

    def _fleet_excavate(
        self, job: FleetJob, requested_mass_kg: float
    ) -> tuple[int, float, bool]:
        """Remove real units from one exposed geology layer."""
        x, y = int(job.target_x), int(job.target_y)
        if self._is_construction_protected_cell(x, y):
            return 0, 0.0, False
        layer = self._current_geology_layer(x, y)
        if (
            layer is None
            or layer.get("material") != job.resource
            or (x, y) not in self.discovered_resources.get(job.resource, set())
            or (
                job.resource == "basalt"
                and not bool(self.surface_fleet.excavator_spec.get(
                    "hard_rock_capable", False
                ))
            )
        ):
            return 0, 0.0, False
        density = float(MATERIAL_DENSITY_KG.get(job.resource, 2.0))
        production_face = str(job.objective_resource) == str(job.resource)
        inventory_key = (
            "recoverable_remaining"
            if production_face and "recoverable_remaining" in layer
            else "remaining"
        )
        units = min(
            int(layer.get(inventory_key, 0)),
            max(0, int(float(requested_mass_kg) // max(0.001, density))),
        )
        if units <= 0:
            return 0, 0.0, False
        remaining = max(0, int(layer.get(inventory_key, 0)) - units)
        layer[inventory_key] = remaining
        self.cell_resource_units[(x, y, job.resource)] = remaining
        depleted = remaining <= 0
        if depleted and inventory_key == "remaining":
            geology = self._get_cell_geology(x, y)
            self.cell_excavation_depth[(x, y)] = max(
                int(self.cell_excavation_depth.get((x, y), 0)),
                int(geology.get("current_index", 0)) + 1,
            )
            self._advance_depleted_geology_layer(x, y)
        return units, units * density, depleted

    def _unload_fleet_payload(self, payload: dict[str, int]) -> None:
        for resource, quantity in payload.items():
            if int(quantity) > 0:
                self.central_depot_inventory[resource] = (
                    self.central_depot_inventory.get(resource, 0) + int(quantity)
                )

    def _consume_fleet_service_module(
        self, material: str, quantity: int
    ) -> bool:
        """Consume one serialized vehicle spare only at the central base."""
        quantity = max(0, int(quantity))
        available = max(
            0, int(self.central_depot_inventory.get(str(material), 0))
        )
        if quantity <= 0 or available < quantity:
            return False
        remaining = available - quantity
        if remaining:
            self.central_depot_inventory[str(material)] = remaining
        else:
            self.central_depot_inventory.pop(str(material), None)
        return True

    def _tick_surface_fleet(self) -> list[dict]:
        result = self.surface_fleet.tick(
            available_grid_energy_kwh=max(
                0.0, float(self._colony_resources.get("energy_stored_kwh", 0.0))
            ),
            excavation_callback=self._fleet_excavate,
            unload_callback=self._unload_fleet_payload,
            service_callback=self._consume_fleet_service_module,
        )
        draw = float(result.get("grid_energy_draw_kwh", 0.0))
        self._colony_resources["energy_stored_kwh"] = max(
            0.0,
            float(self._colony_resources.get("energy_stored_kwh", 0.0)) - draw,
        )
        for event in result.get("events", []):
            if event.get("type") == "cargo_delivery_complete":
                self._receive_construction_cargo(event)
            if event.get("type") == "excavator_job_complete":
                job = event.get("job") or {}
                dispatcher = next(
                    (
                        crew for crew in self.agents
                        if crew.id == job.get("dispatched_by")
                    ),
                    None,
                )
                if dispatcher is not None:
                    delivered_mass = float(job.get("extracted_mass_kg", 0.0))
                    reward = min(12.0, 1.0 + delivered_mass / 20.0)
                    self.decision_engine.apply_delayed_rl_reward(
                        dispatcher,
                        str(job.get("rl_state_key", "")),
                        str(job.get("rl_action_key", "")),
                        reward if delivered_mass > 0 else -2.0,
                    )
            if self._on_event:
                self._on_event({**event, "tick": self.current_tick})
        return list(result.get("events", []))

    def _spoil_drop_cell(self, excavation_x: int, excavation_y: int, _quantity: int) -> tuple[int, int]:
        """Choose an adjacent, structure-free grid for excavated material."""
        offsets = (
            (1, 0), (0, 1), (-1, 0), (0, -1),
            (1, 1), (-1, 1), (-1, -1), (1, -1),
        )
        candidates: list[tuple[int, int]] = []
        for dx, dy in offsets:
            px, py = excavation_x + dx, excavation_y + dy
            if not (0 <= px < self.world.map_size and 0 <= py < self.world.map_size):
                continue
            if any(
                structure.get("x") == px
                and structure.get("y") == py
                and not structure.get("destroyed", False)
                for structure in getattr(self, "placed_structures", [])
            ):
                continue
            if self._is_construction_protected_cell(px, py):
                continue
            if max(abs(px - self.lz_x), abs(py - self.lz_y)) <= 1:
                continue
            candidates.append((px, py))

        # Keep using the same designated neighboring dump grid so deposited
        # materials form an ordered physical stack. The simulation has no
        # calibrated grid-area/heap-angle model, so do not invent an arbitrary
        # pile-capacity number here.
        for coord in candidates:
            pile = self.spoil_piles.get(coord)
            if pile:
                return coord
        for coord in candidates:
            if coord not in self.spoil_piles:
                return coord
        if candidates:
            return candidates[0]
        # Returning the excavation cell here silently put loose overburden on
        # top of the active face (and, for an invalid face, inside a protected
        # habitat or landing-zone envelope).  Mining callers are required to
        # select a legal face with at least one adjacent spoil pad; failing
        # closed preserves both material placement and civil-site safety.
        raise RuntimeError(
            "no safe adjacent spoil pad for excavation face "
            f"({excavation_x}, {excavation_y})"
        )

    def _add_to_spoil_pile(
        self, x: int, y: int, material: str, quantity: int
    ) -> tuple[int, int]:
        if quantity <= 0:
            return x, y
        drop_coord = self._spoil_drop_cell(x, y, int(quantity))
        pile = self.spoil_piles.setdefault(drop_coord, {})
        # Dict insertion order is the physical bottom-to-top stack order.
        # Once a named layer exists, adding more of that material changes only
        # its quantity. Popping and reinserting it made labels jump between
        # rows whenever two haulers revisited the same pile.
        pile[material] = pile.get(material, 0) + int(quantity)
        return drop_coord

    def _spoil_coordinates(self, material: str) -> set[tuple[int, int]]:
        return {
            coord for coord, pile in self.spoil_piles.items()
            if pile.get(material, 0) > 0
        }

    def _is_construction_protected_cell(self, x: int, y: int) -> bool:
        """Protect the planned campus and every built site's service apron."""
        lz_x = getattr(self, "lz_x", self.world.center)
        lz_y = getattr(self, "lz_y", self.world.center)
        protected_radius = (
            self.CONSTRUCTION_CAMPUS_RADIUS_CELLS
            + self.CONSTRUCTION_GROUND_BUFFER_CELLS
        )
        if max(abs(int(x) - lz_x), abs(int(y) - lz_y)) <= protected_radius:
            return True
        landing_profile = self._structure_site_profile("landing_zone")
        landing_dx, landing_dy = self._site_plan_offset(
            "landing_zone", landing_profile
        )
        planned_landing_cells = self._structure_footprint_cells(
            "landing_zone", lz_x + int(landing_dx), lz_y + int(landing_dy)
        )
        if min(
            max(abs(int(x) - lx), abs(int(y) - ly))
            for lx, ly in planned_landing_cells
        ) <= self._structure_pair_clearance_cells("landing_zone", "structure"):
            return True
        return any(
            not structure.get("destroyed", False)
            and max(
                abs(int(x) - int(structure.get("x", x))),
                abs(int(y) - int(structure.get("y", y))),
            ) <= self.CONSTRUCTION_GROUND_BUFFER_CELLS
            for structure in getattr(self, "placed_structures", [])
        )

    def _structure_site_profile(self, recipe_name: str) -> dict:
        """Return the deterministic civil-layout envelope for one asset."""
        profiles = {
            "storage_crate": {
                "zone": "logistics_yard", "preferred_offset": (-5, 4),
                "preferred_offsets": [(-5, 4), (-7, 5), (-4, 7)],
                "min_radius": 4, "max_radius": 10,
                "render_scale": 0.56,
                "related_types": {"eclss_lander_hub", "forge"},
            },
            "life_support_distribution_grid": {
                "zone": "utility_spine", "preferred_offset": (-4, -4),
                "preferred_offsets": [
                    (-4, -4), (5, -5), (-6, 6), (6, 6), (1, -10),
                ],
                "min_radius": 4, "max_radius": 12,
                "render_scale": 0.58,
                "same_type_spacing_cells": 3,
                "related_types": {"eclss_lander_hub", "habitat_module"},
            },
            "potable_water_tank": {
                "zone": "water_tank_farm", "preferred_offset": (-7, -6),
                "preferred_offsets": [(-7, -6), (-10, -6), (-7, -10)],
                "min_radius": 6, "max_radius": 14,
                "render_scale": 0.58,
                "same_type_spacing_cells": 3,
                "related_types": {"life_support_distribution_grid", "water_collector", "water_purifier"},
            },
            "oxygen_buffer_tank": {
                "zone": "oxygen_tank_farm", "preferred_offset": (-9, -4),
                "preferred_offsets": [(-9, -4), (-11, -2), (-10, -7)],
                "min_radius": 7, "max_radius": 14,
                "render_scale": 0.58,
                "same_type_spacing_cells": 3,
                "related_types": {"life_support_distribution_grid", "isru_o2_unit"},
            },
            "habitat_module": {
                "zone": "habitation", "preferred_offset": (7, -5),
                "preferred_offsets": [(7, -5), (10, -6), (7, -9)],
                "min_radius": 6, "max_radius": 14,
                "render_scale": 0.72,
                "same_type_spacing_cells": 3,
                "related_types": {"eclss_lander_hub", "habitat_module"},
            },
            "medical_station": {
                "zone": "habitation", "preferred_offset": (5, -4),
                "min_radius": 5, "max_radius": 11,
                "render_scale": 0.64,
                "related_types": {"eclss_lander_hub", "habitat_module"},
            },
            "water_collector": {
                "zone": "water_process_yard", "preferred_offset": (-7, -9),
                "preferred_offsets": [(-7, -9), (-10, -9), (-8, -12)],
                "min_radius": 7, "max_radius": 15,
                "render_scale": 0.66,
                "same_type_spacing_cells": 3,
                "related_types": {"life_support_distribution_grid", "water_purifier"},
            },
            "water_purifier": {
                "zone": "water_process_yard", "preferred_offset": (-6, -8),
                "preferred_offsets": [(-6, -8), (-9, -7), (-7, -11)],
                "min_radius": 6, "max_radius": 14,
                "render_scale": 0.60,
                "same_type_spacing_cells": 3,
                "related_types": {"water_collector", "life_support_distribution_grid"},
            },
            "isru_o2_unit": {
                "zone": "oxygen_process_yard", "preferred_offset": (-9, -6),
                "preferred_offsets": [(-9, -6), (-12, -5), (-10, -9)],
                "min_radius": 7, "max_radius": 15,
                "render_scale": 0.64,
                "same_type_spacing_cells": 3,
                "related_types": {"oxygen_buffer_tank", "life_support_distribution_grid"},
            },
            "greenhouse": {
                "zone": "agriculture", "preferred_offset": (8, 9),
                "preferred_offsets": [(8, 9), (11, 8), (9, 12)],
                "min_radius": 8, "max_radius": 15,
                "render_scale": 0.78,
                "same_type_spacing_cells": 3,
                "related_types": {"greenhouse", "life_support_distribution_grid"},
            },
            "power_distribution_grid": {
                "zone": "power_yard", "preferred_offset": (-7, 7),
                "preferred_offsets": [(-7, 7), (-10, 6), (-8, 10)],
                "min_radius": 6, "max_radius": 14,
                "render_scale": 0.58,
                "same_type_spacing_cells": 3,
                "related_types": {"solar_panel", "eclss_lander_hub"},
            },
            "solar_panel": {
                "zone": "solar_field", "preferred_offset": (-11, 11),
                "preferred_offsets": [(-11, 11), (-14, 10), (-10, 14)],
                "min_radius": 10, "max_radius": 15,
                "render_scale": 0.82,
                "same_type_spacing_cells": 2,
                "related_types": {"solar_panel", "power_distribution_grid"},
            },
            "communications_array": {
                "zone": "communications", "preferred_offset": (2, -12),
                "min_radius": 10, "max_radius": 15,
                "render_scale": 0.54,
                "related_types": {"power_distribution_grid", "eclss_lander_hub"},
            },
            "landing_zone": {
                "zone": "arrival_landing_ellipse", "preferred_offset": (20, 0),
                "min_radius": 18, "max_radius": 24,
                "footprint_half_width_cells": 2,
                "footprint_half_height_cells": 1,
                "render_scale": 0.92,
                "related_types": {"communications_array"},
            },
        }
        default = {
            "zone": "industrial_yard",
            "preferred_offset": (-6, 3),
            "preferred_offsets": [(-6, 3), (-9, 4), (-6, 7)],
            "min_radius": 5,
            "max_radius": 13,
            "render_scale": 0.66,
            "related_types": {"eclss_lander_hub", "forge", "stone_furnace"},
        }
        specific = profiles.get(str(recipe_name), {})
        resolved = {**default, **specific}
        if "preferred_offsets" not in specific:
            resolved["preferred_offsets"] = [resolved["preferred_offset"]]
        return resolved

    def _layout_seed_value(self, *parts: object) -> int:
        """Return a stable integer for reproducible civil-layout variation."""
        payload = "|".join((
            str(self.planet.id), str(self.seed), "campus-layout",
            *(str(part) for part in parts),
        ))
        return int.from_bytes(
            hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
        )

    def _site_plan_offset(
        self, recipe_name: str, profile: dict, occurrence: int = 0
    ) -> tuple[int, int]:
        """Rotate/mirror one zoned master plan for this episode's seed.

        This is deliberate design uncertainty, not nondeterministic physics:
        the same planet and seed reproduce the same parcels. The whole plan
        shares one transform, preserving separation between habitation,
        process, agriculture and power zones.
        """
        options = profile.get("preferred_offsets") or [
            profile.get("preferred_offset", (0, 0))
        ]
        raw_dx, raw_dy = options[int(occurrence) % len(options)]
        if recipe_name == "landing_zone":
            # The damaged ship's certified eastern approach corridor is known
            # before surface deployment and must remain clear of the delivered
            # western machine yard. Randomising occupied buildings is safe;
            # rotating an already-declared descent corridor is not.
            return int(raw_dx), int(raw_dy)
        variant = self._layout_seed_value("master-plan") % 8
        dx, dy = int(raw_dx), int(raw_dy)
        if variant >= 4:
            dx = -dx
        for _ in range(variant % 4):
            dx, dy = -dy, dx
        return dx, dy

    def _site_layout_noise(
        self, recipe_name: str, occurrence: int, x: int, y: int
    ) -> float:
        """Small seeded tie-break that never overrides a serious hazard."""
        return (
            self._layout_seed_value(recipe_name, occurrence, x, y) % 1000
        ) / 1000.0 * 5.0

    def _structure_footprint_cells(
        self, recipe_name: str, x: int, y: int, structure: dict | None = None
    ) -> set[tuple[int, int]]:
        profile = self._structure_site_profile(recipe_name)
        source = structure or {}
        half_x = int(source.get(
            "footprint_half_width_cells",
            profile.get("footprint_half_width_cells", 0),
        ))
        half_y = int(source.get(
            "footprint_half_height_cells",
            profile.get("footprint_half_height_cells", 0),
        ))
        min_dx = int(source.get(
            "footprint_min_dx_cells",
            profile.get("footprint_min_dx_cells", -half_x),
        ))
        max_dx = int(source.get(
            "footprint_max_dx_cells",
            profile.get("footprint_max_dx_cells", half_x),
        ))
        min_dy = int(source.get(
            "footprint_min_dy_cells",
            profile.get("footprint_min_dy_cells", -half_y),
        ))
        max_dy = int(source.get(
            "footprint_max_dy_cells",
            profile.get("footprint_max_dy_cells", half_y),
        ))
        if min_dx > max_dx or min_dy > max_dy:
            raise ValueError(f"invalid footprint bounds for {recipe_name}")
        return {
            (int(x) + dx, int(y) + dy)
            for dx in range(min_dx, max_dx + 1)
            for dy in range(min_dy, max_dy + 1)
        }

    @staticmethod
    def _structure_pair_clearance_cells(
        candidate_type: str, existing_type: str
    ) -> int:
        pair = {str(candidate_type), str(existing_type)}
        if "landing_zone" in pair:
            # A ten-cell clear strip is one kilometre on the configured grid,
            # outside the certified ellipse itself. It is a conservative
            # blast/debris exclusion zone, not an invented habitable radius.
            return 10
        if pair & {"oxygen_buffer_tank"} and pair & {
            "habitat_module", "greenhouse", "medical_station"
        }:
            return 2
        if pair & {"solar_panel"} and pair & {
            "habitat_module", "greenhouse"
        }:
            return 2
        if pair & {"forge", "stone_furnace"} and pair & {
            "habitat_module", "greenhouse", "medical_station"
        }:
            return 2
        # Adjacent parcel centres are already 100 m apart. Requiring another
        # empty parcel between every harmless pair would turn a compact base
        # into a multi-kilometre sprawl and consume fictitious pipe/cable.
        # Exact overlap is still forbidden; equal assets and hazardous pairs
        # receive the larger explicit clearances above, while the renderer
        # shows each asset's smaller real envelope inside its 100 m parcel.
        return 0

    @staticmethod
    def _footprint_chebyshev_gap(
        first_cells: set[tuple[int, int]],
        second_cells: set[tuple[int, int]],
    ) -> int:
        """Return exact edge gap for the rectangular structure footprints."""
        first_x = [cell[0] for cell in first_cells]
        first_y = [cell[1] for cell in first_cells]
        second_x = [cell[0] for cell in second_cells]
        second_y = [cell[1] for cell in second_cells]
        x_gap = max(
            min(second_x) - max(first_x),
            min(first_x) - max(second_x),
            0,
        )
        y_gap = max(
            min(second_y) - max(first_y),
            min(first_y) - max(second_y),
            0,
        )
        return max(x_gap, y_gap)

    def _site_footprint_is_clear(
        self, recipe_name: str, x: int, y: int, *,
        candidate_cells: set[tuple[int, int]] | None = None,
        existing_footprints: list[tuple[dict, set[tuple[int, int]]]] | None = None,
        future_landing_cells: set[tuple[int, int]] | None = None,
    ) -> bool:
        candidate_profile = self._structure_site_profile(recipe_name)
        candidate_cells = candidate_cells or self._structure_footprint_cells(
            recipe_name, x, y
        )
        if any(
            not (0 <= cx < self.world.map_size and 0 <= cy < self.world.map_size)
            for cx, cy in candidate_cells
        ):
            return False
        if recipe_name != "landing_zone" and hasattr(self, "lz_x"):
            # Two one-cell-wide cardinal spines remain free through the whole
            # campus. They stand in for graded rover/fire access and leave a
            # predictable unobstructed trunk for buried utilities. A 100 m
            # planning cell is coarse; the visible lane is narrower inside it.
            if any(
                cx == int(self.lz_x) or cy == int(self.lz_y)
                for cx, cy in candidate_cells
            ):
                return False
        if recipe_name != "landing_zone" and hasattr(self, "lz_x"):
            # The damaged ship's 100 civilians have a known arrival vehicle
            # before surface construction begins. Reserve its eastern landing
            # corridor from day one so a late-built certification pad cannot
            # discover that solar arrays or habitats occupy the blast zone.
            if future_landing_cells is None:
                landing_profile = self._structure_site_profile("landing_zone")
                landing_dx, landing_dy = self._site_plan_offset(
                    "landing_zone", landing_profile
                )
                future_landing_cells = self._structure_footprint_cells(
                    "landing_zone",
                    int(self.lz_x) + int(landing_dx),
                    int(self.lz_y) + int(landing_dy),
                )
            future_gap = self._footprint_chebyshev_gap(
                candidate_cells, future_landing_cells
            )
            if future_gap <= self._structure_pair_clearance_cells(
                recipe_name, "landing_zone"
            ):
                return False
        if existing_footprints is None:
            existing_footprints = [
                (
                    existing,
                    self._structure_footprint_cells(
                        str(existing.get("type", "structure")),
                        int(existing.get("x", x)),
                        int(existing.get("y", y)),
                        existing,
                    ),
                )
                for existing in getattr(self, "placed_structures", [])
                if not existing.get("destroyed", False)
            ]
        for existing, existing_cells in existing_footprints:
            existing_type = str(existing.get("type", "structure"))
            min_gap = self._footprint_chebyshev_gap(
                candidate_cells, existing_cells
            )
            clearance = self._structure_pair_clearance_cells(
                recipe_name, existing_type
            )
            if min_gap <= clearance:
                return False
        same_type_spacing = int(
            candidate_profile.get("same_type_spacing_cells", 1)
        )
        if same_type_spacing > 1 and any(
            existing.get("type") == recipe_name
            and max(
                abs(int(x) - int(existing.get("x", x))),
                abs(int(y) - int(existing.get("y", y))),
            ) < same_type_spacing
            for existing in getattr(self, "placed_structures", [])
            if not existing.get("destroyed", False)
        ):
            return False
        return True

    def _site_congestion_penalty(
        self, recipe_name: str, x: int, y: int,
        structures: list[dict], *,
        candidate_cells: set[tuple[int, int]] | None = None,
        existing_footprints: list[tuple[dict, set[tuple[int, int]]]] | None = None,
    ) -> float:
        """Prefer a service gap without making a full campus impossible.

        Adjacent 100 m parcels can be physically safe because the equipment
        occupies only part of each parcel.  A soft cost nevertheless keeps
        another parcel boundary between unrelated assets when the zoned yard
        has room, while late mission saturation can still use a valid parcel.
        """
        candidate_cells = candidate_cells or self._structure_footprint_cells(
            recipe_name, x, y
        )
        penalty = 0.0
        if existing_footprints is None:
            existing_footprints = [
                (
                    existing,
                    self._structure_footprint_cells(
                        str(existing.get("type", "structure")),
                        int(existing.get("x", x)),
                        int(existing.get("y", y)),
                        existing,
                    ),
                )
                for existing in structures
            ]
        for _existing, existing_cells in existing_footprints:
            min_gap = self._footprint_chebyshev_gap(
                candidate_cells, existing_cells
            )
            if min_gap == 1:
                penalty += 24.0
        return penalty

    def _select_structure_site(self, recipe_name: str) -> tuple[int, int]:
        """Choose the nearest safe, utility-connected grid for a new module."""
        lz_x = getattr(self, "lz_x", self.world.center)
        lz_y = getattr(self, "lz_y", self.world.center)
        if recipe_name == "greenhouse":
            # Utility availability is a campus-wide prerequisite. Reject it
            # before reading hundreds of terrain cells; the prior ordering
            # repeated the same impossible parcel search for every available
            # crew member on every planning tick while no grid existed.
            network = self._life_support_network_snapshot(
                reserve_pending=True
            )
            if (
                not network.get("nodes")
                or network["connected_endpoint_count"]
                >= network["maximum_endpoints"]
            ):
                raise RuntimeError(
                    "no safe utility-feasible surveyed campus site remains "
                    "for greenhouse"
                )
        profile = self._structure_site_profile(recipe_name)
        min_radius = int(profile["min_radius"])
        max_radius = int(profile["max_radius"])
        related_types = set(profile["related_types"])
        structures = [
            structure for structure in getattr(self, "placed_structures", [])
            if not structure.get("destroyed", False)
        ]
        existing_footprints = [
            (
                structure,
                self._structure_footprint_cells(
                    str(structure.get("type", "structure")),
                    int(structure.get("x", lz_x)),
                    int(structure.get("y", lz_y)),
                    structure,
                ),
            )
            for structure in structures
        ]
        future_landing_cells = None
        if recipe_name != "landing_zone":
            landing_profile = self._structure_site_profile("landing_zone")
            landing_dx, landing_dy = self._site_plan_offset(
                "landing_zone", landing_profile
            )
            future_landing_cells = self._structure_footprint_cells(
                "landing_zone",
                int(lz_x) + int(landing_dx),
                int(lz_y) + int(landing_dy),
            )
        same_type_structures = [
            structure for structure in structures
            if structure.get("type") == recipe_name
        ]
        occurrence = len(same_type_structures)
        preferred_dx, preferred_dy = self._site_plan_offset(
            recipe_name, profile, occurrence
        )
        preferred_x = lz_x + int(preferred_dx)
        preferred_y = lz_y + int(preferred_dy)
        anchors = [
            (structure.get("x", lz_x), structure.get("y", lz_y))
            for structure in structures
            if structure.get("type") in related_types
        ] or [(lz_x, lz_y)]

        disturbed_cells = set(self.spoil_piles) | {
            coord for coord, depth in self.cell_excavation_depth.items()
            if depth > 0
        }
        reservations = getattr(self, "_construction_cargo_reservations", {})
        site_state = tuple(sorted(
            (
                str(structure.get("id", "")),
                str(structure.get("type", "")),
                int(structure.get("x", lz_x)),
                int(structure.get("y", lz_y)),
                bool(structure.get("materials_committed", False)),
                str(reservations.get(
                    str(structure.get("id", "")), {}
                ).get("status", "")),
            )
            for structure in structures
        ))
        cache_key = (
            str(recipe_name),
            occurrence,
            site_state,
        )
        site_cache = getattr(self, "_structure_site_selection_cache", {})
        cached_site = site_cache.get(cache_key)
        if cached_site is not None:
            cached_x, cached_y = int(cached_site[0]), int(cached_site[1])
            cached_footprint = self._structure_footprint_cells(
                recipe_name, cached_x, cached_y
            )
            # Excavation elsewhere in a kilometre-scale campus must not force a
            # full terrain rescan.  Reuse the surveyed parcel while its own
            # footprint (and, for solar, dust apron) is still physically clear.
            cached_disturbed = any(
                cell in disturbed_cells for cell in cached_footprint
            )
            if recipe_name == "solar_panel":
                cached_disturbed = cached_disturbed or any(
                    max(abs(cached_x - dx), abs(cached_y - dy)) < 4
                    for dx, dy in disturbed_cells
                )
            cached_valid = (
                not cached_disturbed
                and self._site_footprint_is_clear(
                    recipe_name, cached_x, cached_y,
                    candidate_cells=cached_footprint,
                    existing_footprints=existing_footprints,
                    future_landing_cells=future_landing_cells,
                )
            )
            if cached_valid and recipe_name == "greenhouse":
                cached_valid = self._select_utility_feasible_site(
                    recipe_name, [(0.0, 0, cached_y, cached_x)]
                ) == (cached_x, cached_y)
            if cached_valid:
                return cached_x, cached_y
            site_cache.pop(cache_key, None)

        def remember_site(selected_site: tuple[int, int]) -> tuple[int, int]:
            # This is a state-keyed planning cache, not mutable physics state.
            # Bound it so a long campaign cannot retain every historical
            # excavation/layout combination forever.
            if len(site_cache) >= 256:
                site_cache.clear()
            site_cache[cache_key] = selected_site
            self._structure_site_selection_cache = site_cache
            return selected_site

        solar_field_anchor = (
            (
                int(same_type_structures[0].get("x", lz_x)),
                int(same_type_structures[0].get("y", lz_y)),
            )
            if recipe_name == "solar_panel" and same_type_structures else None
        )
        candidates = []
        cell_info_cache: dict[tuple[int, int], dict] = {}

        def cached_cell_info(cx: int, cy: int) -> dict:
            coord = (int(cx), int(cy))
            if coord not in cell_info_cache:
                cell_info_cache[coord] = self.world.get_cell_info(
                    coord[0], coord[1], self.current_tick
                )
            return cell_info_cache[coord]

        for y in range(max(0, lz_y - max_radius), min(self.world.map_size, lz_y + max_radius + 1)):
            for x in range(max(0, lz_x - max_radius), min(self.world.map_size, lz_x + max_radius + 1)):
                radius = max(abs(x - lz_x), abs(y - lz_y))
                if radius < min_radius or radius > max_radius:
                    continue
                if (
                    recipe_name != "landing_zone"
                    and radius > self.CONSTRUCTION_CAMPUS_RADIUS_CELLS
                ):
                    continue
                footprint = self._structure_footprint_cells(
                    recipe_name, x, y
                )
                if not self._site_footprint_is_clear(
                    recipe_name, x, y,
                    candidate_cells=footprint,
                    existing_footprints=existing_footprints,
                    future_landing_cells=future_landing_cells,
                ):
                    continue
                if any(cell in self.spoil_piles for cell in footprint):
                    continue
                if any(
                    self.cell_excavation_depth.get(cell, 0) > 0
                    for cell in footprint
                ):
                    continue
                footprint_info = [
                    cached_cell_info(cx, cy)
                    for cx, cy in footprint
                ]
                if not all(
                    cell.get("traversable", True) for cell in footprint_info
                ):
                    continue
                cell_info = cached_cell_info(x, y)
                if recipe_name == "solar_panel":
                    # Arrays need a clear solar field, not the dust, traffic,
                    # shadow and maintenance envelope beside a furnace/CNC or
                    # habitat wall. A four-cell service/cable apron separates
                    # the field from occupied modules; excavation and spoil get
                    # a three-cell dust/settlement apron.
                    if any(
                        structure.get("type") != "solar_panel"
                        and max(
                            abs(x - int(structure.get("x", lz_x))),
                            abs(y - int(structure.get("y", lz_y))),
                        ) < 5
                        for structure in structures
                    ):
                        continue
                    if any(
                        max(abs(x - dx), abs(y - dy)) < 4
                        for dx, dy in disturbed_cells
                    ):
                        continue
                related_distance = min(
                    max(abs(x - ax), abs(y - ay)) for ax, ay in anchors
                )
                same_type_distance = min((
                    max(abs(x - structure.get("x", x)), abs(y - structure.get("y", y)))
                    for structure in same_type_structures
                ), default=99)
                congestion_penalty = self._site_congestion_penalty(
                    recipe_name, x, y, structures,
                    candidate_cells=footprint,
                    existing_footprints=existing_footprints,
                )
                layout_noise = self._site_layout_noise(
                    recipe_name, occurrence, x, y
                )
                elevation = float(cell_info.get("elevation", 0.5))
                neighbor_elevations = [
                    float(cached_cell_info(nx, ny).get("elevation", 0.5))
                    for nx, ny in (
                        (max(0, x - 1), y),
                        (min(self.world.map_size - 1, x + 1), y),
                        (x, max(0, y - 1)),
                        (x, min(self.world.map_size - 1, y + 1)),
                    )
                ]
                local_slope = max(
                    (abs(elevation - neighbor) for neighbor in neighbor_elevations),
                    default=0.0,
                )
                hazard_penalty = sum(
                    abs(float(value))
                    for value in cell_info.get("hazard_modifiers", {}).values()
                    if isinstance(value, (int, float))
                )
                terrain_penalty = (
                    max(0.0, float(cell_info.get("traversal_cost", 1.0)) - 1.0)
                    * 5.0
                    + local_slope * 12.0
                    + hazard_penalty * 3.0
                )
                # Equal arrays retain one clear grid for access and cable
                # routing. A planned solar field uses a stable two-cell lattice
                # instead of choosing whichever base-adjacent square is nearest.
                if (
                    recipe_name == "solar_panel"
                    and same_type_distance
                    < int(profile.get("same_type_spacing_cells", 2))
                ):
                    continue
                zone_distance = abs(x - preferred_x) + abs(y - preferred_y)
                if recipe_name == "solar_panel":
                    light_level = float(
                        cell_info.get("light", {}).get("light_level", 1.0)
                    )
                    solar_terrain_penalty = terrain_penalty - light_level
                    if solar_field_anchor is None:
                        preferred_radius = max(
                            abs(int(preferred_dx)), abs(int(preferred_dy))
                        )
                        layout_penalty = abs(radius - preferred_radius) * 4.0
                    else:
                        anchor_dx = abs(x - solar_field_anchor[0])
                        anchor_dy = abs(y - solar_field_anchor[1])
                        on_field_lattice = anchor_dx % 2 == 0 and anchor_dy % 2 == 0
                        layout_penalty = (
                            (0.0 if on_field_lattice else 20.0)
                            + abs(same_type_distance - 2) * 5.0
                        )
                    candidates.append((
                        zone_distance * 3.0 + layout_penalty
                        + solar_terrain_penalty + layout_noise,
                        radius,
                        y,
                        x,
                    ))
                elif recipe_name == "landing_zone":
                    elevations = [
                        float(cell.get("elevation", 0.5))
                        for cell in footprint_info
                    ]
                    relief = max(elevations) - min(elevations)
                    traversal = sum(
                        float(cell.get("traversal_cost", 1.0))
                        for cell in footprint_info
                    ) / max(1, len(footprint_info))
                    candidates.append((
                        zone_distance * 8.0 + relief * 100.0 + traversal
                        + layout_noise,
                        radius,
                        y,
                        x,
                    ))
                else:
                    candidates.append((
                        zone_distance * 6.0
                        + related_distance * 1.5
                        + radius
                        + congestion_penalty
                        # Seeded parcel preference and real terrain only
                        # resolve otherwise comparable sites.
                        + terrain_penalty + layout_noise,
                        radius,
                        y,
                        x,
                    ))
        selected = self._select_utility_feasible_site(recipe_name, candidates)
        if selected is not None:
            return remember_site(selected)
        if not candidates or recipe_name == "greenhouse":
            # A full preferred zone may expand into the remaining surveyed
            # campus, but it must never silently stack a second asset on an
            # occupied coordinate.  The previous preferred-coordinate
            # fallback was the source of late-mission overlapping buildings.
            expanded = []
            campus_radius = self.CONSTRUCTION_CAMPUS_RADIUS_CELLS
            for y in range(
                max(0, lz_y - campus_radius),
                min(self.world.map_size, lz_y + campus_radius + 1),
            ):
                for x in range(
                    max(0, lz_x - campus_radius),
                    min(self.world.map_size, lz_x + campus_radius + 1),
                ):
                    radius = max(abs(x - lz_x), abs(y - lz_y))
                    if radius < min_radius:
                        continue
                    footprint = self._structure_footprint_cells(
                        recipe_name, x, y
                    )
                    if not self._site_footprint_is_clear(
                        recipe_name, x, y,
                        candidate_cells=footprint,
                        existing_footprints=existing_footprints,
                        future_landing_cells=future_landing_cells,
                    ):
                        continue
                    if any(cell in disturbed_cells for cell in footprint):
                        continue
                    if not all(
                        cached_cell_info(cx, cy).get("traversable", True)
                        for cx, cy in footprint
                    ):
                        continue
                    expanded.append((
                        abs(x - preferred_x) + abs(y - preferred_y)
                        + self._site_congestion_penalty(
                            recipe_name, x, y, structures,
                            candidate_cells=footprint,
                            existing_footprints=existing_footprints,
                        )
                        + self._site_layout_noise(
                            recipe_name, occurrence, x, y
                        ),
                        radius,
                        y,
                        x,
                    ))
            selected = self._select_utility_feasible_site(recipe_name, expanded)
            if selected is None:
                raise RuntimeError(
                    f"no safe utility-feasible surveyed campus site remains for {recipe_name}"
                )
            return remember_site(selected)

    def _select_utility_feasible_site(
        self, recipe_name: str, candidates: list[tuple]
    ) -> tuple[int, int] | None:
        """Keep farm siting inside the actual finite, routed utility network.

        Zone proximity is only a preference. Preview commissioning with the
        same router used by production, without changing the live network or
        stealing an already connected consumer's corridor/port allocation.
        Power may return before commissioning, so siting requires a physical
        route; crew work separately requires that route to be powered.
        """
        if not candidates:
            return None
        if recipe_name != "greenhouse":
            _, _, y, x = min(candidates)
            return x, y
        network = self._life_support_network_snapshot(reserve_pending=True)
        nodes = network.get("nodes", [])
        if (
            not nodes
            or network["connected_endpoint_count"] >= network["maximum_endpoints"]
        ):
            return None
        service_radius = int(network["service_radius_cells"])
        grid_cell_m = float(network["grid_cell_meters"])
        maximum_length_m = float(network["maximum_length_m"])
        network_cells = {
            (int(node["x"]), int(node["y"])) for node in nodes
        }
        existing_edges = set()
        for route in network.get("routes", []):
            path = [
                (int(point["x"]), int(point["y"]))
                for point in route.get("path", [])
            ]
            network_cells.update(path)
            existing_edges.update(
                tuple(sorted((first, second)))
                for first, second in zip(path, path[1:])
            )

        # Build the static obstruction map once. The previous implementation
        # rebuilt and rerouted the complete existing utility network for every
        # candidate parcel. A saturated campus therefore spent minutes inside
        # one physics tick even though every accepted route is an append-only
        # extension that must preserve all commissioned consumers.
        routed_structures = [
            structure
            for structure in getattr(self, "placed_structures", [])
            if not structure.get("under_construction", False)
            and not structure.get("destroyed", False)
            and float(structure.get("health", 1.0)) > 0.0
        ]
        consumer_types = {
            "water_collector", "water_purifier", "greenhouse", "hydroponics",
            "isru_o2_unit", "potable_water_tank", "oxygen_buffer_tank",
            "habitat_module", "medical_station",
        }
        reservations = getattr(self, "_construction_cargo_reservations", {})
        for structure in getattr(self, "placed_structures", []):
            reservation = reservations.get(str(structure.get("id", "")), {})
            funded = bool(structure.get("materials_committed", False)) or bool(
                float(reservation.get("total_mass_kg", 0.0)) > 0.0
                and reservation.get("status") not in {"stranded", "cancelled"}
            )
            if (
                funded and structure.get("under_construction", False)
                and structure.get("type") in consumer_types
                and not structure.get("destroyed", False)
                and float(structure.get("health", 1.0)) > 0.0
            ):
                routed_structures.append(structure)
        base_blocked: set[tuple[int, int]] = set(
            getattr(self, "spoil_piles", {})
        )
        base_blocked.update(
            coord for coord, depth in getattr(
                self, "cell_excavation_depth", {}
            ).items() if depth > 0
        )
        for structure in routed_structures:
            base_blocked.update(self._structure_footprint_cells(
                str(structure.get("type", "structure")),
                int(structure.get("x", 0)),
                int(structure.get("y", 0)),
                structure,
            ))
        base_blocked.difference_update(network_cells)

        for _, _, y, x in sorted(candidates):
            if min(
                max(abs(x - node["x"]), abs(y - node["y"])) for node in nodes
            ) > service_radius:
                continue
            start = (int(x), int(y))
            footprint = self._structure_footprint_cells(
                "greenhouse", start[0], start[1]
            )
            route = self._route_utility_trench(
                start,
                network_cells,
                (base_blocked | footprint) - footprint,
                footprint,
            )
            if not route:
                continue
            route_edges = {
                tuple(sorted((first, second)))
                for first, second in zip(route, route[1:])
            }
            projected_length_m = (
                len(existing_edges | route_edges) * grid_cell_m
            )
            if projected_length_m <= maximum_length_m + 1e-9:
                return int(x), int(y)
        return None

    def _construction_productivity(self, agent: Agent) -> float:
        """Effective person-hours contributed per scheduled person-hour."""
        engineering = float(getattr(agent.competency, "engineering", 5))
        strength = float(getattr(agent.genome, "strength", 5))
        base = 0.62 + engineering * 0.045 + strength * 0.02
        energy_factor = max(0.45, min(1.0, agent.needs.energy / 80.0))
        gravity = max(0.3, float(getattr(agent, "gravity_g", 1.0)))
        gravity_factor = min(1.15, 1.0 / math.sqrt(gravity))
        return max(0.35, min(1.35, base * energy_factor * gravity_factor))

    def _movement_effort_state(
        self,
        agent: Agent,
        remaining_cells: int,
        terrain_cost: float,
    ) -> str:
        """Return a compact state for the independent travel-effort policy."""
        if remaining_cells <= 2:
            distance_band = "short"
        elif remaining_cells <= 8:
            distance_band = "medium"
        else:
            distance_band = "long"
        energy_band = "fresh" if agent.needs.energy >= 90.0 else "ready"
        reserve_band = (
            "full"
            if agent.needs.thirst >= 88.0 and agent.needs.o2_supply >= 82.0
            else "adequate"
        )
        terrain_band = "clear" if terrain_cost <= 1.20 else "costly"
        return (
            "effort-v1|activity:surface_move"
            f"|distance:{distance_band}|energy:{energy_band}"
            f"|reserve:{reserve_band}|terrain:{terrain_band}"
        )

    def _surge_movement_allowed(
        self,
        agent: Agent,
        target: dict,
        terrain_cost: float,
        projected_ticks: int = 1,
    ) -> bool:
        """Apply a route-end medical/suit envelope before RL may surge."""
        emergency_route = any(bool(target.get(flag)) for flag in (
            "forced_return", "dehydration_return", "fatigue_return",
            "o2_return", "storm_return", "medical_emergency",
        ))
        o2_return = float(getattr(
            agent, "_routine_eva_o2_return_threshold", 40.0
        ))
        overload = agent.operational_load_fraction(
            float(getattr(agent, "gravity_g", self.planet.gravity_g))
        )
        # The contract is checked again every movement tick. Project the
        # actual configured metabolic load to the route end, then retain a
        # small reserve above the ordinary EVA-return triggers. This permits
        # a deliberate fatigue trade while still masking surge for illness,
        # injury, overload, emergencies or a route that would consume its
        # homeward reserve.
        exposure_ticks = max(0, int(projected_ticks) - 1)
        surge_activity = (
            1.5 * self.MOVEMENT_SURGE_METABOLIC_MULTIPLIER
        )
        projected_energy_margin = (
            self._estimated_eva_energy_drain_per_tick(agent, surge_activity)
            * 1.10 * exposure_ticks
        )
        projected_thirst_margin = (
            float(agent.needs.BASE_THIRST_DECAY)
            * surge_activity * 1.25 * exposure_ticks
        )
        projected_hunger_margin = (
            float(agent.needs.BASE_HUNGER_DECAY)
            * surge_activity * 1.20 * exposure_ticks
        )
        projected_o2_margin = (
            float(agent.plss_o2_percent_per_tick(surge_activity))
            * 1.10 * exposure_ticks
        )
        energy_return = float(getattr(
            agent, "_routine_eva_energy_return_threshold", 55.0
        ))
        return bool(
            not emergency_route
            and not target.get("expedition")
            and terrain_cost <= 1.60
            and agent.needs.energy >= (
                max(66.0, energy_return + 5.0) + projected_energy_margin
            )
            and agent.needs.thirst >= 67.0 + projected_thirst_margin
            and agent.needs.hunger >= 40.0 + projected_hunger_margin
            and agent.needs.o2_supply >= (
                max(55.0, o2_return + 10.0) + projected_o2_margin
            )
            and 35.0 <= agent.needs.temperature_stress <= 70.0
            and float(getattr(agent, "injury_level", 0.0)) < 0.10
            and int(getattr(agent, "radiation_sickness_level", 0)) == 0
            and not getattr(agent, "active_diseases", [])
            and overload <= 1.0
        )

    @staticmethod
    def _movement_route_signature(
        target: dict, tx: int, ty: int
    ) -> tuple:
        """Identify one physical trip while ignoring transient UI metadata."""
        return (
            int(tx), int(ty), str(target.get("destination", "")),
            str(target.get("recipe", "")), str(target.get("resource", "")),
            str(target.get("structure_id", target.get("struct_id", ""))),
        )

    def _settle_movement_effort(
        self,
        agent: Agent,
        *,
        completed: bool,
        reason: str,
    ) -> float:
        """Reward a pace choice from measured trip time and reserve cost."""
        contract = getattr(agent, "_movement_effort_contract", None)
        if not isinstance(contract, dict):
            return 0.0
        elapsed = max(
            1, int(self.current_tick) - int(contract["start_tick"]) + 1
        )
        expected = max(1.0, float(contract["expected_steady_ticks"]))
        energy_loss = max(
            0.0, float(contract["start_energy"]) - float(agent.needs.energy)
        )
        thirst_loss = max(
            0.0, float(contract["start_thirst"]) - float(agent.needs.thirst)
        )
        o2_loss = max(
            0.0, float(contract["start_o2"]) - float(agent.needs.o2_supply)
        )
        mode = str(contract.get("mode", "steady"))
        mode_stats = self._movement_effort_telemetry.setdefault(
            "by_mode", {}
        ).setdefault(mode, {
            "completed_routes": 0,
            "aborted_routes": 0,
            "censored_routes": 0,
            "elapsed_ticks": 0,
            "expected_steady_ticks": 0.0,
            "estimated_ticks_saved": 0.0,
            "energy_loss": 0.0,
            "thirst_loss": 0.0,
            "o2_loss": 0.0,
            "total_reward": 0.0,
        })
        if not completed and reason == "route_changed":
            # A dispatcher replacing the destination censors the trip before
            # its pace outcome is observable. Penalising that policy change as
            # a locomotion failure drove both pace Q-values downward even when
            # movement was healthy.
            self._movement_effort_telemetry["censored_routes"] += 1
            censored = self._movement_effort_telemetry.setdefault(
                "censored_by_reason", {}
            )
            censored[reason] = int(censored.get(reason, 0)) + 1
            mode_stats["censored_routes"] += 1
            self._movement_effort_telemetry["last_outcome"] = {
                "agent_id": agent.id,
                "mode": contract["mode"],
                "completed": False,
                "censored": True,
                "reason": reason,
                "elapsed_ticks": elapsed,
                "expected_steady_ticks": round(expected, 3),
                "energy_loss": round(energy_loss, 3),
                "thirst_loss": round(thirst_loss, 3),
                "o2_loss": round(o2_loss, 3),
                "reward": 0.0,
            }
            delattr(agent, "_movement_effort_contract")
            return 0.0
        if completed:
            time_gain = max(
                -0.75, min(0.75, (expected - elapsed) / expected)
            )
            reward = (
                1.0 + time_gain * 5.0
                - energy_loss * 0.10
                - thirst_loss * 0.06
                - o2_loss * 0.04
            )
            if agent.needs.energy < 65.0:
                reward -= (65.0 - agent.needs.energy) * 0.25
            if agent.needs.thirst < 60.0:
                reward -= (60.0 - agent.needs.thirst) * 0.20
            if agent.needs.o2_supply < max(
                45.0,
                float(getattr(
                    agent, "_routine_eva_o2_return_threshold", 40.0
                )),
            ):
                reward -= 5.0
            self._movement_effort_telemetry["completed_routes"] += 1
            self._movement_effort_telemetry[
                "estimated_ticks_saved"
            ] += max(0.0, expected - elapsed)
            mode_stats["completed_routes"] += 1
            mode_stats["estimated_ticks_saved"] += max(
                0.0, expected - elapsed
            )
        else:
            reward = -2.0 if contract["mode"] == "surge" else -0.5
            self._movement_effort_telemetry["aborted_routes"] += 1
            aborted = self._movement_effort_telemetry.setdefault(
                "aborted_by_reason", {}
            )
            aborted[reason] = int(aborted.get(reason, 0)) + 1
            mode_stats["aborted_routes"] += 1

        reward = max(-15.0, min(6.0, reward))
        self.decision_engine.apply_delayed_rl_reward(
            agent,
            str(contract["state_key"]),
            str(contract["action_key"]),
            reward,
        )
        self._movement_effort_telemetry["total_route_reward"] += reward
        mode_stats["elapsed_ticks"] += elapsed
        mode_stats["expected_steady_ticks"] += expected
        mode_stats["energy_loss"] += energy_loss
        mode_stats["thirst_loss"] += thirst_loss
        mode_stats["o2_loss"] += o2_loss
        mode_stats["total_reward"] += reward
        self._movement_effort_telemetry["last_outcome"] = {
            "agent_id": agent.id,
            "mode": contract["mode"],
            "completed": completed,
            "reason": reason,
            "elapsed_ticks": elapsed,
            "expected_steady_ticks": round(expected, 3),
            "energy_loss": round(energy_loss, 3),
            "thirst_loss": round(thirst_loss, 3),
            "o2_loss": round(o2_loss, 3),
            "reward": round(reward, 3),
        }
        delattr(agent, "_movement_effort_contract")
        return reward

    def _movement_effort_mode(
        self,
        agent: Agent,
        target: dict,
        tx: int,
        ty: int,
        base_speed_cells: int,
        terrain_cost: float,
    ) -> str:
        """Choose or reuse a safe learned pace for one on-foot surface route."""
        signature = self._movement_route_signature(target, tx, ty)
        remaining_cells = abs(int(tx) - int(agent.x)) + abs(
            int(ty) - int(agent.y)
        )
        surge_speed_cells = max(
            int(base_speed_cells) + 1,
            int(math.ceil(
                float(base_speed_cells)
                * self.MOVEMENT_SURGE_SPEED_MULTIPLIER
            )),
        )
        surge_rate = max(
            1.0,
            min(
                float(surge_speed_cells),
                float(surge_speed_cells)
                * agent.movement_speed(float(self.planet.gravity_g))
                / max(1.0, terrain_cost),
            ),
        )
        expected_surge_ticks = max(
            1, math.ceil(remaining_cells / surge_rate)
        )
        existing = getattr(agent, "_movement_effort_contract", None)
        if isinstance(existing, dict) and existing.get("signature") == signature:
            if existing["mode"] == "surge" and not self._surge_movement_allowed(
                agent,
                target,
                terrain_cost,
                projected_ticks=expected_surge_ticks,
            ):
                self._settle_movement_effort(
                    agent, completed=False, reason="safety_envelope_changed"
                )
                return "steady"
            return str(existing["mode"])
        if isinstance(existing, dict):
            self._settle_movement_effort(
                agent, completed=False, reason="route_changed"
            )

        if remaining_cells <= 1 or not self._surge_movement_allowed(
            agent,
            target,
            terrain_cost,
            projected_ticks=expected_surge_ticks,
        ):
            return "steady"

        state_key = self._movement_effort_state(
            agent, remaining_cells, terrain_cost
        )
        selected = self.decision_engine.select_auxiliary_action_via_rl(
            agent,
            state_key,
            [
                {"action": "pace", "target": {"destination": "steady"}},
                {"action": "pace", "target": {"destination": "surge"}},
            ],
            initial_q=1.0,
        )
        mode = str(selected.get("target", {}).get("destination", "steady"))
        nominal_rate = max(
            1.0,
            min(
                float(base_speed_cells),
                float(base_speed_cells)
                * agent.movement_speed(float(self.planet.gravity_g))
                / max(1.0, terrain_cost),
            ),
        )
        expected_steady_ticks = math.ceil(remaining_cells / nominal_rate)
        agent._movement_effort_contract = {
            "signature": signature,
            "mode": mode,
            "state_key": selected["auxiliary_state_key"],
            "action_key": selected["auxiliary_action_key"],
            "start_tick": int(self.current_tick),
            "expected_steady_ticks": int(expected_steady_ticks),
            "expected_surge_ticks": int(expected_surge_ticks),
            "start_energy": float(agent.needs.energy),
            "start_thirst": float(agent.needs.thirst),
            "start_o2": float(agent.needs.o2_supply),
        }
        self._movement_effort_telemetry["eligible_routes"] += 1
        self._movement_effort_telemetry[f"{mode}_routes"] += 1
        return mode

    def _construction_acceptance_record(
        self,
        structure: dict,
        builders: list[Agent],
        active_robot_ids: list[str],
    ) -> dict:
        """Run a reproducible tolerance, inspection and commissioning gate.

        Physics remains deterministic for a seed: the small tolerance offset
        represents landed-part tolerances, alignment and workmanship, while
        engineering competence, robot condition and completed rework determine
        whether the pressure/electrical/functional checklist passes.
        """
        recipe = self._get_recipe(str(structure.get("type", ""))) or {}
        attempt = int(structure.get("acceptance_attempts", 0)) + 1
        engineering_mean = (
            sum(float(getattr(item.competency, "engineering", 0)) for item in builders)
            / max(1, len(builders))
        )
        minimum_engineering = float(recipe.get("min_engineering", 0))
        engineering_adjustment = max(
            -0.08,
            min(0.05, (engineering_mean - minimum_engineering) * 0.01),
        )
        robot_by_id = {item.id: item for item in self.surface_fleet.assembly_robots}
        robot_conditions = [
            float(robot_by_id[robot_id].condition)
            for robot_id in active_robot_ids
            if robot_id in robot_by_id
        ]
        robot_condition_mean = (
            sum(robot_conditions) / len(robot_conditions)
            if robot_conditions else 1.0
        )
        tolerance_key = (
            f"{self.seed}|{structure.get('id')}|{structure.get('type')}|{attempt}"
        ).encode("utf-8")
        tolerance_unit = int(
            hashlib.sha256(tolerance_key).hexdigest()[:12], 16
        ) / float(16 ** 12 - 1)
        tolerance_offset = (tolerance_unit - 0.5) * 0.11
        rework_bonus = 0.08 * max(0, attempt - 1)
        quality_score = max(0.0, min(
            1.0,
            0.86
            + engineering_adjustment
            - (1.0 - robot_condition_mean) * 0.12
            + tolerance_offset
            + rework_bonus,
        ))
        threshold = 0.82
        phases = list((recipe.get("construction", {}) or {}).get("phases", []))
        acceptance_terms = (
            "test", "commission", "checkout", "calibration",
            "certification", "acceptance", "energization",
        )
        checks = [
            str(phase) for phase in phases
            if any(term in str(phase).lower() for term in acceptance_terms)
        ] or ["visual_dimensional_inspection"]
        passed = quality_score >= threshold or attempt >= 3
        return {
            "attempt": attempt,
            "status": "passed" if passed else "rework_required",
            "quality_score": round(quality_score, 4),
            "acceptance_threshold": threshold,
            "engineering_mean": round(engineering_mean, 3),
            "minimum_engineering": minimum_engineering,
            "assembly_robot_condition_mean": round(robot_condition_mean, 4),
            "seeded_tolerance_offset": round(tolerance_offset, 4),
            "checks": checks,
            "deterministic_seeded": True,
        }

    def _manufacturing_cycle_for_agent(self, agent: Agent) -> dict | None:
        """Return the machine-owned cycle currently supervised by ``agent``."""
        target = agent.action.target if isinstance(agent.action.target, dict) else {}
        machine_id = target.get("machine_id")
        if (
            getattr(agent.action, "action_type", "") != "refine"
            or not target.get("completion_pending")
            or not machine_id
        ):
            return None
        cycle = self._manufacturing_cycles.get(str(machine_id))
        if (
            cycle
            and cycle.get("completion_pending")
            and cycle.get("operator_id") == agent.id
            and (
                not cycle.get("cycle_id")
                or target.get("cycle_id") == cycle.get("cycle_id")
            )
        ):
            return cycle
        return None

    def _manufacturing_operator_matches_cycle(
        self, agent: Agent | None, cycle: dict
    ) -> bool:
        """Return whether an active batch has its exact live operator binding."""
        if agent is None:
            return False
        status = getattr(agent.status, "value", str(agent.status))
        if status in {"dead", "incapacitated"}:
            return False
        target = agent.action.target if isinstance(agent.action.target, dict) else {}
        return bool(
            getattr(agent.action, "action_type", "") == "refine"
            and agent.action.ticks_remaining > 0
            and target.get("completion_pending")
            and str(target.get("machine_id")) == str(cycle.get("machine_id"))
            and (
                not cycle.get("cycle_id")
                or target.get("cycle_id") == cycle.get("cycle_id")
            )
            and cycle.get("operator_id") == agent.id
        )

    def _advance_autonomous_manufacturing_cycles(self) -> list[dict]:
        """Advance closed-loop machine time independently of crew actions.

        Materials and the full batch-energy charge are committed when the
        cycle is created.  This clock only advances that same physical WIP;
        it cannot create output.  Finished WIP remains locked in its machine
        until a qualified crew member performs the unload/inspection action.
        """
        if not self.AUTONOMOUS_MANUFACTURING_ENABLED:
            return []

        completed: list[dict] = []
        for machine_id, cycle in self._manufacturing_cycles.items():
            if (
                not cycle.get("completion_pending")
                or not cycle.get("autonomous")
                or cycle.get("output_ready")
                or not cycle.get("active")
            ):
                continue

            last_tick = int(cycle.get(
                "last_process_tick", cycle.get("started_tick", self.current_tick)
            ))
            elapsed_ticks = max(0, int(self.current_tick) - last_tick)
            if elapsed_ticks <= 0:
                continue
            remaining = max(0, int(cycle.get("remaining_ticks", 0)))
            advanced = min(remaining, elapsed_ticks)
            cycle["remaining_ticks"] = max(0, remaining - advanced)
            cycle["last_process_tick"] = int(self.current_tick)
            self._manufacturing_automation_telemetry[
                "process_ticks_elapsed"
            ] += advanced
            if cycle["remaining_ticks"] > 0:
                continue

            cycle["active"] = False
            cycle["autonomous_processing"] = False
            cycle["output_ready"] = True
            cycle["process_completed_tick"] = int(self.current_tick)
            cycle["pause_reason"] = "awaiting_unload_inspection"
            cycle["operator_id"] = None
            cycle["unload_remaining_ticks"] = int(
                self.MANUFACTURING_UNLOAD_TICKS
            )
            cycle.pop("route_operator_id", None)
            cycle.pop("route_reserved_until_tick", None)
            # The initiating worker carries a read-only planner token while
            # the batch runs. Refresh every copy at the state transition so
            # the planner evaluates only the short unload duty, rather than
            # demanding reserves for the now-finished multi-hour process.
            for crew in self.agents:
                paused = getattr(crew, "_paused_manufacturing", None)
                if (
                    isinstance(paused, dict)
                    and str(paused.get("machine_id")) == str(machine_id)
                    and (
                        not cycle.get("cycle_id")
                        or paused.get("cycle_id") == cycle.get("cycle_id")
                    )
                ):
                    refreshed = self._manufacturing_action_target(
                        cycle, resumed=True
                    )
                    refreshed["pause_reason"] = (
                        "awaiting_unload_inspection"
                    )
                    crew._paused_manufacturing = refreshed
            self._manufacturing_automation_telemetry[
                "process_cycles_completed"
            ] += 1
            record = {
                "machine_id": str(machine_id),
                "cycle_id": cycle.get("cycle_id"),
                "output": cycle.get("output"),
                "quantity": int(cycle.get("output_quantity", 1)),
                "process_ticks": int(cycle.get("nominal_process_ticks", 0)),
                "awaiting_unload_inspection": True,
            }
            completed.append(record)
            if self._on_event:
                self._on_event({
                    "type": "manufacturing_process_complete",
                    **record,
                    "tick": self.current_tick,
                })
        return completed

    def _enforce_manufacturing_cycle_invariants(self) -> list[dict]:
        """Pause any batch labelled active without a matching live operator.

        Agent actions can be replaced by self-care, an EVA recall, route logic,
        restored state or emergency handling outside the normal REFINE branch.
        Machine WIP is authoritative, so such a replacement must pause the
        batch and leave it available for a later qualified handoff; it must
        never leave the physical machine permanently occupied by a ghost shift.
        """
        reconciled: list[dict] = []
        agents_by_id = {crew.id: crew for crew in self.agents}
        for machine_id, cycle in self._manufacturing_cycles.items():
            if not cycle.get("completion_pending") or not cycle.get("active"):
                continue
            if cycle.get("autonomous") and not cycle.get("output_ready"):
                # A qualified closed-loop recipe is intentionally unattended.
                # Its machine-owned clock is advanced above; no ghost human
                # operator is required for this state.
                continue
            operator = agents_by_id.get(cycle.get("operator_id"))
            if self._manufacturing_operator_matches_cycle(operator, cycle):
                cycle["last_operator_verified_tick"] = self.current_tick
                continue

            if operator is None:
                reason = "operator_missing"
            elif getattr(operator.status, "value", str(operator.status)) in {
                "dead", "incapacitated"
            }:
                reason = "operator_unavailable"
            elif getattr(operator.action, "action_type", "") != "refine":
                reason = "operator_action_replaced"
            else:
                reason = "operator_binding_lost"

            remaining_ticks = max(1, int(cycle.get("remaining_ticks", 1)))
            if operator is not None:
                self._pause_manufacturing_cycle(
                    operator,
                    reason,
                    cycle=cycle,
                    remaining_ticks=remaining_ticks,
                )
            else:
                cycle["active"] = False
                cycle["pause_reason"] = reason
                cycle["paused_tick"] = self.current_tick

            # A stale route reservation can otherwise hide the newly paused
            # batch from every relief operator until another long timeout.
            cycle.pop("route_operator_id", None)
            cycle.pop("route_reserved_until_tick", None)
            cycle["watchdog_pause_tick"] = self.current_tick
            record = {
                "machine_id": str(machine_id),
                "cycle_id": cycle.get("cycle_id"),
                "operator_id": cycle.get("operator_id"),
                "reason": reason,
                "remaining_ticks": remaining_ticks,
            }
            reconciled.append(record)
            if self._on_event:
                self._on_event({
                    "type": "manufacturing_watchdog_pause",
                    **record,
                    "tick": self.current_tick,
                })
        return reconciled

    def _manufacturing_action_target(
        self, cycle: dict, *, resumed: bool = False, handoff: bool = False
    ) -> dict:
        """Expose the public part of a machine cycle on an operator action."""
        output_ready = bool(cycle.get("output_ready"))
        action_remaining = (
            max(1, int(cycle.get(
                "unload_remaining_ticks", self.MANUFACTURING_UNLOAD_TICKS
            )))
            if output_ready
            else max(0, int(cycle.get("remaining_ticks", 0)))
        )
        target = {
            "x": cycle.get("x"),
            "y": cycle.get("y"),
            "output": cycle.get("output"),
            "output_quantity": int(cycle.get("output_quantity", 1)),
            "completion_pending": bool(cycle.get("completion_pending", True)),
            "energy_kwh": float(cycle.get("energy_kwh", 0.0)),
            "machine_type": cycle.get("machine_type"),
            "machine_id": cycle.get("machine_id"),
            "cycle_id": cycle.get("cycle_id"),
            "remaining_ticks": action_remaining,
            "process_remaining_ticks": max(
                0, int(cycle.get("remaining_ticks", 0))
            ),
            "autonomous": bool(cycle.get("autonomous", False)),
            "autonomous_processing": bool(
                cycle.get("autonomous_processing", False)
            ),
            "output_ready": output_ready,
            "shift_start_tick": int(cycle.get("shift_start_tick", self.current_tick)),
            "shift_end_tick": int(cycle.get(
                "shift_end_tick", self.current_tick + self.MANUFACTURING_SHIFT_TICKS
            )),
        }
        if resumed:
            target["resumed"] = True
        if handoff:
            target["operator_handoff"] = True
        return target

    def _manufacturing_operator_is_fit(
        self, agent: Agent, recipe: dict, duty_ticks: int = 1
    ) -> bool:
        """Require both recipe capability and safe current shift vitals."""
        status = getattr(agent.status, "value", str(agent.status))
        if status in {"dead", "incapacitated"}:
            return False
        non_material_recipe = dict(recipe)
        non_material_recipe["materials"] = {}
        # Fixed manufacturing assets carry their own tooling. This check is
        # for operator competency and physiology; portable field tools are a
        # separate, finite construction resource.
        non_material_recipe["requires_tool"] = False
        if not agent.can_craft(non_material_recipe).get("can_craft", False):
            return False
        needs = agent.needs
        o2_safe = bool(getattr(agent, "_in_habitat", False)) or needs.o2_supply > 40.0
        duty_ticks = min(
            max(1, int(duty_ticks)), self.MANUFACTURING_SHIFT_TICKS
        )
        overtime_active = self.current_tick <= int(getattr(
            agent, "_fatigue_overtime_until_tick", -1
        ))
        energy_floor = (
            min(45.0, 15.0 + duty_ticks * 0.35)
            if overtime_active
            else min(96.0, 65.0 + duty_ticks * 0.4)
        )
        return (
            needs.energy > energy_floor
            and needs.hunger > min(60.0, 35.0 + duty_ticks * 0.25)
            and needs.thirst > min(70.0, 45.0 + duty_ticks * 0.25)
            and 42.0 < needs.temperature_stress < 88.0
            and o2_safe
        )

    def _agent_can_take_manufacturing_cycle(
        self, agent: Agent, cycle: dict, recipe: dict
    ) -> bool:
        """Whether a paused batch can be resumed by this operator now."""
        if cycle.get("active") or not cycle.get("completion_pending"):
            return False
        route_owner = cycle.get("route_operator_id")
        route_until = int(cycle.get("route_reserved_until_tick", -1))
        if route_owner and route_owner != agent.id and self.current_tick <= route_until:
            return False
        duty_ticks = (
            int(cycle.get(
                "unload_remaining_ticks", self.MANUFACTURING_UNLOAD_TICKS
            ))
            if cycle.get("output_ready")
            else min(
                int(cycle.get("remaining_ticks", 1)),
                self.MANUFACTURING_SHIFT_TICKS,
            )
        )
        if not self._manufacturing_operator_is_fit(
            agent,
            recipe,
            duty_ticks=duty_ticks,
        ):
            return False
        if cycle.get("output_ready"):
            # Finished WIP is safe and does not need a shift-relief delay.
            return True
        operator_id = cycle.get("operator_id")
        shift_end = int(cycle.get("shift_end_tick", self.current_tick))
        if operator_id == agent.id:
            # The interrupted operator may return within the same duty block.
            # Once the block expires, give another qualified crewmember a real
            # opportunity to take the machine before this operator cycles back.
            relief_until = int(cycle.get(
                "relief_until_tick", shift_end + self.MANUFACTURING_SHIFT_TICKS
            ))
            return self.current_tick < shift_end or self.current_tick >= relief_until
        return self.current_tick >= shift_end

    def _assign_manufacturing_cycle(
        self, agent: Agent, cycle: dict
    ) -> str | None:
        """Attach a machine-owned batch to one fit operator shift."""
        previous_operator_id = cycle.get("operator_id")
        new_shift = (
            previous_operator_id != agent.id
            or self.current_tick >= int(cycle.get("shift_end_tick", self.current_tick))
        )
        handoff_from = (
            str(previous_operator_id)
            if previous_operator_id and previous_operator_id != agent.id
            else None
        )
        if new_shift:
            cycle["shift_start_tick"] = self.current_tick
            cycle["shift_end_tick"] = (
                self.current_tick + self.MANUFACTURING_SHIFT_TICKS
            )
        cycle["operator_id"] = agent.id
        cycle["active"] = True
        if cycle.get("output_ready"):
            cycle["unloading"] = True
        cycle["pause_reason"] = None
        cycle["paused_tick"] = None
        cycle.pop("relief_until_tick", None)
        cycle.pop("route_operator_id", None)
        cycle.pop("route_reserved_until_tick", None)
        if handoff_from:
            cycle["handoff_count"] = int(cycle.get("handoff_count", 0)) + 1
            cycle["previous_operator_id"] = handoff_from

        # Assignment (including the original operator reclaiming an expired
        # relief offer) invalidates every read-only offer for this machine.
        # Leaving those copies on other crew makes them request a cycle that
        # is already actively supervised, producing long REFINE->IDLE loops.
        for crew in self.agents:
            paused = getattr(crew, "_paused_manufacturing", None)
            if (
                isinstance(paused, dict)
                and str(paused.get("machine_id"))
                == str(cycle.get("machine_id"))
            ):
                delattr(crew, "_paused_manufacturing")

        agent.action.action_type = "refine"
        agent.action.target = self._manufacturing_action_target(
            cycle, resumed=True, handoff=bool(handoff_from)
        )
        agent.action.ticks_remaining = (
            max(1, int(cycle.get(
                "unload_remaining_ticks", self.MANUFACTURING_UNLOAD_TICKS
            )))
            if cycle.get("output_ready")
            else max(1, int(cycle.get("remaining_ticks", 1)))
        )
        agent.action.ticks_elapsed = 0
        if hasattr(agent, "_paused_manufacturing"):
            delattr(agent, "_paused_manufacturing")
        return handoff_from

    def _pause_manufacturing_cycle(
        self,
        agent: Agent,
        reason: str,
        *,
        cycle: dict | None = None,
        remaining_ticks: int | None = None,
    ) -> dict | None:
        """Pause supervision without releasing or recommitting the machine."""
        if cycle is None:
            cycle = self._manufacturing_cycle_for_agent(agent)
        if cycle is None:
            cycle = next((
                candidate for candidate in self._manufacturing_cycles.values()
                if candidate.get("completion_pending")
                and candidate.get("operator_id") == agent.id
                and candidate.get("active")
            ), None)
        if cycle is None:
            return None
        if remaining_ticks is not None:
            if cycle.get("output_ready"):
                cycle["unload_remaining_ticks"] = max(
                    1, int(remaining_ticks)
                )
            else:
                cycle["remaining_ticks"] = max(1, int(remaining_ticks))
        cycle["active"] = False
        cycle["unloading"] = False
        cycle["pause_reason"] = reason
        cycle["paused_tick"] = self.current_tick
        cycle.pop("route_operator_id", None)
        cycle.pop("route_reserved_until_tick", None)
        if reason == "operator_shift_complete":
            cycle["relief_until_tick"] = (
                self.current_tick + self.MANUFACTURING_SHIFT_TICKS
            )
        paused_target = self._manufacturing_action_target(cycle, resumed=True)
        paused_target["pause_reason"] = reason
        agent._paused_manufacturing = paused_target
        return cycle

    def _offer_manufacturing_handoff(self, agent: Agent) -> dict | None:
        """Expose one due machine shift to an available qualified operator."""
        if (
            agent.action.ticks_remaining > 0
            and getattr(agent.action, "action_type", "") not in {None, "", "idle"}
        ):
            return None
        expedition = getattr(agent, "_active_expedition", None)
        if (
            isinstance(expedition, dict)
            and expedition.get("status") in {"outbound", "working", "returning"}
        ):
            return None
        # The initiating operator can tend another cell while their original
        # autonomous batch runs. Its output reservation lives on the machine.
        self.decision_engine._release_autonomous_operator_token(
            agent, self._manufacturing_cycles
        )
        existing = getattr(agent, "_paused_manufacturing", None)
        if isinstance(existing, dict) and existing.get("completion_pending"):
            return None
        operational_ids = {
            str(machine.get("id"))
            for machine in self.placed_structures
            if not machine.get("under_construction", False)
            and not machine.get("destroyed", False)
        }
        candidates = []
        for cycle in self._manufacturing_cycles.values():
            if str(cycle.get("machine_id")) not in operational_ids:
                continue
            recipe = self._get_recipe(str(cycle.get("output", "")))
            if recipe and self._agent_can_take_manufacturing_cycle(
                agent, cycle, recipe
            ):
                candidates.append(cycle)
        if not candidates:
            return None
        cycle = min(
            candidates,
            key=lambda candidate: (
                max(
                    abs(int(candidate.get("x", agent.x)) - agent.x),
                    abs(int(candidate.get("y", agent.y)) - agent.y),
                ),
                str(candidate.get("machine_id")),
            ),
        )
        cycle["route_operator_id"] = agent.id
        cycle["route_reserved_until_tick"] = (
            self.current_tick + self.MANUFACTURING_SHIFT_TICKS
        )
        offer = self._manufacturing_action_target(cycle, resumed=True, handoff=True)
        offer["handoff_from_operator_id"] = cycle.get("operator_id")
        agent._paused_manufacturing = offer
        return cycle

    def _reserved_kit_materials(self) -> dict[str, int]:
        """Return depot quantities still sealed for delivered structure kits."""
        reserved: dict[str, int] = {}
        depot = getattr(self, "central_depot_inventory", {})
        for kit_materials in getattr(self, "delivered_structure_kits", {}).values():
            for material, quantity in kit_materials.items():
                reserved[material] = reserved.get(material, 0) + min(
                    int(quantity), int(depot.get(material, 0))
                )
        return reserved

    def _available_depot_quantity(
        self, material: str, purpose_recipe: str | None = None
    ) -> int:
        """Return stock usable for a task without cannibalising another kit."""
        depot_quantity = int(
            getattr(self, "central_depot_inventory", {}).get(material, 0)
        )
        if purpose_recipe in getattr(self, "delivered_structure_kits", {}):
            return depot_quantity
        return max(
            0,
            depot_quantity - self._reserved_kit_materials().get(material, 0),
        )

    def _staged_material_totals(
        self,
        site_x: int,
        site_y: int,
        purpose_recipe: str | None = None,
    ) -> dict[str, int]:
        """Count task-eligible depot stock and physically staged crew loads."""
        depot = getattr(self, "central_depot_inventory", {})
        totals = {
            material: self._available_depot_quantity(material, purpose_recipe)
            for material in depot
        }
        lz_x = getattr(self, "lz_x", site_x)
        lz_y = getattr(self, "lz_y", site_y)
        for crew in self.agents:
            if getattr(crew.status, "value", str(crew.status)) == "dead":
                continue
            near_depot = max(abs(crew.x - lz_x), abs(crew.y - lz_y)) <= 1
            near_site = max(abs(crew.x - site_x), abs(crew.y - site_y)) <= 1
            if not (near_depot or near_site):
                continue
            for material, quantity in crew.inventory.materials.items():
                totals[material] = totals.get(material, 0) + quantity
        return totals

    def _consume_staged_materials(
        self,
        requirements: dict[str, int],
        site_x: int,
        site_y: int,
        purpose_recipe: str | None = None,
    ) -> bool:
        """Consume a bill of materials from staged stock without teleporting loads."""
        totals = self._staged_material_totals(
            site_x, site_y, purpose_recipe=purpose_recipe
        )
        if any(totals.get(material, 0) < needed for material, needed in requirements.items()):
            return False
        depot = getattr(self, "central_depot_inventory", {})
        lz_x = getattr(self, "lz_x", site_x)
        lz_y = getattr(self, "lz_y", site_y)
        staged_crew = [
            crew for crew in self.agents
            if getattr(crew.status, "value", str(crew.status)) != "dead"
            and (
                max(abs(crew.x - lz_x), abs(crew.y - lz_y)) <= 1
                or max(abs(crew.x - site_x), abs(crew.y - site_y)) <= 1
            )
        ]
        for material, needed in requirements.items():
            remaining = needed
            from_depot = min(
                remaining,
                self._available_depot_quantity(material, purpose_recipe),
            )
            depot[material] = depot.get(material, 0) - from_depot
            remaining -= from_depot
            for crew in staged_crew:
                if remaining <= 0:
                    break
                available = crew.inventory.materials.get(material, 0)
                take = min(remaining, available)
                if take:
                    crew.inventory.remove_material(material, take)
                    remaining -= take
        if purpose_recipe in getattr(self, "delivered_structure_kits", {}):
            # The complete kit is now physically committed to its construction
            # site.  It is no longer available to the depot reservation ledger.
            self.delivered_structure_kits.pop(purpose_recipe, None)
        return True

    @staticmethod
    def _positive_material_manifest(materials: dict) -> dict[str, int]:
        return {
            str(material): int(quantity)
            for material, quantity in (materials or {}).items()
            if int(quantity) > 0
        }

    def _material_manifest_mass_kg(self, materials: dict[str, int]) -> float:
        return sum(
            int(quantity) * float(MATERIAL_DENSITY_KG.get(material, 2.0))
            for material, quantity in materials.items()
        )

    def _reserve_construction_cargo(
        self,
        *,
        site_id: str,
        recipe_name: str,
        requirements: dict,
        site_x: int,
        site_y: int,
        required_item: str | None = None,
    ) -> dict:
        """Atomically move one BOM from the depot into a site haul queue."""
        manifest = self._positive_material_manifest(requirements)
        if required_item and required_item not in manifest:
            manifest[str(required_item)] = 1
        missing = {
            material: quantity - self._available_depot_quantity(
                material, purpose_recipe=recipe_name
            )
            for material, quantity in manifest.items()
            if self._available_depot_quantity(
                material, purpose_recipe=recipe_name
            ) < quantity
        }
        if missing:
            return {"reserved": False, "reason": "depot_stock_missing", "missing": missing}

        transporter_limit = float(
            self.surface_fleet.cargo_transporter_spec.get("payload_kg", 0.0)
        )
        oversized = {
            material: float(MATERIAL_DENSITY_KG.get(material, 2.0))
            for material in manifest
            if float(MATERIAL_DENSITY_KG.get(material, 2.0))
            > transporter_limit + 1e-6
        }
        if not self.surface_fleet.cargo_transporters:
            return {"reserved": False, "reason": "no_cargo_transporter"}
        if oversized:
            return {
                "reserved": False,
                "reason": "indivisible_cargo_exceeds_transporter_deck",
                "oversized": oversized,
            }

        depot = self.central_depot_inventory
        for material, quantity in manifest.items():
            depot[material] = int(depot.get(material, 0)) - quantity
            if depot[material] <= 0:
                depot.pop(material, None)
        if recipe_name in self.delivered_structure_kits:
            self.delivered_structure_kits.pop(recipe_name, None)

        total_mass = self._material_manifest_mass_kg(manifest)
        self.site_material_staging[str(site_id)] = {}
        self._construction_cargo_reservations[str(site_id)] = {
            "site_id": str(site_id),
            "recipe": str(recipe_name),
            "x": int(site_x),
            "y": int(site_y),
            "required": dict(manifest),
            "at_depot": dict(manifest),
            "in_transit": {},
            "delivered": {},
            "total_mass_kg": total_mass,
            "dispatched_mass_kg": 0.0,
            "delivered_mass_kg": 0.0,
            "trips_dispatched": 0,
            "trips_completed": 0,
            "status": "reserved_at_depot",
        }
        return {
            "reserved": True,
            "total_mass_kg": total_mass,
            "site_id": str(site_id),
        }

    def _next_construction_cargo_batch(
        self, materials: dict[str, int], payload_limit_kg: float
    ) -> tuple[dict[str, int], float]:
        """Pack one exact integer-unit transporter manifest."""
        remaining_capacity = max(0.0, float(payload_limit_kg))
        batch: dict[str, int] = {}
        # Heavy serialized modules load first; small loose lots fill the deck's
        # remaining rated mass without splitting an inventory unit.
        ordered = sorted(
            self._positive_material_manifest(materials).items(),
            key=lambda pair: (
                -float(MATERIAL_DENSITY_KG.get(pair[0], 2.0)), pair[0]
            ),
        )
        for material, available in ordered:
            unit_mass = max(
                0.001, float(MATERIAL_DENSITY_KG.get(material, 2.0))
            )
            quantity = min(
                int(available), int((remaining_capacity + 1e-9) // unit_mass)
            )
            if quantity <= 0:
                continue
            batch[material] = quantity
            remaining_capacity -= quantity * unit_mass
        return batch, self._material_manifest_mass_kg(batch)

    def _dispatch_construction_cargo(self) -> list[dict]:
        """Dispatch available heavy transporters to pending construction sites."""
        dispatches: list[dict] = []
        payload_limit = float(
            self.surface_fleet.cargo_transporter_spec.get("payload_kg", 0.0)
        )
        for site_id in sorted(self._construction_cargo_reservations):
            reservation = self._construction_cargo_reservations[site_id]
            if reservation.get("status") in {"delivered", "stranded"}:
                continue
            surface_route = reservation.get("surface_route")
            if not surface_route:
                planned_route = self._surface_vehicle_route(
                    self.lz_x,
                    self.lz_y,
                    int(reservation["x"]),
                    int(reservation["y"]),
                )
                if not planned_route:
                    reservation["status"] = "blocked_no_surface_route"
                    continue
                surface_route = [
                    {"x": x, "y": y} for x, y in planned_route
                ]
                reservation["surface_route"] = surface_route
            while any(int(value) > 0 for value in reservation["at_depot"].values()):
                batch, batch_mass = self._next_construction_cargo_batch(
                    reservation["at_depot"], payload_limit
                )
                if not batch:
                    reservation["status"] = "blocked_oversize_manifest"
                    break
                result = self.surface_fleet.dispatch_cargo_transport(
                    site_id=site_id,
                    target_x=int(reservation["x"]),
                    target_y=int(reservation["y"]),
                    payload=batch,
                    payload_mass_kg=batch_mass,
                    current_tick=self.current_tick,
                    route=surface_route,
                )
                if not result.get("dispatched"):
                    break
                for material, quantity in batch.items():
                    reservation["at_depot"][material] -= quantity
                    if reservation["at_depot"][material] <= 0:
                        reservation["at_depot"].pop(material, None)
                    reservation["in_transit"][material] = (
                        reservation["in_transit"].get(material, 0) + quantity
                    )
                reservation["dispatched_mass_kg"] += batch_mass
                reservation["trips_dispatched"] += 1
                reservation["status"] = "in_transit"
                dispatches.append({
                    "type": "cargo_transport_dispatched",
                    "site_id": site_id,
                    "vehicle_id": result.get("vehicle_id"),
                    "payload_mass_kg": round(batch_mass, 3),
                    "payload": dict(batch),
                })
        return dispatches

    def _receive_construction_cargo(self, event: dict) -> None:
        """Move an arrived vehicle manifest into exactly one site ledger."""
        site_id = str(event.get("site_id", ""))
        reservation = self._construction_cargo_reservations.get(site_id)
        if reservation is None:
            return
        payload = self._positive_material_manifest(event.get("payload", {}))
        delivered = self.site_material_staging.setdefault(site_id, {})
        for material, quantity in payload.items():
            in_transit = int(reservation["in_transit"].get(material, 0))
            accepted = min(quantity, in_transit)
            if accepted <= 0:
                continue
            reservation["in_transit"][material] = in_transit - accepted
            if reservation["in_transit"][material] <= 0:
                reservation["in_transit"].pop(material, None)
            reservation["delivered"][material] = (
                reservation["delivered"].get(material, 0) + accepted
            )
            delivered[material] = delivered.get(material, 0) + accepted
        delivered_mass = self._material_manifest_mass_kg(payload)
        reservation["delivered_mass_kg"] += delivered_mass
        reservation["trips_completed"] += 1

        complete = all(
            int(reservation["delivered"].get(material, 0)) >= int(quantity)
            for material, quantity in reservation["required"].items()
        )
        site = next((
            item for item in self.placed_structures
            if str(item.get("id")) == site_id
        ), None)
        if site is not None:
            site["cargo_trips_completed"] = int(
                reservation["trips_completed"]
            )
            site["materials_delivered_kg"] = round(
                reservation["delivered_mass_kg"], 3
            )
        if not complete:
            reservation["status"] = "in_transit" if reservation["in_transit"] else "awaiting_dispatch"
            return
        reservation["status"] = "delivered"
        if site is None or site.get("destroyed", False):
            reservation["status"] = "stranded"
            return
        site["materials_committed"] = True
        site["construction_phase"] = "assembly"
        site["site_material_manifest"] = dict(reservation["delivered"])
        site["cargo_delivery_completed_tick"] = self.current_tick
        if self._on_event:
            self._on_event({
                "type": "construction_cargo_staged",
                "agent": "Autonomous Cargo Fleet",
                "cause": (
                    f"Delivered the conserved {reservation['recipe'].replace('_', ' ').title()} "
                    f"BOM in {reservation['trips_completed']} transporter trips"
                ),
                "site_id": site_id,
                "payload_mass_kg": round(reservation["delivered_mass_kg"], 3),
                "tick": self.current_tick,
            })

    def _staged_item_total(self, item: str, site_x: int, site_y: int) -> int:
        """Count a non-BOM mission item at the depot or construction site."""
        total = int(getattr(
            self, "central_depot_inventory", {}
        ).get(item, 0))
        lz_x = getattr(self, "lz_x", site_x)
        lz_y = getattr(self, "lz_y", site_y)
        for crew in self.agents:
            if getattr(crew.status, "value", str(crew.status)) == "dead":
                continue
            near_depot = max(abs(crew.x - lz_x), abs(crew.y - lz_y)) <= 1
            near_site = max(abs(crew.x - site_x), abs(crew.y - site_y)) <= 1
            if near_depot or near_site:
                total += int(crew.inventory.items.get(item, 0))
        return total

    def _consume_staged_item(self, item: str, site_x: int, site_y: int) -> bool:
        """Consume one unique mission item from physically staged stock."""
        if self._staged_item_total(item, site_x, site_y) <= 0:
            return False
        depot = getattr(self, "central_depot_inventory", {})
        if int(depot.get(item, 0)) > 0:
            depot[item] -= 1
            if depot[item] <= 0:
                depot.pop(item, None)
            return True
        lz_x = getattr(self, "lz_x", site_x)
        lz_y = getattr(self, "lz_y", site_y)
        for crew in self.agents:
            near_depot = max(abs(crew.x - lz_x), abs(crew.y - lz_y)) <= 1
            near_site = max(abs(crew.x - site_x), abs(crew.y - site_y)) <= 1
            if not (near_depot or near_site):
                continue
            if int(crew.inventory.items.get(item, 0)) > 0:
                crew.inventory.remove_item(item, 1)
                return True
        return False

    def _reveal_detectable_resources(
        self, x: int, y: int, detector_penetration_layers: int
    ) -> list[str]:
        """Reveal only deposits reachable from the measured excavation face."""
        geology = self._get_cell_geology(x, y)
        excavation_depth = self.cell_excavation_depth.get((x, y), 0)
        current_index = int(geology.get("current_index", 0))
        max_detectable_index = current_index + excavation_depth + detector_penetration_layers
        revealed: list[str] = []
        for index, layer in enumerate(geology.get("layers", [])):
            resource = layer["material"]
            if index > max_detectable_index:
                continue
            if (x, y, resource) in self.depleted_cell_resources:
                continue
            self.discovered_resources.setdefault(resource, set()).add((x, y))
            revealed.append(resource)
        if revealed:
            self.revealed_cell_resources.add((x, y))
        return revealed

    def _excavate_test_layer(self, agent: Agent, x: int, y: int) -> list[str]:
        """Open one shallow test-pit layer and expose only reached deposits."""
        # Detector readings are harmless inside the planned campus; opening a
        # test pit is not.  Keep this guard at the physical mutation boundary
        # so every caller preserves foundations, utilities and access lanes.
        if self._is_construction_protected_cell(x, y):
            return []
        key = (x, y)
        self.cell_excavation_depth[key] = self.cell_excavation_depth.get(key, 0) + 1
        revealed = self._reveal_detectable_resources(
            x, y, detector_penetration_layers=0
        )
        if agent.inventory.get_best_tool() is not None:
            agent.inventory.use_tool(1)
        return revealed

    def _perform_portable_resource_scan(
        self, agent: Agent, radius: int | None = None
    ) -> dict[str, list[tuple[int, int]]]:
        """Measure shallow geology around the agent and spend battery charge."""
        radius = (
            self.PORTABLE_SCANNER_SCAN_RADIUS_CELLS if radius is None else int(radius)
        )
        charge = float(agent.inventory.tool_charge_pct.get("portable_scanner", 0.0))
        durability = int(agent.inventory.tool_durability.get("portable_scanner", 0))
        if (
            not agent.inventory.has_item("portable_scanner")
            or durability <= 0
            or charge < self.PORTABLE_SCANNER_SCAN_CHARGE_PCT
        ):
            return {}

        found: dict[str, list[tuple[int, int]]] = {}
        min_x = max(0, agent.x - radius)
        max_x = min(self.world.map_size - 1, agent.x + radius)
        min_y = max(0, agent.y - radius)
        max_y = min(self.world.map_size - 1, agent.y + radius)
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                for resource in self._reveal_detectable_resources(
                    x, y, detector_penetration_layers=1
                ):
                    found.setdefault(resource, []).append((x, y))

        agent.inventory.tool_charge_pct["portable_scanner"] = max(
            0.0, charge - self.PORTABLE_SCANNER_SCAN_CHARGE_PCT
        )
        # Electronics still age, but battery depletion is the normal limiting
        # factor rather than an invented five-scan self-destruction rule.
        agent.inventory.tool_durability["portable_scanner"] = max(0, durability - 1)
        center = (agent.x, agent.y)
        self._portable_scan_attempts[center] = self._portable_scan_attempts.get(center, 0) + 1
        for resource in tuple(self.remote_resource_requests):
            if self.discovered_resources.get(resource):
                self.remote_resource_requests.discard(resource)
        return found

    def _has_buried_target_anomaly(
        self, x: int, y: int, resource: str | None
    ) -> bool:
        """Return a non-identifying detector anomaly for test-pit triage.

        This does not reveal a buried material or add it to discovered
        resources. It only models a tuned geophysical/chemical response strong
        enough to justify opening a pit during an expensive regional traverse.
        """
        if not resource:
            return False
        geology = self._get_cell_geology(x, y)
        current_index = int(geology.get("current_index", 0))
        already_visible = (
            current_index
            + self.cell_excavation_depth.get((x, y), 0)
            + 1
        )
        return any(
            index > already_visible
            and layer.get("material") == resource
            and (x, y, resource) not in self.depleted_cell_resources
            for index, layer in enumerate(geology.get("layers", []))
        )

    def _distance_from_lz(self, x: int, y: int) -> int:
        """Chebyshev distance from the landing hub on the square grid."""
        lz_x = getattr(self, "lz_x", self.world.center)
        lz_y = getattr(self, "lz_y", self.world.center)
        return max(abs(int(x) - lz_x), abs(int(y) - lz_y))

    def _lander_hub(self) -> dict | None:
        return next((
            structure for structure in getattr(self, "placed_structures", [])
            if structure.get("type") == "eclss_lander_hub"
            and not structure.get("destroyed", False)
        ), None)

    def _lander_footprint_cells(self) -> set[tuple[int, int]]:
        hub = self._lander_hub()
        lz_x = int(getattr(self, "lz_x", self.world.center))
        lz_y = int(getattr(self, "lz_y", self.world.center))
        if hub is not None:
            return self._structure_footprint_cells(
                "eclss_lander_hub", lz_x, lz_y, hub
            )
        return {
            (lz_x + dx, lz_y + dy)
            for dx in range(
                self.LANDER_FOOTPRINT_MIN_DX,
                self.LANDER_FOOTPRINT_MAX_DX + 1,
            )
            for dy in range(
                self.LANDER_FOOTPRINT_MIN_DY,
                self.LANDER_FOOTPRINT_MAX_DY + 1,
            )
        }

    def _lander_airlock_position(self) -> tuple[int, int]:
        hub = self._lander_hub() or {}
        lz_x = int(getattr(self, "lz_x", self.world.center))
        lz_y = int(getattr(self, "lz_y", self.world.center))
        return (
            int(hub.get("airlock_x", lz_x + self.LANDER_AIRLOCK_DX)),
            int(hub.get("airlock_y", lz_y + self.LANDER_AIRLOCK_DY)),
        )

    def _lander_airlock_exterior_position(self) -> tuple[int, int]:
        hub = self._lander_hub() or {}
        lz_x = int(getattr(self, "lz_x", self.world.center))
        lz_y = int(getattr(self, "lz_y", self.world.center))
        return (
            int(hub.get(
                "airlock_exterior_x",
                lz_x + self.LANDER_AIRLOCK_EXTERIOR_DX,
            )),
            int(hub.get(
                "airlock_exterior_y",
                lz_y + self.LANDER_AIRLOCK_EXTERIOR_DY,
            )),
        )

    def _is_lander_footprint_cell(self, x: int, y: int) -> bool:
        return (int(x), int(y)) in self._lander_footprint_cells()

    def _is_lander_airlock_cell(self, x: int, y: int) -> bool:
        return (int(x), int(y)) == self._lander_airlock_position()

    def _estimated_eva_energy_drain_per_tick(
        self, agent: Agent, activity_multiplier: float
    ) -> float:
        """Mirror the fatigue model for conservative EVA route budgeting."""
        endurance = max(
            1,
            int(getattr(agent.genome, "endurance", 5))
            - int(getattr(agent, "_radiation_permanent_endurance_penalty", 0)),
        )
        endurance_mod = max(
            0.5, min(1.5, 1.0 - 0.05 * (endurance - 5))
        )
        gravity = max(0.3, float(self.planet.gravity_g))
        activity = max(0.5, float(activity_multiplier))
        gravity_mod = 1.0 + (
            (gravity - 1.0) * max(0.0, activity - 1.0) * 0.5
        )
        morale_mod = 1.0 + (
            1.0 - max(0.0, min(1.0, float(getattr(agent, "morale", 1.0))))
        ) * 0.3
        disease_bonus = 0.0
        for disease in getattr(agent, "active_diseases", []):
            try:
                disease_bonus += float(
                    disease.get_effects().get("energy_drain_bonus", 0.0)
                ) * (self.SIM_MINUTES_PER_TICK / 5.0)
            except (AttributeError, TypeError, ValueError):
                continue
        return max(
            0.0,
            float(agent.needs.BASE_ENERGY_DECAY)
            * endurance_mod
            * gravity_mod
            * activity
            * morale_mod
            * 1.15  # unsheltered circadian/thermal workload factor
            + disease_bonus,
        )

    def _routine_eva_energy_return_threshold(
        self, agent: Agent, distance_cells: int
    ) -> float:
        # Convert distance to physics ticks. Treating every 100 m cell as a
        # full ten-minute tick double-counted travel despite the motor's
        # two-cell steady pace and recalled healthy crews far too early.
        return_ticks = math.ceil(
            max(0, int(distance_cells)) / max(1, self.EVA_WALK_SPEED_CELLS)
        )
        return min(
            90.0,
            55.0
            + return_ticks
            * self._estimated_eva_energy_drain_per_tick(agent, 1.5),
        )

    def _routine_eva_o2_return_threshold(
        self, agent: Agent, distance_cells: int
    ) -> float:
        return_ticks = math.ceil(
            max(0, int(distance_cells)) / max(1, self.EVA_WALK_SPEED_CELLS)
        )
        return min(
            80.0,
            40.0 + return_ticks * agent.plss_o2_percent_per_tick(1.5),
        )

    def _routine_eva_start_energy_threshold(
        self,
        agent: Agent,
        outbound_cells: int,
        target_return_cells: int,
    ) -> float:
        """Energy needed for travel, one useful work hour and safe return."""
        steady_speed = max(1, self.EVA_WALK_SPEED_CELLS)
        outbound_ticks = math.ceil(max(0, int(outbound_cells)) / steady_speed)
        return_ticks = math.ceil(
            max(0, int(target_return_cells)) / steady_speed
        )
        travel_cost = (
            outbound_ticks + return_ticks
        ) * self._estimated_eva_energy_drain_per_tick(agent, 1.5)
        useful_work_ticks = ticks_for_minutes(
            60.0, self.SIM_MINUTES_PER_TICK
        )
        work_cost = useful_work_ticks * self._estimated_eva_energy_drain_per_tick(
            agent, 2.0
        )
        # Finish above 50% plus a five-point navigation/weather contingency.
        return min(95.0, 55.0 + travel_cost + work_cost)

    def _inside_eva_operating_area(self, x: int, y: int) -> bool:
        return self._distance_from_lz(x, y) <= self.LOCAL_EVA_RADIUS_CELLS

    def _resource_local_campaign_exhausted(self, resource: str) -> bool:
        """Return true after adequate local misses and no usable local hit."""
        misses = self._local_resource_miss_centers.get(resource, set())
        if len(misses) < self.LOCAL_RESOURCE_MISS_LIMIT:
            return False
        return not any(
            self._inside_eva_operating_area(x, y)
            and (x, y, resource) not in self.depleted_cell_resources
            for x, y in self.discovered_resources.get(resource, set())
        )

    def _pressurized_structure_at(self, x: int, y: int) -> dict | None:
        """Use the same commissioned pressure hull for physiology and movement."""
        return next((
            structure for structure in getattr(self, "placed_structures", [])
            if (
                structure.get("type") in {
                    "habitat_module", "greenhouse"
                }
                or bool(structure.get("pressurized", False))
            )
            and structure.get("type") != "eclss_lander_hub"
            and not structure.get("under_construction", False)
            and not structure.get("destroyed", False)
            and float(structure.get("health", 1.0)) > 0.0
            and max(
                abs(x - structure.get("x", -999)),
                abs(y - structure.get("y", -999)),
            ) <= 1
        ), None)

    def _is_pressurized_location(self, x: int, y: int) -> bool:
        return (self._is_lander_footprint_cell(x, y)
                or self._pressurized_structure_at(x, y) is not None)

    def _outbound_work_order_rejection(self, agent: Agent, target: dict) -> str | None:
        """Revalidate the physical job immediately before opening the airlock."""
        if not isinstance(target, dict):
            return "missing_work_order"
        destination = target.get("destination")
        if destination in {"habitat", "shelter", "o2_filling_station"}:
            return None
        if target.get("expedition") or target.get("maintenance"):
            return None

        if destination in {"construction_site", "planned_construction_site"}:
            recipe_name = target.get("recipe")
            recipe = self._get_recipe(str(recipe_name)) if recipe_name else None
            if not recipe:
                return "construction_recipe_unavailable"
            requested_id = target.get("struct_id")
            active_sites = [
                site for site in self.placed_structures
                if site.get("under_construction", False)
                and not site.get("destroyed", False)
                and site.get("type") == recipe_name
            ]
            if requested_id and not any(
                site.get("id") == requested_id for site in active_sites
            ):
                return "construction_site_no_longer_active"
            active_site = next((
                site for site in active_sites
                if (
                    requested_id and site.get("id") == requested_id
                ) or (
                    not requested_id
                    and site.get("x") == target.get("x")
                    and site.get("y") == target.get("y")
                )
            ), None)
            if active_site is not None:
                construction = recipe.get("construction", {})
                crew_limit = max(
                    1, int(construction.get("recommended_crew", 2))
                )
                shared_order = getattr(
                    self.decision_engine, "shared_work_order", {}
                )
                assigned_ids = {
                    crew_id
                    for crew_id, role in shared_order.get(
                        "assignments", {}
                    ).items()
                    if role == "construction"
                } if shared_order.get("site_id") == active_site.get("id") else set()

                claimants = []
                for crew in self.agents:
                    crew_target = (
                        crew.action.target
                        if isinstance(crew.action.target, dict) else {}
                    )
                    same_site = (
                        crew_target.get("struct_id") == active_site.get("id")
                        or (
                            crew_target.get("recipe") == recipe_name
                            and crew_target.get("x") == active_site.get("x")
                            and crew_target.get("y") == active_site.get("y")
                        )
                    )
                    if (
                        crew.id == agent.id
                        or (
                            same_site
                            and crew.action.action_type
                            in {"move", "arrived", "build"}
                        )
                    ):
                        claimants.append(crew)
                claimants.sort(key=lambda crew: (
                    0 if crew.id in assigned_ids else 1,
                    crew.id,
                ))
                permitted_ids = {
                    crew.id for crew in claimants[:crew_limit]
                }
                if agent.id not in permitted_ids:
                    return "construction_crew_capacity_claimed"
            if not requested_id and not active_sites:
                required_structure = recipe.get("requires_structure")
                if required_structure and self._operational_structure_count(
                    str(required_structure)
                ) <= 0:
                    return "required_workshop_unavailable"
                site_x = int(target.get("x", agent.x))
                site_y = int(target.get("y", agent.y))
                staged = self._staged_material_totals(
                    site_x, site_y, purpose_recipe=str(recipe_name)
                )
                if any(
                    staged.get(material, 0) < quantity
                    for material, quantity in recipe.get("materials", {}).items()
                ):
                    return "construction_bom_no_longer_staged"

        if destination == "manufacturing_machine":
            machine_id = str(target.get("machine_id") or "")
            output = target.get("output")
            recipe = self._get_recipe(str(output)) if output else None
            machine = next((
                structure for structure in self.placed_structures
                if str(structure.get("id")) == machine_id
                and not structure.get("under_construction", False)
                and not structure.get("destroyed", False)
            ), None)
            if machine is None or not recipe:
                return "manufacturing_order_unavailable"
            cycle = self._manufacturing_cycles.get(machine_id)
            if cycle and cycle.get("completion_pending"):
                route_owner = cycle.get("route_operator_id")
                if (
                    cycle.get("operator_id") != agent.id
                    and route_owner != agent.id
                ):
                    return "machine_claim_owned_by_other_agent"
            else:
                for other in self.agents:
                    if other.id == agent.id or not isinstance(
                        other.action.target, dict
                    ):
                        continue
                    other_target = other.action.target
                    if (
                        str(other_target.get("machine_id") or "") == machine_id
                        and getattr(other.action, "action_type", "")
                        in {"move", "arrived", "refine"}
                    ):
                        return "machine_claim_owned_by_other_agent"
                staged = self._staged_material_totals(
                    int(machine.get("x", agent.x)),
                    int(machine.get("y", agent.y)),
                    purpose_recipe=str(output),
                )
                if any(
                    staged.get(material, 0) < quantity
                    for material, quantity in recipe.get("materials", {}).items()
                ):
                    return "manufacturing_bom_no_longer_staged"

        resource = target.get("resource")
        if resource and target.get("capacity_recipe"):
            outstanding = self.decision_engine._raw_bom_deficits(
                str(target["capacity_recipe"]),
                self.decision_engine._pooled_materials(),
            )
            if int(outstanding.get(str(resource), 0)) <= 0:
                return "resource_contract_already_satisfied"
        if (
            resource
            and target.get("x") is not None
            and target.get("y") is not None
            and (
                int(target["x"]), int(target["y"]), str(resource)
            ) in self.depleted_cell_resources
        ):
            return "resource_face_depleted"
        return None

    def _reject_invalid_work_order(
        self, agent: Agent, target: dict, reason: str
    ) -> None:
        """Release only the rejected physical reservation; never punish RL."""
        resource = target.get("resource") if isinstance(target, dict) else None
        contract = getattr(agent, "_shared_work_contract", None)
        if (
            resource
            and isinstance(contract, dict)
            and str(contract.get("action", "")).lower() == "gather"
            and contract.get("stock_key") == resource
        ):
            self.decision_engine._close_bounded_gather_contract(agent, contract)
        for marker_name in (
            "_hand_prospect_target", "_active_excavation",
            "_detected_resource_recovery",
        ):
            marker = getattr(agent, marker_name, None)
            if isinstance(marker, dict) and (
                not resource or marker.get("resource") == resource
            ):
                setattr(agent, marker_name, None)
        agent._rl_transition_pending = False
        agent._last_invalid_order = {
            "reason": reason,
            "tick": self.current_tick,
            "action": "move",
            "target": {
                key: target.get(key)
                for key in (
                    "destination", "recipe", "output", "resource",
                    "machine_id", "struct_id", "x", "y",
                )
                if key in target
            },
        }
        agent.action.action_type = "invalid_action_rejected"
        agent.action.target = {
            "invalid_action_rejected": True,
            "reason": reason,
            "requested_action": "move",
        }
        agent.action.ticks_remaining = 1

    def _indoor_activity_position(self, agent: Agent, action: str) -> tuple[int, int] | None:
        """Compartments within the existing lander footprint, not new capacity."""
        if not agent._in_habitat or not self._is_lander_footprint_cell(agent.x, agent.y):
            return None
        if action == "sleep":
            berths = [(-1, -1), (0, -1), (1, -1), (2, -1), (-1, 0), (2, 0)]
            crew_index = next((i for i, crew in enumerate(self.agents) if crew.id == agent.id), 0)
            dx, dy = berths[crew_index % len(berths)]
        elif action == "eat":
            dx, dy = 1, 1
        elif action == "plan_construction":
            dx, dy = 0, 0
        elif action == "service_suit":
            dx, dy = 1, 0
        elif action == "stand_watch":
            index = next((i for i, c in enumerate(self.agents) if c.id == agent.id), 0)
            dx, dy = [(-1, 1), (0, 1), (2, 1), (-1, 2), (1, 2), (2, 2)][index % 6]
        else:
            return None
        return self.lz_x + dx, self.lz_y + dy

    def _indoor_activity_decision(self, agent: Agent, action: str, target: dict) -> tuple[str, dict]:
        pending = getattr(agent, "_pending_indoor_activity", None)
        if not agent._in_habitat or not self._is_lander_footprint_cell(agent.x, agent.y):
            agent._pending_indoor_activity = None
            agent.indoor_location = None
            return action, target
        alert = getattr(agent, "_team_emergency_alert", {}) or {}
        if (target.get("medical_response") or alert.get("assigned_sar") or alert.get("assigned_medical")) and not self.decision_engine.sar_recovery_required(agent):
            agent._pending_indoor_activity = None
            return action, target
        # Acute self-care/rescue can interrupt interior transit; no delayed
        # meal or room assignment may suppress an emergency response.
        if action in {"drink", "treat", "rescue", "medical_rest", "refill_o2"}:
            return action, target
        if action == "eat" and (
            agent.needs.hunger <= 45.0
            or not (
                agent.inventory.has_item("emergency_rations")
                or agent.inventory.has_item("ration_pack")
                or self.central_depot_inventory.get("ration_packs", 0) > 0
                or self._colony_resources.get("food_reserve_kcal", 0.0) > 0.0
            )
        ):
            return action, target
        if action == "sleep" and (
            agent.needs.energy <= 15.0
            or (agent.needs.thirst <= 45.0
                and not agent.inventory.has_item("water_packs")
                and self.central_depot_inventory.get("water_packs", 0) <= 0
                and self._colony_resources.get("water_reserve_l", 0.0) <= 0.0)
        ):
            return action, target
        if pending:
            action, target = pending["action"], dict(pending["target"])
        position = self._indoor_activity_position(agent, action)
        if position is None:
            return action, target
        zone = {"sleep": "bunk", "eat": "galley", "plan_construction": "planning_bay",
                "service_suit": "suit_service_bay", "stand_watch": "operations_station"}[action]
        if (agent.x, agent.y) != position:
            agent._pending_indoor_activity = {"action": action, "target": dict(target)}
            agent.indoor_location = {"zone": "corridor", "destination": zone}
            return "move", {"x": position[0], "y": position[1],
                            "destination": "indoor_activity", "indoor_activity": action}
        agent._pending_indoor_activity = None
        agent.indoor_location = {"zone": zone, "berth_owner": agent.id if action == "sleep" else None}
        return action, target

    def _route_queued_indoor_recovery(self, agent: Agent) -> None:
        """EVA-denial recovery uses the same berth route as normal sleep."""
        original = dict(agent.action.target)
        action, target = self._indoor_activity_decision(agent, "sleep", original)
        if action == "move":
            agent.action.action_type = action
            agent.action.target = {**original, **target}
            agent.action.ticks_remaining = 1

    def _return_for_indoor_recovery(self, agent: Agent, reason: str) -> None:
        expedition = getattr(agent, "_active_expedition", None)
        if isinstance(expedition, dict):
            self.decision_engine._recall_expedition_team(agent)
        agent._pending_indoor_activity = None
        agent.action.action_type = "move"
        agent.action.target = {
            "x": self._lander_airlock_position()[0],
            "y": self._lander_airlock_position()[1],
            "destination": "shelter", "indoor_recovery_return": reason,
            "expedition": isinstance(expedition, dict),
        }
        agent.action.ticks_remaining = 1
        agent._rl_transition_pending = False

    def _start_suit_service(self, agent: Agent, pressure_overhaul: bool = False) -> bool:
        """A real indoor maintenance job; its benefit is applied on completion."""
        if not agent._in_habitat:
            self._return_for_indoor_recovery(agent, "suit_service_requires_base")
            return False
        action, target = self._indoor_activity_decision(
            agent, "service_suit", {"pressure_overhaul": pressure_overhaul}
        )
        if action == "move":
            agent.action.action_type, agent.action.target = action, target
            agent.action.ticks_remaining = 1
            return False
        job = getattr(agent, "_suit_service_job", None)
        if not job:
            if pressure_overhaul:
                sources = [agent.inventory.materials, agent.inventory.items,
                           self.central_depot_inventory]
                sources.extend(c.inventory.materials for c in self.agents
                               if c is not agent and c._in_habitat)
                source = next((s for s in sources if s.get("vacuum_gasket_seal", 0) > 0), None)
                if source is None:
                    agent.action.action_type = "rest"
                    agent.action.target = {"eva_denied": "suit_integrity_unsafe",
                                           "missing_part": "vacuum_gasket_seal"}
                    agent.action.ticks_remaining = 1
                    return False
                source["vacuum_gasket_seal"] -= 1
            job = {
                "suit_service_pending": True, "routine_service": True,
                "pressure_overhaul": pressure_overhaul,
                "condition_before": round(agent.suit_condition, 3),
                "remaining_ticks": ticks_for_minutes(
                    120.0 if pressure_overhaul else 60.0, self.SIM_MINUTES_PER_TICK
                ),
            }
            agent._suit_service_job = job
        agent.action.action_type = "service_suit"
        agent.action.target = dict(job)
        agent.action.ticks_remaining = max(1, job["remaining_ticks"])
        return True

    def _eva_water_demand_l(self, agent: Agent, activity: float = 1.5,
                            position: tuple[int, int] | None = None) -> float:
        """Use the physiology model's units/modifiers for a water budget."""
        x, y = position or (agent.x, agent.y)
        temperature = float(self.world.get_cell_info(
            x, y, self.current_tick
        ).get("temperature_c", 20.0))
        thermal = 1.0
        if temperature > 25.0:
            thermal += (temperature - 25.0) * 0.02 * activity
        elif temperature < -10.0:
            thermal += (-10.0 - temperature) * 0.005
        endurance = max(1, agent.genome.endurance - int(getattr(
            agent, "_radiation_permanent_endurance_penalty", 0
        )))
        endurance_mod = max(0.5, min(1.5, 1.0 - 0.05 * (endurance - 5)))
        morale_mod = 1.0 + (1.0 - agent.morale) * 0.3
        disease_l = sum(float(d.get_effects().get("thirst_drain_bonus", 0.0))
                        for d in agent.active_diseases) * (self.SIM_MINUTES_PER_TICK / 5.0) / 35.0
        return (agent.water_requirement_l_per_tick(True, activity)
                * endurance_mod * thermal * morale_mod + disease_l)

    def _movement_route_cell(self, x: int, y: int) -> dict:
        """Reuse deterministic terrain/weather queries only within this tick."""
        getter = self.world.get_cell_info
        key = (self.current_tick, getattr(getter, "__func__", getter))
        if getattr(self, "_movement_route_cell_epoch", None) != key:
            self._movement_route_cell_epoch = key
            self._movement_route_cells = {}
        position = (x, y)
        if position not in self._movement_route_cells:
            self._movement_route_cells[position] = getter(x, y, self.current_tick)
        return self._movement_route_cells[position]

    def _eva_walk_route_ticks(
        self, agent: Agent, route: list[tuple[int, int]]
    ) -> int:
        """Estimate the same terrain/load-limited tick count used by movement."""
        if not route:
            return 0
        route_cost = sum(
            max(
                1.0,
                float(self._movement_route_cell(
                    x, y
                ).get("traversal_cost", 1.0)),
            )
            for x, y in route[1:]
        )
        pace = max(
            0.25,
            float(self.EVA_WALK_SPEED_CELLS)
            * float(agent.movement_speed(float(self.planet.gravity_g))),
        )
        return max(0, int(math.ceil(route_cost / pace)))

    def _eva_return_water_threshold(self, agent: Agent) -> float:
        """Routed walk-back plus 30 minutes, arriving above dehydration risk."""
        cells = self._eva_walk_home_ticks(agent)
        return 35.0 + (cells + ticks_for_minutes(30, self.SIM_MINUTES_PER_TICK)) * self._eva_water_demand_l(agent) * 35.0

    def _eva_walk_home_ticks(self, agent: Agent) -> int | float:
        """Budget the exterior approach actually used by the walking motor."""
        exterior = self._lander_airlock_exterior_position()
        cache_key = (agent.x, agent.y, len(self.placed_structures),
                     len(self.spoil_piles), len(self.cell_excavation_depth),
                     self.current_tick // max(1, int(1440 / self.SIM_MINUTES_PER_TICK)))
        cached = getattr(agent, "_water_return_route_cache", None)
        if not cached or cached[0] != cache_key:
            route = self._surface_vehicle_route(agent.x, agent.y, *exterior)
            agent._water_return_route_cache = (cache_key, route)
        else:
            route = cached[1]
        if not route:
            return float("inf")
        # One extra tick covers the apron-to-chamber leg. The caller adds its
        # own physiological contingency for the pressure cycle and queue.
        return self._eva_walk_route_ticks(agent, route) + 1

    def _prepare_eva_water(self, agent: Agent,
                           target_position: tuple[int, int] | None,
                           *, rover_outbound: bool = False) -> bool:
        """Pack real water at the airlock; never grant hydration or liters."""
        # A route-specific target is live only while its top-off is pending.
        # Drinking consumes time, so normal metabolism can lower thirst by a
        # point before the next decision. Reusing the already-completed target
        # then rearmed another drink forever even though the recalculated route
        # and two carried liters were safe.
        hydration_pending = bool(getattr(agent, "_eva_hydration_pending", False))
        hydration_target = (
            float(getattr(agent, "_eva_hydration_target", 80.0))
            if hydration_pending else 80.0
        )
        if not hydration_pending:
            agent._eva_hydration_target = 80.0
        if agent.needs.thirst < hydration_target:
            agent._eva_hydration_pending = True
            agent.action.action_type = "idle"
            agent.action.target = {"eva_denied": "preflight_hydration_required"}
            agent.action.ticks_remaining = 0
            return False
        agent._eva_hydration_pending = False
        agent._eva_hydration_target = 80.0
        if target_position is None:
            for order in (getattr(agent, "last_decision", {}).get("target", {}),
                          agent.action.target):
                if isinstance(order, dict) and order.get("x") is not None and order.get("y") is not None:
                    target_position = (int(order["x"]), int(order["y"]))
                    break
        expedition = getattr(agent, "_active_expedition", None)
        if target_position is None and isinstance(expedition, dict):
            if expedition.get("target_x") is not None and expedition.get("target_y") is not None:
                target_position = (int(expedition["target_x"]), int(expedition["target_y"]))
        default_exit = target_position is None and agent._in_habitat
        target_position = target_position or self._lander_airlock_exterior_position()
        required_l = float("inf")
        if default_exit:
            # No caller supplied a field destination: this is only the one-cell
            # airlock crossing. Avoid a full world A* search on every routine
            # preflight while retaining the real two-pack stock transaction.
            required_l = ticks_for_minutes(30, self.SIM_MINUTES_PER_TICK) * self._eva_water_demand_l(agent)
            required_l += ticks_for_minutes(60, self.SIM_MINUTES_PER_TICK) * self._eva_water_demand_l(agent, 2.0)
            required_l = max(0.0, required_l - max(0.0, agent.needs.thirst - 35.0) / 35.0)
            outbound = inbound = [(agent.x, agent.y)]
        else:
            # Terrain/routes are stable across most ticks. Cache this
            # read-only budget lookup; otherwise every agent's preflight would
            # rerun the same expensive A* search on the 1000x1000 grid.
            route_cache = getattr(self, "_eva_route_cache", {})
            route_context = (
                len(self.placed_structures), len(self.spoil_piles),
                len(self.cell_excavation_depth),
                self.current_tick // max(1, int(1440 / self.SIM_MINUTES_PER_TICK)),
            )
            def cached_route(start, end):
                key = (start, end, route_context)
                if key not in route_cache:
                    route_cache[key] = self._surface_vehicle_route(*start, *end)
                return route_cache[key]
            if len(route_cache) > 512:
                route_cache.clear()
            self._eva_route_cache = route_cache
            outbound = cached_route((agent.x, agent.y), target_position)
            inbound = cached_route(target_position, self._lander_airlock_position())
        if outbound and inbound and not default_exit:
            expedition = getattr(agent, "_active_expedition", None)
            rover_outbound = rover_outbound or (
                isinstance(expedition, dict) and expedition.get("transport") == "crew_rover"
            )
            outbound_ticks = self._eva_walk_route_ticks(agent, outbound)
            if rover_outbound:
                speed = max(1, int(float(self.mission_profile.surface_fleet["crew_rover"].get(
                    "max_speed_kph", 10.0
                )) * self.SIM_HOURS_PER_TICK * 1000.0 / self.mission_profile.grid_cell_meters))
                outbound_ticks = math.ceil(sum(max(1.0, float(self.world.get_cell_info(
                    x, y, self.current_tick
                ).get("traversal_cost", 1.0))) for x, y in outbound[1:]) / speed)
            # Drive out if a rover is assigned, but carry enough to WALK home
            # after a failure at the farthest point, plus work/contingency.
            travel_ticks = outbound_ticks + self._eva_walk_route_ticks(
                agent, inbound
            )
            required_l = (
                (travel_ticks + ticks_for_minutes(30, self.SIM_MINUTES_PER_TICK))
                * max(self._eva_water_demand_l(agent), self._eva_water_demand_l(agent, position=target_position))
                + ticks_for_minutes(60, self.SIM_MINUTES_PER_TICK)
                * self._eva_water_demand_l(agent, 2.0, target_position)
                - max(0.0, agent.needs.thirst - 35.0) / 35.0
            )
        if required_l > 2.0:
            # If one real pre-departure drink makes the same route safe,
            # request it instead of retrying forever at the ordinary 80% floor.
            needed_hydration = agent.needs.thirst + (required_l - 2.0) * 35.0 + 2.0
            if needed_hydration <= 99.0:
                agent._eva_hydration_pending = True
                agent._eva_hydration_target = needed_hydration
                agent.action.action_type = "idle"
                agent.action.target = {"eva_denied": "preflight_hydration_required"}
                agent.action.ticks_remaining = 0
                return False
            agent.action.action_type = "idle"
            agent.action.target = {"eva_denied": "insufficient_round_trip_water",
                                   "required_carried_l": round(required_l, 2) if math.isfinite(required_l) else None}
            agent.action.ticks_remaining = 1
            return False
        # Refill both existing one-liter drink bags, including departures
        # between the periodic depot restocks. Every liter leaves real stock.
        depot = self.central_depot_inventory
        while agent.inventory.items.get("water_packs", 0) < 2:
            if depot.get("water_packs", 0) > 0:
                depot["water_packs"] -= 1
            elif self._colony_resources.get("water_reserve_l", 0.0) >= 1.0:
                self._colony_resources["water_reserve_l"] -= 1.0
            else:
                break
            agent.inventory.add_item("water_packs", 1)
        if agent.inventory.items.get("water_packs", 0) < max(1, math.ceil(required_l)):
            agent.action.action_type = "idle"
            agent.action.target = {"eva_denied": "portable_water_unavailable"}
            agent.action.ticks_remaining = 1
            return False
        return True

    def _prepare_agent_for_eva(
        self,
        agent: Agent,
        mission_critical_maintenance: bool = False,
        life_support_bootstrap: bool = False,
        defer_pressure_transition: bool = False,
        target_position: tuple[int, int] | None = None,
        rover_outbound: bool = False,
    ) -> bool:
        """Validate real suit consumables before opening the outer airlock."""
        # This is a departure gate, never a portable workshop or outdoor bunk.
        if not agent._in_habitat:
            if (agent.needs.energy < self.MIN_ROUTINE_EVA_START_ENERGY_PCT
                or agent.suit_condition < self.MIN_ROUTINE_EVA_EXIT_SUIT_CONDITION
                or agent.suit_integrity < self.MIN_EVA_EXIT_SUIT_INTEGRITY):
                self._return_for_indoor_recovery(agent, "eva_preflight_requires_base")
                return False
            return True
        if agent._in_habitat and not self._prepare_eva_water(
            agent, target_position, rover_outbound=rover_outbound
        ):
            return False
        needs_o2_support = bool(getattr(agent, "_needs_o2_support", False))
        if not needs_o2_support:
            result = agent.exit_habitat(has_atmosphere=True)
            if result.get("exited") and defer_pressure_transition:
                agent._in_habitat = True
            return bool(result.get("exited"))

        surface_hazard = next((
            event for event in getattr(self, "_active_events", [])
            if (
                event.event_type if hasattr(event, "event_type")
                else event.get("type", "") if isinstance(event, dict)
                else ""
            ) in {
                "solar_flare", "stellar_flare", "flare",
                "dust_storm", "sandstorm",
                "meteor_shower", "micrometeorite_shower",
            }
        ), None)
        if surface_hazard is not None and not mission_critical_maintenance:
            hazard_type = (
                surface_hazard.event_type
                if hasattr(surface_hazard, "event_type")
                else surface_hazard.get("type", "surface_hazard")
            )
            agent.action.action_type = "rest"
            agent.action.target = {
                "eva_denied": "active_surface_hazard",
                "hazard": hazard_type,
            }
            agent.action.ticks_remaining = 1
            return False

        # Every route into an unpressurized work area passes this final
        # interlock.  Individual MOVE routes also make a distance-scaled
        # estimate, but BUILD/SURVEY/expedition paths must not be able to open
        # the airlock with a crew member already too tired for useful work.
        if (
            float(agent.needs.energy) < self.MIN_ROUTINE_EVA_START_ENERGY_PCT
            and not mission_critical_maintenance
        ):
            recovery_ticks = ticks_for_minutes(
                120.0, self.SIM_MINUTES_PER_TICK
            )
            agent.action.action_type = "sleep"
            agent.action.target = {
                "eva_denied": "insufficient_round_trip_energy",
                "required_energy_pct": self.MIN_ROUTINE_EVA_START_ENERGY_PCT,
                "habitat": True,
                "ticks": recovery_ticks,
            }
            agent.action.ticks_remaining = recovery_ticks
            self._route_queued_indoor_recovery(agent)
            return False

        if agent.plss_co2_scrubber_pct < 25.0 or agent.plss_suit_battery_pct < 25.0:
            agent.action.action_type = "idle"
            agent.action.target = {"eva_denied": "plss_not_recharged"}
            agent.action.ticks_remaining = 1
            return False

        # Routine post-EVA work handles dust removal, bearing inspection and
        # joint lubrication. It costs crew time, but does not consume a
        # pressure gasket unless structural integrity was actually damaged.
        minimum_suit_condition = (
            self.MIN_EVA_EXIT_SUIT_CONDITION
            if mission_critical_maintenance
            else self.MIN_ROUTINE_EVA_EXIT_SUIT_CONDITION
        )
        if getattr(agent, "suit_condition", 1.0) < minimum_suit_condition:
            self._start_suit_service(agent)
            return False

        # A worn pressure garment must be serviced before the outer airlock
        # opens. One stocked vacuum gasket represents a joint/seal overhaul;
        # without a service part, the safe outcome is to remain inside.
        if agent.suit_integrity < self.MIN_EVA_EXIT_SUIT_INTEGRITY:
            self._start_suit_service(agent, pressure_overhaul=True)
            return False

        minimum_o2_reserve = (
            self.MIN_EVA_EXIT_O2_PCT
            if mission_critical_maintenance
            else self.MIN_ROUTINE_EVA_EXIT_O2_PCT
        )
        if agent._current_canister_remaining < minimum_o2_reserve:
            if not agent.inventory.has_item("oxygen_canisters"):
                depot = getattr(self, "central_depot_inventory", {})
                isru_pending = self.structures_built.get("isru_o2_unit", 0) == 0
                if life_support_bootstrap:
                    protected_canisters = 0
                elif mission_critical_maintenance and isru_pending:
                    # Exterior assets still receive maintenance, but the final
                    # cylinder remains reserved to commission local O2.
                    protected_canisters = 1
                elif mission_critical_maintenance:
                    protected_canisters = 0
                else:
                    protected_canisters = self.MIN_CENTRAL_MAINTENANCE_O2_CANISTERS
                if depot.get("oxygen_canisters", 0) > protected_canisters:
                    depot["oxygen_canisters"] -= 1
                    agent.inventory.add_item("oxygen_canisters", 1)
                # Bulk O2 is not a magic spare cylinder. Once delivered full
                # bottles are gone, reusable empty vessels must be carried to
                # an operational ISRU filling station.
                if (
                    not agent.inventory.has_item("oxygen_canisters")
                    and self._nearest_o2_filling_station(agent) is not None
                    and self._colony_resources.get("o2_reserve_kg", 0.0)
                    >= Agent.PLSS_CANISTER_O2_KG
                ):
                    # Start the physical transaction now.  Merely queueing a
                    # one-tick action here caused ActionState.tick() to expire
                    # it before the motor dispatcher could ever run it.
                    service = self._execute_o2_refill_action(agent)
                    if service.get("moving"):
                        agent.action.action_type = "move"
                        agent.action.target = {
                            "x": service.get("station_x"),
                            "y": service.get("station_y"),
                            "destination": "o2_filling_station",
                            "o2_service": True,
                            "eva_denied": "o2_service_required",
                        }
                        agent.action.ticks_remaining = 1
                    elif service.get("refilled"):
                        agent.action.action_type = "refill_o2"
                        agent.action.target = {
                            **service,
                            "eva_denied": "o2_service_required",
                        }
                        agent.action.ticks_remaining = ticks_for_minutes(
                            15.0, self.SIM_MINUTES_PER_TICK
                        )
                    else:
                        agent.action.action_type = "idle"
                        agent.action.target = {
                            "eva_denied": service.get(
                                "reason", "o2_refill_unavailable"
                            )
                        }
                        agent.action.ticks_remaining = 1
                    return False

        result = agent.exit_habitat(
            has_atmosphere=False,
            minimum_o2_reserve=minimum_o2_reserve,
            minimum_suit_integrity=self.MIN_EVA_EXIT_SUIT_INTEGRITY,
            minimum_suit_condition=minimum_suit_condition,
        )
        if not result.get("exited"):
            agent.action.action_type = "idle"
            agent.action.target = {
                "eva_denied": result.get("reason", "life_support_unavailable"),
                "suit_o2": round(agent._current_canister_remaining, 1),
            }
            agent.action.ticks_remaining = 1
            return False
        if defer_pressure_transition:
            # A route can begin with one or more steps across the pressurized
            # 4 x 4 lander footprint. Preflight is complete, but cabin pressure
            # remains authoritative until the motor crosses the outer edge.
            agent._in_habitat = True
        return True

    def _nearest_o2_filling_station(self, agent: Agent) -> dict | None:
        """Return a completed powered O2 plant with a cylinder manifold.

        A delivered crew lander includes its own OGS service port and is the
        bootstrap fallback. Once an ISRU plant exists, its outdoor compressor
        skid is preferred so locally produced oxygen is serviced physically at
        the machine shown on the map.
        """
        if self._colony_resources.get("energy_stored_kwh", 0.0) <= 0.0:
            return None
        isru_stations = self._connected_life_support_structures(
            "isru_o2_unit"
        )
        bootstrap_stations = [
            structure for structure in getattr(self, "placed_structures", [])
            if structure.get("type") in {"eclss_lander_hub", "advanced_eclss"}
            and not structure.get("under_construction", False)
            and not structure.get("destroyed", False)
            and float(structure.get("health", 1.0)) > 0.0
        ]
        # A crew member already inside the pressurised campus always uses its
        # protected service manifold.  Sending them outdoors to the local ISRU
        # solely because that skid is newer caused a deadlock when the PLSS was
        # due for refill and the suit was simultaneously due for service.  An
        # astronaut already outdoors still visits the physical ISRU compressor.
        stations = (
            bootstrap_stations
            if agent._in_habitat and bootstrap_stations
            else isru_stations or bootstrap_stations
        )
        if not stations:
            return None
        return min(
            stations,
            key=lambda structure: (
                abs(agent.x - int(structure.get("x", agent.x)))
                + abs(agent.y - int(structure.get("y", agent.y)))
            ),
        )

    def _fill_agent_o2_at_isru(
        self,
        agent: Agent,
        station: dict,
        *,
        internal_manifold: bool = False,
    ) -> dict:
        """Top off the active PLSS and recharge reusable empty cylinders.

        The delivered lander ECLSS feeds service manifolds inside the
        pressurised bootstrap campus.  Using one of those ports is an indoor
        operation; an astronaut with a nearly empty PLSS must not be asked to
        perform an otherwise impossible EVA merely to reach the outdoor lander
        coordinate.  Local ISRU compressor skids still require physical
        proximity.
        """
        sx, sy = int(station.get("x", agent.x)), int(station.get("y", agent.y))
        if (
            max(abs(agent.x - sx), abs(agent.y - sy)) > 1
            and not internal_manifold
        ):
            return {"filled": False, "reason": "not_at_o2_station"}
        if self._colony_resources.get("energy_stored_kwh", 0.0) <= 0.0:
            return {"filled": False, "reason": "o2_station_unpowered"}

        reserve_floor = (
            self.MIN_CENTRAL_MAINTENANCE_O2_CANISTERS
            * Agent.PLSS_CANISTER_O2_KG
        )
        reserve = float(self._colony_resources.get("o2_reserve_kg", 0.0))
        available = max(0.0, reserve - reserve_floor)
        active_fraction = max(
            0.0, min(1.0, float(agent._current_canister_remaining) / 100.0)
        )
        active_topoff_kg = (1.0 - active_fraction) * Agent.PLSS_CANISTER_O2_KG
        topped_off = False
        if active_topoff_kg > 1e-6 and available + 1e-9 >= active_topoff_kg:
            available -= active_topoff_kg
            reserve -= active_topoff_kg
            agent._current_canister_remaining = 100.0
            agent.needs.o2_supply = 100.0
            topped_off = True

        empty_count = int(agent.inventory.items.get("empty_oxygen_canisters", 0))
        filled_spares = 0
        # One cylinder per service cycle keeps the compressor transaction and
        # UI duration visible instead of converting an arbitrary stack at once.
        if empty_count > 0 and available + 1e-9 >= Agent.PLSS_CANISTER_O2_KG:
            agent.inventory.items["empty_oxygen_canisters"] = empty_count - 1
            if agent.inventory.items["empty_oxygen_canisters"] <= 0:
                del agent.inventory.items["empty_oxygen_canisters"]
            agent.inventory.items["oxygen_canisters"] = (
                agent.inventory.items.get("oxygen_canisters", 0) + 1
            )
            reserve -= Agent.PLSS_CANISTER_O2_KG
            filled_spares = 1

        self._colony_resources["o2_reserve_kg"] = max(0.0, reserve)
        return {
            "filled": topped_off or filled_spares > 0,
            "active_topoff_kg": round(active_topoff_kg if topped_off else 0.0, 4),
            "spares_filled": filled_spares,
            "station_x": sx,
            "station_y": sy,
        }

    def _execute_o2_refill_action(self, agent: Agent) -> dict:
        """Swap a delivered bottle or physically visit an ISRU filling point."""
        depot = getattr(self, "central_depot_inventory", {})
        lz_x = getattr(self, "lz_x", agent.spawn_x)
        lz_y = getattr(self, "lz_y", agent.spawn_y)
        airlock_x, airlock_y = self._lander_airlock_position()
        depot_accessible = agent._in_habitat or max(
            abs(agent.x - lz_x), abs(agent.y - lz_y)
        ) <= 1

        if agent.inventory.has_item("oxygen_canisters"):
            # A full active PLSS plus a full carried bottle is the desired EVA
            # configuration.  Loading the spare into an already-full regulator
            # used to delete a pressure vessel from inventory and trigger an
            # endless refill/reload cycle.
            if (
                getattr(agent, "_has_active_o2_canister", False)
                and agent._current_canister_remaining >= 99.9
            ):
                return {
                    "refilled": True,
                    "source": "carried_spare_canister",
                    "spare_ready": True,
                }
            loaded = bool(agent.load_o2_canister().get("loaded"))
            return {"refilled": loaded, "source": "carried_full_canister"}
        if (
            depot_accessible
            and depot.get("oxygen_canisters", 0)
            > self.MIN_CENTRAL_MAINTENANCE_O2_CANISTERS
        ):
            depot["oxygen_canisters"] -= 1
            agent.inventory.add_item("oxygen_canisters", 1)
            if (
                getattr(agent, "_has_active_o2_canister", False)
                and agent._current_canister_remaining >= 99.9
            ):
                return {
                    "refilled": True,
                    "source": "central_depot_spare_canister",
                    "spare_ready": True,
                }
            loaded = bool(agent.load_o2_canister().get("loaded"))
            return {"refilled": loaded, "source": "central_depot_full_canister"}

        station = self._nearest_o2_filling_station(agent)
        if station is None:
            return {"refilled": False, "reason": "no_powered_o2_filling_station"}

        # Reusable shells are kept in the habitat depot between sorties. The
        # astronaut physically carries one to the compressor skid.
        if (
            depot_accessible
            and agent.inventory.items.get("empty_oxygen_canisters", 0) <= 0
            and depot.get("empty_oxygen_canisters", 0) > 0
        ):
            depot["empty_oxygen_canisters"] -= 1
            agent.inventory.add_item("empty_oxygen_canisters", 1)

        # Bootstrap pressurised modules share the delivered ECLSS service
        # manifold in this base-campus abstraction.  The transfer is still
        # metered against bulk O2 and power in _fill_agent_o2_at_isru.
        internal_manifold = bool(
            agent._in_habitat
            and station.get("type") in {"eclss_lander_hub", "advanced_eclss"}
        )
        if internal_manifold:
            result = self._fill_agent_o2_at_isru(
                agent,
                station,
                internal_manifold=True,
            )
            return {
                "refilled": bool(result.get("filled")),
                "source": station.get("type", "o2_filling_station"),
                "internal_manifold": True,
                **result,
            }

        sx, sy = int(station.get("x", agent.x)), int(station.get("y", agent.y))
        distance = max(abs(agent.x - sx), abs(agent.y - sy))
        active_expedition = getattr(agent, "_active_expedition", None)
        active_rover = (
            self.surface_fleet.crew_rover_for_expedition(
                str(active_expedition.get("id", ""))
            )
            if isinstance(active_expedition, dict)
            else None
        )
        if (
            distance > 1
            and not agent._in_habitat
            and isinstance(active_expedition, dict)
            and active_expedition.get("transport") == "crew_rover"
            and active_expedition.get("status")
            in {"outbound", "working", "returning"}
            and active_rover is not None
            and active_rover.state == "in_use"
            and (agent.x, agent.y) == (active_rover.x, active_rover.y)
        ):
            # The compressor may be only a few cells away, but an assigned
            # passenger cannot silently leave an in-use vehicle and invalidate
            # the two-person mission. Recall the pair; service happens after
            # the rover completes its physical return to the hub.
            self.decision_engine._recall_expedition_team(agent)
            return {
                "refilled": False,
                "rover_returning": True,
                "return_x": airlock_x,
                "return_y": airlock_y,
                "station_x": sx,
                "station_y": sy,
            }
        if distance > 1:
            if agent._in_habitat:
                # A direct 3-8 cell service traverse needs much less O2 than a
                # normal EVA. It is still denied if the remaining tank cannot
                # cover the trip plus a 10% contingency.
                service_reserve = max(
                    10.0,
                    (distance + 2) * agent.plss_o2_percent_per_tick(1.5) * 2.0,
                )
                exit_result = agent.exit_habitat(
                    has_atmosphere=False,
                    minimum_o2_reserve=service_reserve,
                    minimum_suit_integrity=self.MIN_EVA_EXIT_SUIT_INTEGRITY,
                    minimum_suit_condition=self.MIN_EVA_EXIT_SUIT_CONDITION,
                )
                if not exit_result.get("exited"):
                    return {
                        "refilled": False,
                        "reason": exit_result.get("reason", "o2_service_requires_assistance"),
                        "station_x": sx,
                        "station_y": sy,
                    }
            dx, dy = self._cardinal_step_toward(agent, sx, sy)
            agent.x = max(0, min(self.world.map_size - 1, agent.x + dx))
            agent.y = max(0, min(self.world.map_size - 1, agent.y + dy))
            return {
                "refilled": False,
                "moving": True,
                "station_x": sx,
                "station_y": sy,
            }

        result = self._fill_agent_o2_at_isru(agent, station)
        return {
            "refilled": bool(result.get("filled")),
            "source": station.get("type", "o2_filling_station"),
            **result,
        }

    def _mining_staging_point(self, agent: Agent) -> tuple[int, int]:
        """Choose an extraction point beyond the reserved building campus."""
        directions = (
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1),
        )
        direction_index = sum(ord(char) for char in agent.id) % len(directions)
        lz_x = getattr(self, "lz_x", self.world.center)
        lz_y = getattr(self, "lz_y", self.world.center)
        occupied = {
            (structure.get("x"), structure.get("y"))
            for structure in getattr(self, "placed_structures", [])
            if not structure.get("destroyed", False)
        }
        for extra_radius in range(0, 4):
            radius = self.MIN_EXTRACTION_RADIUS_CELLS + extra_radius
            reachable = []
            for offset in range(len(directions)):
                dx, dy = directions[(direction_index + offset) % len(directions)]
                candidate = (
                    max(0, min(self.world.map_size - 1, lz_x + dx * radius)),
                    max(0, min(self.world.map_size - 1, lz_y + dy * radius)),
                )
                if (
                    candidate not in occupied
                    and not self._is_construction_protected_cell(*candidate)
                ):
                    route = self._surface_vehicle_route(
                        *self._lander_airlock_position(), *candidate
                    )
                    if route:
                        reachable.append((len(route), offset, candidate))
            if reachable:
                # Cardinal travel makes a diagonal 18-cell sector twice as
                # far away. Prefer the shortest real apron route, retaining
                # the seeded sector only as a tie-break, not a safety penalty.
                return min(reachable)[2]
        # Never excavate beneath an occupied structure if every apron is full.
        return lz_x, lz_y

    def _send_loaded_agent_to_base(
        self, agent: Agent, current_weight: float, carry_cap_kg: float
    ):
        """Start a physical return when no payload capacity remains."""
        logger.info(
            f"{agent.name} physical carry capacity reached "
            f"({current_weight:.1f}kg/{carry_cap_kg:.1f}kg) — returning to depot"
        )
        lz_x = getattr(self, "lz_x", agent.spawn_x)
        lz_y = getattr(self, "lz_y", agent.spawn_y)
        expedition = getattr(agent, "_active_expedition", None)
        airlock_x, airlock_y = self._lander_airlock_position()
        is_expedition = isinstance(expedition, dict)
        if is_expedition and expedition.get("transport") == "crew_rover":
            self.decision_engine._recall_expedition_team(agent)
            return_target = {
                "x": airlock_x, "y": airlock_y,
                "destination": "shelter", "expedition": True,
            }
            agent.action.action_type = "move"
            agent.action.target = return_target
            agent.action.ticks_remaining = 1
            if self._move_resource_rover_team(agent, return_target):
                return
        step_speed = (
            int(expedition.get(
                "move_speed_cells", self.EXPEDITION_MOVE_SPEED_CELLS
            )) if is_expedition else self.EVA_WALK_SPEED_CELLS
        )
        previous_x, previous_y = agent.x, agent.y
        dx, dy = self._cardinal_step_toward(
            agent, airlock_x, airlock_y, step_speed
        )
        agent.x += dx
        agent.y += dy
        if (
            is_expedition
            and expedition.get("transport") == "crew_rover"
            and expedition.get("role") == "lead"
        ):
            self.surface_fleet.record_crew_rover_movement(
                str(expedition.get("id")),
                previous_x,
                previous_y,
                agent.x,
                agent.y,
            )
        if self._is_lander_airlock_cell(agent.x, agent.y):
            agent.enter_habitat()
            if (
                is_expedition
                and expedition.get("transport") == "crew_rover"
                and expedition.get("role") == "lead"
            ):
                self._complete_crew_rover_expedition(expedition)
            agent._active_expedition = None
            agent._pending_expedition = None
        elif is_expedition:
            agent._active_expedition["status"] = "returning"
        agent.action.action_type = "move"
        agent.action.target = {
            "x": airlock_x,
            "y": airlock_y,
            "dx": dx,
            "dy": dy,
            "destination": "shelter",
            "expedition": is_expedition,
        }
        agent.action.ticks_remaining = 1

    def _survey_radius_cells(self) -> int:
        """Remote geological knowledge radius unlocked by science infrastructure."""
        if self.structures_built.get("laboratory_module", 0) > 0:
            return 200
        if self.structures_built.get("research_workbench", 0) > 0:
            return 50
        return self.LOCAL_EVA_RADIUS_CELLS

    def _operational_structure_count(self, structure_type: str) -> int:
        """Count commissioned, intact physical assets of one type."""
        return sum(
            1
            for structure in getattr(self, "placed_structures", [])
            if structure.get("type") == structure_type
            and not structure.get("under_construction", False)
            and not structure.get("destroyed", False)
            and float(structure.get("health", 1.0)) > 0.0
        )

    def _operational_structures(self, structure_type: str) -> list[dict]:
        """Return commissioned physical assets eligible for capacity credit."""
        return [
            structure
            for structure in getattr(self, "placed_structures", [])
            if structure.get("type") == structure_type
            and not structure.get("under_construction", False)
            and not structure.get("destroyed", False)
            and float(structure.get("health", 1.0)) > 0.0
        ]

    def _life_support_network_nodes(self) -> list[dict]:
        """Return powered, commissioned campus water/O2 distribution nodes."""
        if float(getattr(self, "_colony_resources", {}).get(
            "energy_stored_kwh", 0.0
        )) <= 0.0:
            return []
        return self._operational_structures("life_support_distribution_grid")

    @staticmethod
    def _life_support_structure_key(structure: dict) -> str:
        """Stable identity for topology checks, including compact test assets."""
        structure_id = str(structure.get("id", "")).strip()
        if structure_id:
            return structure_id
        return "{}@{},{}".format(
            structure.get("type", "structure"),
            int(structure.get("x", 0)),
            int(structure.get("y", 0)),
        )

    def _route_utility_trench(
        self,
        start: tuple[int, int],
        goals: set[tuple[int, int]],
        blocked: set[tuple[int, int]],
        start_footprint: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Route one orthogonal buried corridor around occupied ground.

        The 100 m tactical cell is a civil-planning parcel, so a diagonal
        corner cut would have an undefined pipe length and could clip a
        foundation.  A small turn penalty produces inspectable trunk routes
        while terrain traversal cost still decides between equal corridors.
        """
        if start in goals:
            return [start]
        lz_x = getattr(self, "lz_x", self.world.center)
        lz_y = getattr(self, "lz_y", self.world.center)
        margin = self.CONSTRUCTION_GROUND_BUFFER_CELLS + 1
        radius = self.CONSTRUCTION_CAMPUS_RADIUS_CELLS + margin
        min_x = max(0, int(lz_x) - radius)
        max_x = min(self.world.map_size - 1, int(lz_x) + radius)
        min_y = max(0, int(lz_y) - radius)
        max_y = min(self.world.map_size - 1, int(lz_y) + radius)

        def heuristic(x: int, y: int) -> float:
            return float(min(
                abs(x - gx) + abs(y - gy) for gx, gy in goals
            ))

        start_state = (int(start[0]), int(start[1]), 0, 0)
        frontier: list[tuple[float, float, int, tuple[int, int, int, int]]] = []
        serial = 0
        heapq.heappush(
            frontier,
            (heuristic(*start), 0.0, serial, start_state),
        )
        best_cost = {start_state: 0.0}
        previous: dict[
            tuple[int, int, int, int],
            tuple[int, int, int, int] | None,
        ] = {start_state: None}
        directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        terminal = None
        while frontier:
            _, cost, _, state = heapq.heappop(frontier)
            x, y, prior_dx, prior_dy = state
            if cost > best_cost.get(state, float("inf")) + 1e-9:
                continue
            if (x, y) in goals:
                terminal = state
                break
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                coord = (nx, ny)
                if not (min_x <= nx <= max_x and min_y <= ny <= max_y):
                    continue
                if (
                    coord in blocked
                    and coord not in goals
                    and coord not in start_footprint
                ):
                    continue
                cell = self.world.get_cell_info(nx, ny, self.current_tick)
                if not cell.get("traversable", True):
                    continue
                traversal = max(1.0, float(cell.get("traversal_cost", 1.0)))
                turn = (
                    0.18
                    if (prior_dx or prior_dy)
                    and (dx, dy) != (prior_dx, prior_dy)
                    else 0.0
                )
                next_cost = cost + 1.0 + (traversal - 1.0) * 0.08 + turn
                next_state = (nx, ny, dx, dy)
                if next_cost + 1e-9 >= best_cost.get(
                    next_state, float("inf")
                ):
                    continue
                best_cost[next_state] = next_cost
                previous[next_state] = state
                serial += 1
                heapq.heappush(
                    frontier,
                    (
                        next_cost + heuristic(nx, ny),
                        next_cost,
                        serial,
                        next_state,
                    ),
                )
        if terminal is None:
            return []
        route: list[tuple[int, int]] = []
        cursor = terminal
        while cursor is not None:
            route.append((cursor[0], cursor[1]))
            cursor = previous[cursor]
        route.reverse()
        return route

    def _surface_vehicle_route(
        self,
        start_x: int,
        start_y: int,
        target_x: int,
        target_y: int,
    ) -> list[tuple[int, int]]:
        """Plan a four-neighbour vehicle route around occupied surface cells.

        Commissioned utility corridors are buried and compacted, so they are
        deliberately crossable.  Active construction footprints, exposed
        excavation and spoil are surface obstacles.  The start/target site's
        own footprint is opened as an access apron.
        """
        start = (int(start_x), int(start_y))
        target = (int(target_x), int(target_y))
        if start == target:
            return [start]
        blocked: set[tuple[int, int]] = set(getattr(self, "spoil_piles", {}))
        blocked.update(
            coord for coord, depth in getattr(
                self, "cell_excavation_depth", {}
            ).items() if depth > 0
        )
        endpoint_access: set[tuple[int, int]] = {start, target}
        for structure in getattr(self, "placed_structures", []):
            if structure.get("destroyed", False):
                continue
            footprint = self._structure_footprint_cells(
                str(structure.get("type", "structure")),
                int(structure.get("x", 0)),
                int(structure.get("y", 0)),
                structure,
            )
            if start in footprint or target in footprint:
                endpoint_access.update(footprint)
            else:
                blocked.update(footprint)
        blocked.difference_update(endpoint_access)

        # The same cell is visited by both direct candidates and by several
        # A* headings. Weather is identical within this one route query.
        cell_cache: dict[tuple[int, int], dict] = {}

        def route_cell(x: int, y: int) -> dict:
            key = (x, y)
            if key not in cell_cache:
                cell_cache[key] = self._movement_route_cell(x, y)
            return cell_cache[key]

        def direct_route(x_first: bool) -> list[tuple[int, int]]:
            points = [start]
            x, y = start
            axes = ((0, target[0] - x), (1, target[1] - y))
            if not x_first:
                axes = (axes[1], axes[0])
            for axis, delta in axes:
                step = 1 if delta > 0 else -1
                for _ in range(abs(delta)):
                    if axis == 0:
                        x += step
                    else:
                        y += step
                    if (x, y) in blocked or not route_cell(x, y).get("traversable", True):
                        return []
                    points.append((x, y))
            return points

        # Most audited routes have no crossing obstruction. Validate the two
        # cardinal orderings cheaply before expanding an A* search window.
        direct_routes = [direct_route(True), direct_route(False)]
        direct_routes = [route for route in direct_routes if route]
        if direct_routes:
            return min(direct_routes, key=len)

        direct_span = abs(target[0] - start[0]) + abs(target[1] - start[1])
        search_margin = max(4, min(12, direct_span // 4 + 3))
        min_x = max(0, min(start[0], target[0]) - search_margin)
        max_x = min(
            self.world.map_size - 1,
            max(start[0], target[0]) + search_margin,
        )
        min_y = max(0, min(start[1], target[1]) - search_margin)
        max_y = min(
            self.world.map_size - 1,
            max(start[1], target[1]) + search_margin,
        )
        start_state = (start[0], start[1], 0, 0)
        queue: list[
            tuple[float, float, int, tuple[int, int, int, int]]
        ] = [(float(direct_span), 0.0, 0, start_state)]
        costs = {start_state: 0.0}
        previous: dict[
            tuple[int, int, int, int],
            tuple[int, int, int, int] | None,
        ] = {start_state: None}
        serial = 0
        terminal = None
        for_pop_directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        while queue:
            _, cost, _, state = heapq.heappop(queue)
            x, y, previous_dx, previous_dy = state
            if cost > costs.get(state, float("inf")) + 1e-9:
                continue
            if (x, y) == target:
                terminal = state
                break
            for dx, dy in for_pop_directions:
                nx, ny = x + dx, y + dy
                coord = (nx, ny)
                if not (min_x <= nx <= max_x and min_y <= ny <= max_y):
                    continue
                if coord in blocked:
                    continue
                cell = route_cell(nx, ny)
                if not cell.get("traversable", True):
                    continue
                traversal = max(1.0, float(cell.get("traversal_cost", 1.0)))
                turn_penalty = (
                    0.12
                    if (previous_dx or previous_dy)
                    and (dx, dy) != (previous_dx, previous_dy)
                    else 0.0
                )
                next_cost = (
                    cost + 1.0 + (traversal - 1.0) * 0.12 + turn_penalty
                )
                next_state = (nx, ny, dx, dy)
                if next_cost + 1e-9 >= costs.get(
                    next_state, float("inf")
                ):
                    continue
                costs[next_state] = next_cost
                previous[next_state] = state
                serial += 1
                heuristic = abs(nx - target[0]) + abs(ny - target[1])
                heapq.heappush(
                    queue,
                    (next_cost + heuristic, next_cost, serial, next_state),
                )
        if terminal is None:
            return []
        route = []
        cursor = terminal
        while cursor is not None:
            route.append((cursor[0], cursor[1]))
            cursor = previous[cursor]
        route.reverse()
        return route

    def _life_support_network_snapshot(
        self, *, proposed_structure: dict | None = None,
        reserve_pending: bool = False,
    ) -> dict:
        """Build the finite, routed water/O2 network visible to physics and UI.

        A utility vault is a manifold and pump/control location; it does not
        grant radius-based teleportation.  Commissioned consumers are joined
        by orthogonal buried corridors, foundations and disturbed excavation
        are avoided, and the unique installed corridor length cannot exceed
        the delivered loop's certified line-length budget.
        """
        powered = float(getattr(self, "_colony_resources", {}).get(
            "energy_stored_kwh", 0.0
        )) > 0.0
        structures = [
            structure
            for structure in getattr(self, "placed_structures", [])
            if not structure.get("under_construction", False)
            and not structure.get("destroyed", False)
            and float(structure.get("health", 1.0)) > 0.0
        ]
        if reserve_pending:
            # Only paid-for sites reserve future endpoints. Uncommitted plans
            # grant neither demand nor capacity; unfinished vaults never count
            # as an available pump/manifold during this preview.
            consumer_types = {
                "water_collector", "water_purifier", "greenhouse", "hydroponics",
                "isru_o2_unit", "potable_water_tank", "oxygen_buffer_tank",
                "habitat_module", "medical_station",
            }
            reservations = getattr(self, "_construction_cargo_reservations", {})
            for structure in getattr(self, "placed_structures", []):
                reservation = reservations.get(str(structure.get("id", "")), {})
                funded = bool(structure.get("materials_committed", False)) or bool(
                    float(reservation.get("total_mass_kg", 0.0)) > 0.0
                    and reservation.get("status") not in {"stranded", "cancelled"}
                )
                if (
                    funded and structure.get("under_construction", False)
                    and structure.get("type") in consumer_types
                    and not structure.get("destroyed", False)
                    and float(structure.get("health", 1.0)) > 0.0
                ):
                    structures.append({
                        **structure,
                        "under_construction": False,
                        "_utility_pending": True,
                    })
        if proposed_structure is not None:
            structures.append({
                **proposed_structure,
                "_utility_pending": True,
            })
        signature = (
            powered,
            tuple(sorted(
                (
                    self._life_support_structure_key(structure),
                    str(structure.get("type", "")),
                    int(structure.get("x", 0)),
                    int(structure.get("y", 0)),
                    round(float(structure.get("health", 1.0)), 4),
                )
                for structure in structures
            )),
            tuple(sorted(
                (int(x), int(y))
                for (x, y), depth in getattr(
                    self, "cell_excavation_depth", {}
                ).items()
                if depth > 0
                and max(
                    abs(int(x) - getattr(self, "lz_x", self.world.center)),
                    abs(int(y) - getattr(self, "lz_y", self.world.center)),
                ) <= self.CONSTRUCTION_CAMPUS_RADIUS_CELLS + 2
            )),
            tuple(sorted(
                (int(x), int(y))
                for x, y in getattr(self, "spoil_piles", {})
                if max(
                    abs(int(x) - getattr(self, "lz_x", self.world.center)),
                    abs(int(y) - getattr(self, "lz_y", self.world.center)),
                ) <= self.CONSTRUCTION_CAMPUS_RADIUS_CELLS + 2
            )),
        )
        if (
            proposed_structure is None and not reserve_pending
            and signature == getattr(self, "_utility_network_cache_signature", None)
        ):
            return self._utility_network_cache

        nodes = [
            structure for structure in structures
            if structure.get("type") == "life_support_distribution_grid"
        ]
        effects = (
            (self._get_recipe("life_support_distribution_grid") or {})
            .get("output", {}).get("effects", {})
        )
        service_radius = max(1, int(effects.get("service_radius_cells", 10)))
        per_node_length_m = max(
            0.0, float(effects.get("installed_line_length_m", 0.0))
        )
        maximum_length_m = per_node_length_m * len(nodes)
        endpoints_per_node = max(
            1, int(effects.get("max_connected_endpoints", 1))
        )
        maximum_endpoints = endpoints_per_node * len(nodes)
        grid_cell_m = float(self.mission_profile.grid_cell_meters)
        circuits = ["potable_water", "reclaimed_water", "oxygen"]
        node_cells = {
            (int(node.get("x", 0)), int(node.get("y", 0)))
            for node in nodes
        }
        network_cells = set(node_cells)
        blocked: set[tuple[int, int]] = set(getattr(self, "spoil_piles", {}))
        blocked.update(
            coord for coord, depth in getattr(
                self, "cell_excavation_depth", {}
            ).items() if depth > 0
        )
        for structure in structures:
            blocked.update(self._structure_footprint_cells(
                str(structure.get("type", "structure")),
                int(structure.get("x", 0)),
                int(structure.get("y", 0)),
                structure,
            ))
        blocked.difference_update(node_cells)

        consumer_priority = {
            "eclss_lander_hub": 0,
            "water_collector": 1,
            "water_purifier": 2,
            "isru_o2_unit": 3,
            "potable_water_tank": 4,
            "oxygen_buffer_tank": 5,
            "habitat_module": 6,
            "medical_station": 7,
            "greenhouse": 9,
            "hydroponics": 9,
        }
        consumers = [
            structure for structure in structures
            if structure.get("type") in consumer_priority
        ]
        consumers.sort(key=lambda structure: (
            # Utility trenches are commissioned physical assets. A later
            # process skid may use remaining line and ports, but must not make
            # an older endpoint disappear merely because its structure type
            # has a lower planning priority. Funded construction previews are
            # considered only after every commissioned endpoint.
            bool(structure.get("_utility_pending", False)),
            int(structure.get(
                "commissioned_tick",
                structure.get("completed_tick", 0),
            ) or 0),
            consumer_priority[str(structure.get("type"))],
            min((
                abs(int(structure.get("x", 0)) - nx)
                + abs(int(structure.get("y", 0)) - ny)
                for nx, ny in node_cells
            ), default=9999),
            self._life_support_structure_key(structure),
        ))

        unique_edges: set[
            tuple[tuple[int, int], tuple[int, int]]
        ] = set()
        routes = []
        endpoints = []
        physically_connected = {
            self._life_support_structure_key(node) for node in nodes
        }
        connected_endpoint_count = 0
        for structure in consumers:
            key = self._life_support_structure_key(structure)
            start = (
                int(structure.get("x", 0)),
                int(structure.get("y", 0)),
            )
            nearest_node_distance = min((
                max(abs(start[0] - nx), abs(start[1] - ny))
                for nx, ny in node_cells
            ), default=9999)
            if not nodes:
                endpoints.append({
                    "structure_id": key,
                    "structure_type": structure.get("type"),
                    "x": start[0], "y": start[1],
                    "connected": False, "physically_connected": False,
                    "reason": "no_utility_vault",
                })
                continue
            if nearest_node_distance > service_radius:
                endpoints.append({
                    "structure_id": key,
                    "structure_type": structure.get("type"),
                    "x": start[0], "y": start[1],
                    "connected": False, "physically_connected": False,
                    "reason": "outside_service_envelope",
                })
                continue
            if connected_endpoint_count >= maximum_endpoints:
                endpoints.append({
                    "structure_id": key,
                    "structure_type": structure.get("type"),
                    "x": start[0], "y": start[1],
                    "connected": False, "physically_connected": False,
                    "reason": "utility_vault_port_capacity_exhausted",
                })
                continue
            start_footprint = self._structure_footprint_cells(
                str(structure.get("type", "structure")),
                start[0], start[1], structure,
            )
            route = self._route_utility_trench(
                start,
                network_cells,
                blocked - start_footprint,
                start_footprint,
            )
            if not route:
                endpoints.append({
                    "structure_id": key,
                    "structure_type": structure.get("type"),
                    "x": start[0], "y": start[1],
                    "connected": False, "physically_connected": False,
                    "reason": "route_obstructed",
                })
                continue
            route_edges = set()
            for first, second in zip(route, route[1:]):
                route_edges.add(tuple(sorted((first, second))))
            added_edges = route_edges - unique_edges
            projected_length_m = (
                len(unique_edges) + len(added_edges)
            ) * grid_cell_m
            if projected_length_m > maximum_length_m + 1e-9:
                endpoints.append({
                    "structure_id": key,
                    "structure_type": structure.get("type"),
                    "x": start[0], "y": start[1],
                    "connected": False, "physically_connected": False,
                    "reason": "line_length_budget_exhausted",
                })
                continue
            unique_edges.update(route_edges)
            # The consumer service port is an endpoint, not a free manifold
            # for future branches; all other corridor cells may be reused.
            network_cells.update(route[1:])
            physically_connected.add(key)
            connected_endpoint_count += 1
            route_payload = [{"x": x, "y": y} for x, y in route]
            routes.append({
                "structure_id": key,
                "structure_type": structure.get("type"),
                "path": route_payload,
                "length_m": round(max(0, len(route) - 1) * grid_cell_m, 1),
                "new_installed_length_m": round(
                    len(added_edges) * grid_cell_m, 1
                ),
                "buried": True,
                "commissioned": True,
                "commissioned_tick": max(
                    int(structure.get(
                        "commissioned_tick",
                        structure.get("completed_tick", 0),
                    ) or 0),
                    min((
                        int(node.get(
                            "commissioned_tick",
                            node.get("completed_tick", 0),
                        ) or 0)
                        for node in nodes
                    ), default=0),
                ),
                "flow_enabled": powered,
            })
            endpoints.append({
                "structure_id": key,
                "structure_type": structure.get("type"),
                "x": start[0], "y": start[1],
                "connected": powered,
                "physically_connected": True,
                "reason": None if powered else "network_unpowered",
                "commissioned": True,
                "flow_enabled": powered,
            })

        connected = set(physically_connected) if powered else set()
        segments = [
            {
                "x1": first[0], "y1": first[1],
                "x2": second[0], "y2": second[1],
                "length_m": round(grid_cell_m, 1),
                "buried": True,
            }
            for first, second in sorted(unique_edges)
        ]
        snapshot = {
            "powered": powered,
            "transfer_enabled": bool(powered and nodes),
            "activation_gate": (
                "accepted_grid_and_endpoint_commissioning_plus_bus_power"
            ),
            "topology": "buried_radial_spine",
            "redundant_loop": False,
            "vehicle_crossing_policy": "buried_compacted_crossable",
            "open_trenches_block_vehicles": True,
            "circuits": circuits,
            "service_radius_cells": service_radius,
            "grid_cell_meters": grid_cell_m,
            "installed_length_m": round(len(unique_edges) * grid_cell_m, 1),
            "maximum_length_m": round(maximum_length_m, 1),
            "connected_endpoint_count": connected_endpoint_count,
            "maximum_endpoints": maximum_endpoints,
            "endpoints_per_node": endpoints_per_node,
            "nodes": [
                {
                    "id": self._life_support_structure_key(node),
                    "x": int(node.get("x", 0)),
                    "y": int(node.get("y", 0)),
                    "powered": powered,
                    "commissioned": True,
                    "commissioned_tick": int(node.get(
                        "commissioned_tick", node.get("completed_tick", 0)
                    ) or 0),
                }
                for node in nodes
            ],
            "routes": routes,
            "segments": segments,
            "endpoints": endpoints,
            "physically_connected_structure_ids": sorted(
                physically_connected
            ),
            "connected_structure_ids": sorted(connected),
        }
        if proposed_structure is None and not reserve_pending:
            self._utility_network_cache_signature = signature
            self._utility_network_cache = snapshot
        return snapshot

    def _life_support_connected(self, structure: dict) -> bool:
        """Whether an asset has a powered, finite physical utility route."""
        connected = set(
            self._life_support_network_snapshot().get(
                "connected_structure_ids", []
            )
        )
        return self._life_support_structure_key(structure) in connected

    def _connected_life_support_structures(
        self, structure_type: str, network_snapshot: dict | None = None,
    ) -> list[dict]:
        network = (
            network_snapshot
            if network_snapshot is not None
            else self._life_support_network_snapshot()
        )
        connected = set(network.get("connected_structure_ids", []))
        return [
            structure for structure in self._operational_structures(
                structure_type
            )
            if self._life_support_structure_key(structure) in connected
        ]

    def _settlement_storage_capacity(
        self, structure_type: str, effect_key: str,
        network_snapshot: dict | None = None,
    ) -> float:
        effects = (
            (self._get_recipe(structure_type) or {})
            .get("output", {}).get("effects", {})
        )
        per_tank = max(0.0, float(effects.get(effect_key, 0.0)))
        return self._capacity_with_health(
            self._connected_life_support_structures(
                structure_type, network_snapshot
            ),
            per_tank,
        )

    def _settlement_stored_inventory(
        self,
        resource_key: str,
        lander_capacity: float,
        structure_type: str,
        effect_key: str,
        network_snapshot: dict | None = None,
    ) -> float:
        """Return physically stored settlement inventory eligible for readiness.

        Life-support production currently mixes into one conserved resource
        ledger. The precursor lander's finite tankage is deliberately not a
        colony asset, so it is reserved first and only inventory above that
        volume can be credited to commissioned, networked settlement tanks.
        This conservative allocation prevents an empty tank from earning a
        thirty-day readiness score merely because its nameplate volume exists.
        """
        installed_capacity = self._settlement_storage_capacity(
            structure_type, effect_key, network_snapshot
        )
        total_inventory = max(
            0.0,
            float(getattr(self, "_colony_resources", {}).get(
                resource_key, 0.0
            )),
        )
        settlement_inventory = max(
            0.0, total_inventory - max(0.0, float(lander_capacity))
        )
        return min(installed_capacity, settlement_inventory)

    def _landing_zone_operational(self) -> bool:
        """Require a certified, correctly rated and powered arrival site."""
        if self._operational_structure_count("landing_zone") <= 0:
            return False
        if self._operational_structure_count("power_distribution_grid") <= 0:
            return False
        recipe = self._get_recipe("landing_zone") or {}
        effects = recipe.get("output", {}).get("effects", {})
        if int(effects.get("rated_arrival_population", 0)) < int(
            self.mission_profile.operational_support_population
        ):
            return False
        power_kwh = (
            float(effects.get("power_consumption_w", 500.0))
            / 1000.0 * self.SIM_HOURS_PER_TICK
        )
        return float(getattr(self, "_colony_resources", {}).get(
            "energy_stored_kwh", 0.0
        )) >= max(1e-9, power_kwh)

    def _greenhouse_maturity_ticks(self) -> int:
        """Return the first-crop service time in canonical simulation ticks."""
        recipe = self._get_recipe("greenhouse") or {}
        effects = recipe.get("output", {}).get("effects", {})
        maturity_ticks = int(effects.get("growth_cycle_ticks", 0) or 0)
        if maturity_ticks <= 0:
            maturity_ticks = int(math.ceil(
                float(effects.get("first_harvest_days", 28.0))
                * self.mission_profile.clock.ticks_per_earth_day
            ))
        return max(1, maturity_ticks)

    def _greenhouse_service_limits(self) -> tuple[int, int]:
        """Return crop-inspection interval and tolerated scheduling grace."""
        effects = (
            (self._get_recipe("greenhouse") or {})
            .get("output", {}).get("effects", {})
        )
        interval = ticks_for_minutes(
            float(effects.get("crop_service_interval_hours", 24.0)) * 60.0,
            self.SIM_MINUTES_PER_TICK,
        )
        grace = ticks_for_minutes(
            float(effects.get("crop_service_grace_hours", 12.0)) * 60.0,
            self.SIM_MINUTES_PER_TICK,
        )
        return interval, grace

    def _greenhouse_crop_service_current(self, greenhouse: dict) -> bool:
        """Whether a chamber has received its required human crop inspection."""
        interval, grace = self._greenhouse_service_limits()
        service_anchor = int(greenhouse.get(
            "last_crop_service_tick",
            greenhouse.get(
                "commissioned_tick", greenhouse.get("completed_tick", 0)
            ),
        ) or 0)
        return max(0, self.current_tick - service_anchor) <= interval + grace

    def _mature_greenhouses(self) -> list[dict]:
        """Return farms that accumulated a full powered, irrigated crop cycle."""
        maturity_ticks = self._greenhouse_maturity_ticks()
        return [
            structure for structure in self._operational_structures("greenhouse")
            if int(structure.get("crop_growth_ticks", 0)) >= maturity_ticks
        ]

    @staticmethod
    def _capacity_with_health(
        structures: list[dict], contribution_per_structure: float
    ) -> float:
        return sum(
            max(0.0, min(1.0, float(structure.get("health", 1.0))))
            * float(contribution_per_structure)
            for structure in structures
        )

    def _colony_capacity_overrides(self) -> dict:
        """Derate nominal BOM capacity by live commissioning constraints."""
        resources = getattr(self, "_colony_resources", {})
        network_snapshot = self._life_support_network_snapshot()
        power_state = getattr(self, "_power_cycle_telemetry", {})
        powered = (
            not bool(power_state.get("deficit", False))
            and float(resources.get("energy_stored_kwh", 0.0)) > 0.0
        )
        composition = self.planet.atmosphere.get("composition") or {}
        water_effects = (
            (self._get_recipe("water_collector") or {})
            .get("output", {}).get("effects", {})
        )
        atmosphere_source = bool(
            self.planet.atmosphere.get("present", False)
            and float(composition.get("H2O", 0.0))
            >= float(water_effects.get(
                "minimum_atmospheric_h2o_fraction", 0.001
            ))
        )
        ice_source = float(getattr(
            self, "central_depot_inventory", {}
        ).get("water_ice", 0.0)) > 0.0
        extraction_source = atmosphere_source or ice_source
        process_water_available = (
            float(resources.get("water_reserve_l", 0.0)) > 0.0
            or extraction_source
        )

        def contribution(recipe_name: str, category: str) -> float:
            value = (self._get_recipe(recipe_name) or {}).get(
                "colony_contribution", {}
            ).get(category, 0.0)
            return (
                float(value)
                if isinstance(value, (int, float))
                and not isinstance(value, bool) else 0.0
            )

        o2_capacity = self._capacity_with_health(
            self._connected_life_support_structures(
                "isru_o2_unit", network_snapshot
            ),
            contribution("isru_o2_unit", "oxygen_production"),
        ) if powered and process_water_available else 0.0
        o2_storage_inventory = self._settlement_stored_inventory(
            "o2_reserve_kg",
            float(getattr(self, "_lander_o2_storage_capacity_kg", 0.0)),
            "oxygen_buffer_tank",
            "oxygen_storage_capacity_kg",
            network_snapshot,
        ) if powered else 0.0
        extraction_capacity = self._capacity_with_health(
            self._connected_life_support_structures(
                "water_collector", network_snapshot
            ),
            contribution("water_collector", "water_extraction"),
        ) if powered and extraction_source else 0.0
        recycling_capacity = self._capacity_with_health(
            self._connected_life_support_structures(
                "water_purifier", network_snapshot
            ),
            contribution("water_purifier", "water_recycling"),
        ) if powered else 0.0
        water_storage_inventory = self._settlement_stored_inventory(
            "water_reserve_l",
            float(getattr(self, "_lander_water_storage_capacity_l", 0.0)),
            "potable_water_tank",
            "potable_water_storage_capacity_liters",
            network_snapshot,
        ) if powered else 0.0
        active_greenhouse_ids = set(getattr(
            self, "_active_greenhouse_ids", set()
        ))
        mature_active_greenhouses = [
            structure for structure in self._mature_greenhouses()
            if str(structure.get("id")) in active_greenhouse_ids
            and self._greenhouse_crop_service_current(structure)
        ]
        food_capacity = self._capacity_with_health(
            mature_active_greenhouses,
            contribution("greenhouse", "food_production"),
        ) if powered and process_water_available else 0.0
        shelter_capacity = self._capacity_with_health(
            self._connected_life_support_structures(
                "habitat_module", network_snapshot
            ),
            contribution("habitat_module", "shelter_capacity"),
        )
        # Every connected habitat includes its own shielded storm-safe section;
        # score only capacity that is physically commissioned and routed to the
        # life-support network, just like ordinary habitation capacity.
        hazard_capacity = self._capacity_with_health(
            self._connected_life_support_structures(
                "habitat_module", network_snapshot
            ),
            contribution("habitat_module", "hazard_protection"),
        )

        solar = self._operational_structures("solar_panel")
        grid_count = self._operational_structure_count("power_distribution_grid")
        grid_effects = (
            (self._get_recipe("power_distribution_grid") or {})
            .get("output", {}).get("effects", {})
        )
        feeder_count = max(1, int(grid_effects.get(
            "max_connected_solar_arrays", self.POWER_GRID_DEFAULT_SOLAR_FEEDERS
        )))
        connected = min(
            len(solar),
            self.LANDER_DIRECT_SOLAR_INPUTS
            if grid_count <= 0 else grid_count * feeder_count,
        )
        solar_flux = max(0.0, float(self.planet.data.get(
            "surface", {}
        ).get("solar_flux_relative_to_earth", 1.0)))
        # A recipe's 1 MW credit is already half its 2 MW peak. A stationary
        # tidally locked site can use both peak and that reserve margin;
        # rotating worlds retain the local-flux derating in firm capacity.
        firm_multiplier = min(
            1.0,
            solar_flux * (2.0 if self.planet.tidally_locked else 1.0),
        )
        energy_capacity = self._capacity_with_health(
            solar[:connected],
            contribution("solar_panel", "energy_infrastructure")
            * firm_multiplier,
        )
        return {
            "o2": {
                "oxygen_production": o2_capacity,
                "oxygen_storage": o2_storage_inventory,
            },
            "water": {
                "water_extraction": extraction_capacity,
                "water_recycling": recycling_capacity,
                "water_storage": water_storage_inventory,
            },
            "food": food_capacity,
            "shelter": shelter_capacity,
            "energy": energy_capacity,
            "hazard_protection": hazard_capacity,
        }

    def _update_colony_score(self) -> float:
        self.colony_score.update_counts(
            self.structures_built,
            communications_operational=self._communications_array_operational(),
            capacity_overrides=self._colony_capacity_overrides(),
        )
        return self.colony_score.get_overall_score()

    def _mission_state(self) -> dict:
        return self.mission_profile.mission_state(
            self.current_tick,
            self.colony_score.to_dict(),
            self.structures_built,
            communications_operational=self._communications_array_operational(),
            landing_zone_operational=self._landing_zone_operational(),
            support_soak_ticks=self._support_soak_ticks,
        )

    def _update_support_soak(self) -> None:
        """Advance one qualification tick only while every other gate holds."""
        state = self._mission_state()
        if self.strategic_policy.deadline_learning and (
            self.current_tick <= state["support_soak"]["infrastructure_deadline_tick"]
        ):
            self._strategy_deadline_readiness = max(
                self._strategy_deadline_readiness,
                min(self.colony_score.get_scores().values(), default=0.0),
            )
        non_soak_gates = {
            name: passed for name, passed in state["arrival_gates"].items()
            if name != "thirty_day_support_model"
        }
        if non_soak_gates and all(non_soak_gates.values()):
            self._support_soak_ticks += 1
        else:
            self._support_soak_ticks = 0

    def _communications_array_operational(self) -> bool:
        """A completed antenna is useful only with PMAD and live bus power."""
        if self._operational_structure_count("communications_array") <= 0:
            return False
        if self._operational_structure_count("power_distribution_grid") <= 0:
            return False
        resources = getattr(self, "_colony_resources", {})
        recipe = self._get_recipe("communications_array") or {}
        power_w = float(
            recipe.get("output", {}).get("effects", {}).get(
                "power_consumption_w", 150.0
            )
        )
        one_tick_energy_kwh = power_w / 1000.0 * self.SIM_HOURS_PER_TICK
        return float(resources.get("energy_stored_kwh", 0.0)) >= max(
            1e-9, one_tick_energy_kwh
        )

    def _communication_relay_operational(self) -> bool:
        """Local relay can use the lander bus but still needs electrical power."""
        if self._operational_structure_count("communication_relay") <= 0:
            return False
        recipe = self._get_recipe("communication_relay") or {}
        power_w = float(
            recipe.get("output", {}).get("effects", {}).get(
                "power_consumption_w", 20.0
            )
        )
        required_kwh = power_w / 1000.0 * self.SIM_HOURS_PER_TICK
        return float(getattr(self, "_colony_resources", {}).get(
            "energy_stored_kwh", 0.0
        )) >= max(1e-9, required_kwh)

    def _communication_radius_cells(self) -> int:
        """Reliable crew-to-base communications radius."""
        if self._communications_array_operational():
            return 200
        if self._communication_relay_operational():
            return 100
        return self.LOCAL_EVA_RADIUS_CELLS

    def _process_water_recovery_queue(self, *, online: bool) -> float:
        """Retain unpowered wastewater without an ever-growing batch scan.

        Once batches finish their warm-up, their individual ready times no
        longer affect processing. Combine only mature batches of the same
        source; future batches keep their original deadlines and volumes.
        """
        future = []
        mature: dict[str, list[dict]] = {}
        for batch in getattr(self, "_water_recovery_queue", []):
            if int(batch.get("ready_tick", self.current_tick + 1)) > self.current_tick:
                future.append(batch)
            else:
                mature.setdefault(batch.get("source", ""), []).append(batch)
        recovered_l = 0.0
        retained = []
        for batches in mature.values():
            liters = sum(float(batch.get("liters", 0.0)) for batch in batches)
            if online:
                recovered_l += liters
            else:
                retained.append({**batches[0], "liters": liters})
        self._water_recovery_queue = retained + future
        return recovered_l

    def _water_mass_ledger(self) -> dict[str, float]:
        """Return every tracked mission-water pool in equivalent litres."""
        depot_packs = float(getattr(
            self, "central_depot_inventory", {}
        ).get("water_packs", 0))
        crew_packs = sum(
            float(agent.inventory.items.get("water_packs", 0))
            for agent in getattr(self, "agents", [])
        )
        body_water = sum(
            max(0.0, float(getattr(
                agent, "_recoverable_body_water_l", 0.0
            )))
            for agent in getattr(self, "agents", [])
        )
        treatment = sum(
            max(0.0, float(batch.get("liters", 0.0)))
            for batch in getattr(self, "_water_recovery_queue", [])
        )
        potable = max(0.0, float(getattr(
            self, "_colony_resources", {}
        ).get("water_reserve_l", 0.0)))
        packaged = depot_packs + crew_packs
        return {
            "potable_tank_l": round(potable, 4),
            "sealed_packs_l": round(packaged, 4),
            "crew_body_pool_l": round(body_water, 4),
            "treatment_queue_l": round(treatment, 4),
            "tracked_total_l": round(
                potable + packaged + body_water + treatment, 4
            ),
        }

    @staticmethod
    def _patient_can_take_oral_fluids(patient: Agent) -> bool:
        """Whether a patient is conscious and explicitly safe to swallow."""
        status = getattr(patient.status, "value", str(patient.status))
        if status in {"dead", "incapacitated"}:
            return False
        target = (
            patient.action.target
            if isinstance(patient.action.target, dict) else {}
        )
        if target.get("conscious") is False:
            return False
        if target.get("oral_fluids_allowed") is False:
            return False
        return bool(
            patient.needs.o2_supply > 0.0
            and patient.needs.temperature_stress > 0.0
            and patient.injury_level < 0.80
        )

    def _apply_hydration_dose(self, patient: Agent, dose_l: float) -> None:
        """Transfer an already-accounted liquid dose into the body-water pool."""
        dose_l = max(0.0, float(dose_l))
        if dose_l <= 0.0:
            return
        patient.needs.thirst = min(
            100.0, patient.needs.thirst + dose_l * 35.0
        )
        patient.needs._thirst_death_timer = 0
        patient._recoverable_body_water_l = max(
            0.0,
            float(getattr(patient, "_recoverable_body_water_l", 0.0)),
        ) + dose_l
        patient.total_water_consumed_l += dose_l
        patient._last_drink_tick = self.current_tick

    def _consume_nutrition_source(
        self, patient: Agent, *, maximum_kcal: float = 700.0
    ) -> tuple[float, str | None]:
        """Consume co-located food for a meal or measured enteral feed."""
        maximum_kcal = max(0.0, float(maximum_kcal))
        if maximum_kcal <= 0.0 or not getattr(patient, "_in_habitat", False):
            return 0.0, None
        ration_kcal = float(
            Agent.FOOD_TYPES["emergency_rations"]["kcal"]
        )
        if patient.inventory.has_item("emergency_rations"):
            patient.inventory.remove_item("emergency_rations", 1)
            return ration_kcal, "patient_emergency_ration"
        if patient.inventory.has_item("ration_pack"):
            patient.inventory.remove_item("ration_pack", 1)
            return ration_kcal, "patient_ration_pack"
        depot = getattr(self, "central_depot_inventory", {})
        if depot.get("ration_packs", 0) > 0:
            depot["ration_packs"] -= 1
            return ration_kcal, "depot_ration_pack"
        food = max(0.0, float(
            self._colony_resources.get("food_reserve_kcal", 0.0)
        ))
        dose_kcal = min(maximum_kcal, food)
        if dose_kcal <= 0.0:
            return 0.0, None
        self._colony_resources["food_reserve_kcal"] = food - dose_kcal
        return dose_kcal, "habitat_food_reserve"

    def _apply_nutrition_dose(
        self, patient: Agent, dose_kcal: float
    ) -> None:
        """Apply already-accounted nutrition to physiology and food ledger."""
        dose_kcal = max(0.0, float(dose_kcal))
        if dose_kcal <= 0.0:
            return
        patient.needs.hunger = min(
            100.0,
            patient.needs.hunger
            + patient.hunger_points_for_kcal(dose_kcal),
        )
        patient.needs._hunger_death_timer = 0
        patient.total_kcal_consumed += int(round(dose_kcal))
        patient._last_meal_tick = self.current_tick
        self._crew_food_consumed_kcal_this_tick = (
            getattr(self, "_crew_food_consumed_kcal_this_tick", 0.0)
            + dose_kcal
        )

    def _consume_oral_hydration_source(
        self, patient: Agent, *, maximum_l: float = 1.0
    ) -> tuple[float, str | None]:
        """Consume one accessible, potable oral-water source without creating mass."""
        maximum_l = max(0.0, float(maximum_l))
        if maximum_l <= 0.0 or not getattr(patient, "_in_habitat", False):
            return 0.0, None
        if patient.inventory.has_item("water_packs"):
            patient.inventory.remove_item("water_packs", 1)
            # Water packs are indivisible one-litre sealed provisions. Once
            # opened, the entire litre enters the recoverable body-water pool.
            return 1.0, "patient_sealed_water_pack"
        depot = getattr(self, "central_depot_inventory", {})
        if depot.get("water_packs", 0) > 0:
            depot["water_packs"] -= 1
            return 1.0, "depot_sealed_water_pack"
        reserve_l = max(0.0, float(
            self._colony_resources.get("water_reserve_l", 0.0)
        ))
        dose_l = min(maximum_l, reserve_l)
        if dose_l <= 0.0:
            return 0.0, None
        self._colony_resources["water_reserve_l"] = reserve_l - dose_l
        return dose_l, "habitat_potable_water"

    def _medical_item_stocks(self, responder: Agent, patient: Agent, item: str) -> list[dict]:
        """Accessible medical stock in the same lander, not remote inventories.

        A ten-minute treatment step can collect a kit from the nearby locker
        or another person in the cabin. No stock is duplicated or manufactured.
        """
        if not responder._in_habitat or not patient._in_habitat:
            return []
        stocks = []
        for holder in self.agents:
            same_cabin = (
                holder._in_habitat
                and self._is_lander_footprint_cell(responder.x, responder.y)
                and self._is_lander_footprint_cell(holder.x, holder.y)
            )
            if ((holder.id in {responder.id, patient.id} or same_cabin)
                and max(abs(holder.x-responder.x), abs(holder.y-responder.y)) <= 2
                and holder.inventory.items.get(item, 0) > 0):
                stocks.append(holder.inventory.items)
        if (self._is_lander_footprint_cell(responder.x, responder.y)
            and self._is_lander_footprint_cell(patient.x, patient.y)
            and self.central_depot_inventory.get(item, 0) > 0):
            stocks.append(self.central_depot_inventory)
        return stocks

    def _oral_hydration_during_medical_rest(
        self, patient: Agent
    ) -> dict | None:
        """Let a conscious convalescent drink without ending monitored rest."""
        if (
            getattr(patient.action, "action_type", "") != "medical_rest"
            or patient.action.ticks_remaining <= 0
            or not getattr(patient, "_in_habitat", False)
            or patient.needs.thirst > 45.0
            or not self._patient_can_take_oral_fluids(patient)
        ):
            return None
        dose_l, source = self._consume_oral_hydration_source(patient)
        if dose_l <= 0.0 or source is None:
            return None
        self._apply_hydration_dose(patient, dose_l)
        target = (
            dict(patient.action.target)
            if isinstance(patient.action.target, dict) else {}
        )
        target.update({
            "conscious": True,
            "oral_fluids_allowed": True,
            "last_hydration_route": "oral",
            "last_hydration_source": source,
            "last_hydration_dose_l": dose_l,
            "last_hydration_tick": self.current_tick,
        })
        patient.action.target = target
        record = {
            "route": "oral",
            "source": source,
            "dose_l": dose_l,
            "patient_id": patient.id,
        }
        if self._on_event:
            self._on_event({
                "type": "medical_hydration",
                "agent": patient.name,
                **record,
                "tick": self.current_tick,
            })
        return record

    def _oral_nutrition_during_medical_rest(
        self, patient: Agent
    ) -> dict | None:
        """Feed a conscious convalescent without cancelling monitored rest."""
        if (
            getattr(patient.action, "action_type", "") != "medical_rest"
            or patient.action.ticks_remaining <= 0
            or not getattr(patient, "_in_habitat", False)
            or patient.needs.hunger > 35.0
            or not self._patient_can_take_oral_fluids(patient)
        ):
            return None
        dose_kcal, source = self._consume_nutrition_source(patient)
        if dose_kcal <= 0.0 or source is None:
            return None
        self._apply_nutrition_dose(patient, dose_kcal)
        target = (
            dict(patient.action.target)
            if isinstance(patient.action.target, dict) else {}
        )
        target.update({
            "conscious": True,
            "last_nutrition_route": "oral",
            "last_nutrition_source": source,
            "last_nutrition_dose_kcal": dose_kcal,
            "last_nutrition_tick": self.current_tick,
        })
        patient.action.target = target
        record = {
            "route": "oral",
            "source": source,
            "dose_kcal": dose_kcal,
            "patient_id": patient.id,
        }
        if self._on_event:
            self._on_event({
                "type": "medical_nutrition",
                "agent": patient.name,
                **record,
                "tick": self.current_tick,
            })
        return record

    def _infrastructure_expedition_radius_cells(self) -> int:
        """Area that is both surveyed and covered by communications."""
        return min(self._survey_radius_cells(), self._communication_radius_cells())

    def _agent_eva_radius_cells(self, agent: Agent) -> int:
        expedition = getattr(agent, "_active_expedition", None)
        if isinstance(expedition, dict) and expedition.get("status") in {"outbound", "working", "returning"}:
            return max(
                self.LOCAL_EVA_RADIUS_CELLS,
                int(expedition.get("authorized_radius", self.LOCAL_EVA_RADIUS_CELLS)),
            )
        return self.LOCAL_EVA_RADIUS_CELLS

    def _expedition_available_ticks(self, agent: Agent) -> int:
        """Conservative EVA time remaining across O2, scrubber and battery."""
        walking_o2_per_tick = agent.plss_o2_percent_per_tick(1.5)
        current_o2 = max(0.0, float(getattr(agent, "_current_canister_remaining", 0.0)))
        spare_o2 = max(0, int(agent.inventory.items.get("oxygen_canisters", 0))) * 100.0
        o2_ticks = int((current_o2 + spare_o2) / walking_o2_per_tick)

        scrubber_pct = max(0.0, float(getattr(agent, "plss_co2_scrubber_pct", 0.0)) - 18.0)
        tick_scale = self.SIM_MINUTES_PER_TICK / 5.0
        scrubber_ticks = int(scrubber_pct / (0.35 * 1.5 * tick_scale))
        battery_pct = max(0.0, float(getattr(agent, "plss_suit_battery_pct", 0.0)) - 18.0)
        battery_ticks = int(battery_pct / (0.40 * tick_scale))
        continuous_ticks = max(
            0, self.MAX_CONTINUOUS_EVA_TICKS - int(getattr(agent, "_eva_ticks_continuous", 0))
        )
        return max(0, min(o2_ticks, scrubber_ticks, battery_ticks, continuous_ticks))

    def _expedition_required_ticks(self, target_distance: int, current_distance: int = 0) -> int:
        """Travel from current position to target, then home, plus field work."""
        outbound_cells = max(0, int(target_distance) - max(0, int(current_distance)))
        travel_cells = outbound_cells + max(0, int(target_distance))
        travel_ticks = math.ceil(travel_cells / self.EXPEDITION_MOVE_SPEED_CELLS)
        return travel_ticks + self.EXPEDITION_WORK_RESERVE_TICKS

    def _agent_ready_for_expedition(
        self,
        agent: Agent,
        target_distance: int,
        require_spare_canister: bool = True,
    ) -> bool:
        if getattr(agent.status, "value", str(agent.status)) != "alive":
            return False
        if self._distance_from_lz(agent.x, agent.y) > 2:
            return False
        if isinstance(getattr(agent, "_active_expedition", None), dict):
            return False
        # A remote EVA uses an explicit medical go/no-go rule. The learner
        # still decides when to eat, drink and rest, and loses productive time
        # when it presents an unfit crew; it cannot turn a preventable
        # dehydration mistake into an implausible mission-authorized launch.
        if (
            agent.needs.energy < 70.0
            or agent.needs.hunger < 55.0
            or agent.needs.thirst < 60.0
            or agent.needs.o2_supply < 75.0
            or getattr(agent, "plss_co2_scrubber_pct", 0.0) < 80.0
            or getattr(agent, "plss_suit_battery_pct", 0.0) < 80.0
            or getattr(agent, "suit_integrity", 0.0) < 0.70
            or (
                require_spare_canister
                and agent.inventory.items.get("oxygen_canisters", 0) < 1
            )
        ):
            return False
        required = self._expedition_required_ticks(target_distance)
        return (
            self._expedition_available_ticks(agent)
            >= required + self.EXPEDITION_LAUNCH_MARGIN_TICKS
        )

    def _find_expedition_buddy(
        self,
        agent: Agent,
        target_distance: int,
        require_spare_canister: bool = True,
    ) -> Agent | None:
        def available_for_assignment(other: Agent) -> bool:
            action_type = getattr(other.action, "action_type", "idle")
            action_target = (
                other.action.target
                if isinstance(other.action.target, dict) else {}
            )
            if action_target.get("completion_pending"):
                return False
            return not (
                int(getattr(other.action, "ticks_remaining", 0)) > 0
                and action_type not in {
                    "idle", "rest", "enter_habitat", "stand_watch"
                }
            )

        candidates = [
            other for other in self.agents
            if other.id != agent.id
            and available_for_assignment(other)
            and self._agent_ready_for_expedition(
                other,
                target_distance,
                require_spare_canister=require_spare_canister,
            )
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda other: (
                1 if (
                    isinstance(
                        getattr(other, "_expedition_buddy_reservation", None),
                        dict,
                    )
                    and other._expedition_buddy_reservation.get("lead_id")
                    == agent.id
                ) else 0,
                getattr(other.competency, "physics", 0)
                + getattr(other.competency, "medical", 0)
                + getattr(other.genome, "endurance", 0)
            ),
        )

    def _ensure_recovery_buddy_reservation(
        self,
        lead: Agent,
        resource: str,
        target: tuple[int, int],
    ) -> Agent | None:
        """Reserve one base crew member across scheduler turn boundaries.

        A crew member may finish sleep, food, suit service or another short
        action before joining. Without this handshake they immediately took a
        new ordinary job on their own turn, so the detector lead could wait
        thousands of ticks beside an idle rover without ever observing an
        idle buddy.
        """
        contract = getattr(lead, "_detected_resource_recovery", None)
        if not isinstance(contract, dict):
            return None

        existing_id = contract.get("buddy_id")
        if existing_id:
            existing = next(
                (crew for crew in self.agents if crew.id == existing_id),
                None,
            )
            reservation = (
                getattr(existing, "_expedition_buddy_reservation", None)
                if existing is not None else None
            )
            if (
                existing is not None
                and isinstance(reservation, dict)
                and reservation.get("lead_id") == lead.id
                and int(reservation.get("expires_tick", -1))
                >= self.current_tick
                and getattr(existing.status, "value", str(existing.status))
                in {"alive", "critical"}
            ):
                return existing
            contract.pop("buddy_id", None)

        candidates = []
        for crew in self.agents:
            if crew.id == lead.id:
                continue
            if getattr(crew.status, "value", str(crew.status)) not in {
                "alive", "critical"
            }:
                continue
            if self._distance_from_lz(crew.x, crew.y) > 2:
                continue
            if isinstance(getattr(crew, "_active_expedition", None), dict):
                continue
            action_target = (
                crew.action.target
                if isinstance(crew.action.target, dict) else {}
            )
            # Let an already-materialed machine cycle finish; its output must
            # not disappear merely to create a buddy scheduling window.
            if action_target.get("completion_pending"):
                continue
            if getattr(crew.action, "action_type", "") in {
                "rescue", "treat", "medical_rest", "unconscious", "dead"
            }:
                continue
            if getattr(crew.action, "action_type", "") in {
                "build", "refine", "gather", "prospect",
                "survey_resources", "repair",
            }:
                continue
            if int(getattr(crew.action, "ticks_remaining", 0)) > 24:
                continue
            candidates.append(crew)
        if not candidates:
            return None

        buddy = min(
            candidates,
            key=lambda crew: (
                int(getattr(crew.action, "ticks_remaining", 0)),
                -(
                    getattr(crew.competency, "physics", 0)
                    + getattr(crew.competency, "medical", 0)
                    + getattr(crew.genome, "endurance", 0)
                ),
                crew.id,
            ),
        )
        reservation = {
            "lead_id": lead.id,
            "resource": resource,
            "target_x": int(target[0]),
            "target_y": int(target[1]),
            # A rover crew is a short mobilization handshake, not permission
            # to remove a builder from the critical path for a whole day.
            "expires_tick": int(self.current_tick + 36),
        }
        buddy._expedition_buddy_reservation = reservation
        contract["buddy_id"] = buddy.id
        return buddy

    def _clear_expedition_buddy_reservations(self, lead_id: str) -> None:
        for crew in self.agents:
            reservation = getattr(
                crew, "_expedition_buddy_reservation", None
            )
            if (
                isinstance(reservation, dict)
                and reservation.get("lead_id") == lead_id
            ):
                crew._expedition_buddy_reservation = None

    def _provision_expedition_spare(self, agent: Agent) -> bool:
        """Fill one reusable cylinder at the physical base O2 manifold."""
        if agent.inventory.items.get("oxygen_canisters", 0) >= 1:
            return True
        if self._distance_from_lz(agent.x, agent.y) > 1 and not agent._in_habitat:
            return False
        station = self._nearest_o2_filling_station(agent)
        if station is None:
            return False
        depot = self.central_depot_inventory
        if (
            agent.inventory.items.get("empty_oxygen_canisters", 0) < 1
            and depot.get("empty_oxygen_canisters", 0) > 0
        ):
            depot["empty_oxygen_canisters"] -= 1
            agent.inventory.add_item("empty_oxygen_canisters", 1)
        result = self._fill_agent_o2_at_isru(
            agent, station, internal_manifold=bool(
                agent._in_habitat and station.get("type") in {"eclss_lander_hub", "advanced_eclss"}
            )
        )
        if result.get("spares_filled", 0) < 1:
            return False
        agent.action.action_type = "refill_o2"
        agent.action.target = {
            "x": station.get("x"),
            "y": station.get("y"),
            "destination": "o2_filling_station",
            "expedition_provisioning": True,
            **result,
        }
        # Cylinder connection, pressure equalisation and leak check remain a
        # visible 15-minute preparation step.
        agent.action.ticks_remaining = ticks_for_minutes(
            15.0, self.SIM_MINUTES_PER_TICK
        )
        return True

    def _can_launch_expedition(
        self,
        agent: Agent,
        target_distance: int,
        target: tuple[int, int] | None = None,
        resource: str | None = None,
    ) -> bool:
        """Allow a deliberate remote trip only from base with round-trip margin."""
        if target_distance <= self.LOCAL_EVA_RADIUS_CELLS:
            return True
        scanner_confirmed = bool(
            target is not None
            and resource
            and target in self.discovered_resources.get(resource, set())
        )
        if target_distance > self._survey_radius_cells() and not scanner_confirmed:
            return False
        if not self._agent_ready_for_expedition(agent, target_distance):
            return False
        if target_distance <= self._communication_radius_cells():
            return True
        # Before a relay is available, a six-person team may conduct the
        # first regional survey only as a two-person buddy expedition.
        return self._find_expedition_buddy(agent, target_distance) is not None

    def _find_resource_from_lz(
        self, resource: str, min_radius: int, max_radius: int
    ) -> tuple[int, int] | None:
        """Return the nearest *physically confirmed* deposit in a radius band.

        Research infrastructure expands where a crew may travel; it does not
        grant omniscient access to procedural biome tables.  Unknown cells are
        measured by the portable-scanner buddy traverse and enter
        ``discovered_resources`` only after that real field action.
        """
        candidates = []
        for x, y in self.discovered_resources.get(resource, set()):
            distance = self._distance_from_lz(x, y)
            if not (max(1, int(min_radius)) <= distance <= int(max_radius)):
                continue
            if (x, y, resource) in self.depleted_cell_resources:
                continue
            candidates.append((distance, int(x), int(y)))
        if not candidates:
            return None
        _, x, y = min(candidates)
        return x, y

    def _start_construction_rover_trip(self, agent: Agent, site: dict) -> bool:
        """Mobilize the assigned two-person site crew, never a solo long walk."""
        airlock = self._lander_airlock_position()
        exterior = self._lander_airlock_exterior_position()
        target = (int(site["x"]), int(site["y"]))
        surface_route = self._surface_vehicle_route(*exterior, *target)
        route = [airlock] + surface_route
        distance_cells = (
            len(route) - 1 if surface_route else
            abs(target[0] - airlock[0]) + abs(target[1] - airlock[1])
        )
        if distance_cells < self.CREW_ROVER_MIN_DISTANCE_CELLS:
            return False

        def wait_for_crew(reason: str) -> None:
            agent.action.action_type = "rest"
            agent.action.target = {
                "construction_rover_preparation": True,
                "struct_id": site["id"], "recipe": site["type"],
                "reason": reason,
            }
            agent.action.ticks_remaining = 1

        if not surface_route:
            wait_for_crew("no_safe_construction_rover_route")
            return True
        def rendezvous(crew: Agent) -> None:
            crew.action.action_type = "move"
            crew.action.target = {
                "x": airlock[0], "y": airlock[1],
                "destination": "shelter",
                "construction_rover_rendezvous": site["id"],
            }
            crew.action.ticks_remaining = 1

        order = getattr(self.decision_engine, "shared_work_order", {})
        assigned_ids = set(order.get("construction_crew_ids", []))
        assigned_ids.update(
            crew_id for crew_id, role in order.get("assignments", {}).items()
            if role == "construction"
        )
        if site.get("planned") or not (assigned_ids - {agent.id}):
            assigned_ids.update(
                crew.id for crew in self.agents
                if self._distance_from_lz(crew.x, crew.y) <= 2
                and getattr(crew.action, "action_type", "idle") in {
                    "idle", "rest", "arrived", "plan_construction", "stand_watch"
                }
            )
        reservation = getattr(self, "_construction_sortie", None)
        if reservation and (
            reservation["site"]["id"] != site["id"]
            or any(isinstance(getattr(c, "_active_expedition", None), dict)
                   for c in self.agents if c.id in reservation["crew_ids"])
            or any(not any(c.id == cid and str(getattr(c.status, "value", c.status))
                          not in {"dead", "incapacitated"} for c in self.agents)
                   for cid in reservation["crew_ids"])
        ):
            self._release_construction_preparation()
            reservation = None
        candidates = [
            crew for crew in self.agents
            if crew.id != agent.id and crew.id in assigned_ids
            and getattr(crew.status, "value", str(crew.status)) == "alive"
            and not isinstance(getattr(crew, "_active_expedition", None), dict)
            and getattr(crew.action, "action_type", "") not in {
                "medical_rest", "refine", "craft", "operate_machine",
                "rescue", "treat",
            }
            and not (
                isinstance(crew.action.target, dict)
                and crew.action.target.get("completion_pending")
            )
        ]
        if reservation:
            if agent.id not in reservation["crew_ids"]:
                wait_for_crew("reserved_site_crew_preparing")
                return True
            candidates = [c for c in self.agents if c.id in reservation["crew_ids"]
                          and c.id != agent.id]
        buddy = min(candidates, key=lambda crew: (
            abs(crew.x - airlock[0]) + abs(crew.y - airlock[1]), crew.id
        ), default=None)
        if buddy is None:
            wait_for_crew("awaiting_assigned_rover_buddy")
            return True
        if not reservation:
            recipe = self._get_recipe(site["type"]) or {}
            qualification_recipe = {**recipe, "materials": {}}
            pair = [agent, buddy]
            qualified_leads = [
                member for member in pair
                if member.can_craft(qualification_recipe).get("can_craft", False)
            ]
            lead = max(
                qualified_leads,
                key=lambda member: (
                    int(getattr(member.competency, "engineering", 0)),
                    int(getattr(member.genome, "strength", 0)),
                    member.id,
                ),
                default=agent,
            )
            reservation = {
                "site": dict(site),
                "crew_ids": [agent.id, buddy.id],
                "lead_id": lead.id,
                "created_tick": self.current_tick,
            }
            self._construction_sortie = reservation
            for member in (agent, buddy):
                member._construction_preparation = reservation
                # An old planning-room visit may not replace this mission.
                pending = getattr(member, "_pending_indoor_activity", None)
                if pending and pending.get("action") == "plan_construction":
                    member._pending_indoor_activity = None
        reserved_rover = next((r for r in self.surface_fleet.crew_rovers
                               if r.reservation_id == site["id"]), None)
        if reserved_rover is None:
            reserved_rover = next((r for r in self.surface_fleet.crew_rovers
                                   if r.state in {"idle", "charging"} and not r.reservation_id), None)
            if reserved_rover is not None:
                reserved_rover.reservation_id = site["id"]
        if getattr(agent, "_pending_indoor_activity", None):
            return True
        if (agent.x, agent.y) != airlock:
            rendezvous(agent)
            return True
        if (getattr(buddy, "_pending_indoor_activity", None)
            or buddy.action.action_type in {"sleep", "eat", "drink", "refill_o2", "service_suit"}):
            wait_for_crew("buddy_completing_preflight")
            return True
        if (buddy.x, buddy.y) != airlock:
            rendezvous(buddy)
            wait_for_crew("awaiting_buddy_at_airlock")
            return True
        # Charging can be interrupted when the existing route-energy audit
        # covers the round trip plus emergency reserve; full charge is not
        # a physical departure prerequisite.
        if not any(rover.state in {"idle", "charging"} and not rover.mission
                   and rover.reservation_id in {None, site["id"]}
                   for rover in self.surface_fleet.crew_rovers):
            wait_for_crew("crew_rover_unavailable")
            return True

        for member in (agent, buddy):
            if member.inventory.items.get("oxygen_canisters", 0) < 1:
                depot = self.central_depot_inventory
                if depot.get("oxygen_canisters", 0) > self.MIN_CENTRAL_MAINTENANCE_O2_CANISTERS:
                    depot["oxygen_canisters"] -= 1
                    member.inventory.add_item("oxygen_canisters", 1)
                elif self._provision_expedition_spare(member):
                    if member is not agent:
                        wait_for_crew("buddy_o2_service")
                    return True
            if not self._prepare_agent_for_eva(
                member, defer_pressure_transition=True, target_position=target,
                rover_outbound=True,
            ):
                if member is not agent:
                    wait_for_crew("buddy_eva_preflight")
                return True
            if not self._agent_ready_for_expedition(member, distance_cells):
                wait_for_crew("construction_crew_not_ready")
                return True

        speed = max(1, int(
            float(self.surface_fleet.crew_rover_spec.get("max_speed_kph", 10.0))
            * self.SIM_HOURS_PER_TICK * 1000.0
            / self.mission_profile.grid_cell_meters
        ))
        expedition_id = f"construction-{self.current_tick}-{agent.id[:8]}"
        dispatch = self.surface_fleet.begin_crew_rover_trip(
            expedition_id=expedition_id, crew_ids=[agent.id, buddy.id],
            target_x=target[0], target_y=target[1], route=route,
            reservation_id=site["id"],
        )
        if not dispatch.get("reserved"):
            wait_for_crew(str(dispatch.get("reason", "crew_rover_unavailable")))
            return True
        state = {
            "id": expedition_id, "kind": "construction_support",
            "resource": "construction", "site_id": site["id"],
            "planned_site": bool(site.get("planned")),
            "construction_crew_limit": max(1, int(
                (self._get_recipe(site["type"]) or {}).get("construction", {}).get("recommended_crew", 2)
            )),
            "recipe": site["type"], "target_x": target[0], "target_y": target[1],
            "authorized_radius": self._distance_from_lz(*target),
            "launched_tick": self.current_tick, "status": "outbound",
            "lead_id": agent.id, "lead_name": agent.name,
            "buddy_id": buddy.id, "buddy_name": buddy.name,
            "transport": "crew_rover", "rover_id": dispatch["vehicle_id"],
            "move_speed_cells": speed, "route": route, "route_index": 0,
            "required_ticks_at_launch": math.ceil(2 * distance_cells / speed)
            + self.EXPEDITION_WORK_RESERVE_TICKS,
        }
        for member, role in ((agent, "lead"), (buddy, "buddy")):
            member._construction_preparation = None
            member._active_expedition = {
                **state, "role": role,
                "available_ticks_at_launch": self._expedition_available_ticks(member),
            }
            member.action.action_type = "move"
            member.action.target = {
                "x": target[0], "y": target[1], "expedition": True,
                "destination": "construction_site", "struct_id": site["id"],
                "recipe": site["type"], "construction_route": True,
            }
            member.action.ticks_remaining = 1
        self._construction_sortie = None
        return True

    def _release_construction_preparation(self) -> None:
        reservation = getattr(self, "_construction_sortie", None)
        if reservation:
            for rover in self.surface_fleet.crew_rovers:
                if rover.reservation_id == reservation["site"]["id"]:
                    rover.reservation_id = None
        self._construction_sortie = None
        for crew in self.agents:
            crew._construction_preparation = None

    def _construction_preparation_decision(self, agent: Agent) -> dict | None:
        """Keep the assigned pair on one recoverable, mode-aware preflight.

        This is execution of an already selected job, not a learned build order.
        Medical alarms and active hazards retain priority in the caller.
        """
        reservation = getattr(agent, "_construction_preparation", None)
        if not reservation or isinstance(getattr(agent, "_active_expedition", None), dict):
            return None
        site = reservation["site"]
        live = next((s for s in self.placed_structures if s["id"] == site["id"]), None)
        if (not site.get("planned") and (not live or not live.get("under_construction")
                                        or live.get("destroyed"))):
            self._release_construction_preparation()
            return None
        if any(str(getattr(c.status, "value", c.status)) in {"dead", "incapacitated"}
               for c in self.agents if c.id in reservation["crew_ids"]):
            self._release_construction_preparation()
            return None
        target = {"construction_preparation": site["id"], "habitat": True}
        ready = reservation.setdefault("energy_ready", {})
        if agent.needs.energy >= 85.0:
            ready[agent.id] = True
        elif agent.needs.energy < self.MIN_ROUTINE_EVA_START_ENERGY_PCT:
            ready[agent.id] = False
        action = "build"
        if not agent._in_habitat or not self._is_lander_footprint_cell(agent.x, agent.y):
            # Preparing the *next* sortie must never send a tired returnee
            # from the airlock back to an external compressor for a spare.
            # A pressurized greenhouse is not the lander's service rack.
            agent._construction_o2_service = False
            action = "move"
            target.update(zip(("x", "y"), self._lander_airlock_position()))
            target["destination"] = "shelter"
        elif getattr(agent, "_pending_indoor_activity", None):
            pending = agent._pending_indoor_activity
            action, target = pending["action"], dict(pending["target"])
        elif agent.needs.hunger < 75.0:
            action = "eat"
        elif agent.needs.thirst < 85.0:
            action = "drink"
        elif not ready.get(agent.id, False):
            action = "sleep"
            target["ticks"] = ticks_for_minutes(120.0, self.SIM_MINUTES_PER_TICK)
        elif agent.suit_condition < self.MIN_ROUTINE_EVA_EXIT_SUIT_CONDITION or agent.suit_integrity < 0.70:
            action = "service_suit"
            target["pressure_overhaul"] = agent.suit_integrity < 0.70
        elif (agent._current_canister_remaining < 75.0
              or not agent.inventory.has_item("oxygen_canisters")):
            action = "refill_o2"
            agent._construction_o2_service = True
        elif agent.plss_co2_scrubber_pct < 80.0 or agent.plss_suit_battery_pct < 80.0:
            action = "stand_watch"
            target["reason"] = "plss_rack_recharging"
        elif (agent.x, agent.y) != self._lander_airlock_position():
            # Execute the route, do not reissue BUILD every tick: BUILD only
            # queues rendezvous and never advances an interior motor step.
            action = "move"
            target = {"x": self._lander_airlock_position()[0],
                      "y": self._lander_airlock_position()[1],
                      "destination": "shelter",
                      "construction_rover_rendezvous": site["id"]}
        else:
            agent._construction_o2_service = False
            lead_id = reservation.get("lead_id")
            if not lead_id:
                recipe = self._get_recipe(site["type"]) or {}
                qualification_recipe = {**recipe, "materials": {}}
                members = [
                    crew for crew in self.agents
                    if crew.id in reservation["crew_ids"]
                ]
                qualified = [
                    crew for crew in members
                    if crew.can_craft(qualification_recipe).get(
                        "can_craft", False
                    )
                ]
                lead = max(
                    qualified,
                    key=lambda crew: (
                        int(getattr(crew.competency, "engineering", 0)),
                        int(getattr(crew.genome, "strength", 0)),
                        crew.id,
                    ),
                    default=members[0] if members else agent,
                )
                lead_id = lead.id
                reservation["lead_id"] = lead_id
            if agent.id != lead_id:
                # The second seat supplies buddy safety and construction labor,
                # but only the qualified lead may open a planned site. Letting
                # the helper execute BUILD produced a permanent craft_blocked
                # loop between otherwise-ready airlock partners.
                action = "stand_watch"
                target = {
                    "construction_preparation": site["id"],
                    "habitat": True,
                    "reason": "awaiting_construction_lead",
                    "construction_lead_id": lead_id,
                }
            else:
                target = {"recipe": site["type"], "struct_id": site["id"],
                          "x": site["x"], "y": site["y"],
                          "construction_preparation": site["id"]}
        agent._rl_transition_pending = False
        return {"action": action, "target": target, "deterministic": True,
                "reasoning": "Completing the reserved two-person construction sortie preflight"}

    def _rover_departure_rendezvous(self, team: list[Agent], target: dict) -> bool:
        """Finish individual recovery, then walk both occupants to the suitlock."""
        recovering = [member for member in team if (
            getattr(member, "_pending_indoor_activity", None)
            or (member.action.action_type in {
                "sleep", "eat", "drink", "refill_o2", "service_suit", "medical_rest"
            } and member.action.ticks_remaining > 0)
        )]
        if recovering:
            for member in team:
                if member not in recovering:
                    member.action.action_type = "stand_watch"
                    member.action.target = {
                        **target, "expedition": True,
                        "rover_rendezvous": True,
                        "reason": "buddy_completing_preflight",
                    }
                    member.action.ticks_remaining = 1
            return False
        airlock = self._lander_airlock_position()
        if all((member.x, member.y) == airlock for member in team):
            return True
        for member in team:
            moving = (member.x, member.y) != airlock
            if (member.x, member.y) != airlock:
                dx, dy = self._cardinal_step_toward(member, *airlock, 1)
                member.x += dx
                member.y += dy
            # A stationary driver waiting for a passenger is not walking.
            # Preserve the passenger's real route and avoid charging the
            # driver's movement metabolism while standing at the suitlock.
            member.action.action_type = "move" if moving else "stand_watch"
            member.action.target = {
                **target, "expedition": True, "rover_rendezvous": True,
                "reason": "walking_to_rover" if moving else "awaiting_buddy_at_airlock",
            }
            member.action.ticks_remaining = 1
        return False

    def _recover_incomplete_rover_team(self, state: dict, team: list[Agent]) -> bool:
        """Cancel an orphaned assignment without moving an absent passenger."""
        expedition_id = state.get("id")
        matching = [crew for crew in team
                    if isinstance(getattr(crew, "_active_expedition", None), dict)
                    and crew._active_expedition.get("id") == expedition_id]
        if len(team) == 2 and len(matching) == 2 and all(
            getattr(crew.status, "value", str(crew.status)) != "dead" for crew in team
        ):
            return False
        rover = self.surface_fleet.crew_rover_for_expedition(str(expedition_id))
        all_home = all(
            getattr(crew.status, "value", str(crew.status)) == "dead"
            or (crew._in_habitat and self._is_lander_footprint_cell(crew.x, crew.y))
            for crew in team
        )
        rover_home = rover is None or self._is_lander_footprint_cell(rover.x, rover.y)
        if all_home and rover_home:
            if rover is not None:
                self._complete_crew_rover_expedition({**state, "role": "lead"})
            for crew in matching:
                crew._active_expedition = None
                pending = getattr(crew, "_pending_expedition", None)
                if isinstance(pending, dict) and pending.get("id") == expedition_id:
                    crew._pending_expedition = None
                if crew.action.action_type in {"move", "arrived"}:
                    crew.action.clear()
            return True
        # An inconsistent field party cannot be reassembled by teleporting
        # passengers. Retain the vehicle on site and use the normal foot return.
        if rover is not None:
            rover.state = "fault"
            rover.fault_reason = "crew_expedition_state_mismatch"
        for crew in matching:
            if getattr(crew.status, "value", str(crew.status)) == "dead":
                continue
            crew._active_expedition.update(
                status="returning", transport="on_foot", move_speed_cells=1,
                rover_fault=True,
            )
            if not crew._in_habitat:
                crew.action.action_type = "move"
                crew.action.target = {
                    "x": self.lz_x, "y": self.lz_y,
                    "destination": "shelter", "expedition": True,
                    "transport": "on_foot",
                }
                crew.action.ticks_remaining = 1
        return True

    def _move_resource_rover_team(self, agent: Agent, target: dict) -> bool:
        """Move a non-construction rover expedition as one physical vehicle.

        The earlier generic MOVE path advanced the two occupants separately
        and forced every homeward rover through the one-cell pedestrian
        airlock approach.  Its planner still budgeted the real rover speed,
        so a nominally safe return could exhaust PLSS oxygen kilometres from
        the hub.  This dispatcher keeps both seats co-located, follows the
        obstacle-aware surface route and charges traction energy once.
        """
        state = getattr(agent, "_active_expedition", None)
        if (
            not isinstance(state, dict)
            or state.get("kind") == "construction_support"
            or state.get("transport") != "crew_rover"
            or (agent._in_habitat and target.get("destination") == "indoor_activity")
        ):
            return False
        team = [
            crew for crew in self.agents
            if crew.id in {state.get("lead_id"), state.get("buddy_id")}
        ]
        if self._recover_incomplete_rover_team(state, team):
            return True
        lead = next(
            (crew for crew in team if crew.id == state.get("lead_id")), None
        )
        if lead is None or len(team) != 2:
            return False
        master = getattr(lead, "_active_expedition", None)
        if not isinstance(master, dict):
            return False
        rover = self.surface_fleet.crew_rover_for_expedition(
            str(master.get("id", ""))
        )
        if rover is None or rover.state != "in_use":
            for member in team:
                member_state = getattr(member, "_active_expedition", None)
                if isinstance(member_state, dict):
                    member_state["transport"] = "on_foot"
                    member_state["move_speed_cells"] = 1
                    member_state["rover_fault"] = True
            return False

        # A vehicle can carry only its actual occupants. Do not repair a
        # legacy split-party state by assigning the absent passenger's position.
        if not any(member._in_habitat for member in team) and any(
            (member.x, member.y) != (rover.x, rover.y) for member in team
        ):
            rover.state = "fault"
            rover.fault_reason = "crew_rover_passenger_separated"
            for member in team:
                member._active_expedition.update(
                    transport="on_foot", move_speed_cells=1,
                    rover_fault=True, status="returning",
                )
            return False

        # Only the driver advances the vehicle, atomically updating both
        # occupants. The passenger never performs a second coordinate assignment.
        if agent.id != lead.id:
            return True

        returning = (
            target.get("destination") in {"habitat", "shelter"}
            or any(
                isinstance(getattr(member, "_active_expedition", None), dict)
                and member._active_expedition.get("status") == "returning"
                for member in team
            )
        )
        if returning:
            for member in team:
                member._active_expedition["status"] = "returning"
        if master.get("last_movement_tick") == self.current_tick:
            return True

        exterior = self._lander_airlock_exterior_position()
        airlock = self._lander_airlock_position()
        destination = (
            exterior if returning else (
                int(master.get("target_x", target.get("x", lead.x))),
                int(master.get("target_y", target.get("y", lead.y))),
            )
        )

        if not returning and any(member._in_habitat for member in team):
            # Let each occupant finish their own berth/galley/service route.
            # Pulling the buddy back to the door here repeatedly cancelled sleep.
            if not self._rover_departure_rendezvous(team, target):
                master["last_movement_tick"] = self.current_tick
                return True
            for member in team:
                if not self._prepare_agent_for_eva(
                    member,
                    defer_pressure_transition=True,
                    target_position=destination,
                    rover_outbound=True,
                ):
                    return True
            if not self._airlock_pass(team, "out"):
                return True
            for member in team:
                member.x, member.y = exterior
                member._in_habitat = False
                member._managed_airlock_tick = self.current_tick
            rover.x, rover.y = exterior
            master["last_movement_tick"] = self.current_tick
            return True

        if returning and (lead.x, lead.y) == exterior:
            if not self._airlock_pass(team, "in"):
                return True
            self._complete_crew_rover_expedition(master)
            for member in team:
                member.x, member.y = airlock
                member.enter_habitat()
                member._managed_airlock_tick = self.current_tick
                member._active_expedition = None
                member._pending_expedition = None
                member.action.clear()
            return True

        route = self._surface_vehicle_route(
            lead.x, lead.y, destination[0], destination[1]
        )
        if (lead.x, lead.y) == destination and not returning:
            for member in team:
                member._active_expedition["status"] = "working"
            master["movement_credit"] = 0.0
            return True
        if len(route) < 2:
            # Keep the occupied rover in place; a transient obstruction must
            # not silently turn a vehicle trip into a teleport or solo walk.
            for member in team:
                member.action.action_type = "move"
                member.action.target = {
                    **target,
                    "expedition": True,
                    "transport": "crew_rover",
                    "reason": "no_safe_rover_route",
                }
                member.action.ticks_remaining = 1
            return True

        speed = max(1, int(master.get("move_speed_cells", 1)))
        credit = float(master.pop("movement_credit", 0.0)) + speed
        route_index = 0
        while route_index < len(route) - 1:
            next_cell = route[route_index + 1]
            cell = self.world.get_cell_info(
                next_cell[0], next_cell[1], self.current_tick
            )
            cost = max(1.0, float(cell.get("traversal_cost", 1.0)))
            if credit + 1e-9 < cost:
                break
            credit -= cost
            route_index += 1
        master["movement_credit"] = credit
        if route_index == 0:
            master["last_movement_tick"] = self.current_tick
            return True

        # Charge every routed edge, including detours whose length exceeds
        # the Manhattan distance between this tick's start and end positions.
        travelled_index = 0
        for edge_index in range(1, route_index + 1):
            if not self.surface_fleet.record_crew_rover_movement(
                str(master.get("id", "")),
                *route[edge_index - 1], *route[edge_index],
            ):
                for member in team:
                    member._active_expedition.update(
                        transport="on_foot", move_speed_cells=1,
                        rover_fault=True, status="returning",
                    )
                break
            travelled_index = edge_index
        new_x, new_y = route[travelled_index]

        master["last_movement_tick"] = self.current_tick
        reached = (new_x, new_y) == destination
        if reached or travelled_index < route_index:
            master["movement_credit"] = 0.0
        for member in team:
            member.x, member.y = new_x, new_y
            member._in_habitat = False
            member_state = getattr(member, "_active_expedition", None)
            if isinstance(member_state, dict):
                member_state["last_movement_tick"] = self.current_tick
                if reached and not returning:
                    member_state["status"] = "working"
            member.action.action_type = (
                "arrived" if reached and not returning else "move"
            )
            member.action.target = {
                **target,
                "x": destination[0],
                "y": destination[1],
                "destination": "shelter" if returning else target.get(
                    "destination", "expedition_site"
                ),
                "expedition": True,
                "transport": member_state.get("transport", "on_foot"),
            }
            member.action.ticks_remaining = 1
        return True

    def _move_construction_rover_team(self, agent: Agent, target: dict) -> bool:
        """Move both seats along the audited route, charging each edge once."""
        state = getattr(agent, "_active_expedition", None)
        if (
            not isinstance(state, dict)
            or state.get("kind") != "construction_support"
            or state.get("rover_fault")
            or (agent._in_habitat and target.get("destination") == "indoor_activity")
        ):
            return False
        team = [crew for crew in self.agents if crew.id in {
            state.get("lead_id"), state.get("buddy_id")
        }]
        if self._recover_incomplete_rover_team(state, team):
            return True
        lead = next(crew for crew in team if crew.id == state["lead_id"])
        master = lead._active_expedition
        returning = target.get("destination") in {"habitat", "shelter"} or any(
            crew._active_expedition.get("status") == "returning" for crew in team
        )
        if returning:
            for crew in team:
                crew._active_expedition["status"] = "returning"
        if master.get("last_movement_tick") == self.current_tick:
            return True
        master["last_movement_tick"] = self.current_tick
        route = master["route"]
        index = int(master["route_index"])
        departing_chamber = not returning and index == 0
        if departing_chamber and not self._rover_departure_rendezvous(team, target):
            return True
        if departing_chamber and not self._airlock_pass(team, "out"):
            return True
        direction = -1 if returning else 1
        end = 0 if returning else len(route) - 1
        credit = float(master.pop("movement_credit", 0.0)) + float(master["move_speed_cells"])
        while index != end:
            next_index = index + direction
            if returning and next_index == 0 and not self._airlock_pass(team, "in"):
                break
            cell = self.world.get_cell_info(*route[next_index], self.current_tick)
            cost = max(1.0, float(cell.get("traversal_cost", 1.0)))
            if credit < cost:
                break
            if not self.surface_fleet.record_crew_rover_movement(
                str(master["id"]), *route[index], *route[next_index]
            ):
                for crew in team:
                    crew._active_expedition["status"] = "returning"
                    crew._active_expedition["rover_fault"] = True
                    crew._active_expedition["transport"] = "on_foot"
                    crew._active_expedition["move_speed_cells"] = 1
                return True
            credit -= cost
            index = next_index
            if departing_chamber:
                # The completed pressure cycle ends on the exterior apron.
                # Driving starts next tick, never as an apparent jump from
                # the closed suitlock directly to a remote building site.
                credit = 0.0
                break
        master["route_index"] = index
        master["movement_credit"] = credit if index != end else 0.0
        for crew in team:
            crew.x, crew.y = route[index]
            crew._in_habitat = index == 0
            if (not returning and int(master.get("route_index", 0)) > 0) or (returning and index == 0):
                crew._managed_airlock_tick = self.current_tick
            crew._active_expedition["route_index"] = index
            if index == end and not returning:
                crew._active_expedition["status"] = "working"
                crew.action.action_type = "arrived"
            else:
                crew.action.action_type = "move"
            crew.action.target = {
                "x": route[end][0], "y": route[end][1], "expedition": True,
                "destination": "shelter" if returning else "construction_site",
                "struct_id": master["site_id"], "recipe": master["recipe"],
                "construction_route": not returning,
            }
            crew.action.ticks_remaining = 1
        if returning and index == 0:
            self._complete_crew_rover_expedition(master)
            for crew in team:
                crew.enter_habitat()
                crew._active_expedition = None
                crew._pending_expedition = None
                crew.action.clear()
        return True

    def _start_expedition(
        self,
        agent: Agent,
        resource: str,
        target: tuple[int, int],
        *,
        require_rover: bool = False,
        expedition_kind: str | None = None,
        capacity_recipe: str | None = None,
        recovery_goal_units: int | None = None,
    ) -> dict | None:
        target_distance = self._distance_from_lz(*target)
        # Every launch path, including deterministic fallbacks, crosses the
        # same physical PLSS/suit/range gate. Callers cannot accidentally
        # create an expedition by bypassing preflight validation.
        if not self._agent_ready_for_expedition(agent, target_distance):
            return None
        buddy = None
        if target_distance >= self.CREW_ROVER_MIN_DISTANCE_CELLS:
            buddy = self._find_expedition_buddy(agent, target_distance)
        if require_rover and (
            target_distance < self.CREW_ROVER_MIN_DISTANCE_CELLS
            or buddy is None
        ):
            return None
        # Check water before reserving a vehicle or publishing an expedition.
        # A rejected departure must not leave a phantom active field mission.
        for member in ([agent, buddy] if buddy is not None else [agent]):
            if not self._prepare_eva_water(member, target, rover_outbound=buddy is not None):
                return None
        authorized_radius = (
            self._survey_radius_cells() if buddy
            else self._infrastructure_expedition_radius_cells()
        )
        if target in self.discovered_resources.get(resource, set()):
            authorized_radius = max(authorized_radius, target_distance)
        expedition_id = f"exp-{self.current_tick}-{agent.id[:8]}"
        transport = "on_foot"
        rover_id = None
        move_speed_cells = self.EXPEDITION_MOVE_SPEED_CELLS
        if (
            buddy is not None
            and target_distance >= self.CREW_ROVER_MIN_DISTANCE_CELLS
        ):
            rover_dispatch = self.surface_fleet.begin_crew_rover_trip(
                expedition_id=expedition_id,
                crew_ids=[agent.id, buddy.id],
                target_x=int(target[0]),
                target_y=int(target[1]),
                route=[
                    {"x": x, "y": y}
                    for x, y in self._surface_vehicle_route(
                        self.lz_x,
                        self.lz_y,
                        int(target[0]),
                        int(target[1]),
                    )
                ],
            )
            if rover_dispatch.get("reserved"):
                transport = "crew_rover"
                rover_id = rover_dispatch.get("vehicle_id")
                speed_kph = float(
                    self.mission_profile.surface_fleet["crew_rover"].get(
                        "max_speed_kph", 10.0
                    )
                )
                move_speed_cells = max(
                    1,
                    int(
                        speed_kph
                        * self.SIM_HOURS_PER_TICK
                        * 1000.0
                        / self.mission_profile.grid_cell_meters
                    ),
                )
        if require_rover and transport != "crew_rover":
            # A confirmed distant production face is a cargo mission, not a
            # licence for a solo astronaut to walk kilometres with ore.  No
            # crew state has been changed yet, so a failed physical rover
            # reservation can be retried safely after charging/availability.
            return None
        if transport != "crew_rover":
            for member in ([agent, buddy] if buddy is not None else [agent]):
                if not self._prepare_eva_water(member, target):
                    return None
        required_ticks = (
            math.ceil(target_distance * 2 / max(1, move_speed_cells))
            + self.EXPEDITION_WORK_RESERVE_TICKS
        )
        expedition = {
            "id": expedition_id,
            "resource": resource,
            "target_x": int(target[0]),
            "target_y": int(target[1]),
            "authorized_radius": int(authorized_radius),
            "launched_tick": int(self.current_tick),
            "required_ticks_at_launch": int(required_ticks),
            "available_ticks_at_launch": int(self._expedition_available_ticks(agent)),
            "status": "outbound",
            "role": "lead",
            "lead_id": agent.id,
            "lead_name": agent.name,
            "buddy_id": buddy.id if buddy else None,
            "buddy_name": buddy.name if buddy else None,
            "transport": transport,
            "rover_id": rover_id,
            "move_speed_cells": move_speed_cells,
        }
        if expedition_kind:
            expedition["kind"] = expedition_kind
        if capacity_recipe:
            expedition["capacity_recipe"] = capacity_recipe
        if recovery_goal_units is not None:
            expedition["recovery_goal_units"] = max(
                1, int(recovery_goal_units)
            )
        agent._active_expedition = expedition
        agent._pending_expedition = None
        self._clear_expedition_buddy_reservations(agent.id)
        recovery_contract = getattr(
            agent, "_detected_resource_recovery", None
        )
        if isinstance(recovery_contract, dict):
            recovery_contract.pop("buddy_id", None)
        self.discovered_resources.setdefault(resource, set()).add(target)
        self.revealed_cell_resources.add(target)
        if buddy is not None:
            buddy._active_expedition = {
                **expedition,
                "role": "buddy",
                "available_ticks_at_launch": int(self._expedition_available_ticks(buddy)),
            }
            buddy._pending_expedition = None
            buddy.action.action_type = "move"
            buddy.action.target = {
                "x": int(target[0]), "y": int(target[1]),
                "resource": resource, "expedition": True,
                "transport": transport,
                "rover_id": rover_id,
                "move_speed_cells": move_speed_cells,
            }
            buddy.action.ticks_remaining = 1
        return expedition

    def _cardinal_step_toward(
        self, agent: Agent, target_x: int, target_y: int, step: int | None = None
    ) -> tuple[int, int]:
        """Return a one-axis grid step toward a target.

        Routine surface traverses should follow an inspectable route instead of
        teleporting one cell on both axes and drawing diagonal X-shaped tracks.
        The stable tie-break distributes the first leg across crew members.
        """
        target_x = int(target_x)
        target_y = int(target_y)
        if step is None:
            step = getattr(self, "EVA_WALK_SPEED_CELLS", 1)
        step = max(1, int(step))
        in_lander = self._is_lander_footprint_cell(agent.x, agent.y)
        target_in_lander = self._is_lander_footprint_cell(
            target_x, target_y
        )
        airlock_x, airlock_y = self._lander_airlock_position()
        if in_lander and bool(getattr(agent, "_in_habitat", False)):
            if not target_in_lander:
                # Interior travel converges on one visible suitlock. From the
                # suitlock the next motor step is the fixed exterior apron,
                # even if the ultimate job lies north/east/west of the hub.
                if (agent.x, agent.y) != (airlock_x, airlock_y):
                    target_x, target_y = airlock_x, airlock_y
                else:
                    if not self._airlock_pass([agent], "out"):
                        return 0, 0
                    target_x, target_y = (
                        self._lander_airlock_exterior_position()
                    )
        elif not bool(getattr(agent, "_in_habitat", False)) and target_in_lander:
            # No external route may enter through an arbitrary hull cell.
            exterior = self._lander_airlock_exterior_position()
            if (agent.x, agent.y) != exterior and not in_lander:
                # Route to the *outside* apron, so the pathfinder keeps the
                # hull blocked instead of opening the whole footprint as an
                # endpoint. This also prevents entry through a north wall.
                route = self._surface_vehicle_route(agent.x, agent.y, *exterior)
                if len(route) < 2:
                    return 0, 0
                # Respect the same stride as outbound travel. The old special
                # approach silently limited the entire return to one cell per
                # tick while the life-support budget assumed the normal pace.
                first = (route[1][0] - agent.x, route[1][1] - agent.y)
                count = 1
                while count < min(step, len(route) - 1):
                    if route[count + 1] != (
                        agent.x + first[0] * (count + 1),
                        agent.y + first[1] * (count + 1),
                    ):
                        break
                    count += 1
                return first[0] * count, first[1] * count
            target_x, target_y = airlock_x, airlock_y

        delta_x = target_x - agent.x
        delta_y = target_y - agent.y
        prefer_x = abs(delta_x) > abs(delta_y) or (
            abs(delta_x) == abs(delta_y)
            and sum(ord(char) for char in agent.id) % 2 == 0
        )
        if delta_x and (prefer_x or not delta_y):
            movement = (max(-step, min(step, delta_x)), 0)
        elif delta_y:
            movement = (0, max(-step, min(step, delta_y)))
        else:
            movement = (0, 0)

        if not agent._in_habitat and not in_lander and not target_in_lander and movement != (0, 0):
            unit_x = (1 if movement[0] > 0 else -1) if movement[0] else 0
            unit_y = (1 if movement[1] > 0 else -1) if movement[1] else 0
            distance = abs(movement[0]) + abs(movement[1])
            if any(self._is_lander_footprint_cell(
                agent.x + unit_x * offset, agent.y + unit_y * offset
            ) for offset in range(1, distance + 1)):
                # An outside-to-outside traverse must go around the hull.
                # Cutting through it spent the crew's reserved return cycle
                # while the outbound action kept its outside physiology flag.
                route = self._surface_vehicle_route(agent.x, agent.y, target_x, target_y)
                if len(route) < 2:
                    return 0, 0
                first = (route[1][0] - agent.x, route[1][1] - agent.y)
                count = 1
                while count < min(step, len(route) - 1):
                    next_cell = route[count + 1]
                    if next_cell != (agent.x + first[0] * (count + 1),
                                     agent.y + first[1] * (count + 1)):
                        break
                    count += 1
                return first[0] * count, first[1] * count

        if (not bool(getattr(agent, "_in_habitat", False)) and target_in_lander
            and self._is_lander_footprint_cell(agent.x + movement[0], agent.y + movement[1])):
            # Stop on the exterior apron, then finish the pressure cycle before
            # entering. A two-cell stride must not jump across the chamber.
            exterior = self._lander_airlock_exterior_position()
            if (agent.x, agent.y) != exterior:
                if agent.x == exterior[0]:
                    return 0, max(-step, min(step, exterior[1] - agent.y))
                return max(-step, min(step, exterior[0] - agent.x)), 0
            if not self._airlock_pass([agent], "in"):
                return 0, 0

        # Ten-minute ticks permit two-cell walking steps, but an indoor route
        # may not jump across the final pressurized grid.  Every caller gets a
        # visible airlock boundary stop even if that action has its own travel
        # executor (gather, prospect, service, ordinary move, and so on).
        if (
            step > 1
            and bool(getattr(agent, "_in_habitat", False))
            and movement != (0, 0)
        ):
            proposed_x = agent.x + movement[0]
            proposed_y = agent.y + movement[1]
            if not self._is_pressurized_location(proposed_x, proposed_y):
                unit = (
                    (1 if movement[0] > 0 else -1, 0)
                    if movement[0]
                    else (0, 1 if movement[1] > 0 else -1)
                )
                if self._is_pressurized_location(
                    agent.x + unit[0], agent.y + unit[1]
                ):
                    return unit
        return movement

    def _airlock_pass(self, crew: list[Agent], direction: str) -> bool:
        # Explicit engineering scenario budget: recovered-gas suitlock loses
        # 0.03 kg O2 and draws 0.1 kWh per 10-minute pressure cycle. Neither
        # these consumables nor the chamber create civilian capacity.
        def charge():
            return self.airlock.charge_cycle(
                [c.id for c in crew], direction, self._colony_resources
            )
        allowed = self.airlock.request([c.id for c in crew], direction, self.current_tick, charge)
        for member in crew:
            if allowed:
                member._airlock_wait = None
            else:
                member._airlock_wait = {
                    "direction": direction,
                    "reason": self.airlock.blocked_reason or "pressure_cycle_or_queue",
                    "tick": self.current_tick,
                }
                member._rl_transition_pending = False
        return allowed

    def _nearest_unexplored_step(self, agent: Agent) -> tuple[int, int]:
        """Walk a staged, base-centred survey instead of drawing diagonal rays.

        Each astronaut clears an 8-cell local zone before the authorised
        frontier grows in four-cell increments.  Targets are selected on
        landing-zone-centred rings and movement is cardinal (one axis per
        tick), producing practical traverses rather than an ever-widening X.
        """
        explored = getattr(agent, "explored_cells", set())
        lz_x = getattr(self, "lz_x", agent.spawn_x)
        lz_y = getattr(self, "lz_y", agent.spawn_y)
        frontier_limit = max(
            8, min(
                self.LOCAL_EVA_RADIUS_CELLS,
                int(getattr(agent, "_local_exploration_radius", 8)),
            )
        )
        stable_sector = sum(ord(char) for char in agent.id) % 4
        sector_dx, sector_dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[stable_sector]

        while frontier_limit <= self.LOCAL_EVA_RADIUS_CELLS:
            candidates: list[tuple[int, int, int, int, int]] = []
            for ring in range(1, frontier_limit + 1):
                for ry in range(-ring, ring + 1):
                    for rx in range(-ring, ring + 1):
                        if abs(rx) != ring and abs(ry) != ring:
                            continue
                        cx, cy = lz_x + rx, lz_y + ry
                        if not (0 <= cx < self.world.map_size and 0 <= cy < self.world.map_size):
                            continue
                        if (cx, cy) in explored:
                            continue
                        travel = abs(cx - agent.x) + abs(cy - agent.y)
                        cross_track = abs(rx * sector_dy - ry * sector_dx)
                        forward = -(rx * sector_dx + ry * sector_dy)
                        candidates.append((travel, cross_track, forward, cx, cy))
            if candidates:
                _, _, _, target_x, target_y = min(candidates)
                return self._cardinal_step_toward(agent, target_x, target_y)

            if frontier_limit >= self.LOCAL_EVA_RADIUS_CELLS:
                break
            frontier_limit = min(
                self.LOCAL_EVA_RADIUS_CELLS, frontier_limit + 4
            )
            agent._local_exploration_radius = frontier_limit

        # The local operating area is fully explored: return toward the hub
        # instead of random-walking the frontier outward.
        return self._cardinal_step_toward(agent, lz_x, lz_y)

    def _next_hand_prospect_station(
        self, agent: Agent, resource: str
    ) -> tuple[int, int] | None:
        """Choose a shared test-pit station instead of sampling every footstep."""
        active = getattr(agent, "_hand_prospect_target", None)
        used = getattr(self, "_hand_prospect_centers", set())
        if (
            isinstance(active, dict)
            and active.get("resource") == resource
            and (int(active.get("x")), int(active.get("y"))) not in used
            and not self._is_construction_protected_cell(
                int(active.get("x")), int(active.get("y"))
            )
        ):
            return int(active["x"]), int(active["y"])

        reserved = {
            (int(target["x"]), int(target["y"]))
            for crew in self.agents
            for target in [getattr(crew, "_hand_prospect_target", None)]
            if crew.id != agent.id
            and isinstance(target, dict)
            and target.get("x") is not None
            and target.get("y") is not None
        }
        lz_x = getattr(self, "lz_x", agent.spawn_x)
        lz_y = getattr(self, "lz_y", agent.spawn_y)
        spacing = 4
        sector = sum(ord(char) for char in agent.id) % 4
        directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        candidates: list[tuple[int, int, int, int, int]] = []
        frontier_limit = max(
            8,
            min(
                self.LOCAL_EVA_RADIUS_CELLS,
                int(getattr(agent, "_local_exploration_radius", 8)),
            ),
        )
        for radius in range(
            self.MIN_EXTRACTION_RADIUS_CELLS, frontier_limit + 1, spacing
        ):
            for offset in range(-radius, radius + 1, spacing):
                for dx, dy in (
                    (offset, -radius), (radius, offset),
                    (-offset, radius), (-radius, -offset),
                ):
                    station = (lz_x + dx, lz_y + dy)
                    if station in used or station in reserved:
                        continue
                    if not (
                        0 <= station[0] < self.world.map_size
                        and 0 <= station[1] < self.world.map_size
                    ):
                        continue
                    if self._is_construction_protected_cell(*station):
                        continue
                    if any(
                        structure.get("x") == station[0]
                        and structure.get("y") == station[1]
                        and not structure.get("destroyed", False)
                        for structure in self.placed_structures
                    ):
                        continue
                    radial = self._distance_from_lz(*station)
                    if radial < self.MIN_EXTRACTION_RADIUS_CELLS:
                        continue
                    travel = abs(station[0] - agent.x) + abs(station[1] - agent.y)
                    direction_rank = abs(
                        dx * directions[sector][1] - dy * directions[sector][0]
                    )
                    candidates.append(
                        (radial, travel, direction_rank, station[0], station[1])
                    )
            if candidates:
                break
        if not candidates and frontier_limit < self.LOCAL_EVA_RADIUS_CELLS:
            agent._local_exploration_radius = min(
                self.LOCAL_EVA_RADIUS_CELLS, frontier_limit + 4
            )
            return self._next_hand_prospect_station(agent, resource)
        if not candidates:
            agent._hand_prospect_target = None
            return None
        _, _, _, target_x, target_y = min(candidates)
        agent._hand_prospect_target = {
            "resource": resource,
            "x": target_x,
            "y": target_y,
        }
        return target_x, target_y

    ABUNDANCE_UNITS = {
        "very_abundant": 180,
        "very_rich": 150,
        "abundant": 120,
        "rich": 80,
        "moderate": 45,
        "sparse": 18,
        "trace": 8,
    }

    RECOVERABLE_LAYER_UNITS = {
        "very_abundant": 500_000,
        "very_rich": 350_000,
        "abundant": 220_000,
        "rich": 120_000,
        "moderate": 50_000,
        "sparse": 10_000,
        "trace": 2_000,
    }

    def _get_initial_cell_resource_capacity(
        self,
        x: int,
        y: int,
        resource: str,
        base_resources: dict | None = None,
    ) -> int:
        """Calculate realistic initial deposit units based on planet biome abundance and cell coordinates."""
        base_res = base_resources
        if base_res is None:
            base_res = self.world.get_cell_info(
                x, y, self.current_tick
            ).get("base_resources", {})
        abundance_rating = base_res.get(resource, "moderate")
        base_units = self.ABUNDANCE_UNITS.get(abundance_rating, 45)
        resource_hash = int.from_bytes(
            hashlib.sha256(resource.encode("utf-8")).digest()[:8], "big"
        )
        cell_hash = (x * 73856093 ^ y * 19349663 ^ resource_hash) % 30 - 15
        return max(5, int(base_units * (1.0 + cell_hash / 100.0)))

    def _get_recoverable_cell_resource_capacity(
        self,
        x: int,
        y: int,
        resource: str,
        base_resources: dict | None = None,
    ) -> int:
        """Return the mass-backed inventory along one exposed working face.

        One cell represents 10,000 m².  A 0.10 m regolith cut at 1,500 kg/m³
        contains about 1.5 million kg, or one million 1.5 kg inventory units.
        Subsurface seams use conservative abundance-scaled recoverable panels;
        haul capacity and energy, rather than a fictitious 200 kg planet cell,
        remain the production bottleneck.
        """
        if resource == "regolith":
            return max(1, int(
                self.mission_profile.grid_cell_meters
                * self.mission_profile.grid_cell_meters
                * 0.10 * 1500.0
                / max(0.001, MATERIAL_DENSITY_KG["regolith"])
            ))
        base_res = base_resources
        if base_res is None:
            base_res = self.world.get_cell_info(
                x, y, self.current_tick
            ).get("base_resources", {})
        abundance = str(base_res.get(resource, "moderate"))
        base_units = self.RECOVERABLE_LAYER_UNITS.get(abundance, 50_000)
        digest = hashlib.sha256(
            f"recoverable:{x}:{y}:{resource}".encode("utf-8")
        ).digest()
        variation = (digest[0] % 21 - 10) / 100.0
        return max(100, int(base_units * (1.0 + variation)))

    def add_agent(self, agent: Agent):
        """Add an agent to the simulation."""
        agent.configure_timebase(self.SIM_MINUTES_PER_TICK)
        # Stochastic physiology/hazard outcomes remain replayable for a
        # planet attempt without making every crew member draw the same
        # sequence.
        agent._rng = np.random.default_rng(
            self.seed * 1009 + len(self.agents) * 9176 + 1
        )
        # Policy exploration must not depend on other threads or UI requests.
        # Keep this separate from stochastic physiology draws.
        agent._policy_rng = random.Random(self.seed * 1009 + len(self.agents) * 9176 + 71)
        # The planet schema exposes ``gravity_g``.  Persist it on the agent so
        # physiology, carrying and construction all use the same local value;
        # looking for a non-existent ``gravity_multiplier`` silently ran those
        # systems at 1 g on every planet.
        agent.gravity_g = float(self.planet.gravity_g)
        agent._gravity_g = float(self.planet.gravity_g)
        self.agents.append(agent)
        # Position agent near world center
        cx, cy = self.world.center, self.world.center
        agent.x = cx + len(self.agents) - 1
        agent.y = cy
        
        # Initialize decision engine state & prompt for this agent
        self.decision_engine.init_agent(agent)
        self._system_prompts[agent.id] = build_system_prompt(agent)
    
    def _init_agents(self):
        """Initialize agent positions and system prompts."""
        if getattr(self, "_initialized", False):
            return
        try:
            distribute_capsule_inventory(
                self.agents,
                central_depot_inventory=self.central_depot_inventory,
            )
        except Exception as e:
            logger.warning(f"Failed to distribute capsule inventory: {e}")
            
        # Pre-flight orbital spectroscopy selects a safe landing corridor that
        # puts all planet-present bootstrap feedstock classes within the rover
        # survey envelope. Exact deposits are not exposed to agents; they must
        # still traverse, scan, excavate overburden and recover the material.
        try:
            lz_x, lz_y = self.world.find_spawn_location(
                required_resources=self._bootstrap_planetary_raw_resources(),
                operational_radius_cells=(
                    self.ORBITAL_RECON_OPERATIONAL_RADIUS_CELLS
                ),
            )
        except Exception:
            lz_x, lz_y = self.world.center, self.world.center
        
        self.lz_x = lz_x
        self.lz_y = lz_y
        
        industry_capabilities = dict(
            self.mission_profile.delivered_industry.get("capabilities", {})
        )
        self.placed_structures = [
            {
                "id": "struct_lander_hub",
                "type": "eclss_lander_hub",
                "x": lz_x,
                "y": lz_y,
                "health": 1.0,
                "built_by": "Mission Lander",
                "built_tick": 0,
                "dust_fouling_level": 0.0,
                "site_zone": "precursor_lander_hub",
                "footprint_min_dx_cells": self.LANDER_FOOTPRINT_MIN_DX,
                "footprint_max_dx_cells": self.LANDER_FOOTPRINT_MAX_DX,
                "footprint_min_dy_cells": self.LANDER_FOOTPRINT_MIN_DY,
                "footprint_max_dy_cells": self.LANDER_FOOTPRINT_MAX_DY,
                "airlock_x": lz_x + self.LANDER_AIRLOCK_DX,
                "airlock_y": lz_y + self.LANDER_AIRLOCK_DY,
                "airlock_exterior_x": (
                    lz_x + self.LANDER_AIRLOCK_EXTERIOR_DX
                ),
                "airlock_exterior_y": (
                    lz_y + self.LANDER_AIRLOCK_EXTERIOR_DY
                ),
                "airlock_side": "south",
                "pressurized": True,
                "render_scale": 1.0,
            },
        ]
        industry_layout = {
            # Separate 100 m work cells and a 100 m aisle between equal
            # machines.  The old single-column offsets made furnace #1 share
            # forge #2's coordinate and forge #1 share CNC #2's coordinate.
            # The north-west yard also stays clear of the fixed south suitlock
            # route; passing a pressurised CNC enclosure must not look like a
            # surveyor re-entered the lander during an outbound traverse.
            "cnc_fabricator": (-4, -4),
            "forge": (-6, -4),
            "stone_furnace": (-8, -4),
        }
        for machine_type, (offset_x, offset_y) in industry_layout.items():
            count = int(
                self.mission_profile.delivered_industry.get(
                    f"{machine_type}_count", 0
                )
            )
            for machine_index in range(count):
                self.placed_structures.append({
                    "id": f"struct_{machine_type}_{machine_index + 1}",
                    "type": machine_type,
                    "x": lz_x + offset_x,
                    "y": lz_y + offset_y + machine_index * 2,
                    "health": 1.0,
                    "built_by": "Deployable Cargo Kit",
                    "built_tick": 0,
                    "site_zone": "industrial_yard",
                    "footprint_half_width_cells": 0,
                    "footprint_half_height_cells": 0,
                    "render_scale": 0.66,
                    "pressurized": machine_type == "cnc_fabricator",
                    "capabilities": list(
                        industry_capabilities.get(machine_type, [])
                    ),
                })
        self.structures_built["eclss_lander_hub"] = 1
        self.structures_built["stone_furnace"] = int(
            self.mission_profile.delivered_industry.get(
                "stone_furnace_count", 2
            )
        )
        self.structures_built["forge"] = int(
            self.mission_profile.delivered_industry.get("forge_count", 2)
        )
        self.structures_built["cnc_fabricator"] = int(
            self.mission_profile.delivered_industry.get(
                "cnc_fabricator_count", 2
            )
        )
        self.structure_health["stone_furnace"] = 1.0
        self.structure_health["forge"] = 1.0
        self.structure_health["cnc_fabricator"] = 1.0
        self.structure_health["eclss_lander_hub"] = 1.0
        self.surface_fleet.place_at_base(lz_x, lz_y)
        
        # Clean start: terrain starts with surface regolith intact (unexcavated)
        if not hasattr(self, "revealed_cell_resources"):
            self.revealed_cell_resources = set()
        self.revealed_cell_resources.clear()
        if hasattr(self, "discovered_resources"):
            self.discovered_resources.clear()
        self.cell_geology.clear()
        self.spoil_piles.clear()
        self.cell_excavation_depth.clear()
        self._portable_scan_centers.clear()
        self._portable_scan_attempts.clear()
        self._portable_resource_miss_centers.clear()
        self._hand_prospect_centers.clear()
        self._local_resource_miss_centers.clear()

        # Initial scan around landing zone (discovers surface regolith across field of view)
        self.scan_surroundings(self.lz_x, self.lz_y, radius=7)

        # One shared high-level policy is isolated by planet, just like the
        # individual tactical policies.  Physical site state is reset between
        # attempts; only learned strategy is restored.
        if getattr(self, "_db", None) and not self._strategic_policy_loaded:
            try:
                saved_strategy = self._db.load_agent_q_table(
                    self.strategic_policy.persistence_id, planet_id=self.planet.id
                )
                if saved_strategy and isinstance(saved_strategy, dict):
                    self.strategic_policy.load(saved_strategy)
                    logger.info(
                        "RL PERSISTENCE: Loaded %s shared %s strategy states.",
                        len(self.strategic_policy.q_table), self.planet.id,
                    )
                self._strategic_policy_loaded = True
            except Exception as exc:
                logger.warning("Could not load shared strategy policy: %s", exc)
        elif not getattr(self, "_db", None):
            self._strategic_policy_loaded = True
        
        for i, agent in enumerate(self.agents):
            agent.configure_timebase(self.SIM_MINUTES_PER_TICK)
            # Place in a cluster near the landing zone
            agent.x = lz_x + (i % 3) - 1
            agent.y = lz_y + (i // 3)
            
            # Store spawn locations
            agent.spawn_x = agent.x
            agent.spawn_y = agent.y
            
            # Build and cache system prompt
            self._system_prompts[agent.id] = build_system_prompt(agent)
            self.decision_engine.init_agent(agent)
            
            # Load persistent RL experience (Q-table) from previous simulation runs
            if getattr(self, "_db", None):
                try:
                    saved_q = self._db.load_agent_q_table(
                        agent.id, planet_id=self.planet.id
                    )
                    if saved_q and isinstance(saved_q, dict):
                        agent.q_table = saved_q
                        logger.info(f"RL PERSISTENCE: Loaded {len(saved_q)} {self.planet.id} Q-states for {agent.name}.")
                except Exception as e:
                    logger.warning(f"Could not load previous Q-table for {agent.id}: {e}")
            agent.rl_planet_id = self.planet.id
            agent.rl_episode_trace.clear()
            
            # All agents start inside the landing module (pressurized habitat)
            # This is scientifically accurate — any planetary landing includes
            # a pressurized lander/habitat for initial environmental protection.
            agent._in_habitat = True
            
            # Auto-equip suit if atmosphere is unbreathable
            # Condition: no atmosphere OR PO2 < 16 kPa (below which hypoxia begins)
            # Kepler-442b has atmosphere but only 2% O2 → PO2 ~2 kPa → lethal without suit
            atm = self.planet.atmosphere
            has_atm = atm.get("present", atm.get("has_atmosphere", True))
            if has_atm is None:
                has_atm = True
            o2_frac = atm.get("o2_fraction")
            if o2_frac is None:
                o2_frac = 0.21
            sp_atm = atm.get("surface_pressure_atm")
            if sp_atm is None:
                sp_atm = 1.0 if has_atm else 0.0
            surface_pressure_kpa = sp_atm * 101.325
            po2_kpa = surface_pressure_kpa * o2_frac
            needs_suit = (not has_atm) or (po2_kpa < 16.0)
            if needs_suit:
                if not agent.inventory.has_item("protective_suit"):
                    agent.inventory.add_item("protective_suit", 1)
                agent.equip_suit()
                # Even without a suit item, mark as needing O2 support
                agent._needs_o2_support = True
            else:
                agent._needs_o2_support = False

        lz_cell = self.world.get_cell_info(lz_x, lz_y, 0)
        logger.info(
            f"Landing zone: ({lz_x},{lz_y}) "
            f"biome={lz_cell.get('biome','?')} "
            f"temp={lz_cell.get('temperature_c',0):.1f}°C"
        )
        self._initialized = True
    
    # ================================================================
    # MAIN TICK LOOP
    # ================================================================
    
    def run(self):
        """Run the simulation until an end condition is met."""
        self.running = True
        self._init_agents()
        
        logger.info(f"Simulation starting: {len(self.agents)} agents on {self.planet.name}")
        
        while self.running and self.current_tick < self.max_ticks:
            if self.paused:
                time.sleep(0.1)
                continue
            
            tick_start = time.time()
            
            # === RUN ONE TICK ===
            tick_events = self._run_tick()
            
            # === CHECK END CONDITIONS ===
            if self._check_end_conditions():
                break
            
            # === PACING (maintain target tick speed) ===
            elapsed = time.time() - tick_start
            self._tick_times.append(elapsed)
            if len(self._tick_times) > 100:
                self._tick_times = self._tick_times[-50:]
            
            sleep_time = max(0, self.tick_speed - elapsed)
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)
            
            self.current_tick += 1

        # A high-speed run may end between scheduled live publications. Push
        # the authoritative terminal state once so the UI never stops on an
        # older display frame.
        self._publish_state_snapshot(force=True)
        
        self.running = False
        if self.end_reason == "running":
            self.end_reason = (
                "timeout"
                if self.current_tick >= self.max_ticks
                else "manual_stop"
            )
        return self._get_final_report()
    
    def run_headless(self, ticks: int = None):
        """Run without pacing (as fast as possible). For testing."""
        self._init_agents()
        max_t = ticks or self.max_ticks
        
        import time as _time
        while self.current_tick < max_t:
            tick_start = _time.time()
            self._run_tick()
            elapsed = _time.time() - tick_start
            self._tick_times.append(elapsed)
            if len(self._tick_times) > 100:
                self._tick_times = self._tick_times[-50:]
                
            if self._check_end_conditions():
                break
            self.current_tick += 1

        self._publish_state_snapshot(force=True)

        if self.end_reason == "running" and self.current_tick >= max_t:
            self.end_reason = "timeout"
        return self._get_final_report()
    
    def _run_tick(self) -> dict:
        """Execute one simulation tick. Returns tick events."""
        if not getattr(self, "_initialized", False):
            self._init_agents()
            self._initialized = True

        for crew in self.agents:
            crew._telemetry_tick = self.current_tick

        # Meals are charged when an eat action consumes a physical source.
        # This counter is telemetry only; it must never drain a store again.
        self._crew_food_consumed_kcal_this_tick = 0.0
        # Calories enter packaged colony storage only through an explicit
        # greenhouse harvest action executed by a crew member this tick.
        self._greenhouse_harvested_kcal_this_tick = 0.0

        tick_events = {
            "tick": self.current_tick,
            "world_events": [],
            "agent_events": {},
            "agent_processing_errors": [],
            "deaths": [],
            "fleet_events": [],
            "colony_score": 0,
        }

        # Construction is progressed near the end of a physics tick.  Settle
        # that verified completion before this tick asks the policy for a new
        # objective; otherwise ``choose`` could replace the completed
        # transition and the real module would receive no delayed reward.
        commitment = self.strategic_policy.commitment
        if (
            commitment is not None
            and int(self.structures_built.get(commitment.recipe, 0))
            >= int(commitment.target_count)
        ):
            settled_score = self._update_colony_score()
            settled_reward = self.strategic_policy.observe_completion(
                structures_built=self.structures_built,
                tick=self.current_tick,
                colony_score=settled_score,
                tick_minutes=self.mission_profile.clock.tick_minutes,
                next_state_key=self.decision_engine.get_colony_strategy_state(
                    self.structures_built
                ),
            )
            if settled_reward:
                tick_events["strategy_learning"] = dict(
                    self.strategic_policy.last_outcome or {}
                )
        
        # 1. UPDATE WORLD EVENTS
        active_events = self.event_scheduler.get_active_events(self.current_tick)
        self._active_events = list(active_events)
        for event in active_events:
            event_dict = {
                "type": event.event_type,
                "severity": event.severity,
                "effects": event.effects,
                "description": event.event_type.replace('_', ' ').title(),
            }
            tick_events["world_events"].append(event_dict)
            if self._on_event:
                self._on_event({"type": "world_event", "tick": self.current_tick, **event_dict})

        completed_machine_processes = (
            self._advance_autonomous_manufacturing_cycles()
        )
        if completed_machine_processes:
            tick_events["manufacturing_processes_completed"] = (
                completed_machine_processes
            )
        watchdog_pauses = self._enforce_manufacturing_cycle_invariants()
        if watchdog_pauses:
            tick_events["manufacturing_watchdog_pauses"] = watchdog_pauses
        
        # 2. PROCESS EACH AGENT
        alive_agents = [a for a in self.agents if getattr(a.status, 'value', str(a.status)) != 'dead']
        nearby_count = len(alive_agents)  # Simplified: all agents "near" each other
        
        for agent in alive_agents:
            try:
                agent_events = self._process_agent_tick(agent, active_events, nearby_count)
                if not isinstance(agent_events, dict):
                    agent_events = {}
                tick_events["agent_events"][agent.id] = agent_events
                
                if agent_events.get("died"):
                    if not getattr(agent, "death_cause", None):
                        cause_str = agent_events.get("death_cause", "hypothermia")
                        agent.death_cause = cause_str
                    agent.death_tick = self.current_tick
                    cause_val = agent.death_cause.value if hasattr(agent.death_cause, 'value') else str(agent.death_cause)
                    tick_events["deaths"].append({
                        "agent": agent.name,
                        "cause": cause_val,
                        "tick": self.current_tick,
                    })
                    if self._on_event:
                        self._on_event({
                            "type": "death",
                            "agent": agent.name,
                            "cause": cause_val,
                            "tick": self.current_tick,
                        })
            except Exception as ex:
                error_record = {
                    "tick": int(self.current_tick),
                    "agent_id": agent.id,
                    "agent": agent.name,
                    "exception_type": type(ex).__name__,
                    "message": str(ex),
                }
                tick_events["agent_processing_errors"].append(error_record)
                logger.exception(
                    "Error processing tick for agent %s", agent.name
                )

        # 2.25 Surface robots operate on their own physical clocks. Heavy
        # construction BOMs first leave the depot aboard a rated transporter;
        # they are credited to a named site only after physical unloading.
        cargo_dispatches = self._dispatch_construction_cargo()
        for dispatch in cargo_dispatches:
            if self._on_event:
                self._on_event({**dispatch, "tick": self.current_tick})
        tick_events["fleet_events"] = [
            *cargo_dispatches,
            *self._tick_surface_fleet(),
        ]
        
        # 2.5 PERIODIC AGENT ENCOUNTERS & DIALOGUES (LLM Social Interaction)
        if self.current_tick % 25 == 0 and len(alive_agents) >= 2:
            for i, a1 in enumerate(alive_agents):
                for a2 in alive_agents[i+1:]:
                    dist = max(abs(a1.x - a2.x), abs(a1.y - a2.y))
                    if dist <= 3 or (a1._in_habitat and a2._in_habitat):
                        try:
                            self.decision_engine.handle_encounter(a1, a2, self.current_tick)
                        except Exception as e:
                            logger.warning(f"Encounter dialogue error: {e}")
                        break
        
        # 3. UPDATE COLONY SCORE & REINFORCEMENT LEARNING REWARD SIGNAL
        prev_score = getattr(self, "_last_colony_score", 0.0)
        current_score = self._update_colony_score()
        self.colony_score.record_tick(self.current_tick)
        tick_events["colony_score"] = current_score
        score_delta = current_score - prev_score
        self._last_colony_score = current_score

        strategy_reward = self.strategic_policy.observe_completion(
            structures_built=self.structures_built,
            tick=self.current_tick,
            colony_score=current_score,
            tick_minutes=self.mission_profile.clock.tick_minutes,
            next_state_key=self.decision_engine.get_colony_strategy_state(
                self.structures_built
            ),
        )
        if strategy_reward:
            tick_events["strategy_learning"] = dict(
                self.strategic_policy.last_outcome or {}
            )
        
        # Apply Bellman RL Rewards to All Active Agents
        for agent in alive_agents:
            # Deterministic survival/medical actions do not create a new RL
            # transition. Reusing the last RL action every tick caused stale
            # penalties to accumulate into values such as -1500.
            if not getattr(agent, "_rl_transition_pending", False):
                continue
            # Macro strategic reward from colony progress
            progress_reward = max(-5.0, min(5.0, score_delta * 20.0))
            acute_hazard_reward = 0.0
            outdoor_cold_reward = 0.0
            rl_reward = progress_reward
            
            # Never reward the label of a repeated build/refine/gather tick.
            # Physical completions already issue delayed rewards at their
            # audited event sites, while capacity gain is represented by the
            # score delta above.  This prevents reward farming without output.

            # Survival vitals & thermal reward balance
            temp = getattr(agent.needs, "temperature_stress", 50.0)
            # Being healthy is a constraint, not a repeatable reward source.
            # Physical output/completion and terminal survival supply rewards;
            # waiting or reselecting sleep must not farm a healthy-state bonus.
            if temp < 30 or agent.needs.hunger < 15 or agent.needs.thirst < 15:
                acute_hazard_reward = -3.0
                rl_reward += acute_hazard_reward
                
            # Outdoor cold penalty (learning to not loiter far outside at freezing night)
            if not getattr(agent, "_in_habitat", False) and temp < 40:
                outdoor_cold_reward = -1.5
                rl_reward += outdoor_cold_reward
                
            colony_mats = {}
            for other in self.agents:
                for k, v in other.inventory.materials.items():
                    colony_mats[k] = colony_mats.get(k, 0) + v
            agent_cell = self.world.get_cell_info(agent.x, agent.y, self.current_tick) if hasattr(self, 'world') and self.world else {}
            rl_context = dict(agent_cell)
            rl_context["colony_resources"] = dict(self._colony_resources)
            rl_context["day_night_phase"] = tick_events.get(
                "day_night", {}
            ).get("phase", "day")
            next_state_key = self.decision_engine.get_rl_state_key(
                agent, self.structures_built, colony_mats, rl_context
            )
            self.decision_engine.apply_rl_reward(
                agent,
                rl_reward,
                next_state_key,
                reason="colony progress and survival",
                components={
                    "colony_progress": progress_reward,
                    "acute_hazard": acute_hazard_reward,
                    "outdoor_cold": outdoor_cold_reward,
                },
            )
            agent._rl_transition_pending = False
            
        # RL Death Penalty (Agents learn from fatal mistakes across simulations)
        for d in tick_events.get("deaths", []):
            death_name = d.get("agent") or d.get("name")
            d_agent = next((a for a in self.agents if a.name == death_name), None)
            if d_agent and hasattr(d_agent, "q_table"):
                updated = self.decision_engine.apply_terminal_rl_reward(
                    d_agent, -50.0
                )
                logger.info(
                    f"RL FATAL PENALTY: Applied -50.0 across {updated} "
                    f"recent decisions for {d_agent.name} ({d.get('cause')})"
                )
        
        # 3.4.1 RL Q-Table Persistence Checkpoint
        if (
            self.current_tick > 0
            and self.current_tick % 25 == 0
            and getattr(self, "_db", None)
        ):
            policies = [
                {
                    "agent_id": crew.id,
                    "q_table": crew.q_table,
                    "total_reward": getattr(
                        crew, "total_accumulated_reward", 0.0
                    ),
                }
                for crew in self.agents
                if getattr(crew, "q_table", None)
            ]
            if self._strategic_policy_loaded:
                policies.append({
                    "agent_id": self.strategic_policy.persistence_id,
                    "q_table": self.strategic_policy.dump(),
                    "total_reward": self.strategic_policy.total_reward,
                })
            if policies:
                if hasattr(self._db, "save_agent_q_tables"):
                    self._db.save_agent_q_tables(
                        policies, tick=self.current_tick
                    )
                else:
                    for policy in policies:
                        self._db.save_agent_q_table(
                            policy["agent_id"],
                            policy["q_table"],
                            policy["total_reward"],
                            self.current_tick,
                        )
        
        # 3.4.2 Construction progression: only real person-hours performed at
        # the physical site advance a project. Empty sites no longer build
        # themselves and helpers contribute continuously, not as a one-time
        # duration discount.
        for struct in getattr(self, "placed_structures", []):
            if struct.get("under_construction", False):
                # A laid-out project is visible on the map, but survey stakes
                # and a reserved footprint are not a self-building structure.
                # Physical person-hours begin only after the complete BOM has
                # been staged and consumed at the site.
                if not struct.get("materials_committed", True):
                    struct["active_builder_count"] = 0
                    struct["progress"] = 0.0
                    continue
                sx, sy = struct.get("x", 1000), struct.get("y", 1000)
                assisting_builders = [
                    a for a in self.agents
                    if getattr(a.status, 'value', str(a.status)) != 'dead'
                    and getattr(a.action, 'action_type', '') == 'build'
                    and max(abs(a.x - sx), abs(a.y - sy)) <= 1
                    and (
                        not isinstance(getattr(a.action, "target", None), dict)
                        or a.action.target.get("struct_id") in (None, struct.get("id"))
                    )
                ]
                recipe = self._get_recipe(str(struct.get("type", ""))) or {}
                construction = recipe.get("construction", {})
                productive_crew_limit = max(
                    1, int(construction.get("recommended_crew", 2))
                )
                if len(assisting_builders) > productive_crew_limit:
                    # A compact pressure-system or array work front cannot
                    # gain linearly from unlimited bodies. Prefer the crew
                    # explicitly assigned by the shared order, then the most
                    # productive builders, and release everyone else before
                    # their agent tick so they can take logistics/maintenance
                    # work instead of crowding the site.
                    shared_order = getattr(
                        self.decision_engine, "shared_work_order", {}
                    )
                    assigned_ids = {
                        crew_id
                        for crew_id, role in shared_order.get(
                            "assignments", {}
                        ).items()
                        if role == "construction"
                    }
                    assisting_builders.sort(key=lambda builder: (
                        0 if builder.id in assigned_ids else 1,
                        -self._construction_productivity(builder),
                        builder.id,
                    ))
                    surplus_builders = assisting_builders[
                        productive_crew_limit:
                    ]
                    assisting_builders = assisting_builders[
                        :productive_crew_limit
                    ]
                    for builder in surplus_builders:
                        builder.action.clear()
                robot_assist = self.surface_fleet.assembly_assist(
                    site_id=str(struct.get("id", f"site-{sx}-{sy}")),
                    builder_count=len(assisting_builders),
                    available_grid_energy_kwh=max(
                        0.0,
                        float(self._colony_resources.get(
                            "energy_stored_kwh", 0.0
                        )),
                    ),
                    site_x=int(sx),
                    site_y=int(sy),
                    route=(
                        self._construction_cargo_reservations.get(
                            str(struct.get("id")), {}
                        ).get("surface_route")
                        or [
                            {"x": x, "y": y}
                            for x, y in self._surface_vehicle_route(
                                self.lz_x, self.lz_y, int(sx), int(sy)
                            )
                        ]
                    ),
                )
                if not assisting_builders:
                    struct["active_builder_count"] = 0
                    struct["active_assembly_robot_count"] = 0
                    continue

                human_work_this_tick = sum(
                    self._construction_productivity(builder)
                    for builder in assisting_builders
                ) * self.SIM_HOURS_PER_TICK
                robot_work_this_tick = float(
                    robot_assist.get("work_hours", 0.0)
                )
                work_this_tick = human_work_this_tick + robot_work_this_tick
                required_hours = max(0.1, float(struct.get("required_work_hours", 1.0)))
                completed_hours = min(
                    required_hours,
                    float(struct.get("work_hours_completed", 0.0)) + work_this_tick,
                )
                struct["work_hours_completed"] = completed_hours
                struct["active_builder_count"] = len(assisting_builders)
                struct["active_assembly_robot_count"] = len(
                    robot_assist.get("active_robot_ids", [])
                )
                struct["assembly_robot_equivalent_workers"] = float(
                    robot_assist.get("equivalent_workers", 0.0)
                )
                struct["progress"] = min(1.0, completed_hours / required_hours)
                remaining_hours = max(0.0, required_hours - completed_hours)
                current_rate = max(0.01, work_this_tick / self.SIM_HOURS_PER_TICK)
                struct["ticks_remaining"] = math.ceil(
                    remaining_hours / (current_rate * self.SIM_HOURS_PER_TICK)
                )
                if completed_hours >= required_hours:
                    acceptance = self._construction_acceptance_record(
                        struct,
                        assisting_builders,
                        list(robot_assist.get("active_robot_ids", [])),
                    )
                    struct["acceptance_attempts"] = acceptance["attempt"]
                    struct["acceptance_test"] = acceptance
                    if acceptance["status"] != "passed":
                        original_hours = max(
                            0.1,
                            float(struct.get(
                                "original_required_work_hours", required_hours
                            )),
                        )
                        rework_hours = max(4.0, original_hours * 0.08)
                        struct["required_work_hours"] = (
                            required_hours + rework_hours
                        )
                        struct["construction_phase"] = "rework_and_retest"
                        struct["quality_hold"] = True
                        struct["progress"] = min(
                            0.999,
                            completed_hours / struct["required_work_hours"],
                        )
                        struct["ticks_remaining"] = math.ceil(
                            rework_hours
                            / (current_rate * self.SIM_HOURS_PER_TICK)
                        )
                        if self._on_event:
                            self._on_event({
                                "type": "construction_quality_hold",
                                "agent": struct.get("built_by", "Crew"),
                                "cause": (
                                    f"{struct.get('type', 'structure').replace('_', ' ').title()} "
                                    f"failed acceptance; {rework_hours:.1f} work-hours "
                                    "of correction and retest required"
                                ),
                                "acceptance": dict(acceptance),
                                "tick": self.current_tick,
                            })
                        continue
                    struct["under_construction"] = False
                    struct["progress"] = 1.0
                    struct["ticks_remaining"] = 0
                    struct["completed_tick"] = self.current_tick
                    struct["commissioned_tick"] = self.current_tick
                    struct["construction_phase"] = "commissioned"
                    struct["quality_hold"] = False
                    s_type = struct.get("type", "structure")
                    if s_type == "greenhouse":
                        # Crop commissioning includes planting the first
                        # staggered beds and performing the initial nutrient /
                        # health inspection.  Subsequent growth is automatic
                        # only while daily human service remains current.
                        struct["pressurized"] = True
                        struct["crop_growth_ticks"] = 0
                        struct["last_crop_service_tick"] = self.current_tick
                        struct["last_crop_harvest_tick"] = self.current_tick
                        struct["unharvested_food_kcal"] = 0.0
                        struct["crop_service_due"] = False
                        struct["harvest_due"] = False
                    self.surface_fleet.release_assembly_site(
                        str(struct.get("id", f"site-{sx}-{sy}"))
                    )
                    self.structures_built[s_type] = self.structures_built.get(s_type, 0) + 1
                    self.structure_health[s_type] = 1.0
                    for builder in assisting_builders:
                        if (
                            isinstance(builder.action.target, dict)
                            and builder.action.target.get("struct_id") == struct.get("id")
                        ):
                            builder.action.clear()
                    logger.info(f"CONSTRUCTION COMPLETE: {s_type} is now operational on grid!")
                    if self._on_event:
                        self._on_event({
                            "type": "construction_complete",
                            "agent": struct.get("built_by", "Crew"),
                            "cause": f"Commissioned {s_type.replace('_', ' ').title()} into active colony service!",
                            "tick": self.current_tick
                        })
        
        # 3.4.3 Solar Panel Regolith Dust Accumulation
        is_dust_storm = any(
            (ev.event_type if hasattr(ev, 'event_type') else ev.get('type', '')) in ('dust_storm', 'sandstorm')
            for ev in active_events
        )
        for struct in getattr(self, "placed_structures", []):
            if struct.get("type") in {"solar_panel", "eclss_lander_hub"}:
                physical_tick_scale = self.SIM_MINUTES_PER_TICK / 5.0
                if is_dust_storm:
                    struct["dust_fouling_level"] = min(
                        1.0,
                        struct.get("dust_fouling_level", 0.0)
                        + 0.00025 * physical_tick_scale,
                    )
                elif not struct.get("under_construction", False):
                    struct["dust_fouling_level"] = min(
                        1.0,
                        struct.get("dust_fouling_level", 0.0)
                        + 0.000011 * physical_tick_scale,
                    )

        # 3.5 STRUCTURE DEGRADATION
        self._tick_structure_degradation()
        
        # 3.6 DAY/NIGHT PHASE
        tick_events["day_night"] = self.get_day_night_phase()
        
        # ================================================================
        # 3.7 PRODUCTION SYSTEMS — Energy, O₂, Water, Food (per-tick)
        # ================================================================
        if not hasattr(self, "_colony_resources"):
            # Load initial surface resource stores from planet/presets or NASA DRA 5.0 baseline
            initial_res = getattr(self.planet, "initial_colony_resources", {})
            crew_consumables = self.mission_profile.advance_crew_consumables
            crew_power = self.mission_profile.advance_crew_power
            self._colony_resources = {
                "energy_stored_kwh": float(initial_res.get(
                    "energy_stored_kwh", crew_power["initial_charge_kwh"]
                )),
                "lander_integrated_solar_peak_kw": float(initial_res.get(
                    "lander_integrated_solar_peak_kw",
                    crew_power["integrated_lander_solar_peak_kw"],
                )),
                "lander_auxiliary_power_kw": float(
                    initial_res.get(
                        "lander_auxiliary_power_kw",
                        crew_power["auxiliary_fuel_cell_power_kw"],
                    )
                ),
                "lander_auxiliary_energy_remaining_kwh": float(initial_res.get(
                    "lander_auxiliary_energy_remaining_kwh",
                    crew_power["auxiliary_reactant_energy_kwh"],
                )),
                "o2_reserve_kg": float(initial_res.get(
                    "o2_reserve_kg", crew_consumables["oxygen_kg"]
                )),
                "water_reserve_l": float(initial_res.get(
                    "water_reserve_l", crew_consumables["potable_water_l"]
                )),
                "food_reserve_kcal": float(initial_res.get(
                    "food_reserve_kcal", crew_consumables["food_kcal"]
                )),
            }
            self._food_storage_capacity_kcal = max(
                float(crew_consumables["food_kcal"]),
                self._colony_resources["food_reserve_kcal"]
            )
            self._lander_water_storage_capacity_l = max(
                float(crew_consumables.get("potable_water_l", 0.0)),
                self._colony_resources["water_reserve_l"],
            )
            self._water_storage_capacity_l = (
                self._lander_water_storage_capacity_l
            )
            self._lander_o2_storage_capacity_kg = float(
                crew_consumables.get(
                    "oxygen_buffer_capacity_kg",
                    self._colony_resources["o2_reserve_kg"],
                )
            )
            self._o2_storage_capacity_kg = (
                self._lander_o2_storage_capacity_kg
            )
        
        dn_phase = tick_events.get("day_night", {})
        solar_intensity = dn_phase.get("solar_intensity", 1.0)
        lz_event_effects = self.event_scheduler.get_combined_effects(
            self.current_tick,
            int(getattr(self, "lz_x", 0)),
            int(getattr(self, "lz_y", 0)),
        )
        solar_intensity *= max(
            0.0,
            1.0 - float(lz_event_effects.get(
                "solar_panel_efficiency_reduction", 0.0
            )),
        )
        
        # --- FIX #6: ENERGY PRODUCTION (Factoring in Solar Dust Fouling) ---
        solar_panels = [
            structure for structure in getattr(self, "placed_structures", [])
            if structure.get("type") == "solar_panel"
            and not structure.get("under_construction", False)
            and not structure.get("destroyed", False)
        ]
        avg_dust = sum(s.get("dust_fouling_level", 0.0) for s in solar_panels) / len(solar_panels) if solar_panels else 0.0
        dust_efficiency = max(0.20, 1.0 - avg_dust * 0.80)

        solar_count = len(solar_panels)
        tick_hours = self.SIM_HOURS_PER_TICK
        solar_effects = self._get_recipe("solar_panel").get("output", {}).get(
            "effects", {}
        ) if self._get_recipe("solar_panel") else {}
        solar_peak_kw = float(solar_effects.get("power_output_kw", 25.0))
        solar_flux = float(
            self.planet.data.get("surface", {}).get(
                "solar_flux_relative_to_earth", 1.0
            )
        )
        gross_solar_production_kwh = (
            solar_count * solar_peak_kw * tick_hours
            * solar_flux * solar_intensity * dust_efficiency
        )
        grid_count = self._operational_structure_count(
            "power_distribution_grid"
        )
        grid_recipe = self._get_recipe("power_distribution_grid") or {}
        grid_effects = grid_recipe.get("output", {}).get("effects", {})
        grid_efficiency = min(1.0, max(0.0, float(
            grid_effects.get(
                "power_distribution_efficiency",
                self.POWER_GRID_DEFAULT_EFFICIENCY,
            )
        )))
        feeder_count = max(1, int(grid_effects.get(
            "max_connected_solar_arrays",
            self.POWER_GRID_DEFAULT_SOLAR_FEEDERS,
        )))
        connected_solar_count = min(
            solar_count,
            (
                self.LANDER_DIRECT_SOLAR_INPUTS
                if grid_count <= 0
                else grid_count * feeder_count
            ),
        )
        per_panel_gross_kwh = (
            gross_solar_production_kwh / solar_count
            if solar_count > 0 else 0.0
        )
        direct_solar_count = min(
            connected_solar_count, self.LANDER_DIRECT_SOLAR_INPUTS
        )
        grid_connected_solar_count = max(
            0, connected_solar_count - direct_solar_count
        )
        solar_production_kwh = (
            direct_solar_count * per_panel_gross_kwh
            + grid_connected_solar_count * per_panel_gross_kwh
            * grid_efficiency
        )
        solar_distribution_loss_kwh = max(
            0.0,
            connected_solar_count * per_panel_gross_kwh
            - solar_production_kwh,
        )
        unconnected_solar_count = max(0, solar_count - connected_solar_count)
        lander_hub = next((
            structure
            for structure in getattr(self, "placed_structures", [])
            if structure.get("type") == "eclss_lander_hub"
            and not structure.get("under_construction", False)
            and not structure.get("destroyed", False)
            and float(structure.get("health", 1.0)) > 0.0
        ), None)
        lander_hub_operational = lander_hub is not None
        # This is the lander's own deployable array. It follows local light,
        # stellar flux and dust, and feeds the lander battery directly. It is
        # intentionally absent from structures_built["solar_panel"], so it
        # cannot inflate settlement readiness or constructed-array counts.
        lander_solar_dust_efficiency = max(
            0.20,
            1.0 - float(
                lander_hub.get("dust_fouling_level", 0.0)
                if lander_hub else 0.0
            ) * 0.80,
        )
        lander_solar_production_kwh = (
            float(self._colony_resources.get(
                "lander_integrated_solar_peak_kw", 0.0
            ))
            * tick_hours * solar_flux * solar_intensity
            * lander_solar_dust_efficiency
            if lander_hub_operational else 0.0
        )
        field_solar_production_kwh = solar_production_kwh
        field_gross_solar_production_kwh = gross_solar_production_kwh
        solar_production_kwh += lander_solar_production_kwh
        gross_solar_production_kwh += lander_solar_production_kwh
        auxiliary_power_limit_kwh = (
            float(self._colony_resources.get("lander_auxiliary_power_kw", 10.0))
            * tick_hours if lander_hub_operational else 0.0
        )
        living_count = sum(
            1 for crew in self.agents
            if getattr(crew.status, "value", str(crew.status)) != "dead"
        )
        # Delivered crew ECLSS: selectable water electrolysis sized only for
        # the six-person precursor crew. It does not count toward the
        # 106-person settlement's ISRU capacity. NASA reports 0.84 kg O2/person/day
        # and selectable station OGS output, so this loop follows actual living
        # crew demand up to 4.2 kg/day.
        ticks_per_day = self.mission_profile.clock.ticks_per_earth_day
        lander_ogs_o2_per_tick = (
            min(4.2, living_count * 0.84) / ticks_per_day
        )
        lander_ogs_gross_water_l = lander_ogs_o2_per_tick * 1.125
        # Standard Sabatier recovers approximately half of electrolyser water
        # through metabolic CO2.  Cabin humidity belongs to the separate WRS
        # ledger below and must not be credited here a second time.
        lander_ogs_net_water_l = lander_ogs_gross_water_l * (
            1.0 - self.SABATIER_WATER_RETURN_FRACTION
        )
        # Delivered O2 is a finite buffer, not a request to electrolyse water
        # continuously. Maintain a three-day crew contingency only when it is
        # needed; a full flight tank must not run a 5 kW stack all night.
        lander_ogs_active = (
            lander_hub_operational
            and living_count > 0
            and float(self._colony_resources.get("o2_reserve_kg", 0.0))
                < living_count * 0.84 * 3.0
            and self._colony_resources.get("water_reserve_l", 0.0)
            >= lander_ogs_net_water_l
        )
        
        # Battery storage is physical. Each commissioned solar field includes
        # the serialized storage module declared in its BOM; omitting it here
        # previously left megawatt fields backed only by the 120 kWh lander.
        field_storage_kwh = solar_count * float(
            solar_effects.get("battery_storage_kwh", 0.0)
        )
        max_storage = float(
            self.mission_profile.advance_crew_power["battery_capacity_kwh"]
        ) + field_storage_kwh
        max_storage = max(0.0, max_storage - float(
            self._colony_resources.get("airlock_reserved_energy_kwh", 0.0)
        ))

        emergency_reserve_kwh = max_storage * float(
            self.mission_profile.advance_crew_power[
                "emergency_battery_reserve_fraction"
            ]
        )

        # Base loads are mandatory. Crop chambers are dispatchable industrial
        # loads: a dormant/unwatered farm cannot silently drain the grid, and
        # a partial-power tick earns neither crop growth nor water turnover.
        # All consumers in this phase must see one identical network snapshot.
        # Re-querying once per greenhouse/tank/process unit rebuilt the same
        # topology signature dozens of times per tick.
        life_support_network = self._life_support_network_snapshot()
        operational_greenhouses = self._operational_structures("greenhouse")
        networked_greenhouses = self._connected_life_support_structures(
            "greenhouse", life_support_network
        )
        greenhouse_dispatch_effects = (
            (self._get_recipe("greenhouse") or {})
            .get("output", {}).get("effects", {})
        )
        crop_service_interval_ticks, crop_service_grace_ticks = (
            self._greenhouse_service_limits()
        )
        for greenhouse in operational_greenhouses:
            greenhouse["pressurized"] = True
            greenhouse.setdefault("crop_growth_ticks", 0)
            greenhouse.setdefault(
                "last_crop_service_tick",
                int(greenhouse.get(
                    "commissioned_tick", greenhouse.get("completed_tick", 0)
                ) or 0),
            )
            greenhouse.setdefault("last_crop_harvest_tick", int(
                greenhouse.get(
                    "commissioned_tick", greenhouse.get("completed_tick", 0)
                ) or 0
            ))
            greenhouse.setdefault("unharvested_food_kcal", 0.0)
            crop_service_age = max(
                0,
                self.current_tick
                - int(greenhouse.get("last_crop_service_tick", 0) or 0),
            )
            greenhouse["crop_service_due"] = (
                crop_service_age >= crop_service_interval_ticks
            )
            greenhouse["crop_service_overdue"] = (
                crop_service_age
                > crop_service_interval_ticks + crop_service_grace_ticks
            )
        greenhouse_power_kwh_per_tick = max(
            0.0,
            float(greenhouse_dispatch_effects.get(
                "power_consumption_w", 0.0
            )) / 1000.0 * tick_hours,
        )
        greenhouse_gross_water_l_per_tick = max(
            0.0,
            float(greenhouse_dispatch_effects.get(
                "water_consumption_liters_per_tick", 0.0
            )),
        )
        greenhouse_net_makeup_l_per_tick = (
            greenhouse_gross_water_l_per_tick * 0.10
        )
        # Keep a finite potable contingency for the six-person precursor crew.
        # Crop production is important, but a controller may not spend the last
        # several days of drinking water merely to advance plant age.
        water_buffer_target_l = max(150.0, 30.0 * len(self.agents))
        available_crop_water_l = max(
            0.0,
            float(self._colony_resources.get("water_reserve_l", 0.0))
            - water_buffer_target_l,
        )
        max_greenhouses_by_water = (
            len(networked_greenhouses)
            if greenhouse_net_makeup_l_per_tick <= 1e-9
            else int(available_crop_water_l // greenhouse_net_makeup_l_per_tick)
        )

        # Energy consumption by structures + Sub-zero Trace Heating.
        # Greenhouses are added after the discretionary-load calculation.
        # Process skids are dispatched against live demand below.  Charging
        # every installed electrolyser at nameplate power even with a full O2
        # tank exhausted the lander bus and permanently blocked PLSS service.
        installed_isru_count = self._operational_structure_count(
            "isru_o2_unit"
        )
        networked_isru_count = len(
            self._connected_life_support_structures(
                "isru_o2_unit", life_support_network
            )
        )
        settlement_o2_storage_capacity_kg = (
            self._settlement_storage_capacity(
                "oxygen_buffer_tank", "oxygen_storage_capacity_kg",
                life_support_network,
            )
        )
        settlement_water_storage_capacity_l = (
            self._settlement_storage_capacity(
                "potable_water_tank",
                "potable_water_storage_capacity_liters",
                life_support_network,
            )
        )
        self._o2_storage_capacity_kg = (
            float(getattr(self, "_lander_o2_storage_capacity_kg", 0.0))
            + settlement_o2_storage_capacity_kg
        )
        self._water_storage_capacity_l = (
            float(getattr(
                self, "_lander_water_storage_capacity_l", 0.0
            )) + settlement_water_storage_capacity_l
        )
        # A disconnected/destroyed tank cannot keep its former contents in an
        # invisible global reservoir. Capacity loss vents oxygen or isolates
        # water outside the usable network; usable inventory is clamped here.
        self._colony_resources["o2_reserve_kg"] = min(
            float(self._colony_resources.get("o2_reserve_kg", 0.0)),
            max(0.0, self._o2_storage_capacity_kg - float(
                self._colony_resources.get("airlock_reserved_o2_kg", 0.0)
            )),
        )
        self._colony_resources["water_reserve_l"] = min(
            float(self._colony_resources.get("water_reserve_l", 0.0)),
            self._water_storage_capacity_l,
        )
        isru_dispatch_effects = (
            (self._get_recipe("isru_o2_unit") or {})
            .get("output", {}).get("effects", {})
        )
        isru_o2_per_tick_dispatch = max(
            0.0,
            float(isru_dispatch_effects.get(
                "o2_production_kg_per_day", 0.0
            )) / ticks_per_day,
        )
        o2_storage_headroom_kg = max(
            0.0,
            float(self._o2_storage_capacity_kg)
            - float(self._colony_resources.get("o2_reserve_kg", 0.0)),
        )
        desired_isru_count = (
            min(
                networked_isru_count,
                max(1, int(math.ceil(
                    o2_storage_headroom_kg
                    / max(1e-9, isru_o2_per_tick_dispatch)
                ))),
            )
            if networked_isru_count > 0
            and o2_storage_headroom_kg > 0.001
            and float(self._colony_resources.get("water_reserve_l", 0.0))
                > water_buffer_target_l
            else 0
        )
        # Commissioned expansion habitats remain isolated in cold standby
        # until precursor crew actually occupy them. Keeping every empty
        # 15-kW pressure shell online before the first field array existed
        # could exceed the 18-kW lander array forever, leaving no 0.1-kWh
        # airlock cycle with which to install that array. This is load
        # shedding: nameplate demand and readiness requirements are unchanged.
        occupied_habitat_ids = {
            str(habitat.get("id"))
            for habitat in self._operational_structures("habitat_module")
            if any(
                getattr(crew.status, "value", str(crew.status)) != "dead"
                and bool(getattr(crew, "_in_habitat", False))
                and not self._is_lander_footprint_cell(crew.x, crew.y)
                and max(
                    abs(crew.x - int(habitat.get("x", crew.x))),
                    abs(crew.y - int(habitat.get("y", crew.y))),
                ) <= 1
                for crew in self.agents
            )
        }
        active_habitat_count = len(occupied_habitat_ids)
        base_consumption = 0.0
        for structure_name, count in self.structures_built.items():
            if structure_name in {
                "greenhouse", "isru_o2_unit", "water_collector",
                "water_purifier",
            }:
                continue
            recipe = self._get_recipe(structure_name) or {}
            effects = recipe.get("output", {}).get("effects", {})
            power_w = float(effects.get("power_consumption_w", 0.0))
            powered_count = (
                active_habitat_count
                if structure_name == "habitat_module" else count
            )
            base_consumption += powered_count * power_w / 1000.0 * tick_hours
        # One precursor wastewater train handles the six-person metabolic
        # loop. Additional installed trains remain cold standby until the
        # 106-person load arrives; full readiness still requires firm power
        # for all rated infrastructure through the capacity audit.
        installed_water_purifier_count = self._operational_structure_count(
            "water_purifier"
        )
        networked_water_purifier_count = len(
            self._connected_life_support_structures(
                "water_purifier", life_support_network
            )
        )
        active_water_purifier_count = min(
            networked_water_purifier_count,
            1 if living_count > 0 else 0,
        )
        purifier_effects = (
            (self._get_recipe("water_purifier") or {})
            .get("output", {}).get("effects", {})
        )
        base_consumption += (
            active_water_purifier_count
            * float(purifier_effects.get("power_consumption_w", 0.0))
            / 1000.0 * tick_hours
        )
        if lander_ogs_active:
            # A six-person-class electrolysis stack and its power module.
            base_consumption += 5.0 * tick_hours
        
        # Sub-zero freeze protection (Trace Heating): 0.25 kWh per water collector in sub-zero temps
        current_ambient_temp = float(self.world.get_cell_info(
            int(getattr(self, "lz_x", 0)),
            int(getattr(self, "lz_y", 0)),
            self.current_tick,
        ).get("temperature_c", 20.0))
        water_collector_count = self._operational_structure_count(
            "water_collector"
        )
        networked_water_collector_count = len(
            self._connected_life_support_structures(
                "water_collector", life_support_network
            )
        )
        # Crew contingency and commissioned settlement reserves are distinct
        # demands. The readiness ledger excludes the lander's tankage first,
        # so filling only the 180 L crew buffer can never fill settlement tanks.
        crop_makeup_buffer_l = (
            len(networked_greenhouses)
            * greenhouse_net_makeup_l_per_tick
            * ticks_per_day
        )
        water_storage_status = self.colony_score.get_structure_capacity_status(
            "potable_water_tank"
        ) or {}
        settlement_water_target_l = min(
            settlement_water_storage_capacity_l,
            max(0.0, float(water_storage_status.get("target_capacity", 0.0))),
        )
        water_collection_target_l = min(
            self._water_storage_capacity_l,
            max(
                water_buffer_target_l,
                (
                    float(getattr(self, "_lander_water_storage_capacity_l", 0.0))
                    + settlement_water_target_l
                    if settlement_water_target_l > 0.0 else 0.0
                ),
            ) + crop_makeup_buffer_l,
        )
        water_dispatch_effects = (
            (self._get_recipe("water_collector") or {})
            .get("output", {}).get("effects", {})
        )
        water_per_collector_l = max(0.0, float(water_dispatch_effects.get(
            "water_collection_rate_liters_per_tick", 0.0
        )))
        water_reserve_l = float(self._colony_resources.get("water_reserve_l", 0.0))
        desired_water_collector_count = (
            min(networked_water_collector_count, int(math.ceil(
                max(0.0, water_collection_target_l - water_reserve_l)
                / water_per_collector_l
            )))
            if water_per_collector_l > 1e-9 else 0
        )
        urgent_water_collector_count = min(
            desired_water_collector_count,
            int(math.ceil(
                max(0.0, water_buffer_target_l - water_reserve_l)
                / max(1e-9, water_per_collector_l)
            )),
        )
        water_collector_kwh_per_tick = max(
            0.0,
            float(water_dispatch_effects.get("power_consumption_w", 0.0))
            / 1000.0 * tick_hours,
        ) + (0.25 if current_ambient_temp < 0.0 else 0.0)
        # Urgent drinking water keeps priority. Filling future-colony reserves
        # is optional: it may use solar surplus or battery energy above the
        # emergency floor, never the protected reserve or auxiliary reactants.
        base_consumption += urgent_water_collector_count * water_collector_kwh_per_tick
        stored_energy_kwh = float(self._colony_resources["energy_stored_kwh"])
        water_fill_energy_budget_kwh = max(
            0.0,
            stored_energy_kwh + solar_production_kwh
            - base_consumption - emergency_reserve_kwh,
        )
        optional_water_collector_count = min(
            desired_water_collector_count - urgent_water_collector_count,
            (
                desired_water_collector_count
                if water_collector_kwh_per_tick <= 1e-9
                else int(water_fill_energy_budget_kwh // water_collector_kwh_per_tick)
            ),
        )
        active_water_collector_count = (
            urgent_water_collector_count + optional_water_collector_count
        )
        base_consumption += optional_water_collector_count * water_collector_kwh_per_tick

        isru_power_kwh_per_tick = (
            float(isru_dispatch_effects.get("power_consumption_w", 0.0))
            / 1000.0 * tick_hours
        )
        process_energy_budget_kwh = max(
            0.0,
            max(0.0, stored_energy_kwh - emergency_reserve_kwh)
            + solar_production_kwh - base_consumption,
        )
        active_isru_count = min(
            desired_isru_count,
            (
                desired_isru_count
                if isru_power_kwh_per_tick <= 1e-9
                else int(process_energy_budget_kwh // isru_power_kwh_per_tick)
            ),
        )
        base_consumption += active_isru_count * isru_power_kwh_per_tick
        discretionary_energy_kwh = max(
            0.0,
            max(0.0, stored_energy_kwh - emergency_reserve_kwh)
            + solar_production_kwh - base_consumption,
        )
        max_greenhouses_by_energy = (
            len(networked_greenhouses)
            if greenhouse_power_kwh_per_tick <= 1e-9
            else int(discretionary_energy_kwh // greenhouse_power_kwh_per_tick)
        )
        active_greenhouse_count = min(
            len(networked_greenhouses),
            max_greenhouses_by_water,
            max_greenhouses_by_energy,
        )
        active_greenhouses = networked_greenhouses[:active_greenhouse_count]
        self._active_greenhouse_ids = {
            str(structure.get("id")) for structure in active_greenhouses
        }
        total_consumption = (
            base_consumption
            + active_greenhouse_count * greenhouse_power_kwh_per_tick
        )
        
        auxiliary_remaining = max(0.0, float(self._colony_resources.get(
            "lander_auxiliary_energy_remaining_kwh", 0.0
        )))
        projected_energy_without_auxiliary = (
            stored_energy_kwh + solar_production_kwh - total_consumption
        )
        # Consume finite reactants only when the next power balance would dip
        # below the emergency reserve, never just to pin the battery at 100%.
        auxiliary_needed = max(
            0.0, emergency_reserve_kwh - projected_energy_without_auxiliary
        )
        auxiliary_production_kwh = min(
            auxiliary_power_limit_kwh, auxiliary_remaining, auxiliary_needed
        )
        self._colony_resources["lander_auxiliary_energy_remaining_kwh"] = max(
            0.0, auxiliary_remaining - auxiliary_production_kwh
        )
        energy_delta = solar_production_kwh + auxiliary_production_kwh - total_consumption
        self._colony_resources["energy_stored_kwh"] = max(0.0, min(
            max_storage,
            self._colony_resources["energy_stored_kwh"] + energy_delta
        ))
        
        # Energy deficit shuts down non-essential structures & trace heating
        energy_deficit = self._colony_resources["energy_stored_kwh"] <= 0.0
        self._power_cycle_telemetry = {
            "gross_solar_production_kwh": gross_solar_production_kwh,
            "delivered_solar_production_kwh": solar_production_kwh,
            "field_gross_solar_production_kwh": (
                field_gross_solar_production_kwh
            ),
            "field_solar_production_kwh": field_solar_production_kwh,
            "lander_solar_production_kwh": lander_solar_production_kwh,
            "distribution_loss_kwh": solar_distribution_loss_kwh,
            "connected_solar_arrays": connected_solar_count,
            "unconnected_solar_arrays": unconnected_solar_count,
            "grid_nodes_online": grid_count,
            "distribution_efficiency": grid_efficiency,
            "load_kwh": total_consumption,
            "deficit": energy_deficit,
            "installed_greenhouses": len(operational_greenhouses),
            "networked_greenhouses": len(networked_greenhouses),
            "active_greenhouses": active_greenhouse_count,
            "active_habitats": active_habitat_count,
            "installed_isru_units": installed_isru_count,
            "networked_isru_units": networked_isru_count,
            "active_isru_units": active_isru_count,
            "lander_ogs_active": lander_ogs_active,
            "airlock_reserved_energy_kwh": self._colony_resources.get("airlock_reserved_energy_kwh", 0.0),
            "installed_water_purifiers": installed_water_purifier_count,
            "networked_water_purifiers": networked_water_purifier_count,
            "active_water_purifiers": active_water_purifier_count,
            "installed_water_collectors": water_collector_count,
            "networked_water_collectors": networked_water_collector_count,
            "active_water_collectors": active_water_collector_count,
            "water_collection_target_l": water_collection_target_l,
        }
        
        # --- HABITAT THERMAL BALANCE & RADIATOR LOOP ---
        # A deployed workshop does not emit its rated process heat forever.
        # Count steady life-support loads by installed module and furnaces/
        # forges only while a real manufacturing cycle occupies the machine.
        steady_structure_heat_kw = {
            "eclss_lander_hub": 1.5,
            "isru_o2_unit": 1.2,
            "habitat_module": 0.8,
            "greenhouse": 0.6,
            "water_purifier": 0.3,
        }
        active_machine_cycles: dict[str, int] = {}
        registered_machine_ids = set()
        for machine_id, cycle in self._manufacturing_cycles.items():
            if not cycle.get("completion_pending"):
                continue
            registered_machine_ids.add(str(machine_id))
            if not cycle.get("active"):
                continue
            machine_type = str(cycle.get("machine_type", ""))
            if machine_type:
                active_machine_cycles[machine_type] = (
                    active_machine_cycles.get(machine_type, 0) + 1
                )
        for crew in self.agents:
            action_target = (
                crew.action.target if isinstance(crew.action.target, dict) else {}
            )
            if (
                getattr(crew.action, "action_type", "") == "refine"
                and action_target.get("completion_pending")
                and str(action_target.get("machine_id"))
                not in registered_machine_ids
            ):
                machine_type = str(action_target.get("machine_type", ""))
                if machine_type:
                    active_machine_cycles[machine_type] = (
                        active_machine_cycles.get(machine_type, 0) + 1
                    )
        process_heat_kw = {
            "forge": 2.5,
            "stone_furnace": 1.5,
            "cnc_fabricator": 1.2,
        }
        active_thermal_loads = {
            "eclss_lander_hub": int(lander_hub_operational),
            "isru_o2_unit": active_isru_count,
            "habitat_module": active_habitat_count,
            "greenhouse": active_greenhouse_count,
            "water_purifier": active_water_purifier_count,
        }
        steady_waste_heat_kw = sum(
            active_thermal_loads[struct] * heat
            for struct, heat in steady_structure_heat_kw.items()
        )
        active_process_heat_kw = sum(
            min(
                active_count,
                int(self.structures_built.get(machine_type, 0)),
            ) * process_heat_kw.get(machine_type, 0.0)
            for machine_type, active_count in active_machine_cycles.items()
        )
        total_waste_heat_kw = steady_waste_heat_kw + active_process_heat_kw
        radiator_count = self._operational_structure_count("radiator_panel")
        # Lander hub base radiator capacity = 4.0 kW; each radiator panel adds 8.0 kW
        radiator_effects = (self._get_recipe("radiator_panel") or {}).get("output", {}).get("effects", {})
        total_cooling_capacity_kw = 4.0 + radiator_count * float(radiator_effects.get("heat_rejection_kw", 8.0))
        net_heat_flux = total_waste_heat_kw - total_cooling_capacity_kw
        
        if not hasattr(self, "base_temperature_c"):
            self.base_temperature_c = 21.0
            
        # Lumped thermal mass of the pressure shell, equipment and water loop.
        # Temperature changes from energy (kWh), not directly from power (kW)
        # on every simulation tick. This removes the old permanent-forge
        # runaway while retaining genuine overheating under sustained load.
        habitat_heat_capacity_kwh_per_c = max(
            20.0,
            20.0 + 8.0 * self.structures_built.get("habitat_module", 0),
        )
        temperature_delta_c = (
            net_heat_flux * self.SIM_HOURS_PER_TICK
            / habitat_heat_capacity_kwh_per_c
        )
        self.base_temperature_c = min(
            65.0,
            max(21.0, self.base_temperature_c + temperature_delta_c),
        )
        self._thermal_state = {
            "temperature_c": round(self.base_temperature_c, 3),
            "lander_cooling_capacity_kw": 4.0,
            "active_structure_loads": dict(active_thermal_loads),
            "steady_waste_heat_kw": round(steady_waste_heat_kw, 3),
            "active_process_heat_kw": round(active_process_heat_kw, 3),
            "total_waste_heat_kw": round(total_waste_heat_kw, 3),
            "cooling_capacity_kw": round(total_cooling_capacity_kw, 3),
            "net_heat_kw": round(net_heat_flux, 3),
            "active_machine_cycles": dict(active_machine_cycles),
        }
            
        # Thermal throttling: Overheated machinery suffers 30% performance penalty
        thermal_throttle_mult = 0.70 if self.base_temperature_c > 35.0 else 1.0
        
        # --- FIX #8: O₂ PRODUCTION & CRYOGENIC LOX BOIL-OFF ---
        # ISRU units use water electrolysis. Oxygen is capped by physically
        # available liquid water; no medical/rest action can create feedstock.
        isru_count = active_isru_count
        greenhouse_count = active_greenhouse_count
        isru_recipe = self._get_recipe("isru_o2_unit") or {}
        isru_effects = isru_recipe.get("output", {}).get("effects", {})
        greenhouse_recipe = self._get_recipe("greenhouse") or {}
        greenhouse_effects = greenhouse_recipe.get("output", {}).get("effects", {})
        isru_o2_per_tick = float(
            isru_effects.get("o2_production_kg_per_day", 0.0)
        ) / ticks_per_day
        isru_water_ratio = float(
            isru_effects.get("water_consumption_kg_per_day", 0.0)
        ) / max(
            1e-9, float(isru_effects.get("o2_production_kg_per_day", 0.0))
        )
        if not energy_deficit:
            lander_ogs_o2 = lander_ogs_o2_per_tick if lander_ogs_active else 0.0
            potential_isru_o2 = (
                isru_count * isru_o2_per_tick * thermal_throttle_mult
            )
            available_process_water = max(
                0.0, float(self._colony_resources.get("water_reserve_l", 0.0))
            )
            isru_o2 = min(
                potential_isru_o2,
                available_process_water / max(1e-9, isru_water_ratio),
            )
            isru_water_used_l = isru_o2 * isru_water_ratio
            o2_production = isru_o2 + lander_ogs_o2
            self._colony_resources["hydrogen_byproduct_kg"] = (
                float(self._colony_resources.get("hydrogen_byproduct_kg", 0.0))
                + isru_o2 * 0.125
            )
        else:
            lander_ogs_o2 = 0.0
            isru_o2 = 0.0
            isru_water_used_l = 0.0
            o2_production = 0.0  # Photosynthesis grow lights shut down in energy deficit
        
        # Only oxygen above the lander's separate high-pressure reserve is
        # treated as settlement LOX. An unpowered cryogenic store boils off by
        # the declared fractional rate; a fixed kg/tick loss would make tank
        # size and tick duration physically meaningless.
        oxygen_tank_effects = (
            (self._get_recipe("oxygen_buffer_tank") or {})
            .get("output", {}).get("effects", {})
        )
        cryogenic_o2_kg = max(
            0.0,
            float(self._colony_resources.get("o2_reserve_kg", 0.0))
            - float(getattr(self, "_lander_o2_storage_capacity_kg", 0.0)),
        )
        cryo_boiloff = (
            cryogenic_o2_kg
            * float(oxygen_tank_effects.get(
                "unpowered_boiloff_fraction_per_day", 0.0
            )) / ticks_per_day
            if energy_deficit else 0.0
        )
        
        # Habitat metabolic O₂ load. EVA crew breathe from their PLSS tanks,
        # so charging the central habitat reserve for them would double-count.
        alive_agents = [
            agent for agent in self.agents
            if getattr(agent.status, 'value', str(agent.status)) != 'dead'
        ]
        alive_count = len(alive_agents)
        o2_consumption = sum(
            agent.oxygen_consumption_kg_per_tick()
            for agent in alive_agents if agent._in_habitat
        )
        
        self._colony_resources["o2_reserve_kg"] = max(0.0, min(
            self._o2_storage_capacity_kg,
            self._colony_resources["o2_reserve_kg"]
            + o2_production - o2_consumption - cryo_boiloff,
        ))
        
        # Restore blood/suit-loop O₂ saturation in the habitat. Metabolic
        # oxygen has already been charged continuously above, so deducting an
        # extra fixed amount here would count the same breathing twice.
        for agent in self.agents:
            if agent._in_habitat and agent.needs.o2_supply < 80 and self._colony_resources["o2_reserve_kg"] > 0.0:
                agent.needs.o2_supply = min(100.0, agent.needs.o2_supply + 5.0)
        
        # --- FIX #9: WATER CYCLE & FREEZE PROTECTION ---
        water_purifier_count = networked_water_purifier_count
        
        # Water collection uses exactly one valid source mode. Humid
        # atmospheres need the configured H2O fraction; vacuum worlds consume
        # already recovered water-ice stock. A detector hit alone is not water.
        planet_has_atmosphere = self.planet.atmosphere.get("present", True)
        atmosphere_h2o = float(
            (self.planet.atmosphere.get("composition") or {}).get("H2O", 0.0)
        )
        water_recipe = self._get_recipe("water_collector") or {}
        water_effects = water_recipe.get("output", {}).get("effects", {})
        water_rate = float(
            water_effects.get("water_collection_rate_liters_per_tick", 0.0)
        )
        planned_water_outflow_l = (
            greenhouse_count * float(greenhouse_effects.get(
                "water_consumption_liters_per_tick", 0.0
            )) * 0.10
            + (lander_ogs_net_water_l if lander_ogs_o2 > 0.0 else 0.0)
            + isru_water_used_l
        )
        water_storage_headroom_l = max(
            0.0,
            float(getattr(self, "_water_storage_capacity_l", 0.0))
            - float(self._colony_resources.get("water_reserve_l", 0.0))
            + planned_water_outflow_l,
        )
        requested_water = min(
            active_water_collector_count * water_rate * thermal_throttle_mult,
            water_storage_headroom_l,
            max(
                0.0,
                water_collection_target_l
                - float(self._colony_resources.get("water_reserve_l", 0.0))
                + planned_water_outflow_l,
            ),
        )
        minimum_h2o = float(
            water_effects.get("minimum_atmospheric_h2o_fraction", 0.001)
        )
        if current_ambient_temp < 0.0 and energy_deficit:
            water_production = 0.0  # Pipeline frozen without active trace heating
        elif planet_has_atmosphere and atmosphere_h2o >= minimum_h2o:
            water_production = requested_water
        else:
            depot = getattr(self, "central_depot_inventory", {})
            available_ice_units = max(0.0, float(depot.get("water_ice", 0.0)))
            available_ice_liters = (
                available_ice_units * MATERIAL_DENSITY_KG["water_ice"]
            )
            water_production = min(requested_water, available_ice_liters)
            if water_production > 0.0:
                depot["water_ice"] = max(
                    0.0,
                    available_ice_units
                    - water_production / MATERIAL_DENSITY_KG["water_ice"],
                )
            
        # A purifier changes water quality and wastewater recovery capacity;
        # it cannot multiply the kilograms extracted by a collector.  The old
        # +50% modifier violated conservation of mass and was the apparent
        # source of water that users saw appearing in the base tank.
        
        # Crew water is deducted when an actual 1 L pack/reserve drink occurs.
        # This value is demand telemetry only; subtracting it here as well was
        # the old double-counting bug. Greenhouse process water remains direct.
        crew_water_demand = sum(
            agent.water_requirement_l_per_tick(
                is_eva=not agent._in_habitat,
                activity_multiplier=agent.get_activity_multiplier(),
            )
            for agent in alive_agents
        )
        greenhouse_water_rate = float(
            greenhouse_effects.get("water_consumption_liters_per_tick", 0.0)
        )
        water_consumption = (
            greenhouse_count * greenhouse_water_rate
            + (lander_ogs_net_water_l if lander_ogs_o2 > 0.0 else 0.0)
            + isru_water_used_l
        )
        # Only crop transpiration is condensed here. Electrolyser feed becomes
        # O2 + stored H2 and the lander OGS term is already a net loss.
        transpiration_recovery = (
            greenhouse_count * greenhouse_water_rate * 0.90
        )

        # Water first remains in an explicit crew-body pool. Physiological
        # turnover, not the drink action itself, creates a wastewater stream.
        # In-habitat turnover is collected; EVA perspiration/respiratory loss
        # is conservatively unrecovered until a future suit recovery subsystem
        # is explicitly modelled.  Unused drink-bag water remains in inventory.
        advanced_water_loop = (
            self._operational_structure_count("advanced_eclss") > 0
        )
        water_recovery_fraction = (
            self.ADVANCED_WATER_RECOVERY_FRACTION
            if advanced_water_loop
            else self.HABITAT_WATER_RECOVERY_FRACTION
        )
        recovery_hardware_present = (
            lander_hub_operational
            or advanced_water_loop
            or water_purifier_count > 0
        )
        body_water_turnover_l = 0.0
        wastewater_captured_l = 0.0
        unrecovered_body_water_l = 0.0
        for agent in alive_agents:
            body_pool = max(0.0, float(getattr(
                agent, "_recoverable_body_water_l", 0.0
            )))
            turnover = min(
                body_pool,
                agent.water_requirement_l_per_tick(
                    is_eva=not agent._in_habitat,
                    activity_multiplier=agent.get_activity_multiplier(),
                ),
            )
            if turnover <= 0.0:
                continue
            agent._recoverable_body_water_l = body_pool - turnover
            body_water_turnover_l += turnover
            if agent._in_habitat and recovery_hardware_present:
                recovered_l = turnover * water_recovery_fraction
                wastewater_captured_l += recovered_l
                unrecovered_body_water_l += turnover - recovered_l
                self._water_recovery_queue.append({
                    "ready_tick": self.current_tick + ticks_for_minutes(
                        self.WATER_PROCESSOR_WARMUP_MINUTES,
                        self.SIM_MINUTES_PER_TICK,
                    ),
                    "liters": recovered_l,
                    "source": "habitat_metabolic_wastewater",
                })
            else:
                unrecovered_body_water_l += turnover

        # The one-hour value now represents the documented WPA recirculation /
        # warm-up after wastewater collection, rather than an impossible
        # one-hour drink-to-potable shortcut.  An unpowered processor retains
        # its batch instead of destroying or purifying it for free.
        recovery_processor_online = recovery_hardware_present and not energy_deficit
        potable_recovery = self._process_water_recovery_queue(
            online=recovery_processor_online,
        )
        
        self._colony_resources["water_reserve_l"] = max(
            0.0,
            min(
                float(getattr(self, "_water_storage_capacity_l", 0.0)),
                self._colony_resources["water_reserve_l"] + water_production
                + transpiration_recovery + potable_recovery
                - water_consumption,
            ),
        )
        # Crop age is accumulated productive service time, not wall-clock age.
        # A completed but dormant or human-neglected chamber cannot mature
        # through power, irrigation or crop-health inspection gaps.
        if not energy_deficit:
            for greenhouse in active_greenhouses:
                if self._greenhouse_crop_service_current(greenhouse):
                    greenhouse["crop_growth_ticks"] = int(
                        greenhouse.get("crop_growth_ticks", 0)
                    ) + 1
        self._water_cycle_telemetry = {
            "external_extraction_l": water_production,
            "metabolic_turnover_l": body_water_turnover_l,
            "wastewater_captured_l": wastewater_captured_l,
            "potable_recovered_l": potable_recovery,
            "unrecovered_body_loss_l": unrecovered_body_water_l,
            "greenhouse_recovery_l": transpiration_recovery,
            "process_consumption_l": water_consumption,
            "processor_online": recovery_processor_online,
            "recovery_fraction": water_recovery_fraction,
        }
        
        # --- FIX #10: FOOD BIOMASS, HARVEST LABOR & PACKAGED RESERVE ---
        # A pressure-tested farm is not food on commissioning day, and ripe
        # biomass is not packaged food.  Mature, powered, irrigated and
        # recently inspected chambers accumulate a bounded in-chamber harvest
        # buffer. Only an explicit crew FARM/harvest action transfers calories
        # into the colony food store.
        mature_active_greenhouses = [
            greenhouse for greenhouse in self._mature_greenhouses()
            if str(greenhouse.get("id")) in self._active_greenhouse_ids
        ]
        mature_greenhouse_count = len(mature_active_greenhouses)
        greenhouse_food_rate = float(
            greenhouse_effects.get("food_production_kcal_per_tick", 0.0)
        )
        unharvested_buffer_ticks = max(
            1,
            int(math.ceil(
                float(greenhouse_effects.get(
                    "unharvested_buffer_days", 2.0
                )) * ticks_per_day
            )),
        )
        unharvested_capacity_per_greenhouse = (
            greenhouse_food_rate * unharvested_buffer_ticks
        )
        crop_biomass_generated = 0.0
        productive_greenhouse_count = 0
        for greenhouse in mature_active_greenhouses:
            service_current = self._greenhouse_crop_service_current(greenhouse)
            greenhouse["crop_service_current"] = service_current
            if service_current and not energy_deficit:
                productive_greenhouse_count += 1
                before_biomass = max(
                    0.0,
                    float(greenhouse.get("unharvested_food_kcal", 0.0)),
                )
                after_biomass = min(
                    unharvested_capacity_per_greenhouse,
                    before_biomass + greenhouse_food_rate,
                )
                greenhouse["unharvested_food_kcal"] = after_biomass
                crop_biomass_generated += after_biomass - before_biomass
            harvest_interval_ticks = ticks_for_minutes(
                float(greenhouse_effects.get(
                    "harvest_batch_interval_hours", 24.0
                )) * 60.0,
                self.SIM_MINUTES_PER_TICK,
            )
            greenhouse["harvest_due"] = bool(
                float(greenhouse.get("unharvested_food_kcal", 0.0)) > 0.0
                and self.current_tick - int(greenhouse.get(
                    "last_crop_harvest_tick", 0
                ) or 0) >= harvest_interval_ticks
            )
        food_production = max(
            0.0,
            float(getattr(
                self, "_greenhouse_harvested_kcal_this_tick", 0.0
            )),
        )
        # Agent hunger already advances from the individualized physiological
        # model. Storage mass changes only when a meal is physically issued or
        # consumed; the old continuous subtraction charged those calories a
        # second time, followed by a third charge during passive feeding.
        food_storage_capacity = max(
            460000.0,
            float(getattr(self, "_food_storage_capacity_kcal", 0.0)),
            float(self._colony_resources["food_reserve_kcal"]),
        )
        self._food_storage_capacity_kcal = food_storage_capacity
        self._colony_resources["food_reserve_kcal"] = max(
            0.0,
            min(
                food_storage_capacity,
                self._colony_resources["food_reserve_kcal"],
            ),
        )

        # --- AUTOMATIC EVA FIELD PROVISIONS RESTOCK AT BASE ---
        # Food/water are issued from habitat stores. Used O2 pressure vessels
        # are returned to the depot, but bulk oxygen is never converted into a
        # full portable cylinder without a physical ISRU service action.
        for agent in self.agents:
            if (
                agent._in_habitat
                and getattr(agent.status, 'value', str(agent.status))
                not in {'dead', 'incapacitated'}
            ):
                # 1. Restock food rations (up to 2 packs = 1,400 kcal)
                cur_rations = agent.inventory.items.get("emergency_rations", 0) + agent.inventory.items.get("ration_pack", 0)
                if cur_rations < 2 and self._colony_resources["food_reserve_kcal"] >= 700:
                    agent.inventory.items["emergency_rations"] = agent.inventory.items.get("emergency_rations", 0) + 1
                    self._colony_resources["food_reserve_kcal"] = max(0.0, self._colony_resources["food_reserve_kcal"] - 700.0)
                
                # 2. Restock water packs (up to 2 packs = 2.0 L)
                cur_water = agent.inventory.items.get("water_packs", 0)
                if cur_water < 2 and self._colony_resources["water_reserve_l"] >= 1.0:
                    agent.inventory.items["water_packs"] = agent.inventory.items.get("water_packs", 0) + 1
                    self._colony_resources["water_reserve_l"] = max(0.0, self._colony_resources["water_reserve_l"] - 1.0)

                # 3. Return reusable empty PLSS cylinders to central storage.
                empty_o2 = int(agent.inventory.items.pop(
                    "empty_oxygen_canisters", 0
                ))
                if empty_o2 > 0:
                    self.central_depot_inventory["empty_oxygen_canisters"] = (
                        self.central_depot_inventory.get(
                            "empty_oxygen_canisters", 0
                        ) + empty_o2
                    )

                # Raw materials remain with their carrier until the explicit
                # deposit_materials action transfers them to Central Depot.
                # The former automatic engineer hand-off raced that action,
                # emptied the backpack first and caused endless zero deposits.
        tick_events["colony_production"] = {
            "energy_stored_kwh": round(self._colony_resources["energy_stored_kwh"], 1),
            "energy_delta": round(energy_delta, 2),
            "solar_production_kwh": round(solar_production_kwh, 3),
            "field_solar_production_kwh": round(
                field_solar_production_kwh, 3
            ),
            "lander_solar_production_kwh": round(
                lander_solar_production_kwh, 3
            ),
            "gross_solar_production_kwh": round(
                gross_solar_production_kwh, 3
            ),
            "solar_distribution_loss_kwh": round(
                solar_distribution_loss_kwh, 3
            ),
            "connected_solar_arrays": connected_solar_count,
            "unconnected_solar_arrays": unconnected_solar_count,
            "power_grid_online": grid_count > 0,
            "power_distribution_efficiency": round(grid_efficiency, 3),
            "auxiliary_production_kwh": round(auxiliary_production_kwh, 3),
            "auxiliary_energy_remaining_kwh": round(
                self._colony_resources.get(
                    "lander_auxiliary_energy_remaining_kwh", 0.0
                ), 2
            ),
            "structure_consumption_kwh": round(total_consumption, 3),
            "installed_isru_units": installed_isru_count,
            "networked_isru_units": networked_isru_count,
            "active_isru_units": active_isru_count,
            "installed_water_purifiers": installed_water_purifier_count,
            "networked_water_purifiers": networked_water_purifier_count,
            "active_water_purifiers": active_water_purifier_count,
            "installed_water_collectors": water_collector_count,
            "networked_water_collectors": networked_water_collector_count,
            "active_water_collectors": active_water_collector_count,
            "o2_reserve_kg": round(self._colony_resources["o2_reserve_kg"], 1),
            "o2_storage_capacity_kg": round(
                self._o2_storage_capacity_kg, 1
            ),
            "settlement_o2_storage_capacity_kg": round(
                settlement_o2_storage_capacity_kg, 1
            ),
            "o2_boiloff_kg": round(cryo_boiloff, 6),
            "o2_production_kg": round(o2_production, 4),
            "lander_ogs_o2_production_kg": round(lander_ogs_o2, 4),
            "crew_o2_consumption_kg": round(o2_consumption, 4),
            "water_reserve_l": round(self._colony_resources["water_reserve_l"], 1),
            "water_storage_capacity_l": round(
                self._water_storage_capacity_l, 1
            ),
            "settlement_water_storage_capacity_l": round(
                settlement_water_storage_capacity_l, 1
            ),
            "life_support_distribution_nodes_online": len(
                self._life_support_network_nodes()
            ),
            "water_production_l": round(water_production, 4),
            "water_recovered_l": round(potable_recovery, 4),
            "water_process_consumption_l": round(water_consumption, 4),
            "crew_water_demand_l": round(crew_water_demand, 4),
            "water_cycle": {
                **{
                    key: round(value, 4) if isinstance(value, float) else value
                    for key, value in self._water_cycle_telemetry.items()
                },
                **self._water_mass_ledger(),
            },
            "food_reserve_kcal": round(self._colony_resources["food_reserve_kcal"], 0),
            "food_production_kcal": round(food_production, 2),
            "crop_biomass_generated_kcal": round(
                crop_biomass_generated, 2
            ),
            "installed_greenhouses": len(operational_greenhouses),
            "networked_greenhouses": len(networked_greenhouses),
            "active_greenhouses": active_greenhouse_count,
            "mature_greenhouses": mature_greenhouse_count,
            "productive_greenhouses": productive_greenhouse_count,
            "serviced_greenhouses": sum(
                1 for greenhouse in operational_greenhouses
                if self._greenhouse_crop_service_current(greenhouse)
            ),
            "greenhouses_needing_crop_service": sum(
                1 for greenhouse in operational_greenhouses
                if greenhouse.get("crop_service_due", False)
            ),
            "greenhouse_unharvested_food_kcal": {
                str(greenhouse.get("id")): round(float(
                    greenhouse.get("unharvested_food_kcal", 0.0)
                ), 2)
                for greenhouse in operational_greenhouses
            },
            "greenhouse_crop_growth_ticks": {
                str(greenhouse.get("id")): int(
                    greenhouse.get("crop_growth_ticks", 0)
                )
                for greenhouse in operational_greenhouses
            },
            "crew_food_consumed_kcal": round(
                self._crew_food_consumed_kcal_this_tick, 2
            ),
            "energy_deficit": energy_deficit,
            "communications_online": self._communications_array_operational(),
        }

        # Qualification is assessed after construction, hazards and resource
        # flows for this tick have settled. It therefore cannot be earned by
        # an unpowered nameplate or survive an outage hidden later in a tick.
        tick_events["colony_score"] = self._update_colony_score()
        self._update_support_soak()
        tick_events["support_soak_ticks"] = self._support_soak_ticks
        
        # 4-5: Strategic + Reflection handled by DecisionEngine.process_tick()
        
        # 6. LIVE STATE / PERSISTENCE CALLBACK
        # Physics above always advances on every tick. Only the expensive
        # full-state serialization is rate-limited at accelerated UI speeds.
        self._publish_state_snapshot()
        
        return tick_events

    def _live_state_interval_ticks(self) -> int:
        """Return the display publication cadence without changing physics."""
        seconds_per_tick = max(0.001, float(self.tick_speed))
        return max(
            1,
            int(math.ceil(
                self.LIVE_STATE_MIN_INTERVAL_SECONDS / seconds_per_tick
            )),
        )

    def _publish_state_snapshot(self, force: bool = False) -> bool:
        """Publish current state while retaining persistence checkpoints.

        The web callback performs the SQLite checkpoint on ten-tick
        boundaries, so those ticks remain mandatory even when display frames
        are downsampled. ``force`` is used only for a terminal state.
        """
        if self._on_tick is None:
            return False
        if self._last_state_snapshot_tick == self.current_tick:
            return False
        interval = self._live_state_interval_ticks()
        checkpoint_due = self.current_tick % self.CHECKPOINT_INTERVAL == 0
        if (
            not getattr(self, "capture_every_tick", False)
            and not force
            and not checkpoint_due
            and self.current_tick % interval != 0
        ):
            return False
        self._on_tick(self.current_tick, self._get_state_snapshot())
        self._last_state_snapshot_tick = self.current_tick
        return True
    
    def _process_agent_tick(self, agent: Agent, active_events: list,
                            nearby_count: int) -> dict:
        """Process one tick for a single agent."""
        if getattr(agent.status, 'value', str(agent.status)) == 'dead':
            self._pause_manufacturing_cycle(
                agent,
                "operator_unavailable",
                remaining_ticks=max(1, int(agent.action.ticks_remaining or 1)),
            )
            agent.action.action_type = "dead"
            agent.action.ticks_remaining = 0
            return {"died": True}
        # Mark surroundings as explored by agent and scan resources
        if not hasattr(self, "revealed_cell_resources"):
            self.revealed_cell_resources = set()

        perception_position = (agent.x, agent.y)
        if getattr(agent, "_last_perception_scan_position", None) != perception_position:
            agent.explore_surroundings(agent.x, agent.y, radius=7)
            self.scan_surroundings(agent.x, agent.y, radius=7)
            agent._last_perception_scan_position = perception_position
        
        # Get environment at agent position & filter base_resources to only revealed ones
        raw_cell_info = self.world.get_cell_info(agent.x, agent.y, self.current_tick)
        cell_info = dict(raw_cell_info)
        raw_res = raw_cell_info.get("base_resources", {})
        filtered_res = {}
        for r_name, r_val in raw_res.items():
            if (
                r_name == "regolith"
                or (agent.x, agent.y) in self.discovered_resources.get(r_name, set())
            ):
                if (agent.x, agent.y, r_name) not in getattr(self, "depleted_cell_resources", set()):
                    filtered_res[r_name] = r_val
        cell_info["base_resources"] = filtered_res
        
        # Determine physical shelter and habitat status. The main lander uses
        # its explicit asymmetric 4 x 4 footprint and one south-side airlock.
        lz_x = getattr(self, "lz_x", 1000)
        lz_y = getattr(self, "lz_y", 1000)
        was_in_habitat = bool(getattr(agent, "_in_habitat", False))
        is_inside_lander = self._is_lander_footprint_cell(
            agent.x, agent.y
        )
        agent_expedition = getattr(agent, "_active_expedition", None)
        expedition_rover = (
            self.surface_fleet.crew_rover_for_expedition(
                str(agent_expedition.get("id", ""))
            )
            if isinstance(agent_expedition, dict)
            and agent_expedition.get("transport") == "crew_rover"
            else None
        )
        # A coordinate denotes the footprint of an exterior pressure hull,
        # not an open doorway.  A crew member riding the assigned rover does
        # not enter that hull merely because the routed vehicle crosses one
        # of its map cells.  Explicit expedition/airlock logic owns vehicle
        # pressure transitions and keeps both seats co-located with the rover.
        active_rover_occupant = bool(
            not is_inside_lander
            and expedition_rover is not None
            and expedition_rover.state == "in_use"
            and (agent.x, agent.y) == (expedition_rover.x, expedition_rover.y)
        )
        occupies_pressurized_lander = is_inside_lander and (
            was_in_habitat
            or self._is_lander_airlock_cell(agent.x, agent.y)
        )
        
        # Check if standing at or adjacent (dist <= 1) to an actual placed shelter structure.
        # Only commissioned structures can protect an agent: an unfinished
        # pressure core or a destroyed refuge is not a physical shelter.
        placed_shelter_match = None if active_rover_occupant else next((
            s for s in getattr(self, "placed_structures", [])
            if s.get("type") in [
                "basic_shelter", "habitat_module"
            ]
            and not s.get("under_construction", False)
            and not s.get("destroyed", False)
            and float(s.get("health", 1.0)) > 0.0
            and max(abs(agent.x - s.get("x", -999)), abs(agent.y - s.get("y", -999))) <= 1
        ), None)
        pressurized_structure_match = (
            None if active_rover_occupant
            else self._pressurized_structure_at(agent.x, agent.y)
        )
        
        dn_phase = self.get_day_night_phase()
        cell_info["day_night_phase"] = dn_phase.get("phase", "day")
        
        # Physical presence determines habitat and shelter status.  Values
        # named ``radiation_reduction`` are dose pass-through fractions:
        # 0.02 means 2% of the external dose reaches the occupant.
        if occupies_pressurized_lander:
            agent._in_habitat = True
            has_shelter = True
            ambient_temp = 21.0
            # The flight lander includes a small, water/consumables-lined
            # storm vault for the precursor crew.  It protects the six crew
            # but contributes no civilian readiness capacity.
            shelter_effects = {
                "temperature_stress_reduction": 1.0,
                "radiation_reduction": 0.02,
            }
        elif pressurized_structure_match:
            agent._in_habitat = True
            has_shelter = True
            ambient_temp = 21.0
            structure_type = str(pressurized_structure_match.get("type", ""))
            structure_recipe = self._get_recipe(structure_type) or {}
            structure_protection = (
                structure_recipe.get("output", {}).get("effects", {})
            )
            shelter_effects = {
                "temperature_stress_reduction": float(
                    structure_protection.get(
                        "temperature_stress_reduction", 1.0
                    )
                ),
                # A generic pressure-rated workshop keeps pressure and heat,
                # but is not assumed to have metres of regolith shielding.
                "radiation_reduction": float(
                    structure_protection.get(
                        "radiation_reduction_factor", 0.50
                    )
                ),
            }
        elif placed_shelter_match:
            agent._in_habitat = False
            has_shelter = True
            outdoor_temp = cell_info.get("temperature_c", 20.0)
            ambient_temp = outdoor_temp
            shelter_recipe = self._get_recipe(
                str(placed_shelter_match.get("type", ""))
            ) or {}
            shelter_protection = (
                shelter_recipe.get("output", {}).get("effects", {})
            )
            shelter_effects = {
                "temperature_stress_reduction": float(
                    shelter_protection.get(
                        "temperature_stress_reduction", 0.60
                    )
                ),
                # A 10 cm regolith-bag refuge provides only modest shielding;
                # absent a recipe value, 95% of external dose passes through.
                "radiation_reduction": float(
                    shelter_protection.get(
                        "radiation_reduction_factor", 0.95
                    )
                ),
            }
        else:
            agent._in_habitat = False
            has_shelter = False
            ambient_temp = cell_info.get("temperature_c", 20.0)
            shelter_effects = {}

        # Monitored rest is not a coma. A conscious patient who can swallow
        # may drink from an actual habitat source while remaining in the bunk;
        # this runs before metabolic/death timers so an active rest action
        # cannot conceal available water until fatal dehydration.
        if (not agent._in_habitat and agent.action.action_type in {"sleep", "service_suit"}):
            self._return_for_indoor_recovery(agent, "routine_recovery_requires_pressure")
        medical_rest_oral_hydration = (
            self._oral_hydration_during_medical_rest(agent)
        )
        medical_rest_oral_nutrition = (
            self._oral_nutrition_during_medical_rest(agent)
        )

        # A returning expedition is physically complete only when the whole
        # buddy team is accounted for at the 4 x 4 landing hub. Releasing the
        # lead as soon as *they* crossed the airlock let them accept another
        # EVA while the buddy was still returning, so a completed expedition
        # was never observable as a completed team event.
        returned_expedition = getattr(agent, "_active_expedition", None)
        if (
            occupies_pressurized_lander
            and isinstance(returned_expedition, dict)
            and returned_expedition.get("transport") == "crew_rover"
            and returned_expedition.get("status") == "outbound"
            and returned_expedition.get("kind") != "construction_support"
            and self.current_tick - int(returned_expedition.get("launched_tick", self.current_tick))
            >= max(3 * self.airlock.cycle_ticks,
                   int(returned_expedition.get("available_ticks_at_launch", 36)))
        ):
            team_ids = {returned_expedition.get("lead_id"), returned_expedition.get("buddy_id")}
            if all(
                crew._in_habitat and self._is_lander_footprint_cell(crew.x, crew.y)
                for crew in self.agents if crew.id in team_ids
            ):
                # The original departure window expired without an EVA.
                # Release the vehicle through the normal, safely-home path;
                # no resources or positions are synthesized to make progress.
                self.decision_engine._recall_expedition_team(agent)
        if (
            occupies_pressurized_lander
            and isinstance(returned_expedition, dict)
            and returned_expedition.get("status") in {
                "returning", "awaiting_buddy"
            }
        ):
            self._order_expedition_partner_home(returned_expedition)
            lead = next((
                crew for crew in self.agents
                if crew.id == returned_expedition.get("lead_id")
            ), None)
            buddy_id = returned_expedition.get("buddy_id")
            buddy = next((
                crew for crew in self.agents if crew.id == buddy_id
            ), None) if buddy_id else None
            expedition_team = [
                crew for crew in (lead, buddy) if crew is not None
            ] or [agent]

            def safely_home(crew: Agent) -> bool:
                crew_status = getattr(
                    crew.status, "value", str(crew.status)
                )
                return (
                    crew_status == "dead"
                    or (
                        bool(getattr(crew, "_in_habitat", False))
                        and self._is_lander_footprint_cell(crew.x, crew.y)
                    )
                )

            if all(safely_home(crew) for crew in expedition_team):
                lead_state = (
                    getattr(lead, "_active_expedition", None)
                    if lead is not None else None
                )
                completion_state = (
                    lead_state
                    if isinstance(lead_state, dict) else returned_expedition
                )
                if (
                    completion_state.get("transport") == "crew_rover"
                    and completion_state.get("role") == "lead"
                ):
                    self._complete_crew_rover_expedition(completion_state)
                for member in expedition_team:
                    member._active_expedition = None
                    member._pending_expedition = None
                    member_target = (
                        member.action.target
                        if isinstance(member.action.target, dict) else {}
                    )
                    if (
                        member.action.action_type in {
                            "move", "arrived", "stand_watch"
                        }
                        and (
                            member_target.get("expedition")
                            or member_target.get("awaiting_expedition_member")
                            or member_target.get("destination")
                            in {"habitat", "shelter"}
                        )
                    ):
                        member.action.clear()
            else:
                returned_expedition["status"] = "awaiting_buddy"
                # This is a mission assignment, not a physiology override.
                # Replacing sleep/drink on every tick with STAND_WATCH made
                # safe crew collapse inside while waiting for their buddy.
                if agent.action.action_type in {"move", "arrived"}:
                    agent.action.clear()
        
        has_atmosphere = self.planet.atmosphere.get("present", self.planet.atmosphere.get("has_atmosphere", True))
        if has_atmosphere is None:
            has_atmosphere = True
            
        # Build biome hazards from cell + events. Radiation values are
        # authoritative absorbed-dose increments in Sv/tick. Biome fields
        # live under hazard_modifiers; the previous flat lookup silently
        # ignored them.
        baseline_rad = 0.0
        flare_profile = getattr(self.planet, 'flare_profile', {})
        radiation_system = getattr(self.planet, "radiation_system", {}) or {}
        baseline_by_biome = radiation_system.get(
            "baseline_dose_by_biome_sv_per_hour", {}
        ) or {}
        baseline_hourly = float(
            baseline_by_biome.get(
                str(cell_info.get("biome_id", "")),
                radiation_system.get(
                    "baseline_dose_sv_per_hour",
                    flare_profile.get("radiation_baseline_sv_per_hour", 0.0),
                ),
            )
        )
        baseline_rad = max(0.0, baseline_hourly) * self.SIM_HOURS_PER_TICK
        
        cell_hazards = cell_info.get("hazard_modifiers", {}) or {}
        biome_hazards = {
            **{
                str(key): value for key, value in cell_hazards.items()
                if isinstance(value, (int, float, bool))
            },
            "radiation_per_tick": float(
                cell_hazards.get(
                    "radiation_per_tick",
                    cell_info.get("radiation_per_tick", baseline_rad),
                )
            ),
            "o2_fraction": self.planet.atmosphere.get("o2_fraction", 0.21),
        }

        event_effects = self.event_scheduler.get_combined_effects(
            self.current_tick, agent.x, agent.y
        )
        biome_hazards["environmental_injury_probability_per_tick"] = max(
            0.0,
            float(event_effects.get("injury_probability_per_tick", 0.0)),
        )
        biome_hazards["suit_damage_probability_per_tick"] = max(
            0.0,
            float(event_effects.get("suit_damage_probability_per_tick", 0.0)),
        )
        biome_hazards["event_toxic_exposure_probability_per_tick"] = max(
            0.0,
            float(event_effects.get(
                "toxic_exposure_probability_per_tick", 0.0
            )),
        )
        biome_hazards["uv_lethal_unprotected"] = bool(
            event_effects.get("uv_lethal_unprotected", False)
        )
        cell_info["visibility_factor"] = max(
            0.0,
            1.0 - float(event_effects.get("visibility_reduction", 0.0)),
        )
        if not has_shelter:
            ambient_temp += float(event_effects.get(
                "temperature_bonus_c", 0.0
            ))
        
        # Apply active events. ``radiation_multiplier`` is dimensionless and
        # must never be added as though it were Sv/tick. If a profile supplies
        # a measured/scenario flare rate, severity scales that rate. Otherwise
        # the dimensionless event multiplier scales the configured baseline.
        for ev in active_events:
            ev_type = ev.event_type if hasattr(ev, 'event_type') else ev.get('type', '') if isinstance(ev, dict) else ''
            if ev_type in ("stellar_flare", "solar_flare", "flare"):
                ev_effects = ev.effects if hasattr(ev, 'effects') else ev.get('effects', {}) if isinstance(ev, dict) else {}
                event_severity = float(
                    ev.severity if hasattr(ev, 'severity')
                    else ev.get('severity', 1.0) if isinstance(ev, dict)
                    else 1.0
                )
                explicit_flare_rate = ev_effects.get(
                    "radiation_sv_per_hour",
                    flare_profile.get("radiation_flare_sv_per_hour"),
                )
                if explicit_flare_rate is not None:
                    flare_rad = (
                        max(0.0, float(explicit_flare_rate))
                        * self.SIM_HOURS_PER_TICK
                        * max(0.0, min(1.0, event_severity))
                    )
                else:
                    baseline_rate = max(
                        0.0,
                        float(flare_profile.get(
                            "radiation_baseline_sv_per_hour", 0.0
                        )),
                    )
                    flare_rad = (
                        baseline_rate
                        * max(0.0, float(ev_effects.get(
                            "radiation_multiplier", 1.0
                        )))
                        * self.SIM_HOURS_PER_TICK
                    )
                flare_rad *= max(
                    0.0,
                    float(cell_hazards.get("flare_damage_multiplier", 1.0)),
                )
                biome_hazards["radiation_per_tick"] += flare_rad
            elif ev_type == "dust_storm":
                biome_hazards["visibility"] = cell_info["visibility_factor"]
        
        planet_has_atmosphere = self.planet.atmosphere.get("present", True)
        effective_has_atmosphere = planet_has_atmosphere or agent._in_habitat
        
        # FIX #4: Vacuum pressure — agents outside on atmosphereless planet need suit
        sp_atm = self.planet.atmosphere.get("surface_pressure_atm", None)
        if sp_atm is None:
            surface_pressure = 101.325 if planet_has_atmosphere else 0.0
        else:
            surface_pressure = float(sp_atm) * 101.325
        if not effective_has_atmosphere and surface_pressure < 6.3:  # Armstrong limit
            biome_hazards["pressure_kpa"] = surface_pressure
            # Auto-equip suit if agent has one and is going outside
            if not agent.suit_equipped and not agent._in_habitat:
                agent.suit_equipped = True

        # Physical pressure and breathable oxygen are separate properties. A
        # dense CO2/N2 atmosphere (Kepler-442b) prevents vacuum exposure but
        # still requires a suit/PLSS when PO2 is below 16 kPa.
        effective_pressure = 101.3 if agent._in_habitat else surface_pressure

        # Preserve the actual day/night-adjusted temperature for proactive
        # return decisions and realism telemetry.
        cell_info["effective_temperature_c"] = ambient_temp
        agent._last_effective_temperature_c = ambient_temp

        status_before_tick = getattr(agent.status, 'value', str(agent.status))
        manufacturing_cycle = self._manufacturing_cycle_for_agent(agent)
        manufacturing_remaining_before = None
        if manufacturing_cycle is not None:
            # The operator timer advances the engine-owned WIP record, while
            # still allowing deterministic tests/restores to adjust the timer.
            manufacturing_remaining_before = max(
                1, int(agent.action.ticks_remaining or 1)
            )
            if manufacturing_cycle.get("output_ready"):
                manufacturing_cycle["unload_remaining_ticks"] = (
                    manufacturing_remaining_before
                )
            else:
                manufacturing_cycle["remaining_ticks"] = (
                    manufacturing_remaining_before
                )

        # Capture the suit-service state. Agent physiology proposes the
        # regenerative-rack gain; the authoritative engine then limits that
        # gain by real bus energy instead of allowing free sorbent/battery
        # regeneration inside any habitat.
        plss_before = (
            float(getattr(agent, "plss_co2_scrubber_pct", 100.0)),
            float(getattr(agent, "plss_suit_battery_pct", 100.0)),
            float(getattr(agent, "hypercapnia_level", 0.0)),
        )

        # Update agent physical needs & environment tick
        base_wind = cell_info.get("wind", {}) or {}
        effective_wind_kmh = max(
            0.0,
            float(base_wind.get("speed_kmh", 0.0))
            + float(event_effects.get("wind_speed_bonus_kmh", 0.0)),
        )
        physical_wind_modifier = (
            1.0 + (effective_wind_kmh - 20.0) * 0.02
            if effective_wind_kmh > 20.0 else 1.0
        )
        physical_wind_modifier *= max(
            1.0, float(cell_hazards.get("wind_speed_modifier", 1.0))
        )

        agent_events = agent.tick_update(
            ambient_temp_c=ambient_temp,
            gravity_multiplier=float(self.planet.gravity_g),
            has_shelter=has_shelter,
            shelter_effects=shelter_effects,
            has_atmosphere=effective_has_atmosphere,
            wind_speed_modifier=physical_wind_modifier,
            biome_hazards=biome_hazards,
            pressure_kpa=effective_pressure,
            nearby_agent_count=nearby_count,
        )
        if medical_rest_oral_hydration is not None:
            agent_events["medical_rest_hydration"] = (
                medical_rest_oral_hydration
            )
        if medical_rest_oral_nutrition is not None:
            agent_events["medical_rest_nutrition"] = (
                medical_rest_oral_nutrition
            )

        # Physiology is authoritative. A terminal tick cannot continue into
        # tactical planning and overwrite the visible state with a new drink,
        # build or movement order after the astronaut has already died.
        if getattr(agent.status, "value", str(agent.status)) == "dead":
            self._pause_manufacturing_cycle(
                agent,
                "operator_deceased",
                remaining_ticks=max(1, int(agent.action.ticks_remaining or 1)),
            )
            agent.action.action_type = "dead"
            agent.action.target = {
                "death_cause": (
                    getattr(agent.death_cause, "value", str(agent.death_cause))
                    if agent.death_cause else "unknown"
                )
            }
            agent.action.ticks_remaining = 0
            return agent_events

        plss_after = (
            float(getattr(agent, "plss_co2_scrubber_pct", 100.0)),
            float(getattr(agent, "plss_suit_battery_pct", 100.0)),
            float(getattr(agent, "hypercapnia_level", 0.0)),
        )
        service_requested = bool(
            agent._in_habitat
            and (
                plss_after[0] > plss_before[0]
                or plss_after[1] > plss_before[1]
                or plss_after[2] < plss_before[2]
            )
        )
        if service_requested:
            service_energy_kwh = 0.75 * self.SIM_HOURS_PER_TICK
            available_kwh = max(0.0, float(
                self._colony_resources.get("energy_stored_kwh", 0.0)
            ))
            service_fraction = min(
                1.0, available_kwh / max(1e-9, service_energy_kwh)
            )
            self._colony_resources["energy_stored_kwh"] = max(
                0.0,
                available_kwh - service_energy_kwh * service_fraction,
            )
            agent.plss_co2_scrubber_pct = (
                plss_before[0] + (plss_after[0] - plss_before[0])
                * service_fraction
            )
            agent.plss_suit_battery_pct = (
                plss_before[1] + (plss_after[1] - plss_before[1])
                * service_fraction
            )
            agent.hypercapnia_level = (
                plss_before[2] + (plss_after[2] - plss_before[2])
                * service_fraction
            )
            agent_events["plss_service"] = {
                "energy_draw_kwh": round(
                    service_energy_kwh * service_fraction, 4
                ),
                "service_fraction": round(service_fraction, 3),
            }

        # Manufacturing output becomes available only when the scheduled
        # machine cycle actually finishes. Previously parts appeared instantly
        # and the operator merely remained busy afterward.
        completed_target = (
            agent.action.target if isinstance(agent.action.target, dict) else {}
        )
        suit_job = getattr(agent, "_suit_service_job", None)
        if suit_job and agent.action.action_type == "service_suit":
            suit_job["remaining_ticks"] = max(0, agent.action.ticks_remaining)
            if agent_events.get("action_completed") and agent._in_habitat:
                agent.suit_condition = max(agent.suit_condition, 0.95)
                if suit_job.get("pressure_overhaul"):
                    agent.suit_integrity = max(agent.suit_integrity, 0.85)
                    agent.suit_ticks_used = int((1.0 - agent.suit_integrity) * agent.suit_durability_ticks)
                    agent.has_micro_puncture = False
                agent_events["suit_service_completed"] = dict(suit_job)
                agent._suit_service_job = None
        if (
            manufacturing_cycle is not None
            and agent.action.action_type == "refine"
            and str(completed_target.get("machine_id"))
            == str(manufacturing_cycle.get("machine_id"))
        ):
            if manufacturing_cycle.get("output_ready"):
                manufacturing_cycle["unload_remaining_ticks"] = max(
                    0, int(agent.action.ticks_remaining)
                )
                completed_target["remaining_ticks"] = (
                    manufacturing_cycle["unload_remaining_ticks"]
                )
                completed_target["process_remaining_ticks"] = 0
            else:
                manufacturing_cycle["remaining_ticks"] = max(
                    0, int(agent.action.ticks_remaining)
                )
                completed_target["remaining_ticks"] = manufacturing_cycle[
                    "remaining_ticks"
                ]
        if (
            agent_events.get("action_completed")
            and agent.action.action_type == "refine"
            and completed_target.get("completion_pending")
            and (
                manufacturing_cycle is not None
                or not completed_target.get("machine_id")
            )
        ):
            completion_record = manufacturing_cycle or completed_target
            output = completion_record.get("output")
            quantity = int(completion_record.get("output_quantity", 1))
            if output and quantity > 0:
                agent.inventory.add_material(output, quantity)
                completed_target["completion_pending"] = False
                machine_id = completion_record.get("machine_id")
                if manufacturing_cycle is not None:
                    manufacturing_cycle["completion_pending"] = False
                    manufacturing_cycle["active"] = False
                    if manufacturing_cycle.get("autonomous"):
                        manufacturing_cycle["unloading"] = False
                        manufacturing_cycle["unloaded_tick"] = int(
                            self.current_tick
                        )
                        self._manufacturing_automation_telemetry[
                            "cycles_unloaded"
                        ] += 1
                    if (
                        machine_id
                        and self._manufacturing_cycles.get(str(machine_id))
                        is manufacturing_cycle
                    ):
                        self._manufacturing_cycles.pop(str(machine_id), None)
                agent_events["manufacturing_completed"] = {
                    "output": output,
                    "quantity": quantity,
                    "machine_id": machine_id,
                }
                for crew in self.agents:
                    paused = getattr(crew, "_paused_manufacturing", None)
                    if (
                        isinstance(paused, dict)
                        and machine_id
                        and str(paused.get("machine_id")) == str(machine_id)
                    ):
                        delattr(crew, "_paused_manufacturing")
                self.decision_engine.apply_rl_reward(agent, 4.0, "refine:success")
                if self._on_event:
                    self._on_event({
                        "type": "refine",
                        "agent": agent.name,
                        "output": output,
                        "quantity": quantity,
                        "output_mass_kg": round(
                            quantity * MATERIAL_DENSITY_KG.get(output, 1.0), 6
                        ),
                        "machine_id": machine_id,
                        "cycle_id": completion_record.get("cycle_id"),
                        "cause": (
                            f"Completed {quantity}x {output.replace('_', ' ').title()} "
                            f"after the full machine cycle"
                        ),
                        "tick": self.current_tick,
                    })

        # Every cycle started by the physical workshop has a machine ID and a
        # canonical entry in ``_manufacturing_cycles``.  A stale action copy
        # must never mint another batch after a shift handoff or after the real
        # cycle has completed.  Machine-less targets are retained solely for
        # compatibility with old saves/tests created before machine-owned WIP.
        if (
            agent_events.get("action_completed")
            and agent.action.action_type == "refine"
            and completed_target.get("completion_pending")
            and completed_target.get("machine_id")
            and manufacturing_cycle is None
        ):
            completed_target["completion_pending"] = False
            agent_events["stale_manufacturing_completion_rejected"] = {
                "machine_id": completed_target.get("machine_id"),
                "output": completed_target.get("output"),
            }

        if (
            manufacturing_cycle is not None
            and manufacturing_cycle.get("completion_pending")
            and (
                agent.action.action_type != "refine"
                or str(completed_target.get("machine_id"))
                != str(manufacturing_cycle.get("machine_id"))
            )
        ):
            self._pause_manufacturing_cycle(
                agent,
                "operator_unavailable",
                cycle=manufacturing_cycle,
                remaining_ticks=manufacturing_remaining_before,
            )

        status_after_tick = getattr(agent.status, 'value', str(agent.status))
        if status_before_tick != 'incapacitated' and status_after_tick == 'incapacitated':
            cause = min(
                (
                    (agent.needs.o2_supply, "oxygen depletion"),
                    (agent.needs.temperature_stress, "hypothermia"),
                    (agent.needs.energy, "fatigue collapse"),
                    (agent.needs.thirst, "severe dehydration"),
                    (agent.needs.hunger, "severe caloric depletion"),
                    ((1.0 - agent.injury_level) * 100.0, "traumatic injury"),
                ),
                key=lambda item: item[0],
            )[1]
            alert = {
                "active": True,
                "tick": self.current_tick,
                "victim_ids": [agent.id],
                "victim_names": [agent.name],
                "cause": cause,
            }
            agent_events["team_emergency_broadcast"] = True
            for crew_member in self.agents:
                if getattr(crew_member.status, 'value', str(crew_member.status)) != 'dead':
                    crew_member._team_emergency_alert = dict(alert)
            if self._on_event:
                self._on_event({
                    "type": "team_medical_emergency",
                    "agent": agent.name,
                    "victim_id": agent.id,
                    "cause": cause,
                    "message": f"Crew-wide medical alarm: {agent.name} incapacitated ({cause})",
                    "tick": self.current_tick,
                })

        # Long-running sleep actions normally skip the decision engine. A
        # current crew collapse must break that lock here so every sleeper
        # receives the alarm within at most one simulation tick.
        incapacitated_crew = [
            other for other in self.agents
            if other.id != agent.id
            and getattr(other.status, 'value', str(other.status)) == 'incapacitated'
        ]
        if incapacitated_crew:
            # A casualty who collapsed inside a pressurized habitat is
            # already in the medical bay: admit them to monitored rest
            # without running a fictitious carry/teleport action or restoring
            # any vital for free.
            for victim in incapacitated_crew:
                if (
                    getattr(victim, "_in_habitat", False)
                    and getattr(victim.action, "action_type", "")
                    != "medical_rest"
                ):
                    victim.action.action_type = "medical_rest"
                    victim.action.target = {
                        "habitat": True,
                        "medical_recovery": True,
                        "conscious": False,
                        "oral_fluids_allowed": False,
                        "awaiting_resources": bool(
                            victim.needs.o2_supply <= 0.0
                            or victim.needs.thirst <= 0.0
                            or victim.needs.hunger <= 0.0
                            or victim.needs.temperature_stress <= 0.0
                        ),
                    }
                    victim.action.ticks_remaining = 24

            assigned_sar_victims = [
                victim for victim in incapacitated_crew
                if self.decision_engine.select_sar_rescuer_id(victim, self.agents) == agent.id
            ]
            is_assigned_sar_responder = bool(assigned_sar_victims)
            assigned_medical_victims = []
            for victim in incapacitated_crew:
                if not getattr(victim, "_in_habitat", False):
                    continue
                candidates = [
                    crew for crew in self.agents
                    if crew.id != victim.id
                    and getattr(crew.status, "value", str(crew.status))
                    not in {"dead", "incapacitated"}
                    and float(getattr(crew.competency, "medical", 0)) >= 4
                ]
                if not candidates:
                    continue
                medical_lead = min(
                    candidates,
                    key=lambda crew: (
                        -float(getattr(crew.competency, "medical", 0)),
                        max(abs(crew.x - victim.x), abs(crew.y - victim.y)),
                        crew.id,
                    ),
                )
                if medical_lead.id == agent.id:
                    assigned_medical_victims.append(victim)
            is_assigned_medical_responder = bool(
                assigned_medical_victims
            )
            agent._team_emergency_alert = {
                "active": True,
                "tick": self.current_tick,
                "victim_ids": [other.id for other in incapacitated_crew],
                "victim_names": [other.name for other in incapacitated_crew],
                "assigned_sar": is_assigned_sar_responder,
                "assigned_medical": is_assigned_medical_responder,
            }
            if (
                (is_assigned_sar_responder or is_assigned_medical_responder)
                and getattr(agent.action, "action_type", "") == "sleep"
                and not self.decision_engine.sar_recovery_required(agent)
            ):
                agent.action.action_type = "idle"
                agent.action.ticks_remaining = 0
                agent.action.target = {
                    "woken_by_team_emergency": True,
                    "victims": [other.name for other in incapacitated_crew],
                }
                agent.needs._consecutive_sleep_ticks = 0
                agent._last_emergency_wake_tick = self.current_tick
        elif isinstance(getattr(agent, "_team_emergency_alert", None), dict):
            agent._team_emergency_alert["active"] = False

        # Long construction/crafting/gathering actions used to bypass the
        # decision engine until their whole timer elapsed. Re-open the
        # decision loop as soon as a physiological return/rest threshold is
        # crossed, so an astronaut cannot keep working into a preventable
        # collapse merely because the current job is unfinished.
        current_action = getattr(agent.action, "action_type", "") or "idle"
        protected_recovery_actions = {
            "sleep", "medical_rest", "eat", "drink", "move", "rest",
            "refill_o2", "service_suit", "treat", "rescue", "dead", "unconscious",
        }
        distance_from_base = max(abs(agent.x - lz_x), abs(agent.y - lz_y))
        active_expedition_state = getattr(agent, "_active_expedition", None)
        expedition_in_progress = (
            isinstance(active_expedition_state, dict)
            and active_expedition_state.get("status") in {
                "outbound", "working", "returning"
            }
        )
        overtime_active = self.current_tick <= int(getattr(
            agent, "_fatigue_overtime_until_tick", -1
        ))
        routine_energy_return_threshold = (
            self._routine_eva_energy_return_threshold(
                agent, distance_from_base
            )
        )
        routine_o2_return_threshold = self._routine_eva_o2_return_threshold(
            agent, distance_from_base
        )
        if not agent._in_habitat:
            return_ticks = self._eva_walk_home_ticks(agent)
            routine_energy_return_threshold = min(
                95.0, 55.0 + return_ticks
                * self._estimated_eva_energy_drain_per_tick(agent, 1.5)
            )
            routine_o2_return_threshold = min(
                80.0, 40.0 + return_ticks * agent.plss_o2_percent_per_tick(1.5)
            )
        # DecisionEngine also handles one-tick tactical returns. Publishing
        # the physics-kernel estimates keeps both layers on the same units.
        agent._routine_eva_energy_return_threshold = (
            routine_energy_return_threshold
        )
        agent._routine_eva_o2_return_threshold = routine_o2_return_threshold
        water_return_threshold = (
            0.0 if agent._in_habitat else self._eva_return_water_threshold(agent)
        )
        agent._routine_eva_water_return_threshold = water_return_threshold
        fatigue_interrupt_threshold = (
            15.0 if agent._in_habitat and overtime_active
            else 65.0 if agent._in_habitat
            else 55.0 if expedition_in_progress
            else routine_energy_return_threshold
        )
        plss_return_threshold = (
            0.0 if agent._in_habitat
            else 40.0 if expedition_in_progress
            else routine_o2_return_threshold
        )
        physiological_interrupt = (
            agent.action.ticks_remaining > 0
            and current_action not in protected_recovery_actions
            and (
                agent.needs.energy <= fatigue_interrupt_threshold
                or agent.needs.o2_supply <= plss_return_threshold
                or agent.needs.thirst <= (
                    45.0 if agent._in_habitat else
                    65.0 if agent.inventory.has_item("water_packs") else water_return_threshold
                )
                or agent.needs.hunger <= 35.0
                or agent.needs.temperature_stress <= 42.0
                or agent.needs.temperature_stress >= 88.0
            )
        )
        if physiological_interrupt:
            interrupted_action = current_action
            if (
                interrupted_action == "refine"
                and isinstance(agent.action.target, dict)
                and agent.action.target.get("completion_pending")
            ):
                paused_cycle = self._pause_manufacturing_cycle(
                    agent,
                    "physiological_safety_interrupt",
                    remaining_ticks=max(1, agent.action.ticks_remaining),
                )
                if paused_cycle is None:
                    # Compatibility for legacy/tests that construct an action
                    # without a physical machine ID.
                    agent._paused_manufacturing = {
                        **agent.action.target,
                        "remaining_ticks": max(1, agent.action.ticks_remaining),
                    }
            agent.action.action_type = "idle"
            agent.action.ticks_remaining = 0
            agent.action.target = {
                "interrupted_action": interrupted_action,
                "physiological_safety_interrupt": True,
            }
            agent_events["physiological_safety_interrupt"] = interrupted_action

        if agent_events.get("fatigue_collapse"):
            self._settle_movement_effort(
                agent, completed=False, reason="fatigue_collapse"
            )
            transition = getattr(agent, "_fatigue_risk_transition", None)
            if transition:
                self.decision_engine.apply_delayed_rl_reward(
                    agent, transition[0], transition[1], -15.0
                )
                delattr(agent, "_fatigue_risk_transition")

        active_cycle = self._manufacturing_cycle_for_agent(agent)
        if (
            not physiological_interrupt
            and active_cycle is not None
            and agent.action.ticks_remaining > 0
            and self.current_tick >= int(active_cycle.get("shift_end_tick", 0))
        ):
            self._pause_manufacturing_cycle(
                agent,
                "operator_shift_complete",
                cycle=active_cycle,
                remaining_ticks=agent.action.ticks_remaining,
            )
            machine_id = active_cycle.get("machine_id")
            agent.action.action_type = "idle"
            agent.action.ticks_remaining = 0
            agent.action.target = {
                "interrupted_action": "refine",
                "manufacturing_shift_complete": True,
                "machine_id": machine_id,
            }
            agent_events["manufacturing_shift_ended"] = {
                "machine_id": machine_id,
                "output": active_cycle.get("output"),
                "remaining_ticks": int(active_cycle.get("remaining_ticks", 0)),
            }
            if self._on_event:
                self._on_event({
                    "type": "manufacturing_shift_end",
                    "agent": agent.name,
                    "machine_id": machine_id,
                    "output": active_cycle.get("output"),
                    "tick": self.current_tick,
                })
        
        # Execute active agent action
        action_name = agent.action.action_type
        position_before_action = (agent.x, agent.y)
        started_action_pressurized = bool(agent._in_habitat)
        
        if agent.action.ticks_remaining > 0 and action_name not in ("idle", "move", "explore"):
            # Agent.tick_update() already advances ActionState once per tick.
            # A second decrement here halved sleep/recovery and work durations.
            pass
        else:
            offered_manufacturing_cycle = None
            if (not incapacitated_crew and not active_events
                and not getattr(agent, "_construction_preparation", None)):
                # Claim due machine-owned WIP before ordinary colony planning.
                # Materials and batch energy were already committed, so a
                # replacement must not be rejected because current stores no
                # longer contain the original BOM or power charge.
                offered_manufacturing_cycle = (
                    self._offer_manufacturing_handoff(agent)
                )
            self.decision_engine.structures_built = self.structures_built
            dispatch_resources = set(self.discovered_resources)
            dispatch_resources.update(self.planetary_resources)
            dispatch_resources.add("regolith")
            self.decision_engine.robot_dispatch_targets = {
                resource: target
                for resource in sorted(dispatch_resources)
                if (target := self._robot_dispatch_target(resource)) is not None
            }
            self.decision_engine.structure_health = self.structure_health
            self.decision_engine.placed_structures = self.placed_structures
            self.decision_engine.agents = self.agents
            self.decision_engine.lz_x = getattr(self, "lz_x", 1000)
            self.decision_engine.lz_y = getattr(self, "lz_y", 1000)
            self.decision_engine.max_eva_radius_cells = self.MAX_EVA_RADIUS_CELLS
            self.decision_engine.min_extraction_radius_cells = self.MIN_EXTRACTION_RADIUS_CELLS
            self.decision_engine.expedition_move_speed_cells = self.EXPEDITION_MOVE_SPEED_CELLS
            self.decision_engine.crew_rover_min_distance_cells = (
                self.CREW_ROVER_MIN_DISTANCE_CELLS
            )
            self.decision_engine.expedition_available_ticks = self._expedition_available_ticks(agent)
            self.decision_engine.remote_resource_requests = self.remote_resource_requests
            self.decision_engine.planetary_resources = self.planetary_resources
            self.decision_engine.min_central_maintenance_o2_canisters = (
                self.MIN_CENTRAL_MAINTENANCE_O2_CANISTERS
            )
            self.decision_engine.portable_scanner_scan_charge_pct = (
                self.PORTABLE_SCANNER_SCAN_CHARGE_PCT
            )
            # Exhausting the blind hand-sampling budget should dispatch the
            # portable detector, not falsely declare the whole local survey
            # complete and send a buddy team straight into the regional zone.
            # The local campaign is complete only after its physical detector
            # stations have actually been measured.
            self.decision_engine.local_survey_exhausted = (
                self._next_portable_scan_station(None) is None
            )
            self.decision_engine.local_survey_exhausted_resources = {
                resource
                for resource, centers in self._portable_resource_miss_centers.items()
                if len(centers) >= self.LOCAL_RESOURCE_MISS_LIMIT
                and self._resource_local_campaign_exhausted(resource)
            }
            self.decision_engine.portable_regional_survey_available = any(
                crew.inventory.has_item("portable_scanner")
                and crew.inventory.tool_durability.get("portable_scanner", 0) > 0
                and getattr(crew.status, "value", str(crew.status)) != "dead"
                for crew in self.agents
            )
            self.decision_engine.solar_cleaning_dust_threshold = (
                self.SOLAR_CLEANING_DUST_THRESHOLD
            )
            self.decision_engine.surface_requires_plss = any(
                bool(getattr(crew, "_needs_o2_support", False))
                for crew in self.agents
                if getattr(crew.status, "value", str(crew.status)) != "dead"
            )
            
            # Bug Fix #1: Inject active_events into world_context so decision engine
            # can trigger storm/flare evacuation protocols
            event_dicts = []
            for ev in active_events:
                if hasattr(ev, 'event_type'):
                    event_dicts.append({"type": ev.event_type, "severity": getattr(ev, 'severity', 1.0), "effects": getattr(ev, 'effects', {})})
                elif isinstance(ev, dict):
                    event_dicts.append(ev)
            cell_info["active_events"] = event_dicts
            cell_info["colony_resources"] = dict(self._colony_resources)
            
            # Bug Fix #2: Build nearby_agents list with agent_obj references
            # so medic (Dr. Kwame) can detect and treat injured colonists
            NEARBY_RADIUS = 10
            nearby_agents_list = []
            for other in self.agents:
                if other.id != agent.id and getattr(other.status, 'value', str(other.status)) != 'dead':
                    dist = max(abs(other.x - agent.x), abs(other.y - agent.y))
                    if dist <= NEARBY_RADIUS:
                        nearby_agents_list.append({
                            "agent_obj": other,
                            "name": other.name,
                            "id": other.id,
                            "distance": dist,
                            "injury": other.injury_level,
                            "o2": other.needs.o2_supply,
                            "temp_stress": other.needs.temperature_stress,
                        })
            self.decision_engine.agents = self.agents
            self.decision_engine.current_tick = self.current_tick
            self.decision_engine.central_depot_inventory = getattr(self, "central_depot_inventory", {})
            self.decision_engine.delivered_structure_kits = getattr(
                self, "delivered_structure_kits", {}
            )
            self.decision_engine.spoil_piles = self.spoil_piles
            self.decision_engine._colony_resources = self._colony_resources
            self.decision_engine._thermal_state = dict(getattr(self, "_thermal_state", {}))
            self.decision_engine._power_cycle_telemetry = dict(getattr(self, "_power_cycle_telemetry", {}))
            self.decision_engine._water_storage_capacity_l = getattr(self, "_water_storage_capacity_l", 0.0)
            self.decision_engine.medical_item_available = lambda responder, patient, item: bool(
                self._medical_item_stocks(responder, patient, item)
            )
            self.decision_engine.food_storage_capacity_kcal = float(
                getattr(self, "_food_storage_capacity_kcal", 0.0)
            )
            self.decision_engine.structures_built = self.structures_built
            
            preparation_decision = (
                self._construction_preparation_decision(agent)
                if not incapacitated_crew and not active_events else None
            )
            if preparation_decision is not None:
                decision = preparation_decision
            elif offered_manufacturing_cycle is not None:
                decision = {
                    "action": "refine",
                    "target": {
                        "output": offered_manufacturing_cycle.get("output"),
                        "resume_manufacturing": True,
                        "operator_handoff": True,
                    },
                    "reasoning": (
                        f"{agent.name} accepts the due operator shift on "
                        f"{offered_manufacturing_cycle.get('machine_type')} "
                        f"{offered_manufacturing_cycle.get('machine_id')}"
                    ),
                    "deterministic": True,
                }
            else:
                decision = self.decision_engine.process_tick(
                    agent=agent,
                    tick=self.current_tick,
                    tick_events=agent_events if 'agent_events' in locals() else {},
                    world_context=cell_info,
                    nearby_agents=nearby_agents_list,
                )
            
            action = decision.get("action", "explore")
            target = decision.get("target", {})
            if isinstance(target, str):
                target = {"resource": target, "recipe": target, "name": target}
            elif not isinstance(target, dict):
                target = {}
            reasoning = decision.get("reasoning", "")

            # Authoritative pre-duty self-care interlock. Route persistence,
            # manufacturing handoff or an EVA preflight may produce a valid
            # work order before the tactical policy sees a newly crossed need
            # threshold. Inside a pressurised location, issue the physical
            # drink/meal first instead of allowing that work order to replace
            # self-care with another sleep or airlock retry.
            depot = getattr(self, "central_depot_inventory", {})
            hydration_available_here = bool(
                agent.inventory.has_item("water_packs")
                or depot.get("water_packs", 0) > 0
                or float(self._colony_resources.get(
                    "water_reserve_l", 0.0
                )) > 0.0
            )
            food_available_here = bool(
                agent.inventory.has_item("emergency_rations")
                or agent.inventory.has_item("ration_pack")
                or depot.get("ration_packs", 0) > 0
                or float(self._colony_resources.get(
                    "food_reserve_kcal", 0.0
                )) > 0.0
            )
            if (
                not agent._in_habitat
                and agent.inventory.has_item("oxygen_canisters")
                and agent.needs.o2_supply <= plss_return_threshold
            ):
                # A hydration return must never overwrite a real canister
                # swap selected by the tactical life-support policy.
                action = "refill_o2"
                target = {"carried_spare": True, "pre_duty_interlock": True}
                reasoning = f"{agent.name} swapping carried oxygen before continuing EVA"
            elif (
                not agent._in_habitat
                and not agent.inventory.has_item("oxygen_canisters")
                and agent.needs.o2_supply <= plss_return_threshold
                and action != "refill_o2"
                and not target.get("o2_service")
            ):
                # Use the destination against which the oxygen reserve was
                # budgeted. A tactical detour to another shelter followed by
                # a hydration recall to the lander could otherwise spend the
                # same return reserve twice and strand the crew at the lock.
                expedition = getattr(agent, "_active_expedition", None)
                if isinstance(expedition, dict):
                    self.decision_engine._recall_expedition_team(agent)
                return_x, return_y = self._lander_airlock_position()
                action = "move"
                target = {"x": return_x, "y": return_y, "destination": "shelter",
                          "o2_return": True, "expedition": isinstance(expedition, dict)}
                reasoning = f"{agent.name} preserving the oxygen budget for the planned return"
            elif (
                not agent._in_habitat
                and agent.needs.thirst <= 65.0
                and agent.inventory.has_item("water_packs")
            ):
                action = "drink"
                target = {"field": True, "pre_duty_interlock": True}
                reasoning = f"{agent.name} drinking carried water before continuing EVA"
            elif (
                not agent._in_habitat
                and agent.needs.hunger <= 20.0
                and self._patient_can_take_oral_fluids(agent)
                and (agent.inventory.has_item("emergency_rations")
                     or agent.inventory.has_item("ration_pack"))
            ):
                # An emergency return outranks the ordinary meal policy, but
                # cannot suppress a carried in-suit ration until starvation.
                # The normal EAT executor consumes the real 700-kcal packet;
                # no distant depot or new clinical consumable is substituted.
                action = "eat"
                target = {"field": True, "pre_duty_interlock": True,
                          "emergency_nutrition": True}
                reasoning = f"{agent.name} consuming carried nutrition before continuing the return"
            elif (
                not agent._in_habitat
                and not agent.inventory.has_item("water_packs")
                and agent.needs.thirst <= water_return_threshold
            ):
                expedition = getattr(agent, "_active_expedition", None)
                if isinstance(expedition, dict):
                    self.decision_engine._recall_expedition_team(agent)
                return_x, return_y = self._lander_airlock_position()
                action = "move"
                target = {"x": return_x, "y": return_y, "destination": "shelter",
                          "dehydration_return": True, "expedition": isinstance(expedition, dict)}
                reasoning = f"{agent.name} preserving routed walk-back hydration reserve"
            elif (
                agent._in_habitat
                and agent.needs.thirst < (float(getattr(agent, "_eva_hydration_target", 80.0)) if getattr(agent, "_eva_hydration_pending", False) else 45.0)
                and hydration_available_here
            ):
                action = "drink"
                target = {
                    "habitat": True,
                    "pre_duty_interlock": True,
                    "interrupted_action": action_name,
                }
                reasoning = (
                    f"{agent.name} completing mandatory hydration before duty"
                )
            elif (
                agent._in_habitat
                and agent.needs.hunger <= 45.0
                and food_available_here
            ):
                action = "eat"
                target = {
                    "habitat": True,
                    "pre_duty_interlock": True,
                    "interrupted_action": action_name,
                }
                reasoning = (
                    f"{agent.name} completing mandatory meal before duty"
                )

            # Reject stale/LLM/RL logistics orders before they become visible
            # multi-tick actions. A zero-payload deposit has no physical work
            # to perform and must release the agent for useful work/EVA.
            if (
                action == "deposit_materials"
                and not any(qty > 0 for qty in agent.inventory.materials.values())
            ):
                action = "idle"
                target = {
                    "empty_deposit_rejected": True,
                    "requested_action": "deposit_materials",
                }
                reasoning = (
                    f"{agent.name} cancelled an empty depot transfer; "
                    "no materials are being carried"
                )
                agent._last_empty_deposit_tick = self.current_tick
                # Applicability is a deterministic action mask, not an RL
                # outcome. Do not teach the policy from an executor race.
                agent._rl_transition_pending = False
            
            if not agent._in_habitat and action in {"sleep", "service_suit"}:
                self._return_for_indoor_recovery(agent, action)
                action, target = "move", dict(agent.action.target)
                decision["deterministic"] = True
            action, target = self._indoor_activity_decision(agent, action, target)
            agent.last_decision = {
                "action": action,
                "target": target,
                "reasoning": reasoning,
                "tick": self.current_tick,
                "deterministic": bool(decision.get("deterministic", False)),
                "fallback": bool(decision.get("fallback", False)),
                "source": (
                    "validated_policy"
                    if decision.get("deterministic", False)
                    else "rl_policy"
                ),
            }
            
            # Execute action effects
            has_atmosphere = self.planet.atmosphere.get("present", False)
            
            if action == "dispatch_excavator":
                excavator_target_x = int(target.get("x", agent.x))
                excavator_target_y = int(target.get("y", agent.y))
                excavator_route = self._surface_vehicle_route(
                    self.lz_x,
                    self.lz_y,
                    excavator_target_x,
                    excavator_target_y,
                )
                dispatch = self.surface_fleet.dispatch_excavator(
                    resource=str(target.get("resource", "regolith")),
                    target_x=excavator_target_x,
                    target_y=excavator_target_y,
                    current_tick=self.current_tick,
                    dispatched_by=agent.id,
                    verified_exposed_face=bool(
                        target.get("verified_exposed_face", False)
                    ),
                    protected_ground=bool(target.get("protected_ground", True)),
                    objective_resource=str(
                        target.get(
                            "requested_resource",
                            target.get("resource", "regolith"),
                        )
                    ),
                    rl_state_key=str(getattr(agent, "last_state_key", "")),
                    rl_action_key=str(getattr(agent, "last_action_key", "")),
                    route=[
                        {"x": x, "y": y} for x, y in excavator_route
                    ],
                )
                target.update(dispatch)
                if dispatch.get("dispatched"):
                    agent.action.action_type = "dispatch_excavator"
                    agent.action.target = dict(target)
                    agent.action.ticks_remaining = 1
                    # Reward arrives only when a physical payload returns.
                    agent._rl_transition_pending = False
                else:
                    # A robot availability/face validation rejection is a
                    # physical mask. Strategy learns from later mission
                    # outcomes, never from an impossible command label.
                    agent._rl_transition_pending = False
                    self._apply_failed_excavator_fallback(
                        agent,
                        target,
                        str(dispatch.get("reason", "unknown")),
                    )

            elif action == "eat":
                depot = getattr(self, "central_depot_inventory", {})
                at_food_store = (
                    agent._in_habitat
                    or max(
                        abs(agent.x - getattr(self, "lz_x", 1000)),
                        abs(agent.y - getattr(self, "lz_y", 1000)),
                    ) <= 3
                )
                ration_kcal = float(
                    Agent.FOOD_TYPES["emergency_rations"]["kcal"]
                )
                consumed_kcal = 0.0
                if agent.inventory.has_item("emergency_rations"):
                    agent.inventory.remove_item("emergency_rations", 1)
                    consumed_kcal = ration_kcal
                elif agent.inventory.has_item("ration_pack"):
                    agent.inventory.remove_item("ration_pack", 1)
                    consumed_kcal = ration_kcal
                elif depot.get("ration_packs", 0) > 0 and at_food_store:
                    depot["ration_packs"] -= 1
                    consumed_kcal = ration_kcal
                    logger.info(f"{agent.name} consumed ration pack from Central Depot")
                elif (
                    at_food_store
                    and getattr(self, "_colony_resources", {}).get(
                        "food_reserve_kcal", 0.0
                    ) > 0.0
                ):
                    consumed_kcal = min(
                        ration_kcal,
                        float(self._colony_resources["food_reserve_kcal"]),
                    )
                    self._colony_resources["food_reserve_kcal"] -= consumed_kcal

                ate = consumed_kcal > 0.0
                
                if ate:
                    agent.needs.hunger = min(
                        100.0,
                        agent.needs.hunger
                        + agent.hunger_points_for_kcal(consumed_kcal),
                    )
                    agent.total_kcal_consumed += int(round(consumed_kcal))
                    agent._last_meal_tick = self.current_tick
                    self._crew_food_consumed_kcal_this_tick = (
                        getattr(
                            self, "_crew_food_consumed_kcal_this_tick", 0.0
                        )
                        + consumed_kcal
                    )
                    agent.action.action_type = "eat"
                    agent.action.target = dict(target)
                    agent.action.ticks_remaining = ticks_for_minutes(
                        10.0, self.SIM_MINUTES_PER_TICK
                    )
                    if self._on_event:
                        self._on_event({
                            "type": "vital",
                            "agent": agent.name,
                            "cause": (
                                f"{agent.name} consumed {consumed_kcal:.0f} kcal "
                                f"(Hunger: {agent.needs.hunger:.0f}%)"
                            ),
                            "tick": self.current_tick
                        })
                else:
                    agent.action.action_type = "idle"
                    agent.action.target = {"food_unavailable": True}
                    agent.action.ticks_remaining = 1

            elif action == "drink":
                depot = getattr(self, "central_depot_inventory", {})
                drank = False
                water_liters = 0.0
                if agent.inventory.has_item("water_packs"):
                    agent.inventory.remove_item("water_packs", 1)
                    water_liters = 1.0
                    drank = True
                elif depot.get("water_packs", 0) > 0 and (agent._in_habitat or max(abs(agent.x - getattr(self, "lz_x", 1000)), abs(agent.y - getattr(self, "lz_y", 1000))) <= 3):
                    depot["water_packs"] -= 1
                    water_liters = 1.0
                    logger.info(f"{agent.name} drank water pack from Central Depot")
                    drank = True
                elif (
                    agent._in_habitat
                    and getattr(self, "_colony_resources", {}).get("water_reserve_l", 0) > 0.0
                ):
                    water_liters = min(
                        1.0,
                        float(self._colony_resources["water_reserve_l"]),
                    )
                    self._colony_resources["water_reserve_l"] = max(
                        0.0,
                        self._colony_resources["water_reserve_l"] - water_liters,
                    )
                    drank = water_liters > 0.0
                
                if drank:
                    # This is a transfer from pack/tank into the astronaut,
                    # not an immediate transfer back to the potable tank.
                    # Physiological turnover later decides what reaches the
                    # habitat wastewater processor and what is lost on EVA.
                    agent._recoverable_body_water_l = (
                        max(0.0, float(getattr(
                            agent, "_recoverable_body_water_l", 0.0
                        ))) + water_liters
                    )
                    agent.needs.thirst = min(
                        100.0, agent.needs.thirst + water_liters * 35.0
                    )
                    agent.total_water_consumed_l += water_liters
                    if agent.needs.thirst >= float(getattr(agent, "_eva_hydration_target", 80.0)):
                        agent._eva_hydration_pending = False
                        agent._eva_hydration_target = 80.0
                    agent._last_drink_tick = self.current_tick
                    agent.action.action_type = "drink"
                    agent.action.target = dict(target)
                    # Drinking a prepared one-litre bag or habitat serving is
                    # one ten-minute self-care block. The old twenty-minute
                    # timer consumed almost two crew-hours per person per day
                    # without representing any physical treatment process.
                    agent.action.ticks_remaining = 1
                    if self._on_event:
                        self._on_event({
                            "type": "vital",
                            "agent": agent.name,
                            "cause": f"{agent.name} hydrated (Thirst: {agent.needs.thirst:.0f}%)",
                            "tick": self.current_tick
                        })
                else:
                    agent.action.action_type = "idle"
                    agent.action.target = {"water_unavailable": True}
                    agent.action.ticks_remaining = 1

            elif action == "refill_o2":
                refill_result = self._execute_o2_refill_action(agent)
                if refill_result.get("rover_returning"):
                    agent.action.action_type = "move"
                    agent.action.target = {
                        "x": refill_result["return_x"],
                        "y": refill_result["return_y"],
                        "destination": "shelter",
                        "expedition": True,
                        "transport": "crew_rover",
                        "o2_service_recall": True,
                    }
                    agent.action.ticks_remaining = 1
                elif refill_result.get("moving"):
                    agent.action.action_type = "move"
                    agent.action.target = {
                        "x": refill_result["station_x"],
                        "y": refill_result["station_y"],
                        "destination": "o2_filling_station",
                        "o2_service": True,
                    }
                    agent.action.ticks_remaining = 1
                elif refill_result.get("refilled"):
                    agent.action.action_type = "refill_o2"
                    agent.action.target = dict(refill_result)
                    agent.action.ticks_remaining = ticks_for_minutes(
                        15.0, self.SIM_MINUTES_PER_TICK
                    )
                else:
                    agent.action.action_type = "idle"
                    agent.action.target = {
                        "eva_denied": refill_result.get(
                            "reason", "o2_refill_unavailable"
                        ),
                        **{
                            key: refill_result[key]
                            for key in ("station_x", "station_y")
                            if key in refill_result
                        },
                    }
                    agent.action.ticks_remaining = 1
                if self._on_event and refill_result.get("refilled"):
                    self._on_event({
                        "type": "vital",
                        "agent": agent.name,
                        "cause": (
                            f"{agent.name} serviced PLSS O2 from "
                            f"{refill_result.get('source', 'canister')} (O2: 100%)"
                        ),
                        "tick": self.current_tick
                    })
            elif action == "service_suit":
                self._start_suit_service(agent, bool(target.get("pressure_overhaul")))

            elif action in ("sleep", "rest"):
                lz_x = getattr(self, "lz_x", 1000)
                lz_y = getattr(self, "lz_y", 1000)
                dist_base = max(abs(agent.x - lz_x), abs(agent.y - lz_y))
                
                if dist_base <= 3:
                    if not getattr(agent, "_in_habitat", False):
                        airlock_x, airlock_y = self._lander_airlock_position()
                        if self._is_lander_airlock_cell(agent.x, agent.y):
                            agent.enter_habitat()
                        else:
                            dx, dy = self._cardinal_step_toward(
                                agent, airlock_x, airlock_y, 1
                            )
                            agent.x = max(0, min(self.world.map_size - 1, agent.x + dx))
                            agent.y = max(0, min(self.world.map_size - 1, agent.y + dy))
                            if self._is_lander_airlock_cell(
                                agent.x, agent.y
                            ):
                                agent.enter_habitat()
                            else:
                                agent.action.action_type = "move"
                                agent.action.ticks_remaining = 1
                                agent.action.target = {
                                    "x": airlock_x,
                                    "y": airlock_y,
                                    "destination": "shelter",
                                    "rest_return": True,
                                }
                                return agent_events
                    if action == "rest":
                        # REST is conscious seated/thermal recovery. Folding
                        # it into SLEEP made every short re-warming or queue
                        # wait restart a full sleep cycle, including while a
                        # dehydrated astronaut was trying to wake and work.
                        rest_duration = max(
                            1,
                            int(target.get("ticks", 1))
                            if isinstance(target, dict) else 1,
                        )
                        agent.action.action_type = "rest"
                        agent.action.ticks_remaining = rest_duration
                        agent.action.target = {
                            **(target if isinstance(target, dict) else {}),
                            "ticks": rest_duration,
                            "habitat": True,
                            "awake_recovery": True,
                        }
                    else:
                        default_sleep = ticks_for_minutes(
                            120.0, self.SIM_MINUTES_PER_TICK
                        )
                        sleep_duration = (
                            target.get("ticks", default_sleep)
                            if isinstance(target, dict) else default_sleep
                        )
                        agent.action.action_type = "sleep"
                        agent.action.ticks_remaining = sleep_duration
                        agent.action.target = {
                            **(target if isinstance(target, dict) else {}),
                            "ticks": sleep_duration,
                            "habitat": True,
                        }
                else:
                    if action == "rest":
                        agent.action.action_type = "rest"
                        agent.action.ticks_remaining = 1
                        agent.action.target = {
                            **(target if isinstance(target, dict) else {}),
                            "ticks": 1,
                            "field": True,
                            "awake_recovery": True,
                        }
                    else:
                        # Expedition Field Sleep: a short sealed-suit power nap.
                        field_sleep = ticks_for_minutes(
                            80.0, self.SIM_MINUTES_PER_TICK
                        )
                        sleep_duration = min(
                            field_sleep,
                            target.get("ticks", field_sleep)
                            if isinstance(target, dict) else field_sleep,
                        )
                        agent.action.action_type = "sleep"
                        agent.action.ticks_remaining = sleep_duration
                        agent.action.target = {
                            **(target if isinstance(target, dict) else {}),
                            "ticks": sleep_duration,
                            "field": True,
                        }
                if self._on_event:
                    self._on_event({
                        "type": "vital",
                        "agent": agent.name,
                        "cause": f"{agent.name} resting to restore energy ({agent.needs.energy:.0f}%)",
                        "tick": self.current_tick
                    })
            elif action == "gather":
                # Keep excavation away from habitat foundations, pressure
                # lines, solar cabling and the crew's daily living perimeter.
                lz_x = getattr(self, "lz_x", 1000)
                lz_y = getattr(self, "lz_y", 1000)
                is_protected_construction_ground = (
                    self._is_construction_protected_cell(agent.x, agent.y)
                )
                requested_resource = target.get("resource", "regolith")
                known_raw_materials = {
                    "regolith", "iron_ore", "silica_sand", "basalt",
                    "water_ice", "graphite", "sulfur",
                    "chalcopyrite_ore", "olivine", "calcite",
                }
                if (
                    requested_resource in known_raw_materials
                    and requested_resource not in self.planetary_resources
                ):
                    # Absence is global mission knowledge, not local deposit
                    # knowledge. Do not burn EVA time scanning forever.
                    self.remote_resource_requests.discard(requested_resource)
                    if isinstance(getattr(agent, "_active_excavation", None), dict):
                        if agent._active_excavation.get("resource") == requested_resource:
                            agent._active_excavation = None
                    agent.action.action_type = "idle"
                    agent.action.target = {
                        "resource": requested_resource,
                        "resource_unavailable_on_planet": True,
                        "planet": self.planet.id,
                    }
                    agent.action.ticks_remaining = 1
                    return agent_events

                # A portable detector hit outside practical hand-carry range
                # becomes a persistent cargo-recovery contract.  Preflight it
                # while everyone is still inside the base: the previous flow
                # cycled the airlock first, then silently fell back to a solo
                # walk whenever the rover or buddy was not ready.
                detector_recovery_target = None
                prelaunched_recovery_expedition = None
                recovery_x = target.get("x")
                recovery_y = target.get("y")
                if (
                    target.get("detector_recovery")
                    and recovery_x is not None
                    and recovery_y is not None
                ):
                    detector_recovery_target = (
                        int(recovery_x), int(recovery_y)
                    )
                strict_rover_recovery = bool(
                    detector_recovery_target
                    and self._distance_from_lz(*detector_recovery_target)
                    >= self.CREW_ROVER_MIN_DISTANCE_CELLS
                )
                if strict_rover_recovery:
                    recovery_target_valid = bool(
                        0 <= detector_recovery_target[0] < self.world.map_size
                        and 0 <= detector_recovery_target[1] < self.world.map_size
                        and detector_recovery_target
                        in self.discovered_resources.get(
                            requested_resource, set()
                        )
                        and (
                            detector_recovery_target[0],
                            detector_recovery_target[1],
                            requested_resource,
                        ) not in self.depleted_cell_resources
                        and not self._is_construction_protected_cell(
                            *detector_recovery_target
                        )
                    )
                    if not recovery_target_valid:
                        contract = getattr(
                            agent, "_detected_resource_recovery", None
                        )
                        if isinstance(contract, dict) and (
                            int(contract.get("x", -1)),
                            int(contract.get("y", -1)),
                        ) == detector_recovery_target:
                            agent._detected_resource_recovery = None
                        agent.action.action_type = "rest"
                        agent.action.target = {
                            "detector_recovery_cancelled": (
                                "invalid_or_protected_face"
                            ),
                            "resource": requested_resource,
                            "target": list(detector_recovery_target),
                        }
                        agent.action.ticks_remaining = 1
                        return agent_events

                    target_distance = self._distance_from_lz(
                        *detector_recovery_target
                    )
                    if self._distance_from_lz(agent.x, agent.y) > 2:
                        agent.action.action_type = "move"
                        agent.action.target = {
                            "x": lz_x,
                            "y": lz_y,
                            "destination": "shelter",
                            "resource": requested_resource,
                            "detector_recovery": True,
                            "expedition_preparation": True,
                        }
                        agent.action.ticks_remaining = 1
                        return agent_events

                    capacity_recipe = target.get("capacity_recipe")
                    recovery_goal_units = int(
                        target.get("recovery_goal_units") or 0
                    )
                    if capacity_recipe:
                        deficits = self.decision_engine._raw_bom_deficits(
                            capacity_recipe,
                            self.decision_engine._pooled_materials(),
                        )
                        recovery_goal_units = max(
                            1,
                            int(deficits.get(
                                requested_resource,
                                recovery_goal_units or 1,
                            )),
                        )

                    recovery_face_reserved = any(
                        other.id != agent.id
                        and getattr(
                            other.status, "value", str(other.status)
                        ) not in {"dead", "incapacitated"}
                        and (
                            (
                                isinstance(
                                    getattr(other, "_active_excavation", None),
                                    dict,
                                )
                                and (
                                    int(other._active_excavation.get("x", other.x)),
                                    int(other._active_excavation.get("y", other.y)),
                                ) == detector_recovery_target
                            )
                            or (
                                isinstance(other.action.target, dict)
                                and (
                                    other.action.target.get("x"),
                                    other.action.target.get("y"),
                                ) == detector_recovery_target
                                and other.action.target.get("destination") in {
                                    "extraction_face",
                                    "active_excavation_face",
                                }
                            )
                        )
                        for other in self.agents
                    )

                    # A partially opened face can survive an interrupted
                    # ordinary walking sortie.  Once that astronaut is back
                    # at the hub and no longer travelling to or working the
                    # face, the stale local marker must not permanently block
                    # the detector-confirmed cargo expedition.
                    if recovery_face_reserved:
                        for other in self.agents:
                            if other.id == agent.id:
                                continue
                            active_face = getattr(
                                other, "_active_excavation", None
                            )
                            if not isinstance(active_face, dict) or (
                                int(active_face.get("x", other.x)),
                                int(active_face.get("y", other.y)),
                            ) != detector_recovery_target:
                                continue
                            other_target = (
                                other.action.target
                                if isinstance(other.action.target, dict)
                                else {}
                            )
                            at_face = max(
                                abs(other.x - detector_recovery_target[0]),
                                abs(other.y - detector_recovery_target[1]),
                            ) <= 1 and not other._in_habitat
                            travelling_to_face = bool(
                                getattr(other.action, "action_type", "")
                                == "move"
                                and other_target.get("destination") in {
                                    "extraction_face",
                                    "active_excavation_face",
                                }
                                and (
                                    other_target.get("x"),
                                    other_target.get("y"),
                                ) == detector_recovery_target
                            )
                            if not at_face and not travelling_to_face:
                                other._active_excavation = None

                        recovery_face_reserved = any(
                            other.id != agent.id
                            and getattr(
                                other.status, "value", str(other.status)
                            ) not in {"dead", "incapacitated"}
                            and (
                                (
                                    isinstance(
                                        getattr(
                                            other, "_active_excavation", None
                                        ),
                                        dict,
                                    )
                                    and (
                                        int(other._active_excavation.get(
                                            "x", other.x
                                        )),
                                        int(other._active_excavation.get(
                                            "y", other.y
                                        )),
                                    ) == detector_recovery_target
                                )
                                or (
                                    isinstance(other.action.target, dict)
                                    and (
                                        other.action.target.get("x"),
                                        other.action.target.get("y"),
                                    ) == detector_recovery_target
                                    and other.action.target.get(
                                        "destination"
                                    ) in {
                                        "extraction_face",
                                        "active_excavation_face",
                                    }
                                )
                            )
                            for other in self.agents
                        )

                    # Expedition cylinders are reusable hardware.  Draw a
                    # delivered full bottle only above the maintenance floor;
                    # otherwise use the physical O2 manifold and wait through
                    # its visible servicing action before departure.
                    if (
                        agent.inventory.items.get("oxygen_canisters", 0) < 1
                        and self.central_depot_inventory.get(
                            "oxygen_canisters", 0
                        ) > self.MIN_CENTRAL_MAINTENANCE_O2_CANISTERS
                    ):
                        self.central_depot_inventory["oxygen_canisters"] -= 1
                        agent.inventory.add_item("oxygen_canisters", 1)
                    provisioning = False
                    if agent.inventory.items.get("oxygen_canisters", 0) < 1:
                        provisioning = self._provision_expedition_spare(agent)

                    self._ensure_recovery_buddy_reservation(
                        agent,
                        requested_resource,
                        detector_recovery_target,
                    )
                    buddy_candidate = self._find_expedition_buddy(
                        agent,
                        target_distance,
                        require_spare_canister=False,
                    )
                    if (
                        buddy_candidate is not None
                        and buddy_candidate.inventory.items.get(
                            "oxygen_canisters", 0
                        ) < 1
                    ):
                        if self.central_depot_inventory.get(
                            "oxygen_canisters", 0
                        ) > self.MIN_CENTRAL_MAINTENANCE_O2_CANISTERS:
                            self.central_depot_inventory[
                                "oxygen_canisters"
                            ] -= 1
                            buddy_candidate.inventory.add_item(
                                "oxygen_canisters", 1
                            )
                        else:
                            provisioning = (
                                self._provision_expedition_spare(
                                    buddy_candidate
                                )
                                or provisioning
                            )

                    recovery_lead_ready = self._agent_ready_for_expedition(
                        agent, target_distance
                    )
                    ready_buddy = self._find_expedition_buddy(
                        agent, target_distance
                    )
                    eva_preflight_ready = False
                    if (
                        not provisioning
                        and not recovery_face_reserved
                        and recovery_lead_ready
                        and ready_buddy is not None
                    ):
                        lead_preflight = (
                            not agent._in_habitat
                            or self._prepare_agent_for_eva(
                                agent,
                                defer_pressure_transition=True,
                                target_position=detector_recovery_target,
                                rover_outbound=True,
                            )
                        )
                        buddy_preflight = False
                        if lead_preflight:
                            buddy_preflight = (
                                not ready_buddy._in_habitat
                                or self._prepare_agent_for_eva(
                                    ready_buddy,
                                    defer_pressure_transition=True,
                                    target_position=detector_recovery_target,
                                    rover_outbound=True,
                                )
                            )
                        eva_preflight_ready = bool(
                            lead_preflight and buddy_preflight
                        )
                        if eva_preflight_ready:
                            prelaunched_recovery_expedition = (
                                self._start_expedition(
                                    agent,
                                    requested_resource,
                                    detector_recovery_target,
                                    require_rover=True,
                                    expedition_kind="resource_recovery",
                                    capacity_recipe=capacity_recipe,
                                    recovery_goal_units=(
                                        recovery_goal_units or None
                                    ),
                                )
                            )

                    if prelaunched_recovery_expedition is None:
                        if not agent._in_habitat:
                            agent.enter_habitat()
                        preflight_action_pending = bool(
                            agent.action.action_type in {
                                "refill_o2", "service_suit"
                            }
                            or (
                                isinstance(agent.action.target, dict)
                                and agent.action.target.get("eva_denied")
                            )
                        )
                        if not preflight_action_pending:
                            agent.action.action_type = "rest"
                            agent.action.target = {
                                "expedition_preparation": True,
                                "detector_recovery": True,
                                "resource": requested_resource,
                                "target": list(detector_recovery_target),
                                "waiting_for_recovery_face": (
                                    recovery_face_reserved
                                ),
                                "waiting_for_crew_readiness": (
                                    not recovery_lead_ready
                                    or ready_buddy is None
                                ),
                                "waiting_for_eva_preflight": bool(
                                    recovery_lead_ready
                                    and ready_buddy is not None
                                    and not provisioning
                                    and not recovery_face_reserved
                                    and not eva_preflight_ready
                                ),
                            }
                            agent.action.ticks_remaining = 1
                        return agent_events

                raw_current_cell_resources = self.world.get_cell_info(
                    agent.x, agent.y, self.current_tick
                ).get("base_resources", {})
                current_cell_resources = {
                    resource: abundance
                    for resource, abundance in raw_current_cell_resources.items()
                    if resource == "regolith" or (
                        agent.x, agent.y
                    ) in self.discovered_resources.get(resource, set())
                    if (agent.x, agent.y, resource) not in self.depleted_cell_resources
                }
                local_spoil_qty = self.spoil_piles.get(
                    (agent.x, agent.y), {}
                ).get(requested_resource, 0)
                requires_resource_search = (
                    requested_resource != "regolith"
                    and requested_resource not in current_cell_resources
                    and local_spoil_qty <= 0
                )
                if is_protected_construction_ground and not requires_resource_search:
                    if isinstance(getattr(agent, "_active_excavation", None), dict):
                        agent._active_excavation = None
                    target_x, target_y = self._mining_staging_point(agent)
                    dx, dy = self._cardinal_step_toward(
                        agent, target_x, target_y
                    )
                    next_x = max(0, min(self.world.map_size - 1, agent.x + dx))
                    next_y = max(0, min(self.world.map_size - 1, agent.y + dy))
                    if (
                        agent._in_habitat
                        and not self._prepare_agent_for_eva(
                        agent,
                        mission_critical_maintenance=bool(
                            target.get("life_support_bootstrap")
                        ),
                        defer_pressure_transition=True,
                        target_position=(target_x, target_y),
                    )):
                        return agent_events
                    agent.x = next_x
                    agent.y = next_y
                    agent.action.action_type = "move"
                    agent.action.ticks_remaining = 1
                    agent.action.target = {
                        "x": target_x,
                        "y": target_y,
                        "destination": "extraction_zone",
                    }
                    logger.info(
                        f"{agent.name} moving to extraction zone ({target_x}, {target_y}); "
                        f"base buffer={self.MIN_EXTRACTION_RADIUS_CELLS} cells"
                    )
                else:
                    if (
                        agent._in_habitat
                        and not self._prepare_agent_for_eva(
                            agent,
                            mission_critical_maintenance=bool(
                                target.get("life_support_bootstrap")
                            ),
                            defer_pressure_transition=True,
                        )
                    ):
                        return agent_events
                    # Planetary Geology: Check if surface regolith on this cell has been excavated
                    if not hasattr(self, "revealed_cell_resources"):
                        self.revealed_cell_resources = set()
                    is_cell_excavated = (agent.x, agent.y) in self.revealed_cell_resources
                    cell_resources = current_cell_resources

                    # If the requested resource is not regolith and not present in this cell, navigate to matching deposit
                    if (
                        requested_resource != "regolith"
                        and (not cell_resources or requested_resource not in cell_resources)
                        and local_spoil_qty <= 0
                    ):
                        reserved_faces = set()
                        for other in self.agents:
                            if (
                                other.id == agent.id
                                or getattr(
                                    other.status,
                                    "value",
                                    str(other.status),
                                ) in {"dead", "incapacitated"}
                            ):
                                continue
                            active_face = getattr(other, "_active_excavation", None)
                            if (
                                isinstance(active_face, dict)
                                and active_face.get("resource") == requested_resource
                            ):
                                reserved_faces.add((
                                    int(active_face.get("x", other.x)),
                                    int(active_face.get("y", other.y)),
                                ))
                            other_target = (
                                other.action.target
                                if isinstance(other.action.target, dict) else {}
                            )
                            if (
                                other_target.get("resource") == requested_resource
                                and other_target.get("destination") in {
                                    "extraction_face", "active_excavation_face"
                                }
                                and other_target.get("x") is not None
                                and other_target.get("y") is not None
                            ):
                                reserved_faces.add((
                                    int(other_target["x"]),
                                    int(other_target["y"]),
                                ))
                            # Detector-confirmed distant seams belong to their
                            # persistent cargo-recovery contract.  Without
                            # this reservation, a later crew tick could start
                            # a normal walking excavation at the same face;
                            # that local marker then blocked the two-person
                            # rover preflight indefinitely.
                            recovery_contract = getattr(
                                other, "_detected_resource_recovery", None
                            )
                            if (
                                isinstance(recovery_contract, dict)
                                and recovery_contract.get("resource")
                                == requested_resource
                            ):
                                reserved_faces.add((
                                    int(recovery_contract.get("x", other.x)),
                                    int(recovery_contract.get("y", other.y)),
                                ))
                            recovery_expedition = getattr(
                                other, "_active_expedition", None
                            )
                            if (
                                isinstance(recovery_expedition, dict)
                                and recovery_expedition.get("kind")
                                == "resource_recovery"
                                and recovery_expedition.get("resource")
                                == requested_resource
                            ):
                                reserved_faces.add((
                                    int(recovery_expedition.get(
                                        "target_x", other.x
                                    )),
                                    int(recovery_expedition.get(
                                        "target_y", other.y
                                    )),
                                ))
                        # The owner must still see its own exact face while it
                        # executes the strict rover contract; only competing
                        # ordinary gatherers are masked from it.
                        if strict_rover_recovery:
                            reserved_faces.discard(detector_recovery_target)
                        known_deposits = {
                            coord for coord in getattr(self, "discovered_resources", {}).get(requested_resource, set())
                            if (coord[0], coord[1], requested_resource)
                            not in self.depleted_cell_resources
                            and coord not in reserved_faces
                            and not self._is_construction_protected_cell(*coord)
                            and self._distance_from_lz(coord[0], coord[1])
                            >= self.MIN_EXTRACTION_RADIUS_CELLS
                        } | {
                            coord for coord in self._spoil_coordinates(requested_resource)
                            if not self._is_construction_protected_cell(*coord)
                        }
                        discovered = {
                            coord for coord in known_deposits
                            if self._inside_eva_operating_area(coord[0], coord[1])
                            and self._distance_from_lz(coord[0], coord[1])
                            >= self.MIN_EXTRACTION_RADIUS_CELLS
                        }
                        if (
                            strict_rover_recovery
                            and detector_recovery_target in known_deposits
                        ):
                            # A 3x3 measurement footprint centred on the last
                            # local station may confirm a radius-25 cell.  The
                            # audited rover expedition authorizes that exact
                            # face even though routine walking EVA stops at 24.
                            discovered.add(detector_recovery_target)
                        if discovered:
                            closest = (
                                detector_recovery_target
                                if detector_recovery_target in discovered
                                else min(
                                    discovered,
                                    key=lambda c: (
                                        (c[0] - agent.x) ** 2
                                        + (c[1] - agent.y) ** 2
                                    ),
                                )
                            )
                            tx, ty = closest
                            expedition = prelaunched_recovery_expedition
                            target_distance_from_lz = self._distance_from_lz(
                                tx, ty
                            )
                            # A detector-confirmed face can justify transport,
                            # but the rover still needs two fit crew, audited
                            # return energy and a physical reservation.  If
                            # any condition fails the existing on-foot route
                            # remains available.
                            if (
                                expedition is None
                                and not strict_rover_recovery
                                and target_distance_from_lz
                                >= self.CREW_ROVER_MIN_DISTANCE_CELLS
                                and self._distance_from_lz(agent.x, agent.y) <= 2
                                and self._agent_ready_for_expedition(
                                    agent, target_distance_from_lz
                                )
                                and self._find_expedition_buddy(
                                    agent, target_distance_from_lz
                                ) is not None
                            ):
                                expedition = self._start_expedition(
                                    agent, requested_resource, (tx, ty)
                                )
                            if (
                                isinstance(expedition, dict)
                                and expedition.get("transport") == "crew_rover"
                            ):
                                # Vehicle mobilization/airlock transit belongs
                                # to the shared two-seat motor, not this
                                # gather action's individual pedestrian step.
                                agent.action.action_type = "move"
                                agent.action.target = {
                                    "x": tx, "y": ty,
                                    "resource": requested_resource,
                                    "destination": "extraction_face",
                                    "mission_action": "gather",
                                    "expedition": True,
                                    "transport": "crew_rover",
                                }
                                agent.action.ticks_remaining = 1
                                return agent_events
                            dist_to_target = max(abs(tx - agent.x), abs(ty - agent.y))
                            step_speed = (
                                int(expedition.get("move_speed_cells", 1))
                                if isinstance(expedition, dict) else
                                self.EVA_WALK_SPEED_CELLS
                            )
                            dx, dy = self._cardinal_step_toward(
                                agent, tx, ty, step_speed
                            )
                            next_x = max(
                                0,
                                min(self.world.map_size - 1, agent.x + dx),
                            )
                            next_y = max(
                                0,
                                min(self.world.map_size - 1, agent.y + dy),
                            )
                            if (
                                (dx or dy)
                                and agent._in_habitat
                                and not self._is_pressurized_location(
                                    next_x, next_y
                                )
                            ):
                                boundary_dx, boundary_dy = (
                                    self._cardinal_step_toward(
                                        agent, tx, ty, 1
                                    )
                                )
                                boundary_x = max(
                                    0,
                                    min(
                                        self.world.map_size - 1,
                                        agent.x + boundary_dx,
                                    ),
                                )
                                boundary_y = max(
                                    0,
                                    min(
                                        self.world.map_size - 1,
                                        agent.y + boundary_dy,
                                    ),
                                )
                                if self._is_pressurized_location(
                                    boundary_x, boundary_y
                                ):
                                    dx, dy = boundary_dx, boundary_dy
                                    next_x, next_y = boundary_x, boundary_y
                                elif not self._prepare_agent_for_eva(agent, target_position=(int(tx), int(ty))):
                                    return agent_events
                            previous_x, previous_y = agent.x, agent.y
                            agent.x = next_x
                            agent.y = next_y
                            if (
                                isinstance(expedition, dict)
                                and expedition.get("transport") == "crew_rover"
                                and expedition.get("role") == "lead"
                            ):
                                self.surface_fleet.record_crew_rover_movement(
                                    str(expedition.get("id")),
                                    previous_x,
                                    previous_y,
                                    agent.x,
                                    agent.y,
                                )
                            agent.action.action_type = "move"
                            agent.action.target = {
                                "x": tx,
                                "y": ty,
                                "resource": requested_resource,
                                "destination": "extraction_face",
                                "mission_action": "gather",
                                "expedition": bool(expedition),
                                "transport": (
                                    expedition.get("transport")
                                    if isinstance(expedition, dict) else "on_foot"
                                ),
                            }
                            agent.action.ticks_remaining = 1
                        else:
                            # No exact local deposit is known yet. Ordinary EVA
                            # must prospect cells; only a portable/research scan
                            # may reveal a remote coordinate without excavation.
                            found_coord = None
                            expedition = None
                            remote_coord = None
                            handled_remote_preparation = False
                            if not found_coord:
                                # Stage 2: science infrastructure may remotely
                                # survey a wider area. A surveyed deposit becomes
                                # a travel target only when comms and round-trip
                                # life-support budgets also permit an expedition.
                                pending = getattr(agent, "_pending_expedition", None)
                                if isinstance(pending, dict) and pending.get("resource") == requested_resource:
                                    remote_coord = (
                                        int(pending.get("target_x")),
                                        int(pending.get("target_y")),
                                    )
                                elif known_deposits:
                                    remote_coord = min(
                                        known_deposits,
                                        key=lambda coord: self._distance_from_lz(*coord),
                                    )
                                else:
                                    survey_radius = self._survey_radius_cells()
                                    if survey_radius > self.LOCAL_EVA_RADIUS_CELLS:
                                        remote_coord = self._find_resource_from_lz(
                                            requested_resource,
                                            self.LOCAL_EVA_RADIUS_CELLS + 1,
                                            survey_radius,
                                        )

                                if not remote_coord:
                                    self.remote_resource_requests.add(requested_resource)

                                if remote_coord:
                                    remote_distance = self._distance_from_lz(*remote_coord)
                                    agent._pending_expedition = {
                                        "resource": requested_resource,
                                        "target_x": remote_coord[0],
                                        "target_y": remote_coord[1],
                                        "distance": remote_distance,
                                    }
                                    if self._can_launch_expedition(
                                        agent,
                                        remote_distance,
                                        target=remote_coord,
                                        resource=requested_resource,
                                    ):
                                        expedition = self._start_expedition(
                                            agent, requested_resource, remote_coord
                                        )
                                        found_coord = remote_coord
                                        self.remote_resource_requests.discard(requested_resource)
                                    elif self._distance_from_lz(agent.x, agent.y) > 2:
                                        lz_x = getattr(self, "lz_x", agent.spawn_x)
                                        lz_y = getattr(self, "lz_y", agent.spawn_y)
                                        dx, dy = self._cardinal_step_toward(
                                            agent, lz_x, lz_y
                                        )
                                        agent.x += dx
                                        agent.y += dy
                                        agent.action.action_type = "move"
                                        agent.action.target = {
                                            "x": lz_x, "y": lz_y,
                                            "destination": "shelter",
                                            "expedition_preparation": True,
                                            "resource": requested_resource,
                                        }
                                        agent.action.ticks_remaining = 1
                                        handled_remote_preparation = True
                                    else:
                                        agent.enter_habitat()
                                        agent.action.action_type = "rest"
                                        agent.action.target = {
                                            "expedition_preparation": True,
                                            "resource": requested_resource,
                                        }
                                        agent.action.ticks_remaining = 1
                                        handled_remote_preparation = True

                            if handled_remote_preparation:
                                pass
                            elif found_coord:
                                tx, ty = found_coord
                                if (
                                    isinstance(expedition, dict)
                                    and expedition.get("transport") == "crew_rover"
                                ):
                                    agent.action.action_type = "move"
                                    agent.action.target = {
                                        "x": tx, "y": ty,
                                        "resource": requested_resource,
                                        "expedition": True,
                                        "transport": "crew_rover",
                                    }
                                    agent.action.ticks_remaining = 1
                                    return agent_events
                                step_speed = (
                                    int(expedition.get(
                                        "move_speed_cells",
                                        self.EXPEDITION_MOVE_SPEED_CELLS,
                                    )) if expedition else self.EVA_WALK_SPEED_CELLS
                                )
                                dx, dy = self._cardinal_step_toward(
                                    agent, tx, ty, step_speed
                                )
                                if dx or dy:
                                    previous_x, previous_y = agent.x, agent.y
                                    agent.x += dx
                                    agent.y += dy
                                    if (
                                        isinstance(expedition, dict)
                                        and expedition.get("transport") == "crew_rover"
                                        and expedition.get("role") == "lead"
                                    ):
                                        self.surface_fleet.record_crew_rover_movement(
                                            str(expedition.get("id")),
                                            previous_x,
                                            previous_y,
                                            agent.x,
                                            agent.y,
                                        )
                                agent.action.action_type = "move"
                                agent.action.target = {
                                    "x": tx, "y": ty,
                                    "resource": requested_resource,
                                    "expedition": bool(expedition),
                                }
                                agent.action.ticks_remaining = 1
                            else:
                                # Walk to one reserved geological test station;
                                # travel is MOVE and only the actual test pit is
                                # PROSPECT. This avoids excavating a continuous
                                # trench across every cell on the route.
                                station = self._next_hand_prospect_station(
                                    agent, requested_resource
                                )
                                if station is None:
                                    self.remote_resource_requests.add(requested_resource)
                                    if self._distance_from_lz(agent.x, agent.y) > 1:
                                        dx, dy = self._cardinal_step_toward(
                                            agent, self.lz_x, self.lz_y
                                        )
                                        agent.x += dx
                                        agent.y += dy
                                        agent.action.action_type = "move"
                                        agent.action.target = {
                                            "x": self.lz_x,
                                            "y": self.lz_y,
                                            "destination": "shelter",
                                            "survey_complete": True,
                                            "resource": requested_resource,
                                        }
                                    else:
                                        agent.enter_habitat()
                                        agent.action.action_type = "rest"
                                        agent.action.target = {
                                            "local_prospecting_complete": True,
                                            "resource": requested_resource,
                                        }
                                    agent.action.ticks_remaining = 1
                                elif (agent.x, agent.y) != station:
                                    best_dx, best_dy = self._cardinal_step_toward(
                                        agent, station[0], station[1]
                                    )
                                    agent.x = max(
                                        0,
                                        min(
                                            self.world.map_size - 1,
                                            agent.x + best_dx,
                                        ),
                                    )
                                    agent.y = max(
                                        0,
                                        min(
                                            self.world.map_size - 1,
                                            agent.y + best_dy,
                                        ),
                                    )
                                    agent.action.action_type = "move"
                                    agent.action.target = {
                                        "resource": requested_resource,
                                        "x": station[0],
                                        "y": station[1],
                                        "destination": "prospect_station",
                                        "survey_action": "hand_prospect",
                                    }
                                    agent.action.ticks_remaining = 1
                                else:
                                    newly_revealed = self._excavate_test_layer(
                                        agent, agent.x, agent.y
                                    )
                                    if (
                                        requested_resource != "regolith"
                                        and requested_resource not in newly_revealed
                                    ):
                                        misses = self._local_resource_miss_centers.setdefault(
                                            requested_resource, set()
                                        )
                                        misses.add(station)
                                        # Twelve spaced test pits are enough to
                                        # stop blind digging. Escalate the same
                                        # resource to the rechargeable detector;
                                        # this is a request for measured local
                                        # surveying, not magical discovery.
                                        if self._resource_local_campaign_exhausted(
                                            requested_resource
                                        ):
                                            self.remote_resource_requests.add(
                                                requested_resource
                                            )
                                    self._hand_prospect_centers.add(station)
                                    agent._hand_prospect_target = None
                                    agent.action.action_type = "prospect"
                                    agent.action.target = {
                                        "resource": requested_resource,
                                        "x": agent.x,
                                        "y": agent.y,
                                        "survey_action": "hand_prospect",
                                        "excavation_depth": self.cell_excavation_depth.get(
                                            (agent.x, agent.y), 0
                                        ),
                                        "newly_revealed": newly_revealed,
                                    }
                                    agent.action.ticks_remaining = 2
                    else:
                        # Vertical extraction never skips strata. If a sensor
                        # detected a deep target, every complete layer above it
                        # is removed first and staged as a recoverable spoil pile.
                        dn_eff = self.get_day_night_phase().get("gather_efficiency", 1.0)
                        geo_skill = max(
                            getattr(agent.competency, "engineering", 0),
                            getattr(agent.competency, "physics", 0),
                        )
                        extraction_rate = max(2, int((2 + geo_skill * 0.45) * dn_eff))
                        gravity_g = self.planet.physical.get("gravity_g", 1.0)
                        carry_cap_kg = (
                            agent.inventory.BASE_CARRY_CAPACITY_KG
                            * (1 + agent.genome.strength * 0.1)
                            / max(0.3, gravity_g)
                        )
                        # BASE_CARRY_CAPACITY_KG is the payload allowance on
                        # top of the assigned suit/PLSS and worn mission kit.
                        current_weight = sum(
                            quantity * MATERIAL_DENSITY_KG.get(material, 2.0)
                            for material, quantity in agent.inventory.materials.items()
                        )

                        # Previously excavated overburden remains useful and
                        # can be collected later instead of mining a new face.
                        pile = self.spoil_piles.get((agent.x, agent.y), {})
                        pile_available = int(pile.get(requested_resource, 0))
                        if pile_available > 0:
                            resource = requested_resource
                            resource_weight = MATERIAL_DENSITY_KG.get(resource, 2.0)
                            max_can_carry = max(
                                0, int((carry_cap_kg - current_weight) / resource_weight)
                            )
                            amount = min(extraction_rate, pile_available, max_can_carry)
                            if amount > 0:
                                agent.inventory.add_material(resource, amount)
                                pile[resource] -= amount
                                if pile[resource] <= 0:
                                    pile.pop(resource, None)
                                if not pile:
                                    self.spoil_piles.pop((agent.x, agent.y), None)
                                agent.action.action_type = "gather"
                                agent.action.target = {
                                    "x": agent.x,
                                    "y": agent.y,
                                    "resource": requested_resource,
                                    "amount": amount,
                                    "source": "spoil_pile",
                                }
                                agent.action.ticks_remaining = 2
                            else:
                                self._send_loaded_agent_to_base(agent, current_weight, carry_cap_kg)
                        else:
                            geology = self._get_cell_geology(agent.x, agent.y)
                            layer = self._current_geology_layer(agent.x, agent.y)
                            if layer is None:
                                self.remote_resource_requests.add(requested_resource)
                                agent.action.action_type = "prospect"
                                agent.action.target = {
                                    "resource": requested_resource,
                                    "cell_layers_exhausted": True,
                                }
                                agent.action.ticks_remaining = 2
                            else:
                                resource = layer["material"]
                                overburden = resource != requested_resource
                                resource_weight = MATERIAL_DENSITY_KG.get(resource, 2.0)
                                active_expedition = getattr(
                                    agent, "_active_expedition", None
                                )
                                rover = None
                                if (
                                    not overburden
                                    and isinstance(active_expedition, dict)
                                    and active_expedition.get("transport")
                                    == "crew_rover"
                                    and active_expedition.get("role") == "lead"
                                ):
                                    rover = self.surface_fleet.crew_rover_for_expedition(
                                        str(active_expedition.get("id"))
                                    )
                                if rover is not None:
                                    rover_payload_limit = float(
                                        self.mission_profile.surface_fleet[
                                            "crew_rover"
                                        ].get("payload_kg", 490.0)
                                    )
                                    max_can_carry = max(
                                        0,
                                        int(
                                            (
                                                rover_payload_limit
                                                - rover.payload_mass_kg
                                            )
                                            / resource_weight
                                        ),
                                    )
                                else:
                                    max_can_carry = max(
                                        0,
                                        int(
                                            (carry_cap_kg - current_weight)
                                            / resource_weight
                                        ),
                                    )
                                if not overburden and max_can_carry <= 0:
                                    if rover is not None:
                                        active_expedition["status"] = "returning"
                                        agent.action.action_type = "move"
                                        agent.action.target = {
                                            "x": self.lz_x,
                                            "y": self.lz_y,
                                            "destination": "shelter",
                                            "expedition": True,
                                            "rover_payload_full": True,
                                        }
                                        agent.action.ticks_remaining = 1
                                    else:
                                        self._send_loaded_agent_to_base(
                                            agent, current_weight, carry_cap_kg
                                        )
                                else:
                                    inventory_key = (
                                        "recoverable_remaining"
                                        if (
                                            not overburden
                                            and "recoverable_remaining" in layer
                                        )
                                        else "remaining"
                                    )
                                    amount = min(
                                        extraction_rate,
                                        int(layer.get(inventory_key, 0)),
                                    )
                                    if not overburden:
                                        amount = min(amount, max_can_carry)

                                    rover_load = None
                                    recovery_goal_reached = False
                                    if overburden:
                                        # Once a real working face is opened,
                                        # finish removing its overburden before
                                        # choosing another prospect station.
                                        active_face = {
                                            "x": agent.x,
                                            "y": agent.y,
                                            "resource": requested_resource,
                                        }
                                        capacity_recipe = (
                                            target.get("capacity_recipe")
                                            or (
                                                active_expedition.get(
                                                    "capacity_recipe"
                                                )
                                                if isinstance(
                                                    active_expedition, dict
                                                )
                                                else None
                                            )
                                            or (
                                                getattr(
                                                    agent,
                                                    "_detected_resource_recovery",
                                                    {},
                                                ).get("capacity_recipe")
                                                if isinstance(
                                                    getattr(
                                                        agent,
                                                        "_detected_resource_recovery",
                                                        None,
                                                    ),
                                                    dict,
                                                )
                                                else None
                                            )
                                            or getattr(
                                                self.decision_engine,
                                                "shared_work_order",
                                                {},
                                            ).get("recipe")
                                        )
                                        if capacity_recipe:
                                            active_face["capacity_recipe"] = (
                                                capacity_recipe
                                            )
                                        agent._active_excavation = active_face
                                        spoil_coord = self._add_to_spoil_pile(
                                            agent.x, agent.y, resource, amount
                                        )
                                        storage = "spoil_pile"
                                    else:
                                        if rover is not None:
                                            rover_load = (
                                                self.surface_fleet.load_crew_rover_payload(
                                                    str(active_expedition.get("id")),
                                                    resource,
                                                    amount,
                                                    resource_weight,
                                                )
                                            )
                                            amount = int(
                                                rover_load.get("loaded_units", 0)
                                            )
                                            storage = "crew_rover"
                                            recovery_goal = max(
                                                0,
                                                int(
                                                    active_expedition.get(
                                                        "recovery_goal_units",
                                                        0,
                                                    )
                                                ),
                                            )
                                            recovery_goal_reached = bool(
                                                recovery_goal
                                                and rover.payload.get(
                                                    requested_resource, 0
                                                ) >= recovery_goal
                                            )
                                        else:
                                            agent.inventory.add_material(
                                                resource, amount
                                            )
                                            storage = "inventory"

                                    layer[inventory_key] = max(
                                        0,
                                        int(layer.get(inventory_key, 0)) - amount,
                                    )
                                    self.cell_resource_units[
                                        (agent.x, agent.y, resource)
                                    ] = layer[inventory_key]
                                    agent.inventory.use_tool(1)

                                    newly_exposed = None
                                    # Depleting a lateral production reserve
                                    # does not pretend that the access trench
                                    # through this stratum has also vanished.
                                    layer_finished = (
                                        inventory_key == "remaining"
                                        and layer["remaining"] <= 0
                                    )
                                    if layer_finished:
                                        newly_exposed = self._advance_depleted_geology_layer(
                                            agent.x, agent.y
                                        )
                                        # Reaching the requested stratum is
                                        # the start of recovery, not completion
                                        # of the work face.  Keep the contract
                                        # until its BOM deficit is satisfied or
                                        # the target layer is physically spent.
                                        if self._on_event:
                                            next_name = (
                                                newly_exposed["material"]
                                                if newly_exposed else "bedrock floor"
                                            )
                                            self._on_event({
                                                "type": "geology_layer_exposed",
                                                "agent": agent.name,
                                                "cause": (
                                                    f"Finished {resource.replace('_', ' ')} layer at "
                                                    f"({agent.x}, {agent.y}); exposed {next_name.replace('_', ' ')}"
                                                ),
                                                "tick": self.current_tick,
                                            })

                                    if self._on_event:
                                        destination = (
                                            "the recoverable spoil pile"
                                            if overburden else
                                            "the crew rover cargo bed"
                                            if storage == "crew_rover" else
                                            "carried inventory"
                                        )
                                        self._on_event({
                                            "type": "gather",
                                            "agent": agent.name,
                                            "cause": (
                                                f"Excavated {amount}x {resource.replace('_', ' ')} "
                                                f"into {destination}"
                                            ),
                                            "tick": self.current_tick,
                                        })

                                    return_after_target = (
                                        (
                                            layer_finished
                                            or bool(
                                                rover_load
                                                and rover_load.get(
                                                    "payload_full", False
                                                )
                                            )
                                            or recovery_goal_reached
                                        )
                                        and not overburden
                                        and isinstance(
                                            getattr(agent, "_active_expedition", None), dict
                                        )
                                    )
                                    if return_after_target:
                                        agent._active_expedition["status"] = "returning"
                                        lz_x = getattr(self, "lz_x", agent.spawn_x)
                                        lz_y = getattr(self, "lz_y", agent.spawn_y)
                                        agent.action.action_type = "move"
                                        agent.action.target = {
                                            "x": lz_x,
                                            "y": lz_y,
                                            "destination": "shelter",
                                            "expedition": True,
                                            "resource": requested_resource,
                                        }
                                        agent.action.ticks_remaining = 1
                                    else:
                                        agent.action.action_type = "gather"
                                        agent.action.target = {
                                            "x": agent.x,
                                            "y": agent.y,
                                            "resource": requested_resource,
                                            "extracted_material": resource,
                                            "amount": amount,
                                            "stored_as": storage,
                                            "storage_location": (
                                                list(spoil_coord)
                                                if overburden else
                                                active_expedition.get("rover_id")
                                                if storage == "crew_rover" else None
                                            ),
                                            "layer_index": geology["current_index"],
                                        }
                                        agent.action.ticks_remaining = 3
            elif action == "borrow_tool":
                donor_id = target.get("donor_id")
                donor = next((
                    teammate for teammate in self.agents
                    if teammate.id == donor_id
                    and getattr(teammate.status, "value", str(teammate.status)) != "dead"
                    and max(abs(teammate.x - agent.x), abs(teammate.y - agent.y)) <= 1
                ), None)
                tool_name = donor.inventory.get_best_tool() if donor else None
                if donor and tool_name:
                    durability = donor.inventory.tool_durability.get(tool_name, 0)
                    charge = donor.inventory.tool_charge_pct.get(tool_name)
                    donor.inventory.remove_item(tool_name, 1)
                    agent.inventory.add_item(tool_name, 1)
                    agent.inventory.tool_durability[tool_name] = durability
                    if charge is not None:
                        agent.inventory.tool_charge_pct[tool_name] = charge
                    agent.action.action_type = "borrow_tool"
                    agent.action.target = {
                        "donor": donor.name,
                        "tool": tool_name,
                        "durability": durability,
                    }
                    agent.action.ticks_remaining = 1
                    if self._on_event:
                        self._on_event({
                            "type": "tool_transfer",
                            "agent": agent.name,
                            "cause": (
                                f"{donor.name} transferred shared {tool_name.replace('_', ' ')} "
                                f"to {agent.name} ({durability} durability)"
                            ),
                            "tick": self.current_tick,
                        })
                else:
                    agent.action.action_type = "tool_transfer_blocked"
                    agent.action.target = {"donor_id": donor_id}
                    agent.action.ticks_remaining = 1
            elif action == "craft_item":
                recipe_name = target.get("recipe", "")
                recipe = self._get_recipe(recipe_name)
                if (
                    not recipe
                    or recipe.get("field_constructible", True) is False
                    or recipe.get("output", {}).get("type") != "item"
                ):
                    agent.action.action_type = "recipe_unavailable"
                    agent.action.target = {"recipe": recipe_name}
                    agent.action.ticks_remaining = 1
                    return agent_events

                # Check tool/skill/atmosphere constraints before moving any
                # pooled material. This keeps a failed craft transaction
                # atomic and prevents the depot/backpack ping-pong loop.
                non_material_recipe = dict(recipe)
                non_material_recipe["materials"] = {}
                non_material_recipe["requires_tool"] = False
                preflight = agent.can_craft(non_material_recipe)
                if (
                    not preflight.get("can_craft", False)
                    or (
                        recipe.get("requires_atmosphere")
                        and not getattr(agent, "_has_atmosphere_context", True)
                    )
                ):
                    agent.action.action_type = "craft_blocked"
                    agent.action.target = {"recipe": recipe_name, "reason": preflight}
                    agent.action.ticks_remaining = 1
                    return agent_events

                depot = getattr(self, "central_depot_inventory", {})
                for material, needed in recipe.get("materials", {}).items():
                    current = agent.inventory.materials.get(material, 0)
                    available_depot = self._available_depot_quantity(material)
                    if current < needed and available_depot > 0:
                        take = min(needed - current, available_depot)
                        depot[material] -= take
                        agent.inventory.add_material(material, take)
                        current += take
                    if current < needed:
                        for teammate in self.agents:
                            if (
                                teammate.id != agent.id
                                and getattr(teammate.status, "value", str(teammate.status)) != "dead"
                            ):
                                available = teammate.inventory.materials.get(material, 0)
                                if available > 0:
                                    take = min(needed - current, available)
                                    teammate.inventory.remove_material(material, take)
                                    agent.inventory.add_material(material, take)
                                    current += take
                                    if current >= needed:
                                        break

                result = agent.start_crafting(recipe)
                if isinstance(result, dict) and result.get("started"):
                    output = recipe.get("output", {})
                    item_id = output.get("item_id")
                    quantity = int(output.get("quantity", 1))
                    if item_id and quantity > 0:
                        agent.inventory.add_item(item_id, quantity)
                    if self._on_event:
                        self._on_event({
                            "type": "craft_item",
                            "agent": agent.name,
                            "cause": f"Fabricated {quantity}x {item_id.replace('_', ' ').title()}",
                            "tick": self.current_tick,
                        })
                else:
                    agent.action.action_type = "craft_blocked"
                    agent.action.target = {
                        "recipe": recipe_name,
                        "reason": result.get("reason", {}) if isinstance(result, dict) else {},
                    }
                    agent.action.ticks_remaining = 1
            elif action == "plan_construction":
                recipe_name = target.get("recipe", "water_collector")
                recipe = self._get_recipe(recipe_name)
                if recipe and recipe.get("field_constructible", True) is not False:
                    # Planning and BOM preparation are not a physical building.
                    # The map receives a construction site only when the full
                    # BOM is committed and actual field work begins.
                    agent.action.action_type = "plan_construction"
                    agent.action.target = {
                        "recipe": recipe_name,
                        "planning_only": True,
                    }
                    agent.action.ticks_remaining = 1
                else:
                    agent.action.action_type = "rest"
                    agent.action.target = {"recipe": recipe_name, "recipe_unavailable": True}
                    agent.action.ticks_remaining = 1
            elif action == "build":
                recipe_name = target.get("recipe", "water_collector")
                life_support_bootstrap = bool(target.get("life_support_bootstrap"))
                critical_o2_bootstrap = (
                    recipe_name == "isru_o2_unit"
                    and self.structures_built.get("isru_o2_unit", 0) == 0
                )
                def prepare_build_eva(
                    destination_x: int, destination_y: int,
                    mission_target: tuple[int, int] | None = None,
                ) -> bool:
                    """Open the airlock only when the next work step is outside."""
                    if (
                        not agent._in_habitat
                        or self._is_pressurized_location(
                            destination_x, destination_y
                        )
                    ):
                        return True
                    return self._prepare_agent_for_eva(
                        agent,
                        target_position=mission_target or (destination_x, destination_y),
                        mission_critical_maintenance=(
                            critical_o2_bootstrap or life_support_bootstrap
                        ),
                        life_support_bootstrap=critical_o2_bootstrap,
                    )
                recipe = self._get_recipe(recipe_name)
                if recipe and recipe.get("field_constructible", True) is not False:
                    true_singletons = {"storage_crate"}
                    if recipe_name in true_singletons and self.structures_built.get(recipe_name, 0) > 0:
                        agent.action.action_type = "milestone_complete"
                        agent.action.target = {"recipe": recipe_name, "already_built": True}
                        agent.action.ticks_remaining = 1
                        return agent_events

                    required_structure = recipe.get("requires_structure")
                    if (
                        required_structure
                        and self._operational_structure_count(
                            str(required_structure)
                        ) <= 0
                    ):
                        agent.action.action_type = "craft_blocked"
                        agent.action.target = {
                            "recipe": recipe_name,
                            "reason": "missing_operational_structure",
                            "requires_structure": required_structure,
                        }
                        agent.action.ticks_remaining = 1
                        return agent_events

                    # Joining an existing construction site must not consume a
                    # second set of materials or create a duplicate structure.
                    requested_struct_id = target.get("struct_id")
                    active_site = next((
                        site for site in self.placed_structures
                        if site.get("under_construction", False)
                        and not site.get("destroyed", False)
                        and (
                            (requested_struct_id and site.get("id") == requested_struct_id)
                            or (not requested_struct_id and site.get("type") == recipe_name)
                        )
                    ), None)
                    if active_site:
                        if not active_site.get("materials_committed", True):
                            reservation = self._construction_cargo_reservations.get(
                                str(active_site.get("id")), {}
                            )
                            agent.action.action_type = "stage_materials"
                            agent.action.ticks_remaining = 1
                            agent.action.target = {
                                "recipe": recipe_name,
                                "struct_id": active_site.get("id"),
                                "cargo_delivery_pending": True,
                                "cargo_status": reservation.get(
                                    "status", "awaiting_dispatch"
                                ),
                                "trips_completed": reservation.get(
                                    "trips_completed", 0
                                ),
                                "trips_dispatched": reservation.get(
                                    "trips_dispatched", 0
                                ),
                            }
                            return agent_events
                        site_x = int(active_site.get("x", agent.x))
                        site_y = int(active_site.get("y", agent.y))
                        if (
                            not isinstance(getattr(agent, "_active_expedition", None), dict)
                            and self._start_construction_rover_trip(agent, active_site)
                        ):
                            return agent_events
                        distance_to_site = max(abs(agent.x - site_x), abs(agent.y - site_y))
                        if distance_to_site > 1:
                            dx, dy = self._cardinal_step_toward(
                                agent, site_x, site_y, 1
                            )
                            next_x = max(0, min(
                                self.world.map_size - 1, agent.x + dx
                            ))
                            next_y = max(0, min(
                                self.world.map_size - 1, agent.y + dy
                            ))
                            if not prepare_build_eva(next_x, next_y, (site_x, site_y)):
                                return agent_events
                            agent.x = next_x
                            agent.y = next_y
                            agent.action.action_type = "move"
                            agent.action.ticks_remaining = 1
                            agent.action.target = {
                                "recipe": recipe_name,
                                "struct_id": active_site.get("id"),
                                "x": site_x,
                                "y": site_y,
                                "destination": "construction_site",
                                "construction_route": True,
                            }
                            return agent_events
                        if not prepare_build_eva(site_x, site_y):
                            return agent_events
                        agent.action.action_type = "build"
                        agent.action.ticks_remaining = (
                            1 if isinstance(getattr(agent, "_active_expedition", None), dict)
                            and agent._active_expedition.get("kind") == "construction_support"
                            else max(1, int(active_site.get("ticks_remaining", 1)))
                        )
                        agent.action.target = {
                            "recipe": recipe_name,
                            "struct_id": active_site.get("id"),
                            "x": active_site.get("x"),
                            "y": active_site.get("y"),
                            "assisting": True,
                        }
                        return agent_events

                    # Tool, competency and environment constraints must be
                    # validated before the build transaction takes anything
                    # from the depot or teammates.
                    non_material_recipe = dict(recipe)
                    non_material_recipe["materials"] = {}
                    preflight = agent.can_craft(non_material_recipe)
                    atmosphere_blocked = (
                        recipe.get("requires_atmosphere")
                        and not getattr(agent, "_has_atmosphere_context", True)
                    )
                    if not preflight.get("can_craft", False) or atmosphere_blocked:
                        agent.action.action_type = "craft_blocked"
                        agent.action.target = {
                            "recipe": recipe_name,
                            "reason": (
                                "requires_atmosphere" if atmosphere_blocked else preflight
                            ),
                        }
                        agent.action.ticks_remaining = 1
                        return agent_events

                    # The lead must physically reach the selected compact site
                    # before materials are committed and ground is broken.
                    try:
                        planned_trip = getattr(agent, "_active_expedition", None)
                        if (
                            isinstance(planned_trip, dict)
                            and planned_trip.get("kind") == "construction_support"
                            and planned_trip.get("planned_site")
                        ):
                            place_x, place_y = planned_trip["target_x"], planned_trip["target_y"]
                        else:
                            place_x, place_y = self._select_structure_site(recipe_name)
                    except RuntimeError as exc:
                        # A saturated campus is a real planning constraint,
                        # not permission to overlap two structures.  Keep the
                        # BOM untouched and expose a stable blocked action so
                        # the policy can choose a different build priority.
                        self.decision_engine.mark_capacity_site_unavailable(recipe_name)
                        agent.action.action_type = "build_blocked"
                        agent.action.target = {
                            "recipe": recipe_name,
                            "reason": "no_safe_construction_site",
                            "detail": str(exc),
                        }
                        agent.action.ticks_remaining = 1
                        return agent_events
                    distance_to_site = max(abs(agent.x - place_x), abs(agent.y - place_y))
                    if (
                        not isinstance(getattr(agent, "_active_expedition", None), dict)
                        and self._start_construction_rover_trip(agent, {
                            "id": f"planned-{recipe_name}-{place_x}-{place_y}",
                            "type": recipe_name, "x": place_x, "y": place_y,
                            "planned": True,
                        })
                    ):
                        return agent_events
                    if distance_to_site > 1:
                        dx, dy = self._cardinal_step_toward(
                            agent, place_x, place_y, 1
                        )
                        next_x = max(0, min(
                            self.world.map_size - 1, agent.x + dx
                        ))
                        next_y = max(0, min(
                            self.world.map_size - 1, agent.y + dy
                        ))
                        if not prepare_build_eva(next_x, next_y, (place_x, place_y)):
                            return agent_events
                        agent.x = next_x
                        agent.y = next_y
                        agent.action.action_type = "move"
                        agent.action.ticks_remaining = 1
                        agent.action.target = {
                            "recipe": recipe_name,
                            "x": place_x,
                            "y": place_y,
                            "destination": "planned_construction_site",
                            "construction_route": True,
                        }
                        return agent_events

                    if not prepare_build_eva(place_x, place_y):
                        return agent_events

                    # Open a surveyed site and reserve the exact depot BOM.
                    # Physical assembly cannot start until rated cargo
                    # transporters have delivered every conserved unit.
                    req_mats = recipe.get("materials", {})
                    required_item = recipe.get("requires_item")
                    struct_id = f"struct_{len(self.placed_structures)+1}"
                    cargo_reservation = self._reserve_construction_cargo(
                        site_id=struct_id,
                        recipe_name=recipe_name,
                        requirements=req_mats,
                        site_x=place_x,
                        site_y=place_y,
                        required_item=(
                            str(required_item) if required_item else None
                        ),
                    )
                    if cargo_reservation.get("reserved"):
                        construction_trip = getattr(agent, "_active_expedition", None)
                        if (
                            isinstance(construction_trip, dict)
                            and construction_trip.get("kind") == "construction_support"
                        ):
                            for crew in self.agents:
                                crew_trip = getattr(crew, "_active_expedition", None)
                                if isinstance(crew_trip, dict) and crew_trip.get("id") == construction_trip["id"]:
                                    crew_trip["site_id"] = struct_id
                                    crew_trip["planned_site"] = False
                        construction = recipe.get("construction", {})
                        site_profile = self._structure_site_profile(recipe_name)
                        required_work_hours = float(construction.get(
                            "work_person_hours",
                            max(1.0, recipe.get("base_duration_ticks", 12) * self.SIM_HOURS_PER_TICK),
                        ))
                        build_ticks = max(1, math.ceil(
                            required_work_hours / self.SIM_HOURS_PER_TICK
                        ))
                        self.placed_structures.append({
                            "id": struct_id,
                            "type": recipe_name,
                            "x": place_x,
                            "y": place_y,
                            "health": 1.0,
                            "built_by": agent.name,
                            "built_tick": self.current_tick,
                            "under_construction": True,
                            "materials_committed": False,
                            "required_item_committed": (
                                str(required_item) if required_item else None
                            ),
                            "construction_phase": "cargo_staging",
                            "site_zone": site_profile.get(
                                "zone", "industrial_yard"
                            ),
                            "footprint_half_width_cells": int(
                                site_profile.get(
                                    "footprint_half_width_cells", 0
                                )
                            ),
                            "footprint_half_height_cells": int(
                                site_profile.get(
                                    "footprint_half_height_cells", 0
                                )
                            ),
                            "render_scale": float(
                                site_profile.get("render_scale", 0.66)
                            ),
                            "cargo_manifest_mass_kg": round(
                                float(cargo_reservation.get(
                                    "total_mass_kg", 0.0
                                )), 3
                            ),
                            "materials_delivered_kg": 0.0,
                            "cargo_trips_completed": 0,
                            "progress": 0.0,
                            "total_ticks": build_ticks,
                            "ticks_remaining": build_ticks,
                            "required_work_hours": required_work_hours,
                            "original_required_work_hours": required_work_hours,
                            "work_hours_completed": 0.0,
                            "acceptance_attempts": 0,
                            "quality_hold": False,
                            "recommended_crew": int(construction.get("recommended_crew", 2)),
                            "active_builder_count": 1,
                            "dust_fouling_level": 0.0,
                        })
                        
                        # Building wears the best available tool three times as
                        # heavily as one gather operation. Inventory owns the
                        # entire break/removal/salvage transaction so neither a
                        # durability key nor the broken tool's mass is lost.
                        tool_use = agent.inventory.use_tool(uses=3)
                        if tool_use.get("broke"):
                            logger.warning(
                                "%s's %s BROKE during construction!",
                                agent.name,
                                tool_use.get("tool_used", "tool"),
                            )
                        
                        if self._on_event:
                            self._on_event({
                                "type": "construction_site_opened",
                                "agent": agent.name,
                                "cause": (
                                    f"Surveyed {recipe_name.replace('_', ' ').title()} site and "
                                    f"reserved {float(cargo_reservation.get('total_mass_kg', 0.0)):.1f} kg "
                                    "for physical transporter delivery"
                                ),
                                "tick": self.current_tick
                            })
                        agent.action.action_type = "stage_materials"
                        agent.action.ticks_remaining = 1
                        agent.action.ticks_elapsed = 0
                        agent.action.target = {
                            "recipe": recipe_name,
                            "struct_id": struct_id,
                            "total_ticks": build_ticks,
                            "x": place_x,
                            "y": place_y,
                            "cargo_delivery_pending": True,
                        }
                    else:
                        agent.action.action_type = "craft_blocked"
                        agent.action.target = {
                            "recipe": recipe_name,
                            "reason": cargo_reservation.get(
                                "reason", "construction_cargo_unavailable"
                            ),
                            "missing": cargo_reservation.get("missing", {}),
                            "oversized": cargo_reservation.get(
                                "oversized", {}
                            ),
                            "x": place_x,
                            "y": place_y,
                        }
                        agent.action.ticks_remaining = 1
            elif action == "refill_o2":
                refill_result = self._execute_o2_refill_action(agent)
                if refill_result.get("rover_returning"):
                    agent.action.action_type = "move"
                    agent.action.target = {
                        "x": refill_result["return_x"],
                        "y": refill_result["return_y"],
                        "destination": "shelter",
                        "expedition": True,
                        "transport": "crew_rover",
                        "o2_service_recall": True,
                    }
                    agent.action.ticks_remaining = 1
                elif refill_result.get("moving"):
                    agent.action.action_type = "move"
                    agent.action.target = {
                        "x": refill_result["station_x"],
                        "y": refill_result["station_y"],
                        "destination": "o2_filling_station",
                        "o2_service": True,
                    }
                    agent.action.ticks_remaining = 1
                elif refill_result.get("refilled"):
                    agent.action.action_type = "refill_o2"
                    agent.action.target = dict(refill_result)
                    agent.action.ticks_remaining = 3
                else:
                    agent.action.action_type = "idle"
                    agent.action.target = {
                        "eva_denied": refill_result.get(
                            "reason", "o2_refill_unavailable"
                        )
                    }
                    agent.action.ticks_remaining = 1
                if self._on_event and refill_result.get("refilled"):
                    self._on_event({
                        "type": "vital",
                        "agent": agent.name,
                        "cause": (
                            f"{agent.name} serviced PLSS O2 from "
                            f"{refill_result.get('source', 'canister')} "
                            f"({agent.needs.o2_supply:.0f}%)"
                        ),
                        "tick": self.current_tick
                    })
            elif action == "treat_injury":
                agent.treat_injury()
                agent.action.action_type = "treat_injury"
                agent.action.ticks_remaining = 4
            elif action == "treat":
                # Treatment effects happen here so every improvement has a
                # physical source.  Decision selection must never mutate a
                # patient's vital signs by itself.
                patient_name = target.get("patient", "unknown")
                patient_id = target.get("patient_id")
                patient = next((
                    crew for crew in self.agents
                    if (patient_id and crew.id == patient_id)
                    or (not patient_id and crew.name == patient_name)
                ), None)
                protocol = target.get("protocol")
                if patient is not None and max(abs(agent.x-patient.x), abs(agent.y-patient.y)) > 2:
                    agent.action.action_type = "move"
                    agent.action.target = {"x": patient.x, "y": patient.y, "medical_response": True}
                    agent.action.ticks_remaining = 1
                    return agent_events
                applied = False
                source = None
                clinical_details = {}
                medical_skill = float(getattr(
                    agent.competency, "medical", 0
                ))
                is_medic = (
                    medical_skill >= 7
                    or "medic" in getattr(agent, "role", "").lower()
                )
                # Old saves used one ambiguous protocol name. Resolve it using
                # the patient's actual airway/consciousness state, then expose
                # the explicit route on the resulting action telemetry.
                if protocol == "rehydration" and patient is not None:
                    protocol = (
                        "oral_rehydration"
                        if self._patient_can_take_oral_fluids(patient)
                        else "iv_io_rehydration"
                    )
                qualified = (
                    medical_skill >= 4
                    if protocol in {
                        "emergency_oxygen",
                        "rewarming",
                        "oral_rehydration",
                        "iv_io_rehydration",
                        "nutrition_support",
                    }
                    else is_medic
                )
                if (
                    patient is not None
                    and qualified
                    and protocol == "emergency_oxygen"
                ):
                    dose_kg = 0.10  # roughly one five-minute high-flow session
                    reserve = self._colony_resources.get("o2_reserve_kg", 0.0)
                    if (
                        agent._in_habitat
                        and patient._in_habitat
                        and reserve >= dose_kg
                    ):
                        self._colony_resources["o2_reserve_kg"] -= dose_kg
                        source = "habitat_o2_reserve"
                        applied = True
                    else:
                        donor = next((
                            crew for crew in (agent, patient)
                            if crew.inventory.has_item("oxygen_canisters")
                        ), None)
                        if donor is not None:
                            donor.inventory.remove_item("oxygen_canisters", 1)
                            if not patient._in_habitat:
                                # Install the real donated cylinder in the
                                # patient's PLSS. A saturation-only boost is
                                # overwritten by the unchanged empty suit
                                # cylinder on the next physical tick.
                                patient.inventory.items["oxygen_canisters"] = (
                                    patient.inventory.items.get("oxygen_canisters", 0) + 1
                                )
                                patient.load_o2_canister()
                                clinical_details["delivery"] = "plss_canister_replacement"
                            else:
                                donor.inventory.add_item("empty_oxygen_canisters", 1)
                            source = f"{donor.id}_canister"
                            applied = True
                    if applied:
                        patient.needs.o2_supply = min(
                            100.0, patient.needs.o2_supply + 50.0
                        )
                elif (
                    patient is not None
                    and qualified
                    and protocol == "rewarming"
                ):
                    has_blanket = (
                        agent.inventory.has_item("emergency_blanket")
                        or patient.inventory.has_item("emergency_blanket")
                    )
                    if has_blanket:
                        # Insulation limits further loss; warm habitat
                        # physiology performs the gradual rewarming.
                        patient.needs.temperature_stress = min(
                            100.0, patient.needs.temperature_stress + 5.0
                        )
                        source = "emergency_blanket"
                        applied = True
                elif (
                    patient is not None
                    and qualified
                    and protocol == "oral_rehydration"
                ):
                    if (
                        agent._in_habitat
                        and patient._in_habitat
                        and self._patient_can_take_oral_fluids(patient)
                    ):
                        dose_l, source = self._consume_oral_hydration_source(
                            patient
                        )
                        if dose_l > 0.0:
                            self._apply_hydration_dose(patient, dose_l)
                            clinical_details = {
                                "hydration_route": "oral",
                                "dose_l": dose_l,
                                "supplies_consumed": [],
                            }
                            applied = True
                elif (
                    patient is not None
                    and qualified
                    and protocol == "iv_io_rehydration"
                ):
                    fluid_stocks = self._medical_item_stocks(agent, patient, "sterile_iv_fluid_bags")
                    access_stocks = self._medical_item_stocks(agent, patient, "iv_io_administration_sets")
                    if (
                        agent._in_habitat
                        and patient._in_habitat
                        and not self._patient_can_take_oral_fluids(patient)
                        and fluid_stocks
                        and access_stocks
                    ):
                        fluid_stocks[0]["sterile_iv_fluid_bags"] -= 1
                        access_stocks[0]["iv_io_administration_sets"] -= 1
                        dose_l = 1.0
                        self._apply_hydration_dose(patient, dose_l)
                        source = (
                            "sterile_isotonic_crystalloid_bag_and_"
                            "single_use_iv_io_set"
                        )
                        clinical_details = {
                            "hydration_route": "iv_io",
                            "dose_l": dose_l,
                            "supplies_consumed": [
                                "sterile_iv_fluid_bags",
                                "iv_io_administration_sets",
                            ],
                        }
                        applied = True
                elif (
                    patient is not None
                    and qualified
                    and protocol == "nutrition_support"
                ):
                    if (
                        agent._in_habitat
                        and patient._in_habitat
                    ):
                        dose_kcal, source = (
                            self._consume_nutrition_source(patient)
                        )
                        if dose_kcal > 0.0:
                            self._apply_nutrition_dose(
                                patient, dose_kcal
                            )
                            clinical_details = {
                                "nutrition_route": "assisted_enteral",
                                "dose_kcal": dose_kcal,
                                "supplies_consumed": [source],
                            }
                            applied = True
                elif (
                    patient is not None
                    and qualified
                    and protocol == "injury_care"
                ):
                    medical_stocks = self._medical_item_stocks(
                        agent, patient, "medical_supplies"
                    )
                    if medical_stocks:
                        medical_stocks[0]["medical_supplies"] -= 1
                        if medical_stocks[0]["medical_supplies"] <= 0:
                            medical_stocks[0].pop("medical_supplies", None)
                        patient.injury_level = max(
                            0.0, patient.injury_level - 0.4
                        )
                        source = "medical_supplies"
                        applied = True

                agent.action.action_type = "treat" if applied else "idle"
                agent.action.target = {
                    **target,
                    "protocol": protocol,
                    "treatment_applied": applied,
                    "source": source,
                    **clinical_details,
                }
                agent.action.ticks_remaining = 3 if applied else 1
                if self._on_event:
                    self._on_event({
                        "type": "medical",
                        "agent": agent.name,
                        "patient_id": patient_id,
                        "protocol": protocol,
                        "applied": applied,
                        "source": source,
                        "cause": (
                            f"{agent.name} treated {patient_name} using {source}"
                            if applied else
                            f"{agent.name} could not treat {patient_name}: required resources unavailable"
                        ),
                        "tick": self.current_tick
                    })
            elif action == "wash":
                wash_result = agent.wash()
                if not wash_result.get("washed") and agent._in_habitat:
                    depot = getattr(self, "central_depot_inventory", {})
                    if depot.get("water_packs", 0) >= 2:
                        depot["water_packs"] -= 2
                        agent.needs.hygiene = min(90.0, agent.needs.hygiene + 50.0)
                        wash_result = {"washed": True, "water_used": 2.0}
                    elif self._colony_resources.get("water_reserve_l", 0.0) >= 2.0:
                        self._colony_resources["water_reserve_l"] -= 2.0
                        agent.needs.hygiene = min(90.0, agent.needs.hygiene + 50.0)
                        wash_result = {"washed": True, "water_used": 2.0}
                agent.action.action_type = "wash" if wash_result.get("washed") else "idle"
                agent.action.target = {**dict(target), **dict(wash_result)}
                agent.action.ticks_remaining = 2
            elif action == "repair":
                struct_name = target.get("structure", target.get("recipe", ""))
                inspection_only = bool(target.get("inspection_only", False))
                placed_candidates = [
                    struct for struct in getattr(self, "placed_structures", [])
                    if struct.get("type") == struct_name
                    and not struct.get("under_construction", False)
                    and not struct.get("destroyed", False)
                ]
                requested_id = target.get("structure_id")
                if requested_id:
                    requested = [
                        struct for struct in placed_candidates
                        if struct.get("id") == requested_id
                    ]
                    if requested:
                        placed_candidates = requested
                repair_target = min(
                    placed_candidates,
                    key=lambda struct: (
                        max(
                            abs(agent.x - struct.get("x", agent.x)),
                            abs(agent.y - struct.get("y", agent.y)),
                        ),
                        struct.get("health", 1.0),
                    ),
                    default=None,
                )
                if repair_target is not None:
                    sx = int(repair_target.get("x", agent.x))
                    sy = int(repair_target.get("y", agent.y))
                    distance = max(abs(agent.x - sx), abs(agent.y - sy))
                    if distance > 1:
                        dx, dy = self._cardinal_step_toward(
                            agent, sx, sy, self.EVA_WALK_SPEED_CELLS
                        )
                        next_x = agent.x + dx
                        next_y = agent.y + dy
                        if (
                            agent._in_habitat
                            and not self._is_pressurized_location(next_x, next_y)
                            and not self._prepare_agent_for_eva(
                                agent, mission_critical_maintenance=True,
                                target_position=(int(sx), int(sy)),
                            )
                        ):
                            return agent_events
                        agent.x = max(0, min(self.world.map_size - 1, next_x))
                        agent.y = max(0, min(self.world.map_size - 1, next_y))
                        agent.action.action_type = "move"
                        agent.action.target = {
                            "x": sx,
                            "y": sy,
                            "structure": struct_name,
                            "structure_id": repair_target.get("id"),
                            "maintenance": "external_structure_repair",
                            "maintenance_action": (
                                "inspection" if inspection_only else "repair"
                            ),
                            "inspection_only": inspection_only,
                        }
                        agent.action.ticks_remaining = 1
                        return agent_events

                requested_id = repair_target.get("id") if repair_target else None
                if struct_name and self.repair_structure(
                    agent,
                    struct_name,
                    requested_id,
                    inspection_only=inspection_only,
                ):
                    agent.action.action_type = "repair"
                    agent.action.target = {
                        "structure": struct_name,
                        "structure_id": (
                            repair_target.get("id") if repair_target else None
                        ),
                        "maintenance": "external_structure_repair",
                        "maintenance_action": (
                            "inspection" if inspection_only else "repair"
                        ),
                        "inspection_only": inspection_only,
                    }
                    agent.action.ticks_remaining = 3 if inspection_only else 6
                    if self._on_event:
                        health_pct = (
                            repair_target.get("health", 1.0)
                            if repair_target is not None
                            else self.structure_health.get(struct_name, 1.0)
                        )
                        self._on_event({
                            "type": "inspection" if inspection_only else "repair",
                            "agent": agent.name,
                            "cause": (
                                f"{agent.name} inspected "
                                f"{struct_name.replace('_', ' ').title()} at "
                                f"{health_pct:.0%} integrity"
                                if inspection_only else
                                f"{agent.name} repaired "
                                f"{struct_name.replace('_', ' ').title()} to "
                                f"{health_pct:.0%}"
                            ),
                            "tick": self.current_tick,
                        })
                else:
                    # A repair may lose a stock race after the decision was
                    # made (for example, construction consumed the same spare
                    # part first).  Do not leave the previous MOVE/REPAIR
                    # action displayed as if work were still progressing.
                    agent.action.action_type = "idle"
                    agent.action.target = {
                        "structure": struct_name,
                        "structure_id": requested_id,
                        "reason": "repair_materials_unavailable",
                    }
                    agent.action.ticks_remaining = 1
            elif action == "move":
                if self._move_construction_rover_team(agent, target):
                    return agent_events
                if self._move_resource_rover_team(agent, target):
                    return agent_events
                active_expedition = getattr(agent, "_active_expedition", None)
                if isinstance(active_expedition, dict):
                    target = dict(target)
                    home_x = getattr(self, "lz_x", getattr(agent, "spawn_x", 1000))
                    home_y = getattr(self, "lz_y", getattr(agent, "spawn_y", 1000))
                    is_home_route = (
                        target.get("destination") in ("habitat", "shelter")
                        or (target.get("x"), target.get("y")) == (home_x, home_y)
                    )
                    if is_home_route:
                        active_expedition["status"] = "returning"
                        target["expedition"] = True
                dx = target.get("dx")
                dy = target.get("dy")
                destination = target.get("destination")
                tx = target.get("x")
                ty = target.get("y")

                # Revalidate a committed outbound route every tick, not only
                # at the airlock. A work site's BOM, machine owner, resource
                # need or crew capacity may change while somebody is walking.
                # Homeward, expedition and maintenance routes have their own
                # safety/ownership rules and are intentionally excluded.
                if (
                    destination not in {"habitat", "shelter"}
                    and not target.get("expedition")
                    and not target.get("maintenance")
                ):
                    rejection = self._outbound_work_order_rejection(
                        agent, target
                    )
                    if rejection:
                        self._reject_invalid_work_order(
                            agent, target, rejection
                        )
                        return agent_events
                
                if destination in ("habitat", "shelter"):
                    lz_x = getattr(self, "lz_x", getattr(agent, "spawn_x", 1000))
                    lz_y = getattr(self, "lz_y", getattr(agent, "spawn_y", 1000))
                    main_airlock = self._lander_airlock_position()
                    tx, ty = main_airlock
                    explicit_lz_return = (
                        target.get("x"), target.get("y")
                    ) in {(lz_x, lz_y), main_airlock}
                    expedition_return = isinstance(active_expedition, dict)
                    if not (explicit_lz_return or expedition_return):
                        for s in getattr(self, "placed_structures", []):
                            if (
                                s.get("type") == "habitat_module"
                                and not s.get("under_construction", False)
                                and not s.get("destroyed", False)
                            ):
                                sx, sy = s.get("x", lz_x), s.get("y", lz_y)
                                if max(
                                    abs(agent.x - sx), abs(agent.y - sy)
                                ) < max(
                                    abs(agent.x - tx), abs(agent.y - ty)
                                ):
                                    tx, ty = sx, sy
                # Most tactical decisions provide an absolute target. The old
                # executor ignored x/y unless dx/dy was also present, producing
                # hundreds of stationary "move" ticks.
                if tx is not None or ty is not None:
                    tx = agent.x if tx is None else int(tx)
                    ty = agent.y if ty is None else int(ty)
                    allowed_radius = self._agent_eva_radius_cells(agent)
                    if self._distance_from_lz(tx, ty) > allowed_radius:
                        rejected_target = (tx, ty)
                        rejected_destination = destination
                        rejected_resource = target.get("resource")
                        active_face = getattr(agent, "_active_excavation", None)
                        if (
                            rejected_destination == "active_excavation_face"
                            and isinstance(active_face, dict)
                            and (
                                int(active_face.get("x", agent.x)),
                                int(active_face.get("y", agent.y)),
                            ) == rejected_target
                        ):
                            # An interrupted expedition can leave a partially
                            # opened face just outside the routine EVA radius.
                            # Keeping that face mission-locked made the planner
                            # order the same impossible trip every tick while
                            # the motor correctly returned the astronaut home.
                            # Release the local contract and let the normal
                            # regional survey/expedition system recover it.
                            agent._active_excavation = None
                            if rejected_resource:
                                self.remote_resource_requests.add(
                                    rejected_resource
                                )
                        rejected_contract = getattr(
                            agent, "_shared_work_contract", None
                        )
                        if (
                            rejected_resource
                            and isinstance(rejected_contract, dict)
                            and str(rejected_contract.get("action", "")).lower()
                            == "gather"
                            and rejected_contract.get("stock_key")
                            == rejected_resource
                        ):
                            self.decision_engine._close_bounded_gather_contract(
                                agent, rejected_contract
                            )
                        for marker_name in (
                            "_hand_prospect_target",
                            "_detected_resource_recovery",
                        ):
                            marker = getattr(agent, marker_name, None)
                            if (
                                isinstance(marker, dict)
                                and marker.get("resource") == rejected_resource
                                and (
                                    int(marker.get("x", rejected_target[0])),
                                    int(marker.get("y", rejected_target[1])),
                                ) == rejected_target
                            ):
                                setattr(agent, marker_name, None)
                        lz_x = getattr(self, "lz_x", getattr(agent, "spawn_x", 1000))
                        lz_y = getattr(self, "lz_y", getattr(agent, "spawn_y", 1000))
                        tx, ty = lz_x, lz_y
                        destination = "shelter"
                        target = {
                            "x": tx, "y": ty, "destination": destination,
                            "forced_return": True,
                            "cancelled_unreachable_target": list(rejected_target),
                            "cancelled_destination": rejected_destination,
                            "resource": rejected_resource,
                        }
                    route_leaves_lander = bool(
                        agent._in_habitat
                        and self._is_lander_footprint_cell(
                            agent.x, agent.y
                        )
                        and not self._is_lander_footprint_cell(tx, ty)
                        and destination not in {"habitat", "shelter"}
                    )
                    if route_leaves_lander:
                        authorized_expedition = bool(target.get("expedition"))
                        maintenance = bool(target.get("maintenance"))
                        if not authorized_expedition and not maintenance:
                            target_radius = self._distance_from_lz(tx, ty)
                            outbound_cells = max(
                                abs(int(tx) - agent.x),
                                abs(int(ty) - agent.y),
                            )
                            required_energy = (
                                self._routine_eva_start_energy_threshold(
                                    agent, outbound_cells, target_radius
                                )
                            )
                            if agent.needs.energy < required_energy:
                                recovery_ticks = ticks_for_minutes(
                                    120.0, self.SIM_MINUTES_PER_TICK
                                )
                                agent.action.action_type = "sleep"
                                agent.action.target = {
                                    "eva_denied": "insufficient_round_trip_energy",
                                    "target_radius_cells": target_radius,
                                    "required_energy_pct": round(
                                        required_energy, 1
                                    ),
                                    "habitat": True,
                                    "ticks": recovery_ticks,
                                    "preflight_recovery": True,
                                    "resume_destination": destination,
                                    "resume_x": int(tx),
                                    "resume_y": int(ty),
                                }
                                agent.action.ticks_remaining = recovery_ticks
                                self._route_queued_indoor_recovery(agent)
                                return agent_events
                        if not self._prepare_agent_for_eva(
                            agent,
                            mission_critical_maintenance=maintenance,
                            defer_pressure_transition=True,
                            target_position=(int(tx), int(ty)) if tx is not None and ty is not None else None,
                        ):
                            return agent_events
                    move_speed = (
                        int(active_expedition.get(
                            "move_speed_cells", self.EXPEDITION_MOVE_SPEED_CELLS
                        ))
                        if target.get("expedition")
                        and isinstance(active_expedition, dict)
                        else self.EVA_WALK_SPEED_CELLS
                    )
                    if not agent._in_habitat:
                        terrain_cost = max(
                            1.0, float(cell_info.get("traversal_cost", 1.0))
                        )
                        terrain_cost *= max(1.0, float(
                            event_effects.get("traversal_cost_multiplier", 1.0)
                        ))
                        visibility = max(
                            0.1,
                            min(
                                1.0,
                                float(cell_hazards.get(
                                    "visibility_range_modifier", 1.0
                                )) * float(cell_info.get(
                                    "visibility_factor", 1.0
                                )),
                            ),
                        )
                        terrain_cost *= 1.0 + (1.0 - visibility) * 0.75
                        in_rover = bool(
                            isinstance(active_expedition, dict)
                            and active_expedition.get("transport")
                            == "crew_rover"
                        )
                        crew_speed = (
                            1.0 if in_rover else agent.movement_speed(
                                float(self.planet.gravity_g)
                            )
                        )
                        if not in_rover:
                            target = dict(target)
                            effort_mode = self._movement_effort_mode(
                                agent,
                                target,
                                tx,
                                ty,
                                move_speed,
                                terrain_cost,
                            )
                            target["effort_mode"] = effort_mode
                            if effort_mode == "surge":
                                target["effort_multiplier"] = (
                                    self.MOVEMENT_SURGE_METABOLIC_MULTIPLIER
                                )
                                move_speed = max(
                                    move_speed + 1,
                                    int(math.ceil(
                                        move_speed
                                        * self.MOVEMENT_SURGE_SPEED_MULTIPLIER
                                    )),
                                )
                        movement_credit = min(
                            float(move_speed),
                            float(getattr(
                                agent, "_movement_cell_credit", 0.0
                            )) + move_speed * crew_speed / terrain_cost,
                        )
                        # One 100 m cell per ten-minute tick is the minimum
                        # representable motion. Sub-cell waiting looked like a
                        # stuck MOVE loop in the visualization; terrain still
                        # reduces a nominal two-cell stride to this lower pace.
                        allowed_step = max(
                            1,
                            min(move_speed, int(math.floor(movement_credit))),
                        )
                        agent._movement_cell_credit = max(
                            0.0, movement_credit - allowed_step
                        )
                    else:
                        agent._movement_cell_credit = 0.0
                        allowed_step = move_speed
                    if allowed_step > 0:
                        dx, dy = self._cardinal_step_toward(
                            agent, tx, ty, allowed_step
                        )
                    else:
                        dx, dy = 0, 0
                else:
                    dx = max(-1, min(1, int(dx or 0)))
                    dy = max(-1, min(1, int(dy or 0)))
                    if dx and dy:
                        # Coordinates are centres of 100 m parcels. A one-tick
                        # diagonal would cut a parcel corner while charging
                        # only one cardinal leg, so legacy relative orders are
                        # reduced to one deterministic orthogonal step.
                        if (self.current_tick + sum(map(ord, agent.id))) % 2:
                            dx = 0
                        else:
                            dy = 0

                next_x = max(0, min(self.world.map_size - 1, agent.x + (dx or 0)))
                next_y = max(0, min(self.world.map_size - 1, agent.y + (dy or 0)))
                if (
                    (dx or dy)
                    and agent._in_habitat
                    and not self._is_pressurized_location(next_x, next_y)
                ):
                    # A multi-cell ten-minute walking step must not jump over
                    # the pressurized footprint and silently combine indoor
                    # transit with an airlock exit. Stop at the final indoor
                    # grid first; the following tick performs EVA preflight.
                    boundary_dx, boundary_dy = self._cardinal_step_toward(
                        agent, tx, ty, 1
                    )
                    boundary_x = max(
                        0,
                        min(self.world.map_size - 1, agent.x + boundary_dx),
                    )
                    boundary_y = max(
                        0,
                        min(self.world.map_size - 1, agent.y + boundary_dy),
                    )
                    if self._is_pressurized_location(boundary_x, boundary_y):
                        dx, dy = boundary_dx, boundary_dy
                        next_x, next_y = boundary_x, boundary_y
                if (
                    (dx or dy)
                    and agent._in_habitat
                    and not self._is_pressurized_location(next_x, next_y)
                ):
                    homeward = destination in {"habitat", "shelter"}
                    authorized_expedition = bool(target.get("expedition"))
                    maintenance = bool(target.get("maintenance"))
                    if not homeward and not authorized_expedition:
                        rejection = self._outbound_work_order_rejection(
                            agent, target
                        )
                        if rejection:
                            self._reject_invalid_work_order(
                                agent, target, rejection
                            )
                            return agent_events
                    if (
                        not homeward
                        and not authorized_expedition
                        and not maintenance
                        and tx is not None and ty is not None
                    ):
                        target_radius = self._distance_from_lz(tx, ty)
                        outbound_cells = max(
                            abs(int(tx) - agent.x), abs(int(ty) - agent.y)
                        )
                        # Budget in travel ticks, not raw grid cells. At the
                        # ten-minute mission clock a suited walker covers two
                        # 100 m cells per physics tick; the old per-cell rule
                        # double-counted both fatigue and O2 after the clock
                        # conversion.
                        required_energy = self._routine_eva_start_energy_threshold(
                            agent,
                            outbound_cells,
                            target_radius,
                        )
                        if agent.needs.energy < required_energy:
                            # Conscious REST still consumes basal energy; a
                            # one-tick REST here therefore made the same
                            # outbound order fail again on every following
                            # tick until the general fatigue rule eventually
                            # forced sleep.  Keep the conservative round-trip
                            # reserve, but satisfy it with a real protected
                            # sleep block before another airlock attempt.
                            recovery_ticks = ticks_for_minutes(
                                120.0, self.SIM_MINUTES_PER_TICK
                            )
                            agent.action.action_type = "sleep"
                            agent.action.target = {
                                "eva_denied": "insufficient_round_trip_energy",
                                "target_radius_cells": target_radius,
                                "required_energy_pct": round(required_energy, 1),
                                "habitat": True,
                                "ticks": recovery_ticks,
                                "preflight_recovery": True,
                                "resume_destination": destination,
                                "resume_x": int(tx),
                                "resume_y": int(ty),
                            }
                            agent.action.ticks_remaining = recovery_ticks
                            self._route_queued_indoor_recovery(agent)
                            return agent_events
                    if not self._prepare_agent_for_eva(
                        agent,
                        mission_critical_maintenance=maintenance,
                        target_position=(int(tx), int(ty)) if tx is not None and ty is not None else None,
                    ):
                        return agent_events
                
                if dx or dy:
                    previous_x, previous_y = agent.x, agent.y
                    agent.x = next_x
                    agent.y = next_y
                    if (
                        isinstance(active_expedition, dict)
                        and active_expedition.get("transport") == "crew_rover"
                        and active_expedition.get("role") == "lead"
                    ):
                        self.surface_fleet.record_crew_rover_movement(
                            str(active_expedition.get("id")),
                            previous_x,
                            previous_y,
                            agent.x,
                            agent.y,
                        )

                    reached_target = (
                        tx is not None and ty is not None
                        and agent.x == tx and agent.y == ty
                    )
                    lz_x = getattr(self, "lz_x", getattr(agent, "spawn_x", 1000))
                    lz_y = getattr(self, "lz_y", getattr(agent, "spawn_y", 1000))
                    reached_base = self._is_lander_airlock_cell(
                        agent.x, agent.y
                    )
                    if reached_target:
                        self._settle_movement_effort(
                            agent, completed=True, reason="arrived"
                        )
                        if destination in ("habitat", "shelter") or reached_base:
                            was_already_inside = agent._in_habitat
                            if not was_already_inside:
                                agent.enter_habitat()
                            self._order_expedition_partner_home(
                                getattr(agent, "_active_expedition", None)
                            )
                            if (
                                isinstance(active_expedition, dict)
                                and active_expedition.get("transport") == "crew_rover"
                                and active_expedition.get("role") == "lead"
                            ):
                                self._complete_crew_rover_expedition(
                                    active_expedition
                                )
                            agent._active_expedition = None
                            agent._pending_expedition = None
                            # The pressure transition has already completed in
                            # this physics tick. Do not carry an ENTER action
                            # into the next tick and appear to cycle the same
                            # airlock twice.
                            agent.action.action_type = "idle"
                        else:
                            expedition_state = getattr(agent, "_active_expedition", None)
                            if isinstance(expedition_state, dict):
                                expedition_state["status"] = "working"
                            agent.action.action_type = "arrived"
                        # Arrival completes navigation, not the mission. Keep
                        # the payload so the next decision can immediately
                        # perform gather/refine/build instead of recomputing a
                        # different goal and walking away again.
                        agent.action.target = {
                            **target,
                            "arrived": True,
                            "x": agent.x,
                            "y": agent.y,
                        }
                    else:
                        agent.action.action_type = "move"
                        agent.action.target = dict(target)
                    agent.action.ticks_remaining = 1
                else:
                    # A difficult cell may consume this tick's fractional
                    # travel credit without crossing a grid boundary. That is
                    # still an in-progress route, not arrival at the current
                    # cell. Preserve the original absolute destination until
                    # enough physical movement credit has accumulated.
                    waiting_for_travel_credit = bool(
                        tx is not None
                        and ty is not None
                        and (agent.x, agent.y) != (int(tx), int(ty))
                    )
                    if waiting_for_travel_credit:
                        agent.action.action_type = "move"
                        agent.action.target = dict(target)
                        agent.action.ticks_remaining = 1
                        return agent_events

                    self._settle_movement_effort(
                        agent,
                        completed=True,
                        reason="already_at_destination",
                    )

                    # Arrival is a completed navigation action, not movement.
                    # Enter the habitat immediately when that was the target;
                    # otherwise clear the stale movement instruction.
                    lz_x = getattr(self, "lz_x", getattr(agent, "spawn_x", 1000))
                    lz_y = getattr(self, "lz_y", getattr(agent, "spawn_y", 1000))
                    arrived_at_base = self._is_lander_airlock_cell(
                        agent.x, agent.y
                    )
                    if destination in ("habitat", "shelter") or arrived_at_base:
                        was_already_inside = agent._in_habitat
                        if not was_already_inside:
                            agent.enter_habitat()
                        self._order_expedition_partner_home(
                            getattr(agent, "_active_expedition", None)
                        )
                        if (
                            isinstance(active_expedition, dict)
                            and active_expedition.get("transport") == "crew_rover"
                            and active_expedition.get("role") == "lead"
                        ):
                            self._complete_crew_rover_expedition(
                                active_expedition
                            )
                        agent._active_expedition = None
                        agent._pending_expedition = None
                        agent.action.action_type = "idle"
                        agent.action.target = {
                            **target,
                            "arrived": True,
                            "x": agent.x,
                            "y": agent.y,
                        }
                    else:
                        agent.action.action_type = "arrived"
                        agent.action.target = {
                            **target,
                            "arrived": True,
                            "x": agent.x,
                            "y": agent.y,
                        }
                    agent.action.ticks_remaining = 1
            elif action == "enter_habitat":
                # Deliberate airlock cycling to enter pressurized habitat
                lz_x = getattr(self, "lz_x", 1000)
                lz_y = getattr(self, "lz_y", 1000)
                is_at_base = self._is_lander_airlock_cell(
                    agent.x, agent.y
                )
                is_at_habitat = any(
                    s.get("type") == "habitat_module" and max(abs(agent.x - s.get("x", -999)), abs(agent.y - s.get("y", -999))) <= 1
                    for s in getattr(self, "placed_structures", [])
                )
                if is_at_base or is_at_habitat or (agent.x == agent.spawn_x and agent.y == agent.spawn_y):
                    if not agent._in_habitat:
                        agent.enter_habitat()
                        logger.info(f"{agent.name} cycled airlock and entered habitat.")
                        agent.action.action_type = "idle"
                        agent.action.target = {"entered_habitat": True}
                    else:
                        agent.action.action_type = "idle"
                        agent.action.target = {"already_in_habitat": True}
                agent.action.ticks_remaining = 1
            elif action == "exit_habitat":
                # Deliberate airlock cycling to exit to surface EVA
                if (
                    self._is_lander_footprint_cell(agent.x, agent.y)
                    and not self._is_lander_airlock_cell(agent.x, agent.y)
                ):
                    airlock_x, airlock_y = self._lander_airlock_position()
                    dx, dy = self._cardinal_step_toward(
                        agent, airlock_x, airlock_y, 1
                    )
                    agent.x += dx
                    agent.y += dy
                    agent.action.action_type = "move"
                    agent.action.target = {
                        "x": airlock_x,
                        "y": airlock_y,
                        "destination": "lander_airlock",
                        "airlock_transit": True,
                    }
                    agent.action.ticks_remaining = 1
                elif self._prepare_agent_for_eva(agent):
                    if self._is_lander_airlock_cell(agent.x, agent.y):
                        agent.x, agent.y = (
                            self._lander_airlock_exterior_position()
                        )
                    logger.info(f"{agent.name} exited airlock to surface EVA.")
                    agent.action.action_type = "exit_habitat"
                    agent.action.target = {
                        "airlock": list(self._lander_airlock_position()),
                        "exterior": list(
                            self._lander_airlock_exterior_position()
                        ),
                    }
                    agent.action.ticks_remaining = 1
            elif action == "stand_watch":
                agent.action.action_type = "stand_watch"
                agent.action.target = dict(target)
                agent.action.ticks_remaining = max(1, int(target.get("ticks", 1)))
            elif action == "start_regional_survey":
                requested_resource = target.get("resource")
                survey_resources = self._survey_resources_from_target(target)
                if requested_resource and requested_resource not in survey_resources:
                    survey_resources.insert(0, requested_resource)
                station = self._next_regional_scan_station(
                    agent, requested_resource
                )
                lz_x = getattr(self, "lz_x", agent.spawn_x)
                lz_y = getattr(self, "lz_y", agent.spawn_y)
                if station is None:
                    agent.action.action_type = "rest"
                    agent.action.target = {"regional_survey_complete": True}
                    agent.action.ticks_remaining = 1
                elif self._distance_from_lz(agent.x, agent.y) > 2:
                    agent.action.action_type = "move"
                    agent.action.target = {
                        "x": lz_x,
                        "y": lz_y,
                        "destination": "shelter",
                        "expedition_preparation": True,
                        "resource": requested_resource,
                        "survey_resources": list(survey_resources),
                    }
                    agent.action.ticks_remaining = 1
                else:
                    # Regional traverses carry a spare PLSS cylinder. Use a
                    # delivered full bottle first; once that finite stock is at
                    # its maintenance floor, refill reusable shells at the
                    # lander/ISRU manifold for both lead and buddy. Previously
                    # the eighth trip exhausted delivered buddy bottles and the
                    # whole survey campaign silently stopped at radius 26.
                    if (
                        agent.inventory.items.get("oxygen_canisters", 0) < 1
                        and self.central_depot_inventory.get("oxygen_canisters", 0)
                        > self.MIN_CENTRAL_MAINTENANCE_O2_CANISTERS
                    ):
                        self.central_depot_inventory["oxygen_canisters"] -= 1
                        agent.inventory.add_item("oxygen_canisters", 1)
                    provisioning = False
                    if agent.inventory.items.get("oxygen_canisters", 0) < 1:
                        provisioning = self._provision_expedition_spare(agent)

                    buddy_candidate = self._find_expedition_buddy(
                        agent,
                        self._distance_from_lz(*station),
                        require_spare_canister=False,
                    )
                    if (
                        buddy_candidate is not None
                        and buddy_candidate.inventory.items.get(
                            "oxygen_canisters", 0
                        ) < 1
                    ):
                        provisioning = (
                            self._provision_expedition_spare(buddy_candidate)
                            or provisioning
                        )

                    if provisioning:
                        # Do not depart in the same tick as pressure servicing.
                        # The request remains queued and is retried after both
                        # visible refill actions finish.
                        if agent.action.action_type != "refill_o2":
                            agent.action.action_type = "rest"
                            agent.action.target = {
                                "expedition_preparation": True,
                                "waiting_for_buddy_o2_service": True,
                                "resource": requested_resource,
                                "survey_resources": list(survey_resources),
                            }
                            agent.action.ticks_remaining = 3
                        return agent_events
                    expedition = self._start_regional_survey_expedition(
                        agent, requested_resource, station, survey_resources
                    )
                    if expedition is None:
                        agent._regional_survey_retry_tick = self.current_tick + 24
                        agent.enter_habitat()
                        agent.action.action_type = "rest"
                        agent.action.target = {
                            "expedition_preparation": True,
                            "resource": requested_resource,
                            "survey_resources": list(survey_resources),
                            "station": list(station),
                        }
                        agent.action.ticks_remaining = 1
                    else:
                        # Dispatch reserves the crew/vehicle; physical motion
                        # starts on the following tick through the one fixed
                        # suitlock instead of jumping diagonally through the
                        # 4 x 4 pressure hull on this administrative tick.
                        agent.action.action_type = "move"
                        agent.action.target = {
                            "x": station[0],
                            "y": station[1],
                            "resource": requested_resource,
                            "survey_resources": list(survey_resources),
                            "expedition": True,
                            "survey_action": "portable_scanner",
                        }
                        agent.action.ticks_remaining = 1
            elif action == "survey_resources":
                requested_resource = target.get("resource")
                active_survey_state = getattr(agent, "_active_expedition", None)
                survey_resources = self._survey_resources_from_target(
                    target,
                    active_survey_state
                    if isinstance(active_survey_state, dict) else None,
                )
                if requested_resource and requested_resource not in survey_resources:
                    survey_resources.insert(0, requested_resource)
                station = None
                if target.get("survey_action") == "portable_scanner":
                    station = (int(target["x"]), int(target["y"]))
                if station is None:
                    station = self._next_portable_scan_station(agent)

                durability = int(agent.inventory.tool_durability.get("portable_scanner", 0))
                charge = float(agent.inventory.tool_charge_pct.get("portable_scanner", 0.0))
                if charge < self.PORTABLE_SCANNER_SCAN_CHARGE_PCT:
                    agent.action.action_type = "recharge_scanner"
                    agent.action.target = {
                        "resource": requested_resource,
                        "survey_resources": list(survey_resources),
                        "survey_action": "scanner_recharge",
                    }
                    agent.action.ticks_remaining = 1
                elif station is None or durability <= 0:
                    agent.action.action_type = "rest"
                    agent.action.target = {
                        "local_survey_complete": station is None,
                        "scanner_broken": durability <= 0,
                    }
                    agent.action.ticks_remaining = 1
                else:
                    sx, sy = station
                    expedition = getattr(agent, "_active_expedition", None)
                    station_radius = self._distance_from_lz(sx, sy)
                    supported_local_traverse = (
                        station_radius >= self.CREW_ROVER_MIN_DISTANCE_CELLS
                        and not isinstance(expedition, dict)
                        and (agent.x, agent.y) != (sx, sy)
                    )
                    if supported_local_traverse:
                        # The larger construction campus puts blind detector
                        # stations beyond a sensible repeated solo walk. Return
                        # to the hub first, then reserve both a buddy and a
                        # physical crew rover. Failed reservation means no EVA.
                        if self._distance_from_lz(agent.x, agent.y) > 2:
                            airlock_x, airlock_y = (
                                self._lander_airlock_position()
                            )
                            agent.action.action_type = "move"
                            agent.action.target = {
                                "x": airlock_x,
                                "y": airlock_y,
                                "destination": "shelter",
                                "survey_resources": list(survey_resources),
                                "expedition_preparation": True,
                            }
                            agent.action.ticks_remaining = 1
                            return agent_events

                        survey_resource = (
                            requested_resource
                            or (survey_resources[0] if survey_resources else "regolith")
                        )
                        launched = self._start_regional_survey_expedition(
                            agent,
                            survey_resource,
                            station,
                            survey_resources,
                            survey_scope="local",
                        )
                        if launched is None:
                            # A rover/buddy conflict is temporary.  Retrying
                            # every tick trapped the scanner custodian in an
                            # all-year wait loop and withheld that person from
                            # construction.  Keep the survey request queued but
                            # release the worker for four simulated hours.
                            agent._regional_survey_retry_tick = (
                                self.current_tick + 24
                            )
                            agent.action.action_type = "rest"
                            agent.action.target = {
                                "expedition_preparation": True,
                                "waiting_for_buddy_or_rover": True,
                                "retry_tick": agent._regional_survey_retry_tick,
                                "resource": requested_resource,
                                "survey_resources": list(survey_resources),
                                "station": [sx, sy],
                            }
                            agent.action.ticks_remaining = 1
                        else:
                            agent.action.action_type = "move"
                            agent.action.target = {
                                "x": sx,
                                "y": sy,
                                "resource": requested_resource,
                                "survey_resources": list(survey_resources),
                                "expedition": True,
                                "transport": launched.get(
                                    "transport", "crew_rover"
                                ),
                                "rover_id": launched.get("rover_id"),
                                "move_speed_cells": launched.get(
                                    "move_speed_cells",
                                    self.EXPEDITION_MOVE_SPEED_CELLS,
                                ),
                                "survey_action": "portable_scanner",
                            }
                            agent.action.ticks_remaining = 1
                        return agent_events

                    distance = max(abs(agent.x - sx), abs(agent.y - sy))
                    if distance > 0:
                        if self._move_resource_rover_team(agent, {
                            **target, "x": sx, "y": sy,
                            "expedition": True, "transport": "crew_rover",
                        }):
                            return agent_events
                        regional_move = (
                            isinstance(expedition, dict)
                            and expedition.get("kind") == "regional_survey"
                        )
                        move_step = (
                            self.EXPEDITION_MOVE_SPEED_CELLS
                            if regional_move else self.EVA_WALK_SPEED_CELLS
                        )
                        move_dx, move_dy = self._cardinal_step_toward(
                            agent, sx, sy, move_step
                        )
                        next_x = agent.x + move_dx
                        next_y = agent.y + move_dy
                        if (
                            agent._in_habitat
                            and not self._is_pressurized_location(next_x, next_y)
                        ):
                            # A resumed detector traverse must budget the
                            # whole cardinal trip before reopening the
                            # airlock. The generic 72% gate allowed a tired
                            # scanner operator to walk halfway out, retreat,
                            # sleep and repeat without ever making the final
                            # measurement.
                            route_cells = (
                                abs(int(sx) - agent.x)
                                + abs(int(sy) - agent.y)
                            )
                            return_cells = (
                                abs(int(sx) - self.lz_x)
                                + abs(int(sy) - self.lz_y)
                            )
                            travel_speed = max(
                                1, int(self.EVA_WALK_SPEED_CELLS)
                            )
                            required_energy = (
                                self._routine_eva_start_energy_threshold(
                                    agent,
                                    math.ceil(route_cells / travel_speed),
                                    math.ceil(return_cells / travel_speed),
                                )
                            )
                            if agent.needs.energy < required_energy:
                                recovery_ticks = ticks_for_minutes(
                                    120.0, self.SIM_MINUTES_PER_TICK
                                )
                                agent.action.action_type = "sleep"
                                agent.action.target = {
                                    "eva_denied": (
                                        "insufficient_survey_trip_energy"
                                    ),
                                    "required_energy_pct": round(
                                        required_energy, 1
                                    ),
                                    "habitat": True,
                                    "ticks": recovery_ticks,
                                    "resume_action": "survey_resources",
                                    "resume_x": int(sx),
                                    "resume_y": int(sy),
                                }
                                agent.action.ticks_remaining = recovery_ticks
                                self._route_queued_indoor_recovery(agent)
                                return agent_events
                            if not self._prepare_agent_for_eva(agent, target_position=(int(sx), int(sy))):
                                return agent_events
                        agent.x = max(0, min(self.world.map_size - 1, next_x))
                        agent.y = max(0, min(self.world.map_size - 1, next_y))

                    if (agent.x, agent.y) == station:
                        expedition = getattr(agent, "_active_expedition", None)
                        if (
                            isinstance(expedition, dict)
                            and expedition.get("kind") == "regional_survey"
                            and expedition.get("role") == "lead"
                        ):
                            expedition["status"] = "working"
                        found = self._perform_portable_resource_scan(agent)
                        requested_hits = len(found.get(requested_resource, []))
                        hits_by_objective = {
                            resource: len(found.get(resource, []))
                            for resource in survey_resources
                        }
                        center = (agent.x, agent.y)
                        scan_attempt = int(
                            self._portable_scan_attempts.get(center, 0)
                        )
                        expedition = getattr(agent, "_active_expedition", None)
                        is_regional = (
                            isinstance(expedition, dict)
                            and expedition.get("kind") == "regional_survey"
                            and expedition.get("role") == "lead"
                            and expedition.get("survey_scope", "regional")
                            == "regional"
                        )
                        for resource, hits in hits_by_objective.items():
                            if hits == 0:
                                self._local_resource_miss_centers.setdefault(
                                    resource, set()
                                ).add(center)
                        pit_depth = self.cell_excavation_depth.get(center, 0)
                        scan_face_depth = int(pit_depth)
                        test_pit_opened = False
                        test_pit_depth_after = scan_face_depth
                        if (
                            requested_hits == 0
                            and pit_depth < self.PORTABLE_SCANNER_MAX_TEST_PIT_LAYERS
                            and (
                                not is_regional
                                or self._has_buried_target_anomaly(
                                    agent.x, agent.y, requested_resource
                                )
                            )
                        ):
                            newly_revealed = self._excavate_test_layer(
                                agent, agent.x, agent.y
                            )
                            test_pit_depth_after = int(
                                self.cell_excavation_depth.get(
                                    center, scan_face_depth
                                )
                            )
                            test_pit_opened = (
                                test_pit_depth_after > scan_face_depth
                            )
                            if test_pit_opened:
                                agent.action.action_type = "prospect"
                                agent.action.target = {
                                    "resource": requested_resource,
                                    "survey_resources": list(survey_resources),
                                    "x": agent.x,
                                    "y": agent.y,
                                    "survey_action": "portable_scanner",
                                    "rescan_after_test_pit": True,
                                    "excavation_depth": test_pit_depth_after,
                                    "newly_revealed": newly_revealed,
                                }
                                agent.action.ticks_remaining = 3
                            else:
                                # A site can become protected after a traverse
                                # was assigned.  Preserve the non-invasive scan,
                                # close that station and move on; never index a
                                # pit that the civil-safety gate correctly
                                # refused to excavate or retry it forever.
                                for resource, hits in hits_by_objective.items():
                                    if hits == 0:
                                        self._portable_resource_miss_centers.setdefault(
                                            resource, set()
                                        ).add(center)
                                self._portable_scan_centers.add(center)
                                agent.action.action_type = "survey_resources"
                                agent.action.target = {
                                    "resource": requested_resource,
                                    "survey_resources": list(survey_resources),
                                    "x": agent.x,
                                    "y": agent.y,
                                    "survey_action": "portable_scanner",
                                    "test_pit_denied": "protected_ground",
                                    "excavation_depth": scan_face_depth,
                                }
                                agent.action.ticks_remaining = 1
                        else:
                            for resource, hits in hits_by_objective.items():
                                if hits == 0:
                                    self._portable_resource_miss_centers.setdefault(
                                        resource, set()
                                    ).add(center)
                            self._portable_scan_centers.add(center)
                            if is_regional and requested_hits > 0:
                                deposit = min(
                                    found[requested_resource],
                                    key=lambda coord: max(
                                        abs(coord[0] - agent.x),
                                        abs(coord[1] - agent.y),
                                    ),
                                )
                                capacity_recipe = getattr(
                                    self.decision_engine,
                                    "shared_work_order",
                                    {},
                                ).get("recipe")
                                recovery_goal_units = 12
                                if capacity_recipe:
                                    recovery_goal_units = max(
                                        1,
                                        int(
                                            self.decision_engine._raw_bom_deficits(
                                                capacity_recipe,
                                                self.decision_engine._pooled_materials(),
                                            ).get(
                                                requested_resource,
                                                recovery_goal_units,
                                            )
                                        ),
                                    )
                                agent._detected_resource_recovery = {
                                    "x": int(deposit[0]),
                                    "y": int(deposit[1]),
                                    "resource": requested_resource,
                                    "survey_resources": list(survey_resources),
                                    "capacity_recipe": capacity_recipe,
                                    "recovery_goal_units": recovery_goal_units,
                                }
                                buddy = next((
                                    crew for crew in self.agents
                                    if crew.id == expedition.get("buddy_id")
                                ), None)
                                for member in (agent, buddy):
                                    if member is None:
                                        continue
                                    member_state = getattr(
                                        member, "_active_expedition", None
                                    )
                                    if not isinstance(member_state, dict):
                                        continue
                                    member_state.update({
                                        "kind": "resource_recovery",
                                        "target_x": int(deposit[0]),
                                        "target_y": int(deposit[1]),
                                        "capacity_recipe": capacity_recipe,
                                        "recovery_goal_units": (
                                            recovery_goal_units
                                        ),
                                        "authorized_radius": max(
                                            int(member_state.get(
                                                "authorized_radius", 0
                                            )),
                                            self._distance_from_lz(*deposit),
                                        ),
                                        "status": "outbound",
                                    })
                                    member.action.action_type = "move"
                                    member.action.target = {
                                        "x": int(deposit[0]),
                                        "y": int(deposit[1]),
                                        "resource": requested_resource,
                                        "survey_resources": list(survey_resources),
                                        "expedition": True,
                                        "transport": member_state.get(
                                            "transport", "on_foot"
                                        ),
                                    }
                                    member.action.ticks_remaining = 1
                                agent.action.action_type = "move"
                                agent.action.target = {
                                    "x": deposit[0],
                                    "y": deposit[1],
                                    "resource": requested_resource,
                                    "survey_resources": list(survey_resources),
                                    "expedition": True,
                                    "transport": expedition.get(
                                        "transport", "on_foot"
                                    ),
                                }
                                agent.action.ticks_remaining = 1
                            elif requested_resource and requested_hits > 0:
                                # A local survey was launched for this exact
                                # missing feedstock. Preserve the measurement
                                # as a physical recovery contract instead of
                                # dropping it back into the general planner.
                                deposit = min(
                                    found[requested_resource],
                                    key=lambda coord: max(
                                        abs(coord[0] - agent.x),
                                        abs(coord[1] - agent.y),
                                    ),
                                )
                                capacity_recipe = getattr(
                                    self.decision_engine,
                                    "shared_work_order",
                                    {},
                                ).get("recipe")
                                recovery_goal_units = 1
                                if capacity_recipe:
                                    recovery_goal_units = max(
                                        1,
                                        int(
                                            self.decision_engine._raw_bom_deficits(
                                                capacity_recipe,
                                                self.decision_engine._pooled_materials(),
                                            ).get(requested_resource, 1)
                                        ),
                                    )
                                active_face = {
                                    "x": int(deposit[0]),
                                    "y": int(deposit[1]),
                                    "resource": requested_resource,
                                }
                                if capacity_recipe:
                                    active_face["capacity_recipe"] = (
                                        capacity_recipe
                                    )
                                if (
                                    self._distance_from_lz(*deposit)
                                    >= self.CREW_ROVER_MIN_DISTANCE_CELLS
                                ):
                                    # A 1+ km confirmed ore face needs repeated
                                    # cargo sorties.  Return the scanner to the
                                    # hub and launch only when a buddy, reusable
                                    # O2 cylinders and the physical rover pass
                                    # preflight; never degrade to a solo walk.
                                    recovery_contract = {
                                        **active_face,
                                        "recovery_goal_units": (
                                            recovery_goal_units
                                        ),
                                    }
                                    agent._detected_resource_recovery = (
                                        recovery_contract
                                    )
                                    agent._active_excavation = None
                                    if isinstance(expedition, dict):
                                        # The field survey is complete. Without
                                        # this transition the expedition policy
                                        # ordered another scan every tick and
                                        # overwrote the queued homeward move.
                                        self.decision_engine._recall_expedition_team(agent)
                                    agent.action.action_type = "move"
                                    agent.action.target = {
                                        "x": self.lz_x,
                                        "y": self.lz_y,
                                        "destination": "shelter",
                                        "resource": requested_resource,
                                        "detector_recovery": True,
                                        "detector_confirmed": True,
                                        "expedition": isinstance(expedition, dict),
                                    }
                                else:
                                    # Nearby faces remain practical hand-EVA
                                    # jobs. Geological layers are still
                                    # authoritative and must be removed in
                                    # order; the detector never gifts ore.
                                    agent._active_excavation = active_face
                                    if (agent.x, agent.y) == deposit:
                                        agent.action.action_type = "gather"
                                        agent.action.target = {
                                            **active_face,
                                            "continue_excavation": True,
                                            "detector_confirmed": True,
                                        }
                                    else:
                                        agent.action.action_type = "move"
                                        agent.action.target = {
                                            **active_face,
                                            "destination": "active_excavation_face",
                                            "mission_action": "gather",
                                            "detector_confirmed": True,
                                        }
                                agent.action.ticks_remaining = 1
                            elif is_regional:
                                next_station = self._next_regional_scan_station(
                                    agent, requested_resource
                                )
                                continued = bool(
                                    next_station
                                    and self._continue_regional_survey(
                                        agent, expedition, next_station
                                    )
                                )
                                if not continued:
                                    expedition["status"] = "returning"
                                    agent.action.action_type = "move"
                                    agent.action.target = {
                                        "x": self.lz_x,
                                        "y": self.lz_y,
                                        "destination": "shelter",
                                        "resource": requested_resource,
                                        "expedition": True,
                                        "regional_survey_complete": True,
                                    }
                                    agent.action.ticks_remaining = 1
                            else:
                                agent.action.action_type = "survey_resources"
                                agent.action.target = {
                                    "resource": requested_resource,
                                    "survey_resources": list(survey_resources),
                                    "scan_center": [agent.x, agent.y],
                                    "requested_hits": requested_hits,
                                    "scanner_charge_pct": agent.inventory.tool_charge_pct.get(
                                        "portable_scanner", 0.0
                                    ),
                                }
                                agent.action.ticks_remaining = 4
                        if self._on_event:
                            pass_limit = (
                                self.PORTABLE_SCANNER_MAX_TEST_PIT_LAYERS + 1
                            )
                            pit_note = (
                                f"; opened test-pit layer "
                                f"{test_pit_depth_after}"
                                if test_pit_opened else ""
                            )
                            self._on_event({
                                "type": "resource_survey",
                                "agent": agent.name,
                                "cause": (
                                    f"Portable scan pass {scan_attempt}/{pass_limit} "
                                    f"at ({agent.x}, {agent.y}), test-pit depth "
                                    f"{scan_face_depth}, found {requested_hits} "
                                    f"{requested_resource or 'requested'} deposits"
                                    f"{pit_note}"
                                ),
                                "scan_attempt": scan_attempt,
                                "scan_face_depth": scan_face_depth,
                                "test_pit_opened": test_pit_opened,
                                "test_pit_depth_after": test_pit_depth_after,
                                "scan_pass_limit": pass_limit,
                                "survey_resources": list(survey_resources),
                                "hits_by_objective": dict(hits_by_objective),
                                "tick": self.current_tick,
                            })
                    else:
                        agent.action.action_type = "move"
                        agent.action.target = {
                            "x": sx,
                            "y": sy,
                            "resource": requested_resource,
                            "survey_resources": list(survey_resources),
                            "survey_action": "portable_scanner",
                        }
                        agent.action.ticks_remaining = 1
            elif action == "recharge_scanner":
                lz_x = getattr(self, "lz_x", agent.spawn_x)
                lz_y = getattr(self, "lz_y", agent.spawn_y)
                airlock_x, airlock_y = self._lander_airlock_position()
                at_lander_access = bool(agent._in_habitat) or (
                    self._is_lander_airlock_cell(agent.x, agent.y)
                )
                if not at_lander_access:
                    expedition = getattr(agent, "_active_expedition", None)
                    if isinstance(expedition, dict) and expedition.get("transport") == "crew_rover":
                        self.decision_engine._recall_expedition_team(agent)
                        if self._move_resource_rover_team(agent, {
                            "x": airlock_x, "y": airlock_y,
                            "destination": "shelter", "expedition": True,
                            "survey_action": "scanner_recharge",
                        }):
                            return agent_events
                    dx, dy = self._cardinal_step_toward(
                        agent, airlock_x, airlock_y,
                        self.EVA_WALK_SPEED_CELLS,
                    )
                    agent.x += dx
                    agent.y += dy
                    agent.action.action_type = "move"
                    agent.action.target = {
                        "x": airlock_x,
                        "y": airlock_y,
                        "resource": target.get("resource"),
                        "survey_resources": list(
                            target.get("survey_resources", [])
                        ),
                        "survey_action": "scanner_recharge",
                    }
                    agent.action.ticks_remaining = 1
                else:
                    agent.enter_habitat()
                    current_charge = float(
                        agent.inventory.tool_charge_pct.get("portable_scanner", 0.0)
                    )
                    desired_gain = min(
                        self.PORTABLE_SCANNER_CHARGE_STEP_PCT,
                        100.0 - current_charge,
                    )
                    # 100 Wh pack charged at 85% wall-to-battery efficiency.
                    desired_draw = (
                        desired_gain / 100.0
                        * self.PORTABLE_SCANNER_CHARGE_KWH / 0.85
                    )
                    available_kwh = max(
                        0.0, self._colony_resources.get("energy_stored_kwh", 0.0)
                    )
                    actual_draw = min(desired_draw, available_kwh)
                    actual_gain = (
                        actual_draw * 0.85 / self.PORTABLE_SCANNER_CHARGE_KWH * 100.0
                        if self.PORTABLE_SCANNER_CHARGE_KWH > 0 else 0.0
                    )
                    self._colony_resources["energy_stored_kwh"] = max(
                        0.0, available_kwh - actual_draw
                    )
                    agent.inventory.tool_charge_pct["portable_scanner"] = min(
                        100.0, current_charge + actual_gain
                    )
                    agent.action.action_type = "recharge_scanner"
                    agent.action.target = {
                        "resource": target.get("resource"),
                        "survey_resources": list(
                            target.get("survey_resources", [])
                        ),
                        "charged_pct": actual_gain,
                        "energy_draw_kwh": actual_draw,
                        "power_unavailable": actual_gain <= 0.0,
                    }
                    agent.action.ticks_remaining = 1
            elif action == "explore":
                # Frontier-based exploration inside the base-centred EVA area.
                best_dx, best_dy = self._nearest_unexplored_step(agent)
                next_x = max(0, min(self.world.map_size - 1, agent.x + best_dx))
                next_y = max(0, min(self.world.map_size - 1, agent.y + best_dy))
                if (
                    agent._in_habitat
                    and not self._is_pressurized_location(next_x, next_y)
                    and not self._prepare_agent_for_eva(agent)
                ):
                    return agent_events
                agent.x = next_x
                agent.y = next_y
                agent.action.action_type = "explore"
                agent.action.ticks_remaining = 1
            elif action == "use_item":
                item_name = target.get("item")
                if item_name == "oxygen_canisters":
                    agent.load_o2_canister()
                agent.action.action_type = "use_item"
                agent.action.ticks_remaining = 1
            elif action == "refine":
                # Multi-stage Industrial Manufacturing & Pyrometallurgy
                target_output = target.get("output", target.get("raw_material", "reduced_iron_ingot"))
                recipe = self._get_recipe(target_output)
                
                if recipe and recipe.get("output", {}).get("type") == "material":
                    req_struct = recipe.get("requires_structure")
                    mat_cost = recipe.get("materials", {})
                    out_qty = recipe.get("output", {}).get("quantity", 1)
                    duration_ticks = recipe.get("base_duration_ticks", 4)
                    power_cost = float(recipe.get(
                        "energy_kwh_per_batch",
                        0.4 if req_struct == "stone_furnace" else 0.6,
                    ))
                else:
                    agent.action.action_type = "recipe_unavailable"
                    agent.action.target = {"output": target_output}
                    agent.action.ticks_remaining = 1
                    return agent_events

                if not req_struct:
                    agent.action.action_type = "recipe_unavailable"
                    agent.action.target = {
                        "output": target_output,
                        "reason": "material recipe has no explicit manufacturing route",
                    }
                    agent.action.ticks_remaining = 1
                    return agent_events

                non_material_recipe = dict(recipe)
                non_material_recipe["materials"] = {}
                # The delivered enclosed cells own their qualified cutters,
                # fixtures and interlocked tooling. A worn portable field kit
                # must not prevent loading or unloading a fixed machine.
                non_material_recipe["requires_tool"] = False
                preflight = agent.can_craft(non_material_recipe)
                if not preflight.get("can_craft", False):
                    agent.action.action_type = "craft_blocked"
                    agent.action.target = {
                        "output": target_output,
                        "reason": preflight,
                        "machine_type": req_struct,
                    }
                    agent.action.ticks_remaining = 1
                    return agent_events
                
                # Precision work must use the specified physical machine.
                lz_x = getattr(self, "lz_x", 1000)
                lz_y = getattr(self, "lz_y", 1000)
                matching_machines = [
                    s for s in getattr(self, "placed_structures", [])
                    if s.get("type") == req_struct
                    and not s.get("under_construction", False)
                    and not s.get("destroyed", False)
                ]
                if not matching_machines:
                    agent.action.action_type = "recipe_unavailable"
                    agent.action.target = {
                        "output": target_output,
                        "reason": f"requires operational {req_struct}",
                    }
                    agent.action.ticks_remaining = 1
                    return agent_events

                machine_by_id = {
                    str(machine.get("id")): machine for machine in matching_machines
                }
                resumable_cycles = [
                    cycle for cycle in self._manufacturing_cycles.values()
                    if cycle.get("completion_pending")
                    and cycle.get("machine_type") == req_struct
                    and cycle.get("output") == target_output
                    and str(cycle.get("machine_id")) in machine_by_id
                    and self._agent_can_take_manufacturing_cycle(
                        agent, cycle, recipe
                    )
                ]
                resuming_cycle = (
                    min(
                        resumable_cycles,
                        key=lambda cycle: (
                            (cycle.get("x", lz_x) - agent.x) ** 2
                            + (cycle.get("y", lz_y) - agent.y) ** 2,
                            str(cycle.get("machine_id")),
                        ),
                    )
                    if resumable_cycles else None
                )

                paused = getattr(agent, "_paused_manufacturing", None)
                paused_machine_id = (
                    str(paused.get("machine_id"))
                    if isinstance(paused, dict)
                    and paused.get("completion_pending")
                    and paused.get("machine_id")
                    else None
                )
                paused_cycle = (
                    self._manufacturing_cycles.get(paused_machine_id)
                    if paused_machine_id else None
                )
                if (
                    paused_cycle is not None
                    and paused_cycle.get("completion_pending")
                    and paused_cycle.get("output") == target_output
                    and resuming_cycle is None
                ):
                    # The batch still belongs to its original machine.  During
                    # the relief window the old operator may not clone their
                    # paused action onto a second free workshop.
                    agent.action.action_type = "idle"
                    agent.action.target = {
                        "output": target_output,
                        "machine_id": paused_machine_id,
                        "machine_type": req_struct,
                        "reason": "machine_cycle_awaiting_relief",
                        "machine_cycle_reserved": True,
                    }
                    agent.action.ticks_remaining = 1
                    return agent_events
                if (
                    paused_machine_id
                    and (
                        paused_cycle is None
                        or not paused_cycle.get("completion_pending")
                    )
                ):
                    # The canonical batch has already completed or been
                    # cancelled; discard the stale personal view.
                    delattr(agent, "_paused_manufacturing")
                    paused = None

                # Engine-owned WIP reserves a slot even while no astronaut is
                # standing at its controls.  A second order cannot overwrite
                # that batch or charge its BOM and power again.
                occupied_machine_ids = {
                    str(machine_id)
                    for machine_id, cycle in self._manufacturing_cycles.items()
                    if cycle.get("completion_pending")
                }
                for other in self.agents:
                    if other.id == agent.id or not isinstance(
                        other.action.target, dict
                    ):
                        continue
                    other_target = other.action.target
                    other_action = getattr(other.action, "action_type", "")
                    running_cycle = (
                        other_action == "refine"
                        and other_target.get("completion_pending")
                    )
                    reserved_route = (
                        other_action in ("move", "arrived")
                        and other_target.get("destination")
                        == "manufacturing_machine"
                    )
                    if (
                        (running_cycle or reserved_route)
                        and other_target.get("machine_type") == req_struct
                        and other_target.get("machine_id")
                    ):
                        occupied_machine_ids.add(
                            str(other_target.get("machine_id"))
                        )
                if resuming_cycle is not None:
                    closest_machine = machine_by_id[
                        str(resuming_cycle.get("machine_id"))
                    ]
                else:
                    free_machines = [
                        machine for machine in matching_machines
                        if str(machine.get("id")) not in occupied_machine_ids
                    ]
                    if not free_machines:
                        agent.action.action_type = "idle"
                        agent.action.target = {
                            "output": target_output,
                            "reason": "all_physical_machines_occupied",
                            "machine_type": req_struct,
                            "machine_cycle_reserved": True,
                        }
                        agent.action.ticks_remaining = 1
                        return agent_events

                    closest_machine = min(
                        free_machines,
                        key=lambda s: (s.get("x", lz_x) - agent.x)**2 + (s.get("y", lz_y) - agent.y)**2
                    )
                machine_id = str(closest_machine.get("id"))
                mx, my = closest_machine.get("x", lz_x), closest_machine.get("y", lz_y)
                dist = max(abs(agent.x - mx), abs(agent.y - my))
                
                if dist > 1:
                    # Physically walk to the machine cell first
                    dx, dy = self._cardinal_step_toward(agent, mx, my, 1)
                    next_x = max(0, min(self.world.map_size - 1, agent.x + dx))
                    next_y = max(0, min(self.world.map_size - 1, agent.y + dy))
                    if (
                        agent._in_habitat
                        and not self._is_pressurized_location(next_x, next_y)
                        and not self._prepare_agent_for_eva(agent, target_position=(int(mx), int(my)))
                    ):
                        return agent_events
                    agent.x = next_x
                    agent.y = next_y
                    agent.action.action_type = "move"
                    agent.action.target = {
                        "x": mx, "y": my, "dx": dx, "dy": dy,
                        "output": target_output,
                        "machine_id": machine_id,
                        "machine_type": req_struct,
                        "destination": "manufacturing_machine",
                    }
                    if resuming_cycle is not None:
                        resuming_cycle["route_operator_id"] = agent.id
                        resuming_cycle["route_reserved_until_tick"] = (
                            self.current_tick + max(2, dist + 2)
                        )
                        agent.action.target["resume_machine_cycle"] = True
                    agent.action.ticks_remaining = 1
                    logger.info(f"{agent.name} walking to {req_struct.upper()} at ({mx}, {my}) to fabricate {target_output}")
                else:
                    if resuming_cycle is not None:
                        handoff_from = self._assign_manufacturing_cycle(
                            agent, resuming_cycle
                        )
                        event = {
                            "machine_id": machine_id,
                            "output": target_output,
                            "remaining_ticks": int(
                                resuming_cycle.get("remaining_ticks", 0)
                            ),
                            "from_operator_id": handoff_from,
                            "to_operator_id": agent.id,
                        }
                        if handoff_from:
                            agent_events["manufacturing_handoff"] = event
                            previous = next((
                                crew.name for crew in self.agents
                                if crew.id == handoff_from
                            ), handoff_from)
                            if self._on_event:
                                self._on_event({
                                    "type": "manufacturing_handoff",
                                    "agent": agent.name,
                                    "previous_agent": previous,
                                    **event,
                                    "tick": self.current_tick,
                                })
                        else:
                            agent_events["manufacturing_resumed"] = event
                        return agent_events

                    resuming = (
                        isinstance(paused, dict)
                        and paused.get("output") == target_output
                        and paused.get("completion_pending")
                        and not paused.get("machine_id")
                    )
                    if resuming:
                        agent.action.action_type = "refine"
                        agent.action.target = {
                            **paused,
                            "x": mx,
                            "y": my,
                            "machine_id": machine_id,
                            "resumed": True,
                        }
                        agent.action.ticks_remaining = max(
                            1, int(paused.get("remaining_ticks", duration_ticks))
                        )
                        return agent_events

                    energy_available = float(
                        self._colony_resources.get("energy_stored_kwh", 0.0)
                    )
                    has_all_mats = all(
                        self._staged_material_totals(
                            mx, my, purpose_recipe=target_output
                        ).get(mat, 0) >= qty
                        for mat, qty in mat_cost.items()
                    )

                    if has_all_mats and energy_available >= power_cost:
                        self._consume_staged_materials(
                            mat_cost,
                            mx,
                            my,
                            purpose_recipe=target_output,
                        )
                        self._colony_resources["energy_stored_kwh"] = (
                            energy_available - power_cost
                        )
                        
                        # The powered machine owns its cutters and fixtures;
                        # do not silently wear a handheld field kit merely
                        # because its operator happens to carry one.
                        closest_machine["manufacturing_cycles_started"] = int(
                            closest_machine.get("manufacturing_cycles_started", 0)
                        ) + 1
                        
                        total_mass_kg = out_qty * MATERIAL_DENSITY_KG.get(target_output, 1.0)
                        cycle_id = (
                            f"{machine_id}:{self.current_tick}:{target_output}"
                        )
                        logger.info(
                            f"{agent.name} started {req_struct.upper()} cycle at ({mx}, {my}) "
                            f"for {out_qty}x {target_output} ({duration_ticks} ticks)"
                        )
                        if self._on_event:
                            self._on_event({
                                "type": "manufacturing_start",
                                "agent": agent.name,
                                "output": target_output,
                                "quantity": out_qty,
                                "output_mass_kg": round(total_mass_kg, 6),
                                "machine_id": machine_id,
                                "cycle_id": cycle_id,
                                "autonomous_batch_control": bool(
                                    self.AUTONOMOUS_MANUFACTURING_ENABLED
                                ),
                                "cause": (
                                    f"Started {out_qty}x {target_output.replace('_', ' ').title()} "
                                    f"({total_mass_kg:.1f} kg, {duration_ticks * self.SIM_MINUTES_PER_TICK / 60.0:.1f} h, "
                                    f"{power_cost:.1f} kWh) at {req_struct.replace('_', ' ').title()}"
                                    + (
                                        "; qualified closed-loop batch control, "
                                        "human setup and final inspection"
                                        if self.AUTONOMOUS_MANUFACTURING_ENABLED
                                        else "; continuously supervised operation"
                                    )
                                ),
                                "tick": self.current_tick
                            })
                        cycle = {
                            "x": mx,
                            "y": my,
                            "output": target_output,
                            "output_quantity": out_qty,
                            "completion_pending": True,
                            "energy_kwh": power_cost,
                            "machine_type": req_struct,
                            "machine_id": machine_id,
                            "cycle_id": cycle_id,
                            "output_mass_kg": total_mass_kg,
                            "remaining_ticks": int(duration_ticks),
                            "nominal_process_ticks": int(duration_ticks),
                            "operator_id": (
                                None
                                if self.AUTONOMOUS_MANUFACTURING_ENABLED
                                else agent.id
                            ),
                            "initiated_by_id": agent.id,
                            "active": True,
                            "autonomous": bool(
                                self.AUTONOMOUS_MANUFACTURING_ENABLED
                            ),
                            "autonomous_processing": bool(
                                self.AUTONOMOUS_MANUFACTURING_ENABLED
                            ),
                            "output_ready": False,
                            "started_tick": self.current_tick,
                            "last_process_tick": self.current_tick,
                            "shift_start_tick": self.current_tick,
                            "shift_end_tick": (
                                self.current_tick + self.MANUFACTURING_SHIFT_TICKS
                            ),
                            "handoff_count": 0,
                        }
                        self._manufacturing_cycles[machine_id] = cycle
                        if self.AUTONOMOUS_MANUFACTURING_ENABLED:
                            self._manufacturing_automation_telemetry[
                                "cycles_started"
                            ] += 1
                            self._manufacturing_automation_telemetry[
                                "nominal_process_ticks_committed"
                            ] += int(duration_ticks)
                            # Keep one read-only WIP token on the initiating
                            # worker so the shared planner reserves this output
                            # while that worker is free to perform useful work.
                            paused_target = self._manufacturing_action_target(
                                cycle
                            )
                            paused_target["pause_reason"] = (
                                "autonomous_process_running"
                            )
                            agent._paused_manufacturing = paused_target
                            agent.action.action_type = "operate_machine"
                            agent.action.target = {
                                "x": mx,
                                "y": my,
                                "output": target_output,
                                "machine_type": req_struct,
                                "machine_id": machine_id,
                                "cycle_id": cycle_id,
                                "autonomous_cycle_setup": True,
                                "process_ticks": int(duration_ticks),
                            }
                            agent.action.ticks_remaining = max(
                                1, self.MANUFACTURING_SETUP_TICKS
                            )
                        else:
                            agent.action.action_type = "refine"
                            agent.action.target = (
                                self._manufacturing_action_target(cycle)
                            )
                            agent.action.ticks_remaining = duration_ticks
                        agent.action.ticks_elapsed = 0
                    else:
                        reason = (
                            "insufficient_power" if energy_available < power_cost
                            else "materials_not_staged"
                        )
                        logger.info(
                            f"{agent.name} cannot start {target_output} at {req_struct.upper()}: {reason}"
                        )
                        agent.action.action_type = "idle"
                        agent.action.target = {"output": target_output, "reason": reason}
                        agent.action.ticks_remaining = 1
            elif action == "rescue":
                # Search and rescue living incapacitated or critical teammate (never dead)
                victim_id = target.get("victim_id")
                victim = next((other for other in self.agents if other.id == victim_id and getattr(other.status, 'value', str(other.status)) != 'dead'), None)
                if victim:
                    carrying_victim = False
                    dist = max(abs(agent.x - victim.x), abs(agent.y - victim.y))
                    if dist > 1 or agent._in_habitat != victim._in_habitat:
                        # Grid adjacency through a pressure hull is not
                        # physical access. Reach the patient's compartment
                        # through the airlock before attaching the casualty.
                        # Move towards victim
                        dx, dy = self._cardinal_step_toward(agent, victim.x, victim.y, 1)
                        agent.x = max(0, min(self.world.map_size - 1, agent.x + dx))
                        agent.y = max(0, min(self.world.map_size - 1, agent.y + dy))
                        agent.action.action_type = "move"
                        agent.action.target = {"x": victim.x, "y": victim.y}
                        agent.action.ticks_remaining = 1
                        logger.info(f"{agent.name} rushing to rescue {victim.name} at ({victim.x}, {victim.y})")
                    else:
                        # At victim! Stabilize and carry victim towards nearest habitat airlock
                        carrying_victim = True
                        lz_x = getattr(self, "lz_x", 1000)
                        lz_y = getattr(self, "lz_y", 1000)
                        shelter_x, shelter_y = self._lander_airlock_position()
                        main_lander_selected = True
                        min_shelter_dist = max(
                            abs(agent.x - shelter_x),
                            abs(agent.y - shelter_y),
                        )
                        for s in getattr(self, "placed_structures", []):
                            if s.get("type") == "habitat_module" and not s.get("under_construction", False) and not s.get("destroyed", False):
                                sx, sy = s.get("x", lz_x), s.get("y", lz_y)
                                dist = max(abs(agent.x - sx), abs(agent.y - sy))
                                if dist < min_shelter_dist:
                                    min_shelter_dist = dist
                                    shelter_x, shelter_y = sx, sy
                                    main_lander_selected = False

                        if (main_lander_selected and not agent._in_habitat
                            and (agent.x, agent.y) == self._lander_airlock_exterior_position()):
                            # Both people occupy the return cycle. A solo
                            # permit must not carry a second person through
                            # the hull or leave their return allocation stale.
                            if not self._airlock_pass([agent, victim], "in"):
                                agent.action.action_type = "rescue"
                                agent.action.target = {"victim_id": victim.id,
                                                       "carrying_victim": True}
                                agent.action.ticks_remaining = 1
                                return agent_events
                            agent._managed_airlock_tick = self.current_tick
                            victim._managed_airlock_tick = self.current_tick
                            dx, dy = shelter_x - agent.x, shelter_y - agent.y
                        else:
                            dx, dy = self._cardinal_step_toward(
                                agent, shelter_x, shelter_y, 1
                            )
                        agent.x = max(0, min(self.world.map_size - 1, agent.x + dx))
                        agent.y = max(0, min(self.world.map_size - 1, agent.y + dy))
                        victim.x = agent.x
                        victim.y = agent.y
                        
                        # Rescuer field stabilization: halt active death countdown while in transit
                        victim.needs._temp_death_timer = max(0, victim.needs._temp_death_timer - 1)
                        victim.needs._o2_death_timer = max(0, victim.needs._o2_death_timer - 1)
                        victim.needs._thirst_death_timer = max(0, victim.needs._thirst_death_timer - 1)
                        victim.needs._energy_death_timer = max(0, victim.needs._energy_death_timer - 1)
                        
                        reached_shelter = (
                            self._is_lander_airlock_cell(agent.x, agent.y)
                            if main_lander_selected else
                            max(
                                abs(agent.x - shelter_x),
                                abs(agent.y - shelter_y),
                            ) <= 1
                        )
                        if reached_shelter:
                            victim.enter_habitat()
                            agent.enter_habitat()
                            
                            # === CLINICAL RESUSCITATION & REWARMING SHOCK CHECK ===
                            # Severe hypothermia (<28°C core) and prolonged anoxia carry high risk of irreversible arrest / afterdrop fibrillation
                            hypo_sev = victim.needs._temp_death_timer / max(1, victim.needs.TEMP_DEATH_TICKS)
                            anox_sev = victim.needs._o2_death_timer / max(1, victim.needs.SUFFOCATION_TICKS)
                            max_severity = max(hypo_sev, anox_sev)
                            
                            rescuer_med = getattr(agent.competency, "medical", 5)
                            # Mortality risk scales with how long victim was left dying, mitigated by rescuer's medical skill
                            mortality_risk = min(0.60, max(0.0, max_severity * 0.70 - rescuer_med * 0.05)) if max_severity >= 0.40 else 0.0
                            
                            if mortality_risk > 0 and agent._rng.random() < mortality_risk:
                                # Fatal clinical collapse during resuscitation (Rewarming shock / Cerebral anoxia)
                                fatal_cause = DeathCause.HYPOTHERMIA if hypo_sev >= anox_sev else DeathCause.SUFFOCATION
                                victim._die(fatal_cause, self.current_tick)
                                logger.warning(f"RESUSCITATION FAILED: {victim.name} suffered fatal clinical collapse ({fatal_cause.value}) during emergency rewarming.")
                                if self._on_event:
                                    self._on_event({
                                        "type": "medical_fatal",
                                        "agent": victim.name,
                                        "cause": f"💔 RESUSCITATION FAILED: {victim.name} suffered fatal ventricular fibrillation / anoxia during emergency rewarming ({fatal_cause.value.replace('_', ' ').title()}).",
                                        "tick": self.current_tick
                                    })
                            else:
                                # Admission is transport, not a magic resupply.
                                # Habitat O2 recovery is performed later by the
                                # life-support pass only while bulk O2 exists;
                                # hydration still requires an explicit drink.
                                # Sleep physiology may restore fatigue, but no
                                # vital or death timer is reset here for free.
                                physiologically_stable = (
                                    victim.needs.o2_supply > 0.0
                                    and victim.needs.thirst > 0.0
                                    and victim.needs.hunger > 0.0
                                    and victim.needs.energy > 0.0
                                    and victim.needs.temperature_stress > 0.0
                                )
                                victim.status = (
                                    AgentStatus.CRITICAL
                                    if physiologically_stable
                                    else AgentStatus.INCAPACITATED
                                )
                                victim.action.action_type = "medical_rest"
                                victim.action.target = {
                                    "habitat": True,
                                    "medical_recovery": True,
                                    "conscious": physiologically_stable,
                                    "oral_fluids_allowed": physiologically_stable,
                                    "awaiting_resources": not physiologically_stable,
                                }
                                victim.action.ticks_remaining = 24  # Two-hour monitored recovery
                                
                                logger.info(f"RESCUE SUCCESS: {agent.name} delivered {victim.name} into habitat medical bay (ICU rewarming stable).")
                                if self._on_event:
                                    self._on_event({
                                        "type": "rescue",
                                        "agent": agent.name,
                                        "cause": f"🏥 ICU ADMISSION: {agent.name} evacuated {victim.name} into Habitat Medical Bay (critical but stable gradual rewarming).",
                                        "tick": self.current_tick
                                    })
                                # Heroic SAR Reward
                                self.decision_engine.apply_rl_reward(agent, 15.0, "sar:success")
                        agent.action.action_type = "rescue"
                        agent.action.target = {
                            "victim_id": victim.id,
                            "carrying_victim": carrying_victim,
                        }
                        agent.action.ticks_remaining = 2
                else:
                    agent.action.action_type = "idle"
                    agent.action.ticks_remaining = 1
            elif action in ("scavenge_corpse", "scavenge", "loot_corpse"):
                # Salvage equipment, rations, O2 canisters and minerals from fallen teammates
                victim_id = target.get("victim_id")
                victim = next((other for other in self.agents if other.id == victim_id and getattr(other.status, 'value', str(other.status)) == 'dead'), None)
                if not victim:
                    dead_with_loot = [
                        other for other in self.agents
                        if getattr(other.status, 'value', str(other.status)) == 'dead'
                        and not getattr(other, "corpse_salvaged", False)
                        and (other.inventory.items or other.inventory.materials)
                    ]
                    if dead_with_loot:
                        victim = min(dead_with_loot, key=lambda o: (o.x - agent.x)**2 + (o.y - agent.y)**2)
                
                if victim:
                    dist = max(abs(agent.x - victim.x), abs(agent.y - victim.y))
                    if dist > 1:
                        dx = max(-1, min(1, victim.x - agent.x))
                        dy = max(-1, min(1, victim.y - agent.y))
                        agent.x = max(0, min(self.world.map_size - 1, agent.x + dx))
                        agent.y = max(0, min(self.world.map_size - 1, agent.y + dy))
                        agent.action.action_type = "move"
                        agent.action.target = {"x": victim.x, "y": victim.y}
                        agent.action.ticks_remaining = 1
                        logger.info(f"{agent.name} moving to recover supplies from fallen crew member {victim.name}'s suit at ({victim.x}, {victim.y})")
                    else:
                        # Adjacent to corpse! Salvage gear & ores
                        res = victim.salvage_corpse(agent)
                        item_summary = []
                        if res.get("items"):
                            item_summary.extend(f"{q}x {k}" for k, q in res["items"].items())
                        if res.get("materials"):
                            item_summary.extend(f"{q}x {k}" for k, q in res["materials"].items())
                        salvage_str = ", ".join(item_summary) if item_summary else "no salvageable supplies"
                        
                        logger.info(f"{agent.name} recovered survival gear & resources from {victim.name}: {salvage_str}")
                        if self._on_event:
                            self._on_event({
                                "type": "corpse_salvaged",
                                "agent": agent.name,
                                "cause": f"📦 SALVAGED FALLEN CREW: {agent.name} recovered supplies ({salvage_str}) from {victim.name}'s suit.",
                                "tick": self.current_tick
                            })
                        # Reward living agent for securing life-saving resources
                        self.decision_engine.apply_rl_reward(agent, 8.0, "salvage:success")
                        agent.action.action_type = "scavenge_corpse"
                        agent.action.ticks_remaining = 2
                else:
                    agent.action.action_type = "idle"
                    agent.action.ticks_remaining = 1
            elif action == "repair_tool":
                # Restore tool durability at Forge
                forge = next((s for s in getattr(self, "placed_structures", []) if s.get("type") == "forge"), None)
                if forge:
                    dist = max(abs(agent.x - forge.get("x", agent.x)), abs(agent.y - forge.get("y", agent.y)))
                    if dist > 1:
                        fx, fy = forge.get("x", agent.x), forge.get("y", agent.y)
                        dx = max(-1, min(1, fx - agent.x))
                        dy = max(-1, min(1, fy - agent.y))
                        agent.x = max(0, min(self.world.map_size - 1, agent.x + dx))
                        agent.y = max(0, min(self.world.map_size - 1, agent.y + dy))
                        agent.action.action_type = "move"
                        agent.action.target = {"x": fx, "y": fy}
                        agent.action.ticks_remaining = 1
                    else:
                        # At Forge! Repair worn tool
                        tool_name = target.get("tool", "multitool_kit")
                        if hasattr(agent.inventory, 'tool_durability'):
                            agent.inventory.tool_durability[tool_name] = 60
                            logger.info(f"{agent.name} repaired {tool_name} to 60% durability at Forge")
                            if self._on_event:
                                self._on_event({
                                    "type": "repair_tool",
                                    "agent": agent.name,
                                    "cause": f"Repaired {tool_name.replace('_', ' ').title()} to 60% durability at Forge",
                                    "tick": self.current_tick
                                })
                            self.decision_engine.apply_rl_reward(agent, 4.0, "repair:success")
                        agent.action.action_type = "repair_tool"
                        agent.action.ticks_remaining = 3
                else:
                    agent.action.action_type = "idle"
                    agent.action.ticks_remaining = 1
            elif action == "deposit_materials":
                # Haul and store raw minerals/components into Central Storage Depot Silo
                lz_x = getattr(self, "lz_x", 1000)
                lz_y = getattr(self, "lz_y", 1000)
                dist = max(abs(agent.x - lz_x), abs(agent.y - lz_y))
                if dist > 2:
                    # Preserve physical walking speed across tick resolutions
                    # and keep the haul route cardinal rather than diagonal.
                    step_speed = self.EVA_WALK_SPEED_CELLS * (
                        2 if dist > 3 else 1
                    )
                    dx, dy = self._cardinal_step_toward(
                        agent, lz_x, lz_y, step_speed
                    )
                    agent.x = max(0, min(self.world.map_size - 1, agent.x + dx))
                    agent.y = max(0, min(self.world.map_size - 1, agent.y + dy))
                    agent.action.action_type = "move"
                    agent.action.target = {
                        "x": lz_x,
                        "y": lz_y,
                        "destination": "central_depot",
                        "mission_action": "deposit_materials",
                    }
                    agent.action.ticks_remaining = 1
                else:
                    # At Base Silo! Offload materials
                    depot = getattr(self, "central_depot_inventory", {})
                    deposited_count = 0
                    for mat, qty in list(agent.inventory.materials.items()):
                        if qty > 0:
                            depot[mat] = depot.get(mat, 0) + qty
                            agent.inventory.remove_material(mat, qty)
                            deposited_count += qty
                    if deposited_count > 0:
                        logger.info(f"{agent.name} stored {deposited_count}x items in Central Storage Depot Silo")
                    if self._on_event and deposited_count > 0:
                        self._on_event({
                            "type": "deposit_materials",
                            "agent": agent.name,
                            "cause": f"Transferred {deposited_count}x materials to Central Storage Depot Silo",
                            "tick": self.current_tick
                        })
                    if deposited_count > 0:
                        self.decision_engine.apply_rl_reward(agent, 3.0, "deposit:success")
                        agent.action.action_type = "deposit_materials"
                        agent.action.ticks_remaining = 2
                    else:
                        agent.action.action_type = "rest"
                        agent.action.target = {"empty_deposit_rejected": True}
                        agent.action.ticks_remaining = 1
            elif action == "clean_solar_panels":
                # Maintenance EVA: Sweep abrasive dust off solar panel array
                eligible_panels = [
                    s for s in getattr(self, "placed_structures", [])
                    if s.get("type") in {"solar_panel", "eclss_lander_hub"}
                    and not s.get("under_construction", False)
                    and not s.get("destroyed", False)
                    and s.get("dust_fouling_level", 0.0)
                    >= self.SOLAR_CLEANING_DUST_THRESHOLD
                ]
                target_id = target.get("structure_id")
                solar_struct = next(
                    (s for s in eligible_panels if s.get("id") == target_id),
                    None,
                )
                if solar_struct is None and eligible_panels:
                    solar_struct = max(
                        eligible_panels,
                        key=lambda s: s.get("dust_fouling_level", 0.0),
                    )
                if solar_struct:
                    sx, sy = solar_struct.get("x", agent.x), solar_struct.get("y", agent.y)
                    dist = max(abs(agent.x - sx), abs(agent.y - sy))
                    if dist > 1:
                        dx, dy = self._cardinal_step_toward(
                            agent, sx, sy, self.EVA_WALK_SPEED_CELLS
                        )
                        next_x = max(0, min(self.world.map_size - 1, agent.x + dx))
                        next_y = max(0, min(self.world.map_size - 1, agent.y + dy))
                        if (
                            agent._in_habitat
                            and not self._is_pressurized_location(next_x, next_y)
                            and not self._prepare_agent_for_eva(
                                agent, mission_critical_maintenance=True,
                                target_position=(int(sx), int(sy)),
                            )
                        ):
                            return agent_events
                        agent.x = max(0, min(self.world.map_size - 1, agent.x + dx))
                        agent.y = max(0, min(self.world.map_size - 1, agent.y + dy))
                        agent.action.action_type = "move"
                        agent.action.target = {
                            "x": sx,
                            "y": sy,
                            "structure_id": solar_struct.get("id"),
                            "maintenance": "solar_cleaning",
                            "maintenance_action": "clean_solar_panels",
                        }
                        agent.action.ticks_remaining = 1
                    else:
                        # Clean all solar panels in cluster
                        solar_struct["dust_fouling_level"] = 0.0
                        logger.info(f"{agent.name} swept regolith dust off solar photovoltaic array! Power restored to 100%.")
                        if self._on_event:
                            self._on_event({
                                "type": "clean_solar_panels",
                                "agent": agent.name,
                                "cause": f"Swept regolith dust off Solar Array (100% Photovoltaic Efficiency Restored)",
                                "tick": self.current_tick
                            })
                        agent.action.action_type = "clean_solar_panels"
                        agent.action.ticks_remaining = 3
                else:
                    # Another crew member may have cleaned it while this
                    # decision was queued. Do not walk to a clean array.
                    agent.action.action_type = "rest"
                    agent.action.target = {"solar_cleaning_not_needed": True}
                    agent.action.ticks_remaining = 1
            elif action == "seal_patch":
                # Apply emergency pressure seal patch
                patch_res = agent.apply_seal_patch()
                logger.info(f"{agent.name} applied emergency seal patch to EVA suit micro-puncture ({patch_res})")
                if self._on_event:
                    self._on_event({
                        "type": "seal_patch",
                        "agent": agent.name,
                        "cause": f"Applied emergency sealant patch to suit micro-puncture ({agent.suit_integrity:.0%} integrity)",
                        "tick": self.current_tick
                    })
                agent.action.action_type = "seal_patch"
                agent.action.ticks_remaining = 2
            elif action == "recycle_scrap":
                # Smelt broken tool scrap and debris back into refined metal ingots
                scrap_amt = agent.inventory.materials.get("scrap_metal", 0)
                basalt_amt = agent.inventory.materials.get("basalt_scrap", 0)
                if scrap_amt >= 2:
                    agent.inventory.materials["scrap_metal"] -= 2
                    if agent.inventory.materials["scrap_metal"] <= 0:
                        del agent.inventory.materials["scrap_metal"]
                    agent.inventory.materials["reduced_iron_ingot"] = agent.inventory.materials.get("reduced_iron_ingot", 0) + 1
                    logger.info(f"{agent.name} recycled 2x scrap_metal into 1x reduced_iron_ingot in furnace!")
                    if self._on_event:
                        self._on_event({
                            "type": "recycle_scrap",
                            "agent": agent.name,
                            "cause": "Recycled broken tool scrap into 1x Refined Metal Ingot",
                            "tick": self.current_tick
                        })
                    agent.action.action_type = "recycle_scrap"
                    agent.action.ticks_remaining = 2
                elif basalt_amt >= 2:
                    agent.inventory.materials["basalt_scrap"] -= 2
                    if agent.inventory.materials["basalt_scrap"] <= 0:
                        del agent.inventory.materials["basalt_scrap"]
                    agent.inventory.materials["basalt"] = agent.inventory.materials.get("basalt", 0) + 2
                    logger.info(f"{agent.name} crushed and repurposed basalt scrap!")
                    agent.action.action_type = "recycle_scrap"
                    agent.action.ticks_remaining = 2
                else:
                    agent.action.action_type = "idle"
                    agent.action.ticks_remaining = 1
            elif action in ("communicate", "talk", "social"):
                # Comms & Radio Relay Network (Blackout if out of range without beacon)
                message_type = target.get("message_type", "info")
                has_comms_beacons = any(
                    s.get("type") in ("trail_marker", "comms_beacon")
                    and not s.get("under_construction", False)
                    and not s.get("destroyed", False)
                    for s in getattr(self, "placed_structures", [])
                )
                if self._communications_array_operational():
                    base_comm_dist = 200
                    network_label = "Powered Orbital Array"
                elif self._communication_relay_operational():
                    base_comm_dist = 100
                    network_label = "Powered Surface Relay"
                elif has_comms_beacons:
                    base_comm_dist = 25
                    network_label = "Trail Beacon Network"
                else:
                    base_comm_dist = 12
                    network_label = "Direct UHF"
                
                nearby_agents = [
                    other for other in self.agents
                    if other.id != agent.id
                    and getattr(other.status, 'value', str(other.status)) != 'dead'
                    and max(abs(other.x - agent.x), abs(other.y - agent.y)) <= base_comm_dist
                ]
                
                for other in nearby_agents:
                    # Transfer knowledge: explored cells
                    other.explored_cells.update(agent.explored_cells)
                    # Update trust
                    old_trust = other.trust_scores.get(agent.id, 0.0)
                    other.trust_scores[agent.id] = min(1.0, old_trust + 0.05)
                    agent.trust_scores[other.id] = min(1.0, agent.trust_scores.get(other.id, 0.0) + 0.05)
                
                if self._on_event:
                    self._on_event({
                        "type": "communicate",
                        "agent": agent.name,
                        "cause": f"Shared telemetry over {network_label} with {len(nearby_agents)} crew ({message_type})",
                        "tick": self.current_tick
                    })
                agent.action.action_type = "communicate"
                agent.action.ticks_remaining = 2
            elif action == "farm":
                # Automated lighting/irrigation may sustain crops, but a human
                # must inspect each biological loop and physically harvest /
                # pack ripe biomass. This is the sole authoritative food path;
                # the removed legacy state used a five-hour crop and loose
                # inventory "food" units unrelated to the calorie ledger.
                greenhouses = self._connected_life_support_structures("greenhouse")
                requested_id = str(target.get("greenhouse_id") or "")
                target_gh = next((
                    greenhouse for greenhouse in greenhouses
                    if str(greenhouse.get("id")) == requested_id
                ), None)
                if target_gh is None and not requested_id and greenhouses:
                    target_gh = min(
                        greenhouses,
                        key=lambda greenhouse: (
                            max(
                                abs(int(greenhouse.get("x", agent.x)) - agent.x),
                                abs(int(greenhouse.get("y", agent.y)) - agent.y),
                            ),
                            str(greenhouse.get("id", "")),
                        ),
                    )
                if target_gh is None:
                    agent.action.action_type = "invalid_action_rejected"
                    agent.action.target = {
                        "invalid_action_rejected": True,
                        "reason": "greenhouse_unavailable",
                        "greenhouse_id": requested_id,
                    }
                    agent.action.ticks_remaining = 1
                else:
                    gh_id = str(target_gh.get("id"))
                    gh_x = int(target_gh.get("x", agent.x))
                    gh_y = int(target_gh.get("y", agent.y))
                    farm_target = {
                        **target,
                        "greenhouse_id": gh_id,
                        "x": gh_x,
                        "y": gh_y,
                        "destination": "greenhouse_workstation",
                    }
                    if (agent.x, agent.y) != (gh_x, gh_y):
                        agent.action.action_type = "move"
                        agent.action.target = farm_target
                        agent.action.ticks_remaining = 1
                    else:
                        effects = (
                            (self._get_recipe("greenhouse") or {})
                            .get("output", {}).get("effects", {})
                        )
                        target_gh["pressurized"] = True
                        agent._in_habitat = True
                        sub_action = str(
                            target.get("sub_action", "crop_care")
                        ).lower()
                        if sub_action in {"tend", "plant", "water"}:
                            sub_action = "crop_care"
                        if sub_action == "crop_care":
                            target_gh["last_crop_service_tick"] = self.current_tick
                            target_gh["crop_service_due"] = False
                            target_gh["crop_service_overdue"] = False
                            labor_minutes = float(effects.get(
                                "crop_service_labor_minutes", 30.0
                            ))
                            event_cause = (
                                f"{agent.name} inspected crop health, nutrient "
                                f"chemistry and irrigation in greenhouse {gh_id}"
                            )
                        elif sub_action == "harvest":
                            ripe_kcal = max(0.0, float(target_gh.get(
                                "unharvested_food_kcal", 0.0
                            )))
                            storage_capacity = max(
                                0.0,
                                float(getattr(
                                    self, "_food_storage_capacity_kcal", 0.0
                                )),
                            )
                            storage_headroom = max(
                                0.0,
                                storage_capacity - float(
                                    self._colony_resources.get(
                                        "food_reserve_kcal", 0.0
                                    )
                                ),
                            )
                            harvested_kcal = min(ripe_kcal, storage_headroom)
                            if harvested_kcal <= 0.0:
                                agent.action.action_type = (
                                    "invalid_action_rejected"
                                )
                                agent.action.target = {
                                    **farm_target,
                                    "invalid_action_rejected": True,
                                    "reason": (
                                        "no_ripe_crop" if ripe_kcal <= 0.0
                                        else "food_storage_full"
                                    ),
                                }
                                agent.action.ticks_remaining = 1
                                return agent_events
                            target_gh["unharvested_food_kcal"] = (
                                ripe_kcal - harvested_kcal
                            )
                            target_gh["last_crop_harvest_tick"] = self.current_tick
                            target_gh["harvest_due"] = bool(
                                target_gh["unharvested_food_kcal"] > 0.0
                            )
                            self._colony_resources["food_reserve_kcal"] = (
                                float(self._colony_resources.get(
                                    "food_reserve_kcal", 0.0
                                )) + harvested_kcal
                            )
                            self._greenhouse_harvested_kcal_this_tick = (
                                float(getattr(
                                    self,
                                    "_greenhouse_harvested_kcal_this_tick",
                                    0.0,
                                )) + harvested_kcal
                            )
                            labor_minutes = float(effects.get(
                                "harvest_labor_minutes", 30.0
                            ))
                            event_cause = (
                                f"{agent.name} harvested and packed "
                                f"{harvested_kcal:.0f} kcal from greenhouse {gh_id}"
                            )
                        else:
                            agent.action.action_type = "invalid_action_rejected"
                            agent.action.target = {
                                **farm_target,
                                "invalid_action_rejected": True,
                                "reason": "unknown_greenhouse_sub_action",
                            }
                            agent.action.ticks_remaining = 1
                            return agent_events

                        agent.action.action_type = "farm"
                        agent.action.target = {
                            **farm_target,
                            "sub_action": sub_action,
                        }
                        agent.action.ticks_remaining = ticks_for_minutes(
                            labor_minutes, self.SIM_MINUTES_PER_TICK
                        )
                        logger.info(event_cause)
                        if self._on_event:
                            self._on_event({
                                "type": "greenhouse_crop_care"
                                if sub_action == "crop_care"
                                else "greenhouse_harvest",
                                "agent": agent.name,
                                "greenhouse_id": gh_id,
                                "cause": event_cause,
                                "tick": self.current_tick,
                            })
            elif action == "continue":
                # Continue active ongoing task
                pass
            elif action == "unconscious":
                # Incapacitated agent remains immobile on current tile awaiting SAR rescue
                agent.action.action_type = "unconscious"
                agent.action.ticks_remaining = 1
            elif action in ("dead", "death"):
                agent.action.action_type = "dead"
                agent.action.ticks_remaining = 0
            elif action == "idle":
                # Idle is not a field mission. Return an outdoor agent to the
                # pressurised base instead of silently creating exploration.
                if not agent._in_habitat:
                    lz_x = getattr(self, "lz_x", agent.spawn_x)
                    lz_y = getattr(self, "lz_y", agent.spawn_y)
                    dx, dy = self._cardinal_step_toward(agent, lz_x, lz_y)
                    agent.x = max(0, min(self.world.map_size - 1, agent.x + dx))
                    agent.y = max(0, min(self.world.map_size - 1, agent.y + dy))
                    agent.action.action_type = "move"
                    agent.action.target = {
                        "x": lz_x,
                        "y": lz_y,
                        "destination": "shelter",
                        "idle_return": True,
                    }
                else:
                    agent.action.action_type = "rest"
                    agent.action.target = dict(target)
                agent.action.ticks_remaining = 1
            else:
                logger.warning(f"Unknown action: {action}")
                agent._rl_transition_pending = False
                agent._last_invalid_order = {
                    "reason": "unknown_action",
                    "tick": self.current_tick,
                    "action": str(action),
                    "target": dict(target),
                }
                agent.action.action_type = "invalid_action_rejected"
                agent.action.target = {
                    "invalid_action_rejected": True,
                    "reason": "unknown_action",
                    "requested_action": str(action),
                }
                agent.action.ticks_remaining = 1

        # A lander crossing requires a completed FIFO pressure cycle, even if
        # an older action implementation changed the habitat flag itself.
        rover_state = getattr(agent, "_active_expedition", None)
        managed_rover = isinstance(rover_state, dict) and rover_state.get("kind") == "construction_support"
        was_in_lander = self._is_lander_footprint_cell(*position_before_action)
        now_in_lander = self._is_lander_footprint_cell(agent.x, agent.y)
        if not managed_rover and getattr(agent, "_managed_airlock_tick", -1) != self.current_tick:
            crossing_out = started_action_pressurized and was_in_lander and not now_in_lander
            crossing_in = not started_action_pressurized and not was_in_lander and now_in_lander
            if (crossing_out or crossing_in) and not self._airlock_pass([agent], "out" if crossing_out else "in"):
                agent.x, agent.y = position_before_action
                agent._in_habitat = started_action_pressurized
                return agent_events

        # Every movement implementation shares one physical airlock boundary.
        # Some gather routes use a two-cell stride and previously jumped from
        # the base centre directly outdoors, bypassing suit service/preflight.
        # Roll the step back when preflight schedules service or denies EVA.
        if (
            started_action_pressurized
            and agent._in_habitat
            and not self._is_pressurized_location(agent.x, agent.y)
        ):
            outgoing_target = (
                agent.action.target
                if isinstance(getattr(agent.action, "target", None), dict)
                else {}
            )
            mission_critical = bool(
                outgoing_target.get("maintenance")
                or outgoing_target.get("life_support_bootstrap")
            )
            if not self._prepare_agent_for_eva(
                agent, mission_critical_maintenance=mission_critical
            ):
                agent.x, agent.y = position_before_action
                return agent_events

        # Motor-level invariant: no action branch may accidentally step an
        # agent farther beyond the EVA boundary. Existing out-of-bounds saves
        # are allowed to walk inward; new outward movement is rolled back and
        # converted into an explicit return route.
        if getattr(agent.status, "value", str(agent.status)) not in ("dead", "unconscious"):
            old_dist = self._distance_from_lz(*position_before_action)
            new_dist = self._distance_from_lz(agent.x, agent.y)
            allowed_radius = self._agent_eva_radius_cells(agent)
            if new_dist > allowed_radius and new_dist > old_dist:
                agent.x, agent.y = position_before_action
                lz_x = getattr(self, "lz_x", getattr(agent, "spawn_x", 1000))
                lz_y = getattr(self, "lz_y", getattr(agent, "spawn_y", 1000))
                agent.action.action_type = "move"
                agent.action.target = {
                    "x": lz_x, "y": lz_y, "destination": "shelter",
                    "forced_return": True,
                }
                agent.action.ticks_remaining = 1

        return agent_events

    def _get_recipe(self, recipe_name: str) -> Optional[dict]:
        """Get recipe definition from in-memory cache."""
        if not hasattr(self, "_recipes_cache") or not self._recipes_cache:
            recipes_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config', 'recipes.json'
            )
            try:
                if os.path.exists(recipes_path):
                    with open(recipes_path, 'r', encoding='utf-8') as f:
                        self._recipes_cache = normalize_tick_based_config(
                            json.load(f).get("recipes", {}),
                            self.mission_profile.clock.tick_minutes,
                        )
            except Exception:
                self._recipes_cache = {}
        return self._recipes_cache.get(recipe_name)
    
    # ================================================================
    # DAY/NIGHT CYCLE
    # ================================================================
    
    def get_day_night_phase(self) -> dict:
        """Return the world's authoritative local solar phase at the base.

        ``WorldGenerator`` already applies this same phase to cell light and
        temperature. Reusing it here keeps UI light, photovoltaic output,
        EVA visibility and thermal physics on one astronomical clock. The
        zero temperature modifier is intentional: cell temperature already
        contains the diurnal response and must not be cooled a second time.
        """
        if self.planet.tidally_locked:
            light = self.world.get_tidal_zone_light(
                int(getattr(self, "lz_x", self.world.center)),
                int(getattr(self, "lz_y", self.world.center)),
            )
        else:
            light = self.world.get_day_phase(self.current_tick)
        intensity = max(0.0, min(1.0, float(
            light.get("light_level", 1.0)
        )))
        return {
            "phase": str(light.get(
                "phase_name", "day" if intensity > 0.0 else "night"
            )),
            "solar_intensity": intensity,
            "temp_modifier_c": 0.0,
            "gather_efficiency": 0.5 + 0.5 * intensity,
        }
    
    # ================================================================
    # STRUCTURE DEGRADATION
    # ================================================================
    
    def _tick_structure_degradation(self):
        """Apply spatial hazard damage and track condition-based maintenance.

        ``interval_ticks`` is an inspection/service interval.  The former
        implementation subtracted health from tick zero and destroyed a solar
        array or workshop in roughly two weeks even in perfect weather.  A
        missed inspection now raises the damage multiplier of a *real* hazard;
        elapsed calendar time alone never destroys hardware.
        """

        health_by_type: dict[str, list[float]] = {}
        warned_types: set[str] = set()
        for structure in getattr(self, "placed_structures", []):
            if structure.get("under_construction", False) or structure.get("destroyed", False):
                continue

            struct_name = structure.get("type", "")
            recipe = self._get_recipe(struct_name)
            maintenance = recipe.get("maintenance", {}) if recipe else {}
            interval = int(maintenance.get("interval_ticks", 0) or 0)
            missed_loss = float(
                maintenance.get("degradation_per_missed_maintenance", 0.0) or 0.0
            )
            service_anchor = int(structure.get(
                "last_maintenance_tick",
                structure.get("completed_tick", structure.get("built_tick", 0)),
            ) or 0)
            elapsed_since_service = max(0, self.current_tick - service_anchor)
            overdue_cycles = (
                elapsed_since_service // interval if interval > 0 else 0
            )
            structure["maintenance_due"] = bool(
                interval > 0 and elapsed_since_service >= interval
            )
            structure["maintenance_overdue_cycles"] = int(overdue_cycles)

            hazard_effects = self.event_scheduler.get_combined_effects(
                self.current_tick,
                int(structure.get("x", 0)),
                int(structure.get("y", 0)),
            )
            hazard_damage = max(
                0.0,
                float(hazard_effects.get("structure_damage_per_tick", 0.0)),
            )
            overdue_damage_multiplier = min(
                3.0,
                1.0 + max(0, overdue_cycles - 1) * missed_loss,
            )
            old_health = float(structure.get("health", 1.0))
            health = max(
                0.0,
                old_health - hazard_damage * overdue_damage_multiplier,
            )
            structure["health"] = health

            if old_health >= 0.5 > health and struct_name not in warned_types:
                warned_types.add(struct_name)
                logger.warning(f"Structure {struct_name} at 50% health — needs maintenance!")
                if self._on_event:
                    self._on_event({
                        "type": "structure_warning",
                        "agent": "SYSTEM",
                        "cause": f"{struct_name.replace('_', ' ').title()} at 50% integrity — maintenance required!",
                        "tick": self.current_tick,
                    })
            if health <= 0.0:
                structure["destroyed"] = True
                structure["operational"] = False
                self.structures_built[struct_name] = max(
                    0, self.structures_built.get(struct_name, 1) - 1
                )
                logger.warning(f"STRUCTURE FAILED: {struct_name} after environmental damage!")
                if self._on_event:
                    self._on_event({
                        "type": "structure_collapse",
                        "agent": "SYSTEM",
                        "cause": f"{struct_name.replace('_', ' ').title()} failed after environmental damage!",
                        "tick": self.current_tick,
                    })
                continue

            health_by_type.setdefault(struct_name, []).append(health)

        active_types = set(health_by_type)
        for struct_name, healths in health_by_type.items():
            self.structure_health[struct_name] = sum(healths) / len(healths)
        for struct_name in list(self.structure_health):
            if self.structures_built.get(struct_name, 0) <= 0 and struct_name not in active_types:
                self.structure_health.pop(struct_name, None)

    def repair_structure(
        self, agent: Agent, struct_name: str, structure_id: Optional[str] = None,
        inspection_only: bool = False,
    ) -> bool:
        """Inspect or repair one physical asset.

        Routine condition inspection consumes crew time but not a fictional
        replacement-part bundle.  Corrective repair still consumes the exact
        recipe-defined spares before restoring damage.
        """
        candidates = [
            structure for structure in getattr(self, "placed_structures", [])
            if structure.get("type") == struct_name
            and not structure.get("destroyed", False)
            and not structure.get("under_construction", False)
            and (structure_id is None or structure.get("id") == structure_id)
        ]
        if not candidates:
            return False

        target = min(candidates, key=lambda item: float(item.get("health", 1.0)))
        if inspection_only:
            target["last_maintenance_tick"] = self.current_tick
            target["maintenance_due"] = False
            target["maintenance_overdue_cycles"] = 0
            logger.info(f"{agent.name} inspected {struct_name}; condition verified")
            return True

        recipe = self._get_recipe(struct_name) or {}
        repair_materials = {
            material: int(quantity)
            for material, quantity in recipe.get("maintenance", {}).get(
                "repair_materials", {}
            ).items()
            if int(quantity) > 0
        }
        if repair_materials and not self._consume_staged_materials(
            repair_materials,
            int(target.get("x", agent.x)),
            int(target.get("y", agent.y)),
        ):
            logger.info(
                "%s could not repair %s: maintenance spares unavailable",
                agent.name,
                struct_name,
            )
            return False

        # Repair amount scales with engineering competency
        repair_amount = 0.15 + 0.03 * getattr(agent.competency, 'engineering', 0)
        target["health"] = min(1.0, float(target.get("health", 1.0)) + repair_amount)
        target["last_maintenance_tick"] = self.current_tick
        peers = [
            float(structure.get("health", 1.0))
            for structure in getattr(self, "placed_structures", [])
            if structure.get("type") == struct_name
            and not structure.get("destroyed", False)
            and not structure.get("under_construction", False)
        ]
        self.structure_health[struct_name] = sum(peers) / len(peers)

        logger.info(f"{agent.name} repaired {struct_name} to {target['health']:.0%}")
        return True
    
    # ================================================================
    # END CONDITIONS
    # ================================================================

    def _finalize_terminal_learning(self) -> None:
        """Apply one auditable episode outcome and persist its policies."""
        if self._terminal_learning_finalized or self.end_reason == "running":
            return
        outcome = str(self.end_reason)
        if not self._terminal_reward_applied:
            self.strategic_policy.finish_episode(
                outcome,
                elapsed_days=(
                    self.current_tick
                    * self.mission_profile.clock.tick_minutes
                    / 1440.0
                ),
                category_scores=self.colony_score.get_scores(),
                deadline_readiness=self._strategy_deadline_readiness,
                support_soak_fraction=(self._support_soak_ticks / max(
                    1, self._mission_state()["support_soak"]["required_ticks"]
                )),
                surviving_crew_fraction=(sum(
                    getattr(crew.status, "value", str(crew.status)) != "dead"
                    for crew in self.agents
                ) / max(1, len(self.agents))),
            )

            individual_reward = {
                "colony_ready": 50.0,
                "timeout": -20.0,
                "stagnation": -25.0,
                "all_agents_dead": -50.0,
            }.get(outcome)
            if individual_reward is not None:
                for crew in self.agents:
                    if (
                        outcome != "all_agents_dead"
                        and getattr(crew.status, "value", str(crew.status))
                        == "dead"
                    ):
                        continue
                    self.decision_engine.apply_terminal_rl_reward(
                        crew, individual_reward
                    )
            self._terminal_reward_applied = True

        if getattr(self, "_db", None):
            policies = [
                {
                    "agent_id": crew.id,
                    "q_table": crew.q_table,
                    "total_reward": crew.total_accumulated_reward,
                }
                for crew in self.agents
                if getattr(crew, "q_table", None)
            ]
            if self._strategic_policy_loaded:
                policies.append({
                    "agent_id": self.strategic_policy.persistence_id,
                    "q_table": self.strategic_policy.dump(),
                    "total_reward": self.strategic_policy.total_reward,
                })
            try:
                self._db.save_agent_q_tables(
                    policies, tick=self.current_tick, flush=True
                )
            except Exception as exc:
                logger.warning("Could not persist terminal RL policies: %s", exc)
                return
        self._terminal_learning_finalized = True
    
    def _check_end_conditions(self) -> bool:
        """Check if simulation should end."""
        # All agents dead — Bug Fix #4: use safe enum comparison
        alive = [a for a in self.agents if getattr(a.status, 'value', str(a.status)) != 'dead']
        if not alive:
            self.end_reason = "all_agents_dead"
            self.running = False
            logger.info("SIMULATION END: All agents dead")
            return True
        
        # Civilian landing is accepted only at the full, balanced 106-person
        # capacity contract and after orbital communications are operational.
        mission_state = self._mission_state()
        if mission_state["surface_acceptance_ready"]:
            self.end_reason = "colony_ready"
            self.running = False
            logger.info("SIMULATION END: 106-person surface acceptance complete")
            return True
        
        # Max ticks
        if self.current_tick >= self.max_ticks:
            self.end_reason = "timeout"
            self.running = False
            logger.info(f"SIMULATION END: Max ticks ({self.max_ticks}) reached")
            return True
        
        return False
    
    # ================================================================
    # STATE & REPORTING
    # ================================================================
    
    def _get_state_snapshot(self) -> dict:
        """Get current simulation state."""
        discovered_json = {}
        for res, coords in getattr(self, "discovered_resources", {}).items():
            discovered_json[res] = []
            for coord in coords:
                x, y = coord[0], coord[1]
                geology = self.cell_geology.get((x, y))
                if geology is not None:
                    index = int(geology.get("current_index", 0))
                    layers = geology.get("layers", [])
                    current_layer = layers[index] if 0 <= index < len(layers) else None
                    # Sensor knowledge can guide an agent, but the tactical map
                    # displays only the stratum physically exposed at the face.
                    if current_layer is None or current_layer["material"] != res:
                        continue
                    qty = int(current_layer.get("remaining", 0))
                elif res == "regolith":
                    qty = None
                else:
                    qty = self.cell_resource_units.get((x, y, res))
                discovered_json[res].append([x, y, qty])

        cell_geology_json = []
        for (x, y), geology in self.cell_geology.items():
            index = int(geology.get("current_index", 0))
            layers = geology.get("layers", [])
            current_layer = layers[index] if 0 <= index < len(layers) else None
            next_index = index + 1
            next_layer = (
                layers[next_index] if 0 <= next_index < len(layers) else None
            )
            next_layer_detected = bool(
                next_layer
                and (x, y) in self.discovered_resources.get(
                    next_layer["material"], set()
                )
            )
            cell_geology_json.append({
                "x": x,
                "y": y,
                "excavated_layers": index,
                "test_pit_depth": int(self.cell_excavation_depth.get((x, y), 0)),
                "current_layer": (
                    {
                        "material": current_layer["material"],
                        "remaining": int(current_layer.get("remaining", 0)),
                        "initial_quantity": int(current_layer.get("initial_quantity", 0)),
                        "layer_index": index,
                    }
                    if current_layer else None
                ),
                # The tactical display intentionally exposes only the next
                # detected contact beneath the working face. Deeper sensor
                # knowledge can guide an agent's test pits, but must not turn
                # one scan into an x-ray view of the whole column.
                "detected_next_layer": (
                    {
                        "material": next_layer["material"],
                        "layer_index": next_index,
                        "relative_order": 1,
                    }
                    if next_layer_detected else None
                ),
                "detected_below_count": 1 if next_layer_detected else 0,
            })

        spoil_piles_json = [
            {
                "x": x,
                "y": y,
                "materials": dict(pile),
                "stack_order": list(pile.keys()),
                "total_units": int(sum(pile.values())),
            }
            for (x, y), pile in self.spoil_piles.items()
            if any(quantity > 0 for quantity in pile.values())
        ]
        return {
            "engine_id": self.engine_id,
            "tick": self.current_tick,
            "planet": self.planet.name,
            "planet_model_provenance": dict(
                self.planet.data.get("epistemic_model", {})
            ),
            # Mission contract is available to public telemetry and the
            # private ICARUS inspection view.
            "mission": {
                **self._mission_state(),
                "cargo_mass_ledger": self.cargo_mass_ledger,
                "delivered_contingency_feedstocks": dict(
                    self.delivered_contingency_feedstocks
                ),
                "delivered_structure_kits_remaining": sorted(
                    getattr(self, "delivered_structure_kits", {})
                ),
            },
            "surface_fleet": self.surface_fleet.to_dict(),
            "airlock": self.airlock.snapshot(),
            "utility_network": self._life_support_network_snapshot(),
            "construction_logistics": {
                site_id: {
                    **reservation,
                    "at_depot": dict(reservation.get("at_depot", {})),
                    "in_transit": dict(reservation.get("in_transit", {})),
                    "delivered": dict(reservation.get("delivered", {})),
                    "required": dict(reservation.get("required", {})),
                }
                for site_id, reservation in sorted(
                    self._construction_cargo_reservations.items()
                )
            },
            "landing_zone": {"x": getattr(self, "lz_x", 1000), "y": getattr(self, "lz_y", 1000)},
            "exploration_ranges": {
                "local_eva_radius": self.LOCAL_EVA_RADIUS_CELLS,
                "survey_radius": self._survey_radius_cells(),
                "communication_radius": self._communication_radius_cells(),
                "infrastructure_expedition_radius": self._infrastructure_expedition_radius_cells(),
            },
            "discovered_resources": discovered_json,
            "cell_geology": cell_geology_json,
            "spoil_piles": spoil_piles_json,
            "shared_work_order": dict(getattr(self.decision_engine, "shared_work_order", {})),
            # Backend research telemetry; the challenge UI remains intentionally
            # deferred until the learning protocol is validated.
            "strategic_rl": self.strategic_policy.telemetry(),
            "movement_effort_rl": dict(self._movement_effort_telemetry),
            "surface_mobility": {
                "routine_nominal_cells_per_tick": self.EVA_WALK_SPEED_CELLS,
                "expedition_nominal_cells_per_tick": (
                    self.EXPEDITION_MOVE_SPEED_CELLS
                ),
                "grid_cell_meters": self.mission_profile.grid_cell_meters,
                "surge_speed_multiplier": (
                    self.MOVEMENT_SURGE_SPEED_MULTIPLIER
                ),
                "surge_metabolic_multiplier": (
                    self.MOVEMENT_SURGE_METABOLIC_MULTIPLIER
                ),
            },
            "manufacturing_automation": dict(
                self._manufacturing_automation_telemetry
            ),
            "manufacturing_cycles": {
                machine_id: dict(cycle)
                for machine_id, cycle in sorted(self._manufacturing_cycles.items())
                if cycle.get("completion_pending")
            },
            "agents": [
                {
                    "id": a.id,
                    "name": a.name,
                    "status": a.status.value if hasattr(a.status, 'value') else str(a.status),
                    "death_cause": (a.death_cause.value if hasattr(a.death_cause, 'value') else str(a.death_cause)) if a.death_cause else None,
                    "death_tick": a.death_tick,
                    "position": {"x": a.x, "y": a.y},
                    "in_habitat": bool(getattr(a, "_in_habitat", False)),
                    "anthropometrics": a.anthropometrics.to_dict(),
                    "life_support_profile": a.get_life_support_profile(),
                    "needs": a.needs.to_dict(),
                    "action": a.action.to_dict(),
                    "airlock_wait": (
                        dict(a._airlock_wait)
                        if getattr(a, "_airlock_wait", None)
                        and self.current_tick - a._airlock_wait.get("tick", -100) <= 1
                        else None
                    ),
                    "inventory": {
                        "materials": dict(a.inventory.materials),
                        "items": dict(a.inventory.items),
                        "tool_durability": dict(a.inventory.tool_durability),
                        "tool_charge_pct": dict(a.inventory.tool_charge_pct),
                    },
                    "plss": {
                        "o2_canister_pct": round(max(0.0, getattr(a, "_current_canister_remaining", 0.0)), 1),
                        "co2_scrubber_pct": round(getattr(a, "plss_co2_scrubber_pct", 100.0), 1),
                        "suit_battery_pct": round(getattr(a, "plss_suit_battery_pct", 100.0), 1),
                        "suit_integrity_pct": round(getattr(a, "suit_integrity", 1.0) * 100.0, 1),
                        "suit_condition_pct": round(getattr(a, "suit_condition", 1.0) * 100.0, 1),
                        "has_micro_puncture": getattr(a, "has_micro_puncture", False),
                        "hypercapnia_level": round(getattr(a, "hypercapnia_level", 0.0), 2),
                    },
                    "expedition": (
                        dict(a._active_expedition)
                        if isinstance(getattr(a, "_active_expedition", None), dict)
                        else None
                    ),
                    "last_decision": dict(getattr(a, "last_decision", {})),
                    "relationships": [
                        {
                            "agent_id": other.id,
                            "name": other.name,
                            "trust": round(a.trust_scores.get(other.id, 0.0), 3),
                            "source": "simulation_social_graph",
                            "opinion": None,
                        }
                        for other in self.agents
                        if other.id != a.id
                    ],
                    "thoughts": [],
                    "narrative_status": self.llm_client.get_status(),
                    "rl_policy": {
                        "total_reward": round(getattr(a, "total_accumulated_reward", 0.0), 1),
                        "state_key": getattr(a, "last_state_key", "nominal"),
                        "last_action": getattr(a, "last_action_key", "none"),
                        "learned_states": len(getattr(a, "q_table", {})),
                        "reward_history": list(getattr(a, "rl_reward_history", [])),
                        "current_q_values": dict(
                            getattr(a, "q_table", {}).get(
                                getattr(a, "last_state_key", None), {}
                            )
                        ),
                        "epsilon": getattr(a, "rl_epsilon_explore", 0.0),
                        "learning_rate": getattr(a, "rl_learning_rate", 0.0),
                    },
                }
                for a in self.agents
            ],
            "central_depot_inventory": dict(getattr(self, "central_depot_inventory", {})),
            "colony_score": self.colony_score.to_dict(),
            "structures": dict(self.structures_built),
            "placed_structures": [dict(s) for s in getattr(self, "placed_structures", [])],
            "structure_health": {k: round(v, 3) for k, v in self.structure_health.items()},
            "day_night": self.get_day_night_phase(),
            "colony_production": {
                "energy_stored_kwh": round(getattr(self, '_colony_resources', {}).get("energy_stored_kwh", 0), 1),
                "energy_delta": 0.0,
                "power_cycle": {
                    key: round(value, 4) if isinstance(value, float) else value
                    for key, value in getattr(
                        self, "_power_cycle_telemetry", {}
                    ).items()
                },
                "communications_online": self._communications_array_operational(),
                "o2_reserve_kg": round(getattr(self, '_colony_resources', {}).get("o2_reserve_kg", 0), 1),
                "water_reserve_l": round(getattr(self, '_colony_resources', {}).get("water_reserve_l", 0), 1),
                "water_cycle": {
                    **{
                        key: round(value, 4) if isinstance(value, float) else value
                        for key, value in getattr(
                            self, "_water_cycle_telemetry", {}
                        ).items()
                    },
                    **self._water_mass_ledger(),
                },
                "food_reserve_kcal": round(getattr(self, '_colony_resources', {}).get("food_reserve_kcal", 0), 0),
                "base_temperature_c": round(getattr(self, 'base_temperature_c', 21.0), 1),
                "thermal_balance": dict(
                    getattr(self, "_thermal_state", {})
                ),
            },
            "narrative_status": self.llm_client.get_status(),
            "llm_status": self.llm_client.get_status(),
        }
    
    def _get_final_report(self) -> dict:
        """Generate end-of-simulation report."""
        self._finalize_terminal_learning()
        alive = [a for a in self.agents if getattr(a.status, 'value', str(a.status)) != 'dead']
        dead = [a for a in self.agents if getattr(a.status, 'value', str(a.status)) == 'dead']
        
        avg_tick_time = (
            sum(self._tick_times) / len(self._tick_times)
            if self._tick_times else 0
        )
        
        return {
            "engine_id": self.engine_id,
            "end_reason": self.end_reason,
            "total_ticks": self.current_tick,
            "sim_days": round(
                self.current_tick
                / self.mission_profile.clock.ticks_per_earth_day,
                1,
            ),
            "planet": self.planet.name,
            "mission": {
                **self._mission_state(),
                "cargo_mass_ledger": self.cargo_mass_ledger,
                "delivered_contingency_feedstocks": dict(
                    self.delivered_contingency_feedstocks
                ),
                "delivered_structure_kits_remaining": sorted(
                    getattr(self, "delivered_structure_kits", {})
                ),
            },
            "surface_fleet": self.surface_fleet.to_dict(),
            "utility_network": self._life_support_network_snapshot(),
            "construction_logistics": {
                site_id: {
                    "recipe": reservation.get("recipe"),
                    "status": reservation.get("status"),
                    "total_mass_kg": round(
                        float(reservation.get("total_mass_kg", 0.0)), 3
                    ),
                    "delivered_mass_kg": round(
                        float(reservation.get("delivered_mass_kg", 0.0)), 3
                    ),
                    "trips_completed": int(
                        reservation.get("trips_completed", 0)
                    ),
                }
                for site_id, reservation in sorted(
                    self._construction_cargo_reservations.items()
                )
            },
            "colony_score": self.colony_score.to_dict(),
            "strategic_rl": self.strategic_policy.telemetry(),
            "movement_effort_rl": dict(self._movement_effort_telemetry),
            "surface_mobility": {
                "routine_nominal_cells_per_tick": self.EVA_WALK_SPEED_CELLS,
                "expedition_nominal_cells_per_tick": (
                    self.EXPEDITION_MOVE_SPEED_CELLS
                ),
                "grid_cell_meters": self.mission_profile.grid_cell_meters,
                "surge_speed_multiplier": (
                    self.MOVEMENT_SURGE_SPEED_MULTIPLIER
                ),
                "surge_metabolic_multiplier": (
                    self.MOVEMENT_SURGE_METABOLIC_MULTIPLIER
                ),
            },
            "manufacturing_automation": dict(
                self._manufacturing_automation_telemetry
            ),
            "agents_alive": len(alive),
            "agents_dead": len(dead),
            "deaths": [
                {
                    "name": a.name,
                    "cause": (
                        a.death_cause.value
                        if hasattr(a.death_cause, "value")
                        else str(a.death_cause)
                    ),
                    "tick": a.death_tick,
                }
                for a in dead
            ],
            "survivors": [
                {"name": a.name, "ticks_alive": a.ticks_alive}
                for a in alive
            ],
            "structures": self.structures_built,
            "thermal_balance": dict(getattr(self, "_thermal_state", {})),
            "llm_stats": self.llm_client.get_status(),
            "avg_tick_ms": round(avg_tick_time * 1000, 1),
        }
    
    # ================================================================
    # CONTROLS
    # ================================================================
    
    def pause(self):
        self.paused = True
    
    def resume(self):
        self.paused = False
    
    def stop(self):
        if self.end_reason == "running":
            self.end_reason = "manual_stop"
        self._stop_event.set()
        self.running = False
    
    def set_speed(self, multiplier: float):
        """Set development playback speed relative to the 4-second protocol."""
        self.tick_speed = self.TARGET_TICK_SECONDS / max(0.1, multiplier)
