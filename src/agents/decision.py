"""
Decision Engine — Strategic + Tactical Decision Loop.

Architecture:
- Strategic layer: Every 10 ticks (or task completion), LLM picks next goal
  from colony sub-targets. Output: target task + plan steps.
- Tactical layer: Each tick, deterministically executes plan steps.
  LLM called ONLY when: need critical, threat detected, plan impossible,
  agent encounter.

This separation minimizes API calls while maintaining agent intelligence.
Most ticks are pure deterministic execution (< 1ms, no API call).
"""

import json
import hashlib
import logging
import math
import os
from typing import Optional
from dataclasses import dataclass, field

from src.agents.agent import Agent, AgentStatus
from src.agents.prompts import (
    build_system_prompt, build_strategic_prompt,
    build_tactical_prompt, build_social_prompt,
)
from src.orchestration.llm_client import LocalNarrativeClient, LLMCallType
from src.orchestration.fallback import FallbackDecisionEngine, fallback_engine
from src.memory.vector_store import MemoryManager
from src.systems.colony_score import ColonyScore
from src.systems.mission_profile import MissionProfile
from src.systems.timebase import normalize_tick_based_config, ticks_for_minutes
from src.agents.strategic_rl import build_colony_strategy_state

logger = logging.getLogger(__name__)

# Scheduler/config aliases that require an immediate surface return. Keep the
# vocabulary centralized so a generated ``stellar_flare`` cannot bypass a
# policy check written only for the colloquial ``solar_flare`` name.
SURFACE_HAZARD_EVENT_TYPES = frozenset({
    "stellar_flare",
    "solar_flare",
    "flare",
    "dust_storm",
    "sandstorm",
    "meteor_shower",
    "micrometeorite_shower",
})


# ============================================================
# PLAN STEP
# ============================================================

@dataclass
class PlanStep:
    """Single step in a strategic plan."""
    action: str           # move, gather, build, etc.
    target: dict          # Action-specific target params
    description: str      # Human-readable description
    estimated_ticks: int   # Expected duration
    completed: bool = False
    ticks_spent: int = 0
    blocked_until_tick: int = -1
    last_rejection: str = ""


@dataclass
class StrategicPlan:
    """Agent's current strategic plan."""
    goal: str               # High-level goal description
    steps: list[PlanStep]   # Ordered plan steps
    created_tick: int       # When plan was created
    priority: str = "normal"  # critical, high, normal
    
    @property
    def current_step(self) -> Optional[PlanStep]:
        """Get first incomplete step."""
        for step in self.steps:
            if not step.completed:
                return step
        return None
    
    @property
    def is_complete(self) -> bool:
        return all(s.completed for s in self.steps)
    
    @property
    def progress(self) -> float:
        if not self.steps:
            return 1.0
        done = sum(1 for s in self.steps if s.completed)
        return done / len(self.steps)


# ============================================================
# TACTICAL TRIGGER CONDITIONS
# ============================================================

class TacticalTrigger:
    """Determines when a tactical LLM call is needed."""
    
    NEED_CRITICAL = 25.0    # Need value below this triggers LLM
    NEED_WARNING = 40.0     # Below this, flag for awareness
    
    @staticmethod
    def check(agent: Agent, tick_events: dict, has_active_plan: bool) -> Optional[str]:
        """
        Check if tactical LLM call is needed.
        
        Returns trigger reason string, or None if no call needed.
        """
        needs = agent.needs
        
        # Priority 1: Critical survival needs
        if needs.hunger < TacticalTrigger.NEED_CRITICAL:
            return "hunger_critical"
        if needs.thirst < TacticalTrigger.NEED_CRITICAL:
            return "thirst_critical"
        if needs.energy < TacticalTrigger.NEED_CRITICAL:
            return "energy_critical"
        if needs.o2_supply < TacticalTrigger.NEED_CRITICAL:
            return "o2_critical"
        
        # Priority 2: Dangerous warnings
        warnings = tick_events.get("warnings", [])
        critical_warnings = [w for w in warnings if "critical" in str(w).lower() or "DEATH" in str(w)]
        if critical_warnings:
            return f"warning:{critical_warnings[0]}"
        
        # Priority 3: No active task and no plan
        if not agent.action.is_active and not has_active_plan:
            return "task_complete"
        
        # Priority 4: Severe injury
        if agent.injury_level > 0.7:
            return "severe_injury"
        
        # No tactical call needed — continue deterministic execution
        return None


# ============================================================
# DECISION ENGINE
# ============================================================

class DecisionEngine:
    """
    Central decision-making engine for all agents.
    
    Orchestrates:
    - Strategic planning (LLM, every 10 ticks)
    - Tactical decisions (LLM, on-demand triggers)
    - Plan execution (deterministic, every tick)
    - Memory recording (every action)
    - Reflection (LLM, every 20 ticks)
    
    API call budget management:
    - Strategic: 6 agents × 1 call/10 ticks = 0.6 calls/tick
    - Tactical: ~20% of ticks trigger = ~1 call/tick
    - Reflection: 6 agents × 1 call/20 ticks = 0.3 calls/tick
    - Total: ~1.75 calls/tick average → within RPM budget
    """
    
    STRATEGIC_INTERVAL = 20    # Ticks between strategic evaluations (Balanced 20 ticks per user preference)
    REFLECTION_INTERVAL = 60   # Ticks between reflections
    # A processor can release very small batches every tick.  Treating every
    # millilitre as a new full drink order trapped thirsty crew in a perpetual
    # sip loop and prevented physiologically necessary sleep.  250 mL is a
    # meaningful single hydration dose while still allowing partial reserves.
    MIN_MEANINGFUL_DRINK_L = 0.25
    # This is a planning-only bill of materials. It never becomes a physical
    # structure and never earns score. Its direct material quantities are
    # refreshed from the still-unbuilt 106-person mission architecture so the
    # landed workshops can manufacture known deficits before the particular
    # structure that needs them reaches the front of the build queue.
    MISSION_FORECAST_RECIPE = "__mission_capacity_manifest__"
    MISSION_FORECAST_STRUCTURES = (
        "solar_panel", "life_support_distribution_grid",
        "potable_water_tank", "oxygen_buffer_tank",
        "isru_o2_unit", "water_collector", "water_purifier",
        "greenhouse", "habitat_module",
        "power_distribution_grid", "communications_array", "landing_zone",
        "radiator_panel",
    )
    
    def __init__(self, llm_client: LocalNarrativeClient,
                 memory_manager: MemoryManager = None,
                 colony_score: ColonyScore = None,
                 dialogue_generation_enabled: bool = False):
        from src.memory.reflection import ReflectionSystem, SocialGraph
        self.llm = llm_client
        self.memory = memory_manager or MemoryManager()
        self.colony = colony_score or ColonyScore()
        self.social = SocialGraph()
        self.reflection = ReflectionSystem(self.memory, llm_client, self.social)
        self.fallback = fallback_engine
        
        # Per-agent state
        self._plans: dict[str, StrategicPlan] = {}
        # Invalid motor commands invalidate the plan that emitted them.  A
        # short replan cooldown prevents the same stale order being generated
        # on the immediately following tick while deterministic work remains
        # available.
        self._plan_retry_after_tick: dict[str, int] = {}
        self._system_prompts: dict[str, str] = {}
        self._last_strategic_tick: dict[str, int] = {}
        self._last_reflection_tick: dict[str, int] = {}
        self._last_tactical_tick: dict[str, int] = {}
        
        # Stats
        self._strategic_calls = 0
        self._tactical_calls = 0
        self._deterministic_ticks = 0
        self._fallback_decisions = 0
        # Generative language is non-authoritative.  The physics kernel and
        # RL policy select actions from explicit masks; the LLM may still
        # produce crew dialogue, but never strategic/tactical state changes.
        self.language_planning_enabled = False
        # Narrative calls are opt-in so headless audits and unit tests remain
        # hermetic and cannot consume a live provider quota. The API may turn
        # this on explicitly for public, non-authoritative dialogue streams.
        self.dialogue_generation_enabled = bool(dialogue_generation_enabled)
        self.robot_dispatch_targets: dict[str, dict] = {}
        # One mission-level order is shared by the whole crew. It is rebuilt
        # from live depot/site state, but kept here for telemetry and stable
        # role assignment instead of letting six independent RL policies
        # invent unrelated construction goals.
        self.shared_work_order: dict = {}
        mission_profile = MissionProfile.load()
        self.tick_minutes = mission_profile.clock.tick_minutes
        self.ticks_per_hour = 60.0 / self.tick_minutes
        self.mission_attempt_max_ticks = (
            mission_profile.clock.planet_attempt_max_ticks
        )
        self.infrastructure_deadline_tick = ticks_for_minutes(
            float(mission_profile.arrival_contract.get("infrastructure_complete_by_day", 330))
            * 1440.0, self.tick_minutes
        )
        self.production_policy = dict(mission_profile.production_policy)
        self.finite_cargo_materials = {
            str(material) for material in self.production_policy.get(
                "delivered_only_materials", []
            )
        }
        self.severe_radiation_environment = False

        recipes_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config", "recipes.json",
        )
        try:
            with open(recipes_path, "r", encoding="utf-8") as recipe_file:
                raw_recipes = json.load(recipe_file).get("recipes", {})
                self._recipes_cache = normalize_tick_based_config(
                    raw_recipes, self.tick_minutes
                )
        except (OSError, ValueError):
            self._recipes_cache = {}
    
    def init_agent(self, agent: Agent):
        """Initialize decision engine state for a new agent."""
        self._system_prompts[agent.id] = build_system_prompt(agent)
        self._last_strategic_tick[agent.id] = 0  # Staggered evaluation start
        self._last_reflection_tick[agent.id] = 0

    def _hydration_source_available(
        self,
        agent: Agent,
        *,
        near_shelter: bool,
        colony_resources: Optional[dict] = None,
    ) -> bool:
        """Return whether one physically meaningful drink is accessible."""
        if agent.inventory.has_item("water_packs"):
            return True
        if not near_shelter:
            return False
        if getattr(self, "central_depot_inventory", {}).get("water_packs", 0) > 0:
            return True
        resources = (
            getattr(self, "_colony_resources", {})
            if colony_resources is None else colony_resources
        )
        return float(resources.get("water_reserve_l", 0.0)) >= (
            self.MIN_MEANINGFUL_DRINK_L
        )

    @staticmethod
    def sar_recovery_required(agent: Agent) -> bool:
        """Do not keep cancelling the recovery required by the EVA interlock."""
        pending = getattr(agent, "_pending_indoor_activity", None) or {}
        targets = [getattr(agent.action, "target", {}), pending.get("target", {})]
        return any(
            isinstance(t, dict)
            and (t.get("preflight_recovery") or t.get("eva_denied") == "insufficient_round_trip_energy")
            and agent.needs.energy < max(72.0, float(t.get("required_energy_pct", 72.0)))
            for t in targets
        )

    def select_sar_rescuer_id(self, victim: Agent, agents: list[Agent] = None) -> Optional[str]:
        """Select one conscious, medically fit SAR lead for a victim."""
        eligible_rescuers = []
        for candidate in agents or getattr(self, "agents", []):
            hydration_ready = (
                candidate.needs.thirst > 45.0
                or candidate.inventory.has_item("water_packs")
            )
            nutrition_ready = (
                candidate.needs.hunger > 35.0
                or candidate.inventory.has_item("emergency_rations")
                or candidate.inventory.has_item("ration_pack")
            )
            if (
                candidate.id != victim.id
                and getattr(candidate.status, 'value', str(candidate.status)) not in ('dead', 'incapacitated')
                and candidate.needs.o2_supply >= 45.0
                and candidate.needs.temperature_stress >= 35.0
                and candidate.needs.energy >= 25.0
                and hydration_ready
                and nutrition_ready
                and candidate.injury_level < 0.45
            ):
                distance = max(abs(candidate.x - victim.x), abs(candidate.y - victim.y))
                medical = getattr(candidate.competency, "medical", 5)
                strength = getattr(candidate.genome, "strength", 5)
                score = medical * 3.0 + strength * 1.5 + candidate.needs.energy * 0.1 - distance * 2.5
                ready = not self.sar_recovery_required(candidate) and (
                    not candidate._in_habitat or victim._in_habitat or candidate.needs.energy >= 72.0
                )
                eligible_rescuers.append((ready, score, candidate.id))
        if not eligible_rescuers:
            return None
        return max(eligible_rescuers, key=lambda item: (item[0], item[1]))[2]

    def _recall_expedition_team(self, agent: Agent):
        """Mark both expedition members returning and interrupt field sleep."""
        expedition = getattr(agent, "_active_expedition", None)
        if not isinstance(expedition, dict):
            return
        expedition["status"] = "returning"
        partner_id = (
            expedition.get("lead_id")
            if expedition.get("role") == "buddy"
            else expedition.get("buddy_id")
        )
        partner = next((
            crew for crew in getattr(self, "agents", [])
            if crew.id == partner_id
        ), None)
        partner_state = getattr(partner, "_active_expedition", None) if partner else None
        if isinstance(partner_state, dict):
            partner_state["status"] = "returning"
            if (
                not getattr(partner, "_in_habitat", False)
                and getattr(partner.action, "action_type", "") in ("sleep", "rest")
            ):
                partner.action.clear()

    def _balanced_milestone_candidate(
        self, agent: Agent, colony_mats: dict, structures_built: dict
    ) -> Optional[dict]:
        """Return the next recipe- and capacity-driven infrastructure action."""
        capacity_order = (
            "solar_panel", "life_support_distribution_grid",
            "potable_water_tank", "oxygen_buffer_tank",
            "isru_o2_unit", "water_collector", "water_purifier",
            "greenhouse", "habitat_module",
        )
        placed_structures = getattr(self, "placed_structures", [])

        # Finish an existing core construction site before opening another
        # site. This is material-efficient and lets nearby crew assist.
        active_site = next((
            structure for structure in placed_structures
            if structure.get("type") in capacity_order
            and structure.get("under_construction", False)
            and not structure.get("destroyed", False)
        ), None)
        if active_site is not None:
            if not active_site.get("materials_committed", True):
                return None
            recipe_name = active_site["type"]
            status = self.colony.get_structure_capacity_status(recipe_name) or {}
            return {
                "action": "build",
                "target": {
                    "recipe": recipe_name,
                    "struct_id": active_site.get("id"),
                    "capacity_target_count": status.get("required_count"),
                    "capacity_category": status.get("category"),
                    "life_support_bootstrap": (
                        getattr(self, "surface_requires_plss", False)
                        and structures_built.get("isru_o2_unit", 0) == 0
                        and recipe_name in ("solar_panel", "isru_o2_unit")
                    ),
                },
                "reasoning": f"Completing active {recipe_name} capacity module",
                "deterministic": True,
            }

        # The shared planet policy owns strategic order.  This compatibility
        # path is still used to populate an individual agent's validated
        # action candidates, but it must not reintroduce a deterministic
        # solar→water→oxygen script behind the strategic learner.
        order = self._select_shared_capacity_order(
            colony_mats,
            structures_built,
            tick=getattr(self, "current_tick", 0),
        )
        if order is None:
            return None
        recipe_name = order["recipe"]

        recipe = self._recipes_cache.get(recipe_name, {})
        recipe_mats = self._materials_available_for_recipe(
            recipe_name, colony_mats, structures_built
        )
        status = self.colony.get_structure_capacity_status(recipe_name) or {}
        target_metadata = {
            "capacity_target_count": status.get("required_count"),
            "capacity_category": status.get("category"),
            "capacity_current": status.get("current_capacity"),
            "capacity_target": status.get("target_capacity"),
            "life_support_bootstrap": (
                getattr(self, "surface_requires_plss", False)
                and structures_built.get("isru_o2_unit", 0) == 0
                and recipe_name in ("solar_panel", "isru_o2_unit")
            ),
        }
        requirements = recipe.get("materials", {})
        missing = next((
            (material, needed - recipe_mats.get(material, 0))
            for material, needed in requirements.items()
            if recipe_mats.get(material, 0) < needed
        ), None)
        if missing is None:
            return {
                "action": "build",
                "target": {"recipe": recipe_name, **target_metadata},
                "reasoning": (
                    f"Capacity plan: construct {recipe_name} module "
                    f"{structures_built.get(recipe_name, 0) + 1}/"
                    f"{status.get('required_count', '?')} for 100 colonists"
                ),
                "deterministic": True,
            }

        decision = self._material_dependency_action(
            missing[0], missing[1], recipe_name, recipe_mats, set()
        )
        decision.setdefault("target", {}).update(target_metadata)
        decision["target"]["capacity_recipe"] = recipe_name
        return decision

    def _is_infrastructure_planner(self, agent: Agent) -> bool:
        """Reserve strategic construction planning for the two best engineers."""
        living = [
            crew for crew in getattr(self, "agents", [agent])
            if getattr(crew.status, "value", str(crew.status)) != "dead"
        ]
        ranked = sorted(
            living,
            key=lambda crew: (
                -int(getattr(crew.competency, "engineering", 0)),
                str(crew.id),
            ),
        )
        return agent.id in {crew.id for crew in ranked[:2]}

    def _pooled_materials(self) -> dict[str, int]:
        """Materials physically carried or staged in the central depot."""
        pooled = dict(getattr(self, "central_depot_inventory", {}))
        for crew in getattr(self, "agents", []):
            if getattr(crew.status, "value", str(crew.status)) == "dead":
                continue
            for material, quantity in crew.inventory.materials.items():
                pooled[material] = pooled.get(material, 0) + quantity
        return pooled

    def _materials_available_for_recipe(
        self,
        recipe_name: str,
        pooled: Optional[dict] = None,
        structures_built: Optional[dict] = None,
    ) -> dict[str, int]:
        """Hide delivered structure-kit stock from every other recipe.

        The physical kit remains visible in central-depot telemetry.  Its BOM is
        therefore already present in ``pooled`` and must not be added twice.
        Until the intended structure is built, subtract that reserved floor
        from every other recipe's planning view.  Older saves have no
        ``delivered_structure_kits`` field and retain previous behavior.
        """
        available = dict(self._pooled_materials() if pooled is None else pooled)
        if recipe_name == self.MISSION_FORECAST_RECIPE:
            # The aggregate manifest includes the structure assigned to every
            # delivered kit. Count those sealed parts once against that full
            # future demand; the normal one-structure view below must continue
            # hiding them from unrelated construction orders.
            return available
        built = (
            getattr(self, "structures_built", {})
            if structures_built is None else structures_built
        )
        kits = getattr(self, "delivered_structure_kits", {})
        if not isinstance(kits, dict):
            return available
        for kit_recipe, kit in kits.items():
            if (
                kit_recipe == recipe_name
                or int(built.get(kit_recipe, 0)) > 0
                or not isinstance(kit, dict)
                or kit.get("consumed", False)
            ):
                continue
            kit_materials = kit.get("materials", kit)
            if not isinstance(kit_materials, dict):
                continue
            numeric_kit_materials = {
                material: int(quantity)
                for material, quantity in kit_materials.items()
                if material not in {"consumed", "reserved", "kit_id"}
                and isinstance(quantity, (int, float))
                and quantity > 0
            }
            # Reservation metadata cannot claim unrelated replacement stock
            # after a kit has physically ceased to be complete (legacy saves,
            # test fixtures or damage recovery). In normal operation the
            # engine's atomic reservation prevents this condition entirely.
            if not all(
                int(available.get(material, 0)) >= quantity
                for material, quantity in numeric_kit_materials.items()
            ):
                continue
            for material, quantity in numeric_kit_materials.items():
                available[material] = max(
                    0, int(available.get(material, 0)) - quantity
                )
        return available

    def _all_machine_slots_busy(
        self,
        machine_type: str,
        living: list[Agent],
        requesting_agent: Agent,
        tick: Optional[int] = None,
    ) -> bool:
        """Count occupied physical machines instead of treating a type as one slot."""
        machines = [
            structure for structure in getattr(self, "placed_structures", [])
            if structure.get("type") == machine_type
            and not structure.get("under_construction", False)
            and not structure.get("destroyed", False)
        ]
        capacity = len(machines)
        if capacity <= 0:
            return True
        occupied_ids: set[str] = set()
        legacy_occupancy = 0
        planner_tick = int(
            getattr(self, "current_tick", 0) if tick is None else tick
        )
        represented_crew_ids: set[str] = set()
        for cycle in getattr(self, "manufacturing_cycles", {}).values():
            if (
                cycle.get("completion_pending")
                and cycle.get("machine_type") == machine_type
                and cycle.get("machine_id")
            ):
                # The requesting operator's own inactive WIP is the batch
                # they intend to resume, not a competing order. A live route
                # reservation for a different relief operator still owns it.
                route_owner = cycle.get("route_operator_id")
                route_claimed_by_relief = bool(
                    route_owner
                    and route_owner != requesting_agent.id
                    and planner_tick
                    <= int(cycle.get("route_reserved_until_tick", -1))
                )
                if (
                    not cycle.get("active")
                    and route_owner == requesting_agent.id
                ):
                    # This machine is reserved for the requesting relief
                    # operator's physical walk. Counting the old operator's
                    # paused ownership as competing occupancy made the relief
                    # abandon the route on the following tick; both CNCs then
                    # remained locked forever with already-materialed WIP.
                    continue
                if (
                    cycle.get("operator_id") == requesting_agent.id
                    and not cycle.get("active")
                    and not route_claimed_by_relief
                ):
                    continue
                occupied_ids.add(str(cycle["machine_id"]))
                if cycle.get("operator_id"):
                    represented_crew_ids.add(str(cycle["operator_id"]))
                if route_owner:
                    represented_crew_ids.add(str(route_owner))
        for crew in living:
            if crew.id == requesting_agent.id:
                continue
            target = crew.action.target if isinstance(crew.action.target, dict) else {}
            action_type = getattr(crew.action, "action_type", "")
            running_cycle = (
                action_type == "refine" and target.get("completion_pending")
            )
            reserved_route = (
                action_type in ("move", "arrived")
                and target.get("destination") == "manufacturing_machine"
            )
            if (
                (running_cycle or reserved_route)
                and target.get("machine_type") == machine_type
            ):
                machine_id = target.get("machine_id")
                if machine_id:
                    occupied_ids.add(str(machine_id))
                    represented_crew_ids.add(str(crew.id))
                else:
                    legacy_occupancy += 1

        # Planning precedes physical route/cycle creation. Reserve one
        # anonymous slot for each REFINE contract emitted during this exact
        # planning tick so a batch planner cannot assign two astronauts to one
        # free workshop. The claim intentionally expires at the tick boundary:
        # after execution, MOVE/ARRIVED or the canonical machine cycle is the
        # sole source of truth. This is coordination only, not an RL outcome.
        pending_contract_claims = 0
        for crew in living:
            if (
                crew.id == requesting_agent.id
                or str(crew.id) in represented_crew_ids
            ):
                continue
            contract = getattr(crew, "_shared_work_contract", None)
            if (
                not isinstance(contract, dict)
                or str(contract.get("action", "")).lower() != "refine"
                or int(contract.get("slot_claim_tick", -1)) != planner_tick
            ):
                continue
            contract_target = contract.get("target", {})
            if not isinstance(contract_target, dict):
                contract_target = {}
            contract_machine = (
                contract.get("machine_type")
                or contract_target.get("machine_type")
                or contract_target.get("structure")
            )
            if not contract_machine and contract.get("stock_key"):
                contract_machine = self._recipes_cache.get(
                    contract["stock_key"], {}
                ).get("requires_structure")
            if contract_machine == machine_type:
                pending_contract_claims += 1
        return (
            len(occupied_ids) + legacy_occupancy + pending_contract_claims
            >= capacity
        )

    def _executable_refine_support_actions(
        self,
        support_actions: list[dict],
        agent: Agent,
        living: list[Agent],
        staged: dict[str, int],
        tick: Optional[int] = None,
    ) -> list[dict]:
        """Return critical-path batches that can enter a free machine now.

        Raw exploration remains parallel, but it must not take every available
        astronaut while qualified workshops sit idle with a fully staged
        batch. ``support_actions`` is already sorted by downstream critical
        path, so preserving its order starts the longest gating chain first.
        """
        ready: list[dict] = []
        energy_available = float(
            getattr(self, "_colony_resources", {}).get(
                "energy_stored_kwh", 0.0
            )
        )
        for action in support_actions:
            if action.get("action") != "refine":
                continue
            output = action.get("target", {}).get("output")
            recipe = self._recipes_cache.get(output, {})
            required_machine = recipe.get("requires_structure")
            if (
                not output
                or not required_machine
                or self._all_machine_slots_busy(
                    required_machine, living, agent, tick=tick
                )
                or energy_available
                < float(recipe.get("energy_kwh_per_batch", 0.0))
                or not all(
                    int(staged.get(material, 0)) >= int(quantity)
                    for material, quantity in recipe.get("materials", {}).items()
                )
                or not agent.can_craft(
                    {
                        **recipe,
                        "materials": {},
                        # A fixed forge/CNC cell supplies its own cutters and
                        # fixtures. Handheld field kits remain mandatory for
                        # construction and portable crafting.
                        "requires_tool": False,
                    }
                ).get("can_craft", False)
            ):
                continue
            ready.append(action)
        return ready

    def _refine_candidate_is_executable(
        self,
        candidate: dict,
        agent: Agent,
        living: list[Agent],
        structures_built: dict,
        tick: int,
    ) -> bool:
        """Mask a legacy RL fabrication option against the live shop state.

        RL may rank only actions that deterministic physics can start.  In
        particular, material still carried by a remote crewmate is not staged
        at a machine, and a second machine route must reserve its unconsumed
        BOM so another astronaut cannot plan against the same stock.
        """
        target = candidate.get("target", {})
        output = target.get("output") if isinstance(target, dict) else None
        recipe = self._recipes_cache.get(str(output), {}) if output else {}
        machine_type = recipe.get("requires_structure")
        if (
            not output
            or recipe.get("output", {}).get("type") != "material"
            or not machine_type
            or self._all_machine_slots_busy(
                str(machine_type), living, agent, tick=tick
            )
            or float(getattr(self, "_colony_resources", {}).get(
                "energy_stored_kwh", 0.0
            )) < float(recipe.get("energy_kwh_per_batch", 0.0))
            or not agent.can_craft(
                {
                    **recipe,
                    "materials": {},
                    "requires_tool": False,
                }
            ).get("can_craft", False)
        ):
            return False

        lz_x = int(getattr(self, "lz_x", agent.x))
        lz_y = int(getattr(self, "lz_y", agent.y))
        machines = [
            structure for structure in getattr(self, "placed_structures", [])
            if structure.get("type") == machine_type
            and not structure.get("under_construction", False)
            and not structure.get("destroyed", False)
        ]
        physical_stock = dict(getattr(self, "central_depot_inventory", {}))
        for crew in living:
            near_depot = max(abs(crew.x - lz_x), abs(crew.y - lz_y)) <= 1
            near_shop = any(
                max(
                    abs(crew.x - int(machine.get("x", crew.x))),
                    abs(crew.y - int(machine.get("y", crew.y))),
                ) <= 1
                for machine in machines
            )
            # The requester physically carries their own load to the selected
            # shop. Other remote inventories cannot satisfy this batch.
            if crew.id != agent.id and not (near_depot or near_shop):
                continue
            for material, quantity in crew.inventory.materials.items():
                physical_stock[material] = (
                    physical_stock.get(material, 0) + int(quantity)
                )

        available = self._materials_available_for_recipe(
            str(output), physical_stock, structures_built
        )
        # A route to a free machine owns the inputs it is going to collect at
        # the depot. Canonical WIP has already consumed its materials.
        for other in living:
            if other.id == agent.id:
                continue
            other_target = (
                other.action.target
                if isinstance(other.action.target, dict) else {}
            )
            if (
                getattr(other.action, "action_type", "")
                not in {"move", "arrived"}
                or other_target.get("destination")
                != "manufacturing_machine"
                or other_target.get("resume_machine_cycle")
            ):
                continue
            reserved_recipe = self._recipes_cache.get(
                str(other_target.get("output", "")), {}
            )
            for material, quantity in reserved_recipe.get("materials", {}).items():
                available[material] = max(
                    0, int(available.get(material, 0)) - int(quantity)
                )

        return all(
            int(available.get(material, 0)) >= int(quantity)
            for material, quantity in recipe.get("materials", {}).items()
        )

    def _contract_reservations(
        self, recipe_name: str, requesting_agent: Agent, tick: int
    ) -> dict[str, int]:
        """Count unfinished shared-work promises as planned, not completed, stock.

        Without this reservation ledger each of six independently evaluated
        workers saw the same depot deficit and opened an identical multi-batch
        order. The physical machines then correctly rejected the crowd, but the
        colony lost days in machine-door retry loops and accumulated surplus
        parts. Reservations coordinate work; only completed output remains in
        the real depot and construction still consumes only real inventory.
        """
        reserved: dict[str, int] = {}
        # A machine owns committed WIP independently of the person who loaded
        # it. Count its future output once, across all construction objectives;
        # it remains unavailable to physical crafting until unload inspection.
        cycles = getattr(self, "manufacturing_cycles", {})
        committed_cycle_ids = set()
        for cycle in cycles.values():
            if not cycle.get("completion_pending") or not cycle.get("output"):
                continue
            output = str(cycle["output"])
            reserved[output] = reserved.get(output, 0) + max(
                0, int(cycle.get("output_quantity", 1))
            )
            committed_cycle_ids.add(str(cycle.get("machine_id")))
        pooled = self._materials_available_for_recipe(
            recipe_name,
            self._pooled_materials(),
            getattr(self, "structures_built", {}),
        )
        outstanding = self._bom_outstanding_requirements(recipe_name, pooled)
        for crew in getattr(self, "agents", []):
            if crew.id == requesting_agent.id:
                continue
            contract = getattr(crew, "_shared_work_contract", None)
            if (
                not isinstance(contract, dict)
            ):
                continue
            active_wip = self._shared_contract_has_active_wip(crew, contract)
            contract_action = str(contract.get("action", "")).lower()
            if contract_action == "refine" and active_wip:
                token = getattr(crew, "_paused_manufacturing", None)
                if not isinstance(token, dict):
                    token = crew.action.target if isinstance(crew.action.target, dict) else {}
                if str(token.get("machine_id")) in committed_cycle_ids:
                    continue
            if (
                contract.get("recipe") != recipe_name
                or (
                    tick > int(contract.get("review_tick", tick))
                    and not active_wip
                )
            ):
                if contract_action == "gather":
                    self._close_bounded_gather_contract(crew, contract)
                else:
                    crew._shared_work_contract = None
                continue
            stock_key = contract.get("stock_key")
            quantity = max(0, int(contract.get("quantity_reserved", 0)))
            needed = max(
                0,
                int(outstanding.get(stock_key, 0))
                - int(reserved.get(stock_key, 0)),
            )
            target_stock = contract.get("target_stock")
            target_unfilled = (
                quantity
                if target_stock is None
                else max(0, int(target_stock) - int(pooled.get(stock_key, 0)))
            )
            gather_shift_expired = (
                contract_action == "gather"
                and tick > int(contract.get("shift_end_tick", tick))
            )
            if contract_action == "gather" and (
                not stock_key
                or quantity <= 0
                or target_unfilled <= 0
                or gather_shift_expired
            ):
                # Field extraction is a bounded load/shift commitment.  A
                # partially exposed geological face remains in the world, but
                # it must not make one astronaut own the campaign forever.
                self._close_bounded_gather_contract(crew, contract)
                continue
            if not stock_key or quantity <= 0 or needed <= 0 or target_unfilled <= 0:
                # The physical BOM may have changed underneath a bounded work
                # promise: for example a pump module has arrived, so its raw
                # raw-material prerequisite is no longer outstanding. A live
                # field worker may finish the small promised load before the
                # shift ends, but that contingency stock is not counted as
                # material reserved for the now-satisfied BOM branch.
                if not active_wip:
                    if contract_action == "gather":
                        self._close_bounded_gather_contract(crew, contract)
                    else:
                        crew._shared_work_contract = None
                continue
            reserved[stock_key] = reserved.get(stock_key, 0) + min(
                quantity, target_unfilled
            )
        return reserved

    @staticmethod
    def _release_autonomous_operator_token(agent: Agent, cycles: dict) -> None:
        """Release setup labor while the enclosed cell owns its running batch."""
        token = getattr(agent, "_paused_manufacturing", None)
        if not isinstance(token, dict):
            return
        cycle = cycles.get(str(token.get("machine_id")))
        if not (
            cycle and cycle.get("autonomous") and cycle.get("active")
            and not cycle.get("output_ready")
        ):
            return
        delattr(agent, "_paused_manufacturing")
        contract = getattr(agent, "_shared_work_contract", None)
        if (
            isinstance(contract, dict) and contract.get("action") == "refine"
            and contract.get("stock_key") == cycle.get("output")
        ):
            agent._shared_work_contract = None

    @staticmethod
    def _is_productive_contract_progress(
        action_type: str, action_target: dict, stock_key: str
    ) -> bool:
        """Distinguish work travel from an abort/return carrying stale metadata."""
        action_type = str(action_type or "").lower()
        if action_type not in {"move", "prospect", "gather"}:
            return False
        if not isinstance(action_target, dict):
            return False
        if action_target.get("resource") != stock_key:
            return False
        if action_target.get("survey_action") in {
            "portable_scanner", "scanner_recharge",
        }:
            return False
        if action_target.get("destination") in {
            "habitat", "shelter", "central_depot", "o2_filling_station",
        }:
            return False
        if any(action_target.get(marker) for marker in {
            "forced_return", "fatigue_return", "o2_return", "idle_return",
            "dehydration_return", "cancelled_unreachable_target",
            "route_rejected", "eva_denied",
        }):
            return False
        return True

    @staticmethod
    def _shared_contract_has_active_wip(crew: Agent, contract: dict) -> bool:
        """Keep a reservation while its physical batch or field job is live.

        ``review_tick`` is a planning lease, not permission to forget a batch
        whose materials and energy have already been committed.  A running
        cycle stores its state on ``action.target``; a physiological safety
        interruption moves the same state to ``_paused_manufacturing``.
        Engine movement intentionally strips work-contract metadata, so the
        physical output is the stable identity shared by both states.
        """
        contract_action = str(contract.get("action", "")).lower()
        stock_key = contract.get("stock_key")
        if not stock_key:
            return False
        action_target = (
            crew.action.target if isinstance(crew.action.target, dict) else {}
        )
        action_type = str(getattr(crew.action, "action_type", "")).lower()
        if contract_action == "refine":
            if (
                action_type == "refine"
                and action_target.get("completion_pending")
                and action_target.get("output") == stock_key
            ):
                return True
            paused = getattr(crew, "_paused_manufacturing", None)
            return bool(
                isinstance(paused, dict)
                and paused.get("completion_pending")
                and paused.get("output") == stock_key
            )

        if contract_action != "gather":
            return False

        # An abort/rejection state overrides every cached field-work marker.
        # Without this guard, an old excavation face can make a homeward route
        # look productive and reserve the worker indefinitely.
        if (
            action_target.get("destination")
            in {"habitat", "shelter", "central_depot", "o2_filling_station"}
            or any(action_target.get(marker) for marker in {
                "forced_return", "fatigue_return", "o2_return", "idle_return",
                "dehydration_return", "cancelled_unreachable_target",
                "route_rejected", "eva_denied",
            })
        ):
            return False

        if DecisionEngine._is_productive_contract_progress(
            action_type, action_target, str(stock_key)
        ):
            return True

        hand_prospect = getattr(crew, "_hand_prospect_target", None)
        if (
            isinstance(hand_prospect, dict)
            and hand_prospect.get("resource") == stock_key
        ):
            return True

        active_face = getattr(crew, "_active_excavation", None)
        if (
            isinstance(active_face, dict)
            and active_face.get("resource") == stock_key
        ):
            return True
        detected_recovery = getattr(crew, "_detected_resource_recovery", None)
        if (
            isinstance(detected_recovery, dict)
            and detected_recovery.get("resource") == stock_key
        ):
            return True
        expedition = getattr(crew, "_active_expedition", None)
        return bool(
            isinstance(expedition, dict)
            and expedition.get("kind") == "resource_recovery"
            and expedition.get("resource") == stock_key
            and expedition.get("status") in {"outbound", "working"}
        )

    def _close_bounded_gather_contract(
        self, crew: Agent, contract: dict
    ) -> None:
        """Release one completed/expired field shift without erasing geology.

        Refining and fabrication consume inputs when a physical batch starts,
        so those cycles are protected until completion or hand-off.  A gather
        marker is different: it merely points at a persistent world face.  At
        the promised stock target or shift boundary the crew returns/replans;
        the exposed layers and spoil already stored by the world are untouched.
        """
        if str(contract.get("action", "")).lower() != "gather":
            return
        resource = contract.get("stock_key")
        expedition = getattr(crew, "_active_expedition", None)
        if (
            isinstance(expedition, dict)
            and expedition.get("kind") == "resource_recovery"
            and expedition.get("resource") == resource
            and expedition.get("status") in {"outbound", "working", "returning"}
        ):
            # This changes the navigation state to a physical return; it does
            # not teleport either buddy or discard any carried payload.
            self._recall_expedition_team(crew)

        for marker_name in (
            "_hand_prospect_target",
            "_active_excavation",
            "_detected_resource_recovery",
        ):
            marker = getattr(crew, marker_name, None)
            if isinstance(marker, dict) and marker.get("resource") == resource:
                setattr(crew, marker_name, None)
        crew._shared_work_contract = None

    def _bom_outstanding_requirements(
        self, recipe_name: str, pooled: dict
    ) -> dict[str, int]:
        """Return current, inventory-aware demand for every BOM node.

        Unlike a flat dependency-name check, this consumes physically available
        parent subassemblies before expanding their inputs.  Consequently a
        delivered pump invalidates an old contract for its upstream feedstock
        that pump, while a still-missing pump remains visible as a real demand.
        Recipe quantities and configured batch yields are used unchanged.
        """
        available = {
            material: max(0, int(quantity))
            for material, quantity in pooled.items()
        }
        outstanding: dict[str, int] = {}

        def require(material: str, quantity: int, trail: set[str]) -> None:
            quantity = max(0, int(quantity))
            if quantity <= 0:
                return
            available_quantity = min(quantity, available.get(material, 0))
            if available_quantity:
                available[material] -= available_quantity
                quantity -= available_quantity
            if quantity <= 0:
                return

            outstanding[material] = outstanding.get(material, 0) + quantity
            component = self._recipes_cache.get(material, {})
            output = component.get("output", {})
            if (
                material in trail
                or output.get("type") != "material"
                or not component.get("materials")
            ):
                return
            batch_yield = max(1, int(output.get("quantity", 1)))
            batches = (quantity + batch_yield - 1) // batch_yield
            next_trail = set(trail)
            next_trail.add(material)
            for input_material, per_batch in component["materials"].items():
                require(input_material, int(per_batch) * batches, next_trail)
            # One physical batch may supply more than this branch requested.
            # Keep that projected surplus available to sibling branches of an
            # aggregate manifest, matching the batch accounting used by the
            # raw-feedstock projection below.
            surplus = batches * batch_yield - quantity
            if surplus > 0:
                available[material] = available.get(material, 0) + surplus

        recipe = self._recipes_cache.get(recipe_name, {})
        for material, quantity in recipe.get("materials", {}).items():
            require(material, int(quantity), {recipe_name})
        return outstanding

    def _bootstrap_order_feasibility(self, recipe_name: str, pooled: dict) -> dict:
        """Describe whether a bootstrap order can make useful progress now."""
        requirements = self._recipes_cache.get(recipe_name, {}).get("materials", {})
        ready = bool(requirements) and all(
            pooled.get(material, 0) >= needed
            for material, needed in requirements.items()
        )
        raw_deficits = self._raw_bom_deficits(recipe_name, pooled)
        planetary = set(getattr(self, "planetary_resources", set()))
        # A planet-present but not-yet-located feedstock is an exploration
        # task, not an impossible objective. The old mask treated a local
        # survey miss/remote request as permanent infeasibility; once carbon
        # was requested, every capacity recipe disappeared and the six crew
        # idled forever. Only an orbital-survey absence can invalidate a raw
        # BOM path. Discovery/extraction remains mandatory downstream.
        blocked_resources = {
            material
            for material, deficit in raw_deficits.items()
            if deficit > 0 and planetary and material not in planetary
        }
        finite_cargo_deficits = self._finite_cargo_deficits(
            recipe_name, pooled
        )
        return {
            "ready": ready,
            "actionable": ready or (
                not blocked_resources and not finite_cargo_deficits
            ),
            "blocked_resources": tuple(sorted(blocked_resources)),
            "finite_cargo_deficits": finite_cargo_deficits,
            "material_readiness": (
                sum(
                    min(int(pooled.get(material, 0)), int(needed))
                    for material, needed in requirements.items()
                ) / max(1, sum(int(value) for value in requirements.values()))
            ),
        }

    def _finite_cargo_deficits(
        self, recipe_name: str, pooled: dict
    ) -> dict[str, int]:
        """Return missing flight-qualified cores that cannot be made locally."""
        outstanding = self._bom_outstanding_requirements(recipe_name, pooled)
        return {
            material: int(quantity)
            for material, quantity in outstanding.items()
            if material in self.finite_cargo_materials and int(quantity) > 0
        }

    def _planning_fulfillment(
        self, recipe_name: str, status: Optional[dict], structures_built: dict
    ) -> float:
        """Count installed modules while operational capacity comes online.

        Readiness still uses commissioned live throughput. Planning also sees
        an installed-but-maturing or temporarily unpowered module so it does
        not build the same hardware past the configured physical requirement.
        """
        required = self._required_physical_structure_count(recipe_name, status)
        if required <= 0:
            return 1.0
        physical = min(
            1.0, float(structures_built.get(recipe_name, 0)) / required
        )
        if not status:
            return physical
        return max(float(status.get("fulfillment", 0.0)), physical)

    def _required_physical_structure_count(
        self, recipe_name: str, status: Optional[dict] = None
    ) -> int:
        """Derive gate-hardware count from the capacity it must physically route."""
        if status:
            return max(1, int(status.get("required_count", 1)))
        if recipe_name == "radiator_panel":
            thermal = getattr(self, "_thermal_state", {})
            effects = self._recipes_cache.get(recipe_name, {}).get("output", {}).get("effects", {})
            unit_kw = max(1e-9, float(effects.get("heat_rejection_kw", 8.0)))
            return max(0, math.ceil((
                float(thermal.get("total_waste_heat_kw", 0.0))
                - float(thermal.get("lander_cooling_capacity_kw", 4.0))
            ) / unit_kw))
        if recipe_name == "life_support_distribution_grid":
            effects = (
                self._recipes_cache.get(
                    "life_support_distribution_grid", {}
                )
                .get("output", {})
                .get("effects", {})
            )
            endpoints_per_vault = max(
                1, int(effects.get("max_connected_endpoints", 1))
            )
            planned_campus_vaults = max(
                0, int(effects.get("planned_campus_vaults", 0))
            )
            routed_assets = (
                "potable_water_tank", "oxygen_buffer_tank",
                "isru_o2_unit", "water_collector", "water_purifier",
                "greenhouse", "habitat_module",
            )
            # One additional endpoint is the precursor lander's ECLSS tie-in.
            required_endpoints = 1 + sum(
                max(
                    1,
                    int((
                        self.colony.get_structure_capacity_status(asset) or {}
                    ).get("required_count", 1)),
                )
                for asset in routed_assets
            )
            port_based_count = max(
                1, math.ceil(required_endpoints / endpoints_per_vault)
            )
            # Port count is only a lower bound. The surveyed layout can use
            # every metre of installed pipe before it uses every manifold port.
            # Keep expansion eligible when the real router reports this debt;
            # the extra vault still requires its full BOM and construction.
            network_reader = getattr(self, "life_support_network_snapshot", None)
            network = network_reader() if callable(network_reader) else {}
            needs_extension = any(
                not endpoint.get("physically_connected", False)
                and endpoint.get("reason") in {
                    "line_length_budget_exhausted",
                    "utility_vault_port_capacity_exhausted",
                }
                for endpoint in network.get("endpoints", [])
            )
            installed_nodes = len(network.get("nodes", []))
            return max(
                port_based_count,
                planned_campus_vaults,
                installed_nodes + 1 if needs_extension else 0,
            )
        if recipe_name == "power_distribution_grid":
            solar_status = self.colony.get_structure_capacity_status(
                "solar_panel"
            ) or {}
            required_arrays = max(
                1, int(solar_status.get("required_count", 1))
            )
            effects = (
                self._recipes_cache.get("power_distribution_grid", {})
                .get("output", {})
                .get("effects", {})
            )
            arrays_per_node = max(
                1, int(effects.get("max_connected_solar_arrays", 1))
            )
            return max(1, math.ceil(required_arrays / arrays_per_node))
        return 1

    def _open_capacity_orders(self, structures_built: dict) -> list[dict]:
        """Return every unfinished 106-person capacity order."""
        order = (
            "solar_panel", "life_support_distribution_grid",
            "potable_water_tank", "oxygen_buffer_tank",
            "isru_o2_unit", "water_collector", "water_purifier",
            "greenhouse", "habitat_module",
        )
        open_orders = []
        for index, recipe_name in enumerate(order):
            status = self.colony.get_structure_capacity_status(recipe_name)
            if status and self._planning_fulfillment(
                recipe_name, status, structures_built
            ) < 1.0:
                open_orders.append({
                    "recipe": recipe_name,
                    "order_index": index,
                    "fulfillment": self._planning_fulfillment(
                        recipe_name, status, structures_built
                    ),
                    "required_count": status.get("required_count"),
                    "category": status.get("category"),
                })
        return open_orders

    def _refresh_mission_forecast_recipe(
        self, structures_built: dict
    ) -> tuple[str, dict[str, int], dict[str, int]]:
        """Project the conserved direct BOM for every remaining mission module.

        Materials already committed to an active site have left inventory, so
        that site is subtracted from the remaining module count. Completed
        structures are likewise excluded. The resulting virtual recipe can use
        the existing recursive, batch-aware supply planner without changing any
        physical recipe, yield, machine slot, cargo quantity or score target.
        """
        active_counts: dict[str, int] = {}
        for structure in getattr(self, "placed_structures", []):
            if (
                structure.get("under_construction", False)
                and not structure.get("destroyed", False)
            ):
                name = str(structure.get("type", ""))
                active_counts[name] = active_counts.get(name, 0) + 1

        requirements: dict[str, int] = {}
        remaining_counts: dict[str, int] = {}
        for recipe_name in self.MISSION_FORECAST_STRUCTURES:
            recipe = self._recipes_cache.get(recipe_name, {})
            if not recipe or recipe_name == self.MISSION_FORECAST_RECIPE:
                continue
            status = self.colony.get_structure_capacity_status(recipe_name)
            required = self._required_physical_structure_count(
                recipe_name, status
            )
            remaining = max(
                0,
                int(required)
                - int(structures_built.get(recipe_name, 0))
                - int(active_counts.get(recipe_name, 0)),
            )
            if remaining <= 0:
                continue
            remaining_counts[recipe_name] = remaining
            for material, quantity in recipe.get("materials", {}).items():
                amount = int(quantity) * remaining
                if amount > 0:
                    requirements[str(material)] = (
                        requirements.get(str(material), 0) + amount
                    )

        self._recipes_cache[self.MISSION_FORECAST_RECIPE] = {
            "tier": 0,
            "display_name": "Remaining mission capacity material manifest",
            "materials": requirements,
            "base_duration_ticks": 0,
            "requires_tool": False,
            "requires_structure": None,
            "output": {"type": "planning_manifest"},
        }
        # Route capacity enables many independent consumers. Forecast its
        # remaining procurement before the last delivered tube is consumed,
        # rather than waiting for a disconnected habitat to stop field work.
        utility_materials: dict[str, int] = {}
        for utility in ("life_support_distribution_grid", "power_distribution_grid"):
            for material, per_unit in self._recipes_cache.get(utility, {}).get("materials", {}).items():
                utility_materials[material] = utility_materials.get(material, 0) + (
                    int(per_unit) * remaining_counts.get(utility, 0)
                )
        self._recipes_cache["__remaining_utility_supply__"] = {
            "materials": utility_materials,
            "output": {"type": "planning_manifest"},
        }
        return self.MISSION_FORECAST_RECIPE, requirements, remaining_counts

    def _forecast_utility_supply_debt(self, recipe_name: str, pooled: dict) -> dict:
        if recipe_name != self.MISSION_FORECAST_RECIPE:
            return {}
        return self._bom_outstanding_requirements("__remaining_utility_supply__", pooled)

    def get_colony_strategy_state(
        self, structures_built: Optional[dict] = None
    ) -> str:
        """Build the shared strategy observation without selecting an action."""
        structures = (
            getattr(self, "structures_built", {})
            if structures_built is None else structures_built
        )
        pooled = self._pooled_materials()
        unfinished = []
        eligible_recipe_names = []
        ready_recipe_names = []
        for recipe_name in (
            "solar_panel", "life_support_distribution_grid",
            "potable_water_tank", "oxygen_buffer_tank",
            "isru_o2_unit", "water_collector", "water_purifier",
            "greenhouse", "habitat_module",
            "power_distribution_grid", "communications_array", "landing_zone",
            "radiator_panel",
        ):
            if recipe_name not in self._recipes_cache:
                continue
            recipe = self._recipes_cache[recipe_name]
            prerequisite = recipe.get("requires_structure")
            if prerequisite and structures.get(prerequisite, 0) <= 0:
                continue
            status = self.colony.get_structure_capacity_status(recipe_name)
            if self._planning_fulfillment(
                recipe_name, status, structures
            ) >= 1.0:
                continue
            requirements = recipe.get("materials", {})
            available = self._materials_available_for_recipe(
                recipe_name, pooled, structures
            )
            is_ready = all(
                available.get(material, 0) >= needed
                for material, needed in requirements.items()
            )
            unfinished.append(is_ready)
            eligible_recipe_names.append(recipe_name)
            if is_ready:
                ready_recipe_names.append(recipe_name)

        active_site = next((
            site for site in getattr(self, "placed_structures", [])
            if site.get("under_construction", False)
            and not site.get("destroyed", False)
        ), None)
        living = [
            crew for crew in getattr(self, "agents", [])
            if getattr(crew.status, "value", str(crew.status)) != "dead"
        ]
        fit = [
            crew for crew in living
            if getattr(crew.status, "value", str(crew.status))
            not in ("incapacitated", "critical")
            and crew.needs.energy >= 25.0
            and crew.needs.o2_supply >= 15.0
        ]
        experiment_context = {}
        if getattr(getattr(self, "strategic_policy", None), "deadline_learning", False):
            # Coarse observations limit table growth. These describe the live
            # constraint; they do not select a preferred structure or bypass BOMs.
            constraint = ("assembly" if active_site or ready_recipe_names else
                          "fabrication" if any(
                              cycle.get("completion_pending")
                              for cycle in getattr(self, "manufacturing_cycles", {}).values()
                          ) else
                          "supply" if eligible_recipe_names else "qualification")
            experiment_context = {
                "days_until_deadline": (
                    getattr(self, "infrastructure_deadline_tick", 47520)
                    - getattr(self, "current_tick", 0)
                ) * self.tick_minutes / 1440.0,
                "work_constraint": constraint,
            }
        return build_colony_strategy_state(
            category_scores=self.colony.get_scores(),
            colony_resources=dict(getattr(self, "_colony_resources", {})),
            living_crew=len(living),
            fit_crew=len(fit),
            active_site_type=(active_site or {}).get("type"),
            ready_recipe_count=sum(1 for ready in unfinished if ready),
            surface_requires_plss=bool(
                getattr(self, "surface_requires_plss", False)
            ),
            eligible_recipes=eligible_recipe_names,
            ready_recipes=ready_recipe_names,
            **experiment_context,
        )

    def _capacity_site_layout_key(self) -> tuple:
        """Changes that can make a previously rejected campus site feasible."""
        return tuple(sorted(
            (str(s.get("id", "")), str(s.get("type", "")),
             int(s.get("x", 0)), int(s.get("y", 0)),
             bool(s.get("under_construction", False)),
             bool(s.get("destroyed", False)))
            for s in getattr(self, "placed_structures", [])
        ))

    def _water_stock_deadline_status(self, structures: dict, tick: int) -> dict:
        """Forecast reserve filling from measured extraction and process load."""
        storage = self.colony.get_structure_capacity_status("potable_water_tank") or {}
        extraction = self.colony.get_structure_capacity_status("water_collector") or {}
        target_l = float(storage.get("target_capacity", 0.0))
        stored_l = float(storage.get("current_capacity", 0.0))
        greenhouse_effects = self._recipes_cache.get("greenhouse", {}).get("output", {}).get("effects", {})
        # The engine condenses 90% of crop transpiration. Charge the full
        # installed farm load even when today's energy dispatch idles a chamber.
        crop_loss_per_day = (
            int(structures.get("greenhouse", 0))
            * float(greenhouse_effects.get("water_consumption_liters_per_tick", 0.0))
            * 0.10 * 1440.0 / self.tick_minutes
        )
        living = sum(
            getattr(crew.status, "value", str(crew.status)) != "dead"
            for crew in getattr(self, "agents", [])
        )
        # Advance-crew makeup and electrolyser feed are small but real loads;
        # civilians do not consume surface water before their arrival.
        crew_loss_per_day = living * (3.0 + 1.083 * 1.125)
        net_l_per_day = float(extraction.get("current_capacity", 0.0)) - crop_loss_per_day - crew_loss_per_day
        days_remaining = max(0.0, (self.infrastructure_deadline_tick - tick) * self.tick_minutes / 1440.0)
        projected_l = stored_l + max(0.0, net_l_per_day) * days_remaining
        tank_capacity_l = int(structures.get("potable_water_tank", 0)) * float(
            self._recipes_cache.get("potable_water_tank", {}).get("output", {}).get("effects", {}).get(
                "potable_water_storage_capacity_liters", 0.0
            )
        )
        return {
            "at_risk": (
                int(structures.get("solar_panel", 0)) > 0
                and int(structures.get("life_support_distribution_grid", 0)) > 0
                and target_l > 0 and stored_l < target_l and projected_l < target_l
            ),
            "net_l_per_day": net_l_per_day,
            "projected_reserve_l": min(tank_capacity_l, projected_l),
            "target_l": target_l,
            "storage_capacity_short": tank_capacity_l < target_l,
        }

    def mark_capacity_site_unavailable(self, recipe_name: str) -> None:
        blocked = getattr(self, "_blocked_capacity_sites", {})
        blocked[recipe_name] = (self._capacity_site_layout_key(),
            int(getattr(self, "current_tick", 0))
            + ticks_for_minutes(1440.0, self.tick_minutes))
        self._blocked_capacity_sites = blocked

    def _select_shared_capacity_order(
        self, pooled: dict, structures_built: dict, tick: Optional[int] = None
    ) -> Optional[dict]:
        """Let the planet policy choose the crew's next physical objective.

        This method constructs the action mask from real recipes and unmet
        acceptance requirements.  It does not encode a solar/water/O2 build
        order.  A fresh policy can therefore make a poor but physically valid
        strategic choice and learn from the resulting episode.
        """
        active_site = next((
            site for site in getattr(self, "placed_structures", [])
            if site.get("under_construction", False)
            and not site.get("destroyed", False)
            and site.get("type") in {
                "solar_panel", "life_support_distribution_grid",
                "potable_water_tank", "oxygen_buffer_tank",
                "isru_o2_unit", "water_collector", "water_purifier",
                "greenhouse", "habitat_module",
                "power_distribution_grid", "communications_array", "landing_zone",
                "radiator_panel",
            }
        ), None)
        if active_site is not None:
            candidate_names = (active_site["type"],)
        else:
            # Every unfinished capacity/gate recipe is a valid strategic
            # action from landing onward.  Dependency planning below decides
            # how to realize it; the mask does not quietly teach an order.
            candidate_names_list = []
            layout_key = self._capacity_site_layout_key()
            now = int(getattr(self, "current_tick", 0) if tick is None else tick)
            for name in (
                "solar_panel", "life_support_distribution_grid",
                "potable_water_tank", "oxygen_buffer_tank",
                "isru_o2_unit", "water_collector", "water_purifier",
                "greenhouse", "habitat_module",
                "power_distribution_grid", "communications_array", "landing_zone",
                "radiator_panel",
            ):
                recipe = self._recipes_cache.get(name)
                if not recipe:
                    continue
                rejected_site = getattr(self, "_blocked_capacity_sites", {}).get(name)
                if (rejected_site and rejected_site[0] == layout_key
                    and now < rejected_site[1]):
                    # The motor already proved that the current campus cannot
                    # host this job. Replan other work until the layout changes;
                    # retry daily for terrain/survey changes outside this key.
                    continue
                prerequisite = recipe.get("requires_structure")
                if prerequisite and structures_built.get(prerequisite, 0) <= 0:
                    # This is an action-validity mask, not a strategic hint:
                    # an antenna cannot be built before its physical grid.
                    continue
                status = self.colony.get_structure_capacity_status(name)
                unfinished = (
                    self._planning_fulfillment(
                        name, status, structures_built
                    ) < 1.0
                )
                if unfinished:
                    candidate_names_list.append(name)
            candidate_names = tuple(candidate_names_list)
        if not candidate_names:
            return None

        # A commissioned endpoint that cannot join the finite pipe network is
        # direct physical evidence that another distribution vault/line kit is
        # required. Surface this dependency to the strategic mask so RL does
        # not keep installing consumers whose capacity cannot be commissioned.
        network_reader = getattr(
            self, "life_support_network_snapshot", None
        )
        utility_network = (
            network_reader() if callable(network_reader) else {}
        )
        blocked_utility_reasons = {
            "no_utility_vault",
            "outside_service_envelope",
            "utility_vault_port_capacity_exhausted",
            "line_length_budget_exhausted",
        }
        utility_expansion_required = any(
            not endpoint.get("physically_connected", False)
            and endpoint.get("reason") in blocked_utility_reasons
            for endpoint in utility_network.get("endpoints", [])
        )
        water_deadline = self._water_stock_deadline_status(
            structures_built, int(getattr(self, "current_tick", 0) if tick is None else tick)
        )

        orders = []
        for order_index, recipe_name in enumerate(candidate_names):
            status = self.colony.get_structure_capacity_status(recipe_name) or {}
            planning_fulfillment = self._planning_fulfillment(
                recipe_name, status or None, structures_built
            )
            candidate = {
                "recipe": recipe_name,
                "order_index": order_index,
                "fulfillment": planning_fulfillment,
                "operational_fulfillment": float(
                    status.get("fulfillment", 0.0)
                ),
                "required_count": self._required_physical_structure_count(
                    recipe_name, status or None
                ),
                "category": status.get("category", "mission_acceptance"),
                "mission_acceptance_gate": recipe_name in {
                    "communications_array", "landing_zone"
                },
                "utility_prerequisite": (
                    recipe_name in {"power_distribution_grid", "radiator_panel"}
                ),
                "utility_recovery_priority": bool(
                    recipe_name == "life_support_distribution_grid"
                    and utility_expansion_required
                ),
                "water_stock_deadline_priority": bool(
                    water_deadline["at_risk"]
                    and (
                        recipe_name == "water_collector"
                        or (
                            recipe_name == "potable_water_tank"
                            and water_deadline["storage_capacity_short"]
                            and water_deadline["net_l_per_day"] > 0.0
                        )
                    )
                ),
                # These modules consume the finite life-support pipe/port
                # envelope.  When that envelope is exhausted, installing more
                # of them before the next distribution vault is ready merely
                # creates disconnected hardware.  Independent work such as
                # solar, power distribution and landing preparation can still
                # proceed while the shop makes the vault's long-lead parts.
                "life_support_endpoint": recipe_name in {
                    "potable_water_tank", "oxygen_buffer_tank",
                    "isru_o2_unit", "water_collector", "water_purifier",
                    "greenhouse", "habitat_module",
                },
                "thermal_safety_priority": bool(
                    recipe_name == "radiator_panel"
                    and float(getattr(self, "_thermal_state", {}).get("temperature_c", 21.0)) >= 30.0
                    and float(getattr(self, "_thermal_state", {}).get("net_heat_kw", 0.0)) > 0.0
                ),
                "bootstrap_power_priority": bool(
                    recipe_name == "solar_panel"
                    and structures_built.get("solar_panel", 0) == 0
                    and "lander_auxiliary_energy_remaining_kwh" in getattr(self, "_colony_resources", {})
                    and float(self._colony_resources.get("energy_stored_kwh", 0.0)) < 25.0
                    and float(self._colony_resources.get("lander_auxiliary_energy_remaining_kwh", 0.0)) < 288.0
                ),
                "safety_priority": bool(
                    recipe_name == "habitat_module"
                    and self.severe_radiation_environment
                    and structures_built.get("habitat_module", 0) <= 0
                ),
            }
            if active_site is not None:
                candidate["site"] = active_site
            power_state = getattr(self, "_power_cycle_telemetry", {})
            if power_state:
                # A wastewater unit cannot create new water, and a charged
                # electrical system is not the cause of an extraction deficit.
                # Keep only dependencies that can relieve the measured water
                # shortage; normal readiness planning still sees every module.
                power_shortage = bool(power_state.get("deficit", False)) or float(
                    getattr(self, "_colony_resources", {}).get("energy_stored_kwh", 0.0)
                ) < 25.0
                collector_connection_gap = int(power_state.get("installed_water_collectors", 0)) > int(
                    power_state.get("networked_water_collectors", 0)
                )
                candidate["water_reserve_support"] = bool(
                    recipe_name == "water_collector"
                    or (recipe_name in {"solar_panel", "power_distribution_grid"} and power_shortage)
                    or (recipe_name == "life_support_distribution_grid" and collector_connection_gap)
                    or (recipe_name == "potable_water_tank" and float(
                        getattr(self, "_water_storage_capacity_l", 0.0)
                    ) < max(150.0, 30.0 * len(getattr(self, "agents", []))))
                )
            orders.append(candidate)

        for candidate in orders:
            candidate_pool = self._materials_available_for_recipe(
                candidate["recipe"], pooled, structures_built
            )
            recipe = self._recipes_cache.get(candidate["recipe"], {})
            requirements = recipe.get("materials", {})
            committed_site = candidate.get("site")
            if isinstance(committed_site, dict):
                # The BOM was atomically removed from inventory when this
                # physical site opened. Rechecking the now-absent materials
                # falsely marked the objective infeasible, orphaned its RL
                # credit, and let a completed structure look like no progress.
                candidate["ready"] = True
                candidate["material_readiness"] = 1.0
                candidate["blocked_resources"] = []
                candidate["remaining_work_units"] = max(
                    0.0,
                    float(committed_site.get(
                        "ticks_remaining",
                        1.0 - float(committed_site.get("progress", 0.0)),
                    )),
                )
                feasibility = {"actionable": True}
            else:
                candidate["ready"] = all(
                    candidate_pool.get(material, 0) >= needed
                    for material, needed in requirements.items()
                )
                total_required = max(1, sum(requirements.values()))
                candidate["material_readiness"] = sum(
                        min(candidate_pool.get(material, 0), needed)
                    for material, needed in requirements.items()
                ) / total_required
                feasibility = self._bootstrap_order_feasibility(
                    candidate["recipe"], pooled
                )
                candidate["blocked_resources"] = feasibility["blocked_resources"]
                candidate["finite_cargo_deficits"] = dict(
                    feasibility.get("finite_cargo_deficits", {})
                )
                outstanding = self._bom_outstanding_requirements(
                    candidate["recipe"], pooled
                )
                candidate["remaining_work_units"] = sum(outstanding.values())

            living_fit = [
                crew for crew in getattr(self, "agents", [])
                if getattr(crew.status, "value", str(crew.status))
                not in ("dead", "incapacitated", "critical")
                and crew.needs.energy >= 25.0
                and crew.needs.o2_supply >= 15.0
            ]
            recommended_crew = max(
                1, int(recipe.get("construction", {}).get(
                    "recommended_crew", 1
                ))
            )
            productive_crew = max(
                1, min(len(living_fit), recommended_crew)
            )
            minimum_completion_ticks = int(math.ceil(
                max(1.0, float(recipe.get("base_duration_ticks", 1.0)))
                / productive_crew
            ))
            # Base duration covers productive installation work. A credible
            # mission plan must also reserve time for fabrication, transporter
            # staging, crew recovery and hand-offs. Until this planet has a
            # measured delivery sample, use one equal logistics allowance;
            # later attempts and later modules use their real observed latency.
            planning_completion_ticks = minimum_completion_ticks * 2
            policy_for_schedule = getattr(self, "strategic_policy", None)
            if policy_for_schedule is not None and hasattr(
                policy_for_schedule, "expected_completion_ticks"
            ):
                planning_completion_ticks = (
                    policy_for_schedule.expected_completion_ticks(
                        candidate["recipe"], planning_completion_ticks
                    )
                )
            now_tick = int(
                getattr(self, "current_tick", 0) if tick is None else tick
            )
            required_count = max(1, int(candidate["required_count"]))
            installed_count = max(
                0, int(structures_built.get(candidate["recipe"], 0))
            )
            remaining_modules = max(0, required_count - installed_count)
            maturity_ticks = 0
            if candidate["recipe"] == "greenhouse":
                effects = recipe.get("output", {}).get("effects", {})
                maturity_ticks = int(effects.get("growth_cycle_ticks", 0) or 0)
                if maturity_ticks <= 0:
                    maturity_ticks = ticks_for_minutes(
                        float(effects.get("first_harvest_days", 28.0)) * 1440.0,
                        self.tick_minutes,
                    )
            infrastructure_ticks_remaining = max(
                0, int(self.infrastructure_deadline_tick) - now_tick
            )
            # The acceptance clock stops at day 330, not at the end of the
            # 30-day soak.  A greenhouse also has to complete one real crop
            # cycle before it contributes food.  This forecast does not grant
            # progress or prescribe a build order; it tells the strategic
            # policy when an unfinished category is approaching its last
            # credible start window.
            category_lead_ticks = (
                remaining_modules * planning_completion_ticks + maturity_ticks
            )
            schedule_uncertainty_ticks = max(
                ticks_for_minutes(7.0 * 1440.0, self.tick_minutes),
                int(math.ceil(category_lead_ticks * 0.20)),
            )
            acceptance_slack_ticks = (
                infrastructure_ticks_remaining - category_lead_ticks
            )
            candidate.update({
                "remaining_required_modules": remaining_modules,
                "acceptance_maturity_ticks": maturity_ticks,
                "acceptance_category_lead_ticks": category_lead_ticks,
                "acceptance_deadline_ticks_remaining": (
                    infrastructure_ticks_remaining
                ),
                "acceptance_deadline_slack_ticks": acceptance_slack_ticks,
                "acceptance_deadline_priority": bool(
                    remaining_modules > 0
                    and acceptance_slack_ticks <= schedule_uncertainty_ticks
                ),
                # Preserve the broad learned exploration envelope early in the
                # mission. During the final 120 days before acceptance, compare
                # the actual next module and permit only one normalized module
                # of headroom. The policy still chooses among all categories
                # inside that physical envelope.
                "balance_headroom_modules": (
                    None
                    if infrastructure_ticks_remaining
                    > ticks_for_minutes(120.0 * 1440.0, self.tick_minutes)
                    else 1
                ),
            })
            max_ticks = self.mission_attempt_max_ticks
            ticks_remaining = (
                max(0, int(max_ticks) - int(
                    now_tick
                ))
                if max_ticks is not None else None
            )
            deadline_feasible = (
                ticks_remaining is None
                or minimum_completion_ticks <= ticks_remaining
            )
            candidate["minimum_completion_ticks"] = minimum_completion_ticks
            candidate["planning_completion_ticks"] = planning_completion_ticks
            candidate["productive_crew_limit"] = productive_crew
            candidate["ticks_remaining"] = ticks_remaining
            candidate["deadline_feasible"] = deadline_feasible
            candidate["actionable"] = bool(
                feasibility["actionable"] and deadline_feasible
            )
            # A commitment is reviewed after three days without any measured
            # BOM/site progress, and after seven days without a delivered
            # module even if intermediate stock is still improving.
            candidate["stall_timeout_ticks"] = ticks_for_minutes(
                3.0 * 1440.0, self.tick_minutes
            )
            candidate["replan_interval_ticks"] = ticks_for_minutes(
                7.0 * 1440.0, self.tick_minutes
            )

        policy = getattr(self, "strategic_policy", None)
        if policy is None:
            # Unit-level callers that do not install a mission policy still
            # receive a deterministic physical order.  Live engines always
            # install the planet-scoped policy during construction.
            return min(orders, key=lambda item: (
                item["fulfillment"], -item["material_readiness"],
                item["order_index"],
            ))

        state_key = self.get_colony_strategy_state(structures_built)
        return policy.choose(
            state_key=state_key,
            candidates=orders,
            structures_built=structures_built,
            tick=int(
                getattr(self, "current_tick", 0) if tick is None else tick
            ),
            colony_score=self.colony.get_overall_score(),
        )

    def _shared_dependency_actions(
        self, recipe_name: str, pooled: dict
    ) -> list[dict]:
        """Create a deduplicated bill-of-material support queue."""
        requirements = self._recipes_cache.get(recipe_name, {}).get("materials", {})
        actions = []
        action_indexes = {}
        for material, needed in requirements.items():
            deficit = needed - pooled.get(material, 0)
            if deficit <= 0:
                continue
            action = self._material_dependency_action(
                material, deficit, recipe_name, pooled, set()
            )
            target = action.get("target", {})
            key = (action.get("action"), target.get("resource"), target.get("output"))
            if key not in action_indexes:
                action_indexes[key] = len(actions)
                actions.append(action)
            else:
                index = action_indexes[key]
                current = actions[index].setdefault("target", {})
                current["quantity_needed"] = max(
                    int(current.get("quantity_needed", 1)),
                    int(target.get("quantity_needed", 1)),
                )

        # Expand the complete configured BOM now, including batch yields. This
        # does not alter a recipe: it lets logistics discover and stock raw
        # feedstocks before a late subassembly suddenly exhausts one of them.
        # Available intermediate components are consumed virtually so the
        # projection never orders their raw inputs twice.
        for material, deficit in sorted(
            self._raw_bom_deficits(recipe_name, pooled).items(),
            key=lambda item: (-item[1], item[0]),
        ):
            if deficit <= 0:
                continue
            key = ("gather", material, None)
            if key in action_indexes:
                index = action_indexes[key]
                current = actions[index].setdefault("target", {})
                current["quantity_needed"] = max(
                    int(current.get("quantity_needed", 1)), int(deficit)
                )
                continue
            action_indexes[key] = len(actions)
            actions.append({
                "action": "gather",
                "target": {
                    "resource": material,
                    "quantity_needed": int(deficit),
                    "full_bom_feedstock": True,
                },
                "reasoning": (
                    f"Pre-stocking {deficit} {material} required by the full "
                    f"configured {recipe_name} subassembly chain"
                ),
                "deterministic": True,
            })
        criticality = self._bom_critical_path_priorities(recipe_name)
        utility_debt = self._forecast_utility_supply_debt(recipe_name, pooled)
        for action in actions:
            target = action.setdefault("target", {})
            stock_key = target.get("resource") or target.get("output")
            target["critical_path_ticks"] = int(
                criticality.get(stock_key, 0)
            )
            target["utility_supply_priority"] = utility_debt.get(stock_key, 0) > 0
        return sorted(
            actions,
            key=lambda action: (
                0 if action.get("target", {}).get("utility_supply_priority") else 1,
                -int(action.get("target", {}).get("critical_path_ticks", 0)),
                0 if action.get("action") == "refine" else 1,
                str(
                    action.get("target", {}).get("output")
                    or action.get("target", {}).get("resource")
                    or ""
                ),
            ),
        )

    def _bom_critical_path_priorities(self, recipe_name: str) -> dict[str, int]:
        """Return remaining downstream duration for every BOM node.

        A raw feedstock or subassembly that gates a long chain is started
        before a short leaf even when the JSON happens to list that leaf first.
        Durations are configured process ticks; no imaginary productivity or
        random priority is introduced here.
        """
        priorities: dict[str, int] = {}

        def walk(node: str, downstream_ticks: int, trail: set[str]) -> None:
            if node in trail:
                return
            recipe = self._recipes_cache.get(node, {})
            output = recipe.get("output", {})
            is_material_recipe = output.get("type") == "material"
            duration = (
                max(0, int(recipe.get("base_duration_ticks", 0)))
                if is_material_recipe else 0
            )
            remaining = downstream_ticks + duration
            priorities[node] = max(priorities.get(node, 0), remaining)
            materials = recipe.get("materials", {}) if is_material_recipe else {}
            if not materials:
                return
            next_trail = set(trail)
            next_trail.add(node)
            for material in materials:
                walk(str(material), remaining, next_trail)

        root = self._recipes_cache.get(recipe_name, {})
        construction_ticks = max(0, int(root.get("base_duration_ticks", 0)))
        for material in root.get("materials", {}):
            walk(str(material), construction_ticks, {recipe_name})
        return priorities

    def _raw_bom_deficits(self, recipe_name: str, pooled: dict) -> dict[str, int]:
        """Project extractable raw inputs for a complete, batch-aware BOM."""
        raw_materials = {
            "iron_ore", "silica_sand", "basalt", "water_ice",
            "graphite", "sulfur", "chalcopyrite_ore", "regolith",
            "olivine", "calcite",
        }
        available = {
            material: max(0, int(quantity))
            for material, quantity in pooled.items()
        }
        deficits: dict[str, int] = {}
        criticality: dict[str, int] = {}

        def satisfy(material: str, quantity: int, trail: set[str]) -> None:
            quantity = max(0, int(quantity))
            if quantity <= 0:
                return
            use = min(quantity, available.get(material, 0))
            if use:
                available[material] -= use
                quantity -= use
            if quantity <= 0:
                return
            if material in raw_materials:
                deficits[material] = deficits.get(material, 0) + quantity
                return
            component = self._recipes_cache.get(material, {})
            output = component.get("output", {})
            if (
                material in trail
                or output.get("type") != "material"
                or not component.get("materials")
            ):
                # Delivered/salvaged parts without a fabrication recipe are
                # finite mission cargo, not imaginary geological deposits.
                return
            batch_yield = max(1, int(output.get("quantity", 1)))
            batches = (quantity + batch_yield - 1) // batch_yield
            next_trail = set(trail)
            next_trail.add(material)
            for input_material, per_batch in component["materials"].items():
                satisfy(input_material, int(per_batch) * batches, next_trail)
            # A fabrication batch can yield more than this branch requested.
            # Keep that physical surplus available to sibling BOM branches;
            # otherwise the planner orders the same intermediate's raw inputs
            # repeatedly even though one earlier batch already produced it.
            surplus = batches * batch_yield - quantity
            if surplus > 0:
                available[material] = available.get(material, 0) + surplus

        recipe = self._recipes_cache.get(recipe_name, {})
        for material, quantity in recipe.get("materials", {}).items():
            satisfy(material, int(quantity), {recipe_name})
        return deficits

    def _prioritize_live_bom_resources(
        self,
        resources,
        recipe_name: Optional[str] = None,
        pooled: Optional[dict] = None,
    ) -> list[str]:
        """Order requested feedstocks by the live capacity BOM bottleneck.

        Alphabetical request order made a newly critical zero-stock feedstock
        wait behind unrelated historical survey requests.  Use only real
        inventory and the currently selected structure BOM; non-BOM requests
        remain queued deterministically after its live deficits.
        """
        requested = {
            str(resource) for resource in (resources or []) if resource
        }
        if not requested:
            return []

        structures_built = getattr(self, "structures_built", {})
        pooled = self._pooled_materials() if pooled is None else dict(pooled)
        if not recipe_name:
            shared_order = getattr(self, "shared_work_order", {})
            if isinstance(shared_order, dict):
                recipe_name = shared_order.get("recipe")
        if not recipe_name or recipe_name not in self._recipes_cache:
            selected = self._select_shared_capacity_order(
                pooled, structures_built
            )
            recipe_name = selected.get("recipe") if selected else None

        deficits: dict[str, int] = {}
        criticality: dict[str, int] = {}
        planning_pool = pooled
        if recipe_name in self._recipes_cache:
            planning_pool = self._materials_available_for_recipe(
                recipe_name, pooled, structures_built
            )
            deficits = self._raw_bom_deficits(recipe_name, planning_pool)
            criticality = self._bom_critical_path_priorities(recipe_name)

        def priority(resource: str):
            deficit = max(0, int(deficits.get(resource, 0)))
            available = max(0, int(planning_pool.get(resource, 0)))
            readiness = (
                available / (available + deficit)
                if deficit > 0 else 1.0
            )
            return (
                0 if deficit > 0 else 1,
                0 if utility_debt.get(resource, 0) > 0 else 1,
                -int(criticality.get(resource, 0)),
                readiness,
                -deficit,
                resource,
            )

        utility_debt = self._forecast_utility_supply_debt(recipe_name, planning_pool)
        return sorted(requested, key=priority)

    def _scanner_target(
        self,
        agent: Agent,
        target: Optional[dict] = None,
        resources=None,
        *,
        preserve_primary: bool = False,
    ) -> dict:
        """Return a backward-compatible multi-resource scanner target."""
        scanner_target = dict(target or {})
        candidates = set(
            getattr(self, "remote_resource_requests", set())
            if resources is None else resources
        )
        candidates.update(scanner_target.get("survey_resources", []) or [])
        if scanner_target.get("resource"):
            candidates.add(scanner_target["resource"])
        ordered = self._prioritize_live_bom_resources(candidates)

        primary = scanner_target.get("resource") if preserve_primary else None
        if primary:
            ordered = [primary, *(
                resource for resource in ordered if resource != primary
            )]
        elif ordered:
            primary = ordered[0]
        if primary:
            scanner_target["resource"] = primary
        scanner_target["survey_resources"] = ordered

        # The scanner custodian has changed jobs.  Its old extraction promise
        # must become available to another worker instead of looking like a
        # second, still-active gatherer for the same resource.
        contract = getattr(agent, "_shared_work_contract", None)
        if (
            isinstance(contract, dict)
            and str(contract.get("action", "")).lower() == "gather"
        ):
            claimed_resource = contract.get("stock_key")
            agent._shared_work_contract = None
            hand_target = getattr(agent, "_hand_prospect_target", None)
            if (
                isinstance(hand_target, dict)
                and hand_target.get("resource") == claimed_resource
            ):
                agent._hand_prospect_target = None
        return scanner_target

    def _dependency_material_names(
        self, recipe_name: str, visited: Optional[set[str]] = None
    ) -> set[str]:
        """Return top-level and intermediate inputs in a structure BOM."""
        visited = set() if visited is None else visited
        if recipe_name in visited:
            return set()
        visited.add(recipe_name)
        names = set()
        for material in self._recipes_cache.get(recipe_name, {}).get("materials", {}):
            names.add(material)
            component = self._recipes_cache.get(material, {})
            if component.get("output", {}).get("type") == "material":
                names.update(self._dependency_material_names(material, visited))
        return names

    def _active_raw_campaign_claims(
        self,
        recipe_name: str,
        tick: Optional[int] = None,
        exclude_agent_id: Optional[str] = None,
    ) -> set[str]:
        """Return raw resources already owned by a live field campaign."""
        claims: set[str] = set()
        for crew in getattr(self, "agents", []):
            if crew.id == exclude_agent_id:
                continue
            contract = getattr(crew, "_shared_work_contract", None)
            if (
                isinstance(contract, dict)
                and contract.get("recipe") == recipe_name
                and str(contract.get("action", "")).lower() == "gather"
            ):
                within_lease = (
                    tick is None
                    or tick <= int(contract.get("review_tick", tick))
                )
                if within_lease or self._shared_contract_has_active_wip(
                    crew, contract
                ):
                    stock_key = contract.get("stock_key")
                    if stock_key:
                        claims.add(stock_key)

            action_target = (
                crew.action.target
                if isinstance(crew.action.target, dict) else {}
            )
            action_type = str(
                getattr(crew.action, "action_type", "")
            ).lower()
            productive_progress = self._is_productive_contract_progress(
                action_type,
                action_target,
                str(action_target.get("resource", "")),
            )
            if productive_progress:
                claims.add(action_target["resource"])
            safety_return = (
                action_type == "move"
                and action_target.get("destination") in {
                    "habitat", "shelter", "central_depot",
                    "o2_filling_station",
                }
            )
            for marker_name in (
                "_hand_prospect_target",
                "_active_excavation",
                "_detected_resource_recovery",
            ):
                marker = getattr(crew, marker_name, None)
                if (
                    not safety_return
                    and isinstance(marker, dict)
                    and marker.get("resource")
                ):
                    claims.add(marker["resource"])
            expedition = getattr(crew, "_active_expedition", None)
            if (
                isinstance(expedition, dict)
                and expedition.get("kind") == "resource_recovery"
                and expedition.get("status")
                in {"outbound", "working"}
                and expedition.get("resource")
            ):
                claims.add(expedition["resource"])
        return claims

    def _raw_support_action(
        self,
        recipe_name: str,
        pooled: dict,
        worker_index: int,
        requesting_agent: Optional[Agent] = None,
        tick: Optional[int] = None,
    ) -> dict:
        """Keep logistics useful while a machine or power bus is occupied."""
        deficits = self._raw_bom_deficits(recipe_name, pooled)
        ranked = self._prioritize_live_bom_resources(
            deficits, recipe_name, pooled
        )
        if not ranked:
            # The immediate BOM may be staged while the machine is occupied.
            # Derive a small reserve target from this recipe's configured raw
            # graph instead of falling back to a hard-coded material table.
            configured_raw = self._raw_bom_deficits(recipe_name, {})
            ranked = sorted(
                configured_raw,
                key=lambda resource: (
                    int(pooled.get(resource, 0)),
                    -int(configured_raw.get(resource, 0)),
                    resource,
                ),
            )
        if not ranked:
            ranked = ["iron_ore"]

        claims = self._active_raw_campaign_claims(
            recipe_name,
            tick,
            getattr(requesting_agent, "id", None),
        )
        unclaimed = [resource for resource in ranked if resource not in claims]
        resource = (
            unclaimed[0]
            if unclaimed else ranked[worker_index % len(ranked)]
        )
        quantity_needed = max(1, int(deficits.get(resource, 12)))
        return {
            "action": "gather",
            "target": {
                "resource": resource,
                "quantity_needed": quantity_needed,
                "shared_work_order": True,
                "capacity_recipe": recipe_name,
                "live_bom_priority": True,
            },
            "reasoning": (
                f"Shared {recipe_name} logistics: stockpiling {resource} while "
                "the fabrication machine or power bus is occupied"
            ),
            "deterministic": True,
        }

    def _greenhouse_work_available(self, greenhouse_id: str) -> bool:
        """Only dispatch crew into a commissioned, powered utility endpoint."""
        greenhouse = next((
            structure for structure in getattr(self, "placed_structures", [])
            if str(structure.get("id", "")) == str(greenhouse_id)
            and structure.get("type") == "greenhouse"
            and not structure.get("under_construction", False)
            and not structure.get("destroyed", False)
            and float(structure.get("health", 1.0)) > 0.0
        ), None)
        connected = getattr(self, "greenhouse_utility_connected", None)
        return bool(
            greenhouse is not None
            and callable(connected)
            and connected(greenhouse)
        )

    def _greenhouse_duty_decision(
        self, agent: Agent, tick: int
    ) -> Optional[dict]:
        """Schedule bounded crop inspection and harvest work.

        Environmental controls run continuously, so astronauts do not need to
        stand in a farm all day.  Each commissioned chamber does, however,
        require a daily crop/nutrient-loop inspection, and ripe biomass enters
        the food ledger only after a human harvest-and-pack task. A botany lead
        owns routine work; a fit cross-trained crew member may intervene only
        after the configured service grace has actually expired.
        """
        effects = self._recipes_cache.get(
            "greenhouse", {}
        ).get("output", {}).get("effects", {})
        greenhouses = [
            structure for structure in getattr(self, "placed_structures", [])
            if structure.get("type") == "greenhouse"
            and self._greenhouse_work_available(str(structure.get("id", "")))
        ]
        if not greenhouses:
            return None

        service_interval = ticks_for_minutes(
            float(effects.get("crop_service_interval_hours", 24.0)) * 60.0,
            self.tick_minutes,
        )
        service_grace = ticks_for_minutes(
            float(effects.get("crop_service_grace_hours", 12.0)) * 60.0,
            self.tick_minutes,
        )
        harvest_interval = ticks_for_minutes(
            float(effects.get("harvest_batch_interval_hours", 24.0)) * 60.0,
            self.tick_minutes,
        )
        food_store = float(getattr(
            self, "_colony_resources", {}
        ).get("food_reserve_kcal", 0.0))
        food_capacity = max(
            food_store,
            float(getattr(self, "food_storage_capacity_kcal", food_store)),
        )
        harvest_headroom = max(0.0, food_capacity - food_store)

        claimed_greenhouses = set()
        for crew in getattr(self, "agents", []):
            if crew.id == agent.id:
                continue
            crew_target = (
                crew.action.target
                if isinstance(getattr(crew.action, "target", None), dict)
                else {}
            )
            if (
                getattr(crew.action, "action_type", "")
                in {"move", "arrived", "farm"}
                and crew_target.get("greenhouse_id")
            ):
                claimed_greenhouses.add(str(
                    crew_target["greenhouse_id"]
                ))

        jobs = []
        for greenhouse in greenhouses:
            greenhouse_id = str(greenhouse.get("id", ""))
            if greenhouse_id in claimed_greenhouses:
                continue
            commissioned_tick = int(greenhouse.get(
                "commissioned_tick", greenhouse.get("completed_tick", tick)
            ) or 0)
            last_service = int(greenhouse.get(
                "last_crop_service_tick", commissioned_tick
            ) or 0)
            service_age = max(0, tick - last_service)
            if service_age >= service_interval:
                jobs.append({
                    "priority": 0 if service_age > service_interval + service_grace else 1,
                    "critical": service_age > service_interval + service_grace,
                    "sub_action": "crop_care",
                    "greenhouse": greenhouse,
                })

            last_harvest = int(greenhouse.get(
                "last_crop_harvest_tick", commissioned_tick
            ) or 0)
            ripe_kcal = max(0.0, float(greenhouse.get(
                "unharvested_food_kcal", 0.0
            )))
            if (
                harvest_headroom > 0.0
                and ripe_kcal > 0.0
                and tick - last_harvest >= harvest_interval
            ):
                jobs.append({
                    "priority": 2,
                    "critical": False,
                    "sub_action": "harvest",
                    "greenhouse": greenhouse,
                    "ripe_kcal": ripe_kcal,
                })
        if not jobs:
            return None

        jobs.sort(key=lambda job: (
            int(job["priority"]),
            -float(job.get("ripe_kcal", 0.0)),
            str(job["greenhouse"].get("id", "")),
        ))

        def dispatchable(crew: Agent) -> bool:
            if getattr(crew.status, "value", str(crew.status)) in {
                "dead", "incapacitated"
            }:
                return False
            if isinstance(getattr(crew, "_active_expedition", None), dict):
                return False
            if getattr(crew.action, "ticks_remaining", 0) > 0:
                return False
            return bool(
                crew.needs.energy > 65.0
                and crew.needs.hunger > 45.0
                and crew.needs.thirst > 45.0
                and crew.needs.o2_supply > 40.0
                and 42.0 < crew.needs.temperature_stress < 75.0
            )

        available = [
            crew for crew in getattr(self, "agents", [agent])
            if dispatchable(crew)
        ]
        assignments: list[tuple[Agent, dict]] = []
        reserved_greenhouses: set[str] = set()
        remaining_headroom = harvest_headroom
        for job in jobs:
            if not available:
                break
            greenhouse_id = str(job["greenhouse"].get("id", ""))
            if greenhouse_id in reserved_greenhouses:
                continue
            if job["sub_action"] == "harvest" and remaining_headroom <= 0.0:
                continue
            specialists = [
                crew for crew in available
                if crew.competency.get_primary_domain() == "botany_bio"
            ]
            eligible = specialists
            if not eligible and job["critical"]:
                eligible = [
                    crew for crew in available
                    if float(getattr(crew.competency, "botany_bio", 0)) >= 4.0
                ]
            if not eligible:
                continue
            greenhouse = job["greenhouse"]
            assigned = min(eligible, key=lambda crew: (
                -int(getattr(crew.competency, "botany_bio", 0)),
                max(
                    abs(crew.x - int(greenhouse.get("x", crew.x))),
                    abs(crew.y - int(greenhouse.get("y", crew.y))),
                ),
                crew.id,
            ))
            assignments.append((assigned, job))
            available.remove(assigned)
            reserved_greenhouses.add(greenhouse_id)
            if job["sub_action"] == "harvest":
                remaining_headroom -= min(
                    remaining_headroom, float(job.get("ripe_kcal", 0.0))
                )

        assignment = next((
            job for assigned, job in assignments if assigned.id == agent.id
        ), None)
        if assignment is None:
            return None
        greenhouse = assignment["greenhouse"]
        sub_action = assignment["sub_action"]
        return {
            "action": "farm",
            "target": {
                "greenhouse_id": str(greenhouse.get("id", "")),
                "x": int(greenhouse.get("x", agent.x)),
                "y": int(greenhouse.get("y", agent.y)),
                "destination": "greenhouse_workstation",
                "sub_action": sub_action,
                "greenhouse_duty": True,
                "critical_crop_service": bool(assignment["critical"]),
            },
            "reasoning": (
                f"{agent.name} performing scheduled crop-health and nutrient-loop "
                f"inspection in greenhouse {greenhouse.get('id')}"
                if sub_action == "crop_care" else
                f"{agent.name} harvesting and packing ripe crop from "
                f"greenhouse {greenhouse.get('id')}"
            ),
            "deterministic": True,
        }

    def _structure_maintenance_decision(
        self, agent: Agent, structures_built: dict
    ) -> Optional[dict]:
        """Assign one technician before a maintainable asset reaches failure.

        Geological survey and capacity work can run for thousands of ticks.
        Keeping maintenance below those branches let every furnace and forge
        collapse while the scanner kept looking for one missing feedstock.
        This dispatcher is deliberately single-owner and leaves interrupted
        excavation/construction contracts intact for resumption afterward.
        """
        placed = getattr(self, "placed_structures", [])
        candidates = [
            structure for structure in placed
            if not structure.get("destroyed", False)
            and not structure.get("under_construction", False)
            and int(structures_built.get(structure.get("type", ""), 0)) > 0
            and (
                float(structure.get("health", 1.0)) < 0.85
                or bool(structure.get("maintenance_due", False))
            )
            and self._recipes_cache.get(
                structure.get("type", ""), {}
            ).get("maintenance", {}).get("interval_ticks", 0)
        ]
        if not candidates:
            return None
        worst = min(
            candidates,
            key=lambda structure: (
                0 if float(structure.get("health", 1.0)) < 0.85 else 1,
                float(structure.get("health", 1.0)),
                -int(structure.get("maintenance_overdue_cycles", 0)),
                str(structure.get("id", "")),
            ),
        )
        structure_type = worst.get("type")
        structure_id = worst.get("id")
        maintenance = self._recipes_cache.get(
            structure_type, {}
        ).get("maintenance", {})
        inspection_only = float(worst.get("health", 1.0)) >= 0.85
        repair_materials = (
            {} if inspection_only
            else dict(maintenance.get("repair_materials", {}))
        )
        # Delivered starter kits remain visible in depot telemetry but are
        # physically sealed/reserved for their certified structure. Match the
        # engine's consumption view so maintenance cannot promise those parts.
        pooled = self._materials_available_for_recipe(
            "__maintenance__", self._pooled_materials(), structures_built
        )
        missing = next((
            (material, int(quantity) - int(pooled.get(material, 0)))
            for material, quantity in repair_materials.items()
            if int(pooled.get(material, 0)) < int(quantity)
        ), None)

        def continuing(crew: Agent) -> bool:
            target = crew.action.target if isinstance(crew.action.target, dict) else {}
            return (
                target.get("maintenance_action") in {"repair", "inspection"}
                and target.get("structure_id") == structure_id
            )

        def dispatchable(crew: Agent) -> bool:
            """Do not reserve the job to someone trapped in an earlier lock."""
            if continuing(crew):
                return True
            expedition = getattr(crew, "_active_expedition", None)
            if (
                isinstance(expedition, dict)
                and expedition.get("status") in {"outbound", "working", "returning"}
            ):
                return False
            action_type = getattr(crew.action, "action_type", "")
            target = crew.action.target if isinstance(crew.action.target, dict) else {}
            if action_type == "sleep" and crew.action.ticks_remaining > 0:
                return False
            # These routes are persisted before the maintenance dispatcher is
            # reached. Selecting their custodian would make every available
            # lower-ranked engineer stand down while no repair order was ever
            # emitted.
            if action_type in {"move", "prospect"} and (
                target.get("survey_action") == "portable_scanner"
                or target.get("survey_action") == "scanner_recharge"
            ):
                return False
            return True

        living = [
            crew for crew in getattr(self, "agents", [agent])
            if getattr(crew.status, "value", str(crew.status))
            not in {"dead", "incapacitated"}
            and crew.needs.energy > 65.0
            and crew.needs.hunger > 40.0
            and crew.needs.thirst > 40.0
            and crew.needs.o2_supply > 40.0
            and dispatchable(crew)
        ]
        if not living:
            return None

        assigned = min(
            living,
            key=lambda crew: (
                0 if continuing(crew) else 1,
                -int(getattr(crew.competency, "engineering", 0)),
                max(
                    abs(crew.x - int(worst.get("x", crew.x))),
                    abs(crew.y - int(worst.get("y", crew.y))),
                ),
                crew.id,
            ),
        )
        if assigned.id != agent.id:
            return None

        if missing:
            dependency = self._material_dependency_action(
                missing[0], missing[1], f"{structure_type} maintenance",
                pooled, set(),
            )
            return {
                **dependency,
                "target": {
                    **dependency.get("target", {}),
                    "maintenance_for": structure_id,
                    "maintenance_structure": structure_type,
                },
                "reasoning": (
                    f"Preventive {structure_type} service requires "
                    f"{missing[1]} more {missing[0]} before dispatch"
                ),
                "deterministic": True,
            }

        return {
            "action": "repair",
            "target": {
                "structure": structure_type,
                "structure_id": structure_id,
                "x": worst.get("x"),
                "y": worst.get("y"),
                "maintenance": "external_structure_repair",
                "maintenance_action": (
                    "inspection" if inspection_only else "repair"
                ),
                "inspection_only": inspection_only,
                "repair_materials": repair_materials,
            },
            "reasoning": (
                f"{agent.name} performing "
                f"{'scheduled condition inspection' if inspection_only else 'corrective repair'} "
                f"on {structure_type} "
                f"at {float(worst.get('health', 1.0)):.0%} integrity"
            ),
            "deterministic": True,
        }

    def _record_shared_supply_assignment(
        self,
        *,
        agent: Agent,
        action: Optional[str],
        recipe_name: str,
        tick: int,
        order: dict,
        assignments: dict,
        preserve_construction_order: bool,
    ) -> None:
        """Record logistics without replacing an active construction order."""
        assignments[agent.id] = action
        if preserve_construction_order:
            # The site recipe, stage, id and retained builders are the source
            # of truth for the in-progress structure.  Support contracts may
            # target the next milestone, but must never make that future
            # recipe look like a second active build site.
            self.shared_work_order["assignments"] = assignments
            self.shared_work_order["updated_tick"] = tick
            return
        self.shared_work_order = {
            "recipe": recipe_name,
            "stage": "materials",
            "assignments": assignments,
            "updated_tick": tick,
            "required_count": order.get("required_count"),
        }

    def _shared_supply_decision(
        self,
        *,
        agent: Agent,
        tick: int,
        structures_built: dict,
        living: list[Agent],
        recipe_name: str,
        order: dict,
        assignments: dict,
        lead: Optional[Agent] = None,
        preserve_construction_order: bool = False,
        target_context: Optional[dict] = None,
    ) -> Optional[dict]:
        """Continue or allocate one persistent, batch-aware BOM supply job.

        Both ordinary material preparation and the non-builder crew beside an
        active site use this path.  It deliberately never emits ``build``;
        construction authorization remains in the caller, so preparing the
        next milestone cannot create a second site while one is in progress.
        """
        target_context = dict(target_context or {})
        pooled = self._materials_available_for_recipe(
            recipe_name, self._pooled_materials(), structures_built
        )
        depot = getattr(self, "central_depot_inventory", {})
        lz_x = getattr(self, "lz_x", getattr(agent, "spawn_x", agent.x))
        lz_y = getattr(self, "lz_y", getattr(agent, "spawn_y", agent.y))
        staged = self._materials_available_for_recipe(
            recipe_name, dict(depot), structures_built
        )
        for crew in living:
            if max(abs(crew.x - lz_x), abs(crew.y - lz_y)) <= 1:
                for material, quantity in crew.inventory.materials.items():
                    staged[material] = staged.get(material, 0) + quantity

        dependency_materials = self._dependency_material_names(recipe_name)
        useful_carried = any(
            quantity > 0 and material in dependency_materials
            for material, quantity in agent.inventory.materials.items()
        )
        if useful_carried:
            self._record_shared_supply_assignment(
                agent=agent,
                action="stage_materials",
                recipe_name=recipe_name,
                tick=tick,
                order=order,
                assignments=assignments,
                preserve_construction_order=preserve_construction_order,
            )
            return {
                "action": "deposit_materials",
                "target": {
                    "x": lz_x,
                    "y": lz_y,
                    "destination": "central_depot",
                    "mission_action": "deposit_materials",
                    "shared_work_order": True,
                    "capacity_recipe": recipe_name,
                    **target_context,
                },
                "reasoning": f"Staging {recipe_name} materials at the central depot",
                "deterministic": True,
            }

        contract = getattr(agent, "_shared_work_contract", None)
        if isinstance(contract, dict):
            contract_released_for_replan = False
            same_recipe = contract.get("recipe") == recipe_name
            stock_key = contract.get("stock_key")
            target_stock = int(contract.get("target_stock", 0))
            contract_action = str(contract.get("action", "")).lower()
            current_bom_need = self._bom_outstanding_requirements(
                recipe_name, pooled
            ).get(stock_key, 0)
            active_wip = self._shared_contract_has_active_wip(agent, contract)
            within_review_window = tick <= int(contract.get("review_tick", tick))
            within_shift_window = tick <= int(
                contract.get("shift_end_tick", contract.get("review_tick", tick))
            )
            target_unfilled = (
                bool(stock_key)
                and pooled.get(stock_key, 0) < target_stock
            )
            # A started machine batch is irreversible and survives a planning
            # lease through hand-off.  Field work is instead a bounded human
            # shift: even if a parent subassembly makes its immediate BOM need
            # disappear, the astronaut may finish the promised small load, but
            # neither an excavation marker nor an expedition can bypass the
            # stock target or shift boundary forever.
            irreversible_wip = active_wip and contract_action == "refine"
            bounded_field_wip = (
                active_wip
                and contract_action == "gather"
                and target_unfilled
                and within_review_window
                and within_shift_window
            )
            still_needed = (
                bool(stock_key)
                and (
                    irreversible_wip
                    or bounded_field_wip
                    or (
                        current_bom_need > 0
                        and target_unfilled
                    )
                )
            )
            if same_recipe and still_needed and (
                (within_review_window and within_shift_window)
                or irreversible_wip
            ):
                contract_action = contract["action"]
                contract_target = dict(contract.get("target", {}))
                if contract_action == "refine":
                    component_recipe = self._recipes_cache.get(stock_key, {})
                    required_machine = component_recipe.get("requires_structure")
                    machine_busy = self._all_machine_slots_busy(
                        required_machine, living, agent, tick=tick
                    )
                    missing_input = next((
                        (material, needed - staged.get(material, 0))
                        for material, needed in component_recipe.get(
                            "materials", {}
                        ).items()
                        if staged.get(material, 0) < needed
                    ), None)
                    energy_needed = float(
                        component_recipe.get("energy_kwh_per_batch", 0.0)
                    )
                    energy_available = float(
                        getattr(self, "_colony_resources", {}).get(
                            "energy_stored_kwh", 0.0
                        )
                    )
                    if missing_input and not active_wip:
                        dependency = self._material_dependency_action(
                            missing_input[0], missing_input[1],
                            recipe_name, pooled, set()
                        )
                        dependency_executable = (
                            dependency.get("action") != "refine"
                            or self._refine_candidate_is_executable(
                                dependency,
                                agent,
                                living,
                                structures_built,
                                tick,
                            )
                        )
                        if dependency_executable:
                            decision = {
                                **dependency,
                                "target": {
                                    **dependency.get("target", {}),
                                    "shared_work_order": True,
                                    "capacity_recipe": recipe_name,
                                    "supporting_work_contract": contract.get("id"),
                                    **target_context,
                                },
                                "reasoning": (
                                    f"Persistent {recipe_name} work contract: "
                                    f"prepare {missing_input[0]} before the "
                                    f"scheduled {stock_key} batch"
                                ),
                                "deterministic": True,
                            }
                            self._record_shared_supply_assignment(
                                agent=agent,
                                action=decision["action"],
                                recipe_name=recipe_name,
                                tick=tick,
                                order=order,
                                assignments=assignments,
                                preserve_construction_order=(
                                    preserve_construction_order
                                ),
                            )
                            return decision

                        # A dependency can become unavailable between two crew
                        # turns when another operator reserves the last machine
                        # slot or its staged BOM.  Drop only this reversible
                        # planning lease and let the live support queue choose a
                        # different executable job below.  The physical parent
                        # BOM remains outstanding and no RL penalty is emitted.
                        agent._shared_work_contract = None
                        contract_released_for_replan = True
                    if not contract_released_for_replan and (
                        (machine_busy or energy_available < energy_needed)
                        and not active_wip
                    ):
                        standby = self._raw_support_action(
                            recipe_name, pooled,
                            next((i for i, crew in enumerate(living)
                                  if crew.id == agent.id), 0),
                            agent,
                            tick,
                        )
                        decision = {
                            **standby,
                            "target": {
                                **standby.get("target", {}),
                                "supporting_work_contract": contract.get("id"),
                                **target_context,
                            },
                            "reasoning": (
                                f"Persistent {recipe_name} work contract: "
                                f"stockpile feedstock while the {required_machine} "
                                "or its power bus is occupied"
                            ),
                        }
                        self._record_shared_supply_assignment(
                            agent=agent,
                            action=decision["action"],
                            recipe_name=recipe_name,
                            tick=tick,
                            order=order,
                            assignments=assignments,
                            preserve_construction_order=(
                                preserve_construction_order
                            ),
                        )
                        return decision
                if not contract_released_for_replan:
                    decision = {
                        "action": contract_action,
                        "target": {
                            **contract_target,
                            "shared_work_order": True,
                            "work_contract_id": contract.get("id"),
                            "target_stock": target_stock,
                            **target_context,
                        },
                        "reasoning": (
                            f"Persistent {recipe_name} work contract: continue "
                            f"{contract_action} for {stock_key} until pooled stock "
                            f"reaches {target_stock}"
                        ),
                        "deterministic": True,
                    }
                    self._record_shared_supply_assignment(
                        agent=agent,
                        action=decision["action"],
                        recipe_name=recipe_name,
                        tick=tick,
                        order=order,
                        assignments=assignments,
                        preserve_construction_order=preserve_construction_order,
                    )
                    return decision
            if contract_released_for_replan:
                pass
            elif contract_action == "gather":
                self._close_bounded_gather_contract(agent, contract)
            elif not active_wip:
                agent._shared_work_contract = None

        planning_pooled = dict(pooled)
        for material, quantity in self._contract_reservations(
            recipe_name, agent, tick
        ).items():
            planning_pooled[material] = (
                planning_pooled.get(material, 0) + quantity
            )
        support_actions = self._shared_dependency_actions(
            recipe_name, planning_pooled
        )
        if not support_actions:
            # The next milestone is fully staged.  Waiting/planning is useful
            # here, but starting it would violate the one-site construction
            # invariant enforced by the caller.
            if preserve_construction_order:
                decision = {
                    "action": "stand_watch",
                    "target": {
                        "recipe": recipe_name,
                        "operations_watch": True,
                        "materials_ready": True,
                        "shared_work_order": True,
                        "ticks": max(
                            1, ticks_for_minutes(60.0, self.tick_minutes)
                        ),
                        **target_context,
                    },
                    "reasoning": (
                        f"{recipe_name} BOM is staged; monitoring base, rover "
                        "and life-support telemetry while the fixed site crew "
                        "completes the active assembly shift"
                    ),
                    "deterministic": True,
                }
                self._record_shared_supply_assignment(
                    agent=agent,
                    action=decision["action"],
                    recipe_name=recipe_name,
                    tick=tick,
                    order=order,
                    assignments=assignments,
                    preserve_construction_order=True,
                )
                return decision
            return None

        ranked_support = sorted(living, key=lambda crew: (
            0 if crew.id != getattr(lead, "id", None) else 1,
            crew.id,
        ))
        support_index = next(
            (index for index, crew in enumerate(ranked_support)
             if crew.id == agent.id),
            0,
        )
        gather_actions = {
            action.get("target", {}).get("resource"): action
            for action in support_actions
            if action.get("action") == "gather"
            and action.get("target", {}).get("resource")
        }
        claimed_gather_resources = self._active_raw_campaign_claims(
            recipe_name, tick, agent.id
        )
        critical_gathers = self._prioritize_live_bom_resources(
            gather_actions, recipe_name, planning_pooled
        )
        unclaimed_gathers = [
            resource for resource in critical_gathers
            if resource not in claimed_gather_resources
        ]
        executable_refines = self._executable_refine_support_actions(
            support_actions, agent, living, staged, tick=tick
        )
        if executable_refines:
            # Keep each real machine fed while the remaining crew prospect and
            # recover distinct raw bottlenecks in parallel.
            chosen = executable_refines[0]
        elif unclaimed_gathers:
            # Give each free worker a different live BOM bottleneck before
            # opening a second extraction claim for a large single deficit.
            chosen = gather_actions[unclaimed_gathers[0]]
        else:
            chosen = support_actions[support_index % len(support_actions)]
        if chosen.get("action") == "refine":
            output = chosen.get("target", {}).get("output")
            component_recipe = self._recipes_cache.get(output, {})
            required_machine = component_recipe.get("requires_structure")
            machine_busy = self._all_machine_slots_busy(
                required_machine, living, agent, tick=tick
            )
            energy_needed = float(
                component_recipe.get("energy_kwh_per_batch", 0.0)
            )
            energy_available = float(
                getattr(self, "_colony_resources", {}).get(
                    "energy_stored_kwh", 0.0
                )
            )
            component_staged = all(
                staged.get(material, 0) >= needed
                for material, needed in component_recipe.get(
                    "materials", {}
                ).items()
            )
            if not agent.can_craft(
                {
                    **component_recipe,
                    "materials": {},
                    "requires_tool": False,
                }
            ).get("can_craft", False):
                capable = [
                    crew for crew in ranked_support
                    if crew.can_craft(
                        {
                            **component_recipe,
                            "materials": {},
                            "requires_tool": False,
                        }
                    ).get("can_craft", False)
                ]
                if not capable or agent.id != capable[0].id:
                    gather_actions = [
                        action for action in support_actions
                        if action.get("action") == "gather"
                    ]
                    if not gather_actions:
                        chosen = self._raw_support_action(
                            recipe_name, pooled, support_index, agent, tick
                        )
                    else:
                        chosen = gather_actions[
                            support_index % len(gather_actions)
                        ]
            elif (
                machine_busy
                or energy_available < energy_needed
                or not component_staged
            ):
                chosen = self._raw_support_action(
                    recipe_name, pooled, support_index, agent, tick
                )

        chosen = {
            **chosen,
            "target": {
                **chosen.get("target", {}),
                "shared_work_order": True,
                "capacity_recipe": recipe_name,
                **target_context,
            },
            "reasoning": (
                f"Shared {recipe_name} bill of materials: "
                f"{chosen.get('reasoning', 'support task')}"
            ),
            "deterministic": True,
        }
        chosen_target = chosen.get("target", {})
        stock_key = chosen_target.get("resource") or chosen_target.get("output")
        if stock_key:
            quantity_needed = max(
                1, int(chosen_target.get("quantity_needed", 1))
            )
            if chosen.get("action") == "refine":
                # A machine cycle cannot reserve an arbitrary slice of its
                # output.  One contract is exactly one configured batch.
                quantity_reserved = max(
                    1,
                    int(
                        self._recipes_cache.get(stock_key, {})
                        .get("output", {})
                        .get("quantity", 1)
                    ),
                )
            else:
                quantity_reserved = min(12, quantity_needed)
            # Include earlier workers' reservations in this worker's bounded
            # stock target.  Thus two free machines can own batch 1 and batch
            # 2 instead of both believing they are responsible for batch 1.
            target_stock = (
                planning_pooled.get(stock_key, 0) + quantity_reserved
            )
            if chosen.get("action") == "gather":
                review_window = 96
            else:
                component_duration = int(
                    self._recipes_cache.get(stock_key, {}).get(
                        "base_duration_ticks", 48
                    )
                )
                review_window = max(48, component_duration + 24)
            agent._shared_work_contract = {
                "id": f"{recipe_name}:{agent.id}:{tick}:{stock_key}",
                "recipe": recipe_name,
                "action": chosen.get("action"),
                "target": dict(chosen_target),
                "stock_key": stock_key,
                "target_stock": target_stock,
                "quantity_reserved": quantity_reserved,
                "review_tick": tick + review_window,
                # A NASA-length operations shift prevents permanent ownership
                # of one stalled branch. Irreversibly committed machine or
                # excavation WIP remains protected across the handover.
                "shift_start_tick": tick,
                "shift_end_tick": tick + int(getattr(
                    self, "manufacturing_shift_ticks", 78
                )),
                "critical_path_ticks": int(
                    chosen_target.get("critical_path_ticks", 0)
                ),
            }
            if chosen.get("action") == "refine":
                component_recipe = self._recipes_cache.get(stock_key, {})
                agent._shared_work_contract.update({
                    "machine_type": (
                        chosen_target.get("machine_type")
                        or chosen_target.get("structure")
                        or component_recipe.get("requires_structure")
                    ),
                    "slot_claim_tick": int(tick),
                })
        self._record_shared_supply_assignment(
            agent=agent,
            action=chosen.get("action"),
            recipe_name=recipe_name,
            tick=tick,
            order=order,
            assignments=assignments,
            preserve_construction_order=preserve_construction_order,
        )
        return chosen

    def _shared_colony_work_decision(
        self, agent: Agent, tick: int, structures_built: dict
    ) -> Optional[dict]:
        """Assign one coordinated capacity job to each fit crew member."""
        living = [
            crew for crew in getattr(self, "agents", [agent])
            if getattr(crew.status, "value", str(crew.status))
            not in ("dead", "incapacitated")
        ]
        if not living:
            return None
        pooled = self._pooled_materials()
        order = self._select_shared_capacity_order(
            pooled, structures_built, tick
        )
        if order is None:
            self.shared_work_order = {}
            return None

        recipe_name = order["recipe"]
        pooled = self._materials_available_for_recipe(
            recipe_name, pooled, structures_built
        )
        recipe = self._recipes_cache.get(recipe_name, {})
        site = order.get("site")
        existing_order = getattr(self, "shared_work_order", {})
        assignments = dict(existing_order.get("assignments", {})) if (
            existing_order.get("recipe") == recipe_name
            and existing_order.get("updated_tick") == tick
        ) else {}

        if site:
            if not site.get("materials_committed", True):
                ranked_logistics = sorted(living, key=lambda crew: (
                    -int(getattr(crew.competency, "engineering", 0)),
                    -int(getattr(crew.genome, "strength", 0)),
                    crew.id,
                ))
                coordinator = ranked_logistics[0] if ranked_logistics else agent
                required_count = int(order.get("required_count") or 1)
                another_unit_needed = (
                    int(structures_built.get(recipe_name, 0)) + 1
                    < required_count
                )
                if agent.id == coordinator.id or not another_unit_needed:
                    return {
                        "action": "stand_watch",
                        "target": {
                            "cargo_logistics": True,
                            "recipe": recipe_name,
                            "struct_id": site.get("id"),
                            "x": site.get("x"),
                            "y": site.get("y"),
                            "construction_phase": site.get(
                                "construction_phase", "cargo_staging"
                            ),
                        },
                        "reasoning": (
                            f"{agent.name} auditing the restrained {recipe_name} "
                            "manifest while dedicated heavy transporters stage it; "
                            "assembly cannot begin before final delivery"
                        ),
                        "deterministic": True,
                    }
                return self._shared_supply_decision(
                    agent=agent,
                    tick=tick,
                    structures_built=structures_built,
                    living=living,
                    recipe_name=recipe_name,
                    order=order,
                    assignments=assignments,
                    preserve_construction_order=True,
                    target_context={
                        "construction_support_for": site.get("id"),
                        "cargo_delivery_pending": True,
                    },
                )
            construction = recipe.get("construction", {})
            crew_needed = max(1, int(construction.get("recommended_crew", 2)))
            ranked = sorted(living, key=lambda crew: (
                -int(getattr(crew.competency, "engineering", 0)),
                -int(getattr(crew.genome, "strength", 0)),
                max(abs(crew.x - site.get("x", crew.x)), abs(crew.y - site.get("y", crew.y))),
                crew.id,
            ))
            # Keep the same site crew across ticks. Re-ranking solely by current
            # distance made an astronaut lose the assignment while walking to
            # the site; the new, closer replacement then displaced somebody
            # else and the original worker immediately returned to the lander.
            # Only death/incapacitation removes a retained builder. Ordinary
            # sleep or EVA servicing pauses that person's contribution without
            # abandoning the construction contract.
            same_site_order = (
                existing_order.get("recipe") == recipe_name
                and existing_order.get("stage") == "construction"
                and existing_order.get("site_id") == site.get("id")
            )
            previous_crew_ids = (
                list(existing_order.get("construction_crew_ids", []))
                if same_site_order else []
            )
            if same_site_order and not previous_crew_ids:
                previous_crew_ids = [
                    crew_id
                    for crew_id, role in existing_order.get("assignments", {}).items()
                    if role == "construction"
                ]
            living_by_id = {crew.id: crew for crew in living}
            primary_crew = []
            primary_ids = set()
            for crew_id in previous_crew_ids:
                retained = living_by_id.get(crew_id)
                if retained is None or crew_id in primary_ids:
                    continue
                primary_crew.append(retained)
                primary_ids.add(crew_id)
                if len(primary_crew) >= crew_needed:
                    break
            for crew in ranked:
                if len(primary_crew) >= min(crew_needed, len(ranked)):
                    break
                if crew.id in primary_ids:
                    continue
                primary_crew.append(crew)
                primary_ids.add(crew.id)

            def available_for_site(crew: Agent) -> bool:
                expedition = getattr(crew, "_active_expedition", None)
                if isinstance(expedition, dict):
                    # A rover pair owns its work front until both return;
                    # temporary field drinking must not dispatch a third walker.
                    return (
                        expedition.get("kind") == "construction_support"
                        and expedition.get("site_id") == site.get("id")
                    )
                action_type = str(getattr(crew.action, "action_type", "") or "")
                if action_type in {
                    "sleep", "medical_rest", "eat", "drink", "refill_o2",
                    "service_suit", "refine", "craft", "operate_machine",
                }:
                    return False
                if isinstance(getattr(crew, "_active_expedition", None), dict):
                    return False
                # This is availability for a local work assignment, not an
                # EVA certificate. Do not filter out a worn suit here: its
                # owner must receive the order to reach the real service gate.
                # Remote paired preparation is handled separately below.
                if crew.needs.energy < 65.0:
                    return False
                if crew.needs.thirst <= 30.0 or crew.needs.hunger <= 25.0:
                    return False
                return True

            # Primary owners keep the work contract, while rested teammates
            # cover their sleep/self-care periods. This is a real shift handoff:
            # it preserves one supervised site and does not create extra labor.
            lz_x = int(getattr(self, "lz_x", site.get("x", 0)))
            lz_y = int(getattr(self, "lz_y", site.get("y", 0)))
            rover_min_distance = int(getattr(
                self, "crew_rover_min_distance_cells", 12
            ))
            remote_rover_site = (
                abs(int(site.get("x", lz_x)) - lz_x)
                + abs(int(site.get("y", lz_y)) - lz_y)
                >= rover_min_distance
            )
            if remote_rover_site:
                # A two-seat rover sortie is a fixed crew operation.  Replacing
                # one sleeping owner with a different person every scheduler
                # pass created three-way rendezvous churn at the airlock.  The
                # owners retain the assignment while completing bounded food,
                # sleep, O2 and suit preparation; only incapacity replaces one.
                assigned = list(primary_crew)
            else:
                assigned = [
                    crew for crew in primary_crew if available_for_site(crew)
                ]
                active_ids = {crew.id for crew in assigned}
                for crew in ranked:
                    if len(assigned) >= min(crew_needed, len(ranked)):
                        break
                    if crew.id in active_ids or not available_for_site(crew):
                        continue
                    assigned.append(crew)
                    active_ids.add(crew.id)
            # A materials-stage assignment must not leak into the construction
            # stage.  Previously every crew member that had helped with the BOM
            # remained present in ``assignments`` and therefore all six were
            # sent to the same site despite the recipe's recommended crew size.
            assignments = {}
            for crew in assigned:
                assignments[crew.id] = "construction"
            self.shared_work_order = {
                "recipe": recipe_name,
                "stage": "construction",
                "site_id": site.get("id"),
                "construction_crew_ids": [crew.id for crew in primary_crew],
                "assignments": assignments,
                "updated_tick": tick,
            }
            # A relief assignment replaces, rather than supplements, the
            # previous site's active crew. Multi-tick BUILD/MOVE actions can
            # otherwise outlive yesterday's assignment and make the physics
            # layer clear the same surplus worker every tick while the planner
            # immediately sends them back. Release only stale routes to this
            # exact site; unrelated work remains untouched.
            for crew in living:
                if crew.id in assignments:
                    continue
                crew_target = (
                    crew.action.target
                    if isinstance(getattr(crew.action, "target", None), dict)
                    else {}
                )
                same_site = (
                    crew_target.get("struct_id") == site.get("id")
                    or (
                        crew_target.get("recipe") == recipe_name
                        and crew_target.get("x") == site.get("x")
                        and crew_target.get("y") == site.get("y")
                    )
                )
                if (
                    same_site
                    and getattr(crew.action, "action_type", "")
                    in {"move", "arrived", "build"}
                ):
                    crew.action.clear()
            if agent.id in assignments:
                return {
                    "action": "build",
                    "target": {
                        "recipe": recipe_name,
                        "struct_id": site.get("id"),
                        "x": site.get("x"),
                        "y": site.get("y"),
                        "shared_work_order": True,
                    },
                    "reasoning": (
                        f"Shared work order: {agent.name} reports to {recipe_name} "
                        f"site ({site.get('progress', 0.0):.0%} complete)"
                    ),
                    "deterministic": True,
                }
            # The remaining crew must not fall through to stale individual
            # plans that also say "build".  Keep them on distinct upstream
            # logistics jobs for the next module while the bounded site crew
            # performs the current installation.
            # Site labor and supply-chain labor run in parallel.  Non-builders
            # prepare the same learned objective; choosing a different "next"
            # recipe here would quietly restore a hard-coded build order below
            # the strategic policy.
            support_recipe = recipe_name
            support_context = {
                "construction_support_for": site.get("id"),
                "next_bootstrap_milestone": False,
            }
            support_pool = self._materials_available_for_recipe(
                support_recipe, pooled, structures_built
            )
            if not self._shared_dependency_actions(support_recipe, support_pool):
                # The current module is already staged. Put spare crew and the
                # landed shop on the total known remaining manifest. This
                # exposes the mission's planned local-manufacturing share at
                # the beginning of the year instead of waiting until a later
                # structure consumes the last delivered pipe or seal.
                forecast_recipe, _, remaining_counts = (
                    self._refresh_mission_forecast_recipe(structures_built)
                )
                forecast_pool = self._materials_available_for_recipe(
                    forecast_recipe, pooled, structures_built
                )
                if self._shared_dependency_actions(
                    forecast_recipe, forecast_pool
                ):
                    support_recipe = forecast_recipe
                    support_pool = forecast_pool
                    support_context.update({
                        "mission_capacity_forecast": True,
                        "remaining_structure_counts": remaining_counts,
                    })
                else:
                    # No aggregate material debt remains. Keep the bounded
                    # learned next-order path for maintenance or an objective
                    # introduced dynamically during the run.
                    queued = getattr(self, "_queued_construction_supply", {})
                    if (queued.get("site_id") != site.get("id")
                        or (not queued.get("recipe") and tick >= queued.get("review_tick", 0))):
                        queued = {}
                    if not queued:
                        options = []
                        for name in self.MISSION_FORECAST_STRUCTURES:
                            next_recipe = self._recipes_cache.get(name)
                            if not next_recipe or name == recipe_name:
                                continue
                            prerequisite = next_recipe.get("requires_structure")
                            if prerequisite and not structures_built.get(prerequisite, 0):
                                continue
                            status = self.colony.get_structure_capacity_status(name)
                            if self._planning_fulfillment(name, status, structures_built) >= 1.0:
                                continue
                            next_pool = self._materials_available_for_recipe(name, pooled, structures_built)
                            if not self._shared_dependency_actions(name, next_pool):
                                continue
                            if not self._bootstrap_order_feasibility(name, pooled).get("actionable"):
                                continue
                            options.append({"action": "prepare_capacity", "target": {"recipe": name}})
                        if options:
                            selected = self.select_action_via_rl(
                                agent, self.get_colony_strategy_state(structures_built) + "|planning:upstream_bom", options
                            )
                            queued = {"site_id": site.get("id"), "recipe": selected["target"]["recipe"]}
                            self._queued_construction_supply = queued
                        else:
                            # Do not rebuild the whole candidate/BOM graph for
                            # every idle crew member on every tick.
                            self._queued_construction_supply = {
                                "site_id": site.get("id"), "recipe": None,
                                "review_tick": tick + ticks_for_minutes(60.0, self.tick_minutes),
                            }
                    if queued.get("recipe"):
                        support_recipe = queued["recipe"]

            support_order = {
                "recipe": support_recipe,
                "required_count": (
                    self.colony.get_structure_capacity_status(support_recipe)
                    or {}
                ).get("required_count"),
            }
            return self._shared_supply_decision(
                agent=agent,
                tick=tick,
                structures_built=structures_built,
                living=living,
                recipe_name=support_recipe,
                order=support_order,
                assignments=assignments,
                preserve_construction_order=True,
                target_context=support_context,
            )

        requirements = recipe.get("materials", {})
        depot = getattr(self, "central_depot_inventory", {})
        lz_x = getattr(self, "lz_x", getattr(agent, "spawn_x", agent.x))
        lz_y = getattr(self, "lz_y", getattr(agent, "spawn_y", agent.y))
        staged = self._materials_available_for_recipe(
            recipe_name, dict(depot), structures_built
        )
        for crew in living:
            if max(abs(crew.x - lz_x), abs(crew.y - lz_y)) <= 1:
                for material, quantity in crew.inventory.materials.items():
                    staged[material] = staged.get(material, 0) + quantity
        depot_ready = all(
            staged.get(material, 0) >= needed
            for material, needed in requirements.items()
        )
        capable_leads = [
            crew for crew in living
            if crew.can_craft({**recipe, "materials": {}}).get("can_craft", False)
        ]
        lead = max(capable_leads, key=lambda crew: (
            int(getattr(crew.competency, "engineering", 0)),
            int(getattr(crew.genome, "strength", 0)),
            crew.id,
        ), default=None)

        dependency_materials = self._dependency_material_names(recipe_name)
        useful_carried = any(
            quantity > 0 and material in dependency_materials
            for material, quantity in agent.inventory.materials.items()
        )
        if useful_carried:
            assignments[agent.id] = "stage_materials"
            self.shared_work_order = {
                "recipe": recipe_name, "stage": "logistics",
                "assignments": assignments, "updated_tick": tick,
            }
            return {
                "action": "deposit_materials",
                "target": {
                    "x": lz_x,
                    "y": lz_y,
                    "destination": "central_depot",
                    "mission_action": "deposit_materials",
                    "shared_work_order": True,
                },
                "reasoning": f"Staging {recipe_name} materials at the central depot",
                "deterministic": True,
            }

        if depot_ready and lead and agent.id == lead.id:
            agent._shared_work_contract = None
            assignments[agent.id] = "construction_lead"
            self.shared_work_order = {
                "recipe": recipe_name, "stage": "ready_to_build",
                "assignments": assignments, "updated_tick": tick,
                "required_count": order.get("required_count"),
            }
            return {
                "action": "build",
                "target": {
                    "recipe": recipe_name,
                    "capacity_target_count": order.get("required_count"),
                    "capacity_category": order.get("category"),
                    "shared_work_order": True,
                    "life_support_bootstrap": bool(order.get("bootstrap")),
                },
                "reasoning": (
                    f"Shared capacity order: lead assembly of {recipe_name} "
                    f"{structures_built.get(recipe_name, 0) + 1}/"
                    f"{order.get('required_count', '?')}"
                ),
                "deterministic": True,
            }

        return self._shared_supply_decision(
            agent=agent,
            tick=tick,
            structures_built=structures_built,
            living=living,
            recipe_name=recipe_name,
            order=order,
            assignments=assignments,
            lead=lead,
        )
    def _material_dependency_action(
        self,
        material: str,
        deficit: int,
        milestone: str,
        colony_mats: dict,
        visited: set[str],
    ) -> dict:
        """Resolve a raw/refined material dependency into one executable action."""
        if material in self.finite_cargo_materials:
            return {
                "action": "idle",
                "target": {
                    "finite_cargo_unavailable": material,
                    "quantity_needed": max(1, int(deficit)),
                },
                "reasoning": (
                    f"Cannot fabricate flight-qualified {material} in the "
                    f"surface shop; finite emergency cargo is exhausted"
                ),
                "deterministic": True,
            }
        raw_materials = {
            "iron_ore", "silica_sand", "basalt", "water_ice", "graphite",
            "sulfur", "chalcopyrite_ore", "regolith", "olivine",
        }
        if material in raw_materials:
            return {
                "action": "gather",
                "target": {
                    "resource": material,
                    "quantity_needed": max(1, int(deficit)),
                },
                "reasoning": f"Gathering {deficit} more {material} for {milestone}",
                "deterministic": True,
            }

        if material in visited:
            return {
                "action": "gather", "target": {"resource": "iron_ore"},
                "reasoning": f"Breaking cyclic dependency while preparing {milestone}",
                "deterministic": True,
            }
        visited.add(material)
        component_recipe = self._recipes_cache.get(material, {})
        component_costs = component_recipe.get("materials", {})
        missing_input = next((
            (input_material, needed - colony_mats.get(input_material, 0))
            for input_material, needed in component_costs.items()
            if colony_mats.get(input_material, 0) < needed
        ), None)
        if missing_input:
            return self._material_dependency_action(
                missing_input[0], missing_input[1], milestone, colony_mats, visited
            )
        return {
            "action": "refine",
            "target": {
                "output": material,
                "structure": component_recipe.get("requires_structure"),
                "quantity_needed": max(1, int(deficit)),
            },
            "reasoning": f"Fabricating {material} for balanced milestone {milestone}",
            "deterministic": True,
        }

    def _exploration_infrastructure_candidate(
        self, colony_mats: dict, structures_built: dict
    ) -> Optional[dict]:
        """Build the next survey/comms tier when a mission resource is remote."""
        requests = sorted(getattr(self, "remote_resource_requests", set()))
        if not requests:
            return None

        if structures_built.get("research_workbench", 0) == 0:
            recipe_name = "research_workbench"
        elif (
            structures_built.get("communication_relay", 0) == 0
            and any(
                isinstance(getattr(other, "_pending_expedition", None), dict)
                and other._pending_expedition.get("resource") in requests
                for other in getattr(self, "agents", [])
            )
        ):
            # After the research bench surveys a real remote deposit, attempt
            # the built-in buddy expedition first. A relay is constructed only
            # when a concrete pending expedition still cannot launch; building
            # it pre-emptively delayed solar/O2/food infrastructure for days.
            recipe_name = "communication_relay"
        elif (
            structures_built.get("laboratory_module", 0) == 0
            and structures_built.get("habitat_module", 0) > 0
        ):
            recipe_name = "laboratory_module"
        else:
            return None

        recipe = self._recipes_cache.get(recipe_name, {})
        requirements = recipe.get("materials", {})
        missing = next((
            (material, needed - colony_mats.get(material, 0))
            for material, needed in requirements.items()
            if colony_mats.get(material, 0) < needed
        ), None)
        blocked_resource = requests[0]
        if missing:
            return self._material_dependency_action(
                missing[0], missing[1], recipe_name, colony_mats, set()
            )
        return {
            "action": "build",
            "target": {"recipe": recipe_name},
            "reasoning": (
                f"Building {recipe_name} to reach surveyed {blocked_resource} "
                "needed by the colony mission"
            ),
            "deterministic": True,
        }
    
    def _finalize_decision(self, agent: Agent, decision: Optional[dict], tick: int) -> dict:
        """Validate and record a decision before execution."""
        if not decision:
            decision = {"action": "idle", "target": {}, "reasoning": "Resting / idle", "deterministic": True}

        # Relative movement without a destination is not an EVA mission. Live
        # plans occasionally supplied only dx/dy, so the astronaut stepped out
        # once and navigation commitment sent them straight home next tick.
        if decision.get("action") == "move":
            target = decision.get("target")
            target = target if isinstance(target, dict) else {}
            has_absolute_target = (
                target.get("x") is not None and target.get("y") is not None
            )
            destination = target.get("destination")
            if not has_absolute_target and destination in ("habitat", "shelter"):
                decision = {
                    **decision,
                    "target": {
                        **target,
                        "x": getattr(
                            self, "lz_x", getattr(agent, "spawn_x", agent.x)
                        ),
                        "y": getattr(
                            self, "lz_y", getattr(agent, "spawn_y", agent.y)
                        ),
                    },
                }
            elif not has_absolute_target and getattr(agent, "_in_habitat", False):
                decision = {
                    "action": "rest",
                    "target": {
                        "route_rejected": "missing_absolute_mission_target"
                    },
                    "reasoning": (
                        f"{agent.name} rejected an unbounded MOVE order before EVA; "
                        "no physical destination was supplied"
                    ),
                    "deterministic": True,
                }
            
        # Extraction safety: heavy excavation is kept away from foundations,
        # pressure vessels, utilities and the crew's immediate living area.
        if decision.get("action") in ("gather", "mine"):
            original_target = (
                dict(decision.get("target", {}))
                if isinstance(decision.get("target"), dict) else {}
            )
            original_reasoning = str(decision.get("reasoning", "")).strip()
            lz_x = getattr(self, "lz_x", getattr(agent, "spawn_x", 1000))
            lz_y = getattr(self, "lz_y", getattr(agent, "spawn_y", 1000))
            res_type = decision.get("target", {}).get("resource", "regolith") if isinstance(decision.get("target"), dict) else "minerals"
            exclusion_radius = int(getattr(self, "min_extraction_radius_cells", 6))
            is_inside_extraction_buffer = (
                getattr(agent, "_in_habitat", False)
                or max(abs(agent.x - lz_x), abs(agent.y - lz_y)) < exclusion_radius
            )
            is_on_building = any(
                s.get("x") == agent.x and s.get("y") == agent.y
                for s in getattr(self, "placed_structures", [])
            )
            # Named subsurface resources must first reach the engine's
            # geological site-selection path. Surface regolith and generic
            # mine commands can be routed directly to the extraction apron.
            needs_geological_site_selection = (
                decision.get("action") == "gather" and res_type != "regolith"
            )
            if (is_inside_extraction_buffer or is_on_building) and not needs_geological_site_selection:
                directions = (
                    (1, 0), (1, 1), (0, 1), (-1, 1),
                    (-1, 0), (-1, -1), (0, -1), (1, -1),
                )
                start = sum(ord(ch) for ch in agent.id) % len(directions)
                occupied = {
                    (structure.get("x"), structure.get("y"))
                    for structure in getattr(self, "placed_structures", [])
                    if not structure.get("destroyed", False)
                }
                target_x, target_y = lz_x, lz_y
                origin = (agent.x, agent.y)
                reserved = {
                    (crew.action.target.get("x"), crew.action.target.get("y"))
                    for crew in getattr(self, "agents", [])
                    if crew.id != agent.id and isinstance(crew.action.target, dict)
                    and crew.action.target.get("destination") == "extraction_zone"
                }
                for extra_radius in range(0, 4):
                    radius = exclusion_radius + extra_radius
                    candidates = []
                    for offset in range(len(directions)):
                        direction = directions[(start + offset) % len(directions)]
                        candidate = (
                            lz_x + direction[0] * radius,
                            lz_y + direction[1] * radius,
                        )
                        if candidate not in occupied:
                            # Equal radial safety does not imply equal travel:
                            # diagonal corners can double suited walking time.
                            candidates.append((candidate in reserved,
                                abs(candidate[0] - origin[0]) + abs(candidate[1] - origin[1]),
                                offset, candidate))
                    if candidates:
                        target_x, target_y = min(candidates)[-1]
                    if (target_x, target_y) != (lz_x, lz_y):
                        break
                dx = max(-1, min(1, target_x - agent.x))
                dy = max(-1, min(1, target_y - agent.y))
                decision = {
                    "action": "move",
                    "target": {
                        **original_target,
                        "dx": dx,
                        "dy": dy,
                        "destination": "extraction_zone",
                        "x": target_x,
                        "y": target_y,
                    },
                    "reasoning": (
                        f"{original_reasoning}; " if original_reasoning else ""
                    ) + (
                        f"{agent.name} moving beyond the {exclusion_radius}-cell "
                        f"base extraction safety buffer to gather {res_type}"
                    ),
                    "deterministic": True
                }

        # Record memory
        if decision.get("action") != "continue":
            self.memory.record_action(
                agent.id,
                action=decision.get("action", "explore"),
                result=decision,
                tick=tick,
            )
        
        # Reflection check
        if (
            self.dialogue_generation_enabled
            and tick - self._last_reflection_tick.get(agent.id, 0)
            >= self.REFLECTION_INTERVAL
        ):
            self._run_reflection(agent, tick)
            self._last_reflection_tick[agent.id] = tick
            
        return decision

    def process_tick(self, agent: Agent, tick: int,
                     tick_events: dict,
                     world_context: dict = None,
                     nearby_agents: list = None) -> dict:
        """Main entry point: executes decision logic and finalizes RL policy registration."""
        raw_decision = self._process_tick_internal(agent, tick, tick_events, world_context, nearby_agents)
        return self._finalize_decision(agent, raw_decision, tick)

    def _consider_excavator_dispatch(
        self,
        agent: Agent,
        shared_decision: dict,
        world_context: dict | None,
        structures_built: dict,
    ) -> dict:
        """Let RL choose between human extraction and a valid robot job.

        The engine supplies only currently exposed, detector-verified and
        construction-safe targets.  Therefore this method cannot invent a
        deposit or bypass the physical action mask.
        """
        if (
            shared_decision.get("action") != "gather"
            or not getattr(agent, "_in_habitat", False)
        ):
            return shared_decision
        shared_target = shared_decision.get("target", {})
        requested_resource = (
            shared_target.get("resource") if isinstance(shared_target, dict) else None
        )
        robot_target = getattr(self, "robot_dispatch_targets", {}).get(
            requested_resource
        )
        if not isinstance(robot_target, dict):
            return shared_decision

        robot_decision = {
            "action": "dispatch_excavator",
            "target": {
                **robot_target,
                "requested_resource": requested_resource,
                "capacity_recipe": shared_target.get("capacity_recipe"),
                # If the physical state changes between policy selection and
                # execution, retain the already validated crew alternative.
                # The engine can release the astronaut directly back to this
                # work instead of displaying a synthetic idle cycle.
                "human_fallback": {
                    "action": "gather",
                    "target": dict(shared_target),
                    "reasoning": shared_decision.get("reasoning", ""),
                },
            },
            "reasoning": (
                f"{agent.name} dispatching a charged excavator to the verified "
                f"{robot_target.get('resource')} face while crew labor remains "
                "available for the colony work order"
            ),
            "validated_action": True,
        }
        state_key = self.get_rl_state_key(
            agent,
            structures_built,
            self._pooled_materials(),
            world_context,
        )
        # Put the useful automation option first for optimistic-value ties.
        # Subsequent outcomes are learned from actual delivered payload, not
        # merely from sending the command.
        return self.select_action_via_rl(
            agent,
            state_key,
            [robot_decision, dict(shared_decision)],
        )

    def _process_tick_internal(self, agent: Agent, tick: int,
                               tick_events: dict,
                               world_context: dict = None,
                               nearby_agents: list = None) -> dict:
        """
        Process one tick for an agent.
        
        This is the main entry point called by the engine each tick.
        
        Args:
            agent: The agent to process
            tick: Current simulation tick
            tick_events: Events from agent.tick_update()
            world_context: Environmental data
            nearby_agents: List of nearby agent info dicts
            
        Returns:
            Decision dict with action, target, reasoning, etc.
        """
        if agent.status == AgentStatus.DEAD or getattr(agent.status, 'value', str(agent.status)) == 'dead':
            return {"action": "dead", "reasoning": "Agent is dead"}

        if agent.status == AgentStatus.INCAPACITATED or getattr(agent.status, 'value', str(agent.status)) == 'incapacitated':
            return {
                "action": "unconscious",
                "target": {},
                "reasoning": f"{agent.name} is unconscious / incapacitated awaiting emergency SAR rescue",
                "deterministic": True
            }

        # A crew collapse is a habitat-wide medical alarm, not merely a local
        # observation. It is allowed to interrupt even the protected first
        # stages of sleep; SAR triage below still dispatches only one lead.
        team_emergency_victims = [
            other for other in getattr(self, "agents", [])
            if other.id != agent.id
            and getattr(other.status, 'value', str(other.status)) == 'incapacitated'
        ]
        team_medical_emergency = bool(team_emergency_victims)
        assigned_sar_victims = [
            victim for victim in team_emergency_victims
            if self.select_sar_rescuer_id(victim) == agent.id
        ]
        is_assigned_sar_responder = bool(assigned_sar_victims)
        if team_medical_emergency:
            victim_names = [victim.name for victim in team_emergency_victims]
            agent._team_emergency_alert = {
                "active": True,
                "tick": tick,
                "victim_ids": [victim.id for victim in team_emergency_victims],
                "victim_names": victim_names,
                "assigned_sar": is_assigned_sar_responder,
            }
            # Everyone receives the alert, but only the designated SAR lead is
            # awakened. Repeatedly waking non-responders made them select sleep
            # again in the same tick and permanently reset sleep architecture.
            if (is_assigned_sar_responder and getattr(agent.action, "action_type", "") == "sleep"
                and not self.sar_recovery_required(agent)):
                agent.action.action_type = "idle"
                agent.action.ticks_remaining = 0
                agent.action.target = {
                    "woken_by_team_emergency": True,
                    "victims": victim_names,
                }
                agent.needs._consecutive_sleep_ticks = 0
                agent._last_emergency_wake_tick = tick
        elif isinstance(getattr(agent, "_team_emergency_alert", None), dict):
            agent._team_emergency_alert["active"] = False

        # === 0. POLYSOMNOGRAPHIC SLEEP LOCKOUT (NASA-STD-3001) ===
        # Sleeping is a continuous biological process. Protect the first 8 ticks (40 min)
        # of restorative deep sleep unless life-threatening hazard occurs.
        active_events = world_context.get("active_events", []) if world_context else []
        is_hazard_event = any(
            ev.get("type") in SURFACE_HAZARD_EVENT_TYPES
            for ev in active_events
        )
        early_colony_resources = (
            world_context.get(
                "colony_resources", getattr(self, "_colony_resources", {})
            )
            if world_context else getattr(self, "_colony_resources", {})
        )
        early_lz_x = getattr(self, "lz_x", getattr(agent, "spawn_x", agent.x))
        early_lz_y = getattr(self, "lz_y", getattr(agent, "spawn_y", agent.y))
        early_near_shelter = (
            getattr(agent, "_in_habitat", False)
            or max(abs(agent.x - early_lz_x), abs(agent.y - early_lz_y)) <= 1
        )
        actionable_hydration = self._hydration_source_available(
            agent,
            near_shelter=early_near_shelter,
            colony_resources=early_colony_resources,
        )
        self_life_emergency = (
            agent.needs.o2_supply <= 25.0
            # Wake a sleeper for thirst only when waking permits an actual
            # drink.  With a dry base, repeatedly aborting sleep cannot treat
            # dehydration and instead creates a preventable fatigue collapse.
            or (agent.needs.thirst <= 25.0 and actionable_hydration)
            or agent.needs.temperature_stress <= 30.0
            or agent.needs.temperature_stress >= 92.0
        )
        has_life_emergency = (
            is_hazard_event
            or (is_assigned_sar_responder and not self.sar_recovery_required(agent))
            or self_life_emergency
        )
        if agent.is_in_sleep_lockout(has_emergency=has_life_emergency):
            return {
                "action": "sleep",
                "target": agent.action.target if isinstance(agent.action.target, dict) else {},
                "reasoning": f"{agent.name} resting in continuous physiological sleep cycle (Stage N1/N2/N3)",
                "deterministic": True
            }

        # === ONGOING SLEEP CYCLE PERSISTENCE ===
        if agent.action.action_type == "sleep" and agent.action.ticks_remaining > 0 and agent.needs.energy < 90.0:
            critical_warnings = [w for w in tick_events.get("warnings", []) if "critical" in str(w).lower() or "DEATH" in str(w)]
            if not critical_warnings and not has_life_emergency:
                return {
                    "action": "sleep",
                    "target": agent.action.target if isinstance(agent.action.target, dict) else {},
                    "reasoning": f"{agent.name} resting in continuous sleep cycle ({agent.needs.energy:.0f}% energy, {agent.action.ticks_remaining}t remaining)",
                    "deterministic": True
                }

        # Preserve every scheduled exterior-maintenance route across the
        # pressurized base cells. Dropping this marker on an intermediate move
        # converted the next airlock crossing back into an ordinary EVA and
        # hid the protected maintenance O2 reserve.
        active_target = (
            agent.action.target if isinstance(agent.action.target, dict) else {}
        )
        route_vitals_safe = (
            agent.needs.energy > 65.0
            and agent.needs.hunger > 45.0
            and agent.needs.thirst > 45.0
            and agent.needs.o2_supply > float(getattr(
                agent, "_routine_eva_o2_return_threshold", 40.0
            ))
            and 42.0 < agent.needs.temperature_stress < 75.0
        )
        if (
            agent.action.action_type == "move"
            and active_target.get("maintenance")
            and active_target.get("maintenance_action")
            and route_vitals_safe
        ):
            maintenance_action = active_target["maintenance_action"]
            executable_action = (
                "repair"
                if maintenance_action in {"repair", "inspection"}
                else maintenance_action
            )
            return {
                # ``inspection`` describes the service mode; the executable
                # engine action remains ``repair`` for both inspection and
                # corrective work. Other maintenance routes, notably solar
                # cleaning, already name their executable engine action and
                # must retain it while travelling.
                "action": executable_action,
                "target": dict(active_target),
                "reasoning": (
                    f"{agent.name} continuing scheduled exterior maintenance "
                    f"({maintenance_action})"
                ),
                "deterministic": True,
            }
        if (
            agent.action.action_type == "move"
            and active_target.get("survey_action") == "portable_scanner"
            and route_vitals_safe
        ):
            return {
                "action": "survey_resources",
                "target": self._scanner_target(
                    agent, active_target, preserve_primary=True
                ),
                "reasoning": (
                    f"{agent.name} continuing the scheduled portable-scanner "
                    "survey route"
                ),
                "deterministic": True,
            }
        if (
            agent.action.action_type == "prospect"
            and active_target.get("survey_action") == "portable_scanner"
            and active_target.get("rescan_after_test_pit")
            and route_vitals_safe
        ):
            return {
                "action": "survey_resources",
                "target": self._scanner_target(
                    agent, active_target, preserve_primary=True
                ),
                "reasoning": (
                    f"{agent.name} rescanning from the newly exposed test-pit face"
                ),
                "deterministic": True,
            }
        if (
            agent.action.action_type == "move"
            and active_target.get("survey_action") == "scanner_recharge"
            and route_vitals_safe
        ):
            return {
                "action": "recharge_scanner",
                "target": self._scanner_target(
                    agent, active_target, preserve_primary=True
                ),
                "reasoning": f"{agent.name} returning the field scanner to base power",
                "deterministic": True,
            }
        
        if world_context is None:
            world_context = {}
        if nearby_agents is None:
            nearby_agents = []
        
        colony_res = world_context.get("colony_resources", getattr(self, "_colony_resources", {}))
        
        decision = None
        
        # === 1. AUTOMATIC STORM / SOLAR FLARE / PLSS CRITICAL EMERGENCY EVACUATION ===
        active_events = world_context.get("active_events", [])
        is_hazard_event = any(
            ev.get("type") in SURFACE_HAZARD_EVENT_TYPES
            for ev in active_events
        )
        
        # PLSS Critical Life Support Trigger (CO2 Scrubber / Battery / O2).
        # A depot canister is not available remotely: without a carried spare,
        # reserve enough oxygen for the actual walk back plus contingency.
        lz_x_for_reserve = getattr(self, "lz_x", getattr(agent, "spawn_x", agent.x))
        lz_y_for_reserve = getattr(self, "lz_y", getattr(agent, "spawn_y", agent.y))
        return_distance = max(
            abs(agent.x - lz_x_for_reserve), abs(agent.y - lz_y_for_reserve)
        )
        has_carried_o2_spare = agent.inventory.has_item("oxygen_canisters")
        # Life-safety planning uses the conservative realized grid pace;
        # fair-weather motion may cover two cells, but rough routes must still
        # leave enough energy/O2 for one-cell ticks on the way home.
        routine_walk_speed = 1
        return_ticks = math.ceil(return_distance / routine_walk_speed)
        distance_adjusted_o2_return = float(getattr(
            agent,
            "_routine_eva_o2_return_threshold",
            min(
                80.0,
                40.0
                + return_ticks * agent.plss_o2_percent_per_tick(1.5),
            ),
        ))
        active_o2_budgeted_expedition = (
            isinstance(getattr(agent, "_active_expedition", None), dict)
            and agent._active_expedition.get("status") in {"outbound", "working", "returning"}
        )
        active_o2_service_route = (
            getattr(agent.action, "action_type", "") == "move"
            and isinstance(getattr(agent.action, "target", None), dict)
            and agent.action.target.get("o2_service", False)
            and getattr(agent.needs, "o2_supply", 0.0) > 10.0
        )
        plss_critical = (
            getattr(agent, "plss_co2_scrubber_pct", 100.0) < 18.0 or
            getattr(agent, "plss_suit_battery_pct", 100.0) < 18.0 or
            getattr(agent, "suit_integrity", 1.0) < 0.45 or
            getattr(agent, "suit_condition", 1.0) < 0.50 or
            (
                getattr(agent.needs, "o2_supply", 100.0) < 22.0
                and not active_o2_service_route
            ) or
            (
                not getattr(agent, "_in_habitat", False)
                and not active_o2_budgeted_expedition
                and not active_o2_service_route
                and not has_carried_o2_spare
                and agent.needs.o2_supply <= distance_adjusted_o2_return
            )
        )
        
        # Suit Micro-Puncture Immediate Patching
        if getattr(agent, "has_micro_puncture", False) and not agent._in_habitat:
            return {
                "action": "seal_patch",
                "target": {},
                "reasoning": f"CRITICAL SUIT BREACH: {agent.name} deploying emergency sealant patch to arrest rapid decompression leak!",
                "deterministic": True
            }

        if (
            not getattr(agent, "_in_habitat", False)
            and has_carried_o2_spare
            and agent.needs.o2_supply <= (
                40.0 if active_o2_budgeted_expedition else distance_adjusted_o2_return
            )
        ):
            return {
                "action": "refill_o2",
                "target": {"carried_spare": True},
                "reasoning": (
                    f"{agent.name} swapping a carried O₂ canister at "
                    f"{agent.needs.o2_supply:.0f}% to preserve a {return_distance}-cell return reserve"
                ),
                "deterministic": True,
            }

        if (is_hazard_event or plss_critical) and not agent._in_habitat:
            # Find nearest operational habitat_module or Landing Base
            lz_x = getattr(self, "lz_x", getattr(agent, "spawn_x", agent.x))
            lz_y = getattr(self, "lz_y", getattr(agent, "spawn_y", agent.y))
            nearest_x, nearest_y = lz_x, lz_y
            min_dist = max(abs(agent.x - lz_x), abs(agent.y - lz_y))
            
            for s in getattr(self, "placed_structures", []):
                if (
                    s.get("type") == "habitat_module"
                    and not s.get("under_construction", False)
                    and not s.get("destroyed", False)
                    and float(s.get("health", 1.0)) > 0.0
                ):
                    sx, sy = s.get("x", lz_x), s.get("y", lz_y)
                    dist = max(abs(agent.x - sx), abs(agent.y - sy))
                    if dist < min_dist:
                        min_dist = dist
                        nearest_x, nearest_y = sx, sy
                        
            dx = 1 if nearest_x > agent.x else (-1 if nearest_x < agent.x else 0)
            dy = 1 if nearest_y > agent.y else (-1 if nearest_y < agent.y else 0)
            if is_hazard_event:
                cause_desc = "environmental hazard"
            elif getattr(agent, "suit_condition", 1.0) < 0.50:
                cause_desc = (
                    f"PLSS EVA suit mobility/service limit "
                    f"({getattr(agent, 'suit_condition', 1.0):.0%} condition)"
                )
            elif getattr(agent, "suit_integrity", 1.0) < 0.45:
                cause_desc = (
                    f"PLSS pressure-suit integrity reserve "
                    f"({getattr(agent, 'suit_integrity', 1.0):.0%})"
                )
            elif getattr(agent, "plss_co2_scrubber_pct", 100.0) < 18.0:
                cause_desc = "PLSS CO₂ scrubber reserve"
            elif getattr(agent, "plss_suit_battery_pct", 100.0) < 18.0:
                cause_desc = "PLSS battery reserve"
            else:
                cause_desc = (
                    f"PLSS return reserve ({agent.needs.o2_supply:.0f}% O₂ "
                    f"for {min_dist} cells)"
                )
            return {
                "action": "move",
                "target": {"dx": dx, "dy": dy, "x": nearest_x, "y": nearest_y},
                "reasoning": f"EMERGENCY RETREAT: {agent.name} evacuating to Habitat Airlock ({nearest_x}, {nearest_y}) due to {cause_desc}!",
                "deterministic": True
            }

        # === 1.5 INTELLIGENT SEARCH AND RESCUE (SAR) TRIAGE DISPATCH ===
        # When an astronaut collapses on EVA, select the SINGLE optimal rescuer based on distance, fitness, and medical skill.
        # Colonists engaged in critical life-support repairs or with compromised vitals stay on their vital posts.
        if agent.status != AgentStatus.INCAPACITATED and getattr(agent.status, 'value', str(agent.status)) != 'incapacitated':
            # Check if this agent itself is physically fit for EVA rescue
            is_fit_rescuer = (
                agent.needs.o2_supply >= 50.0 and
                agent.needs.temperature_stress >= 35.0 and
                agent.needs.energy >= 30.0 and
                agent.injury_level < 0.40 and
                not getattr(agent, "has_micro_puncture", False)
            )
            
            # Check if agent is currently engaged in a vital colony life-support mission that must NOT be abandoned
            is_doing_vital_job = False
            cur_act = getattr(agent.action, "action_type", "")
            if cur_act == "repair":
                struct_target = agent.action.target.get("structure", "") if isinstance(agent.action.target, dict) else ""
                if struct_target in ("isru_o2_unit", "greenhouse", "water_collector", "water_purifier", "solar_panel", "habitat_module", "eclss_lander_hub"):
                    is_doing_vital_job = True
            elif cur_act == "treat":
                is_doing_vital_job = True

            if is_fit_rescuer and not is_doing_vital_job:
                # Find every unconscious victim. A collapse inside the base is
                # still a medical emergency and must receive a designated lead.
                unconscious_victims = [
                    other for other in getattr(self, "agents", [])
                    if other.id != agent.id
                    and getattr(other.status, 'value', str(other.status)) == 'incapacitated'
                    and not getattr(other, "_in_habitat", False)
                ]
                
                for victim in unconscious_victims:
                    best_rescuer_id = self.select_sar_rescuer_id(victim)
                    if best_rescuer_id:
                        
                        # Only the designated primary SAR Lead rushes to rescue; others continue colony operations
                        if agent.id == best_rescuer_id:
                            # Rescue work is heavy and may involve carrying a
                            # second suited adult. If the assigned lead has a
                            # physical field provision, consume it before
                            # joining the casualty instead of creating a
                            # second patient halfway through the extraction.
                            if (
                                agent.needs.thirst <= 45.0
                                and agent.inventory.has_item("water_packs")
                            ):
                                return {
                                    "action": "drink",
                                    "target": {
                                        "field": not agent._in_habitat,
                                        "sar_self_care": True,
                                        "victim_id": victim.id,
                                    },
                                    "reasoning": (
                                        f"SAR PREFLIGHT: {agent.name} hydrating "
                                        f"from a carried pack before extracting "
                                        f"{victim.name}"
                                    ),
                                    "deterministic": True,
                                }
                            if (
                                agent.needs.hunger <= 35.0
                                and (
                                    agent.inventory.has_item(
                                        "emergency_rations"
                                    )
                                    or agent.inventory.has_item("ration_pack")
                                )
                            ):
                                return {
                                    "action": "eat",
                                    "target": {
                                        "field": not agent._in_habitat,
                                        "sar_self_care": True,
                                        "victim_id": victim.id,
                                    },
                                    "reasoning": (
                                        f"SAR PREFLIGHT: {agent.name} consuming "
                                        f"a carried ration before extracting "
                                        f"{victim.name}"
                                    ),
                                    "deterministic": True,
                                }
                            return {
                                "action": "rescue",
                                "target": {"victim_id": victim.id, "victim_name": victim.name, "x": victim.x, "y": victim.y},
                                "reasoning": f"PRIMARY SAR LEAD: {agent.name} (optimal proximity & triage fitness) dispatched to extract unconscious astronaut {victim.name} at ({victim.x}, {victim.y})",
                                "deterministic": True
                            }

        # === 2. CREW MEDICAL & PATHOLOGY PROTOCOLS (Zero Magic Kits) ===
        # A six-person expedition cannot make its CMO a single point of
        # failure. Flight crew with medical >= 4 are cross-trained first
        # responders; invasive injury care remains CMO-only (>= 7).
        medical_skill = float(getattr(agent.competency, "medical", 0))
        is_medic = (
            medical_skill >= 7
            or "medic" in getattr(agent, "role", "").lower()
        )
        is_first_responder = medical_skill >= 4
        if is_medic or is_first_responder:
            for patient in nearby_agents:
                p_agent = patient.get("agent_obj") if isinstance(patient, dict) else None
                if p_agent and p_agent.id != agent.id and getattr(p_agent.status, 'value', str(p_agent.status)) != 'dead':
                    if (
                        p_agent.injury_level > 0.25
                        or p_agent.needs.temperature_stress < 25
                        or p_agent.needs.o2_supply < 30
                        or p_agent.needs.thirst < 25
                        or p_agent.needs.hunger < 25
                    ):
                        dist = max(abs(p_agent.x - agent.x), abs(p_agent.y - agent.y))
                        if dist <= 2:
                            # When several vital systems are critical, service
                            # the failure clock that has advanced furthest.
                            # A fixed hydration-first order let repeated IV
                            # doses suppress enteral nutrition until an
                            # otherwise fully supplied patient starved.
                            hunger_critical = p_agent.needs.hunger < 25
                            thirst_critical = p_agent.needs.thirst < 25
                            patient_target = (
                                p_agent.action.target
                                if isinstance(p_agent.action.target, dict)
                                else {}
                            )
                            patient_status = getattr(
                                p_agent.status, "value", str(p_agent.status)
                            )
                            can_take_oral_fluids = bool(
                                patient_status not in {"dead", "incapacitated"}
                                and patient_target.get("conscious") is not False
                                and patient_target.get(
                                    "oral_fluids_allowed"
                                ) is not False
                                and p_agent.needs.o2_supply > 0.0
                                and p_agent.needs.temperature_stress > 0.0
                                and p_agent.injury_level < 0.80
                            )
                            medical_item_available = getattr(
                                self, "medical_item_available", None
                            )
                            fluid_available = bool(
                                agent.inventory.has_item(
                                    "sterile_iv_fluid_bags"
                                )
                                or p_agent.inventory.has_item(
                                    "sterile_iv_fluid_bags"
                                )
                            )
                            access_set_available = bool(
                                agent.inventory.has_item(
                                    "iv_io_administration_sets"
                                )
                                or p_agent.inventory.has_item(
                                    "iv_io_administration_sets"
                                )
                            )
                            if medical_item_available:
                                fluid_available = medical_item_available(
                                    agent, p_agent,
                                    "sterile_iv_fluid_bags",
                                )
                                access_set_available = medical_item_available(
                                    agent, p_agent,
                                    "iv_io_administration_sets",
                                )
                            both_in_habitat = bool(
                                getattr(agent, "_in_habitat", False)
                                and getattr(p_agent, "_in_habitat", False)
                            )
                            iv_io_treatment_available = bool(
                                thirst_critical
                                and not can_take_oral_fluids
                                and both_in_habitat
                                and fluid_available
                                and access_set_available
                            )
                            depot_inventory = getattr(
                                self, "central_depot_inventory", {}
                            )
                            nutrition_source_available = bool(
                                p_agent.inventory.has_item(
                                    "emergency_rations"
                                )
                                or p_agent.inventory.has_item("ration_pack")
                                or depot_inventory.get("ration_packs", 0) > 0
                                or float(colony_res.get(
                                    "food_reserve_kcal", 0.0
                                )) >= 700.0
                            )
                            nutrition_treatment_available = bool(
                                hunger_critical
                                # A conscious casualty can consume the same
                                # physical food through ordinary self-care.
                                # Reserve the finite enteral consumable for a
                                # patient who cannot safely feed themselves.
                                and not can_take_oral_fluids
                                and both_in_habitat
                                and nutrition_source_available
                            )
                            hunger_clock_fraction = float(getattr(
                                p_agent.needs, "_hunger_death_timer", 0
                            )) / max(1, int(getattr(
                                p_agent.needs, "STARVATION_TICKS", 1
                            )))
                            thirst_clock_fraction = float(getattr(
                                p_agent.needs, "_thirst_death_timer", 0
                            )) / max(1, int(getattr(
                                p_agent.needs, "DEHYDRATION_TICKS", 1
                            )))
                            nutrition_priority = bool(
                                nutrition_treatment_available
                                and (
                                    not thirst_critical
                                    or can_take_oral_fluids
                                    or not iv_io_treatment_available
                                    or hunger_clock_fraction
                                    > thirst_clock_fraction
                                )
                            )
                            # Apply targeted space medical protocol
                            if p_agent.needs.o2_supply < 30:
                                protocol_msg = "High-Flow O2 Hyperbaric Purge"
                                protocol = "emergency_oxygen"
                                treatment_available = bool(
                                    (
                                        getattr(agent, "_in_habitat", False)
                                        and getattr(
                                            p_agent, "_in_habitat", False
                                        )
                                        and float(colony_res.get(
                                            "o2_reserve_kg", 0.0
                                        )) >= 0.10
                                    )
                                    or agent.inventory.has_item(
                                        "oxygen_canisters"
                                    )
                                    or p_agent.inventory.has_item(
                                        "oxygen_canisters"
                                    )
                                )
                            elif p_agent.needs.temperature_stress < 25:
                                protocol_msg = "Insulated Fabric Thermal Blanket Re-warming"
                                protocol = "rewarming"
                                treatment_available = bool(
                                    agent.inventory.has_item(
                                        "emergency_blanket"
                                    )
                                    or p_agent.inventory.has_item(
                                        "emergency_blanket"
                                    )
                                )
                            elif (
                                p_agent.needs.thirst < 25
                                and not nutrition_priority
                            ):
                                if can_take_oral_fluids:
                                    # The patient's own monitored-rest oral
                                    # interlock consumes potable water. Do not
                                    # spend an invasive kit on someone who can
                                    # safely drink for themselves.
                                    continue
                                protocol_msg = (
                                    "Sterile IV/IO Isotonic Crystalloid Infusion"
                                )
                                protocol = "iv_io_rehydration"
                                treatment_available = (
                                    iv_io_treatment_available
                                )
                            elif p_agent.needs.hunger < 25:
                                protocol_msg = "Measured Enteral Nutrition"
                                protocol = "nutrition_support"
                                treatment_available = (
                                    nutrition_treatment_available
                                )
                            elif is_medic:
                                protocol_msg = "Tissue Debridement & Carbon-Fiber Splinting"
                                protocol = "injury_care"
                                treatment_available = agent.inventory.has_item(
                                    "medical_supplies"
                                )
                                if medical_item_available:
                                    treatment_available = bool(
                                        medical_item_available(
                                            agent, p_agent, "medical_supplies"
                                        )
                                    )
                            else:
                                # Cross-trained responders stabilize vitals;
                                # definitive wound care requires the CMO.
                                continue
                            if not treatment_available:
                                # Do not turn a medically correct intent into
                                # an endless impossible-action loop.  The
                                # patient's own O2/thermal return override (or
                                # the crew-wide incapacitation alarm) remains
                                # authoritative until a physical treatment
                                # source is actually co-located.
                                continue
                            return {
                                "action": "treat",
                                "target": {
                                    "patient": p_agent.name,
                                    "patient_id": p_agent.id,
                                    "protocol": protocol,
                                },
                                "reasoning": f"{agent.name} attempting {protocol_msg} on {p_agent.name}",
                                "deterministic": True
                            }
                        elif (
                            route_vitals_safe
                            and (
                                is_medic
                                or p_agent.needs.o2_supply < 30
                                or p_agent.needs.temperature_stress < 25
                            )
                        ):
                            # A remote response must not cancel the responder's
                            # own fatigue/O2 return on alternating ticks.  An
                            # injury-only case also needs the CMO: a first
                            # responder who cannot execute definitive wound
                            # care would merely walk to the patient and leave.
                            dx = 1 if p_agent.x > agent.x else (-1 if p_agent.x < agent.x else 0)
                            dy = 1 if p_agent.y > agent.y else (-1 if p_agent.y < agent.y else 0)
                            return {
                                "action": "move",
                                "target": {"dx": dx, "dy": dy, "x": p_agent.x, "y": p_agent.y,
                                           "medical_response": True},
                                "reasoning": f"{agent.name} rushing to treat injured colonist {p_agent.name}",
                                "deterministic": True
                            }

        # Priority B: Self-treatment using insulated_fabric thermal blankets/splints
        if agent.injury_level > 0.3 and agent.inventory.materials.get("insulated_fabric", 0) > 0:
            agent.inventory.remove_material("insulated_fabric", 1)
            agent.injury_level = max(0.0, agent.injury_level - 0.35)
            return {
                "action": "treat",
                "target": {},
                "reasoning": f"{agent.name} applied insulated_fabric thermal bandage/splint to self",
                "deterministic": True
            }

        # === 3. BIOLOGICAL SELF-PRESERVATION OVERRIDES ===
        # Determine nearest operational shelter / habitat module or Lander Base
        lz_x = getattr(self, "lz_x", getattr(agent, "spawn_x", 1000))
        lz_y = getattr(self, "lz_y", getattr(agent, "spawn_y", 1000))
        shelter_x, shelter_y = lz_x, lz_y
        min_shelter_dist = max(abs(agent.x - lz_x), abs(agent.y - lz_y))

        for s in getattr(self, "placed_structures", []):
            if s.get("type") == "habitat_module" and not s.get("under_construction", False) and not s.get("destroyed", False):
                sx, sy = s.get("x", lz_x), s.get("y", lz_y)
                dist = max(abs(agent.x - sx), abs(agent.y - sy))
                if dist < min_shelter_dist:
                    min_shelter_dist = dist
                    shelter_x, shelter_y = sx, sy

        # Recovery resources become available only after a completed
        # pressure transition.  Treating an adjacent 100 m surface parcel as
        # "inside" let the policy toggle the habitat flag through a wall, then
        # schedule sleep/eating outside until physics corrected it next tick.
        is_near_shelter = bool(getattr(agent, "_in_habitat", False))

        # A walking precursor crew has a finite EVA operating area. Survival
        # needs may still be handled first on a given tick, but normal work and
        # route commitment must never keep pushing an agent farther from LZ.
        local_eva_radius = int(getattr(self, "max_eva_radius_cells", 24))
        active_expedition = getattr(agent, "_active_expedition", None)
        is_authorized_expedition = (
            isinstance(active_expedition, dict)
            and active_expedition.get("status") in {"outbound", "working", "returning"}
        )
        max_eva_radius = (
            max(local_eva_radius, int(active_expedition.get("authorized_radius", local_eva_radius)))
            if is_authorized_expedition else local_eva_radius
        )
        dist_from_lz = max(abs(agent.x - lz_x), abs(agent.y - lz_y))
        if dist_from_lz > max_eva_radius and not getattr(agent, "_in_habitat", False):
            dx = 1 if lz_x > agent.x else (-1 if lz_x < agent.x else 0)
            dy = 1 if lz_y > agent.y else (-1 if lz_y < agent.y else 0)
            return {
                "action": "move",
                "target": {
                    "dx": dx, "dy": dy, "x": lz_x, "y": lz_y,
                    "destination": "shelter", "forced_return": True,
                    "expedition": is_authorized_expedition,
                },
                "reasoning": (
                    f"{agent.name} exceeded the {max_eva_radius}-cell EVA "
                    f"operating radius — aborting field task and returning to base"
                ),
                "deterministic": True,
            }

        # Remote expeditions reserve enough time for the trip home. They abort
        # proactively instead of waiting for the generic 18% PLSS emergency.
        if is_authorized_expedition:
            move_speed = max(1, int(active_expedition.get(
                "move_speed_cells",
                getattr(self, "expedition_move_speed_cells", 2),
            )))
            # Route authorization may use the nominal two-cell fair-terrain
            # pace, but the realized walker frequently advances only one cell
            # per tick on rough terrain.  Return safety must use that slower
            # pace; otherwise a crew member can begin the trip home with an
            # apparently positive reserve and empty the active bottle outside
            # the airlock.
            return_speed = (
                move_speed
                if active_expedition.get("transport") == "crew_rover"
                else routine_walk_speed
            )
            sheltered_field_recovery = bool(
                getattr(agent, "_in_habitat", False)
                and (
                    agent.needs.energy <= 65.0
                    or agent.needs.hunger <= 65.0
                    or agent.needs.thirst <= 65.0
                    or agent.needs.temperature_stress <= 42.0
                )
            )
            # A working survey/construction watch cannot be maintained from a
            # pressurized building.  Recall the pair, then let the sheltered
            # member eat, drink or sleep before attempting the trip home.
            detached_from_field = bool(
                getattr(agent, "_in_habitat", False)
                and active_expedition.get("status") == "working"
            )
            if detached_from_field:
                self._recall_expedition_team(agent)
            if (
                active_expedition.get("status") == "returning"
                and dist_from_lz > 1
                and not sheltered_field_recovery
            ):
                self._recall_expedition_team(agent)
                return {
                    "action": "move",
                    "target": {
                        "x": lz_x,
                        "y": lz_y,
                        "destination": "shelter",
                        "expedition": True,
                        "forced_return": True,
                    },
                    "reasoning": (
                        f"{agent.name} continuing the committed expedition "
                        "return to the landing hub"
                    ),
                    "deterministic": True,
                }
            return_ticks = (dist_from_lz + return_speed - 1) // return_speed
            available_ticks = int(getattr(self, "expedition_available_ticks", 0))
            # The modelled walk can consume slightly more PLSS capacity than
            # the nominal planning activity factor under high gravity. Keep an
            # 90-minute operational contingency so the recorded margin never
            # crosses zero between successive decisions.
            return_margin_low = available_ticks <= return_ticks + ticks_for_minutes(
                90.0, self.tick_minutes
            )
            partner_id = (
                active_expedition.get("lead_id")
                if active_expedition.get("role") == "buddy"
                else active_expedition.get("buddy_id")
            )
            partner = next(
                (other for other in getattr(self, "agents", []) if other.id == partner_id),
                None,
            )
            partner_state = getattr(partner, "_active_expedition", None) if partner else None
            partner_returning = (
                isinstance(partner_state, dict)
                and partner_state.get("status") == "returning"
            )
            # Only the audited PLSS return margin (or the buddy already
            # returning) forcibly recalls an expedition. Hunger, thirst and
            # fatigue remain policy states and medical consequences, not
            # hidden mission locks.
            if (
                not sheltered_field_recovery
                and (return_margin_low or partner_returning)
            ):
                self._recall_expedition_team(agent)
                dx = move_speed if lz_x > agent.x else (-move_speed if lz_x < agent.x else 0)
                dy = move_speed if lz_y > agent.y else (-move_speed if lz_y < agent.y else 0)
                return {
                    "action": "move",
                    "target": {
                        "dx": dx, "dy": dy, "x": lz_x, "y": lz_y,
                        "destination": "shelter", "expedition": True,
                        "forced_return": True,
                    },
                    "reasoning": (
                        f"{agent.name} ending planned {active_expedition.get('resource', 'survey')} "
                        f"expedition with {available_ticks - return_ticks} reserve ticks for return"
                    ),
                    "deterministic": True,
                }

        if (
            is_authorized_expedition
            and active_expedition.get("role") == "buddy"
            and active_expedition.get("status") == "working"
            and active_expedition.get("kind") != "construction_support"
            and not getattr(agent, "_in_habitat", False)
        ):
            return {
                "action": "stand_watch",
                "target": {
                    "expedition_id": active_expedition.get("id"),
                    "resource": active_expedition.get("resource"),
                },
                "reasoning": f"{agent.name} maintaining buddy watch during field survey",
                "deterministic": True,
            }

        # --- 0. DYNAMIC COGNITIVE HAUL UTILITY & FATIGUE EVALUATION ---
        from src.agents.agent import MATERIAL_DENSITY_KG
        carry_weight = sum(
            qty * MATERIAL_DENSITY_KG.get(mat, 2.0)
            for mat, qty in agent.inventory.materials.items()
        )
        carried_units = sum(agent.inventory.materials.values())
        material_haul_target = {
            "x": lz_x,
            "y": lz_y,
            "destination": "material_storage",
            "mission_action": "deposit_materials",
        }
        committed_material_haul = bool(
            agent.action.action_type == "move"
            and active_target.get("destination") == "material_storage"
            and active_target.get("mission_action") == "deposit_materials"
        )
        if (
            route_vitals_safe
            and committed_material_haul
            and carried_units > 0
        ):
            # Once an offload route has selected a real depot/crate access
            # point, keep that route across subsequent policy ticks.  Without
            # this marker the nearest-habitat rule pulled the hauler back after
            # every step and created an endless two-way shuttle.
            return {
                "action": "deposit_materials",
                "target": dict(active_target),
                "reasoning": (
                    f"{agent.name} continuing the committed physical material "
                    "offload route"
                ),
                "deterministic": True,
            }
        if (
            route_vitals_safe
            and is_near_shelter
            and carried_units > 0
            and carry_weight >= 8.0
        ):
            return {
                "action": "deposit_materials",
                "target": dict(material_haul_target),
                "reasoning": (
                    f"{agent.name} offloading {carried_units} gathered units "
                    f"({carry_weight:.1f}kg) before the next EVA"
                ),
                "deterministic": True,
            }
        if carry_weight > 1.0 and not is_near_shelter:
            strength = getattr(agent.genome, "strength", 5)
            energy = getattr(agent.needs, "energy", 70.0)
            temp_stress = getattr(agent.needs, "temperature_stress", 50.0)
            dist_to_base = min_shelter_dist

            # Dynamic comfort threshold based on strength, stamina, and return trip distance
            base_comfort_kg = 6.0 + (strength * 1.8) + ((energy / 100.0) * 6.0)
            dist_penalty = min(12.0, dist_to_base * 0.75)
            comfort_threshold = max(3.5, base_comfort_kg - dist_penalty)

            # If weather is dangerously cold or agent is tired, drop threshold to return sooner
            if temp_stress < 44 or energy < 45:
                comfort_threshold *= 0.65

            if route_vitals_safe and carry_weight >= comfort_threshold:
                dx = 1 if shelter_x > agent.x else (-1 if shelter_x < agent.x else 0)
                dy = 1 if shelter_y > agent.y else (-1 if shelter_y < agent.y else 0)
                return {
                    "action": "move",
                    "target": {
                        "dx": dx,
                        "dy": dy,
                        "x": shelter_x,
                        "y": shelter_y,
                        "material_haul_staging": True,
                    },
                    "reasoning": (
                        f"{agent.name} feeling physical strain "
                        f"({carry_weight:.1f}kg load over {dist_to_base} tiles) "
                        "— returning to the nearest pressure refuge before offload"
                    ),
                    "deterministic": True,
                }

        # --- 0.5 THERMAL / FREEZING OVERRIDE (Proactive Core Protection) ---
        # Cold ambient conditions need a wider return margin: waiting for
        # severe cold stress leaves no time for the walk back to a heated hub.
        effective_temp_c = float(world_context.get(
            "effective_temperature_c", world_context.get("temperature_c", 20.0)
        ))
        thermal_return_threshold = 46.0 if effective_temp_c < 5.0 else 42.0
        if (
            agent.needs.temperature_stress <= thermal_return_threshold
            and not getattr(agent, "_in_habitat", False)
        ):
            if is_near_shelter:
                agent.enter_habitat()
                return {
                    "action": "rest",
                    "target": {},
                    "reasoning": f"{agent.name} entered habitat airlock for proactive re-warming ({agent.needs.temperature_stress:.0f}% thermal reserve)",
                    "deterministic": True
                }
            else:
                dx = 1 if shelter_x > agent.x else (-1 if shelter_x < agent.x else 0)
                dy = 1 if shelter_y > agent.y else (-1 if shelter_y < agent.y else 0)
                return {
                    "action": "move",
                    "target": {"dx": dx, "dy": dy, "x": shelter_x, "y": shelter_y},
                    "reasoning": (
                        f"{agent.name} detected declining thermal reserve "
                        f"({agent.needs.temperature_stress:.0f}%, ambient {effective_temp_c:.1f}°C) "
                        "— returning to the heated habitat before hypothermia"
                    ),
                    "deterministic": True
                }

        # --- A. OXYGEN LIFE SUPPORT EMERGENCY (Canister Swap / ISRU Fill) ---
        # Inside the habitat, physiological O2 saturation returns to 100% but
        # the separate PLSS cylinder remains depleted. Looking only at
        # needs.o2_supply caused the airlock loop: choose outdoor work, discover
        # the empty suit bottle at the door, idle, then choose outdoor work
        # again. Service the actual tank state before planning another EVA.
        plss_o2_remaining = float(getattr(
            agent, "_current_canister_remaining", agent.needs.o2_supply
        ))
        plss_service_due = (
            getattr(agent, "_in_habitat", False)
            and getattr(agent, "_needs_o2_support", False)
            and plss_o2_remaining <= 55.0
        )
        if agent.needs.o2_supply <= 40 or plss_service_due:
            depot = getattr(self, "central_depot_inventory", {})
            depot_accessible = is_near_shelter or getattr(agent, "_in_habitat", False)
            protected_canisters = int(getattr(
                self, "min_central_maintenance_o2_canisters", 2
            ))
            isru_o2_stations = [
                structure for structure in getattr(self, "placed_structures", [])
                if structure.get("type") == "isru_o2_unit"
                and not structure.get("under_construction", False)
                and not structure.get("destroyed", False)
                and float(structure.get("health", 1.0)) > 0.0
            ]
            bootstrap_o2_stations = [
                structure for structure in getattr(self, "placed_structures", [])
                if structure.get("type") in {"eclss_lander_hub", "advanced_eclss"}
                and not structure.get("under_construction", False)
                and not structure.get("destroyed", False)
                and float(structure.get("health", 1.0)) > 0.0
            ]
            o2_stations = isru_o2_stations or bootstrap_o2_stations
            o2_station = min(
                o2_stations,
                key=lambda structure: max(
                    abs(agent.x - int(structure.get("x", agent.x))),
                    abs(agent.y - int(structure.get("y", agent.y))),
                ),
                default=None,
            )
            isru_fill_available = (
                o2_station is not None
                and colony_res.get("energy_stored_kwh", 0.0) > 0.0
                and colony_res.get("o2_reserve_kg", 0.0)
                >= (protected_canisters + 1) * Agent.PLSS_CANISTER_O2_KG
            )
            if agent.inventory.has_item("oxygen_canisters") or (
                depot_accessible
                and depot.get("oxygen_canisters", 0) > protected_canisters
            ) or isru_fill_available:
                refill_target = {}
                if isru_fill_available and not (
                    agent.inventory.has_item("oxygen_canisters")
                    or (
                        depot_accessible
                        and depot.get("oxygen_canisters", 0) > protected_canisters
                    )
                ):
                    refill_target = {
                        "x": int(o2_station.get("x")),
                        "y": int(o2_station.get("y")),
                        "destination": "o2_filling_station",
                    }
                return {
                    "action": "refill_o2",
                    "target": refill_target,
                    "reasoning": (
                        f"{agent.name} servicing depleted PLSS tank "
                        f"({plss_o2_remaining:.0f}% tank, "
                        f"{agent.needs.o2_supply:.0f}% physiological O2)"
                    ),
                    "deterministic": True
                }
            if not getattr(agent, "_in_habitat", False):
                dx = 1 if shelter_x > agent.x else (-1 if shelter_x < agent.x else 0)
                dy = 1 if shelter_y > agent.y else (-1 if shelter_y < agent.y else 0)
                return {
                    "action": "move",
                    "target": {
                        "dx": dx, "dy": dy, "x": shelter_x, "y": shelter_y,
                        "destination": "shelter", "o2_return": True,
                    },
                    "reasoning": f"{agent.name} has no carried O₂ spare — returning to the depot",
                    "deterministic": True,
                }

        # --- B. THIRST OVERRIDE (In-Suit IDB or Base Hydration) ---
        # Proactive drinking at base (<= 65%) or field hydration (<= 45%)
        thirst_thresh = 65 if is_near_shelter else 45
        hydration_available = self._hydration_source_available(
            agent,
            near_shelter=is_near_shelter,
            colony_resources=colony_res,
        )
        last_drink_tick = int(getattr(agent, "_last_drink_tick", -10_000))
        hydration_cooldown_complete = tick - last_drink_tick >= 4
        acute_dehydration = agent.needs.thirst <= 20.0
        if (
            agent.needs.thirst <= thirst_thresh
            and hydration_available
            and (hydration_cooldown_complete or acute_dehydration)
        ):
            if is_near_shelter:
                if not getattr(agent, "_in_habitat", False):
                    agent.enter_habitat()
                return {
                    "action": "drink",
                    "target": {"habitat": True},
                    "reasoning": f"{agent.name} hydrating safely inside base habitat ({agent.needs.thirst:.0f}%)",
                    "deterministic": True
                }
            else:
                # In-Suit Drink Bag (IDB): Astronaut sips from in-helmet valve on EVA
                return {
                    "action": "drink",
                    "target": {"field": True},
                    "reasoning": f"{agent.name} drinking from EVA suit In-Suit Drink Bag (IDB) ({agent.needs.thirst:.0f}%)",
                    "deterministic": True
                }

        # Once physiological telemetry reaches the model's critical
        # dehydration threshold, continuing an ordinary field assignment is
        # no longer a rational policy choice. Abort to the real water source;
        # this does not create water or guarantee survival at a dry base.
        if (
            agent.needs.thirst <= float(getattr(
                agent, "_routine_eva_water_return_threshold", 35.0
            ))
            and not hydration_available
            and not getattr(agent, "_in_habitat", False)
        ):
            if is_authorized_expedition:
                self._recall_expedition_team(agent)
            return {
                "action": "move",
                "target": {
                    "x": shelter_x,
                    "y": shelter_y,
                    "destination": "shelter",
                    "dehydration_return": True,
                    "expedition": is_authorized_expedition,
                },
                "reasoning": (
                    f"{agent.name} making an emergency return to the base "
                    f"water system at {agent.needs.thirst:.0f}% hydration reserve"
                ),
                "deterministic": True,
            }

        # A fatigued sleeper must eat before starting another recovery block.
        # The general hunger branch is intentionally later in the policy, but
        # putting sleep ahead of an available meal created a pathological loop:
        # hunger woke the astronaut, fatigue immediately scheduled sleep again,
        # and no calories were ever issued despite a stocked habitat.
        pre_sleep_food_available = (
            agent.inventory.has_item("emergency_rations")
            or agent.inventory.has_item("ration_pack")
            or getattr(self, "central_depot_inventory", {}).get(
                "ration_packs", 0
            ) > 0
            or float(colony_res.get("food_reserve_kcal", 0.0)) > 0.0
        )
        pre_sleep_meal_due = (
            tick - int(getattr(agent, "_last_meal_tick", -10_000))
            >= ticks_for_minutes(60.0, self.tick_minutes)
            or agent.needs.hunger <= 20.0
        )
        if (
            is_near_shelter
            and agent.needs.hunger <= 65.0
            and pre_sleep_meal_due
            and pre_sleep_food_available
        ):
            if not getattr(agent, "_in_habitat", False):
                agent.enter_habitat()
            return {
                "action": "eat",
                "target": {"habitat": True, "pre_sleep_meal": True},
                "reasoning": (
                    f"{agent.name} eating before fatigue recovery "
                    f"({agent.needs.hunger:.0f}% nutrition reserve)"
                ),
                "deterministic": True,
            }

        # --- C. EXHAUSTION / SLEEP OVERRIDE (Return Before Collapse) ---
        # Fatigue is an acute safety condition and therefore outranks hygiene.
        # Previously a water-starved crew retried a failed wash every tick and
        # never reached this sleep branch, eventually collapsing in the habitat.
        if agent.needs.energy <= 65 and is_near_shelter:
            if not getattr(agent, "_in_habitat", False):
                agent.enter_habitat()
            active_target = (
                agent.action.target
                if isinstance(getattr(agent.action, "target", None), dict)
                else {}
            )
            sleep_decision = {
                "action": "sleep",
                "target": {
                    "ticks": ticks_for_minutes(120.0, self.tick_minutes),
                    "habitat": True,
                    # When there is no accessible water, waking for thirst
                    # cannot produce treatment. Preserve real sleep until a
                    # meaningful dose becomes available; the next decision
                    # tick will then interrupt this flag and issue DRINK.
                    "dry_base_conservation_sleep": not hydration_available,
                },
                "reasoning": f"{agent.name} sleeping inside heated habitat bunk ({agent.needs.energy:.0f}%)",
                "deterministic": agent.needs.energy <= 15,
            }
            # Once the deterministic physiology layer has already stopped a
            # multi-tick duty at its safe fatigue limit, the policy may not
            # immediately reinterpret that stop as optional overtime.  Doing
            # so turned one genuine safety interrupt into an IDLE/REST tick and
            # let the same long job be selected again.  Overtime remains an RL
            # choice while approaching the limit; crossing the enforced limit
            # requires a real sleep block before duty can resume.
            if active_target.get("physiological_safety_interrupt"):
                agent._fatigue_overtime_until_tick = -1
                sleep_decision["target"]["safety_interrupt_recovery"] = True
                sleep_decision["target"]["interrupted_action"] = (
                    active_target.get("interrupted_action")
                )
                sleep_decision["deterministic"] = True
                return sleep_decision
            dry_base_dehydration_conservation = bool(
                not hydration_available and agent.needs.thirst <= 45.0
            )
            if agent.needs.energy <= 15 or dry_base_dehydration_conservation:
                agent._fatigue_overtime_until_tick = -1
                return sleep_decision

            overtime_until = int(getattr(
                agent, "_fatigue_overtime_until_tick", -1
            ))
            if tick > overtime_until:
                fatigue_band = "severe" if agent.needs.energy < 35 else "tired"
                duty = getattr(agent.action, "action_type", "idle") or "idle"
                state_key = (
                    self.get_rl_state_key(
                        agent, getattr(self, "structures_built", {}), {}, world_context
                    )
                    + f"|fatigue:{fatigue_band}|duty:{duty}"
                )
                choice = self.select_action_via_rl(agent, state_key, [
                    sleep_decision,
                    {
                        "action": "continue_duty",
                        "target": {"destination": "overtime", "duty": duty},
                        "reasoning": "Accepting fatigue risk to pursue the current objective",
                    },
                ])
                if choice["action"] == "sleep":
                    agent._fatigue_overtime_until_tick = -1
                    return choice
                agent._fatigue_overtime_until_tick = tick + ticks_for_minutes(
                    120.0, self.tick_minutes
                )
                agent._fatigue_risk_transition = (
                    state_key, self._get_action_key(choice)
                )
        # Reserve is measured in conservative remaining travel ticks. The
        # motor can cover two cells in fair terrain, while the safety budget
        # assumes one 100 m cell under rough/low-visibility return conditions.
        fatigue_return_ticks = math.ceil(
            min_shelter_dist / routine_walk_speed
        )
        fatigue_return_threshold = float(getattr(
            agent,
            "_routine_eva_energy_return_threshold",
            min(90.0, 55.0 + fatigue_return_ticks * 1.25),
        ))
        expedition_local_return = (
            is_authorized_expedition
            and (
                active_expedition.get("status") == "returning"
                or agent.needs.energy <= 55.0
            )
        )
        if (
            not getattr(agent, "_in_habitat", False)
            and (
                expedition_local_return
                or (
                    not is_authorized_expedition
                    and agent.needs.energy <= fatigue_return_threshold
                )
            )
        ):
            if is_authorized_expedition:
                self._recall_expedition_team(agent)
            dx = 1 if shelter_x > agent.x else (-1 if shelter_x < agent.x else 0)
            dy = 1 if shelter_y > agent.y else (-1 if shelter_y < agent.y else 0)
            return {
                "action": "move",
                "target": {
                    "dx": dx, "dy": dy, "x": shelter_x, "y": shelter_y,
                    "destination": "shelter", "fatigue_return": True,
                    "expedition": is_authorized_expedition,
                },
                "reasoning": (
                    f"{agent.name} ending EVA at {agent.needs.energy:.0f}% energy "
                    f"with a distance-scaled return reserve for the {dist_from_lz}-cell "
                    "route and heading to a "
                    "heated bunk before fatigue collapse"
                ),
                "deterministic": True,
            }

        # --- D. HYGIENE / DECONTAMINATION CYCLE ---
        # Only schedule a wash when the requested water actually exists. A dry
        # habitat should conserve energy and continue other work, not issue an
        # impossible wash command on every decision tick.
        depot = getattr(self, "central_depot_inventory", {})
        wash_water_available = (
            getattr(self, "structures_built", {}).get("water_collector", 0) > 0
            and float(colony_res.get("water_reserve_l", 0.0))
            >= max(2.0, 5.0 * sum(
                1 for crew in getattr(self, "agents", [agent])
                if getattr(crew.status, "value", str(crew.status)) != "dead"
            ) + 2.0)
        )
        if agent.needs.hygiene <= 25.0 and wash_water_available:
            if is_near_shelter:
                if not getattr(agent, "_in_habitat", False):
                    agent.enter_habitat()
                return {
                    "action": "wash",
                    "target": {"habitat": True, "water_liters": 2.0},
                    "reasoning": (
                        f"{agent.name} performing scheduled habitat wash and "
                        f"EVA decontamination ({agent.needs.hygiene:.0f}% hygiene)"
                    ),
                    "deterministic": True,
                }
            if agent.needs.hygiene <= 15.0:
                dx = 1 if shelter_x > agent.x else (-1 if shelter_x < agent.x else 0)
                dy = 1 if shelter_y > agent.y else (-1 if shelter_y < agent.y else 0)
                return {
                    "action": "move",
                    "target": {
                        "dx": dx, "dy": dy, "x": shelter_x, "y": shelter_y,
                        "destination": "decontamination",
                    },
                    "reasoning": f"{agent.name} returning for mandatory hygiene and EVA decontamination",
                    "deterministic": True,
                }

        # --- D. HUNGER OVERRIDE (Proactive Base Meals & In-Suit Ration Bars) ---
        hunger_thresh = 65 if is_near_shelter else 45
        carried_food_available = (
            agent.inventory.has_item("emergency_rations")
            or agent.inventory.has_item("ration_pack")
        )
        stored_food_available = is_near_shelter and (
            getattr(self, "central_depot_inventory", {}).get(
                "ration_packs", 0
            ) > 0
            or float(colony_res.get("food_reserve_kcal", 0.0)) > 0.0
        )
        if (
            agent.needs.hunger <= hunger_thresh
            and (
                tick - int(getattr(agent, "_last_meal_tick", -10_000))
                >= ticks_for_minutes(60.0, self.tick_minutes)
                or agent.needs.hunger <= 20.0
            )
            and (carried_food_available or stored_food_available)
        ):
            if is_near_shelter:
                if not getattr(agent, "_in_habitat", False):
                    agent.enter_habitat()
                return {
                    "action": "eat",
                    "target": {"habitat": True},
                    "reasoning": f"{agent.name} eating meal inside base habitat ({agent.needs.hunger:.0f}%)",
                    "deterministic": True
                }
            else:
                # In-Suit Nutrient Bar: Consumes helmet bite-bar during EVA
                return {
                    "action": "eat",
                    "target": {"field": True},
                    "reasoning": f"{agent.name} consuming in-suit nutrient bar during EVA ({agent.needs.hunger:.0f}%)",
                    "deterministic": True
                }

        # --- D. THERMAL RE-WARMING & ACUTE HYPOTHERMIA REFLEX ---
        # Inside habitat re-warming
        if getattr(agent, "_in_habitat", False) and (agent.needs.temperature_stress < 48 or agent.needs.temperature_stress > 52):
            return {
                "action": "rest",
                "target": {},
                "reasoning": f"{agent.name} re-warming in 21°C ECLSS habitat (temp: {agent.needs.temperature_stress:.0f}%)",
                "deterministic": True
            }
        # Acute lethal hypothermia emergency reflex (< 15% core threshold)
        elif agent.needs.temperature_stress <= 15.0 or agent.needs.temperature_stress >= 85.0:
            if is_near_shelter:
                return {
                    "action": "enter_habitat",
                    "target": {},
                    "reasoning": f"ACUTE THERMAL EMERGENCY: {agent.name} rushing into habitat airlock ({agent.needs.temperature_stress:.0f}%)",
                    "deterministic": True
                }

        # A detector-confirmed rover mission needs two people. Preserve the
        # dispatcher's buddy assignment after this astronaut finishes current
        # self-care; otherwise agent turn order immediately assigns another
        # ordinary job and the lead can wait beside an idle rover for days.
        buddy_reservation = getattr(
            agent, "_expedition_buddy_reservation", None
        )
        if (
            isinstance(buddy_reservation, dict)
            and not isinstance(getattr(agent, "_active_expedition", None), dict)
        ):
            reservation_lead = next((
                crew for crew in getattr(self, "agents", [])
                if crew.id == buddy_reservation.get("lead_id")
            ), None)
            lead_contract = (
                getattr(reservation_lead, "_detected_resource_recovery", None)
                if reservation_lead is not None else None
            )
            reservation_valid = (
                reservation_lead is not None
                and getattr(
                    reservation_lead.status,
                    "value",
                    str(reservation_lead.status),
                ) not in {"dead", "incapacitated"}
                and isinstance(lead_contract, dict)
                and lead_contract.get("resource")
                == buddy_reservation.get("resource")
                and int(lead_contract.get("x", -1))
                == int(buddy_reservation.get("target_x", -2))
                and int(lead_contract.get("y", -1))
                == int(buddy_reservation.get("target_y", -2))
                and int(buddy_reservation.get("expires_tick", -1)) >= tick
            )
            if not reservation_valid:
                agent._expedition_buddy_reservation = None
            elif max(abs(agent.x - lz_x), abs(agent.y - lz_y)) > 1:
                return {
                    "action": "move",
                    "target": {
                        "x": lz_x,
                        "y": lz_y,
                        "destination": "shelter",
                        "expedition_buddy_standby": True,
                    },
                    "reasoning": (
                        f"{agent.name} returning to base for the reserved "
                        f"{buddy_reservation.get('resource')} rover crew"
                    ),
                    "deterministic": True,
                }
            else:
                return {
                    "action": "stand_watch",
                    "target": {
                        "expedition_buddy_standby": True,
                        "lead_id": buddy_reservation.get("lead_id"),
                        "resource": buddy_reservation.get("resource"),
                    },
                    "reasoning": (
                        f"{agent.name} standing by as the assigned buddy for "
                        f"the {buddy_reservation.get('resource')} rover sortie"
                    ),
                    "deterministic": True,
                }

        # Self-care at the departure rack can legitimately replace MOVE for a
        # few ticks. Reconstruct the still-active outbound contract afterward
        # so the reserved rover/crew pair cannot fall through to an unrelated
        # local work order while the vehicle remains in use.
        if (
            is_authorized_expedition
            and active_expedition.get("status") == "outbound"
            and active_expedition.get("kind") != "construction_support"
        ):
            expedition_kind = active_expedition.get("kind")
            destination = (
                "survey_station"
                if expedition_kind == "regional_survey"
                else "extraction_face"
            )
            return {
                "action": "move",
                "target": {
                    "x": int(active_expedition.get("target_x", agent.x)),
                    "y": int(active_expedition.get("target_y", agent.y)),
                    "destination": destination,
                    "resource": active_expedition.get(
                        "resource", "regolith"
                    ),
                    "mission_action": (
                        "survey_resources"
                        if expedition_kind == "regional_survey"
                        else "gather"
                    ),
                    "expedition": True,
                    "transport": active_expedition.get(
                        "transport", "on_foot"
                    ),
                    "rover_id": active_expedition.get("rover_id"),
                    "move_speed_cells": int(active_expedition.get(
                        "move_speed_cells", 1
                    )),
                },
                "reasoning": (
                    f"{agent.name} resuming the committed outbound "
                    f"{active_expedition.get('resource', 'field')} expedition"
                ),
                "deterministic": True,
            }

        rover_field_assignment_locked = bool(
            is_authorized_expedition
            and active_expedition.get("transport") == "crew_rover"
            and active_expedition.get("status") == "working"
            and active_expedition.get("kind") != "construction_support"
        )

        # --- E. ARRIVAL FOLLOW-THROUGH ---
        # Navigation is only a transport phase of a concrete job. The motor
        # layer preserves the mission payload on ARRIVED so the astronaut does
        # the work at the destination instead of selecting a fresh unrelated
        # route on the following tick.
        if (
            getattr(agent.action, "action_type", "") == "arrived"
            and not rover_field_assignment_locked
        ):
            arrived = (
                agent.action.target
                if isinstance(agent.action.target, dict) else {}
            )
            destination = arrived.get("destination")
            resource = arrived.get("resource")
            recipe_name = arrived.get("recipe")
            output = arrived.get("output")
            if arrived.get("mission_action") == "gather" or destination in (
                "extraction_face", "active_excavation_face"
            ):
                return {
                    "action": "gather",
                    "target": {**arrived, "resource": resource or "regolith"},
                    "reasoning": (
                        f"{agent.name} arrived at the assigned {resource or 'regolith'} "
                        "working face and is starting extraction"
                    ),
                    "deterministic": True,
                }
            if destination in ("prospect_station", "survey_station"):
                return {
                    "action": "gather",
                    "target": {**arrived, "resource": resource or "regolith"},
                    "reasoning": (
                        f"{agent.name} arrived at the reserved geological station "
                        f"and is prospecting for {resource or 'materials'}"
                    ),
                    "deterministic": True,
                }
            if destination == "manufacturing_machine" and output:
                return {
                    "action": "refine",
                    "target": {**arrived, "output": output},
                    "reasoning": (
                        f"{agent.name} arrived at the fabrication machine and is "
                        f"starting the scheduled {output} batch"
                    ),
                    "deterministic": True,
                }
            if (
                destination in ("construction_site", "planned_construction_site")
                and recipe_name
                and not (
                    isinstance(getattr(agent, "_active_expedition", None), dict)
                    and agent._active_expedition.get("kind") == "construction_support"
                )
            ):
                return {
                    "action": "build",
                    "target": {**arrived, "recipe": recipe_name},
                    "reasoning": (
                        f"{agent.name} arrived at the {recipe_name} site and is "
                        "continuing physical construction"
                    ),
                    "deterministic": True,
                }
            if destination == "o2_filling_station":
                return {
                    "action": "refill_o2",
                    "target": dict(arrived),
                    "reasoning": f"{agent.name} arrived at the O2 service station",
                    "deterministic": True,
                }
            if (
                destination == "greenhouse_workstation"
                and arrived.get("greenhouse_id")
                and self._greenhouse_work_available(arrived["greenhouse_id"])
            ):
                return {
                    "action": "farm",
                    "target": dict(arrived),
                    "reasoning": (
                        f"{agent.name} arrived at greenhouse "
                        f"{arrived.get('greenhouse_id')} for "
                        f"{arrived.get('sub_action', 'crop care')}"
                    ),
                    "deterministic": True,
                }

        # --- F. NAVIGATION COMMITMENT ---
        # One grid step completes every tick, but the destination remains a
        # multi-tick task. Preserve that destination until arrival unless one
        # of the survival overrides above interrupts it.
        if (
            getattr(agent.action, "action_type", "") == "move"
            and not rover_field_assignment_locked
            and (
                not isinstance(agent.action.target, dict)
                or agent.action.target.get("destination") != "greenhouse_workstation"
                or self._greenhouse_work_available(
                    agent.action.target.get("greenhouse_id", "")
                )
            )
        ):
            previous_target = agent.action.target if isinstance(agent.action.target, dict) else {}
            target_x = previous_target.get("x")
            target_y = previous_target.get("y")
            if target_x is not None and target_y is not None:
                target_from_lz = max(abs(int(target_x) - lz_x), abs(int(target_y) - lz_y))
                if target_from_lz > max_eva_radius:
                    dx = 1 if lz_x > agent.x else (-1 if lz_x < agent.x else 0)
                    dy = 1 if lz_y > agent.y else (-1 if lz_y < agent.y else 0)
                    return {
                        "action": "move",
                        "target": {
                            "dx": dx, "dy": dy, "x": lz_x, "y": lz_y,
                            "destination": "shelter", "forced_return": True,
                            "expedition": is_authorized_expedition,
                        },
                        "reasoning": (
                            f"{agent.name} cancelled route to ({target_x}, {target_y}); "
                            f"target lies outside the {max_eva_radius}-cell EVA radius"
                        ),
                        "deterministic": True,
                    }
                remaining = max(abs(agent.x - target_x), abs(agent.y - target_y))
                if remaining > 0:
                    return {
                        "action": "move",
                        "target": dict(previous_target),
                        "reasoning": f"{agent.name} continuing committed route to ({target_x}, {target_y}); {remaining} cells remain",
                        "deterministic": True,
                    }
            else:
                dx = 1 if shelter_x > agent.x else (-1 if shelter_x < agent.x else 0)
                dy = 1 if shelter_y > agent.y else (-1 if shelter_y < agent.y else 0)
                return {
                    "action": "move",
                    "target": {"dx": dx, "dy": dy, "x": shelter_x, "y": shelter_y, "destination": "shelter"},
                    "reasoning": f"ACUTE THERMAL EMERGENCY: {agent.name} sprinting for shelter ({agent.needs.temperature_stress:.0f}%)",
                    "deterministic": True
                }

        if (
            isinstance(active_expedition, dict)
            and active_expedition.get("kind") == "construction_support"
        ):
            site = next((
                structure for structure in getattr(self, "placed_structures", [])
                if structure.get("id") == active_expedition.get("site_id")
                and structure.get("under_construction", False)
                and not structure.get("destroyed", False)
            ), None)
            if site is None and active_expedition.get("planned_site"):
                site = {
                    "id": active_expedition["site_id"],
                    "type": active_expedition["recipe"],
                    "x": active_expedition["target_x"],
                    "y": active_expedition["target_y"],
                }
            if site is None or active_expedition.get("status") == "returning":
                self._recall_expedition_team(agent)
                return {
                    "action": "move",
                    "target": {
                        "x": lz_x, "y": lz_y, "destination": "shelter",
                        "expedition": True,
                    },
                    "reasoning": "Construction rover crew returning together to the hub",
                    "deterministic": True,
                }
            if (
                active_expedition.get("status") == "working"
                and active_expedition.get("role") == "buddy"
                and int(active_expedition.get("construction_crew_limit", 2)) < 2
            ):
                return {
                    "action": "stand_watch",
                    "target": {"expedition": True, "struct_id": site["id"]},
                    "reasoning": "Second rover occupant maintains watch at the single-person work front",
                    "deterministic": True,
                }
            return {
                "action": "build" if active_expedition.get("status") == "working" else "move",
                "target": {
                    "x": site["x"], "y": site["y"], "struct_id": site["id"],
                    "recipe": site["type"], "destination": "construction_site",
                    "expedition": True, "construction_route": True,
                },
                "reasoning": "Assigned rover crew supervising the physical construction site",
                "deterministic": True,
            }

        # Waiting for a returning partner does not prohibit drinking, eating
        # or a sleep block above. It does prohibit a new unrelated sortie.
        if (
            isinstance(active_expedition, dict)
            and active_expedition.get("status") == "awaiting_buddy"
        ):
            return {
                "action": "stand_watch",
                "target": {"awaiting_expedition_member": True, "habitat": True},
                "reasoning": "Remaining at the hub until the returning expedition partner is accounted for",
                "deterministic": True,
            }

        # At a remote survey site the expedition remains mission-locked. The
        # lead extracts the named resource; the buddy stands by rather than
        # wandering into an unrelated RL task.
        if is_authorized_expedition and active_expedition.get("status") == "working":
            if active_expedition.get("role") == "buddy":
                return {
                    "action": "stand_watch",
                    "target": {"field": True, "expedition": True},
                    "reasoning": (
                        f"{agent.name} maintaining buddy watch for "
                        f"{active_expedition.get('resource', 'survey')} expedition"
                    ),
                    "deterministic": True,
                }
            if active_expedition.get("kind") == "regional_survey":
                return {
                    "action": "survey_resources",
                    "target": self._scanner_target(
                        agent,
                        {
                            "resource": active_expedition.get("resource"),
                            "x": active_expedition.get("survey_station_x"),
                            "y": active_expedition.get("survey_station_y"),
                            "survey_action": "portable_scanner",
                            "regional_expedition": True,
                        },
                        preserve_primary=True,
                    ),
                    "reasoning": f"{agent.name} scanning the regional field station",
                    "deterministic": True,
                }
            return {
                "action": "gather",
                "target": {
                    "resource": active_expedition.get("resource", "regolith"),
                    "expedition": True,
                    "capacity_recipe": active_expedition.get(
                        "capacity_recipe"
                    ),
                    "recovery_goal_units": active_expedition.get(
                        "recovery_goal_units"
                    ),
                },
                "reasoning": (
                    f"{agent.name} extracting expedition target "
                    f"{active_expedition.get('resource', 'resource')} before return"
                ),
                "deterministic": True,
            }

        # Manufactured tools are finite.  If every issued tool has worn out,
        # recover through the Tier-0 stone-hammer recipe from recipes.json
        # instead of repeatedly pulling a structure's materials into the
        # engineer's backpack and failing the craft precondition.
        if not agent.inventory.has_usable_tool():
            primary_domain = (
                agent.competency.get_primary_domain()
                if hasattr(agent, "competency") else "engineering"
            )
            nearby_tool_donor = next((
                other for other in getattr(self, "agents", [])
                if other.id != agent.id
                and getattr(other.status, "value", str(other.status)) != "dead"
                and other.inventory.has_usable_tool()
                and max(abs(other.x - agent.x), abs(other.y - agent.y)) <= 1
                and bool(getattr(other, "_in_habitat", False))
                == bool(getattr(agent, "_in_habitat", False))
            ), None)
            # A scarce working tool is mission equipment, not personal
            # property. Give it to the chief engineer at the shared base before
            # manufacturing a primitive replacement.
            if primary_domain == "engineering" and nearby_tool_donor is not None:
                return {
                    "action": "borrow_tool",
                    "target": {"donor_id": nearby_tool_donor.id},
                    "reasoning": (
                        f"{agent.name} taking custody of a working shared tool "
                        f"from {nearby_tool_donor.name} for critical construction"
                    ),
                    "deterministic": True,
                }
            fallback_recipe = self._recipes_cache.get("stone_hammer", {})
            fallback_costs = fallback_recipe.get("materials", {})
            colony_mats = dict(getattr(self, "central_depot_inventory", {}))
            for other in getattr(self, "agents", [agent]):
                if getattr(other.status, "value", str(other.status)) != "dead":
                    for material, quantity in other.inventory.materials.items():
                        colony_mats[material] = colony_mats.get(material, 0) + quantity
            missing_fallback = next((
                material for material, needed in fallback_costs.items()
                if colony_mats.get(material, 0) < needed
            ), None)
            if fallback_recipe and not missing_fallback:
                return {
                    "action": "craft_item",
                    "target": {"recipe": "stone_hammer", "tool_recovery": True},
                    "reasoning": (
                        f"{agent.name} fabricating the configured Tier-0 stone "
                        "hammer after the issued tools wore out"
                    ),
                    "deterministic": True,
                }
            if missing_fallback and primary_domain == "engineering":
                return {
                    "action": "gather",
                    "target": {"resource": missing_fallback, "tool_recovery": True},
                    "reasoning": (
                        f"{agent.name} gathering {missing_fallback} for the "
                        "configured Tier-0 stone-hammer recipe"
                    ),
                    "deterministic": True,
                }

        # Biological production needs visible, bounded human work. Crop duty
        # follows medical/survival and committed navigation, but precedes
        # ordinary asset maintenance and raw-material campaigns so six farms
        # cannot silently remain unattended for months.
        greenhouse_decision = self._greenhouse_duty_decision(agent, tick)
        if greenhouse_decision is not None:
            return greenhouse_decision

        # Maintenance is a physical mission dependency, not optional cleanup.
        # Dispatch it before unfinished excavation and geological scanning so
        # a long search for one scarce mineral cannot destroy the very
        # furnace/forge needed to process that mineral. Physiological, SAR,
        # expedition and tool-safety branches above remain authoritative.
        structures_built = getattr(self, "structures_built", {})
        maintenance_decision = self._structure_maintenance_decision(
            agent, structures_built
        )
        if maintenance_decision is not None:
            return maintenance_decision

        # Do not abandon a partially cleared production face to open another
        # test pit. Survival, navigation and tool-recovery overrides above may
        # interrupt it, but after recovery the crew returns to the same grid
        # until the requested stratum is physically exposed.
        detected_recovery = getattr(
            agent, "_detected_resource_recovery", None
        )
        if isinstance(detected_recovery, dict):
            recovery_resource = detected_recovery.get("resource")
            recovery_target = (
                int(detected_recovery.get("x", agent.x)),
                int(detected_recovery.get("y", agent.y)),
            )
            capacity_recipe = detected_recovery.get("capacity_recipe")
            recovery_goal_units = detected_recovery.get(
                "recovery_goal_units"
            )
            recovery_needed = True
            if capacity_recipe and recovery_resource:
                recovery_needed = self._raw_bom_deficits(
                    capacity_recipe, self._pooled_materials()
                ).get(recovery_resource, 0) > 0
            recovery_depleted = (
                recovery_target[0], recovery_target[1], recovery_resource
            ) in getattr(self, "depleted_cell_resources", set())
            if not recovery_needed or recovery_depleted:
                agent._detected_resource_recovery = None
                detected_recovery = None
            elif dist_from_lz > 2:
                return {
                    "action": "move",
                    "target": {
                        "x": lz_x,
                        "y": lz_y,
                        "destination": "shelter",
                        "resource": recovery_resource,
                        "detector_recovery": True,
                    },
                    "reasoning": (
                        f"{agent.name} returning to base to organize rover "
                        f"recovery of the confirmed {recovery_resource} face"
                    ),
                    "deterministic": True,
                }
            else:
                return {
                    "action": "gather",
                    "target": {
                        "x": recovery_target[0],
                        "y": recovery_target[1],
                        "resource": recovery_resource,
                        "capacity_recipe": capacity_recipe,
                        "recovery_goal_units": recovery_goal_units,
                        "detector_recovery": True,
                    },
                    "reasoning": (
                        f"{agent.name} organizing rover recovery of the "
                        f"detector-confirmed {recovery_resource} face"
                    ),
                    "deterministic": True,
                }

        active_excavation = getattr(agent, "_active_excavation", None)
        if isinstance(active_excavation, dict):
            active_resource = active_excavation.get("resource", "regolith")
            capacity_recipe = (
                active_excavation.get("capacity_recipe")
                or getattr(self, "shared_work_order", {}).get("recipe")
            )
            # Excavated layers and spoil piles persist in the world. Once the
            # associated mission BOM no longer has a deficit for this raw
            # material, release the astronaut while preserving the partially
            # opened face for later use. The old unconditional lock stranded
            # most of the crew on surplus feedstocks while sulfur stayed at 0.
            if capacity_recipe:
                outstanding_raw = self._raw_bom_deficits(
                    capacity_recipe, self._pooled_materials()
                )
                if int(outstanding_raw.get(active_resource, 0)) <= 0:
                    agent._active_excavation = None
                    active_excavation = None
        if isinstance(active_excavation, dict):
            target_x = int(active_excavation.get("x", agent.x))
            target_y = int(active_excavation.get("y", agent.y))
            resource = active_excavation.get("resource", "regolith")
            if (agent.x, agent.y) != (target_x, target_y):
                return {
                    "action": "move",
                    "target": {
                        "x": target_x,
                        "y": target_y,
                        "destination": "active_excavation_face",
                        "resource": resource,
                        "capacity_recipe": capacity_recipe,
                    },
                    "reasoning": (
                        f"{agent.name} returning to the unfinished {resource} "
                        "working face instead of opening another test pit"
                    ),
                    "deterministic": True,
                }
            return {
                "action": "gather",
                "target": {
                    "x": target_x,
                    "y": target_y,
                    "resource": resource,
                    "continue_excavation": True,
                    "capacity_recipe": capacity_recipe,
                },
                "reasoning": (
                    f"{agent.name} finishing overburden removal at the active "
                    f"{resource} working face"
                ),
                "deterministic": True,
            }

        # A geological scanner reveals only the ground physically surveyed by
        # its custodian. Its battery is rechargeable from base power and is
        # separate from the instrument's much slower physical wear.
        planetary_resources = set(getattr(self, "planetary_resources", set()))
        remote_requests = sorted(
            resource
            for resource in getattr(self, "remote_resource_requests", set())
            if not planetary_resources or resource in planetary_resources
        )
        locally_exhausted_resources = set(
            getattr(self, "local_survey_exhausted_resources", set())
        )
        priority_recipe = None
        if getattr(self, "local_survey_exhausted", False):
            pooled_materials = dict(getattr(self, "central_depot_inventory", {}))
            for crew in getattr(self, "agents", [agent]):
                if getattr(crew.status, "value", str(crew.status)) != "dead":
                    for material, quantity in crew.inventory.materials.items():
                        pooled_materials[material] = pooled_materials.get(material, 0) + quantity
            capacity_probe = self._balanced_milestone_candidate(
                agent, pooled_materials, getattr(self, "structures_built", {})
            )
            capacity_resource = (
                capacity_probe.get("target", {}).get("resource")
                if isinstance(capacity_probe, dict) else None
            )
            priority_recipe = (
                capacity_probe.get("target", {}).get("capacity_recipe")
                if isinstance(capacity_probe, dict) else None
            )
            if capacity_resource:
                self.remote_resource_requests.add(capacity_resource)
                remote_requests.append(capacity_resource)
        remote_requests = self._prioritize_live_bom_resources(
            remote_requests, priority_recipe
        )
        scanner_durability = int(
            agent.inventory.tool_durability.get("portable_scanner", 0)
        )
        scanner_charge = float(
            agent.inventory.tool_charge_pct.get("portable_scanner", 0.0)
        )
        scan_charge_cost = float(getattr(
            self, "portable_scanner_scan_charge_pct", 8.0
        ))
        active_expedition = getattr(agent, "_active_expedition", None)
        regional_survey = (
            isinstance(active_expedition, dict)
            and active_expedition.get("kind") == "regional_survey"
            and active_expedition.get("role") == "lead"
        )
        regional_requests = [
            resource for resource in remote_requests
            if getattr(self, "local_survey_exhausted", False)
            or resource in locally_exhausted_resources
        ]
        local_requests = [
            resource for resource in remote_requests
            if resource not in regional_requests
        ]
        if (
            isinstance(active_expedition, dict)
            and active_expedition.get("kind") == "resource_recovery"
            and active_expedition.get("role") == "lead"
            and active_expedition.get("status") == "working"
        ):
            return {
                "action": "gather",
                "target": {
                    "resource": active_expedition.get("resource"),
                    "expedition": True,
                },
                "reasoning": (
                    f"{agent.name} recovering the deposit physically confirmed "
                    "by the regional scanner traverse"
                ),
                "deterministic": True,
            }
        if regional_survey and active_expedition.get("status") == "working":
            return {
                "action": "survey_resources",
                "target": self._scanner_target(
                    agent,
                    {
                        "resource": active_expedition.get("resource"),
                        "x": active_expedition.get("survey_station_x"),
                        "y": active_expedition.get("survey_station_y"),
                        "survey_action": "portable_scanner",
                        "regional_expedition": True,
                    },
                    remote_requests,
                    preserve_primary=True,
                ),
                "reasoning": f"{agent.name} scanning the authorized regional field station",
                "deterministic": True,
            }
        if (
            regional_requests
            and agent.inventory.has_item("portable_scanner")
            and scanner_durability > 0
            and not isinstance(active_expedition, dict)
            and tick >= int(getattr(agent, "_regional_survey_retry_tick", 0))
        ):
            if scanner_charge < scan_charge_cost:
                return {
                    "action": "recharge_scanner",
                    "target": self._scanner_target(
                        agent,
                        {"resource": regional_requests[0]},
                        remote_requests,
                        preserve_primary=True,
                    ),
                    "reasoning": f"{agent.name} charging the scanner before a regional traverse",
                    "deterministic": True,
                }
            return {
                "action": "start_regional_survey",
                "target": self._scanner_target(
                    agent,
                    {"resource": regional_requests[0]},
                    remote_requests,
                    preserve_primary=True,
                ),
                "reasoning": (
                    f"{agent.name} organizing a buddy survey beyond the exhausted "
                    "local geological campaign"
                ),
                "deterministic": True,
            }
        if (
            local_requests
            and agent.inventory.has_item("portable_scanner")
            and scanner_durability > 0
            and tick >= int(getattr(agent, "_regional_survey_retry_tick", 0))
        ):
            if (
                scanner_charge < scan_charge_cost
                and float(colony_res.get("energy_stored_kwh", 0.0)) > 0.0
            ):
                return {
                    "action": "recharge_scanner",
                    "target": self._scanner_target(
                        agent,
                        {"resource": local_requests[0]},
                        remote_requests,
                        preserve_primary=True,
                    ),
                    "reasoning": (
                        f"{agent.name} returning the scanner to colony power "
                        f"at {scanner_charge:.0f}% charge"
                    ),
                    "deterministic": True,
                }
            if scanner_charge >= scan_charge_cost:
                return {
                    "action": "survey_resources",
                    "target": self._scanner_target(
                        agent,
                        {"resource": local_requests[0]},
                        remote_requests,
                        preserve_primary=True,
                    ),
                    "reasoning": (
                        f"{agent.name} conducting an on-site shallow survey for "
                        f"{local_requests[0]} ({scanner_charge:.0f}% battery)"
                    ),
                    "deterministic": True,
                }
        
        # === DETERMINISTIC SOLAR-ARRAY DUST MAINTENANCE ===
        # This used to live only at the end of the RL candidate list. The
        # chief engineer returned an infrastructure milestone earlier in this
        # method, so a panel could reach 100% fouling without ever generating
        # a cleaning order. Assign one healthy, awake technician and preserve
        # the physical structure target across the EVA route.
        solar_cleaning_threshold = float(getattr(
            self, "solar_cleaning_dust_threshold", 0.20
        ))
        dirty_solar_panels = [
            structure for structure in getattr(self, "placed_structures", [])
            if structure.get("type") in {"solar_panel", "eclss_lander_hub"}
            and not structure.get("under_construction", False)
            and not structure.get("destroyed", False)
            and float(structure.get("dust_fouling_level", 0.0))
            >= solar_cleaning_threshold
        ]
        if dirty_solar_panels:
            dirtiest_panel = max(
                dirty_solar_panels,
                key=lambda structure: float(
                    structure.get("dust_fouling_level", 0.0)
                ),
            )
            panel_id = dirtiest_panel.get("id")

            def available_solar_technician(crew_member: Agent) -> bool:
                action_type = getattr(crew_member.action, "action_type", "")
                active_target = (
                    crew_member.action.target
                    if isinstance(crew_member.action.target, dict) else {}
                )
                continuing_this_job = (
                    active_target.get("maintenance_action") == "clean_solar_panels"
                    and active_target.get("structure_id") == panel_id
                )
                return (
                    getattr(crew_member.status, "value", str(crew_member.status))
                    != "dead"
                    and getattr(crew_member.competency, "engineering", 0) >= 5
                    and crew_member.needs.energy > 65.0
                    and crew_member.needs.hunger > 45.0
                    and crew_member.needs.thirst > 45.0
                    and crew_member.needs.o2_supply > 40.0
                    and 42.0 < crew_member.needs.temperature_stress < 75.0
                    and (
                        continuing_this_job
                        or not crew_member.action.is_active
                        or action_type in ("idle", "rest", "enter_habitat")
                    )
                )

            technicians = [
                crew_member for crew_member in getattr(self, "agents", [agent])
                if available_solar_technician(crew_member)
            ]
            if technicians:
                assigned = min(
                    technicians,
                    key=lambda crew_member: (
                        0 if (
                            isinstance(crew_member.action.target, dict)
                            and crew_member.action.target.get("maintenance_action")
                            == "clean_solar_panels"
                            and crew_member.action.target.get("structure_id") == panel_id
                        ) else 1,
                        max(
                            abs(crew_member.x - dirtiest_panel.get("x", crew_member.x)),
                            abs(crew_member.y - dirtiest_panel.get("y", crew_member.y)),
                        ),
                        -getattr(crew_member.competency, "engineering", 0),
                        crew_member.id,
                    ),
                )
                if assigned.id == agent.id:
                    return {
                        "action": "clean_solar_panels",
                        "target": {
                            "structure_id": panel_id,
                            "x": dirtiest_panel.get("x"),
                            "y": dirtiest_panel.get("y"),
                            "dust_fouling_level": dirtiest_panel.get(
                                "dust_fouling_level", 0.0
                            ),
                            "maintenance": "solar_cleaning",
                            "maintenance_action": "clean_solar_panels",
                        },
                        "reasoning": (
                            f"{agent.name} assigned to clean solar array "
                            f"{panel_id} at "
                            f"{dirtiest_panel.get('dust_fouling_level', 0.0):.0%} "
                            "dust fouling"
                        ),
                        "deterministic": True,
                    }

        self._release_autonomous_operator_token(
            agent, getattr(self, "manufacturing_cycles", {})
        )
        paused_manufacturing = getattr(agent, "_paused_manufacturing", None)
        if isinstance(paused_manufacturing, dict) and paused_manufacturing.get("output"):
            paused_output = paused_manufacturing["output"]
            paused_recipe = self._recipes_cache.get(paused_output, {})
            required_machine = (
                paused_manufacturing.get("machine_type")
                or paused_recipe.get("requires_structure")
                or "stone_furnace"
            )
            living = [
                crew for crew in getattr(self, "agents", [agent])
                if getattr(crew.status, "value", str(crew.status))
                not in ("dead", "incapacitated")
            ]
            machine_busy = self._all_machine_slots_busy(
                required_machine, living, agent, tick=tick
            )
            paused_machine_id = paused_manufacturing.get("machine_id")
            canonical_cycle = getattr(
                self, "manufacturing_cycles", {}
            ).get(str(paused_machine_id)) if paused_machine_id else None
            autonomous_process_running = bool(
                canonical_cycle is not None
                and canonical_cycle.get("autonomous")
                and canonical_cycle.get("active")
                and not canonical_cycle.get("output_ready")
            )
            if autonomous_process_running:
                # The WIP token keeps the committed batch represented in the
                # shared BOM, but the qualified closed-loop cell needs no
                # continuous operator. Give this astronaut productive feeder
                # work until the engine transitions the same cycle to its
                # short unload/inspection state.
                pooled = self._pooled_materials()
                active_order = self._select_shared_capacity_order(
                    pooled, structures_built, tick
                )
                if active_order is not None:
                    worker_index = next((
                        index for index, crew in enumerate(living)
                        if crew.id == agent.id
                    ), 0)
                    support = self._raw_support_action(
                        active_order["recipe"], pooled, worker_index,
                        agent, tick,
                    )
                    return {
                        **support,
                        "target": {
                            **support.get("target", {}),
                            "paused_output": paused_output,
                            "autonomous_process_running": True,
                        },
                        "reasoning": (
                            f"{agent.name} leaves the interlocked "
                            f"{paused_output} batch under closed-loop control "
                            "and advances the live mission BOM"
                        ),
                    }
                return {
                    "action": "stand_watch",
                    "target": {
                        "paused_output": paused_output,
                        "autonomous_process_running": True,
                        "machine_id": paused_machine_id,
                    },
                    "reasoning": (
                        f"{agent.name} monitors autonomous {paused_output} "
                        "telemetry while no feeder work is due"
                    ),
                    "deterministic": True,
                }
            if canonical_cycle is not None and canonical_cycle.get(
                "output_ready"
            ):
                # This astronaut's exact finished WIP can be claimed even if
                # every other machine also contains a completed batch.
                machine_busy = False
            needs = agent.needs
            o2_safe = bool(getattr(agent, "_in_habitat", False)) or (
                needs.o2_supply > 40.0
            )
            duty_ticks = min(
                max(1, int(paused_manufacturing.get("remaining_ticks", 1))),
                int(getattr(self, "manufacturing_shift_ticks", 78)),
            )
            operator_fit = (
                needs.energy > min(96.0, 65.0 + duty_ticks * 0.4)
                and needs.hunger > min(60.0, 35.0 + duty_ticks * 0.25)
                and needs.thirst > min(70.0, 45.0 + duty_ticks * 0.25)
                and 42.0 < needs.temperature_stress < 88.0
                and o2_safe
            )
            # A completed duty block is held for a relief operator for one
            # full shift. The original operator must not hammer RESUME every
            # tick while the engine correctly denies that handoff window.
            shift_end_tick = int((canonical_cycle or paused_manufacturing).get(
                "shift_end_tick", tick
            ))
            relief_until_tick = int((canonical_cycle or {}).get(
                "relief_until_tick",
                shift_end_tick + int(
                    getattr(self, "manufacturing_shift_ticks", 78)
                ),
            ))
            relief_route_owner = (
                (canonical_cycle or {}).get("route_operator_id")
            )
            relief_route_active = bool(
                relief_route_owner
                and relief_route_owner != agent.id
                and tick <= int((canonical_cycle or {}).get(
                    "route_reserved_until_tick", -1
                ))
            )
            relief_hold = (
                canonical_cycle is not None
                and canonical_cycle.get("operator_id") == agent.id
                and tick >= shift_end_tick
                and tick < relief_until_tick
            )
            if not operator_fit:
                in_pressurized_recovery_space = bool(
                    getattr(agent, "_in_habitat", False)
                )
                if not in_pressurized_recovery_space:
                    return {
                        "action": "move",
                        "target": {
                            "x": getattr(self, "lz_x", agent.x),
                            "y": getattr(self, "lz_y", agent.y),
                            "destination": "habitat",
                            "paused_output": paused_output,
                            "resume_when_operator_fit": True,
                        },
                        "reasoning": (
                            f"{agent.name} returning to the heated habitat "
                            f"before recovering for {paused_output} precision work"
                        ),
                        "deterministic": True,
                    }
                hydration_available = self._hydration_source_available(
                    agent,
                    near_shelter=True,
                    colony_resources=getattr(self, "_colony_resources", {}),
                )
                if needs.thirst <= 45.0 and not hydration_available:
                    # With no physical water, DRINK would execute as IDLE and
                    # the next tick would restart SLEEP because fatigue was
                    # also low. Preserve the machine-owned WIP, but release
                    # this operator to the shared objective so the crew can
                    # still attempt a real recovery before dehydration wins.
                    desperate_work = self._shared_colony_work_decision(
                        agent, tick, structures_built
                    )
                    if desperate_work is not None:
                        return {
                            **desperate_work,
                            "reasoning": (
                                f"{agent.name} has no physical water reserve; "
                                f"abandoning a futile sleep/drink loop and "
                                f"continuing {desperate_work.get('reasoning', 'colony recovery work')}"
                            ),
                        }
                if needs.thirst <= 45.0:
                    recovery_action = "drink"
                elif needs.hunger <= 35.0:
                    recovery_action = "eat"
                else:
                    # This branch exists because the operator is not fit for
                    # another precision-work shift; when hydration/nutrition
                    # are adequate, actual sleep—not an awake queue wait—is
                    # the physiological recovery that can restore readiness.
                    recovery_action = "sleep"
                return {
                    "action": recovery_action,
                    "target": {
                        "paused_output": paused_output,
                        "resume_when_operator_fit": True,
                    },
                    "reasoning": (
                        f"{agent.name} recovering to safe precision-work "
                        f"thresholds before resuming {paused_output}"
                    ),
                    "deterministic": True,
                }
            if relief_hold or relief_route_active:
                in_pressurized_recovery_space = bool(
                    getattr(agent, "_in_habitat", False)
                )
                return {
                    "action": (
                        "rest" if in_pressurized_recovery_space else "move"
                    ),
                    "target": {
                        "x": (
                            agent.x if in_pressurized_recovery_space
                            else getattr(self, "lz_x", agent.x)
                        ),
                        "y": (
                            agent.y if in_pressurized_recovery_space
                            else getattr(self, "lz_y", agent.y)
                        ),
                        "destination": "pressurized_recovery",
                        "paused_output": paused_output,
                        "awaiting_relief": True,
                    },
                    "reasoning": (
                        f"{agent.name} taking the required off-shift recovery "
                        f"while relief owns {paused_output} WIP"
                    ),
                    "deterministic": True,
                }
            if machine_busy:
                pooled = self._pooled_materials()
                active_order = self._select_shared_capacity_order(
                    pooled, structures_built, tick
                )
                if active_order is not None:
                    worker_index = next((
                        index for index, crew in enumerate(living)
                        if crew.id == agent.id
                    ), 0)
                    support = self._raw_support_action(
                        active_order["recipe"], pooled, worker_index,
                        agent, tick,
                    )
                    return {
                        **support,
                        "target": {
                            **support.get("target", {}),
                            "paused_output": paused_output,
                            "resume_when_machine_available": True,
                        },
                        "reasoning": (
                            f"{agent.name} preserving the already-materialed "
                            f"{paused_output} cycle and doing useful logistics "
                            f"while {required_machine} capacity or relief is occupied"
                        ),
                    }
            return {
                "action": "refine",
                "target": {
                    "output": paused_output,
                    "resume_manufacturing": True,
                    "shared_work_order": True,
                },
                "reasoning": (
                    f"{agent.name} resuming the paused, already-materialed "
                    f"{paused_output} machine cycle"
                ),
                "deterministic": True,
            }

        # Once the deployable workshop exists, all six crew members receive
        # role-compatible tasks from one colony work order. Survival, SAR and
        # exterior maintenance above still pre-empt construction safely.
        if (
            structures_built.get("stone_furnace", 0) > 0
            and structures_built.get("forge", 0) > 0
            and structures_built.get("cnc_fabricator", 0) > 0
        ):
            colony_mats = self._pooled_materials()

            # EVA suit service is mission-enabling life support, not optional
            # surplus manufacturing. If worn suits outnumber ready seals, the
            # engineer fabricates a batch before planning another structure.
            worn_suit_count = sum(
                1 for other in getattr(self, "agents", [agent])
                if getattr(other.status, "value", str(other.status)) != "dead"
                and getattr(other, "suit_integrity", 1.0) < 0.45
            )
            ready_seals = int(colony_mats.get("vacuum_gasket_seal", 0))
            if (
                worn_suit_count > ready_seals
                and colony_mats.get("fluoroelastomer_seal_stock", 0) >= 1
            ):
                return {
                    "action": "refine",
                    "target": {
                        "output": "vacuum_gasket_seal",
                        "structure": "cnc_fabricator",
                        "life_support_maintenance": True,
                    },
                    "reasoning": (
                        f"{agent.name} fabricating pressure-seal kits for "
                        f"{worn_suit_count} EVA suits awaiting service"
                    ),
                    "deterministic": True,
                }
            shared_decision = self._shared_colony_work_decision(
                agent, tick, structures_built
            )
            if shared_decision is not None:
                shared_target = shared_decision.get("target", {})
                blocked_resource = shared_target.get("resource")
                bootstrap_complete = (
                    not getattr(self, "surface_requires_plss", False)
                    or (
                        structures_built.get("solar_panel", 0) > 0
                        and structures_built.get("isru_o2_unit", 0) > 0
                    )
                )
                if (
                    (
                        bootstrap_complete
                        or getattr(self, "local_survey_exhausted", False)
                        or blocked_resource
                        in getattr(self, "local_survey_exhausted_resources", set())
                    )
                    and blocked_resource
                    in getattr(self, "remote_resource_requests", set())
                    and not getattr(self, "portable_regional_survey_available", False)
                ):
                    exploration_decision = self._exploration_infrastructure_candidate(
                        colony_mats, structures_built
                    )
                    if exploration_decision is not None:
                        return exploration_decision
                return self._consider_excavator_dispatch(
                    agent,
                    shared_decision,
                    world_context,
                    structures_built,
                )
        
        # === STRATEGIC RE-EVALUATION (Staggered strictly every 20 ticks) ===
        stable_agent_hash = int.from_bytes(
            hashlib.sha256(agent.id.encode("utf-8")).digest()[:8], "big"
        )
        agent_offset = (stable_agent_hash % 5) * 4
        time_for_strategic = (
            (tick + agent_offset) % self.STRATEGIC_INTERVAL == 0
            and tick != self._last_strategic_tick.get(agent.id, -1)
        )
        plan = self._plans.get(agent.id)

        replan_after_tick = int(
            self._plan_retry_after_tick.get(agent.id, -1)
        )
        replan_allowed = tick >= replan_after_tick
        if replan_allowed:
            self._plan_retry_after_tick.pop(agent.id, None)

        if (
            self.language_planning_enabled
            and replan_allowed
            and (time_for_strategic or plan is None)
        ):
            self._run_strategic(agent, tick, world_context)
            self._last_strategic_tick[agent.id] = tick
            plan = self._plans.get(agent.id)
        elif not self.language_planning_enabled:
            # Discard legacy cached plans: loading an old run must not restore
            # generative-model authority over the physical simulation.
            self._plans.pop(agent.id, None)
            plan = None
        
        # === TACTICAL CHECK ===
        plan = self._plans.get(agent.id)
        has_plan = plan is not None and not plan.is_complete
        
        # Enforce a 15-tick cooldown between tactical LLM calls for the same agent
        time_since_last_tactical = tick - self._last_tactical_tick.get(agent.id, -99)
        if (
            self.language_planning_enabled
            and replan_allowed
            and time_since_last_tactical >= 15
        ):
            trigger = TacticalTrigger.check(agent, tick_events, has_plan)
        else:
            trigger = None
        
        is_in_distress = (
            agent.needs.temperature_stress < 45 or
            agent.needs.thirst < 48 or
            agent.needs.hunger < 48 or
            agent.needs.energy < 40
        )
        if trigger:
            # Need LLM tactical decision
            decision = self._run_tactical(agent, tick, trigger, world_context, nearby_agents)
            self._last_tactical_tick[agent.id] = tick
        elif has_plan and not is_in_distress:
            # Execute current plan step deterministically
            decision = self._execute_plan_step(agent, plan)
            if decision is not None:
                self._deterministic_ticks += 1
        
        if decision is None:
            # RL-Guided Specialized Action & Tech Tree Progression
            primary = agent.competency.get_primary_domain() if hasattr(agent, 'competency') else 'engineering'
            
            # Check team & central depot pooled stock for structure building & refining
            colony_mats = dict(getattr(self, "central_depot_inventory", {}))
            for other in getattr(self, "agents", [agent]):
                if getattr(other.status, 'value', str(other.status)) != 'dead':
                    for k, v in other.inventory.materials.items():
                        colony_mats[k] = colony_mats.get(k, 0) + v

            structures_built = getattr(self, "structures_built", {})
            placed_structures = getattr(self, "placed_structures", [])
            candidates = []

            # Dynamic carry weight & inventory item count
            from src.agents.agent import MATERIAL_DENSITY_KG
            cur_weight = sum(
                qty * MATERIAL_DENSITY_KG.get(mat, 2.0)
                for mat, qty in agent.inventory.materials.items()
            )
            total_items = sum(agent.inventory.materials.values())
            
            lz_x = getattr(self, "lz_x", getattr(agent, "spawn_x", 1000))
            lz_y = getattr(self, "lz_y", getattr(agent, "spawn_y", 1000))
            dist_to_base = max(abs(agent.x - lz_x), abs(agent.y - lz_y))

            # === 1. LOGISTICS: HAUL GATHERED MATERIALS TO CENTRAL STORAGE DEPOT SILO ===
            # Efficient logistics: Agents haul only when carrying substantial payload (>= 8 items or >= 14.0kg) or when at base
            gravity_g = getattr(agent, "gravity_g", 1.31)
            carry_cap_kg = getattr(agent.inventory, "BASE_CARRY_CAPACITY_KG", 30.0) * (1 + agent.genome.strength * 0.1) / max(0.3, gravity_g)
            if total_items >= 8 or cur_weight >= 14.0 or cur_weight >= (carry_cap_kg * 0.75) or (dist_to_base <= 2 and total_items >= 3):
                candidates.append({
                    "action": "deposit_materials",
                    "target": {"x": lz_x, "y": lz_y},
                    "reasoning": f"{agent.name} hauling {total_items}x gathered resources ({cur_weight:.1f}kg) to Central Depot Silo",
                    "deterministic": True
                })

            # === 2. ACTIVE CONSTRUCTION SITE ASSISTANCE (CO-OP TEAMWORK) ===
            # If any structure is currently under construction on the colony grid, nearby agents help assemble
            active_builds = [
                s for s in placed_structures
                if s.get("under_construction", False)
                and s.get("materials_committed", True)
                and not s.get("destroyed", False)
            ]
            if active_builds:
                active_site = active_builds[0]
                site_recipe = active_site.get("type", "basic_shelter")
                dist_to_site = max(abs(agent.x - active_site.get("x", lz_x)), abs(agent.y - active_site.get("y", lz_y)))
                if dist_to_site <= 3 or primary == 'engineering':
                    candidates.append({
                        "action": "build",
                        "target": {"recipe": site_recipe, "struct_id": active_site.get("id"), "x": active_site.get("x"), "y": active_site.get("y")},
                        "reasoning": f"{agent.name} assisting on-site construction of {site_recipe.replace('_', ' ').title()} ({active_site.get('progress', 0):.0%} done)",
                        "deterministic": True
                    })

            # === 3. ROLE-SPECIFIC DOMAIN PROGRESSION ===
            if primary == 'engineering':
                # Query weakest colony category dynamically
                weakest_cat = "water"
                if hasattr(self, 'colony_score') and self.colony_score:
                    weakest_cat, score_val = self.colony_score.get_weakest_category()

                wc_count = structures_built.get("water_collector", 0)
                sp_count = structures_built.get("solar_panel", 0)
                gh_count = structures_built.get("greenhouse", 0)
                o2_count = structures_built.get("isru_o2_unit", 0)
                hab_count = structures_built.get("habitat_module", 0)
                crate_count = structures_built.get("storage_crate", 0)

                if self._is_infrastructure_planner(agent):
                    milestone_candidate = self._balanced_milestone_candidate(
                        agent, colony_mats, structures_built
                    )
                    if milestone_candidate:
                        candidates.append(milestone_candidate)

                # --- A. CHECK ALL READY-TO-BUILD SCORE-BOOSTING STRUCTURES ---

                # 1. Central Storage Container
                if crate_count == 0 and colony_mats.get("iron_ore", 0) >= 10 and colony_mats.get("basalt", 0) >= 8:
                    candidates.append({"action": "build", "target": {"recipe": "storage_crate"},
                                       "reasoning": "Constructing Central Storage Depot Container at Base!", "deterministic": True})

                # Capacity structures are selected only by the recipe-driven
                # milestone planner above. Keeping a second hardcoded material
                # table here caused one-off builds, recipe drift and accidental
                # overbuilding after a category reached its 106-person target.

                # --- B. INDUSTRIAL REFINING PIPELINE (Smelting & Precision Fabrication) ---
                if (colony_mats.get("iron_ore", 0) >= 4 and colony_mats.get("graphite", 0) >= 1 and
                    colony_mats.get("reduced_iron_ingot", 0) < 40):
                    candidates.append({"action": "refine", "target": {"raw_material": "iron_ore", "output": "reduced_iron_ingot", "structure": "stone_furnace"},
                                       "reasoning": f"{agent.name} reducing graded iron ore with graphite and casting iron ingots", "deterministic": True})

                if (colony_mats.get("silica_sand", 0) >= 6
                        and colony_mats.get("glass_pane", 0) < 14):
                    candidates.append({"action": "refine", "target": {"raw_material": "silica_sand", "output": "glass_pane", "structure": "stone_furnace"},
                                       "reasoning": f"{agent.name} smelting optical glass panes for Solar/Greenhouse/Habitat at furnace", "deterministic": True})

                if colony_mats.get("basalt", 0) >= 2 and colony_mats.get("rope_cordage", 0) < 6:
                    candidates.append({"action": "refine", "target": {"output": "rope_cordage", "structure": "cnc_fabricator"},
                                       "reasoning": f"{agent.name} braiding and testing fibre cordage in the enclosed fabrication cell", "deterministic": True})

                if (colony_mats.get("basalt", 0) >= 2 and colony_mats.get("insulated_fabric", 0) < 8):
                    candidates.append({"action": "refine", "target": {"output": "insulated_fabric", "structure": "cnc_fabricator"},
                                       "reasoning": f"{agent.name} laying up basalt-fibre insulation in the enclosed fabrication cell", "deterministic": True})

                # Only fabricate secondary components if base ingot reserve is healthy
                ingot_reserve_min = 6 if structures_built.get("water_collector", 0) == 0 else 2

                if (colony_mats.get("reduced_iron_ingot", 0) >= (ingot_reserve_min + 1) and
                    colony_mats.get("machined_bolts_fasteners", 0) < 16):
                    candidates.append({"action": "refine", "target": {"output": "machined_bolts_fasteners", "structure": "cnc_fabricator"},
                                       "reasoning": f"{agent.name} machining precision aerospace fasteners at CNC", "deterministic": True})

                if (colony_mats.get("reduced_iron_ingot", 0) >= (ingot_reserve_min + 2) and
                    colony_mats.get("metal_pipe", 0) < 4):
                    candidates.append({"action": "refine", "target": {"output": "metal_pipe", "structure": "cnc_fabricator"},
                                       "reasoning": f"{agent.name} forming, finishing and proof-testing fluid pipe at CNC", "deterministic": True})

                if (colony_mats.get("fluoroelastomer_seal_stock", 0) >= 1 and
                    colony_mats.get("vacuum_gasket_seal", 0) < 8):
                    candidates.append({"action": "refine", "target": {"output": "vacuum_gasket_seal", "structure": "cnc_fabricator"},
                                       "reasoning": f"{agent.name} molding qualified vacuum seals in the enclosed fabrication cell", "deterministic": True})

                if (colony_mats.get("reduced_iron_ingot", 0) >= (ingot_reserve_min + 2) and
                    colony_mats.get("finned_heat_sink", 0) < 2):
                    candidates.append({"action": "refine", "target": {"output": "finned_heat_sink", "structure": "cnc_fabricator"},
                                       "reasoning": f"{agent.name} machining finned radiative heat sink at CNC", "deterministic": True})

                if (colony_mats.get("qualified_composite_resin_pack", 0) >= 6 and colony_mats.get("reduced_iron_ingot", 0) >= (ingot_reserve_min + 4) and
                    colony_mats.get("glass_pane", 0) >= 2 and colony_mats.get("composite_panel", 0) < 10):
                    candidates.append({"action": "refine", "target": {"output": "composite_panel", "structure": "cnc_fabricator"},
                                       "reasoning": f"{agent.name} laying up and curing structural composite panels", "deterministic": True})

                if (colony_mats.get("qualified_control_board_core", 0) >= 1 and
                    colony_mats.get("electronic_component", 0) < 3):
                    candidates.append({"action": "refine", "target": {"output": "electronic_component", "structure": "cnc_fabricator"},
                                       "reasoning": f"{agent.name} configuring and acceptance-testing a flight-qualified control board", "deterministic": True})

                # --- C. FALLBACK: GATHER RAW MATERIALS FOR REFINING PIPELINE ---
                if not candidates:
                    if colony_mats.get("basalt", 0) < 12:
                        target_res = "basalt"
                    elif colony_mats.get("silica_sand", 0) < 8:
                        target_res = "silica_sand"
                    elif colony_mats.get("graphite", 0) < 8:
                        target_res = "graphite"
                    elif colony_mats.get("iron_ore", 0) < 16:
                        target_res = "iron_ore"
                    elif colony_mats.get("silica_sand", 0) < 16:
                        target_res = "silica_sand"
                    elif colony_mats.get("graphite", 0) < 16:
                        target_res = "graphite"
                    elif colony_mats.get("regolith", 0) < 20:
                        target_res = "regolith"
                    else:
                        target_res = "iron_ore"
                    candidates.append({"action": "gather", "target": {"resource": target_res},
                                       "reasoning": f"{agent.name} gathering {target_res} for base industrial pipeline", "deterministic": True})

            elif primary == 'physics':
                # Physicist / geologist: extracts only live industrial feedstock deficits.
                if colony_mats.get("basalt", 0) < 12:
                    res = "basalt"
                elif colony_mats.get("iron_ore", 0) < 16:
                    res = "iron_ore"
                elif colony_mats.get("silica_sand", 0) < 8:
                    res = "silica_sand"
                elif colony_mats.get("graphite", 0) < 8:
                    res = "graphite"
                elif colony_mats.get("iron_ore", 0) < 32:
                    res = "iron_ore"
                else:
                    res = "iron_ore"
                candidates.append({
                    "action": "gather",
                    "target": {"resource": res},
                    "reasoning": f"{agent.name} (Physicist/Geologist) mining {res} for colony construction pipeline",
                    "deterministic": True
                })

            elif primary in ('botany_bio', 'medical'):
                # Botanist / Medic: Extracts agricultural, optical & composite precursor materials
                if colony_mats.get("silica_sand", 0) < 8:
                    res = "silica_sand"
                elif colony_mats.get("graphite", 0) < 8:
                    res = "graphite"
                elif colony_mats.get("regolith", 0) < 15:
                    res = "regolith"
                elif colony_mats.get("iron_ore", 0) < 12:
                    res = "iron_ore"
                elif colony_mats.get("graphite", 0) < 20:
                    res = "graphite"
                else:
                    res = "silica_sand"
                candidates.append({
                    "action": "gather",
                    "target": {"resource": res},
                    "reasoning": f"{agent.name} gathering {res} for life support & environmental systems",
                    "deterministic": True
                })

            else:
                # Commander / Logistics: Gathers high-priority manufacturing materials
                if colony_mats.get("iron_ore", 0) < 20:
                    res = "iron_ore"
                elif colony_mats.get("silica_sand", 0) < 15:
                    res = "silica_sand"
                elif colony_mats.get("graphite", 0) < 25:
                    res = "graphite"
                else:
                    res = "iron_ore"
                candidates.append({
                    "action": "gather",
                    "target": {"resource": res},
                    "reasoning": f"{agent.name} (Commander) extracting {res} for Colony Score infrastructure",
                    "deterministic": True
                })

            # === TOOL REPAIR AT FORGE CANDIDATE ===
            has_forge = any(s.get("type") == "forge" for s in getattr(self, "placed_structures", []))
            if has_forge and hasattr(agent.inventory, 'tool_durability'):
                for t_name, dur in list(agent.inventory.tool_durability.items()):
                    if dur < 35 and colony_mats.get("reduced_iron_ingot", 0) >= 1:
                        candidates.append({
                            "action": "repair_tool",
                            "target": {"tool": t_name},
                            "reasoning": f"{agent.name} restoring worn {t_name} at Forge (durability {dur}%)",
                            "deterministic": True
                        })
                        break

            # === RESCUE INCAPACITATED / CRITICAL TEAMMATE CANDIDATE ===
            for other in getattr(self, "agents", []):
                if other.id != agent.id and getattr(other.status, 'value', str(other.status)) != 'dead':
                    is_critical = (
                        other.needs.temperature_stress <= 15 or
                        other.needs.o2_supply <= 15 or
                        other.injury_level >= 0.7 or
                        getattr(other.status, 'value', str(other.status)) in ('incapacitated', 'critical')
                    )
                    if is_critical and not other._in_habitat:
                        dist = max(abs(other.x - agent.x), abs(other.y - agent.y))
                        if dist <= 12:
                            candidates.append({
                                "action": "rescue",
                                "target": {"victim_id": other.id, "victim_name": other.name, "x": other.x, "y": other.y},
                                "reasoning": f"SEARCH & RESCUE: {agent.name} rushing to evacuate critical colonist {other.name} to Base Airlock",
                                "deterministic": True
                            })
                            break

            # === SCAVENGE DECEASED CREW INVENTORY CANDIDATE ===
            for other in getattr(self, "agents", []):
                if getattr(other.status, 'value', str(other.status)) == 'dead' and not getattr(other, "corpse_salvaged", False):
                    has_valuable_loot = bool(other.inventory.items or other.inventory.materials)
                    if has_valuable_loot:
                        candidates.append({
                            "action": "scavenge_corpse",
                            "target": {"victim_id": other.id, "victim_name": other.name, "x": other.x, "y": other.y},
                            "reasoning": f"SALVAGE EXPEDITION: {agent.name} retrieving survival gear & minerals from fallen astronaut {other.name} at ({other.x}, {other.y})",
                            "deterministic": True
                        })
                        break

            # === SHELTER & AIRLOCK RL CANDIDATES ===
            # Returning to shelter is a survival/logistics decision, not a
            # generic action to explore randomly. Offering it every outdoor
            # tick made fresh RL policies oscillate between a work site and the
            # airlock without completing either task.
            if not getattr(agent, "_in_habitat", False):
                lz_x = getattr(self, "lz_x", getattr(agent, "spawn_x", 1000))
                lz_y = getattr(self, "lz_y", getattr(agent, "spawn_y", 1000))
                shelter_x, shelter_y = lz_x, lz_y
                min_shelter_dist = max(abs(agent.x - lz_x), abs(agent.y - lz_y))
                for s in getattr(self, "placed_structures", []):
                    if s.get("type") == "habitat_module" and not s.get("under_construction", False) and not s.get("destroyed", False):
                        sx, sy = s.get("x", lz_x), s.get("y", lz_y)
                        dist = max(abs(agent.x - sx), abs(agent.y - sy))
                        if dist < min_shelter_dist:
                            min_shelter_dist = dist
                            shelter_x, shelter_y = sx, sy

                needs_recovery = (
                    agent.needs.energy < 45.0
                    or agent.needs.o2_supply < 45.0
                    or agent.needs.temperature_stress < 35.0
                    or getattr(agent, "_eva_ticks_continuous", 0)
                    >= int(getattr(
                        self,
                        "max_continuous_eva_ticks",
                        ticks_for_minutes(360.0, self.tick_minutes),
                    ))
                )
                carrying_payload = total_items >= 4 or cur_weight >= 8.0
                if needs_recovery or carrying_payload:
                    if min_shelter_dist <= 1:
                        candidates.append({
                            "action": "enter_habitat",
                            "target": {},
                            "reasoning": f"{agent.name} cycling airlock for recovery/resupply",
                            "deterministic": True
                        })
                    else:
                        dx = 1 if shelter_x > agent.x else (-1 if shelter_x < agent.x else 0)
                        dy = 1 if shelter_y > agent.y else (-1 if shelter_y < agent.y else 0)
                        candidates.append({
                            "action": "move",
                            "target": {"dx": dx, "dy": dy, "x": shelter_x, "y": shelter_y, "destination": "shelter"},
                            "reasoning": f"{agent.name} returning towards base shelter for recovery/resupply",
                            "deterministic": True
                        })

            # === CENTRAL DEPOT LOGISTICS CANDIDATE ===
            # If agent is carrying >25kg or >5 items of gathered raw ores, deposit to Central Silo
            carried_mats = sum(agent.inventory.materials.values())
            if carried_mats >= 4:
                candidates.append({
                    "action": "deposit_materials",
                    "target": {},
                    "reasoning": f"{agent.name} hauling {carried_mats}x materials to Central Storage Depot Silo",
                    "deterministic": True
                })

            # === RECYCLE SCRAP METAL CANDIDATE ===
            if agent.inventory.materials.get("scrap_metal", 0) >= 2:
                candidates.append({
                    "action": "recycle_scrap",
                    "target": {},
                    "reasoning": f"{agent.name} recycling salvaged scrap metal into refined metal ingots in furnace",
                    "deterministic": True
                })

            # Add fallback exploration candidate if no other candidates
            if not candidates:
                candidates.append({"action": "explore", "target": {}, "reasoning": f"{agent.name} exploring terrain", "deterministic": True})

            # Re-check legacy free-choice fabrication candidates at the
            # physical boundary immediately before RL selection.  A high
            # Q-value cannot authorize an occupied machine, duplicate a BOM
            # already reserved by another route, or use a remote inventory.
            living = [
                crew for crew in getattr(self, "agents", [agent])
                if getattr(crew.status, "value", str(crew.status))
                not in {"dead", "incapacitated"}
            ]
            candidates = [
                candidate for candidate in candidates
                if candidate.get("action") != "refine"
                or self._refine_candidate_is_executable(
                    candidate, agent, living, structures_built, tick
                )
            ]
            if not candidates:
                candidates.append({
                    "action": "stand_watch",
                    "target": {
                        "operations_watch": True,
                        "ticks": ticks_for_minutes(60.0, self.tick_minutes),
                        "reason": "no_currently_executable_shop_or_field_order",
                    },
                    "reasoning": (
                        f"{agent.name} reviewing the work queue while physical "
                        "shop slots and staged inputs are committed"
                    ),
                    "deterministic": True,
                })

            # Select best action via RL Q-table policy with exploration
            state_key = self.get_rl_state_key(agent, structures_built, colony_mats, world_context)
            decision = self.select_action_via_rl(agent, state_key, candidates)
            self._deterministic_ticks += 1
            
        return decision
    
    def _run_strategic(self, agent: Agent, tick: int,
                       world_context: dict):
        """Run strategic LLM evaluation."""
        # Get relevant memories
        query = f"colony building, resource gathering, {self.colony.get_weakest_category()[0]}"
        relevant_memories = self.memory.get_context_for_prompt(
            agent.id, query, tick, top_k=3
        )
        
        # Get insights
        insights = self.reflection.get_insights_for_agent(agent.id)[-3:]
        
        # Get detailed team inventory & status summary
        agents_list = getattr(self, "agents", [agent])
        agent_names = {a.id: a.name for a in agents_list}
        team_parts = []
        colony_mats = {}
        for other in agents_list:
            if other.status != getattr(other.status, 'DEAD', 'dead') and getattr(other.status, 'value', str(other.status)) != 'dead':
                mats_str = ", ".join(f"{v} {k}" for k, v in other.inventory.materials.items() if v > 0)
                if not mats_str: mats_str = "empty"
                team_parts.append(f"{other.name} ({other.status.value if hasattr(other.status, 'value') else other.status}, carrying: {mats_str})")
                for k, v in other.inventory.materials.items():
                    colony_mats[k] = colony_mats.get(k, 0) + v
        
        colony_stock_str = ", ".join(f"{v} {k}" for k, v in colony_mats.items() if v > 0) or "no materials yet"
        team_summary = f"{'; '.join(team_parts)} | COLONY COMBINED STOCK: {colony_stock_str}"
        
        # Build prompt
        system = self._system_prompts.get(agent.id, "")
        user = build_strategic_prompt(
            agent,
            colony_score=self.colony.get_scores(),
            world_summary=f"Biome: {world_context.get('biome', 'unknown')}, "
                         f"Temp: {world_context.get('temperature_c', 20)}C",
            recent_insights=insights,
            team_status=team_summary,
        )
        
        # Call LLM
        result = self.llm.call(
            system, user,
            call_type=LLMCallType.STRATEGIC,
            current_tick=tick,
            temperature=0.7,
            max_tokens=300,
        )
        
        self._strategic_calls += 1
        
        if result is None or not isinstance(result, dict) or not result.get("plan_steps"):
            primary_domain = agent.competency.get_primary_domain() if hasattr(agent, 'competency') else 'engineering'
            weakest_cat = "water"
            if hasattr(self, 'colony_score') and self.colony_score:
                weakest_cat, _ = self.colony_score.get_weakest_category()
                
            structures_built = getattr(self, "structures_built", {})
            wc_count = structures_built.get("water_collector", 0)
            wp_count = structures_built.get("water_purifier", 0)
            sp_count = structures_built.get("solar_panel", 0)
            gh_count = structures_built.get("greenhouse", 0)
            o2_count = structures_built.get("isru_o2_unit", 0)
            hab_count = structures_built.get("habitat_module", 0)
            furnace_count = structures_built.get("stone_furnace", 0)
            
            if primary_domain == 'engineering':
                if furnace_count == 0:
                    result = {
                        "action": "build",
                        "target": {"recipe": "stone_furnace"},
                        "goal": "Construct Stone Smelting Furnace for industrial metallurgy",
                        "plan_steps": ["Mine basalt and iron ore for furnace", "Construct Stone Smelting Furnace"],
                        "estimated_ticks": 10
                    }
                elif weakest_cat == "water" and wc_count == 0:
                    result = {
                        "action": "build",
                        "target": {"recipe": "water_collector"},
                        "goal": "Construct Thermal Condenser & Water Collector to secure colony water supply",
                        "plan_steps": ["Reduce graded iron ore into structural iron ingots", "Form metal pipes and qualified vacuum gaskets", "Construct Water Collector"],
                        "estimated_ticks": 14
                    }
                elif weakest_cat == "water" and wp_count < 4:
                    result = {
                        "action": "build",
                        "target": {"recipe": "water_purifier"},
                        "goal": "Expand the colony wastewater recovery train without treating purification as a water source",
                        "plan_steps": ["Fabricate pumps, pressure vessels, seals and controls", "Construct and commission Water Purification Unit"],
                        "estimated_ticks": 18
                    }
                elif weakest_cat == "energy" or sp_count == 0:
                    result = {
                        "action": "build",
                        "target": {"recipe": "solar_panel"},
                        "goal": "Deploy High-Efficiency Solar Panel Array to power ECLSS systems",
                        "plan_steps": ["Melt optical glass panes from silica sand", "Fabricate electronic inverter circuits", "Construct Solar Panel Array"],
                        "estimated_ticks": 14
                    }
                elif weakest_cat == "food" or gh_count == 0:
                    result = {
                        "action": "build",
                        "target": {"recipe": "greenhouse"},
                        "goal": "Construct Hydroponic Biosphere Greenhouse for sustainable food production",
                        "plan_steps": ["Press composite panels and glass for greenhouse", "Construct Hydroponic Greenhouse"],
                        "estimated_ticks": 16
                    }
                elif weakest_cat == "o2" or o2_count == 0:
                    result = {
                        "action": "build",
                        "target": {"recipe": "isru_o2_unit"},
                        "goal": "Construct In-Situ O2 Generation Reactor for atmospheric replenishment",
                        "plan_steps": ["Reduce iron ore and form proof-tested metal pipes", "Construct ISRU O2 Unit"],
                        "estimated_ticks": 16
                    }
                else:
                    result = {
                        "action": "build",
                        "target": {"recipe": "habitat_module"},
                        "goal": "Construct Pressurized Habitat Module to expand shirtsleeve living space",
                        "plan_steps": ["Fabricate aerospace structural beams", "Construct Habitat Module"],
                        "estimated_ticks": 20
                    }
            elif primary_domain == 'physics':
                # Mining Specialist
                result = {
                    "action": "gather",
                    "target": {"resource": "iron_ore"},
                    "goal": "Extract iron ore and basalt minerals to feed colony manufacturing pipeline",
                    "plan_steps": ["Mine high-grade iron ore deposits", "Extract basalt foundation blocks", "Haul raw materials to Central Storage Depot"],
                    "estimated_ticks": 10
                }
            elif primary_domain in ('botany_bio', 'medical'):
                # Life Support & Bio Specialist
                planet_resources = set(getattr(self, "planetary_resources", set()))
                res_target = (
                    "water_ice"
                    if weakest_cat == "water" and "water_ice" in planet_resources
                    else "silica_sand"
                )
                result = {
                    "action": "gather",
                    "target": {"resource": res_target},
                    "goal": f"Gather {res_target} for colony life-support construction",
                    "plan_steps": [f"Prospect and gather {res_target} deposits", "Deliver materials to Base Habitat"],
                    "estimated_ticks": 10
                }
            else:
                # Commander / Generalist
                result = {
                    "action": "gather",
                    "target": {"resource": "iron_ore"},
                    "goal": "Extract industrial minerals (iron ore, chalcopyrite, silica sand) for colony infrastructure",
                    "plan_steps": ["Mine high-grade iron ore deposits", "Deposit materials at Central Storage Depot", "Assist Chief Engineer in assembly"],
                    "estimated_ticks": 10
                }
        
        if hasattr(self, 'on_event') and self.on_event and result:
            self.on_event({
                "type": "llm_thought",
                "agent": agent.name,
                "call_type": "strategic",
                "tokens": result.get("_tokens_used", 120),
                "goal": result.get("goal", result.get("action", "explore")),
                "reasoning": result.get("reasoning", result.get("goal", "Strategic colony planning")),
                "plan_steps": result.get("plan_steps", []),
                "tick": tick
            })

        # Parse into strategic plan
        self._parse_strategic_plan(agent.id, result, tick)
    
    def _parse_strategic_plan(self, agent_id: str, result: dict, tick: int):
        """Parse LLM response into a StrategicPlan."""
        steps = []
        
        # Parse plan_steps from LLM
        raw_steps = result.get("plan_steps", [])
        if isinstance(raw_steps, list):
            for i, step_text in enumerate(raw_steps):
                if isinstance(step_text, str):
                    # Infer action and target from step text
                    action = self._infer_action(step_text)
                    step_target = self._infer_step_target(action, step_text, result.get("target", {}))
                    steps.append(PlanStep(
                        action=action,
                        target=step_target,
                        description=step_text,
                        estimated_ticks=5,  # Default
                    ))
        
        # If no steps parsed, create single step from action
        if not steps:
            steps.append(PlanStep(
                action=result.get("action", "explore"),
                target=result.get("target", {}),
                description=result.get("goal", result.get("reasoning", "colony task")),
                estimated_ticks=result.get("estimated_ticks", 10),
            ))
        
        self._plans[agent_id] = StrategicPlan(
            goal=result.get("goal", result.get("reasoning", "survival")),
            steps=steps,
            created_tick=tick,
            priority=result.get("priority", "normal"),
        )
    
    def _infer_action(self, step_text: str) -> str:
        """Infer action type from plan step text."""
        text = step_text.lower()
        if any(w in text for w in ["smelt", "refine", "forge", "extrude", "fabricate", "vulcanize", "press"]):
            return "refine"
        if any(w in text for w in ["build", "construct", "assemble", "erect", "deploy"]):
            return "build"
        if any(w in text for w in ["gather", "collect", "mine", "harvest", "extract", "prospect", "excavate", "scoop"]):
            return "gather"
        if any(w in text for w in ["deposit", "haul", "offload", "unload", "deliver", "transfer", "store"]):
            return "deposit_materials"
        if any(w in text for w in ["repair", "fix", "maintain", "service"]):
            return "repair"
        if any(w in text for w in ["move", "go", "walk", "travel", "head", "return", "retreat"]):
            return "move"
        if any(w in text for w in ["eat", "food", "ration"]):
            return "eat"
        if any(w in text for w in ["drink", "water", "hydrate"]):
            return "drink"
        if any(w in text for w in ["sleep", "rest", "nap", "bunk"]):
            return "sleep"
        if any(w in text for w in ["talk", "discuss", "communicate"]):
            return "talk"
        if any(w in text for w in ["explore", "search", "scout", "survey", "find", "scan"]):
            return "explore"
        return "explore"
        
    def _infer_step_target(self, action: str, step_text: str, default_target: dict) -> dict:
        """Infer specific recipe or resource target from plan step text."""
        text = step_text.lower()
        if action == "build":
            if "purifier" in text or "purification" in text or "recycl" in text:
                return {"recipe": "water_purifier"}
            if "water" in text or "condenser" in text:
                return {"recipe": "water_collector"}
            elif "solar" in text or "panel" in text:
                return {"recipe": "solar_panel"}
            elif "greenhouse" in text or "hydroponic" in text:
                return {"recipe": "greenhouse"}
            elif "isru" in text or "o2" in text or "oxygen" in text:
                return {"recipe": "isru_o2_unit"}
            elif "habitat" in text or "shelter" in text:
                return {"recipe": "habitat_module"}
            elif "furnace" in text or "smelter" in text:
                return {"recipe": "stone_furnace"}
            elif "forge" in text:
                return {"recipe": "forge"}
            return default_target if default_target else {"recipe": "water_collector"}
            
        elif action == "refine":
            if "pipe" in text:
                return {"output": "metal_pipe", "structure": "forge"}
            elif "glass" in text:
                return {"output": "glass_pane", "structure": "stone_furnace"}
            elif "gasket" in text or "seal" in text:
                return {"output": "vacuum_gasket_seal", "structure": "forge"}
            elif "bolt" in text or "fastener" in text:
                return {"output": "machined_bolts_fasteners", "structure": "forge"}
            elif "composite" in text:
                return {"output": "composite_panel", "structure": "cnc_fabricator"}
            elif "electronic" in text or "circuit" in text:
                return {"output": "electronic_component", "structure": "forge"}
            elif "heat" in text or "sink" in text:
                return {"output": "finned_heat_sink", "structure": "forge"}
            return {"output": "reduced_iron_ingot", "structure": "stone_furnace"}
            
        elif action == "gather":
            if "iron" in text:
                return {"resource": "iron_ore"}
            elif "basalt" in text:
                return {"resource": "basalt"}
            elif "silica" in text:
                return {"resource": "silica_sand"}
            elif "ice" in text or "water" in text:
                return {"resource": "water_ice"}
            elif "graphite" in text or "carbon" in text:
                return {"resource": "graphite"}
            elif "sulfur" in text:
                return {"resource": "sulfur"}
            elif "regolith" in text:
                return {"resource": "regolith"}
            elif "chalcopyrite" in text or "copper ore" in text:
                return {"resource": "chalcopyrite_ore"}
            return default_target if default_target else {"resource": "iron_ore"}
            
        return default_target
        
    def _run_tactical(self, agent: Agent, tick: int, trigger: str,
                      world_context: dict,
                      nearby_agents: list) -> dict:
        """Run tactical LLM decision."""
        system = self._system_prompts.get(agent.id, "")
        
        # Get relevant memories for context
        memories = self.memory.get_context_for_prompt(
            agent.id, trigger, tick, top_k=3
        )
        
        user = build_tactical_prompt(
            agent,
            trigger=trigger,
            world_context=world_context,
            nearby_agents=[
                {"name": a.get("name", "?"), "distance": a.get("distance", 0)}
                for a in nearby_agents[:3]
            ],
        )
        
        # Add memory context if available
        if memories:
            user += f"\nRELEVANT MEMORIES: {'; '.join(memories[:3])}"
        
        result = self.llm.call(
            system, user,
            call_type=LLMCallType.TACTICAL,
            current_tick=tick,
            temperature=0.6,  # Lower temp for tactical (more deterministic)
            max_tokens=200,
        )
        
        self._tactical_calls += 1
        
        if result is None or not isinstance(result, dict):
            # Context-Aware Safe Fallback: Prioritize self-preservation over blind exploration
            t_lower = trigger.lower()
            lz_x = getattr(self, "lz_x", 1000)
            lz_y = getattr(self, "lz_y", 1000)
            
            if any(w in t_lower for w in ["cold", "hypotherm", "temp", "freeze"]):
                safe_action = "move"
                dx = 1 if lz_x > agent.x else (-1 if lz_x < agent.x else 0)
                dy = 1 if lz_y > agent.y else (-1 if lz_y < agent.y else 0)
                target_dict = {"x": lz_x, "y": lz_y, "dx": dx, "dy": dy}
                reason = f"Emergency retreat to Base Hub for thermal re-warming ({trigger})"
            elif any(w in t_lower for w in ["thirst", "dehydrat", "water"]):
                safe_action = "drink"
                target_dict = {}
                reason = f"Emergency hydration from life support reserves ({trigger})"
            elif any(w in t_lower for w in ["hunger", "starv", "food"]):
                safe_action = "eat"
                target_dict = {}
                reason = f"Emergency caloric consumption ({trigger})"
            elif any(w in t_lower for w in ["o2", "suffocat", "asphyx"]):
                safe_action = "use_item"
                target_dict = {"item": "oxygen_canisters"}
                reason = f"Emergency O2 canister purge/reload ({trigger})"
            elif any(w in t_lower for w in ["storm", "flare", "hazard", "quake"]):
                safe_action = "move"
                dx = 1 if lz_x > agent.x else (-1 if lz_x < agent.x else 0)
                dy = 1 if lz_y > agent.y else (-1 if lz_y < agent.y else 0)
                target_dict = {"x": lz_x, "y": lz_y, "dx": dx, "dy": dy}
                reason = f"Hazard shelter evacuation to Base Hub ({trigger})"
            elif any(w in t_lower for w in ["exhaust", "tired", "sleep"]):
                safe_action = "sleep"
                target_dict = {}
                reason = f"Fatigue recovery rest in shelter ({trigger})"
            else:
                safe_action = "idle"
                target_dict = {}
                reason = f"Tactical stabilization standby ({trigger})"
            
            result = {
                "action": safe_action,
                "target": target_dict,
                "reasoning": reason,
                "priority": "critical"
            }
        
        if hasattr(self, 'on_event') and self.on_event and result:
            self.on_event({
                "type": "llm_thought",
                "agent": agent.name,
                "call_type": "tactical",
                "tokens": result.get("_tokens_used", 110),
                "goal": result.get("action", "tactical response"),
                "reasoning": result.get("reasoning", f"Tactical response to {trigger}"),
                "tick": tick
            })

        # If tactical decision overrides plan, update plan
        if result.get("priority") == "critical":
            # Override strategic plan with survival action
            self._plans[agent.id] = StrategicPlan(
                goal=f"Emergency: {trigger}",
                steps=[PlanStep(
                    action=result.get("action", "explore"),
                    target=result.get("target", {}),
                    description=result.get("reasoning", trigger),
                    estimated_ticks=1,
                )],
                created_tick=tick,
                priority="critical",
            )
        
        return result
    
    def _get_missing_materials_for_step(self, step: PlanStep, colony_mats: dict) -> list[str]:
        """Check if any raw or refined materials are missing for the plan step."""
        action = step.action
        target = step.target or {}
        
        if action == "refine":
            output = target.get("output", target.get("raw_material", "reduced_iron_ingot"))
            recipe = self._recipes_cache.get(output, {})
            cost = recipe.get("materials", {})
            if not recipe or recipe.get("output", {}).get("type") != "material":
                return [f"recipe:{output}"]
            missing = []
            for mat, needed in cost.items():
                if colony_mats.get(mat, 0) < needed:
                    if mat == "reduced_iron_ingot" and colony_mats.get("iron_ore", 0) < 4:
                        missing.append("iron_ore")
                    else:
                        missing.append(mat)
            return missing
            
        elif action == "build":
            recipe_name = target.get("recipe", "water_collector")
            recipe = self._recipes_cache.get(recipe_name, {})
            if not recipe:
                return [f"recipe:{recipe_name}"]
            cost = recipe.get("materials", {})
            missing = []
            for mat, needed in cost.items():
                if colony_mats.get(mat, 0) < needed:
                    if mat == "reduced_iron_ingot" and colony_mats.get("iron_ore", 0) < 4:
                        missing.append("iron_ore")
                    elif mat in ("metal_pipe", "machined_bolts_fasteners", "finned_heat_sink") and colony_mats.get("reduced_iron_ingot", 0) < 1 and colony_mats.get("iron_ore", 0) < 4:
                        missing.append("iron_ore")
                    elif mat == "glass_pane" and colony_mats.get("silica_sand", 0) < 6:
                        missing.append("silica_sand")
                    elif mat == "composite_panel" and colony_mats.get("qualified_composite_resin_pack", 0) < 6:
                        missing.append("qualified_composite_resin_pack")
                    else:
                        missing.append(mat)
            return missing
            
        return []

    def _execute_plan_step(self, agent: Agent,
                           plan: StrategicPlan) -> Optional[dict]:
        """Execute current plan step deterministically with material prerequisite checks."""
        step = plan.current_step
        if step is None:
            return {"action": "explore", "target": {}, "reasoning": "Plan complete, exploring planetary surface", "deterministic": True}

        # A motor rejection is not task progress and is not an RL outcome.
        # Release the visible one-tick failure state and let validated colony
        # candidates run during a short, state-conditioned retry interval.
        current_action = str(
            getattr(agent.action, "action_type", "") or ""
        ).lower()
        current_target = (
            agent.action.target
            if isinstance(getattr(agent.action, "target", None), dict)
            else {}
        )
        rejection = ""
        if current_action in {
            "craft_blocked", "recipe_unavailable",
            "tool_transfer_blocked", "invalid_action_rejected",
        }:
            rejection = current_action
        elif current_target.get("recipe_unavailable"):
            rejection = "recipe_unavailable"
        elif current_target.get("invalid_action_rejected"):
            rejection = str(current_target.get("reason", "invalid_action"))
        if rejection:
            step.last_rejection = rejection
            current_tick = int(getattr(self, "current_tick", 0))
            # The plan produced a command that the physical layer rejected.
            # Invalidate it instead of treating waiting as a reason to repeat
            # the same command. This is a coordination reset, not an RL loss.
            self._plans.pop(agent.id, None)
            self._plan_retry_after_tick[agent.id] = current_tick + 6
            agent.action.action_type = "idle"
            agent.action.target = {
                "plan_replan_after_rejection": rejection,
            }
            agent.action.ticks_remaining = 0
            return None
        if int(getattr(self, "current_tick", 0)) <= int(
            step.blocked_until_tick
        ):
            return None
        step.blocked_until_tick = -1

        # Complete a stale logistics step once its payload is gone. Otherwise
        # a persistent strategic plan can issue deposit_materials forever.
        if (
            step.action == "deposit_materials"
            and not any(qty > 0 for qty in agent.inventory.materials.values())
        ):
            step.completed = True
            return self._execute_plan_step(agent, plan)

        # A strategic plan can outlive its requested capacity. Scalable
        # infrastructure remains valid until its configured target count is
        # reached; only genuinely singleton storage stops after one instance.
        if step.action == "build":
            recipe_name = (step.target or {}).get("recipe")
            target_count = (step.target or {}).get("capacity_target_count")
            if (
                (recipe_name == "storage_crate" and getattr(
                    self, "structures_built", {}
                ).get(recipe_name, 0) > 0)
                or (
                    target_count is not None
                    and getattr(self, "structures_built", {}).get(recipe_name, 0)
                    >= int(target_count)
                )
            ):
                step.completed = True
                return self._execute_plan_step(agent, plan)
        
        # Only genuinely productive active work advances a plan step.
        if agent.action.is_active and getattr(agent.action, 'action_type', '') not in ('idle', 'rest', None, ''):
            step.ticks_spent += 1
            if step.ticks_spent >= step.estimated_ticks:
                step.completed = True
            return {
                "action": "continue",
                "target": step.target,
                "reasoning": f"Executing: {step.description}",
                "deterministic": True,
                "plan_progress": plan.progress,
            }
        
        # Calculate pooled colony materials across agent, teammates, and central depot
        colony_mats = dict(getattr(self, "central_depot_inventory", {}))
        for other in getattr(self, "agents", [agent]):
            if getattr(other.status, 'value', str(other.status)) != 'dead':
                for k, v in other.inventory.materials.items():
                    colony_mats[k] = colony_mats.get(k, 0) + v
        
        # Check if materials are missing for this plan step
        missing_mats = self._get_missing_materials_for_step(step, colony_mats)
        
        if missing_mats:
            # Plan step is blocked by missing materials!
            primary = agent.competency.get_primary_domain() if hasattr(agent, 'competency') else 'engineering'
            
            # If Chief Engineer, fall through to candidates to immediately build ready structures or refine available materials
            if primary == 'engineering':
                return None
                
            raw_resources = {"iron_ore", "silica_sand", "basalt", "water_ice", "graphite", "sulfur", "chalcopyrite_ore", "calcite", "food"}
            missing_raw = next((m for m in missing_mats if m in raw_resources), None)
            
            # Check carry weight before gathering
            from src.agents.agent import MATERIAL_DENSITY_KG
            cur_weight = sum(
                qty * MATERIAL_DENSITY_KG.get(mat, 2.0)
                for mat, qty in agent.inventory.materials.items()
            )
            gravity_g = getattr(agent, "_gravity_g", 1.0)
            carry_cap = agent.inventory.BASE_CARRY_CAPACITY_KG * (1 + agent.genome.strength * 0.1) / max(0.3, gravity_g)
            
            if missing_raw and cur_weight < carry_cap - 2.0:
                logger.info(f"{agent.name}'s plan step '{step.description}' needs {missing_raw} — dynamically gathering prerequisite")
                return {
                    "action": "gather",
                    "target": {"resource": missing_raw},
                    "reasoning": f"{agent.name} gathering {missing_raw} required for plan: {step.description}",
                    "deterministic": True,
                    "plan_progress": plan.progress,
                }
            elif cur_weight >= 6.0 or sum(agent.inventory.materials.values()) >= 4:
                # Carry load substantial — deposit to central depot
                lz_x = getattr(self, "lz_x", 1000)
                lz_y = getattr(self, "lz_y", 1000)
                return {
                    "action": "deposit_materials",
                    "target": {"x": lz_x, "y": lz_y},
                    "reasoning": f"{agent.name} offloading gathered materials ({cur_weight:.1f}kg) to Central Depot",
                    "deterministic": True,
                    "plan_progress": plan.progress,
                }
            else:
                # Cannot gather this raw material directly, fall back to smart domain candidates
                return None
        
        # Step has all required materials or is a non-material action
        step.ticks_spent += 1
        if step.ticks_spent >= step.estimated_ticks:
            step.completed = True
            
        return {
            "action": step.action,
            "target": step.target,
            "reasoning": step.description,
            "deterministic": True,
            "plan_progress": plan.progress,
        }
    
    def _run_reflection(self, agent: Agent, tick: int):
        """Run reflection cycle."""
        if not self.dialogue_generation_enabled:
            return
        # Skip reflection if LLM pool is exhausted to save daily API limits
        if self.llm.all_exhausted:
            return
        insights = self.reflection.reflect(agent.id, agent.name, tick)
        if insights:
            logger.debug(f"{agent.name} reflected: {insights}")
            if hasattr(self, 'on_event') and self.on_event:
                for ins in insights:
                    ins_text = ins if isinstance(ins, str) else getattr(ins, 'text', str(ins))
                    self.on_event({
                        "type": "llm_reflection",
                        "agent": agent.name,
                        "insight": ins_text,
                        "tick": tick
                    })
    
    # ================================================================
    # SOCIAL INTERACTIONS
    # ================================================================
    
    def handle_encounter(self, agent: Agent, other_agent: Agent,
                         tick: int) -> Optional[dict]:
        """Handle two agents encountering each other."""
        if not self.dialogue_generation_enabled:
            return None
        trust = self.social.get_trust(agent.id, other_agent.id)
        
        system = self._system_prompts.get(agent.id, "")
        user = build_social_prompt(
            agent, other_agent.name,
            context="general encounter during colony work",
            trust_level=trust,
        )
        
        result = self.llm.call(
            system, user,
            call_type=LLMCallType.SOCIAL,
            current_tick=tick,
            temperature=0.9,  # More creative for dialogue
            max_tokens=150,
        )
        
        if result and result.get("dialogue"):
            # Emit dialogue event for frontend LLM stream
            if hasattr(self, 'on_event') and self.on_event:
                self.on_event({
                    "type": "llm_dialogue",
                    "agent": agent.name,
                    "target_agent": other_agent.name,
                    "dialogue": result.get("dialogue"),
                    "social_action": result.get("social_action", "talk"),
                    "tokens": result.get("_tokens_used", 90),
                    "tick": tick
                })
            
            # Dialogue is telemetry/narrative only.  Neither generated wording
            # nor a generated ``social_action`` may change trust, memory,
            # morale, inventories or any other simulation state.
            result["non_authoritative"] = True
        
        return result
    
    # ================================================================
    # REINFORCEMENT LEARNING (RL) COGNITIVE VALUE POLICY
    # ================================================================
    
    def get_rl_state_key(self, agent: Agent, structures_built: dict, colony_mats: dict, world_context: dict = None) -> str:
        """Discretize continuous simulation state into a structured RL State key."""
        agent.ensure_tactical_policy_schema()
        primary = agent.competency.get_primary_domain() if hasattr(agent, 'competency') else 'general'
        
        # 1. Distance zone to nearest operational shelter airlock
        lz_x = getattr(self, "lz_x", getattr(agent, "spawn_x", 1000))
        lz_y = getattr(self, "lz_y", getattr(agent, "spawn_y", 1000))
        min_shelter_dist = max(abs(agent.x - lz_x), abs(agent.y - lz_y))
        for s in getattr(self, "placed_structures", []):
            if s.get("type") == "habitat_module" and not s.get("under_construction", False) and not s.get("destroyed", False):
                sx, sy = s.get("x", lz_x), s.get("y", lz_y)
                dist = max(abs(agent.x - sx), abs(agent.y - sy))
                if dist < min_shelter_dist:
                    min_shelter_dist = dist
                    
        if getattr(agent, "_in_habitat", False):
            dist_zone = "in_hab"
        elif min_shelter_dist <= 1:
            dist_zone = "at_airlock"
        elif min_shelter_dist <= 6:
            dist_zone = "dist_near"
        elif min_shelter_dist <= 14:
            dist_zone = "dist_mid"
        else:
            dist_zone = "dist_far"
            
        # 2. Thermal state zone
        temp = getattr(agent.needs, "temperature_stress", 50.0)
        if temp < 30:
            therm_state = "freezing"
        elif temp < 45:
            therm_state = "chilled"
        elif temp > 65:
            therm_state = "overheating"
        else:
            therm_state = "nominal"
            
        # 3. Day / Night / Weather phase
        dn_phase = "day"
        if world_context and isinstance(world_context, dict):
            dn_phase = world_context.get("day_night_phase", "day")
            
        # 4. Critical need. O2 and severe dehydration must be visible to the
        # policy as consequences, not converted into hard-coded EVA locks.
        if agent.needs.o2_supply < 15:
            need_state = "o2_critical"
        elif agent.needs.thirst <= 0:
            need_state = "dehydrated"
        elif agent.needs.thirst < 40:
            need_state = "thirsty"
        elif agent.needs.hunger < 40:
            need_state = "hungry"
        elif agent.needs.energy < 35:
            need_state = "exhausted"
        else:
            need_state = "nominal"
            
        # 5. Shared life-support reserves. These categories let the same crew
        # learn that an otherwise sensible EVA/build action is dangerous when
        # the settlement has no recovery buffer.
        colony_resources = {}
        if world_context and isinstance(world_context, dict):
            colony_resources = world_context.get("colony_resources", {}) or {}

        def reserve_zone(value: float, critical: float, low: float) -> str:
            if value <= 0.0:
                return "empty"
            if value < critical:
                return "critical"
            if value < low:
                return "low"
            return "buffered"

        water_zone = reserve_zone(
            float(colony_resources.get("water_reserve_l", 0.0)), 10.0, 30.0
        )
        o2_zone = reserve_zone(
            float(colony_resources.get("o2_reserve_kg", 0.0)), 3.0, 10.0
        )

        # 6. Mission horizon and the shared physical objective. These values
        # are included only when the engine exposes them; no deadline or goal
        # is invented by the tactical policy.
        context = world_context if isinstance(world_context, dict) else {}
        max_ticks = context.get("max_ticks", self.mission_attempt_max_ticks)
        ticks_remaining = context.get("ticks_remaining")
        current_tick = context.get(
            "current_tick", getattr(self, "current_tick", None)
        )
        if (
            ticks_remaining is None
            and max_ticks is not None
            and current_tick is not None
        ):
            ticks_remaining = max(0, int(max_ticks) - int(current_tick))
        if ticks_remaining is None or not max_ticks:
            time_zone = "unknown"
        else:
            fraction = max(0.0, min(
                1.0, float(ticks_remaining) / float(max_ticks)
            ))
            if fraction > 0.75:
                time_zone = "early"
            elif fraction > 0.40:
                time_zone = "mid"
            elif fraction > 0.15:
                time_zone = "late"
            else:
                time_zone = "final"

        objective = context.get("active_objective")
        if isinstance(objective, dict):
            objective = objective.get("recipe") or objective.get("type")
        if not objective:
            policy = getattr(self, "strategic_policy", None)
            commitment = getattr(policy, "commitment", None)
            objective = getattr(commitment, "recipe", None)
        objective_zone = str(objective or "none").replace("|", "_")

        return (
            f"role:{primary}|dist:{dist_zone}|temp:{therm_state}|dn:{dn_phase}"
            f"|need:{need_state}|water:{water_zone}|o2:{o2_zone}"
            f"|time:{time_zone}|objective:{objective_zone}"
        )

    def _get_action_key(self, cand: dict) -> str:
        """Extract canonical action key string for Q-table indexing."""
        act = cand.get("action", "idle")
        tgt = cand.get("target", {})
        if isinstance(tgt, dict):
            tgt_param = (
                tgt.get("resource") or tgt.get("destination") or tgt.get("recipe") or
                tgt.get("output") or tgt.get("structure") or tgt.get("patient") or
                tgt.get("tool") or ""
            )
        else:
            tgt_param = str(tgt)
        return f"{act}:{tgt_param}" if tgt_param else act

    def select_action_via_rl(self, agent: Agent, state_key: str, candidate_actions: list[dict]) -> dict:
        """Select an action using Q-Learning policy with epsilon-greedy exploration."""
        if not candidate_actions:
            return {"action": "idle", "target": {}, "reasoning": "No candidate actions available", "deterministic": True}
            
        if state_key not in agent.q_table:
            agent.q_table[state_key] = {}
            
        # Ensure all candidate actions have an initial Q-value entry
        for cand in candidate_actions:
            act_key = self._get_action_key(cand)
            if act_key not in agent.q_table[state_key]:
                agent.q_table[state_key][act_key] = 1.0  # Optimistic initial value for exploration
                
        import random
        policy_rng = getattr(agent, "_policy_rng", None)
        if policy_rng is None:
            # Standalone unit-test agents have no engine seed; retain a local
            # stable stream rather than consulting process-global randomness.
            policy_rng = random.Random(sum(map(ord, agent.id)))
            agent._policy_rng = policy_rng
        # Epsilon-greedy exploration
        if policy_rng.random() < agent.rl_epsilon_explore:
            chosen = policy_rng.choice(candidate_actions)
            act_key = self._get_action_key(chosen)
            agent.last_state_key = state_key
            agent.last_action_key = act_key
            agent._rl_transition_pending = True
            trace = getattr(agent, "rl_episode_trace", None)
            if trace is None:
                trace = []
                agent.rl_episode_trace = trace
            transition = (state_key, act_key)
            if not trace or trace[-1] != transition:
                trace.append(transition)
                if len(trace) > 128:
                    del trace[:-128]
            chosen["reasoning"] = f"[RL EXPLORE ε={agent.rl_epsilon_explore:.2f}] Testing {act_key} (Q={agent.q_table[state_key].get(act_key, 0.0):.2f})"
            return chosen
            
        # Exploitation: Select action with maximum Q-value
        best_cand = candidate_actions[0]
        best_q = -999999.0
        for cand in candidate_actions:
            act_key = self._get_action_key(cand)
            q_val = agent.q_table[state_key].get(act_key, 0.0)
            if q_val > best_q:
                best_q = q_val
                best_cand = cand
                
        act_key = self._get_action_key(best_cand)
        agent.last_state_key = state_key
        agent.last_action_key = act_key
        agent._rl_transition_pending = True
        trace = getattr(agent, "rl_episode_trace", None)
        if trace is None:
            trace = []
            agent.rl_episode_trace = trace
        transition = (state_key, act_key)
        if not trace or trace[-1] != transition:
            trace.append(transition)
            if len(trace) > 128:
                del trace[:-128]
        best_cand["reasoning"] = f"[RL POLICY] Executing optimal {act_key} (Q-Value: {best_q:+.2f})"
        return best_cand

    def select_auxiliary_action_via_rl(
        self,
        agent: Agent,
        state_key: str,
        candidate_actions: list[dict],
        *,
        initial_q: float = 1.0,
    ) -> dict:
        """Choose a bounded sub-action without replacing the main RL transition.

        Navigation pace is an asynchronous choice whose outcome is known only
        when the route ends. Reusing ``select_action_via_rl`` here would
        overwrite the construction/gather decision awaiting its normal reward.
        This selector shares the persisted per-agent Q-table and epsilon but
        leaves ``last_state_key``, ``last_action_key`` and the episode trace
        untouched; the engine later applies the measured route reward through
        ``apply_delayed_rl_reward``.
        """
        if not candidate_actions:
            return {
                "action": "idle", "target": {},
                "reasoning": "No auxiliary action available",
                "deterministic": True,
            }

        actions = agent.q_table.setdefault(state_key, {})
        for candidate in candidate_actions:
            actions.setdefault(
                self._get_action_key(candidate), float(initial_q)
            )

        import random
        policy_rng = getattr(agent, "_policy_rng", None)
        if policy_rng is None:
            policy_rng = random.Random(sum(map(ord, agent.id)))
            agent._policy_rng = policy_rng

        exploring = policy_rng.random() < agent.rl_epsilon_explore
        if exploring:
            selected = policy_rng.choice(candidate_actions)
        else:
            best_q = max(
                actions[self._get_action_key(candidate)]
                for candidate in candidate_actions
            )
            tied = [
                candidate for candidate in candidate_actions
                if actions[self._get_action_key(candidate)] == best_q
            ]
            selected = policy_rng.choice(tied)

        result = {
            **selected,
            "target": dict(selected.get("target", {})),
        }
        action_key = self._get_action_key(result)
        result["auxiliary_state_key"] = state_key
        result["auxiliary_action_key"] = action_key
        result["auxiliary_exploring"] = exploring
        result["reasoning"] = (
            f"[AUX RL {'EXPLORE' if exploring else 'POLICY'}] "
            f"{action_key} (Q={actions[action_key]:+.2f})"
        )
        return result

    def _record_rl_reward(
        self,
        agent: Agent,
        reward: float,
        reason: str,
        *,
        state_key: Optional[str] = None,
        action_key: Optional[str] = None,
        q_before: Optional[float] = None,
        q_after: Optional[float] = None,
        components: Optional[dict] = None,
    ) -> None:
        history = getattr(agent, "rl_reward_history", [])
        history.append({
            "tick": int(getattr(agent, "_telemetry_tick", agent.ticks_alive)),
            "reward": round(float(reward), 4),
            "reason": str(reason),
            "state_key": state_key,
            "action": action_key,
            "q_before": q_before,
            "q_after": q_after,
            "components": dict(components or {}),
            "source": "reinforcement_learning",
        })
        agent.rl_reward_history = history[-120:]

    def apply_rl_reward(
        self,
        agent: Agent,
        reward: float,
        next_state_key: str,
        reason: Optional[str] = None,
        components: Optional[dict] = None,
    ):
        """Apply Bellman Temporal Difference (TD) Q-table update."""
        s = agent.last_state_key
        a = agent.last_action_key
        if not s or not a:
            return
            
        agent.total_accumulated_reward += reward
        
        if s not in agent.q_table:
            agent.q_table[s] = {}
        if a not in agent.q_table[s]:
            agent.q_table[s][a] = 0.0
            
        old_q = agent.q_table[s][a]
        
        # Max future Q in next state
        next_qs = agent.q_table.get(next_state_key, {})
        max_next_q = max(next_qs.values()) if next_qs else 0.0
        
        # TD Bellman update
        alpha = agent.rl_learning_rate
        gamma = agent.rl_discount_factor
        td_target = reward + gamma * max_next_q
        new_q = old_q + alpha * (td_target - old_q)
        agent.q_table[s][a] = round(new_q, 3)
        self._record_rl_reward(
            agent, reward, reason or next_state_key,
            state_key=s, action_key=a,
            q_before=round(old_q, 3), q_after=agent.q_table[s][a],
            components=components,
        )

    def apply_delayed_rl_reward(
        self,
        agent: Agent,
        state_key: str,
        action_key: str,
        reward: float,
    ) -> bool:
        """Credit a completed asynchronous physical job to its dispatcher."""
        if not state_key or not action_key:
            return False
        actions = agent.q_table.setdefault(state_key, {})
        old_q = float(actions.get(action_key, 0.0))
        alpha = float(agent.rl_learning_rate)
        # The delivered payload is a completed outcome, so there is no
        # speculative bootstrap term at dispatch time.
        actions[action_key] = round(old_q + alpha * (reward - old_q), 3)
        agent.total_accumulated_reward += float(reward)
        self._record_rl_reward(
            agent, reward, "completed physical job",
            state_key=state_key, action_key=action_key,
            q_before=round(old_q, 3), q_after=actions[action_key],
        )
        return True

    def apply_terminal_rl_reward(
        self,
        agent: Agent,
        reward: float,
        decay: float = 0.92,
        max_transitions: int = 64,
    ) -> int:
        """Back-propagate an episode outcome through recent RL decisions."""
        trace = list(getattr(agent, "rl_episode_trace", []) or [])
        if not trace:
            return 0

        agent.total_accumulated_reward += reward
        alpha = agent.rl_learning_rate
        credit = float(reward)
        updated = 0
        seen: set[tuple[str, str]] = set()
        for state_key, action_key in reversed(trace[-max_transitions:]):
            transition = (state_key, action_key)
            if transition in seen:
                continue
            seen.add(transition)
            actions = agent.q_table.setdefault(state_key, {})
            old_q = float(actions.get(action_key, 0.0))
            # Terminal transitions have no bootstrap value.
            actions[action_key] = round(old_q + alpha * (credit - old_q), 3)
            credit *= decay
            updated += 1

        agent.rl_episode_trace.clear()
        agent._rl_transition_pending = False
        self._record_rl_reward(agent, reward, "episode outcome")
        return updated

    # ================================================================
    # STATS & INFO
    # ================================================================
    
    def get_agent_plan(self, agent_id: str) -> Optional[dict]:
        """Get current plan for an agent."""
        plan = self._plans.get(agent_id)
        if plan is None:
            return None
        return {
            "goal": plan.goal,
            "progress": plan.progress,
            "current_step": plan.current_step.description if plan.current_step else "complete",
            "priority": plan.priority,
        }
    
    def get_stats(self) -> dict:
        """Get decision engine statistics."""
        return {
            "strategic_calls": self._strategic_calls,
            "tactical_calls": self._tactical_calls,
            "deterministic_ticks": self._deterministic_ticks,
            "fallback_decisions": self._fallback_decisions,
            "llm_call_rate": round(
                (self._strategic_calls + self._tactical_calls) / 
                max(1, self._deterministic_ticks + self._strategic_calls + self._tactical_calls),
                3
            ),
            "active_plans": len([p for p in self._plans.values() if not p.is_complete]),
            "memory": self.memory.get_all_stats(),
            "reflection": self.reflection.get_stats(),
        }
