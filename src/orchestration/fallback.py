"""
Deterministic Fallback Decision Engine.

This engine makes rule-based decisions so the simulation remains independent
from the optional local narrative layer.

Decision priority (based on survival psychology):
1. Immediate survival threats (O2, thirst, hunger, hypothermia)
2. Continue current strategic task
3. Address degrading needs before they become critical
4. Social interaction if nearby agents
5. Explore if nothing else to do
"""

import random
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FallbackDecisionEngine:
    """
    Rule-based decision engine for simulation actions.
    
    Mimics rational agent behavior using simple priority rules.
    Output format matches LLM JSON schema exactly.
    """
    
    # Need thresholds
    CRITICAL_THRESHOLD = 20.0   # Below this = life-threatening
    WARNING_THRESHOLD = 40.0    # Below this = should address soon
    
    def make_decision(self, agent, world_context: dict = None,
                      colony_score: dict = None) -> dict:
        """
        Generate a deterministic decision for an agent.
        
        Args:
            agent: Agent instance with needs, inventory, position, etc.
            world_context: Environmental data (biome, resources nearby, etc.)
            colony_score: Current colony readiness breakdown
            
        Returns:
            dict matching LLM output format:
            {"action": str, "target": dict, "reasoning": str, "priority": str}
        """
        if world_context is None:
            world_context = {}
        if colony_score is None:
            colony_score = {}
        
        needs = agent.needs
        
        # === PRIORITY 1: IMMEDIATE SURVIVAL ===
        
        # O2 critical (atmosphereless planets)
        if needs.o2_supply < self.CRITICAL_THRESHOLD:
            if agent.inventory.has_item("oxygen_canisters"):
                return self._action("use_item", 
                    target={"item": "oxygen_canisters"},
                    reasoning="O2 critically low, loading canister",
                    priority="critical")
            else:
                # Must return to habitat or find canisters
                return self._action("move",
                    target={"destination": "habitat", "reason": "o2_emergency"},
                    reasoning="O2 depleting, no canisters — returning to habitat",
                    priority="critical")
        
        # Dying of thirst
        if needs.thirst < self.CRITICAL_THRESHOLD:
            if agent.inventory.has_item("water_packs"):
                return self._action("drink",
                    target={"source": "water_packs"},
                    reasoning="Dehydration imminent, drinking water",
                    priority="critical")
            else:
                return self._action("gather",
                    target={"resource": "water_ice"},
                    reasoning="No water, must find water ice urgently",
                    priority="critical")
        
        # Dying of hunger
        if needs.hunger < self.CRITICAL_THRESHOLD:
            if agent.inventory.has_item("emergency_rations"):
                return self._action("eat",
                    target={"source": "emergency_rations"},
                    reasoning="Starvation imminent, eating rations",
                    priority="critical")
            else:
                return self._action("gather",
                    target={"resource": "food"},
                    reasoning="No food available, foraging",
                    priority="critical")
        
        # Freezing to death
        if needs.temperature_stress < self.CRITICAL_THRESHOLD:
            if any(s.get("type") == "basic_shelter" for s in world_context.get("nearby_structures", [])):
                return self._action("move",
                    target={"destination": "shelter"},
                    reasoning="Hypothermia risk, moving to shelter",
                    priority="critical")
            else:
                # Return to landing habitat if no built shelter nearby
                return self._action("move",
                    target={"destination": "habitat"},
                    reasoning="Hypothermia risk, returning to pressurized landing habitat",
                    priority="critical")
        
        # Exhaustion
        if needs.energy < self.CRITICAL_THRESHOLD:
            return self._action("sleep",
                target={"location": "current"},
                reasoning="Exhaustion imminent, must rest",
                priority="critical")
        
        # === PRIORITY 2: CONTINUE CURRENT TASK ===
        if agent.action.is_active:
            return self._action("continue",
                target={"current_action": agent.action.action_type,
                        "ticks_remaining": agent.action.ticks_remaining},
                reasoning="Continuing current task",
                priority="normal")
        
        # === PRIORITY 3: ADDRESS WARNING-LEVEL NEEDS ===
        
        if needs.thirst < self.WARNING_THRESHOLD and agent.inventory.has_item("water_packs"):
            return self._action("drink",
                target={"source": "water_packs"},
                reasoning="Thirst getting low, drinking proactively",
                priority="high")
        
        if needs.hunger < self.WARNING_THRESHOLD and agent.inventory.has_item("emergency_rations"):
            return self._action("eat",
                target={"source": "emergency_rations"},
                reasoning="Hunger getting low, eating proactively",
                priority="high")
        
        if needs.energy < self.WARNING_THRESHOLD:
            return self._action("sleep",
                target={"location": "current"},
                reasoning="Energy declining, resting proactively",
                priority="high")
        
        # Injury treatment
        if agent.injury_level > 0.3 and agent.inventory.has_item("medical_supplies"):
            if not (agent.suit_equipped and not agent._in_habitat):
                return self._action("treat_injury",
                    target={},
                    reasoning=f"Injury level {agent.injury_level:.0%}, treating",
                    priority="high")
        
        # === PRIORITY 4: STRATEGIC TASK (colony building) ===
        
        # If has strategic goal, work toward it
        if agent.strategic_goal:
            goal_type = agent.strategic_goal.get("type", "gather")
            return self._action(goal_type,
                target=agent.strategic_goal.get("target", {}),
                reasoning=f"Working on strategic goal: {agent.strategic_goal.get('description', 'colony task')}",
                priority="normal")
        
        # All agents contribute to strategic colony building
        recipe_name = self._pick_needed_recipe(colony_score)
        recipe = self._get_recipe(recipe_name)
        if recipe:
            missing_materials = []
            for mat, qty in recipe.get("materials", {}).items():
                current_qty = agent.inventory.materials.get(mat, 0)
                if current_qty < qty:
                    missing_materials.append((mat, qty - current_qty))
            
            if missing_materials:
                target_mat = missing_materials[0][0]
                cell_resources = world_context.get("base_resources", {})
                if target_mat != "regolith" and target_mat not in cell_resources:
                    # If surface regolith is present on this unexcavated tile, excavate it to uncover deposits!
                    if "regolith" in cell_resources:
                        return self._action("gather",
                            target={"resource": "regolith"},
                            reasoning=f"Excavating surface regolith to uncover sub-surface {target_mat} deposits to build {recipe_name}",
                            priority="normal")
                    return self._action("explore",
                        target={"resource": target_mat},
                        reasoning=f"No {target_mat} in current cell, exploring to find a source to build {recipe_name}",
                        priority="normal")
                return self._action("gather",
                    target={"resource": target_mat},
                    reasoning=f"Gathering {target_mat} to build {recipe_name} (needs {missing_materials[0][1]} more)",
                    priority="normal")
            else:
                return self._action("build",
                    target={"recipe": recipe_name},
                    reasoning=f"All materials available, building {recipe_name}",
                    priority="normal")
        
        return self._action("explore",
            target={},
            reasoning="Exploring alien environment for resources",
            priority="normal")
    
    def _action(self, action: str, target: dict, reasoning: str,
                priority: str = "normal") -> dict:
        """Build standard action response dict."""
        return {
            "action": action,
            "target": target,
            "reasoning": reasoning,
            "priority": priority,
            "fallback": True,  # Flag that this came from fallback engine
        }
    
    def _get_recipe(self, recipe_name: str) -> Optional[dict]:
        """Load recipe from config."""
        import os, json
        try:
            config_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(config_dir, '..', '..', 'config', 'recipes.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    recipes = json.load(f)
                    return recipes.get("recipes", {}).get(recipe_name)
        except Exception as e:
            logger.warning(f"Fallback failed to load recipe {recipe_name}: {e}")
        return None

    def _pick_needed_recipe(self, colony_score: dict) -> str:
        """Pick the most needed recipe based on colony score gaps."""
        # Default priority order if no colony score available
        priority_recipes = [
            "water_collector",
            "solar_panel",
            "isru_o2_unit",
            "greenhouse",
            "habitat_module",
            "storage_crate",
        ]
        
        if not colony_score:
            return priority_recipes[0]
        
        # Find lowest scoring category and map to recipe
        category_to_recipe = {
            "shelter": "habitat_module",
            "water": "water_collector",
            "food": "greenhouse",
            "energy": "solar_panel",
            "o2": "isru_o2_unit",
            "hazard_protection": "habitat_module",
        }
        
        lowest = min(colony_score.items(), key=lambda x: x[1], default=("water", 0))
        return category_to_recipe.get(lowest[0], priority_recipes[0])


# Singleton instance
fallback_engine = FallbackDecisionEngine()
