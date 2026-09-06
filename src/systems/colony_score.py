"""
Colony Readiness Score Calculator.

Tracks operational progress toward supporting every surface occupant.
Targets loaded from config/colony_targets.json.
Overall score is a balanced weighted average; mission acceptance requires 100%.
"""

import json
import math
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ColonyScore:
    """
    Colony readiness scoring system.
    
    Evaluates commissioned infrastructure against the full arrival contract.
    
    Categories:
    - O2: Breathable atmosphere capacity (ISRU units)
    - Water: External make-up extraction AND wastewater treatment
    - Food: Calorie production (greenhouses)
    - Shelter: Pressurized living space (habitat modules)
    - Energy: Power generation (solar panels)
    - Hazard protection: Storm-safe sections integrated into habitats
    """
    
    # Default targets if config not found
    DEFAULT_TARGETS = {
        "o2": {"target": 3, "unit": "isru_o2_unit", "weight": 0.20},
        "water": {"target": 5, "unit": "water_collector+water_purifier", "weight": 0.20},
        "food": {"target": 3, "unit": "greenhouse", "weight": 0.15},
        "shelter": {"target": 4, "unit": "habitat_module", "weight": 0.15},
        "energy": {"target": 6, "unit": "solar_panel", "weight": 0.15},
        "hazard_protection": {"target": 4, "unit": "habitat_module", "weight": 0.15},
    }
    
    CATEGORY_MAPPING = {
        "oxygen_production": "o2",
        "oxygen_storage": "o2",
        "water_supply": "water",
        "water_extraction": "water",
        "water_recycling": "water",
        "water_storage": "water",
        "food_production": "food",
        "shelter_capacity": "shelter",
        "energy_infrastructure": "energy",
        "hazard_protection": "hazard_protection",
    }
    
    def __init__(self, config_path: str = None):
        """Load colony targets from config."""
        self._targets = self.DEFAULT_TARGETS.copy()
        self._readiness_threshold = 100.0
        self._population_target = 106
        self._component_targets: dict[str, dict[str, float]] = {}
        
        # Resolve config directory relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_dir = os.path.join(base_dir, 'config')
        
        if not config_path:
            config_path = os.path.join(config_dir, 'colony_targets.json')
            
        # Load recipes to map structure types to their contribution values
        self._recipes = {}
        recipes_path = os.path.join(config_dir, 'recipes.json')
        if os.path.exists(recipes_path):
            try:
                with open(recipes_path, 'r', encoding='utf-8') as f:
                    self._recipes = json.load(f).get("recipes", {})
            except Exception as e:
                logger.warning(f"Failed to load recipes in ColonyScore: {e}")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                self._readiness_threshold = float(
                    loaded.get("readiness_threshold_percent", 100.0)
                )
                self._population_target = int(loaded.get("population_target", 106))
                    
                sub_targets = loaded.get("sub_targets", {})
                if sub_targets:
                    # Clear default target values to overwrite them
                    self._targets = {}
                    for long_key, val in sub_targets.items():
                        mapped_key = self.CATEGORY_MAPPING.get(long_key, long_key)
                        
                        # Extract target capacity value
                        target_val = 1.0
                        for k, v in val.items():
                            if k.startswith("target_capacity"):
                                target_val = float(v)
                                break
                                
                        self._targets[mapped_key] = {
                            "target": target_val,
                            "weight": val.get("weight_in_score", 0.1),
                            "display_name": val.get("display_name", long_key)
                        }
                        components = val.get("capacity_components", {})
                        if isinstance(components, dict) and components:
                            self._component_targets[mapped_key] = {}
                            for component_name, component in components.items():
                                if not isinstance(component, dict):
                                    continue
                                component_target = next((
                                    float(value)
                                    for key, value in component.items()
                                    if key.startswith("target_capacity")
                                ), 0.0)
                                if component_target > 0.0:
                                    self._component_targets[mapped_key][
                                        str(component_name)
                                    ] = component_target
                logger.info(f"Colony targets loaded from {config_path}: {self._targets}")
            except Exception as e:
                logger.warning(f"Failed to load colony targets from {config_path}: {e}")
        
        # Current counts (updated by engine)
        self._current: dict[str, float] = {cat: 0.0 for cat in self._targets}
        self._component_current: dict[str, dict[str, float]] = {
            category: {component: 0.0 for component in components}
            for category, components in self._component_targets.items()
        }
        self._communications_ready = False
        # Planet physics may derate the usable output of a structure without
        # changing its nameplate recipe.  Planning must use the same derating
        # as scoring or a policy can stop at the nominal count while the
        # operational category remains incomplete.
        self._structure_capacity_multipliers: dict[str, float] = {}
        
        # Score history for trending
        self._history: list[dict] = []

    def set_structure_capacity_multiplier(
        self, structure_id: str, multiplier: float
    ) -> None:
        """Set a deterministic planet-specific planning derate (0, 1]."""
        self._structure_capacity_multipliers[str(structure_id)] = max(
            1e-9, min(1.0, float(multiplier))
        )
    
    def update_counts(
        self,
        structures_built: dict[str, int],
        communications_operational: bool | None = None,
        capacity_overrides: dict | None = None,
    ):
        """
        Update current structure counts from world state.
        
        Args:
            structures_built: Dict of structure_id → count
        """
        # Reset counts/capacities
        counts = {cat: 0.0 for cat in self._targets}
        component_counts = {
            category: {component: 0.0 for component in components}
            for category, components in self._component_targets.items()
        }
        self._communications_ready = (
            structures_built.get("communications_array", 0) > 0
            if communications_operational is None
            else bool(communications_operational)
        )
        
        for struct_id, count in structures_built.items():
            if count <= 0:
                continue
                
            recipe = self._recipes.get(struct_id)
            if recipe and "colony_contribution" in recipe:
                contribution = recipe.get("colony_contribution")
                if isinstance(contribution, dict):
                    for long_cat, val in contribution.items():
                        mapped_key = self.CATEGORY_MAPPING.get(long_cat, long_cat)
                        if mapped_key in counts:
                            if long_cat in component_counts.get(mapped_key, {}):
                                component_counts[mapped_key][long_cat] += (
                                    float(val) * count
                                )
                            else:
                                counts[mapped_key] += float(val) * count
            else:
                # Fallback to default hardcoded structure mapping if recipe contribution missing
                structure_to_category = {
                    "isru_o2_unit": "o2",
                    "water_collector": "water",
                    "greenhouse": "food",
                    "habitat_module": "shelter",
                    "solar_panel": "energy",
                }
                cat = structure_to_category.get(struct_id)
                if cat and cat in counts:
                    defaults_contrib = {
                        "isru_o2_unit": 6.0,
                        "water_collector": 200.0,
                        "greenhouse": 44000.0,
                        "habitat_module": 8.0,
                        "solar_panel": 25.0,
                    }
                    val = defaults_contrib.get(struct_id, 1.0)
                    counts[cat] += val * count
        
        for category, components in component_counts.items():
            component_targets = self._component_targets.get(category, {})
            fulfillment = min(
                components[name] / max(target, 1e-9)
                for name, target in component_targets.items()
            ) if component_targets else 0.0
            counts[category] = (
                float(self._targets[category].get("target", 1.0))
                * min(1.0, fulfillment)
            )

        # The engine may replace nominal nameplate values with audited,
        # operational capacities (for example mature greenhouse output or a
        # water extractor that has a real source).  This never increases a
        # recipe's physical capacity; it only removes unavailable throughput.
        for category, override in (capacity_overrides or {}).items():
            if category not in counts:
                continue
            if isinstance(override, dict) and category in component_counts:
                for component, value in override.items():
                    if component in component_counts[category]:
                        component_counts[category][component] = max(
                            0.0, float(value)
                        )
                targets = self._component_targets[category]
                fulfillment = min(
                    component_counts[category][name] / max(target, 1e-9)
                    for name, target in targets.items()
                )
                counts[category] = (
                    float(self._targets[category].get("target", 1.0))
                    * min(1.0, fulfillment)
                )
            elif isinstance(override, (int, float)):
                counts[category] = max(0.0, float(override))

        self._component_current = component_counts
        self._current = counts

    def get_structure_capacity_status(self, structure_id: str) -> Optional[dict]:
        """Return configured 106-person capacity status for one structure.

        Recipes remain the single source of truth: required module counts are
        derived from each recipe's colony contribution and colony_targets.json,
        never from a hardcoded one-building milestone.
        """
        recipe = self._recipes.get(structure_id, {})
        contribution = recipe.get("colony_contribution")
        if not isinstance(contribution, dict):
            return None

        statuses = []
        for raw_category, per_structure in contribution.items():
            category = self.CATEGORY_MAPPING.get(raw_category, raw_category)
            target_config = self._targets.get(category)
            # ``colony_contribution`` also contains explanatory metadata for
            # mission-gate structures (for example ``notes`` and boolean
            # flags on the communications array).  Those fields are not
            # physical capacities and must never be parsed as numbers.
            if (
                target_config is None
                or isinstance(per_structure, bool)
                or not isinstance(per_structure, (int, float))
            ):
                continue
            amount = float(per_structure)
            if amount <= 0.0:
                continue
            component_target = self._component_targets.get(
                category, {}
            ).get(raw_category)
            target = float(
                component_target
                if component_target is not None
                else target_config.get("target", 0.0)
            )
            current = float(
                self._component_current.get(category, {}).get(raw_category, 0.0)
                if component_target is not None
                else self._current.get(category, 0.0)
            )
            statuses.append({
                "category": category,
                "component": raw_category if component_target is not None else None,
                "target_capacity": target,
                "current_capacity": current,
                "contribution_per_structure": amount,
                "required_count": max(1, int(math.ceil(
                    target / (
                        amount * self._structure_capacity_multipliers.get(
                            structure_id, 1.0
                        )
                    )
                ))),
                "fulfillment": min(1.0, current / max(target, 1.0)),
            })
        if not statuses:
            return None
        # Multi-purpose structures are primarily planned against the category
        # furthest from its configured target; their other contributions still
        # enter the normal colony score calculation.
        return min(statuses, key=lambda status: status["fulfillment"])
    
    def get_scores(self) -> dict:
        """Get per-category scores (0.0 to 1.0+)."""
        scores = {}
        for category, config in self._targets.items():
            target = config.get("target", 1)
            current = self._current.get(category, 0)
            scores[category] = min(1.0, current / max(1, target))
        return scores
    
    def get_overall_score(self) -> float:
        """Get weighted overall score (0-100%)."""
        scores = self.get_scores()
        total_weight = sum(c.get("weight", 0.1) for c in self._targets.values())
        
        weighted_sum = sum(
            scores.get(cat, 0) * self._targets[cat].get("weight", 0.1)
            for cat in self._targets
        )
        
        score = weighted_sum / max(0.01, total_weight) * 100
        # The capacity hardware cannot call down a civilian ship by itself.
        # The configured communications array is a physical final gate and
        # caps readiness at 90 until it is operational.
        if not self._communications_ready:
            score = min(score, 90.0)
        return round(score, 1)
    
    def get_weakest_category(self) -> tuple[str, float]:
        """Get the lowest-scoring category (for strategic priority)."""
        scores = self.get_scores()
        if not scores:
            return ("shelter", 0.0)
        weakest = min(scores.items(), key=lambda x: x[1])
        return weakest
    
    def record_tick(self, tick: int):
        """Record current score for history/trending."""
        self._history.append({
            "tick": tick,
            "overall": self.get_overall_score(),
            "categories": self.get_scores(),
        })
        # Keep last 200 entries
        if len(self._history) > 200:
            self._history = self._history[-100:]
    
    def get_summary(self) -> str:
        """Get compact summary for LLM context."""
        scores = self.get_scores()
        parts = []
        for cat, score in scores.items():
            current = self._current.get(cat, 0)
            target = self._targets[cat].get("target", 1)
            parts.append(f"{cat}: {current}/{target} ({score:.0%})")
        overall = self.get_overall_score()
        return f"Colony readiness: {overall:.0f}%. {', '.join(parts)}"
    
    def is_colony_ready(self) -> bool:
        """Check the configured capacity threshold (100% for ASAC-6)."""
        return self.get_overall_score() >= self._readiness_threshold
    
    def to_dict(self) -> dict:
        """Full serialization for API/persistence."""
        return {
            "overall": self.get_overall_score(),
            "categories": self.get_scores(),
            "current_counts": self._current,
            "component_counts": self._component_current,
            "targets": {k: v.get("target", 0) for k, v in self._targets.items()},
            "population_target": self._population_target,
            "readiness_threshold_percent": self._readiness_threshold,
            "ready": self.is_colony_ready(),
            "communications_ready": self._communications_ready,
        }
