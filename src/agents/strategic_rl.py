"""Planet-scoped reinforcement learning for colony-level strategy.

The low-level simulation remains authoritative for physics, recipes, travel,
health and construction.  This policy chooses *which physically valid colony
objective to pursue next*.  It deliberately contains no preferred build
order: a fresh policy may choose water, power, oxygen, food or shelter first
and must learn the consequences on that planet.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Iterable, Optional


STRATEGY_POLICY_ID = "__shared_colony_strategy__"
META_KEY = "__strategy_meta__"
# Version 8 retains normalized completion credit and adds deterministic
# acceptance-envelope masks: severe radiation needs an initial real shelter,
# and no already-ahead category may crowd every zero/low readiness category.
# RL still chooses among all physically valid objectives inside that envelope.
STATE_SCHEMA_VERSION = 8


def _band(value: float) -> str:
    value = max(0.0, min(1.0, float(value)))
    if value <= 0.0:
        return "none"
    if value < 0.34:
        return "low"
    if value < 0.67:
        return "mid"
    if value < 1.0:
        return "near"
    return "done"


def build_colony_strategy_state(
    *,
    category_scores: dict[str, float],
    colony_resources: dict[str, float],
    living_crew: int,
    fit_crew: int,
    active_site_type: Optional[str],
    ready_recipe_count: int,
    surface_requires_plss: bool,
    eligible_recipes: Optional[Iterable[str]] = None,
    ready_recipes: Optional[Iterable[str]] = None,
) -> str:
    """Return a compact, auditable state made only from observable facts."""
    ordered_categories = (
        "energy", "o2", "water", "food", "shelter", "hazard_protection"
    )
    capacity = ",".join(
        f"{name}:{_band(category_scores.get(name, 0.0))}"
        for name in ordered_categories
    )

    def reserve(value: float, critical: float, buffered: float) -> str:
        if value <= 0.0:
            return "empty"
        if value < critical:
            return "critical"
        if value < buffered:
            return "low"
        return "buffered"

    population = max(1, int(living_crew))

    def days_of_supply(
        value: float,
        per_person_day: float,
        critical_days: float,
        buffered_days: float,
    ) -> str:
        daily_demand = population * max(1e-9, float(per_person_day))
        return reserve(
            float(value) / daily_demand,
            critical_days,
            buffered_days,
        )

    water = days_of_supply(
        float(colony_resources.get("water_reserve_l", 0.0)),
        3.5, 25.0, 40.0,
    )
    oxygen = days_of_supply(
        float(colony_resources.get("o2_reserve_kg", 0.0)),
        0.84, 10.0, 25.0,
    )
    # Full mixed-crop rated output follows 56 serviced days. The critical band
    # leaves a modest assembly, checkout and crop-loss contingency.
    food = days_of_supply(
        float(colony_resources.get("food_reserve_kcal", 0.0)),
        2700.0, 100.0, 120.0,
    )
    energy = reserve(
        float(colony_resources.get("energy_stored_kwh", 0.0)), 5.0, 25.0
    )
    crew_band = f"{max(0, int(fit_crew))}/{max(0, int(living_crew))}"
    site = active_site_type or "none"
    ready = "none" if ready_recipe_count <= 0 else (
        "one" if ready_recipe_count == 1 else "many"
    )
    eligible = ",".join(sorted(str(name) for name in (eligible_recipes or ())))
    ready_names = ",".join(sorted(str(name) for name in (ready_recipes or ())))
    return (
        f"cap[{capacity}]|reserve[w:{water},o:{oxygen},f:{food},e:{energy}]"
        f"|crew:{crew_band}|site:{site}|ready:{ready}[{ready_names}]"
        f"|eligible[{eligible}]|plss:{int(bool(surface_requires_plss))}"
    )


@dataclass
class StrategyCommitment:
    state_key: str
    action_key: str
    recipe: str
    target_count: int
    started_tick: int
    score_at_start: float
    last_progress_tick: int
    best_material_readiness: float = 0.0
    lowest_remaining_work: Optional[float] = None
    best_site_progress: float = 0.0
    target_fraction: float = 0.0
    mission_acceptance_gate: bool = False
    stall_timeout_ticks: int = 432
    replan_interval_ticks: int = 1008


class ColonyStrategicPolicy:
    """Small tabular Q-policy shared by the precursor crew on one planet."""

    def __init__(
        self,
        planet_id: str,
        *,
        seed: int = 0,
        q_table: Optional[dict] = None,
        learning_rate: float = 0.22,
        discount_factor: float = 0.90,
        initial_epsilon: float = 0.35,
        minimum_epsilon: float = 0.05,
        epsilon_decay_per_attempt: float = 0.90,
        default_stall_timeout_ticks: int = 432,
        default_replan_interval_ticks: int = 1008,
    ):
        self.planet_id = str(planet_id)
        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        self.initial_epsilon = float(initial_epsilon)
        self.minimum_epsilon = float(minimum_epsilon)
        self.epsilon_decay_per_attempt = float(epsilon_decay_per_attempt)
        self.default_stall_timeout_ticks = max(
            1, int(default_stall_timeout_ticks)
        )
        self.default_replan_interval_ticks = max(
            self.default_stall_timeout_ticks,
            int(default_replan_interval_ticks),
        )
        self.q_table: dict[str, dict[str, float]] = {}
        self.attempt_count = 0
        self.exploration_age = 0
        self.total_reward = 0.0
        self.decision_count = 0
        self.commitment: Optional[StrategyCommitment] = None
        self.episode_trace: list[tuple[str, str]] = []
        self.episode_capacity_high_water: dict[str, int] = {}
        self.episode_verified_completions = 0
        self.last_outcome: Optional[dict] = None
        if q_table:
            self.load(q_table)

    @property
    def epsilon(self) -> float:
        return max(
            self.minimum_epsilon,
            self.initial_epsilon
            * (self.epsilon_decay_per_attempt ** self.exploration_age),
        )

    def _rng(self, tick: int) -> random.Random:
        material = (
            f"{self.planet_id}|{self.seed}|{self.attempt_count}|"
            f"{self.decision_count}|{int(tick)}"
        )
        digest = hashlib.sha256(material.encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    @staticmethod
    def action_key(recipe: str) -> str:
        return f"capacity:{recipe}"

    @staticmethod
    def _reserve_context(state_key: str) -> str:
        """Extract only the life-support reserve bands from an audited state."""
        marker = "|reserve["
        start = str(state_key).find(marker)
        if start < 0:
            return ""
        end = str(state_key).find("]", start)
        return str(state_key)[start:end + 1] if end >= 0 else ""

    @staticmethod
    def _critical_reserve_recipes(state_key: str, *, include_water: bool = True) -> set[str]:
        """Map last-safe-start reserve bands to relevant safe actions."""
        context = ColonyStrategicPolicy._reserve_context(state_key)
        if not context:
            return set()
        body = context.removeprefix("|reserve[").removesuffix("]")
        bands: dict[str, str] = {}
        for item in body.split(","):
            key, separator, value = item.partition(":")
            if separator:
                bands[key.strip()] = value.strip()
        urgent: set[str] = set()
        if include_water and bands.get("w") in {"critical", "empty"}:
            urgent.update({
                "water_collector", "water_purifier", "solar_panel",
                "power_distribution_grid", "life_support_distribution_grid",
                "potable_water_tank",
            })
        if bands.get("o") in {"critical", "empty"}:
            urgent.update({
                "isru_o2_unit", "water_collector", "solar_panel",
                "power_distribution_grid", "life_support_distribution_grid",
                "oxygen_buffer_tank", "potable_water_tank",
            })
        if bands.get("f") in {"critical", "empty"}:
            urgent.update({
                "greenhouse", "solar_panel", "power_distribution_grid",
                "life_support_distribution_grid", "potable_water_tank",
                "water_collector", "water_purifier",
            })
        if bands.get("e") in {"critical", "empty"}:
            urgent.update({"solar_panel", "power_distribution_grid"})
        return urgent

    def choose(
        self,
        *,
        state_key: str,
        candidates: Iterable[dict],
        structures_built: dict[str, int],
        tick: int,
        colony_score: float,
    ) -> Optional[dict]:
        """Choose and commit to one candidate until one real module completes."""
        current = self.commitment
        if current is not None and int(
            structures_built.get(current.recipe, 0)
        ) >= current.target_count:
            # The physics tick has completed the objective, but the engine has
            # not necessarily reached its delayed-reward phase yet.  Never
            # discard or replace this transition in ``choose``: doing so made
            # every real construction completion worth zero and could attach
            # terminal success to a brand-new, unexecuted objective.
            return None

        raw_options = [dict(candidate) for candidate in candidates]
        if not raw_options:
            return None

        raw_by_recipe = {
            str(item["recipe"]): item for item in raw_options
        }
        replan_reason: Optional[str] = None
        if current is not None:
            current_candidate = raw_by_recipe.get(current.recipe)
            if (
                current_candidate is not None
                and current_candidate.get("actionable", True) is False
            ):
                # The action mask is built from measured inventories,
                # discovered resources and the real mission deadline. Do not
                # keep dispatching crew to an objective that those facts now
                # show cannot advance.
                self._release_unproductive_commitment(
                    current,
                    state_key=state_key,
                    tick=tick,
                    reward=-10.0,
                    reason="infeasible",
                )
                current = None
                replan_reason = "infeasible"

        options = [
            item for item in raw_options
            if item.get("actionable", True) is not False
        ]
        if not options:
            return None

        by_recipe = {str(item["recipe"]): item for item in options}
        bootstrap_power = {name: item for name, item in by_recipe.items()
                           if item.get("bootstrap_power_priority", False)}
        thermal_recovery = {name: item for name, item in by_recipe.items()
                            if item.get("thermal_safety_priority", False)}
        recovery_options = bootstrap_power or thermal_recovery
        recovery_reason = "bootstrap_power_reserve" if bootstrap_power else "thermal_overload"
        if recovery_options:
            # The lander's finite emergency energy is already being spent.
            # Storage and distribution cannot recharge the construction rover
            # without generation. Preserve physical sites, but do not let an
            # unstarted, staged objective consume the last launch window.
            if current is not None and current.recipe not in recovery_options:
                self._release_unproductive_commitment(
                    current, state_key=state_key, tick=tick,
                    reward=-2.0, reason=recovery_reason,
                )
                current = None
                replan_reason = recovery_reason
            by_recipe = recovery_options
        # Once the exact BOM is staged, give the crew a bounded window to
        # rendezvous at the airlock and open the physical site.  Reserve bands
        # can cross a threshold while two people are walking to the rover; if
        # that ordinary fluctuation cancels the order, the rendezvous action is
        # overwritten and no structure ever starts.  This is not an immortal
        # scripted build order: the existing stall timer still releases it
        # after three days without a site or measurable progress.
        current_candidate = (
            by_recipe.get(current.recipe) if current is not None else None
        )
        staged_site_commitment = bool(
            current is not None
            and current_candidate is not None
            and current_candidate.get("ready", False)
            and not isinstance(current_candidate.get("site"), dict)
            and int(structures_built.get(current.recipe, 0))
            < current.target_count
            and int(tick) - current.last_progress_tick
            < current.stall_timeout_ticks
        )
        if staged_site_commitment:
            selected = current_candidate
            selected.update({
                "strategy_state": current.state_key,
                "strategy_action": current.action_key,
                "strategy_committed": True,
                "strategy_epsilon": self.epsilon,
                "strategy_site_mobilization": True,
            })
            return selected

        if current is not None:
            built = int(structures_built.get(current.recipe, 0))
            reserve_changed = (
                self._reserve_context(current.state_key)
                != self._reserve_context(state_key)
            )
            # A BOM objective is a commitment, not a suicide pact. Before a
            # physical construction site exists (multiple candidates remain),
            # a real change in water/O2/energy reserve bands must reopen the
            # strategic decision. Exact material-readiness changes do not.
            has_physical_site = isinstance(
                (current_candidate or {}).get("site"), dict
            )
            if (
                built < current.target_count
                and reserve_changed
                and len(by_recipe) > 1
                and not has_physical_site
            ):
                self.commitment = None
                current = None
                replan_reason = "reserve_band_changed"

        # Critical means the last physically credible start window has been
        # reached, not merely that a bar is visually low. At that point the
        # deterministic safety mask removes unrelated objectives; RL still
        # chooses among all relevant, feasible recovery paths.
        urgent_recipes = self._critical_reserve_recipes(state_key)
        other_reserve_recipes = self._critical_reserve_recipes(state_key, include_water=False)
        urgent_options = {
            recipe: candidate for recipe, candidate in by_recipe.items()
            if recipe in urgent_recipes
            and (candidate.get("water_reserve_support", True)
                 or recipe in other_reserve_recipes)
        }
        if urgent_options:
            if current is not None and current.recipe not in urgent_options:
                self._release_unproductive_commitment(
                    current,
                    state_key=state_key,
                    tick=tick,
                    reward=-2.0,
                    reason="survival_deadline",
                )
                current = None
                replan_reason = "survival_deadline"
            by_recipe = urgent_options

        # On a surface whose measured dose/flare envelope is immediately
        # mission-threatening, the first real storm shelter is a safety case,
        # not an optional preference to be discovered after repeated deaths.
        safety_options = {
            recipe: candidate for recipe, candidate in by_recipe.items()
            if candidate.get("safety_priority", False)
        }
        utility_recovery_active = False
        if safety_options:
            if current is not None and current.recipe not in safety_options:
                self._release_unproductive_commitment(
                    current,
                    state_key=state_key,
                    tick=tick,
                    reward=-2.0,
                    reason="mandatory_hazard_control",
                )
                current = None
                replan_reason = "mandatory_hazard_control"
            by_recipe = safety_options
        elif not urgent_options:
            utility_recovery = {
                recipe: candidate
                for recipe, candidate in by_recipe.items()
                if candidate.get("utility_recovery_priority", False)
            }
            if utility_recovery:
                utility_recovery_active = True
                # Once commissioned consumers exceed the real pipe/port
                # envelope, another distribution node is a measured dependency.
                # If its BOM is still in production, use otherwise idle field
                # labor on already-staged work that does not consume that
                # envelope.  No additional disconnected endpoint is installed.
                ready_recovery = {
                    recipe: candidate
                    for recipe, candidate in utility_recovery.items()
                    if candidate.get("ready", False)
                }
                ready_independent_work = {
                    recipe: candidate
                    for recipe, candidate in by_recipe.items()
                    if candidate.get("ready", False)
                    and not candidate.get("life_support_endpoint", False)
                    and not candidate.get("utility_recovery_priority", False)
                }
                recovery_mask = (
                    ready_recovery
                    or ready_independent_work
                    or utility_recovery
                )
                parallel_work = bool(
                    not ready_recovery and ready_independent_work
                )
                if current is not None and current.recipe not in recovery_mask:
                    neutral_grid_wait = bool(
                        parallel_work and current.recipe in utility_recovery
                    )
                    self._release_unproductive_commitment(
                        current,
                        state_key=state_key,
                        tick=tick,
                        reward=0.0 if neutral_grid_wait else -2.0,
                        reason=(
                            "utility_materials_in_production"
                            if neutral_grid_wait
                            else "utility_capacity_exhausted"
                        ),
                    )
                    current = None
                    replan_reason = (
                        "utility_materials_in_production"
                        if neutral_grid_wait
                        else "utility_capacity_exhausted"
                    )
                by_recipe = recovery_mask
        water_stock_options = {
            recipe: candidate for recipe, candidate in by_recipe.items()
            if candidate.get("water_stock_deadline_priority", False)
        }
        water_stock_recovery = bool(
            water_stock_options and not safety_options and not urgent_options
            and not utility_recovery_active
        )
        if water_stock_recovery:
            if current is not None and current.recipe not in water_stock_options:
                self._release_unproductive_commitment(
                    current, state_key=state_key, tick=tick, reward=0.0,
                    reason="water_stock_fill_deadline",
                )
                current = None
                replan_reason = "water_stock_fill_deadline"
            by_recipe = water_stock_options
        if (
            not safety_options
            and not urgent_options
            and not utility_recovery_active
            and not water_stock_recovery
        ):
            # Balance is a deterministic arrival constraint: excess power or
            # water cannot compensate for zero habitat, food or radiation
            # capacity. Keep candidates within one meaningful target fraction
            # of the least-complete category. This prevents easy-module spam
            # without teaching a fixed build order.
            capacity_options = {
                recipe: candidate for recipe, candidate in by_recipe.items()
                if not candidate.get("mission_acceptance_gate", False)
                and not candidate.get("utility_prerequisite", False)
            }
            if capacity_options:
                minimum_fulfillment = min(
                    float(candidate.get("fulfillment", 0.0))
                    for candidate in capacity_options.values()
                )
                balanced_capacity = {
                    recipe: candidate
                    for recipe, candidate in capacity_options.items()
                    if float(candidate.get("fulfillment", 0.0))
                    <= minimum_fulfillment + 0.20 + 1e-9
                }
                utility_options = {
                    recipe: candidate for recipe, candidate in by_recipe.items()
                    if candidate.get("utility_prerequisite", False)
                }
                gate_options = {
                    recipe: candidate for recipe, candidate in by_recipe.items()
                    if candidate.get("mission_acceptance_gate", False)
                    and minimum_fulfillment >= 0.50
                }
                balanced_options = {
                    **balanced_capacity, **utility_options, **gate_options
                }
                if balanced_options:
                    if (
                        current is not None
                        and current.recipe not in balanced_options
                    ):
                        self._release_unproductive_commitment(
                            current,
                            state_key=state_key,
                            tick=tick,
                            reward=-1.0,
                            reason="acceptance_balance",
                        )
                        current = None
                        replan_reason = "acceptance_balance"
                    by_recipe = balanced_options

        # A six-person precursor crew must keep the field workface productive
        # while the autonomous shop manufactures a different objective's
        # long-lead parts.  Prefer an already-complete BOM within the safety,
        # reserve and balance masks above.  Materials stay physical and the
        # deferred objective remains eligible as soon as no ready package is
        # available; this changes scheduling, not recipes or yields.
        ready_options = {
            recipe: candidate for recipe, candidate in by_recipe.items()
            if candidate.get("ready", False)
        }
        if ready_options:
            if current is not None and current.recipe not in ready_options:
                self._release_unproductive_commitment(
                    current,
                    state_key=state_key,
                    tick=tick,
                    reward=0.0,
                    reason="ready_work_available",
                )
                current = None
                replan_reason = "ready_work_available"
            by_recipe = ready_options

        if current is not None and current.recipe in by_recipe:
            candidate = by_recipe[current.recipe]
            material_readiness = float(candidate.get(
                "material_readiness", current.best_material_readiness
            ))
            remaining_value = candidate.get("remaining_work_units")
            remaining_work = (
                max(0.0, float(remaining_value))
                if remaining_value is not None else None
            )
            site = candidate.get("site")
            site_progress = float(
                (site or {}).get(
                    "progress", candidate.get("site_progress", 0.0)
                )
            )
            progressed = False
            if material_readiness > current.best_material_readiness + 1e-6:
                current.best_material_readiness = material_readiness
                progressed = True
            if (
                remaining_work is not None
                and (
                    current.lowest_remaining_work is None
                    or remaining_work < current.lowest_remaining_work - 1e-6
                )
            ):
                current.lowest_remaining_work = remaining_work
                progressed = True
            if site_progress > current.best_site_progress + 1e-6:
                current.best_site_progress = site_progress
                progressed = True
            if progressed:
                current.last_progress_tick = int(tick)

            has_physical_site = isinstance(site, dict)
            stalled = (
                int(tick) - current.last_progress_tick
                >= current.stall_timeout_ticks
            )
            review_due = (
                not has_physical_site
                and int(tick) - current.started_tick
                >= current.replan_interval_ticks
            )
            if len(by_recipe) > 1 and (stalled or review_due):
                # Materials and WIP remain physical; replanning neither
                # teleports nor discards them. A true stall receives a larger
                # penalty than a progressing objective that simply failed to
                # deliver a module within its bounded strategy window.
                replan_reason = "stalled" if stalled else "review_due"
                released_recipe = current.recipe
                self._release_unproductive_commitment(
                    current,
                    state_key=state_key,
                    tick=tick,
                    reward=-8.0 if stalled else -1.0,
                    reason=replan_reason,
                )
                current = None
                # A one-decision cooldown guarantees that an exploratory draw
                # cannot immediately recreate the exact deadlock just proven
                # by physical telemetry. The recipe remains eligible at the
                # next strategic review; no long-term build order is encoded.
                by_recipe.pop(released_recipe, None)

        if current is not None:
            built = int(structures_built.get(current.recipe, 0))
            if built < current.target_count and current.recipe in by_recipe:
                selected = by_recipe[current.recipe]
                selected.update({
                    "strategy_state": current.state_key,
                    "strategy_action": current.action_key,
                    "strategy_committed": True,
                    "strategy_epsilon": self.epsilon,
                })
                return selected
            # An action-mask change can invalidate a still-unfinished restored
            # objective (for example a prerequisite structure was destroyed).
            # Only that case may release the commitment without a completion.
            if built < current.target_count:
                self.commitment = None

        actions = self.q_table.setdefault(state_key, {})
        for recipe in by_recipe:
            actions.setdefault(self.action_key(recipe), 0.0)

        rng = self._rng(tick)
        exploring = rng.random() < self.epsilon
        if exploring:
            recipe = rng.choice(sorted(by_recipe))
        else:
            best_value = max(actions[self.action_key(name)] for name in by_recipe)
            tied = sorted(
                name for name in by_recipe
                if actions[self.action_key(name)] == best_value
            )
            # Equal fresh Q-values must not secretly encode a build order.
            recipe = rng.choice(tied)

        action = self.action_key(recipe)
        current_count = int(structures_built.get(recipe, 0))
        self.episode_capacity_high_water[recipe] = max(
            current_count,
            self.episode_capacity_high_water.get(recipe, current_count),
        )
        target_count = current_count + 1
        selected = by_recipe[recipe]
        required_count = max(1, int(selected.get("required_count", 1)))
        mission_acceptance_gate = bool(
            selected.get("mission_acceptance_gate", False)
        )
        # A one-off acceptance gate is important but must not dwarf an entire
        # life-support category. Normal capacity credit is the marginal share
        # of that category's configured 106-person target.
        target_fraction = (
            0.20 if mission_acceptance_gate
            else min(1.0, 1.0 / required_count)
        )
        self.commitment = StrategyCommitment(
            state_key=state_key,
            action_key=action,
            recipe=recipe,
            target_count=target_count,
            started_tick=int(tick),
            score_at_start=float(colony_score),
            last_progress_tick=int(tick),
            best_material_readiness=float(selected.get(
                "material_readiness", 0.0
            )),
            lowest_remaining_work=(
                max(0.0, float(selected["remaining_work_units"]))
                if selected.get("remaining_work_units") is not None else None
            ),
            best_site_progress=float(
                (selected.get("site") or {}).get(
                    "progress", selected.get("site_progress", 0.0)
                )
            ),
            target_fraction=target_fraction,
            mission_acceptance_gate=mission_acceptance_gate,
            stall_timeout_ticks=max(1, int(selected.get(
                "stall_timeout_ticks", self.default_stall_timeout_ticks
            ))),
            replan_interval_ticks=max(1, int(selected.get(
                "replan_interval_ticks", self.default_replan_interval_ticks
            ))),
        )
        transition = (state_key, action)
        if not self.episode_trace or self.episode_trace[-1] != transition:
            self.episode_trace.append(transition)
            if len(self.episode_trace) > 256:
                del self.episode_trace[:-256]
        self.decision_count += 1

        selected.update({
            "strategy_state": state_key,
            "strategy_action": action,
            "strategy_committed": True,
            "strategy_exploring": exploring,
            "strategy_epsilon": self.epsilon,
            "strategy_target_count": target_count,
        })
        if replan_reason:
            selected["strategy_replan_reason"] = replan_reason
        return selected

    def _release_unproductive_commitment(
        self,
        current: StrategyCommitment,
        *,
        state_key: str,
        tick: int,
        reward: float,
        reason: str,
    ) -> None:
        """Learn from an observable dead end and reopen the action mask."""
        self._td_update(
            current.state_key, current.action_key, reward, state_key
        )
        self.total_reward += float(reward)
        self.last_outcome = {
            "type": "objective_replanned",
            "recipe": current.recipe,
            "reason": str(reason),
            "reward": float(reward),
            "elapsed_ticks": max(0, int(tick) - current.started_tick),
        }
        self.commitment = None

    def observe_completion(
        self,
        *,
        structures_built: dict[str, int],
        tick: int,
        colony_score: float,
        tick_minutes: float,
        next_state_key: str,
    ) -> float:
        """Reward a verified completed objective, never a repeated action tick."""
        current = self.commitment
        if current is None:
            return 0.0
        if int(structures_built.get(current.recipe, 0)) < current.target_count:
            return 0.0

        elapsed_days = max(
            0.0,
            (int(tick) - current.started_tick) * float(tick_minutes) / 1440.0,
        )
        score_gain = max(0.0, float(colony_score) - current.score_at_start)
        built_count = int(structures_built.get(current.recipe, 0))
        previous_peak = self.episode_capacity_high_water.get(
            current.recipe, 0
        )
        net_new_capacity = built_count > previous_peak
        if net_new_capacity:
            # Reward verified mission value, not raw object count. This keeps
            # physically necessary zero-score prerequisites learnable while
            # making actual balanced-readiness gain much more valuable than
            # repeating an easy module in one category.
            target_fraction_credit = 24.0 * current.target_fraction
            reward = (
                2.0
                + target_fraction_credit
                + score_gain * 6.0
                - min(5.0, elapsed_days * 0.15)
            )
            self.episode_capacity_high_water[current.recipe] = built_count
            self.episode_verified_completions += 1
        else:
            # Replacing a collapsed copy restores physical capacity but must
            # not be an infinite positive-reward loop. Episode success still
            # credits necessary recovery through terminal return.
            reward = -min(2.0, elapsed_days * 0.20)
        self._td_update(current.state_key, current.action_key, reward, next_state_key)
        self.total_reward += reward
        self.last_outcome = {
            "type": "objective_completed",
            "recipe": current.recipe,
            "reward": round(reward, 3),
            "elapsed_days": round(elapsed_days, 3),
            "score_gain": round(score_gain, 3),
            "target_fraction": round(current.target_fraction, 4),
            "net_new_capacity": net_new_capacity,
        }
        self.commitment = None
        return reward

    def _td_update(
        self, state_key: str, action_key: str, reward: float, next_state_key: str
    ) -> None:
        actions = self.q_table.setdefault(state_key, {})
        old = float(actions.get(action_key, 0.0))
        next_actions = self.q_table.get(next_state_key, {})
        next_best = max(next_actions.values()) if next_actions else 0.0
        target = float(reward) + self.discount_factor * next_best
        actions[action_key] = round(
            old + self.learning_rate * (target - old), 4
        )

    def finish_episode(self, outcome: str, *, elapsed_days: float = 0.0) -> int:
        """Back-propagate success/failure through high-level decisions."""
        if str(outcome) == "manual_stop":
            # Developer/UI inspection stops are not experimental outcomes and
            # must not contaminate the learned planet policy.
            self.last_outcome = {
                "type": "episode_aborted",
                "outcome": "manual_stop",
                "reward": 0.0,
                "updated_transitions": 0,
            }
            self.episode_trace.clear()
            self.episode_capacity_high_water.clear()
            self.episode_verified_completions = 0
            self.commitment = None
            return 0
        terminal_rewards = {
            "colony_ready": 100.0,
            "timeout": -35.0,
            "stagnation": -45.0,
        }
        if str(outcome) == "all_agents_dead":
            # Flat -100 made a near-successful 42-day attempt indistinguishable
            # from a collapse on day 12. Survival time is an observable outcome,
            # so it provides dense credit without prescribing any build order.
            reward = min(-40.0, -100.0 + max(0.0, float(elapsed_days)) * 1.5)
        else:
            reward = float(terminal_rewards.get(str(outcome), -20.0))
        credit = reward
        updated = 0
        seen: set[tuple[str, str]] = set()
        for state_key, action_key in reversed(self.episode_trace):
            transition = (state_key, action_key)
            if transition in seen:
                continue
            seen.add(transition)
            actions = self.q_table.setdefault(state_key, {})
            old = float(actions.get(action_key, 0.0))
            actions[action_key] = round(
                old + self.learning_rate * (credit - old), 4
            )
            credit *= 0.94
            updated += 1
        self.total_reward += reward
        verified_progress = (
            self.episode_verified_completions > 0
            or str(outcome) == "colony_ready"
        )
        self.attempt_count += 1
        if verified_progress:
            self.exploration_age += 1
        self.last_outcome = {
            "type": "episode_terminal",
            "outcome": str(outcome),
            "reward": reward,
            "elapsed_days": round(max(0.0, float(elapsed_days)), 3),
            "updated_transitions": updated,
            "verified_completions": self.episode_verified_completions,
            "exploration_aged": verified_progress,
        }
        self.episode_trace.clear()
        self.episode_capacity_high_water.clear()
        self.episode_verified_completions = 0
        self.commitment = None
        return updated

    def dump(self) -> dict:
        payload = {
            state: dict(actions)
            for state, actions in self.q_table.items()
            if state != META_KEY
        }
        payload[META_KEY] = {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "attempt_count": int(self.attempt_count),
            "exploration_age": int(self.exploration_age),
            "total_reward": round(float(self.total_reward), 6),
            "decision_count": int(self.decision_count),
        }
        return payload

    def load(self, payload: dict) -> None:
        metadata = dict(payload.get(META_KEY, {}) or {})
        if int(metadata.get("state_schema_version", 0)) != STATE_SCHEMA_VERSION:
            # Observation keys are part of the experiment definition.  A
            # policy trained against an older state encoding cannot be mixed
            # with the new one: its Q rows are unreachable while a restored
            # low epsilon would suppress the exploration needed to relearn.
            self.q_table = {}
            self.attempt_count = 0
            self.exploration_age = 0
            self.total_reward = 0.0
            self.decision_count = 0
            self.last_outcome = {
                "type": "incompatible_state_schema_reset",
                "loaded_version": metadata.get("state_schema_version"),
                "required_version": STATE_SCHEMA_VERSION,
            }
            return
        self.q_table = {
            str(state): {
                str(action): float(value)
                for action, value in dict(actions).items()
            }
            for state, actions in payload.items()
            if state != META_KEY and isinstance(actions, dict)
        }
        self.attempt_count = max(0, int(metadata.get("attempt_count", 0)))
        self.exploration_age = max(
            0, int(metadata.get("exploration_age", 0))
        )
        self.total_reward = float(metadata.get("total_reward", 0.0))
        self.decision_count = max(0, int(metadata.get("decision_count", 0)))

    def telemetry(self) -> dict:
        current = self.commitment
        return {
            "planet_id": self.planet_id,
            "state_schema_version": STATE_SCHEMA_VERSION,
            "attempt_count": self.attempt_count,
            "exploration_age": self.exploration_age,
            "epsilon": round(self.epsilon, 4),
            "learned_states": len(self.q_table),
            "total_reward": round(self.total_reward, 3),
            "current_objective": current.recipe if current else None,
            "current_action": current.action_key if current else None,
            "last_outcome": dict(self.last_outcome) if self.last_outcome else None,
        }
