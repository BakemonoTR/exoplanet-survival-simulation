"""Small pre-terminal strategic/BOM diagnostic for one deterministic run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("GROQ_API_KEY", "offline-strategy-diagnostic")

from src.agents.agent import create_team_from_presets
from src.memory import vector_store
from src.orchestration.engine import SimulationEngine


ROOT = Path(__file__).resolve().parent
vector_store._use_tfidf_fallback = True
CAPACITY_RECIPES = (
    "solar_panel", "isru_o2_unit", "water_collector", "water_purifier",
    "greenhouse", "habitat_module",
    "power_distribution_grid", "communications_array", "landing_zone",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planet", default="teegardens-star-b")
    parser.add_argument("--ticks", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engine = SimulationEngine(
        str(ROOT / "config" / "planets" / f"{args.planet}.json"),
        seed=args.seed,
        max_ticks=args.ticks,
        db=False,
    )
    for agent in create_team_from_presets(
        str(ROOT / "config" / "agent_presets.json")
    )[:6]:
        engine.add_agent(agent)
    while engine.current_tick < args.ticks:
        engine._run_tick()
        engine.current_tick += 1

    decision = engine.decision_engine
    pooled = decision._pooled_materials()
    feasibility = {}
    for recipe in CAPACITY_RECIPES:
        if recipe not in decision._recipes_cache:
            continue
        check = decision._bootstrap_order_feasibility(recipe, pooled)
        feasibility[recipe] = {
            "actionable": bool(check.get("actionable", False)),
            "blocked_resources": check.get("blocked_resources", []),
            "raw_bom_deficits": decision._raw_bom_deficits(recipe, pooled),
        }
    print(json.dumps({
        "tick": engine.current_tick,
        "alive": sum(
            getattr(a.status, "value", str(a.status)) != "dead"
            for a in engine.agents
        ),
        "structures": engine.structures_built,
        "colony_score": engine.colony_score.to_dict(),
        "agent_actions": {
            agent.name: {
                "action": agent.action.action_type,
                "target": agent.action.target,
                "energy": round(agent.needs.energy, 1),
                "status": getattr(agent.status, "value", str(agent.status)),
                "death_cause": (
                    getattr(agent.death_cause, "value", str(agent.death_cause))
                    if agent.death_cause else None
                ),
            }
            for agent in engine.agents
        },
        "fleet_summary": {
            "service_modules_used": engine.surface_fleet.service_modules_used,
            "excavators": [
                {
                    "id": vehicle.id,
                    "state": vehicle.state,
                    "condition": round(vehicle.condition, 3),
                    "jobs": vehicle.completed_jobs,
                    "excavated_kg": round(vehicle.total_excavated_kg, 1),
                }
                for vehicle in engine.surface_fleet.excavators
            ],
            "assembly_robots": [
                {
                    "id": vehicle.id,
                    "state": vehicle.state,
                    "condition": round(vehicle.condition, 3),
                    "work_hours": round(vehicle.total_assembly_work_hours, 1),
                }
                for vehicle in engine.surface_fleet.assembly_robots
            ],
        },
        "quality": {
            "commissioned_with_acceptance": sum(
                bool(site.get("acceptance_test"))
                and not site.get("under_construction", False)
                for site in engine.placed_structures
            ),
            "rework_attempts": sum(
                max(0, int(site.get("acceptance_attempts", 0)) - 1)
                for site in engine.placed_structures
            ),
        },
        "mission": engine._mission_state(),
        "support_soak_ticks": engine._support_soak_ticks,
        "shared_work_order": decision.shared_work_order,
        "strategic": engine.strategic_policy.telemetry(),
        "active_sites": [
            {
                "type": site.get("type"),
                "progress": site.get("progress"),
                "ticks_remaining": site.get("ticks_remaining"),
            }
            for site in engine.placed_structures
            if site.get("under_construction", False)
            and not site.get("destroyed", False)
        ],
        "feasibility": feasibility,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
