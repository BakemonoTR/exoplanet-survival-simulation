"""Replay the sixth learning episode and trace its first construction stall.

This is a read-only simulation diagnostic: it loads the policy snapshots saved
before train_6 and writes observations to a separate report directory.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np

from campaign_audit import ROOT, PLANET_DIR, _offline_llm
from src.agents.agent import create_team_from_presets
from src.memory import vector_store
from src.orchestration.engine import SimulationEngine


def json_copy(value):
    return json.loads(json.dumps(value, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=20000)
    parser.add_argument("--detail-start", type=int, default=15730)
    parser.add_argument("--detail-end", type=int, default=15750)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("reports/rl_learning_verification_final_20260915/training/train_6"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/diagnose_train6_stall_20260916"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    random.seed(42)
    np.random.seed(42)
    logging.basicConfig(level=logging.ERROR)
    vector_store._use_tfidf_fallback = True
    events = []

    def on_event(event):
        if event.get("type") in {
            "construction_complete", "utility_route_segment_installed",
            "utility_route_installed", "structure_collapse",
        }:
            events.append(json_copy(event))

    engine = SimulationEngine(
        str(PLANET_DIR / "kepler-442b.json"), seed=42,
        max_ticks=args.ticks, db=False, on_event=on_event,
        strategic_deadline_learning=True,
    )
    engine.llm_client.call = _offline_llm
    engine.llm_client.send = _offline_llm

    tactical = json.loads(
        (args.checkpoint / "tactical_policy_start.json").read_text(encoding="utf-8")
    )
    for agent in create_team_from_presets(str(ROOT / "config/agent_presets.json"))[
        : engine.mission_profile.advance_crew
    ]:
        engine.add_agent(agent)
        agent.q_table = copy.deepcopy(tactical[agent.id])
    engine.strategic_policy.load(json.loads(
        (args.checkpoint / "policy_start.json").read_text(encoding="utf-8")
    ))

    trace_path = args.output / "stall_trace.jsonl"
    trace = trace_path.open("w", encoding="utf-8")
    last_site_state = None
    last_progress_tick = None
    last_snapshot_tick = -1

    original_start = engine._start_construction_rover_trip
    def traced_start(agent, site):
        before = (engine.current_tick, agent.id, agent.action.action_type)
        result = original_start(agent, site)
        if args.detail_start <= engine.current_tick <= args.detail_end:
            trace.write(json.dumps({
                "kind": "rover_start_attempt", "tick": engine.current_tick,
                "agent": agent.id, "site": site.get("id"), "result": result,
                "before": before, "action": json_copy(vars(agent.action)),
                "preparation": json_copy(getattr(agent, "_construction_preparation", None)),
                "expedition": json_copy(getattr(agent, "_active_expedition", None)),
            }) + "\n")
        return result
    engine._start_construction_rover_trip = traced_start

    original_tick = engine._run_tick
    def snapshot(site, reason):
        workfront = engine._construction_workfront(site)
        order = copy.deepcopy(getattr(engine.decision_engine, "shared_work_order", {}))
        commitment = getattr(engine.strategic_policy, "commitment", None)
        return {
            "kind": "snapshot", "reason": reason, "tick": engine.current_tick,
            "score": engine.colony_score.to_dict()["overall"],
            "site": {key: copy.deepcopy(site.get(key)) for key in (
                "id", "type", "x", "y", "construction_phase", "progress",
                "materials_committed", "materials_delivered_kg",
                "cargo_delivery_completed_tick", "work_hours_completed",
                "required_work_hours", "utility_route_segments_completed",
                "utility_route_total_segments", "utility_route_work_hours_completed",
                "utility_route_work_hours", "active_builder_count",
                "active_assembly_robot_count", "assembly_robot_equivalent_workers",
            )},
            "workfront": workfront,
            "workfront_pressurized": engine._is_pressurized_location(*workfront),
            "order": order,
            "commitment": json_copy(vars(commitment)) if commitment else None,
            "crew": [{
                "id": a.id, "pos": [a.x, a.y], "inside": a._in_habitat,
                "status": str(getattr(a.status, "value", a.status)),
                "needs": {k: round(float(getattr(a.needs, k)), 3) for k in (
                    "energy", "hunger", "thirst", "o2_supply", "temperature_stress"
                )},
                "action": json_copy(vars(a.action)),
                "expedition": json_copy(getattr(a, "_active_expedition", None)),
                "preparation": json_copy(getattr(a, "_construction_preparation", None)),
                "paused_manufacturing": json_copy(getattr(a, "_paused_manufacturing", None)),
                "contract": json_copy(getattr(a, "_shared_work_contract", None)),
            } for a in engine.agents],
            "rovers": [json_copy(vars(r)) for r in engine.surface_fleet.crew_rovers],
        }

    def audited_tick():
        nonlocal last_site_state, last_progress_tick, last_snapshot_tick
        result = original_tick()
        active = next((s for s in engine.placed_structures
                       if s.get("under_construction") and not s.get("destroyed")), None)
        if active is None:
            return result
        state = (
            active.get("id"), active.get("construction_phase"),
            int(active.get("utility_route_segments_completed", 0)),
            round(float(active.get("work_hours_completed", 0.0)), 9),
        )
        changed = state != last_site_state
        if changed:
            last_site_state = state
            last_progress_tick = engine.current_tick
        stalled = last_progress_tick is not None and engine.current_tick - last_progress_tick >= 36
        periodic = stalled and engine.current_tick - last_snapshot_tick >= 36
        detailed = args.detail_start <= engine.current_tick <= args.detail_end
        if changed or periodic or detailed:
            trace.write(json.dumps(snapshot(active, "progress" if changed else "stalled")) + "\n")
            trace.flush()
            last_snapshot_tick = engine.current_tick
        return result

    engine._run_tick = audited_tick
    report = engine.run_headless(ticks=args.ticks)
    trace.close()
    summary = {
        "ticks": engine.current_tick,
        "score": engine.colony_score.to_dict(),
        "structures": dict(engine.structures_built),
        "events": events,
        "active_sites": [json_copy(s) for s in engine.placed_structures
                         if s.get("under_construction") and not s.get("destroyed")],
        "end_reason": report.get("end_reason"),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "ticks": engine.current_tick,
        "score": summary["score"]["overall"],
        "active_sites": [(s.get("id"), s.get("type"), s.get("work_hours_completed"))
                         for s in summary["active_sites"]],
        "trace": str(trace_path),
    }))


if __name__ == "__main__":
    main()
