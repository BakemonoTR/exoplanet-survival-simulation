"""Offline 5-planet x 15-attempt campaign audit.

The production physics loop is used without wall-clock pacing or language
model calls.  Each planet keeps its own strategic and tactical Q-tables across
attempts while the physical site and crew are reset after every failure.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
PLANET_DIR = ROOT / "config" / "planets"
DEFAULT_PLANETS = tuple(sorted(path.stem for path in PLANET_DIR.glob("*.json")))
DELIVERED_BASELINE = {
    "eclss_lander_hub",
    "stone_furnace",
    "forge",
    "cnc_fabricator",
}

os.environ.setdefault("GROQ_API_KEY", "offline-campaign-audit")
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.agents.agent import create_team_from_presets  # noqa: E402
from src.memory import vector_store  # noqa: E402
from src.orchestration.engine import SimulationEngine  # noqa: E402
from src.systems.mission_profile import MissionProfile  # noqa: E402


def _offline_llm(*_args: Any, **_kwargs: Any) -> None:
    return None


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 3) if values else 0.0


def _field_structures(structures: dict[str, int]) -> dict[str, int]:
    return {
        name: int(count)
        for name, count in sorted(structures.items())
        if name not in DELIVERED_BASELINE and int(count) > 0
    }


def _attempt(
    planet: str,
    attempt: int,
    *,
    seed: int,
    max_ticks: int,
    strategic_policy: dict[str, Any] | None,
    tactical_policies: dict[str, dict[str, dict[str, float]]],
    event_log_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, dict[str, float]]]]:
    random.seed(seed)
    np.random.seed(seed)
    # Keep every event for post-run forensic analysis, but stream it to disk.
    # Holding a 360-day event trace in one Python list makes the audit itself
    # consume hundreds of MB and changes the timing characteristics we are
    # trying to measure.  Construction completions remain in memory because
    # the compact campaign summary needs only those records.
    completion_events: list[dict[str, Any]] = []
    event_log = None
    event_count = 0
    if event_log_path is not None:
        event_log_path.parent.mkdir(parents=True, exist_ok=True)
        event_log = event_log_path.open("w", encoding="utf-8")

    def record_event(event: dict[str, Any]) -> None:
        nonlocal event_count
        payload = dict(event)
        event_count += 1
        if event_log is not None:
            # default=str makes a diagnostic journal robust to a future event
            # carrying an Enum or other display-only object.
            event_log.write(json.dumps(payload, ensure_ascii=False, default=str))
            event_log.write("\n")
        if payload.get("type") == "construction_complete":
            completion_events.append(payload)

    engine = SimulationEngine(
        str(PLANET_DIR / f"{planet}.json"),
        seed=seed,
        max_ticks=max_ticks,
        db=False,
        on_event=record_event,
    )
    engine.llm_client.call = _offline_llm
    if hasattr(engine.llm_client, "send"):
        engine.llm_client.send = _offline_llm
    if strategic_policy:
        engine.strategic_policy.load(copy.deepcopy(strategic_policy))

    agents = create_team_from_presets(str(ROOT / "config" / "agent_presets.json"))
    crew_count = int(engine.mission_profile.advance_crew)
    if len(agents) < crew_count:
        raise ValueError(
            f"Mission requires {crew_count} crew but only {len(agents)} presets exist"
        )
    for index, agent in enumerate(agents[:crew_count]):
        # Preset IDs participate in route tie-breaks and duty assignments.
        # Planet isolation is already supplied by the outer policy maps.
        agent.q_table = copy.deepcopy(tactical_policies.get(agent.id, {}))
        engine.add_agent(agent)

    strategic_before = engine.strategic_policy.dump()
    tactical_before = {
        agent.id: copy.deepcopy(agent.q_table) for agent in engine.agents
    }
    engine_error: dict[str, Any] | None = None
    action_ticks = Counter()
    preparation_waits = Counter()
    outdoor_recovery_ticks = 0
    processing_errors = []
    run_tick = engine._run_tick

    def audited_tick():
        nonlocal outdoor_recovery_ticks
        events = run_tick()
        processing_errors.extend(events.get("agent_processing_errors", []))
        for crew in engine.agents:
            action_ticks[str(crew.action.action_type)] += 1
            if not crew._in_habitat and crew.action.action_type in {"sleep", "service_suit"}:
                outdoor_recovery_ticks += 1
            target = crew.action.target if isinstance(crew.action.target, dict) else {}
            if target.get("construction_rover_preparation"):
                preparation_waits[str(target.get("reason"))] += 1
        return events

    engine._run_tick = audited_tick
    started = time.perf_counter()
    try:
        try:
            report = engine.run_headless(ticks=max_ticks)
        except Exception as exc:  # Keep the matrix running and expose engine crashes.
            engine_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "tick": int(engine.current_tick),
            }
            report = {
                "end_reason": "engine_error",
                "total_ticks": int(engine.current_tick),
                "sim_days": round(
                    engine.current_tick
                    / engine.mission_profile.clock.ticks_per_earth_day,
                    3,
                ),
                "colony_score": engine.colony_score.to_dict(),
                "agents_alive": sum(
                    getattr(agent.status, "value", str(agent.status)) != "dead"
                    for agent in engine.agents
                ),
                "agents_dead": sum(
                    getattr(agent.status, "value", str(agent.status)) == "dead"
                    for agent in engine.agents
                ),
                "deaths": [],
                "structures": dict(engine.structures_built),
                "mission": {},
                "strategic_rl": engine.strategic_policy.telemetry(),
            }
    finally:
        if event_log is not None:
            event_log.close()
    wall_seconds = time.perf_counter() - started

    structures = {
        name: int(count) for name, count in report.get("structures", {}).items()
    }
    deaths = [
        {
            "name": death.get("name"),
            "cause": death.get("cause"),
            "tick": death.get("tick"),
        }
        for death in report.get("deaths", [])
    ]
    result = {
        "planet": planet,
        "attempt": attempt,
        "seed": seed,
        "end_reason": report.get("end_reason"),
        "ticks": int(report.get("total_ticks", 0)),
        "sim_days": float(report.get("sim_days", 0.0)),
        "score_percent": round(
            float(report.get("colony_score", {}).get("overall", 0.0)), 3
        ),
        "agents_alive": int(report.get("agents_alive", 0)),
        "agents_dead": int(report.get("agents_dead", 0)),
        "death_causes": dict(Counter(str(item.get("cause")) for item in deaths)),
        "deaths": deaths,
        "field_structures": _field_structures(structures),
        "field_structure_count": sum(_field_structures(structures).values()),
        "all_structures": structures,
        "construction_completions": [
            {
                "tick": int(event.get("tick", 0)),
                "cause": str(event.get("cause", "")),
            }
            for event in completion_events
        ],
        "communications_online": bool(
            report.get("mission", {}).get("communications_operational", False)
        ),
        "water_reserve_l": round(
            float(engine._colony_resources.get("water_reserve_l", 0.0)), 3
        ),
        "o2_reserve_kg": round(
            float(engine._colony_resources.get("o2_reserve_kg", 0.0)), 3
        ),
        "food_reserve_kcal": round(
            float(engine._colony_resources.get("food_reserve_kcal", 0.0)), 1
        ),
        "strategic_rl": dict(report.get("strategic_rl", {})),
        "final_shared_work_order": copy.deepcopy(
            getattr(engine.decision_engine, "shared_work_order", {})
        ),
        "active_construction_sites": [
            {
                "type": site.get("type"),
                "progress": round(float(site.get("progress", 0.0)), 4),
                "ticks_remaining": site.get("ticks_remaining"),
                "materials_committed": bool(
                    site.get("materials_committed", False)
                ),
            }
            for site in getattr(engine, "placed_structures", [])
            if site.get("under_construction", False)
            and not site.get("destroyed", False)
        ],
        "tactical_q_states": {
            agent.name: len(agent.q_table) for agent in engine.agents
        },
        "engine_error": engine_error,
        "agent_processing_errors": processing_errors,
        "action_ticks": dict(action_ticks),
        "construction_preparation_wait_ticks": dict(preparation_waits),
        "outdoor_routine_recovery_ticks": outdoor_recovery_ticks,
        "airlock": engine.airlock.snapshot(),
        "final_agents": [
            {"id": a.id, "position": [a.x, a.y], "in_habitat": a._in_habitat,
             "action": a.action.to_dict(), "needs": a.needs.to_dict(),
             "spare_o2_cylinders": a.inventory.items.get("oxygen_canisters", 0),
             "preparing_construction_sortie": bool(getattr(a, "_construction_preparation", None))}
            for a in engine.agents
        ],
        "event_count": event_count,
        "event_log": str(event_log_path) if event_log_path is not None else None,
        "wall_seconds": round(wall_seconds, 3),
    }
    if engine_error:
        return result, strategic_before, tactical_before
    next_tactical = {
        agent.id: copy.deepcopy(agent.q_table) for agent in engine.agents
    }
    return result, engine.strategic_policy.dump(), next_tactical


def _planet_summary(planet: str, attempts: list[dict[str, Any]]) -> dict[str, Any]:
    first = attempts[:5]
    last = attempts[-5:]
    return {
        "planet": planet,
        "attempts": len(attempts),
        "successes": sum(row["end_reason"] == "colony_ready" for row in attempts),
        "timeouts": sum(row["end_reason"] == "timeout" for row in attempts),
        "crew_extinctions": sum(
            row["end_reason"] == "all_agents_dead" for row in attempts
        ),
        "engine_errors": sum(row["end_reason"] == "engine_error" for row in attempts),
        "best_score_percent": max((row["score_percent"] for row in attempts), default=0.0),
        "mean_score_percent": _mean([row["score_percent"] for row in attempts]),
        "first5_mean_score_percent": _mean([row["score_percent"] for row in first]),
        "last5_mean_score_percent": _mean([row["score_percent"] for row in last]),
        "best_survival_days": max((row["sim_days"] for row in attempts), default=0.0),
        "first5_mean_survival_days": _mean([row["sim_days"] for row in first]),
        "last5_mean_survival_days": _mean([row["sim_days"] for row in last]),
        "attempts_with_any_field_structure": sum(
            row["field_structure_count"] > 0 for row in attempts
        ),
        "best_field_structure_count": max(
            (row["field_structure_count"] for row in attempts), default=0
        ),
        "field_structure_types": dict(Counter(
            name
            for row in attempts
            for name, count in row["field_structures"].items()
            for _ in range(count)
        )),
        "death_causes": dict(Counter(
            cause
            for row in attempts
            for cause, count in row["death_causes"].items()
            for _ in range(count)
        )),
        "final_strategic_attempt_count": int(
            attempts[-1].get("strategic_rl", {}).get("attempt_count", 0)
            if attempts else 0
        ),
        "final_strategic_epsilon": float(
            attempts[-1].get("strategic_rl", {}).get("epsilon", 0.0)
            if attempts else 0.0
        ),
    }


def run_campaign(
    planets: list[str], *, attempts_per_planet: int, max_ticks: int, seed: int,
    event_log_dir: Path | None = None,
) -> dict[str, Any]:
    vector_store._use_tfidf_fallback = True
    logging.disable(logging.CRITICAL)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for planet_index, planet in enumerate(planets):
            strategic_policy: dict[str, Any] | None = None
            tactical_policies: dict[str, dict[str, dict[str, float]]] = {}
            for attempt in range(1, attempts_per_planet + 1):
                attempt_seed = seed + planet_index * 100_000 + attempt - 1
                row, strategic_policy, tactical_policies = _attempt(
                    planet,
                    attempt,
                    seed=attempt_seed,
                    max_ticks=max_ticks,
                    strategic_policy=strategic_policy,
                    tactical_policies=tactical_policies,
                    event_log_path=(
                        event_log_dir
                        / f"events_{planet}_attempt_{attempt:02}.jsonl"
                        if event_log_dir is not None else None
                    ),
                )
                rows.append(row)
                print(
                    f"{planet} {attempt:02d}/{attempts_per_planet}: "
                    f"{row['end_reason']} t={row['ticks']} "
                    f"score={row['score_percent']:.2f}% "
                    f"field={row['field_structure_count']}"
                )
    finally:
        logging.disable(logging.NOTSET)

    summaries = [
        _planet_summary(planet, [row for row in rows if row["planet"] == planet])
        for planet in planets
    ]
    return {
        "meta": {
            "planets": planets,
            "attempts_per_planet": attempts_per_planet,
            "attempt_cap_ticks": max_ticks,
            "tick_minutes": 10,
            "advance_crew": MissionProfile.load().advance_crew,
            "base_seed": seed,
            "seed_schedule": "base + planet_index*100000 + attempt_index",
            "learning_scope": "planet-scoped; physical state reset per attempt",
            "language_model": "disabled",
            "wall_seconds": round(time.perf_counter() - started, 3),
        },
        "summary": {
            "total_attempts": len(rows),
            "successes": sum(row["end_reason"] == "colony_ready" for row in rows),
            "timeouts": sum(row["end_reason"] == "timeout" for row in rows),
            "crew_extinctions": sum(
                row["end_reason"] == "all_agents_dead" for row in rows
            ),
            "engine_errors": sum(row["end_reason"] == "engine_error" for row in rows),
            "attempts_with_any_field_structure": sum(
                row["field_structure_count"] > 0 for row in rows
            ),
            "best_score_percent": max((row["score_percent"] for row in rows), default=0.0),
        },
        "planets": summaries,
        "attempts": rows,
    }


def render_markdown(result: dict[str, Any]) -> str:
    meta = result["meta"]
    total = result["summary"]
    lines = [
        f"# {len(meta['planets'])} Gezegen × {meta['attempts_per_planet']} Deneme / {meta['advance_crew']} Kişilik Ekip",
        "",
        f"- Deneme: {total['total_attempts']} (gezegen başına {meta['attempts_per_planet']})",
        f"- Deneme üst sınırı: {meta['attempt_cap_ticks']} tick / "
        f"{meta['attempt_cap_ticks'] * meta['tick_minutes'] / 1440:.1f} simülasyon günü",
        f"- LLM: {meta['language_model']}",
        f"- Başarı: {total['successes']}",
        f"- Ekip kaybı: {total['crew_extinctions']}",
        f"- Motor hatası: {total['engine_errors']}",
        f"- Süre aşımı: {total['timeouts']}",
        f"- Herhangi bir saha yapısı tamamlanan deneme: {total['attempts_with_any_field_structure']}",
        f"- En iyi hazırlık skoru: %{total['best_score_percent']:.3f}",
        f"- Duvar saati: {meta['wall_seconds']:.1f} saniye",
        "",
        "Aynı gezegendeki denemeler Q tablolarını devralır; fiziksel saha, ekip ve sarflar sıfırlanır. "
        "Gezegenler arasında politika aktarımı yapılmaz. Her deneme farklı fakat rapora kaydedilen "
        "deterministik bir seed kullanır; böylece politika tek bir harita/olay dizisini ezberleyemez.",
        "",
        "## Gezegen özeti",
        "",
        "| Gezegen | Başarı | Ekip kaybı | Motor hatası | En iyi skor | İlk 5 → son 5 skor | İlk 5 → son 5 gün | Yapılı deneme | En çok saha yapısı |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["planets"]:
        lines.append(
            f"| {row['planet']} | {row['successes']}/{row['attempts']} | "
            f"{row['crew_extinctions']} | {row['engine_errors']} | "
            f"%{row['best_score_percent']:.3f} | "
            f"%{row['first5_mean_score_percent']:.3f} → %{row['last5_mean_score_percent']:.3f} | "
            f"{row['first5_mean_survival_days']:.2f} → {row['last5_mean_survival_days']:.2f} | "
            f"{row['attempts_with_any_field_structure']}/{row['attempts']} | "
            f"{row['best_field_structure_count']} |"
        )

    lines.extend(["", "## Deneme dökümü", ""])
    for row in result["planets"]:
        structures = ", ".join(
            f"{name}={count}" for name, count in row["field_structure_types"].items()
        ) or "yok"
        causes = ", ".join(
            f"{name}={count}" for name, count in row["death_causes"].items()
        ) or "yok"
        lines.extend([
            f"- **{row['planet']}** — saha yapıları: {structures}; ölüm nedenleri: {causes}; "
            f"son ε={row['final_strategic_epsilon']:.4f}, RL deneme sayacı={row['final_strategic_attempt_count']}",
        ])
    return "\n".join(lines) + "\n"


def main() -> int:
    profile = MissionProfile.load()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=15)
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=profile.clock.planet_attempt_max_ticks,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planets", nargs="+", default=list(DEFAULT_PLANETS))
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reports" / "campaign_5x15"
    )
    args = parser.parse_args()
    unknown = [planet for planet in args.planets if not (PLANET_DIR / f"{planet}.json").exists()]
    if unknown:
        parser.error(f"Unknown planets: {', '.join(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_campaign(
        args.planets,
        attempts_per_planet=max(1, args.attempts),
        max_ticks=max(1, args.max_ticks),
        seed=args.seed,
        event_log_dir=args.output_dir,
    )
    json_path = args.output_dir / "campaign_audit.json"
    markdown_path = args.output_dir / "campaign_audit.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
