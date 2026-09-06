"""Deterministic 500-tick realism audit for the colony simulation.

Live LLM calls are disabled so the fallback planner and fresh RL policy can be
tested repeatably.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.agents.agent import create_team_from_presets  # noqa: E402
from src.orchestration.engine import SimulationEngine  # noqa: E402
from src.memory import vector_store  # noqa: E402
from src.systems.mission_profile import MissionProfile  # noqa: E402

DEFAULT_TICKS = 500
AUDITED_AGENTS = MissionProfile.load().advance_crew
DEFAULT_SEED = 42


def _offline_llm(*_args: Any, **_kwargs: Any) -> None:
    # The real client returns None when unavailable. This deliberately invokes
    # DecisionEngine's local fallback instead of injecting a fake idle command.
    return None


def _normalise_target(target: Any) -> str:
    if not isinstance(target, dict):
        return str(target or "")
    useful = {
        key: target[key]
        for key in (
            "recipe", "resource", "x", "y", "dx", "dy", "structure",
            "destination", "expedition", "maintenance", "survey_action",
            "eva_denied", "empty_deposit_rejected", "output",
            "capacity_recipe", "reason",
        )
        if key in target
    }
    return json.dumps(useful, sort_keys=True, ensure_ascii=False)


def _coordination_capacity(
    engine: SimulationEngine, action: str, normalised_target: str
) -> int:
    """Return the physically defensible crew capacity for one shared target.

    A fixed ``four astronauts == copied task`` rule mislabeled a four-person
    ISRU commissioning crew as weak specialisation. Capacity is instead tied
    to the recipe or physical machine count. The audit still catches excess
    staffing; it simply stops treating intended teamwork as duplication.
    """
    try:
        target = json.loads(normalised_target) if normalised_target else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        target = {}
    if not isinstance(target, dict):
        target = {}

    if action == "build":
        recipe_name = target.get("recipe")
        recipe = engine._get_recipe(str(recipe_name)) if recipe_name else None
        construction = (recipe or {}).get("construction", {})
        return max(1, int(construction.get("recommended_crew", 2)))

    if action == "refine":
        output = target.get("output")
        recipe = engine._get_recipe(str(output)) if output else None
        machine_type = (recipe or {}).get("requires_structure")
        if machine_type:
            return max(1, int(engine._operational_structure_count(
                str(machine_type)
            )))
        return 1

    if action == "move":
        if target.get("destination") in {
            "habitat", "shelter", "central_depot", "decontamination"
        }:
            # A common return or several independent payload deliveries are
            # coordinated logistics, not duplicate use of one work station.
            return max(1, len(engine.agents))
        if target.get("expedition"):
            return 2
        recipe_name = target.get("recipe")
        recipe = engine._get_recipe(str(recipe_name)) if recipe_name else None
        if recipe:
            return max(1, int(
                recipe.get("construction", {}).get("recommended_crew", 2)
            ))
        if target.get("destination") == "manufacturing_machine":
            return 1
        if target.get("destination") in {
            "extraction_face", "active_excavation_face",
            "prospect_station", "survey_station",
        }:
            return 2
        # Sharing a corridor, base waypoint or return coordinate is not use of
        # one finite workstation. Crowding is assessed at the destination's
        # productive action, not while independent people are in transit.
        return max(1, len(engine.agents))

    # Two-person handling, buddy work, rescue and inspection are defensible;
    # persistent crowding beyond that remains a coordination defect.
    if action in {
        "gather", "repair", "clean_solar_panels", "prospect",
        "survey_resources", "deposit_materials", "rescue", "treat",
    }:
        return 2
    return 1


def _is_repeated_short_sortie_pattern(short_count: int, total: int) -> bool:
    """Flag a systematic airlock loop, not rare legitimate task aborts."""
    return short_count >= 3 and short_count / max(1, total) >= 0.03


def _is_sustained_coordination_overage(
    copied_ticks: int, total_ticks: int
) -> bool:
    """Require persistent overstaffing across both short and long audits."""
    return copied_ticks >= max(20, math.ceil(total_ticks * 0.005))


def _is_thermally_unsafe_field_sleep(record: dict[str, Any]) -> bool:
    """Separate dangerous cold/thermal depletion from small reserve drift."""
    return (
        float(record.get("effective_temperature_c", 20.0)) < 5.0
        or float(record.get("temperature_stress", 50.0)) < 30.0
    )


def _action_runs(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    for record in records:
        if not runs or runs[-1][-1]["action"] != record["action"]:
            runs.append([record])
        else:
            runs[-1].append(record)
    return runs


def _finding(
    code: str,
    severity: str,
    title: str,
    evidence: str,
    recommendation: str,
    agent: str | None = None,
    tick_range: str | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "title": title,
        "agent": agent,
        "tick_range": tick_range,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def _analyse(
    engine: SimulationEngine,
    records: list[dict[str, Any]],
    processing_failures: list[dict[str, Any]],
    notable_events: list[dict[str, Any]],
    production_samples: list[dict[str, Any]],
    ticks_requested: int,
    seed: int,
) -> dict[str, Any]:
    tick_minutes = float(engine.mission_profile.clock.tick_minutes)
    ticks_per_hour = max(1, round(60.0 / tick_minutes))
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_tick: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_agent[record["agent"]].append(record)
        by_tick[record["tick"]].append(record)

    findings: list[dict[str, Any]] = []
    agent_summaries: dict[str, Any] = {}

    if processing_failures:
        examples = "; ".join(
            f"tick {row.get('tick')} {row.get('agent')}: "
            f"{row.get('exception_type', 'unknown')}({row.get('message', 'no detail')})"
            for row in processing_failures[:3]
        )
        findings.append(_finding(
            "ENGINE_AGENT_TICK_FAILURE", "critical",
            "Motor bazı ajan ticklerini işleyemedi",
            f"{len(processing_failures)} ajan-tick işlenemedi. İlk kayıtlar: {examples}",
            "_process_agent_tick istisnalarını test ortamında yeniden fırlatın.",
        ))

    for agent_name, agent_records in by_agent.items():
        action_counts = Counter(r["action"] for r in agent_records)
        move_records = [r for r in agent_records if r["action"] == "move"]
        positions = [(r["x"], r["y"]) for r in agent_records]
        lz_x = getattr(engine, "lz_x", 1000)
        lz_y = getattr(engine, "lz_y", 1000)
        radial_distances = [
            max(abs(x - lz_x), abs(y - lz_y)) for x, y in positions
        ]
        max_base_distance = max(radial_distances, default=0)
        distance = sum(
            abs(positions[i][0] - positions[i - 1][0])
            + abs(positions[i][1] - positions[i - 1][1])
            for i in range(1, len(positions))
        )
        runs = _action_runs(agent_records)
        longest_runs: dict[str, int] = defaultdict(int)
        idle_runs: list[list[dict[str, Any]]] = []
        movement_loops: list[tuple[list[dict[str, Any]], int, float]] = []
        collapse_onsets: list[dict[str, Any]] = []
        collapse_recoveries: list[dict[str, Any]] = []
        sleep_incapacitated_edges: list[dict[str, Any]] = []
        sorties: list[list[dict[str, Any]]] = []
        active_sortie: list[dict[str, Any]] = []
        medical_emergency_events = [
            event for event in notable_events
            if event.get("type") == "team_medical_emergency"
            and event.get("agent") == agent_name
        ]

        for previous, current in zip(agent_records, agent_records[1:]):
            if previous["status"] != "incapacitated" and current["status"] == "incapacitated":
                collapse_onsets.append(current)
            if previous["status"] == "incapacitated" and current["status"] != "incapacitated":
                collapse_recoveries.append(current)
            if {previous["action"], current["action"]} == {"sleep", "unconscious"}:
                sleep_incapacitated_edges.append({
                    "tick": current["tick"],
                    "from": previous["action"],
                    "to": current["action"],
                    "in_habitat": current["in_habitat"],
                    "energy": current["energy"],
                    "hunger": current["hunger"],
                    "thirst": current["thirst"],
                    "o2": current["o2"],
                    "temperature_stress": current["temperature_stress"],
                })

        for record in agent_records:
            # A pressurized CNC/workshop is not a return through the lander
            # airlock. Segment sorties only at the actual 3x3 base habitat;
            # otherwise a legitimate route through a workshop footprint looks
            # like repeated one-tick EVA exits.
            in_base_habitat = record.get(
                "in_base_habitat", record["in_habitat"]
            )
            if not in_base_habitat:
                active_sortie.append(record)
            elif active_sortie:
                sorties.append(active_sortie)
                active_sortie = []
        completed_sorties = list(sorties)
        if active_sortie:
            sorties.append(active_sortie)
        next_record_by_tick = {
            previous["tick"]: current
            for previous, current in zip(agent_records, agent_records[1:])
        }

        productive_field_actions = {
            "gather", "build", "repair", "clean_solar_panels", "prospect",
            "survey_resources", "rescue", "treat", "refill_o2", "mine",
        }
        short_unproductive_sorties = [
            sortie for sortie in completed_sorties
            if len(sortie) <= 3
            and not any(
                record["action"] in productive_field_actions for record in sortie
            )
        ]
        if _is_repeated_short_sortie_pattern(
            len(short_unproductive_sorties), len(completed_sorties)
        ):
            examples = ", ".join(
                f"{sortie[0]['tick']}-{sortie[-1]['tick']} "
                f"actions={'/'.join(record['action'] for record in sortie)} "
                f"energy={sortie[0]['energy']:.0f}->{sortie[-1]['energy']:.0f} "
                f"o2={sortie[0]['o2']:.0f}->{sortie[-1]['o2']:.0f} "
                f"decision={sortie[0].get('decision_action', '?')} "
                f"reason={sortie[0].get('decision_reasoning', '')[:90]} "
                f"post={next_record_by_tick.get(sortie[-1]['tick'], {}).get('decision_reasoning', '')[:90]}"
                for sortie in short_unproductive_sorties[:5]
            )
            findings.append(_finding(
                "SHORT_UNPRODUCTIVE_SORTIES", "warning",
                "Ajan airlocktan çıkıp iş yapmadan geri döndü",
                f"{len(short_unproductive_sorties)} sonuçsuz kısa EVA; ilk aralıklar: {examples}.",
                "Görev hedefini ve fizyolojik gidiş-dönüş rezervini airlock açılmadan doğrulayın.",
                agent_name,
            ))

        prospect_positions = []
        for record in agent_records:
            if record["action"] != "prospect":
                continue
            position = (record["x"], record["y"])
            if not prospect_positions or position != prospect_positions[-1]:
                prospect_positions.append(position)
        adjacent_prospect_steps = sum(
            1 for previous, current in zip(
                prospect_positions, prospect_positions[1:]
            )
            if max(
                abs(current[0] - previous[0]), abs(current[1] - previous[1])
            ) <= 1
        )
        prospect_transition_count = max(0, len(prospect_positions) - 1)
        if (
            len(prospect_positions) >= 20
            and prospect_transition_count > 0
            and adjacent_prospect_steps / prospect_transition_count > 0.70
        ):
            findings.append(_finding(
                "PROSPECT_TRENCH_WALK", "warning",
                "Jeolojik arama her yürüyüş gridini test çukuruna çevirdi",
                f"{len(prospect_positions)} ayrı prospect noktası; ardışık noktaların "
                f"%{adjacent_prospect_steps / prospect_transition_count * 100:.0f} kadarı bitişik.",
                "Yürüyüşü MOVE olarak koruyup yalnız aralıklı, paylaşılan test istasyonlarında PROSPECT yapın.",
                agent_name,
            ))

        for run in runs:
            action = run[0]["action"]
            longest_runs[action] = max(longest_runs[action], len(run))

            # Sub-hour queue arbitration is normal workshop latency; flag only
            # an hour or more of continuous no-op behavior.
            if action == "idle" and len(run) >= ticks_per_hour:
                idle_runs.append(run)

            if action == "move" and len(run) >= ticks_per_hour:
                # A real pressure-cycle queue is not a walking loop. Require
                # at least one hour of actual movement attempts as evidence.
                walking = [r for r in run if not r.get("airlock_cycle_wait", False)]
                if len(walking) < ticks_per_hour:
                    continue
                run_positions = [(r["x"], r["y"]) for r in walking]
                net = (
                    abs(run_positions[-1][0] - run_positions[0][0])
                    + abs(run_positions[-1][1] - run_positions[0][1])
                )
                unique_ratio = len(set(run_positions)) / len(run_positions)
                if (net <= 3 and unique_ratio < 0.50) or unique_ratio < 0.35:
                    movement_loops.append((run, net, unique_ratio))

            if action == "sleep" and len(run) > ticks_per_hour * 10:
                findings.append(_finding(
                    "EXCESSIVE_SLEEP", "warning",
                    "Kesintisiz uyku fizyolojik sınırı aştı",
                    f"{len(run)} tick = {len(run) * tick_minutes / 60:.1f} saat uyku.",
                    "Uyku hedefini sirkadiyen faz, uyku borcu ve azami süreyle sonlandırın.",
                    agent_name, f"{run[0]['tick']}-{run[-1]['tick']}",
                ))

        # A no-op caused by a stale/applicability bug is irrational.  A crew
        # physically unable to staff a reserved machine after exhausting its
        # last litre of water is instead a terminal mission-resource failure;
        # it is already reported by the collapse and water-balance evidence
        # and must not be mislabeled as an unexplained idle loop.
        def is_terminal_resource_wait(run: list[dict[str, Any]]) -> bool:
            explicit_exhaustion = sum(
                1 for record in run
                if "no physical water reserve" in record.get(
                    "decision_reasoning", ""
                ).lower()
            )
            # One route-arrival frame can retain the immediately preceding
            # reasoning, so use a strong majority instead of requiring every
            # sample to carry the same diagnostic string.
            return explicit_exhaustion / max(1, len(run)) >= 0.80

        unexplained_idle_runs = [
            run for run in idle_runs if not is_terminal_resource_wait(run)
        ]
        terminal_resource_idle_ticks = sum(
            len(run) for run in idle_runs if run not in unexplained_idle_runs
        )
        if unexplained_idle_runs:
            longest = max(unexplained_idle_runs, key=len)
            ranges = ", ".join(
                f"{r[0]['tick']}-{r[-1]['tick']}"
                for r in unexplained_idle_runs[:5]
            )
            idle_decisions = Counter(
                r.get("decision_action", "unknown")
                for run in unexplained_idle_runs for r in run
            )
            idle_reasons = Counter(
                r.get("decision_reasoning", "")
                for run in unexplained_idle_runs for r in run
            )
            findings.append(_finding(
                "REPEATED_IDLE", "warning",
                "Ajan aktif görev varken art arda boş durdu",
                f"{len(unexplained_idle_runs)} idle serisi; en uzunu {len(longest)} tick. İlk aralıklar: {ranges}. "
                f"Karar kaynakları: {dict(idle_decisions.most_common(3))}; nedenler: {dict(idle_reasons.most_common(3))}. "
                f"Koloni skoru %{longest[-1]['colony_score']:.2f}.",
                "Uygulanamaz kararı bir sonraki eksik malzeme veya altyapı görevine bağlayın.",
                agent_name,
            ))

        if movement_loops:
            longest, net, unique_ratio = max(movement_loops, key=lambda item: len(item[0]))
            ranges = ", ".join(
                f"{run[0]['tick']}-{run[-1]['tick']}" for run, _, _ in movement_loops[:5]
            )
            loop_targets = Counter(r["target"] for r in longest)
            findings.append(_finding(
                "MOVEMENT_LOOP", "critical",
                "Uzun hareket dizileri anlamlı ilerleme üretmedi",
                f"{len(movement_loops)} döngü; en uzunu {len(longest)} tick, net {net} hücre, benzersiz konum oranı %{unique_ratio * 100:.0f}. "
                f"İlk aralıklar: {ranges}. Başlıca hedefler: {dict(loop_targets.most_common(3))}.",
                "Hedefe ulaşma toleransı, rota geçersizleştirme ve son-konum tabu belleği ekleyin.",
                agent_name,
            ))

        stationary_moves = 0
        for index, record in enumerate(agent_records[1:], start=1):
            if record["action"] == "move":
                previous = agent_records[index - 1]
                if (
                    (record["x"], record["y"]) == (previous["x"], previous["y"])
                    and not record.get("airlock_cycle_wait", False)
                    and not record.get("indoor_recovery_pending", False)
                ):
                    stationary_moves += 1
        if move_records and stationary_moves / len(move_records) > 0.20:
            findings.append(_finding(
                "STATIONARY_MOVE", "critical",
                "Hareket eylemlerinin önemli kısmında konum değişmedi",
                f"{len(move_records)} move tickinin {stationary_moves} tanesi (%{stationary_moves / len(move_records) * 100:.1f}) sıfır mesafe üretti.",
                "Engellenmiş hareketi başarı saymayın; birkaç denemeden sonra rota/hedef değiştirin.",
                agent_name,
            ))

        local_eva_limit = int(getattr(engine, "LOCAL_EVA_RADIUS_CELLS", 24))
        unauthorized_distance = [
            record for record in agent_records
            if record["base_distance"] > local_eva_limit
            and not record.get("expedition_active", False)
        ]
        if unauthorized_distance:
            first = unauthorized_distance[0]
            findings.append(_finding(
                "AIMLESS_BASE_DISTANCE", "critical",
                "Ajan planlı sefer olmadan yerel EVA bölgesinin dışına çıktı",
                f"Base'den azami uzaklık {max_base_distance} hücre; yerel sınır {local_eva_limit}. "
                f"İlk ihlal tick {first['tick']} konum=({first['x']},{first['y']}).",
                "Uzak hareketi kaynak hedefi, altyapı menzili ve dönüş bütçesi olan keşif seferine bağlayın.",
                agent_name, f"{first['tick']}-{unauthorized_distance[-1]['tick']}",
            ))

        over_authorized_radius = [
            record for record in agent_records
            if record["base_distance"] > record.get("authorized_eva_radius", local_eva_limit)
        ]
        if over_authorized_radius:
            first = over_authorized_radius[0]
            findings.append(_finding(
                "EXPEDITION_RADIUS_BREACH", "critical",
                "Planlı sefer yetkili altyapı menzilini aştı",
                f"Tick {first['tick']}: uzaklık={first['base_distance']}, "
                f"yetkili sınır={first['authorized_eva_radius']}.",
                "Hedefi tarama ve iletişim menzillerinin kesişiminde sınırlandırın.",
                agent_name, f"{first['tick']}-{over_authorized_radius[-1]['tick']}",
            ))

        purposeless_expedition = [
            record for record in agent_records
            if record.get("expedition_active")
            and (not record.get("expedition_resource") or not record.get("expedition_target"))
        ]
        if purposeless_expedition:
            first = purposeless_expedition[0]
            findings.append(_finding(
                "PURPOSELESS_EXPEDITION", "critical",
                "Uzak keşif seferinin somut kaynak hedefi yok",
                f"İlk örnek tick {first['tick']}; durum={first.get('expedition_status')}.",
                "Her sefere kaynak, koordinat, yetkili menzil ve dönüş bütçesi kaydedin.",
                agent_name,
            ))

        negative_return_margin = [
            record for record in agent_records
            if record.get("expedition_active")
            and record.get("expedition_return_margin_ticks") is not None
            and record["expedition_return_margin_ticks"] < 0
        ]
        if negative_return_margin:
            first = negative_return_margin[0]
            findings.append(_finding(
                "EXPEDITION_NO_RETURN_RESERVE", "critical",
                "Ajanın planlı seferden base'e dönecek PLSS zamanı kalmadı",
                f"Tick {first['tick']}: dönüş marjı {first['expedition_return_margin_ticks']} tick.",
                "Seferi gidiş-dönüş O₂, scrubber, batarya ve EVA süresi rezerviyle başlatıp erken sonlandırın.",
                agent_name,
            ))

        unsafe_work = [
            r for r in agent_records
            if r["action"] in {"move", "gather", "build", "refine"}
            and (r["energy"] < 15 or r["hunger"] < 10 or r["thirst"] < 10)
            and not (
                r["action"] == "move"
                and any(word in r.get("decision_reasoning", "").lower() for word in (
                    "retreat", "return", "shelter", "emergency", "evacuat", "airlock"
                ))
            )
        ]
        if unsafe_work:
            example = unsafe_work[0]
            findings.append(_finding(
                "WORKING_WHILE_PHYSIOLOGICALLY_CRITICAL", "critical",
                "Ajan kritik fizyolojik durumda ağır işe devam etti",
                f"{len(unsafe_work)} tick; ilk örnek tick {example['tick']} ({example['action']}, enerji={example['energy']:.1f}, açlık={example['hunger']:.1f}, susuzluk={example['thirst']:.1f}).",
                (
                    "Görev öncesi su/enerji planlamasını ve RL fizyolojik-risk "
                    "cezasını güçlendirin; susuzluk için yapay bir EVA kilidi eklemeyin."
                ),
                agent_name,
            ))

        min_extraction_radius = int(getattr(engine, "MIN_EXTRACTION_RADIUS_CELLS", 6))
        unsafe_extraction = [
            r for r in agent_records
            if r["action"] in {"gather", "mine"}
            and r["base_distance"] < min_extraction_radius
        ]
        if unsafe_extraction:
            example = unsafe_extraction[0]
            findings.append(_finding(
                "MINING_TOO_CLOSE_TO_BASE", "critical",
                "Kazı habitat ve yaşam alanına tehlikeli ölçüde yakın yapıldı",
                f"{len(unsafe_extraction)} tick; ilk örnek tick {example['tick']}, "
                f"base uzaklığı={example['base_distance']} hücre, asgari güvenli tampon={min_extraction_radius}.",
                "Kazı hedeflerini habitat temelleri ve altyapı hatlarının dışındaki çıkarım bölgesine yönlendirin.",
                agent_name,
            ))

        unsafe_field_sleep = [
            r for r in agent_records
            if r["action"] == "sleep"
            and not r["in_habitat"]
            and _is_thermally_unsafe_field_sleep(r)
        ]
        if unsafe_field_sleep:
            example = unsafe_field_sleep[0]
            findings.append(_finding(
                "COLD_FIELD_SLEEP", "critical",
                "Ajan soğuk EVA ortamında uyumaya çalıştı",
                f"{len(unsafe_field_sleep)} tick; ilk örnek tick {example['tick']}, "
                f"ortam={example['effective_temperature_c']:.1f}°C, termal rezerv={example['temperature_stress']:.1f}%.",
                "Yorgun EVA ajanını sahada uyutmak yerine dönüş rezervi varken ısıtılmış habitata gönderin.",
                agent_name,
            ))

        collapses = [r for r in agent_records if r["status"] == "incapacitated"]
        if collapse_onsets or medical_emergency_events:
            first_event = medical_emergency_events[0] if medical_emergency_events else None
            collapse_cause = (
                first_event.get("cause", "") if first_event else ""
            ).lower()
            collapse_recommendation = (
                "Zorunlu susuzluk EVA kilidi yerine ajanların taşıdığı suyu, "
                "görev süresini ve dönüş maliyetini RL kararında planlatın."
                if "dehydrat" in collapse_cause or "susuz" in collapse_cause
                else "O₂, termal ve enerji rezervlerini rota süresiyle birlikte görev öncesinde planlatın."
            )
            first = collapse_onsets[0] if collapse_onsets else min(
                agent_records,
                key=lambda record: abs(record["tick"] - int(first_event.get("tick", 0))),
            )
            first_index = agent_records.index(first)
            prelude = agent_records[max(0, first_index - 5):first_index + 1]
            prelude_text = " -> ".join(
                f"t{r['tick']}:{r['action']}/O2={r['o2']:.1f}/"
                f"E={r['energy']:.1f}/H={r['hunger']:.1f}/T={r['thirst']:.1f}/"
                f"d={r['base_distance']}/base={r['in_habitat']}"
                for r in prelude
            )
            findings.append(_finding(
                "PREVENTABLE_PHYSIOLOGICAL_COLLAPSE", "critical",
                "Ajan denetim sırasında bilincini kaybetti",
                f"{max(len(collapse_onsets), len(medical_emergency_events))} ayrı bayılma alarmı, "
                f"{len(collapses)} tick-sonu baygın kayıt. "
                f"İlk bayılma tick {first_event.get('tick', first['tick']) if first_event else first['tick']}: "
                f"neden={first_event.get('cause', 'physiological collapse') if first_event else 'physiological collapse'}, "
                f"enerji={first['energy']:.1f}, "
                f"O₂={first['o2']:.1f}, termal rezerv={first['temperature_stress']:.1f}%. "
                f"Öncesi: {prelude_text}.",
                collapse_recommendation,
                agent_name,
            ))

        if len(sleep_incapacitated_edges) >= 4:
            first_edges = ", ".join(
                f"t{edge['tick']} {edge['from']}→{edge['to']} "
                f"(base={edge['in_habitat']}, O₂={edge['o2']:.1f}, E={edge['energy']:.1f})"
                for edge in sleep_incapacitated_edges[:6]
            )
            findings.append(_finding(
                "SLEEP_INCAPACITATION_TOGGLE", "critical",
                "Uyku ve baygınlık durumları tekrarlı biçimde birbirine geçti",
                f"{len(sleep_incapacitated_edges)} doğrudan geçiş. İlk örnekler: {first_edges}.",
                "Baygın ajanı normal uyku kararından ayırın; tıbbi stabilizasyon tamamlanmadan uyku durumuna geçirmeyin.",
                agent_name,
            ))

        unsupported_eva = [
            r for r in agent_records
            if not r["in_habitat"] and r["needs_o2_support"]
        ]
        if len(unsupported_eva) >= 10:
            consecutive_eva_pairs = [
                (previous, current)
                for previous, current in zip(agent_records, agent_records[1:])
                if not previous["in_habitat"] and not current["in_habitat"]
                and previous["needs_o2_support"] and current["needs_o2_support"]
            ]
            consuming_pairs = sum(
                1 for previous, current in consecutive_eva_pairs
                if current["plss_o2"] < previous["plss_o2"] - 0.1
            )
            if len(consecutive_eva_pairs) >= 10 and consuming_pairs == 0:
                findings.append(_finding(
                    "UNBREATHABLE_ATMOSPHERE_FREE_OXYGEN", "critical",
                    "Öldürücü düşük O₂ atmosferinde EVA yaşam desteği tüketilmedi",
                    f"{len(consecutive_eva_pairs)} ardışık dış ortam geçişinin hiçbirinde PLSS azalmadı.",
                    "has_atmosphere yerine solunabilirlik/PO₂ kullanın; _needs_o2_support true ise PLSS tüketimini çalıştırın.",
                    agent_name,
                ))

        starts: dict[str, list[int]] = defaultdict(list)
        for run in runs:
            starts[run[0]["action"]].append(run[0]["tick"])
        for vital_action in ("eat", "drink"):
            close_gaps = [
                b - a for a, b in zip(starts[vital_action], starts[vital_action][1:])
                if b - a <= 3
            ]
            if len(close_gaps) >= 3:
                findings.append(_finding(
                    "VITAL_ACTION_SPAM", "warning",
                    "Beslenme eylemi gerçekçilik dışı kısa aralıkla tekrarlandı",
                    f"{vital_action}: {len(close_gaps)} tekrarın aralığı 15 dakika veya daha azdı.",
                    "Tokluk/susuzluk hedefi ve asgari tekrar bekleme süresi uygulayın.",
                    agent_name,
                ))

        # Count only genuinely redundant cycles. Many separated, legitimate
        # returns over two simulated days are not an airlock loop.
        redundant_habitat_entries = [
            current
            for previous, current in zip(agent_records, agent_records[1:])
            if current["action"] == "enter_habitat" and previous["in_habitat"]
        ]
        if len(redundant_habitat_entries) > 5:
            entry_reasons = Counter(
                r.get("decision_reasoning", "") for r in redundant_habitat_entries
            )
            findings.append(_finding(
                "REPEATED_HABITAT_ENTRY", "critical",
                "Ajan zaten güvenli bölgedeyken hava kilidine tekrar tekrar girdi",
                f"{len(redundant_habitat_entries)} gereksiz enter_habitat ticki. "
                f"Başlıca nedenler: {dict(entry_reasons.most_common(3))}.",
                "Habitat içindeyken tahliye/geri dönüş planını tamamlayıp stratejik görevi yeniden planlayın.",
                agent_name,
            ))

        preflight_retry_records = [
            record for record in agent_records
            if record.get("eva_denied") == "insufficient_round_trip_energy"
            and record.get("action") not in {"sleep", "medical_rest"}
            and not record.get("indoor_recovery_pending", False)
        ]
        if len(preflight_retry_records) >= 3:
            first_ticks = ", ".join(
                f"{record['tick']}:{record.get('action', 'unknown')}"
                for record in preflight_retry_records[:8]
            )
            findings.append(_finding(
                "EVA_PREFLIGHT_RETRY_LOOP", "warning",
                "EVA enerji kilidi gerçek recovery yerine emri yeniden denedi",
                (
                    f"{len(preflight_retry_records)} tickte yetersiz gidiş-dönüş "
                    f"enerjisi reddi uyku/medikal recovery olmadan kaldı; "
                    f"ilk tick:eylemler: {first_ticks}."
                ),
                "Mesafe ölçekli çıkış reddini kesintisiz uyku bloğuna bağlayın.",
                agent_name,
            ))

        final = agent_records[-1]
        eva_denials = Counter(
            r.get("eva_denied") for r in agent_records if r.get("eva_denied")
        )
        late_records = [r for r in agent_records if r["tick"] > 3000]
        late_positions = [(r["x"], r["y"]) for r in late_records]
        late_action_counts = Counter(r["action"] for r in late_records)
        final_agent = next(
            (candidate for candidate in engine.agents if candidate.name == agent_name),
            None,
        )
        agent_summaries[agent_name] = {
            "final_status": final["status"],
            "final_position": [final["x"], final["y"]],
            "distance_cells": distance,
            "max_base_distance_cells": max_base_distance,
            "max_authorized_eva_radius_cells": max(
                (r.get("authorized_eva_radius", local_eva_limit) for r in agent_records),
                default=local_eva_limit,
            ),
            "expedition_ticks": sum(1 for r in agent_records if r.get("expedition_active")),
            "sorties": len(sorties),
            "short_unproductive_sorties": len(short_unproductive_sorties),
            "terminal_resource_exhaustion_idle_ticks": (
                terminal_resource_idle_ticks
            ),
            "short_sortie_examples": [
                {
                    "ticks": [sortie[0]["tick"], sortie[-1]["tick"]],
                    "actions": [record["action"] for record in sortie],
                    "targets": [record["target"] for record in sortie],
                    "decision_actions": [
                        record.get("decision_action", "") for record in sortie
                    ],
                    "decision_reasons": [
                        record.get("decision_reasoning", "") for record in sortie
                    ],
                    "energy": [round(record["energy"], 1) for record in sortie],
                    "temperature_stress": [
                        round(record["temperature_stress"], 1)
                        for record in sortie
                    ],
                    "post_sortie_decision": next_record_by_tick.get(
                        sortie[-1]["tick"], {}
                    ).get("decision_reasoning", ""),
                    "plss": [
                        {
                            "o2": round(record["plss_o2"], 1),
                            "scrubber": round(record["plss_scrubber"], 1),
                            "battery": round(record["plss_battery"], 1),
                            "suit_integrity": round(record["suit_integrity"], 3),
                            "suit_condition": round(
                                record.get("suit_condition", 1.0), 3
                            ),
                        }
                        for record in sortie
                    ],
                }
                for sortie in short_unproductive_sorties[:5]
            ],
            "unique_positions": len(set(positions)),
            "action_counts": dict(action_counts.most_common()),
            "eva_denials": dict(eva_denials),
            "post_3000": {
                "ticks": len(late_records),
                "outside_ticks": sum(
                    1 for r in late_records if not r.get("in_habitat", False)
                ),
                "distance_cells": sum(
                    abs(late_positions[i][0] - late_positions[i - 1][0])
                    + abs(late_positions[i][1] - late_positions[i - 1][1])
                    for i in range(1, len(late_positions))
                ),
                "max_base_distance_cells": max(
                    (r["base_distance"] for r in late_records), default=0
                ),
                "action_counts": dict(late_action_counts.most_common()),
                "eva_denials": dict(Counter(
                    r.get("eva_denied")
                    for r in late_records if r.get("eva_denied")
                )),
            },
            "longest_action_runs": dict(sorted(longest_runs.items())),
            "collapse_onsets": len(collapse_onsets),
            "incapacitated_ticks": len(collapses),
            "collapse_recoveries": len(collapse_recoveries),
            "medical_emergency_events": len(medical_emergency_events),
            "sleep_incapacitated_edges": sleep_incapacitated_edges[:10],
            "final_needs": {
                key: final[key]
                for key in ("hunger", "thirst", "energy", "o2", "temperature_stress")
            },
            "final_inventory": (
                dict(final_agent.inventory.materials) if final_agent else {}
            ),
            "final_items": (
                dict(final_agent.inventory.items) if final_agent else {}
            ),
            "final_tool_durability": (
                dict(final_agent.inventory.tool_durability) if final_agent else {}
            ),
        }

    copied_ticks = 0
    coordination_hotspots: list[dict[str, Any]] = []
    coordinated_work_actions = {
        "move", "gather", "build", "repair", "refine", "prospect",
        "survey_resources", "dispatch_excavator", "clean_solar_panels",
        "deposit_materials", "rescue", "treat",
    }
    for tick_records in by_tick.values():
        groups = Counter(
            (r["action"], r["target"])
            for r in tick_records
            if r["action"] in coordinated_work_actions
        )
        over_capacity = [
            (action, target, count, _coordination_capacity(
                engine, action, target
            ))
            for (action, target), count in groups.items()
            if count > _coordination_capacity(engine, action, target)
        ]
        if over_capacity:
            copied_ticks += 1
            if len(coordination_hotspots) < 20:
                action, target, count, capacity = max(
                    over_capacity, key=lambda row: row[2] - row[3]
                )
                coordination_hotspots.append({
                    "tick": tick_records[0]["tick"],
                    "action": action,
                    "target": target,
                    "agents": count,
                    "physical_capacity": capacity,
                })
    if _is_sustained_coordination_overage(copied_ticks, ticks_requested):
        findings.append(_finding(
            "WEAK_TASK_SPECIALISATION", "warning",
            "Ajanlar uzun süre aynı fiziksel hedefin ekip kapasitesini aştı",
            f"Toplam {copied_ticks} tickte en az bir üretken hedefin fiziksel/önerilen ekip kapasitesi aşıldı.",
            "Yetkinlik bazlı görev sahipliği, hedef kapasitesi ve ekip çapında görev rezervasyonu uygulayın.",
        ))

    final_structures = dict(engine.structures_built)
    final_score = float(engine.colony_score.get_overall_score())
    construction_projects = [
        {
            "id": structure.get("id"),
            "type": structure.get("type"),
            "phase": structure.get("construction_phase", "assembly"),
            "materials_committed": structure.get("materials_committed", True),
            "progress": round(float(structure.get("progress", 0.0)), 4),
            "planned_tick": structure.get("planned_tick"),
            "built_tick": structure.get("built_tick"),
        }
        for structure in getattr(engine, "placed_structures", [])
        if structure.get("under_construction", False)
        and not structure.get("destroyed", False)
    ]
    structure_timeline: dict[str, dict[str, int | None]] = {}
    for struct in getattr(engine, "placed_structures", []):
        struct_type = struct.get("type")
        if not struct_type:
            continue
        start_tick = int(struct.get("built_tick", struct.get("planned_tick", 0)))
        completed_tick = struct.get("completed_tick")
        timeline = structure_timeline.setdefault(
            struct_type,
            {"first_started_tick": start_tick, "first_completed_tick": None},
        )
        timeline["first_started_tick"] = min(
            int(timeline["first_started_tick"]), start_tick
        )
        if completed_tick is not None:
            previous_completion = timeline["first_completed_tick"]
            timeline["first_completed_tick"] = (
                int(completed_tick) if previous_completion is None
                else min(int(previous_completion), int(completed_tick))
            )

    production_summary = {
        "sampled_ticks": len(production_samples),
        "energy_deficit_ticks": sum(
            1 for sample in production_samples if sample.get("energy_deficit")
        ),
        "oxygen_production_ticks": sum(
            1 for sample in production_samples
            if sample.get("o2_production_kg", 0.0) > 0.0
        ),
        "greenhouse_food_production_ticks": sum(
            1 for sample in production_samples
            if sample.get("food_production_kcal", 0.0) > 0.0
        ),
        "oxygen_produced_kg": round(sum(
            sample.get("o2_production_kg", 0.0) for sample in production_samples
        ), 3),
        "lander_ogs_oxygen_produced_kg": round(sum(
            sample.get("lander_ogs_o2_production_kg", 0.0)
            for sample in production_samples
        ), 3),
        "isru_oxygen_produced_kg": round(sum(
            max(
                0.0,
                sample.get("o2_production_kg", 0.0)
                - sample.get("lander_ogs_o2_production_kg", 0.0),
            )
            for sample in production_samples
        ), 3),
        "greenhouse_food_produced_kcal": round(sum(
            sample.get("food_production_kcal", 0.0) for sample in production_samples
        ), 1),
        "final": dict(production_samples[-1]) if production_samples else {},
    }

    # A physical machine batch is an exactly-once transaction: every emitted
    # start must end in one completion or remain as one canonical open WIP
    # cycle.  This catches phantom output caused by stale shift-handoff action
    # copies, including cases where inventory totals happen to look plausible.
    manufacturing_starts = [
        event for event in notable_events
        if event.get("type") == "manufacturing_start" and event.get("cycle_id")
    ]
    manufacturing_completions = [
        event for event in notable_events
        if event.get("type") == "refine" and event.get("cycle_id")
    ]
    open_manufacturing_cycles = [
        cycle for cycle in getattr(engine, "_manufacturing_cycles", {}).values()
        if cycle.get("completion_pending") and cycle.get("cycle_id")
    ]
    start_ids = Counter(str(event["cycle_id"]) for event in manufacturing_starts)
    completion_ids = Counter(
        str(event["cycle_id"]) for event in manufacturing_completions
    )
    open_ids = Counter(
        str(cycle["cycle_id"]) for cycle in open_manufacturing_cycles
    )
    all_cycle_ids = set(start_ids) | set(completion_ids) | set(open_ids)
    imbalanced_cycle_ids = sorted(
        cycle_id for cycle_id in all_cycle_ids
        if start_ids[cycle_id]
        != completion_ids[cycle_id] + open_ids[cycle_id]
    )
    duplicate_completion_ids = sorted(
        cycle_id for cycle_id, count in completion_ids.items() if count > 1
    )
    batch_delta = (
        len(manufacturing_starts)
        - len(manufacturing_completions)
        - len(open_manufacturing_cycles)
    )
    started_output_mass_kg = sum(
        float(event.get("output_mass_kg", 0.0))
        for event in manufacturing_starts
    )
    completed_output_mass_kg = sum(
        float(event.get("output_mass_kg", 0.0))
        for event in manufacturing_completions
    )
    open_output_mass_kg = sum(
        float(cycle.get("output_mass_kg", 0.0))
        for cycle in open_manufacturing_cycles
    )
    output_mass_delta_kg = round(
        started_output_mass_kg
        - completed_output_mass_kg
        - open_output_mass_kg,
        6,
    )
    manufacturing_summary = {
        "started_batches": len(manufacturing_starts),
        "completed_batches": len(manufacturing_completions),
        "open_batches": len(open_manufacturing_cycles),
        "batch_balance_delta": batch_delta,
        "started_output_mass_kg": round(started_output_mass_kg, 3),
        "completed_output_mass_kg": round(completed_output_mass_kg, 3),
        "open_output_mass_kg": round(open_output_mass_kg, 3),
        "output_mass_balance_delta_kg": output_mass_delta_kg,
        "duplicate_completion_cycle_ids": duplicate_completion_ids,
        "imbalanced_cycle_ids": imbalanced_cycle_ids,
        "exactly_once_balanced": (
            batch_delta == 0
            and abs(output_mass_delta_kg) <= 1e-6
            and not duplicate_completion_ids
            and not imbalanced_cycle_ids
        ),
    }
    if not manufacturing_summary["exactly_once_balanced"]:
        findings.append(_finding(
            "MANUFACTURING_BATCH_BALANCE_BROKEN", "critical",
            "Üretim batch olayları fiziksel WIP ile dengelenmiyor",
            (
                f"Başlangıç={len(manufacturing_starts)}, "
                f"tamamlanma={len(manufacturing_completions)}, "
                f"açık={len(open_manufacturing_cycles)}, fark={batch_delta}; "
                f"çıktı-kütle farkı={output_mass_delta_kg} kg."
            ),
            "Her makine çevrimini değişmez cycle_id ile tam olarak bir kez tamamlayın.",
        ))
    # The default horizon is still an early field campaign. A recorded
    # geological survey is valid progress while critical feedstock is being
    # located, without prescribing a hidden power/O2/water strategy to RL.
    # Survey activity stops excusing missing infrastructure beyond that
    # initial horizon.
    critical_missing = []
    surface_requires_plss = any(
        bool(getattr(agent, "_needs_o2_support", False))
        for agent in engine.agents
        if getattr(agent.status, "value", str(agent.status)) != "dead"
    )
    early_survey_progress = (
        ticks_requested <= DEFAULT_TICKS
        and any(event.get("type") == "resource_survey" for event in notable_events)
    )
    # Delivered reserves are sized for 40 days. Requiring local water and O2
    # after only the old 500-tick (~3.5 day) debug horizon creates a false
    # warning for a rational prefab/power commissioning phase. Half the
    # consumables design life is the latest acceptable bootstrap checkpoint.
    consumable_design_days = float(
        engine.mission_profile.advance_crew_consumables.get(
            "design_days", 40.0
        )
    )
    life_support_bootstrap_tick = int(
        math.floor(
            0.5 * consumable_design_days
            * engine.mission_profile.clock.ticks_per_earth_day
        )
    )
    long_horizon = ticks_requested >= life_support_bootstrap_tick
    if (
        (long_horizon or not surface_requires_plss)
        and final_structures.get("water_collector", 0) == 0
    ):
        critical_missing.append("water_collector")
    if (
        long_horizon
        and surface_requires_plss
        and final_structures.get("isru_o2_unit", 0) == 0
    ):
        critical_missing.append("isru_o2_unit")
    if (
        ticks_requested >= 10_000
        and final_structures.get("greenhouse", 0) == 0
    ):
        critical_missing.append("greenhouse")
    # A solar array still needs several local bulk feedstocks even though its
    # qualified PV blankets and copper conductors are finite delivered cargo.
    # A research workbench plus relay is valid early dependency-resolution
    # progress when a remaining ore or mineral lies outside local EVA; do not
    # label that survey stage irrationally incomplete.
    solar_dependency_path_active = (
        final_structures.get("research_workbench", 0) > 0
        and final_structures.get("communication_relay", 0) > 0
    ) or early_survey_progress
    # At the 3.5-day inspection an actually commissioned storage/utility
    # component is also valid bootstrap progress. Requiring solar to be the
    # first RL choice contradicts the strategy contract. This exception never
    # excuses absent generation in a longer campaign.
    solar_dependency_path_active = solar_dependency_path_active or (
        ticks_requested <= DEFAULT_TICKS
        and any(final_structures.get(name, 0) > 0 for name in (
            "oxygen_buffer_tank", "potable_water_tank", "isru_o2_unit",
            "water_collector", "power_distribution_grid",
            "life_support_distribution_grid",
        ))
    )
    if final_structures.get("solar_panel", 0) == 0 and not solar_dependency_path_active:
        critical_missing.append("solar_panel")
    if critical_missing:
        in_progress = {
            project["type"]: round(float(project.get("progress", 0.0)) * 100, 1)
            for project in construction_projects
            if project.get("type") in critical_missing
        }
        progress_text = (
            "; devam eden: "
            + ", ".join(
                f"{name} %{progress}" for name, progress in in_progress.items()
            )
            if in_progress else ""
        )
        findings.append(_finding(
            "MISSION_CRITICAL_INFRASTRUCTURE_MISSING", "warning",
            f"{ticks_requested} tick sonunda temel koloni altyapısı dengeli ilerlemedi",
            f"Eksik: {', '.join(critical_missing)}{progress_text}; "
            f"hazırlık skoru %{final_score:.2f}.",
            "Bağımlılık grafiği ve minimum dengeli kapasite kilometre taşları kullanın.",
        ))

    severity_counts = Counter(f["severity"] for f in findings)
    colony_capacity = engine.colony_score.to_dict()
    arrival_qualification = engine._mission_state()
    return {
        "meta": {
            "planet": engine.planet.name,
            "seed": seed,
            "ticks_requested": ticks_requested,
            "ticks_completed": len(by_tick),
            "simulated_hours": round(
                ticks_requested * tick_minutes / 60, 2
            ),
            "agent_count": len(by_agent),
            "landing_zone": {
                "x": getattr(engine, "lz_x", 1000),
                "y": getattr(engine, "lz_y", 1000),
            },
            "local_eva_radius_cells": int(getattr(engine, "LOCAL_EVA_RADIUS_CELLS", 24)),
            "min_extraction_radius_cells": int(getattr(engine, "MIN_EXTRACTION_RADIUS_CELLS", 6)),
            "max_eva_radius_cells": int(getattr(engine, "LOCAL_EVA_RADIUS_CELLS", 24)),
            "survey_radius_cells": int(engine._survey_radius_cells()),
            "communication_radius_cells": int(engine._communication_radius_cells()),
            "decision_mode": "offline deterministic fallback + fresh RL policy",
        },
        "summary": {
            "critical_findings": severity_counts.get("critical", 0),
            "warning_findings": severity_counts.get("warning", 0),
            "final_colony_score_percent": final_score,
            "colony_capacity": colony_capacity,
            "arrival_qualification": arrival_qualification,
            "final_structures": final_structures,
            "construction_projects": construction_projects,
            "structure_timeline": structure_timeline,
            "production": production_summary,
            "manufacturing": manufacturing_summary,
            "coordination_hotspots": coordination_hotspots,
            "final_depot": dict(engine.central_depot_inventory),
            "resource_search": {
                "pending_requests": sorted(engine.remote_resource_requests),
                "discovered_counts": {
                    resource: len(coords)
                    for resource, coords in sorted(
                        engine.discovered_resources.items()
                    )
                },
                "hand_miss_counts": {
                    resource: len(coords)
                    for resource, coords in sorted(
                        engine._local_resource_miss_centers.items()
                    )
                },
                "detector_miss_counts": {
                    resource: len(coords)
                    for resource, coords in sorted(
                        engine._portable_resource_miss_centers.items()
                    )
                },
                "portable_scan_stations": len(engine._portable_scan_centers),
            },
            "shared_work_order": dict(
                getattr(engine.decision_engine, "shared_work_order", {})
            ),
            "mission_diagnostics": {
                "surface_fleet": engine.surface_fleet.to_dict(),
                "open_excavation_faces": {
                    agent.name: dict(agent._active_excavation)
                    for agent in engine.agents
                    if isinstance(
                        getattr(agent, "_active_excavation", None), dict
                    )
                },
                "water": {
                    "crew_consumed_l": round(sum(
                        float(getattr(agent, "total_water_consumed_l", 0.0))
                        for agent in engine.agents
                    ), 2),
                    "recovery_queue_l": round(sum(
                        float(batch.get("liters", 0.0))
                        for batch in getattr(engine, "_water_recovery_queue", [])
                    ), 2),
                    "portable_packs": sum(
                        int(agent.inventory.items.get("water_packs", 0))
                        for agent in engine.agents
                    ),
                },
                "critical_feedstock_faces": {
                    resource: [
                        {
                            "x": int(x),
                            "y": int(y),
                            "distance": engine._distance_from_lz(x, y),
                            "current_layer": (
                                engine._current_geology_layer(x, y) or {}
                            ).get("material"),
                            "target_remaining": next((
                                int(layer.get("remaining", 0))
                                for layer in engine._get_cell_geology(
                                    x, y
                                ).get("layers", [])
                                if layer.get("material") == resource
                            ), 0),
                        }
                        for x, y in sorted(
                            engine.discovered_resources.get(resource, set())
                        )
                        if (x, y, resource)
                        not in engine.depleted_cell_resources
                    ][:12]
                    for resource in ("sulfur", "chalcopyrite_ore", "iron_ore")
                },
                "mission_event_timeline": [
                    {
                        "tick": int(event.get("tick", 0)),
                        "type": event.get("type"),
                        "agent": event.get("agent"),
                        "cause": str(event.get("cause", "")),
                    }
                    for event in notable_events
                    if event.get("type") in {
                        "resource_survey", "geology_layer_exposed",
                        "manufacturing_start", "construction_start",
                        "construction_complete", "rover_unload",
                    }
                    and (
                        event.get("type") != "resource_survey"
                        or "sulfur" in str(event.get("cause", "")).lower()
                    )
                ][-200:],
            },
            "agents_alive": sum(1 for a in engine.agents if getattr(a.status, "value", str(a.status)) != "dead"),
            "agents_dead": sum(1 for a in engine.agents if getattr(a.status, "value", str(a.status)) == "dead"),
            "processing_failures": len(processing_failures),
        },
        "agents": agent_summaries,
        "findings": findings,
    }


def run_audit(ticks: int = DEFAULT_TICKS, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Run a true tick-advancing simulation and return structured audit data."""
    random.seed(seed)
    np.random.seed(seed)
    # A neural embedder is irrelevant to this offline behavior regression and
    # makes a 500-tick test needlessly slow/non-hermetic.
    vector_store._use_tfidf_fallback = True
    notable_events: list[dict[str, Any]] = []
    engine = SimulationEngine(
        str(ROOT / "config" / "planets" / "kepler-442b.json"),
        seed=seed,
        max_ticks=ticks + 1,
        db=False,
        on_event=lambda event: notable_events.append(dict(event)),
    )
    engine.llm_client.call = _offline_llm
    if hasattr(engine.llm_client, "send"):
        engine.llm_client.send = _offline_llm
    agents = create_team_from_presets(str(ROOT / "config" / "agent_presets.json"))
    crew_count = int(MissionProfile.load().advance_crew)
    if len(agents) < crew_count:
        raise ValueError(
            f"Mission requires {crew_count} crew but only {len(agents)} presets exist"
        )
    for index, agent in enumerate(agents[:crew_count]):
        # Agent UUIDs otherwise change role offsets and RL exploration ordering
        # between identical audit runs, defeating the stated reproducibility.
        agent.id = f"realism-audit-agent-{index + 1}"
        engine.add_agent(agent)

    records: list[dict[str, Any]] = []
    processing_failures: list[dict[str, Any]] = []
    production_samples: list[dict[str, Any]] = []
    logging.disable(logging.CRITICAL)
    try:
        for step in range(1, ticks + 1):
            tick_events = engine._run_tick()
            production_samples.append(dict(tick_events.get("colony_production", {})))
            processed_ids = set(tick_events.get("agent_events", {}))
            explicit_errors = {
                str(error.get("agent_id")): dict(error)
                for error in tick_events.get("agent_processing_errors", [])
            }
            for agent in engine.agents:
                if getattr(agent.status, "value", str(agent.status)) != "dead" and agent.id not in processed_ids:
                    processing_failures.append({
                        "tick": step,
                        "agent": agent.name,
                        **explicit_errors.get(str(agent.id), {}),
                    })
                expedition = getattr(agent, "_active_expedition", None)
                expedition_active = (
                    isinstance(expedition, dict)
                    and expedition.get("status") in {"outbound", "working", "returning"}
                )
                radial_distance = engine._distance_from_lz(agent.x, agent.y)
                return_ticks = (
                    radial_distance + engine.EXPEDITION_MOVE_SPEED_CELLS - 1
                ) // engine.EXPEDITION_MOVE_SPEED_CELLS
                records.append({
                    "tick": step,
                    "engine_tick": engine.current_tick,
                    "agent": agent.name,
                    "action": agent.action.action_type or "idle",
                    "target": _normalise_target(agent.action.target),
                    "airlock_cycle_wait": bool(
                        (getattr(agent, "_airlock_wait", None) or {}).get("tick")
                        == engine.current_tick
                    ),
                    "indoor_recovery_pending": bool(
                        agent._in_habitat
                        and (getattr(agent, "_pending_indoor_activity", None) or {}).get("action")
                        in {"sleep", "medical_rest"}
                    ),
                    "eva_denied": (
                        agent.action.target.get("eva_denied")
                        if isinstance(agent.action.target, dict) else None
                    ),
                    "x": agent.x,
                    "y": agent.y,
                    "base_distance": radial_distance,
                    "authorized_eva_radius": engine._agent_eva_radius_cells(agent),
                    "expedition_active": expedition_active,
                    "expedition_status": expedition.get("status", "") if expedition_active else "",
                    "expedition_resource": expedition.get("resource", "") if expedition_active else "",
                    "expedition_target": (
                        [expedition.get("target_x"), expedition.get("target_y")]
                        if expedition_active else None
                    ),
                    "expedition_return_margin_ticks": (
                        engine._expedition_available_ticks(agent) - return_ticks
                        if expedition_active else None
                    ),
                    "status": getattr(agent.status, "value", str(agent.status)),
                    "in_habitat": bool(getattr(agent, "_in_habitat", False)),
                    "in_base_habitat": bool(
                        getattr(agent, "_in_habitat", False)
                        and engine._is_lander_footprint_cell(agent.x, agent.y)
                    ),
                    "suit_equipped": bool(getattr(agent, "suit_equipped", False)),
                    "needs_o2_support": bool(getattr(agent, "_needs_o2_support", False)),
                    "plss_o2": float(getattr(agent, "_current_canister_remaining", 100.0)),
                    "plss_scrubber": float(getattr(agent, "plss_co2_scrubber_pct", 100.0)),
                    "plss_battery": float(getattr(agent, "plss_suit_battery_pct", 100.0)),
                    "suit_integrity": float(getattr(agent, "suit_integrity", 1.0)),
                    "suit_condition": float(getattr(agent, "suit_condition", 1.0)),
                    "hunger": float(agent.needs.hunger),
                    "thirst": float(agent.needs.thirst),
                    "energy": float(agent.needs.energy),
                    "o2": float(agent.needs.o2_supply),
                    "temperature_stress": float(agent.needs.temperature_stress),
                    "hygiene": float(agent.needs.hygiene),
                    "effective_temperature_c": float(getattr(agent, "_last_effective_temperature_c", 20.0)),
                    "team_emergency_alert": bool(
                        getattr(agent, "_team_emergency_alert", {}).get("active", False)
                        if isinstance(getattr(agent, "_team_emergency_alert", {}), dict)
                        else False
                    ),
                    "injury": float(agent.injury_level),
                    "water_reserve_l": float(
                        engine._colony_resources.get("water_reserve_l", 0.0)
                    ),
                    "colony_score": float(engine.colony_score.get_overall_score()),
                    "decision_action": getattr(agent, "last_decision", {}).get("action", ""),
                    "decision_reasoning": getattr(agent, "last_decision", {}).get("reasoning", ""),
                })
            # Production run() increments after _run_tick(); reproduce it here.
            engine.current_tick += 1
    finally:
        logging.disable(logging.NOTSET)

    return _analyse(
        engine, records, processing_failures, notable_events,
        production_samples, ticks, seed,
    )


def render_markdown(result: dict[str, Any]) -> str:
    meta, summary = result["meta"], result["summary"]
    capacity = summary.get("colony_capacity", {})
    qualification = summary.get("arrival_qualification", {})
    support_soak = qualification.get("support_soak", {})
    lines = [
        f"# {meta['ticks_completed']} Tick Ajan Gerçekçilik Denetimi", "",
        f"- Gezegen: {meta['planet']}",
        f"- Süre: {meta['ticks_completed']} tick / {meta['simulated_hours']} simülasyon saati",
        f"- Ajan: {meta['agent_count']}",
        f"- Karar modu: {meta['decision_mode']}",
        f"- Son koloni hazırlığı: %{summary['final_colony_score_percent']:.2f}",
        f"- 30 günlük kesintisiz yeterlilik: "
        f"{support_soak.get('completed_days', 0)} / "
        f"{support_soak.get('required_days', 30)} gün",
        f"- Yüzey kabulü: "
        f"{'HAZIR' if qualification.get('surface_acceptance_ready') else 'HAZIR DEĞİL'}",
        f"- Bulgular: {summary['critical_findings']} kritik, {summary['warning_findings']} uyarı",
        f"- Oksijen üretimi: {summary['production']['oxygen_production_ticks']} tick / "
        f"{summary['production']['oxygen_produced_kg']} kg "
        f"(lander OGS {summary['production']['lander_ogs_oxygen_produced_kg']} kg, "
        f"yerel ISRU {summary['production']['isru_oxygen_produced_kg']} kg)",
        f"- Sera üretimi: {summary['production']['greenhouse_food_production_ticks']} tick / "
        f"{summary['production']['greenhouse_food_produced_kcal']} kcal",
        f"- Üretim batch dengesi: {summary['manufacturing']['started_batches']} başlangıç / "
        f"{summary['manufacturing']['completed_batches']} tamamlanma / "
        f"{summary['manufacturing']['open_batches']} açık; "
        f"fark {summary['manufacturing']['batch_balance_delta']}",
        f"- Planlanan çıktı-kütle dengesi: "
        f"{summary['manufacturing']['started_output_mass_kg']} kg başlangıç / "
        f"{summary['manufacturing']['completed_output_mass_kg']} kg tamam / "
        f"{summary['manufacturing']['open_output_mass_kg']} kg açık; "
        f"fark {summary['manufacturing']['output_mass_balance_delta_kg']} kg",
        f"- Keşif menzilleri: yerel EVA {meta['local_eva_radius_cells']}, "
        f"jeolojik tarama {meta['survey_radius_cells']}, iletişim {meta['communication_radius_cells']} grid",
        "",
        "Bu koşu canlı LLM kullanmaz; yerel fallback planlayıcı ile temiz RL politikasını denetler. "
        "Bulgular tekrar üretilebilir, fakat canlı model davranışını temsil etmez.",
        "", "## Kapasite ve kabul kapıları", "",
        "| Kategori | Gerçek kapasite | Hedef | Tamamlanma |",
        "|---|---:|---:|---:|",
    ]
    current_counts = capacity.get("current_counts", {})
    targets = capacity.get("targets", {})
    category_scores = capacity.get("categories", {})
    for category in targets:
        lines.append(
            f"| {category} | {float(current_counts.get(category, 0.0)):.2f} | "
            f"{float(targets.get(category, 0.0)):.2f} | "
            f"%{100.0 * float(category_scores.get(category, 0.0)):.1f} |"
        )
    lines.extend(["", "| Kabul kapısı | Sonuç |", "|---|---:|"])
    for gate, passed in qualification.get("arrival_gates", {}).items():
        lines.append(f"| {gate} | {'GEÇTİ' if passed else 'KALDI'} |")

    lines.extend([
        "", "## Ajan özeti", "",
        "| Ajan | Son durum | Toplam mesafe | Base'den azami uzaklık | Sefer ticki | Benzersiz konum | En sık eylemler |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for name, agent in result["agents"].items():
        top = ", ".join(f"{a}={c}" for a, c in list(agent["action_counts"].items())[:4])
        lines.append(
            f"| {name} | {agent['final_status']} | {agent['distance_cells']} | "
            f"{agent['max_base_distance_cells']} | {agent['expedition_ticks']} | "
            f"{agent['unique_positions']} | {top} |"
        )

    lines.extend(["", "## Gerçekçilik dışı / saçma davranış bulguları", ""])
    if not result["findings"]:
        lines.append("Bulgulanmadı.")
    for index, finding in enumerate(result["findings"], start=1):
        context = " / ".join(v for v in (finding.get("agent"), finding.get("tick_range")) if v)
        lines.extend([
            f"### {index}. [{finding['severity'].upper()}] {finding['title']}", "",
            f"- Kod: `{finding['code']}`",
            f"- Bağlam: {context or 'sistem geneli'}",
            f"- Kanıt: {finding['evidence']}",
            f"- Öneri: {finding['recommendation']}", "",
        ])

    structures = ", ".join(f"{n}={c}" for n, c in summary["final_structures"].items()) or "yok"
    lines.extend([
        "## Son durum", "",
        f"- Hayatta: {summary['agents_alive']}, ölü: {summary['agents_dead']}",
        f"- Yapılar: {structures}",
        f"- İşleme hatası: {summary['processing_failures']}", "",
    ])
    return "\n".join(lines)


def write_reports(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ticks_completed = int(result.get("meta", {}).get("ticks_completed", DEFAULT_TICKS))
    markdown_path = output_dir / f"realism_audit_{ticks_completed}.md"
    json_path = output_dir / f"realism_audit_{ticks_completed}.json"
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return markdown_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--strict", action="store_true", help="Kritik bulguda hata koduyla çık")
    args = parser.parse_args()
    result = run_audit(args.ticks, args.seed)
    markdown_path, json_path = write_reports(result, args.output_dir)
    print(render_markdown(result))
    print(f"\nMarkdown raporu: {markdown_path}\nJSON raporu: {json_path}")
    return 1 if args.strict and result["summary"]["critical_findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
