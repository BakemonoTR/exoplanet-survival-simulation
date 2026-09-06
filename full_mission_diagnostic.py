"""Read-only/offline production-identity mission diagnostic; never writes live RL.

Runs the actual physics loop to the requested deadline or a real terminal
condition. Per-tick crew traces are compressed, not retained in memory.
"""
from __future__ import annotations

import argparse
import faulthandler
from collections import Counter
import gzip
import hashlib
import json
import logging
from pathlib import Path
import random
import time

import numpy as np

from campaign_audit import ROOT, PLANET_DIR, _offline_llm
from src.agents.agent import create_team_from_presets
from src.memory import vector_store
from src.orchestration.engine import SimulationEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ticks', type=int, default=51840)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--planet', default='kepler-442b')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile-start', type=int, default=-1,
                        help='Profile 100 physical ticks starting here; disabled by default')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_paths = ('src/orchestration/engine.py', 'src/agents/agent.py',
                    'src/agents/decision.py', 'src/agents/strategic_rl.py',
                    'config/recipes.json', 'config/colony_targets.json',
                    'config/mission_profile.json',
                    'src/systems/airlock.py', 'src/systems/surface_fleet.py',
                    'full_mission_diagnostic.py')
    (args.output / 'loaded_sources.json').write_text(json.dumps({
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in source_paths}, indent=2), encoding='utf-8')
    logging.basicConfig(level=logging.ERROR)
    vector_store._use_tfidf_fallback = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    started = time.perf_counter()
    stall_log = (args.output / 'slow_tick_stack.log').open('w', encoding='utf-8')
    counts = Counter()
    waits = Counter()
    per_agent = {}
    longest = {}
    runs = {}
    errors = []
    critical_active = set()
    critical_onsets = []
    completed = []
    checkpoints = []
    tick_profiler = None
    with gzip.open(args.output / 'events.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) as journal, gzip.open(args.output / 'crew_trace.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) as trace:
        def event_callback(event):
            journal.write(json.dumps(event, ensure_ascii=False, default=str) + '\n')
            if event.get('type') == 'construction_complete':
                completed.append(event)

        engine = SimulationEngine(str(PLANET_DIR / (args.planet + '.json')),
                                  seed=args.seed, max_ticks=args.ticks,
                                  db=False, on_event=event_callback)
        engine.llm_client.call = _offline_llm
        engine.llm_client.send = _offline_llm
        # Keep the actual preset IDs (agent_01 ... agent_06), as in the API.
        for agent in create_team_from_presets(str(ROOT / 'config' / 'agent_presets.json'))[:engine.mission_profile.advance_crew]:
            engine.add_agent(agent)
        original_tick = engine._run_tick

        def audited_tick():
            nonlocal tick_profiler
            if engine.current_tick == args.profile_start:
                import cProfile
                tick_profiler = cProfile.Profile()
                tick_profiler.enable()
            faulthandler.dump_traceback_later(30, file=stall_log)
            events = original_tick()
            faulthandler.cancel_dump_traceback_later()
            tick = int(engine.current_tick)
            if tick_profiler is not None and tick == args.profile_start + 99:
                tick_profiler.disable()
                tick_profiler.dump_stats(str(args.output / 'tick_profile.prof'))
                tick_profiler = None
            errors.extend({'tick': tick, 'error': e} for e in events.get('agent_processing_errors', []))
            crew_rows = []
            for crew in engine.agents:
                action = str(crew.action.action_type)
                alive = getattr(crew.status, 'value', str(crew.status)) != 'dead'
                target = crew.action.target if isinstance(crew.action.target, dict) else {}
                critical = alive and (
                    getattr(crew.status, 'value', str(crew.status)) == 'incapacitated'
                    or min(crew.needs.o2_supply, crew.needs.thirst, crew.needs.hunger) <= 0
                )
                if critical and crew.id not in critical_active:
                    critical_onsets.append({'tick': tick, 'agent': crew.id,
                        'action': action, 'inside': crew._in_habitat,
                        'o2': crew.needs.o2_supply, 'thirst': crew.needs.thirst,
                        'hunger': crew.needs.hunger})
                    critical_active.add(crew.id)
                elif not critical:
                    critical_active.discard(crew.id)
                if alive:
                    counts[action] += 1
                    per_agent.setdefault(crew.id, Counter())[action] += 1
                    if not crew._in_habitat and action in {'sleep', 'service_suit'}:
                        counts['OUTDOOR_ROUTINE_RECOVERY'] += 1
                    if target.get('reason'):
                        waits[str(target['reason'])] += 1
                key = (crew.x, crew.y, crew._in_habitat, action,
                       target.get('x'), target.get('y'), target.get('reason'),
                       target.get('indoor_activity'))
                previous, length = runs.get(crew.id, (None, 0))
                length = length + 1 if key == previous and alive else 1
                runs[crew.id] = (key, length)
                if alive and length > longest.get(crew.id, {}).get('ticks', 0):
                    longest[crew.id] = {'ticks': length, 'start': tick-length+1,
                        'end': tick, 'position': [crew.x, crew.y], 'inside': crew._in_habitat,
                        'action': action, 'target': dict(target)}
                crew_rows.append({'id': crew.id, 'alive': alive, 'x': crew.x, 'y': crew.y,
                    'inside': crew._in_habitat, 'action': action, 'target': target,
                    'remaining': crew.action.ticks_remaining,
                    'energy': round(crew.needs.energy, 3), 'hunger': round(crew.needs.hunger, 3),
                    'thirst': round(crew.needs.thirst, 3), 'o2': round(crew.needs.o2_supply, 3),
                    'clinical': {'status': getattr(crew.status, 'value', str(crew.status)),
                        'unconscious': getattr(crew, '_is_unconscious', None),
                        'canister': round(crew._current_canister_remaining, 3),
                        'timers': {name: getattr(crew.needs, '_' + name + '_death_timer', 0)
                                   for name in ('hunger', 'thirst', 'energy', 'temp', 'o2')},
                        'inventory': dict(crew.inventory.items)},
                    'prep': getattr(crew, '_construction_preparation', None),
                    'pending_indoor': getattr(crew, '_pending_indoor_activity', None),
                    'expedition': getattr(crew, '_active_expedition', None)})
            rovers = [{'id': rover.id, 'x': rover.x, 'y': rover.y, 'state': rover.state,
                       'mission': rover.mission, 'distance_km': rover.total_distance_km,
                       'battery_kwh': rover.battery_kwh}
                      for rover in engine.surface_fleet.crew_rovers]
            trace.write(json.dumps({'tick': tick, 'airlock': engine.airlock.snapshot(), 'rovers': rovers,
                                   'crew': crew_rows}, ensure_ascii=False, default=str) + '\n')
            if tick % 100 == 0 or events.get('deaths'):
                checkpoint = {'tick': tick, 'alive': sum(r['alive'] for r in crew_rows),
                    'score': engine.colony_score.to_dict(), 'structures': dict(engine.structures_built),
                    'airlock': engine.airlock.snapshot(), 'resources': dict(engine._colony_resources),
                    'power': dict(getattr(engine, '_power_cycle_telemetry', {})),
                    'water_cycle': dict(getattr(engine, '_water_cycle_telemetry', {})),
                    'thermal': dict(getattr(engine, '_thermal_state', {})),
                    'wall_seconds': round(time.perf_counter()-started, 2)}
                checkpoint['processing_error_count'] = len(errors)
                checkpoint['critical_episode_count'] = len(critical_onsets)
                checkpoints.append(checkpoint)
                (args.output / 'progress.json').write_text(json.dumps(checkpoint, indent=2, default=str), encoding='utf-8')
                print(json.dumps({k: checkpoint[k] for k in ('tick','alive','structures','wall_seconds')}), flush=True)
                journal.flush()
                trace.flush()
            return events

        engine._run_tick = audited_tick
        failure = None
        try:
            report = engine.run_headless(ticks=args.ticks)
        except Exception as exc:
            import traceback
            failure = traceback.format_exc()
            report = {'end_reason': 'engine_error', 'total_ticks': engine.current_tick,
                      'error': repr(exc)}
        finally:
            faulthandler.cancel_dump_traceback_later()
            stall_log.close()
        result = {'requested_ticks': args.ticks, 'seed': args.seed, 'planet': args.planet,
            'policy_source': 'fresh; live database untouched',
            'agent_ids': [a.id for a in engine.agents], 'report': report,
            'action_ticks': dict(counts), 'per_agent_actions': per_agent,
            'wait_reasons': dict(waits), 'longest_unchanged_action': longest,
            'processing_errors': errors, 'exception': failure,
            'critical_onsets': critical_onsets,
            'completions': completed, 'checkpoints': checkpoints,
            'wall_seconds': round(time.perf_counter()-started, 3)}
        (args.output / 'diagnostic.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        print(json.dumps({'finished': True, 'tick': engine.current_tick,
              'end_reason': report.get('end_reason'), 'wall_seconds': result['wall_seconds']}), flush=True)
    return 1 if failure else 0


if __name__ == '__main__':
    raise SystemExit(main())
