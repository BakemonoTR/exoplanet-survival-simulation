"""Isolated cumulative strategy trial; no LLM calls or live database writes.

The tactical crew policy starts fresh in every episode to isolate strategic
learning. --control runs a fresh frozen strategic policy; the training path
persists every checkpoint and evaluates the last policy with exploration and
strategic Q updates disabled. World seed and evaluation tie-break counters match.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
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


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def run_episode(args, name, learned=None, evaluation=False):
    output = args.output / name
    output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    start = time.perf_counter()
    windows = defaultdict(Counter)
    completions, checkpoints, errors = [], [], []
    critical_crew = set()
    critical_onsets = 0

    def event_callback(event):
        if event.get('type') in ('construction_complete', 'structure_collapse'):
            completions.append(event)

    engine = SimulationEngine(str(PLANET_DIR / (args.planet + '.json')),
                              seed=args.seed, max_ticks=args.ticks, db=False,
                              on_event=event_callback, strategic_deadline_learning=True)
    engine.llm_client.call = _offline_llm
    engine.llm_client.send = _offline_llm
    for agent in create_team_from_presets(str(ROOT / 'config/agent_presets.json'))[:engine.mission_profile.advance_crew]:
        engine.add_agent(agent)
    policy = engine.strategic_policy
    if learned:
        policy.load(copy.deepcopy(learned))
    policy.evaluation_mode = evaluation
    if evaluation:
        # Identical tie-breaking stream for control and learned evaluation.
        policy.attempt_count = 0
        policy.decision_count = 0
    before = copy.deepcopy(policy.q_table)
    write_json(output / 'policy_start.json', policy.dump())
    original_tick = engine._run_tick

    def audited_tick():
        nonlocal critical_onsets
        events = original_tick()
        tick = engine.current_tick
        errors.extend({'tick': tick, 'error': error} for error in events.get('agent_processing_errors', []))
        window = 'days_300_360' if tick >= 300 * 144 else 'days_0_300'
        counts = windows[window]
        for crew in engine.agents:
            status = getattr(crew.status, 'value', str(crew.status))
            if status == 'dead':
                continue
            action = str(crew.action.action_type)
            target = crew.action.target if isinstance(crew.action.target, dict) else {}
            counts['crew_ticks'] += 1
            counts['action:' + action] += 1
            if (action in ('repair', 'clean_solar_panels') or
                    target.get('maintenance_action') in ('repair', 'inspection', 'clean_solar_panels')):
                counts['asset_service_and_tagged_travel'] += 1
            critical = status == 'incapacitated' or min(crew.needs.o2_supply, crew.needs.thirst, crew.needs.hunger) <= 0
            if critical and crew.id not in critical_crew:
                critical_onsets += 1
                critical_crew.add(crew.id)
            elif not critical:
                critical_crew.discard(crew.id)
        if tick % 5000 == 0 or tick == 330 * 144:
            row = {'tick': tick, 'score': engine.colony_score.to_dict(),
                   'structures': dict(engine.structures_built),
                   'strategy': policy.telemetry(), 'errors': len(errors),
                   'wall_seconds': round(time.perf_counter() - start, 2)}
            checkpoints.append(row)
            write_json(output / 'progress.json', row)
            print(json.dumps({'run': name, 'tick': tick, 'score': row['score']['overall'],
                              'wall_seconds': row['wall_seconds']}), flush=True)
        return events

    engine._run_tick = audited_tick
    report = engine.run_headless(ticks=args.ticks)
    engine._finalize_terminal_learning()
    result = {'name': name, 'seed': args.seed, 'ticks': engine.current_tick,
              'evaluation': evaluation, 'report': report,
              'score': engine.colony_score.to_dict(), 'mission': engine._mission_state(),
              'alive': sum(getattr(a.status, 'value', str(a.status)) != 'dead' for a in engine.agents),
              'critical_onsets': critical_onsets, 'processing_errors': errors,
              'strategy': policy.telemetry(), 'windows': dict(windows),
              'completions': completions, 'checkpoints': checkpoints,
              'q_values_unchanged': before == policy.q_table,
              'wall_seconds': round(time.perf_counter() - start, 3),
              'tactical_policy': 'fresh per episode; only strategic policy transferred'}
    write_json(output / 'result.json', result)
    if not evaluation:
        write_json(output / 'policy_end.json', policy.dump())
    if evaluation and not result['q_values_unchanged']:
        raise AssertionError('Frozen strategy changed Q values')
    print(json.dumps({'finished': name, 'score': result['score']['overall'],
                      'alive': result['alive'], 'errors': len(errors)}), flush=True)
    if errors or result['alive'] != len(engine.agents) or critical_onsets:
        raise RuntimeError('Safety regression; stopping experiment, checkpoint retained')
    return policy.dump(), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--planet', default='kepler-442b')
    parser.add_argument('--ticks', type=int, default=51840)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--control', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.ERROR)
    vector_store._use_tfidf_fallback = True
    label = 'control' if args.control else 'training'
    sources = ['src/agents/decision.py', 'src/agents/strategic_rl.py', 'src/orchestration/engine.py',
               'config/recipes.json', 'config/mission_profile.json', 'rl_learning_experiment.py']
    hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources}
    write_json(args.output / (label + '_sources.json'), hashes)
    if args.control:
        run_episode(args, 'fresh_frozen_control', evaluation=True)
        return
    learned = None
    summaries = []
    for index in range(1, args.episodes + 1):
        learned, result = run_episode(args, f'train_{index}', learned)
        summaries.append({key: result[key] for key in ('name', 'score', 'alive', 'strategy', 'wall_seconds')})
        write_json(args.output / 'training_summary.json', summaries)
    run_episode(args, 'trained_frozen_evaluation', learned, evaluation=True)


if __name__ == '__main__':
    main()
