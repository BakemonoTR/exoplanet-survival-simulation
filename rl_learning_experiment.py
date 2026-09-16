"""Isolated cumulative RL trial; no LLM calls or live database writes.

By default the tactical crew policy starts fresh in every episode to isolate
strategic learning. ``--carry-tactical`` instead transfers each crew member's
tactical and movement-effort Q-table along with the shared strategic policy.
``--control`` runs a fresh frozen policy; ``--skip-evaluation`` limits a run to
training episodes. World seed stays fixed so policy learning is the variable.
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


def run_episode(
    args,
    name,
    learned=None,
    learned_tactical=None,
    evaluation=False,
):
    output = args.output / name
    output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    start = time.perf_counter()
    windows = defaultdict(Counter)
    completions, checkpoints, errors = [], [], []
    critical_crew = set()
    critical_onsets = 0
    manufacturing_starts = Counter()
    machine_ticks = Counter()
    seen_cycles = set()

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
        if args.carry_tactical and learned_tactical:
            saved = learned_tactical.get(agent.id)
            if isinstance(saved, dict):
                agent.q_table = copy.deepcopy(saved)
        if evaluation:
            # Greedy choices with zero learning freeze tactical and movement
            # values while allowing harmless initial entries for unseen states.
            agent.rl_epsilon_explore = 0.0
            agent.rl_learning_rate = 0.0
    policy = engine.strategic_policy
    if learned:
        policy.load(copy.deepcopy(learned))
    policy.evaluation_mode = evaluation
    if evaluation:
        # Identical tie-breaking stream for control and learned evaluation.
        policy.attempt_count = 0
        policy.decision_count = 0
        policy.operational_decision_count = 0
    operational_before = copy.deepcopy(policy.operational_q_table)
    before = copy.deepcopy(policy.q_table)
    tactical_before = {
        agent.id: copy.deepcopy(agent.q_table) for agent in engine.agents
    }
    write_json(output / 'policy_start.json', policy.dump())
    if args.carry_tactical:
        write_json(output / 'tactical_policy_start.json', tactical_before)
    original_tick = engine._run_tick

    def audited_tick():
        nonlocal critical_onsets
        events = original_tick()
        tick = engine.current_tick
        errors.extend({'tick': tick, 'error': error} for error in events.get('agent_processing_errors', []))
        window = 'days_300_360' if tick >= 300 * 144 else 'days_0_300'
        counts = windows[window]
        # Read-only shop audit: separate physical process time, blocked output,
        # and unused installed machine slots instead of inferring from REFINE.
        for machine in engine.placed_structures:
            kind = machine.get('type')
            if kind not in ('stone_furnace', 'forge', 'cnc_fabricator'):
                continue
            if machine.get('destroyed') or machine.get('under_construction'):
                continue
            cycle = engine._manufacturing_cycles.get(str(machine.get('id')), {})
            state = ('awaiting_unload' if cycle.get('output_ready')
                     else 'processing' if cycle.get('active')
                     else 'paused' if cycle.get('completion_pending') else 'idle')
            machine_ticks[kind + ':' + state] += 1
            cycle_id = cycle.get('cycle_id')
            if cycle_id and cycle_id not in seen_cycles:
                seen_cycles.add(cycle_id)
                manufacturing_starts[str(cycle.get('output'))] += 1
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
                   'wall_seconds': round(time.perf_counter() - start, 2),
                   'manufacturing_starts': dict(manufacturing_starts),
                   'machine_ticks': dict(machine_ticks)}
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
              'structures': dict(engine.structures_built),
              'movement_effort_rl': dict(engine._movement_effort_telemetry),
              'completions': completions, 'checkpoints': checkpoints,
              'q_values_unchanged': before == policy.q_table,
              'tactical_state_counts': {
                  agent.id: sum(
                      1 for state in agent.q_table
                      if state != '__tactical_meta__'
                  )
                  for agent in engine.agents
              },
              'wall_seconds': round(time.perf_counter() - start, 3),
              'tactical_policy': (
                  'persisted across episodes'
                  if args.carry_tactical
                  else 'fresh per episode; only strategic policy transferred'
              )}
    planner = engine.decision_engine
    forecast, requirements, remaining = planner._refresh_mission_forecast_recipe(
        engine.structures_built
    )
    pooled = planner._pooled_materials()
    result['supply_audit'] = {
        'manufacturing_starts': dict(manufacturing_starts),
        'machine_ticks': dict(machine_ticks),
        'central_depot': dict(engine.central_depot_inventory),
        'crew_inventory': {a.id: dict(a.inventory.materials) for a in engine.agents},
        'remaining_structure_counts': remaining,
        'remaining_direct_materials': requirements,
        'remaining_bom_deficits': planner._bom_outstanding_requirements(forecast, pooled),
        'shared_work_order': copy.deepcopy(getattr(planner, 'shared_work_order', {})),
        'work_contracts': {a.id: copy.deepcopy(getattr(a, '_shared_work_contract', None))
                           for a in engine.agents},
        'machine_cycles': copy.deepcopy(engine._manufacturing_cycles),
        'active_sites': [copy.deepcopy(s) for s in engine.placed_structures
                         if s.get('under_construction') and not s.get('destroyed')],
    }
    write_json(output / 'result.json', result)
    tactical_end = {
        agent.id: copy.deepcopy(agent.q_table) for agent in engine.agents
    }
    if not evaluation:
        write_json(output / 'policy_end.json', policy.dump())
        if args.carry_tactical:
            write_json(output / 'tactical_policy_end.json', tactical_end)
    if evaluation:
        if operational_before != policy.operational_q_table:
            raise AssertionError('Frozen operational policy changed Q values')
        for agent in engine.agents:
            for state, actions in tactical_before[agent.id].items():
                if state == '__tactical_meta__' or not isinstance(actions, dict):
                    continue
                for action, value in actions.items():
                    if agent.q_table.get(state, {}).get(action) != value:
                        raise AssertionError('Frozen tactical policy changed learned Q values')
        write_json(output / 'freeze_checks.json', {
            'strategic': result['q_values_unchanged'], 'operational': True,
            'existing_tactical_values': True,
            'note': 'Unseen tactical actions may receive their fixed initial Q values; no learning occurs.'})
    if evaluation and not result['q_values_unchanged']:
        raise AssertionError('Frozen strategy changed Q values')
    print(json.dumps({'finished': name, 'score': result['score']['overall'],
                      'alive': result['alive'], 'errors': len(errors)}), flush=True)
    if errors:
        raise RuntimeError('Processing regression; stopping experiment, checkpoint retained')
    return policy.dump(), tactical_end, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--planet', default='kepler-442b')
    parser.add_argument('--ticks', type=int, default=51840)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--control', action='store_true')
    parser.add_argument(
        '--carry-tactical', action='store_true',
        help='Carry crew tactical and movement Q-tables between episodes',
    )
    parser.add_argument(
        '--skip-evaluation', action='store_true',
        help='Do not add a final frozen evaluation episode',
    )
    parser.add_argument(
        '--evaluate-from', type=Path,
        help='Run one frozen evaluation from a training checkpoint directory',
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.ERROR)
    vector_store._use_tfidf_fallback = True
    label = (
        'checkpoint_evaluation' if args.evaluate_from
        else 'control' if args.control else 'training'
    )
    sources = ['src/agents/decision.py', 'src/agents/strategic_rl.py', 'src/orchestration/engine.py',
               'config/recipes.json', 'config/mission_profile.json', 'rl_learning_experiment.py']
    hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources}
    write_json(args.output / (label + '_sources.json'), hashes)
    if args.evaluate_from:
        checkpoint = args.evaluate_from
        learned = json.loads(
            (checkpoint / 'policy_end.json').read_text(encoding='utf-8')
        )
        tactical_path = checkpoint / 'tactical_policy_end.json'
        learned_tactical = (
            json.loads(tactical_path.read_text(encoding='utf-8'))
            if tactical_path.exists() else None
        )
        if learned_tactical is not None:
            args.carry_tactical = True
        run_episode(
            args, 'trained_frozen_evaluation', learned, learned_tactical,
            evaluation=True,
        )
        return
    if args.control:
        run_episode(args, 'fresh_frozen_control', evaluation=True)
        return
    learned = None
    learned_tactical = None
    summaries = []
    for index in range(1, args.episodes + 1):
        learned, learned_tactical, result = run_episode(
            args, f'train_{index}', learned, learned_tactical,
        )
        summaries.append({
            key: result[key]
            for key in (
                'name', 'score', 'alive', 'strategy', 'structures',
                'movement_effort_rl', 'tactical_state_counts', 'wall_seconds',
            )
        })
        write_json(args.output / 'training_summary.json', summaries)
    if not args.skip_evaluation:
        run_episode(
            args, 'trained_frozen_evaluation', learned, learned_tactical,
            evaluation=True,
        )


if __name__ == '__main__':
    main()
