"""Replay the archived rejected planner in isolation and record the casualty window."""
import argparse
import dataclasses
import hashlib
import json
import logging
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import rl_learning_experiment as experiment
import src.orchestration.engine as engine_module
from src.memory import vector_store

ROOT = Path(__file__).resolve().parent

def candidate_source():
    source = (ROOT / 'src/agents/decision.py').read_text(encoding='utf-8')
    patch = (ROOT / 'reports/last_unit_supply_annual_20260914/rejected_candidate.patch').read_text(encoding='utf-8')
    patch = patch.split('--- a/test_shared_construction_realism.py')[0]
    for hunk in patch.split('@@')[2::2]:
        old, new = [], []
        for line in hunk.splitlines(keepends=True):
            if line.startswith((' ', '-')):
                old.append(line[1:])
            if line.startswith((' ', '+')):
                new.append(line[1:])
        before, after = ''.join(old), ''.join(new)
        assert before and source.count(before) == 1, before[:100]
        source = source.replace(before, after, 1)
    return source

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ticks', type=int, default=43300)
    parser.add_argument('--current-planner', action='store_true')
    parser.add_argument('--trace-start', type=int, default=42500)
    parser.add_argument('--trace-end', type=int, default=43300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = ((ROOT / 'src/agents/decision.py').read_text(encoding='utf-8')
              if args.current_planner else candidate_source())
    module = types.ModuleType('candidate_decision')
    module.__file__ = str(ROOT / 'src/agents/decision.py')
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    engine_module.DecisionEngine = module.DecisionEngine
    experiment.write_json(args.output / 'replay_sources.json', {
        'candidate_decision_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'engine_sha256': hashlib.sha256((ROOT/'src/orchestration/engine.py').read_bytes()).hexdigest(),
    })
    original = engine_module.SimulationEngine._process_agent_tick
    with (args.output / 'casualty_trace.jsonl').open('w', encoding='utf-8') as trace:
        def snapshot(agent):
            return {
                'id': agent.id, 'position': [agent.x, agent.y],
                'status': str(agent.status), 'inside': agent._in_habitat,
                'injury': agent.injury_level, 'needs': vars(agent.needs).copy(),
                'action': vars(agent.action).copy(),
                'expedition': getattr(agent, '_active_expedition', None),
                'medical': {k: v for k,v in vars(agent).items()
                            if any(part in k for part in ('dehyd', 'drink', 'water', 'medical', 'rescu'))},
            }
        def audited(self, agent, *a, **kw):
            record = args.trace_start <= self.current_tick <= args.trace_end
            before = json.loads(json.dumps(snapshot(agent), default=str)) if record else None
            result = original(self, agent, *a, **kw)
            if record:
                trace.write(json.dumps({
                    'tick': self.current_tick, 'before': before,
                    'after': snapshot(agent), 'events': result,
                    'rover': vars(self.surface_fleet.crew_rovers[0]).copy(),
                }, default=str) + '\n')
            return result
        engine_module.SimulationEngine._process_agent_tick = audited
        logging.basicConfig(level=logging.ERROR)
        vector_store._use_tfidf_fallback = True
        experiment.run_episode(SimpleNamespace(
            output=args.output, seed=42, planet='kepler-442b',
            ticks=args.ticks, carry_tactical=False,
        ), 'replay')

if __name__ == '__main__':
    main()
