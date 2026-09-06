"""Summarize completed or still-streaming full-mission diagnostic traces."""
import argparse
from collections import Counter, deque
import gzip
import json
from pathlib import Path


def summarize(path):
    stationary, longest, blocked, blocked_longest = {}, {}, {}, {}
    indoor_blocked, indoor_longest = {}, {}
    recent, first_medical, deaths = {}, {}, []
    action_counts = Counter()
    airlock_overdue = 0
    last = None
    try:
        with gzip.open(path / 'crew_trace.jsonl.gz', 'rt', encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                last = row
                tick = row['tick']
                lock = row['airlock']
                if lock['ready_tick'] is not None and tick > lock['ready_tick'] + 3:
                    airlock_overdue += 1
                for crew in row['crew']:
                    name = crew['id']
                    history = recent.setdefault(name, deque(maxlen=8))
                    compact = {k: crew[k] for k in ('x','y','inside','action','target','energy','hunger','thirst','o2')}
                    compact['tick'] = tick
                    if crew['alive']:
                        action_counts[crew['action']] += 1
                        key = (crew['x'], crew['y'], crew['inside'])
                        old, count = stationary.get(name, (None, 0))
                        count = count+1 if key == old else 1
                        stationary[name] = key, count
                        if count > longest.get(name, {}).get('ticks', 0):
                            longest[name] = {'ticks': count, 'start': tick-count+1, 'end': tick, **compact}
                        moving = crew['action'] in {'move', 'arrived'}
                        old, count = blocked.get(name, (None, 0))
                        count = count+1 if moving and old == key else (1 if moving else 0)
                        blocked[name] = key, count
                        if count > blocked_longest.get(name, {}).get('ticks', 0):
                            blocked_longest[name] = {'ticks': count, 'start': tick-count+1, 'end': tick, **compact}
                        old, count = indoor_blocked.get(name, (None, 0))
                        indoor_moving = moving and crew['inside']
                        count = count+1 if indoor_moving and old == key else int(indoor_moving)
                        indoor_blocked[name] = key, count
                        if count > indoor_longest.get(name, {}).get('ticks', 0):
                            indoor_longest[name] = {'ticks': count, 'start': tick-count+1, 'end': tick, **compact}
                        if crew['action'] in {'unconscious', 'medical_rest'} and name not in first_medical:
                            first_medical[name] = {'onset': compact, 'preceding': list(history), 'airlock': lock}
                    if not crew['alive'] and history and history[-1].get('alive', True):
                        deaths.append({'id': name, 'tick': tick, 'last_alive': list(history)})
                    compact['alive'] = crew['alive']
                    history.append(compact)
    except EOFError:
        pass  # The writer is still running; preceding complete lines are valid.
    result = {'last_tick': last['tick'] if last else None, 'longest_stationary': longest,
              'longest_motion_without_displacement': blocked_longest, 'first_medical': first_medical,
              'longest_indoor_motion_without_displacement': indoor_longest,
              'deaths': deaths, 'airlock_ready_overdue_ticks': airlock_overdue,
              'action_ticks': dict(action_counts), 'last_state': last}
    (path / 'trace_analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('last_tick','longest_indoor_motion_without_displacement','airlock_ready_overdue_ticks')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)
