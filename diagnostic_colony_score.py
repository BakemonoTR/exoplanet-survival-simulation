"""
Full 51,840-tick diagnostic — uses run_headless with tick_speed=0.
Captures score/structure changes via on_tick callback at intervals.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.orchestration.engine import SimulationEngine
from src.agents.agent import Agent
from src.systems.mission_profile import MissionProfile

PLANET = "kepler-442b"
SEED = 42
SAMPLE_INTERVAL = 1000

config_dir = os.path.join(os.path.dirname(__file__), 'config')
planet_path = os.path.join(config_dir, 'planets', f'{PLANET}.json')

t0 = time.time()
last_score = -1.0
last_structures = {}
action_summary = {}
score_log = []
deaths_log = []

def on_tick(tick, state):
    global last_score, last_structures
    score = state.get("colony_score", {}).get("overall", 0)
    structs = state.get("structures", {})
    agents = state.get("agents", [])
    alive = sum(1 for a in agents if a.get("status") != "dead")

    # Track actions
    for a in agents:
        if a.get("status") == "dead":
            continue
        act = a.get("action", {})
        act_type = act if isinstance(act, str) else act.get("action_type", "?")
        action_summary[act_type] = action_summary.get(act_type, 0) + 1

    structs_changed = structs != last_structures
    score_changed = abs(score - last_score) > 0.05
    is_sample = tick % SAMPLE_INTERVAL == 0

    if is_sample or score_changed or structs_changed:
        elapsed = time.time() - t0
        tps = max(0.1, (tick + 1) / max(0.01, elapsed))
        eta_m = (51840 - tick) / tps / 60

        colony_keys = {"solar_panel","isru_o2_unit","water_collector","water_purifier",
                      "greenhouse","habitat_module",
                      "communications_array","power_distribution_grid",
                      "oxygen_buffer_tank","potable_water_tank","radiator_panel"}
        relevant = {k: v for k, v in structs.items() if k in colony_keys and v > 0}

        parts = [f"T={tick:>5}"]
        parts.append(f"S={score:>5.1f}%")
        parts.append(f"A={alive}/6")
        parts.append(f"{tps:.0f}t/s")
        parts.append(f"ETA={eta_m:.1f}m")
        if relevant:
            parts.append(f"C={relevant}")

        if structs_changed:
            for k in set(structs.keys()) - set(last_structures.keys()):
                parts.append(f"+{k}={structs[k]}")
            for k in set(structs.keys()) & set(last_structures.keys()):
                if structs[k] != last_structures.get(k, 0):
                    parts.append(f"^{k}:{last_structures.get(k,0)}->{structs[k]}")

        print(" | ".join(parts), flush=True)
        score_log.append({"tick": tick, "score": score, "alive": alive, "colony": dict(relevant)})
        last_score = score
        last_structures = dict(structs)

engine = SimulationEngine(
    planet_config_path=planet_path,
    seed=SEED,
    tick_speed=0,
    on_tick=on_tick,
)
# Override checkpoint interval to reduce DB overhead
engine.CHECKPOINT_INTERVAL = 500

preset_path = os.path.join(config_dir, 'agent_presets.json')
if os.path.exists(preset_path):
    with open(preset_path, 'r', encoding='utf-8') as f:
        presets = json.load(f)
    crew_count = MissionProfile.load().advance_crew
    for preset in presets.get("agents", [])[:crew_count]:
        engine.add_agent(Agent(preset))

MAX_TICKS = engine.max_ticks
print(f"=== FULL {MAX_TICKS}-TICK SIMULATION (tick_speed=0) ===", flush=True)
print(f"Planet: {PLANET} | Agents: {len(engine.agents)}", flush=True)

report = engine.run_headless(ticks=MAX_TICKS)

elapsed = time.time() - t0
print(f"\n{'='*70}")
print(f"COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}m)")
print(f"  End: {report['end_reason']}")
print(f"  Ticks: {report['total_ticks']}")
print(f"  Score: {report['colony_score'].get('overall', 0):.1f}%")
print(f"  Alive: {report['agents_alive']}/{report['agents_alive']+report['agents_dead']}")
print(f"  Structures: {json.dumps(dict(report.get('structures', report['colony_score'])), indent=2)}")
print(f"\n  Category scores:")
for cat, sc in report['colony_score'].get('categories', {}).items():
    print(f"    {cat:25s}: {sc*100:5.1f}%")
print(f"\n  Action distribution:")
for act, count in sorted(action_summary.items(), key=lambda x: -x[1])[:15]:
    print(f"    {act:25s}: {count:>8}")
if report.get('deaths'):
    print(f"\n  Deaths:")
    for d in report['deaths']:
        print(f"    {d['name']}: {d['cause']} (tick {d.get('tick','?')})")

# Dump score progression
print(f"\n  Score progression ({len(score_log)} samples):")
for s in score_log:
    print(f"    T={s['tick']:>5} S={s['score']:>5.1f}% A={s['alive']} {s.get('colony',{})}")
print(f"{'='*70}")
