"""
Exoplanet Colony Simulation — Main Entry Point.

Usage:
    python run.py                     # Start web server (default)
    python run.py --headless          # Run simulation without UI
    python run.py --planet kepler-442b --seed 42
    python run.py --port 8080
"""

import argparse
import logging
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def mission_default_max_ticks() -> int:
    """Read the CLI attempt cap from the same validated mission contract."""
    from src.systems.mission_profile import MissionProfile

    max_ticks = MissionProfile.load().clock.planet_attempt_max_ticks
    if max_ticks is None:
        raise ValueError("mission profile must define an attempt tick limit")
    return int(max_ticks)


def run_headless(planet: str, seed: int, max_ticks: int):
    """Run simulation without web server."""
    from src.orchestration.engine import SimulationEngine
    from src.agents.agent import Agent
    from src.systems.mission_profile import MissionProfile
    
    config_dir = os.path.join(os.path.dirname(__file__), 'config')
    planet_path = os.path.join(config_dir, 'planets', f'{planet}.json')
    
    if not os.path.exists(planet_path):
        available = [f.replace('.json', '') for f in os.listdir(os.path.join(config_dir, 'planets')) if f.endswith('.json')]
        print(f"Planet '{planet}' not found. Available: {available}")
        sys.exit(1)
    
    # Live status display every N ticks
    STATUS_INTERVAL = 10  # Print every 10 ticks

    def on_tick(tick: int, state: dict):
        if tick % STATUS_INTERVAL != 0:
            return
        agents = state.get("agents", [])
        score = state.get("colony_score", {}).get("overall", 0)
        alive = sum(1 for a in agents if a.get("status") not in ("dead",))
        sim_day = round(
            tick / engine.mission_profile.clock.ticks_per_earth_day, 2
        )
        print(f"\n  -- Tick {tick:>4} | Day {sim_day:>5.2f} | Score: {score:>5.1f}% | Alive: {alive}/{len(agents)} --")
        for a in agents:
            needs = a.get("needs", {})
            h  = needs.get("hunger", 0)
            t  = needs.get("thirst", 0)
            e  = needs.get("energy", 0)
            ts = needs.get("temperature_stress", 50)
            status = a.get("status", "?")
            status_display = status.upper()
            action = a.get("action", {})
            act = action if isinstance(action, str) else action.get("action_type", "?")
            # ASCII level indicators: [!!!]=critical <20, [! ]=low <50, [ok]=fine
            def lvl(v): return "[!!!]" if v < 20 else ("[! ]" if v < 50 else "[ ok]")
            def tmp(v): return "[!!!]" if v < 10 or v > 90 else ("[! ]" if v < 25 or v > 75 else "[ ok]")
            name_short = a.get("name", "?")[:20]
            print(f"    {name_short:<20} {status_display:<11} "
                  f"H:{lvl(h)}{h:>5.1f}  T:{lvl(t)}{t:>5.1f}  "
                  f"E:{lvl(e)}{e:>5.1f}  Tmp:{tmp(ts)}{ts:>5.1f}  -> {act}")

    # Create engine with live callback
    engine = SimulationEngine(
        planet_config_path=planet_path,
        seed=seed,
        max_ticks=max_ticks,
        on_tick=on_tick,
    )
    
    # Load agents
    preset_path = os.path.join(config_dir, 'agent_presets.json')
    if os.path.exists(preset_path):
        with open(preset_path, 'r', encoding='utf-8') as f:
            presets = json.load(f)
        crew_count = MissionProfile.load().advance_crew
        for preset in presets.get("agents", [])[:crew_count]:
            engine.add_agent(Agent(preset))
    
    print(f"\n{'='*60}")
    print(f"  EXOPLANET COLONY SIMULATION")
    print(f"  Planet: {planet}")
    print(f"  Seed: {seed}")
    print(f"  Agents: {len(engine.agents)}")
    print(f"  Max ticks: {max_ticks}")
    print(f"  Status update every {STATUS_INTERVAL} ticks")
    print(f"{'='*60}\n")
    
    # Run
    report = engine.run_headless(ticks=max_ticks)
    
    # Print report
    print(f"\n{'='*60}")
    print(f"  SIMULATION COMPLETE")
    print(f"  Result: {report['end_reason']}")
    print(f"  Duration: {report['total_ticks']} ticks ({report['sim_days']} sim days)")
    print(f"  Colony Score: {report['colony_score'].get('overall', 0):.1f}%")
    print(f"  Survivors: {report['agents_alive']}/{report['agents_alive'] + report['agents_dead']}")
    
    if report['deaths']:
        print(f"\n  Deaths:")
        for d in report['deaths']:
            print(f"    - {d['name']}: {d['cause']} (tick {d.get('tick', '?')})")
    
    llm = report.get('llm_stats', {})
    print(f"\n  LLM Stats:")
    print(f"    API calls: {llm.get('total_calls', 0)}")
    print(f"    Tokens used: {llm.get('total_tokens', 0)}")
    print(f"    Fallback calls: {llm.get('fallback_calls', 0)}")
    cache = llm.get('cache', {})
    print(f"    Cache hit rate: {cache.get('hit_rate', 0):.1%}")
    print(f"    Avg tick time: {report.get('avg_tick_ms', 0):.1f}ms")
    print(f"{'='*60}\n")
    
    return report


def run_server(host: str, port: int):
    """Start web server with dashboard."""
    print(f"\n{'='*60}")
    print(f"  EXOPLANET COLONY SIMULATION SERVER")
    print(f"  Dashboard: http://{host}:{port}")
    print(f"  API docs:  http://{host}:{port}/docs")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")
    
    import uvicorn
    # Simulation checkpoints update the local database frequently. Uvicorn's
    # project-wide file watcher treated those writes as source edits and
    # restarted the live engine, producing intermittent/mixed frontend state.
    uvicorn.run("src.api.server:app", host=host, port=port, reload=False, log_level="info")


def run_analysis(mode: str, planet: str = None, export: str = None):
    """Run post-simulation analysis tools."""
    import subprocess
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'simulations.db')
    if not os.path.exists(db_path):
        print("No simulation database found. Run a simulation first.")
        sys.exit(1)
    script = {
        "compare": "analysis/compare.py",
        "survival": "analysis/survival_curve.py",
    }.get(mode)
    if not script:
        print(f"Unknown analysis mode: {mode}")
        sys.exit(1)
    cmd = [sys.executable, script]
    cmd += ["--db", db_path]
    if planet:
        cmd += ["--planet", planet]
    if export:
        cmd += ["--export", export]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description='Exoplanet Colony Simulation')
    parser.add_argument('--headless', action='store_true', help='Run without web UI')
    parser.add_argument('--planet', default='kepler-442b', help='Planet config name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--max-ticks', type=int, default=mission_default_max_ticks(),
        help='Max simulation ticks (default: mission profile attempt limit)',
    )
    parser.add_argument('--host', default='127.0.0.1', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--analyze', choices=['compare', 'survival'],
                        help='Run analysis on past simulation runs')
    parser.add_argument('--export', type=str, default=None,
                        help='Export analysis to file (with --analyze)')
    
    args = parser.parse_args()
    
    if args.analyze:
        run_analysis(args.analyze, planet=args.planet, export=args.export)
    elif args.headless:
        run_headless(args.planet, args.seed, args.max_ticks)
    else:
        run_server(args.host, args.port)


if __name__ == "__main__":
    main()
