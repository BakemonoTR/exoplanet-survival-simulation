"""Opt-in, full-physics seed-42 readiness regression."""
import os
import json
import sqlite3
import time
import unittest
from pathlib import Path

from src.agents.agent import create_team_from_presets
from src.memory import vector_store
from src.orchestration.engine import SimulationEngine

ROOT = Path(__file__).resolve().parent
vector_store._use_tfidf_fallback = True


class _ReadOnlyPolicySnapshot:
    """Load a recorded policy without creating another simulation DB run."""
    def __init__(self, run_id):
        self.run_id = int(run_id)
        self.conn = sqlite3.connect(
            f"file:{ROOT / 'data/simulations.db'}?mode=ro", uri=True
        )

    def load_agent_q_table(self, agent_id, planet_id=None):
        row = self.conn.execute(
            "SELECT q_table_json FROM agent_q_tables WHERE run_id=? AND agent_id=? "
            "ORDER BY updated_tick DESC, id DESC LIMIT 1",
            (self.run_id, agent_id),
        ).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def save_agent_q_tables(self, *args, **kwargs):
        return 0

    def save_agent_q_table(self, *args, **kwargs):
        return False


@unittest.skipUnless(os.getenv("RUN_LONG_READINESS") == "1", "opt-in long regression")
class AcceleratedReadinessRegressionTest(unittest.TestCase):
    def test_seed42_reaches_eighty_percent_by_tick_27040(self):
        policy_run = os.getenv("READINESS_Q_RUN")
        engine = SimulationEngine(
            str(ROOT / "config/planets/kepler-442b.json"),
            seed=42, max_ticks=27_040, tick_speed=0.001,
            db=_ReadOnlyPolicySnapshot(policy_run) if policy_run else False,
        )
        engine.llm_client.call = lambda *_args, **_kwargs: None
        for agent in create_team_from_presets(str(ROOT / "config/agent_presets.json")):
            engine.add_agent(agent)

        engine._init_agents()
        started = time.monotonic()
        score = 0.0
        while engine.current_tick < 27_040 and score < 80.0:
            engine._run_tick()
            score = engine.colony_score.get_overall_score()
            engine.current_tick += 1
            if engine.current_tick % 1000 == 0:
                print(f" tick={engine.current_tick} score={score:.1f}", flush=True)
            if engine._check_end_conditions():
                break
        report = engine._get_final_report()
        print(f" elapsed={time.monotonic() - started:.1f}s", flush=True)
        self.assertGreaterEqual(
            score, 80.0,
            f"tick={engine.current_tick}, score={score}, alive={report['agents_alive']}, "
            f"categories={report['colony_score']['categories']}, "
            f"structures={report['structures_built']}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
