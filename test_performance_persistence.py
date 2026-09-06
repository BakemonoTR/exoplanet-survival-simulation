"""Focused regressions for tick-safe LLM and compact RL persistence."""

import os
import sqlite3
import tempfile
import threading
import time
import unittest

from src.api.database import SimulationDB
from src.orchestration.llm_client import LocalNarrativeClient, LLMCallType


class NonBlockingSocialLLMTest(unittest.TestCase):
    def test_social_network_wait_never_blocks_physics_caller(self):
        worker_started = threading.Event()
        release_worker = threading.Event()

        class SlowLocalProvider:
            def generate(self, **_kwargs):
                worker_started.set()
                release_worker.wait(timeout=1.0)
                return {"dialogue": "ready", "_tokens_used": 7}

        client = LocalNarrativeClient(
            provider=SlowLocalProvider(), enabled=True, max_pending=2
        )
        try:
            started = time.perf_counter()
            result = client.call(
                "system", "same encounter",
                call_type=LLMCallType.SOCIAL,
                current_tick=25,
            )
            elapsed = time.perf_counter() - started

            self.assertIsNone(result)
            self.assertLess(elapsed, 0.15)
            self.assertTrue(worker_started.wait(timeout=0.5))

            # The same in-flight prompt is deduplicated rather than queued.
            client.call(
                "system", "same encounter",
                call_type=LLMCallType.SOCIAL,
                current_tick=25,
            )
            status = client.get_status()
            self.assertEqual(1, status["pending"])

            release_worker.set()
            deadline = time.time() + 1.0
            while client.get_status()["pending"]:
                self.assertLess(time.time(), deadline)
                time.sleep(0.005)

            # A later identical encounter can consume the completed cache entry.
            cached = client.call(
                "system", "same encounter",
                call_type=LLMCallType.SOCIAL,
                current_tick=26,
            )
            self.assertEqual("ready", cached["dialogue"])
            self.assertEqual(1, client.get_status()["total_calls"])
        finally:
            release_worker.set()
            client.close(wait=True)


class CompactQTablePersistenceTest(unittest.TestCase):
    def test_failed_flush_retains_policy_until_transaction_retry_succeeds(self):
        class FailingConnection:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def execute(self, *_args, **_kwargs):
                raise sqlite3.OperationalError("simulated disk failure")

        with tempfile.TemporaryDirectory() as directory:
            db = SimulationDB(os.path.join(directory, "retry.db"))
            db.start_run("test-planet", seed=4, max_ticks=100)
            db.save_agent_q_tables(
                [{
                    "agent_id": "shared",
                    "q_table": {"state": {"capacity:solar_panel": 7.0}},
                    "total_reward": 12.0,
                }],
                tick=20,
                flush=False,
            )
            original = db._conn
            db._conn = FailingConnection()
            self.assertFalse(db._flush_pending())
            self.assertEqual(1, len(db._pending_q_tables))

            db._conn = original
            self.assertTrue(db._flush_pending())
            self.assertEqual(0, len(db._pending_q_tables))
            self.assertEqual(
                {"state": {"capacity:solar_panel": 7.0}},
                db.load_agent_q_table("shared", planet_id="test-planet"),
            )
            db.close()

    def test_existing_history_schema_migrates_without_deleting_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "legacy.db")
            conn = sqlite3.connect(path)
            conn.execute(
                """CREATE TABLE agent_q_tables (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       run_id INTEGER,
                       agent_id TEXT NOT NULL,
                       q_table_json TEXT NOT NULL,
                       total_reward REAL DEFAULT 0,
                       updated_tick INTEGER NOT NULL,
                       timestamp REAL NOT NULL
                   )"""
            )
            conn.execute(
                """INSERT INTO agent_q_tables
                   (run_id, agent_id, q_table_json, total_reward,
                    updated_tick, timestamp)
                   VALUES (1, 'legacy', '{"state": {}}', 2, 25, 1.0)"""
            )
            conn.commit()
            conn.close()

            db = SimulationDB(path)
            columns = {
                row[1] for row in db._conn.execute(
                    "PRAGMA table_info(agent_q_tables)"
                ).fetchall()
            }
            self.assertIn("q_table_hash", columns)
            self.assertIn("is_latest", columns)
            self.assertEqual(
                1,
                db._conn.execute(
                    "SELECT COUNT(*) FROM agent_q_tables"
                ).fetchone()[0],
            )
            self.assertEqual({"state": {}}, db.load_agent_q_table("legacy"))
            db.close()

    def test_legacy_single_saves_batch_into_one_latest_row_per_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            db = SimulationDB(os.path.join(directory, "policies.db"))
            run_id = db.start_run("test-planet", seed=7, max_ticks=1000)

            for index in range(5):
                db.save_agent_q_table(
                    f"agent-{index}",
                    {"state": {"move": float(index)}},
                    total_reward=float(index),
                    tick=25,
                )

            # Legacy calls queue; the next normal checkpoint performs one
            # transaction for all five crew policies.
            self.assertEqual(5, len(db._pending_q_tables))
            db.checkpoint(30, [], {"overall": 0, "categories": {}})
            full_rows = db._conn.execute(
                "SELECT COUNT(*) FROM agent_q_tables WHERE run_id=?",
                (run_id,),
            ).fetchone()[0]
            self.assertEqual(5, full_rows)

            # Changed policies update those same rows instead of appending five
            # more full JSON snapshots.
            for index in range(5):
                db.save_agent_q_table(
                    f"agent-{index}",
                    {"state": {"move": float(index + 10)}},
                    total_reward=float(index + 10),
                    tick=50,
                )
            db.checkpoint(60, [], {"overall": 0, "categories": {}})
            full_rows = db._conn.execute(
                "SELECT COUNT(*) FROM agent_q_tables WHERE run_id=?",
                (run_id,),
            ).fetchone()[0]
            progress_rows = db._conn.execute(
                "SELECT COUNT(*) FROM agent_q_table_progress WHERE run_id=?",
                (run_id,),
            ).fetchone()[0]
            self.assertEqual(5, full_rows)
            self.assertEqual(5, progress_rows)
            self.assertEqual(
                {"state": {"move": 10.0}},
                db.load_agent_q_table("agent-0", planet_id="test-planet"),
            )
            db.close()

    def test_explicit_batch_coalesces_repeated_agent_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            db = SimulationDB(os.path.join(directory, "batch.db"))
            run_id = db.start_run("test-planet", seed=8, max_ticks=1000)
            queued = db.save_agent_q_tables(
                [
                    {"agent_id": "a", "q_table": {"old": {}}, "total_reward": 1},
                    {"agent_id": "a", "q_table": {"new": {}}, "total_reward": 2},
                ],
                tick=25,
                flush=True,
            )

            self.assertEqual(2, queued)
            self.assertEqual(
                1,
                db._conn.execute(
                    "SELECT COUNT(*) FROM agent_q_tables WHERE run_id=?",
                    (run_id,),
                ).fetchone()[0],
            )
            self.assertEqual(
                {"new": {}},
                db.load_agent_q_table("a", planet_id="test-planet"),
            )
            db.close()


if __name__ == "__main__":
    unittest.main()
