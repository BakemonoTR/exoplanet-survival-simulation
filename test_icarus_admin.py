"""Focused tests for ICARUS authentication and scoped telemetry controls."""

import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import server
from src.api.database import SimulationDB


class IcarusAdminHTTPTest(unittest.TestCase):
    def setUp(self):
        server._admin_sessions.clear()
        server._login_attempts.clear()

    def tearDown(self):
        server._admin_sessions.clear()
        server._login_attempts.clear()

    def test_admin_requires_login_and_csrf_for_mutation(self):
        with patch.dict(os.environ, {"ICARUS_ADMIN_PASSWORD": "orbit-test"}):
            client = TestClient(server.app)
            self.assertEqual(401, client.get("/api/admin/status").status_code)

            login = client.post(
                "/api/admin/login", json={"password": "orbit-test"}
            )
            self.assertEqual(200, login.status_code)
            csrf = login.json()["csrf_token"]
            self.assertTrue(client.cookies.get("icarus_session"))
            self.assertEqual(200, client.get("/api/admin/status").status_code)

            denied = client.post(
                "/api/admin/control", json={"action": "stop"}
            )
            self.assertEqual(403, denied.status_code)
            accepted = client.post(
                "/api/admin/control",
                json={"action": "stop"},
                headers={"X-CSRF-Token": csrf},
            )
            self.assertEqual(200, accepted.status_code)

    def test_icarus_page_is_served_with_private_cache_headers(self):
        client = TestClient(server.app)
        response = client.get("/icarus")
        self.assertEqual(200, response.status_code)
        self.assertIn("ICARUS Mission Control", response.text)
        self.assertEqual("no-store", response.headers["cache-control"])
        self.assertEqual("DENY", response.headers["x-frame-options"])


class IcarusPersistenceTest(unittest.TestCase):
    def test_planet_reset_and_attempt_discard_are_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            db = SimulationDB(os.path.join(directory, "icarus.db"))
            first_run = db.start_run("kepler-442b", 1, 100)
            db.save_agent_q_table(
                "agent", {"kepler": {"build": 2.0}}, 2.0, 10
            )
            db.end_run("timeout", 10, 20)
            second_run = db.start_run("trappist-1e", 2, 100)
            db.save_agent_q_table(
                "agent", {"trappist": {"gather": 3.0}}, 3.0, 10
            )
            db.end_run("timeout", 10, 30)

            deleted = db.discard_run_rl(second_run)
            self.assertEqual(1, deleted["policies_deleted"])
            self.assertEqual(
                {"kepler": {"build": 2.0}},
                db.load_agent_q_table("agent", "kepler-442b"),
            )
            self.assertIsNone(db.load_agent_q_table("agent", "trappist-1e"))

            db.reset_rl_policies("kepler-442b")
            self.assertIsNone(db.load_agent_q_table("agent", "kepler-442b"))
            self.assertGreater(first_run, 0)
            db.close()

    def test_tick_telemetry_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            db = SimulationDB(os.path.join(directory, "telemetry.db"))
            run_id = db.start_run("ross-128b", 4, 100)
            db.record_tick_telemetry(7, {
                "tick": 7,
                "agents": [{"id": "a", "rl_policy": {"total_reward": 1.5}}],
                "events": [{"type": "move"}],
            })
            db.record_tick_telemetry(8, {"tick": 8, "agents": [], "events": []})
            db.record_tick_telemetry(9, {"tick": 9, "agents": [], "events": []})
            db.end_run("manual_stop", 9, 0)
            rows = db.get_tick_telemetry(run_id)
            self.assertEqual(7, rows[0]["tick"])
            self.assertEqual("move", rows[0]["events"][0]["type"])
            self.assertEqual(
                [8, 9],
                [row["tick"] for row in db.get_tick_telemetry(
                    run_id, limit=2, latest=True
                )],
            )
            db.close()


if __name__ == "__main__":
    unittest.main()
