import threading
import time
import unittest
from unittest.mock import patch

from src.api import server
from src.systems.mission_profile import MissionProfile


class DebugAutoRestartTests(unittest.TestCase):
    def tearDown(self):
        server._disable_debug_loop(emit=False)
        server._tick_buffer.clear()

    def _request(
        self, *, enabled=True, delay=0.1, max_attempts=15,
        challenge_mode=False,
    ):
        return server.SimulationStartRequest(
            planet="kepler-442b",
            seed=42,
            max_ticks=100,
            tick_speed=0.01,
            debug_auto_restart=enabled,
            debug_max_attempts=max_attempts,
            debug_restart_delay_seconds=delay,
            challenge_mode=challenge_mode,
        )

    def test_request_default_attempt_limit_comes_from_mission_profile(self):
        request = server.SimulationStartRequest()
        profile = MissionProfile.load()
        self.assertEqual(
            profile.clock.planet_attempt_max_ticks, request.max_ticks
        )
        self.assertEqual(
            profile.clock.realtime_seconds_per_tick, request.tick_speed
        )
        self.assertFalse(request.dialogue_generation_enabled)

    def test_dialogue_generation_is_explicit_opt_in(self):
        request = self._request()
        request.dialogue_generation_enabled = True
        server._configure_debug_session(request)
        snapshot = server._debug_session_snapshot()
        self.assertTrue(snapshot["dialogue_generation_enabled"])

    def test_attempt_history_records_active_seed(self):
        token = server._configure_debug_session(self._request(max_attempts=1))
        server._debug_session["active_seed"] = 42
        server._finish_debug_attempt(
            token,
            1,
            {"end_reason": "timeout", "total_ticks": 100, "deaths": []},
            1,
        )
        self.assertEqual(42, server._debug_session_snapshot()["history"][0]["seed"])

    def test_challenge_default_budget_matches_public_clock_capacity(self):
        request = server.SimulationStartRequest(
            challenge_mode=True, debug_auto_restart=True
        )
        token = server._configure_debug_session(request)
        self.assertGreater(token, 0)
        profile = MissionProfile.load()
        expected = int(
            profile.clock.public_challenge_days * 24 * 60 * 60
            // (
                profile.clock.planet_attempt_max_ticks
                * profile.clock.realtime_seconds_per_tick
            )
        )
        self.assertEqual(
            expected, server._debug_session_snapshot()["max_attempts"]
        )

    def test_fatal_attempt_is_recorded_and_restarted(self):
        token = server._configure_debug_session(self._request())
        restarted = threading.Event()

        def fake_launch(session_token, *, is_restart=False):
            self.assertEqual(token, session_token)
            self.assertTrue(is_restart)
            restarted.set()

        report = {
            "end_reason": "all_agents_dead",
            "total_ticks": 1730,
            "deaths": [{
                "name": "Dr. Elena Vasquez",
                "cause": "DeathCause.DEHYDRATION",
                "tick": 1730,
            }],
        }
        with patch.object(server, "_launch_simulation_attempt", fake_launch):
            server._finish_debug_attempt(token, 1, report, 184)
            self.assertTrue(restarted.wait(1.0))

        snapshot = server._debug_session_snapshot()
        self.assertEqual(1, len(snapshot["history"]))
        self.assertEqual("dehydration", snapshot["history"][0]["deaths"][0]["cause"])

    def test_manual_stop_cancels_pending_restart(self):
        token = server._configure_debug_session(self._request(delay=0.5))
        restarted = threading.Event()
        with patch.object(
            server,
            "_launch_simulation_attempt",
            lambda *args, **kwargs: restarted.set(),
        ):
            server._finish_debug_attempt(
                token, 1,
                {"end_reason": "all_agents_dead", "total_ticks": 10, "deaths": []},
                1,
            )
            server._disable_debug_loop(emit=False)
            time.sleep(0.6)
        self.assertFalse(restarted.is_set())
        self.assertFalse(server._debug_session_snapshot()["restart_pending"])

    def test_timeout_is_recorded_and_restarted(self):
        token = server._configure_debug_session(self._request())
        restarted = threading.Event()

        with patch.object(
            server,
            "_launch_simulation_attempt",
            lambda *args, **kwargs: restarted.set(),
        ):
            server._finish_debug_attempt(
                token, 1,
                {"end_reason": "timeout", "total_ticks": 100, "deaths": []},
                4,
            )
            self.assertTrue(restarted.wait(1.0))

        snapshot = server._debug_session_snapshot()
        self.assertEqual("timeout", snapshot["history"][0]["end_reason"])
        self.assertTrue(snapshot["history"][0]["consumed_attempt"])

    def test_engine_error_is_visible_but_not_consumed_or_restarted(self):
        token = server._configure_debug_session(self._request())
        # Launch normally increments before the worker can fail.
        server._debug_session["attempt"] = 1
        with patch.object(server, "_launch_simulation_attempt") as launch:
            server._finish_debug_attempt(
                token, 1,
                {
                    "end_reason": "engine_error",
                    "total_ticks": 0,
                    "deaths": [],
                    "error": "invalid atmosphere configuration",
                },
                5,
            )
            time.sleep(0.15)
            launch.assert_not_called()

        snapshot = server._debug_session_snapshot()
        self.assertEqual(0, snapshot["attempt"])
        self.assertFalse(snapshot["enabled"])
        self.assertEqual(
            "invalid atmosphere configuration", snapshot["last_error"]
        )
        self.assertFalse(snapshot["history"][0]["consumed_attempt"])

    def test_challenge_mode_records_terminal_score_before_restart(self):
        class FakeChallenge:
            def __init__(self):
                self.scores = []

            def finish_active_attempt(self, score):
                self.scores.append(score)
                return {
                    "planet": "kepler-442b",
                    "succeeded": False,
                    "challenge_status": "running",
                    "remaining_planets": ["kepler-442b"],
                }

            def to_dict(self):
                return {"state": {"status": "running"}}

        fake = FakeChallenge()
        previous = server._challenge
        server._challenge = fake
        try:
            token = server._configure_debug_session(
                self._request(challenge_mode=True)
            )
            restarted = threading.Event()
            with patch.object(
                server,
                "_launch_simulation_attempt",
                lambda *args, **kwargs: restarted.set(),
            ):
                server._finish_debug_attempt(
                    token, 1,
                    {
                        "end_reason": "timeout",
                        "total_ticks": 100,
                        "deaths": [],
                        "colony_score": {"overall": 17.5},
                    },
                    6,
                    planet="kepler-442b",
                )
                self.assertTrue(restarted.wait(1.0))
            self.assertEqual([17.5], fake.scores)
        finally:
            server._disable_debug_loop(emit=False)
            server._challenge = previous

    def test_nonfatal_end_does_not_restart(self):
        token = server._configure_debug_session(self._request())
        with patch.object(server, "_launch_simulation_attempt") as launch:
            server._finish_debug_attempt(
                token, 1,
                {"end_reason": "colony_ready", "total_ticks": 50, "deaths": []},
                2,
            )
            time.sleep(0.15)
            launch.assert_not_called()
        snapshot = server._debug_session_snapshot()
        self.assertFalse(snapshot["restart_pending"])
        self.assertEqual("colony_ready", snapshot["history"][0]["end_reason"])

    def test_attempt_limit_stops_fatal_restart_loop(self):
        token = server._configure_debug_session(
            self._request(max_attempts=1)
        )
        with patch.object(server, "_launch_simulation_attempt") as launch:
            server._finish_debug_attempt(
                token, 1,
                {"end_reason": "all_agents_dead", "total_ticks": 1730, "deaths": []},
                3,
            )
            time.sleep(0.15)
            launch.assert_not_called()
        snapshot = server._debug_session_snapshot()
        self.assertTrue(snapshot["limit_reached"])
        self.assertFalse(snapshot["enabled"])
        self.assertEqual(1, snapshot["max_attempts"])


if __name__ == "__main__":
    unittest.main()
