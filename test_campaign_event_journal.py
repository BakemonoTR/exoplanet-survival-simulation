"""Regression coverage for the streamed campaign event journal."""

import json
import tempfile
import unittest
from pathlib import Path

from campaign_audit import _attempt
from src.memory import vector_store


class CampaignEventJournalTests(unittest.TestCase):
    def test_every_emitted_event_is_preserved_in_jsonl(self):
        """The audit must retain all events without retaining them in RAM."""
        vector_store._use_tfidf_fallback = True
        with tempfile.TemporaryDirectory() as temp_dir:
            journal = Path(temp_dir) / "events.jsonl"
            with self.assertNoLogs(
                "src.orchestration.engine", level="ERROR"
            ):
                result, _strategic, _tactical = _attempt(
                    "kepler-442b",
                    1,
                    seed=42,
                    max_ticks=120,
                    strategic_policy=None,
                    tactical_policies={},
                    event_log_path=journal,
                )

            self.assertTrue(journal.exists())
            rows = [
                json.loads(line)
                for line in journal.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(result["event_count"], len(rows))
            self.assertTrue(all("type" in event for event in rows))
            self.assertEqual(
                len(result["construction_completions"]),
                sum(event.get("type") == "construction_complete" for event in rows),
            )
            self.assertEqual(result["event_log"], str(journal))


if __name__ == "__main__":
    unittest.main()
