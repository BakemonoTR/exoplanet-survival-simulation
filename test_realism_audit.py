"""500-tick system/regression tests for agent realism."""

import json
import unittest

from realism_audit import (
    AUDITED_AGENTS,
    DEFAULT_TICKS,
    _coordination_capacity,
    _is_repeated_short_sortie_pattern,
    _is_sustained_coordination_overage,
    _is_thermally_unsafe_field_sleep,
    run_audit,
)


class CoordinationCapacityAuditTest(unittest.TestCase):
    class StubEngine:
        agents = [object() for _ in range(6)]

        @staticmethod
        def _get_recipe(name):
            return {
                "isru_o2_unit": {
                    "construction": {"recommended_crew": 4}
                },
                "reduced_iron_ingot": {
                    "requires_structure": "stone_furnace"
                },
            }.get(name)

        @staticmethod
        def _operational_structure_count(name):
            return 2 if name == "stone_furnace" else 0

    def test_recommended_construction_crew_is_not_duplication(self):
        target = json.dumps({"recipe": "isru_o2_unit"})
        self.assertEqual(
            4,
            _coordination_capacity(self.StubEngine(), "build", target),
        )

    def test_refine_capacity_tracks_physical_machine_count(self):
        target = json.dumps({"output": "reduced_iron_ingot"})
        self.assertEqual(
            2,
            _coordination_capacity(self.StubEngine(), "refine", target),
        )

    def test_shared_emergency_return_is_not_productive_crowding(self):
        target = json.dumps({"destination": "shelter"})
        self.assertEqual(
            6,
            _coordination_capacity(self.StubEngine(), "move", target),
        )

    def test_common_base_waypoint_is_not_a_two_person_workstation(self):
        target = json.dumps({"x": 1027, "y": 1089, "dx": 1, "dy": 1})
        self.assertEqual(
            6,
            _coordination_capacity(self.StubEngine(), "move", target),
        )

    def test_long_campaign_uses_rate_not_three_isolated_sorties(self):
        self.assertFalse(_is_repeated_short_sortie_pattern(13, 1055))
        self.assertTrue(_is_repeated_short_sortie_pattern(4, 40))

    def test_coordination_warning_scales_with_audit_horizon(self):
        self.assertFalse(_is_sustained_coordination_overage(37, 51_840))
        self.assertTrue(_is_sustained_coordination_overage(300, 51_840))

    def test_temperate_field_nap_is_not_mislabeled_as_cold_emergency(self):
        self.assertFalse(_is_thermally_unsafe_field_sleep({
            "effective_temperature_c": 21.0,
            "temperature_stress": 45.7,
        }))
        self.assertTrue(_is_thermally_unsafe_field_sleep({
            "effective_temperature_c": 0.0,
            "temperature_stress": 50.0,
        }))


class RealismAudit500TickTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_audit(ticks=DEFAULT_TICKS, seed=42)

    def test_simulation_processes_true_500_ticks_for_six_agents(self):
        meta, summary = self.result["meta"], self.result["summary"]
        self.assertEqual(DEFAULT_TICKS, meta["ticks_completed"])
        self.assertEqual(AUDITED_AGENTS, meta["agent_count"])
        self.assertEqual(0, summary["processing_failures"])
        for name, agent in self.result["agents"].items():
            self.assertEqual(
                DEFAULT_TICKS,
                sum(agent["action_counts"].values()),
                f"{name} için eksik eylem kaydı",
            )

    def test_no_realism_violations(self):
        """Regression gate for every critical or warning-level audit finding."""
        findings = self.result["findings"]
        details = "\n".join(
            f"- {f['severity']} {f['code']}: {f['agent'] or 'sistem'} — {f['evidence']}"
            for f in findings
        )
        self.assertFalse(findings, f"Gerçekçilik ihlalleri:\n{details}")

    def test_agents_remain_inside_base_eva_operating_radius(self):
        local_limit = self.result["meta"]["local_eva_radius_cells"]
        for name, agent in self.result["agents"].items():
            allowed_limit = (
                agent["max_authorized_eva_radius_cells"]
                if agent["expedition_ticks"] else local_limit
            )
            self.assertLessEqual(
                agent["max_base_distance_cells"],
                allowed_limit,
                f"{name} base'den {agent['max_base_distance_cells']} hücre uzaklaştı (yetkili sınır={allowed_limit})",
            )

    def test_manufacturing_batches_are_exactly_once_balanced(self):
        manufacturing = self.result["summary"]["manufacturing"]
        self.assertTrue(manufacturing["exactly_once_balanced"])
        self.assertEqual(0, manufacturing["batch_balance_delta"])
        self.assertEqual(0.0, manufacturing["output_mass_balance_delta_kg"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
