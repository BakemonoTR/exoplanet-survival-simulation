"""Fast, dependency-free acceptance checks for the ASAC-6 contract."""

import json
import math
import unittest
from copy import deepcopy
from pathlib import Path

from src.systems.colony_score import ColonyScore
from src.systems.cargo_ledger import CargoLedgerError, CargoMassLedger
from src.systems.mission_profile import MissionProfile
from src.systems.timebase import normalize_tick_based_config


class MissionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = MissionProfile.load()
        with open("config/recipes.json", encoding="utf-8") as handle:
            raw = json.load(handle)["recipes"]
        cls.recipes = normalize_tick_based_config(
            raw, cls.profile.clock.tick_minutes
        )
        cls.cargo_ledger = CargoMassLedger(cls.profile, cls.recipes).to_dict()

    def test_campaign_arithmetic(self):
        clock = self.profile.clock
        self.assertEqual(self.profile.advance_crew, 6)
        self.assertEqual(self.profile.operational_support_population, 106)
        self.assertEqual(clock.tick_minutes, 10.0)
        self.assertEqual(clock.planet_attempt_max_ticks, 51_840)
        self.assertEqual(clock.support_window_ticks, 4_320)
        self.assertAlmostEqual(clock.full_attempts_in_public_challenge, 75.0)

    def test_browser_launch_defaults_to_a_full_attempt(self):
        frontend = Path("frontend/index.html").read_text(encoding="utf-8")
        self.assertIn(
            'id="maxTicksInput" value="51840" min="100" max="51840"',
            frontend,
        )
        self.assertIn(
            "parseInt(maxTicksInput.value) : 51840",
            frontend,
        )

    def test_infrastructure_deadline_plus_soak_fits_attempt(self):
        full_categories = {
            "o2": 1.0, "water": 1.0, "food": 1.0,
            "shelter": 1.0, "energy": 1.0,
            "hazard_protection": 1.0,
        }
        score = {"overall": 100.0, "categories": full_categories}
        structures = {"communications_array": 1, "landing_zone": 1}
        final_tick = self.profile.clock.planet_attempt_max_ticks - 1

        almost = self.profile.mission_state(
            final_tick, score, structures,
            communications_operational=True,
            landing_zone_operational=True,
            support_soak_ticks=self.profile.clock.support_window_ticks - 1,
        )
        self.assertFalse(almost["surface_acceptance_ready"])

        ready = self.profile.mission_state(
            final_tick, score, structures,
            communications_operational=True,
            landing_zone_operational=True,
            support_soak_ticks=self.profile.clock.support_window_ticks,
        )
        self.assertTrue(ready["surface_acceptance_ready"])
        self.assertEqual(
            ready["support_soak"]["qualification_start_tick"], 47_520
        )

    def test_water_is_conjunctive_and_includes_crop_makeup(self):
        score = ColonyScore()
        complete = {
            "isru_o2_unit": 20,
            "water_collector": 14,
            "water_purifier": 6,
            "potable_water_tank": 6,
            "oxygen_buffer_tank": 2,
            "greenhouse": 6,
            "habitat_module": 9,
            "solar_panel": 18,
            "communications_array": 1,
        }
        score.update_counts(complete, communications_operational=True)
        self.assertEqual(score.get_scores()["water"], 1.0)
        self.assertEqual(score.get_scores()["shelter"], 1.0)
        self.assertEqual(score.get_scores()["hazard_protection"], 1.0)

        complete["water_collector"] = 13
        score.update_counts(complete, communications_operational=True)
        self.assertLess(score.get_scores()["water"], 1.0)

    def test_greenhouse_rates_match_declared_capacity(self):
        effects = self.recipes["greenhouse"]["output"]["effects"]
        ticks_per_day = self.profile.clock.ticks_per_earth_day
        self.assertAlmostEqual(
            effects["food_production_kcal_per_tick"] * ticks_per_day,
            47_700.0,
        )
        self.assertAlmostEqual(
            effects["water_consumption_liters_per_tick"] * ticks_per_day,
            4_000.0,
            places=3,
        )
        self.assertEqual(effects["first_harvest_days"], 28)
        self.assertNotIn("growth_cycle_ticks", effects)
        self.assertEqual(effects["crop_service_interval_hours"], 24)
        self.assertEqual(effects["harvest_batch_interval_hours"], 24)

    def test_prefab_payload_stays_inside_declared_ceiling(self):
        ledger = self.cargo_ledger
        self.assertTrue(ledger["closed"])
        self.assertFalse(ledger["readiness_credit"])
        self.assertAlmostEqual(
            2_590_866.127, ledger["uncrewed"]["payload_mass_kg"], places=3
        )
        self.assertAlmostEqual(
            9_133.873, ledger["uncrewed"]["reserve_kg"], places=3
        )
        self.assertEqual(
            0.0,
            ledger["uncrewed"]["reservations"][0][
                "additional_payload_mass_kg"
            ],
        )
        self.assertEqual(37_272.4, ledger["crew_lander"]["payload_mass_kg"])
        self.assertEqual(12_727.6, ledger["crew_lander"]["reserve_kg"])
        self.assertNotIn("radiation_shelter", self.recipes)
        self.assertIn(
            "integrated_habitat_storm_vault_liner",
            self.recipes["habitat_module"]["materials"],
        )
        self.assertIn(
            "qualified_pv_blanket_segment",
            self.recipes["solar_panel"]["materials"],
        )
        self.assertNotIn(
            "photovoltaic_laminate",
            self.recipes["solar_panel"]["materials"],
        )

    def test_payload_overflow_fails_before_a_trial_can_start(self):
        invalid_data = deepcopy(self.profile.data)
        invalid_data["cargo_architecture"][
            "total_delivered_payload_ceiling_kg"
        ] = 2_400_000.0

        with self.assertRaises(CargoLedgerError):
            CargoMassLedger(MissionProfile(invalid_data), self.recipes)


if __name__ == "__main__":
    unittest.main()
