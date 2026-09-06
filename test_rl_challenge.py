"""Regression tests for planet-scoped learning and global challenge state."""

import os
import tempfile
import unittest

from src.api.database import SimulationDB
from src.agents.agent import (
    TACTICAL_POLICY_META_KEY,
    TACTICAL_STATE_SCHEMA_VERSION,
    create_team_from_presets,
)
from src.agents.strategic_rl import (
    ColonyStrategicPolicy,
    META_KEY,
    STATE_SCHEMA_VERSION,
    STRATEGY_POLICY_ID,
    build_colony_strategy_state,
)
from src.memory import vector_store
from src.orchestration.challenge import (
    ChallengeConfig,
    PlanetChallengeCoordinator,
)
from src.orchestration.engine import SimulationEngine


ROOT = os.path.dirname(os.path.abspath(__file__))
vector_store._use_tfidf_fallback = True


class PlanetScopedRLTest(unittest.TestCase):
    def test_q_tables_are_loaded_only_for_the_same_planet(self):
        with tempfile.TemporaryDirectory() as directory:
            db = SimulationDB(os.path.join(directory, "test.db"))
            db.start_run("planet-a", seed=1, max_ticks=10)
            db.save_agent_q_table("agent-1", {"a": {"build": 3.0}}, 1, 2)
            db.end_run("timeout", 2, 0.0)
            db.start_run("planet-b", seed=2, max_ticks=10)
            db.save_agent_q_table("agent-1", {"b": {"explore": 4.0}}, 2, 3)

            self.assertEqual(
                {"a": {"build": 3.0}},
                db.load_agent_q_table("agent-1", planet_id="planet-a"),
            )
            self.assertEqual(
                {"b": {"explore": 4.0}},
                db.load_agent_q_table("agent-1", planet_id="planet-b"),
            )
            db.close()

    def test_terminal_penalty_reaches_recent_decision_chain(self):
        engine = SimulationEngine(
            os.path.join(ROOT, "config", "planets", "kepler-442b.json"),
            db=False,
        )
        agent = create_team_from_presets(
            os.path.join(ROOT, "config", "agent_presets.json")
        )[0]
        agent.q_table = {
            "s1": {"explore": 2.0},
            "s2": {"build:solar_panel": 2.0},
        }
        agent.rl_episode_trace = [
            ("s1", "explore"),
            ("s2", "build:solar_panel"),
        ]

        updated = engine.decision_engine.apply_terminal_rl_reward(agent, -50.0)

        self.assertEqual(2, updated)
        self.assertLess(agent.q_table["s1"]["explore"], 2.0)
        self.assertLess(agent.q_table["s2"]["build:solar_panel"], 2.0)
        self.assertEqual([], agent.rl_episode_trace)

    def test_dialogue_and_reflection_are_opt_in_for_headless_runs(self):
        engine = SimulationEngine(
            os.path.join(ROOT, "config", "planets", "kepler-442b.json"),
            db=False,
        )
        crew = create_team_from_presets(
            os.path.join(ROOT, "config", "agent_presets.json")
        )

        def unexpected_call(*_args, **_kwargs):
            raise AssertionError("disabled narrative path called provider")

        engine.decision_engine.llm.call = unexpected_call
        engine.decision_engine.reflection.reflect = unexpected_call

        self.assertIsNone(engine.decision_engine.handle_encounter(
            crew[0], crew[1], tick=25
        ))
        self.assertIsNone(engine.decision_engine._run_reflection(
            crew[0], tick=60
        ))

    def test_real_first_structure_completion_reaches_strategic_credit(self):
        engine = SimulationEngine(
            os.path.join(
                ROOT, "config", "planets", "proxima-centauri-b.json"
            ),
            seed=42,
            max_ticks=1_000,
            db=False,
        )
        for agent in create_team_from_presets(
            os.path.join(ROOT, "config", "agent_presets.json")
        )[:6]:
            engine.add_agent(agent)
        # This regression isolates exactly-once strategic completion credit;
        # severe-radiation shelter-first masking has its own policy test.
        engine.decision_engine.severe_radiation_environment = False

        learning_events = []
        while engine.current_tick < 900:
            tick_result = engine._run_tick()
            if tick_result.get("strategy_learning"):
                learning_events.append(tick_result["strategy_learning"])
            engine.current_tick += 1

        delivered_types = {
            "eclss_lander_hub", "stone_furnace", "forge", "cnc_fabricator",
        }
        accepted_field_completions = sum(
            count for recipe, count in engine.structures_built.items()
            if recipe not in delivered_types
        )
        # The planner may now rationally bootstrap stored water/O2 and their
        # distribution grid before the first large solar field.  This test is
        # about exactly-once learning credit for a real field completion, not
        # hard-coding one obsolete build order.
        self.assertGreaterEqual(accepted_field_completions, 1)
        self.assertTrue(
            learning_events,
            {
                "telemetry": engine.strategic_policy.telemetry(),
                "trace": list(engine.strategic_policy.episode_trace),
                "shared_work_order": dict(
                    engine.decision_engine.shared_work_order
                ),
            },
        )
        # Faster construction may finish several modules inside this window;
        # each physical QA-accepted completion must receive exactly one credit.
        self.assertEqual(accepted_field_completions, len(learning_events))
        self.assertEqual(
            accepted_field_completions,
            engine.strategic_policy.episode_verified_completions,
        )


class GlobalChallengeTest(unittest.TestCase):
    def test_one_deadline_and_failed_planet_stays_on_wheel(self):
        clock = [1_000.0]
        coordinator = PlanetChallengeCoordinator(
            ["a", "b"],
            config=ChallengeConfig(),
            seed=7,
            now=lambda: clock[0],
        )
        coordinator.start()
        deadline = coordinator.state.deadline_at
        first = coordinator.spin_next_planet()
        failed = coordinator.finish_active_attempt(43.0)

        self.assertTrue(failed["site_reset_required"])
        self.assertIn(first, coordinator.state.remaining_planets)
        self.assertEqual(deadline, coordinator.state.deadline_at)
        coordinator.spin_next_planet()
        self.assertEqual(deadline, coordinator.state.deadline_at)

    def test_only_score_100_removes_planet_and_protocol_is_frozen(self):
        coordinator = PlanetChallengeCoordinator(["only"], now=lambda: 10.0)
        coordinator.start()
        self.assertEqual(51_840, coordinator.config.episode_max_ticks)
        self.assertEqual(10.0, coordinator.config.sim_minutes_per_tick)
        self.assertEqual(2.0, coordinator.config.realtime_seconds_per_tick)
        self.assertIsNone(coordinator.config.stagnation_timeout_ticks)
        coordinator.spin_next_planet()
        outcome = coordinator.finish_active_attempt(100.0)

        self.assertTrue(outcome["succeeded"])
        self.assertEqual("completed", coordinator.state.status)
        self.assertEqual([], coordinator.state.remaining_planets)

    def test_protocol_yields_seventy_five_total_full_attempts(self):
        config = ChallengeConfig()
        public_ticks = int(
            config.global_duration_seconds / config.realtime_seconds_per_tick
        )
        self.assertEqual(3_888_000, public_ticks)
        self.assertEqual(75, public_ticks // config.episode_max_ticks)
        self.assertEqual(15, (public_ticks // config.episode_max_ticks) // 5)


class SharedStrategicRLTest(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {"recipe": "solar_panel"},
            {"recipe": "isru_o2_unit"},
            {"recipe": "water_collector"},
        ]

    def test_fresh_equal_values_do_not_encode_a_fixed_build_order(self):
        first_choices = set()
        for seed in range(16):
            policy = ColonyStrategicPolicy("planet-a", seed=seed)
            selected = policy.choose(
                state_key="fresh",
                candidates=self.candidates,
                structures_built={},
                tick=0,
                colony_score=0.0,
            )
            first_choices.add(selected["recipe"])
        self.assertGreater(len(first_choices), 1)

    def test_acceptance_balance_masks_an_already_ahead_category(self):
        policy = ColonyStrategicPolicy(
            "planet-a", seed=1, initial_epsilon=0.0, minimum_epsilon=0.0
        )
        policy.q_table["s0"] = {
            "capacity:solar_panel": 100.0,
            "capacity:isru_o2_unit": 0.0,
        }
        selected = policy.choose(
            state_key="s0",
            candidates=[
                {"recipe": "solar_panel", "fulfillment": 0.8},
                {"recipe": "isru_o2_unit", "fulfillment": 0.0},
            ],
            structures_built={}, tick=0, colony_score=0.0,
        )
        self.assertEqual("isru_o2_unit", selected["recipe"])

    def test_exhausting_bootstrap_power_preempts_an_unstarted_commitment(self):
        policy = ColonyStrategicPolicy(
            "planet-a", seed=1, initial_epsilon=0.0, minimum_epsilon=0.0
        )
        policy.q_table["s0"] = {
            "capacity:solar_panel": -100.0,
            "capacity:isru_o2_unit": 100.0,
        }
        first = policy.choose(
            state_key="s0", candidates=self.candidates,
            structures_built={}, tick=0, colony_score=0.0,
        )
        self.assertEqual("isru_o2_unit", first["recipe"])
        selected = policy.choose(
            state_key="s0",
            candidates=[
                {"recipe": "solar_panel", "bootstrap_power_priority": True},
                {"recipe": "isru_o2_unit", "ready": True},
            ],
            structures_built={}, tick=1, colony_score=0.0,
        )
        self.assertEqual("solar_panel", selected["recipe"])

    def test_severe_radiation_safety_mask_requires_first_shielded_habitat(self):
        policy = ColonyStrategicPolicy(
            "planet-a", seed=1, initial_epsilon=0.0, minimum_epsilon=0.0
        )
        policy.q_table["s0"] = {
            "capacity:solar_panel": 100.0,
            "capacity:habitat_module": -100.0,
        }
        selected = policy.choose(
            state_key="s0",
            candidates=[
                {"recipe": "solar_panel", "fulfillment": 0.0},
                {
                    "recipe": "habitat_module", "fulfillment": 0.0,
                    "safety_priority": True,
                },
            ],
            structures_built={}, tick=0, colony_score=0.0,
        )
        self.assertEqual("habitat_module", selected["recipe"])

    def test_exhausted_utility_envelope_requires_distribution_expansion(self):
        policy = ColonyStrategicPolicy(
            "planet-a", seed=1, initial_epsilon=0.0, minimum_epsilon=0.0
        )
        policy.q_table["s0"] = {
            "capacity:solar_panel": 100.0,
            "capacity:life_support_distribution_grid": -100.0,
        }
        selected = policy.choose(
            state_key="s0",
            candidates=[
                {"recipe": "solar_panel", "fulfillment": 0.0},
                {
                    "recipe": "life_support_distribution_grid",
                    "fulfillment": 0.75,
                    "utility_recovery_priority": True,
                },
            ],
            structures_built={"life_support_distribution_grid": 3},
            tick=0,
            colony_score=60.0,
        )
        self.assertEqual("life_support_distribution_grid", selected["recipe"])

    def test_water_fill_deadline_preempts_expansion_of_crop_water_load(self):
        policy = ColonyStrategicPolicy("planet-a", seed=1, initial_epsilon=0.0, minimum_epsilon=0.0)
        policy.q_table["s0"] = {"capacity:greenhouse": 100.0, "capacity:water_collector": -100.0}
        selected = policy.choose(
            state_key="s0", candidates=[
                {"recipe": "greenhouse", "fulfillment": 0.4, "ready": True},
                {"recipe": "water_collector", "fulfillment": 0.8, "ready": True,
                 "water_stock_deadline_priority": True},
            ], structures_built={}, tick=300 * 144, colony_score=70.0,
        )
        self.assertEqual("water_collector", selected["recipe"])

    def test_long_lead_utility_grid_does_not_idle_ready_independent_work(self):
        policy = ColonyStrategicPolicy(
            "planet-a", seed=1, initial_epsilon=0.0, minimum_epsilon=0.0
        )
        first = policy.choose(
            state_key="s0",
            candidates=[{
                "recipe": "life_support_distribution_grid",
                "fulfillment": 0.75,
                "utility_recovery_priority": True,
                "ready": False,
            }],
            structures_built={"life_support_distribution_grid": 3},
            tick=0,
            colony_score=60.0,
        )
        self.assertEqual("life_support_distribution_grid", first["recipe"])

        selected = policy.choose(
            state_key="s0",
            candidates=[
                {
                    "recipe": "life_support_distribution_grid",
                    "fulfillment": 0.75,
                    "utility_recovery_priority": True,
                    "ready": False,
                },
                {
                    "recipe": "solar_panel",
                    "fulfillment": 0.5,
                    "ready": True,
                    "life_support_endpoint": False,
                },
                {
                    "recipe": "water_collector",
                    "fulfillment": 0.5,
                    "ready": True,
                    "life_support_endpoint": True,
                },
            ],
            structures_built={"life_support_distribution_grid": 3},
            tick=1,
            colony_score=60.0,
        )

        self.assertEqual("solar_panel", selected["recipe"])
        self.assertEqual(
            "utility_materials_in_production",
            selected["strategy_replan_reason"],
        )

    def test_ready_bom_preempts_unstarted_long_lead_objective(self):
        policy = ColonyStrategicPolicy(
            "planet-a", seed=1, initial_epsilon=0.0, minimum_epsilon=0.0
        )
        policy.q_table["s0"] = {
            "capacity:greenhouse": 100.0,
            "capacity:solar_panel": -100.0,
        }
        selected = policy.choose(
            state_key="s0",
            candidates=[
                {
                    "recipe": "greenhouse", "fulfillment": 0.0,
                    "ready": False,
                },
                {
                    "recipe": "solar_panel", "fulfillment": 0.0,
                    "ready": True,
                },
            ],
            structures_built={}, tick=0, colony_score=0.0,
        )
        self.assertEqual("solar_panel", selected["recipe"])

    def test_water_shortage_does_not_select_unneeded_electrical_capacity(self):
        policy = ColonyStrategicPolicy("planet-a", seed=1, initial_epsilon=0.0, minimum_epsilon=0.0)
        state = "test|reserve[w:critical,o:buffered,f:buffered,e:buffered]"
        policy.q_table[state] = {"capacity:solar_panel": 100.0, "capacity:water_collector": 0.0}
        chosen = policy.choose(
            state_key=state,
            candidates=[{"recipe": "solar_panel", "water_reserve_support": False},
                        {"recipe": "water_collector", "water_reserve_support": True}],
            structures_built={}, tick=1, colony_score=0.0,
        )
        self.assertEqual("water_collector", chosen["recipe"])

    def test_concurrent_energy_emergency_retains_its_recovery_path(self):
        policy = ColonyStrategicPolicy("planet-a", seed=1, initial_epsilon=0.0, minimum_epsilon=0.0)
        state = "test|reserve[w:critical,o:buffered,f:buffered,e:critical]"
        policy.q_table[state] = {"capacity:solar_panel": 100.0, "capacity:water_collector": 0.0}
        chosen = policy.choose(
            state_key=state,
            candidates=[{"recipe": "solar_panel", "water_reserve_support": False},
                        {"recipe": "water_collector", "water_reserve_support": True}],
            structures_built={}, tick=1, colony_score=0.0,
        )
        self.assertEqual("solar_panel", chosen["recipe"])

    def test_repeated_ticks_do_not_reward_unfinished_work(self):
        policy = ColonyStrategicPolicy("planet-a", seed=3)
        selected = policy.choose(
            state_key="s0",
            candidates=self.candidates,
            structures_built={},
            tick=0,
            colony_score=0.0,
        )
        recipe = selected["recipe"]
        q_before = policy.q_table["s0"][f"capacity:{recipe}"]
        for tick in range(1, 100):
            reward = policy.observe_completion(
                structures_built={},
                tick=tick,
                colony_score=0.0,
                tick_minutes=10.0,
                next_state_key="s0",
            )
            self.assertEqual(0.0, reward)
        self.assertEqual(q_before, policy.q_table["s0"][f"capacity:{recipe}"])

    def test_verified_module_completion_rewards_exactly_once(self):
        policy = ColonyStrategicPolicy("planet-a", seed=9)
        selected = policy.choose(
            state_key="s0",
            candidates=self.candidates,
            structures_built={},
            tick=10,
            colony_score=0.0,
        )
        recipe = selected["recipe"]
        built = {recipe: 1}
        first = policy.observe_completion(
            structures_built=built,
            tick=20,
            colony_score=5.0,
            tick_minutes=10.0,
            next_state_key="s1",
        )
        second = policy.observe_completion(
            structures_built=built,
            tick=21,
            colony_score=5.0,
            tick_minutes=10.0,
            next_state_key="s1",
        )
        self.assertGreater(first, 0.0)
        self.assertEqual(0.0, second)
        self.assertIsNone(policy.commitment)

    def test_readiness_gain_dominates_equal_raw_completion(self):
        stagnant = ColonyStrategicPolicy("planet-a", seed=9)
        useful = ColonyStrategicPolicy("planet-a", seed=9)
        candidate = [{"recipe": "water_collector", "required_count": 14}]
        for policy in (stagnant, useful):
            policy.choose(
                state_key="s0", candidates=candidate,
                structures_built={}, tick=0, colony_score=0.0,
            )
        stagnant_reward = stagnant.observe_completion(
            structures_built={"water_collector": 1}, tick=10,
            colony_score=0.0, tick_minutes=10.0, next_state_key="s1",
        )
        useful_reward = useful.observe_completion(
            structures_built={"water_collector": 1}, tick=10,
            colony_score=2.0, tick_minutes=10.0, next_state_key="s1",
        )

        self.assertGreater(stagnant_reward, 0.0)
        self.assertGreater(useful_reward, stagnant_reward + 10.0)

    def test_choose_before_delayed_observation_preserves_completed_transition(self):
        policy = ColonyStrategicPolicy("planet-a", seed=13)
        selected = policy.choose(
            state_key="s0",
            candidates=self.candidates,
            structures_built={},
            tick=10,
            colony_score=0.0,
        )
        recipe = selected["recipe"]
        built = {recipe: 1}

        # The decision phase can run before the engine's delayed-reward phase.
        # It must not replace the just-completed commitment.
        self.assertIsNone(policy.choose(
            state_key="s1",
            candidates=self.candidates,
            structures_built=built,
            tick=20,
            colony_score=5.0,
        ))
        self.assertEqual(recipe, policy.commitment.recipe)
        first = policy.observe_completion(
            structures_built=built,
            tick=20,
            colony_score=5.0,
            tick_minutes=10.0,
            next_state_key="s1",
        )
        second = policy.observe_completion(
            structures_built=built,
            tick=21,
            colony_score=5.0,
            tick_minutes=10.0,
            next_state_key="s1",
        )
        self.assertGreater(first, 0.0)
        self.assertEqual(0.0, second)

    def test_collapse_and_rebuild_cannot_farm_completion_reward(self):
        policy = ColonyStrategicPolicy(
            "planet-a", seed=2, initial_epsilon=0.0, minimum_epsilon=0.0
        )
        solar = [{"recipe": "solar_panel"}]
        policy.choose(
            state_key="empty",
            candidates=solar,
            structures_built={},
            tick=0,
            colony_score=0.0,
        )
        first = policy.observe_completion(
            structures_built={"solar_panel": 1},
            tick=10,
            colony_score=5.0,
            tick_minutes=10.0,
            next_state_key="one",
        )
        self.assertGreater(first, 0.0)

        # A later loss and replacement restores capacity, but is not net-new.
        policy.choose(
            state_key="collapsed",
            candidates=solar,
            structures_built={"solar_panel": 0},
            tick=20,
            colony_score=0.0,
        )
        rebuilt = policy.observe_completion(
            structures_built={"solar_panel": 1},
            tick=30,
            colony_score=5.0,
            tick_minutes=10.0,
            next_state_key="one",
        )
        self.assertLessEqual(rebuilt, 0.0)
        self.assertFalse(policy.last_outcome["net_new_capacity"])

    def test_ready_recipe_identity_is_part_of_the_observed_state(self):
        common = {
            "category_scores": {},
            "colony_resources": {},
            "living_crew": 5,
            "fit_crew": 5,
            "active_site_type": None,
            "ready_recipe_count": 1,
            "surface_requires_plss": True,
            "eligible_recipes": {"solar_panel", "greenhouse"},
        }
        solar_ready = build_colony_strategy_state(
            **common, ready_recipes={"solar_panel"}
        )
        greenhouse_ready = build_colony_strategy_state(
            **common, ready_recipes={"greenhouse"}
        )
        self.assertNotEqual(solar_ready, greenhouse_ready)

    def test_food_days_of_supply_is_visible_before_starvation(self):
        common = {
            "category_scores": {},
            "living_crew": 6,
            "fit_crew": 6,
            "active_site_type": None,
            "ready_recipe_count": 1,
            "surface_requires_plss": True,
        }
        buffered = build_colony_strategy_state(
            **common,
            colony_resources={"food_reserve_kcal": 2_916_000.0},
        )
        critical = build_colony_strategy_state(
            **common,
            colony_resources={"food_reserve_kcal": 32 * 6 * 2700.0},
        )

        self.assertIn("f:buffered", buffered)
        self.assertIn("f:critical", critical)

    def test_critical_food_window_masks_unrelated_objectives(self):
        policy = ColonyStrategicPolicy("planet-a", seed=7)
        candidates = [
            *self.candidates,
            {"recipe": "greenhouse"},
            {"recipe": "habitat_module"},
        ]
        selected = policy.choose(
            state_key=(
                "cap[]|reserve[w:buffered,o:buffered,f:critical,e:buffered]"
                "|crew:6/6"
            ),
            candidates=candidates,
            structures_built={},
            tick=0,
            colony_score=0.0,
        )

        self.assertIn(selected["recipe"], {
            "greenhouse", "solar_panel", "water_collector",
        })
        self.assertNotEqual("habitat_module", selected["recipe"])

    def test_empty_energy_reserve_still_forces_power_recovery(self):
        policy = ColonyStrategicPolicy("planet-a", seed=7)
        selected = policy.choose(
            state_key=(
                "cap[]|reserve[w:buffered,o:buffered,f:buffered,e:empty]"
                "|crew:6/6"
            ),
            candidates=[*self.candidates, {"recipe": "habitat_module"}],
            structures_built={}, tick=0, colony_score=0.0,
        )
        self.assertEqual("solar_panel", selected["recipe"])

    def test_legacy_state_schema_resets_q_values_and_exploration_age(self):
        legacy = {
            "old-state": {"capacity:solar_panel": 99.0},
            META_KEY: {"attempt_count": 9, "total_reward": 123.0},
        }
        policy = ColonyStrategicPolicy("planet-a", q_table=legacy)

        self.assertEqual({}, policy.q_table)
        self.assertEqual(0, policy.attempt_count)
        self.assertEqual(0, policy.exploration_age)
        self.assertEqual(policy.initial_epsilon, policy.epsilon)
        self.assertEqual(
            STATE_SCHEMA_VERSION,
            policy.last_outcome["required_version"],
        )

    def test_episode_outcome_persists_attempt_and_decays_exploration(self):
        policy = ColonyStrategicPolicy("planet-a", seed=1)
        selected = policy.choose(
            state_key="s0",
            candidates=self.candidates,
            structures_built={},
            tick=0,
            colony_score=0.0,
        )
        policy.observe_completion(
            structures_built={selected["recipe"]: 1},
            tick=10,
            colony_score=1.0,
            tick_minutes=10.0,
            next_state_key="s1",
        )
        epsilon_before = policy.epsilon
        policy.finish_episode("timeout")
        payload = policy.dump()
        restored = ColonyStrategicPolicy("planet-a", q_table=payload)

        self.assertEqual(1, payload[META_KEY]["attempt_count"])
        self.assertEqual(1, restored.attempt_count)
        self.assertEqual(1, restored.exploration_age)
        self.assertLess(restored.epsilon, epsilon_before)
        self.assertEqual("timeout", policy.last_outcome["outcome"])

    def test_zero_progress_failure_does_not_decay_exploration(self):
        policy = ColonyStrategicPolicy("planet-a", seed=1)
        policy.choose(
            state_key="s0",
            candidates=self.candidates,
            structures_built={},
            tick=0,
            colony_score=0.0,
        )
        epsilon_before = policy.epsilon

        policy.finish_episode("all_agents_dead", elapsed_days=12.0)

        self.assertEqual(1, policy.attempt_count)
        self.assertEqual(0, policy.exploration_age)
        self.assertEqual(epsilon_before, policy.epsilon)
        self.assertFalse(policy.last_outcome["exploration_aged"])

    def test_observable_stall_penalizes_and_reopens_objective(self):
        policy = ColonyStrategicPolicy(
            "planet-a",
            seed=4,
            initial_epsilon=0.0,
            minimum_epsilon=0.0,
            default_stall_timeout_ticks=2,
            default_replan_interval_ticks=20,
        )
        candidates = [
            {**item, "material_readiness": 0.0, "remaining_work_units": 100}
            for item in self.candidates
        ]
        first = policy.choose(
            state_key="s0",
            candidates=candidates,
            structures_built={},
            tick=0,
            colony_score=0.0,
        )
        second = policy.choose(
            state_key="s0",
            candidates=candidates,
            structures_built={},
            tick=2,
            colony_score=0.0,
        )

        self.assertNotEqual(first["recipe"], second["recipe"])
        self.assertEqual("stalled", second["strategy_replan_reason"])
        self.assertLess(
            policy.q_table["s0"][f"capacity:{first['recipe']}"], 0.0
        )

    def test_infeasible_committed_objective_is_removed_from_action_mask(self):
        policy = ColonyStrategicPolicy(
            "planet-a", seed=5, initial_epsilon=0.0, minimum_epsilon=0.0
        )
        first = policy.choose(
            state_key="s0",
            candidates=self.candidates,
            structures_built={},
            tick=0,
            colony_score=0.0,
        )
        candidates = [
            {**item, "actionable": item["recipe"] != first["recipe"]}
            for item in self.candidates
        ]
        second = policy.choose(
            state_key="s0",
            candidates=candidates,
            structures_built={},
            tick=1,
            colony_score=0.0,
        )

        self.assertNotEqual(first["recipe"], second["recipe"])
        self.assertEqual("infeasible", second["strategy_replan_reason"])

    def test_tactical_schema_resets_legacy_rows_and_observes_horizon(self):
        engine = SimulationEngine(
            os.path.join(ROOT, "config", "planets", "kepler-442b.json"),
            db=False,
        )
        agent = create_team_from_presets(
            os.path.join(ROOT, "config", "agent_presets.json")
        )[0]
        agent.q_table = {"legacy-state": {"explore": 99.0}}
        agent.rl_planet_id = "kepler-442b"
        state = engine.decision_engine.get_rl_state_key(
            agent,
            {},
            {},
            {
                "max_ticks": 1000,
                "ticks_remaining": 100,
                "active_objective": "solar_panel",
            },
        )

        self.assertNotIn("legacy-state", agent.q_table)
        self.assertEqual(
            TACTICAL_STATE_SCHEMA_VERSION,
            agent.q_table[TACTICAL_POLICY_META_KEY]["state_schema_version"],
        )
        self.assertEqual(
            "kepler-442b",
            agent.q_table[TACTICAL_POLICY_META_KEY]["planet_id"],
        )
        self.assertIn("|time:final|objective:solar_panel", state)

    def test_terminal_failure_changes_the_next_greedy_strategy(self):
        policy = ColonyStrategicPolicy(
            "planet-a",
            seed=42,
            initial_epsilon=0.0,
            minimum_epsilon=0.0,
        )
        first = policy.choose(
            state_key="same-landing-state",
            candidates=self.candidates,
            structures_built={},
            tick=0,
            colony_score=0.0,
        )
        failed_recipe = first["recipe"]
        policy.finish_episode("all_agents_dead")

        second = policy.choose(
            state_key="same-landing-state",
            candidates=self.candidates,
            structures_built={},
            tick=0,
            colony_score=0.0,
        )

        self.assertNotEqual(failed_recipe, second["recipe"])
        self.assertLess(
            policy.q_table["same-landing-state"][
                f"capacity:{failed_recipe}"
            ],
            policy.q_table["same-landing-state"][
                f"capacity:{second['recipe']}"
            ],
        )

    def test_longer_survival_receives_less_severe_death_penalty(self):
        early = ColonyStrategicPolicy("planet-a", seed=1)
        late = ColonyStrategicPolicy("planet-a", seed=1)
        for policy in (early, late):
            policy.choose(
                state_key="same-state",
                candidates=self.candidates,
                structures_built={},
                tick=0,
                colony_score=0.0,
            )
        early.finish_episode("all_agents_dead", elapsed_days=12.0)
        late.finish_episode("all_agents_dead", elapsed_days=42.0)
        self.assertGreater(
            late.last_outcome["reward"], early.last_outcome["reward"]
        )

    def test_reserve_band_change_reopens_unbuilt_strategy(self):
        policy = ColonyStrategicPolicy(
            "planet-a", seed=1, initial_epsilon=0.0, minimum_epsilon=0.0
        )
        buffered = "cap[]|reserve[w:buffered,o:buffered,e:buffered]|crew:5/5"
        critical = "cap[]|reserve[w:critical,o:buffered,e:buffered]|crew:5/5"
        first = policy.choose(
            state_key=buffered,
            candidates=self.candidates,
            structures_built={},
            tick=0,
            colony_score=0.0,
        )
        policy.q_table.setdefault(critical, {})[
            "capacity:water_collector"
        ] = 10.0
        second = policy.choose(
            state_key=critical,
            candidates=self.candidates,
            structures_built={},
            tick=100,
            colony_score=0.0,
        )
        self.assertNotEqual(first["strategy_state"], second["strategy_state"])
        self.assertEqual("water_collector", second["recipe"])

    def test_ready_bom_survives_reserve_change_during_site_mobilization(self):
        policy = ColonyStrategicPolicy(
            "planet-a", seed=1, initial_epsilon=0.0, minimum_epsilon=0.0
        )
        buffered = "cap[]|reserve[w:buffered,o:buffered,e:buffered]|crew:6/6"
        critical = "cap[]|reserve[w:critical,o:buffered,e:buffered]|crew:6/6"
        candidates = [
            {"recipe": "solar_panel", "ready": True, "material_readiness": 1.0},
            {"recipe": "water_collector", "ready": True, "material_readiness": 1.0},
        ]
        policy.q_table[buffered] = {
            "capacity:solar_panel": 10.0,
            "capacity:water_collector": 0.0,
        }
        first = policy.choose(
            state_key=buffered, candidates=candidates,
            structures_built={}, tick=0, colony_score=0.0,
        )
        self.assertEqual("solar_panel", first["recipe"])

        policy.q_table[critical] = {
            "capacity:solar_panel": 0.0,
            "capacity:water_collector": 10.0,
        }
        second = policy.choose(
            state_key=critical, candidates=candidates,
            structures_built={}, tick=20, colony_score=0.0,
        )
        self.assertEqual("solar_panel", second["recipe"])
        self.assertTrue(second["strategy_site_mobilization"])

    def test_reserve_band_change_does_not_orphan_active_construction_credit(self):
        policy = ColonyStrategicPolicy(
            "planet-a", seed=1, initial_epsilon=0.0, minimum_epsilon=0.0
        )
        buffered = "cap[]|reserve[w:buffered,o:buffered,e:buffered]|crew:6/6"
        critical = "cap[]|reserve[w:critical,o:buffered,e:buffered]|crew:6/6"
        candidates = [{"recipe": "solar_panel", "material_readiness": 1.0}]
        policy.choose(
            state_key=buffered,
            candidates=candidates,
            structures_built={},
            tick=0,
            colony_score=0.0,
        )

        active_site = {
            "type": "solar_panel", "under_construction": True, "progress": 0.5
        }
        selected = policy.choose(
            state_key=critical,
            candidates=[{**candidates[0], "site": active_site}],
            structures_built={},
            tick=10,
            colony_score=0.0,
        )

        self.assertTrue(selected["strategy_committed"])
        self.assertEqual("solar_panel", policy.commitment.recipe)
        reward = policy.observe_completion(
            structures_built={"solar_panel": 1},
            tick=20,
            colony_score=1.0,
            tick_minutes=10.0,
            next_state_key=critical,
        )
        self.assertGreater(reward, 0.0)
        self.assertEqual(1, policy.episode_verified_completions)

    def test_manual_inspection_stop_does_not_become_training_data(self):
        policy = ColonyStrategicPolicy("planet-a", seed=1)
        policy.choose(
            state_key="s0",
            candidates=self.candidates,
            structures_built={},
            tick=0,
            colony_score=0.0,
        )
        before = policy.dump()
        self.assertEqual(0, policy.finish_episode("manual_stop"))
        after = policy.dump()
        self.assertEqual(before[META_KEY]["attempt_count"], after[META_KEY]["attempt_count"])
        self.assertEqual(before["s0"], after["s0"])

    def test_terminal_persistence_retry_does_not_apply_reward_twice(self):
        class FlakyDB:
            def __init__(self):
                self.calls = 0
                self.saved = None

            def save_agent_q_tables(self, policies, tick, flush=False):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("simulated transaction rollback")
                self.saved = list(policies)
                return len(self.saved)

        engine = SimulationEngine(
            os.path.join(ROOT, "config", "planets", "kepler-442b.json"),
            db=False,
        )
        engine.strategic_policy.choose(
            state_key="s0",
            candidates=[{"recipe": "solar_panel"}],
            structures_built={},
            tick=0,
            colony_score=0.0,
        )
        engine._db = FlakyDB()
        engine._strategic_policy_loaded = True
        engine.end_reason = "timeout"

        engine._finalize_terminal_learning()
        first_q = engine.strategic_policy.q_table["s0"]["capacity:solar_panel"]
        self.assertEqual(1, engine.strategic_policy.attempt_count)
        self.assertTrue(engine._terminal_reward_applied)
        self.assertFalse(engine._terminal_learning_finalized)

        engine._finalize_terminal_learning()
        self.assertEqual(1, engine.strategic_policy.attempt_count)
        self.assertEqual(
            first_q,
            engine.strategic_policy.q_table["s0"]["capacity:solar_panel"],
        )
        self.assertTrue(engine._terminal_learning_finalized)
        self.assertEqual(2, engine._db.calls)
        shared = next(
            item for item in engine._db.saved
            if item["agent_id"] == STRATEGY_POLICY_ID
        )
        self.assertEqual(1, shared["q_table"][META_KEY]["attempt_count"])

    def test_failed_shared_policy_load_is_never_overwritten(self):
        class LoadFailureDB:
            def __init__(self):
                self.saved = None

            def load_agent_q_table(self, *_args, **_kwargs):
                raise RuntimeError("simulated read failure")

            def save_agent_q_tables(self, policies, tick, flush=False):
                self.saved = list(policies)
                return len(self.saved)

        engine = SimulationEngine(
            os.path.join(ROOT, "config", "planets", "kepler-442b.json"),
            db=False,
        )
        engine._db = LoadFailureDB()
        engine._strategic_policy_loaded = False
        engine._init_agents()
        self.assertFalse(engine._strategic_policy_loaded)

        engine.end_reason = "timeout"
        engine._finalize_terminal_learning()
        self.assertTrue(engine._terminal_learning_finalized)
        self.assertFalse(any(
            item["agent_id"] == STRATEGY_POLICY_ID
            for item in engine._db.saved
        ))


if __name__ == "__main__":
    unittest.main()
