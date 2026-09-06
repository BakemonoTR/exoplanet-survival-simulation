"""Isolation and reward regressions for the opt-in strategy experiment."""
import copy
import unittest
import tempfile
from pathlib import Path

from src.agents.strategic_rl import ColonyStrategicPolicy, build_colony_strategy_state
from src.orchestration.engine import SimulationEngine
from src.memory import vector_store
from src.api.database import SimulationDB

vector_store._use_tfidf_fallback = True
ROOT = Path(__file__).parent
CATEGORIES = ('energy', 'o2', 'water', 'food', 'shelter', 'hazard_protection')


class DeadlineLearningTest(unittest.TestCase):
    def test_legacy_policy_is_default_and_experiment_persists_separately(self):
        old = ColonyStrategicPolicy('planet')
        new = ColonyStrategicPolicy('planet', deadline_learning=True)
        new.q_table = {'state': {'capacity:solar_panel': 3.0}}
        new.attempt_count = 3
        restored = ColonyStrategicPolicy('planet', deadline_learning=True, q_table=new.dump())
        self.assertEqual(new.dump(), restored.dump())
        self.assertNotEqual(old.persistence_id, new.persistence_id)
        old.load(new.dump())
        self.assertEqual({}, old.q_table)

    def test_deadline_and_observed_constraint_distinguish_identical_inventory(self):
        common = dict(category_scores={}, colony_resources={}, living_crew=6,
                      fit_crew=6, active_site_type=None, ready_recipe_count=0,
                      surface_requires_plss=True)
        legacy = build_colony_strategy_state(**common)
        early = build_colony_strategy_state(**common, days_until_deadline=290, work_constraint='supply')
        late = build_colony_strategy_state(**common, days_until_deadline=20, work_constraint='supply')
        busy = build_colony_strategy_state(**common, days_until_deadline=20, work_constraint='fabrication')
        self.assertNotIn('deadline:', legacy)
        self.assertTrue(early.startswith(legacy))
        self.assertEqual(3, len({early, late, busy}))

    def test_experiment_reuses_state_across_exact_action_mask_changes(self):
        policy = ColonyStrategicPolicy('planet', deadline_learning=True)
        common = dict(category_scores={'energy': .5}, colony_resources={},
                      living_crew=6, fit_crew=6, active_site_type=None,
                      ready_recipe_count=1, surface_requires_plss=True,
                      days_until_deadline=100, work_constraint='assembly')
        first = build_colony_strategy_state(**common,
                    eligible_recipes={'solar_panel', 'greenhouse'},
                    ready_recipes={'solar_panel'})
        second = build_colony_strategy_state(**{**common, 'active_site_type': 'solar_panel'},
                    eligible_recipes={'habitat_module', 'greenhouse'},
                    ready_recipes={'greenhouse'})
        self.assertNotEqual(first, second)
        self.assertEqual(policy._learning_state_key(first), policy._learning_state_key(second))

    def test_terminal_credit_orders_partial_success_without_acceptance_shortcut(self):
        def reward(level, **kwargs):
            policy = ColonyStrategicPolicy('planet', deadline_learning=True)
            policy.finish_episode('timeout', category_scores={k: level for k in CATEGORIES}, **kwargs)
            return policy.last_outcome['reward']
        self.assertGreater(reward(.98), reward(.60))
        self.assertGreater(reward(.98, deadline_readiness=.98), reward(.98, deadline_readiness=.60))
        self.assertGreater(reward(.98, support_soak_fraction=.5), reward(.98))
        self.assertLess(reward(.98, surviving_crew_fraction=.5), reward(.98))
        self.assertLessEqual(reward(2, deadline_readiness=2, support_soak_fraction=2), 0)
        old = ColonyStrategicPolicy('planet')
        old.finish_episode('timeout', category_scores={k: .98 for k in CATEGORIES})
        self.assertEqual(-35, old.last_outcome['reward'])

    def test_frozen_evaluation_does_not_update_q_values(self):
        policy = ColonyStrategicPolicy('planet', deadline_learning=True, evaluation_mode=True)
        policy.q_table = {'state': {'capacity:solar_panel': 8.0}}
        before = copy.deepcopy(policy.q_table)
        selected = policy.choose(state_key='unknown', candidates=[{
            'recipe': 'solar_panel', 'required_count': 26,
        }], structures_built={}, tick=0, colony_score=0)
        self.assertEqual('solar_panel', selected['recipe'])
        self.assertEqual(before, policy.q_table)
        policy._td_update('state', 'capacity:solar_panel', -40, 'next')
        policy.episode_trace = [('state', 'capacity:solar_panel')]
        policy.finish_episode('timeout')
        self.assertEqual(before, policy.q_table)
        self.assertEqual(0, policy.epsilon)
        self.assertEqual(0, policy.attempt_count)

    def test_experimental_engine_cannot_overwrite_legacy_strategy(self):
        with tempfile.TemporaryDirectory() as directory:
            db = SimulationDB(str(Path(directory) / 'trial.db'))
            old = ColonyStrategicPolicy('kepler-442b')
            old.q_table = {'legacy': {'capacity:solar_panel': 5.0}}
            db.start_run('kepler-442b', seed=42, max_ticks=51840)
            db.save_agent_q_table(old.persistence_id, old.dump(), 1, 0)
            engine = SimulationEngine(str(ROOT / 'config/planets/kepler-442b.json'),
                                      db=db, strategic_deadline_learning=True)
            engine._strategic_policy_loaded = True
            engine.end_reason = 'timeout'
            engine._finalize_terminal_learning()
            self.assertEqual(old.dump(), db.load_agent_q_table(old.persistence_id, planet_id='kepler-442b'))
            new = db.load_agent_q_table(engine.strategic_policy.persistence_id, planet_id='kepler-442b')
            restored = ColonyStrategicPolicy('kepler-442b', deadline_learning=True, q_table=new)
            self.assertEqual(1, restored.attempt_count)
            db.close()

    def test_engine_observation_uses_mission_clock(self):
        engine = SimulationEngine(str(ROOT / 'config/planets/kepler-442b.json'),
                                  db=False, strategic_deadline_learning=True)
        engine.decision_engine.current_tick = 320 * 144
        state = engine.decision_engine.get_colony_strategy_state({})
        self.assertIn('|deadline:final|constraint:', state)
        engine.end_reason = 'timeout'
        engine.current_tick = 51840
        engine._finalize_terminal_learning()
        self.assertEqual(1, engine.strategic_policy.attempt_count)


if __name__ == '__main__':
    unittest.main()
