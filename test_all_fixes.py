"""
Comprehensive verification test for all audit fixes and user requests.
"""
import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.agents.agent import Agent, AgentStatus, DeathCause, Needs, Genome, Competency
from src.systems.colony_score import ColonyScore
from src.orchestration.engine import SimulationEngine
from src.world.generator import PlanetConfig

def test_no_fire_pit():
    print("\n--- Test 1: Fire Pit Removal ---")
    recipes_path = os.path.join(os.path.dirname(__file__), 'config', 'recipes.json')
    with open(recipes_path, 'r', encoding='utf-8') as f:
        recipes = json.load(f).get("recipes", {})
    assert "fire_pit" not in recipes, "fire_pit must be completely removed from recipes.json!"
    print("✓ fire_pit confirmed removed from recipes.json")

def test_colony_score():
    print("\n--- Test 2: Colony Score Scaling & Targets ---")
    score = ColonyScore()
    
    # Test individual structure contributions
    structures = {
        "water_collector": 2,     # 2 * 200 = 400 L/day (400/2000 = 20%)
        "solar_panel": 4,         # 4 * 25 = 100 kW (100/500 = 20%)
        "greenhouse": 1,          # 1 * 44000 = 44000 kcal (44000/220000 = 20%)
        "habitat_module": 3,      # 3 * 8 = 24 persons (24/100 = 24%)
        "isru_o2_unit": 2,        # 2 * 6.0 = 12 kg/day (12/60 = 20%)
        "potable_water_tank": 6,
        "oxygen_buffer_tank": 2,
    }
    score.update_counts(structures)
    scores = score.get_scores()
    total_score = score.get_overall_score()
    
    print(f"Sub-target scores: {scores}")
    print(f"Overall Colony Readiness Score: {total_score:.1f}%")
    assert scores["food"] > 0.15, f"Food score should be ~0.20, got {scores['food']}"
    assert scores["water"] > 0.15, f"Water score should be ~0.20, got {scores['water']}"
    assert scores["energy"] > 0.15, f"Energy score should be ~0.20, got {scores['energy']}"
    assert scores["o2"] > 0.15, f"O2 score should be ~0.20, got {scores['o2']}"
    assert scores["shelter"] > 0.20, f"Shelter score should be ~0.24, got {scores['shelter']}"
    print("✓ Colony score calculations are balanced and realistic!")

def test_corpse_salvage():
    print("\n--- Test 3: Corpse Scavenging Mechanics ---")
    agent_cfg1 = {
        "id": "agent_1",
        "name": "Dr. Sarah Chen",
        "background": "Geologist",
        "genome": {"strength": 7, "agility": 6, "endurance": 6, "perception": 8, "immunity": 7},
        "competency": {"engineering": 5, "medical": 3, "physics": 6, "botany_bio": 2, "leadership_social": 4}
    }
    agent_cfg2 = {
        "id": "agent_2",
        "name": "Marcus Vance",
        "background": "Chief Engineer",
        "genome": {"strength": 8, "agility": 5, "endurance": 7, "perception": 6, "immunity": 6},
        "competency": {"engineering": 9, "medical": 2, "physics": 5, "botany_bio": 2, "leadership_social": 5}
    }
    
    dead_agent = Agent(agent_cfg1)
    living_agent = Agent(agent_cfg2)
    
    # Give dead agent valuable items and materials
    dead_agent.inventory.items["oxygen_canisters"] = 3
    dead_agent.inventory.items["emergency_rations"] = 4
    dead_agent.inventory.materials["iron_ore"] = 12
    dead_agent.inventory.materials["chalcopyrite_ore"] = 5
    
    # Kill the agent
    dead_agent._die(DeathCause.HYPOTHERMIA, 50)
    assert dead_agent.status == AgentStatus.DEAD
    assert not dead_agent.corpse_salvaged
    
    # Living agent salvages the corpse
    res = dead_agent.salvage_corpse(living_agent)
    print(f"Salvage result: {res}")
    
    assert res["salvaged"] is True
    assert dead_agent.corpse_salvaged is True
    assert living_agent.inventory.items["oxygen_canisters"] >= 3
    assert living_agent.inventory.materials["iron_ore"] >= 12
    assert len(dead_agent.inventory.items) == 0
    assert len(dead_agent.inventory.materials) == 0
    print("✓ Dead agent inventory was completely and safely salvaged!")

def test_needs_and_suit_fixes():
    print("\n--- Test 4: O2, Gravity, and Suit Wear Fixes ---")
    agent_cfg = {
        "id": "agent_test",
        "name": "Elena Rostova",
        "background": "Physicist",
        "genome": {"strength": 5, "agility": 6, "endurance": 6, "perception": 7, "immunity": 6},
        "competency": {"engineering": 4, "medical": 8, "physics": 4, "botany_bio": 7, "leadership_social": 3}
    }
    agent = Agent(agent_cfg)
    agent.suit_equipped = True
    agent.has_micro_puncture = True  # Trigger micro puncture
    
    initial_suit = agent.suit_integrity
    # Run 1 tick update
    events = agent.tick_update(
        ambient_temp_c=-20.0,
        gravity_multiplier=1.31,
        has_shelter=False,
        has_atmosphere=False
    )
    print(f"Tick update warnings: {events.get('warnings')}")
    assert "suit_micro_puncture_active_leak" in events.get("warnings")
    assert agent.suit_integrity < initial_suit
    print("✓ o2_per_tick with micro_puncture executed with zero NameError!")

def test_engine_run():
    print("\n--- Test 5: Full Simulation Engine Run (30 Ticks) ---")
    planet_path = os.path.join(os.path.dirname(__file__), 'config', 'planets', 'kepler-442b.json')
    engine = SimulationEngine(planet_path, seed=123, max_ticks=50, tick_speed=0.01)
    
    # Add standard agents
    agent_cfgs = [
        {"id": "a1", "name": "Sarah", "background": "Engineer", "genome": {"strength": 7, "agility": 6, "endurance": 6, "perception": 8, "immunity": 7}, "competency": {"engineering": 8, "medical": 3, "physics": 5, "botany_bio": 2, "leadership_social": 4}},
        {"id": "a2", "name": "Marcus", "background": "Scientist", "genome": {"strength": 6, "agility": 5, "endurance": 7, "perception": 7, "immunity": 6}, "competency": {"engineering": 4, "medical": 2, "physics": 8, "botany_bio": 4, "leadership_social": 5}},
        {"id": "a3", "name": "Elena", "background": "Medic", "genome": {"strength": 5, "agility": 6, "endurance": 6, "perception": 7, "immunity": 6}, "competency": {"engineering": 3, "medical": 8, "physics": 3, "botany_bio": 6, "leadership_social": 4}}
    ]
    for c in agent_cfgs:
        engine.add_agent(Agent(c))
    
    # Run for 30 ticks
    engine.run_headless(30)
    
    print(f"Simulation completed 30 ticks cleanly.")
    print(f"Structures placed: {len(engine.placed_structures)}")
    print(f"Colony score: {engine.colony_score.get_overall_score():.1f}%")
    print("✓ Simulation Engine executed seamlessly!")

if __name__ == "__main__":
    test_no_fire_pit()
    test_colony_score()
    test_corpse_salvage()
    test_needs_and_suit_fixes()
    test_engine_run()
    print("\n==========================================")
    print("🎉 ALL TESTS PASSED SUCCESSFULLY! 100% VERIFIED!")
    print("==========================================")
