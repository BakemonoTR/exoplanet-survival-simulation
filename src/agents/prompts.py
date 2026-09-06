"""
Prompt Templates for Agent LLM Calls.

Optimized for minimal token usage (~450 tokens total per call).
All prompts produce structured JSON output for reliable parsing.

Prompt types:
1. System: Character identity (genome, skills, personality) — cached once
2. Strategic: Goal selection from colony sub-targets — every 10 ticks
3. Tactical: Immediate action selection — on-demand
4. Reflection: Memory synthesis — every 20 ticks
"""

from typing import Optional


# ============================================================
# SYSTEM PROMPT (per-agent, cached for entire simulation)
# ============================================================

def build_system_prompt(agent) -> str:
    """
    Build character-defining system prompt. ~150 tokens.
    
    This is sent with EVERY call but cached by the LLM client.
    Defines who the agent IS — genome, skills, personality.
    """
    g = agent.genome
    c = agent.competency
    primary = c.get_primary_domain()
    
    # Map primary domain to role description
    role_map = {
        "engineering": "structural engineer and fabricator",
        "medical": "field medic and life support specialist",
        "physics": "planetary physicist and radiation analyst",
        "botany_bio": "astrobiologist and resource specialist",
        "leadership_social": "mission coordinator and team leader",
    }
    role = role_map.get(primary, "generalist colonist")
    
    # Build compact skill summary
    skills = []
    for domain in ["engineering", "medical", "physics", "botany_bio", "leadership_social"]:
        score = getattr(c, domain)
        if score >= 7:
            skills.append(f"{domain}:{score}(expert)")
        elif score >= 4:
            skills.append(f"{domain}:{score}")
    
    # Personality traits
    traits = ", ".join(agent.personality_traits[:3]) if hasattr(agent, 'personality_traits') else "analytical"
    
    return (
        f"You are {agent.name}, a {role} in a 6-person exoplanet colonization team. "
        f"Traits: {traits}. "
        f"Physical: STR:{g.strength} AGI:{g.agility} END:{g.endurance} PER:{g.perception} IMM:{g.immunity}. "
        f"Skills: {', '.join(skills)}. "
        f"You must survive and prepare for 100 arriving civilians, with infrastructure sized for all 106 surface occupants. "
        f"Respond ONLY with valid JSON: "
        f'{{\"action\": \"<type>\", \"target\": {{...}}, \"reasoning\": \"<brief>\"}}'
    )


# ============================================================
# STRATEGIC PROMPT (goal selection, every 10 ticks)
# ============================================================

def build_strategic_prompt(agent, colony_score: dict,
                           world_summary: str = "",
                           recent_insights: list = None,
                           team_status: str = "") -> str:
    """
    Build strategic decision prompt. ~200 tokens.
    
    Agent selects next high-level goal based on colony needs.
    """
    if recent_insights is None:
        recent_insights = []
    
    # Colony score breakdown
    score_parts = []
    for category, pct in colony_score.items():
        if isinstance(pct, (int, float)):
            marker = "!" if pct < 30 else ("*" if pct < 60 else "")
            score_parts.append(f"{category}:{pct:.0f}%{marker}")
    score_str = ", ".join(score_parts) if score_parts else "no data"
    
    # Agent state summary
    state = agent.to_state_summary() if hasattr(agent, 'to_state_summary') else "stable"
    
    # Available actions based on inventory + skills
    prompt = (
        f"COLONY STATUS: {score_str}\n"
        f"YOUR STATE: {state}\n"
    )
    
    if team_status:
        prompt += f"TEAM: {team_status}\n"
    
    if recent_insights:
        prompt += f"INSIGHTS: {'; '.join(recent_insights[:3])}\n"
    
    if world_summary:
        prompt += f"ENVIRONMENT: {world_summary}\n"
    
    recipes_str = "water_collector(pumps,thermal loop,pipes,seals,structure), water_purifier(pumps,pressure vessel,pipes,seals,controls), solar_panel(PV laminates,truss,drive,power conditioner), isru_o2_unit(electrolysis stack,pipes,controls), greenhouse(panels,glass,hydroponics), habitat_module(pressure shell,ECLSS interfaces), storage_crate(iron,basalt)"
        
    primary = agent.competency.get_primary_domain() if hasattr(agent, 'competency') else 'engineering'
    duty_map = {
        "engineering": "ROLE DUTY: You are the team Fabricator/Builder. Primary focus: CRAFT & CONSTRUCT Colony Score infrastructure (water_collector, water_purifier, solar_panel, isru_o2_unit, greenhouse, habitat_module). The collector conditions newly extracted water; water_purifier is the separately rated wastewater-reclamation train. When materials exist in team stock, refine and build!",
        "physics": "ROLE DUTY: You are the Miner/Prospector. Primary focus: survey and extract only verified industrial feedstocks (iron_ore, basalt, silica_sand, graphite and explicitly requested chalcopyrite_ore) needed by a live bill of materials.",
        "botany_bio": "ROLE DUTY: You are the Astrobiologist. Primary focus: GATHER water ice and characterize mineral feedstocks; crop nutrients and polymers remain finite qualified cargo.",
        "medical": "ROLE DUTY: You are the Field Medic. Primary focus: Health, hygiene, safety, and surface gathering (regolith, water_ice).",
        "leadership_social": "ROLE DUTY: You are Mission Coordinator. Primary focus: Coordinate team, balance colony metrics, supply missing materials.",
    }
    role_duty = duty_map.get(primary, "ROLE DUTY: Support colony survival and structure building.")

    prompt += (
        f"\nYOUR SPECIALIZATION:\n{role_duty}\n"
        f"\nPLANETARY GEOLOGY:\n"
        f"Sub-surface ores (basalt, iron, silicates, olivine, carbon) are buried under surface regolith. Gathering regolith on a tile excavates the ground and unearths underlying mineral veins.\n"
        f"\nCONSTRUCTION RECIPES:\n"
        f"{recipes_str}\n"
        f"\nChoose next strategic goal aligned with your role duty. "
        f"Output JSON: "
        f'{{"goal": "<description>", "action": "<type>", '
        f'"target": {{...}}, "plan_steps": ["step1", "step2"], '
        f'"estimated_ticks": <N>, "reasoning": "<why>"}}'
    )
    
    return prompt


# ============================================================
# TACTICAL PROMPT (immediate action, on-demand)
# ============================================================

def build_tactical_prompt(agent, trigger: str = "need_critical",
                          world_context: dict = None,
                          nearby_agents: list = None,
                          available_actions: list = None) -> str:
    """
    Build tactical decision prompt. ~200 tokens.
    
    Agent selects immediate action in response to a trigger event.
    Only called when something changes — not every tick.
    
    Triggers:
    - need_critical: A physiological need dropped below threshold
    - threat_detected: Environmental hazard (flare, quake, cold)
    - plan_impossible: Current plan step can't execute
    - agent_encounter: New agent entered interaction range
    - task_complete: Previous task finished, need new one
    """
    if world_context is None:
        world_context = {}
    if nearby_agents is None:
        nearby_agents = []
    if available_actions is None:
        available_actions = [
            "move", "gather", "build", "eat", "drink", "sleep",
            "wash", "talk", "trade", "flee", "explore", "treat_injury"
        ]
    
    # Agent state
    state = agent.needs.to_compact_string()
    
    prompt = f"TRIGGER: {trigger}\n"
    prompt += f"STATE: {state}\n"
    
    # Injury/disease status
    if agent.injury_level > 0.1:
        prompt += f"INJURY: {agent.injury_level:.0%}\n"
    if agent.active_diseases:
        prompt += f"DISEASES: {', '.join(d.disease_type.value for d in agent.active_diseases)}\n"
    
    # EVA status
    if agent.suit_equipped:
        prompt += f"EVA: suit {agent.suit_integrity:.0%}, O2 {agent._current_canister_remaining:.0f}/100\n"
    
    # Environment
    if world_context:
        env_parts = []
        if "temperature_c" in world_context:
            env_parts.append(f"temp:{world_context['temperature_c']}C")
        if "biome" in world_context:
            env_parts.append(world_context["biome"])
        if world_context.get("active_event"):
            env_parts.append(f"EVENT:{world_context['active_event']}")
        if env_parts:
            prompt += f"ENV: {', '.join(env_parts)}\n"
    
    # Nearby agents
    if nearby_agents:
        agents_str = "; ".join(
            f"{a.get('name', '?')}({a.get('distance', '?')}m)"
            for a in nearby_agents[:3]
        )
        prompt += f"NEARBY: {agents_str}\n"
    
    # Inventory summary
    inv = agent.inventory.to_compact_string() if hasattr(agent.inventory, 'to_compact_string') else ""
    if inv:
        prompt += f"INVENTORY: {inv}\n"
    
    prompt += (
        f"\nChoose immediate action from: {', '.join(available_actions)}. "
        f"Output JSON: "
        f'{{\"action\": \"<type>\", \"target\": {{...}}, '
        f'\"reasoning\": \"<brief>\", \"priority\": \"critical|high|normal\"}}'
    )
    
    return prompt


# ============================================================
# REFLECTION PROMPT (memory synthesis, every 20 ticks)
# ============================================================

def build_reflection_prompt(agent_name: str,
                           recent_memories: list[str],
                           current_tick: int) -> str:
    """
    Build reflection prompt. ~150 tokens.
    
    Agent synthesizes recent experiences into high-level insights.
    These insights are stored back in memory with high importance.
    """
    memories_str = "\n".join(f"- {m}" for m in recent_memories[:10])
    
    return (
        f"You are {agent_name}. Review your recent experiences "
        f"(tick {current_tick}):\n"
        f"{memories_str}\n\n"
        f"Extract 2-3 high-level insights about survival strategy, "
        f"resource availability, or environmental patterns. "
        f"Respond ONLY with valid JSON (no + signs before numbers):\n"
        f'{{"insights": ["insight1", "insight2"], '
        f'"trust_updates": {{"agent_name": 0.1}}, '
        f'"mood": "hopeful"}}'
    )


# ============================================================
# SOCIAL PROMPT (agent dialogue)
# ============================================================

def build_social_prompt(agent, other_agent_name: str,
                       context: str = "general",
                       trust_level: float = 0.0) -> str:
    """
    Build social interaction prompt. ~100 tokens.
    
    Generates agent dialogue for social interactions.
    Trust level affects willingness to share/cooperate.
    """
    trust_desc = (
        "trusted ally" if trust_level > 0.5 else
        "acquaintance" if trust_level > 0 else
        "distrusted" if trust_level > -0.5 else
        "antagonist"
    )
    
    return (
        f"You encounter {other_agent_name} ({trust_desc}). "
        f"Context: {context}. "
        f"Respond with a brief in-character statement and action. "
        f"Output JSON: "
        f'{{\"dialogue\": \"<what you say>\", '
        f'\"social_action\": \"cooperate|share|request|refuse|ignore\", '
        f'\"share_item\": null, \"reasoning\": \"<brief>\"}}'
    )


# ============================================================
# TOKEN ESTIMATION
# ============================================================

def estimate_tokens(text: str) -> int:
    """
    Rough token count estimation.
    English: ~4 chars/token. JSON overhead: ~1.3x.
    """
    return int(len(text) / 3.5)


def get_prompt_budget() -> dict:
    """Get token budget breakdown per call type."""
    return {
        "system_prompt": 150,     # Cached, sent with every call
        "strategic_user": 200,    # Colony status + state
        "tactical_user": 200,     # Trigger + state + env
        "reflection_user": 150,   # Recent memories
        "response_avg": 150,      # JSON action output
        "total_per_call": 500,    # System + user + response
        "calls_per_sim_day": 30,  # ~6 per agent per sim day
        "tokens_per_sim_day": 15000,  # Per agent
        "tokens_per_sim_day_5_agents": 75000,  # Total
    }
