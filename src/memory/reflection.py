"""
Reflection System — Memory Synthesis.

Every 20 ticks (~100 sim minutes), agents reflect on recent experiences:
1. Pull last 10 memories from vector store
2. Send to LLM for insight extraction
3. Store insights back with high importance (8.0)
4. Update social trust graph based on interactions

Inspired by: Park et al., "Generative Agents" (2023)
"""

import logging
from typing import Optional

from src.memory.vector_store import MemoryManager
from src.orchestration.llm_client import LocalNarrativeClient, LLMCallType
from src.agents.prompts import build_reflection_prompt

logger = logging.getLogger(__name__)


class SocialGraph:
    """
    Trust scores between agent pairs.
    
    Trust range: -1.0 (enemy) to +1.0 (trusted ally)
    Default: 0.0 (neutral stranger)
    
    Updated by:
    - Reflection insights (LLM evaluates interactions)
    - Direct actions (sharing resources = +, refusing = -)
    """
    
    def __init__(self):
        # {agent_id: {other_agent_id: trust_score}}
        self._trust: dict[str, dict[str, float]] = {}
    
    def get_trust(self, agent_id: str, other_id: str) -> float:
        """Get trust from agent toward another."""
        return self._trust.get(agent_id, {}).get(other_id, 0.0)
    
    def update_trust(self, agent_id: str, other_id: str, delta: float):
        """
        Modify trust score.
        
        Args:
            agent_id: Agent whose trust changes
            other_id: Agent being evaluated
            delta: Change amount (-0.2 to +0.2 typical)
        """
        if agent_id not in self._trust:
            self._trust[agent_id] = {}
        
        current = self._trust[agent_id].get(other_id, 0.0)
        new_trust = max(-1.0, min(1.0, current + delta))
        self._trust[agent_id][other_id] = round(new_trust, 3)
    
    def set_trust(self, agent_id: str, other_id: str, value: float):
        """Set absolute trust value."""
        if agent_id not in self._trust:
            self._trust[agent_id] = {}
        self._trust[agent_id][other_id] = max(-1.0, min(1.0, value))
    
    def get_all_trust(self, agent_id: str) -> dict[str, float]:
        """Get all trust scores for an agent."""
        return dict(self._trust.get(agent_id, {}))
    
    def get_team_summary(self, agent_id: str, agent_names: dict) -> str:
        """Get compact trust summary for LLM context."""
        trusts = self._trust.get(agent_id, {})
        if not trusts:
            return "No established relationships."
        
        parts = []
        for other_id, score in trusts.items():
            name = agent_names.get(other_id, other_id)
            if score > 0.5:
                parts.append(f"{name}(ally)")
            elif score > 0:
                parts.append(f"{name}(+)")
            elif score < -0.5:
                parts.append(f"{name}(hostile)")
            elif score < 0:
                parts.append(f"{name}(-)")
            else:
                parts.append(f"{name}(neutral)")
        
        return ", ".join(parts)
    
    def to_dict(self) -> dict:
        return dict(self._trust)


class ReflectionSystem:
    """
    Manages periodic agent reflection cycles.
    
    Reflection process:
    1. Get recent memories (last 10-15)
    2. Build reflection prompt
    3. Call LLM for insight extraction
    4. Parse insights + trust updates
    5. Store insights in memory (high importance)
    6. Update social graph
    """
    
    def __init__(self, memory_manager: MemoryManager,
                 llm_client: LocalNarrativeClient,
                 social_graph: SocialGraph = None):
        self.memory = memory_manager
        self.llm = llm_client
        self.social = social_graph or SocialGraph()
        
        # Track reflection history
        self._reflection_count: dict[str, int] = {}
    
    def reflect(self, agent_id: str, agent_name: str,
                current_tick: int) -> list[str]:
        """
        Run one reflection cycle for an agent.
        
        Args:
            agent_id: Agent's unique ID
            agent_name: Agent's display name
            current_tick: Current simulation tick
            
        Returns:
            List of generated insights (strings)
        """
        store = self.memory.get_store(agent_id)
        
        # Get recent memories (last 10)
        recent = store.get_recent(n=10)
        if len(recent) < 3:
            # Not enough memories to reflect on
            return []
        
        recent_texts = [m.text for m in recent]
        
        # Build prompt
        prompt = build_reflection_prompt(agent_name, recent_texts, current_tick)
        
        # Call LLM
        result = self.llm.call(
            system_prompt=f"You are {agent_name}, reflecting on recent experiences.",
            user_prompt=prompt,
            call_type=LLMCallType.REFLECTION,
            current_tick=current_tick,
            temperature=0.8,  # Slightly creative for insights
            max_tokens=250,
        )
        
        insights = []
        
        if result is None:
            # API exhausted — generate rule-based reflection
            insights = self._fallback_reflect(recent)
        else:
            # Parse LLM response
            insights = result.get("insights", [])
            
            # Process trust updates
            trust_updates = result.get("trust_updates", {})
            for other_name, delta in trust_updates.items():
                if isinstance(delta, (int, float)):
                    # Find agent ID by name (simplified — use name as key)
                    self.social.update_trust(agent_id, other_name, delta)
        
        # Store insights in memory
        for insight in insights:
            if isinstance(insight, str) and insight.strip():
                self.memory.add_insight(agent_id, insight, current_tick)
        
        # Track
        self._reflection_count[agent_id] = self._reflection_count.get(agent_id, 0) + 1
        
        logger.debug(
            f"Reflection for {agent_name}: {len(insights)} insights generated"
        )
        
        return insights
    
    def _fallback_reflect(self, recent_memories) -> list[str]:
        """Generate simple rule-based insights when LLM unavailable."""
        insights = []
        
        # Check for patterns in recent memories
        texts = [m.text.lower() for m in recent_memories]
        all_text = " ".join(texts)
        
        # Resource scarcity pattern
        if any("water" in t and ("low" in t or "critical" in t) for t in texts):
            insights.append("Water supply is a recurring concern. Should prioritize water infrastructure.")
        
        # Construction progress
        if any("built" in t or "constructed" in t for t in texts):
            insights.append("Making progress on colony construction. Need to maintain momentum.")
        
        # Health issues
        if any("injur" in t or "disease" in t or "radiation" in t for t in texts):
            insights.append("Health risks are mounting. Medical supplies and shelter should be prioritized.")
        
        # Social observations
        social_memories = [m for m in recent_memories if m.memory_type == "social"]
        if social_memories:
            insights.append("Team interactions are important for morale and efficiency.")
        
        # Generic insight if nothing specific
        if not insights:
            insights.append("Need to stay focused on survival priorities and colony readiness goals.")
        
        return insights
    
    def get_insights_for_agent(self, agent_id: str) -> list[str]:
        """Get all stored insights for an agent."""
        store = self.memory.get_store(agent_id)
        insight_memories = store.get_insights()
        return [m.text for m in insight_memories]
    
    def get_stats(self) -> dict:
        """Get reflection system statistics."""
        return {
            "reflections_per_agent": dict(self._reflection_count),
            "total_reflections": sum(self._reflection_count.values()),
            "social_graph_size": len(self.social._trust),
        }
