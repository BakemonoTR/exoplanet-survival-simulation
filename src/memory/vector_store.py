"""
Lightweight Vector Memory Store for Agents.

Each agent has its own memory collection. Memories are stored as:
- text: Natural language description of event
- embedding: Sentence vector (via sentence-transformers or TF-IDF fallback)
- metadata: tick, importance, type, related agents

Retrieval uses weighted scoring:
  score = 0.4×similarity + 0.3×recency + 0.3×importance

This implementation uses numpy for cosine similarity, avoiding
heavy ChromaDB dependency. Can be upgraded to ChromaDB later.
"""

import numpy as np
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# EMBEDDING MODEL (lazy-loaded)
# ============================================================

_embedding_model = None
_embedding_dim = 384  # all-MiniLM-L6-v2 output dimension
_use_tfidf_fallback = False


def get_embedding_model():
    """Lazy-load sentence-transformers model."""
    global _embedding_model, _use_tfidf_fallback
    
    if _use_tfidf_fallback:
        return None
        
    if _embedding_model is not None:
        return _embedding_model
    
    try:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Loaded sentence-transformers: all-MiniLM-L6-v2")
        return _embedding_model
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "Using TF-IDF fallback (lower quality). "
            "Install with: pip install sentence-transformers"
        )
        _use_tfidf_fallback = True
        return None


def embed_text(text: str) -> np.ndarray:
    """
    Convert text to embedding vector.
    
    Uses sentence-transformers if available, otherwise TF-IDF hash fallback.
    """
    try:
        model = get_embedding_model()
        if model is not None:
            embedding = model.encode(text, show_progress_bar=False)
            return np.array(embedding, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Embedding model execution failed, using hash fallback: {e}")
    
    # TF-IDF hash fallback: deterministic, fast, lower quality
    return _tfidf_hash_embed(text)


def _tfidf_hash_embed(text: str, dim: int = 384) -> np.ndarray:
    """
    Simple hash-based embedding fallback.
    
    Uses character n-gram hashing to produce a fixed-size vector.
    Not as good as neural embeddings but works without dependencies.
    """
    vec = np.zeros(dim, dtype=np.float32)
    words = text.lower().split()
    
    for i, word in enumerate(words):
        # Hash each word to a position
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        idx = h % dim
        # Use word position for sign
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign * (1.0 / (1 + i * 0.1))  # Position decay
        
        # Bigrams for context
        if i > 0:
            bigram = f"{words[i-1]}_{word}"
            h2 = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
            idx2 = h2 % dim
            vec[idx2] += sign * 0.5
    
    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ============================================================
# MEMORY ENTRY
# ============================================================

@dataclass
class MemoryEntry:
    """Single memory item."""
    text: str
    embedding: np.ndarray
    tick: int
    importance: float          # 0.0-10.0 scale
    memory_type: str           # "action", "observation", "insight", "social"
    related_agents: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    access_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tick": self.tick,
            "importance": self.importance,
            "type": self.memory_type,
            "related_agents": self.related_agents,
            "metadata": self.metadata,
            "access_count": self.access_count,
        }


# ============================================================
# IMPORTANCE SCORING
# ============================================================

# Keywords → importance score adjustments
IMPORTANCE_KEYWORDS = {
    # Critical events (8-10)
    "died": 10.0, "death": 10.0, "killed": 10.0,
    "critical": 8.0, "emergency": 8.0, "lethal": 8.0,
    "vacuum": 9.0, "ebullism": 9.0,
    
    # Significant events (5-7)
    "built": 6.0, "constructed": 6.0, "completed": 6.0,
    "discovered": 7.0, "found": 5.0,
    "injured": 6.0, "disease": 6.0, "radiation": 6.0,
    "flare": 7.0, "quake": 7.0, "storm": 7.0,
    "trust": 5.0, "conflict": 6.0, "cooperation": 5.0,
    
    # Moderate events (3-4)
    "gathered": 3.0, "collected": 3.0,
    "ate": 2.0, "drank": 2.0, "slept": 2.0,
    "explored": 3.0, "moved": 2.0,
    "repaired": 4.0, "crafted": 4.0,
    
    # Routine (1-2)
    "idle": 1.0, "waiting": 1.0, "resting": 1.0,
}


def calculate_importance(text: str, base_importance: float = 3.0) -> float:
    """
    Calculate importance score for a memory.
    
    Uses keyword matching + base importance.
    Range: 0.0 to 10.0
    """
    score = base_importance
    text_lower = text.lower()
    
    for keyword, weight in IMPORTANCE_KEYWORDS.items():
        if keyword in text_lower:
            score = max(score, weight)
    
    return min(10.0, score)


# ============================================================
# VECTOR MEMORY STORE
# ============================================================

class VectorMemoryStore:
    """
    Per-agent vector memory store.
    
    Stores memories as text + embedding vectors.
    Retrieves using weighted scoring formula from Generative Agents paper:
      score = 0.4×similarity + 0.3×recency + 0.3×importance
    
    Capacity: max_entries memories per agent (oldest pruned).
    """
    
    # Retrieval weights (Generative Agents inspired)
    WEIGHT_SIMILARITY = 0.4
    WEIGHT_RECENCY = 0.3
    WEIGHT_IMPORTANCE = 0.3
    
    def __init__(self, agent_id: str, max_entries: int = 200):
        self.agent_id = agent_id
        self.max_entries = max_entries
        self._memories: list[MemoryEntry] = []
    
    def add(self, text: str, tick: int,
            importance: float = None,
            memory_type: str = "observation",
            related_agents: list[str] = None,
            metadata: dict = None) -> MemoryEntry:
        """
        Add a new memory.
        
        Args:
            text: Natural language description
            tick: Simulation tick when event occurred
            importance: Importance score (auto-calculated if None)
            memory_type: Type tag (action/observation/insight/social)
            related_agents: IDs of agents involved
            metadata: Additional key-value data
        
        Returns:
            The created MemoryEntry
        """
        if importance is None:
            importance = calculate_importance(text)
        
        embedding = embed_text(text)
        
        entry = MemoryEntry(
            text=text,
            embedding=embedding,
            tick=tick,
            importance=importance,
            memory_type=memory_type,
            related_agents=related_agents or [],
            metadata=metadata or {},
        )
        
        self._memories.append(entry)
        
        # Prune if over capacity
        if len(self._memories) > self.max_entries:
            self._prune()
        
        return entry
    
    def retrieve(self, query: str, current_tick: int,
                 top_k: int = 5,
                 memory_type: str = None) -> list[MemoryEntry]:
        """
        Retrieve most relevant memories for a query.
        
        Uses weighted scoring:
          score = 0.4×similarity + 0.3×recency + 0.3×importance
        
        Args:
            query: Search query text
            current_tick: Current simulation tick (for recency calc)
            top_k: Number of results to return
            memory_type: Optional filter by memory type
        
        Returns:
            List of MemoryEntry sorted by relevance score
        """
        if not self._memories:
            return []
        
        query_embedding = embed_text(query)
        candidates = self._memories
        
        # Filter by type if specified
        if memory_type:
            candidates = [m for m in candidates if m.memory_type == memory_type]
        
        if not candidates:
            return []
        
        # Score each memory
        scored = []
        max_tick = max(m.tick for m in candidates)
        tick_range = max(1, current_tick - min(m.tick for m in candidates))
        
        for memory in candidates:
            # Similarity (cosine, 0 to 1)
            sim = max(0.0, cosine_similarity(query_embedding, memory.embedding))
            
            # Recency (exponential decay, 0 to 1)
            # More recent = higher score
            age = current_tick - memory.tick
            recency = np.exp(-age / max(1, tick_range * 0.5))
            
            # Importance (normalized, 0 to 1)
            imp = memory.importance / 10.0
            
            # Weighted score
            score = (
                self.WEIGHT_SIMILARITY * sim +
                self.WEIGHT_RECENCY * recency +
                self.WEIGHT_IMPORTANCE * imp
            )
            
            scored.append((score, memory))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Update access counts
        results = []
        for score, memory in scored[:top_k]:
            memory.access_count += 1
            results.append(memory)
        
        return results
    
    def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Get N most recent memories."""
        return sorted(self._memories, key=lambda m: m.tick, reverse=True)[:n]
    
    def get_by_type(self, memory_type: str) -> list[MemoryEntry]:
        """Get all memories of a specific type."""
        return [m for m in self._memories if m.memory_type == memory_type]
    
    def get_insights(self) -> list[MemoryEntry]:
        """Get all insight-type memories (from reflection)."""
        return self.get_by_type("insight")
    
    def _prune(self):
        """Remove lowest-value memories when over capacity."""
        if len(self._memories) <= self.max_entries:
            return
        
        # Score for pruning: low importance + old + low access = prune candidate
        max_tick = max(m.tick for m in self._memories)
        
        def prune_score(m: MemoryEntry) -> float:
            age_factor = (max_tick - m.tick) / max(1, max_tick)
            return m.importance * 0.4 + (1 - age_factor) * 0.3 + m.access_count * 0.3
        
        self._memories.sort(key=prune_score, reverse=True)
        self._memories = self._memories[:self.max_entries]
    
    def count(self) -> int:
        return len(self._memories)
    
    def to_text_list(self, n: int = None) -> list[str]:
        """Get memory texts as list (for reflection prompts)."""
        memories = self._memories[-n:] if n else self._memories
        return [m.text for m in memories]
    
    def stats(self) -> dict:
        """Get memory store statistics."""
        if not self._memories:
            return {"count": 0, "types": {}, "avg_importance": 0}
        
        types = {}
        for m in self._memories:
            types[m.memory_type] = types.get(m.memory_type, 0) + 1
        
        return {
            "count": len(self._memories),
            "types": types,
            "avg_importance": round(
                sum(m.importance for m in self._memories) / len(self._memories), 2
            ),
            "oldest_tick": min(m.tick for m in self._memories),
            "newest_tick": max(m.tick for m in self._memories),
        }


# ============================================================
# MEMORY MANAGER (manages all agent memories)
# ============================================================

class MemoryManager:
    """
    Global memory manager — one VectorMemoryStore per agent.
    
    Provides centralized access to all agent memories and
    handles cross-agent memory operations (social interactions).
    """
    
    def __init__(self, max_memories_per_agent: int = 200):
        self._stores: dict[str, VectorMemoryStore] = {}
        self._max_per_agent = max_memories_per_agent
    
    def get_store(self, agent_id: str) -> VectorMemoryStore:
        """Get or create memory store for an agent."""
        if agent_id not in self._stores:
            self._stores[agent_id] = VectorMemoryStore(
                agent_id, max_entries=self._max_per_agent
            )
        return self._stores[agent_id]
    
    def record_action(self, agent_id: str, action: str, result: dict,
                      tick: int, related_agents: list[str] = None):
        """Record an agent action as a memory."""
        store = self.get_store(agent_id)
        
        # Build natural language description
        text = f"Tick {tick}: {action}"
        if isinstance(result, dict):
            if result.get("reasoning"):
                text += f" — {result['reasoning']}"
            elif result.get("target"):
                text += f" targeting {result['target']}"
        
        store.add(
            text=text,
            tick=tick,
            memory_type="action",
            related_agents=related_agents,
            metadata={"action": action, "result_keys": list(result.keys()) if isinstance(result, dict) else []},
        )
    
    def record_event(self, agent_id: str, event_text: str,
                     tick: int, importance: float = 5.0):
        """Record a world event as observed by an agent."""
        store = self.get_store(agent_id)
        store.add(
            text=f"Tick {tick}: {event_text}",
            tick=tick,
            importance=importance,
            memory_type="observation",
        )
    
    def record_social(self, agent_id: str, other_id: str,
                      interaction: str, tick: int):
        """Record a social interaction."""
        store = self.get_store(agent_id)
        store.add(
            text=f"Tick {tick}: {interaction}",
            tick=tick,
            memory_type="social",
            related_agents=[other_id],
        )
    
    def add_insight(self, agent_id: str, insight: str, tick: int):
        """Add a reflection insight (high importance)."""
        store = self.get_store(agent_id)
        store.add(
            text=insight,
            tick=tick,
            importance=8.0,  # Insights are always high importance
            memory_type="insight",
        )
    
    def get_context_for_prompt(self, agent_id: str, query: str,
                               current_tick: int, top_k: int = 5) -> list[str]:
        """Get relevant memories formatted for LLM prompt context."""
        store = self.get_store(agent_id)
        memories = store.retrieve(query, current_tick, top_k=top_k)
        return [m.text for m in memories]
    
    def get_all_stats(self) -> dict:
        """Get stats for all agent memory stores."""
        return {
            agent_id: store.stats()
            for agent_id, store in self._stores.items()
        }
