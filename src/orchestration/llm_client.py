"""
LLM Client with API Key Pool, Rate Limiting, Caching, and Fallback.

Architecture:
- APIKeyPool: Round-robin across multiple Groq accounts
- RateLimiter: Per-key RPM/RPD/TPM/TPD tracking
- ResponseCache: SHA-256 hash-based prompt→response cache
- GroqLLMClient: Async-capable client with retry + backoff

Groq Free Tier Limits (per account, June 2026):
- RPM: 30 requests/minute
- RPD: 1,000 requests/day
- TPM: 12,000 tokens/minute
- TPD: 100,000 tokens/day

Reference: https://console.groq.com/docs/rate-limits
"""

import hashlib
import json
import time
import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================
# API KEY POOL
# ============================================================

@dataclass
class KeyStats:
    """Per-key usage tracking."""
    key_id: str
    calls_today: int = 0
    tokens_today: int = 0
    calls_this_minute: int = 0
    tokens_this_minute: int = 0
    last_call_time: float = 0.0
    last_minute_reset: float = 0.0
    last_day_reset: float = 0.0
    errors_consecutive: int = 0
    is_exhausted: bool = False
    exhausted_until: float = 0.0  # Unix timestamp when key becomes available again


class APIKeyPool:
    """
    Round-robin API key pool with per-key rate tracking.
    
    Each key represents a separate Groq account with independent limits.
    When a key hits its limit, it's marked exhausted and skipped.
    """
    
    # Groq free tier limits
    RPM_LIMIT = 30
    RPD_LIMIT = 1000
    TPM_LIMIT = 12000
    TPD_LIMIT = 100000
    
    def __init__(self, api_keys: list[str]):
        if not api_keys:
            raise ValueError("At least one API key required")
        
        self._keys = api_keys
        self._stats: dict[str, KeyStats] = {}
        self._current_index = 0
        self._lock = threading.Lock()
        
        for i, key in enumerate(api_keys):
            key_id = f"key_{i}"
            self._stats[key_id] = KeyStats(key_id=key_id)
        
        logger.info(f"APIKeyPool initialized with {len(api_keys)} keys")
    
    def get_next_key(self) -> Optional[tuple[str, str]]:
        """
        Get next available API key (round-robin with exhaustion skip).
        
        Returns:
            (api_key, key_id) tuple, or None if all keys exhausted.
        """
        with self._lock:
            now = time.time()
            attempts = 0

            while attempts < len(self._keys):
                idx = self._current_index % len(self._keys)
                self._current_index += 1
                attempts += 1

                key = self._keys[idx]
                key_id = f"key_{idx}"
                stats = self._stats[key_id]

                # Reset minute counters if >60s elapsed
                if now - stats.last_minute_reset > 60:
                    stats.calls_this_minute = 0
                    stats.tokens_this_minute = 0
                    stats.last_minute_reset = now

                # Reset daily counters if >24h elapsed
                if now - stats.last_day_reset > 86400:
                    stats.calls_today = 0
                    stats.tokens_today = 0
                    stats.last_day_reset = now
                    stats.is_exhausted = False

                # Check if key recovered from exhaustion
                if stats.is_exhausted and now > stats.exhausted_until:
                    stats.is_exhausted = False
                    stats.errors_consecutive = 0

                # Skip exhausted keys
                if stats.is_exhausted:
                    continue

                # Check rate limits
                if stats.calls_this_minute >= self.RPM_LIMIT:
                    continue  # Try next key
                if stats.calls_today >= self.RPD_LIMIT:
                    stats.is_exhausted = True
                    stats.exhausted_until = stats.last_day_reset + 86400
                    continue
                if stats.tokens_today >= self.TPD_LIMIT:
                    stats.is_exhausted = True
                    stats.exhausted_until = stats.last_day_reset + 86400
                    continue

                return (key, key_id)
        
        # All keys exhausted
        return None
    
    def record_usage(self, key_id: str, tokens_used: int):
        """Record successful API call usage."""
        with self._lock:
            stats = self._stats[key_id]
            stats.calls_today += 1
            stats.calls_this_minute += 1
            stats.tokens_today += tokens_used
            stats.tokens_this_minute += tokens_used
            stats.last_call_time = time.time()
            stats.errors_consecutive = 0
    
    def record_error(self, key_id: str, is_rate_limit: bool = False):
        """Record API error. Rate limit errors exhaust the key temporarily."""
        with self._lock:
            stats = self._stats[key_id]
            stats.errors_consecutive += 1

            if is_rate_limit:
                # Exponential backoff: 60s, 120s, 240s, ...
                backoff = min(60 * (2 ** stats.errors_consecutive), 3600)
                stats.is_exhausted = True
                stats.exhausted_until = time.time() + backoff
                logger.warning(f"{key_id} rate-limited, backing off {backoff}s")
    
    def get_pool_status(self) -> dict:
        """Get usage summary for all keys."""
        with self._lock:
            return {
                kid: {
                    "calls_today": s.calls_today,
                    "tokens_today": s.tokens_today,
                    "exhausted": s.is_exhausted,
                    "rpm_used": s.calls_this_minute,
                }
                for kid, s in self._stats.items()
            }
    
    @property
    def all_exhausted(self) -> bool:
        with self._lock:
            return all(s.is_exhausted for s in self._stats.values())
    
    @property
    def total_calls_today(self) -> int:
        with self._lock:
            return sum(s.calls_today for s in self._stats.values())
    
    @property
    def total_tokens_today(self) -> int:
        with self._lock:
            return sum(s.tokens_today for s in self._stats.values())


# ============================================================
# RESPONSE CACHE
# ============================================================

@dataclass
class CacheEntry:
    """Cached LLM response with TTL."""
    response: dict
    created_tick: int
    ttl_ticks: int = 50  # Cache expires after 50 ticks (~4 sim hours)
    hit_count: int = 0


class ResponseCache:
    """
    SHA-256 hash-based prompt→response cache.
    
    Identical prompts (same system + user message) return cached response.
    TTL prevents stale context from causing bad decisions.
    """
    
    def __init__(self, ttl_ticks: int = 50, max_entries: int = 500):
        self._cache: dict[str, CacheEntry] = {}
        self._ttl = ttl_ticks
        self._max_entries = max_entries
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()
    
    def _hash_prompt(self, system: str, user: str) -> str:
        """Generate deterministic hash for prompt pair."""
        content = f"{system}|||{user}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def get(self, system: str, user: str, current_tick: int) -> Optional[dict]:
        """Retrieve cached response if fresh."""
        key = self._hash_prompt(system, user)
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            # Check TTL
            if current_tick - entry.created_tick > entry.ttl_ticks:
                del self._cache[key]
                self._misses += 1
                return None

            entry.hit_count += 1
            self._hits += 1
            return entry.response
    
    def put(self, system: str, user: str, response: dict, current_tick: int):
        """Store response in cache."""
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self._max_entries:
                oldest_key = min(self._cache, key=lambda k: self._cache[k].created_tick)
                del self._cache[oldest_key]

            key = self._hash_prompt(system, user)
            self._cache[key] = CacheEntry(
                response=response,
                created_tick=current_tick,
                ttl_ticks=self._ttl,
            )
    
    def clear_expired(self, current_tick: int):
        """Remove all expired entries."""
        with self._lock:
            expired = [
                k for k, v in self._cache.items()
                if current_tick - v.created_tick > v.ttl_ticks
            ]
            for k in expired:
                del self._cache[k]
    
    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / max(1, total), 3),
            }


# ============================================================
# GROQ LLM CLIENT
# ============================================================

class LLMCallType(Enum):
    """Types of LLM calls for logging and budgeting."""
    STRATEGIC = "strategic"    # Goal selection (~every 10 ticks)
    TACTICAL = "tactical"      # Immediate action (on-demand)
    REFLECTION = "reflection"  # Memory synthesis (~every 20 ticks)
    SOCIAL = "social"          # Agent dialogue


class GroqLLMClient:
    """
    Production LLM client for the simulation.
    
    Features:
    - Async-ready (sync wrapper for now, async later with FastAPI)
    - API key pool with round-robin
    - Response caching
    - Automatic fallback detection
    - JSON output parsing with validation
    - Usage tracking and logging
    """
    
    def __init__(self, api_keys: list[str] = None, model: str = None,
                 cache_ttl: int = 50, request_timeout_s: float = 3.0,
                 async_social: bool = True, max_async_pending: int = 8):
        """
        Initialize client.
        
        Args:
            api_keys: List of Groq API keys. If None, loads from .env
            model: Groq model name. Default: llama-3.3-70b-versatile
            cache_ttl: Cache TTL in simulation ticks
        """
        # Load keys from environment if not provided
        if api_keys is None:
            api_keys = self._load_keys_from_env()
        
        if not api_keys:
            logger.warning("No API keys provided — running in fallback-only mode")
            self._pool = None
        else:
            self._pool = APIKeyPool(api_keys)
        
        self._model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self._cache = ResponseCache(ttl_ticks=cache_ttl)
        self._groq_client = None  # Lazy init
        self._request_timeout_s = max(0.25, float(request_timeout_s))
        self._async_social = bool(async_social)
        self._max_async_pending = max(1, int(max_async_pending))
        self._async_executor: Optional[ThreadPoolExecutor] = None
        self._async_pending: set[str] = set()
        self._async_lock = threading.Lock()
        self._closed = False
        self._stats_lock = threading.Lock()
        self._async_submitted = 0
        self._async_dropped = 0
        
        # Usage stats
        self._total_calls = 0
        self._total_tokens = 0
        self._fallback_calls = 0
        self._call_log: list[dict] = []  # Recent calls for debugging
    
    def _load_keys_from_env(self) -> list[str]:
        """Load API keys from .env or environment variables."""
        keys = []
        # Try dotenv
        env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('GROQ_API_KEY') and '=' in line:
                        val = line.split('=', 1)[1].strip()
                        if val and val != 'your_key_here':
                            keys.append(val)
        
        # Also check environment variables
        for i in range(1, 6):
            key = os.getenv(f"GROQ_API_KEY_{i}")
            if key and key not in keys:
                keys.append(key)
        
        return keys
    
    def _get_groq_client(self, api_key: str):
        """Get or create Groq client for a specific key."""
        try:
            from groq import Groq
            return Groq(
                api_key=api_key,
                max_retries=0,
                timeout=self._request_timeout_s,
            )
        except ImportError:
            logger.error("groq package not installed. Run: pip install groq")
            return None
    
    def call(self, system_prompt: str, user_prompt: str,
             call_type: LLMCallType = LLMCallType.TACTICAL,
             current_tick: int = 0,
             temperature: float = 0.7,
             max_tokens: int = 150) -> dict:
        """
        Make an LLM call with caching, rate limiting, and fallback.
        
        Args:
            system_prompt: System message (character definition)
            user_prompt: User message (current state + decision request)
            call_type: Type of call for logging
            current_tick: Current simulation tick (for cache TTL)
            temperature: LLM temperature (0.0=deterministic, 1.0=creative)
            max_tokens: Maximum response tokens
            
        Returns:
            dict with: {"action": str, "target": dict, "reasoning": str, ...}
            OR {"fallback": True, ...} if all APIs exhausted
        """
        # 1. Check cache first
        cached = self._cache.get(system_prompt, user_prompt, current_tick)
        if cached is not None:
            logger.debug(f"Cache hit for {call_type.value}")
            return cached

        # Social dialogue is narrative-only. Never let a network request pause
        # the authoritative physics tick: schedule it on a bounded worker and
        # return immediately. If an identical encounter recurs, its completed
        # result is served from the normal response cache above.
        if call_type == LLMCallType.SOCIAL and self._async_social:
            self._submit_social_call(
                system_prompt,
                user_prompt,
                current_tick=current_tick,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return None
        
        # 2. Try API call
        if self._pool is not None:
            result = self._try_api_call(
                system_prompt, user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if result is not None:
                self._record_success(
                    system_prompt, user_prompt, result,
                    call_type=call_type,
                    current_tick=current_tick,
                )
                return result
        
        # 3. All APIs exhausted → return None (caller handles fallback)
        with self._stats_lock:
            self._fallback_calls += 1
        logger.info(f"All APIs exhausted, fallback needed for {call_type.value}")
        return None

    def _record_success(self, system_prompt: str, user_prompt: str,
                        result: dict, call_type: LLMCallType,
                        current_tick: int):
        """Record and cache one completed call from either execution path."""
        self._cache.put(system_prompt, user_prompt, result, current_tick)
        with self._stats_lock:
            self._total_calls += 1
            self._call_log.append({
                "tick": current_tick,
                "type": call_type.value,
                "tokens": result.get("_tokens_used", 0),
                "cached": False,
            })
            if len(self._call_log) > 100:
                self._call_log = self._call_log[-50:]

    def _submit_social_call(self, system_prompt: str, user_prompt: str,
                            current_tick: int, temperature: float,
                            max_tokens: int) -> bool:
        """Queue a deduplicated social call without blocking the tick thread."""
        prompt_key = self._cache._hash_prompt(system_prompt, user_prompt)
        with self._async_lock:
            if (self._closed or prompt_key in self._async_pending or
                    len(self._async_pending) >= self._max_async_pending):
                self._async_dropped += 1
                return False
            if self._async_executor is None:
                # One worker deliberately preserves rate-limit ordering while
                # the bounded pending set prevents an unbounded encounter queue.
                self._async_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="social-llm",
                )
            self._async_pending.add(prompt_key)
            self._async_submitted += 1

        try:
            self._async_executor.submit(
                self._run_social_call,
                prompt_key,
                system_prompt,
                user_prompt,
                current_tick,
                temperature,
                max_tokens,
            )
            return True
        except RuntimeError:
            with self._async_lock:
                self._async_pending.discard(prompt_key)
                self._async_dropped += 1
            return False

    def _run_social_call(self, prompt_key: str, system_prompt: str,
                         user_prompt: str, current_tick: int,
                         temperature: float, max_tokens: int):
        """Worker body for non-authoritative dialogue generation."""
        try:
            if self._pool is None:
                with self._stats_lock:
                    self._fallback_calls += 1
                return
            result = self._try_api_call(
                system_prompt,
                user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                max_attempts=1,
            )
            if result is not None:
                self._record_success(
                    system_prompt, user_prompt, result,
                    call_type=LLMCallType.SOCIAL,
                    current_tick=current_tick,
                )
            else:
                with self._stats_lock:
                    self._fallback_calls += 1
        finally:
            with self._async_lock:
                self._async_pending.discard(prompt_key)
    
    def _try_api_call(self, system: str, user: str,
                      temperature: float = 0.7,
                      max_tokens: int = 300,
                      max_attempts: Optional[int] = None) -> Optional[dict]:
        """Attempt API call with key rotation and retry."""
        if "json" not in (system + user).lower():
            system = (system or "You are a specialized colony AI.") + "\nRespond strictly in valid JSON format."
            
        max_retries = min(3, len(self._pool._keys) if self._pool else 1)
        if max_attempts is not None:
            max_retries = min(max_retries, max(1, int(max_attempts)))
        
        for attempt in range(max_retries):
            key_pair = self._pool.get_next_key()
            if key_pair is None:
                return None  # All exhausted
            
            api_key, key_id = key_pair
            
            try:
                client = self._get_groq_client(api_key)
                if client is None:
                    return None
                
                # Try primary model first, fallback to standard mode or secondary model if needed
                try:
                    response = client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                except Exception as model_err:
                    err_str = str(model_err)
                    if "json_validate_failed" in err_str or "Failed to validate JSON" in err_str or "400" in err_str:
                        # Retry without strict json_object enforcement
                        try:
                            response = client.chat.completions.create(
                                model=self._model,
                                messages=[
                                    {"role": "system", "content": system + "\nOutput valid JSON only."},
                                    {"role": "user", "content": user},
                                ],
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                        except Exception as retry_err:
                            raise retry_err
                    else:
                        fallback_model = "openai/gpt-oss-120b"
                        if ("429" in err_str or "404" in err_str or "model_not_found" in err_str) and self._model != fallback_model:
                            logger.info(f"Primary model unavailable; attempting fallback model {fallback_model}")
                            response = client.chat.completions.create(
                                model=fallback_model,
                                messages=[
                                    {"role": "system", "content": system},
                                    {"role": "user", "content": user},
                                ],
                                temperature=temperature,
                                max_tokens=max_tokens,
                                response_format={"type": "json_object"},
                            )
                        else:
                            raise model_err
                
                # Parse response
                content = response.choices[0].message.content
                tokens_used = (
                    response.usage.prompt_tokens + 
                    response.usage.completion_tokens
                )
                
                # Record usage
                self._pool.record_usage(key_id, tokens_used)
                with self._stats_lock:
                    self._total_tokens += tokens_used
                
                # Parse JSON
                try:
                    cleaned_content = content.strip()
                    if "```json" in cleaned_content:
                        cleaned_content = cleaned_content.split("```json")[1].split("```")[0].strip()
                    elif "```" in cleaned_content:
                        cleaned_content = cleaned_content.split("```")[1].split("```")[0].strip()
                    result = json.loads(cleaned_content)
                    result["_tokens_used"] = tokens_used
                    result["_key_id"] = key_id
                    return result
                except json.JSONDecodeError:
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            result = json.loads(json_match.group())
                            result["_tokens_used"] = tokens_used
                            result["_key_id"] = key_id
                            return result
                        except Exception:
                            pass
                    logger.warning(f"Invalid JSON from LLM: {content[:100]}")
                    # Return raw text wrapped in dict
                    return {
                        "action": "idle",
                        "reasoning": content[:200],
                        "_tokens_used": tokens_used,
                        "_parse_error": True,
                    }
                    
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in error_str or "rate" in error_str
                self._pool.record_error(key_id, is_rate_limit=is_rate_limit)
                logger.warning(f"API error on {key_id} (attempt {attempt+1}): {e}")
                
                if "404" in error_str or "model_not_found" in error_str or "invalid_api_key" in error_str:
                    # Model not available or invalid credentials — return None immediately to activate fallback engine
                    return None
                if not is_rate_limit:
                    continue
                # Rate limit: try next key
                continue
        
        return None
    
    @property
    def all_exhausted(self) -> bool:
        """Return True if all keys are exhausted."""
        return self._pool.all_exhausted if self._pool else True

    def get_status(self) -> dict:
        """Get client status for monitoring."""
        with self._stats_lock:
            total_calls = self._total_calls
            total_tokens = self._total_tokens
            fallback_calls = self._fallback_calls
        with self._async_lock:
            async_status = {
                "pending": len(self._async_pending),
                "submitted": self._async_submitted,
                "dropped": self._async_dropped,
            }
        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "fallback_calls": fallback_calls,
            "cache": self._cache.stats,
            "pool": self._pool.get_pool_status() if self._pool else {},
            "all_exhausted": self.all_exhausted,
            "model": self._model,
            "async_social": async_status,
        }

    def close(self, wait: bool = False):
        """Stop accepting narrative work; pending calls are safely cancellable."""
        with self._async_lock:
            executor = self._async_executor
            self._async_executor = None
            self._closed = True
            self._async_pending.clear()
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)
