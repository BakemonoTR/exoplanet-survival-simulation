"""Local-only narrative adapter for an optional fine-tuned GPT-2 model.

Physics and reinforcement learning remain authoritative. Generated text may
only provide dialogue and reflections. This module performs no downloads and
contains no hosted API integration.
"""

from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import hashlib
import logging
import os
import threading
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class LLMCallType(Enum):
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    REFLECTION = "reflection"
    SOCIAL = "social"


class LocalNarrativeProvider(Protocol):
    """Interface implemented later by the locally loaded GPT-2 runtime."""

    def generate(self, *, system_prompt: str, user_prompt: str,
                 call_type: str, temperature: float,
                 max_tokens: int) -> dict:
        """Return narrative data using only local model weights."""


class LocalNarrativeClient:
    """Non-blocking, bounded bridge to a local narrative provider."""

    def __init__(self, provider: Optional[LocalNarrativeProvider] = None,
                 enabled: Optional[bool] = None, model: Optional[str] = None,
                 max_pending: int = 8):
        self._provider = provider
        self._enabled = (
            os.getenv("LOCAL_NARRATIVE_ENABLED", "false").lower()
            in {"1", "true", "yes"}
            if enabled is None else bool(enabled)
        )
        self._model = model or os.getenv("LOCAL_NARRATIVE_MODEL", "gpt2")
        self._max_pending = max(1, int(max_pending))
        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending: set[str] = set()
        self._completed: dict[str, dict] = {}
        self._closed = False
        self._error: Optional[str] = None
        self._total_calls = 0
        self._total_tokens = 0
        self._dropped = 0

    @property
    def ready(self) -> bool:
        return self._enabled and self._provider is not None and not self._closed

    @property
    def all_exhausted(self) -> bool:
        return not self.ready

    def call(self, system_prompt: str, user_prompt: str,
             call_type: LLMCallType = LLMCallType.SOCIAL,
             current_tick: int = 0, temperature: float = 0.7,
             max_tokens: int = 150) -> Optional[dict]:
        # Local text is deliberately barred from strategic/tactical decisions.
        if call_type not in {LLMCallType.SOCIAL, LLMCallType.REFLECTION}:
            return None
        if not self.ready:
            return None
        key = hashlib.sha256(
            f"{call_type.value}\0{system_prompt}\0{user_prompt}".encode()
        ).hexdigest()
        with self._lock:
            completed = self._completed.pop(key, None)
            if completed is not None:
                return dict(completed)
            if self._closed or key in self._pending:
                return None
            if len(self._pending) >= self._max_pending:
                self._dropped += 1
                return None
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="local-narrative"
                )
            self._pending.add(key)
            self._executor.submit(
                self._generate, key, system_prompt, user_prompt,
                call_type, current_tick, temperature, max_tokens,
            )
        return None

    def _generate(self, key, system_prompt, user_prompt, call_type,
                  current_tick, temperature, max_tokens) -> None:
        try:
            result = self._provider.generate(
                system_prompt=system_prompt, user_prompt=user_prompt,
                call_type=call_type.value, temperature=temperature,
                max_tokens=max_tokens,
            )
            if not isinstance(result, dict):
                raise ValueError("Local narrative provider must return a dictionary")
            output = {"provider": "local_gpt2", "non_authoritative": True,
                      "generated_at_tick": int(current_tick)}
            if call_type == LLMCallType.SOCIAL:
                dialogue = result.get("dialogue")
                if not isinstance(dialogue, str) or not dialogue.strip():
                    raise ValueError("Local narrative provider returned no dialogue")
                output["dialogue"] = dialogue.strip()
            else:
                insights = result.get("insights")
                if not isinstance(insights, list):
                    raise ValueError("Local narrative provider returned no insights")
                output["insights"] = [
                    item.strip() for item in insights
                    if isinstance(item, str) and item.strip()
                ]
            with self._lock:
                if not self._closed:
                    self._completed[key] = output
                    while len(self._completed) > 128:
                        self._completed.pop(next(iter(self._completed)))
                    self._total_calls += 1
                    self._total_tokens += max(
                        0, int(result.get("_tokens_used", 0))
                    )
                    self._error = None
        except Exception:
            logger.exception("Local narrative generation failed")
            with self._lock:
                self._error = "Local model generation failed; inspect server logs."
        finally:
            with self._lock:
                self._pending.discard(key)

    def get_status(self) -> dict:
        with self._lock:
            status = ("closed" if self._closed else
                      "disabled" if not self._enabled else
                      "unconfigured" if self._provider is None else
                      "error" if self._error else "ready")
            reason = {
                "closed": "Narrative worker is closed.",
                "disabled": "Local GPT-2 narration is disabled.",
                "unconfigured": "Fine-tuned local GPT-2 provider is not connected.",
                "ready": "Local GPT-2 narrative provider is ready.",
                "error": self._error,
            }[status]
            return {
                "provider": "local_gpt2", "model": self._model,
                "status": status, "enabled": self._enabled,
                "configured": self._provider is not None,
                "ready": status == "ready", "reason": reason,
                "total_calls": self._total_calls,
                "total_tokens": self._total_tokens,
                "fallback_calls": 0, "pending": len(self._pending),
                "dropped": self._dropped,
                "cache": {"size": len(self._completed), "hit_rate": 0.0},
                "all_exhausted": status != "ready",
                "non_authoritative": True,
            }

    def close(self, wait: bool = False) -> None:
        with self._lock:
            self._closed = True
            executor, self._executor = self._executor, None
            self._completed.clear()
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)
