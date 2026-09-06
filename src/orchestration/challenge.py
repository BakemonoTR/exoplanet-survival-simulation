"""Backend-only global deployment challenge state.

The challenge clock is global.  Individual planet attempts may reset, while
planet-specific learned policies survive those physical resets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
import random
import time
from typing import Callable, Optional

from src.systems.mission_profile import MissionProfile

_MISSION_PROFILE = MissionProfile.load()
GLOBAL_CHALLENGE_SECONDS = int(
    _MISSION_PROFILE.clock.public_challenge_days * 24 * 60 * 60
)
SIM_MINUTES_PER_TICK = float(_MISSION_PROFILE.clock.tick_minutes)
REALTIME_SECONDS_PER_TICK = float(
    _MISSION_PROFILE.clock.realtime_seconds_per_tick
)
EPISODE_MAX_SIM_DAYS = int(
    _MISSION_PROFILE.clock.planet_attempt_max_earth_days
)
EPISODE_MAX_TICKS = int(_MISSION_PROFILE.clock.planet_attempt_max_ticks)


@dataclass
class ChallengeConfig:
    global_duration_seconds: int = GLOBAL_CHALLENGE_SECONDS
    completion_score: float = 100.0
    realtime_seconds_per_tick: float = REALTIME_SECONDS_PER_TICK
    sim_minutes_per_tick: float = SIM_MINUTES_PER_TICK
    episode_max_ticks: Optional[int] = EPISODE_MAX_TICKS
    # Stagnation remains deliberately unset. A failed physical episode ends
    # at death or its configured attempt cap until a preregistered stagnation
    # definition is chosen.
    stagnation_timeout_ticks: Optional[int] = None


@dataclass
class ChallengeState:
    planet_ids: list[str]
    remaining_planets: list[str]
    completed_planets: list[str] = field(default_factory=list)
    attempts_by_planet: dict[str, int] = field(default_factory=dict)
    best_score_by_planet: dict[str, float] = field(default_factory=dict)
    active_planet: Optional[str] = None
    started_at: Optional[float] = None
    deadline_at: Optional[float] = None
    status: str = "pending"
    wheel_seed: int = 0
    wheel_spins: int = 0
    last_planet: Optional[str] = None


class PlanetChallengeCoordinator:
    """Persist and advance the one-clock, many-planet backend challenge."""

    def __init__(
        self,
        planet_ids: list[str],
        *,
        config: ChallengeConfig | None = None,
        state_path: str | None = None,
        seed: int = 0,
        now: Callable[[], float] = time.time,
    ):
        normalized = list(dict.fromkeys(str(p) for p in planet_ids))
        if not normalized:
            raise ValueError("At least one planet is required")
        self.config = config or ChallengeConfig()
        self.state_path = state_path
        self._now = now
        self.state = ChallengeState(
            planet_ids=normalized,
            remaining_planets=list(normalized),
            attempts_by_planet={planet: 0 for planet in normalized},
            best_score_by_planet={planet: 0.0 for planet in normalized},
            wheel_seed=int(seed),
        )
        if state_path and os.path.exists(state_path):
            self._load()

    def start(self) -> ChallengeState:
        """Start the configured public clock; repeated calls do not reset it."""
        if self.state.started_at is None:
            self.state.started_at = float(self._now())
            self.state.deadline_at = (
                self.state.started_at + self.config.global_duration_seconds
            )
            self.state.status = "running"
            self._save()
        return self.state

    def remaining_seconds(self) -> float:
        if self.state.deadline_at is None:
            return float(self.config.global_duration_seconds)
        return max(0.0, self.state.deadline_at - float(self._now()))

    def _expire_if_needed(self) -> bool:
        if (
            self.state.status == "running"
            and self.state.deadline_at is not None
            and float(self._now()) >= self.state.deadline_at
        ):
            self.state.status = "failed_deadline"
            self.state.active_planet = None
            self._save()
            return True
        return False

    def spin_next_planet(self) -> Optional[str]:
        """Choose randomly from unfinished planets only."""
        if self.state.started_at is None:
            self.start()
        if self._expire_if_needed() or self.state.status != "running":
            return None
        if self.state.active_planet is not None:
            return self.state.active_planet
        if not self.state.remaining_planets:
            self.state.status = "completed"
            self._save()
            return None

        rng = random.Random(
            f"{self.state.wheel_seed}:{self.state.wheel_spins}"
        )
        candidates = [
            planet
            for planet in self.state.remaining_planets
            if planet != self.state.last_planet
        ] or list(self.state.remaining_planets)
        planet = rng.choice(sorted(candidates))
        self.state.wheel_spins += 1
        self.state.active_planet = planet
        self._save()
        return planet

    def finish_active_attempt(self, colony_score: float) -> dict:
        """Record a reset/success; a planet leaves the wheel only at 100."""
        planet = self.state.active_planet
        if planet is None:
            raise RuntimeError("No active planet attempt")

        score = max(0.0, min(100.0, float(colony_score)))
        self.state.attempts_by_planet[planet] += 1
        self.state.best_score_by_planet[planet] = max(
            self.state.best_score_by_planet.get(planet, 0.0), score
        )
        self.state.active_planet = None
        self.state.last_planet = planet

        expired = self._expire_if_needed()
        succeeded = (
            not expired and score >= self.config.completion_score
        )
        if succeeded:
            self.state.remaining_planets.remove(planet)
            self.state.completed_planets.append(planet)
            if not self.state.remaining_planets:
                self.state.status = "completed"
        self._save()
        return {
            "planet": planet,
            "succeeded": succeeded,
            "site_reset_required": not succeeded,
            "challenge_status": self.state.status,
            "remaining_planets": list(self.state.remaining_planets),
        }

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "state": asdict(self.state),
            "remaining_seconds": self.remaining_seconds(),
        }

    def _save(self) -> None:
        if not self.state_path:
            return
        absolute = os.path.abspath(self.state_path)
        os.makedirs(os.path.dirname(absolute), exist_ok=True)
        temporary = f"{absolute}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
        os.replace(temporary, absolute)

    def _load(self) -> None:
        with open(self.state_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        loaded_config = ChallengeConfig(**payload.get("config", {}))
        loaded_state = ChallengeState(**payload["state"])
        if set(loaded_state.planet_ids) != set(self.state.planet_ids):
            raise ValueError("Persisted challenge planet list does not match")
        self.config = loaded_config
        self.state = loaded_state
