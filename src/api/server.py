"""
FastAPI Server with WebSocket for real-time simulation streaming.

Endpoints:
- POST /api/simulation/start — Start new simulation
- GET  /api/simulation/state — Current state snapshot
- GET  /api/simulation/status — Engine status (running/paused/stopped)
- POST /api/simulation/control — Pause/resume/stop/speed
- GET  /api/simulation/history — Past runs list
- GET  /api/simulation/history/{run_id} — Run detail + score chart data
- WS   /ws/live — Real-time tick updates

Frontend served from /frontend/ directory.
"""

import asyncio
import json
import logging
import math
import os
import threading
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.orchestration.engine import SimulationEngine
from src.orchestration.challenge import (
    ChallengeConfig,
    PlanetChallengeCoordinator,
)
from src.api.database import SimulationDB
from src.systems.mission_profile import MissionProfile

logger = logging.getLogger(__name__)

# ============================================================
# GLOBALS
# ============================================================

_engine: Optional[SimulationEngine] = None
_db: Optional[SimulationDB] = None
_engine_thread: Optional[threading.Thread] = None
_ws_clients: list[WebSocket] = []
_tick_buffer: list[dict] = []  # Buffer for tick events
_telemetry_event_buffer: list[dict] = []
_lifecycle_lock = threading.RLock()
_debug_session_token = 0
_debug_restart_cancel = threading.Event()
_debug_restart_thread: Optional[threading.Thread] = None
_challenge: Optional[PlanetChallengeCoordinator] = None
_debug_session: dict = {
    "enabled": False,
    "attempt": 0,
    "max_attempts": 15,
    "limit_reached": False,
    "restart_pending": False,
    "restart_at": None,
    "restart_delay_seconds": 2.0,
    "planet": None,
    "seed": None,
    "max_ticks": None,
    "tick_speed": None,
    "history": [],
    "challenge_mode": False,
    "dialogue_generation_enabled": False,
    "last_error": None,
}


# ============================================================
# MODELS
# ============================================================

def _mission_default_max_ticks() -> int:
    """Use the validated mission clock as the single attempt-length source."""
    max_ticks = MissionProfile.load().clock.planet_attempt_max_ticks
    if max_ticks is None:
        raise ValueError("mission profile must define an attempt tick limit")
    return int(max_ticks)


def _mission_default_tick_speed() -> float:
    return float(MissionProfile.load().clock.realtime_seconds_per_tick)


class SimulationStartRequest(BaseModel):
    planet: str = "kepler-442b"
    seed: int = 42
    max_ticks: int = Field(default_factory=_mission_default_max_ticks, ge=1)
    tick_speed: float = Field(
        default_factory=_mission_default_tick_speed, gt=0.0
    )
    debug_auto_restart: bool = False
    debug_max_attempts: int = Field(default=15, ge=1, le=1000)
    debug_restart_delay_seconds: float = Field(
        default=2.0, ge=0.1, le=300.0
    )
    # Opt-in public challenge mode chooses from unfinished planets. Existing
    # manual/debug callers retain their explicitly requested planet.
    challenge_mode: bool = False
    # Narrative only: physics/actions remain deterministic and RL-authoritative.
    dialogue_generation_enabled: bool = False


class ControlRequest(BaseModel):
    action: str  # "pause", "resume", "stop", "speed"
    value: Optional[float] = None  # For speed: multiplier


# ============================================================
# DEBUG AUTO-RESTART SESSION
# ============================================================

def _get_challenge_coordinator() -> PlanetChallengeCoordinator:
    """Lazily create the persistent public challenge from mission config."""
    global _challenge
    with _lifecycle_lock:
        if _challenge is not None:
            return _challenge
        project_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", ".."
        ))
        planets_dir = os.path.join(project_root, "config", "planets")
        planet_ids = sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(planets_dir)
            if name.endswith(".json")
        )
        profile = MissionProfile.load()
        _challenge = PlanetChallengeCoordinator(
            planet_ids,
            config=ChallengeConfig(
                global_duration_seconds=(
                    profile.clock.public_challenge_days * 24 * 60 * 60
                ),
                completion_score=float(
                    profile.arrival_contract.get(
                        "required_colony_score_percent", 100.0
                    )
                ),
                realtime_seconds_per_tick=(
                    profile.clock.realtime_seconds_per_tick
                ),
                sim_minutes_per_tick=profile.clock.tick_minutes,
                episode_max_ticks=profile.clock.planet_attempt_max_ticks,
                stagnation_timeout_ticks=profile.data.get(
                    "simulation_clock", {}
                ).get("stagnation_timeout_ticks"),
            ),
            state_path=os.path.join(project_root, "data", "challenge_state.json"),
        )
        return _challenge


def _challenge_snapshot() -> Optional[dict]:
    with _lifecycle_lock:
        if _challenge is None:
            return None
        return make_safe_serializable(_challenge.to_dict())

def _debug_session_snapshot() -> dict:
    """Return a JSON-safe view of the active debug session.

    The countdown is derived when the snapshot is requested so both REST and
    newly connected WebSocket clients see a current value rather than the
    value captured when the previous attempt ended.
    """
    with _lifecycle_lock:
        pending = bool(_debug_session.get("restart_pending"))
        restart_at = _debug_session.get("restart_at")
        restart_in = None
        if pending and restart_at is not None:
            restart_in = round(max(0.0, float(restart_at) - time.time()), 1)
        history = [
            {
                **entry,
                "deaths": [dict(death) for death in entry.get("deaths", [])],
            }
            for entry in _debug_session.get("history", [])
        ]
        return {
            "enabled": bool(_debug_session.get("enabled", False)),
            "attempt": int(_debug_session.get("attempt", 0)),
            "max_attempts": int(_debug_session.get("max_attempts", 15)),
            "limit_reached": bool(_debug_session.get("limit_reached", False)),
            "restart_pending": pending,
            "restart_in_seconds": restart_in,
            "restart_at": restart_at if pending else None,
            "restart_delay_seconds": float(
                _debug_session.get("restart_delay_seconds", 2.0)
            ),
            "planet": _debug_session.get("planet"),
            "seed": _debug_session.get("seed"),
            "active_seed": _debug_session.get("active_seed"),
            "max_ticks": _debug_session.get("max_ticks"),
            "tick_speed": _debug_session.get("tick_speed"),
            "challenge_mode": bool(
                _debug_session.get("challenge_mode", False)
            ),
            "dialogue_generation_enabled": bool(
                _debug_session.get("dialogue_generation_enabled", False)
            ),
            "last_error": _debug_session.get("last_error"),
            "history": history,
            "challenge": _challenge_snapshot(),
        }


def _emit_debug_status() -> None:
    """Queue one debug-session update for every live WebSocket client."""
    global _tick_buffer
    item = {
        "type": "debug_status",
        "data": _debug_session_snapshot(),
    }
    with _lifecycle_lock:
        _tick_buffer.append(make_safe_serializable(item))
        if len(_tick_buffer) > 200:
            _tick_buffer = _tick_buffer[-100:]


def _configure_debug_session(req: SimulationStartRequest) -> int:
    """Replace any prior loop and return the new internal session token."""
    global _debug_session_token, _debug_restart_cancel, _debug_session
    max_attempts = int(req.debug_max_attempts)
    fields_set = getattr(req, "model_fields_set", None)
    if fields_set is None:  # Pydantic v1 compatibility
        fields_set = getattr(req, "__fields_set__", set())
    if req.challenge_mode and "debug_max_attempts" not in fields_set:
        profile = MissionProfile.load()
        attempt_ticks = profile.clock.planet_attempt_max_ticks
        if attempt_ticks:
            public_seconds = profile.clock.public_challenge_days * 24 * 60 * 60
            attempt_seconds = (
                attempt_ticks * profile.clock.realtime_seconds_per_tick
            )
            max_attempts = max(1, int(public_seconds // attempt_seconds))
    with _lifecycle_lock:
        _debug_restart_cancel.set()
        _debug_session_token += 1
        _debug_restart_cancel = threading.Event()
        _debug_session = {
            "enabled": bool(req.debug_auto_restart),
            "attempt": 0,
            "max_attempts": max_attempts,
            "limit_reached": False,
            "restart_pending": False,
            "restart_at": None,
            "restart_delay_seconds": float(
                req.debug_restart_delay_seconds
            ),
            "planet": req.planet,
            "seed": int(req.seed),
            "active_seed": None,
            "max_ticks": int(req.max_ticks),
            "tick_speed": float(req.tick_speed),
            "history": [],
            "challenge_mode": bool(req.challenge_mode),
            "dialogue_generation_enabled": bool(
                req.dialogue_generation_enabled
            ),
            "last_error": None,
        }
        return _debug_session_token


def _disable_debug_loop(*, emit: bool = True) -> None:
    """Cancel a pending restart without discarding completed attempt history."""
    with _lifecycle_lock:
        _debug_restart_cancel.set()
        _debug_session["enabled"] = False
        _debug_session["restart_pending"] = False
        _debug_session["restart_at"] = None
    if emit:
        _emit_debug_status()


def _normalize_deaths(report: dict) -> list[dict]:
    """Normalize engine death records for the stable debug API contract."""
    normalized = []
    for death in report.get("deaths", []) or []:
        cause = death.get("cause")
        if hasattr(cause, "value"):
            cause = cause.value
        elif isinstance(cause, str) and cause.startswith("DeathCause."):
            cause = cause.split(".", 1)[1].lower()
        normalized.append({
            "agent": death.get("agent") or death.get("name"),
            "cause": cause,
            "tick": death.get("tick"),
        })
    return normalized


def _state_snapshot_with_debug(engine: SimulationEngine) -> dict:
    state = engine._get_state_snapshot()
    state["debug_session"] = _debug_session_snapshot()
    return state


# ============================================================
# APP LIFECYCLE
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown handlers."""
    global _db
    _db = SimulationDB()
    logger.info("Server started, database connected")
    yield
    # Shutdown
    _disable_debug_loop(emit=False)
    if _engine:
        _engine.stop()
    if _engine_thread and _engine_thread.is_alive():
        await asyncio.to_thread(_engine_thread.join, 10.0)
    if _db:
        _db.close()
    logger.info("Server stopped")


app = FastAPI(
    title="Exoplanet Survival Simulation",
    description="AI-driven multi-agent colony simulation on exoplanets",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# TICK CALLBACK (runs in engine thread)
# ============================================================

def make_safe_serializable(obj):
    """Recursively convert Enums, NumPy types, sets, etc. into JSON-serializable primitives."""
    if isinstance(obj, dict):
        return {str(k): make_safe_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_safe_serializable(v) for v in obj]
    elif isinstance(obj, set):
        return [make_safe_serializable(v) for v in list(obj)]
    elif hasattr(obj, 'value'):  # Enums
        return obj.value
    elif hasattr(obj, 'item'):  # NumPy scalars
        return obj.item()
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def on_tick_callback(tick: int, state: dict):
    """Called by engine on each tick. Buffers state for WebSocket broadcast."""
    global _tick_buffer
    
    # Filter out ticks from duplicate/orphaned background engine threads
    if _engine is None or state.get("engine_id") != _engine.engine_id:
        return
    
    compact = {
        "tick": tick,
        "agents": [
            {
                "id": a["id"],
                "name": a["name"],
                "status": getattr(a.get("status"), "value", str(a.get("status"))),
                "death_cause": getattr(a.get("death_cause"), "value", str(a.get("death_cause"))) if a.get("death_cause") else None,
                "death_tick": a.get("death_tick"),
                "pos": [a["position"]["x"], a["position"]["y"]],
                "position": dict(a.get("position", {})),
                "anthropometrics": dict(a.get("anthropometrics", {})),
                "life_support_profile": dict(a.get("life_support_profile", {})),
                "needs": {
                    k: round(float(v), 1) for k, v in a["needs"].items()
                } if isinstance(a.get("needs"), dict) else {},
                "action": a.get("action", {}) if isinstance(a.get("action"), dict) else {"action_type": str(a.get("action", "idle"))},
                "inventory": a.get("inventory", {}),
                "plss": dict(a.get("plss", {})),
                "in_habitat": bool(a.get("in_habitat", False)),
                "last_decision": a.get("last_decision", {}),
                "rl_policy": a.get("rl_policy", {}),
                "rl_reward": round(float(a.get("rl_reward_accumulated", 0.0)), 2),
                "q_states": a.get("q_table_states", 0),
            }
            for a in state.get("agents", [])
        ],
        "colony": state.get("colony_score", {}),
        "structures": state.get("structures", {}),
        "placed_structures": state.get("placed_structures", []),
        "structure_health": state.get("structure_health", {}),
        "day_night": state.get("day_night", {}),
        "colony_production": state.get("colony_production", {}),
        "landing_zone": state.get("landing_zone", {"x": 1000, "y": 1000}),
        "discovered_resources": {k: list(v) if isinstance(v, set) else v for k, v in state.get("discovered_resources", {}).items()},
        "cell_geology": state.get("cell_geology", []),
        "spoil_piles": state.get("spoil_piles", []),
        "surface_fleet": state.get("surface_fleet", {}),
        "debug_session": _debug_session_snapshot(),
    }
    _tick_buffer.append(make_safe_serializable(compact))
    if len(_tick_buffer) > 200:
        _tick_buffer = _tick_buffer[-100:]
    
    # DB checkpoint every 10 ticks
    if _engine and _db and tick % 10 == 0:
        try:
            checkpoint_actions = []
            for agent_state in state.get("agents", []):
                action_state = agent_state.get("action", {})
                decision = agent_state.get("last_decision", {})
                checkpoint_actions.append({
                    "agent_id": agent_state.get("id", "unknown"),
                    "action": action_state.get(
                        "action_type", decision.get("action", "idle")
                    ),
                    "target": action_state.get(
                        "target", decision.get("target", {})
                    ),
                    "reasoning": decision.get("reasoning", ""),
                    "is_fallback": bool(decision.get("fallback", False)),
                    "is_llm": decision.get("source") == "language_model",
                    "tokens": 0,
                })
            checkpoint_events = list(_telemetry_event_buffer)
            _telemetry_event_buffer.clear()
            _db.checkpoint(
                tick, _engine.agents,
                _engine.colony_score.to_dict(),
                events=checkpoint_events,
                actions=checkpoint_actions,
            )
        except Exception as e:
            logger.error(f"Checkpoint error: {e}")


def on_event_callback(event: dict):
    """Called by engine on notable events."""
    global _tick_buffer, _telemetry_event_buffer
    _telemetry_event_buffer.append(dict(event))
    compact = {
        "type": "event",
        "data": event,
    }
    _tick_buffer.append(compact)
    if len(_tick_buffer) > 200:
        _tick_buffer = _tick_buffer[-100:]


def _finish_debug_attempt(
    session_token: int,
    attempt: int,
    report: dict,
    run_id: Optional[int],
    *,
    planet: Optional[str] = None,
) -> None:
    """Record a valid attempt and restart after death or time exhaustion."""
    global _debug_restart_thread
    should_restart = False
    with _lifecycle_lock:
        if session_token != _debug_session_token:
            return
        end_reason = str(report.get("end_reason", "unknown"))
        consumed_attempt = end_reason != "engine_error"
        entry = {
            "attempt": attempt,
            "run_id": run_id,
            "planet": planet or _debug_session.get("planet"),
            "seed": _debug_session.get("active_seed"),
            "end_reason": end_reason,
            "total_ticks": int(report.get("total_ticks", 0) or 0),
            "deaths": _normalize_deaths(report),
            "ended_at": time.time(),
            "consumed_attempt": consumed_attempt,
        }
        if report.get("error"):
            entry["error"] = str(report["error"])
        _debug_session["history"].append(entry)

        # A software/configuration failure is not a simulated outcome. Keep it
        # visible, stop the loop, and return its attempt budget to the session.
        if not consumed_attempt:
            _debug_session["attempt"] = max(0, attempt - 1)
            _debug_session["last_error"] = entry.get(
                "error", "Simulation engine failed"
            )
            _debug_session["enabled"] = False
            _debug_session["limit_reached"] = False
            _debug_session["restart_pending"] = False
            _debug_session["restart_at"] = None
            cancel_event = _debug_restart_cancel
        else:
            _debug_session["last_error"] = None

            challenge_can_continue = True
            challenge_terminal = end_reason in {
                "all_agents_dead", "timeout", "colony_ready"
            }
            if _debug_session.get("challenge_mode") and challenge_terminal:
                try:
                    coordinator = _get_challenge_coordinator()
                    score_payload = report.get("colony_score", {}) or {}
                    score = (
                        float(score_payload.get("overall", 0.0))
                        if isinstance(score_payload, dict)
                        else float(score_payload or 0.0)
                    )
                    challenge_result = coordinator.finish_active_attempt(score)
                    entry["challenge_result"] = challenge_result
                    challenge_can_continue = (
                        challenge_result.get("challenge_status") == "running"
                    )
                except Exception as exc:
                    logger.exception("Could not advance public challenge")
                    _debug_session["last_error"] = str(exc)
                    challenge_can_continue = False
            elif _debug_session.get("challenge_mode"):
                # Manual stops and unknown terminations do not advance the
                # persisted wheel or consume its per-planet attempt count.
                challenge_can_continue = False

            max_attempts = int(_debug_session.get("max_attempts", 15))
            limit_reached = attempt >= max_attempts
            restart_reasons = {"all_agents_dead", "timeout"}
            if _debug_session.get("challenge_mode"):
                # A successful planet leaves the wheel; continue with the
                # unfinished set while the global challenge remains active.
                restart_reasons.add("colony_ready")
            should_restart = bool(
                _debug_session.get("enabled")
                and end_reason in restart_reasons
                and challenge_can_continue
                and not limit_reached
            )
            _debug_session["limit_reached"] = bool(
                limit_reached and end_reason in restart_reasons
            )
            if _debug_session["limit_reached"] or (
                end_reason == "colony_ready"
                and not _debug_session.get("challenge_mode")
            ) or not challenge_can_continue:
                _debug_session["enabled"] = False
            _debug_session["restart_pending"] = should_restart
            _debug_session["restart_at"] = (
                time.time() + float(_debug_session["restart_delay_seconds"])
                if should_restart else None
            )
            cancel_event = _debug_restart_cancel

    _emit_debug_status()
    if not should_restart:
        return

    def countdown_and_restart():
        last_second = None
        while True:
            with _lifecycle_lock:
                if (
                    session_token != _debug_session_token
                    or not _debug_session.get("enabled")
                    or not _debug_session.get("restart_pending")
                ):
                    return
                remaining = max(
                    0.0,
                    float(_debug_session["restart_at"]) - time.time(),
                )
            current_second = int(math.ceil(remaining))
            if current_second != last_second:
                last_second = current_second
                _emit_debug_status()
            if remaining <= 0:
                with _lifecycle_lock:
                    if (
                        session_token != _debug_session_token
                        or not _debug_session.get("enabled")
                    ):
                        return
                    _debug_session["restart_pending"] = False
                    _debug_session["restart_at"] = None
                _launch_simulation_attempt(session_token, is_restart=True)
                return
            if cancel_event.wait(timeout=min(0.25, remaining)):
                return

    _debug_restart_thread = threading.Thread(
        target=countdown_and_restart,
        name="debug-simulation-restart",
        daemon=True,
    )
    _debug_restart_thread.start()


def _launch_simulation_attempt(
    session_token: int, *, is_restart: bool = False
) -> Optional[dict]:
    """Create one clean engine while retaining planet-scoped DB policies."""
    global _engine, _engine_thread, _tick_buffer, _telemetry_event_buffer
    with _lifecycle_lock:
        if session_token != _debug_session_token:
            return None
        if is_restart and not _debug_session.get("enabled"):
            return None
        if is_restart and int(_debug_session.get("attempt", 0)) >= int(
            _debug_session.get("max_attempts", 15)
        ):
            _debug_session["enabled"] = False
            _debug_session["limit_reached"] = True
            return None
        if _debug_session.get("challenge_mode"):
            coordinator = _get_challenge_coordinator()
            coordinator.start()
            selected_planet = coordinator.spin_next_planet()
            if selected_planet is None:
                _debug_session["enabled"] = False
                _debug_session["restart_pending"] = False
                _debug_session["last_error"] = (
                    "Public challenge is complete or its deadline has expired"
                )
                return None
            _debug_session["planet"] = selected_planet
        _debug_session["attempt"] += 1
        attempt = int(_debug_session["attempt"])
        _debug_session["restart_pending"] = False
        _debug_session["restart_at"] = None
        planet = str(_debug_session["planet"])
        # Use a recorded but distinct world/event realization for each valid
        # episode. Planet-scoped Q-tables persist, while the policy cannot win
        # by memorising one fixed geology/event seed.
        seed = int(_debug_session["seed"]) + attempt - 1
        _debug_session["active_seed"] = seed
        max_ticks = int(_debug_session["max_ticks"])
        tick_speed = float(_debug_session["tick_speed"])

        config_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "config"
        )
        planet_path = os.path.join(config_dir, "planets", f"{planet}.json")
        _tick_buffer = []
        _telemetry_event_buffer = []
        engine = SimulationEngine(
            planet_config_path=planet_path,
            seed=seed,
            max_ticks=max_ticks,
            tick_speed=tick_speed,
            on_tick=on_tick_callback,
            on_event=on_event_callback,
            db=_db,
        )
        engine.decision_engine.dialogue_generation_enabled = bool(
            _debug_session.get("dialogue_generation_enabled", False)
        )
        preset_path = os.path.join(config_dir, "agent_presets.json")
        if os.path.exists(preset_path):
            from src.agents.agent import create_team_from_presets
            for agent in create_team_from_presets(preset_path):
                engine.add_agent(agent)

        run_id = None
        if _db:
            run_id = _db.start_run(
                planet=planet,
                seed=seed,
                max_ticks=max_ticks,
                agents_count=len(engine.agents),
                config={
                    "debug_auto_restart": bool(
                        _debug_session.get("enabled")
                    ),
                    "debug_attempt": attempt,
                    "debug_max_attempts": int(
                        _debug_session.get("max_attempts", 15)
                    ),
                    "challenge_mode": bool(
                        _debug_session.get("challenge_mode", False)
                    ),
                    "dialogue_generation_enabled": bool(
                        _debug_session.get(
                            "dialogue_generation_enabled", False
                        )
                    ),
                },
            )
        _engine = engine

        def run_engine():
            try:
                report = engine.run()
            except Exception as exc:
                logger.exception("Engine error")
                engine.running = False
                engine.end_reason = "engine_error"
                report = {
                    "end_reason": "engine_error",
                    "total_ticks": getattr(engine, "current_tick", 0),
                    "deaths": [],
                    "error": str(exc),
                }
            if _db:
                try:
                    _db.end_run(
                        end_reason=report.get("end_reason", "unknown"),
                        total_ticks=report.get("total_ticks", 0),
                        final_score=report.get("colony_score", {}).get(
                            "overall", 0
                        ),
                        result=report,
                        run_id=run_id,
                    )
                except Exception:
                    logger.exception("Could not close simulation DB run")
            _finish_debug_attempt(
                session_token, attempt, report, run_id, planet=planet
            )

        _engine_thread = threading.Thread(
            target=run_engine,
            name=f"simulation-attempt-{attempt}",
            daemon=True,
        )
        _engine_thread.start()

    _emit_debug_status()
    return {
        "planet": planet,
        "seed": seed,
        "agents": len(engine.agents),
        "max_ticks": max_ticks,
        "attempt": attempt,
        "run_id": run_id,
        "dialogue_generation_enabled": bool(
            _debug_session.get("dialogue_generation_enabled", False)
        ),
        "challenge": _challenge_snapshot(),
    }


# ============================================================
# SIMULATION ENDPOINTS
# ============================================================

@app.post("/api/simulation/start")
async def start_simulation(req: SimulationStartRequest):
    """Start a new simulation."""
    config_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'config')
    planet_path = os.path.join(config_dir, 'planets', f'{req.planet}.json')
    if not req.challenge_mode and not os.path.exists(planet_path):
        available = [
            f.replace('.json', '')
            for f in os.listdir(os.path.join(config_dir, 'planets'))
            if f.endswith('.json')
        ]
        raise HTTPException(404, f"Planet not found. Available: {available}")

    # Invalidate/cancel the old loop before its worker can launch another
    # attempt. Wait for its final DB flush before assigning a new active run.
    session_token = _configure_debug_session(req)
    previous_engine = _engine
    previous_thread = _engine_thread
    if previous_engine and previous_engine.running:
        previous_engine.stop()
        logger.info("Stopped previous simulation to launch new one.")
    if (
        previous_thread
        and previous_thread.is_alive()
        and previous_thread is not threading.current_thread()
    ):
        await asyncio.to_thread(previous_thread.join, 10.0)
        if previous_thread.is_alive():
            _disable_debug_loop()
            raise HTTPException(
                409, "Previous simulation is still shutting down"
            )

    launched = _launch_simulation_attempt(session_token)
    if not launched:
        raise HTTPException(409, "Simulation launch was cancelled")
    return {
        "status": "started",
        **launched,
        "debug_session": _debug_session_snapshot(),
    }


@app.get("/api/simulation/state")
async def get_state():
    """Get current simulation state."""
    if not _engine:
        raise HTTPException(404, "No simulation running")
    
    return _state_snapshot_with_debug(_engine)


@app.get("/api/simulation/status")
async def get_status():
    """Get engine status."""
    if not _engine:
        return {
            "status": "idle",
            "running": False,
            "debug_session": _debug_session_snapshot(),
        }
    
    return {
        "status": "running" if _engine.running else "stopped",
        "running": _engine.running,
        "paused": _engine.paused,
        "tick": _engine.current_tick,
        "max_ticks": _engine.max_ticks,
        "end_reason": getattr(_engine, 'end_reason', getattr(_engine, '_end_reason', None)),
        "planet": _engine.planet.name if _engine.planet else None,
        "agents_alive": sum(
            1 for a in _engine.agents
            if getattr(a.status, 'value', str(a.status)) == "alive"
        ),
        "colony_score": _engine.colony_score.get_overall_score(),
        "llm": _engine.llm_client.get_status(),
        "debug_session": _debug_session_snapshot(),
    }


@app.post("/api/simulation/control")
async def control_simulation(req: ControlRequest):
    """Control running simulation."""
    if not _engine:
        if req.action == "stop":
            _disable_debug_loop()
            return {
                "status": "idle",
                "debug_session": _debug_session_snapshot(),
            }
        raise HTTPException(400, "No simulation running")
    
    if req.action == "pause":
        _engine.pause()
        return {"status": "paused"}
    elif req.action == "resume":
        _engine.resume()
        return {"status": "resumed"}
    elif req.action == "stop":
        _disable_debug_loop()
        _engine.stop()
        return {
            "status": "stopped",
            "debug_session": _debug_session_snapshot(),
        }
    elif req.action == "speed":
        if req.value:
            _engine.set_speed(req.value)
            return {"status": "speed_changed", "multiplier": req.value}
    
    raise HTTPException(400, f"Unknown action: {req.action}")


@app.post("/api/debug/reset-rl")
async def reset_rl_policies():
    """Stop the active attempt and reset learning without deleting audits."""
    if not _db:
        raise HTTPException(503, "Database not available")
    _disable_debug_loop()
    active_engine = _engine
    active_thread = _engine_thread
    if active_engine and active_engine.running:
        active_engine.stop()
    if (
        active_thread
        and active_thread.is_alive()
        and active_thread is not threading.current_thread()
    ):
        await asyncio.to_thread(active_thread.join, 10.0)
        if active_thread.is_alive():
            raise HTTPException(409, "Simulation is still shutting down")
    deleted = _db.reset_rl_policies()
    return {"status": "reset", **deleted}


@app.get("/api/challenge/status")
async def get_challenge_status():
    """Return the persisted public wheel/deadline state without starting it."""
    coordinator = _get_challenge_coordinator()
    return make_safe_serializable(coordinator.to_dict())


# ============================================================
# HISTORY ENDPOINTS
# ============================================================

@app.get("/api/history")
async def get_history():
    """Get list of past simulation runs."""
    if not _db:
        return {"runs": []}
    return {"runs": _db.get_runs()}


@app.get("/api/history/{run_id}")
async def get_run_detail(run_id: int):
    """Get detailed info about a past run."""
    if not _db:
        raise HTTPException(404, "Database not available")
    
    detail = _db.get_run_detail(run_id)
    if not detail:
        raise HTTPException(404, f"Run #{run_id} not found")
    
    detail["score_history"] = _db.get_score_history(run_id)
    detail["deaths"] = _db.get_deaths(run_id)
    return detail


@app.get("/api/simulation/analytics")
async def get_simulation_analytics(run_id: Optional[int] = None):
    """Serve live and historical metrics for the Simulation Analytics tab."""
    if not _db:
        raise HTTPException(404, "Database not available")
    data = _db.get_analytics_data(run_id)
    if _engine and _engine.running:
        data["live_vitals"] = [a.to_telemetry_dict() for a in _engine.agents]
        data["live_scores"] = _engine.colony_score.to_dict()
        data["live_resources"] = getattr(_engine, "_colony_resources", {})
        data["live_tick"] = _engine.current_tick
        data["placed_structures"] = getattr(_engine, "placed_structures", [])
    data["debug_session"] = _debug_session_snapshot()
    return JSONResponse(content=make_safe_serializable(data))


# ============================================================
# WEBSOCKET (real-time tick streaming)
# ============================================================

@app.websocket("/ws/live")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket for real-time simulation updates."""
    await ws.accept()
    _ws_clients.append(ws)
    logger.info(f"WebSocket client connected ({len(_ws_clients)} total)")
    
    try:
        last_engine_id = getattr(_engine, 'engine_id', None) if _engine else None
        end_sent_engine_id = None
        
        # Send current state on connect
        if _engine:
            await ws.send_json({
                "type": "init",
                "data": _state_snapshot_with_debug(_engine),
            })
        
        # Stream tick updates persistently from the current point onwards
        last_sent_idx = len(_tick_buffer)
        while True:
            cur_engine_id = getattr(_engine, 'engine_id', None) if _engine else None
            
            # Detect fresh simulation launch
            if cur_engine_id != last_engine_id:
                last_engine_id = cur_engine_id
                last_sent_idx = 0
                end_sent_engine_id = None
                if _engine:
                    await ws.send_json({
                        "type": "init",
                        "data": _state_snapshot_with_debug(_engine),
                    })
                
            # If tick buffer was cleared/reset
            if last_sent_idx > len(_tick_buffer):
                last_sent_idx = len(_tick_buffer)
                
            # Check for new tick data
            if last_sent_idx < len(_tick_buffer):
                for item in _tick_buffer[last_sent_idx:]:
                    try:
                        await ws.send_json(item)
                    except Exception:
                        break
                last_sent_idx = len(_tick_buffer)
            
            # Check if simulation ended (send end event once per engine run)
            if _engine and not _engine.running and _engine.end_reason and end_sent_engine_id != cur_engine_id:
                end_sent_engine_id = cur_engine_id
                try:
                    final_report = _engine._get_final_report()
                    final_report["debug_session"] = _debug_session_snapshot()
                    await ws.send_json({
                        "type": "end",
                        "data": final_report,
                    })
                except Exception:
                    pass
            
            await asyncio.sleep(0.08)  # 80ms poll interval
            
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)
        logger.info(f"WebSocket client disconnected ({len(_ws_clients)} total)")


# ============================================================
# STATIC FILES (frontend)
# ============================================================

# Serve frontend directory
frontend_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend')
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="frontend")


@app.get("/")
async def serve_index():
    """Serve the main dashboard page."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({
        "message": "Exoplanet Survival Simulation API",
        "docs": "/docs",
        "endpoints": [
            "POST /api/simulation/start",
            "GET /api/simulation/state",
            "GET /api/simulation/status",
            "POST /api/simulation/control",
            "GET /api/challenge/status",
            "GET /api/history",
            "WS /ws/live",
        ]
    })


# ============================================================
# CLI RUNNER
# ============================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the server with uvicorn."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
