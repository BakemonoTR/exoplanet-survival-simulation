"""
SQLite Persistence Layer.

Stores simulation state for:
- Replay and analysis after simulation ends
- Checkpoint/restore (every 10 ticks)
- Post-sim comparison across runs

Tables:
- simulation_runs: Run metadata (planet, seed, result)
- agent_snapshots: Per-tick agent state
- world_events: Events that occurred
- colony_scores: Score history
- agent_actions: Action log for analysis
- agent_memories: Persisted memories
"""

import sqlite3
import json
import os
import time
import logging
import hashlib
import zlib
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SimulationDB:
    """
    SQLite database for simulation persistence.
    
    All writes are batched and committed at checkpoint intervals
    to minimize I/O overhead during tick loop.
    """

    # Reward/Q-state telemetry stays useful at this cadence without storing a
    # complete, increasingly large JSON policy every 25 simulation ticks.
    Q_TABLE_PROGRESS_INTERVAL_TICKS = 250
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'data', 'simulations.db'
            )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._current_run_id: Optional[int] = None
        self._pending_writes: list[tuple] = []  # Batched inserts
        self._pending_q_tables: dict[tuple[int, str], tuple] = {}
        self._last_q_table_hashes: dict[tuple[int, str], Optional[str]] = {}
        self._last_q_progress_tick: dict[tuple[int, str], int] = {}
        
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist."""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent perf
        self._conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS simulation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                planet TEXT NOT NULL,
                seed INTEGER NOT NULL,
                max_ticks INTEGER NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                end_reason TEXT,
                total_ticks INTEGER DEFAULT 0,
                agents_count INTEGER DEFAULT 5,
                final_colony_score REAL DEFAULT 0,
                config_json TEXT,
                result_json TEXT
            );
            
            CREATE TABLE IF NOT EXISTS agent_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                tick INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                agent_name TEXT,
                status TEXT,
                x INTEGER, y INTEGER,
                hunger REAL, thirst REAL, energy REAL,
                hygiene REAL, o2_supply REAL, temperature_stress REAL,
                morale REAL,
                injury_level REAL,
                suit_equipped INTEGER DEFAULT 0,
                suit_integrity REAL DEFAULT 1.0,
                action_type TEXT,
                strategic_goal TEXT,
                FOREIGN KEY (run_id) REFERENCES simulation_runs(id)
            );
            
            CREATE TABLE IF NOT EXISTS world_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                tick INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                description TEXT,
                severity TEXT,
                data_json TEXT,
                FOREIGN KEY (run_id) REFERENCES simulation_runs(id)
            );
            
            CREATE TABLE IF NOT EXISTS colony_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                tick INTEGER NOT NULL,
                overall_score REAL NOT NULL,
                o2_score REAL, water_score REAL,
                food_score REAL, shelter_score REAL,
                energy_score REAL, hazard_score REAL,
                FOREIGN KEY (run_id) REFERENCES simulation_runs(id)
            );
            
            CREATE TABLE IF NOT EXISTS agent_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                tick INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                target_json TEXT,
                reasoning TEXT,
                is_fallback INTEGER DEFAULT 0,
                is_llm INTEGER DEFAULT 0,
                tokens_used INTEGER DEFAULT 0,
                FOREIGN KEY (run_id) REFERENCES simulation_runs(id)
            );
            
            CREATE TABLE IF NOT EXISTS agent_deaths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                tick INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                agent_name TEXT,
                death_cause TEXT,
                ticks_alive INTEGER,
                FOREIGN KEY (run_id) REFERENCES simulation_runs(id)
            );
            
            CREATE TABLE IF NOT EXISTS agent_q_tables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                agent_id TEXT NOT NULL,
                q_table_json TEXT NOT NULL,
                total_reward REAL DEFAULT 0,
                updated_tick INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                q_table_hash TEXT,
                is_latest INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS agent_q_table_progress (
                run_id INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                total_reward REAL DEFAULT 0,
                updated_tick INTEGER NOT NULL,
                q_state_count INTEGER DEFAULT 0,
                q_table_hash TEXT,
                timestamp REAL NOT NULL,
                PRIMARY KEY (run_id, agent_id, updated_tick)
            );

            CREATE TABLE IF NOT EXISTS tick_telemetry (
                run_id INTEGER NOT NULL,
                tick INTEGER NOT NULL,
                payload BLOB NOT NULL,
                PRIMARY KEY (run_id, tick)
            );

            -- Indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_snapshots_run_tick 
                ON agent_snapshots(run_id, tick);
            CREATE INDEX IF NOT EXISTS idx_actions_run_tick 
                ON agent_actions(run_id, tick);
            CREATE INDEX IF NOT EXISTS idx_scores_run 
                ON colony_scores(run_id, tick);
            CREATE INDEX IF NOT EXISTS idx_qtables_agent 
                ON agent_q_tables(agent_id, updated_tick);
            CREATE INDEX IF NOT EXISTS idx_qprogress_run_tick
                ON agent_q_table_progress(run_id, updated_tick);
            CREATE INDEX IF NOT EXISTS idx_tick_telemetry_run_tick
                ON tick_telemetry(run_id, tick);
        """)

        # In-place migration for databases created before compact Q-policy
        # persistence existed. Historical rows remain readable and untouched;
        # all new writes use one upserted full snapshot per run/agent.
        q_columns = {
            row[1] for row in self._conn.execute(
                "PRAGMA table_info(agent_q_tables)"
            ).fetchall()
        }
        if "q_table_hash" not in q_columns:
            self._conn.execute(
                "ALTER TABLE agent_q_tables ADD COLUMN q_table_hash TEXT"
            )
        if "is_latest" not in q_columns:
            self._conn.execute(
                "ALTER TABLE agent_q_tables "
                "ADD COLUMN is_latest INTEGER NOT NULL DEFAULT 0"
            )
        self._conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS idx_qtables_latest
               ON agent_q_tables(run_id, agent_id) WHERE is_latest=1"""
        )
        self._conn.commit()
        logger.info(f"Database initialized: {self._db_path}")
    
    # ================================================================
    # RUN MANAGEMENT
    # ================================================================
    
    def start_run(self, planet: str, seed: int, max_ticks: int,
                  agents_count: int = 5, config: dict = None) -> int:
        """Start a new simulation run. Returns run_id."""
        cursor = self._conn.execute(
            """INSERT INTO simulation_runs 
               (planet, seed, max_ticks, start_time, agents_count, config_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (planet, seed, max_ticks, time.time(), agents_count,
             json.dumps(config) if config else None)
        )
        self._conn.commit()
        self._current_run_id = cursor.lastrowid
        logger.info(f"Started run #{self._current_run_id}: {planet} (seed={seed})")
        return self._current_run_id
    
    def end_run(self, end_reason: str, total_ticks: int,
                final_score: float, result: dict = None,
                run_id: int = None):
        """Mark current run as completed."""
        target_run_id = run_id if run_id is not None else self._current_run_id
        if target_run_id is None:
            return
        
        self._flush_pending()
        self._conn.execute(
            """UPDATE simulation_runs 
               SET end_time=?, end_reason=?, total_ticks=?, 
                   final_colony_score=?, result_json=?
               WHERE id=?""",
            (time.time(), end_reason, total_ticks, final_score,
             json.dumps(result) if result else None,
             target_run_id)
        )
        self._conn.commit()
        logger.info(f"Run #{target_run_id} ended: {end_reason}")
    
    # ================================================================
    # CHECKPOINT (batched writes)
    # ================================================================
    
    def checkpoint(self, tick: int, agents: list, colony_score: dict,
                   events: list = None, actions: list = None):
        """
        Save simulation state at checkpoint tick.
        
        This is the main persistence method, called every N ticks.
        All data is batched and committed in one transaction.
        """
        if self._current_run_id is None:
            return
        
        run_id = self._current_run_id
        
        # Agent snapshots
        for agent in agents:
            needs = agent.needs
            self._pending_writes.append((
                'agent_snapshots',
                (run_id, tick, agent.id, agent.name,
                 agent.status.value, agent.x, agent.y,
                 needs.hunger, needs.thirst, needs.energy,
                 needs.hygiene, needs.o2_supply, needs.temperature_stress,
                 agent.morale, agent.injury_level,
                 1 if agent.suit_equipped else 0,
                 agent.suit_integrity,
                 agent.action.action_type if agent.action else None,
                 json.dumps(getattr(agent, 'strategic_goal', None)))
            ))
        
        # Colony score
        scores = colony_score.get("categories", {})
        self._pending_writes.append((
            'colony_scores',
            (run_id, tick, colony_score.get("overall", 0),
             scores.get("o2", 0), scores.get("water", 0),
             scores.get("food", 0), scores.get("shelter", 0),
             scores.get("energy", 0), scores.get("hazard_protection", 0))
        ))
        
        # World events
        if events:
            for event in events:
                self._pending_writes.append((
                    'world_events',
                    (run_id, tick, event.get("type", "unknown"),
                     event.get("description", ""),
                     event.get("severity", "normal"),
                     json.dumps(event))
                ))

        # Research telemetry: one auditable action sample per agent at each
        # checkpoint. This is compact enough for long runs while preserving
        # the policy, physical target and rationale trajectory.
        if actions:
            for action in actions:
                self._pending_writes.append((
                    'agent_actions',
                    (
                        run_id,
                        tick,
                        str(action.get("agent_id", "unknown")),
                        str(action.get("action", "idle")),
                        json.dumps(action.get("target") or {}),
                        str(action.get("reasoning", "")),
                        1 if action.get("is_fallback") else 0,
                        1 if action.get("is_llm") else 0,
                        int(action.get("tokens", 0) or 0),
                    )
                ))
        
        # Flush all pending writes
        self._flush_pending()
    
    def record_action(self, tick: int, agent_id: str, action: str,
                      target: dict = None, reasoning: str = None,
                      is_fallback: bool = False, is_llm: bool = False,
                      tokens: int = 0):
        """Record an agent action (called per-tick per-agent)."""
        if self._current_run_id is None:
            return
        
        self._pending_writes.append((
            'agent_actions',
            (self._current_run_id, tick, agent_id, action,
             json.dumps(target) if target else None,
             reasoning, 1 if is_fallback else 0,
             1 if is_llm else 0, tokens)
        ))
    
    def record_death(self, tick: int, agent_id: str, agent_name: str,
                     death_cause: str, ticks_alive: int):
        """Record agent death."""
        if self._current_run_id is None:
            return
        
        self._conn.execute(
            """INSERT INTO agent_deaths 
               (run_id, tick, agent_id, agent_name, death_cause, ticks_alive)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (self._current_run_id, tick, agent_id, agent_name,
             death_cause, ticks_alive)
        )
        self._conn.commit()
    
    def _flush_pending(self) -> bool:
        """Commit pending writes atomically and retain them after failure."""
        if not self._pending_writes and not self._pending_q_tables:
            return True
        
        insert_templates = {
            'tick_telemetry': """INSERT OR REPLACE INTO tick_telemetry
                (run_id, tick, payload) VALUES (?, ?, ?)""",
            'agent_snapshots': """INSERT INTO agent_snapshots 
                (run_id, tick, agent_id, agent_name, status, x, y,
                 hunger, thirst, energy, hygiene, o2_supply, temperature_stress,
                 morale, injury_level, suit_equipped, suit_integrity,
                 action_type, strategic_goal) 
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            'colony_scores': """INSERT INTO colony_scores 
                (run_id, tick, overall_score, o2_score, water_score,
                 food_score, shelter_score, energy_score, hazard_score)
                VALUES (?,?,?,?,?,?,?,?,?)""",
            'world_events': """INSERT INTO world_events 
                (run_id, tick, event_type, description, severity, data_json)
                VALUES (?,?,?,?,?,?)""",
            'agent_actions': """INSERT INTO agent_actions 
                (run_id, tick, agent_id, action_type, target_json,
                 reasoning, is_fallback, is_llm, tokens_used)
                VALUES (?,?,?,?,?,?,?,?,?)""",
        }
        
        q_hash_updates = {}
        progress_tick_updates = {}
        try:
            with self._conn:
                for table, data in self._pending_writes:
                    template = insert_templates.get(table)
                    if template:
                        self._conn.execute(template, data)
                for key, data in self._pending_q_tables.items():
                    (run_id, agent_id, q_json, q_hash, total_reward,
                     tick, timestamp, q_state_count) = data
                    previous_hash = self._last_q_table_hashes.get(key)
                    if key not in self._last_q_table_hashes:
                        row = self._conn.execute(
                            """SELECT q_table_hash FROM agent_q_tables
                               WHERE run_id=? AND agent_id=? AND is_latest=1
                               LIMIT 1""",
                            (run_id, agent_id),
                        ).fetchone()
                        previous_hash = row[0] if row else None

                    if previous_hash == q_hash:
                        # Do not rewrite the large JSON BLOB when learning did
                        # not change it; only advance lightweight metadata.
                        self._conn.execute(
                            """UPDATE agent_q_tables
                               SET total_reward=?, updated_tick=?, timestamp=?
                               WHERE run_id=? AND agent_id=? AND is_latest=1""",
                            (total_reward, tick, timestamp, run_id, agent_id),
                        )
                    else:
                        self._conn.execute(
                            """INSERT INTO agent_q_tables
                               (run_id, agent_id, q_table_json, total_reward,
                                updated_tick, timestamp, q_table_hash, is_latest)
                               VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                               ON CONFLICT(run_id, agent_id) WHERE is_latest=1
                               DO UPDATE SET
                                 q_table_json=excluded.q_table_json,
                                 total_reward=excluded.total_reward,
                                 updated_tick=excluded.updated_tick,
                                 timestamp=excluded.timestamp,
                                 q_table_hash=excluded.q_table_hash""",
                            (run_id, agent_id, q_json, total_reward, tick,
                             timestamp, q_hash),
                        )
                    q_hash_updates[key] = q_hash

                    last_progress = self._last_q_progress_tick.get(key)
                    if last_progress is None:
                        row = self._conn.execute(
                            """SELECT MAX(updated_tick)
                               FROM agent_q_table_progress
                               WHERE run_id=? AND agent_id=?""",
                            (run_id, agent_id),
                        ).fetchone()
                        last_progress = int(row[0]) if row and row[0] is not None else -10**9
                    if tick - last_progress >= self.Q_TABLE_PROGRESS_INTERVAL_TICKS:
                        self._conn.execute(
                            """INSERT OR REPLACE INTO agent_q_table_progress
                               (run_id, agent_id, total_reward, updated_tick,
                                q_state_count, q_table_hash, timestamp)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (run_id, agent_id, total_reward, tick,
                             q_state_count, q_hash, timestamp),
                        )
                        progress_tick_updates[key] = tick
            self._last_q_table_hashes.update(q_hash_updates)
            self._last_q_progress_tick.update(progress_tick_updates)
            self._pending_writes.clear()
            self._pending_q_tables.clear()
            return True
        except Exception as e:
            logger.error(f"Database flush error: {e}")
            # SQLite rolled the transaction back. Keep both queues and leave
            # deduplication caches untouched so a later retry writes exactly
            # the same policies instead of silently losing learned state.
            return False
    
    # ================================================================
    # QUERIES (for API and analysis)
    # ================================================================
    
    def get_runs(self, limit: int = 20) -> list[dict]:
        """Get list of simulation runs."""
        cursor = self._conn.execute(
            """SELECT id, planet, seed, total_ticks, end_reason, 
                      final_colony_score, start_time
               FROM simulation_runs ORDER BY id DESC LIMIT ?""",
            (limit,)
        )
        return [
            {"id": r[0], "planet": r[1], "seed": r[2], "ticks": r[3],
             "result": r[4], "score": r[5], "started": r[6]}
            for r in cursor.fetchall()
        ]
    
    def get_run_detail(self, run_id: int) -> Optional[dict]:
        """Get full run details including result JSON."""
        cursor = self._conn.execute(
            "SELECT * FROM simulation_runs WHERE id=?", (run_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            "id": row[0], "planet": row[1], "seed": row[2],
            "max_ticks": row[3], "start_time": row[4], "end_time": row[5],
            "end_reason": row[6], "total_ticks": row[7],
            "agents_count": row[8], "final_score": row[9],
            "result": json.loads(row[11]) if row[11] else None,
        }
    
    def get_score_history(self, run_id: int) -> list[dict]:
        """Get colony score progression for a run."""
        cursor = self._conn.execute(
            """SELECT tick, overall_score, o2_score, water_score,
                      food_score, shelter_score, energy_score, hazard_score
               FROM colony_scores WHERE run_id=? ORDER BY tick""",
            (run_id,)
        )
        return [
            {"tick": r[0], "overall": r[1],
             "o2": r[2], "water": r[3], "food": r[4],
             "shelter": r[5], "energy": r[6], "hazard": r[7]}
            for r in cursor.fetchall()
        ]
    
    def get_agent_timeline(self, run_id: int, agent_id: str) -> list[dict]:
        """Get agent state over time."""
        cursor = self._conn.execute(
            """SELECT tick, status, hunger, thirst, energy, morale, 
                      injury_level, action_type
               FROM agent_snapshots 
               WHERE run_id=? AND agent_id=? ORDER BY tick""",
            (run_id, agent_id)
        )
        return [
            {"tick": r[0], "status": r[1], "hunger": r[2], "thirst": r[3],
             "energy": r[4], "morale": r[5], "injury": r[6], "action": r[7]}
            for r in cursor.fetchall()
        ]
    
    def get_deaths(self, run_id: int) -> list[dict]:
        """Get all deaths in a run."""
        cursor = self._conn.execute(
            """SELECT tick, agent_name, death_cause, ticks_alive
               FROM agent_deaths WHERE run_id=? ORDER BY tick""",
            (run_id,)
        )
        return [
            {"tick": r[0], "agent": r[1], "cause": r[2], "ticks_alive": r[3]}
            for r in cursor.fetchall()
        ]

    # ================================================================
    # Q-TABLE PERSISTENCE & ANALYTICS
    # ================================================================

    def save_agent_q_table(self, agent_id: str, q_table: dict,
                           total_reward: float, tick: int, run_id: int = None):
        """Queue one trained policy for the next checkpoint transaction.

        This legacy single-agent API remains source compatible. Consecutive
        calls (the engine currently makes one per crew member) are coalesced
        and committed together by the next checkpoint/end/close operation.
        """
        return self.save_agent_q_tables(
            [{
                "agent_id": agent_id,
                "q_table": q_table,
                "total_reward": total_reward,
            }],
            tick=tick,
            run_id=run_id,
        )

    def save_agent_q_tables(self, policies, tick: int,
                            run_id: int = None, flush: bool = False) -> int:
        """Queue multiple policies as compact, deduplicated upserts.

        ``policies`` may contain dictionaries with ``agent_id``, ``q_table``
        and optional ``total_reward`` fields, or four-item tuples in the form
        ``(agent_id, q_table, total_reward, tick)``. Only the newest pending
        policy for a run/agent is retained before the batch is flushed.
        """
        r_id = run_id if run_id is not None else self._current_run_id
        # SQLite treats NULL values as distinct in unique indexes; use a
        # sentinel for policies saved outside an active run.
        normalized_run_id = int(r_id) if r_id is not None else 0
        queued = 0
        try:
            for policy in policies:
                if isinstance(policy, dict):
                    agent_id = policy["agent_id"]
                    q_table = policy.get("q_table", {})
                    total_reward = policy.get("total_reward", 0.0)
                    policy_tick = int(policy.get("tick", tick))
                else:
                    agent_id, q_table, total_reward, policy_tick = policy
                    policy_tick = int(policy_tick)
                q_json = json.dumps(
                    q_table,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                q_hash = hashlib.sha256(q_json.encode("utf-8")).hexdigest()
                key = (normalized_run_id, str(agent_id))
                self._pending_q_tables[key] = (
                    normalized_run_id,
                    str(agent_id),
                    q_json,
                    q_hash,
                    float(total_reward),
                    policy_tick,
                    time.time(),
                    len(q_table),
                )
                queued += 1
        except Exception as e:
            logger.warning(f"Failed to queue Q-table batch: {e}")
            return 0
        if flush and not self._flush_pending():
            raise RuntimeError("Q-table transaction was rolled back")
        return queued

    def load_agent_q_table(self, agent_id: str,
                           planet_id: str = None) -> Optional[dict]:
        """Load an agent policy, optionally isolated to one planet."""
        try:
            # Preserve read-after-write behavior of the old immediate-commit
            # API while normal tick persistence remains batched.
            if not self._flush_pending():
                return None
            if planet_id is None:
                cursor = self._conn.execute(
                    """SELECT q_table_json FROM agent_q_tables
                       WHERE agent_id=?
                       ORDER BY timestamp DESC, updated_tick DESC,
                                is_latest DESC, id DESC
                       LIMIT 1""",
                    (agent_id,),
                )
            else:
                cursor = self._conn.execute(
                    """SELECT qt.q_table_json
                       FROM agent_q_tables AS qt
                       JOIN simulation_runs AS sr ON sr.id = qt.run_id
                       WHERE qt.agent_id=? AND sr.planet=?
                       ORDER BY qt.timestamp DESC, qt.updated_tick DESC,
                                qt.is_latest DESC, qt.id DESC LIMIT 1""",
                    (agent_id, planet_id),
                )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
        except Exception as e:
            logger.warning(f"Failed to load Q-table for {agent_id}: {e}")
        return None

    def discard_run_rl(self, run_id: int) -> dict[str, int]:
        """Remove one attempt's policy rows so the prior policy is visible."""
        return self._delete_rl_rows([int(run_id)])

    def _delete_rl_rows(self, run_ids: Optional[list[int]]) -> dict[str, int]:
        selected = set(run_ids) if run_ids is not None else None
        for cache in (
            self._pending_q_tables,
            self._last_q_table_hashes,
            self._last_q_progress_tick,
        ):
            for key in list(cache):
                if selected is None or key[0] in selected:
                    del cache[key]
        if selected == set():
            return {"policies_deleted": 0, "progress_rows_deleted": 0}
        where = "" if selected is None else " WHERE run_id IN (%s)" % ",".join(
            "?" for _ in selected
        )
        parameters = tuple(selected or ())
        with self._conn:
            policy_count = int(self._conn.execute(
                "SELECT COUNT(*) FROM agent_q_tables" + where,
                parameters,
            ).fetchone()[0])
            progress_count = int(self._conn.execute(
                "SELECT COUNT(*) FROM agent_q_table_progress" + where,
                parameters,
            ).fetchone()[0])
            self._conn.execute(
                "DELETE FROM agent_q_table_progress" + where, parameters
            )
            self._conn.execute("DELETE FROM agent_q_tables" + where, parameters)
        return {
            "policies_deleted": policy_count,
            "progress_rows_deleted": progress_count,
        }

    def reset_rl_policies(self, planet_id: str = None) -> dict[str, int]:
        """Delete one planet's policies while preserving audit history."""
        run_ids = None if planet_id is None else [
            int(row[0])
            for row in self._conn.execute(
                "SELECT id FROM simulation_runs WHERE planet=?", (planet_id,)
            ).fetchall()
        ]
        return self._delete_rl_rows(run_ids)

    def get_planet_rl_summary(self) -> dict:
        rows = self._conn.execute(
            """SELECT sr.planet, COUNT(qt.id), MAX(qt.updated_tick),
                      MAX(qt.timestamp)
               FROM agent_q_tables AS qt
               JOIN simulation_runs AS sr ON sr.id=qt.run_id
               GROUP BY sr.planet"""
        ).fetchall()
        return {
            row[0]: {
                "policy_snapshots": row[1],
                "last_tick": row[2],
                "updated_at": row[3],
            }
            for row in rows
        }

    def record_tick_telemetry(self, tick: int, payload: dict) -> None:
        if self._current_run_id is None:
            return
        encoded = zlib.compress(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        self._pending_writes.append((
            "tick_telemetry",
            (self._current_run_id, int(tick), encoded),
        ))

    def get_tick_telemetry(
        self, run_id: int, after_tick: int = -1, limit: int = 100,
        latest: bool = False,
    ) -> list[dict]:
        order = "DESC" if latest else "ASC"
        rows = self._conn.execute(
            f"""SELECT payload FROM tick_telemetry
                WHERE run_id=? AND tick>? ORDER BY tick {order} LIMIT ?""",
            (int(run_id), int(after_tick), min(200, max(1, int(limit)))),
        ).fetchall()
        decoded = [json.loads(zlib.decompress(row[0])) for row in rows]
        return list(reversed(decoded)) if latest else decoded

    def get_analytics_data(self, run_id: int = None) -> dict:
        """Fetch comprehensive telemetry and time-series for the Simulation Analytics tab."""
        r_id = run_id if run_id is not None else self._current_run_id
        if r_id is None:
            # Fallback to latest run
            cursor = self._conn.execute("SELECT id FROM simulation_runs ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if not row:
                return {}
            r_id = row[0]

        # 1. Scores over time
        scores = self.get_score_history(r_id)

        # 2. Vitals over time per agent
        cursor = self._conn.execute(
            """SELECT tick, agent_name, temperature_stress, energy, hunger, thirst, o2_supply, injury_level
               FROM agent_snapshots WHERE run_id=? ORDER BY tick""",
            (r_id,)
        )
        vitals_by_agent = {}
        for r in cursor.fetchall():
            name = r[1]
            if name not in vitals_by_agent:
                vitals_by_agent[name] = []
            vitals_by_agent[name].append({
                "tick": r[0], "temp_stress": r[2], "energy": r[3],
                "hunger": r[4], "thirst": r[5], "o2": r[6], "injury": r[7]
            })

        # 3. Q-learning progression
        cursor = self._conn.execute(
            """SELECT updated_tick, agent_id, total_reward
               FROM agent_q_table_progress WHERE run_id=?
               UNION
               SELECT updated_tick, agent_id, total_reward
               FROM agent_q_tables
               WHERE run_id=? AND is_latest=0
               ORDER BY updated_tick""",
            (r_id, r_id)
        )
        rl_progression = [
            {"tick": r[0], "agent": r[1], "total_reward": r[2]}
            for r in cursor.fetchall()
        ]

        return {
            "run_id": r_id,
            "scores": scores,
            "vitals": vitals_by_agent,
            "rl_progression": rl_progression,
            "deaths": self.get_deaths(r_id)
        }
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._flush_pending()
            self._conn.close()
            self._conn = None
