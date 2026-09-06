"""
World Event Scheduler — Deterministic planetary event generation.

Generates time-sequenced planetary events (flares, quakes, storms,
volcanic eruptions, micrometeorite showers) based on planet config
hazard profiles.

Design principles:
- NO randomness at event generation time — events are pre-computed from seed
- Events affect ALL agents and structures within their radius
- Event severity scales with planet hazard profile
- Deterministic replay: same seed + tick = same events

Scientific basis:
- Stellar flare rates: Davenport et al. 2019 (Proxima Centauri observations)
- Seismic activity: scaled from Earth tectonic models
- Dust storms: Mars analogue (Lemmon et al. 2015)
"""

import math
import hashlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorldEvent:
    """A scheduled planetary event."""
    event_type: str           # flare, quake, dust_storm, volcanic_eruption, micrometeorite
    start_tick: int
    duration_ticks: int
    severity: float           # 0.0 - 1.0 (normalized intensity)
    radius_units: int         # Affected area radius (0 = global)
    epicenter: Optional[tuple[int, int]] = None  # For localized events

    # Computed effects (populated by event type)
    effects: dict = field(default_factory=dict)

    @property
    def end_tick(self) -> int:
        return self.start_tick + self.duration_ticks

    def is_active(self, current_tick: int) -> bool:
        return self.start_tick <= current_tick < self.end_tick

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "start_tick": self.start_tick,
            "duration_ticks": self.duration_ticks,
            "severity": self.severity,
            "radius_units": self.radius_units,
            "epicenter": self.epicenter,
            "effects": self.effects,
        }


class EventScheduler:
    """
    Pre-computes and serves planetary events for the simulation.

    Events are generated deterministically from the planet config +
    simulation seed. The scheduler maintains a sorted timeline and
    efficiently queries active events per tick.
    """

    def __init__(
        self,
        planet_config,
        seed: int = 42,
        max_ticks: int = 1000,
        tick_minutes: float = 5.0,
    ):
        self.planet = planet_config
        self.seed = seed
        self.max_ticks = max_ticks
        self.tick_minutes = max(0.1, float(tick_minutes))
        self._legacy_tick_scale = self.tick_minutes / 5.0
        self.events: list[WorldEvent] = []

        self._generate_all_events()
        self.events.sort(key=lambda e: e.start_tick)

    def _legacy_ticks(self, value: float) -> float:
        """Keep the physical duration of a five-minute-authored value."""
        return float(value) * 5.0 / self.tick_minutes

    def _days_to_ticks(self, days: float) -> float:
        """Convert a physical mean interval to the configured tick clock."""
        return max(1.0, float(days) * 24.0 * 60.0 / self.tick_minutes)

    def _duration_ticks(self, legacy_value: float) -> int:
        return max(1, int(math.ceil(self._legacy_ticks(legacy_value))))

    def _linear_per_tick(self, legacy_value: float) -> float:
        return float(legacy_value) * self._legacy_tick_scale

    def _hourly_probability_per_tick(self, hourly_probability: float) -> float:
        """Convert an event-hour conditional risk to this physics tick."""
        probability = max(0.0, min(1.0, float(hourly_probability)))
        return 1.0 - (1.0 - probability) ** (self.tick_minutes / 60.0)

    def _deterministic_hash(self, *args) -> float:
        """Hash-based deterministic pseudo-random in [0, 1]."""
        h = hashlib.sha256(f"{self.seed}:{':'.join(str(a) for a in args)}".encode())
        return int.from_bytes(h.digest()[:4], 'little') / (2**32)

    def _generate_all_events(self):
        """Generate all events for the full simulation timeline."""
        self._generate_flare_events()
        self._generate_seismic_events()
        self._generate_storm_events()
        self._generate_volcanic_events()
        self._generate_micrometeorite_events()

    # === STELLAR FLARES ===

    def _generate_flare_events(self):
        """
        Generate stellar flare events from planet's flare profile.

        Flare rate is specified as events_per_80_days in planet config.
        Each flare has a severity (normal vs superflare) that determines
        radiation multiplier and structure damage.

        Scientific basis:
        - Proxima Centauri: ~85 flares per 80 days (Davenport et al. 2019)
        - Superflare probability: 0.8/year for Proxima
        """
        flare_profile = self.planet.data.get("flare_profile", {})
        if not flare_profile:
            return

        events_per_80_days = flare_profile.get("flare_events_per_80_days", 0)
        if events_per_80_days == 0:
            return

        # Convert physical days to the configured simulation resolution.
        ticks_per_80_days = 80 * 24 * 60 / self.tick_minutes
        flare_interval_avg = ticks_per_80_days / events_per_80_days

        superflare_prob = flare_profile.get("superflare_probability_per_year", 0)
        ticks_per_year = 365.25 * 24 * 60 / self.tick_minutes
        superflare_per_tick = superflare_prob / ticks_per_year

        radiation_system = self.planet.data.get("radiation_system", {}) or {}
        radiation_flare_mult = flare_profile.get(
            "flare_dose_multiplier",
            radiation_system.get("flare_dose_multiplier", 10.0),
        )

        # Generate flares using quasi-random spacing
        tick = 0
        elapsed_ticks = 0.0
        flare_index = 0
        while tick < self.max_ticks:
            # Vary interval by ±50% deterministically
            jitter = self._deterministic_hash("flare", flare_index) * flare_interval_avg
            elapsed_ticks += max(1.0, flare_interval_avg * 0.5 + jitter)
            tick = max(tick + 1, int(round(elapsed_ticks)))

            if tick >= self.max_ticks:
                break

            # Determine severity
            is_super = self._deterministic_hash("super", flare_index) < superflare_per_tick * flare_interval_avg
            severity = 0.9 + self._deterministic_hash("sev", flare_index) * 0.1 if is_super else \
                       0.2 + self._deterministic_hash("sev", flare_index) * 0.5

            duration = self._duration_ticks(
                2 + self._deterministic_hash("dur", flare_index) * 6
            )

            effects = {
                "radiation_multiplier": radiation_flare_mult * (1.0 + severity),
                # Ionizing radiation/UV is a crew-dose and electronics
                # reliability hazard, not generic mechanical hull damage.
                # Applying percentage health loss to every pressure vessel on
                # every ordinary flare guaranteed destruction after ~191
                # Proxima events. Component upsets require a separate,
                # radiation-hardness-aware fault model; until then no general
                # structure integrity is subtracted here.
                "structure_damage_per_tick": 0.0,
                "uv_lethal_unprotected": severity > 0.7,
                "visibility_reduction": 0.0,  # Flares don't reduce visibility
                "is_superflare": is_super,
            }

            self.events.append(WorldEvent(
                event_type="stellar_flare",
                start_tick=tick,
                duration_ticks=duration,
                severity=round(severity, 2),
                radius_units=0,  # Global
                effects=effects,
            ))
            flare_index += 1

    # === SEISMIC EVENTS ===

    def _generate_seismic_events(self):
        """
        Generate seismic events based on tectonic activity level.
        Localized — epicenter is randomized on the map.
        """
        geology = self.planet.data.get("geology", {})
        tectonic = geology.get("tectonic_activity", "none").lower()

        if "none" in tectonic or "minimal" in tectonic:
            return
        elif "high" in tectonic or "active" in tectonic:
            mean_interval_days = 10.0
        elif "moderate" in tectonic:
            mean_interval_days = 30.0
        elif "low" in tectonic:
            mean_interval_days = 90.0
        else:
            mean_interval_days = 60.0

        # These are mission-affecting local events, not every seismometer
        # detection.  The previous 100 legacy-tick interval meant one
        # potentially destructive quake about every 8.3 simulated hours and
        # destroyed a flight habitat in 15 days.  Even Earth records only
        # roughly 16 M7+ events per year worldwide (USGS); Apollo registered
        # only 28 shallow moonquakes in eight years (NASA NTRS 20240009464).
        # Exoplanet rates are uncertain, so the qualitative planet label sets
        # a conservative, explicit physical-day prior which a future planet
        # profile may override.
        seismic_profile = self.planet.data.get("seismic_profile", {}) or {}
        mean_interval_days = float(seismic_profile.get(
            "mission_event_mean_interval_days", mean_interval_days
        ))
        avg_interval = self._days_to_ticks(mean_interval_days)

        tick = 0
        elapsed_ticks = 0.0
        quake_idx = 0
        map_size = self.planet.data.get("world_generation", {}).get("map_size", 2000)
        center = map_size // 2

        while tick < self.max_ticks:
            jitter = self._deterministic_hash("quake_j", quake_idx)
            elapsed_ticks += max(1.0, avg_interval * (0.5 + jitter))
            tick = max(tick + 1, int(round(elapsed_ticks)))
            if tick >= self.max_ticks:
                break

            severity = 0.1 + self._deterministic_hash("quake_s", quake_idx) * 0.6
            duration = self._duration_ticks(
                1 + self._deterministic_hash("quake_d", quake_idx) * 3
            )

            # Random epicenter within habitable zone
            angle = self._deterministic_hash("quake_a", quake_idx) * 2 * math.pi
            radius = self._deterministic_hash("quake_r", quake_idx) * center * 0.6
            ex = int(center + radius * math.cos(angle))
            ey = int(center + radius * math.sin(angle))

            # Engineered surface hardware does not lose 10-30% integrity in
            # every felt tremor.  Only the strong end of the generated range
            # causes permanent structural damage; the total is divided over
            # the event so changing tick resolution cannot change the dose.
            total_structure_damage = max(0.0, severity - 0.40) * 0.12
            effects = {
                "structure_damage_per_tick": (
                    total_structure_damage / max(1, duration)
                ),
                "injury_probability_per_tick": self._hourly_probability_per_tick(
                    0.03 * severity
                ),
                "traversal_cost_multiplier": 1.5 + severity,
                "aftershock_chance": severity * 0.3,
            }

            self.events.append(WorldEvent(
                event_type="seismic_quake",
                start_tick=tick,
                duration_ticks=duration,
                severity=round(severity, 2),
                radius_units=int(50 + severity * 200),  # 50-250 units
                epicenter=(ex, ey),
                effects=effects,
            ))
            quake_idx += 1

    # === DUST STORMS ===

    def _generate_storm_events(self):
        """
        Generate dust/wind storms on planets with atmosphere.
        Require atmosphere + wind hazard.
        """
        if not self.planet.atmosphere.get("present", False):
            return

        hazards = self.planet.data.get("hazards", [])
        has_storm_hazard = any(h in hazards for h in [
            "dust_storm", "dust_storms", "extreme_wind",
            "atmospheric_storms", "dense_atmosphere_storms",
        ])
        if not has_storm_hazard:
            return

        storm_profile = self.planet.data.get("storm_profile", {}) or {}
        avg_interval = self._days_to_ticks(float(storm_profile.get(
            "mission_event_mean_interval_days", 20.0
        )))
        tick = 0
        elapsed_ticks = 0.0
        storm_idx = 0

        while tick < self.max_ticks:
            jitter = self._deterministic_hash("storm_j", storm_idx)
            elapsed_ticks += max(1.0, avg_interval * (0.5 + jitter))
            tick = max(tick + 1, int(round(elapsed_ticks)))
            if tick >= self.max_ticks:
                break

            severity = 0.3 + self._deterministic_hash("storm_s", storm_idx) * 0.7
            duration_hours = (
                3.0 + self._deterministic_hash("storm_d", storm_idx) * 15.0
            )
            duration = max(1, int(math.ceil(
                duration_hours * 60.0 / self.tick_minutes
            )))

            total_structure_damage = max(0.0, severity - 0.70) * 0.02

            effects = {
                "visibility_reduction": 0.3 + severity * 0.5,  # Up to 80% visibility loss
                "traversal_cost_multiplier": 1.3 + severity * 0.7,
                "wind_speed_bonus_kmh": 20 + severity * 40,
                "solar_panel_efficiency_reduction": severity * 0.6,  # Sand blocks sunlight
                "structure_damage_per_tick": (
                    total_structure_damage / max(1, duration)
                ),
            }

            self.events.append(WorldEvent(
                event_type="dust_storm",
                start_tick=tick,
                duration_ticks=duration,
                severity=round(severity, 2),
                radius_units=0,  # Global (affects entire region)
                effects=effects,
            ))
            storm_idx += 1

    # === VOLCANIC ERUPTIONS ===

    def _generate_volcanic_events(self):
        """Generate volcanic gas eruptions in volcanic biomes."""
        geology = self.planet.data.get("geology", {})
        volcanism = geology.get("volcanism", "none").lower()

        if "none" in volcanism or "minimal" in volcanism:
            return

        if "active" in volcanism or "frequent" in volcanism:
            mean_interval_days = 15.0
        elif "moderate" in volcanism:
            mean_interval_days = 60.0
        else:
            mean_interval_days = 180.0
        volcanic_profile = self.planet.data.get("volcanic_profile", {}) or {}
        mean_interval_days = float(volcanic_profile.get(
            "mission_event_mean_interval_days", mean_interval_days
        ))
        avg_interval = self._days_to_ticks(mean_interval_days)

        tick = 0
        elapsed_ticks = 0.0
        vol_idx = 0
        map_size = self.planet.data.get("world_generation", {}).get("map_size", 2000)
        center = map_size // 2

        while tick < self.max_ticks:
            jitter = self._deterministic_hash("vol_j", vol_idx)
            elapsed_ticks += max(1.0, avg_interval * (0.5 + jitter))
            tick = max(tick + 1, int(round(elapsed_ticks)))
            if tick >= self.max_ticks:
                break

            severity = 0.2 + self._deterministic_hash("vol_s", vol_idx) * 0.6
            duration = self._duration_ticks(
                5 + self._deterministic_hash("vol_d", vol_idx) * 15
            )

            # Epicenter in volcanic zone (roughly located)
            angle = self._deterministic_hash("vol_a", vol_idx) * 2 * math.pi
            radius = center * (0.2 + self._deterministic_hash("vol_r", vol_idx) * 0.3)
            ex = int(center + radius * math.cos(angle))
            ey = int(center + radius * math.sin(angle))

            effects = {
                "toxic_gas_concentration": severity * 0.8,
                "temperature_bonus_c": 20 * severity,
                "visibility_reduction": 0.2 + severity * 0.3,
                "disease_toxic_probability_per_tick": self._hourly_probability_per_tick(
                    0.05 * severity
                ),
            }

            self.events.append(WorldEvent(
                event_type="volcanic_eruption",
                start_tick=tick,
                duration_ticks=duration,
                severity=round(severity, 2),
                radius_units=int(30 + severity * 80),
                epicenter=(ex, ey),
                effects=effects,
            ))
            vol_idx += 1

    # === MICROMETEORITE SHOWERS ===

    def _generate_micrometeorite_events(self):
        """
        Generate micrometeorite shower events on atmosphereless planets.
        No atmosphere = no burn-up protection.
        """
        if self.planet.atmosphere.get("present", False):
            return  # Atmosphere burns up micrometeorites

        hazards = self.planet.data.get("hazards", [])
        if "micrometeorite" not in hazards and "micrometeorites" not in hazards:
            return

        # Micrometeoroid flux is continuous, but a mission-affecting cluster
        # striking one small surface worksite is not a planet-wide daily
        # disaster. The legacy 250-tick interval produced roughly one global
        # shower per day and destroyed every structure in about three weeks.
        # Generate sparse, localized impact clusters instead. A planet config
        # can refine this scenario prior when better environment data exists.
        profile = self.planet.data.get("micrometeorite_profile", {}) or {}
        mean_interval_days = max(
            1.0, float(profile.get("mean_local_cluster_interval_days", 30.0))
        )
        avg_interval = self._days_to_ticks(mean_interval_days)
        tick = 0
        elapsed_ticks = 0.0
        met_idx = 0

        while tick < self.max_ticks:
            jitter = self._deterministic_hash("met_j", met_idx)
            elapsed_ticks += max(1.0, avg_interval * (0.5 + jitter))
            tick = max(tick + 1, int(round(elapsed_ticks)))
            if tick >= self.max_ticks:
                break

            severity = 0.2 + self._deterministic_hash("met_s", met_idx) * 0.5
            duration_minutes = (
                10.0 + self._deterministic_hash("met_d", met_idx) * 50.0
            )
            duration = max(
                1, int(math.ceil(duration_minutes / self.tick_minutes))
            )
            total_local_damage = 0.03 * severity
            impact_x = int(
                self._deterministic_hash("met_x", met_idx)
                * max(1, self.planet.map_size - 1)
            )
            impact_y = int(
                self._deterministic_hash("met_y", met_idx)
                * max(1, self.planet.map_size - 1)
            )
            impact_radius = int(
                profile.get("cluster_radius_units", 5 + severity * 15)
            )

            effects = {
                "injury_probability_per_tick": self._hourly_probability_per_tick(
                    0.01 * severity
                ),
                "structure_damage_per_tick": total_local_damage / duration,
                "suit_damage_probability": self._hourly_probability_per_tick(
                    0.02 * severity
                ),
            }

            self.events.append(WorldEvent(
                event_type="micrometeorite_shower",
                start_tick=tick,
                duration_ticks=duration,
                severity=round(severity, 2),
                radius_units=max(1, impact_radius),
                epicenter=(impact_x, impact_y),
                effects=effects,
            ))
            met_idx += 1

    # === QUERY API ===

    def get_active_events(self, current_tick: int) -> list[WorldEvent]:
        """Return all events active at the given tick."""
        return [e for e in self.events if e.is_active(current_tick)]

    def get_events_in_range(self, start_tick: int, end_tick: int) -> list[WorldEvent]:
        """Return all events that overlap with a tick range."""
        return [e for e in self.events
                if e.start_tick < end_tick and e.end_tick > start_tick]

    def is_flare_active(self, current_tick: int) -> bool:
        """Quick check if any stellar flare is active right now."""
        return any(e.event_type == "stellar_flare" and e.is_active(current_tick)
                   for e in self.events)

    def get_combined_effects(self, current_tick: int,
                             agent_x: int = 0, agent_y: int = 0) -> dict:
        """
        Get combined effects of all active events at an agent's position.

        Returns merged effect dict — multiplicative for multipliers,
        additive for flat bonuses, OR for booleans.
        """
        active = self.get_active_events(current_tick)

        combined = {
            "radiation_multiplier": 1.0,
            "structure_damage_per_tick": 0.0,
            "injury_probability_per_tick": 0.0,
            "traversal_cost_multiplier": 1.0,
            "visibility_reduction": 0.0,
            "wind_speed_bonus_kmh": 0.0,
            "solar_panel_efficiency_reduction": 0.0,
            "temperature_bonus_c": 0.0,
            "toxic_exposure_probability_per_tick": 0.0,
            "suit_damage_probability_per_tick": 0.0,
            "uv_lethal_unprotected": False,
            "active_event_types": [],
            "active_event_count": 0,
        }

        for event in active:
            # Check if agent is within event radius (if localized)
            if event.epicenter and event.radius_units > 0:
                dx = agent_x - event.epicenter[0]
                dy = agent_y - event.epicenter[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > event.radius_units:
                    continue  # Agent outside event range
                # Scale effects by distance (linear falloff)
                falloff = max(0.1, 1.0 - dist / event.radius_units)
            else:
                falloff = 1.0  # Global event

            eff = event.effects

            # Multiplicative
            radiation_multiplier = float(
                eff.get("radiation_multiplier", 1.0)
            )
            traversal_multiplier = float(
                eff.get("traversal_cost_multiplier", 1.0)
            )
            # A localized event fades toward neutral (1.0), not toward zero.
            combined["radiation_multiplier"] *= (
                1.0 + (radiation_multiplier - 1.0) * falloff
            )
            combined["traversal_cost_multiplier"] *= (
                1.0 + (traversal_multiplier - 1.0) * falloff
            )

            # Additive (scaled by falloff)
            combined["structure_damage_per_tick"] += eff.get("structure_damage_per_tick", 0) * falloff
            combined["injury_probability_per_tick"] += eff.get("injury_probability_per_tick", 0) * falloff
            combined["visibility_reduction"] += eff.get("visibility_reduction", 0) * falloff
            combined["wind_speed_bonus_kmh"] += eff.get("wind_speed_bonus_kmh", 0) * falloff
            combined["solar_panel_efficiency_reduction"] += eff.get("solar_panel_efficiency_reduction", 0) * falloff
            combined["temperature_bonus_c"] += eff.get("temperature_bonus_c", 0) * falloff
            combined["toxic_exposure_probability_per_tick"] += eff.get(
                "disease_toxic_probability_per_tick", 0
            ) * falloff
            combined["suit_damage_probability_per_tick"] += eff.get(
                "suit_damage_probability", 0
            ) * falloff

            # Boolean OR
            if eff.get("uv_lethal_unprotected", False):
                combined["uv_lethal_unprotected"] = True

            combined["active_event_types"].append(event.event_type)
            combined["active_event_count"] += 1

        # Clamp visibility reduction to 0-1
        combined["visibility_reduction"] = min(1.0, combined["visibility_reduction"])
        combined["injury_probability_per_tick"] = min(
            1.0, combined["injury_probability_per_tick"]
        )
        combined["toxic_exposure_probability_per_tick"] = min(
            1.0, combined["toxic_exposure_probability_per_tick"]
        )
        combined["suit_damage_probability_per_tick"] = min(
            1.0, combined["suit_damage_probability_per_tick"]
        )

        return combined

    def get_event_summary(self, current_tick: int) -> str:
        """Generate a human-readable summary of active events for LLM context."""
        active = self.get_active_events(current_tick)
        if not active:
            return "No active planetary events."

        parts = []
        for e in active:
            remaining = e.end_tick - current_tick
            if e.event_type == "stellar_flare":
                stype = "SUPERFLARE" if e.effects.get("is_superflare") else "Flare"
                parts.append(f"[WARNING] {stype} (severity {e.severity:.0%}, {remaining} ticks remaining)")
            elif e.event_type == "seismic_quake":
                parts.append(f"[QUAKE] Seismic event (severity {e.severity:.0%}, {remaining} ticks)")
            elif e.event_type == "dust_storm":
                vis = e.effects.get('visibility_reduction', 0)
                parts.append(f"[STORM] Dust storm (visibility -{vis:.0%}, {remaining} ticks)")
            elif e.event_type == "volcanic_eruption":
                parts.append(f"[VOLCANIC] Eruption (toxic gas, {remaining} ticks)")
            elif e.event_type == "micrometeorite_shower":
                parts.append(f"[METEOR] Micrometeorite shower ({remaining} ticks)")

        return " | ".join(parts)

    def get_total_event_count(self) -> dict:
        """Statistics: total scheduled events by type."""
        counts = {}
        for e in self.events:
            counts[e.event_type] = counts.get(e.event_type, 0) + 1
        return counts
