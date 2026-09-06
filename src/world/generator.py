"""
Procedural world generator for exoplanet surface simulation.

Generates deterministic terrain from seed + coordinates using:
- Simplex/Perlin noise for non-tidally-locked planets (continental-style)
- Center-distance gradient for tidally-locked planets (concentric ring biomes)

Key design principle: NO pre-generated map. Every (x, y) query is computed
on-the-fly from the noise function. Only modified cells are stored in the
sparse state layer (see sparse_state.py).

Scientific basis:
- Tidally locked gradient: Leconte et al. 2013, Yang et al. 2013
  (GCM models of tidally locked exoplanet climate)
- Noise-based terrain: Standard procedural generation adapted with
  planet-specific parameters (gravity affects elevation distribution,
  volcanism affects biome boundaries)
"""

import math
import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    from noise import snoise2
except ImportError:
    # Fallback: pure-Python simplex noise approximation
    # Used only if the 'noise' package is not installed
    def snoise2(x: float, y: float, octaves: int = 1,
                persistence: float = 0.5, lacunarity: float = 2.0,
                base: int = 0) -> float:
        """Simplified Perlin-like noise fallback using hash-based interpolation."""
        total = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        for _ in range(octaves):
            # Hash-based pseudo-random gradient
            ix = int(math.floor(x * frequency + base))
            iy = int(math.floor(y * frequency + base))
            fx = (x * frequency + base) - ix
            fy = (y * frequency + base) - iy
            # Simple smooth interpolation
            u = fx * fx * (3 - 2 * fx)
            v = fy * fy * (3 - 2 * fy)
            # Hash corners
            def _hash(xi, yi):
                h = hashlib.md5(f"{xi},{yi},{base}".encode()).digest()
                return (int.from_bytes(h[:4], 'little') / 2**32) * 2 - 1
            n00 = _hash(ix, iy)
            n10 = _hash(ix + 1, iy)
            n01 = _hash(ix, iy + 1)
            n11 = _hash(ix + 1, iy + 1)
            nx0 = n00 * (1 - u) + n10 * u
            nx1 = n01 * (1 - u) + n11 * u
            total += (nx0 * (1 - v) + nx1 * v) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        return total / max_value if max_value > 0 else 0.0


class PlanetConfig:
    """Loaded planet configuration from JSON."""

    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.id = self.data["id"]
        self.display_name = self.data["display_name"]
        self.tidally_locked = self.data["surface"]["tidally_locked"]
        self.gravity_g = self.data["physical"]["gravity_g"]
        self.biomes = {b["id"]: b for b in self.data["biomes"]}
        self.biome_list = self.data["biomes"]
        self.world_gen = self.data["world_generation"]
        self.map_size = self.world_gen["map_size"]
        self.resource_modifiers = self.data["resource_modifiers"]
        self.scarcity_factor = self.resource_modifiers["scarcity_factor"]
        self.hazards = self.data["hazards"]
        self.flare_profile = self.data.get("flare_profile", {})
        self.radiation_system = self.data.get("radiation_system", None)
        self.gravity_effects = self.data.get("gravity_effects", None)
        self.day_night_cycle = self.data.get("day_night_cycle", None)
        self.atmosphere = self.data["atmosphere"]
        self.difficulty_rating = self.data["difficulty_rating"]
        self.physical = self.data.get("physical", {})

    @property
    def name(self) -> str:
        """Alias for display_name (used by engine and prompts)."""
        return self.display_name

    @classmethod
    def load_all(cls, config_dir: str = "config/planets") -> dict:
        """Load all planet configs from the config directory."""
        planets = {}
        config_path = Path(config_dir)
        for json_file in config_path.glob("*.json"):
            planet = cls(str(json_file))
            planets[planet.id] = planet
        return planets

    @classmethod
    def load_random(cls, config_dir: str = "config/planets",
                    rng: Optional[np.random.Generator] = None) -> 'PlanetConfig':
        """Load a random planet configuration."""
        planets = cls.load_all(config_dir)
        if rng is None:
            rng = np.random.default_rng()
        planet_id = rng.choice(list(planets.keys()))
        return planets[planet_id]


class WorldGenerator:
    """
    Deterministic procedural world generator.

    For any (x, y) coordinate, returns the biome type and base resource
    availability WITHOUT storing anything in memory. The noise function
    and planet config together fully determine the world state at t=0.

    Two generation modes:
    1. Tidally-locked gradient: concentric rings from substellar point
    2. Procedural noise: Simplex noise for continental-style terrain
    """

    def __init__(
        self, planet: PlanetConfig, seed: int = 42, tick_minutes: float = 5.0
    ):
        self.planet = planet
        self.seed = seed
        self.tick_minutes = max(0.1, float(tick_minutes))
        self.rng = np.random.default_rng(seed)
        self.map_size = planet.map_size
        self.center = self.map_size // 2

        # Pre-compute biome boundaries for tidally-locked planets
        self._elev_cache: dict[tuple[int, int], float] = {}
        self._biome_cache: dict[tuple[int, int], dict] = {}
        if planet.tidally_locked:
            self._setup_tidal_lock_zones()
        else:
            self._setup_noise_params()

    def _physical_cycle_ticks(self, legacy_ticks: float) -> int:
        """Convert a cycle authored for the legacy five-minute clock."""
        return max(1, int(math.ceil(
            float(legacy_ticks) * 5.0 / self.tick_minutes
        )))

    def _setup_tidal_lock_zones(self):
        """
        Define concentric ring boundaries for tidally-locked world generation.

        Zone layout (center outward):
        - Substellar zone: 0 to 20% radius
        - Day-side zone: 20% to 40% radius
        - Terminator zone: 40% to 55% radius (habitable band)
        - Night-side near: 55% to 75% radius
        - Anti-stellar zone: 75% to 100% radius

        These percentages are based on GCM models of tidally-locked
        exoplanet heat distribution (Leconte et al. 2013).
        """
        r_max = self.map_size // 2
        habitable_width = self.planet.world_gen.get("habitable_band_width_units", 300)

        # Map biome zones to their radial boundaries (as fraction of max radius)
        self.zone_boundaries = {}
        zone_order = []
        for biome in self.planet.biome_list:
            zone = biome.get("zone", "")
            zone_order.append((biome["id"], zone))

        # Assign radial fractions based on zone type
        zone_fractions = {
            "substellar": (0.0, 0.20),
            "day_side": (0.20, 0.40),
            "terminator": (0.40, 0.55),
            "night_side_near": (0.55, 0.75),
            "anti_stellar": (0.75, 1.0),
        }

        self.radial_biomes = []
        for biome in self.planet.biome_list:
            zone = biome.get("zone", "terminator")
            frac = zone_fractions.get(zone, (0.4, 0.55))
            self.radial_biomes.append({
                "biome_id": biome["id"],
                "r_min": int(frac[0] * r_max),
                "r_max": int(frac[1] * r_max),
            })

    def _setup_noise_params(self):
        """Configure noise parameters for procedural terrain generation."""
        wg = self.planet.world_gen
        self.noise_octaves = wg.get("noise_octaves", 6)
        self.noise_persistence = wg.get("noise_persistence", 0.5)
        self.noise_lacunarity = wg.get("noise_lacunarity", 2.0)
        self.elevation_range = wg.get("elevation_range", [0, 100])

    def get_elevation(self, x: int, y: int) -> float:
        """
        Compute deterministic elevation for a coordinate.

        For tidally-locked planets, elevation is based on distance from
        the substellar point with noise perturbation.
        For others, pure noise-based continental terrain.

        Returns: elevation in range [0, 1] (normalized)
        """
        coord = (x, y)
        if coord in self._elev_cache:
            return self._elev_cache[coord]
        if self.planet.tidally_locked:
            val = self._tidal_lock_elevation(x, y)
        else:
            val = self._noise_elevation(x, y)
        self._elev_cache[coord] = val
        return val

    def _tidal_lock_elevation(self, x: int, y: int) -> float:
        """
        Tidally-locked elevation: primarily distance from center with
        noise perturbation for local terrain variation.
        """
        dx = x - self.center
        dy = y - self.center
        r_max = self.map_size / 2.0
        distance = math.sqrt(dx * dx + dy * dy) / r_max  # 0..1

        # Base elevation from distance (higher near terminator due to
        # tidal stress creating ridge formations)
        if 0.4 <= distance <= 0.55:
            base = 0.6 + 0.2 * math.sin((distance - 0.4) / 0.15 * math.pi)
        else:
            base = 0.3 + 0.1 * distance

        # Add noise for local terrain variation
        scale = 0.008
        noise_val = snoise2(
            x * scale, y * scale,
            octaves=4,
            persistence=0.4,
            lacunarity=2.0,
            base=self.seed
        )
        return max(0.0, min(1.0, base + noise_val * 0.15))

    def _noise_elevation(self, x: int, y: int) -> float:
        """Pure noise-based elevation for non-tidally-locked planets."""
        scale = 0.025
        noise_val = snoise2(
            x * scale, y * scale,
            octaves=self.noise_octaves,
            persistence=self.noise_persistence,
            lacunarity=self.noise_lacunarity,
            base=self.seed
        )
        # Normalize from [-1, 1] to [0, 1]
        return (noise_val + 1.0) / 2.0

    def get_biome(self, x: int, y: int) -> dict:
        """
        Determine the biome at (x, y).

        Returns the full biome dict from the planet config.

        For tidally-locked: based on radial distance from substellar point.
        For others: based on elevation thresholds derived from noise.
        """
        coord = (x, y)
        if coord in self._biome_cache:
            return self._biome_cache[coord]
        if self.planet.tidally_locked:
            val = self._tidal_lock_biome(x, y)
        else:
            val = self._noise_biome(x, y)
        self._biome_cache[coord] = val
        return val

    def _tidal_lock_biome(self, x: int, y: int) -> dict:
        """Determine biome from radial distance to substellar point."""
        dx = x - self.center
        dy = y - self.center
        distance = math.sqrt(dx * dx + dy * dy)

        for rb in self.radial_biomes:
            if rb["r_min"] <= distance < rb["r_max"]:
                return self.planet.biomes[rb["biome_id"]]

        # Fallback: outermost biome for coordinates beyond map edge
        return self.planet.biomes[self.radial_biomes[-1]["biome_id"]]

    def _noise_biome(self, x: int, y: int) -> dict:
        """
        Determine biome from noise-based elevation and secondary noise channels.

        Uses multi-channel noise:
        - Channel 1 (elevation): primary terrain shape
        - Channel 2 (moisture/temperature): secondary biome differentiation
        - Channel 3 (volcanism): localized volcanic zones

        Biome assignment follows elevation-based thresholds that map to
        the planet's specific biome set.
        """
        elevation = self.get_elevation(x, y)

        # Secondary noise channel for biome variation within elevation bands
        moisture = snoise2(
            x * 0.003 + 500, y * 0.003 + 500,
            octaves=3, persistence=0.4, lacunarity=2.0,
            base=self.seed + 1
        )
        moisture = (moisture + 1.0) / 2.0  # Normalize to [0, 1]

        # Volcanic noise — sharp, localized features
        volcanic = snoise2(
            x * 0.015 + 1000, y * 0.015 + 1000,
            octaves=2, persistence=0.3, lacunarity=3.0,
            base=self.seed + 2
        )

        # Map to biomes based on planet's biome set
        # Sort biomes by their zone/elevation role
        biome_ids = [b["id"] for b in self.planet.biome_list]

        # Check for volcanic zones first (localized, override)
        for biome in self.planet.biome_list:
            if biome.get("zone") in ("localized_volcanic", "localized_geothermal",
                                      "localized_scattered"):
                if volcanic > 0.75:
                    return biome

        # Elevation-based assignment
        if elevation > 0.75:
            # High elevation biomes
            for biome in self.planet.biome_list:
                if biome.get("zone") in ("elevation_high", "polar_high_altitude"):
                    return biome
        elif elevation > 0.55:
            # Mid-high — check for polar/frost
            if moisture < 0.3:
                for biome in self.planet.biome_list:
                    if biome.get("zone") in ("polar", "polar_high_altitude"):
                        return biome
            for biome in self.planet.biome_list:
                if biome.get("zone") == "elevation_high":
                    return biome
        elif elevation > 0.35:
            # Mid elevation — primary habitable biome
            for biome in self.planet.biome_list:
                if biome.get("zone") == "elevation_mid":
                    return biome
        elif elevation > 0.2:
            # Low elevation
            for biome in self.planet.biome_list:
                if biome.get("zone") in ("elevation_low", "erosion_channels",
                                          "transition_depressions"):
                    return biome
        else:
            # Very low — frost/ice accumulation zones
            for biome in self.planet.biome_list:
                if biome.get("zone") in ("polar", "elevation_low"):
                    return biome

        # Fallback: return the first biome marked as the mid-elevation default
        for biome in self.planet.biome_list:
            if biome.get("zone") == "elevation_mid":
                return biome
        return self.planet.biome_list[0]

    def get_temperature(self, x: int, y: int, current_tick: int = 0) -> float:
        """
        Calculate current temperature at (x, y) considering:
        - Biome base temperature range
        - Day/night cycle (for non-tidally-locked planets)
        - Local noise variation

        Returns temperature in Celsius.
        """
        biome = self.get_biome(x, y)
        temp_range = biome["temp_range_c"]
        t_min, t_max = temp_range[0], temp_range[1]

        if self.planet.tidally_locked:
            # Temperature is stable (no day/night cycle)
            # Add small noise variation
            noise_val = snoise2(
                x * 0.01, y * 0.01,
                octaves=2, base=self.seed + 10
            )
            t = (t_min + t_max) / 2.0 + noise_val * (t_max - t_min) * 0.2
        elif self.planet.day_night_cycle and self.planet.day_night_cycle.get("enabled"):
            # Sinusoidal day/night temperature oscillation
            cycle = self.planet.day_night_cycle
            cycle_len = self._physical_cycle_ticks(
                cycle["cycle_length_ticks"]
            )
            phase = (current_tick % cycle_len) / cycle_len * 2 * math.pi
            day_factor = (math.sin(phase) + 1.0) / 2.0  # 0=midnight, 1=noon

            # Check if biome has day/night variation override
            dnv = biome.get("day_night_variation")
            if dnv:
                day_temp = dnv.get("day_temp_c", temp_range)
                night_temp = dnv.get("night_temp_c", temp_range)
                t = (night_temp[0] + (day_temp[1] - night_temp[0]) * day_factor)
            else:
                t = t_min + (t_max - t_min) * day_factor

            # Noise perturbation
            noise_val = snoise2(
                x * 0.01 + current_tick * 0.001, y * 0.01,
                octaves=2, base=self.seed + 10
            )
            t += noise_val * 5.0
        else:
            # Simple day/night from planet surface data
            surface = self.planet.data["surface"]
            day_length = surface.get("day_length_hours", 24)
            ticks_per_day = max(
                1, int(round(day_length * 60 / self.tick_minutes))
            )
            if ticks_per_day > 0:
                phase = (current_tick % ticks_per_day) / ticks_per_day * 2 * math.pi
                day_factor = (math.sin(phase) + 1.0) / 2.0
            else:
                day_factor = 0.5
            t = t_min + (t_max - t_min) * day_factor

        return round(t, 1)

    # Molecular weights for atmospheric components (kg/mol)
    MOLECULAR_WEIGHTS = {
        "N2": 0.028,    # Nitrogen
        "O2": 0.032,    # Oxygen
        "CO2": 0.044,   # Carbon dioxide
        "Ar": 0.040,    # Argon
        "H2": 0.002,    # Hydrogen
        "He": 0.004,    # Helium
        "CH4": 0.016,   # Methane
        "H2O": 0.018,   # Water vapor
        "SO2": 0.064,   # Sulfur dioxide
        "NH3": 0.017,   # Ammonia
    }

    def _calc_mean_molecular_weight(self, composition: dict) -> float:
        """
        Calculate mean molecular weight of atmosphere from composition.
        composition: dict of {component: fraction}, e.g. {"N2": 0.78, "O2": 0.21}
        If empty/missing, defaults to Earth-like (M = 0.029).
        """
        if not composition:
            return 0.029  # Earth default
        total_weight = 0.0
        total_fraction = 0.0
        for component, fraction in composition.items():
            mw = self.MOLECULAR_WEIGHTS.get(component, 0.029)
            total_weight += mw * fraction
            total_fraction += fraction
        if total_fraction > 0:
            return total_weight / total_fraction
        return 0.029

    def get_atmospheric_pressure(self, x: int, y: int) -> float:
        """
        Calculate atmospheric pressure at (x, y) in kPa.

        Uses the full barometric formula: P = P0 * exp(-M*g*h / (R*T))
        where:
          M = mean molecular weight of atmosphere (kg/mol)
          g = surface gravity (m/s2)
          h = altitude (m)
          R = 8.314 J/(mol*K) = universal gas constant
          T = average atmospheric temperature (K)

        Scientific basis: International Standard Atmosphere (ISA)
        """
        atmo = self.planet.atmosphere
        if not atmo.get("present", False):
            return 0.0

        surface_pressure_atm = atmo.get("surface_pressure_atm")
        if surface_pressure_atm is None:
            surface_pressure_atm = 1.0
        surface_pressure_kpa = surface_pressure_atm * 101.325

        # Elevation: normalize 0-1 to estimated altitude in meters
        elevation = self.get_elevation(x, y)
        est_altitude_m = elevation * 5000  # 0-5000m range

        # Planet-specific parameters for barometric formula
        composition = atmo.get("composition", {})
        M = self._calc_mean_molecular_weight(composition)  # kg/mol
        g = self.planet.gravity_g * 9.81  # m/s2
        R = 8.314  # J/(mol*K)
        T = self.planet.data["surface"].get("equilibrium_temp_k", 288)  # K

        # P = P0 * exp(-Mgh / RT)
        coefficient = M * g / (R * T)
        pressure = surface_pressure_kpa * math.exp(-coefficient * est_altitude_m)
        return round(pressure, 2)

    def get_wind(self, x: int, y: int, current_tick: int = 0) -> dict:
        """
        Calculate wind direction and speed at (x, y) at the current tick.

        Uses noise fields for spatiotemporal variation:
        - Base wind pattern from planetary rotation/convection
        - Temporal variation from tick-based noise evolution
        - Terrain channeling: higher elevation = stronger wind

        Returns dict with:
        - direction_deg: wind direction in degrees (0=N, 90=E, 180=S, 270=W)
        - speed_kmh: wind speed in km/h
        - speed_modifier: multiplier for agent cold stress (1.0 = calm)
        - gust_factor: sudden speed spikes (1.0 = no gusts)

        Scientific basis:
        - Tidally locked planets have permanent day→night atmospheric flow
        - Ross 128b (slow rotation) has weak Coriolis → gentle winds
        - Proxima Centauri b (tidally locked) has strong terminator winds
        """
        # Base wind direction from large-scale noise
        dir_noise = snoise2(
            x * 0.002 + current_tick * 0.0005,
            y * 0.002,
            octaves=2, persistence=0.5, lacunarity=2.0,
            base=self.seed + 20
        )

        if self.planet.tidally_locked:
            # Dominant flow: from substellar (center) outward
            dx = x - self.center
            dy = y - self.center
            base_dir = math.degrees(math.atan2(dy, dx)) % 360
            direction = (base_dir + dir_noise * 45) % 360  # ±45° variation
        else:
            # Noise-driven direction with slow temporal evolution
            direction = (dir_noise + 1.0) * 180  # 0-360°

        # Wind speed from second noise channel
        speed_noise = snoise2(
            x * 0.004 + current_tick * 0.001,
            y * 0.004 + 300,
            octaves=3, persistence=0.6, lacunarity=2.0,
            base=self.seed + 21
        )
        speed_noise = (speed_noise + 1.0) / 2.0  # Normalize to 0-1

        # Base wind speed from planet hazards
        hazards = self.planet.data.get("hazards", [])
        if "extreme_wind" in hazards:
            base_speed = 60.0  # km/h
        elif "dust_storms" in hazards:
            base_speed = 40.0
        else:
            base_speed = 15.0

        # Elevation amplification (ridges and peaks are windier)
        elevation = self.get_elevation(x, y)
        elev_multiplier = 1.0 + elevation * 0.8  # Up to 1.8x at peak elevation

        # Atmosphere factor: no atmosphere = no wind
        if not self.planet.atmosphere.get("present", False):
            return {
                "direction_deg": 0,
                "speed_kmh": 0.0,
                "speed_modifier": 1.0,
                "gust_factor": 1.0,
            }

        speed = base_speed * speed_noise * elev_multiplier

        # Gust factor: occasional speed spikes
        gust_noise = snoise2(
            x * 0.01 + current_tick * 0.01, y * 0.01,
            octaves=1, base=self.seed + 22
        )
        gust = 1.0 + max(0, gust_noise) * 0.5  # Up to 1.5x gust

        # Wind chill modifier: speed > 20 km/h starts increasing cold stress
        effective_speed = speed * gust
        if effective_speed > 20:
            speed_modifier = 1.0 + (effective_speed - 20) * 0.02  # +2% per km/h above 20
        else:
            speed_modifier = 1.0

        return {
            "direction_deg": round(direction, 1),
            "speed_kmh": round(speed, 1),
            "speed_modifier": round(speed_modifier, 3),
            "gust_factor": round(gust, 2),
        }

    def get_day_phase(self, current_tick: int) -> dict:
        """
        Calculate the current day/night phase.

        Returns:
        - is_night: bool
        - phase: 0.0 (midnight) → 0.5 (noon) → 1.0 (midnight)
        - light_level: 0.0 (total darkness) → 1.0 (full daylight)
        - phase_name: "dawn", "day", "dusk", "night"

        For tidally locked planets:
        - Always "day" on day-side, always "night" on anti-stellar side
        - Terminator has permanent twilight
        """
        if self.planet.tidally_locked:
            # Tidally locked — no day/night cycle, phase depends on zone
            return {
                "is_night": False,  # Overridden per-cell by zone
                "phase": 0.5,
                "light_level": 1.0,
                "phase_name": "permanent",
                "cycle_length_ticks": 0,
            }

        # Get cycle length
        cycle = self.planet.day_night_cycle
        if cycle and cycle.get("enabled"):
            cycle_len = self._physical_cycle_ticks(
                cycle["cycle_length_ticks"]
            )
        else:
            surface = self.planet.data["surface"]
            day_length_h = surface.get("day_length_hours", 24)
            cycle_len = max(
                1, int(round(day_length_h * 60 / self.tick_minutes))
            )

        if cycle_len <= 0:
            return {
                "is_night": False,
                "phase": 0.5,
                "light_level": 1.0,
                "phase_name": "permanent_day",
                "cycle_length_ticks": 0,
            }

        phase = (current_tick % cycle_len) / cycle_len  # 0.0 - 1.0
        # Phase 0.0 = midnight, 0.25 = dawn, 0.5 = noon, 0.75 = dusk
        light_level = max(0.0, math.sin(phase * math.pi))  # Peak at 0.5

        if phase < 0.2 or phase >= 0.8:
            phase_name = "night"
        elif phase < 0.3:
            phase_name = "dawn"
        elif phase < 0.7:
            phase_name = "day"
        else:
            phase_name = "dusk"

        return {
            "is_night": light_level < 0.15,
            "phase": round(phase, 3),
            "light_level": round(light_level, 3),
            "phase_name": phase_name,
            "cycle_length_ticks": cycle_len,
        }

    def get_tidal_zone_light(self, x: int, y: int) -> dict:
        """
        For tidally locked planets, determine light level based on
        distance from substellar point.

        Substellar = permanent full daylight
        Terminator = permanent twilight
        Anti-stellar = permanent darkness
        """
        dx = x - self.center
        dy = y - self.center
        r_max = self.map_size / 2.0
        distance = math.sqrt(dx * dx + dy * dy) / r_max  # 0..1

        if distance < 0.4:
            return {"is_night": False, "light_level": 1.0, "phase_name": "permanent_day"}
        elif distance < 0.55:
            # Terminator twilight — light decreases across the band
            twilight = 1.0 - ((distance - 0.4) / 0.15)  # 1.0 → 0.0
            return {"is_night": twilight < 0.3, "light_level": round(twilight * 0.6, 2), "phase_name": "terminator_twilight"}
        else:
            # Night side
            return {"is_night": True, "light_level": 0.0, "phase_name": "permanent_night"}

    def get_gravity_construction_modifier(self) -> float:
        """
        Calculate construction time modifier based on planet gravity.

        Higher gravity = harder to move materials = slower construction.
        Lower gravity = easier to lift but harder to stabilize = slightly slower.

        Optimal construction gravity: ~0.5g (Moon-like)
        Earth-like (1.0g): baseline
        Super-Earth (1.3g+): significant penalty

        Returns multiplier applied to recipe base_duration_ticks.
        """
        g = self.planet.gravity_g
        if g <= 0.3:
            return 1.15  # Low-g instability penalty
        elif g <= 0.6:
            return 0.9   # Optimal range — easier lifting
        elif g <= 1.0:
            return 1.0   # Earth-baseline
        elif g <= 1.3:
            return 1.0 + (g - 1.0) * 1.0  # Linear penalty up to 1.3x
        else:
            return 1.3 + (g - 1.3) * 1.5  # Steeper penalty for super-heavy

    # Terrain traversal cost defaults by biome zone type
    # Based on real terrain mobility research (NASA EVA traverse studies)
    TRAVERSAL_COST_DEFAULTS = {
        "substellar": 1.3,            # Extreme heat, thermal cracking
        "day_side": 1.2,              # Hot, baked regolith
        "terminator": 1.4,            # Rocky, steep ridgelines
        "night_side_near": 1.5,       # Ice, low visibility
        "anti_stellar": 1.8,          # Deep frozen, crevasses
        "elevation_high": 1.6,        # Steep, thin air
        "elevation_mid": 1.0,         # Easiest terrain
        "elevation_low": 1.1,         # Soft ground
        "localized_volcanic": 1.5,    # Unstable, toxic
        "localized_geothermal": 1.3,  # Wet, slippery
        "localized_scattered": 1.2,   # Scattered debris
        "polar": 1.7,                 # Ice, wind exposure
        "polar_high_altitude": 1.8,   # Ice + altitude
        "erosion_channels": 1.3,      # Uneven, carved terrain
        "transition_depressions": 1.2, # Low, damp
    }

    def get_cell_info(self, x: int, y: int, current_tick: int = 0) -> dict:
        """
        Full terrain query for a single cell. Returns everything the
        simulation needs to know about this coordinate at the current tick.

        This is the primary API for the simulation engine.
        Includes: biome, elevation, temperature, pressure, wind, light,
        terrain traversal cost, and night temperature adjustments.
        """
        biome = self.get_biome(x, y)
        elevation = self.get_elevation(x, y)
        temperature = self.get_temperature(x, y, current_tick)
        pressure = self.get_atmospheric_pressure(x, y)
        wind = self.get_wind(x, y, current_tick)

        # Day/night phase
        if self.planet.tidally_locked:
            light = self.get_tidal_zone_light(x, y)
        else:
            light = self.get_day_phase(current_tick)

        # Night temperature drop — Stefan-Boltzmann radiative cooling model
        # Atmosphereless planets: no thermal blanket, rapid radiative heat loss
        # q = epsilon * sigma * T^4 (Stefan-Boltzmann law)
        # Practical model: night_delta = T_day * (1 - thermal_inertia) * solar_flux_factor
        if not self.planet.atmosphere.get("present", False):
            solar_flux = self.planet.data["surface"].get("solar_flux_relative_to_earth", 1.0)
            thermal_inertia = 0.15  # Regolith surface (Moon-like, low thermal mass)
            if light.get("is_night", False):
                # Full night: radiative cooling proportional to daytime heating
                # Moon: +127C day -> -173C night (delta ~300C at solar_flux=1.0)
                day_temp_k = max(1, temperature + 273.15)
                night_delta = abs(temperature) * (1.0 - thermal_inertia) * min(1.5, solar_flux * 0.8)
                temperature -= night_delta
            elif light.get("light_level", 1.0) < 0.3:
                # Twilight: partial radiative cooling scaled by darkness fraction
                darkness = 1.0 - light.get("light_level", 0.0) / 0.3
                night_delta = abs(temperature) * (1.0 - thermal_inertia) * min(1.5, solar_flux * 0.8) * darkness
                temperature -= night_delta
        elif light.get("is_night", False):
            # Atmospheric planets: greenhouse effect retains heat
            # Earth DTR (diurnal temperature range): 5-15C, mean ~10C
            # Thicker atmospheres retain more heat
            atm_pressure = self.planet.atmosphere.get("surface_pressure_atm")
            if atm_pressure is None:
                atm_pressure = 1.0
            # Higher pressure = more thermal retention = smaller night drop
            night_drop = 10.0 / max(0.5, atm_pressure)  # Earth (1 atm) = 10C, Venus (90 atm) = 0.1C
            temperature -= night_drop

        # Terrain traversal cost
        biome_zone = biome.get("zone", "elevation_mid")
        base_traversal = biome.get("traversal_cost",
                                   self.TRAVERSAL_COST_DEFAULTS.get(biome_zone, 1.2))
        hazard_modifiers = biome.get("hazard_modifiers", {}) or {}
        # A value below one is an empirically/scenario-derived speed fraction,
        # so its reciprocal belongs in travel time.  Keeping this in the cell
        # query makes route planning and actual motion use the same terrain.
        movement_fraction = max(
            0.1, min(1.0, float(
                hazard_modifiers.get("movement_speed_modifier", 1.0)
            ))
        )
        base_traversal /= movement_fraction
        # Wind increases traversal cost
        if wind["speed_kmh"] > 30:
            base_traversal *= 1.0 + (wind["speed_kmh"] - 30) * 0.01  # +1% per km/h above 30
        # Steep elevation increases cost
        if elevation > 0.7:
            base_traversal *= 1.0 + (elevation - 0.7) * 1.0  # Up to +30% at peak

        return {
            "x": x,
            "y": y,
            "biome_id": biome["id"],
            "biome_name": biome["name"],
            "elevation": round(elevation, 3),
            "temperature_c": round(temperature, 1),
            "pressure_kpa": pressure,
            "wind": wind,
            "light": light,
            "traversable": biome.get("traversable", True),
            "traversal_cost": round(base_traversal, 2),
            "survival_time_ticks": biome.get("survival_time_ticks_unprotected"),
            "hazard_modifiers": hazard_modifiers,
            "base_resources": biome.get("resources", {}),
            "gravity_construction_modifier": self.get_gravity_construction_modifier(),
        }

    def get_area(self, cx: int, cy: int, radius: int = 7,
                 current_tick: int = 0) -> list[dict]:
        """
        Get terrain info for a square area centered at (cx, cy).

        This produces the 'fog of war' view for an agent — the area
        they can perceive around their current position.

        Default radius=7 gives a 15×15 view (matching design doc spec).
        """
        cells = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = cx + dx
                y = cy + dy
                # Clamp to map bounds
                if 0 <= x < self.map_size and 0 <= y < self.map_size:
                    cells.append(self.get_cell_info(x, y, current_tick))
        return cells

    def find_spawn_location(
        self,
        required_resources: Optional[Iterable[str]] = None,
        operational_radius_cells: Optional[int] = None,
    ) -> tuple[int, int]:
        """
        Determine the optimal landing zone for the advance team.

        For tidally-locked planets: center of the terminator band.
        For others: search for a cell in the most habitable biome
        near the map center.

        ``required_resources`` enables a pre-flight landing-site study.  It
        uses only the resource associations of broad geological/biome units
        (the sort of information available to orbital spectroscopy), never
        the simulation's discovered-resource state or an agent-visible exact
        deposit coordinate.  Resources absent from the planet configuration
        are ignored.  Among safe landing candidates, sites with every
        planet-present requested resource inside ``operational_radius_cells``
        are preferred.

        Both arguments are optional so existing callers retain the original
        landing algorithm and its seeded RNG behaviour.
        """
        requested = self._normalise_required_resources(required_resources)
        present = self._planet_present_resources(requested)
        if not present or operational_radius_cells is None:
            return self._find_spawn_location_default()

        try:
            operational_radius = max(1, int(operational_radius_cells))
        except (TypeError, ValueError):
            return self._find_spawn_location_default()

        candidates = self._orbital_landing_candidates()
        if not candidates:
            return self._find_spawn_location_default()

        resource_cells = self._orbital_resource_samples(
            candidates, present, operational_radius
        )
        bucket_size = max(1, operational_radius)
        resource_buckets: dict[str, dict[tuple[int, int], list[tuple[int, int]]]] = {
            resource: {} for resource in present
        }
        for resource, coords in resource_cells.items():
            buckets = resource_buckets[resource]
            for x, y in coords:
                buckets.setdefault((x // bucket_size, y // bucket_size), []).append((x, y))

        best_pos: tuple[int, int] | None = None
        best_rank: tuple | None = None
        required_count = len(present)
        for x, y in candidates:
            habitability = self._landing_habitability_score(x, y)
            if habitability is None:
                continue

            distances = []
            for resource in present:
                distance = self._nearest_orbital_sample_distance(
                    x,
                    y,
                    resource_buckets[resource],
                    bucket_size,
                    operational_radius,
                )
                if distance is not None:
                    distances.append(distance)

            covered = len(distances)
            all_covered = covered == required_count
            # Resource coverage is a hard mission-feasibility constraint.
            # Habitability remains the primary discriminator once all
            # critical resources are reachable.  Coordinate tie-breakers
            # make the result repeatable without consuming the world's RNG.
            rank = (
                1 if all_covered else 0,
                covered,
                round(habitability, 6),
                -max(distances, default=operational_radius + 1),
                -sum(distances),
                -abs(x - self.center) - abs(y - self.center),
                -x,
                -y,
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_pos = (x, y)

        return best_pos if best_pos is not None else self._find_spawn_location_default()

    @staticmethod
    def _normalise_required_resources(
        required_resources: Optional[Iterable[str]],
    ) -> tuple[str, ...]:
        """Return a stable, de-duplicated resource list."""
        if required_resources is None:
            return ()
        if isinstance(required_resources, str):
            required_resources = (required_resources,)
        return tuple(sorted({str(resource) for resource in required_resources if resource}))

    def _planet_present_resources(self, requested: Iterable[str]) -> tuple[str, ...]:
        """Filter mission requirements against coarse planetary geology."""
        available = {
            resource
            for biome in self.planet.biome_list
            for resource in biome.get("resources", {})
        }
        return tuple(resource for resource in requested if resource in available)

    def _find_spawn_location_default(self) -> tuple[int, int]:
        """Original landing algorithm retained for API/backward compatibility."""
        if self.planet.tidally_locked:
            # Terminator is at ~47.5% of radius from center
            r = int(self.center * 0.475)
            angle = self.rng.random() * 2 * math.pi
            x = int(self.center + r * math.cos(angle))
            y = int(self.center + r * math.sin(angle))
            return (x, y)
        else:
            # Score candidates by habitability:
            # 1. Prefer biomes without "polar", "frost" in name
            # 2. Prefer temperature closest to 10°C (comfortable)
            # 3. Search near center
            best_pos = (self.center, self.center)
            best_score = -9999
            search_radius = 150
            
            for _ in range(300):
                x = self.center + int(self.rng.integers(-search_radius, search_radius))
                y = self.center + int(self.rng.integers(-search_radius, search_radius))
                x = max(0, min(self.map_size - 1, x))
                y = max(0, min(self.map_size - 1, y))
                
                biome = self.get_biome(x, y)
                temp_range = biome.get("temp_range_c", [-50, -30])
                avg_temp = (temp_range[0] + temp_range[1]) / 2.0
                
                # Reject non-traversable and deadly biomes
                if not biome.get("traversable", True):
                    continue
                if "frost_cap" in biome["id"] or "polar" in biome.get("zone", ""):
                    continue
                
                # Score: penalize deviation from 10°C, reward habitable zones
                temp_score = -abs(avg_temp - 10.0) * 0.5
                zone_bonus = {
                    "elevation_mid": 10.0,
                    "localized_geothermal": 5.0,
                    "elevation_low": 2.0,
                }.get(biome.get("zone", ""), 0.0)
                
                score = temp_score + zone_bonus
                if score > best_score:
                    best_score = score
                    best_pos = (x, y)
            
            return best_pos

    def _landing_habitability_score(self, x: int, y: int) -> float | None:
        """Score a traversable landing cell using the legacy safety criteria."""
        biome = self.get_biome(x, y)
        if not biome.get("traversable", True):
            return None
        biome_id = biome.get("id", "")
        zone = biome.get("zone", "")
        if "frost_cap" in biome_id or "polar" in zone:
            return None

        temp_range = biome.get("temp_range_c", [-50, -30])
        avg_temp = (temp_range[0] + temp_range[1]) / 2.0
        temp_score = -abs(avg_temp - 10.0) * 0.5
        zone_bonus = {
            "terminator": 12.0,
            "elevation_mid": 10.0,
            "localized_geothermal": 5.0,
            "elevation_low": 2.0,
        }.get(zone, 0.0)
        return temp_score + zone_bonus

    def _orbital_landing_candidates(self) -> list[tuple[int, int]]:
        """Generate deterministic safe-site candidates for orbital analysis.

        Candidate generation is intentionally independent from ``self.rng``:
        asking mission-planning questions must not perturb later procedural
        randomness.  The grid spacing represents a coarse orbital map rather
        than metre-scale knowledge.
        """
        candidates: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        def add(x: int, y: int):
            pos = (
                max(0, min(self.map_size - 1, int(x))),
                max(0, min(self.map_size - 1, int(y))),
            )
            if pos not in seen:
                seen.add(pos)
                candidates.append(pos)

        if self.planet.tidally_locked:
            terminator = next(
                (zone for zone in self.radial_biomes
                 if self.planet.biomes[zone["biome_id"]].get("zone") == "terminator"),
                None,
            )
            if terminator is None:
                return []
            r_min = int(terminator["r_min"]) + 1
            r_max = max(r_min, int(terminator["r_max"]) - 1)
            radial_step = max(1, (r_max - r_min) // 12)
            radii = list(range(r_min, r_max + 1, radial_step))
            if radii[-1] != r_max:
                radii.append(r_max)
            # A seed-derived angular offset avoids privileging a map axis but
            # remains identical across repeated calls.
            angle_offset = (self.seed % 360) * math.pi / 180.0
            for radius in radii:
                for angle_index in range(72):
                    angle = angle_offset + angle_index * (2.0 * math.pi / 72.0)
                    add(
                        self.center + radius * math.cos(angle),
                        self.center + radius * math.sin(angle),
                    )
            return candidates

        local_rng = np.random.default_rng(self.seed)
        search_radius = 150
        # Keep the legacy statistical candidate set, then add a low-resolution
        # orbital survey lattice so a small random sample cannot miss an
        # otherwise valid landing corridor.
        for _ in range(300):
            add(
                self.center + int(local_rng.integers(-search_radius, search_radius)),
                self.center + int(local_rng.integers(-search_radius, search_radius)),
            )
        lattice_step = 15
        for y in range(self.center - search_radius, self.center + search_radius + 1,
                       lattice_step):
            for x in range(self.center - search_radius, self.center + search_radius + 1,
                           lattice_step):
                add(x, y)
        add(self.center, self.center)
        return candidates

    def _orbital_resource_samples(
        self,
        candidates: list[tuple[int, int]],
        resources: tuple[str, ...],
        operational_radius: int,
    ) -> dict[str, list[tuple[int, int]]]:
        """Sample broad geological units at orbital-survey resolution."""
        x_min = max(0, min(x for x, _ in candidates) - operational_radius)
        x_max = min(self.map_size - 1, max(x for x, _ in candidates) + operational_radius)
        y_min = max(0, min(y for _, y in candidates) - operational_radius)
        y_max = min(self.map_size - 1, max(y for _, y in candidates) + operational_radius)

        # One cell is 100 m in the simulation.  A 0.4--1.2 km orbital
        # reconnaissance grid is fine enough for broad units while avoiding a
        # costly metre-by-metre pre-generation of the procedural world.
        stride = max(4, min(12, operational_radius // 8 or 4))
        samples: dict[str, list[tuple[int, int]]] = {
            resource: [] for resource in resources
        }

        def record(x: int, y: int):
            biome_resources = self.get_biome(x, y).get("resources", {})
            for resource in resources:
                if resource in biome_resources:
                    samples[resource].append((x, y))

        for y in range(y_min, y_max + 1, stride):
            for x in range(x_min, x_max + 1, stride):
                record(x, y)

        # Candidate cells themselves are exact observations in the coarse map
        # and protect small-radius calls from sampling-phase artefacts.
        for x, y in candidates:
            record(x, y)
        return samples

    @staticmethod
    def _nearest_orbital_sample_distance(
        x: int,
        y: int,
        buckets: dict[tuple[int, int], list[tuple[int, int]]],
        bucket_size: int,
        max_distance: int,
    ) -> int | None:
        """Return nearest Chebyshev distance within the operational radius."""
        bx, by = x // bucket_size, y // bucket_size
        nearest: int | None = None
        for bucket_y in range(by - 1, by + 2):
            for bucket_x in range(bx - 1, bx + 2):
                for rx, ry in buckets.get((bucket_x, bucket_y), ()):
                    distance = max(abs(rx - x), abs(ry - y))
                    if distance <= max_distance and (
                        nearest is None or distance < nearest
                    ):
                        nearest = distance
        return nearest
