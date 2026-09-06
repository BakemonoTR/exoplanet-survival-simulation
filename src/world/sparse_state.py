"""
Sparse state management for the simulation world.

Design principle: The world is procedurally generated and deterministic.
At tick 0, every cell's state can be computed from the noise function.
This module stores ONLY changes from the procedural baseline:
- Resource nodes that have been partially/fully depleted
- Structures placed by agents
- Environmental modifications (craters from events, etc.)

This keeps memory usage proportional to the number of changes, not
the map size (2000×2000 = 4M cells would be ~1.6GB if fully stored).

Implementation: dict-backed with (x, y) tuple keys.
"""

import json
import math
from typing import Optional
from dataclasses import dataclass, field, asdict

import numpy as np

from .generator import WorldGenerator, PlanetConfig


@dataclass
class ResourceNode:
    """
    A mineable resource deposit at a specific location.

    Resource nodes are spawned during world initialization based on
    biome resource probabilities. Once depleted, they do NOT respawn
    (permanent scarcity pressure).
    """
    node_id: str
    material: str
    position: tuple[int, int]
    quantity: int             # Current remaining units
    max_quantity: int         # Initial quantity (for depletion tracking)
    gather_rate: float        # Base units per tick (modified by agent stats)
    gather_difficulty: float  # Multiplier (biome/terrain penalty)
    depleted: bool = False

    # Regeneration (for geothermal/volcanic water ice deposits)
    regeneration_rate: float = 0.0  # Units per tick regenerated (0 = no regen)
    max_regen_quantity: int = 0     # Max quantity the node can regenerate to

    def tick_regeneration(self):
        """
        Apply per-tick resource regeneration.
        Only water_ice near geothermal sources regenerates.
        Mineral deposits do NOT regenerate (permanent scarcity).
        """
        if self.regeneration_rate <= 0 or self.max_regen_quantity <= 0:
            return
        if self.quantity < self.max_regen_quantity:
            self.quantity = min(
                self.max_regen_quantity,
                self.quantity + int(max(1, self.regeneration_rate))
            )
            if self.quantity > 0:
                self.depleted = False

    def extract(self, agent_strength: int, agent_has_tools: bool,
                ticks: int = 1) -> int:
        """
        Extract resources from this node.

        Returns the number of units actually extracted.
        Modifies the node's quantity in-place.
        """
        if self.depleted or self.quantity <= 0:
            return 0

        # Effective gather rate: base * strength modifier * tool modifier / difficulty
        strength_mod = 1.0 + 0.08 * (agent_strength - 5)  # 5 is baseline
        tool_mod = 1.3 if agent_has_tools else 1.0
        effective_rate = self.gather_rate * strength_mod * tool_mod / self.gather_difficulty

        extracted = min(int(effective_rate * ticks), self.quantity)
        extracted = max(1, extracted)  # Always extract at least 1 if node has resources

        self.quantity -= extracted
        if self.quantity <= 0:
            self.quantity = 0
            self.depleted = True

        return extracted


@dataclass
class PlacedStructure:
    """
    A structure built by an agent at a specific location.

    Structures degrade over time and require periodic maintenance.
    Environmental events (flares, storms, quakes) deal direct damage.
    Unmaintained structures lose operational efficiency and may break down.
    """
    structure_id: str         # Matches recipe output structure_id
    position: tuple[int, int]
    builder_agent_id: str
    build_tick: int           # Tick when construction completed
    recipe_id: str            # Which recipe was used
    health: float = 1.0       # 0.0 = destroyed, 1.0 = pristine
    condition: float = 1.0    # Operational efficiency (degrades without maintenance)
    powered: bool = False     # Whether connected to power source
    operational: bool = True  # Whether functioning (health > 0 AND condition > 0.2)

    # Real-time Construction Progression (NASA Shantiye Prosedürü)
    under_construction: bool = False
    construction_progress: float = 1.0       # 0.0 = just started, 1.0 = completed
    required_construction_ticks: int = 1     # Total ticks needed to finish
    construction_ticks_elapsed: int = 0

    # Solar Panel Regolith Dust Fouling (Apollo/Curiosity/InSight empirical data)
    dust_fouling_level: float = 0.0          # 0.0 = clean, 1.0 = heavy dust layer (80% power loss)

    # Maintenance tracking
    last_maintenance_tick: int = 0       # Last tick maintenance was performed
    maintenance_interval_ticks: int = 0  # From recipe (0 = no maintenance needed)
    maintenance_materials: dict = field(default_factory=dict)  # Materials needed for repair
    degradation_per_missed: float = 0.1  # Condition loss per missed maintenance cycle

    # Colony contribution (copied from recipe on build)
    colony_contribution: dict = field(default_factory=dict)
    effects: dict = field(default_factory=dict)

    # Light emission (for night visibility system)
    light_radius: int = 0  # 0 = no light, >0 = illumination radius in cells

    # Fuel tracking (for campfire, stone_furnace, forge)
    # Structures with fuel_consumption_*_per_tick in effects consume fuel each tick.
    # When fuel runs out, the structure deactivates (no heat, no light, no smelting).
    fuel_reserve: float = 0.0     # Current fuel units
    fuel_type: str = ""           # e.g., "sulfur"
    fuel_consumption_per_tick: float = 0.0  # From recipe effects
    active: bool = True           # False when fuel runs out (distinct from operational)

    def tick_fuel_consumption(self):
        """
        Consume fuel each tick. Deactivate structure when fuel runs out.
        Structures without fuel consumption are always active.

        Returns dict of fuel status changes.
        """
        if self.fuel_consumption_per_tick <= 0 or not self.operational:
            return {"fuel_consumed": False}

        if self.fuel_reserve <= 0:
            self.active = False
            return {"fuel_consumed": False, "deactivated": True, "reason": f"no_{self.fuel_type}"}

        consumed = min(self.fuel_reserve, self.fuel_consumption_per_tick)
        self.fuel_reserve -= consumed
        self.active = True

        if self.fuel_reserve <= 0:
            self.active = False
            return {"fuel_consumed": True, "amount": consumed, "deactivated": True}

        return {"fuel_consumed": True, "amount": consumed, "fuel_remaining": round(self.fuel_reserve, 2)}

    def refuel(self, amount: float):
        """Add fuel to the structure. Called when agent delivers fuel."""
        self.fuel_reserve += amount
        if self.fuel_reserve > 0 and self.operational:
            self.active = True

    def tick_maintenance(self, current_tick: int):
        """
        Check and apply maintenance degradation.
        Called once per tick by the simulation engine.
        """
        if self.maintenance_interval_ticks <= 0 or self.health <= 0:
            return

        ticks_since = current_tick - self.last_maintenance_tick
        if ticks_since >= self.maintenance_interval_ticks:
            # Maintenance overdue — degrade condition
            missed_cycles = ticks_since // self.maintenance_interval_ticks
            # Only degrade once per overdue period (not retroactively)
            if missed_cycles > 0 and ticks_since % self.maintenance_interval_ticks == 0:
                self.condition = max(0.0, self.condition - self.degradation_per_missed)

        # Structure becomes non-operational if condition drops too low
        if self.condition <= 0.2:
            self.operational = False

    def apply_damage(self, damage: float, damage_source: str = "unknown") -> dict:
        """
        Apply damage to the structure from environmental events.

        Args:
            damage: 0.0-1.0 fraction of max health
            damage_source: flare, storm, quake, etc.

        Returns dict with damage report.
        """
        old_health = self.health
        self.health = max(0.0, self.health - damage)

        result = {
            "structure_id": self.structure_id,
            "damage_taken": round(damage, 3),
            "health_before": round(old_health, 2),
            "health_after": round(self.health, 2),
            "source": damage_source,
            "destroyed": self.health <= 0,
        }

        if self.health <= 0:
            self.operational = False
            self.condition = 0.0

        return result

    def repair(self, repair_amount: float = 0.3, full_maintenance: bool = True,
               current_tick: int = 0):
        """
        Repair structure health and/or perform maintenance.

        Args:
            repair_amount: health restored (0.0-1.0)
            full_maintenance: if True, resets maintenance timer and condition
            current_tick: current simulation tick
        """
        self.health = min(1.0, self.health + repair_amount)

        if full_maintenance:
            self.last_maintenance_tick = current_tick
            self.condition = min(1.0, self.condition + 0.5)

        # Restore operational status if health is sufficient
        if self.health > 0 and self.condition > 0.2:
            self.operational = True

    def get_effective_output(self) -> dict:
        """
        Get effects scaled by current condition.
        A structure at 50% condition produces 50% of its rated output.
        """
        if not self.operational:
            return {}

        scaled = {}
        for key, value in self.effects.items():
            if isinstance(value, (int, float)) and key != "notes":
                scaled[key] = value * self.condition
            else:
                scaled[key] = value
        return scaled


@dataclass
class CellModification:
    """
    Tracks all modifications to a single cell from its procedural baseline.

    A cell may have:
    - Resource nodes (spawned during init, depleted over time)
    - Structures (placed by agents)
    - Environmental flags (radiation contamination, fire, crater, etc.)
    """
    position: tuple[int, int]
    resource_nodes: list[ResourceNode] = field(default_factory=list)
    structures: list[PlacedStructure] = field(default_factory=list)
    environmental_flags: dict = field(default_factory=dict)
    # Examples: {"irradiated": True, "crater": True, "on_fire": False}


class SparseWorldState:
    """
    Dict-backed sparse state layer over the procedural world.

    Only stores cells that have been modified from their procedural
    baseline. Unmodified cells are always computed from the WorldGenerator.

    Primary API:
    - query(x, y, tick) → full cell state (procedural + modifications)
    - place_structure(x, y, structure) → add structure to cell
    - spawn_resource_node(x, y, node) → add resource node to cell
    - extract_resource(x, y, material, agent_stats) → mine resources
    """

    def __init__(self, world_gen: WorldGenerator):
        self.world_gen = world_gen
        self.planet = world_gen.planet
        self._modifications: dict[tuple[int, int], CellModification] = {}
        self._resource_index: dict[str, list[tuple[int, int]]] = {}
        self._structure_index: dict[str, list[tuple[int, int]]] = {}
        self._total_structures: dict[str, int] = {}
        # Physical Base Logistics: Central Storage Depot at Landing Zone
        self.central_depot_inventory: dict[str, int] = {
            "regolith": 20,
            "basalt": 10,
            "reduced_iron_ingot": 4,
            "oxygen_canisters": 8,
            "ration_packs": 15,
            "water_packs": 20,
            "vacuum_gasket_seal": 6,
        }

    def get_modification(self, x: int, y: int) -> Optional[CellModification]:
        """Get modifications for a cell, or None if unmodified."""
        return self._modifications.get((x, y))

    def ensure_cell(self, x: int, y: int) -> CellModification:
        """Get or create a CellModification for this position."""
        key = (x, y)
        if key not in self._modifications:
            self._modifications[key] = CellModification(position=key)
        return self._modifications[key]

    def query(self, x: int, y: int, current_tick: int = 0) -> dict:
        """
        Full state query for a cell. Merges procedural terrain with
        any modifications (resource nodes, structures, environmental flags).

        This is the single source of truth for 'what is at (x, y) right now?'
        """
        # Get procedural baseline
        cell = self.world_gen.get_cell_info(x, y, current_tick)

        # Overlay modifications
        mod = self.get_modification(x, y)
        if mod:
            cell["resource_nodes"] = [
                {
                    "node_id": n.node_id,
                    "material": n.material,
                    "quantity": n.quantity,
                    "max_quantity": n.max_quantity,
                    "depleted": n.depleted,
                    "gather_rate": n.gather_rate,
                    "gather_difficulty": n.gather_difficulty,
                }
                for n in mod.resource_nodes
            ]
            cell["structures"] = [
                {
                    "structure_id": s.structure_id,
                    "builder": s.builder_agent_id,
                    "health": s.health,
                    "condition": s.condition,
                    "powered": s.powered,
                    "operational": s.operational,
                    "effects": s.get_effective_output(),
                    "colony_contribution": s.colony_contribution,
                    "light_radius": s.light_radius,
                    "maintenance_overdue": (
                        (current_tick - s.last_maintenance_tick) >= s.maintenance_interval_ticks
                        if s.maintenance_interval_ticks > 0 else False
                    ),
                }
                for s in mod.structures
            ]
            cell["environmental_flags"] = mod.environmental_flags
        else:
            cell["resource_nodes"] = []
            cell["structures"] = []
            cell["environmental_flags"] = {}

        return cell

    def spawn_resource_node(self, x: int, y: int, node: ResourceNode):
        """Add a resource node to the world at the given position."""
        cell = self.ensure_cell(x, y)
        cell.resource_nodes.append(node)

        # Update index
        if node.material not in self._resource_index:
            self._resource_index[node.material] = []
        self._resource_index[node.material].append((x, y))

    def place_structure(self, x: int, y: int, structure: PlacedStructure):
        """Place a completed structure at the given position."""
        cell = self.ensure_cell(x, y)
        cell.structures.append(structure)

        # Update index
        sid = structure.structure_id
        if sid not in self._structure_index:
            self._structure_index[sid] = []
        self._structure_index[sid].append((x, y))
        self._total_structures[sid] = self._total_structures.get(sid, 0) + 1

    def extract_resource(self, x: int, y: int, material: str,
                         agent_strength: int, agent_has_tools: bool,
                         ticks: int = 1) -> int:
        """
        Attempt to extract a specific material from resource nodes at (x, y).

        Returns the amount extracted (0 if no matching non-depleted nodes).
        """
        mod = self.get_modification(x, y)
        if not mod:
            return 0

        total_extracted = 0
        for node in mod.resource_nodes:
            if node.material == material and not node.depleted:
                extracted = node.extract(agent_strength, agent_has_tools, ticks)
                total_extracted += extracted
                break  # Extract from one node at a time

        return total_extracted

    def find_nearest_resource(self, x: int, y: int, material: str,
                              max_distance: float = 200) -> Optional[tuple[int, int]]:
        """Find the nearest non-depleted resource node of a given material."""
        positions = self._resource_index.get(material, [])
        best_pos = None
        best_dist = max_distance + 1

        for pos in positions:
            mod = self.get_modification(pos[0], pos[1])
            if mod:
                for node in mod.resource_nodes:
                    if node.material == material and not node.depleted:
                        dist = math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
                        if dist < best_dist:
                            best_dist = dist
                            best_pos = pos
                        break

        return best_pos if best_dist <= max_distance else None

    def find_nearest_structure(self, x: int, y: int, structure_id: str,
                                max_distance: float = 200) -> Optional[tuple[int, int]]:
        """Find the nearest operational structure of a given type."""
        positions = self._structure_index.get(structure_id, [])
        best_pos = None
        best_dist = max_distance + 1

        for pos in positions:
            mod = self.get_modification(pos[0], pos[1])
            if mod:
                for s in mod.structures:
                    if s.structure_id == structure_id and s.operational:
                        dist = math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
                        if dist < best_dist:
                            best_dist = dist
                            best_pos = pos
                        break

        return best_pos if best_dist <= max_distance else None

    def get_structures_in_radius(self, x: int, y: int,
                                  radius: float) -> list[PlacedStructure]:
        """Get all structures within a radius of a position."""
        results = []
        for key, mod in self._modifications.items():
            if mod.structures:
                dist = math.sqrt((key[0] - x) ** 2 + (key[1] - y) ** 2)
                if dist <= radius:
                    results.extend(mod.structures)
        return results

    def get_colony_capacity(self) -> dict:
        """
        Calculate total colony capacity across all built structures.

        Returns dict mapping colony sub-target names to current capacity values.
        Used by the colony_score module.
        """
        capacity = {}
        for key, mod in self._modifications.items():
            for structure in mod.structures:
                if structure.operational and structure.colony_contribution:
                    for target, value in structure.colony_contribution.items():
                        if target == "notes":
                            continue
                        if isinstance(value, (int, float)):
                            capacity[target] = capacity.get(target, 0) + value
        return capacity

    def count_modified_cells(self) -> int:
        """Return the number of cells with modifications (memory usage indicator)."""
        return len(self._modifications)

    def get_stats(self) -> dict:
        """Summary statistics for the current world state."""
        total_nodes = 0
        depleted_nodes = 0
        total_structures = 0

        for mod in self._modifications.values():
            total_nodes += len(mod.resource_nodes)
            depleted_nodes += sum(1 for n in mod.resource_nodes if n.depleted)
            total_structures += len(mod.structures)

        return {
            "modified_cells": self.count_modified_cells(),
            "total_resource_nodes": total_nodes,
            "depleted_resource_nodes": depleted_nodes,
            "active_resource_nodes": total_nodes - depleted_nodes,
            "total_structures": total_structures,
            "structure_counts": dict(self._total_structures),
        }

    def tick_world(self, current_tick: int, event_damage: dict = None):
        """
        Per-tick world state update. Called by the simulation engine.

        Handles:
        - Structure maintenance degradation
        - Resource regeneration
        - Event damage to structures (flares, storms, quakes)

        Args:
            current_tick: Current simulation tick
            event_damage: Optional dict {damage_source: damage_amount}
                          applied to all exposed structures
        """
        if event_damage is None:
            event_damage = {}

        damage_reports = []

        for key, mod in self._modifications.items():
            # Tick resource regeneration
            for node in mod.resource_nodes:
                node.tick_regeneration()

            # Tick structure maintenance and apply event damage
            for structure in mod.structures:
                structure.tick_maintenance(current_tick)

                # Apply event damage (flare, storm, quake)
                for source, damage in event_damage.items():
                    # Sheltered structures take reduced damage
                    effective_damage = damage
                    flare_red = structure.effects.get("flare_damage_reduction", 0)
                    if source == "flare" and flare_red > 0:
                        effective_damage *= (1.0 - flare_red)

                    if effective_damage > 0.01:
                        report = structure.apply_damage(effective_damage, source)
                        if report["damage_taken"] > 0:
                            damage_reports.append(report)

        return damage_reports

    def get_light_sources(self, x: int, y: int, radius: int = 15) -> list[dict]:
        """
        Find all light-emitting structures within radius of a position.
        Used by the night visibility system.

        Returns list of {position, light_radius, structure_id}.
        """
        lights = []
        for key, mod in self._modifications.items():
            dist = math.sqrt((key[0] - x) ** 2 + (key[1] - y) ** 2)
            if dist <= radius:
                for s in mod.structures:
                    if s.light_radius > 0 and s.operational:
                        lights.append({
                            "position": key,
                            "light_radius": s.light_radius,
                            "structure_id": s.structure_id,
                            "distance": round(dist, 1),
                        })
        return lights

    def get_structures_needing_maintenance(self, current_tick: int) -> list[dict]:
        """
        Find all structures that are overdue for maintenance.
        Used by the strategic planning layer to prioritize repair tasks.
        """
        overdue = []
        for key, mod in self._modifications.items():
            for s in mod.structures:
                if (s.maintenance_interval_ticks > 0
                        and s.operational
                        and (current_tick - s.last_maintenance_tick) >= s.maintenance_interval_ticks):
                    overdue.append({
                        "structure_id": s.structure_id,
                        "position": key,
                        "condition": round(s.condition, 2),
                        "ticks_overdue": current_tick - s.last_maintenance_tick - s.maintenance_interval_ticks,
                        "repair_materials": s.maintenance_materials,
                    })
        return overdue


class ResourceSpawner:
    """
    Spawns resource nodes across the world based on biome resource availability.

    Resource density is controlled by:
    - Biome resource ratings (sparse/moderate/rich/abundant/very_abundant)
    - Planet scarcity factor (0.25 for Proxima → 0.8 for Ross 128b)
    - Random variation within bounds

    Resources are spawned in a deterministic pattern (seed-based) so the
    same seed always produces the same resource distribution.
    """

    # Resource rating → (nodes per 100×100 area, quantity per node range)
    DENSITY_MAP = {
        "sparse":        (2,  (8, 20)),
        "moderate":      (4,  (15, 35)),
        "rich":          (6,  (25, 55)),
        "abundant":      (8,  (35, 75)),
        "very_abundant": (12, (50, 100)),
        "very_rich":     (8,  (40, 80)),
    }

    # Gather rate and difficulty by material type
    MATERIAL_PROPERTIES = {
        "regolith":        {"gather_rate": 5.0, "difficulty": 0.8},
        "iron_ore":        {"gather_rate": 2.5, "difficulty": 1.5},
        "silica_sand":     {"gather_rate": 3.0, "difficulty": 1.2},
        "water_ice":       {"gather_rate": 4.0, "difficulty": 1.0},
        "sulfur":          {"gather_rate": 3.5, "difficulty": 1.1},
        "calcite":         {"gather_rate": 3.0, "difficulty": 1.3},
        "olivine":         {"gather_rate": 2.0, "difficulty": 1.6},
        "graphite":        {"gather_rate": 2.5, "difficulty": 1.4},
        "basalt":          {"gather_rate": 3.5, "difficulty": 1.0},
        "chalcopyrite_ore": {"gather_rate": 1.5, "difficulty": 2.0},
    }

    def __init__(self, world_gen: WorldGenerator, sparse_state: SparseWorldState,
                 seed: int = 42):
        self.world_gen = world_gen
        self.sparse_state = sparse_state
        self.planet = world_gen.planet
        self.rng = np.random.default_rng(seed)
        self.scarcity = self.planet.scarcity_factor
        self._node_counter = 0

    def _next_node_id(self, material: str) -> str:
        """Generate a unique node ID."""
        self._node_counter += 1
        return f"{material}_{self._node_counter:04d}"

    def spawn_all(self):
        """
        Spawn resource nodes across the entire map.

        Divides the map into 100×100 chunks and spawns nodes in each chunk
        based on the biome at the chunk center.
        """
        chunk_size = 100
        map_size = self.world_gen.map_size

        for chunk_y in range(0, map_size, chunk_size):
            for chunk_x in range(0, map_size, chunk_size):
                # Determine biome at chunk center
                cx = chunk_x + chunk_size // 2
                cy = chunk_y + chunk_size // 2
                cx = min(cx, map_size - 1)
                cy = min(cy, map_size - 1)

                biome = self.world_gen.get_biome(cx, cy)
                resources = biome.get("resources", {})

                for material, rating in resources.items():
                    if rating not in self.DENSITY_MAP:
                        continue

                    nodes_per_chunk, qty_range = self.DENSITY_MAP[rating]

                    # Apply scarcity factor
                    adjusted_nodes = max(1, int(nodes_per_chunk * self.scarcity))

                    for _ in range(adjusted_nodes):
                        # Random position within chunk
                        nx = chunk_x + int(self.rng.integers(0, chunk_size))
                        ny = chunk_y + int(self.rng.integers(0, chunk_size))
                        nx = min(nx, map_size - 1)
                        ny = min(ny, map_size - 1)

                        # Random quantity within range, scaled by scarcity
                        base_qty = int(self.rng.integers(qty_range[0], qty_range[1] + 1))
                        qty = max(5, int(base_qty * self.scarcity))

                        # Get material properties
                        props = self.MATERIAL_PROPERTIES.get(material, {
                            "gather_rate": 2.0, "difficulty": 1.5
                        })

                        node = ResourceNode(
                            node_id=self._next_node_id(material),
                            material=material,
                            position=(nx, ny),
                            quantity=qty,
                            max_quantity=qty,
                            gather_rate=props["gather_rate"],
                            gather_difficulty=props["difficulty"],
                        )

                        self.sparse_state.spawn_resource_node(nx, ny, node)

    def spawn_starting_crates(self, landing_x: int, landing_y: int,
                               rng: Optional[np.random.Generator] = None):
        """
        Spawn 1-3 random supply crates near the landing zone.

        These add early exploration incentive and replayability.
        Crates are deployed via parachute and scatter around the LZ.
        """
        if rng is None:
            rng = self.rng

        num_crates = int(rng.integers(1, 4))  # 1-3 crates

        material_options = ["regolith", "iron_ore", "silica_sand", "water_ice", "basalt"]

        for i in range(num_crates):
            # Random position within 50 units of crash
            angle = rng.random() * 2 * math.pi
            dist = 15 + rng.random() * 35  # 15-50 units away
            cx = int(landing_x + dist * math.cos(angle))
            cy = int(landing_y + dist * math.sin(angle))
            cx = max(0, min(self.world_gen.map_size - 1, cx))
            cy = max(0, min(self.world_gen.map_size - 1, cy))

            material = rng.choice(material_options)
            qty = int(rng.integers(10, 31))

            props = self.MATERIAL_PROPERTIES.get(material, {
                "gather_rate": 3.0, "difficulty": 1.0
            })

            node = ResourceNode(
                node_id=f"crate_{i}_{material}",
                material=material,
                position=(cx, cy),
                quantity=qty,
                max_quantity=qty,
                gather_rate=props["gather_rate"] * 2,  # Crates are easier to extract
                gather_difficulty=0.5,  # Pre-packaged, easy access
            )

            self.sparse_state.spawn_resource_node(cx, cy, node)
