"""Auditable surface-vehicle state machines for the precursor mission.

The excavators are RASSOR/IPEx-class logistics machines, not omniscient ore
finders.  A job may be dispatched only to a verified, physically exposed face;
the caller remains responsible for geology, terrain and inventory conservation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Callable, Optional


@dataclass
class FleetJob:
    id: str
    resource: str
    target_x: int
    target_y: int
    dispatched_tick: int
    dispatched_by: str
    verified_exposed_face: bool
    protected_ground: bool = False
    # The bucket wheel can only remove the material that is physically
    # exposed now.  ``objective_resource`` records the detector-confirmed
    # deeper seam that this layer-removal job is supporting, without letting
    # the vehicle discover or extract that seam early.
    objective_resource: str = ""
    rl_state_key: str = ""
    rl_action_key: str = ""
    travel_ticks_each_way: int = 0
    extracted_units: int = 0
    extracted_mass_kg: float = 0.0
    route: list[dict[str, int]] = field(default_factory=list)


@dataclass
class SurfaceVehicle:
    id: str
    vehicle_type: str
    x: int = 0
    y: int = 0
    state: str = "idle"
    battery_kwh: float = 0.0
    battery_capacity_kwh: float = 0.0
    condition: float = 1.0
    dust_load: float = 0.0
    payload: dict[str, int] = field(default_factory=dict)
    payload_mass_kg: float = 0.0
    job: Optional[FleetJob] = None
    phase_ticks_remaining: int = 0
    total_distance_km: float = 0.0
    total_excavated_kg: float = 0.0
    total_assembly_work_hours: float = 0.0
    completed_jobs: int = 0
    fault_reason: Optional[str] = None
    service_due: bool = False
    service_count: int = 0
    last_service_material: Optional[str] = None
    mission: Optional[dict] = None
    reservation_id: Optional[str] = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["battery_pct"] = round(
            self.battery_kwh / max(0.001, self.battery_capacity_kwh) * 100.0,
            1,
        )
        result["condition_pct"] = round(self.condition * 100.0, 1)
        result["payload_mass_kg"] = round(self.payload_mass_kg, 2)
        result["total_distance_km"] = round(self.total_distance_km, 3)
        result["total_excavated_kg"] = round(self.total_excavated_kg, 2)
        result["total_assembly_work_hours"] = round(
            self.total_assembly_work_hours, 2
        )
        return result


class SurfaceFleet:
    """Crew mobility, excavation, and supervised modular-assembly robots."""

    def __init__(self, mission_profile):
        self.profile = mission_profile
        self.grid_cell_m = mission_profile.grid_cell_meters
        fleet = mission_profile.surface_fleet
        self.crew_rover_spec = dict(fleet.get("crew_rover", {}))
        self.excavator_spec = dict(fleet.get("excavator", {}))
        self.assembly_robot_spec = dict(fleet.get("assembly_robot", {}))
        self.cargo_transporter_spec = dict(fleet.get("cargo_transporter", {}))
        self.base_x = 0
        self.base_y = 0
        self._job_serial = 0
        self.service_modules_used: dict[str, int] = {}
        self.crew_rovers = [
            SurfaceVehicle(
                id=f"crew_rover_{index + 1}",
                vehicle_type="crew_rover",
                battery_kwh=float(self.crew_rover_spec.get("battery_capacity_kwh", 30.0)),
                battery_capacity_kwh=float(
                    self.crew_rover_spec.get("battery_capacity_kwh", 30.0)
                ),
            )
            for index in range(int(self.crew_rover_spec.get("count", 1)))
        ]
        self.excavators = [
            SurfaceVehicle(
                id=f"excavator_{index + 1}",
                vehicle_type="excavator",
                battery_kwh=float(self.excavator_spec.get("battery_capacity_kwh", 2.0)),
                battery_capacity_kwh=float(
                    self.excavator_spec.get("battery_capacity_kwh", 2.0)
                ),
            )
            for index in range(int(self.excavator_spec.get("count", 2)))
        ]
        self.assembly_robots = [
            SurfaceVehicle(
                id=f"assembly_robot_{index + 1}",
                vehicle_type="assembly_robot",
                battery_kwh=float(
                    self.assembly_robot_spec.get("battery_capacity_kwh", 20.0)
                ),
                battery_capacity_kwh=float(
                    self.assembly_robot_spec.get("battery_capacity_kwh", 20.0)
                ),
            )
            for index in range(int(self.assembly_robot_spec.get("count", 0)))
        ]
        self.cargo_transporters = [
            SurfaceVehicle(
                id=f"cargo_transporter_{index + 1}",
                vehicle_type="cargo_transporter",
                battery_kwh=float(
                    self.cargo_transporter_spec.get("battery_capacity_kwh", 120.0)
                ),
                battery_capacity_kwh=float(
                    self.cargo_transporter_spec.get("battery_capacity_kwh", 120.0)
                ),
            )
            for index in range(int(self.cargo_transporter_spec.get("count", 0)))
        ]

    @property
    def vehicles(self) -> list[SurfaceVehicle]:
        return [
            *self.crew_rovers,
            *self.excavators,
            *self.assembly_robots,
            *self.cargo_transporters,
        ]

    def place_at_base(self, x: int, y: int) -> None:
        self.base_x, self.base_y = int(x), int(y)
        for index, vehicle in enumerate(self.vehicles):
            vehicle.x = self.base_x + (index % 2)
            vehicle.y = self.base_y + 2 + index // 2

    @staticmethod
    def _serialized_route(route) -> list[dict[str, int]]:
        result = []
        for point in route or []:
            if isinstance(point, dict):
                x, y = point.get("x"), point.get("y")
            elif isinstance(point, (tuple, list)) and len(point) >= 2:
                x, y = point[0], point[1]
            else:
                continue
            result.append({"x": int(x), "y": int(y)})
        return result

    def _route_distance_cells(
        self, route, target_x: int, target_y: int
    ) -> int:
        serialized = self._serialized_route(route)
        if len(serialized) >= 2:
            return len(serialized) - 1
        # Backward-compatible direct API behavior for callers that have no
        # terrain planner. Engine dispatches always provide a routed path.
        return max(
            abs(int(target_x) - self.base_x),
            abs(int(target_y) - self.base_y),
        )

    def _assembly_approach_route(
        self,
        route,
        site_x: int,
        site_y: int,
        slot: int,
    ) -> tuple[list[dict[str, int]], int, int]:
        """Terminate robots at distinct cells around a construction parcel."""
        site = (int(site_x), int(site_y))
        serialized = self._serialized_route(route)
        if not serialized:
            # Direct API callers do not have the engine's terrain planner.
            serialized = [{"x": self.base_x, "y": self.base_y}]
            x, y = self.base_x, self.base_y
            while x != site[0]:
                x += 1 if site[0] > x else -1
                serialized.append({"x": x, "y": y})
            while y != site[1]:
                y += 1 if site[1] > y else -1
                serialized.append({"x": x, "y": y})
        if (
            serialized[-1]["x"] != site[0]
            or serialized[-1]["y"] != site[1]
        ):
            serialized.append({"x": site[0], "y": site[1]})

        ring = [
            (-1, -1), (0, -1), (1, -1), (1, 0),
            (1, 1), (0, 1), (-1, 1), (-1, 0),
        ]
        # Four delivered robots occupy W/N/E/S slots. Additional future units
        # use diagonal service parcels without sharing the structure cell.
        preferred_slots = (7, 1, 3, 5, 0, 2, 4, 6)
        desired_index = preferred_slots[int(slot) % len(preferred_slots)]
        if len(serialized) == 1:
            approach = ring[desired_index]
            return [
                dict(serialized[0]),
                {
                    "x": site[0] + approach[0],
                    "y": site[1] + approach[1],
                },
            ], site[0] + approach[0], site[1] + approach[1]
        if len(serialized) >= 2:
            entry = (
                serialized[-2]["x"] - site[0],
                serialized[-2]["y"] - site[1],
            )
        else:
            entry = ring[desired_index]
        if entry not in ring:
            entry = ring[desired_index]
        entry_index = ring.index(entry)
        trimmed = serialized[:-1]
        clockwise = (desired_index - entry_index) % len(ring)
        counter = (entry_index - desired_index) % len(ring)
        direction = 1 if clockwise <= counter else -1
        cursor = entry_index
        if not trimmed or (
            trimmed[-1]["x"], trimmed[-1]["y"]
        ) != (site[0] + entry[0], site[1] + entry[1]):
            trimmed.append({
                "x": site[0] + entry[0], "y": site[1] + entry[1]
            })
        while cursor != desired_index:
            cursor = (cursor + direction) % len(ring)
            offset = ring[cursor]
            trimmed.append({
                "x": site[0] + offset[0],
                "y": site[1] + offset[1],
            })
        approach = ring[desired_index]
        return trimmed, site[0] + approach[0], site[1] + approach[1]

    @staticmethod
    def _place_vehicle_on_route(
        vehicle: SurfaceVehicle,
        route: list[dict[str, int]],
        *,
        returning: bool,
        ticks_remaining: int,
        total_ticks: int,
    ) -> None:
        """Publish an auditable intermediate cell during a timed route leg."""
        if len(route) < 2:
            return
        points = list(reversed(route)) if returning else route
        completed_fraction = 1.0 - (
            max(0, int(ticks_remaining)) / max(1, int(total_ticks))
        )
        index = min(
            len(points) - 1,
            max(0, int(math.floor(
                completed_fraction * (len(points) - 1) + 1e-9
            ))),
        )
        vehicle.x = int(points[index]["x"])
        vehicle.y = int(points[index]["y"])

    def _service_at_base(
        self,
        vehicle: SurfaceVehicle,
        spec: dict,
        service_callback: Optional[Callable[[str, int], bool]],
        events: list[dict],
    ) -> None:
        """Replace a finite class-specific wear module at the base.

        Condition loss is deterministic and workload-based.  A vehicle can be
        serviced only after returning to the landing hub, and the replacement
        module must exist in the closed cargo inventory.  This avoids both
        immortal robots and unrepairable one-shot equipment.
        """
        threshold = min(0.95, max(
            0.05, float(spec.get("service_condition_threshold", 0.75))
        ))
        vehicle.service_due = vehicle.condition < threshold
        if not vehicle.service_due:
            return
        material = str(spec.get("service_module_material", "")).strip()
        if not material or service_callback is None:
            return
        if not service_callback(material, 1):
            if vehicle.condition < 0.50 and vehicle.state != "fault":
                vehicle.state = "fault"
                vehicle.fault_reason = "maintenance_spares_exhausted"
                events.append({
                    "type": "vehicle_fault",
                    "vehicle_id": vehicle.id,
                    "reason": vehicle.fault_reason,
                    "required_service_material": material,
                })
            return
        before = vehicle.condition
        vehicle.condition = max(
            vehicle.condition,
            min(1.0, float(spec.get("service_restore_condition", 0.98))),
        )
        vehicle.dust_load = max(0.0, vehicle.dust_load - 0.50)
        vehicle.service_due = False
        vehicle.service_count += 1
        vehicle.last_service_material = material
        self.service_modules_used[material] = (
            self.service_modules_used.get(material, 0) + 1
        )
        events.append({
            "type": "vehicle_service_complete",
            "vehicle_id": vehicle.id,
            "service_material": material,
            "condition_before": round(before, 4),
            "condition_after": round(vehicle.condition, 4),
        })

    def assembly_assist(
        self,
        *,
        site_id: str,
        builder_count: int,
        available_grid_energy_kwh: float = 0.0,
        site_x: int | None = None,
        site_y: int | None = None,
        route=None,
    ) -> dict:
        """Return finite robot work for one supervised construction tick.

        A new assignment spends one tick travelling. Work drains the robot's
        battery and adds wear/dust; charging energy is accounted by ``tick``
        after the robot returns to base. Robots pause when supervision is
        absent and never create materials or readiness on their own.
        """
        result = {
            "equivalent_workers": 0.0,
            "work_hours": 0.0,
            "grid_energy_draw_kwh": 0.0,
            "active_robot_ids": [],
        }
        if not self.assembly_robots:
            return result

        site_id = str(site_id)
        requires_supervisor = bool(
            self.assembly_robot_spec.get("requires_onsite_supervisor", True)
        )
        if requires_supervisor and int(builder_count) <= 0:
            for robot in self.assembly_robots:
                if (
                    isinstance(robot.mission, dict)
                    and robot.mission.get("site_id") == site_id
                    and robot.state == "working"
                ):
                    robot.state = "waiting_supervision"
            return result

        max_per_crew = max(
            1, int(self.assembly_robot_spec.get("max_supervised_per_crew", 2))
        )
        allowed = min(
            len(self.assembly_robots),
            max_per_crew * max(1, int(builder_count)),
        )
        tick_hours = self.profile.clock.tick_minutes / 60.0
        energy_per_tick = max(
            0.0, float(self.assembly_robot_spec.get("operating_power_kw", 2.0))
        ) * tick_hours
        reserve_fraction = min(0.8, max(
            0.0, float(self.assembly_robot_spec.get("reserve_fraction", 0.20))
        ))
        equivalent_per_robot = max(
            0.0,
            float(
                self.assembly_robot_spec.get(
                    "construction_equivalent_workers", 1.5
                )
            ),
        )
        candidates = [
            robot for robot in self.assembly_robots
            if robot.condition >= 0.50
            and (
                robot.state == "idle"
                or (
                    robot.state in {"working", "waiting_supervision"}
                    and isinstance(robot.mission, dict)
                    and robot.mission.get("site_id") == site_id
                )
            )
        ]
        selected = sorted(candidates, key=lambda item: item.id)[:allowed]
        active: list[SurfaceVehicle] = []
        for robot in selected:
            if robot.state == "idle":
                physical_site_x = int(
                    site_x if site_x is not None else self.base_x
                )
                physical_site_y = int(
                    site_y if site_y is not None else self.base_y
                )
                robot_slot = self.assembly_robots.index(robot)
                serialized_route, target_site_x, target_site_y = (
                    self._assembly_approach_route(
                        route, physical_site_x, physical_site_y, robot_slot
                    )
                )
                distance_cells = self._route_distance_cells(
                    serialized_route, target_site_x, target_site_y
                )
                distance_km = distance_cells * self.grid_cell_m / 1000.0
                travel_speed = max(
                    0.05,
                    float(self.assembly_robot_spec.get(
                        "travel_speed_kph", 3.0
                    )),
                )
                travel_ticks = max(
                    1,
                    int(math.ceil(
                        distance_km / travel_speed * 60.0
                        / self.profile.clock.tick_minutes
                    )),
                )
                travel_power = max(
                    0.0,
                    float(self.assembly_robot_spec.get(
                        "travel_power_kw", 2.0
                    )),
                )
                travel_energy = travel_ticks * tick_hours * travel_power
                reserve_kwh = robot.battery_capacity_kwh * reserve_fraction
                if robot.battery_kwh + 1e-9 < (
                    travel_energy * 2.0 + energy_per_tick + reserve_kwh
                ):
                    robot.state = "charging"
                    continue
                robot.state = "outbound"
                robot.phase_ticks_remaining = travel_ticks
                robot.mission = {
                    "site_id": site_id,
                    "site_x": physical_site_x,
                    "site_y": physical_site_y,
                    "target_x": target_site_x,
                    "target_y": target_site_y,
                    "travel_ticks_each_way": travel_ticks,
                    "one_way_distance_km": round(distance_km, 3),
                    "route": serialized_route,
                }
                continue
            return_ticks = max(
                1, int((robot.mission or {}).get("travel_ticks_each_way", 1))
            )
            return_energy = (
                return_ticks
                * tick_hours
                * max(0.0, float(self.assembly_robot_spec.get(
                    "travel_power_kw", 2.0
                )))
            )
            reserve_kwh = robot.battery_capacity_kwh * reserve_fraction
            if robot.battery_kwh + 1e-9 < (
                energy_per_tick + return_energy + reserve_kwh
            ):
                robot.state = "returning"
                robot.phase_ticks_remaining = return_ticks
                continue
            robot.state = "working"
            robot.battery_kwh = max(0.0, robot.battery_kwh - energy_per_tick)
            robot.condition = max(
                0.0,
                robot.condition
                - float(
                    self.assembly_robot_spec.get(
                        "condition_loss_per_operating_hour", 0.0002
                    )
                ) * tick_hours,
            )
            robot.dust_load = min(
                1.0,
                robot.dust_load
                + float(
                    self.assembly_robot_spec.get(
                        "dust_gain_per_operating_hour", 0.001
                    )
                ) * tick_hours,
            )
            robot.total_assembly_work_hours += (
                equivalent_per_robot * tick_hours
            )
            active.append(robot)

        equivalent_workers = len(active) * equivalent_per_robot
        result.update({
            "equivalent_workers": equivalent_workers,
            "work_hours": equivalent_workers * tick_hours,
            "active_robot_ids": [robot.id for robot in active],
        })
        return result

    def release_assembly_site(self, site_id: str, completed: bool = True) -> None:
        """Return robots assigned to a completed or cancelled project."""
        for robot in self.assembly_robots:
            if (
                isinstance(robot.mission, dict)
                and robot.mission.get("site_id") == str(site_id)
            ):
                if completed:
                    robot.completed_jobs += 1
                robot.state = "returning"
                robot.phase_ticks_remaining = max(
                    1,
                    int((robot.mission or {}).get(
                        "travel_ticks_each_way", 1
                    )),
                )

    def cargo_transport_feasibility(
        self,
        target_x: int,
        target_y: int,
        payload_mass_kg: float,
        route=None,
    ) -> dict:
        """Validate a conserved depot-to-site heavy-cargo trip.

        Assembly robots and crew rovers are deliberately excluded: neither has
        the deck load, restraints, braking or stability system needed for
        multi-tonne flight hardware.
        """
        payload_mass_kg = max(0.0, float(payload_mass_kg))
        payload_limit = max(
            0.0, float(self.cargo_transporter_spec.get("payload_kg", 0.0))
        )
        if payload_mass_kg <= 0.0:
            return {"feasible": False, "reason": "empty_cargo_manifest"}
        if payload_mass_kg > payload_limit + 1e-6:
            return {
                "feasible": False,
                "reason": "cargo_payload_limit_exceeded",
                "payload_mass_kg": payload_mass_kg,
                "payload_limit_kg": payload_limit,
            }
        idle = [
            vehicle for vehicle in self.cargo_transporters
            if vehicle.state == "idle" and vehicle.condition >= 0.55
        ]
        if not idle:
            return {"feasible": False, "reason": "cargo_transporter_unavailable"}

        serialized_route = self._serialized_route(route)
        distance_cells = self._route_distance_cells(
            serialized_route, target_x, target_y
        )
        distance_km = distance_cells * self.grid_cell_m / 1000.0
        speed_kph = max(
            0.05,
            float(self.cargo_transporter_spec.get("travel_speed_kph", 3.0)),
        )
        travel_ticks = max(
            1,
            int(math.ceil(
                distance_km / speed_kph * 60.0 / self.profile.clock.tick_minutes
            )),
        )
        unload_ticks = max(
            1,
            int(math.ceil(
                float(self.cargo_transporter_spec.get("unload_minutes", 30.0))
                / self.profile.clock.tick_minutes
            )),
        )
        tick_hours = self.profile.clock.tick_minutes / 60.0
        travel_power = max(
            0.0, float(self.cargo_transporter_spec.get("travel_power_kw", 20.0))
        )
        unload_power = max(
            0.0,
            float(self.cargo_transporter_spec.get("unloading_power_kw", 5.0)),
        )
        feasible_vehicles = []
        for vehicle in idle:
            reserve = vehicle.battery_capacity_kwh * max(
                0.0,
                float(self.cargo_transporter_spec.get(
                    "return_reserve_fraction", 0.20
                )),
            )
            required = (
                travel_ticks * 2 * tick_hours * travel_power
                + unload_ticks * tick_hours * unload_power
                + reserve
            )
            if vehicle.battery_kwh + 1e-9 >= required:
                feasible_vehicles.append((vehicle, required))
        if not feasible_vehicles:
            return {
                "feasible": False,
                "reason": "insufficient_cargo_return_energy",
            }
        vehicle, required_energy = min(
            feasible_vehicles, key=lambda pair: pair[0].id
        )
        return {
            "feasible": True,
            "vehicle_id": vehicle.id,
            "distance_km": round(distance_km, 3),
            "travel_ticks_each_way": travel_ticks,
            "unload_ticks": unload_ticks,
            "required_energy_kwh": round(required_energy, 4),
            "payload_mass_kg": round(payload_mass_kg, 3),
            "payload_limit_kg": payload_limit,
            "route": serialized_route,
        }

    def dispatch_cargo_transport(
        self,
        *,
        site_id: str,
        target_x: int,
        target_y: int,
        payload: dict[str, int],
        payload_mass_kg: float,
        current_tick: int,
        route=None,
    ) -> dict:
        """Load one exact restrained manifest and send it to one site."""
        feasibility = self.cargo_transport_feasibility(
            target_x, target_y, payload_mass_kg, route=route
        )
        if not feasibility.get("feasible"):
            return {"dispatched": False, **feasibility}
        vehicle = next(
            item for item in self.cargo_transporters
            if item.id == feasibility["vehicle_id"]
        )
        vehicle.payload = {
            str(material): int(quantity)
            for material, quantity in payload.items()
            if int(quantity) > 0
        }
        vehicle.payload_mass_kg = float(payload_mass_kg)
        vehicle.mission = {
            "site_id": str(site_id),
            "target_x": int(target_x),
            "target_y": int(target_y),
            "dispatched_tick": int(current_tick),
            "travel_ticks_each_way": int(
                feasibility["travel_ticks_each_way"]
            ),
            "unload_ticks": int(feasibility["unload_ticks"]),
            "route": list(feasibility.get("route", [])),
        }
        vehicle.state = "cargo_outbound"
        vehicle.phase_ticks_remaining = int(
            feasibility["travel_ticks_each_way"]
        )
        return {"dispatched": True, **feasibility, "site_id": str(site_id)}

    def available_excavator(self) -> Optional[SurfaceVehicle]:
        threshold = float(
            self.excavator_spec.get("minimum_dispatch_charge_fraction", 0.8)
        )
        candidates = [
            vehicle for vehicle in self.excavators
            if vehicle.state == "idle"
            and vehicle.condition >= 0.55
            and vehicle.battery_kwh
            >= vehicle.battery_capacity_kwh * threshold
        ]
        return min(candidates, key=lambda item: item.id, default=None)

    def excavator_dispatch_feasibility(
        self,
        target_x: int,
        target_y: int,
        route=None,
    ) -> dict:
        """Return the physical round-trip mask for the next ready excavator.

        A robot is not a valid policy option merely because it is charged past
        the generic dispatch threshold.  Its present battery must also cover
        outbound travel, at least one excavation tick, return travel and the
        configured emergency reserve for this particular face.
        """
        vehicle = self.available_excavator()
        if vehicle is None:
            return {"feasible": False, "reason": "no_ready_excavator"}

        serialized_route = self._serialized_route(route)
        distance_cells = self._route_distance_cells(
            serialized_route, target_x, target_y
        )
        distance_km = distance_cells * self.grid_cell_m / 1000.0
        speed_kph = max(
            0.05,
            float(self.excavator_spec.get("travel_speed_kph", 0.5)),
        )
        travel_ticks = max(
            1,
            int(math.ceil(
                distance_km
                / speed_kph
                * 60.0
                / self.profile.clock.tick_minutes
            )),
        )
        dt_hours = self.profile.clock.tick_minutes / 60.0
        travel_power = float(self.excavator_spec.get("travel_power_kw", 0.2))
        excavation_power = float(
            self.excavator_spec.get("excavation_power_kw", 0.45)
        )
        round_trip_energy = travel_ticks * 2 * dt_hours * travel_power
        reserve = vehicle.battery_capacity_kwh * float(
            self.excavator_spec.get("return_reserve_fraction", 0.25)
        )
        required_energy = (
            round_trip_energy + excavation_power * dt_hours + reserve
        )
        result = {
            "feasible": vehicle.battery_kwh + 1e-9 >= required_energy,
            "vehicle_id": vehicle.id,
            "distance_km": round(distance_km, 3),
            "travel_ticks_each_way": travel_ticks,
            "required_energy_kwh": round(required_energy, 4),
            "available_energy_kwh": round(vehicle.battery_kwh, 4),
            "route": serialized_route,
        }
        if not result["feasible"]:
            result["reason"] = "insufficient_return_energy"
        return result

    def begin_crew_rover_trip(
        self,
        *,
        expedition_id: str,
        crew_ids: list[str],
        target_x: int,
        target_y: int,
        route=None,
        reservation_id: str | None = None,
    ) -> dict:
        """Reserve the unpressurized two-seat rover for a known destination."""
        seats = int(self.crew_rover_spec.get("crew_seats", 2))
        if len(crew_ids) != 2 or len(crew_ids) > seats:
            return {"reserved": False, "reason": "two_person_rover_crew_required"}
        rover = next(
            (vehicle for vehicle in self.crew_rovers if vehicle.state in {"idle", "charging"}
             and not vehicle.mission
             and vehicle.reservation_id in {None, reservation_id}),
            None,
        )
        if rover is None:
            return {"reserved": False, "reason": "crew_rover_unavailable"}
        serialized_route = self._serialized_route(route)
        distance_cells = self._route_distance_cells(
            serialized_route, target_x, target_y
        )
        one_way_km = distance_cells * self.grid_cell_m / 1000.0
        if one_way_km > float(self.crew_rover_spec.get("walkback_limit_km", 7.6)):
            return {"reserved": False, "reason": "outside_walkback_limit"}
        energy_per_km = float(
            self.crew_rover_spec.get("traction_energy_kwh_per_km", 1.2)
        )
        trip_energy = one_way_km * 2.0 * energy_per_km
        reserve = rover.battery_capacity_kwh * float(
            self.crew_rover_spec.get("reserve_fraction", 0.30)
        )
        if rover.battery_kwh < trip_energy + reserve:
            return {"reserved": False, "reason": "insufficient_return_energy"}
        rover.state = "in_use"
        rover.reservation_id = None
        rover.mission = {
            "expedition_id": str(expedition_id),
            "crew_ids": [str(crew_id) for crew_id in crew_ids],
            "target_x": int(target_x),
            "target_y": int(target_y),
            "planned_round_trip_km": round(one_way_km * 2.0, 3),
            "route": serialized_route,
        }
        return {
            "reserved": True,
            "vehicle_id": rover.id,
            "one_way_km": round(one_way_km, 3),
        }

    def record_crew_rover_movement(
        self,
        expedition_id: str,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
    ) -> bool:
        rover = next((
            vehicle for vehicle in self.crew_rovers
            if vehicle.state == "in_use"
            and isinstance(vehicle.mission, dict)
            and vehicle.mission.get("expedition_id") == expedition_id
        ), None)
        if rover is None:
            return False
        # Orthogonal cell travel has Manhattan length; a diagonal coordinate
        # update is two 100 m legs, never one free corner cut.
        distance_cells = (
            abs(int(to_x) - int(from_x))
            + abs(int(to_y) - int(from_y))
        )
        distance_km = distance_cells * self.grid_cell_m / 1000.0
        energy = distance_km * float(
            self.crew_rover_spec.get("traction_energy_kwh_per_km", 1.2)
        )
        if rover.battery_kwh < energy:
            rover.battery_kwh = 0.0
            rover.state = "fault"
            rover.fault_reason = "crew_rover_battery_depleted"
            return False
        rover.battery_kwh -= energy
        rover.total_distance_km += distance_km
        rover.condition = max(
            0.0,
            rover.condition
            - distance_km * float(
                self.crew_rover_spec.get("condition_loss_per_km", 0.0005)
            ),
        )
        rover.dust_load = min(1.0, rover.dust_load + distance_km * 0.0005)
        if rover.condition < 0.50:
            rover.state = "fault"
            rover.fault_reason = "traction_or_suspension_condition_limit"
            return False
        rover.x, rover.y = int(to_x), int(to_y)
        return True

    def crew_rover_for_expedition(
        self, expedition_id: str
    ) -> Optional[SurfaceVehicle]:
        """Return the physical rover reserved for an active expedition."""
        return next((
            vehicle for vehicle in self.crew_rovers
            if isinstance(vehicle.mission, dict)
            and vehicle.mission.get("expedition_id") == expedition_id
        ), None)

    def load_crew_rover_payload(
        self,
        expedition_id: str,
        resource: str,
        requested_units: int,
        unit_mass_kg: float,
    ) -> dict:
        """Load physically excavated material without creating any resource.

        The engine remains the geological authority and removes the exact
        returned unit count from the working face.  This method only enforces
        the delivered rover's audited payload limit.
        """
        rover = self.crew_rover_for_expedition(expedition_id)
        if rover is None or rover.state != "in_use":
            return {"loaded_units": 0, "payload_full": False,
                    "reason": "crew_rover_unavailable"}
        unit_mass_kg = max(0.001, float(unit_mass_kg))
        payload_limit = float(self.crew_rover_spec.get("payload_kg", 490.0))
        remaining_kg = max(0.0, payload_limit - rover.payload_mass_kg)
        loadable_units = min(
            max(0, int(requested_units)),
            max(0, int(remaining_kg // unit_mass_kg)),
        )
        if loadable_units > 0:
            rover.payload[str(resource)] = (
                rover.payload.get(str(resource), 0) + loadable_units
            )
            loaded_mass = loadable_units * unit_mass_kg
            rover.payload_mass_kg += loaded_mass
        return {
            "loaded_units": loadable_units,
            "payload_mass_kg": rover.payload_mass_kg,
            "payload_limit_kg": payload_limit,
            "payload_full": (
                payload_limit - rover.payload_mass_kg < unit_mass_kg
            ),
        }

    def unload_crew_rover_payload(self, expedition_id: str) -> dict[str, int]:
        """Remove and return the rover cargo once it is physically at base."""
        rover = self.crew_rover_for_expedition(expedition_id)
        if rover is None:
            return {}
        payload = dict(rover.payload)
        rover.payload.clear()
        rover.payload_mass_kg = 0.0
        return payload

    def complete_crew_rover_trip(self, expedition_id: str) -> bool:
        rover = self.crew_rover_for_expedition(expedition_id)
        if rover is None:
            return False
        rover.x, rover.y = self.base_x, self.base_y
        rover.state = "charging"
        rover.completed_jobs += 1
        rover.mission = None
        return True

    def dispatch_excavator(
        self,
        *,
        resource: str,
        target_x: int,
        target_y: int,
        current_tick: int,
        dispatched_by: str,
        verified_exposed_face: bool,
        protected_ground: bool,
        objective_resource: str = "",
        rl_state_key: str = "",
        rl_action_key: str = "",
        route=None,
    ) -> dict:
        if self.excavator_spec.get("requires_verified_exposed_face", True) and not verified_exposed_face:
            return {"dispatched": False, "reason": "unverified_exposed_face"}
        if protected_ground:
            return {"dispatched": False, "reason": "construction_ground_protected"}
        feasibility = self.excavator_dispatch_feasibility(
            target_x, target_y, route=route
        )
        if not feasibility.get("feasible"):
            return {
                "dispatched": False,
                "reason": feasibility.get("reason", "dispatch_infeasible"),
            }
        vehicle = next(
            item for item in self.excavators
            if item.id == feasibility["vehicle_id"]
        )
        travel_ticks = int(feasibility["travel_ticks_each_way"])

        self._job_serial += 1
        job = FleetJob(
            id=f"exc-{current_tick}-{self._job_serial}",
            resource=str(resource),
            target_x=int(target_x),
            target_y=int(target_y),
            dispatched_tick=int(current_tick),
            dispatched_by=str(dispatched_by),
            verified_exposed_face=True,
            protected_ground=False,
            objective_resource=str(objective_resource or resource),
            rl_state_key=str(rl_state_key or ""),
            rl_action_key=str(rl_action_key or ""),
            travel_ticks_each_way=travel_ticks,
            route=list(feasibility.get("route", [])),
        )
        vehicle.job = job
        vehicle.state = "outbound"
        vehicle.phase_ticks_remaining = travel_ticks
        return {
            "dispatched": True,
            "vehicle_id": vehicle.id,
            "job_id": job.id,
            "travel_ticks_each_way": travel_ticks,
        }

    def tick(
        self,
        *,
        available_grid_energy_kwh: float,
        excavation_callback: Callable[[FleetJob, float], tuple[int, float, bool]],
        unload_callback: Callable[[dict[str, int]], None],
        service_callback: Optional[Callable[[str, int], bool]] = None,
    ) -> dict:
        """Advance all vehicles one physical tick.

        ``excavation_callback`` returns (units, mass_kg, face_depleted).
        No geological material is created inside this module.
        """
        events: list[dict] = []
        grid_draw = 0.0
        dt_hours = self.profile.clock.tick_minutes / 60.0
        charge_power = float(self.excavator_spec.get("charge_power_kw", 0.45))
        charge_eff = float(self.excavator_spec.get("charge_efficiency", 0.85))
        travel_power = float(self.excavator_spec.get("travel_power_kw", 0.2))
        excavation_power = float(
            self.excavator_spec.get("excavation_power_kw", 0.45)
        )
        payload_limit = float(self.excavator_spec.get("payload_kg", 80.0))
        nominal_mass_tick = float(
            self.excavator_spec.get("nominal_loose_regolith_kg_per_day", 2700.0)
        ) / self.profile.clock.ticks_per_earth_day

        rover_charge_power = float(
            self.crew_rover_spec.get("charge_power_kw", 5.0)
        )
        rover_charge_efficiency = float(
            self.crew_rover_spec.get("charge_efficiency", 0.90)
        )
        for rover in self.crew_rovers:
            if rover.state not in {"idle", "charging"}:
                continue
            self._service_at_base(
                rover, self.crew_rover_spec, service_callback, events
            )
            if rover.state == "fault":
                continue
            missing = rover.battery_capacity_kwh - rover.battery_kwh
            possible_draw = min(
                rover_charge_power * dt_hours,
                missing / max(0.01, rover_charge_efficiency),
            )
            actual_draw = min(
                possible_draw,
                max(0.0, available_grid_energy_kwh - grid_draw),
            )
            if actual_draw > 0:
                rover.state = "charging"
                rover.battery_kwh = min(
                    rover.battery_capacity_kwh,
                    rover.battery_kwh + actual_draw * rover_charge_efficiency,
                )
                grid_draw += actual_draw
            if rover.battery_kwh >= rover.battery_capacity_kwh - 1e-6:
                rover.state = "idle"

        assembly_charge_power = max(
            0.0, float(self.assembly_robot_spec.get("charge_power_kw", 4.0))
        )
        assembly_charge_efficiency = min(1.0, max(
            0.01, float(self.assembly_robot_spec.get("charge_efficiency", 0.90))
        ))
        assembly_travel_power = max(
            0.0,
            float(self.assembly_robot_spec.get("travel_power_kw", 2.0)),
        )
        for robot in self.assembly_robots:
            if robot.state == "fault":
                continue
            if robot.state in {"idle", "charging"}:
                self._service_at_base(
                    robot, self.assembly_robot_spec, service_callback, events
                )
                if robot.state == "fault":
                    continue
            if robot.state in {"outbound", "returning"}:
                travel_draw = assembly_travel_power * dt_hours
                if robot.battery_kwh <= travel_draw:
                    robot.battery_kwh = 0.0
                    robot.state = "fault"
                    robot.fault_reason = "assembly_robot_battery_depleted_in_transit"
                    events.append({
                        "type": "vehicle_fault",
                        "vehicle_id": robot.id,
                        "reason": robot.fault_reason,
                        "site_id": (robot.mission or {}).get("site_id"),
                    })
                    continue
                robot.battery_kwh -= travel_draw
                robot.phase_ticks_remaining -= 1
                mission = robot.mission or {}
                route = self._serialized_route(mission.get("route", []))
                distance_km = float(mission.get("one_way_distance_km", 0.0))
                travel_ticks = max(
                    1, int(mission.get("travel_ticks_each_way", 1))
                )
                self._place_vehicle_on_route(
                    robot,
                    route,
                    returning=robot.state == "returning",
                    ticks_remaining=robot.phase_ticks_remaining,
                    total_ticks=travel_ticks,
                )
                robot.total_distance_km += distance_km / travel_ticks
                if robot.phase_ticks_remaining <= 0:
                    robot.condition = max(
                        0.0,
                        robot.condition
                        - distance_km * float(self.assembly_robot_spec.get(
                            "condition_loss_per_km", 0.0005
                        )),
                    )
                    if robot.state == "outbound":
                        robot.x = int(mission.get("target_x", self.base_x))
                        robot.y = int(mission.get("target_y", self.base_y))
                        robot.state = "waiting_supervision"
                    else:
                        robot.x, robot.y = self.base_x, self.base_y
                        robot.state = "charging"
                        robot.mission = None
                continue
            if robot.state in {"working", "waiting_supervision"}:
                continue
            missing = robot.battery_capacity_kwh - robot.battery_kwh
            possible_draw = min(
                assembly_charge_power * dt_hours,
                missing / assembly_charge_efficiency,
            )
            actual_draw = min(
                possible_draw,
                max(0.0, available_grid_energy_kwh - grid_draw),
            )
            if actual_draw > 0.0:
                robot.state = "charging"
                robot.battery_kwh = min(
                    robot.battery_capacity_kwh,
                    robot.battery_kwh + actual_draw * assembly_charge_efficiency,
                )
                grid_draw += actual_draw
                robot.dust_load = max(0.0, robot.dust_load - 0.002)
            if robot.battery_kwh >= robot.battery_capacity_kwh - 1e-6:
                robot.state = "idle"

        cargo_charge_power = max(
            0.0, float(self.cargo_transporter_spec.get("charge_power_kw", 20.0))
        )
        cargo_charge_efficiency = min(1.0, max(
            0.01, float(self.cargo_transporter_spec.get("charge_efficiency", 0.90))
        ))
        cargo_travel_power = max(
            0.0, float(self.cargo_transporter_spec.get("travel_power_kw", 20.0))
        )
        cargo_unload_power = max(
            0.0,
            float(self.cargo_transporter_spec.get("unloading_power_kw", 5.0)),
        )
        for transporter in self.cargo_transporters:
            if transporter.state == "fault":
                continue
            if transporter.state in {"idle", "charging"}:
                self._service_at_base(
                    transporter,
                    self.cargo_transporter_spec,
                    service_callback,
                    events,
                )
                if transporter.state == "fault":
                    continue
                missing = transporter.battery_capacity_kwh - transporter.battery_kwh
                possible_draw = min(
                    cargo_charge_power * dt_hours,
                    missing / cargo_charge_efficiency,
                )
                actual_draw = min(
                    possible_draw,
                    max(0.0, available_grid_energy_kwh - grid_draw),
                )
                if actual_draw > 0.0:
                    transporter.state = "charging"
                    transporter.battery_kwh = min(
                        transporter.battery_capacity_kwh,
                        transporter.battery_kwh
                        + actual_draw * cargo_charge_efficiency,
                    )
                    grid_draw += actual_draw
                if (
                    transporter.battery_kwh
                    >= transporter.battery_capacity_kwh - 1e-6
                ):
                    transporter.state = "idle"
                continue

            mission = transporter.mission or {}
            if transporter.state in {"cargo_outbound", "cargo_returning"}:
                draw = cargo_travel_power * dt_hours
                if transporter.battery_kwh <= draw:
                    transporter.battery_kwh = 0.0
                    transporter.state = "fault"
                    transporter.fault_reason = "cargo_transporter_battery_depleted"
                    events.append({
                        "type": "vehicle_fault",
                        "vehicle_id": transporter.id,
                        "reason": transporter.fault_reason,
                        "site_id": mission.get("site_id"),
                    })
                    continue
                transporter.battery_kwh -= draw
                transporter.phase_ticks_remaining -= 1
                route = self._serialized_route(mission.get("route", []))
                leg_km = self._route_distance_cells(
                    route,
                    int(mission.get("target_x", self.base_x)),
                    int(mission.get("target_y", self.base_y)),
                ) * self.grid_cell_m / 1000.0
                self._place_vehicle_on_route(
                    transporter,
                    route,
                    returning=transporter.state == "cargo_returning",
                    ticks_remaining=transporter.phase_ticks_remaining,
                    total_ticks=max(
                        1, int(mission.get("travel_ticks_each_way", 1))
                    ),
                )
                transporter.total_distance_km += leg_km / max(
                    1, int(mission.get("travel_ticks_each_way", 1))
                )
                if transporter.phase_ticks_remaining <= 0:
                    transporter.condition = max(
                        0.0,
                        transporter.condition
                        - leg_km * float(self.cargo_transporter_spec.get(
                            "condition_loss_per_km", 0.0004
                        )),
                    )
                    if transporter.state == "cargo_outbound":
                        transporter.x = int(mission.get("target_x", self.base_x))
                        transporter.y = int(mission.get("target_y", self.base_y))
                        transporter.state = "cargo_unloading"
                        transporter.phase_ticks_remaining = max(
                            1, int(mission.get("unload_ticks", 1))
                        )
                    else:
                        transporter.x, transporter.y = self.base_x, self.base_y
                        transporter.state = "charging"
                        transporter.mission = None
                continue

            if transporter.state == "cargo_unloading":
                draw = cargo_unload_power * dt_hours
                if transporter.battery_kwh <= draw:
                    transporter.battery_kwh = 0.0
                    transporter.state = "fault"
                    transporter.fault_reason = "cargo_unloading_power_depleted"
                    events.append({
                        "type": "vehicle_fault",
                        "vehicle_id": transporter.id,
                        "reason": transporter.fault_reason,
                        "site_id": mission.get("site_id"),
                    })
                    continue
                transporter.battery_kwh -= draw
                transporter.phase_ticks_remaining -= 1
                if transporter.phase_ticks_remaining <= 0:
                    events.append({
                        "type": "cargo_delivery_complete",
                        "vehicle_id": transporter.id,
                        "site_id": mission.get("site_id"),
                        "payload": dict(transporter.payload),
                        "payload_mass_kg": round(
                            transporter.payload_mass_kg, 3
                        ),
                    })
                    transporter.payload.clear()
                    transporter.payload_mass_kg = 0.0
                    transporter.completed_jobs += 1
                    transporter.condition = max(
                        0.0,
                        transporter.condition
                        - float(self.cargo_transporter_spec.get(
                            "condition_loss_per_trip", 0.0005
                        )),
                    )
                    transporter.state = "cargo_returning"
                    transporter.phase_ticks_remaining = max(
                        1, int(mission.get("travel_ticks_each_way", 1))
                    )
                continue

        for vehicle in self.excavators:
            if vehicle.state == "fault":
                continue
            if vehicle.state in {"idle", "charging"}:
                self._service_at_base(
                    vehicle, self.excavator_spec, service_callback, events
                )
                if vehicle.state == "fault":
                    continue
                missing = vehicle.battery_capacity_kwh - vehicle.battery_kwh
                possible_draw = min(charge_power * dt_hours, missing / max(0.01, charge_eff))
                actual_draw = min(
                    possible_draw,
                    max(0.0, available_grid_energy_kwh - grid_draw),
                )
                if actual_draw > 0:
                    vehicle.state = "charging"
                    vehicle.battery_kwh = min(
                        vehicle.battery_capacity_kwh,
                        vehicle.battery_kwh + actual_draw * charge_eff,
                    )
                    grid_draw += actual_draw
                if vehicle.battery_kwh >= vehicle.battery_capacity_kwh - 1e-6:
                    vehicle.state = "idle"
                continue

            if vehicle.state in {"outbound", "returning"}:
                draw = travel_power * dt_hours
                if vehicle.battery_kwh <= draw:
                    vehicle.battery_kwh = 0.0
                    vehicle.state = "fault"
                    vehicle.fault_reason = "battery_depleted_away_from_base"
                    events.append({"type": "vehicle_fault", "vehicle_id": vehicle.id, "reason": vehicle.fault_reason})
                    continue
                vehicle.battery_kwh -= draw
                vehicle.phase_ticks_remaining -= 1
                if vehicle.job:
                    route = self._serialized_route(vehicle.job.route)
                    leg_km = self._route_distance_cells(
                        route, vehicle.job.target_x, vehicle.job.target_y
                    ) * self.grid_cell_m / 1000.0
                    self._place_vehicle_on_route(
                        vehicle,
                        route,
                        returning=vehicle.state == "returning",
                        ticks_remaining=vehicle.phase_ticks_remaining,
                        total_ticks=max(1, vehicle.job.travel_ticks_each_way),
                    )
                    vehicle.total_distance_km += leg_km / max(1, vehicle.job.travel_ticks_each_way)
                if vehicle.phase_ticks_remaining <= 0:
                    if vehicle.state == "outbound" and vehicle.job:
                        vehicle.x, vehicle.y = vehicle.job.target_x, vehicle.job.target_y
                        vehicle.state = "excavating"
                    else:
                        vehicle.x, vehicle.y = self.base_x, self.base_y
                        vehicle.state = "unloading"
                        vehicle.phase_ticks_remaining = max(
                            1,
                            int(math.ceil(float(self.excavator_spec.get("unload_minutes", 10)) / self.profile.clock.tick_minutes)),
                        )
                continue

            if vehicle.state == "excavating" and vehicle.job:
                draw = excavation_power * dt_hours
                reserve = vehicle.battery_capacity_kwh * float(
                    self.excavator_spec.get("return_reserve_fraction", 0.25)
                )
                return_energy = vehicle.job.travel_ticks_each_way * travel_power * dt_hours
                if vehicle.battery_kwh <= draw + reserve + return_energy:
                    vehicle.state = "returning"
                    vehicle.phase_ticks_remaining = vehicle.job.travel_ticks_each_way
                    continue
                capacity_kg = max(0.0, payload_limit - vehicle.payload_mass_kg)
                rate_multipliers = self.excavator_spec.get(
                    "excavation_rate_multipliers", {}
                ) or {}
                material_rate = max(
                    0.05, min(1.0, float(rate_multipliers.get(
                        vehicle.job.resource, 1.0
                    )))
                )
                requested_kg = min(
                    nominal_mass_tick * material_rate, capacity_kg
                )
                units, mass_kg, face_depleted = excavation_callback(
                    vehicle.job, requested_kg
                )
                vehicle.battery_kwh -= draw
                if units > 0 and mass_kg > 0:
                    resource = vehicle.job.resource
                    vehicle.payload[resource] = vehicle.payload.get(resource, 0) + units
                    vehicle.payload_mass_kg += mass_kg
                    vehicle.total_excavated_kg += mass_kg
                    vehicle.job.extracted_units += units
                    vehicle.job.extracted_mass_kg += mass_kg
                    vehicle.dust_load = min(
                        1.0,
                        vehicle.dust_load
                        + float(self.excavator_spec.get(
                            "dust_gain_per_excavated_kg", 0.0001
                        )) * mass_kg,
                    )
                    vehicle.condition = max(
                        0.0,
                        vehicle.condition
                        - float(self.excavator_spec.get(
                            "condition_loss_per_excavated_kg", 0.000005
                        )) * mass_kg,
                    )
                if face_depleted or units <= 0 or vehicle.payload_mass_kg >= payload_limit - 0.01:
                    vehicle.state = "returning"
                    vehicle.phase_ticks_remaining = vehicle.job.travel_ticks_each_way
                continue

            if vehicle.state == "unloading":
                vehicle.phase_ticks_remaining -= 1
                if vehicle.phase_ticks_remaining <= 0:
                    unload_callback(dict(vehicle.payload))
                    completed_job = vehicle.job
                    vehicle.completed_jobs += 1
                    events.append({
                        "type": "excavator_job_complete",
                        "vehicle_id": vehicle.id,
                        "job": asdict(completed_job) if completed_job else None,
                        "payload": dict(vehicle.payload),
                    })
                    vehicle.payload.clear()
                    vehicle.payload_mass_kg = 0.0
                    vehicle.state = "cooldown"
                    vehicle.phase_ticks_remaining = max(
                        1,
                        int(math.ceil(
                            (float(self.excavator_spec.get("cooldown_minutes", 20))
                             + float(self.excavator_spec.get("inspection_minutes", 15)))
                            / self.profile.clock.tick_minutes
                        )),
                    )
                    vehicle.job = None
                continue

            if vehicle.state == "cooldown":
                vehicle.phase_ticks_remaining -= 1
                vehicle.dust_load = max(0.0, vehicle.dust_load - 0.01)
                if vehicle.phase_ticks_remaining <= 0:
                    vehicle.state = "charging"

        return {"grid_energy_draw_kwh": grid_draw, "events": events}

    def to_dict(self) -> dict:
        return {
            "basis": self.excavator_spec.get("design_basis"),
            "grid_cell_meters": self.grid_cell_m,
            "capability_matrix": {
                "crew_rover": {
                    "capabilities": list(self.crew_rover_spec.get("capabilities", [])),
                    "prohibited_roles": list(self.crew_rover_spec.get("prohibited_roles", [])),
                },
                "excavator": {
                    "capabilities": list(self.excavator_spec.get("capabilities", [])),
                    "prohibited_roles": list(self.excavator_spec.get("prohibited_roles", [])),
                },
                "assembly_robot": {
                    "capabilities": list(self.assembly_robot_spec.get("capabilities", [])),
                    "prohibited_roles": list(self.assembly_robot_spec.get("prohibited_roles", [])),
                },
                "cargo_transporter": {
                    "capabilities": list(self.cargo_transporter_spec.get("capabilities", [])),
                    "prohibited_roles": list(self.cargo_transporter_spec.get("prohibited_roles", [])),
                },
            },
            "crew_rovers": [vehicle.to_dict() for vehicle in self.crew_rovers],
            "excavators": [vehicle.to_dict() for vehicle in self.excavators],
            "assembly_robots": [
                vehicle.to_dict() for vehicle in self.assembly_robots
            ],
            "cargo_transporters": [
                vehicle.to_dict() for vehicle in self.cargo_transporters
            ],
            "ready_excavators": sum(
                1 for vehicle in self.excavators if vehicle.state == "idle"
            ),
            "active_assembly_robots": sum(
                1 for vehicle in self.assembly_robots
                if vehicle.state == "working"
            ),
            "active_cargo_transports": sum(
                1 for vehicle in self.cargo_transporters
                if vehicle.state in {
                    "cargo_outbound", "cargo_unloading", "cargo_returning"
                }
            ),
            "service_modules_used": dict(self.service_modules_used),
        }
