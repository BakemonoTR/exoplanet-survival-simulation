"""Focused acceptance tests for supervised prefab assembly robots."""

import unittest

from src.systems.mission_profile import MissionProfile
from src.systems.surface_fleet import SurfaceFleet


def _advance_fleet(fleet: SurfaceFleet, grid_energy_kwh: float = 0.0) -> dict:
    return fleet.tick(
        available_grid_energy_kwh=grid_energy_kwh,
        excavation_callback=lambda _job, _mass: (0, 0.0, True),
        unload_callback=lambda _payload: None,
    )


class RoboticAssemblyTests(unittest.TestCase):
    def test_vehicle_classes_publish_non_overlapping_physical_roles(self):
        fleet = SurfaceFleet(MissionProfile.load())
        matrix = fleet.to_dict()["capability_matrix"]

        self.assertIn(
            "excavate_verified_exposed_face",
            matrix["excavator"]["capabilities"],
        )
        self.assertIn("excavation", matrix["assembly_robot"]["prohibited_roles"])
        self.assertIn(
            "restrained_heavy_module_transport",
            matrix["cargo_transporter"]["capabilities"],
        )
        self.assertIn(
            "structural_assembly",
            matrix["cargo_transporter"]["prohibited_roles"],
        )
        self.assertLessEqual(
            fleet.assembly_robot_spec["payload_kg"], 500.0
        )
        self.assertGreater(
            fleet.cargo_transporter_spec["payload_kg"],
            fleet.assembly_robot_spec["payload_kg"],
        )

    def test_requires_supervision_and_has_finite_work_rate(self):
        profile = MissionProfile.load()
        fleet = SurfaceFleet(profile)

        self.assertEqual(len(fleet.assembly_robots), 4)

        # One crew member can supervise at most two robots. Dispatch itself
        # earns no work, so delivered hardware is never free readiness.
        dispatch = fleet.assembly_assist(
            site_id="habitat-01", builder_count=1, site_x=5, site_y=7
        )
        self.assertEqual(dispatch["work_hours"], 0.0)
        self.assertEqual(
            sum(robot.state == "outbound" for robot in fleet.assembly_robots),
            2,
        )
        dispatched_robots = [
            robot for robot in fleet.assembly_robots
            if robot.state == "outbound"
        ]
        approach_cells = {
            (robot.mission["target_x"], robot.mission["target_y"])
            for robot in dispatched_robots
        }
        self.assertEqual(2, len(approach_cells))
        self.assertNotIn((5, 7), approach_cells)
        for robot in dispatched_robots:
            route = robot.mission["route"]
            self.assertGreaterEqual(len(route), 2)
            self.assertTrue(all(
                abs(first["x"] - second["x"])
                + abs(first["y"] - second["y"]) == 1
                for first, second in zip(route, route[1:])
            ))

        before_battery = sum(
            robot.battery_kwh for robot in fleet.assembly_robots[:2]
        )
        for _ in range(10):
            _advance_fleet(fleet)
            if not any(
                robot.state == "outbound"
                for robot in fleet.assembly_robots[:2]
            ):
                break
        self.assertLess(
            sum(robot.battery_kwh for robot in fleet.assembly_robots[:2]),
            before_battery,
        )
        assisted = fleet.assembly_assist(site_id="habitat-01", builder_count=1)
        self.assertEqual(len(assisted["active_robot_ids"]), 2)
        self.assertAlmostEqual(assisted["equivalent_workers"], 4.0)
        self.assertAlmostEqual(assisted["work_hours"], 2.0 / 3.0)
        self.assertAlmostEqual(
            sum(robot.total_assembly_work_hours for robot in fleet.assembly_robots),
            2.0 / 3.0,
        )

        unsupervised = fleet.assembly_assist(
            site_id="habitat-01", builder_count=0
        )
        self.assertEqual(unsupervised["work_hours"], 0.0)
        self.assertTrue(all(
            robot.state == "waiting_supervision"
            for robot in fleet.assembly_robots
            if robot.mission and robot.mission.get("site_id") == "habitat-01"
        ))

    def test_release_requires_return_before_reassignment(self):
        fleet = SurfaceFleet(MissionProfile.load())
        fleet.assembly_assist(site_id="solar-01", builder_count=1)
        _advance_fleet(fleet)
        fleet.release_assembly_site("solar-01")

        self.assertEqual(
            sum(robot.state == "returning" for robot in fleet.assembly_robots),
            2,
        )
        self.assertEqual(
            sum(robot.completed_jobs for robot in fleet.assembly_robots), 2
        )
        _advance_fleet(fleet)
        returned = [
            robot for robot in fleet.assembly_robots if robot.mission is None
        ]
        self.assertEqual(len(returned), 4)
        self.assertTrue(
            all(robot.state in {"idle", "charging"} for robot in returned)
        )

    def test_base_service_consumes_a_finite_serialized_module(self):
        fleet = SurfaceFleet(MissionProfile.load())
        robot = fleet.assembly_robots[0]
        robot.condition = 0.70
        inventory = {"assembly_robot_service_module": 1}

        def consume(material, quantity):
            if inventory.get(material, 0) < quantity:
                return False
            inventory[material] -= quantity
            return True

        result = fleet.tick(
            available_grid_energy_kwh=0.0,
            excavation_callback=lambda _job, _mass: (0, 0.0, True),
            unload_callback=lambda _payload: None,
            service_callback=consume,
        )

        self.assertEqual(0, inventory["assembly_robot_service_module"])
        self.assertEqual(1, robot.service_count)
        self.assertEqual(0.98, robot.condition)
        self.assertEqual("vehicle_service_complete", result["events"][0]["type"])

        stranded = fleet.assembly_robots[1]
        stranded.condition = 0.49
        fleet.tick(
            available_grid_energy_kwh=0.0,
            excavation_callback=lambda _job, _mass: (0, 0.0, True),
            unload_callback=lambda _payload: None,
            service_callback=consume,
        )
        self.assertEqual("fault", stranded.state)
        self.assertEqual("maintenance_spares_exhausted", stranded.fault_reason)


if __name__ == "__main__":
    unittest.main()
