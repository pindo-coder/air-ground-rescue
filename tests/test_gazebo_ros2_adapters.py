"""Unit tests for Gazebo Harmonic + ROS 2 Jazzy sensor adapters, evaluator helpers, and mission prep."""

import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gazebo"))
sys.path.insert(0, str(ROOT / "gazebo/ros2_rescue"))
sys.path.insert(0, str(ROOT / "controllers/disaster_environment"))
sys.path.insert(0, str(ROOT / "controllers/uav_mapper"))
sys.path.insert(0, str(ROOT / "controllers/uav_map_rover"))

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None


class GazeboMissionHelpersTest(unittest.TestCase):
    @unittest.skipUnless(HAS_NUMPY and HAS_SCIPY and importlib.util.find_spec("cv2"),
                         "Gazebo mapper dependencies required")
    def test_mapper_selects_pose_at_image_acquisition(self):
        from uav_mapper_node import _nearest_sample

        pose_history = [(9.9, "old"), (10.0, "at frame"), (10.6, "latest")]
        self.assertEqual(_nearest_sample(pose_history, 10.02, 0.075), (10.0, "at frame"))
        self.assertIsNone(_nearest_sample(pose_history, 10.2, 0.075))

    def test_bridge_script_and_mission_flight_duration(self):
        from run_gazebo_mission import compute_flight_end_s

        cfg = {
            "flight": {
                "waypoints_xy_enu_m": [[0.0, 0.0], [30.0, 0.0], [30.0, 40.0]],
                "speed_mps": 2.5,
            }
        }
        self.assertEqual(compute_flight_end_s(cfg), 28)

    def test_evaluator_sdf_box_builder_valid_xml(self):
        import xml.etree.ElementTree as ET
        if not HAS_NUMPY:
            self.skipTest("numpy not installed in host Python")
        from evaluator_node import build_box_sdf

        xml_str = build_box_sdf("injected_obstacle_01", (0.24, 0.18, 0.18), (1.0, 2.0, 0.5), 0.7, (0.94, 0.36, 0.06))
        root = ET.fromstring(xml_str)
        self.assertEqual(root.tag, "sdf")
        self.assertEqual(root.find("./model").attrib["name"], "injected_obstacle_01")
        self.assertIsNotNone(root.find("./model/link/collision/geometry/box"))

    @unittest.skipUnless(HAS_NUMPY and HAS_SCIPY, "numpy and scipy required for evaluator")
    def test_supply_crate_spawns_with_visual_only(self):
        import xml.etree.ElementTree as ET
        from evaluator_node import spawn_supply_crate

        terrain = dict(origin=[0.0, 0.0], spacing=1.0, heights=[[0.0, 0.0], [0.0, 0.0]])
        delivery = dict(rover_xy_m=[0.25, 0.25], rover_yaw_rad=0.0)
        with patch("evaluator_node._gz_service") as service:
            spawn_supply_crate("search_d", terrain, delivery, 1)
        request = service.call_args.args[3]
        model = ET.fromstring(json.loads(request.removeprefix("sdf: "))).find("model")
        self.assertEqual(model.attrib["name"], "supply_crate_01")
        self.assertIsNotNone(model.find("link/visual/geometry/box"))
        self.assertIsNone(model.find("link/collision"))

    @unittest.skipUnless(HAS_NUMPY and HAS_SCIPY, "numpy and scipy required for evaluator")
    def test_rescue_evaluator_checks_independent_delivery_position_and_still_time(self):
        from evaluator_node import evaluate_run_offline

        terrain = {"victims": [{"id": 1, "xy": [1.0, 2.0]}]}
        report = dict(type="report", victim_id=1, xy_enu_m=[1.0, 2.0], time_s=2.0, cross_view=True)
        cases = (
            ("stationary despite zero claimed time", [1.0] * 21, [0.0] * 21, 0.0, True),
            ("stationary out of reach", [6.0] * 21, [0.0] * 21, 2.2, False),
            ("moving despite claimed stop", [0.8 + 0.01 * i for i in range(21)], [0.0] * 21, 2.2, False),
            ("turning in place", [1.0] * 21, [0.02 * i for i in range(21)], 2.2, False),
            ("single sample", [1.0], [0.0], 2.2, False),
        )
        with TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "rover").mkdir()
            for label, xs, yaws, claimed_s, success in cases:
                with self.subTest(label=label):
                    delivery = dict(
                        type="delivery", victim_id=1, rover_xy_m=[1.0, 2.0],
                        time_s=3.0, settled_s=claimed_s,
                    )
                    status = {"events": [report, delivery]}
                    (run / "rover/rescue_status.json").write_text(json.dumps(status), encoding="utf-8")
                    rows = [
                        dict(time_s=round(3.0 - 0.1 * (len(xs) - i - 1), 3), x=x, y=2.0, yaw_rad=yaw)
                        for i, (x, yaw) in enumerate(zip(xs, yaws))
                    ]
                    summary = evaluate_run_offline(
                        run, terrain, mode="rescue", trajectory_rows=rows, target_deliveries=1,
                    )
                    self.assertEqual(summary["valid_cross_view_reports"], 1)
                    self.assertEqual(summary["mission_success"], success)

    @unittest.skipUnless(HAS_NUMPY and HAS_SCIPY, "numpy and scipy required for evaluator")
    def test_evaluator_ignores_parking_platform_and_terrain_contacts(self):
        from evaluator_node import _is_obstacle_contact

        rover = "rover::chassis::collision"
        for ignored in ("terrain::body::tile_0_0", "rover_parking_platform::support::support"):
            with self.subTest(collision=ignored):
                self.assertFalse(_is_obstacle_contact(rover, ignored))
                self.assertFalse(_is_obstacle_contact(ignored, rover))
        self.assertTrue(_is_obstacle_contact(rover, "boulder::body::collision"))


@unittest.skipUnless(HAS_NUMPY, "numpy required for sensor adapter tests")
class GazeboRos2SensorAdaptersTest(unittest.TestCase):
    def test_survey_and_oblique_drone_pose_schedule(self):
        from rescue_d_environment import compute_survey_drone_pose, oblique_station_pose

        mission = {
            "flight": {
                "waypoints_xy_enu_m": [[0.0, 0.0], [10.0, 0.0]],
                "speed_mps": 2.0,
                "altitude_enu_m": 19.0,
            },
            "flight_end_s": 5.0,
        }
        xyz_mid, r, p, y = compute_survey_drone_pose(mission, None, 1.25)
        self.assertAlmostEqual(xyz_mid[0], 3.0, places=5)
        self.assertAlmostEqual(xyz_mid[2], 19.0, places=5)
        self.assertEqual((r, p), (0.0, 0.0))

        xyz_hold1, _, _, _ = compute_survey_drone_pose(mission, None, 1.55)
        xyz_hold2, _, _, _ = compute_survey_drone_pose(mission, None, 1.95)
        self.assertEqual(xyz_hold1, xyz_hold2)

        schedule = {
            "start_s": 6.0,
            "start_xyz_m": [10.0, 0.0, 19.0],
            "stations": [
                {"xyz_m": [14.0, 4.0, 17.0], "arrival_s": 8.0, "capture_s": 9.0,
                 "yaw_rad": 1.2, "pitch_rad": -0.45},
            ],
        }
        xyz_ob, r_ob, p_ob, y_ob = oblique_station_pose(schedule, 7.0)
        self.assertEqual(xyz_ob, [12.0, 2.0, 18.0])
        self.assertEqual((r_ob, p_ob, y_ob), (0.0, -0.45, 1.2))

    def test_navsat_enu_roundtrip_and_correlated_noise(self):
        import numpy as np
        from sensor_adapters import NavSatToEnu

        conv = NavSatToEnu(lat0_deg=31.2304, lon0_deg=121.4737, alt0_m=5.0, noise_std_m=0.0)
        target = np.array([23.5, -18.25, 19.0])
        lat, lon, alt = conv.enu_to_navsat(target)
        recovered = conv.convert(lat, lon, alt, apply_noise=False)
        np.testing.assert_allclose(recovered, target, atol=1e-6)

        noisy = NavSatToEnu(noise_std_m=0.04, noise_correlation=0.2, seed=42)
        samples = np.array([noisy.convert(0.0, 0.0, 0.0, apply_noise=True) for _ in range(500)])
        self.assertLess(float(np.abs(samples.mean(axis=0)).max()), 0.015)
        self.assertTrue(0.025 < float(samples.std(axis=0).mean()) < 0.055)

    def test_quaternion_and_rpy_roundtrip(self):
        import numpy as np
        from sensor_adapters import quat_xyzw_to_matrix, quat_xyzw_to_rpy, rpy_to_quat_xyzw

        r0, p0, y0 = 0.12, -0.35, 1.42
        q = rpy_to_quat_xyzw(r0, p0, y0)
        r1, p1, y1 = quat_xyzw_to_rpy(*q)
        self.assertAlmostEqual(r0, r1, places=6)
        self.assertAlmostEqual(p0, p1, places=6)
        self.assertAlmostEqual(y0, y1, places=6)
        mat = quat_xyzw_to_matrix(*q)
        np.testing.assert_allclose(mat @ mat.T, np.eye(3), atol=1e-6)

    def test_ros_image_depth_and_policy_rgbd_3ch_matches_4ch(self):
        import numpy as np
        from policy_features import encode_rgbd
        from sensor_adapters import ros_depth_to_meters, ros_image_to_bgr

        h, w = 24, 32
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = 180  # R
        rgb[:, :, 1] = 90   # G
        rgb[:, :, 2] = 30   # B
        bgr = ros_image_to_bgr(rgb.tobytes(), w, h, encoding="rgb8")
        self.assertEqual(tuple(bgr[0, 0]), (30, 90, 180))

        bgra = np.dstack([bgr, np.full((h, w), 255, dtype=np.uint8)])
        depth_raw = np.full((h, w), 2.5, dtype=np.float32)
        depth_raw[0, 0] = np.inf
        depth_raw[0, 1] = 0.0
        depth = ros_depth_to_meters(depth_raw.tobytes(), w, h, encoding="32FC1")
        self.assertTrue(np.isnan(depth[0, 0]) and np.isnan(depth[0, 1]))
        self.assertAlmostEqual(float(depth[5, 5]), 2.5, places=5)

        f_bgra = encode_rgbd(bgra.tobytes(), depth, encoding="bgra")
        f_rgb = encode_rgbd(rgb, depth, encoding="rgb8")
        f_bgr = encode_rgbd(bgr, depth, encoding="bgr8")
        np.testing.assert_allclose(f_rgb, f_bgra, atol=1e-6)
        np.testing.assert_allclose(f_bgr, f_bgra, atol=1e-6)

    def test_pointcloud2_decoding_and_wheel_averaging(self):
        import numpy as np
        from sensor_adapters import extract_wheel_positions_rad, ros_pointcloud2_to_xyz

        pts = [(1.5, -0.4, 0.25, 1.0, 1), (2.0, 0.8, 0.1, 1.0, 2)]
        blob = b"".join(struct.pack("<ffffH", *p) for p in pts)
        fields = [
            {"name": "x", "offset": 0, "datatype": 7},
            {"name": "y", "offset": 4, "datatype": 7},
            {"name": "z", "offset": 8, "datatype": 7},
            {"name": "intensity", "offset": 12, "datatype": 7},
            {"name": "ring", "offset": 16, "datatype": 4},
        ]
        xyz = ros_pointcloud2_to_xyz(blob, width=2, height=1, point_step=18, row_step=36, fields=fields)
        np.testing.assert_allclose(xyz, np.array([[1.5, -0.4, 0.25], [2.0, 0.8, 0.1]], dtype=np.float32))

        wheels = extract_wheel_positions_rad(
            ["front_left_wheel_joint", "rear_left_wheel_joint", "front_right_wheel_joint", "rear_right_wheel_joint"],
            [1.0, 1.2, 2.0, 2.4],
        )
        np.testing.assert_allclose(wheels, np.array([1.1, 2.2]))

    @unittest.skipUnless(HAS_SCIPY, "scipy required for route_tracker obstacle extraction")
    def test_pointcloud_and_ccw_laserscan_match_webots_lidar_obstacles(self):
        import numpy as np
        from route_tracker import extract_lidar_obstacles, extract_pointcloud_obstacles
        from survey_map import Grid

        grid = Grid((-10.0, -10.0), 0.5, (40, 40))
        ground_z = np.zeros((40, 40), dtype=float)
        pose_xyz = np.array([0.0, 0.0, 0.0])
        rpy = (0.0, 0.0, 0.0)

        # Place obstacle cluster at +X = 2.0 m, Y = 0.0 m (azimuth 0)
        n = 360
        cw_angles = np.linspace(math.pi, -math.pi, n, endpoint=False)
        ccw_angles = np.linspace(-math.pi, math.pi, n, endpoint=False)
        ranges_cw = np.full(n, np.inf)
        ranges_ccw = np.full(n, np.inf)
        for idx, a in enumerate(cw_angles):
            if abs(a) < 0.06:
                ranges_cw[idx] = 2.0
        for idx, a in enumerate(ccw_angles):
            if abs(a) < 0.06:
                ranges_ccw[idx] = 2.0

        b_cw, e_cw = extract_lidar_obstacles([ranges_cw], pose_xyz, rpy, grid, ground_z, elevations=(0.0,), azimuth_order="cw")
        b_ccw, e_ccw = extract_lidar_obstacles([ranges_ccw], pose_xyz, rpy, grid, ground_z, elevations=(0.0,), azimuth_order="ccw")
        pc_pts = np.array([[2.0 * math.cos(a), 2.0 * math.sin(a), 0.0] for a in ccw_angles if abs(a) < 0.06])
        b_pc, e_pc = extract_pointcloud_obstacles(pc_pts, pose_xyz, rpy, grid, ground_z)

        self.assertGreater(len(b_cw), 0)
        np.testing.assert_allclose(b_ccw.mean(axis=0), b_cw.mean(axis=0), atol=0.05)
        np.testing.assert_allclose(b_pc.mean(axis=0), b_cw.mean(axis=0), atol=0.05)
        np.testing.assert_allclose(e_pc.mean(axis=0), e_cw.mean(axis=0), atol=0.05)


if __name__ == "__main__":
    unittest.main()
