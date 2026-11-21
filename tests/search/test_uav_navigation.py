"""Unit tests for closed-loop UAV-map route tracking and LiDAR/RGB-D obstacle checking."""
from pathlib import Path
import math
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for name in ("uav_mapper", "uav_map_rover"):
    sys.path.insert(0, str(ROOT / "controllers" / name))

from grid_route import plan
from route_tracker import (
    DEPTH_OFFSET_BODY,
    DEPTH_PITCH_RAD,
    LIDAR_LAYER_ELEVATIONS_RAD,
    LIDAR_OFFSET_BODY,
    LocalPlannerInput,
    OnboardObstacleMap,
    RouteNavigationController,
    RoverPoseEstimator,
    RuleLocalPlanner,
    extract_lidar_obstacles,
    extract_rgbd_obstacles,
    gravity_rotation,
    rotation_2d,
    sample_ground_z,
    wheel_velocities_rad_s,
)
from survey_map import BLOCKED, FREE, UNKNOWN, Grid


def _sloped_ground(grid, slope_x=0.25, base_z=1.0):
    rows, cols = np.indices(grid.shape)
    x = grid.origin[0] + (cols + 0.5) * grid.resolution
    return base_z + slope_x * x


class UavNavigationTests(unittest.TestCase):
    def test_lidar_rejects_14_deg_uphill_slope_but_detects_obstacle_on_slope(self):
        grid = Grid.from_bounds([[0.0, -5.0], [10.0, 5.0]], 0.5)
        slope = math.tan(math.radians(14.0))
        ground_z = _sloped_ground(grid, slope_x=slope, base_z=1.0)
        rover_xy = np.array([2.0, 0.0])
        rover_z = 1.0 + slope * rover_xy[0]
        # Rover pitched nose-up by 14 deg -> negative pitch in Webots ENU (+Y left).
        pitch = -math.radians(14.0)
        pose_xyz = np.array([rover_xy[0], rover_xy[1], rover_z])

        # Synthesize LiDAR hitting smooth uphill ground ahead (no obstacles).
        n_rays = 360
        angles = np.linspace(math.pi, -math.pi, n_rays, endpoint=False)
        layers = []
        r_grav = gravity_rotation(0.0, pitch)
        for el in LIDAR_LAYER_ELEVATIONS_RAD:
            ranges = np.full(n_rays, 8.0)
            for i, az in enumerate(angles):
                dir_body = np.array([math.cos(el) * math.cos(az),
                                     math.cos(el) * math.sin(az),
                                     math.sin(el)])
                dir_enu = dir_body @ r_grav.T
                orig_enu = pose_xyz + LIDAR_OFFSET_BODY @ r_grav.T
                # Intersect ray with plane z = 1.0 + slope * x
                denom = dir_enu[2] - slope * dir_enu[0]
                if denom < -1e-6:
                    t = (1.0 + slope * orig_enu[0] - orig_enu[2]) / denom
                    if 0.3 < t < 7.5:
                        ranges[i] = t
            layers.append(ranges)

        body_obs, enu_obs = extract_lidar_obstacles(layers, pose_xyz, (0.0, pitch, 0.0), grid, ground_z)
        self.assertEqual(len(body_obs), 0)

        # Now inject a 0.45 m tall boulder straight ahead at range 2.0 m across front 10 rays.
        for layer in layers:
            layer[175:185] = 2.0
        body_obs2, enu_obs2 = extract_lidar_obstacles(layers, pose_xyz, (0.0, pitch, 0.0), grid, ground_z)
        self.assertGreater(len(body_obs2), 0)
        self.assertAlmostEqual(float(np.median(enu_obs2[:, 0])), rover_xy[0] + 2.0 * math.cos(math.radians(14.0)), delta=0.35)

    def test_rgbd_rejects_flat_ground_and_detects_raised_obstacle_and_drop(self):
        grid = Grid.from_bounds([[0.0, -5.0], [10.0, 5.0]], 0.5)
        ground_z = np.full(grid.shape, 1.0, dtype=float)
        pose_xyz = np.array([2.0, 0.0, 1.0])

        h, w = 240, 320
        focal = w / 2.0
        yy, xx = np.indices((h, w), dtype=float)
        cp, sp = math.cos(DEPTH_PITCH_RAD), math.sin(DEPTH_PITCH_RAD)
        # Ray direction in body frame for unit optical-axis depth:
        # cam_dir = (1, -(u-cx)/f, -(v-cy)/f)
        # body_z = 0.30 + d * (-sp * 1 + cp * cam_z)
        cam_z = -(yy - (h - 1) / 2.0) / focal
        dz_dd = -sp + cp * cam_z
        depth = np.where(dz_dd < -0.02, -DEPTH_OFFSET_BODY[2] / np.where(dz_dd < -0.02, dz_dd, -1.0), 12.0)
        depth = np.clip(depth, 0.2, 12.0)

        b0, e0, drops0 = extract_rgbd_obstacles(depth, pose_xyz, (0.0, 0.0, 0.0), grid, ground_z)
        self.assertEqual(len(b0), 0)
        self.assertEqual(drops0, 0)

        # Place a low obstacle block in center of image at 1.6 m depth.
        depth_obs = depth.copy()
        depth_obs[80:140, 130:190] = 1.6
        b1, e1, drops1 = extract_rgbd_obstacles(depth_obs, pose_xyz, (0.0, 0.0, 0.0), grid, ground_z)
        self.assertGreater(len(b1), 0)
        self.assertEqual(drops1, 0)
        self.assertAlmostEqual(float(np.median(e1[:, 0])), 2.0 + 0.26 + 1.6, delta=0.25)

        # Simulate a gully drop at bottom-center (depth suddenly 2.1 m instead of ground ~0.6 m).
        depth_drop = depth.copy()
        depth_drop[180:225, 130:190] = 2.1
        b2, e2, drops2 = extract_rgbd_obstacles(depth_drop, pose_xyz, (0.0, 0.0, 0.0), grid, ground_z)
        self.assertGreater(drops2, 0)
        self.assertGreater(len(b2), 0)

    def test_onboard_map_replans_detour_and_stops_when_corridor_blocked_by_unknowns(self):
        grid = Grid.from_bounds([[0.0, 0.0], [10.0, 10.0]], 0.5)
        states = np.full(grid.shape, FREE, dtype=np.int8)
        ground_z = np.zeros(grid.shape, dtype=float)
        cost = np.ones(grid.shape, dtype=float)
        start = grid.center((10, 3))
        goal = grid.center((10, 16))

        omap = OnboardObstacleMap(grid, states, ground_z, cost, rover_radius_m=0.40, confirm_frames=2)
        initial = plan(grid, states, start, goal, 0.40, cost)

        # Inject a single boulder on the open route -> should replan cleanly inside FREE terrain.
        boulder_pt = np.tile(grid.center((10, 9)), (5, 1))
        omap.update(start + np.array([1.5, 0.0]), boulder_pt, np.empty((0, 2)))
        omap.update(start + np.array([1.5, 0.0]), boulder_pt, np.empty((0, 2)))
        needs, _ = omap.route_needs_replan(initial, 0, start, boulder_pt)
        self.assertTrue(needs)
        detour, reason = omap.replan(start, goal, boulder_pt)
        self.assertIsNotNone(detour)
        self.assertIn("REPLANNED", reason)
        self.assertTrue(all(omap.effective_states[omap.fine_grid.cell(p)] == FREE for p in detour))

        # A 3-cell-wide (1.5 m) confirmed corridor flanked by UNKNOWN cells:
        # 1) a localized 0.15 m rock still leaves a confirmed bypass inside FREE terrain;
        # 2) a wider barrier across the corridor forces a safe stop with explicit reason.
        narrow = np.full(grid.shape, UNKNOWN, dtype=np.int8)
        narrow[9:12, 2:18] = FREE
        omap2 = OnboardObstacleMap(grid, narrow, ground_z, cost, rover_radius_m=0.40, confirm_frames=2)
        for _ in range(2):
            omap2.update(start + np.array([1.5, 0.0]), boulder_pt, boulder_pt)
        bypass, _ = omap2.replan(start, goal, boulder_pt)
        self.assertIsNotNone(bypass)
        self.assertTrue(all(narrow[grid.cell(p)] == FREE for p in bypass))

        barrier_pts = np.column_stack((np.full(11, grid.center((10, 9))[0]), np.linspace(4.65, 5.85, 11)))
        for _ in range(2):
            omap2.update(start + np.array([1.5, 0.0]), barrier_pts, barrier_pts)
        no_route, stop_reason = omap2.replan(start, goal, barrier_pts)
        self.assertIsNone(no_route)
        self.assertIn("OBSTACLE_BLOCKED_NO_CONFIRMED_DETOUR", stop_reason)
        self.assertIn("unknown", stop_reason.lower())

    def test_controller_tracks_open_route_and_settles_at_goal(self):
        grid = Grid.from_bounds([[0.0, 0.0], [10.0, 10.0]], 0.5)
        states = np.full(grid.shape, FREE, dtype=np.int8)
        ground_z = np.zeros(grid.shape, dtype=float)
        cost = np.ones(grid.shape, dtype=float)
        start = grid.center((10, 3))
        goal = grid.center((10, 12))
        route = plan(grid, states, start, goal, 0.40, cost)

        omap = OnboardObstacleMap(grid, states, ground_z, cost, rover_radius_m=0.40)
        ctrl = RouteNavigationController(omap, route, goal, settle_time_s=1.5)

        x, y, yaw = float(start[0]), float(start[1]), 0.0
        v_cur, w_cur = 0.0, 0.0
        dt = 0.1
        for step in range(400):
            now = step * dt
            v_cmd, w_cmd, _ = ctrl.step(
                now, dt, np.array([x, y, 0.0]), (0.0, 0.0, yaw),
                abs(v_cur), abs(w_cur),
                np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)),
            )
            # Simple kinematic vehicle response.
            v_cur = 0.4 * v_cur + 0.6 * v_cmd
            w_cur = 0.4 * w_cur + 0.6 * w_cmd
            x += v_cur * math.cos(yaw) * dt
            y += v_cur * math.sin(yaw) * dt
            yaw += w_cur * dt
            if ctrl.report.status == "ARRIVED":
                break

        self.assertEqual(ctrl.report.status, "ARRIVED")
        self.assertLessEqual(math.hypot(x - goal[0], y - goal[1]), 0.35)
        self.assertLess(abs(v_cur), 0.03)

    def test_wheel_velocity_saturation_preserves_turn_ratio(self):
        left, right = wheel_velocities_rad_s(0.42, 0.85)
        self.assertLessEqual(max(abs(left), abs(right)), 9.5 + 1e-6)
        self.assertGreater(right, left)

    def test_pose_estimator_fuses_encoders_and_gps(self):
        est = RoverPoseEstimator([1.0, 2.0, 0.5], 0.0)
        xyz, yaw = est.update([1.05, 2.0, 0.5], (0.0, 0.0, 0.0), [0.5, 0.5], 0.1)
        self.assertGreater(xyz[0], 1.0)
        self.assertAlmostEqual(yaw, 0.0)

    def test_pose_estimator_remains_settled_under_realistic_gps_and_imu_noise(self):
        rng = np.random.default_rng(7)
        est = RoverPoseEstimator([1.0, 2.0, 0.5], 0.5)
        for _ in range(30):
            gps = np.array([1.0, 2.0, 0.5]) + rng.normal(0.0, 0.025, 3)
            yaw = 0.5 + float(rng.normal(0.0, 0.003))
            est.update(gps, (0.0, 0.0, yaw), [0.0, 0.0], 0.1)
        self.assertLess(est.speed_mps, 0.03)
        self.assertLess(est.yaw_rate_rad_s, 0.08)


if __name__ == "__main__":
    unittest.main()
