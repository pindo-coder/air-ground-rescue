from pathlib import Path
import math
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for rel in ("controllers/uav_mapper", "controllers/uav_map_rover", "controllers/search_agent"):
    p = str(ROOT / rel)
    if p not in sys.path:
        sys.path.insert(0, p)

from aerial_victims import AerialVictimTracker
from rescue_mission import confirm_victim_onboard, plan_next_victim, select_victim_approach_point
from route_tracker import OnboardObstacleMap
from survey_map import BLOCKED, FREE, Grid, UNKNOWN
from vctp import FineTerrainSurface, VctpChecker


class _FakeDetector:
    def __init__(self, detections):
        self._detections = detections

    def localize(self, bgr, depth, pose, pose_cov, attitude=(0.0, 0.0)):
        return list(self._detections)


class TestUavRescue(unittest.TestCase):
    def test_aerial_victim_tracker_clusters_and_filters_single_view(self):
        tracker = AerialVictimTracker(association_radius_m=1.8, min_score=0.35)
        cov = np.diag([0.25, 0.25])
        # Two detections of victim A near (10.0, 5.0)
        tracker.ingest_candidates(
            [dict(mean_enu=np.array([10.1, 4.9]), covariance_enu=cov, score=0.78, depth=18.5)],
            frame_index=1, timestamp_s=10.0, observer_xy=(9.0, 5.0), evidence_path="frames/001.jpg",
        )
        tracker.ingest_candidates(
            [dict(mean_enu=np.array([9.9, 5.1]), covariance_enu=cov, score=0.84, depth=18.4)],
            frame_index=2, timestamp_s=11.0, observer_xy=(11.0, 5.0), evidence_path="frames/002.jpg",
        )
        # One isolated false positive far away -> must be filtered out by min_views=2
        tracker.ingest_candidates(
            [dict(mean_enu=np.array([-5.0, -5.0]), covariance_enu=cov, score=0.71, depth=18.6)],
            frame_index=3, timestamp_s=12.0, observer_xy=(-5.0, -4.0), evidence_path="frames/003.jpg",
        )

        confirmed = tracker.export(min_views=2, high_score_single=0.95)
        self.assertEqual(len(confirmed), 1)
        cand = confirmed[0]
        self.assertEqual(cand["views"], 2)
        self.assertGreaterEqual(cand["mean_enu_m"][0], 9.9)
        self.assertLessEqual(cand["mean_enu_m"][0], 10.1)
        self.assertGreaterEqual(cand["mean_enu_m"][1], 4.9)
        self.assertLessEqual(cand["mean_enu_m"][1], 5.1)
        self.assertEqual(cand["evidence"], ["frames/001.jpg", "frames/002.jpg"])

    def test_select_victim_approach_point_respects_standoff_and_vctp(self):
        grid = Grid(origin=np.array([0.0, 0.0]), resolution=0.5, shape=(24, 24))
        states = np.full(grid.shape, FREE, dtype=np.int8)
        # Put an obstacle/victim footprint at (8.0, 6.0) so approach must stay outside it.
        states[grid.cell((8.0, 6.0))] = BLOCKED
        cost = np.ones(grid.shape, dtype=float)
        fine_grid = Grid(origin=grid.origin, resolution=0.125, shape=(grid.shape[0] * 4, grid.shape[1] * 4))
        fine_z = np.zeros(fine_grid.shape, dtype=float)
        vctp = VctpChecker(FineTerrainSurface(fine_grid, fine_z, grid, states))

        res = select_victim_approach_point(
            grid, states, cost,
            start_xy=(2.0, 6.0), victim_xy=(8.0, 6.0),
            rover_radius_m=0.40, vctp=vctp, planner_mode="igha_vctp",
        )
        self.assertEqual(res["status"], "PLANNED")
        self.assertEqual(res["planner"], "igha_vctp")
        self.assertGreaterEqual(res["standoff_m"], 0.80)
        self.assertLessEqual(res["standoff_m"], 1.58)
        # Approach yaw should point from approach_xy toward victim_xy.
        ax, ay = res["approach_xy_m"]
        expected_yaw = math.atan2(6.0 - ay, 8.0 - ax)
        self.assertAlmostEqual(res["approach_yaw_rad"], expected_yaw, places=3)
        self.assertEqual(res["vctp_audit"]["fail_count"], 0)

    def test_select_victim_approach_point_reports_no_approach_in_unknown_zone(self):
        grid = Grid(origin=np.array([0.0, 0.0]), resolution=0.5, shape=(20, 20))
        states = np.full(grid.shape, UNKNOWN, dtype=np.int8)
        states[2:6, 2:6] = FREE
        cost = np.ones(grid.shape, dtype=float)

        res = select_victim_approach_point(
            grid, states, cost,
            start_xy=(2.0, 2.0), victim_xy=(8.0, 8.0),
            rover_radius_m=0.40, vctp=None, planner_mode="astar",
        )
        self.assertEqual(res["status"], "NO_APPROACH")
        self.assertIn("dual_route", res)

    def test_plan_next_victim_picks_nearest_confirmed_reachable(self):
        grid = Grid(origin=np.array([0.0, 0.0]), resolution=0.5, shape=(28, 28))
        states = np.full(grid.shape, FREE, dtype=np.int8)
        # Wall off the far-right region with UNKNOWN so victim 3 is unreachable.
        states[:, 20:] = UNKNOWN
        cost = np.ones(grid.shape, dtype=float)

        candidates = [
            dict(id=1, mean_enu_m=[8.5, 6.0], covariance_enu=[[0.2, 0.0], [0.0, 0.2]]),
            dict(id=2, mean_enu_m=[4.5, 6.0], covariance_enu=[[0.2, 0.0], [0.0, 0.2]]),
            dict(id=3, mean_enu_m=[12.0, 6.0], covariance_enu=[[0.2, 0.0], [0.0, 0.2]]),
        ]
        chosen, all_plans = plan_next_victim(
            grid, states, cost, rover_xy=(2.0, 6.0), rover_yaw=0.0,
            remaining_victims=candidates, rover_radius_m=0.40, vctp=None, planner_mode="astar",
        )
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["victim_candidate"]["id"], 2)
        statuses = {p["victim_candidate"]["id"]: p["status"] for p in all_plans}
        self.assertEqual(statuses[1], "PLANNED")
        self.assertEqual(statuses[2], "PLANNED")
        self.assertEqual(statuses[3], "NO_APPROACH")

    def test_confirm_victim_onboard_fuses_match_and_rejects_far_detection(self):
        prior_mean = [10.0, 5.0]
        prior_cov = [[0.25, 0.0], [0.0, 0.25]]
        good_det = _FakeDetector([
            dict(mean=np.array([10.12, 4.92]), cov=np.diag([0.09, 0.09]), score=0.86, depth=1.35)
        ])
        res = confirm_victim_onboard(
            good_det, None, None, pose_xy=(8.8, 5.0), yaw=0.0, roll=0.0, pitch=0.0,
            score_min=0.68, victim_mean_enu=prior_mean, victim_cov_enu=prior_cov,
        )
        self.assertTrue(res["confirmed"])
        self.assertGreater(res["fused_xy_m"][0], 10.0)
        self.assertLess(res["fused_xy_m"][0], 10.12)
        self.assertLess(np.linalg.det(res["fused_cov_enu"]), np.linalg.det(prior_cov))

        far_det = _FakeDetector([
            dict(mean=np.array([14.0, 5.0]), cov=np.diag([0.09, 0.09]), score=0.86, depth=5.2)
        ])
        res_far = confirm_victim_onboard(
            far_det, None, None, pose_xy=(8.8, 5.0), yaw=0.0, roll=0.0, pitch=0.0,
            score_min=0.68, victim_mean_enu=prior_mean, victim_cov_enu=prior_cov,
        )
        self.assertFalse(res_far["confirmed"])

    def test_person_detector_dual_view_yolo_routing(self):
        from perception import PersonDetector
        if not (ROOT / "models/yolo11n_victim.pt").is_file():
            self.skipTest("Fine-tuned aerial weights not present")
        air = PersonDetector("yolo", ROOT, max_depth=24.0)
        gnd = PersonDetector("yolo", ROOT, max_depth=8.0)
        self.assertEqual(air.view, "aerial")
        self.assertEqual(air.weights_path.name, "yolo11n_victim.pt")
        self.assertTrue(air.yolo_single_pass)
        self.assertEqual(gnd.view, "ground")
        self.assertEqual(gnd.weights_path.name, "yolo11n.pt")
        self.assertFalse(gnd.yolo_single_pass)

    def test_sample_random_victims_d_valid_and_reproducible(self):
        tools_dir = str(ROOT / "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        eval_dir = str(ROOT / "evaluation")
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)
        if not (ROOT / "evaluation/search/D.json").is_file():
            self.skipTest("D.json not built")
        import json
        from run_uav_rescue import sample_random_victims_d

        cfg = json.loads((ROOT / "config/scenes/uav_mapping_full_D.json").read_text(encoding="utf-8"))
        v1, _ = sample_random_victims_d(cfg, seed=42, count=3, random_clothing=True)
        v2, _ = sample_random_victims_d(cfg, seed=42, count=3, random_clothing=True)
        v3, _ = sample_random_victims_d(cfg, seed=99, count=3, random_clothing=True)
        self.assertEqual(v1, v2)
        self.assertNotEqual([v["xy"] for v in v1], [v["xy"] for v in v3])
        self.assertEqual(len(v1), 3)
        for v in v1:
            self.assertIsNotNone(v["shirt_rgb"])
            self.assertGreaterEqual(math.hypot(v["xy"][0] - (-18.0), v["xy"][1] - (-11.0)), 4.5)
        for i in range(len(v1)):
            for j in range(i + 1, len(v1)):
                self.assertGreaterEqual(math.dist(v1[i]["xy"], v1[j]["xy"]), 7.5)

    def test_victim_no_runover_exclusion_and_near_side_approach(self):
        grid = Grid(origin=np.array([0.0, 0.0]), resolution=0.5, shape=(32, 32))
        states = np.full(grid.shape, FREE, dtype=np.int8)
        cost = np.ones(grid.shape, dtype=float)
        fine_grid = Grid(origin=grid.origin, resolution=0.125, shape=(grid.shape[0] * 4, grid.shape[1] * 4))
        fine_z = np.zeros(fine_grid.shape, dtype=float)
        vctp = VctpChecker(FineTerrainSurface(fine_grid, fine_z, grid, states))

        victim_a = [6.0, 8.0]
        victim_b = [12.0, 8.0]
        # Approaching victim A from the west must choose a west-side approach point (x < 6.0).
        leg_a = select_victim_approach_point(
            grid, states, cost,
            start_xy=(2.0, 8.0), victim_xy=victim_a,
            rover_radius_m=0.35, vctp=vctp, planner_mode="igha_vctp",
            exclude_victims_xy=[victim_a, victim_b],
        )
        self.assertEqual(leg_a["status"], "PLANNED")
        self.assertLess(leg_a["approach_xy_m"][0], victim_a[0])
        self.assertGreaterEqual(leg_a["standoff_m"], 1.05)

        # Departing from the west of victim A toward victim B (due east) must route around victim A.
        leg_b = select_victim_approach_point(
            grid, states, cost,
            start_xy=leg_a["approach_xy_m"], victim_xy=victim_b,
            rover_radius_m=0.35, vctp=vctp, planner_mode="igha_vctp",
            exclude_victims_xy=[victim_a, victim_b],
        )
        self.assertEqual(leg_b["status"], "PLANNED")
        route_b = np.asarray(leg_b["route_xy_m"], dtype=float)
        min_dist_to_a = float(np.min(np.linalg.norm(route_b - np.asarray(victim_a), axis=1)))
        self.assertGreaterEqual(min_dist_to_a, 0.85)

        # Protected onboard victim zone must not be erased by dynamic-obstacle stale clearing.
        omap = OnboardObstacleMap(grid, states, np.zeros(grid.shape, dtype=float), cost, rover_radius_m=0.35)
        added = omap.mark_no_runover_zone(victim_a, radius_m=0.68, keep_clear_xy=leg_a["approach_xy_m"], keep_clear_radius_m=0.50)
        self.assertGreater(added, 0)
        empty = np.empty((0, 2), dtype=float)
        for _ in range(30):
            omap.update(leg_a["approach_xy_m"], empty, empty)
        self.assertTrue(omap.fine_onboard_blocked[omap.fine_grid.cell(victim_a)])


if __name__ == "__main__":
    unittest.main()
