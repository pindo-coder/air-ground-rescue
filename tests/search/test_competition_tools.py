"""Unit tests for IGHA* start-pose jitter recovery, safety/smoothness metrics, DSM import & dashboard."""
from pathlib import Path
import json
import math
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "controllers/uav_mapper"))
sys.path.insert(0, str(ROOT / "controllers/uav_map_rover"))
sys.path.insert(0, str(ROOT / "tools"))

from igha_star import plan_igha_star
from import_real_dsm import dsm_to_uav_map
from posture_casualty import ALL_POSTURES, POSTURE_LOCAL_POINTS, build_posture_casualty_vrml
from rescue_dashboard import build_html
from route_tracker import OnboardObstacleMap, RouteNavigationController
from run_uav_rescue import apply_weather_to_world
from safety_metrics import compute_run_safety_smoothness
from survey_map import BLOCKED, FREE, UNKNOWN, Grid
from survey_refinement import schedule_dynamic_oblique, select_dynamic_patches
from vctp import FineTerrainSurface, VctpChecker


class CompetitionToolsTests(unittest.TestCase):
    def test_igha_recovers_from_single_pixel_belly_noise_at_start(self):
        coarse = Grid.from_bounds([[0.0, 0.0], [8.0, 6.0]], 0.5)
        fine = Grid.from_bounds([[0.0, 0.0], [8.0, 6.0]], 0.125)
        states = np.full(coarse.shape, FREE, dtype=np.int8)
        fine_z = np.zeros(fine.shape, dtype=float)
        # Place a single-pixel bump right at front belly offset (+0.21, 0) for yaw=0 (clear when rotated).
        rc = fine.cell((2.21, 3.0))
        fine_z[rc] = 0.26
        vctp = VctpChecker(FineTerrainSurface(fine, fine_z, coarse, states))
        self.assertEqual(vctp.check_pose(2.0, 3.0, 0.0).status, "FAIL")
        res = plan_igha_star(
            coarse, states, vctp,
            start_xy=[2.0, 3.0], goal_xy=[6.0, 3.0],
            rover_radius_m=0.40, start_yaw=0.0,
        )
        self.assertEqual(res["status"], "PLANNED")
        self.assertEqual(res["vctp"]["fail_count"], 0)

    def test_real_dsm_import_produces_valid_vctp_prior(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            # Synthetic 80x80 DSM at 0.125 m (10 m x 10 m) with a gentle 5 deg slope & one 0.6 m boulder.
            rr, cc = np.indices((80, 80))
            dsm = (cc * 0.125 * math.tan(math.radians(5.0))).astype(np.float32)
            dsm[36:44, 36:44] += 0.65
            dsm_tif = td_path / "synthetic_dsm.tif"
            Image.fromarray(dsm, mode="F").save(dsm_tif)
            map_path, meta = dsm_to_uav_map(dsm_tif, td_path / "out", pixel_size_m=0.125, seed_xy=(1.5, 1.5))
            self.assertTrue(map_path.is_file())
            self.assertGreater(meta["free_cells"], 200)
            self.assertGreater(meta["blocked_cells"], 0)
            surf = FineTerrainSurface.from_uav_map(map_path)
            res = plan_igha_star(surf.coarse_grid, surf.coarse_states, VctpChecker(surf),
                                 start_xy=[1.5, 1.5], goal_xy=[8.5, 8.5], rover_radius_m=0.40)
            self.assertEqual(res["status"], "PLANNED")
            self.assertEqual(res["vctp"]["pass_rate"], 1.0)

    def test_safety_metrics_and_dashboard_html_render(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "evaluation").mkdir(parents=True)
            lines = ["time_s,x,y,z,yaw_rad,speed_mps,dist_to_goal_m,est_x,est_y,loc_error_m,obstacle_contact,chassis_contact,total_collisions,path_length_m,stationary_s,roll_deg,pitch_deg"]
            for i in range(25):
                t = i * 0.1
                lines.append(f"{t:.2f},{i*0.04:.3f},0.0,0.0,0.0,0.40,1.0,{i*0.04:.3f},0.0,0.02,0,0,0,{i*0.04:.3f},0.0,1.0,2.0")
            (run_dir / "evaluation/trajectory.csv").write_text("\n".join(lines), encoding="utf-8")
            (run_dir / "evaluation/summary.json").write_text(json.dumps({"injected_obstacles": [{"xy_enu_m": [0.5, 1.2], "size_m": [0.4, 0.4, 0.4]}]}), encoding="utf-8")
            sm = compute_run_safety_smoothness(run_dir)
            self.assertGreater(sm["min_safety_margin_m"], 0.6)
            self.assertAlmostEqual(sm["j_avoid_integral"], 0.0, places=3)
            html = build_html(static_payload={"run_id": "test_run", "evaluation": {"status": "MISSION_SUCCESS"}}, runs_index=[])
            self.assertIn("援来有路", html)
            self.assertIn("test_run", html)

    def test_dynamic_stale_cell_clearing_and_weather_world_patch(self):
        coarse = Grid.from_bounds([[0.0, 0.0], [6.0, 6.0]], 0.5)
        states = np.full(coarse.shape, FREE, dtype=np.int8)
        omap = OnboardObstacleMap(coarse, states, np.zeros(coarse.shape), None, rover_radius_m=0.40)
        rover_xy = np.array([2.0, 3.0])
        dyn_pt = np.array([[3.2, 3.0]])
        for _ in range(3):
            omap.update(rover_xy, dyn_pt, np.empty((0, 2)))
        self.assertTrue(omap.fine_onboard_blocked.any())
        # Once the dynamic obstacle leaves close LiDAR range, cell clears after stale_clear_ticks.
        for _ in range(omap.stale_clear_ticks + 1):
            omap.update(rover_xy, np.empty((0, 2)), np.empty((0, 2)))
        self.assertFalse(omap.fine_onboard_blocked.any())

        with tempfile.TemporaryDirectory() as td:
            w = Path(td) / "test.wbt"
            w.write_text("Background { luminosity 1.05 }\nFog { color 0.6 0.6 0.6 visibilityRange 650 }\nDirectionalLight { ambientIntensity 0.30 intensity 2.1 }\nDirectionalLight { intensity 0.70 }\n", encoding="utf-8")
            apply_weather_to_world(w, "fog")
            txt = w.read_text(encoding="utf-8")
            self.assertIn("visibilityRange 38.0", txt)
            self.assertIn("luminosity 0.64", txt)

    def test_multi_posture_casualty_vrml_and_keypoints(self):
        flat = lambda x, y: 0.25
        for idx, posture in enumerate(ALL_POSTURES, start=1):
            pts = POSTURE_LOCAL_POINTS[posture]
            self.assertEqual(pts.ndim, 2)
            self.assertEqual(pts.shape[1], 3)
            self.assertTrue(np.isfinite(pts).all())
            vrml = build_posture_casualty_vrml(
                f"VICTIM_{idx}", f"victim_{idx}", [1.0, -2.0], 0.4, flat,
                posture=posture, shirt_def=f"SHIRT_APP_{idx}", pants_def=f"PANTS_APP_{idx}",
            )
            self.assertIn(f"DEF VICTIM_{idx} Solid", vrml)
            self.assertIn(f"DEF SHIRT_APP_{idx}", vrml)
            self.assertIn(f"DEF PANTS_APP_{idx}", vrml)
        # half_buried keypoints strictly cover exposed upper body only (x <= 0.0).
        self.assertLessEqual(float(POSTURE_LOCAL_POINTS["half_buried"][:, 0].max()), 0.0)

    def test_event_triggered_dynamic_uav_refinement_unlocks_blocked_corridor(self):
        # 12 m x 8 m grid: south corridor FREE (rows 2..6), north bypass UNKNOWN (rows 7..13).
        coarse = Grid.from_bounds([[0.0, 0.0], [12.0, 8.0]], 0.5)
        states = np.full(coarse.shape, UNKNOWN, dtype=np.int8)
        states[2:7, 2:22] = FREE  # 2.5 m wide confirmed south corridor
        states[7:14, 2:5] = FREE  # west staging free
        states[7:14, 19:22] = FREE  # east staging free
        gz = np.where(states == FREE, 0.0, np.nan)
        omap = OnboardObstacleMap(coarse, states, gz, None, rover_radius_m=0.40)

        # Stamp a full barrier across the south corridor around x = 6.0 m (coarse col 12).
        barrier_pts = np.array([[6.0, y] for y in np.linspace(1.1, 3.4, 24)])
        for _ in range(3):
            omap.update(np.array([4.5, 2.25]), barrier_pts, np.empty((0, 2)))

        route_init = np.array([[2.0, 2.25], [6.0, 2.25], [10.0, 2.25]])
        ctrl = RouteNavigationController(
            omap, route_init, np.array([10.0, 2.25]),
            dynamic_refinement_enabled=True, max_refinement_requests=2,
        )
        # Step once: conservative replan fails -> event-triggered UAV refinement request fires!
        v, w, info = ctrl.step(
            now=10.0, dt=0.1, pose_xyz=np.array([4.5, 2.25, 0.0]), rpy=(0.0, 0.0, 0.0),
            speed_mps=0.0, yaw_rate_rad_s=0.0,
            lidar_body_xy=np.empty((0, 2)), lidar_enu_xy=np.empty((0, 2)),
            rgbd_body_xy=np.empty((0, 2)), rgbd_enu_xy=np.empty((0, 2)),
        )
        self.assertEqual((v, w), (0.0, 0.0))
        self.assertEqual(ctrl.state, "WAITING_UAV_REFINEMENT")
        self.assertIsNotNone(ctrl.pending_refinement_request)
        patches = ctrl.pending_refinement_request["patches"]
        self.assertGreaterEqual(len(patches), 1)

        sched = schedule_dynamic_oblique(coarse, patches, None, [4.5, 2.25, 16.5], now=10.0)
        self.assertGreaterEqual(len(sched["stations"]), 6)

        # Simulate UAV confirming the north bypass corridor (rows 7..13, cols 5..19) as FREE.
        refined_states = states.copy()
        refined_states[7:14, 5:19] = FREE
        refined_gz = np.where(refined_states == FREE, 0.0, np.nan)
        unlocked = ctrl.apply_uav_refinement(
            now=28.0, pose_xy=np.array([4.5, 2.25]), local_obstacles_enu=np.empty((0, 2)),
            new_prior_states=refined_states, new_ground_z_m=refined_gz,
            response_meta=dict(frames_captured=12, flight_duration_s=18.0, patches=patches),
        )
        self.assertTrue(unlocked)
        self.assertEqual(ctrl.state, "TRACKING")
        # Rover's onboard barrier must remain strictly preserved after UAV merge!
        self.assertTrue(omap.fine_onboard_blocked.any())

    def test_triple_cargo_rack_proto_and_config(self):
        proto_txt = (ROOT / "protos/ResearchRover.proto").read_text(encoding="utf-8")
        for bay in ("cargo1Scale", "cargo2Scale", "cargo3Scale"):
            self.assertIn(f"field SFVec3f {bay}", proto_txt)
            self.assertIn(f"scale IS {bay}", proto_txt)
        common = json.loads((ROOT / "config/search/common.json").read_text(encoding="utf-8"))
        self.assertEqual(int(common["cargo_capacity"]), 3)


if __name__ == "__main__":
    unittest.main()
