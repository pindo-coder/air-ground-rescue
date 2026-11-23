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
from rescue_dashboard import build_html
from route_tracker import OnboardObstacleMap
from run_uav_rescue import apply_weather_to_world
from safety_metrics import compute_run_safety_smoothness
from survey_map import FREE, Grid
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


if __name__ == "__main__":
    unittest.main()
