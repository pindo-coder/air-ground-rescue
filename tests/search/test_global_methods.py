"""Unit tests for Traversing Mars dual-route, DUCPP refinement, VCTP, and IGHA*."""
from pathlib import Path
import math
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for name in ("uav_mapper", "uav_map_rover"):
    sys.path.insert(0, str(ROOT / "controllers" / name))

from grid_route import plan_dual
from igha_star import plan_igha_star
from survey_map import BLOCKED, FREE, UNKNOWN, Grid
from survey_refinement import select_patches_density, select_patches_route_value
from vctp import FineTerrainSurface, VctpChecker


class GlobalPlanningMethodsTests(unittest.TestCase):
    def test_traversing_mars_detects_shortcut_and_reachability_segments(self):
        grid = Grid.from_bounds([[0.0, 0.0], [16.0, 12.0]], 0.5)
        states = np.full(grid.shape, FREE, dtype=np.int8)
        # Put an UNKNOWN strip cutting across the middle (rows 0..16), forcing conservative detour far north.
        states[:16, 14:18] = UNKNOWN
        start = grid.center((6, 4))
        goal = grid.center((6, 26))

        dual = plan_dual(grid, states, start, goal, radius=0.40)
        self.assertEqual(dual["conservative_status"], "PLANNED")
        self.assertEqual(dual["optimistic_status"], "PLANNED")
        self.assertFalse(dual["changes_reachability"])
        self.assertGreater(dual["potential_saving_m"], 4.0)
        self.assertGreater(dual["corridor_unknown_cell_count"], 0)
        self.assertGreaterEqual(len(dual["unconfirmed_segments"]), 1)

        # Now surround goal with UNKNOWN so conservative fails while optimistic succeeds.
        states[:, 14:18] = UNKNOWN
        dual_reach = plan_dual(grid, states, start, goal, radius=0.40)
        self.assertEqual(dual_reach["conservative_status"], "NO_ROUTE")
        self.assertEqual(dual_reach["optimistic_status"], "PLANNED")
        self.assertTrue(dual_reach["changes_reachability"])

    def test_ducpp_selects_route_corridor_over_unrelated_dense_unknown_blob(self):
        grid = Grid.from_bounds([[0.0, 0.0], [20.0, 20.0]], 0.5)
        states = np.full(grid.shape, FREE, dtype=np.int8)
        # Goal 1 straight east has a huge irrelevant unknown blob to its south-east,
        # while Goal 2 straight north is blocked from conservative reachability by a small unknown band.
        states[4:14, 16:28] = UNKNOWN  # large irrelevant dense blob
        states[18:22, :] = UNKNOWN  # thin unknown band blocking north goal reachability
        layers = dict(
            states=states,
            ground_z_m=np.where(states == FREE, 1.0, np.nan),
            observed_views=np.full(grid.shape, 4, dtype=np.uint16),
            planning_cost=np.where(states == FREE, 1.0, np.nan),
        )
        settings = dict(
            deployment_xy_enu_m=[5.25, 5.25],
            rover_radius_m=0.40,
            max_slope_deg=18,
            route_goals=[
                {"id": "east", "label": "east", "xy_enu_m": [14.75, 2.25]},
                {"id": "north_blocked", "label": "north", "xy_enu_m": [5.25, 16.25]},
            ],
            refinement=dict(strategy="route_value", patch_count=1, patch_radius_m=2.5),
        )
        rv_patches, duals = select_patches_route_value(grid, layers, settings)
        self.assertEqual(len(rv_patches), 1)
        # DUCPP must target the unknown band blocking north_blocked (near x=5.25, y=10.0), not the east blob.
        px, py = rv_patches[0]["xy_m"]
        self.assertLess(abs(px - 5.25), 3.0)
        self.assertLess(abs(py - 10.0), 3.0)
        self.assertIn("north_blocked", rv_patches[0]["target_goals"])
        self.assertTrue(any(d["changes_reachability"] for d in duals))

        dens_patches, _ = select_patches_density(grid, layers, settings)
        self.assertEqual(len(dens_patches), 1)

    def test_vctp_checks_pitch_roll_belly_clearance_twist_and_unknown_evidence(self):
        coarse = Grid.from_bounds([[0.0, 0.0], [6.0, 6.0]], 0.5)
        fine = Grid.from_bounds([[0.0, 0.0], [6.0, 6.0]], 0.125)
        states = np.full(coarse.shape, FREE, dtype=np.int8)
        rows, cols = np.indices(fine.shape)
        x = fine.origin[0] + (cols + 0.5) * fine.resolution
        y = fine.origin[1] + (rows + 0.5) * fine.resolution

        # 1. Gentle 8 deg pitch slope -> PASS
        gentle_z = x * math.tan(math.radians(8.0))
        vctp = VctpChecker(FineTerrainSurface(fine, gentle_z, coarse, states))
        r_ok = vctp.check_pose(3.0, 3.0, 0.0)
        self.assertEqual(r_ok.status, "PASS")
        self.assertAlmostEqual(abs(r_ok.pitch_deg), 8.0, delta=0.5)

        # 2. Steep 22 deg pitch slope -> FAIL(EXCESSIVE_PITCH)
        steep_z = x * math.tan(math.radians(22.0))
        r_pitch = VctpChecker(FineTerrainSurface(fine, steep_z, coarse, states)).check_pose(3.0, 3.0, 0.0)
        self.assertEqual(r_pitch.status, "FAIL")
        self.assertIn("EXCESSIVE_PITCH", r_pitch.reason)

        # 3. Steep 19 deg lateral roll -> FAIL(EXCESSIVE_ROLL)
        roll_z = y * math.tan(math.radians(19.0))
        r_roll = VctpChecker(FineTerrainSurface(fine, roll_z, coarse, states)).check_pose(3.0, 3.0, 0.0)
        self.assertEqual(r_roll.status, "FAIL")
        self.assertIn("EXCESSIVE_ROLL", r_roll.reason)

        # 4. Small 0.14 m rock directly under chassis belly -> FAIL(BELLY_CLEARANCE)
        rock_z = np.zeros(fine.shape)
        rc = fine.cell((3.0, 3.0))
        rock_z[rc[0] - 1:rc[0] + 1, rc[1] - 1:rc[1] + 1] = 0.14
        r_belly = VctpChecker(FineTerrainSurface(fine, rock_z, coarse, states)).check_pose(3.0, 3.0, 0.0)
        self.assertEqual(r_belly.status, "FAIL")
        self.assertIn("BELLY_CLEARANCE", r_belly.reason)

        # 5. Missing ground evidence under one wheel -> UNKNOWN
        unk_states = states.copy()
        unk_states[coarse.cell((3.2, 3.2))] = UNKNOWN
        r_unk = VctpChecker(FineTerrainSurface(fine, np.zeros(fine.shape), coarse, unk_states)).check_pose(3.0, 3.0, 0.0)
        self.assertEqual(r_unk.status, "UNKNOWN")
        self.assertEqual(r_unk.reason, "INSUFFICIENT_GROUND_EVIDENCE")

    def test_igha_star_avoids_vctp_rock_and_outputs_heading_aware_waypoints(self):
        coarse = Grid.from_bounds([[0.0, 0.0], [10.0, 8.0]], 0.5)
        fine = Grid.from_bounds([[0.0, 0.0], [10.0, 8.0]], 0.125)
        states = np.full(coarse.shape, FREE, dtype=np.int8)
        fine_z = np.zeros(fine.shape, dtype=float)
        # Put a 0.18 m tall rock patch right on the straight line between start and goal.
        rr, cc = np.indices(fine.shape)
        x = fine.origin[0] + (cc + 0.5) * fine.resolution
        y = fine.origin[1] + (rr + 0.5) * fine.resolution
        fine_z[np.hypot(x - 5.0, y - 4.0) <= 0.40] = 0.18

        vctp = VctpChecker(FineTerrainSurface(fine, fine_z, coarse, states))
        # Start facing West (pi) when goal is East -> must execute skid-steer turn + bypass rock.
        res = plan_igha_star(
            coarse, states, vctp,
            start_xy=[1.75, 4.0], goal_xy=[8.25, 4.0],
            rover_radius_m=0.40, start_yaw=math.pi,
        )
        self.assertEqual(res["status"], "PLANNED")
        self.assertEqual(res["vctp"]["pass_rate"], 1.0)
        self.assertEqual(res["vctp"]["fail_count"], 0)
        self.assertEqual(res["vctp"]["unknown_count"], 0)
        self.assertGreater(res["stats"]["turn_in_place_count"], 0)
        # Verify route bypassed the (5.0, 4.0) rock cleanly.
        pts = np.asarray(res["route_xy_m"])
        self.assertGreater(float(np.min(np.linalg.norm(pts - [5.0, 4.0], axis=1))), 0.35)


if __name__ == "__main__":
    unittest.main()
