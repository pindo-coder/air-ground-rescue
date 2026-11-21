"""Regression tests for the aerial prior's unknown cells and handoff to A*."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for name in ("uav_mapper", "uav_map_rover", "search_agent"):
    sys.path.insert(0, str(ROOT / "controllers" / name))

from grid_route import direct_line_cells, plan
from multiview import aerial_pixels_enu
from survey_map import BLOCKED, FREE, UNKNOWN, Grid, classify, load_prior, save_prior
from survey_refinement import refine_ground


class AerialPriorTests(unittest.TestCase):
    def test_observed_canopy_and_unobserved_terrain_are_unknown(self):
        grid = Grid.from_bounds([[0, 0], [10, 10]], 0.5)
        surface = np.zeros(grid.shape)
        surface[5:8, 7:10] = 3.5
        surface[12:14, 11:13] = 0.50
        surface[4:8, 2] = -0.50
        surface[:2, :3] = np.nan
        second_view = surface + 0.02
        second_view[2, 5] = np.nan
        result = classify(grid, [surface, second_view], [surface + 0.08, second_view + 0.08],
                          seed_xy=[1.75, 1.75])

        states = result["states"]
        self.assertTrue(np.all(states[5:8, 7:10] == UNKNOWN))
        self.assertTrue(np.all(states[:2, :3] == UNKNOWN))
        self.assertEqual(states[2, 5], UNKNOWN)
        self.assertTrue(np.all(states[12:14, 11:13] == BLOCKED))
        self.assertTrue(np.all(states[4:8, 2] == BLOCKED))
        self.assertEqual(states[1, 8], FREE)
        self.assertTrue(np.isnan(result["ground_z_m"][6, 8]))

        with tempfile.TemporaryDirectory() as temp:
            filename = Path(temp) / "uav_map.npz"
            save_prior(filename, grid, result, 2.0)
            loaded_grid, loaded_states = load_prior(filename)
        self.assertEqual(loaded_grid, grid)
        np.testing.assert_array_equal(loaded_states, states)

    def test_route_goes_around_hazard_with_clearance_and_stops_at_unknown_barrier(self):
        grid = Grid.from_bounds([[-4, -4], [5, 5]], 0.5)
        states = np.full(grid.shape, FREE, dtype=np.int8)
        states[:12, 8:10] = UNKNOWN
        states[6:12, 8:10] = BLOCKED
        start, goal = grid.center((8, 3)), grid.center((8, 14))
        path = plan(grid, states, start, goal, radius=0.40)
        self.assertGreater(direct_line_cells(grid, states, start, goal)["blocked"], 0)
        self.assertGreater(np.max(path[:, 1]), grid.center((12, 9))[1])
        np.testing.assert_array_equal(path[0], start)
        np.testing.assert_array_equal(path[-1], goal)

        states[:, 8:10] = UNKNOWN
        with self.assertRaisesRegex(ValueError, "No route"):
            plan(grid, states, start, goal, radius=0.40)

    def test_batch_projection_retains_axis_depth_and_heading(self):
        points = aerial_pixels_enu([[319.5, 239.5, 8], [359.5, 239.5, 6]],
                                   640, 480, 1.1, [10, 20, 9], [0, 0, np.pi / 2])
        expected = [[10, 20, 1], [10 + 40 * 6 * (2 * np.tan(0.55) / 640), 20, 3]]
        np.testing.assert_allclose(points, expected, atol=1e-9)

    def test_steep_observed_ground_preserves_elevation_but_blocks_route(self):
        grid = Grid.from_bounds([[0, 0], [4, 4]], 0.5)
        plane = np.broadcast_to(np.arange(8) * 0.5 * np.tan(np.radians(22)), grid.shape).copy()
        result = classify(grid, [plane, plane + 0.02], [plane + 0.08, plane + 0.10], [0.75, 0.75],
                          max_slope_deg=18)
        self.assertTrue(np.isfinite(result["ground_z_m"]).all())
        np.testing.assert_allclose(result["slope_deg"], 22, atol=1e-6)
        self.assertTrue(np.all(result["states"] == BLOCKED))
        self.assertTrue(np.all(result["hazard_type"] == 1))

    def test_terrain_cost_can_select_a_flatter_detour_on_the_same_grid(self):
        grid = Grid.from_bounds([[0, 0], [9, 9]], 0.5)
        states = np.full(grid.shape, FREE, dtype=np.int8)
        cost = np.ones(grid.shape)
        cost[7:10, 5:13] = 8
        start, goal = grid.center((8, 3)), grid.center((8, 14))
        baseline = plan(grid, states, start, goal, radius=0.40)
        weighted = plan(grid, states, start, goal, radius=0.40, planning_cost=cost)
        self.assertTrue(any(cost[grid.cell(point)] > 1 for point in baseline))
        self.assertTrue(all(cost[grid.cell(point)] == 1 for point in weighted))

    def test_oblique_views_recover_floor_but_keep_low_obstacles_and_old_hazards(self):
        grid = Grid.from_bounds([[0, 0], [7, 7]], 0.5)
        clear, obstructed, incomplete, hazard = (5, 5), (5, 7), (7, 5), (7, 7)
        surface = np.zeros(grid.shape)
        for cell in (clear, obstructed, incomplete):
            surface[cell] = np.nan
        surface[hazard] = 0.5
        baseline = classify(grid, [surface, surface], [surface + .02, surface + .02],
                            [1.25, 1.25], max_slope_deg=18)
        survey = {"patches": [{"xy_m": [3.25, 3.25]}], "radius_m": 2.0}
        points = []
        # At least two depth points in every observed quarter.
        for cell in (clear, obstructed, incomplete, hazard):
            x, y = grid.center(cell)
            for east in (-.17, -.09, .09, .17):
                for north in (-.17, -.09, .09, .17):
                    if cell == incomplete and east > 0 and north > 0:
                        continue
                    points.append([x + east, y + north, .01])
            if cell == obstructed:
                points.extend([[x + offset, y, .4] for offset in (-.16, 0, .16)])
        views = [np.asarray(points)] * 3
        positions = [[2, 0, 12], [5, 0, 12], [2, 3, 12]]
        result = refine_ground(grid, baseline, views, positions, 18, survey)
        self.assertEqual(result["states"][clear], FREE)
        self.assertEqual(result["states"][obstructed], UNKNOWN)
        self.assertEqual(result["states"][incomplete], UNKNOWN)
        self.assertEqual(result["oblique_ground_quarters"][incomplete], 3)
        self.assertEqual(result["states"][hazard], BLOCKED)
        self.assertEqual(int(result["refined_ground"].sum()), 1)
        repeated = refine_ground(grid, baseline, views, [positions[0]] * 3, 18, survey)
        self.assertEqual(repeated["states"][clear], UNKNOWN)


if __name__ == "__main__":
    unittest.main()
