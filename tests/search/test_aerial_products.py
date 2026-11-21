"""Checks for inverse projection, depth visibility and raster registration."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from aerial_products import orthorectify, world_to_image, write_raster
from multiview import aerial_camera_rotation, aerial_pixels_enu
from survey_map import Grid


class AerialProductTests(unittest.TestCase):
    def test_inverse_projection_preserves_pixels_with_oblique_camera_attitude(self):
        pixels = np.array([[319.5, 239.5, 10.0], [250, 320, 13.1], [580, 37, 8.5]])
        origin, attitude = [3, -6, 17], [0.10, -0.61, 1.12]
        points = aerial_pixels_enu(pixels, 640, 480, 1.1, origin, attitude)
        optical = aerial_camera_rotation(attitude)
        result = world_to_image(points, origin, optical, 640, 480, 1.1)
        np.testing.assert_allclose(np.column_stack(result), pixels, atol=1e-10)

    def test_orthomosaic_rejects_occlusion_before_preferring_nadir(self):
        grid = Grid.from_bounds([[-0.5, -0.5], [0.5, 0.5]], 1.0)
        dsm = np.zeros(grid.shape)
        observations = {
            "size_wh": [9, 9],
            "fov_rad": 1.1,
            "stride_px": 1,
            "range_m": [0, 6],
            "gps_enu_m": np.array([[0, 0, 2], [0.5, 0, 2], [0, 0, 2.5]]),
            "attitude_rpy_rad": np.zeros((3, 3)),
            "depth_m": np.broadcast_to(np.array([1.0, 2.0, 2.5])[:, None, None], (3, 9, 9)),
            "rgb": np.broadcast_to(np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])[:, None, None],
                                   (3, 9, 9, 3)).astype(np.uint8),
        }
        footprint = (0, 1, 0, 1)
        with tempfile.TemporaryDirectory() as folder:
            color, source, _, rejected = orthorectify(
                grid, dsm, observations, [footprint, footprint, None], Path(folder))
            np.testing.assert_array_equal(color[0, 0], [0, 255, 0, 255])
            self.assertEqual(source.item(), 1)
            self.assertEqual(rejected, 1)
            color, source, _, _ = orthorectify(
                grid, dsm, observations, [footprint] * 3, Path(folder))
            np.testing.assert_array_equal(color[0, 0], [0, 0, 255, 255])
            self.assertEqual(source.item(), 2)

    def test_raster_pixel_order_matches_enu_tags_and_world_file(self):
        grid = Grid.from_bounds([[-2, -1], [0, 1]], 0.5)
        heights = np.arange(16, dtype=np.float32).reshape(grid.shape)
        heights[0, 0] = -9999
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / "elevation.tif"
            write_raster(filename, grid, heights, -9999)
            with Image.open(filename) as image:
                self.assertEqual(image.tag_v2[33550], (0.5, 0.5, 0.0))
                self.assertEqual(image.tag_v2[33922], (0, 0, 0, -2, 1, 0))
                self.assertEqual(image.tag_v2[42113].rstrip("\0"), "-9999")
                np.testing.assert_array_equal(np.asarray(image), heights[::-1])
            coefficients = list(map(float, filename.with_suffix(".tfw").read_text().splitlines()))
            self.assertEqual(coefficients, [0.5, 0, 0, -0.5, -1.75, 0.75])


if __name__ == "__main__":
    unittest.main()
