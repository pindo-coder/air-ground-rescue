"""Regression checks for the D scene's ground collider and onboard cameras."""

from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from export_gazebo_d import add_gui, terrain_collision_tiles
from gazebo_d_robots import add_drone, add_rover
from webots_static_scene import Node, StaticScene


class GazeboDExportTest(unittest.TestCase):
    def test_gui_opens_both_rover_camera_topics(self):
        world = ET.Element("world")
        viewpoint = Node("Viewpoint", {"position": (-58, -69, 48)}, ROOT / "worlds/search_D.wbt")
        add_gui(world, viewpoint)
        panels = world.findall("./gui/plugin[@filename='ImageDisplay']")
        self.assertEqual({panel.get("name"): panel.findtext("topic") for panel in panels}, {
            "Rover RGB": "/rover/rgb",
            "Rover Depth": "/rover/depth",
            "Drone RGB": "/drone/rgb",
        })

    def test_terrain_faces_reference_original_vertices_and_normals(self):
        with tempfile.TemporaryDirectory() as directory:
            grid = 2, 2, .125, .125, [0, .2, .4, .1]
            tile, = terrain_collision_tiles(Path(directory), grid)
            lines = tile.read_text(encoding="ascii").splitlines()
        self.assertEqual([line for line in lines if line.startswith("v ")],
                         ["v 0 0 0", "v 0.125 0 0.2", "v 0 0.125 0.4", "v 0.125 0.125 0.1"])
        self.assertEqual(sum(line.startswith("vn ") for line in lines), 4)
        self.assertEqual([line for line in lines if line.startswith("f ")],
                         ["f 1//1 2//2 4//4", "f 1//1 4//4 3//3"])

    def test_cameras_are_mounted_on_the_corresponding_robot_links(self):
        world = ET.Element("world")
        source = ROOT / "worlds/search_D.wbt"
        with tempfile.TemporaryDirectory() as directory:
            scene = StaticScene(ROOT, Path(directory))
            add_rover(world, Node("ResearchRover", {}, source), scene, {"rubber": .95, "caster": .015})
            add_drone(world, Node("ObserverDrone", {}, source), scene, image_period=1.0)

        for model, link, width, height, mount in (
            ("rover", "chassis", "320", "240", [0.26, 0, 0.30]),
            ("drone", "body", "640", "480", [0, 0, -0.20]),
        ):
            parent = world.find(f"./model[@name='{model}']/link[@name='{link}']")
            for name, kind in (("rgb", "camera"), ("depth", "depth_camera")):
                camera = parent.find(f"./sensor[@name='{name}']")
                self.assertEqual(camera.get("type"), kind)
                self.assertEqual(camera.findtext("topic"), f"/{model}/{name}")
                self.assertEqual(camera.findtext("camera/image/width"), width)
                self.assertEqual(camera.findtext("camera/image/height"), height)
                self.assertEqual(list(map(float, camera.findtext("pose").split()[:3])), mount)


if __name__ == "__main__":
    unittest.main()
