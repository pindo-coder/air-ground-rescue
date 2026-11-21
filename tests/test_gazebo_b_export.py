"""Check B's portable SDF bundle without a simulator installation."""

import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from export_gazebo_b import SOURCE, export


class GazeboBExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.bundle = Path(temporary.name)
        cls.world = ET.parse(export(cls.bundle)).find("world")
        cls.models = {model.get("name"): model for model in cls.world.findall("model")}

    def test_entire_static_scene_and_original_heights(self):
        source = SOURCE.read_text(encoding="utf-8")
        expected = {name.lower() for name in re.findall(r"^DEF (\w+) (?:Solid|Shape)\s*\{", source, re.M)}
        self.assertEqual(self.models.keys(), expected)
        for model in self.models.values():
            self.assertEqual(model.findtext("static"), "true")
            self.assertTrue(model.findall("./link/visual"), model.get("name"))
        data = json.loads((ROOT / "evaluation/scenarios/B.json").read_text(encoding="utf-8"))
        step = data["spacing"]
        with (self.bundle / "terrain.obj").open(encoding="ascii") as mesh:
            vertices = (tuple(map(float, line.split()[1:])) for line in mesh if line.startswith("v "))
            for y, row in enumerate(data["heights"]):
                for x, height in enumerate(row):
                    self.assertEqual(next(vertices), (x * step, y * step, height))
            self.assertIsNone(next(vertices, None))
        ground = self.models["terrain"]
        self.assertEqual(ground.findtext("./link/visual/geometry/mesh/uri"), "terrain.obj")
        self.assertEqual(ground.findtext("./link/collision/geometry/mesh/uri"), "terrain.obj")

    def test_vegetation_and_casualty_collision_proxies(self):
        tree_meshes = set()
        for name, model in self.models.items():
            if name.startswith("tree_"):
                tree_meshes.add(tuple(v.text for v in model.findall("./link/visual/geometry/mesh/uri")))
                collision, = model.findall("./link/collision")
                self.assertIsNotNone(collision.find("./geometry/cylinder"), name)
            elif "shrub" in name:
                self.assertGreater(len(model.findall("./link/visual")), 1, name)
                stem, = model.findall("./link/collision")
                self.assertIsNotNone(stem.find("./geometry/cylinder"), name)
            elif name.startswith("grass_") or name in {"residual_stream", "distant_mountains"}:
                self.assertFalse(model.findall("./link/collision"), name)
        self.assertEqual(len(tree_meshes), 1)
        self.assertEqual(len(next(iter(tree_meshes))), 2)

        casualty = self.models["casualty"]
        meshes = [Path(v.text).name for v in casualty.findall("./link/visual/geometry/mesh/uri")]
        self.assertCountEqual(meshes, ["casualty_B_pants.obj", "casualty_B_shirt.obj", "casualty_B_shoes.obj",
                                       "casualty_B_skin.obj", "head_scan.obj"])
        self.assertEqual(len(casualty.findall("./link/visual")), 5)
        torso = casualty.find("./link/collision[@name='collision_1_geo_casualty_torso']")
        self.assertEqual(torso.findtext("./geometry/box/size"), "0.48 0.33 0.21")
        self.assertGreater(len(casualty.findall("./link/collision")), 5)

    def test_bundle_contains_materials_and_meshes(self):
        for model in self.models.values():
            for link in model.findall("link"):
                names = [entry.get("name") for entry in link]
                self.assertEqual(len(names), len(set(names)), model.get("name"))
        kinds = {"uri", "albedo_map", "normal_map", "roughness_map"}
        referenced = {item.text for item in self.world.iter() if item.tag in kinds}
        self.assertIn("resources/worlds/textures/B_ground.png", referenced)
        self.assertIn("resources/protos/textures/pine_leaves.png", referenced)
        for uri in referenced:
            path = self.bundle / uri
            self.assertTrue(path.is_file(), uri)
            if uri.startswith("resources/"):
                original = ROOT / uri.removeprefix("resources/")
                self.assertEqual(path.read_bytes(), original.read_bytes(), uri)


if __name__ == "__main__":
    unittest.main()
