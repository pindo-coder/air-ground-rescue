"""Human visuals must load offline without changing physics or task inputs."""
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from casualty_visual import casualty_render, human_catalog
from scene_elements import RescueElements
from upgrade_casualties import node_span, upgrade_world_text


class CasualtyVisual(unittest.TestCase):
    def test_scan_neck_overlaps_body_collar_without_a_second_neck(self):
        def vertices(name):
            return [[float(v) for v in line.split()[1:]]
                    for line in (ROOT / "assets/render" / name).read_text(encoding="utf-8").splitlines()
                    if line.startswith("v ")]
        head = vertices("head_scan.obj")
        cut_x = max(v[0] for v in head)
        cut = [v for v in head if v[0] > cut_x - .01]
        for key in "ABC":
            collar = [v for v in vertices(f"casualty_{key}_shirt.obj") if v[0] < -.4 and abs(v[1]) < .09]
            # The raw scan edge ends inside the collar. The overlapping old
            # skin surface is removed, avoiding both a gap and depth flicker.
            self.assertGreaterEqual(cut_x - min(v[0] for v in collar), .02)
            self.assertLessEqual(cut_x - min(v[0] for v in collar), .065)
            self.assertLess(abs(max(v[2] for v in cut) - max(v[2] for v in collar)), .025)
            old_neck = [v for v in vertices(f"casualty_{key}_skin.obj") if v[0] < -.45 and abs(v[1]) < .09]
            self.assertFalse(old_neck)
            self.assertNotIn("Cylinder", casualty_render(key))

    def test_world_visual_assets_are_local_and_fit_the_target(self):
        for key, entry in human_catalog().items():
            vertices = []
            for item in entry["parts"]:
                mesh = ROOT / "assets/render" / item["mesh"]
                self.assertTrue(mesh.is_file())
                for field in ("albedo", "normal"):
                    if item.get(field):
                        self.assertTrue((ROOT / "assets" / item[field]).is_file())
                for line in mesh.read_text(encoding="utf-8").splitlines():
                    if line.startswith("v "):
                        vertices.append([float(value) for value in line.split()[1:]])
            lo = [min(v[i] for v in vertices) for i in range(3)]
            hi = [max(v[i] for v in vertices) for i in range(3)]
            self.assertAlmostEqual(hi[0] - lo[0], 1.67, places=2)
            self.assertGreaterEqual(lo[2], 0)
            self.assertLessEqual(hi[2], .29)
            self.assertLessEqual(max(abs(lo[1]), abs(hi[1])), .425)

            meta = json.loads((ROOT / "evaluation/scenarios" / f"{key}.json").read_text(encoding="utf-8"))
            world = (ROOT / "worlds" / meta["world_file"]).read_text(encoding="utf-8")
            start, end = node_span(world, r"\bDEF\s+CASUALTY\s+Solid")
            node = world[start:end]
            self.assertEqual(node.count("DEF CASUALTY_RENDER Group"), 1)
            self.assertEqual(len(re.findall(r"\bgeometry\s+Mesh\s*\{", node)), 5)
            collision = node.split("boundingObject", 1)[1]
            self.assertNotIn("Mesh {", collision)
            self.assertEqual(upgrade_world_text(world, key), world)

    def test_generator_keeps_collision_proxies_and_adds_textured_visuals(self):
        for key in "ABC":
            props = RescueElements({"key": key, "seed": 101, "goal": [2, 3]}, lambda x, y: 0)
            target = props.casualty()
            node = props.nodes[0]
            self.assertIn(casualty_render(key), node)
            self.assertIn("transparency 1", node)
            self.assertIn("USE GEO_CASUALTY_HEAD", node)
            self.assertIn("USE GEO_CASUALTY_TORSO", node)
            self.assertEqual(props.manifest[0]["visual_model"], "textured_mesh_v2")
            self.assertEqual(target["xy"], [2, 3])
            self.assertFalse(props.obstacles)


if __name__ == "__main__":
    unittest.main()
