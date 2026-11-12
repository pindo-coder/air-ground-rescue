"""Check observability, terrain consistency, and geometric route connectivity."""
from collections import deque
import ast
import json
import math
from pathlib import Path
import re
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "tools"))
sys.path.insert(0,str(ROOT / "evaluation"))
from build_scenarios import path_distance, parameters, terrain, surface_texture
from metrics import height_at, target_in_view


def reachable(meta, reference_paths=None):
    spacing=meta["spacing"]
    height=np.asarray(meta["heights"])
    x,y=np.meshgrid(meta["origin"][0]+np.arange(height.shape[1])*spacing,meta["origin"][1]+np.arange(height.shape[0])*spacing)
    gy,gx=np.gradient(height,spacing)
    mask=(np.hypot(gx,gy)<math.tan(math.radians(8)))
    mask &= (x > -20.6) & (x < 20.6) & (y > -14.6) & (y < 14.6)
    if reference_paths is not None:
        mask &= path_distance(x,y,reference_paths)<.54
    for obstacle in [*meta["obstacles"], meta["target_body"]]:
        if obstacle.get("traversal")=="low_candidate":
            continue
        ox,oy=obstacle["xy"]
        c,s=math.cos(obstacle["yaw"]),math.sin(obstacle["yaw"])
        lx,ly=c*(x-ox)+s*(y-oy),-s*(x-ox)+c*(y-oy)
        mask&=~((np.abs(lx)<obstacle["size"][0]/2+.32)&(np.abs(ly)<obstacle["size"][1]/2+.32))
    def cell(point):
        return (round((point[1]-meta["origin"][1])/spacing),round((point[0]-meta["origin"][0])/spacing))
    start=cell(meta["spawn"]); goal=cell(meta["goal"])
    queue=deque([start]); seen={start}
    while queue:
        j,i=queue.popleft()
        if math.hypot(j-goal[0],i-goal[1])*spacing<1.35:
            return True
        for dj,di in ((1,0),(-1,0),(0,1),(0,-1)):
            nj,ni=j+dj,i+di
            if 0<=nj<mask.shape[0] and 0<=ni<mask.shape[1] and mask[nj,ni] and (nj,ni) not in seen:
                seen.add((nj,ni));queue.append((nj,ni))
    return False


class SceneContract(unittest.TestCase):
    def test_saved_worlds_contain_initial_public_tasks(self):
        for key in "ABC":
            meta=json.loads((ROOT/"evaluation"/"scenarios"/f"{key}.json").read_text(encoding="utf-8"))
            text=(ROOT/"worlds"/meta["world_file"]).read_text(encoding="utf-8")
            rover=text.split("DEF ROVER",1)[1].split("DEF EVALUATOR",1)[0]
            match=re.search(r'\bcustomData\s+("(?:\\.|[^"\\])*")',rover)
            self.assertIsNotNone(match,f"{key}: missing ROVER.customData")
            actual=json.loads(json.loads(match.group(1)))
            expected=json.loads((ROOT/"config"/"public"/f"{key}_task.json").read_text(encoding="utf-8"))
            self.assertEqual(actual,expected,f"{key}: restore startup parameters with python tools/restore_public_tasks.py")

    def test_camera_only_rover(self):
        proto=(ROOT/"protos"/"VisionRover.proto").read_text(encoding="utf-8")
        for name in ("Lidar {","GPS {","Compass {","InertialUnit {","PositionSensor {","supervisor TRUE"):
            self.assertNotIn(name,proto)
        self.assertEqual(proto.count("Camera {"),2)

    def test_policy_adapter_does_not_import_privileged_sources(self):
        for name in ("vision_agent.py","policy.py"):
            text=(ROOT/"controllers"/"vision_agent"/name).read_text(encoding="utf-8")
            tree=ast.parse(text)
            identifiers={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
            self.assertNotIn("Supervisor",identifiers)
            self.assertNotIn("getFromDef",{n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)})
            self.assertNotIn('evaluation/scenarios',text)

    def test_world_height_and_collision_match(self):
        for key in "ABC":
            meta=json.loads((ROOT/"evaluation"/"scenarios"/f"{key}.json").read_text())
            text=(ROOT/"worlds"/meta["world_file"]).read_text()
            # Webots Save may insert VRML commas and reformat whitespace.
            actual=np.fromstring(text.split("height [",1)[1].split("]",1)[0].replace(","," "),sep=" ")
            np.testing.assert_array_equal(actual,np.asarray(meta["heights"]).ravel())
            self.assertTrue(re.search(r"\bboundingObject\s+USE\s+GROUND_GRID\b",text),f"{key}: ground collision must reuse the visible height grid")
            self.assertNotIn("supervisor TRUE",text.split("DEF ROVER",1)[1].split("DEF EVALUATOR",1)[0])

    def test_initial_hint_contains_goal_but_has_no_coordinates(self):
        for key in "ABC":
            public=json.loads((ROOT/"config"/"public"/f"{key}_task.json").read_text())
            private=json.loads((ROOT/"evaluation"/"scenarios"/f"{key}.json").read_text())
            hint=public["goal_hint"]
            self.assertLessEqual(abs(hint["center_deg_left_positive"]-private["actual_initial_bearing_deg"]),hint["half_width_deg"])
            self.assertTrue(hint["issued_once"])
            for name in ("goal","roads","reference_paths","heights","spawn","current_bearing","distance_to_goal","elements","foliage_spheres","target_body","inspection_views"):
                self.assertNotIn(name,public)

    def test_each_world_has_an_offroad_route_with_clearance(self):
        for key in "ABC":
            meta=json.loads((ROOT/"evaluation"/"scenarios"/f"{key}.json").read_text())
            with self.subTest(scenario=key):
                self.assertTrue(reachable(meta))

    def test_washout_blocks_shortcut(self):
        meta=json.loads((ROOT/"evaluation"/"scenarios"/"B.json").read_text())
        self.assertFalse(reachable(meta,[meta["reference_paths"][0]]))

    def test_C_has_different_geometry(self):
        a=json.loads((ROOT/"evaluation"/"scenarios"/"A.json").read_text())
        c=json.loads((ROOT/"evaluation"/"scenarios"/"C.json").read_text())
        self.assertNotEqual(a["heights"],c["heights"])
        self.assertNotEqual(a["goal"],c["goal"])

    def test_surface_and_relief_do_not_trace_reference_paths(self):
        x,y=np.meshgrid(np.linspace(-21,21,80),np.linspace(-15,15,60))
        for key in "ABC":
            p=parameters(key,101)
            changed={**p,"reference_paths":[[(0,0),(20,-10)]]}
            np.testing.assert_array_equal(terrain(x,y,p),terrain(x,y,changed))
            np.testing.assert_array_equal(np.asarray(surface_texture(p,128)),np.asarray(surface_texture(changed,128)))

    def test_elevation_varies_across_unmarked_ground(self):
        for key in "ABC":
            meta=json.loads((ROOT/"evaluation"/"scenarios"/f"{key}.json").read_text())
            z=np.asarray(meta["heights"])
            self.assertEqual(meta["surface_mode"],"unmarked_offroad")
            self.assertGreater(float(np.ptp(z)),1.8)
            # More than a global tilted plane: residual hills/valleys are substantial.
            y,x=np.indices(z.shape)
            design=np.column_stack((np.ones(z.size),x.ravel(),y.ravel()))
            fit=design @ np.linalg.lstsq(design,z.ravel(),rcond=None)[0]
            self.assertGreater(float(np.std(z.ravel()-fit)),.25)

    def test_casualty_is_occluded_at_spawn(self):
        for key in "ABC":
            meta=json.loads((ROOT/"evaluation"/"scenarios"/f"{key}.json").read_text())
            public=json.loads((ROOT/"config"/"public"/f"{key}_task.json").read_text())
            yaw=meta["spawn_yaw"]
            c,s=math.cos(yaw),math.sin(yaw)
            position=[*meta["spawn"],height_at(meta,*meta["spawn"])]
            self.assertFalse(target_in_view(meta,position,[c,-s,0,s,c,0,0,0,1],public["camera"]))

    def test_height_interpolation_on_planar_surface(self):
        meta={"origin":[0,0],"spacing":1,"heights":[[0,1],[2,3]]}
        self.assertAlmostEqual(height_at(meta,.25,.5),1.25)

    def test_rescue_elements_and_collision_geometry(self):
        for key in "ABC":
            meta=json.loads((ROOT/"evaluation"/"scenarios"/f"{key}.json").read_text())
            text=(ROOT/"worlds"/meta["world_file"]).read_text()
            kinds={element["kind"] for element in meta["elements"]}
            self.assertTrue({"fallen_log","rubble","broken_wall","shrub","grass","casualty","low_root"}<=kinds)
            for element in meta["elements"]:
                for name in element["hard_parts"]:
                    reference=r"\bUSE\s+"+re.escape(f"GEO_{element['def']}_{name}")+r"\b"
                    self.assertTrue(re.search(reference,text),f"{key}: missing collision reference {element['def']}/{name}")
                for name in element["visual_only_parts"]:
                    reference=r"\bUSE\s+"+re.escape(f"GEO_{element['def']}_{name}")+r"\b"
                    self.assertFalse(re.search(reference,text),f"{key}: visual-only part used for collision {element['def']}/{name}")
            casualty=next(e for e in meta["elements"] if e["kind"]=="casualty")
            self.assertTrue({"HEAD","TORSO","PELVIS","HAND_L","HAND_R","THIGH_L","THIGH_R","BOOT_L","BOOT_R"}<=set(casualty["hard_parts"]))
            self.assertTrue(meta["foliage_spheres"])
            grades={o.get("traversal") for o in meta["obstacles"]}
            self.assertTrue({"low_candidate","detour","stem"}<=grades)

    def test_collapsed_houses_match_registered_positions(self):
        for key in "ABC":
            meta=json.loads((ROOT/"evaluation"/"scenarios"/f"{key}.json").read_text())
            world=(ROOT/"worlds"/meta["world_file"]).read_text()
            for element in meta["elements"]:
                if element["kind"] != "collapsed_house":
                    continue
                node=world.split(f"DEF {element['def']} Solid",1)[1].split("children",1)[0]
                translation=re.search(r"translation ([-.0-9]+) ([-.0-9]+) ([-.0-9]+)",node)
                self.assertIsNotNone(translation)
                np.testing.assert_allclose([float(translation[1]),float(translation[2])],element["xy"],atol=1e-4)

    def test_target_has_a_visible_close_approach(self):
        for key in "ABC":
            meta=json.loads((ROOT/"evaluation"/"scenarios"/f"{key}.json").read_text())
            public=json.loads((ROOT/"config"/"public"/f"{key}_task.json").read_text())
            gx,gy=meta["goal"]
            visible=False
            for angle in np.linspace(-math.pi,math.pi,48,endpoint=False):
                x,y=gx+1.3*math.cos(angle),gy+1.3*math.sin(angle)
                yaw=angle+math.pi;c,s=math.cos(yaw),math.sin(yaw)
                visible |= target_in_view(meta,[x,y,height_at(meta,x,y)],[c,-s,0,s,c,0,0,0,1],public["camera"])
            self.assertTrue(visible,key)


if __name__=="__main__":
    unittest.main()
