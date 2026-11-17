"""Offline world construction. Coordinates remain outside the Robot input API.

A/B/C preserve the repository's terrain, props, casualty appearance and position.
smoke is an explicitly separate controlled fixture with three relocated casualties.
"""
from pathlib import Path
import argparse
import json
import math
import re
import numpy as np
from scene_elements import RescueElements
from upgrade_casualties import node_span
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"evaluation"))
from metrics import height_at,look_at_rotation

ROOT=Path(__file__).resolve().parents[1]
CASES={"A":"A", "B":"B", "C":"C", "smoke":"A", "D":"B"}


def replace_node(text, pattern, new):
    start,end=node_span(text,pattern)
    return text[:start]+new+text[end:]


def build(case):
    key=CASES[case]
    meta=json.loads((ROOT/"evaluation/scenarios"/f"{key}.json").read_text(encoding="utf-8"))
    scene=(ROOT/"worlds"/meta["world_file"]).read_text(encoding="utf-8")
    scene=replace_node(scene,r"\bDEF\s+ROVER\s+VisionRover", "")
    scene=replace_node(scene,r"\bDEF\s+EVALUATOR\s+Robot", "")
    scene=re.sub(r'EXTERNPROTO\s+"\.\./protos/VisionRover.proto"',
                 'EXTERNPROTO "../protos/ResearchRover.proto"',scene)
    if re.search(r"\brandomSeed\s+\d+",scene):
        scene=re.sub(r"\brandomSeed\s+\d+","randomSeed 71",scene,count=1)
    else:
        scene=re.sub(r"\bbasicTimeStep\s+10\b","basicTimeStep 10 randomSeed 71",scene,count=1)
    # A useful overview, independent of agent sensing and decision making.
    scene=replace_node(scene,r"\bDEF\s+SCENE_VIEW\s+Viewpoint",
        'DEF SCENE_VIEW Viewpoint { orientation -0.2060822 0.4268119 0.8805463 1.0031393 '
        'position -27 -34 23 fieldOfView .95 far 240 exposure 1.35 ambientOcclusionRadius .85 }')
    victims=[dict(id=1,xy=meta["goal"])]
    if case=="smoke":
        # Separate flat fixture: removes locomotion/terrain confounds during
        # pipeline verification. This is not an independent disaster test layout.
        meta=dict(meta,spawn=[0.,0.],spawn_yaw=0.,origin=[-10.,-10.],
                  spacing=20.,heights=[[0.,0.],[0.,0.]])
        orientation=" ".join(map(str,look_at_rotation([-8,-11,14],[3,1,0])))
        scene=f'''#VRML_SIM R2025a utf8
EXTERNPROTO "../protos/ResearchRover.proto"
WorldInfo {{ basicTimeStep 10 randomSeed 71 coordinateSystem "ENU" }}
DEF SCENE_VIEW Viewpoint {{ position -8 -11 14 orientation {orientation} }}
Background {{ skyColor [.65 .73 .8] }}
DirectionalLight {{ direction -.3 -.4 -1 intensity 1.8 ambientIntensity .5 }}
DEF TEST_GROUND Solid {{ name "test ground" translation 2 1 -.05 children [ Shape {{
 appearance PBRAppearance {{ baseColorMap ImageTexture {{ url ["textures/A_ground.png"] }} roughness 1 metalness 0 }}
 geometry Box {{ size 20 20 .1 }} }} ] boundingObject Box {{ size 20 20 .1 }} }}
DEF TEST_OBSTACLE Solid {{ name "test obstacle" translation 3 2 .25 children [ Shape {{
 appearance PBRAppearance {{ baseColor .4 .35 .3 roughness 1 metalness 0 }} geometry Box {{ size .7 .6 .5 }} }} ]
 boundingObject Box {{ size .7 .6 .5 }} }}
'''
        positions=[(3.,.5),(6.,-2.),(6.,3.5)]
        victims=[]
        # Generation may use the private height field. The runtime agent may not.
        for index,xy in enumerate(positions,1):
            elements=RescueElements(dict(key="B",seed=202+index,goal=xy),
                                   lambda x,y:height_at(meta,x,y))
            elements.casualty()
            node=elements.nodes[0].replace("CASUALTY",f"SEARCH_VICTIM_{index}")
            node=node.replace('name "casualty"',f'name "search victim {index}"')
            scene+="\n"+node
            victims.append(dict(id=index,xy=xy))
    complex_config=None
    if case=="D":
        from complex_disaster import add_complex_scene
        scene,meta,victims,complex_config=add_complex_scene(scene,meta,height_at)
    sx,sy=meta["spawn"];yaw=meta["spawn_yaw"]
    # Read spawn elevation from the already committed height field.
    task=json.dumps(json.dumps(dict(search_case=case)))
    scene+=f'''
DEF ROVER ResearchRover {{
 name "Search Rescue Rover"
 translation {sx} {sy} {height_at(meta,sx,sy)+.035:.6f}
 rotation 0 0 1 {yaw}
 customData {task}
}}
DEF SEARCH_REFEREE Robot {{
 name "Search mission referee" supervisor TRUE controller "mission_referee"
 customData "{case}" window "<none>"
}}
'''
    c,s=math.cos(yaw),math.sin(yaw)
    inverse=np.array([[c,s],[-s,c]])
    corners=((-5,-5),(10,-5),(10,8),(-5,8)) if case=="smoke" else ((-20.5,-14.5),(20.5,-14.5),(20.5,14.5),(-20.5,14.5))
    if case=='D':
        (xmin,ymin),(xmax,ymax)=meta['bounds_xy']
        corners=((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax))
    boundary=[(inverse @ (np.array(xy)-[sx,sy])).tolist() for xy in corners]
    public=dict(search_case=case,search_boundary_xy=boundary,depot_xy=[0,0],
                target_positions_provided=False,assumed_sensor_configuration=True,
                delivery_deadline_after_report_s=35 if case=="smoke" else 120)
    private=dict(base_scenario=key,search_case=case,victims=victims,spawn=meta["spawn"],
                 spawn_yaw=yaw,duration_s=150 if case=="smoke" else 600,
                 report_tolerance_m=1.0,delivery_tolerance_m=1.8,
                 delivery_deadline_after_report_s=public["delivery_deadline_after_report_s"])
    if complex_config is not None:
        # Calibration only: flight routes and person placements stay with the scene.
        public_aerial={key:complex_config['aerial'][key] for key in (
            'radio_channel','camera_fov_rad','gps_std_m','attitude_std_rad','image_period_s','flight_model')}
        public.update(aerial=public_aerial,aerial_enabled=True,
                      require_cross_source_confirmation=False,
                      mapping_side_m=2*math.ceil(np.max(np.abs(boundary))+2),
                      features=dict(moving_obstacles=2,added_trees=complex_config['vegetation']['trees'],
                                    total_trees=len(meta['forest']['trees']),sphere_shrubs=0,
                                    textured_bushes=len(meta['forest']['bushes']),extent_m=complex_config['extent_m'],
                                    excavated_gully=True,rough_terrain=True))
        # The rover never loads these private coordinates/height values.
        private.update(origin=meta['origin'],spacing=meta['spacing'],heights=meta['heights'],
                       forest=meta['forest'],bounds_xy=meta['bounds_xy'],
                       scenario_features='textured 3D forest and physical moving hazards')
    if case=="smoke":
        private.update(origin=meta["origin"],spacing=meta["spacing"],heights=meta["heights"],fixture="flat controlled pipeline test")
    for folder in ("config/search/public","evaluation/search"):
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    (ROOT/"config/search/public"/f"{case}.json").write_text(json.dumps(public,indent=2),encoding="utf-8")
    (ROOT/"evaluation/search"/f"{case}.json").write_text(json.dumps(private,indent=2),encoding="utf-8")
    world=ROOT/"worlds"/f"search_{case}.wbt"
    world.write_text(scene,encoding="utf-8")
    layout=ROOT/"config/view_layout.wbproj"
    layout_path=world.parent/f".{world.stem}.wbproj"
    if not layout_path.exists():
        layout_path.write_text(layout.read_text(encoding="utf-8"),encoding="utf-8")
    print(f"Built search_{case}: {len(victims)} people; true positions are evaluator-only.")
    return world


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--case",choices=[*CASES,"all"],default="all")
    args=parser.parse_args()
    for case in CASES if args.case=="all" else [args.case]:build(case)
