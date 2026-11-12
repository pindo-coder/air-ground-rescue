"""Build camera-only outdoor tasks. Geometry data goes only to evaluation/.

NumPy vectorizes height/texture generation. No navigation policy is generated.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
from scene_elements import RescueElements
import sys
from render_style import appearance, flood_ground, skyline, residual_water, sky as hdr_sky

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from metrics import look_at_rotation
CFG = json.loads((ROOT / "config" / "suite.json").read_text(encoding="utf-8"))


def smooth(x):
    t = np.clip(x, 0, 1)
    return t * t * (3 - 2 * t)


def smooth_path(points, passes=2):
    points = np.asarray(points, dtype=float)
    for _ in range(passes):
        pairs = []
        for a, b in zip(points, points[1:]):
            pairs.extend((.75 * a + .25 * b, .25 * a + .75 * b))
        points = np.vstack((points[0], pairs, points[-1]))
    return points.tolist()


def path_distance(x, y, roads):
    result = np.full(np.broadcast(x, y).shape, np.inf)
    for road in roads:
        for a, b in zip(road, road[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            t = np.clip(((x - a[0]) * dx + (y - a[1]) * dy) / (dx * dx + dy * dy), 0, 1)
            result = np.minimum(result, np.hypot(x - a[0] - t * dx, y - a[1] - t * dy))
    return result


def parameters(key, seed):
    rng = random.Random(seed)
    if key == "A":
        roads = [[(-18, -11), (-14, -10), (-10, -7), (-7, -4), (-8, 0), (-4, 2), (1, 1), (5, 3), (8, 7), (13, 8), (17, 11)]]
        goal = (16.9, 10.8)
        cut = None
    elif key == "B":
        roads = [
            [(-18, -11), (-14, -10), (-10, -8), (-6, -5.8), (-3, -5), (1, -2), (6, 1), (10, 5), (14, 8), (17, 11)],
            [(-10, -8), (-13, -5), (-13, -1), (-10, 3), (-6, 5), (-1, 4), (3, 5), (6, 7), (10, 5)],
        ]
        goal = (16.9, 10.8)
        cut = (-3, -5)
    else:
        # New bend order and branch topology, not merely a recolored training road.
        roads = [
            [(-18, -10), (-15, -6), (-11, -5), (-8, -8), (-3, -6), (0, -2), (-2, 3), (1, 7), (6, 6), (10, 9), (16, 9)],
            [(-11, -5), (-10, -1), (-7, 1)],
        ]
        goal = (15.9, 9.0)
        cut = None
    # Keep spawn and shared junctions exact, while varying interior geometry coherently.
    jitter = {(float(x), float(y)): (rng.uniform(-.16, .16), rng.uniform(-.16, .16)) for road in roads for x, y in road}
    edited = []
    for road in roads:
        edited.append([(x + (jitter[x, y][0] if i not in (0, len(road) - 1) else 0), y + (jitter[x, y][1] if i not in (0, len(road) - 1) else 0)) for i, (x, y) in enumerate(road)])
    # Shared fork/end points remain unjittered in all incident roads.
    if key == "B":
        edited[0][2] = edited[1][0] = (-10, -8)
        edited[0][7] = edited[1][-1] = (10, 5)
    if key == "C":
        edited[0][2] = edited[1][0] = (-11, -5)
    roads = [smooth_path(road) for road in edited]
    # Reference paths are private QA probes, never painted or flattened into terrain.
    return {"key": key, "seed": seed, "reference_paths": roads, "goal": goal, "cut": cut, "spawn": roads[0][0], "title": {"A": "A - Off-road Rolling Ground", "B": "B - Off-road Washout Detour", "C": "C - Unseen Off-road Terrain"}[key]}


def natural_elevation(x, y, p):
    phase = (p["seed"] % 37) * .07
    # Continuous broad hills plus shorter undulations; no path-dependent grading.
    z = .9 + .022 * (x + 18) + .010 * (y + 11)
    z += .38 * np.sin(.18 * x + .10 * y + phase)
    z += .28 * np.cos(.24 * y - .09 * x + phase)
    shift = 1.4 if p["key"] == "C" else 0
    for hx, hy, amp, rx, ry in [(-8, -2, 1.15, 6, 5), (5, 5, 1.25, 7, 6), (13, -8, 1.0, 5.5, 4.5), (-14, 9, .75, 4, 5), (-4, 9, -.35, 3.5, 4)]:
        z += amp * np.exp(-((x - hx - shift) / rx) ** 2 - ((y - hy + shift) / ry) ** 2)
    z += .08 * np.sin(.62 * x + .25 * y + phase) * np.cos(.55 * y - .12 * x)
    z += .004 * np.sin(3.1 * x + .9 * y) * np.cos(2.5 * y)
    return z


def terrain(x, y, p):
    z = natural_elevation(x, y, p)
    # A small, unpainted smooth start area prevents a zero-command preview sliding.
    sx, sy = p["spawn"]
    blend = smooth((np.hypot(x - sx, y - sy) - .8) / 1.5)
    z = natural_elevation(sx, sy, p) * (1 - blend) + z * blend
    if p["cut"]:
        cx, cy = p["cut"]
        along = .88 * (x - cx) + .475 * (y - cy)
        across = -.475 * (x - cx) + .88 * (y - cy)
        z -= 1.15 * (1 - smooth((np.abs(along + .12 * np.sin(across)) - .34) / .42)) * (1 - smooth((np.abs(across) - 3.0) / 1.0))
    return z


def surface_texture(p, size):
    return flood_ground(p, size)[0]


def box(name, x, y, z, size, color, yaw=0, collision=True):
    dims = " ".join(map(str, size))
    return f'''DEF {name} Solid {{ name "{name.lower()}" translation {x:.4f} {y:.4f} {z:.4f} rotation 0 0 1 {yaw:.5f}
      children [ Shape {{ appearance PBRAppearance {{ baseColor {color} roughness .9 metalness 0 }} geometry Box {{ size {dims} }} }} ]
      {'boundingObject Box { size '+dims+' }' if collision else ''}
    }}'''


def rock(name, x, y, w, d, h, yaw, p):
    z = float(terrain(x, y, p)) + h / 2 - .015
    shape=[(-.47,-.33,-.5),(.31,-.48,-.5),(.5,.22,-.5),(.08,.46,-.5),(-.39,.31,-.5),(-.36,-.28,.16),(.08,-.43,.40),(.41,.15,.19),(.02,.30,.50),(-.32,.25,.28)]
    points=", ".join(f"{a*w:.5f} {b*d:.5f} {c*h:.5f}" for a,b,c in shape)
    faces='0 4 3 2 1 -1 0 1 6 5 -1 1 2 7 6 -1 2 3 8 7 -1 3 4 9 8 -1 4 0 5 9 -1 5 6 8 -1 6 7 8 -1 5 8 9 -1'
    return f'''DEF {name} Solid {{ name "{name.lower()}" translation {x:.4f} {y:.4f} {z:.4f} rotation 0 0 1 {yaw:.5f}
      children [ Shape {{ appearance {appearance("rock", ".78 .79 .77")}
        geometry DEF GEO_{name} IndexedFaceSet {{ coord Coordinate {{ point [ {points} ] }} coordIndex [ {faces} ] }}
      }} ] boundingObject USE GEO_{name}
    }}'''


def tree(name, x, y, size, p):
    ground = float(terrain(x, y, p))
    scale = .33 * size
    return f'''DEF {name} Solid {{ name "{name.lower()}" translation {x:.4f} {y:.4f} {ground:.4f}
      children [ Pine {{ enableBoundingObject FALSE scale {scale:.5f} {scale:.5f} {scale:.5f} rotation 0 0 1 {x * .27:.4f} }} ]
      boundingObject Pose {{ translation 0 0 {.65*size:.4f} children [ Cylinder {{ radius {.11*size:.4f} height {1.3*size:.4f} }} ] }}
    }}'''


def build(key, seed=None, texture_size=None):
    p = parameters(key, seed if seed is not None else CFG["scenario_seeds"][key])
    out = ROOT / "worlds" / "textures"
    out.mkdir(parents=True, exist_ok=True)
    diffuse, normal, roughness = flood_ground(p, texture_size or 4096)
    diffuse.save(out / f"{key}_ground.png")
    normal.save(out / f"{key}_ground_normal.jpg", quality=91)
    roughness.save(out / f"{key}_ground_rough.jpg", quality=91)
    spacing = CFG["terrain_spacing_m"]
    xx, yy = np.meshgrid(np.arange(-21, 21 + spacing / 2, spacing), np.arange(-15, 15 + spacing / 2, spacing))
    zz = np.round(terrain(xx, yy, p), 5)
    height_text = "\n".join(" ".join(f"{v:.5f}" for v in row) for row in zz)
    quiet = 'bumpSound "" rollSound "" slideSound ""'
    shadows = "TRUE"
    light = '-.65 -.35 -.8' if key != "C" else '.65 -.3 -.45'
    sky = '.62 .73 .81' if key != "C" else '.66 .70 .73'
    view_eye = [-27, -34, 23]
    view_target = [0, 0, 1.5]
    view_rotation = " ".join(f"{v:.7f}" for v in look_at_rotation(view_eye, view_target))
    nodes = [f'''#VRML_SIM R2025a utf8
EXTERNPROTO "../protos/VisionRover.proto"
EXTERNPROTO "../protos/Pine.proto"
WorldInfo {{ title "Vision Rescue | {p['title']}" basicTimeStep 10 coordinateSystem "ENU"
  contactProperties [
    ContactProperties {{ material1 "rubber" material2 "default" coulombFriction [ .95 ] bounce 0 {quiet} }}
    ContactProperties {{ material1 "caster" material2 "default" coulombFriction [ .015 ] bounce 0 {quiet} }}
  ]
}}
DEF SCENE_VIEW Viewpoint {{ orientation {view_rotation} position {' '.join(map(str, view_eye))} fieldOfView .95 far 240 exposure 1.35 ambientOcclusionRadius .85 }}
{hdr_sky()}
DirectionalLight {{ direction {light} color 1 .94 .83 intensity 1.65 ambientIntensity .18 castShadows {shadows} }}
DirectionalLight {{ direction .3 .4 -1 color .73 .83 1 intensity .40 ambientIntensity .08 castShadows FALSE }}
DEF TERRAIN Solid {{ name "terrain" translation -21 -15 0
  children [ Shape {{ appearance PBRAppearance {{ baseColorMap ImageTexture {{ url [ "textures/{key}_ground.png" ] }}
      normalMap ImageTexture {{ url [ "textures/{key}_ground_normal.jpg" ] }} normalMapFactor .70
      roughnessMap ImageTexture {{ url [ "textures/{key}_ground_rough.jpg" ] }} roughness .84 metalness 0 IBLStrength .85 }}
    geometry DEF GROUND_GRID ElevationGrid {{ xDimension {zz.shape[1]} yDimension {zz.shape[0]} xSpacing {spacing} ySpacing {spacing} height [
{height_text}
    ] }}
  }} ] boundingObject USE GROUND_GRID
}}
''']
    nodes.append(skyline(p, lambda x, y: terrain(x, y, p)))
    nodes.append(residual_water(p, lambda x, y: terrain(x, y, p)))
    rng = random.Random(p["seed"])
    obstacles = []
    # Scatter objects independently of reference paths; reserve only spawn/target.
    for i in range(55):
        x, y = rng.uniform(-19.5, 19.5), rng.uniform(-13.7, 13.7)
        if math.dist((x, y), p["goal"]) < 2 or math.dist((x, y), p["spawn"]) < 1.5:
            continue
        w = rng.uniform(.30, .85); depth = w * rng.uniform(.6, .9); h = w * rng.uniform(.45, .85)
        yaw = rng.uniform(-3, 3)
        name = f"ROCK_{i}"
        nodes.append(rock(name, x, y, w, depth, h, yaw, p))
        obstacles.append({"def": name, "xy": [x, y], "size": [w, depth, h], "yaw": yaw})
    for i in range(28):
        x, y = rng.uniform(-19, 19), rng.uniform(-13, 13)
        if math.dist((x, y), p["goal"]) < 2 or math.dist((x, y), p["spawn"]) < 1.5:
            continue
        size = rng.uniform(.8, 1.35)
        house_shift = {"A": -.5, "B": .0, "C": 1.2}[key]
        houses = [(-7.5 + house_shift, -1.2), (1.9 + house_shift, -8.4), (6.8 + house_shift, -.2)]
        if any(math.dist((x, y), center) < 3.8 for center in houses):
            continue
        name = f"TREE_{i}"
        nodes.append(tree(name, x, y, size, p))
        obstacles.append({"def": name, "xy": [x, y], "size": [.22 * size, .22 * size, 1.3 * size], "yaw": 0})
    for i in range(14):
        x, y = rng.uniform(-19, 19), rng.uniform(-13, 13)
        if math.dist((x, y), p["spawn"]) < 1.5 or math.dist((x, y), p["goal"]) < 2:
            continue
        nodes.append(box(f"ROOT_{i}", x, y, float(terrain(x, y, p)) + .01, (.075, .43, .020), ".30 .24 .15", rng.uniform(-math.pi, math.pi)))
    rescue = RescueElements(p, lambda x, y: float(terrain(x, y, p))).build()
    nodes.extend(rescue["nodes"])
    obstacles.extend(rescue["obstacles"])
    gx, gy = p["goal"]
    sx, sy = p["spawn"]
    dx, dy = np.subtract(p["reference_paths"][0][1], p["reference_paths"][0][0])
    yaw = math.atan2(dy, dx)
    actual_bearing = math.degrees((math.atan2(gy-sy,gx-sx)-yaw+math.pi)%(2*math.pi)-math.pi)
    coarse = 15 * round(actual_bearing / 15) + random.Random(p["seed"] + 81).choice((-5, 0, 5))
    public = {"scenario_id": key, "mission": "Search natural off-road terrain for a casualty; no marked route or prior map.", "goal_hint": {"reference": "initial_camera_forward", "center_deg_left_positive": coarse, "half_width_deg": CFG["goal_hint_half_width_deg"], "issued_once": True}, "camera": CFG["camera"], "action_limits": {"linear_m_s": CFG["nominal_speed_limit_m_s"], "angular_rad_s": CFG["turn_rate_limit_rad_s"]}}
    public_json = json.dumps(json.dumps(public, ensure_ascii=True))
    nodes.append(f'''DEF ROVER VisionRover {{ name "Vision Rescue Rover" translation {sx} {sy} {float(terrain(sx,sy,p))+.035:.5f} rotation 0 0 1 {yaw:.7f}
      customData {public_json} cameraNoise {0.003 if key != 'C' else 0.006}
    }}
    DEF EVALUATOR Robot {{ name "Independent evaluator" supervisor TRUE controller "scene_evaluator" window "<none>" customData "{key}" }}''')
    world = ROOT / "worlds" / f"{key}_{ {'A':'trail','B':'washout','C':'generalization'}[key]}.wbt"
    world.write_text("\n".join(nodes), encoding="utf-8")
    log_x, log_y = -7.5, -1.2
    inspection_views = {
        "casualty": {"eye": [gx-2.6, gy-3.3, float(terrain(gx,gy,p))+2.4], "target": [gx,gy,float(terrain(gx,gy,p))+.13]},
        "obstacles": {"eye": [log_x-4.2,log_y-4.5,float(terrain(log_x,log_y,p))+5.5], "target": [log_x,log_y,float(terrain(log_x,log_y,p))+1.0]},
    }
    metadata = {**p, "surface_mode": "unmarked_offroad", "elements_revision": 3, "elements": rescue["elements"], "foliage_spheres": rescue["foliage"], "target_body": rescue["target_body"], "inspection_views": inspection_views, "elevation_range_m": [float(zz.min()), float(zz.max())], "origin": CFG["origin"], "spacing": spacing, "heights": zz.tolist(), "spawn_yaw": yaw, "obstacles": obstacles, "success_distance_m": CFG["success_distance_m"], "required_stop_duration_s": CFG["required_stop_duration_s"], "episode_timeout_s": CFG["episode_timeout_s"], "actual_initial_bearing_deg": actual_bearing, "world_file": world.name}
    (ROOT / "evaluation" / "scenarios" / f"{key}.json").write_text(json.dumps(metadata), encoding="utf-8")
    (ROOT / "config" / "public" / f"{key}_task.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
    layout = (ROOT / "config" / "view_layout.wbproj").read_text(encoding="utf-8").rstrip() + '\nrenderingDevicePerspectives: Vision Rescue Rover:camera_left;1;0.42;0.02;0.73\nrenderingDevicePerspectives: Vision Rescue Rover:camera_right;1;0.42;0.235;0.73\n'
    layout_file = world.parent / f".{world.stem}.wbproj"
    # Webots marks project files Hidden on Windows; truncating with 'w' then fails.
    if layout_file.exists():
        with layout_file.open("r+", encoding="utf-8") as handle:
            handle.seek(0)
            handle.write(layout)
            handle.truncate()
    else:
        layout_file.write_text(layout, encoding="utf-8")
    print(f"Built {world.name}: {zz.shape[1]}x{zz.shape[0]} terrain, seed={p['seed']}, hint={coarse:+.0f}+/-20 deg", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["A","B","C","all"], default="all")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--texture-size", type=int)
    args = parser.parse_args()
    if args.seed is not None and args.scenario == "all":
        parser.error("Use --scenario A/B/C when overriding a seed")
    for key in "ABC" if args.scenario == "all" else args.scenario:
        build(key, args.seed, args.texture_size)
