"""Flood rescue props with local textured visuals and simplified collision."""
from __future__ import annotations

import math
import random

import numpy as np
from render_style import appearance, foliage_cluster
from flood_ruins import add_flood_ruins
from casualty_visual import casualty_render, HEAD_PROXY_POSITIONS, NECK_PROXY_DIMENSIONS, VISUAL_MODEL


def numbers(values):
    return " ".join(f"{float(v):.6f}" for v in values)


def axis_angle(matrix):
    angle = math.acos(float(np.clip((np.trace(matrix) - 1) / 2, -1, 1)))
    if angle < 1e-8:
        return [0, 0, 1, 0]
    axis = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    if np.linalg.norm(axis) < 1e-7:
        # Stable eigenvector for the 180 degree case.
        values, vectors = np.linalg.eigh(matrix)
        axis = vectors[:, np.argmax(values)]
    axis = axis / np.linalg.norm(axis)
    return [*axis, angle]


def ground_frame(x, y, yaw, elevation):
    delta = .05
    dx = (elevation(x + delta, y) - elevation(x - delta, y)) / (2 * delta)
    dy = (elevation(x, y + delta) - elevation(x, y - delta)) / (2 * delta)
    up = np.array([-dx, -dy, 1.0]); up /= np.linalg.norm(up)
    forward = np.array([math.cos(yaw), math.sin(yaw), dx * math.cos(yaw) + dy * math.sin(yaw)])
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    return np.column_stack((forward, left, up))


def part(name, shape, dimensions, at, color, rotation=(0, 0, 1, 0), hard=True):
    return dict(name=name, shape=shape, dimensions=dimensions, at=at, color=color, rotation=rotation, hard=hard)


def link(name, start, end, radius, color):
    start, end = np.asarray(start), np.asarray(end)
    direction = end - start
    length = float(np.linalg.norm(direction))
    direction /= length
    axis = np.cross([0, 0, 1], direction)
    norm = np.linalg.norm(axis)
    rotation = [1, 0, 0, math.pi] if norm < 1e-8 and direction[2] < 0 else [0, 0, 1, 0]
    if norm >= 1e-8:
        rotation = [*(axis / norm), math.acos(float(direction[2]))]
    return part(name, "Cylinder", [radius, length], ((start + end) / 2).tolist(), color, rotation)


class RescueElements:
    def __init__(self, parameters, elevation):
        self.p = parameters
        self.elevation = elevation
        self.rng = random.Random(parameters["seed"] + 1907)
        self.nodes = []
        self.obstacles = []
        self.foliage = []
        self.manifest = []

    def add(self, name, kind, xy, yaw, parts, footprint=None, grade="rigid", target=False):
        x, y = xy
        ground = float(self.elevation(x, y))
        frame = ground_frame(x, y, yaw, self.elevation)
        visual, collision = [], []
        for item in parts:
            shape, dims = item["shape"], item["dimensions"]
            if shape == "Box":
                geometry = f"Box {{ size {numbers(dims)} }}"
            elif shape == "Cylinder":
                geometry = f"Cylinder {{ radius {dims[0]:.5f} height {dims[1]:.5f} subdivision 10 }}"
            elif shape == "Sphere":
                geometry = f"Sphere {{ radius {dims[0]:.5f} subdivision 3 }}"
            elif shape in ("JaggedWall", "BrokenBrick"):
                w, d, h = dims
                profiles = [
                    [1.00, .98, .97, .83, .79, .81, .55, .57, .42, .45],
                    [.57, .60, .85, .83, .94, .89, .72, .61, .59, .43],
                    [.92, .91, .68, .71, .53, .47, .52, .33, .29, .34],
                ]
                profile = profiles[sum(ord(c) for c in name + item["name"]) % len(profiles)]
                vertices = []
                count = len(profile)
                for depth in (-d / 2, d / 2):
                    vertices.extend([(-w / 2, depth, -h / 2), (w / 2, depth, -h / 2)])
                    vertices.extend((-w / 2 + w * i / (count - 1), depth, -h / 2 + h * q)
                                    for i, q in enumerate(profile))
                span = count + 2
                back = " ".join(str(i) for i in [0, 1, *range(count + 1, 1, -1)])
                front = " ".join(str(i + span) for i in [1, 0, *range(2, count + 2)])
                indices = [back + " -1", front + " -1",
                           f"0 {span} {span+1} 1 -1",
                           f"0 2 {span+2} {span} -1",
                           f"1 {span+1} {span+count+1} {count+1} -1"]
                for i in range(count - 1):
                    a = i + 2
                    indices.append(f"{a} {a+1} {a+1+span} {a+span} -1")
                uv = ", ".join(
                    f"{((vx+w/2)/w if w>=d else (vy+d/2)/d):.5f} {(vz+h/2)/max(h,.05):.5f}"
                    for vx, vy, vz in vertices
                )
                geometry = f"IndexedFaceSet {{ coord Coordinate {{ point [ {', '.join(numbers(v) for v in vertices)} ] }} texCoord TextureCoordinate {{ point [ {uv} ] }} coordIndex [ {' '.join(indices)} ] texCoordIndex [ {' '.join(indices)} ] creaseAngle .15 }}"
            elif shape == "RoofPanel":
                w, d, thickness = dims
                columns = 36
                vertices = []
                for panel_y in (-d / 2, d / 2):
                    for i in range(columns + 1):
                        panel_x = -w / 2 + w * i / columns
                        z = .014 * math.cos(i * math.pi) + .018 * math.sin(i * .44)
                        vertices.append((panel_x, panel_y, z))
                span = columns + 1
                indices = []
                for i in range(columns):
                    indices.extend((f"{i} {i+1} {i+1+span} {i+span} -1",
                                    f"{i+span} {i+1+span} {i+1} {i} -1"))
                geometry = f"IndexedFaceSet {{ coord Coordinate {{ point [ {', '.join(numbers(v) for v in vertices)} ] }} coordIndex [ {' '.join(indices)} ] }}"
            else:
                raise ValueError(shape)
            geo_name = f"GEO_{name}_{item['name']}"
            pose = f"translation {numbers(item['at'])} rotation {numbers(item['rotation'])}"
            style = item.get("material")
            if style is None and kind in ("fallen_log", "low_root") and not item["name"].startswith("END"):
                style = "wood"
            if style is None and kind in ("rubble", "broken_wall"):
                style = "plaster"
            material = appearance(style, "0.9 0.86 0.78") if style else f"PBRAppearance {{ baseColor {item['color']} roughness .81 metalness 0 }}"
            if target:
                material = "PBRAppearance { transparency 1 roughness 1 metalness 0 }"
            if kind == "shrub" and item["name"].startswith("LEAVES"):
                geometry = foliage_cluster(dims[0])
                material = 'PBRAppearance { baseColor .72 .83 .63 baseColorMap ImageTexture { url [ "../protos/textures/pine_leaves.png" ] } roughness .92 metalness 0 }'
            visual.append(f'''Pose {{ {pose} children [ Shape {{
              appearance {material}
              geometry DEF {geo_name} {geometry}
            }} ] }}''')
            if item["hard"]:
                collision.append(f"Pose {{ {pose} children [ USE {geo_name} ] }}")
            elif shape == "Sphere":
                center = np.array([x, y, ground]) + frame @ np.asarray(item["at"])
                self.foliage.append({"def": name, "center": center.tolist(), "radius": dims[0]})
        if target:
            visual.append(casualty_render(self.p["key"]))
        bounding = "boundingObject Group { children [ " + "\n".join(collision) + " ] }" if collision else ""
        self.nodes.append(f'''DEF {name} Solid {{ name "{name.lower()}"
          translation {x:.5f} {y:.5f} {ground:.5f} rotation {numbers(axis_angle(frame))}
          children [ {' '.join(visual)} ] {bounding}
        }}''')
        record = {"def": name, "kind": kind, "xy": list(xy), "yaw": yaw, "grade": grade,
                  "hard_parts": [q["name"] for q in parts if q["hard"]],
                  "visual_only_parts": [q["name"] for q in parts if not q["hard"]]}
        self.manifest.append(record)
        if footprint and not target:
            self.obstacles.append({"def": name, "xy": list(xy), "size": footprint, "yaw": yaw,
                                   "kind": kind, "traversal": grade, "ground_z": ground})
        return record

    def fallen_log(self, name, xy, yaw, length, diameter):
        radius = diameter / 2
        wood = ".26 .17 .095"
        pieces = [part("TRUNK", "Cylinder", [radius, length], [0, 0, radius], wood, [0, 1, 0, math.pi / 2])]
        # Light wood at broken ends, with the same physical surface.
        for side in (-1, 1):
            pieces.append(part(f"END_{'L' if side < 0 else 'R'}", "Cylinder", [radius * .93, .008], [side * length / 2, 0, radius], ".52 .40 .24", [0, 1, 0, math.pi / 2]))
        pieces.append(link("BROKEN_BRANCH", [-.3, 0, radius], [-.1, .34, radius + .15], .035, wood))
        self.add(name, "fallen_log", xy, yaw, pieces, [length + .02, max(diameter, .70), diameter + .19], "detour")

    def shrub(self, name, xy, scale=1.0):
        pieces = [part("STEM", "Cylinder", [.028 * scale, .42 * scale], [0, 0, .21 * scale], ".24 .19 .10")]
        for i in range(7):
            angle = i * 2 * math.pi / 7 + self.rng.uniform(-.18, .18)
            distance = self.rng.uniform(.12, .28) * scale
            center = [distance * math.cos(angle), distance * math.sin(angle), self.rng.uniform(.28, .46) * scale]
            pieces.append(part(f"LEAVES_{i}", "Sphere", [self.rng.uniform(.20, .27) * scale], center,
                               self.rng.choice((".20 .29 .13", ".25 .32 .15", ".17 .27 .12")), hard=False))
        self.add(name, "shrub", xy, self.rng.uniform(-math.pi, math.pi), pieces, [.06 * scale, .06 * scale, .42 * scale], "stem")

    def casualty(self):
        key = self.p["key"]
        shirt = {"A": ".48 .26 .20", "B": ".19 .29 .44", "C": ".46 .43 .29"}[key]
        pants, skin = ".16 .19 .22", ".62 .44 .33"
        pieces = [part("TORSO", "Box", [.48, .33, .21], [-.25, 0, .145], shirt),
                  part("PELVIS", "Box", [.22, .31, .18], [.09, 0, .12], pants),
                  part("NECK", "Cylinder", [.065, .10], [-.54, 0, .14], skin, [0, 1, 0, math.pi / 2]),
                  part("HEAD", "Sphere", [.115], [-.66, 0, .145], skin),
                  part("HAIR", "Sphere", [.105], [-.69, 0, .14], ".12 .10 .085"),
                  part("NOSE", "Sphere", [.025], [-.64, 0, .25], skin),
                  part("EYE_L", "Sphere", [.010], [-.68, .043, .248], ".08 .07 .065"),
                  part("EYE_R", "Sphere", [.010], [-.68, -.043, .248], ".08 .07 .065")]
        for side in (-1, 1):
            suffix = "L" if side > 0 else "R"
            shoulder = [-.41, side * .20, .14]
            elbow = [-.16, side * .30, .10]
            wrist = [.06, side * (.38 if key == "C" else .29), .075]
            hip = [.13, side * .095, .12]
            knee = [.43, side * (.18 if key == "B" and side > 0 else .12), .11]
            ankle = [.75, side * .14, .09]
            for label, a, b, radius, color in (("UPPER_ARM", shoulder, elbow, .066, shirt), ("FOREARM", elbow, wrist, .052, skin), ("THIGH", hip, knee, .083, pants), ("SHIN", knee, ankle, .065, pants)):
                pieces.append(link(f"{label}_{suffix}", a, b, radius, color))
            pieces.extend([part(f"HAND_{suffix}", "Sphere", [.052], wrist, skin),
                           part(f"KNEE_{suffix}", "Sphere", [.082], knee, pants),
                           part(f"BOOT_{suffix}", "Box", [.20, .115, .115], [.84, side * .14, .085], ".10 .095 .085")])
        for item in pieces:
            if item["name"] in HEAD_PROXY_POSITIONS:
                item["at"] = list(HEAD_PROXY_POSITIONS[item["name"]])
            if item["name"] == "NECK":
                item["dimensions"] = list(NECK_PROXY_DIMENSIONS)
        yaw = {"A": -.45, "B": .75, "C": -1.0}[key]
        record = self.add("CASUALTY", "casualty", self.p["goal"], yaw, pieces, target=True)
        record.update({"shirt_color": shirt, "pose": "supine", "body_length_m": 1.67, "visual_model": VISUAL_MODEL})
        return {"xy": list(self.p["goal"]), "size": [1.76, .85, .29], "yaw": yaw}

    def build(self):
        key = self.p["key"]
        shift = {"A": 0, "B": .65, "C": -1.15}[key]
        for i, (x, y, yaw, length, diameter) in enumerate([(-14, -8.8, 1.25, 2.5, .28), (-5, 3.6, -.45, 2.8, .34), (8.6, -.5, .7, 2.2, .24)]):
            self.fallen_log(f"FALLEN_LOG_{i}", (x + shift, y), yaw + shift * .2, length, diameter)
        # Two small rubble clusters, rather than continuous walls defining a road.
        for cluster, (cx, cy) in enumerate([(-10 + shift, -2.9), (8.5, 6.8 + shift)]):
            for i in range(5):
                x, y = cx + self.rng.uniform(-.9, .9), cy + self.rng.uniform(-.65, .65)
                w, d, h = self.rng.uniform(.3, .65), self.rng.uniform(.25, .50), self.rng.uniform(.18, .43)
                self.add(f"RUBBLE_{cluster}_{i}", "rubble", (x, y), self.rng.uniform(-3, 3),
                         [part("BLOCK", "Box", [w, d, h], [0, 0, h / 2], self.rng.choice((".42 .41 .37", ".47 .44 .38", ".37 .38 .36")))], [w, d, h], "detour")
        for i, (x, y, yaw) in enumerate([(-2 + shift, 5.8, .55), (11.8, -4.0 + shift, -1.1)]):
            parts = [part("BASE", "Box", [1.8, .28, .42], [0, 0, .21], ".45 .44 .40"),
                     part("BROKEN_TOP", "Box", [.85, .28, .28], [-.42, 0, .56], ".49 .47 .42"),
                     part("FALLEN_PIECE", "Box", [.55, .46, .18], [.67, .26, .09], ".42 .41 .37")]
            self.add(f"BROKEN_WALL_{i}", "broken_wall", (x, y), yaw, parts, [1.9, 1.0, .70], "detour")
        # Low rounded roots are candidates for crossing; higher trunks require detours.
        for i, (x, y) in enumerate([(-15.6, -10.5), (-7, -8.8), (3.5, 8.2)]):
            self.add(f"LOW_ROOT_{i}", "low_root", (x + shift, y), .9 + i * .6,
                     [part("ROOT", "Cylinder", [.018, .65], [0, 0, .005], ".32 .24 .14", [0, 1, 0, math.pi / 2])],
                     [.65, .036, .023], "low_candidate")
        for i in range(16):
            x, y = self.rng.uniform(-18, 18), self.rng.uniform(-12, 12)
            if math.dist((x, y), self.p["spawn"]) < 2 or math.dist((x, y), self.p["goal"]) < 2:
                continue
            self.shrub(f"SHRUB_{i}", (x, y), self.rng.uniform(.75, 1.2))
        self.shrub("NEAR_SHRUB", (-15.5 + shift * .25, -8.0), 1.0)
        gx, gy = self.p["goal"]
        # Off-center foliage partially hides the casualty, leaving an open south approach.
        self.shrub("TARGET_SHRUB", (gx - .25, gy + .78), .75)
        target = self.casualty()
        self.add("BACKPACK", "backpack", (gx - 1.0, gy + .85), -.5,
                 [part("BAG", "Box", [.32, .27, .34], [0, 0, .17], ".30 .29 .21"),
                  part("POCKET", "Box", [.21, .045, .19], [0, -.15, .14], ".24 .24 .18")],
                 [.32, .32, .34], "detour")
        # Grass blades have visible geometry only, not an invisible collision volume.
        for i in range(10):
            x, y = self.rng.uniform(-18, 18), self.rng.uniform(-12, 12)
            points, faces = [], []
            for blade in range(16):
                bx, by = self.rng.uniform(-.3, .3), self.rng.uniform(-.3, .3)
                z = float(self.elevation(x + bx, y + by))
                h = self.rng.uniform(.18, .40)
                index = len(points)
                points.extend(((x + bx - .018, y + by, z), (x + bx + .018, y + by, z), (x + bx + .05, y + by + .025, z + h)))
                # Explicit back face avoids the installed build's unsupported solid flag.
                faces.extend((f"{index} {index+1} {index+2} -1", f"{index+2} {index+1} {index} -1"))
            self.nodes.append(f'''DEF GRASS_{i} Solid {{ name "grass_{i}" children [ Shape {{
              appearance PBRAppearance {{ baseColor .27 .34 .16 roughness 1 metalness 0 }}
              geometry IndexedFaceSet {{ coord Coordinate {{ point [ {', '.join(numbers(q) for q in points)} ] }} coordIndex [ {' '.join(faces)} ] }}
            }} ] }}''')
            self.manifest.append({"def": f"GRASS_{i}", "kind": "grass", "xy": [x, y], "grade": "visual_only", "hard_parts": [], "visual_only_parts": ["BLADES"]})
        add_flood_ruins(self, part, link)
        return {"nodes": self.nodes, "obstacles": self.obstacles, "foliage": self.foliage,
                "elements": self.manifest, "target_body": target}
