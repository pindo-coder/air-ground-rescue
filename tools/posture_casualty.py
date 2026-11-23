"""Multi-posture 3D disaster casualty VRML generator & exact visible 3D keypoint sets.

Supports 6 realistic disaster victim postures:
- supine      : textured high-res supine human mesh (face-up)
- prone       : textured high-res human mesh face-down (back & hair up)
- side        : textured high-res human mesh lying laterally on its side
- curled      : articulated PBR human curled tightly on its side (fetal/drawn-knees compact oval)
- seated      : articulated PBR human slumped/seated on ground with bent knees (compact vertical)
- half_buried : textured supine human with concrete rubble slab burying legs/hips (upper body only)
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation

from casualty_visual import casualty_render
from scene_elements import ground_frame, link, numbers, part

ALL_POSTURES = ("supine", "prone", "side", "curled", "seated", "half_buried")
NON_SUPINE_POSTURES = ("prone", "side", "curled", "seated", "half_buried")

POSTURE_CN = {
    "supine": "仰卧",
    "prone": "俯卧",
    "side": "侧卧",
    "curled": "蜷缩成团",
    "seated": "抱膝坐姿",
    "half_buried": "废墟半掩埋",
}


def _supine_points():
    return np.array(
        [[x, y, z] for x in (-0.68, -0.45, -0.20, 0.05, 0.35, 0.62, 0.86)
         for y in (-0.25, 0.0, 0.25) for z in (0.08, 0.22)]
        + [[-0.62, 0.0, 0.25], [-0.30, -0.32, 0.12], [-0.30, 0.32, 0.12],
           [0.05, -0.32, 0.10], [0.05, 0.32, 0.10], [0.88, -0.14, 0.10], [0.88, 0.14, 0.10]],
        dtype=float,
    )


def _prone_points():
    pts = _supine_points().copy()
    pts[:, 1] = -pts[:, 1]
    pts[:, 2] = np.clip(0.255 - pts[:, 2], 0.04, 0.26)
    return pts


def _side_points():
    # Rotated +90 deg around X -> old Y becomes Z, old Z becomes -Y, shifted +0.175 in Z
    pts = _supine_points().copy()
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0]
    out[:, 1] = -pts[:, 2] + 0.11
    out[:, 2] = np.clip(pts[:, 1] + 0.22, 0.05, 0.44)
    return out


def _curled_points():
    # Compact fetal/curled silhouette (~0.92 m long x 0.58 m wide)
    return np.array(
        [[x, y, z] for x in (-0.42, -0.22, 0.0, 0.20, 0.38)
         for y in (-0.20, 0.0, 0.22) for z in (0.08, 0.24)]
        + [[-0.44, -0.06, 0.18], [-0.12, 0.26, 0.16], [0.16, 0.28, 0.16],
           [0.42, 0.04, 0.12], [0.22, -0.18, 0.15]],
        dtype=float,
    )


def _seated_points():
    # Slumped seated figure: torso upright z=0.14..0.78, knees bent forward x=+0.34
    return np.array(
        [[-0.14, y, z] for y in (-0.19, 0.0, 0.19) for z in (0.18, 0.38, 0.56, 0.72)]
        + [[-0.10, 0.0, 0.80], [-0.10, -0.09, 0.78], [-0.10, 0.09, 0.78]]
        + [[x, y, z] for x in (0.08, 0.26, 0.44) for y in (-0.16, 0.0, 0.16) for z in (0.10, 0.28)],
        dtype=float,
    )


def _half_buried_points():
    # ONLY exposed upper torso, head & arms (x <= -0.04); rubble slab covers legs/hips!
    return np.array(
        [[x, y, z] for x in (-0.68, -0.52, -0.34, -0.16, -0.04)
         for y in (-0.24, 0.0, 0.24) for z in (0.08, 0.22)]
        + [[-0.62, 0.0, 0.25], [-0.36, -0.32, 0.12], [-0.36, 0.32, 0.12],
           [-0.12, -0.30, 0.10], [-0.12, 0.30, 0.10]],
        dtype=float,
    )


POSTURE_LOCAL_POINTS = {
    "supine": _supine_points(),
    "prone": _prone_points(),
    "side": _side_points(),
    "curled": _curled_points(),
    "seated": _seated_points(),
    "half_buried": _half_buried_points(),
}


def _recolor_mesh_vrml(mesh_vrml, shirt_rgb=None, pants_rgb=None, shirt_def=None, pants_def=None):
    out = []
    s_done = False
    p_done = False
    for line in mesh_vrml.splitlines():
        if "_shirt.obj" in line:
            if shirt_rgb:
                parts = line.split("baseColor ")
                if len(parts) == 2:
                    rest = parts[1].split(" ", 3)
                    if len(rest) == 4:
                        line = parts[0] + f"baseColor {shirt_rgb} " + rest[3]
            if shirt_def and not s_done and "appearance PBRAppearance {" in line:
                line = line.replace("appearance PBRAppearance {", f"appearance DEF {shirt_def} PBRAppearance {{", 1)
                s_done = True
        elif "_pants.obj" in line:
            if pants_rgb:
                parts = line.split("baseColor ")
                if len(parts) == 2:
                    rest = parts[1].split(" ", 3)
                    if len(rest) == 4:
                        line = parts[0] + f"baseColor {pants_rgb} " + rest[3]
            if pants_def and not p_done and "appearance PBRAppearance {" in line:
                line = line.replace("appearance PBRAppearance {", f"appearance DEF {pants_def} PBRAppearance {{", 1)
                p_done = True
        out.append(line)
    return "\n".join(out)


def _render_articulated_pieces(pieces, shirt_rgb, pants_rgb, shirt_def=None, pants_def=None):
    """Render articulated limb pieces with shared DEF PBRAppearances for live color randomization."""
    shirt_app_name = shirt_def or "ART_SHIRT_APP"
    pants_app_name = pants_def or "ART_PANTS_APP"
    shirt_used = False
    pants_used = False
    nodes = []
    for item in pieces:
        role = item.get("role")
        if role == "shirt":
            if not shirt_used:
                app = f"DEF {shirt_app_name} PBRAppearance {{ baseColor {shirt_rgb} roughness 0.92 metalness 0 }}"
                shirt_used = True
            else:
                app = f"USE {shirt_app_name}"
        elif role == "pants":
            if not pants_used:
                app = f"DEF {pants_app_name} PBRAppearance {{ baseColor {pants_rgb} roughness 0.92 metalness 0 }}"
                pants_used = True
            else:
                app = f"USE {pants_app_name}"
        else:
            app = f"PBRAppearance {{ baseColor {item['color']} roughness 0.88 metalness 0 }}"
        kind = item["shape"]
        dims = item["dimensions"]
        if kind == "Box":
            geom = f"Box {{ size {numbers(dims)} }}"
        elif kind == "Cylinder":
            geom = f"Cylinder {{ radius {dims[0]} height {dims[1]} subdivision 14 }}"
        elif kind == "Sphere":
            geom = f"Sphere {{ radius {dims[0]} subdivision 2 }}"
        else:
            geom = f"Capsule {{ radius {dims[0]} height {dims[1]} subdivision 14 }}"
        rot_str = f"rotation {numbers(item['rotation'])}" if item.get("rotation") else ""
        nodes.append(
            f"Pose {{ translation {numbers(item['at'])} {rot_str} children [ Shape {{ appearance {app} geometry {geom} }} ] }}"
        )
    return "\n".join(nodes)


def _curled_pieces(shirt_rgb, pants_rgb):
    skin, hair, boot = ".62 .44 .33", ".12 .10 .085", ".10 .095 .085"
    pieces = [
        {**part("TORSO", "Box", [.40, .24, .24], [-.14, -.06, .16], shirt_rgb, [0, 0, 1, .28]), "role": "shirt"},
        {**part("PELVIS", "Box", [.22, .24, .22], [.14, -.02, .14], pants_rgb, [0, 0, 1, -.22]), "role": "pants"},
        part("HEAD", "Sphere", [.115], [-.39, .05, .165], skin),
        part("HAIR", "Sphere", [.108], [-.42, .01, .165], hair),
        # Curled upper & lower legs drawn tightly forward to chest
        {**link("THIGH_L", [.16, .04, .18], [.08, .28, .20], .082, pants_rgb), "role": "pants"},
        {**link("THIGH_R", [.16, -.02, .10], [.06, .24, .11], .082, pants_rgb), "role": "pants"},
        {**link("SHIN_L", [.08, .28, .20], [.36, .16, .16], .064, pants_rgb), "role": "pants"},
        {**link("SHIN_R", [.06, .24, .11], [.35, .12, .09], .064, pants_rgb), "role": "pants"},
        part("BOOT_L", "Box", [.18, .10, .10], [.40, .14, .14], boot),
        part("BOOT_R", "Box", [.18, .10, .10], [.39, .10, .08], boot),
        # Arms tucked defensively across chest/knees
        {**link("UARM_L", [-.24, .02, .22], [-.10, .16, .24], .064, shirt_rgb), "role": "shirt"},
        {**link("FARM_L", [-.10, .16, .24], [-.20, .22, .20], .050, skin)},
        {**link("UARM_R", [-.24, -.04, .10], [-.08, .14, .11], .062, shirt_rgb), "role": "shirt"},
        {**link("FARM_R", [-.08, .14, .11], [-.24, .18, .12], .048, skin)},
    ]
    return pieces


def _seated_pieces(shirt_rgb, pants_rgb):
    skin, hair, boot = ".62 .44 .33", ".12 .10 .085", ".10 .095 .085"
    pieces = [
        # Upright / slightly slumped torso & pelvis
        {**part("PELVIS", "Box", [.26, .32, .18], [-.06, 0, .10], pants_rgb), "role": "pants"},
        {**part("TORSO", "Box", [.24, .34, .44], [-.13, 0, .40], shirt_rgb, [0, 1, 0, .18]), "role": "shirt"},
        part("NECK", "Cylinder", [.055, .09], [-.09, 0, .64], skin),
        part("HEAD", "Sphere", [.115], [-.07, 0, .74], skin),
        part("HAIR", "Sphere", [.108], [-.10, 0, .76], hair),
    ]
    for side in (-1, 1):
        sf = "L" if side > 0 else "R"
        sh = [-.12, side * .20, .54]
        el = [.05, side * .23, .36]
        wr = [.22, side * .14, .32]
        hp = [.02, side * .10, .12]
        kn = [.28, side * .15, .28]
        an = [.44, side * .14, .08]
        pieces.extend([
            {**link(f"UARM_{sf}", sh, el, .064, shirt_rgb), "role": "shirt"},
            link(f"FARM_{sf}", el, wr, .050, skin),
            part(f"HAND_{sf}", "Sphere", [.048], wr, skin),
            {**link(f"THIGH_{sf}", hp, kn, .082, pants_rgb), "role": "pants"},
            {**link(f"SHIN_{sf}", kn, an, .064, pants_rgb), "role": "pants"},
            {**part(f"KNEE_{sf}", "Sphere", [.080], kn, pants_rgb), "role": "pants"},
            part(f"BOOT_{sf}", "Box", [.19, .11, .11], [.50, side * .14, .07], boot),
        ])
    return pieces


def build_posture_casualty_vrml(
    def_name, name, xy, yaw, elevation, posture="supine",
    mesh_key="B", shirt_rgb=None, pants_rgb=None,
    shirt_def=None, pants_def=None,
):
    """Build a complete Webots Solid node for a casualty in any of the 6 disaster postures."""
    if posture not in ALL_POSTURES:
        raise ValueError(f"Unsupported posture: {posture}")
    shirt_rgb = shirt_rgb or ".19 .29 .44"
    pants_rgb = pants_rgb or ".16 .19 .22"
    frame = ground_frame(xy[0], xy[1], yaw, elevation)
    rv = Rotation.from_matrix(frame).as_rotvec()
    ang = float(np.linalg.norm(rv))
    rot_aa = [*(rv / ang).tolist(), ang] if ang >= 1e-8 else [0.0, 0.0, 1.0, 0.0]
    trans_xyz = [float(xy[0]), float(xy[1]), float(elevation(xy[0], xy[1]))]

    if posture in ("supine", "prone", "side", "half_buried"):
        mesh_vrml = _recolor_mesh_vrml(
            casualty_render(mesh_key),
            shirt_rgb=shirt_rgb, pants_rgb=pants_rgb,
            shirt_def=shirt_def, pants_def=pants_def,
        )
        if posture == "supine":
            body_children = mesh_vrml
            bbox_pose = "Pose { translation 0.08 0 0.14 children [ Box { size 1.66 0.62 0.26 } ] }"
        elif posture == "prone":
            body_children = f"Pose {{ translation 0 0 0.255 rotation 1 0 0 3.14159265 children [\n{mesh_vrml}\n] }}"
            bbox_pose = "Pose { translation 0.08 0 0.14 children [ Box { size 1.66 0.62 0.26 } ] }"
        elif posture == "side":
            body_children = f"Pose {{ translation 0 0.11 0.22 rotation 1 0 0 1.57079633 children [\n{mesh_vrml}\n] }}"
            bbox_pose = "Pose { translation 0.08 0 0.21 children [ Box { size 1.66 0.34 0.42 } ] }"
        else:  # half_buried: weathered concrete rubble slab burying legs/hips
            rubble = (
                "Pose { translation 0.44 0.03 0.15 rotation 0 0 1 0.14 children [ Shape { "
                "appearance PBRAppearance { baseColor 0.48 0.45 0.41 roughness 0.94 metalness 0 } "
                "geometry Box { size 0.96 0.78 0.26 } } ] }\n"
                "Pose { translation 0.28 -0.22 0.23 rotation 0.2 0.3 0.9 0.5 children [ Shape { "
                "appearance PBRAppearance { baseColor 0.42 0.38 0.34 roughness 0.95 metalness 0 } "
                "geometry Box { size 0.42 0.36 0.18 } } ] }"
            )
            body_children = mesh_vrml + "\n" + rubble
            bbox_pose = "Pose { translation 0.12 0 0.16 children [ Box { size 1.72 0.78 0.30 } ] }"
    elif posture == "curled":
        body_children = _render_articulated_pieces(
            _curled_pieces(shirt_rgb, pants_rgb), shirt_rgb, pants_rgb,
            shirt_def=shirt_def, pants_def=pants_def,
        )
        bbox_pose = "Pose { translation 0.0 0.06 0.16 children [ Box { size 0.94 0.56 0.28 } ] }"
    else:  # seated
        body_children = _render_articulated_pieces(
            _seated_pieces(shirt_rgb, pants_rgb), shirt_rgb, pants_rgb,
            shirt_def=shirt_def, pants_def=pants_def,
        )
        bbox_pose = "Pose { translation 0.14 0 0.42 children [ Box { size 0.78 0.52 0.84 } ] }"

    return (
        f'DEF {def_name} Solid {{\n'
        f'  translation {numbers(trans_xyz)}\n'
        f'  rotation {numbers(rot_aa)}\n'
        f'  name "{name}"\n'
        f'  children [\n{body_children}\n  ]\n'
        f'  boundingObject {bbox_pose}\n'
        f'}}\n'
    )
