"""Supervisor camera-rig controller for color-invariant & position-randomized YOLO aerial dataset collection.

Runs ONLY on offline training/validation worlds (Scenes A, B, C; NEVER Scene D).
Dynamically randomizes casualty positions (terrain-slope-aligned), yaws, and 16-color
shirt / 8-color pants PBR appearances, then projects 3D casualty geometry with RangeFinder
depth occlusion verification to produce tight YOLO bounding-box labels.
"""
from pathlib import Path
import json
import math
import os
import random
import sys

import cv2
import numpy as np
from controller import Supervisor
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "tools"))
from metrics import height_at, look_at_rotation
from posture_casualty import POSTURE_LOCAL_POINTS


BODY_LOCAL_POINTS = np.array(
    [
        [x, y, z]
        for x in (-0.68, -0.45, -0.20, 0.05, 0.35, 0.62, 0.86)
        for y in (-0.25, 0.0, 0.25)
        for z in (0.08, 0.22)
    ]
    + [
        [-0.62, 0.0, 0.25],
        [-0.30, -0.32, 0.12],
        [-0.30, 0.32, 0.12],
        [0.05, -0.32, 0.10],
        [0.05, 0.32, 0.10],
        [0.88, -0.14, 0.10],
        [0.88, 0.14, 0.10],
    ],
    dtype=float,
)

# Diverse 16-color shirt palette (bright rescue colors, dark colors, and low-contrast earth/gray tones).
SHIRT_COLORS = [
    (0.76, 0.14, 0.14),  # crimson red
    (0.90, 0.42, 0.08),  # safety orange
    (0.84, 0.72, 0.16),  # mustard yellow
    (0.16, 0.42, 0.20),  # forest green
    (0.42, 0.44, 0.26),  # olive / khaki
    (0.84, 0.84, 0.82),  # off-white / light gray
    (0.24, 0.25, 0.27),  # charcoal gray
    (0.16, 0.25, 0.42),  # dark navy
    (0.14, 0.38, 0.78),  # royal blue
    (0.42, 0.66, 0.86),  # sky blue
    (0.46, 0.14, 0.22),  # burgundy / maroon
    (0.46, 0.30, 0.20),  # warm brown
    (0.14, 0.50, 0.52),  # teal
    (0.44, 0.24, 0.54),  # muted purple
    (0.66, 0.58, 0.44),  # tan / beige
    (0.82, 0.42, 0.44),  # coral / dusty pink
]

PANTS_COLORS = [
    (0.16, 0.18, 0.22),  # dark slate
    (0.18, 0.26, 0.38),  # denim blue
    (0.46, 0.42, 0.32),  # khaki
    (0.28, 0.32, 0.22),  # olive drab
    (0.32, 0.24, 0.18),  # dark brown
    (0.54, 0.54, 0.54),  # medium gray
    (0.10, 0.10, 0.11),  # black
    (0.22, 0.22, 0.30),  # dark navy
]


def _node_pose(node):
    pos = np.asarray(node.getPosition(), dtype=float)
    rot = np.asarray(node.getOrientation(), dtype=float).reshape(3, 3)
    return pos, rot


def _ground_aligned_axis_angle(meta, x, y, yaw):
    """Compute Webots axis-angle aligning casualty supine frame to local terrain normal."""
    delta = 0.08
    dx = (height_at(meta, x + delta, y) - height_at(meta, x - delta, y)) / (2.0 * delta)
    dy = (height_at(meta, x, y + delta) - height_at(meta, x, y - delta)) / (2.0 * delta)
    up = np.array([-dx, -dy, 1.0], dtype=float)
    up /= np.linalg.norm(up)
    fwd = np.array([math.cos(yaw), math.sin(yaw), dx * math.cos(yaw) + dy * math.sin(yaw)], dtype=float)
    fwd /= np.linalg.norm(fwd)
    left = np.cross(up, fwd)
    left /= np.linalg.norm(left)
    mat = np.column_stack((fwd, left, up))
    rv = Rotation.from_matrix(mat).as_rotvec()
    ang = float(np.linalg.norm(rv))
    if ang < 1e-8:
        return [0.0, 0.0, 1.0, 0.0]
    return [*(rv / ang).tolist(), ang]


def _project_casualties(cam_pos, cam_rot, width, height, fov, depth_img, victims, min_vis_ratio=0.20):
    """Project all casualties (each with its posture-specific 3D points) to tight YOLO boxes."""
    focal = width / (2.0 * math.tan(fov / 2.0))
    boxes = []
    ambiguous = False

    for entry in victims:
        if len(entry) == 3:
            v_pos, v_rot, posture = entry
        else:
            v_pos, v_rot = entry
            posture = "supine"
        local_pts = POSTURE_LOCAL_POINTS.get(posture, BODY_LOCAL_POINTS)
        pts_world = local_pts @ v_rot.T + v_pos
        pts_cam = (pts_world - cam_pos) @ cam_rot
        in_front = pts_cam[:, 0] > 0.35
        if np.count_nonzero(in_front) < 8:
            continue
        pc = pts_cam[in_front]
        u = (width - 1) * 0.5 - focal * (pc[:, 1] / pc[:, 0])
        v = (height - 1) * 0.5 - focal * (pc[:, 2] / pc[:, 0])
        in_img = (u >= 1.0) & (u < width - 1.0) & (v >= 1.0) & (v < height - 1.0)
        n_in = int(np.count_nonzero(in_img))
        if n_in < 6:
            continue

        ui = np.clip(np.round(u[in_img]).astype(int), 0, width - 1)
        vi = np.clip(np.round(v[in_img]).astype(int), 0, height - 1)
        exp_d = pc[in_img, 0]
        meas_d = depth_img[vi, ui]
        tol = np.maximum(0.65, 0.055 * exp_d)
        visible = np.isfinite(meas_d) & (meas_d >= exp_d - tol) & (meas_d <= exp_d + tol * 1.4)
        vis_ratio = float(np.count_nonzero(visible)) / float(n_in)

        if vis_ratio < min_vis_ratio:
            ambiguous = True
            continue

        uv_use_u = u[in_img][visible]
        uv_use_v = v[in_img][visible]
        x1, x2 = float(np.min(uv_use_u)), float(np.max(uv_use_u))
        y1, y2 = float(np.min(uv_use_v)), float(np.max(uv_use_v))
        bw, bh = x2 - x1, y2 - y1
        pad_x = max(2.5, 0.12 * bw)
        pad_y = max(2.5, 0.12 * bh)
        x1 = max(0.0, x1 - pad_x)
        y1 = max(0.0, y1 - pad_y)
        x2 = min(float(width - 1), x2 + pad_x)
        y2 = min(float(height - 1), y2 + pad_y)
        if (x2 - x1) < 7.0 or (y2 - y1) < 7.0:
            ambiguous = True
            continue

        cx = 0.5 * (x1 + x2) / width
        cy = 0.5 * (y1 + y2) / height
        nw = (x2 - x1) / width
        nh = (y2 - y1) / height
        boxes.append((0, cx, cy, nw, nh))

    return boxes, ambiguous


def _set_pose(trans_field, rot_field, pos, rot_axis_angle):
    trans_field.setSFVec3f([float(pos[0]), float(pos[1]), float(pos[2])])
    rot_field.setSFRotation([float(v) for v in rot_axis_angle])


def _nadir_oblique_rot(yaw, pitch_from_nadir_rad, roll_rad=0.0):
    mat = Rotation.from_euler("ZYX", [yaw, math.pi / 2.0 - pitch_from_nadir_rad, roll_rad]).as_matrix()
    rv = Rotation.from_matrix(mat).as_rotvec()
    ang = float(np.linalg.norm(rv))
    if ang < 1e-8:
        return [0.0, 0.0, 1.0, 0.0]
    return [*(rv / ang).tolist(), ang]


def main():
    spec_path = Path(os.environ["YOLO_COLLECT_SPEC"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    out_images = Path(spec["out_images"])
    out_labels = Path(spec["out_labels"])
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    meta = json.loads((ROOT / "evaluation/scenarios" / f"{spec['scene']}.json").read_text(encoding="utf-8"))
    rng = random.Random(int(spec["seed"]))
    prefix = spec["prefix"]

    sim = Supervisor()
    step_ms = 60
    self_node = sim.getSelf()
    trans_field = self_node.getField("translation")
    rot_field = self_node.getField("rotation")

    uav_rgb = sim.getDevice("uav_rgb")
    uav_depth = sim.getDevice("uav_depth")
    rover_rgb = sim.getDevice("rover_rgb")
    rover_depth = sim.getDevice("rover_depth")
    for dev in (uav_rgb, uav_depth, rover_rgb, rover_depth):
        dev.enable(step_ms)

    uav_w, uav_h, uav_fov = uav_rgb.getWidth(), uav_rgb.getHeight(), uav_rgb.getFov()
    rov_w, rov_h, rov_fov = rover_rgb.getWidth(), rover_rgb.getHeight(), rover_rgb.getFov()

    for _ in range(3):
        sim.step(step_ms)

    raw_defs = spec["victim_defs"]
    raw_postures = spec.get("victim_postures", ["supine"] * len(raw_defs))
    victim_entries = []
    for d, p_name in zip(raw_defs, raw_postures):
        n = sim.getFromDef(d)
        if n is not None:
            victim_entries.append((n, p_name))
    victim_nodes = [n for n, _ in victim_entries]
    shirt_apps = [sim.getFromDef(d) for d in spec.get("shirt_app_defs", [])]
    shirt_apps = [n for n in shirt_apps if n is not None]
    pants_apps = [sim.getFromDef(d) for d in spec.get("pants_app_defs", [])]
    pants_apps = [n for n in pants_apps if n is not None]
    print(
        f"[YOLO Collector] {prefix}: {len(victim_nodes)} victims ({[p for _, p in victim_entries]}), "
        f"{len(shirt_apps)} shirt apps, {len(pants_apps)} pants apps.",
        flush=True,
    )

    def randomize_victims():
        """Relocate every casualty to a fresh terrain-aligned random spot + random clothing colors."""
        placed = []
        for node in victim_nodes:
            for _ in range(25):
                vx = rng.uniform(-17.5, 17.5)
                vy = rng.uniform(-11.5, 11.5)
                if all(math.hypot(vx - px, vy - py) >= 3.2 for px, py in placed):
                    break
            placed.append((vx, vy))
            vz = float(height_at(meta, vx, vy))
            vyaw = rng.uniform(-math.pi, math.pi)
            rot_aa = _ground_aligned_axis_angle(meta, vx, vy, vyaw)
            tf = node.getField("translation")
            rf = node.getField("rotation")
            if tf is not None:
                tf.setSFVec3f([vx, vy, vz])
            if rf is not None:
                rf.setSFRotation(rot_aa)
        for app in shirt_apps:
            cf = app.getField("baseColor")
            if cf is not None:
                base = rng.choice(SHIRT_COLORS)
                jittered = [float(np.clip(c + rng.uniform(-0.06, 0.06), 0.05, 0.95)) for c in base]
                cf.setSFColor(jittered)
        for app in pants_apps:
            cf = app.getField("baseColor")
            if cf is not None:
                base = rng.choice(PANTS_COLORS)
                jittered = [float(np.clip(c + rng.uniform(-0.05, 0.05), 0.04, 0.90)) for c in base]
                cf.setSFColor(jittered)
        for _ in range(2):
            sim.step(step_ms)
        return [(*_node_pose(n), p_name) for n, p_name in victim_entries]

    victims = randomize_victims()
    counts = dict(uav_pos=0, uav_neg=0, rover_pos=0, rover_neg=0, skipped=0, refreshes=1)
    saved_idx = 0

    def capture_sample(mode, eye, rot_aa, expect_positive):
        nonlocal saved_idx
        _set_pose(trans_field, rot_field, eye, rot_aa)
        for _ in range(2):
            if sim.step(step_ms) == -1:
                return False
        cam_pos, cam_rot = _node_pose(self_node)
        if mode == "uav":
            rgb_dev, dep_dev, w, h, fov = uav_rgb, uav_depth, uav_w, uav_h, uav_fov
        else:
            rgb_dev, dep_dev, w, h, fov = rover_rgb, rover_depth, rov_w, rov_h, rov_fov

        raw = rgb_dev.getImage()
        if raw is None:
            return False
        bgr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
        depth = np.asarray(dep_dev.getRangeImage(), dtype=np.float32).reshape(h, w).copy()

        boxes, ambiguous = _project_casualties(cam_pos, cam_rot, w, h, fov, depth, victims)
        if expect_positive and (not boxes or ambiguous):
            counts["skipped"] += 1
            return False
        if (not expect_positive) and (boxes or ambiguous):
            counts["skipped"] += 1
            return False

        stem = f"{prefix}_{mode}_{'pos' if expect_positive else 'neg'}_{saved_idx:04d}"
        cv2.imwrite(str(out_images / f"{stem}.jpg"), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        label_lines = [f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cls, cx, cy, bw, bh in boxes]
        (out_labels / f"{stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")
        saved_idx += 1
        counts[f"{mode}_{'pos' if expect_positive else 'neg'}"] += 1
        return True

    # 1. UAV positive views across randomized victim positions, 16 shirt colors, altitudes 10.5..22 m.
    target_uav_pos = int(spec["uav_pos"])
    refresh_every = max(3, int(spec.get("refresh_every", 4)))
    attempts = 0
    since_refresh = 0
    while counts["uav_pos"] < target_uav_pos and attempts < target_uav_pos * 5:
        attempts += 1
        if since_refresh >= refresh_every:
            victims = randomize_victims()
            counts["refreshes"] += 1
            since_refresh = 0
        v_pos = rng.choice(victims)[0]
        alt = rng.uniform(10.5, 22.0)
        dx = rng.uniform(-6.0, 6.0)
        dy = rng.uniform(-4.4, 4.4)
        pitch_ob = rng.uniform(0.0, 0.25) if rng.random() < 0.35 else 0.0
        yaw = rng.uniform(-math.pi, math.pi)
        roll = rng.uniform(-0.04, 0.04)
        ex, ey = v_pos[0] + dx, v_pos[1] + dy
        gz = float(height_at(meta, ex, ey))
        ez = max(gz + alt, v_pos[2] + alt)
        if pitch_ob > 0.0:
            tx = v_pos[0] + rng.uniform(-2.5, 2.5)
            ty = v_pos[1] + rng.uniform(-2.5, 2.5)
            rot_aa = look_at_rotation([ex, ey, ez], [tx, ty, v_pos[2]])
        else:
            rot_aa = _nadir_oblique_rot(yaw, 0.0, roll)
        if capture_sample("uav", [ex, ey, ez], rot_aa, expect_positive=True):
            since_refresh += 1

    # 2. Optional Rover positive views (0 when focusing pure aerial specialist).
    target_rov_pos = int(spec.get("rover_pos", 0))
    attempts = 0
    while counts["rover_pos"] < target_rov_pos and attempts < target_rov_pos * 5:
        attempts += 1
        v_pos = rng.choice(victims)[0]
        dist = rng.uniform(1.05, 4.4)
        az = rng.uniform(-math.pi, math.pi)
        ex = float(np.clip(v_pos[0] + dist * math.cos(az), -19.0, 19.0))
        ey = float(np.clip(v_pos[1] + dist * math.sin(az), -13.0, 13.0))
        ez = float(height_at(meta, ex, ey)) + rng.uniform(0.28, 0.42)
        tx = v_pos[0] + rng.uniform(-0.35, 0.35)
        ty = v_pos[1] + rng.uniform(-0.35, 0.35)
        tz = v_pos[2] + rng.uniform(0.08, 0.25)
        rot_aa = look_at_rotation([ex, ey, ez], [tx, ty, tz])
        capture_sample("rover", [ex, ey, ez], rot_aa, expect_positive=True)

    # Move casualties out of the scene to capture pure hard-negative background views.
    for node in victim_nodes:
        tf = node.getField("translation")
        if tf is not None:
            tf.setSFVec3f([0.0, 0.0, -60.0])
    victims = []
    for _ in range(2):
        sim.step(step_ms)

    # 3. UAV hard-negative background views.
    target_uav_neg = int(spec["uav_neg"])
    attempts = 0
    while counts["uav_neg"] < target_uav_neg and attempts < target_uav_neg * 3:
        attempts += 1
        ex = rng.uniform(-16.5, 16.5)
        ey = rng.uniform(-11.5, 11.5)
        alt = rng.uniform(11.0, 22.0)
        ez = float(height_at(meta, ex, ey)) + alt
        rot_aa = _nadir_oblique_rot(rng.uniform(-math.pi, math.pi), rng.uniform(0.0, 0.22))
        capture_sample("uav", [ex, ey, ez], rot_aa, expect_positive=False)

    # 4. Optional Rover hard-negative background views.
    target_rov_neg = int(spec.get("rover_neg", 0))
    attempts = 0
    while counts["rover_neg"] < target_rov_neg and attempts < target_rov_neg * 3:
        attempts += 1
        ex = rng.uniform(-16.0, 16.0)
        ey = rng.uniform(-11.0, 11.0)
        ez = float(height_at(meta, ex, ey)) + rng.uniform(0.30, 0.40)
        yaw = rng.uniform(-math.pi, math.pi)
        tx = ex + 4.0 * math.cos(yaw)
        ty = ey + 4.0 * math.sin(yaw)
        tz = float(height_at(meta, tx, ty)) + rng.uniform(0.0, 0.4)
        rot_aa = look_at_rotation([ex, ey, ez], [tx, ty, tz])
        capture_sample("rover", [ex, ey, ez], rot_aa, expect_positive=False)

    print(f"[YOLO Collector] {prefix} done: {counts}", flush=True)
    Path(spec["report_path"]).write_text(json.dumps(counts, indent=2), encoding="utf-8")
    sim.simulationQuit(0)


if __name__ == "__main__":
    main()
