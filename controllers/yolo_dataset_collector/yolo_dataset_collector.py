"""Supervisor camera-rig controller for zero-leakage YOLO victim dataset collection.

Runs ONLY on offline training/validation worlds (Scenes A, B, C; NEVER Scene D).
Uses exact 3D-to-2D projection of casualty geometry verified against the camera's
own RangeFinder depth map to produce tight, occlusion-checked YOLO labels.
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
from metrics import height_at, look_at_rotation


# Local 3D sample points on a supine casualty (from tools/scene_elements.py):
# Head around x=-0.62, shoulders x=-0.41, hips x=+0.13, knees x=+0.43, boots x=+0.84;
# Width y in [-0.28, +0.28], height z in [0.06, 0.24].
BODY_LOCAL_POINTS = np.array(
    [
        [x, y, z]
        for x in (-0.68, -0.45, -0.20, 0.05, 0.35, 0.62, 0.86)
        for y in (-0.25, 0.0, 0.25)
        for z in (0.08, 0.22)
    ]
    + [
        [-0.62, 0.0, 0.25],  # face/head top
        [-0.30, -0.32, 0.12],  # arm R
        [-0.30, 0.32, 0.12],  # arm L
        [0.05, -0.32, 0.10],  # hand R
        [0.05, 0.32, 0.10],  # hand L
        [0.88, -0.14, 0.10],  # boot R
        [0.88, 0.14, 0.10],  # boot L
    ],
    dtype=float,
)


def _node_pose(node):
    pos = np.asarray(node.getPosition(), dtype=float)
    rot = np.asarray(node.getOrientation(), dtype=float).reshape(3, 3)
    return pos, rot


def _project_casualties(cam_pos, cam_rot, width, height, fov, depth_img, victims, min_vis_ratio=0.22):
    """Project all casualties to 2D YOLO boxes and verify unoccluded visibility via depth."""
    focal = width / (2.0 * math.tan(fov / 2.0))
    boxes = []
    ambiguous = False

    for v_pos, v_rot in victims:
        pts_world = BODY_LOCAL_POINTS @ v_rot.T + v_pos
        pts_cam = (pts_world - cam_pos) @ cam_rot  # R_cam.T @ (p - t)
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
        # Visible if measured depth is close to expected body depth (not blocked by a tree canopy in front).
        tol = np.maximum(0.65, 0.055 * exp_d)
        visible = np.isfinite(meas_d) & (meas_d >= exp_d - tol) & (meas_d <= exp_d + tol * 1.4)
        vis_ratio = float(np.count_nonzero(visible)) / float(n_in)

        if vis_ratio < min_vis_ratio:
            # Partially/fully occluded victim in frame -> mark frame ambiguous if negative or unclear.
            ambiguous = True
            continue

        # Tight bounding box around visible points (expanded gently to cover limbs/outline).
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
    """Camera +X points downward (pitch = +pi/2 - pitch_from_nadir_rad)."""
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

    # Warm up two steps so Webots initializes all transforms & depth buffers.
    for _ in range(3):
        sim.step(step_ms)

    victim_nodes = []
    for def_name in spec["victim_defs"]:
        node = sim.getFromDef(def_name)
        if node is not None:
            victim_nodes.append(node)
    victims = [_node_pose(n) for n in victim_nodes]
    print(f"[YOLO Collector] {prefix}: found {len(victims)} casualty node(s).", flush=True)

    counts = dict(uav_pos=0, uav_neg=0, rover_pos=0, rover_neg=0, skipped=0)
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

    # 1. UAV positive views across varied altitudes (11..21.5 m), yaws, offsets, slight obliques.
    target_uav_pos = int(spec["uav_pos"])
    attempts = 0
    while counts["uav_pos"] < target_uav_pos and attempts < target_uav_pos * 4:
        attempts += 1
        v_pos, _ = rng.choice(victims)
        alt = rng.uniform(11.0, 21.5)
        dx = rng.uniform(-5.8, 5.8)
        dy = rng.uniform(-4.2, 4.2)
        pitch_ob = rng.uniform(0.0, 0.24) if rng.random() < 0.35 else 0.0
        yaw = rng.uniform(-math.pi, math.pi)
        roll = rng.uniform(-0.04, 0.04)
        ex, ey = v_pos[0] + dx, v_pos[1] + dy
        gz = float(height_at(meta, ex, ey))
        ez = max(gz + alt, v_pos[2] + alt)
        if pitch_ob > 0.0:
            # Aim slightly toward victim with random jitter for oblique frames.
            tx = v_pos[0] + rng.uniform(-2.5, 2.5)
            ty = v_pos[1] + rng.uniform(-2.5, 2.5)
            rot_aa = look_at_rotation([ex, ey, ez], [tx, ty, v_pos[2]])
        else:
            rot_aa = _nadir_oblique_rot(yaw, 0.0, roll)
        capture_sample("uav", [ex, ey, ez], rot_aa, expect_positive=True)

    # 2. Rover ground-level positive views (distances 1.0..4.5 m, full 360° around casualties).
    target_rov_pos = int(spec["rover_pos"])
    attempts = 0
    while counts["rover_pos"] < target_rov_pos and attempts < target_rov_pos * 5:
        attempts += 1
        v_pos, _ = rng.choice(victims)
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

    # Move casualties out of the scene to capture pure hard-negative background views
    # (collapsed houses, fallen logs, boulders, tree shadows, bushes, gullies).
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
        ex = rng.uniform(-16.0, 16.0)
        ey = rng.uniform(-11.0, 11.0)
        alt = rng.uniform(12.0, 21.5)
        ez = float(height_at(meta, ex, ey)) + alt
        rot_aa = _nadir_oblique_rot(rng.uniform(-math.pi, math.pi), rng.uniform(0.0, 0.20))
        capture_sample("uav", [ex, ey, ez], rot_aa, expect_positive=False)

    # 4. Rover ground-level hard-negative views.
    target_rov_neg = int(spec["rover_neg"])
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
