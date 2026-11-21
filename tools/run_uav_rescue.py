"""Run air-to-ground victim detection, safe approach planning, and supply delivery in Webots D."""
from pathlib import Path
import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aerial_products import export_products
from build_search_worlds import replace_node
from metrics import height_at
from run_search import ROOT, executable_path, simulator_has_errors
from run_uav_mapping import PROFILES, refinement_metrics, survey_metrics
from run_uav_navigation import OBSTACLE_PRESETS, build_nav_world
from scene_elements import RescueElements
from uav_full_preview import draw_aerial_products, draw_full_preview, draw_refinement_preview
from uav_map_preview import audit_alignment, draw_preview, load_prior

sys.path.insert(0, str(ROOT / "controllers/uav_mapper"))
from aerial_victims import detect_victims_from_saved_survey
from survey_map import BLOCKED, FREE, UNKNOWN


RANDOM_SHIRT_COLORS = [
    ("crimson", ".76 .14 .14"),
    ("orange", ".88 .42 .08"),
    ("olive", ".36 .42 .22"),
    ("mustard", ".82 .68 .16"),
    ("light_gray", ".82 .82 .80"),
    ("charcoal", ".24 .25 .27"),
    ("teal", ".14 .48 .50"),
    ("brown", ".46 .28 .18"),
    ("burgundy", ".48 .14 .22"),
    ("sky_blue", ".38 .62 .84"),
]

RANDOM_PANTS_COLORS = [
    ".18 .20 .24",
    ".44 .40 .30",
    ".26 .30 .22",
    ".30 .22 .16",
    ".12 .12 .13",
    ".48 .48 .50",
]


def _terrain_slope_deg(terrain, x, y, delta=0.45):
    dx = (height_at(terrain, x + delta, y) - height_at(terrain, x - delta, y)) / (2.0 * delta)
    dy = (height_at(terrain, x, y + delta) - height_at(terrain, x, y - delta)) / (2.0 * delta)
    return math.degrees(math.atan(math.hypot(dx, dy)))


def sample_random_victims_d(config, seed, count=3, random_clothing=False):
    """Sample valid, non-overlapping random victim poses inside the survey bounds (offline world-gen only)."""
    terrain = json.loads((ROOT / "evaluation/search/D.json").read_text(encoding="utf-8"))
    base_b = json.loads((ROOT / "evaluation/scenarios/B.json").read_text(encoding="utf-8"))
    rng = random.Random(int(seed))

    (bx0, by0), (bx1, by1) = config["bounds_enu_m"]
    margin = 2.5 if (bx1 - bx0) <= 25 else 6.0
    x_lo, x_hi = bx0 + margin, bx1 - margin
    y_lo, y_hi = by0 + margin, by1 - margin
    deploy_xy = tuple(config["deployment_xy_enu_m"])

    trees = terrain.get("forest", {}).get("trees", [])
    bushes = terrain.get("forest", {}).get("bushes", [])
    obstacles = base_b.get("obstacles", [])

    victims = []
    min_pair = 4.0 if (bx1 - bx0) <= 25 else 7.5
    for idx in range(1, count + 1):
        chosen = None
        for _ in range(1200):
            x = rng.uniform(x_lo, x_hi)
            y = rng.uniform(y_lo, y_hi)
            if math.hypot(x - deploy_xy[0], y - deploy_xy[1]) < 4.5:
                continue
            if any(math.hypot(x - v["xy"][0], y - v["xy"][1]) < min_pair for v in victims):
                continue
            if _terrain_slope_deg(terrain, x, y) > 14.5:
                continue
            # Avoid clipping into trees, bushes, collapsed houses, or static boulders.
            if any(math.hypot(x - t["xy"][0], y - t["xy"][1]) < 2.6 for t in trees):
                continue
            if any(math.hypot(x - b["xy"][0], y - b["xy"][1]) < 1.7 for b in bushes):
                continue
            if any(math.hypot(x - o["xy"][0], y - o["xy"][1]) < (max(o["size"][:2]) * 0.65 + 1.5) for o in obstacles):
                continue
            chosen = (round(x, 2), round(y, 2))
            break
        if chosen is None:
            raise RuntimeError(f"Failed to sample valid random victim #{idx} for seed={seed}")
        color_name, shirt_rgb = rng.choice(RANDOM_SHIRT_COLORS) if random_clothing else ("default_blue", None)
        pants_rgb = rng.choice(RANDOM_PANTS_COLORS) if random_clothing else None
        victims.append(dict(
            id=idx,
            xy=list(chosen),
            yaw_rad=round(rng.uniform(-math.pi, math.pi), 4),
            shirt_name=color_name,
            shirt_rgb=shirt_rgb,
            pants_rgb=pants_rgb,
        ))
    return victims, terrain


def apply_random_victims_to_world(world_path, victims, terrain):
    """Replace the 3 victim Solid nodes in the temporary validation world with randomized placements."""
    scene = world_path.read_text(encoding="utf-8")
    # Remove the 3 default victim nodes from the temporary world.
    for pat in (
        r"\bDEF\s+CASUALTY\s+Solid",
        r"\bDEF\s+SEARCH_VICTIM_2\s+Solid",
        r"\bDEF\s+SEARCH_VICTIM_3\s+Solid",
    ):
        if re.search(pat, scene):
            scene = replace_node(scene, pat, "")

    for idx, v in enumerate(victims, 1):
        vkey = ("A", "B", "C")[(idx - 1) % 3]
        el = RescueElements(dict(key=vkey, seed=900 + idx * 31, goal=v["xy"]),
                            lambda x, y: height_at(terrain, x, y))
        el.casualty()
        node = el.nodes[0]
        def_name = "CASUALTY" if idx == 1 else f"SEARCH_VICTIM_{idx}"
        node = node.replace("DEF CASUALTY ", f"DEF {def_name} ", 1)
        node = node.replace('name "casualty"', f'name "search victim {idx}"', 1)
        # Apply randomized yaw via ground_frame rotation directly on the outer Solid node.
        from scene_elements import axis_angle, ground_frame
        rot = axis_angle(ground_frame(v["xy"][0], v["xy"][1], v["yaw_rad"],
                                      lambda x, y: height_at(terrain, x, y)))
        rot_str = " ".join(f"{val:.6f}" for val in rot)
        node = re.sub(r"rotation\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+",
                      f"rotation {rot_str}", node, count=1)
        if v.get("shirt_rgb"):
            lines = []
            for line in node.splitlines():
                if "_shirt.obj" in line:
                    line = re.sub(r"baseColor\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+",
                                  f"baseColor {v['shirt_rgb']}", line, count=1)
                elif "_pants.obj" in line and v.get("pants_rgb"):
                    line = re.sub(r"baseColor\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+",
                                  f"baseColor {v['pants_rgb']}", line, count=1)
                lines.append(line)
            node = "\n".join(lines)
        scene += "\n" + node + "\n"
    world_path.write_text(scene, encoding="utf-8")


def draw_rescue_preview(folder, eval_summary, rover_status, victims_doc):
    """Render UAV map, aerial victim candidates, safe approach routes, truth trajectory, and drops."""
    grid, states = load_prior(folder / "uav_map.npz")
    palette = {UNKNOWN: (84, 94, 107), FREE: (139, 178, 144), BLOCKED: (217, 104, 74)}
    cells = np.zeros((*states.shape, 3), dtype=np.uint8)
    for state, color in palette.items():
        cells[states == state] = color
    for item in rover_status.get("onboard_blocked_cells", []):
        r, c = item["row"], item["col"]
        if 0 <= r < states.shape[0] and 0 <= c < states.shape[1]:
            cells[r, c] = (232, 76, 152)

    w = 780
    h = round(w * states.shape[0] / states.shape[1])
    margin, top, bottom = 28, 94, 148
    canvas = Image.new("RGB", (w + 2 * margin, h + top + bottom), "#111923")
    draw = ImageDraw.Draw(canvas)
    title_f = ImageFont.load_default(size=20)
    body_f = ImageFont.load_default(size=15)
    small_f = ImageFont.load_default(size=13)

    map_img = Image.fromarray(np.flipud(cells)).resize((w, h), Image.Resampling.NEAREST)
    canvas.paste(map_img, (margin, top))
    draw.rectangle((margin, top, margin + w, top + h), outline="#495b6a", width=2)

    extent = np.asarray(grid.shape[::-1]) * grid.resolution

    def to_px(xy):
        east, north = (np.asarray(xy, dtype=float) - grid.origin) / extent
        return float(margin + w * east), float(top + h * (1.0 - north))

    # Draw injected obstacles if any (evaluation-only overlay).
    for obs in eval_summary.get("injected_obstacles", []):
        cx, cy = obs["xy_enu_m"]
        sx, sy = obs["size_m"][:2]
        yaw = obs.get("yaw_rad", 0.0)
        c, s = math.cos(yaw), math.sin(yaw)
        corners = [
            to_px((cx + c * dx - s * dy, cy + s * dx + c * dy))
            for dx, dy in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))
        ]
        draw.polygon(corners, fill="#d93838", outline="#ffe28a")

    # Draw planned / replanned multi-leg routes.
    route_colors = ("#50ddfa", "#ffd166", "#c77dff")
    for idx, r_pts in enumerate(rover_status.get("routes_history", [])):
        if len(r_pts) < 2:
            continue
        poly = [to_px(p) for p in r_pts]
        color = route_colors[idx % len(route_colors)]
        draw.line(poly, fill="#0e2230", width=7, joint="curve")
        draw.line(poly, fill=color, width=4, joint="curve")

    # Draw Webots ground-truth rover trajectory.
    traj_file = folder / "evaluation/trajectory.csv"
    truth_pts = []
    if traj_file.is_file():
        with traj_file.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                truth_pts.append((float(row["x"]), float(row["y"])))
    if len(truth_pts) > 1:
        poly = [to_px(p) for p in truth_pts]
        draw.line(poly, fill="#0b151e", width=7, joint="curve")
        draw.line(poly, fill="#ffffff", width=3, joint="curve")

    # Draw evaluation-only ground-truth victim locations as subtle red crosshairs.
    override_file = folder / "evaluation/truth_override.json"
    truth_file = override_file if override_file.is_file() else ROOT / "evaluation/search/D.json"
    if truth_file.is_file():
        truth = json.loads(truth_file.read_text(encoding="utf-8"))
        for v in truth.get("victims", []):
            px, py = to_px(v["xy"])
            draw.line((px - 10, py, px + 10, py), fill="#ff4d6d", width=2)
            draw.line((px, py - 10, px, py + 10), fill="#ff4d6d", width=2)

    # Draw sensor-detected aerial victim candidates.
    for cand in victims_doc.get("candidates", []):
        px, py = to_px(cand["mean_enu_m"])
        draw.ellipse((px - 12, py - 12, px + 12, py + 12), fill="#331d07", outline="#ff9f1c", width=3)
        draw.text((px, py), f"V{cand['id']}", font=small_f, fill="#ffcf70", anchor="mm")

    # Draw chosen safe approach points and actual delivery locations.
    for leg in rover_status.get("legs", []):
        if "approach_xy_m" in leg:
            ax, ay = to_px(leg["approach_xy_m"])
            draw.rectangle((ax - 7, ay - 7, ax + 7, ay + 7), fill="#102a38", outline="#50ddfa", width=2)
    for drop in eval_summary.get("deliveries", []):
        dx, dy = to_px(drop["actual_drop_xy"])
        col = "#2ec4b6" if drop["valid"] else "#e63946"
        draw.ellipse((dx - 9, dy - 9, dx + 9, dy + 9), fill=col, outline="#ffffff", width=2)

    # Start marker.
    route_doc = json.loads((folder / "route.json").read_text(encoding="utf-8"))
    sx, sy = to_px(route_doc["start_xy_m"])
    draw.ellipse((sx - 11, sy - 11, sx + 11, sy + 11), fill="#142230", outline="#f1f7f9", width=2)
    draw.text((sx, sy), "S", font=small_f, fill="#f1f7f9", anchor="mm")

    status = eval_summary["status"]
    planner = eval_summary.get("planner", "igha_vctp")
    detector = eval_summary.get("detector", "surrogate_rgbd")
    draw.text(
        (margin, 14),
        f"D  /  Air-to-Ground Rescue [{detector} | {planner}]  ->  {status}",
        font=title_f, fill="#edf4f8",
    )
    rmse = eval_summary.get("localization_rmse_m")
    rmse_s = f"{rmse:.2f} m" if rmse is not None else "n/a"
    subtitle = (
        f"reports={eval_summary['effective_reports']}/{eval_summary['truth_targets']} "
        f"(cross_view={eval_summary['valid_cross_view_reports']}, false={eval_summary['false_reports']})  |  "
        f"delivered={eval_summary['delivered_people']}/{eval_summary['cargo_capacity']}  |  "
        f"col={eval_summary['collisions']}  |  RMSE={rmse_s}"
    )
    draw.text((margin, 54), subtitle, font=body_f, fill="#b8c9d7")

    y_leg = top + h + 16
    entries = [
        (palette[FREE], "Free"),
        (palette[BLOCKED], "Hazard"),
        (palette[UNKNOWN], "Unknown"),
        ("#ff9f1c", "Detected victim"),
        ("#ff4d6d", "Truth cross"),
        ("#50ddfa", "Planned leg"),
        ("#ffffff", "Truth traj"),
        ("#2ec4b6", "Valid drop"),
    ]
    x_cur = margin
    for color, label in entries:
        draw.rectangle((x_cur, y_leg, x_cur + 13, y_leg + 13), fill=color)
        draw.text((x_cur + 18, y_leg - 2), label, font=small_f, fill="#d1dee8")
        x_cur += 22 + round(draw.textlength(label, font=small_f))

    drop_summary = ", ".join(
        f"V{d['victim_id']}:{d['true_distance_m']:.2f}m({'OK' if d['valid'] else 'BAD'})"
        for d in eval_summary.get("deliveries", [])
    ) or "none"
    draw.text(
        (margin, y_leg + 30),
        f"Deliveries: {drop_summary}  |  Duration: {eval_summary['total_rescue_duration_s']:.1f} s  |  "
        f"Path: {eval_summary['actual_path_length_m']:.2f} m",
        font=small_f,
        fill="#a3f7bf" if eval_summary["mission_success"] else "#ffd166",
    )
    reason_txt = (eval_summary.get("failure_reason") or "All cargo delivered safely")[:56]
    draw.text(
        (margin, y_leg + 58),
        f"Aerial candidates: {eval_summary['aerial_candidates_total']}  |  "
        f"Replans: {eval_summary['replan_count']}  |  "
        f"Max roll/pitch: {eval_summary['max_abs_roll_deg']:.1f}/{eval_summary['max_abs_pitch_deg']:.1f} deg  |  "
        f"{reason_txt}",
        font=small_f, fill="#b8c9d7",
    )
    canvas.save(folder / "evaluation/rescue_preview.png")


def assemble_rescue_video(folder, rover_status, victims_doc, eval_summary, video_speed=4.0):
    """Compose follow-camera frames + live UAV mini-map PiP into an H.264 MP4 video."""
    frames_dir = folder / "evaluation/video_frames"
    meta_file = folder / "evaluation/video_meta.jsonl"
    if not frames_dir.is_dir() or not meta_file.is_file():
        return None
    meta_rows = [json.loads(line) for line in meta_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not meta_rows:
        return None

    # Build static UAV mini-map base (BGR).
    grid, states = load_prior(folder / "uav_map.npz")
    palette = {UNKNOWN: (107, 94, 84), FREE: (144, 178, 139), BLOCKED: (74, 104, 217)}  # BGR
    cells = np.zeros((*states.shape, 3), dtype=np.uint8)
    for st, bgr in palette.items():
        cells[states == st] = bgr
    pip_w = 270
    pip_h = round(pip_w * states.shape[0] / states.shape[1])
    base_pip = cv2.resize(np.flipud(cells), (pip_w, pip_h), interpolation=cv2.INTER_NEAREST)
    extent = np.asarray(grid.shape[::-1]) * grid.resolution

    def to_pip(xy):
        e, n = (np.asarray(xy, dtype=float) - grid.origin) / extent
        return int(round(pip_w * e)), int(round(pip_h * (1.0 - n)))

    for r_pts in rover_status.get("routes_history", []):
        if len(r_pts) >= 2:
            pts = np.array([to_pip(p) for p in r_pts], dtype=np.int32)
            cv2.polylines(base_pip, [pts], False, (250, 221, 80), 2, cv2.LINE_AA)
    for cand in victims_doc.get("candidates", []):
        px, py = to_pip(cand["mean_enu_m"])
        cv2.circle(base_pip, (px, py), 6, (28, 159, 255), -1)
        cv2.circle(base_pip, (px, py), 6, (255, 255, 255), 1)
    cv2.rectangle(base_pip, (0, 0), (pip_w - 1, pip_h - 1), (210, 225, 235), 2)

    out_w, out_h = 1280, 720
    fps = max(5.0, 5.0 * float(video_speed))
    video_path = folder / "evaluation/rescue_simulation.mp4"
    writer = None
    used_codec = None
    for fourcc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*fourcc), fps, (out_w, out_h))
        if w.isOpened():
            writer = w
            used_codec = fourcc
            break
    if writer is None:
        return None

    trail = []
    written = 0
    try:
        hud_font = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 19)
        hud_font_sm = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 16)
        pip_font = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 16)
    except Exception:
        hud_font = ImageFont.load_default(size=18)
        hud_font_sm = ImageFont.load_default(size=15)
        pip_font = ImageFont.load_default(size=15)

    state_cn_map = {
        "TRACKING": "巡航避障跟踪",
        "ALIGN_HEADING": "原地对准航向",
        "FACE_VICTIM": "转向正对伤员",
        "VERIFY_AND_DELIVER": "抵近停稳复核",
        "DELIVERED": "完成物资投送",
        "ARRIVED": "到达停稳",
        "STOPPED_SAFE": "安全停车",
    }

    def draw_cn_labels(bgr_img, top_text, captions):
        """Render crisp Chinese HUD & PiP captions onto a BGR frame via Pillow."""
        pil = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(pil)
        use_font = hud_font if dr.textlength(top_text, font=hud_font) <= (out_w - 36) else hud_font_sm
        dr.text((18, 14), top_text, font=use_font, fill=(238, 246, 252))
        for cx, cy, ctext in captions:
            tw = round(dr.textlength(ctext, font=pip_font))
            dr.rectangle((cx - 4, cy - 3, cx + tw + 8, cy + 21), fill=(16, 24, 34))
            dr.text((cx + 2, cy - 1), ctext, font=pip_font, fill=(240, 248, 255))
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    # Phase 1 (optional): Drone flight video from drone_fpv_cam + downward aerial_rgb PiP.
    drone_chase_dir = folder / "evaluation/drone_chase_frames"
    nadir_dir = folder / "frames"
    if drone_chase_dir.is_dir():
        drone_files = sorted(drone_chase_dir.glob("*.jpg"))
        nadir_files = sorted(nadir_dir.glob("*.jpg")) if nadir_dir.is_dir() else []
        for idx, dpath in enumerate(drone_files):
            dframe = cv2.imread(str(dpath))
            if dframe is None:
                continue
            dframe = cv2.resize(dframe, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            overlay = dframe.copy()
            cv2.rectangle(overlay, (0, 0), (out_w, 54), (16, 24, 34), -1)
            dframe = cv2.addWeighted(overlay, 0.74, dframe, 0.26, 0.0)
            hud_uav = (
                f"阶段一：无人机航测建图与纯 YOLO 空中搜寻 | "
                f"仿真时间 {idx * 0.2:.1f} s（{video_speed:.0f}倍速）| 第三人称跟飞 + 机腹下视"
            )
            captions = []
            if nadir_files:
                n_idx = min(len(nadir_files) - 1, idx // 5)
                nimg = cv2.imread(str(nadir_files[n_idx]))
                if nimg is not None:
                    nw, nh = 280, 210
                    nimg = cv2.resize(nimg, (nw, nh), interpolation=cv2.INTER_AREA)
                    cv2.rectangle(nimg, (0, 0), (nw - 1, nh - 1), (210, 225, 235), 2)
                    nx0, ny0 = 18, out_h - nh - 18
                    dframe[ny0:ny0 + nh, nx0:nx0 + nw] = nimg
                    captions.append((nx0 + 2, ny0 - 26, "无人机机腹下视航拍"))
            dframe = draw_cn_labels(dframe, hud_uav, captions)
            writer.write(dframe)
            written += 1

    # Phase 2: Rover driving & rescue video + live Rover FPV PiP + live UAV Mini-Map PiP.
    fpv_dir = folder / "rover/fpv_frames"
    fpv_files = sorted(fpv_dir.glob("*.jpg")) if fpv_dir.is_dir() else []
    local_tag = "神经 DAgger 局规" if (rover_status.get("local_policy") or {}).get("policy") == "neural" else "规则局规"
    for row_idx, row in enumerate(meta_rows):
        fpath = frames_dir / f"{row['frame']:05d}.jpg"
        if not fpath.is_file():
            continue
        frame = cv2.imread(str(fpath))
        if frame is None:
            continue
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (out_w, 54), (16, 24, 34), -1)
        frame = cv2.addWeighted(overlay, 0.74, frame, 0.26, 0.0)
        st_cn = state_cn_map.get(row["state"], row["state"])
        hud1 = (
            f"阶段二：四轮车纯 YOLO + VCTP/IGHA* + {local_tag} | "
            f"t={row['time_s']:.1f} s（{video_speed:.0f}x）| 车速 {row['speed']:.2f} m/s | "
            f"已投送 {row['delivered']}/{eval_summary['cargo_capacity']} | {st_cn}"
        )
        captions = []

        if fpv_files:
            f_idx = min(len(fpv_files) - 1, int(round(row_idx * len(fpv_files) / max(1, len(meta_rows)))))
            fpv_img = cv2.imread(str(fpv_files[f_idx]))
            if fpv_img is not None:
                fw, fh = 280, 210
                fpv_img = cv2.resize(fpv_img, (fw, fh), interpolation=cv2.INTER_AREA)
                cv2.rectangle(fpv_img, (0, 0), (fw - 1, fh - 1), (210, 225, 235), 2)
                fx0, fy0 = 18, out_h - fh - 18
                frame[fy0:fy0 + fh, fx0:fx0 + fw] = fpv_img
                captions.append((fx0 + 2, fy0 - 26, "四轮车车头前视相机"))

        pip = base_pip.copy()
        trail.append(to_pip((row["x"], row["y"])))
        if len(trail) >= 2:
            cv2.polylines(pip, [np.array(trail, dtype=np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(pip, trail[-1], 5, (80, 235, 255), -1)
        cv2.circle(pip, trail[-1], 5, (15, 25, 35), 1)
        x0, y0 = out_w - pip_w - 18, out_h - pip_h - 18
        frame[y0:y0 + pip_h, x0:x0 + pip_w] = pip
        captions.append((x0 + 2, y0 - 26, "无人机通行图 + 小车实时轨迹"))

        frame = draw_cn_labels(frame, hud1, captions)
        writer.write(frame)
        written += 1

    writer.release()
    if written > 0 and video_path.is_file() and video_path.stat().st_size > 1024:
        # Save a permanent 2x2 showcase of the drone and rover cameras before cleaning temp frames.
        chase_dir = folder / "rover/chase_frames"
        tiles = []
        for label, directory in (
            ("Drone 3rd-Person (drone_fpv_cam)", drone_chase_dir),
            ("Drone Downward (aerial_rgb)", nadir_dir),
            ("Rover 3rd-Person (rover_chase_cam)", chase_dir),
            ("Rover Front FPV (rgb)", fpv_dir),
        ):
            files = sorted(directory.glob("*.jpg")) if directory.is_dir() else []
            if files:
                img = cv2.imread(str(files[len(files) // 2]))
                tile = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA) if img is not None else np.zeros((270, 480, 3), dtype=np.uint8)
            else:
                tile = np.full((270, 480, 3), 24, dtype=np.uint8)
            cv2.rectangle(tile, (0, 0), (480, 30), (16, 24, 34), -1)
            cv2.putText(tile, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (240, 248, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        if len(tiles) == 4:
            grid_4 = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
            cv2.imwrite(str(folder / "evaluation/four_cameras_preview.jpg"), grid_4, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        shutil.rmtree(frames_dir, ignore_errors=True)
        shutil.rmtree(drone_chase_dir, ignore_errors=True)
        shutil.rmtree(fpv_dir, ignore_errors=True)
        shutil.rmtree(chase_dir, ignore_errors=True)
        print(
            f"[Run rescue] Assembled video: {video_path} ({written} frames, {fps:.0f} fps, "
            f"codec={used_codec}, {video_path.stat().st_size / 1048576:.2f} MB)",
            flush=True,
        )
        return video_path
    return None


def run(profile="full", detector="yolo", planner="igha_vctp",
        max_victims=2, refinement_strategy="route_value", nadir_only=False,
        victim_seed=None, random_clothing=False,
        local_policy="rule", checkpoint=None,
        obstacle="none", map_from=None, visible=False, realtime=False,
        keep_open=False, record_video=False, video_speed=4.0,
        port=18578, timeout=360.0):
    if victim_seed is not None and map_from is not None:
        raise ValueError("--victim-seed / --random-victims requires a fresh UAV survey (omit --map-from).")
    if local_policy == "neural" and not checkpoint:
        raise ValueError("--checkpoint is required when --local-policy neural.")
    checkpoint_path = Path(checkpoint).resolve() if checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Local policy checkpoint not found: {checkpoint_path}")
    config = json.loads((ROOT / "config/scenes" / PROFILES[profile]).read_text(encoding="utf-8"))
    if nadir_only:
        config.pop("refinement", None)
    elif refinement_strategy and "refinement" in config:
        config["refinement"]["strategy"] = refinement_strategy
    injected = [] if obstacle == "none" else OBSTACLE_PRESETS[(profile, obstacle)]
    obs_tag = "clear" if obstacle == "none" else obstacle
    rand_tag = f"_randv{victim_seed}" if victim_seed is not None else ""

    run_id = f"uav_rescue_D_{profile}_{detector}_{planner}_{obs_tag}{rand_tag}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    folder = ROOT / "runs" / run_id
    folder.mkdir(parents=True)
    (folder / "map_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    reuse_map = map_from is not None
    if reuse_map:
        src = Path(map_from)
        if not src.is_absolute():
            src = ROOT / src
        for name in ("uav_map.npz", "uav_map_nadir.npz", "observations.npz", "oblique_survey.json"):
            if (src / name).is_file():
                shutil.copy2(src / name, folder / name)
        if (src / "aerial_products/surface_dsm.tif").is_file():
            (folder / "aerial_products").mkdir(exist_ok=True)
            shutil.copy2(src / "aerial_products/surface_dsm.tif", folder / "aerial_products/surface_dsm.tif")
        if not (folder / "uav_map.npz").is_file():
            raise FileNotFoundError(f"uav_map.npz not found in {src}")
        # Run sensor-only aerial victim detection on saved survey frames (zero truth access).
        existing_av = src / "aerial_victims.json"
        reused_av = False
        if existing_av.is_file():
            doc = json.loads(existing_av.read_text(encoding="utf-8"))
            if doc.get("detector") == detector:
                shutil.copy2(existing_av, folder / "aerial_victims.json")
                reused_av = True
        if not reused_av:
            print(f"[Run rescue] Detecting aerial victim candidates from {src} using {detector}...", flush=True)
            av_doc = detect_victims_from_saved_survey(src, detector_mode=detector)
            (folder / "aerial_victims.json").write_text(json.dumps(av_doc, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[Run rescue] Found {len(av_doc['candidates'])} multi-view aerial candidate(s).", flush=True)

    world = ROOT / "worlds" / f"_validation_{run_id}.wbt"
    build_nav_world(world, config, injected, reuse_map=reuse_map)
    randomized_victims = None
    if victim_seed is not None:
        v_count = 2 if profile == "local" else 3
        randomized_victims, terrain = sample_random_victims_d(
            config, seed=victim_seed, count=v_count, random_clothing=random_clothing,
        )
        apply_random_victims_to_world(world, randomized_victims, terrain)
        eval_dir = folder / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        # Evaluator-only truth override; NEVER read by UAV or rover controllers.
        (eval_dir / "truth_override.json").write_text(
            json.dumps(dict(
                evaluation_only=True,
                victim_seed=int(victim_seed),
                random_clothing=bool(random_clothing),
                victims=randomized_victims,
            ), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"[Run rescue] Spawned {len(randomized_victims)} randomized victim(s) (seed={victim_seed}, "
            f"random_clothing={random_clothing}); truth kept evaluator-only.",
            flush=True,
        )
    executable = executable_path()
    if not executable.is_file():
        raise FileNotFoundError(f"Webots executable not found: {executable}")
    runtime = f"[python]\nCOMMAND = {Path(sys.executable).as_posix()}\n"
    for controller in ("uav_survey", "uav_mapper", "uav_map_rover"):
        (ROOT / f"controllers/{controller}/runtime.ini").write_text(runtime, encoding="utf-8")

    flight = config["flight"]
    points = flight["waypoints_xy_enu_m"]
    flight_duration = 0.0 if reuse_map else sum(math.dist(a, b) for a, b in zip(points, points[1:])) / flight["speed_mps"]

    env = dict(
        os.environ,
        UAV_MAPPING_RUN=run_id,
        UAV_MAPPING_FLIGHT_END=str(flight_duration),
        UAV_NAV_ENABLE="1",
        UAV_RESCUE_MODE="1",
        UAV_RESCUE_DETECTOR=detector,
        UAV_RESCUE_MAX_VICTIMS=str(max_victims),
        UAV_NAV_PLANNER=planner,
        UAV_LOCAL_POLICY=local_policy,
        UAV_LOCAL_CHECKPOINT=str(checkpoint_path) if checkpoint_path else "",
        UAV_NAV_TIMEOUT_S=str(timeout),
        UAV_NAV_KEEP_OPEN="1" if keep_open else "0",
        UAV_RECORD_VIDEO="1" if record_video else "0",
        UAV_VIDEO_SPEED=str(video_speed),
        PYTHONDONTWRITEBYTECODE="1",
    )

    mode = "realtime" if realtime else "fast"
    command = [str(executable), "--batch", f"--mode={mode}", f"--port={port}", "--stdout", "--stderr"]
    if not (visible or record_video):
        command.append("--no-rendering")
    command.append(str(world))
    log = folder / "webots.log"
    print(f"Run: {folder}", flush=True)

    wall_budget = max(480, (flight_duration + timeout + (160 if "refinement" in config and not reuse_map else 0)) * 4)
    try:
        result = subprocess.run(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=wall_budget, text=True, encoding="utf-8", errors="replace",
            creationflags=(subprocess.CREATE_NO_WINDOW if (os.name == "nt" and not (visible or record_video)) else 0),
        )
    finally:
        if not keep_open and world.is_file():
            try:
                world.unlink()
            except OSError:
                pass

    output = result.stdout
    log.write_text(output, encoding="utf-8")
    if result.returncode or simulator_has_errors(output):
        raise RuntimeError(f"Webots exited with code {result.returncode}. See {log}\n{output[-3000:]}")

    eval_summary = json.loads((folder / "evaluation/summary.json").read_text(encoding="utf-8"))
    rover_status = json.loads((folder / "rover/nav_status.json").read_text(encoding="utf-8"))
    victims_doc = (json.loads((folder / "aerial_victims.json").read_text(encoding="utf-8"))
                   if (folder / "aerial_victims.json").is_file() else {"candidates": []})

    mapping_summary = {}
    if not reuse_map and (folder / "observations.npz").is_file():
        route_doc = json.loads((folder / "route.json").read_text(encoding="utf-8"))
        with np.load(folder / "observations.npz", allow_pickle=False) as measurements:
            observations = dict(measurements)
        products = export_products(folder, observations)
        draw_aerial_products(folder, products)
        supplement = folder / "oblique_survey.json"
        schedule = json.loads(supplement.read_text(encoding="utf-8")) if supplement.is_file() else None
        cells = (draw_full_preview(folder, route_doc, observations, config, schedule) if profile == "full"
                 else draw_preview(folder, route_doc, observations))
        mapping_summary = dict(
            captured_views=len(observations["timestamp_s"]),
            survey_end_s=flight_duration,
            alignment=audit_alignment(folder, observations),
            cells=cells,
            mapping=survey_metrics(folder / "uav_map.npz"),
            aerial_products=products,
        )
        if schedule is not None:
            mapping_summary["refinement"] = refinement_metrics(folder, observations, schedule, config)
            if schedule["patches"]:
                draw_refinement_preview(folder, schedule, mapping_summary["refinement"])

    draw_rescue_preview(folder, eval_summary, rover_status, victims_doc)
    video_path = assemble_rescue_video(folder, rover_status, victims_doc, eval_summary, video_speed=video_speed) if record_video else None

    combined = dict(
        run_id=run_id,
        profile=profile,
        detector=detector,
        planner=planner,
        max_victims=max_victims,
        reused_map_from=str(map_from) if map_from else None,
        video_path=str(video_path) if video_path else None,
        evaluation=eval_summary,
        mapping=mapping_summary,
    )
    (folder / "summary.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(eval_summary, indent=2, ensure_ascii=False))
    return folder


def main():
    parser = argparse.ArgumentParser(description="Air-to-ground victim detection & rescue in Webots D.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument("--detector", choices=("yolo", "surrogate_rgbd"), default="yolo")
    parser.add_argument("--planner", choices=("igha_vctp", "astar"), default="igha_vctp")
    parser.add_argument("--max-victims", type=int, default=2,
                        help="Maximum number of supply crates / victims to visit in one sortie (default: 2).")
    parser.add_argument("--refinement-strategy", choices=("density", "route_value"), default="route_value")
    parser.add_argument("--nadir-only", action="store_true",
                        help="Skip oblique refinement pass for faster fresh-flight runs.")
    parser.add_argument("--random-victims", action="store_true",
                        help="Randomize victim positions in Scene D using a fresh seed (requires fresh UAV flight).")
    parser.add_argument("--victim-seed", type=int, default=None,
                        help="Explicit integer seed for randomized victim spawning in Scene D.")
    parser.add_argument("--random-clothing", action="store_true",
                        help="Also randomize victim shirt & pants colors when spawning random victims.")
    parser.add_argument("--local-policy", choices=("rule", "neural"), default="rule",
                        help="Onboard rover local planner: rule baseline or neural network checkpoint.")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained PyTorch local policy checkpoint (.pt) for --local-policy neural.")
    parser.add_argument("--obstacle", choices=("none", "detour", "block"), default="none")
    parser.add_argument("--map-from", default=None,
                        help="Reuse an existing UAV survey folder (runs sensor-only victim detection on its frames).")
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--record-video", action="store_true",
                        help="Record 1280x720 H.264 follow-camera simulation video to evaluation/rescue_simulation.mp4.")
    parser.add_argument("--video-speed", type=float, default=4.0,
                        help="Playback speedup factor for recorded simulation video (default: 4.0x).")
    parser.add_argument("--port", type=int, default=18578)
    parser.add_argument("--timeout", type=float, default=360.0)
    args = parser.parse_args()
    vseed = args.victim_seed
    if args.random_victims and vseed is None:
        vseed = int(time.time()) % 100000
    run(
        profile=args.profile,
        detector=args.detector,
        planner=args.planner,
        max_victims=args.max_victims,
        refinement_strategy=args.refinement_strategy,
        nadir_only=args.nadir_only,
        victim_seed=vseed,
        random_clothing=args.random_clothing,
        local_policy=args.local_policy,
        checkpoint=args.checkpoint,
        obstacle=args.obstacle,
        map_from=args.map_from,
        visible=args.visible,
        realtime=args.realtime,
        keep_open=args.keep_open,
        record_video=args.record_video,
        video_speed=args.video_speed,
        port=args.port,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
