"""Run air-to-ground victim detection, safe approach planning, and supply delivery in Webots D."""
from pathlib import Path
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aerial_products import export_products
from run_search import ROOT, executable_path, simulator_has_errors
from run_uav_mapping import PROFILES, refinement_metrics, survey_metrics
from run_uav_navigation import OBSTACLE_PRESETS, build_nav_world
from uav_full_preview import draw_aerial_products, draw_full_preview, draw_refinement_preview
from uav_map_preview import audit_alignment, draw_preview, load_prior

sys.path.insert(0, str(ROOT / "controllers/uav_mapper"))
from aerial_victims import detect_victims_from_saved_survey
from survey_map import BLOCKED, FREE, UNKNOWN


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
    truth_file = ROOT / "evaluation/search/D.json"
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
    for row in meta_rows:
        fpath = frames_dir / f"{row['frame']:05d}.jpg"
        if not fpath.is_file():
            continue
        frame = cv2.imread(str(fpath))
        if frame is None:
            continue
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # Dark semi-transparent HUD strip at top.
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (out_w, 54), (16, 24, 34), -1)
        frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0)
        hud1 = (
            f"Scene D Air-to-Ground Rescue  |  Planner: {eval_summary.get('planner', 'igha_vctp')}  |  "
            f"t={row['time_s']:.1f}s ({video_speed:.0f}x)  |  v={row['speed']:.2f} m/s  |  "
            f"Drops: {row['delivered']}/{eval_summary['cargo_capacity']}  |  "
            f"State: {row['state']}"
        )
        cv2.putText(frame, hud1, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (238, 246, 252), 2, cv2.LINE_AA)

        # Live UAV mini-map PiP in bottom-right corner.
        pip = base_pip.copy()
        trail.append(to_pip((row["x"], row["y"])))
        if len(trail) >= 2:
            cv2.polylines(pip, [np.array(trail, dtype=np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(pip, trail[-1], 5, (80, 235, 255), -1)
        cv2.circle(pip, trail[-1], 5, (15, 25, 35), 1)
        x0, y0 = out_w - pip_w - 18, out_h - pip_h - 18
        frame[y0:y0 + pip_h, x0:x0 + pip_w] = pip
        cv2.putText(frame, "UAV Prior + Live Rover", (x0 + 6, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 248, 255), 1, cv2.LINE_AA)

        writer.write(frame)
        written += 1

    writer.release()
    if written > 0 and video_path.is_file() and video_path.stat().st_size > 1024:
        shutil.rmtree(frames_dir, ignore_errors=True)
        print(
            f"[Run rescue] Assembled video: {video_path} ({written} frames, {fps:.0f} fps, "
            f"codec={used_codec}, {video_path.stat().st_size / 1048576:.2f} MB)",
            flush=True,
        )
        return video_path
    return None


def run(profile="full", detector="yolo", planner="igha_vctp",
        max_victims=2, refinement_strategy="route_value",
        obstacle="none", map_from=None, visible=False, realtime=False,
        keep_open=False, record_video=False, video_speed=4.0,
        port=18578, timeout=360.0):
    config = json.loads((ROOT / "config/scenes" / PROFILES[profile]).read_text(encoding="utf-8"))
    if refinement_strategy and "refinement" in config:
        config["refinement"]["strategy"] = refinement_strategy
    injected = [] if obstacle == "none" else OBSTACLE_PRESETS[(profile, obstacle)]
    obs_tag = "clear" if obstacle == "none" else obstacle

    run_id = f"uav_rescue_D_{profile}_{detector}_{planner}_{obs_tag}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
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
    run(
        profile=args.profile,
        detector=args.detector,
        planner=args.planner,
        max_victims=args.max_victims,
        refinement_strategy=args.refinement_strategy,
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
