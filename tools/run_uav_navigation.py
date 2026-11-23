"""Run closed-loop UAV-map to four-wheel rover navigation in Webots D and save evidence."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_search_worlds import build, replace_node
from metrics import height_at, look_at_rotation
from run_search import ROOT, executable_path, referenced_assets, simulator_has_errors
from run_uav_mapping import PROFILES, refinement_metrics, survey_metrics
from aerial_products import export_products
from uav_full_preview import draw_aerial_products, draw_full_preview, draw_refinement_preview
from uav_map_preview import audit_alignment, draw_preview, load_prior

sys.path.insert(0, str(ROOT / "controllers/uav_mapper"))
from survey_map import BLOCKED, FREE, UNKNOWN


OBSTACLE_PRESETS = {
    ("local", "detour"): [
        {
            "name": "route_boulder",
            "kind": "detourable_debris",
            "xy_enu_m": [-9.25, 1.75],
            "size_m": [0.55, 0.55, 0.48],
            "yaw_rad": 0.25,
            "color_rgb": [0.54, 0.36, 0.24],
        }
    ],
    ("local", "block"): [
        {
            "name": "corridor_barrier",
            "kind": "blocking_debris",
            "xy_enu_m": [-11.25, -0.75],
            "size_m": [1.80, 0.55, 0.58],
            "yaw_rad": 0.0,
            "color_rgb": [0.58, 0.26, 0.20],
        }
    ],
    ("full", "detour"): [
        {
            "name": "valley_boulder",
            "kind": "detourable_debris",
            "xy_enu_m": [-4.75, -16.75],
            "size_m": [0.65, 0.65, 0.52],
            "yaw_rad": 0.2,
            "color_rgb": [0.54, 0.36, 0.24],
        }
    ],
    ("full", "block"): [
        {
            "name": "valley_barrier",
            "kind": "blocking_debris",
            "xy_enu_m": [12.75, -12.75],
            "size_m": [3.60, 0.65, 0.60],
            "yaw_rad": 0.35,
            "color_rgb": [0.58, 0.26, 0.20],
        }
    ],
    ("full", "dynamic"): [
        {
            "name": "dynamic_crossing_hazard",
            "kind": "dynamic_crossing",
            "xy_enu_m": [-5.2, -18.2],
            "xy_end_enu_m": [-5.2, -15.0],
            "speed_mps": 0.25,
            "size_m": [0.56, 0.56, 0.54],
            "yaw_rad": 0.0,
            "color_rgb": [0.92, 0.42, 0.08],
        }
    ],
}


def build_nav_world(path, config, injected_obstacles, reuse_map=False):
    source = ROOT / "worlds/search_D.wbt"
    terrain_file = ROOT / "evaluation/search/D.json"
    if not source.is_file() or not terrain_file.is_file():
        source = build("D")
    scene = source.read_text(encoding="utf-8")
    terrain = json.loads(terrain_file.read_text(encoding="utf-8"))
    flight = config["flight"]
    x, y = flight["waypoints_xy_enu_m"][0]
    z = flight["altitude_enu_m"]
    drone_controller = "<none>" if reuse_map else "uav_mapper"
    scene = replace_node(
        scene,
        r"\bDEF\s+OBSERVER_DRONE\s+ObserverDrone",
        f'DEF OBSERVER_DRONE ObserverDrone {{ translation {x} {y} {z} controller "{drone_controller}" }}',
    )
    start = config["deployment_xy_enu_m"]
    position = [*start, height_at(terrain, *start) + 0.035]
    fixture = json.dumps(json.dumps({
        "deployment_xyz_m": position,
        "injected_obstacles": injected_obstacles,
    }))
    scene = replace_node(
        scene,
        r"\bDEF\s+DISASTER_ENVIRONMENT\s+Robot",
        'DEF SURVEY Robot { name "Aerial grid survey" supervisor TRUE '
        f'controller "uav_survey" customData {fixture} window "<none>" }}',
    )
    scene = replace_node(scene, r"\bDEF\s+SEARCH_REFEREE\s+Robot", "")
    west, south = config["bounds_enu_m"][0]
    staging = [west - 12, south - 10]
    stage_z = height_at(terrain, *staging) + 0.035
    scene = replace_node(
        scene,
        r"\bDEF\s+ROVER\s+ResearchRover",
        f'DEF ROVER ResearchRover {{ name "Map reading rover" controller "uav_map_rover" '
        f'translation {staging[0]} {staging[1]} {stage_z} }}',
    )
    (xmin, ymin), (xmax, ymax) = terrain["bounds_xy"]
    if not (xmin <= staging[0] <= xmax and ymin <= staging[1] <= ymax):
        scene += (
            f'\nSolid {{ translation {staging[0]} {staging[1]} {stage_z - 0.085} '
            'boundingObject Box { size 2 2 .1 } }\n'
        )
    center = [(low + high) / 2 for low, high in zip(*config["bounds_enu_m"])]
    width = config["bounds_enu_m"][1][0] - config["bounds_enu_m"][0][0]
    eye = [center[0] - 0.6 * width, center[1] - max(20, 0.82 * width), max(25, 0.60 * width)]
    angles = " ".join(map(str, look_at_rotation(eye, [*center, 2.0])))
    scene = replace_node(
        scene,
        r"\bDEF\s+SCENE_VIEW\s+Viewpoint",
        f'DEF SCENE_VIEW Viewpoint {{ position {" ".join(map(str, eye))} '
        f'orientation {angles} fieldOfView 1.0 exposure 1.25 }}',
    )
    path.write_text(scene, encoding="utf-8")
    referenced_assets(path)


def draw_nav_preview(folder, route_doc, eval_summary, rover_status):
    """Render UAV prior map, planned/replanned routes, obstacles, and truth trajectory."""
    grid, states = load_prior(folder / "uav_map.npz")
    palette = {UNKNOWN: (84, 94, 107), FREE: (139, 178, 144), BLOCKED: (217, 104, 74)}
    cells = np.zeros((*states.shape, 3), dtype=np.uint8)
    for state, color in palette.items():
        cells[states == state] = color
    for item in rover_status.get("onboard_blocked_cells", []):
        r, c = item["row"], item["col"]
        if 0 <= r < states.shape[0] and 0 <= c < states.shape[1]:
            cells[r, c] = (232, 76, 152)

    w = 720
    h = round(w * states.shape[0] / states.shape[1])
    margin, top, bottom = 28, 92, 132
    canvas = Image.new("RGB", (w + 2 * margin, h + top + bottom), "#111923")
    draw = ImageDraw.Draw(canvas)
    title_f = ImageFont.load_default(size=24)
    body_f = ImageFont.load_default(size=16)
    small_f = ImageFont.load_default(size=14)

    map_img = Image.fromarray(np.flipud(cells)).resize((w, h), Image.Resampling.NEAREST)
    canvas.paste(map_img, (margin, top))
    draw.rectangle((margin, top, margin + w, top + h), outline="#495b6a", width=2)

    extent = np.asarray(grid.shape[::-1]) * grid.resolution

    def to_px(xy):
        east, north = (np.asarray(xy, dtype=float) - grid.origin) / extent
        return float(margin + w * east), float(top + h * (1.0 - north))

    # Draw injected obstacle boxes (truth footprint, evaluation visualization only).
    for obs in eval_summary.get("injected_obstacles", []):
        cx, cy = obs["xy_enu_m"]
        sx, sy = obs["size_m"][:2]
        yaw = obs.get("yaw_rad", 0.0)
        c, s = math.cos(yaw), math.sin(yaw)
        corners = []
        for dx, dy in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)):
            corners.append(to_px((cx + c * dx - s * dy, cy + s * dx + c * dy)))
        draw.polygon(corners, fill="#d93838", outline="#ffe28a")

    # Draw initial global route and any replanned routes.
    history = rover_status.get("routes_history", [])
    if not history and route_doc.get("route_xy_m"):
        history = [route_doc["route_xy_m"]]
    for idx, r_pts in enumerate(history):
        if len(r_pts) < 2:
            continue
        poly = [to_px(p) for p in r_pts]
        color = "#50ddfa" if idx == 0 else "#ffd166"
        width = 4 if idx == 0 else 5
        draw.line(poly, fill="#0e2230", width=width + 3, joint="curve")
        draw.line(poly, fill=color, width=width, joint="curve")

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

    def mark(xy, label, fill, outline):
        px, py = to_px(xy)
        draw.ellipse((px - 11, py - 11, px + 11, py + 11), fill=fill, outline=outline, width=2)
        draw.text((px, py), label, font=small_f, fill=outline, anchor="mm")

    mark(route_doc["start_xy_m"], "S", "#142230", "#f1f7f9")
    mark(eval_summary["goal_xy_m"], "G", "#142230", "#50ddfa")
    mark(eval_summary["final_position_xy_m"], "F", "#1b3624" if eval_summary["arrived"] else "#4a1e24", "#ffffff")

    status = eval_summary["status"]
    goal_id = eval_summary["goal_id"]
    planner = eval_summary.get("planner", "astar")
    draw.text((margin, 14), f"D  /  Closed-loop Navigation [{goal_id} | {planner}]  ->  {status}",
              font=title_f, fill="#edf4f8")
    t_str = (f"{eval_summary['travel_duration_s']:.1f} s"
             if eval_summary.get("travel_duration_s") is not None else "n/a")
    subtitle = (
        f"arrived={eval_summary['arrived']}  |  settled={eval_summary['settled']}  |  "
        f"collisions={eval_summary['collisions']}  |  travel={t_str}  |  "
        f"goal_err={eval_summary['final_goal_error_m']:.2f} m  |  replans={eval_summary['replan_count']}"
    )
    draw.text((margin, 52), subtitle, font=body_f, fill="#b8c9d7")

    y_leg = top + h + 16
    entries = [
        (palette[FREE], "Confirmed free"),
        (palette[BLOCKED], "Prior hazard"),
        (palette[UNKNOWN], "Unknown"),
        ((232, 76, 152), "Onboard obs"),
        ("#50ddfa", "Initial A*"),
        ("#ffd166", "Replan"),
        ("#ffffff", "Truth traj"),
    ]
    x_cur = margin
    for color, label in entries:
        draw.rectangle((x_cur, y_leg, x_cur + 13, y_leg + 13), fill=color)
        draw.text((x_cur + 18, y_leg - 2), label, font=small_f, fill="#d1dee8")
        x_cur += 26 + round(draw.textlength(label, font=small_f))

    reason = eval_summary.get("failure_reason") or "Arrived and settled cleanly at configured approach point."
    draw.text((margin, y_leg + 30), f"Outcome: {reason[:105]}", font=small_f,
              fill="#a3f7bf" if eval_summary["arrived"] else "#ffd166")
    draw.text(
        (margin, y_leg + 56),
        f"Path: {eval_summary['actual_path_length_m']:.2f} m (planned {eval_summary['planned_route_length_m'] or 0:.2f} m)  |  "
        f"LiDAR frames: {eval_summary['lidar_detection_frames']}  |  RGB-D frames: {eval_summary['rgbd_detection_frames']}  |  "
        f"Max roll/pitch: {eval_summary['max_abs_roll_deg']:.1f}/{eval_summary['max_abs_pitch_deg']:.1f} deg",
        font=small_f, fill="#b8c9d7",
    )
    canvas.save(folder / "evaluation/nav_preview.png")


def run(profile="local", goal=None, planner="astar", refinement_strategy=None,
        obstacle="none", obstacle_xy=None, obstacle_size=None,
        map_from=None, visible=False, realtime=False, keep_open=False,
        port=18556, timeout=240.0, nadir_only=False, local_policy="rule",
        checkpoint=None, record_expert=False, expert_mix=0.0, policy_seed=17):
    if local_policy not in ("rule", "neural"):
        raise ValueError(f"Unknown local policy: {local_policy}")
    if (local_policy == "neural") != (checkpoint is not None):
        raise ValueError("--checkpoint is required with --local-policy neural and only used in that mode.")
    if checkpoint is not None:
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if not 0.0 <= expert_mix <= 1.0 or (expert_mix and (local_policy != "neural" or not record_expert)):
        raise ValueError("--expert-mix must be in [0, 1] and requires neural policy with --record-expert.")
    if not 0 <= policy_seed < 2**32 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Use a uint32 policy seed and a positive finite simulation timeout.")
    config = json.loads((ROOT / "config/scenes" / PROFILES[profile]).read_text(encoding="utf-8"))
    if nadir_only:
        config.pop("refinement", None)
    elif refinement_strategy and "refinement" in config:
        config["refinement"]["strategy"] = refinement_strategy

    if obstacle_xy is not None:
        size = obstacle_size or [0.60, 0.60, 0.50]
        injected = [{
            "name": "custom_obstacle",
            "kind": "custom",
            "xy_enu_m": list(map(float, obstacle_xy)),
            "size_m": list(map(float, size)),
            "yaw_rad": 0.0,
            "color_rgb": [0.56, 0.32, 0.22],
        }]
        obs_tag = "custom"
    elif obstacle == "none":
        injected = []
        obs_tag = "clear"
    else:
        injected = OBSTACLE_PRESETS[(profile, obstacle)]
        obs_tag = obstacle

    planner_tag = f"_{planner}" if planner != "astar" else ""
    policy_tag = "_neural" if local_policy == "neural" else ""
    if record_expert:
        policy_tag += "_capture"
    run_id = f"uav_nav_D_{profile}{planner_tag}_{obs_tag}{policy_tag}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
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
        UAV_RESCUE_MODE="0",
        UAV_NAV_PLANNER=planner,
        UAV_NAV_TIMEOUT_S=str(timeout),
        UAV_NAV_KEEP_OPEN="1" if keep_open else "0",
        UAV_NAV_GOAL=goal or "",
        UAV_LOCAL_POLICY=local_policy,
        UAV_LOCAL_CHECKPOINT=str(checkpoint) if checkpoint else "",
        UAV_LOCAL_RECORD="1" if record_expert else "0",
        UAV_LOCAL_EXPERT_MIX=str(expert_mix),
        UAV_LOCAL_SEED=str(policy_seed),
        PYTHONDONTWRITEBYTECODE="1",
    )

    mode = "realtime" if realtime else "fast"
    command = [str(executable), "--batch", f"--mode={mode}", f"--port={port}", "--stdout", "--stderr"]
    if not visible:
        command.append("--no-rendering")
    command.append(str(world))
    log = folder / "webots.log"
    print(f"Run: {folder}", flush=True)

    wall_budget = max(360, (flight_duration + timeout + (160 if "refinement" in config and not reuse_map else 0)) * 4)
    result = subprocess.run(
        command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=wall_budget, text=True, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    output = result.stdout
    log.write_text(output, encoding="utf-8")
    if result.returncode or simulator_has_errors(output):
        raise RuntimeError(f"Webots exited with code {result.returncode}. See {log}\n{output[-3000:]}")

    route_doc = json.loads((folder / "route.json").read_text(encoding="utf-8"))
    eval_summary = json.loads((folder / "evaluation/summary.json").read_text(encoding="utf-8"))
    rover_status = json.loads((folder / "rover/nav_status.json").read_text(encoding="utf-8"))

    # Export aerial mapping previews if fresh observations were collected.
    mapping_summary = {}
    if not reuse_map and (folder / "observations.npz").is_file():
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

    draw_nav_preview(folder, route_doc, eval_summary, rover_status)

    selected_route = next(item for item in route_doc["routes"] if item["id"] == eval_summary["goal_id"])
    condition = dict(
        profile=profile,
        goal_id=eval_summary["goal_id"],
        goal_xy_m=eval_summary["goal_xy_m"],
        deployment_xy_m=config["deployment_xy_enu_m"],
        global_planner=planner,
        obstacle_scenario=obs_tag,
        injected_obstacles=injected,
        timeout_s=timeout,
        sensor_period_ms=100,
        map_sha256=hashlib.sha256((folder / "uav_map.npz").read_bytes()).hexdigest(),
        world_sha256=hashlib.sha256(world.read_bytes()).hexdigest(),
        initial_route_sha256=hashlib.sha256(json.dumps(selected_route.get("route_xy_m")).encode()).hexdigest(),
    )
    combined = dict(
        run_id=run_id,
        profile=profile,
        planner=planner,
        refinement_strategy=config.get("refinement", {}).get("strategy", "density") if "refinement" in config else None,
        obstacle_scenario=obs_tag,
        reused_map_from=str(map_from) if map_from else None,
        navigation=eval_summary,
        condition=condition,
        local_policy=dict(
            policy=local_policy, checkpoint=str(checkpoint) if checkpoint else None,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
            record_expert=record_expert, expert_mix=expert_mix, seed=policy_seed,
            runtime=rover_status.get("local_policy"),
        ),
        mapping=mapping_summary or None,
    )
    (folder / "summary.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(eval_summary, indent=2, ensure_ascii=False), flush=True)
    return folder, eval_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="local")
    parser.add_argument("--goal", default=None, help="Configured goal id (e.g. local, valley, north, forest).")
    parser.add_argument("--planner", choices=("astar", "igha_vctp"), default="astar",
                        help="Global planner: baseline 2D terrain A* or VCTP + IGHA* heading-aware search.")
    parser.add_argument("--refinement-strategy", choices=("density", "route_value"), default=None,
                        help="Oblique patch selector when running fresh aerial survey.")
    parser.add_argument("--obstacle", choices=("none", "detour", "block", "dynamic"), default="none")
    parser.add_argument("--obstacle-xy", nargs=2, type=float, default=None, metavar=("X", "Y"))
    parser.add_argument("--obstacle-size", nargs=3, type=float, default=None, metavar=("SX", "SY", "SZ"))
    parser.add_argument("--map-from", default=None, help="Optional prior run directory to reuse uav_map.npz.")
    parser.add_argument("--nadir-only", action="store_true")
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--port", type=int, default=18556)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--local-policy", choices=("rule", "neural"), default="rule")
    parser.add_argument("--checkpoint", type=Path, help="PyTorch state_dict checkpoint for the neural policy.")
    parser.add_argument("--record-expert", action="store_true", help="Save onboard observations and expert labels.")
    parser.add_argument("--expert-mix", type=float, default=0.0, help="Per-decision expert action probability in DAgger.")
    parser.add_argument("--policy-seed", type=int, default=17)
    args = parser.parse_args()
    folder, summary = run(**vars(args))
    return 0 if summary["status"] in ("ARRIVED", "STOPPED_SAFE") else 1


if __name__ == "__main__":
    sys.exit(main())
