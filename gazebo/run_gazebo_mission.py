#!/usr/bin/env python3
"""Unified end-to-end Gazebo Harmonic + ROS 2 Jazzy runner for Mapping, Navigation, and Rescue.

Mirrors tools/run_uav_mapping.py, tools/run_uav_navigation.py, and tools/run_uav_rescue.py:
  1. Launches Gazebo Sim Harmonic (`gazebo/generated/D/search_d.sdf`) + `ros_gz_bridge`
  2. Runs UAV aerial mapping + DUCPP route-value oblique refinement + YOLO11n victim search
     (or reuses a saved prior via `--map-from`)
  3. Runs rover global planning (Traversing Mars dual-route, VCTP + IGHA*) and 10 Hz closed-loop
     navigation / victim confirmation / supply delivery with rule or neural local policy
  4. Runs independent ground-truth evaluation (`/eval/*`) with zero truth leakage to controllers.
"""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "gazebo" / "generated" / "D"
BRIDGE_SCRIPT = ROOT / "gazebo" / "ros2_rescue" / "launch_bridge.sh"
MAPPER_NODE = ROOT / "gazebo" / "ros2_rescue" / "uav_mapper_node.py"
ROVER_NODE = ROOT / "gazebo" / "ros2_rescue" / "rover_nav_node.py"
EVAL_NODE = ROOT / "gazebo" / "ros2_rescue" / "evaluator_node.py"

PROFILES = {
    "local": ROOT / "config" / "scenes" / "uav_mapping_D.json",
    "full": ROOT / "config" / "scenes" / "uav_mapping_full_D.json",
}


def compute_flight_end_s(config):
    pts = config["flight"]["waypoints_xy_enu_m"]
    speed = max(float(config["flight"]["speed_mps"]), 0.1)
    dist = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
    return math.ceil(dist / speed)


def load_obstacle_preset(profile, preset):
    if preset == "none":
        return []
    # Reuse the exact Webots obstacle preset definitions from tools/run_uav_navigation.py.
    sys.path.insert(0, str(ROOT / "tools"))
    from run_uav_navigation import OBSTACLE_PRESETS

    key = (profile, preset)
    if key in OBSTACLE_PRESETS:
        return OBSTACLE_PRESETS[key]
    for (_, p), spec in OBSTACLE_PRESETS.items():
        if p == preset:
            return spec
    return []


def prepare_run_directory(args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    tag = f"gz_{args.mode}_D_{args.profile}_{args.planner}_{args.local_policy}_{stamp}_{pid}"
    run_dir = (args.out_dir or (ROOT / "runs" / tag)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(PROFILES[args.profile].read_text(encoding="utf-8"))
    if args.nadir_only:
        cfg.pop("refinement", None)
    elif "refinement" in cfg:
        cfg["refinement"]["strategy"] = args.refinement_strategy

    flight_end_s = compute_flight_end_s(cfg)
    cfg["flight_end_s"] = flight_end_s
    (run_dir / "map_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    if args.map_from:
        src = Path(args.map_from).resolve()
        for fname in ("uav_map.npz", "uav_map_nadir.npz", "observations.npz", "oblique_survey.json", "aerial_victims.json"):
            if (src / fname).is_file():
                shutil.copy2(src / fname, run_dir / fname)
        if (src / "frames").is_dir() and not (run_dir / "frames").exists():
            shutil.copytree(src / "frames", run_dir / "frames")
        if args.mode == "rescue" and not (run_dir / "aerial_victims.json").is_file() and (run_dir / "observations.npz").is_file():
            # Run offline aerial victim detection on reused survey frames.
            sys.path.insert(0, str(ROOT / "controllers/uav_mapper"))
            from aerial_victims import detect_victims_from_saved_survey

            victims_doc = detect_victims_from_saved_survey(run_dir, detector_mode=args.detector)
            (run_dir / "aerial_victims.json").write_text(json.dumps(victims_doc, indent=2), encoding="utf-8")

    obstacles = load_obstacle_preset(args.profile, args.obstacle)
    obstacles_path = run_dir / "injected_obstacles.json"
    if obstacles:
        obstacles_path.write_text(json.dumps(obstacles, indent=2), encoding="utf-8")

    return run_dir, cfg, flight_end_s, (obstacles_path if obstacles else None)


def _terminate_procs(procs):
    for p in reversed(procs):
        if p.poll() is None:
            try:
                p.send_signal(signal.SIGTERM)
            except Exception:
                pass
    deadline = time.time() + 5.0
    for p in reversed(procs):
        while p.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="rescue", choices=("mapping", "navigation", "rescue"))
    parser.add_argument("--profile", default="full", choices=("local", "full"))
    parser.add_argument("--refinement-strategy", default="route_value", choices=("route_value", "density"))
    parser.add_argument("--nadir-only", action="store_true")
    parser.add_argument("--map-from", type=Path, default=None, help="Reuse prior uav_map.npz from an earlier survey run.")
    parser.add_argument("--goal", default=None, help="Goal id for --mode navigation (e.g. valley, north).")
    parser.add_argument("--planner", default="igha_vctp", choices=("astar", "igha_vctp"))
    parser.add_argument("--local-policy", default="rule", choices=("rule", "neural"))
    parser.add_argument("--local-checkpoint", default="")
    parser.add_argument("--local-record", action="store_true")
    parser.add_argument("--local-expert-mix", type=float, default=0.0)
    parser.add_argument("--local-seed", type=int, default=17)
    parser.add_argument("--detector", default="yolo", choices=("yolo", "surrogate_rgbd"))
    parser.add_argument("--max-victims", type=int, default=2)
    parser.add_argument("--obstacle", default="none", choices=("none", "detour", "block"))
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--gps-noise-m", type=float, default=0.04)
    parser.add_argument("--bundle-dir", type=Path, default=BUNDLE_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--headless", action="store_true", help="Run Gazebo server-only (-s -r).")
    parser.add_argument("--record-cameras", action="store_true")
    parser.add_argument("--gz-partition", default="", help="Optional GZ_PARTITION to isolate concurrent Gazebo instances.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for ROS 2 nodes.")
    args = parser.parse_args()

    sdf_path = args.bundle_dir / "search_d.sdf"
    env_json = args.bundle_dir / "environment.json"
    if not sdf_path.is_file():
        parser.error(f"Missing {sdf_path}. Run `python3 tools/export_gazebo_d.py` first.")

    run_dir, cfg, flight_end_s, obstacles_path = prepare_run_directory(args)
    print(f"[Gazebo Mission] Run directory: {run_dir}", flush=True)

    env = os.environ.copy()
    env["GZ_UAV_SURVEY_RUN"] = str(run_dir)
    env["GZ_UAV_FLIGHT_END"] = str(flight_end_s)
    env["GZ_PARK_ROVER_DURING_SURVEY"] = "0" if args.map_from else "1"
    if args.gz_partition:
        env["GZ_PARTITION"] = args.gz_partition

    procs = []
    try:
        # 1. Start Gazebo Harmonic
        gz_cmd = [str(args.bundle_dir / "run.sh")]
        if args.headless:
            gz_cmd.extend(["-s"])
        print(f"[Gazebo Mission] Launching Gazebo: {' '.join(gz_cmd)}", flush=True)
        gz_proc = subprocess.Popen(gz_cmd, env=env)
        procs.append(gz_proc)
        time.sleep(3.0)

        # 2. Start ros_gz_bridge (ROS 2 Jazzy <-> Gazebo Harmonic)
        print("[Gazebo Mission] Launching ros_gz_bridge...", flush=True)
        bridge_proc = subprocess.Popen(["bash", str(BRIDGE_SCRIPT)], env=env)
        procs.append(bridge_proc)
        time.sleep(2.0)

        # 3. Phase 1: UAV Aerial Mapping + DUCPP + YOLO Search (unless --map-from reused)
        if not args.map_from:
            mapper_cmd = [
                args.python, str(MAPPER_NODE),
                "--run-dir", str(run_dir),
                "--flight-end", str(flight_end_s),
                "--gps-noise-m", str(args.gps_noise_m),
                "--environment-json", str(env_json),
            ]
            if args.mode == "rescue":
                mapper_cmd.extend(["--detector", args.detector])
            if args.record_cameras:
                mapper_cmd.append("--record-fpv")
            print(f"[Gazebo Mission] Phase 1 (UAV mapping): {' '.join(mapper_cmd)}", flush=True)
            subprocess.run(mapper_cmd, env=env, check=True)
            if args.mode == "mapping":
                print(f"[Gazebo Mission] Mapping complete: {run_dir / 'uav_map.npz'}", flush=True)
                return

        # Wait briefly for rover deployment onto ground after uav_map.npz appears.
        time.sleep(2.0)

        # 4. Phase 2: Independent Evaluator + Rover Navigation / Rescue
        eval_cmd = [
            args.python, str(EVAL_NODE),
            "--run-dir", str(run_dir),
            "--mode", args.mode,
            "--target-deliveries", str(args.max_victims),
        ]
        if obstacles_path:
            eval_cmd.extend(["--obstacles-json", str(obstacles_path)])
        eval_proc = subprocess.Popen(eval_cmd, env=env)
        procs.append(eval_proc)

        rover_cmd = [
            args.python, str(ROVER_NODE),
            "--run-dir", str(run_dir),
            "--mode", args.mode,
            "--planner", args.planner,
            "--local-policy", args.local_policy,
            "--local-expert-mix", str(args.local_expert_mix),
            "--local-seed", str(args.local_seed),
            "--detector", args.detector,
            "--max-victims", str(args.max_victims),
            "--timeout-s", str(args.timeout_s),
            "--gps-noise-m", str(args.gps_noise_m),
            "--environment-json", str(env_json),
        ]
        if args.goal:
            rover_cmd.extend(["--goal", args.goal])
        if args.local_checkpoint:
            rover_cmd.extend(["--local-checkpoint", args.local_checkpoint])
        if args.local_record:
            rover_cmd.append("--local-record")
        if args.record_cameras:
            rover_cmd.append("--record-cameras")

        print(f"[Gazebo Mission] Phase 2 (Rover {args.mode}): {' '.join(rover_cmd)}", flush=True)
        subprocess.run(rover_cmd, env=env, check=True)
        eval_proc.wait(timeout=30)

        summary_path = run_dir / "evaluation" / "summary.json"
        if summary_path.is_file():
            print("[Gazebo Mission] Evaluation summary:", flush=True)
            print(summary_path.read_text(encoding="utf-8"), flush=True)
    finally:
        _terminate_procs(procs)


if __name__ == "__main__":
    main()
