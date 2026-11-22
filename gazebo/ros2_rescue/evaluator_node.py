#!/usr/bin/env python3
"""Independent privileged ground-truth evaluator & world-spawner for Gazebo Harmonic + ROS 2 Jazzy.

Subscribes ONLY to `/eval/world_poses` (`/world/search_d/pose/info`) and `/eval/rover_contact`.
Strictly isolated from `uav_mapper_node.py` and `rover_nav_node.py` (zero truth leakage).
"""

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for rel in ("evaluation", "gazebo/ros2_rescue", "controllers/uav_mapper"):
    p = str(ROOT / rel)
    if p not in sys.path:
        sys.path.insert(0, p)

from metrics import height_at
from search_metrics import match_reports, physical_delivery_ok
from sensor_adapters import quat_xyzw_to_rpy, rpy_to_quat_xyzw


def _gz_service(service_name, reqtype, reptype, payload, timeout_ms=5000):
    gz_exe = shutil.which("gz")
    if not gz_exe:
        raise FileNotFoundError("`gz` CLI not found in PATH.")
    cmd = [
        gz_exe, "service",
        "-s", service_name,
        "--reqtype", reqtype,
        "--reptype", reptype,
        "--timeout", str(int(timeout_ms)),
        "--req", payload,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def build_box_sdf(name, size_xyz, xyz, yaw_rad=0.0, color_rgb=(0.46, 0.34, 0.24), static=True, *, collidable=True):
    sx, sy, sz = map(float, size_xyz)
    x, y, z = map(float, xyz)
    r, g, b = map(float, color_rgb)
    static_str = "true" if static else "false"
    geometry = f'<geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>'
    collision = f'<collision name="c">{geometry}</collision>' if collidable else ""
    return (
        f'<sdf version="1.10">'
        f'<model name="{name}"><static>{static_str}</static>'
        f'<pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw_rad:.4f}</pose>'
        f'<link name="body">'
        f'<visual name="v">{geometry}'
        f'<material><diffuse>{r:.3f} {g:.3f} {b:.3f} 1</diffuse></material></visual>'
        f"{collision}"
        f'</link></model></sdf>'
    )


def spawn_gazebo_box(world_name, name, size_xyz, xyz, yaw_rad=0.0,
                     color_rgb=(0.46, 0.34, 0.24), static=True, *, collidable=True):
    sdf_xml = build_box_sdf(name, size_xyz, xyz, yaw_rad, color_rgb, static=static, collidable=collidable)
    escaped = sdf_xml.replace("\\", "\\\\").replace('"', '\\"')
    payload = f'sdf: "{escaped}"'
    return _gz_service(
        f"/world/{world_name}/create",
        "gz.msgs.EntityFactory",
        "gz.msgs.Boolean",
        payload,
    )


def set_gazebo_entity_pose(world_name, entity_name, xyz, rpy=(0.0, 0.0, 0.0)):
    qx, qy, qz, qw = rpy_to_quat_xyzw(*rpy)
    x, y, z = map(float, xyz)
    payload = (
        f'name: "{entity_name}" '
        f'position: {{x: {x:.5f}, y: {y:.5f}, z: {z:.5f}}} '
        f'orientation: {{x: {qx:.6f}, y: {qy:.6f}, z: {qz:.6f}, w: {qw:.6f}}}'
    )
    return _gz_service(
        f"/world/{world_name}/set_pose",
        "gz.msgs.Pose",
        "gz.msgs.Boolean",
        payload,
    )


def spawn_test_obstacles(world_name, terrain, obstacles):
    """Spawn post-survey physical obstacles in Gazebo Harmonic."""
    spawned = []
    for idx, item in enumerate(obstacles, start=1):
        x, y = map(float, item["xy_enu_m"])
        sx, sy, sz = map(float, item["size_m"])
        yaw = float(item.get("yaw_rad", 0.0))
        color = item.get("color_rgb", [0.46, 0.34, 0.24])
        gz = float(height_at(terrain, x, y))
        cz = gz + 0.5 * sz - 0.03
        name = item.get("name", f"injected_obstacle_{idx}")
        res = spawn_gazebo_box(world_name, name, (sx, sy, sz), (x, y, cz), yaw, color, static=True)
        spawned.append(dict(
            name=name,
            kind=item.get("kind", "debris"),
            xy_enu_m=[x, y],
            ground_z_m=round(gz, 3),
            size_m=[sx, sy, sz],
            yaw_rad=yaw,
            spawn_ok=(res.returncode == 0),
        ))
    return spawned


def spawn_supply_crate(world_name, terrain, delivery_event, index):
    """Spawn a visual orange rescue supply crate beside the rover on valid delivery."""
    rx, ry = map(float, delivery_event["rover_xy_m"])
    yaw = float(delivery_event.get("rover_yaw_rad", 0.0))
    cx = rx + 0.42 * math.cos(yaw)
    cy = ry + 0.42 * math.sin(yaw)
    cz = float(height_at(terrain, cx, cy)) + 0.10
    name = f"supply_crate_{index:02d}"
    return spawn_gazebo_box(
        world_name, name, (0.24, 0.18, 0.18), (cx, cy, cz), yaw,
        color_rgb=(0.94, 0.36, 0.06), static=True, collidable=False,
    )


def _stationary_durations(rows):
    """Accumulate consecutive still time using only samples from the evaluator."""
    durations = np.zeros(len(rows), dtype=float)
    for i, (previous, current) in enumerate(zip(rows, rows[1:]), start=1):
        dt = float(current["time_s"]) - float(previous["time_s"])
        if not 0.0 < dt <= 0.25:
            continue
        horizontal = math.hypot(current["x"] - previous["x"], current["y"] - previous["y"])
        yaw_change = math.remainder(current["yaw_rad"] - previous["yaw_rad"], math.tau)
        if horizontal / dt < 0.03 and abs(yaw_change) / dt < 0.08:
            durations[i] = durations[i - 1] + dt
    return durations


def _is_obstacle_contact(first, second):
    """Filter the landscape and the survey staging platform from collision counts."""
    if any("rover_parking_platform" in name.split("::") for name in (first, second)):
        return False
    other = second if "rover" in first else first
    return bool(other and "terrain" not in other)


def evaluate_run_offline(run_dir, terrain, mode="navigation", truth_override_path=None,
                         trajectory_rows=None, collisions=0, chassis_contacts=0,
                         spawned_obstacles=None, target_deliveries=2):
    """Pure-Python evaluator computing Webots-compatible evaluation/summary.json."""
    folder = Path(run_dir)
    eval_dir = folder / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    rows = trajectory_rows or []

    path_length = 0.0
    max_roll_deg = 0.0
    max_pitch_deg = 0.0
    for prev, cur in zip(rows, rows[1:]):
        path_length += math.hypot(cur["x"] - prev["x"], cur["y"] - prev["y"])
    for cur in rows:
        max_roll_deg = max(max_roll_deg, abs(cur.get("roll_deg", 0.0)))
        max_pitch_deg = max(max_pitch_deg, abs(cur.get("pitch_deg", 0.0)))

    # Compute rover localization error against rover/telemetry.jsonl if available.
    loc_errors = []
    telem_path = folder / "rover/telemetry.jsonl"
    if telem_path.is_file() and rows:
        truth_times = np.array([r["time_s"] for r in rows], dtype=float)
        truth_xy = np.array([[r["x"], r["y"]] for r in rows], dtype=float)
        for line in telem_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            t = float(rec["time_s"])
            idx = int(np.argmin(np.abs(truth_times - t)))
            if abs(truth_times[idx] - t) <= 0.25:
                est = np.asarray(rec["pose_xy"], dtype=float)
                loc_errors.append(float(np.linalg.norm(est - truth_xy[idx])))

    loc_summary = (
        dict(
            median=round(float(np.median(loc_errors)), 4),
            p90=round(float(np.quantile(loc_errors, 0.9)), 4),
            max=round(float(np.max(loc_errors)), 4),
        )
        if loc_errors else None
    )

    if mode == "rescue":
        truth_doc = (
            json.loads(Path(truth_override_path).read_text(encoding="utf-8"))
            if truth_override_path and Path(truth_override_path).is_file()
            else terrain
        )
        truth_people = truth_doc.get("victims", truth_doc.get("people", []))
        rover_status_path = folder / "rover/rescue_status.json"
        rover_status = (
            json.loads(rover_status_path.read_text(encoding="utf-8"))
            if rover_status_path.is_file() else {}
        )
        events = rover_status.get("events", [])
        reports = [e for e in events if e.get("type") == "report"]
        deliveries = [e for e in events if e.get("type") == "delivery"]

        # Align report schema with evaluation/search_metrics.py
        formatted_reports = [
            dict(
                xy=r["xy_enu_m"],
                timestamp_s=r["time_s"],
                cross_view=bool(r.get("cross_view", False)),
            )
            for r in reports
        ]
        targets = [{"xy": person.get("xy", person.get("xy_enu_m"))} for person in truth_people]
        report_tol = float(truth_doc.get("report_tolerance_m", terrain.get("report_tolerance_m", 1.0)))
        delivery_tol = float(truth_doc.get("delivery_tolerance_m", terrain.get("delivery_tolerance_m", 1.8)))
        matched = match_reports(formatted_reports, targets, (0.0, 0.0), 0.0, report_tol)
        distances = [math.dist(formatted_reports[i]["xy"], targets[j]["xy"]) for i, j in matched.items()]
        rmse = math.sqrt(sum(d * d for d in distances) / len(distances)) if distances else None
        report_for_victim = {report["victim_id"]: i for i, report in enumerate(reports)}
        delivered_ids = set()
        time_samples = np.array([row["time_s"] for row in rows], dtype=float)
        stationary_s = _stationary_durations(rows)
        for d in deliveries:
            target_index = matched.get(report_for_victim.get(d["victim_id"]))
            if target_index is None or target_index in delivered_ids or not rows:
                continue
            delivery_time = float(d["time_s"])
            sample_index = int(np.searchsorted(time_samples, delivery_time, side="right")) - 1
            if sample_index < 0 or delivery_time - time_samples[sample_index] > 0.25:
                continue
            row = rows[sample_index]
            distance = math.dist((row["x"], row["y"]), targets[target_index]["xy"])
            cargo = int(target_deliveries) - len(delivered_ids)
            if physical_delivery_ok(True, round(float(stationary_s[sample_index]), 3), distance, cargo, delivery_tol):
                delivered_ids.add(target_index)

        valid_deliveries = len(delivered_ids)
        cross_view_valid = sum(formatted_reports[i]["cross_view"] for i in matched)
        mission_success = (
            valid_deliveries >= int(target_deliveries)
            and collisions == 0
            and chassis_contacts == 0
        )
        summary = dict(
            simulator="gazebo_harmonic_ros2",
            mode="rescue",
            status="MISSION_SUCCESS" if mission_success else rover_status.get("status", "INCOMPLETE"),
            mission_success=mission_success,
            target_deliveries=int(target_deliveries),
            total_reports=len(reports),
            effective_reports=len(matched),
            valid_cross_view_reports=cross_view_valid,
            delivered_people=valid_deliveries,
            localization_rmse_m=round(float(rmse), 4) if rmse is not None and math.isfinite(rmse) else None,
            collisions=int(collisions),
            chassis_contacts=int(chassis_contacts),
            path_length_m=round(float(path_length), 3),
            max_abs_roll_deg=round(float(max_roll_deg), 2),
            max_abs_pitch_deg=round(float(max_pitch_deg), 2),
            rover_legs=rover_status.get("legs", []),
            injected_obstacles=spawned_obstacles or [],
            rover_pose_error_m=loc_summary,
        )
        (eval_dir / "events.json").write_text(
            json.dumps(dict(reports=reports, deliveries=deliveries), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        nav_status_path = folder / "rover/nav_status.json"
        nav_status = (
            json.loads(nav_status_path.read_text(encoding="utf-8"))
            if nav_status_path.is_file() else {}
        )
        goal_xy = nav_status.get("goal_xy_m")
        final_err = None
        if goal_xy is not None and rows:
            final_err = float(math.hypot(rows[-1]["x"] - goal_xy[0], rows[-1]["y"] - goal_xy[1]))
        summary = dict(
            simulator="gazebo_harmonic_ros2",
            mode="navigation",
            status=nav_status.get("status", "UNKNOWN"),
            reason=nav_status.get("reason", ""),
            goal_id=nav_status.get("goal_id"),
            goal_xy_m=goal_xy,
            final_goal_error_m=round(final_err, 4) if final_err is not None else None,
            collisions=int(collisions),
            chassis_contacts=int(chassis_contacts),
            path_length_m=round(float(path_length), 3),
            max_abs_roll_deg=round(float(max_roll_deg), 2),
            max_abs_pitch_deg=round(float(max_pitch_deg), 2),
            replan_count=int(nav_status.get("replan_count", 0)),
            injected_obstacles=spawned_obstacles or [],
            rover_pose_error_m=loc_summary,
        )

    if rows:
        with (eval_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    (eval_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def run_ros2_evaluator(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter as RosParam
    from rclpy.qos import qos_profile_sensor_data
    from tf2_msgs.msg import TFMessage

    try:
        from ros_gz_interfaces.msg import Contacts
    except ImportError:
        Contacts = None

    rclpy.init()

    class _EvaluatorNode(Node):
        def __init__(self):
            super().__init__(
                "rescue_evaluator_node",
                parameter_overrides=[RosParam("use_sim_time", RosParam.Type.BOOL, True)],
            )
            self.folder = Path(args.run_dir)
            self.terrain = json.loads((ROOT / "evaluation/search/D.json").read_text(encoding="utf-8"))
            self.obstacles_spec = []
            if args.obstacles_json and Path(args.obstacles_json).is_file():
                self.obstacles_spec = json.loads(Path(args.obstacles_json).read_text(encoding="utf-8"))
            self.obstacles_spawned = False
            self.spawned_obstacles = []
            self.crates_spawned = 0

            self._lock = threading.Lock()
            self._latest_rover = None
            self._in_contact = False
            self.collisions = 0
            self.chassis_contacts = 0
            self.trajectory = []
            self._last_sample_s = -1.0

            self.create_subscription(TFMessage, "/eval/world_poses", self._on_poses, qos_profile_sensor_data)
            if Contacts is not None:
                self.create_subscription(Contacts, "/eval/rover_contact", self._on_contact, qos_profile_sensor_data)
            self.create_timer(0.1, self._tick)

        def _on_poses(self, msg):
            for tf in msg.transforms:
                if tf.child_frame_id == "rover":
                    t = tf.transform.translation
                    q = tf.transform.rotation
                    roll, pitch, yaw = quat_xyzw_to_rpy(q.x, q.y, q.z, q.w)
                    with self._lock:
                        self._latest_rover = (
                            float(t.x), float(t.y), float(t.z),
                            math.degrees(roll), math.degrees(pitch), float(yaw),
                        )
                    break

        def _on_contact(self, msg):
            has_obstacle_hit = False
            for c in getattr(msg, "contacts", []):
                c1 = getattr(c.collision1, "name", "")
                c2 = getattr(c.collision2, "name", "")
                if _is_obstacle_contact(c1, c2):
                    has_obstacle_hit = True
                    break
            with self._lock:
                if has_obstacle_hit and not self._in_contact:
                    self.collisions += 1
                    self.chassis_contacts += 1
                self._in_contact = has_obstacle_hit

        def _tick(self):
            now_s = self.get_clock().now().nanoseconds * 1e-9
            if now_s <= 0.0:
                return

            if not self.obstacles_spawned and self.obstacles_spec and (self.folder / "uav_map.npz").is_file():
                self.spawned_obstacles = spawn_test_obstacles(args.world_name, self.terrain, self.obstacles_spec)
                self.obstacles_spawned = True
                print(f"[ROS2 Eval] Spawned {len(self.spawned_obstacles)} post-survey obstacle(s).", flush=True)

            # Spawn orange supply crates in Gazebo as rover publishes deliveries.
            if args.mode == "rescue":
                events_path = self.folder / "rover/events.json"
                if events_path.is_file():
                    try:
                        evts = json.loads(events_path.read_text(encoding="utf-8"))
                        deliveries = [e for e in evts if e.get("type") == "delivery"]
                        while self.crates_spawned < len(deliveries):
                            d = deliveries[self.crates_spawned]
                            self.crates_spawned += 1
                            spawn_supply_crate(args.world_name, self.terrain, d, self.crates_spawned)
                    except Exception:
                        pass

            rover_pose = None
            if now_s - self._last_sample_s >= 0.095:
                with self._lock:
                    rover_pose = self._latest_rover
                    self._latest_rover = None
            if rover_pose is not None:
                self._last_sample_s = now_s
                x, y, z, roll_d, pitch_d, yaw_r = rover_pose
                self.trajectory.append(dict(
                    time_s=round(now_s, 3),
                    x=round(x, 4), y=round(y, 4), z=round(z, 4),
                    yaw_rad=round(yaw_r, 4),
                    roll_deg=round(roll_d, 2), pitch_deg=round(pitch_d, 2),
                ))

            done_file = self.folder / ("rover/rescue_status.json" if args.mode == "rescue" else "rover/nav_status.json")
            if done_file.is_file():
                summary = evaluate_run_offline(
                    run_dir=self.folder,
                    terrain=self.terrain,
                    mode=args.mode,
                    truth_override_path=args.truth_override or None,
                    trajectory_rows=self.trajectory,
                    collisions=self.collisions,
                    chassis_contacts=self.chassis_contacts,
                    spawned_obstacles=self.spawned_obstacles,
                    target_deliveries=args.target_deliveries,
                )
                print(
                    f"[ROS2 Eval] Finished: status={summary['status']} "
                    f"collisions={summary['collisions']} path={summary['path_length_m']}m",
                    flush=True,
                )
                rclpy.shutdown()

    node = _EvaluatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--mode", default="navigation", choices=("navigation", "rescue"))
    parser.add_argument("--world-name", default="search_d")
    parser.add_argument("--obstacles-json", default="")
    parser.add_argument("--truth-override", default="")
    parser.add_argument("--target-deliveries", type=int, default=2)
    args = parser.parse_args()
    run_ros2_evaluator(args)


if __name__ == "__main__":
    main()
