"""Move the kinematic survey platform and optionally run independent navigation evaluation."""
from pathlib import Path
import csv
import json
import math
import os
import sys

import numpy as np
from controller import Supervisor
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "controllers/disaster_environment"))
from motion import path_pose
sys.path.insert(0, str(ROOT / "controllers/uav_mapper"))
from survey_refinement import oblique_pose
sys.path.insert(0, str(ROOT / "evaluation"))
from metrics import height_at, look_at_rotation
from search_metrics import match_reports, physical_delivery_ok


def _spawn_test_obstacles(simulation, terrain, obstacles):
    """Inject post-survey physical obstacles into Webots. Rover gets zero notice."""
    children = simulation.getRoot().getField("children")
    spawned = []
    for index, item in enumerate(obstacles):
        x, y = map(float, item["xy_enu_m"])
        sx, sy, sz = map(float, item["size_m"])
        yaw = float(item.get("yaw_rad", 0.0))
        color = item.get("color_rgb", [0.46, 0.34, 0.24])
        ground_z = height_at(terrain, x, y)
        # Sink slightly (-0.03 m) so obstacle seats solidly on sloped ground.
        cz = ground_z + 0.5 * sz - 0.03
        name = item.get("name", f"injected_obstacle_{index + 1}")
        vrml = (
            f'DEF INJECTED_OBSTACLE_{index + 1} Solid {{ '
            f'translation {x} {y} {cz} rotation 0 0 1 {yaw} name "{name}" '
            f'children [ Shape {{ '
            f'appearance PBRAppearance {{ baseColor {color[0]} {color[1]} {color[2]} roughness 0.92 metalness 0 }} '
            f'geometry Box {{ size {sx} {sy} {sz} }} }} ] '
            f'boundingObject Box {{ size {sx} {sy} {sz} }} }}'
        )
        children.importMFNodeFromString(-1, vrml)
        spawned.append(dict(
            name=name,
            kind=item.get("kind", "debris"),
            xy_enu_m=[x, y],
            ground_z_m=round(float(ground_z), 3),
            size_m=[sx, sy, sz],
            yaw_rad=yaw,
        ))
    return spawned


def _orientation_rpy_deg(orient):
    """Extract (roll_deg, pitch_deg, yaw_rad, up_z) from Webots 3x3 row-major matrix."""
    mat = np.asarray(orient, dtype=float).reshape(3, 3)
    roll, pitch, yaw = Rotation.from_matrix(mat).as_euler("xyz")
    return math.degrees(roll), math.degrees(pitch), float(yaw), float(mat[2, 2])


def _evaluate_navigation(simulation, rover, folder, terrain, deployed_at, spawned_obstacles):
    """Independent ground-truth evaluator for closed-loop rover navigation."""
    eval_dir = folder / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    route_doc = json.loads((folder / "route.json").read_text(encoding="utf-8"))
    target_id = os.environ.get("UAV_NAV_GOAL")
    routes = route_doc.get("routes", [route_doc])
    selected = next((r for r in routes if r["id"] == target_id), routes[0]) if target_id else routes[0]
    goal_xy = np.asarray(selected["goal_xy_m"], dtype=float)
    planned_length = selected.get("route_length_m")

    world_min, world_max = terrain.get("bounds_xy", [[-42.0, -30.0], [42.0, 30.0]])
    period_ms = 100
    dt = period_ms / 1000.0
    rover.enableContactPointsTracking(period_ms, True)
    rover_id = rover.getId()
    rover_data_field = rover.getField("customData")

    nav_budget_s = float(os.environ.get("UAV_NAV_TIMEOUT_S", "240"))
    drive_start_s = simulation.getTime()
    deadline_s = drive_start_s + nav_budget_s + 8.0
    nav_status_file = folder / "rover/nav_status.json"

    prev_xy = None
    path_length = 0.0
    stationary_s = 0.0
    collisions = 0
    chassis_contacts = 0
    in_contact = False
    in_chassis_contact = False
    collision_events = []
    unsafe_state = False
    min_goal_err = float("inf")
    max_abs_roll_deg = 0.0
    max_abs_pitch_deg = 0.0
    loc_errors = []
    arrival_time_s = None
    settled_after_status_s = None

    with (eval_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "time_s", "x", "y", "z", "yaw_rad", "speed_mps", "dist_to_goal_m",
            "est_x", "est_y", "loc_error_m", "obstacle_contact", "chassis_contact",
            "total_collisions", "path_length_m", "stationary_s", "roll_deg", "pitch_deg",
        ])

        while simulation.step(period_ms) != -1:
            now = simulation.getTime()
            pos = rover.getPosition()
            vel = rover.getVelocity()
            orient = rover.getOrientation()
            roll_deg, pitch_deg, yaw_rad, up_z = _orientation_rpy_deg(orient)
            max_abs_roll_deg = max(max_abs_roll_deg, abs(roll_deg))
            max_abs_pitch_deg = max(max_abs_pitch_deg, abs(pitch_deg))

            xy = np.asarray(pos[:2], dtype=float)
            if prev_xy is not None:
                path_length += float(np.linalg.norm(xy - prev_xy))
            prev_xy = xy

            speed = math.hypot(vel[0], vel[1])
            yaw_rate = abs(vel[5]) if len(vel) >= 6 else 0.0
            dist_to_goal = float(np.linalg.norm(xy - goal_xy))
            min_goal_err = min(min_goal_err, dist_to_goal)

            is_still = speed < 0.03 and yaw_rate < 0.08
            stationary_s = stationary_s + dt if is_still else 0.0

            unsafe_state = unsafe_state or up_z < 0.5 or not (
                world_min[0] < pos[0] < world_max[0] and world_min[1] < pos[1] < world_max[1]
            )

            # Ignore transient contacts during the first 1.0 s right after deployment.
            contacts_now = rover.getContactPoints(True) if now > deployed_at + 1.0 else []
            obs_hit = None
            chassis_hit = False
            for point in contacts_now:
                gx, gy, gz = point.point
                ground_z = height_at(terrain, gx, gy)
                if gz > ground_z + 0.055:
                    obs_hit = (gx, gy, gz, ground_z)
                if getattr(point, "node_id", None) == rover_id:
                    chassis_hit = True

            contact = obs_hit is not None
            if contact and not in_contact:
                collisions += 1
                collision_events.append(dict(
                    time_s=round(now, 3),
                    xy_m=[round(obs_hit[0], 3), round(obs_hit[1], 3)],
                    contact_z_m=round(obs_hit[2], 3),
                    ground_z_m=round(obs_hit[3], 3),
                    height_above_ground_m=round(obs_hit[2] - obs_hit[3], 3),
                ))
            in_contact = contact

            if chassis_hit and not in_chassis_contact:
                chassis_contacts += 1
            in_chassis_contact = chassis_hit

            if dist_to_goal <= 0.45 and stationary_s >= 1.5 and arrival_time_s is None and collisions == 0 and not unsafe_state:
                arrival_time_s = now

            try:
                rover_msg = json.loads(rover_data_field.getSFString())
            except json.JSONDecodeError:
                rover_msg = {}
            est = rover_msg.get("pose_est_xy", [None, None])
            loc_err = float(np.linalg.norm(xy - np.asarray(est, dtype=float))) if est[0] is not None else None
            if loc_err is not None:
                loc_errors.append(loc_err)

            writer.writerow([
                round(now, 3), round(pos[0], 4), round(pos[1], 4), round(pos[2], 4),
                round(yaw_rad, 4), round(speed, 4), round(dist_to_goal, 4),
                est[0], est[1], round(loc_err, 4) if loc_err is not None else "",
                int(contact), int(chassis_hit), collisions, round(path_length, 4),
                round(stationary_s, 3), round(roll_deg, 2), round(pitch_deg, 2),
            ])
            csv_file.flush()

            if round(now * 10) % 10 == 0:
                simulation.setLabel(
                    0,
                    f"UAV-NAV | {selected['id']} | dist {dist_to_goal:.2f} m | v {speed:.2f} m/s | col {collisions}",
                    0.02, 0.025, 0.04, 0xFFFFFF, 0, "Arial",
                )
                simulation.setLabel(
                    1,
                    f"state {rover_msg.get('state', 'init')} | replans {rover_msg.get('replan_count', 0)}",
                    0.02, 0.075, 0.032, 0xFFE0A0, 0, "Arial",
                )

            if nav_status_file.is_file():
                if settled_after_status_s is None:
                    settled_after_status_s = now
                elif now - settled_after_status_s >= 0.3:
                    break
            if now >= deadline_s:
                break

    rover_status = (json.loads(nav_status_file.read_text(encoding="utf-8"))
                    if nav_status_file.is_file() else {})
    end_time_s = simulation.getTime()
    final_pos = rover.getPosition()
    final_vel = rover.getVelocity()
    final_speed = math.hypot(final_vel[0], final_vel[1])
    final_goal_err = float(np.linalg.norm(np.asarray(final_pos[:2]) - goal_xy))
    settled_truth = stationary_s >= 1.5 and final_speed < 0.03
    arrived_truth = bool(final_goal_err <= 0.45 and settled_truth and collisions == 0 and not unsafe_state)

    if arrived_truth:
        overall_status = "ARRIVED"
        failure_reason = None
    elif rover_status.get("status") == "STOPPED_SAFE" and collisions == 0 and not unsafe_state and settled_truth:
        overall_status = "STOPPED_SAFE"
        failure_reason = rover_status.get("reason") or "STOPPED_SAFE_UNSPECIFIED"
    else:
        overall_status = "FAILED" if (collisions > 0 or unsafe_state) else "TIMEOUT"
        reasons = []
        if rover_status.get("reason"):
            reasons.append(rover_status["reason"])
        if collisions > 0:
            reasons.append(f"COLLISIONS={collisions}")
        if unsafe_state:
            reasons.append("UNSAFE_STATE")
        if not settled_truth:
            reasons.append(f"NOT_SETTLED(stationary_s={stationary_s:.2f}, speed={final_speed:.3f})")
        if not arrived_truth and final_goal_err > 0.45 and not rover_status.get("reason"):
            reasons.append(f"GOAL_NOT_REACHED(error={final_goal_err:.2f}m)")
        failure_reason = "; ".join(reasons) or "UNKNOWN_FAILURE"

    summary = dict(
        evaluation_only=True,
        planner=selected.get("planner", os.environ.get("UAV_NAV_PLANNER", "astar")),
        goal_id=selected["id"],
        goal_label=selected.get("label", selected["id"]),
        goal_xy_m=goal_xy.tolist(),
        status=overall_status,
        arrived=arrived_truth,
        settled=settled_truth,
        deployment_time_s=round(float(deployed_at), 3),
        drive_start_time_s=round(float(rover_status.get("drive_start_s", drive_start_s)), 3),
        arrival_time_s=round(float(arrival_time_s), 3) if arrival_time_s is not None else None,
        travel_duration_s=(round(float(arrival_time_s - rover_status.get("drive_start_s", drive_start_s)), 3)
                           if arrival_time_s is not None else None),
        end_time_s=round(float(end_time_s), 3),
        final_position_xy_m=[round(float(final_pos[0]), 4), round(float(final_pos[1]), 4)],
        final_goal_error_m=round(final_goal_err, 4),
        min_goal_error_m=round(float(min_goal_err), 4),
        final_speed_mps=round(final_speed, 4),
        settled_duration_s=round(stationary_s, 3),
        planned_route_length_m=round(float(planned_length), 4) if planned_length is not None else None,
        actual_path_length_m=round(path_length, 4),
        collisions=collisions,
        chassis_contacts=chassis_contacts,
        collision_events=collision_events,
        unsafe_state=unsafe_state,
        max_abs_roll_deg=round(max_abs_roll_deg, 2),
        max_abs_pitch_deg=round(max_abs_pitch_deg, 2),
        failure_reason=failure_reason,
        rover_reported_status=rover_status.get("status"),
        rover_reported_reason=rover_status.get("reason"),
        replan_count=rover_status.get("replan_count", 0),
        lidar_detection_frames=rover_status.get("lidar_detection_frames", 0),
        rgbd_detection_frames=rover_status.get("rgbd_detection_frames", 0),
        drop_detection_frames=rover_status.get("drop_detection_frames", 0),
        onboard_blocked_cells=rover_status.get("onboard_blocked_cells", []),
        injected_obstacles=spawned_obstacles,
        vctp_audit=selected.get("vctp_audit") or selected.get("astar_vctp_audit"),
        igha_stats=selected.get("igha_vctp", {}).get("stats") if "igha_vctp" in selected else None,
        dual_route=selected.get("dual_route"),
        localization_error_m=(dict(
            median=round(float(np.median(loc_errors)), 4),
            p90=round(float(np.quantile(loc_errors, 0.9)), 4),
            max=round(float(np.max(loc_errors)), 4),
        ) if loc_errors else None),
    )
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        simulation.exportImage(str(eval_dir / "final_scene.png"), 90)
    except Exception:
        pass
    print(f"[Survey Eval] status={summary['status']} arrived={summary['arrived']} "
          f"settled={summary['settled']} collisions={summary['collisions']} "
          f"goal_err={summary['final_goal_error_m']:.3f}m reason={summary['failure_reason']}",
          flush=True)
    return summary


def _evaluate_rescue(simulation, rover, folder, terrain, deployed_at, spawned_obstacles):
    """Independent ground-truth evaluator for air-to-ground victim rescue & supply delivery."""
    eval_dir = folder / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    common = json.loads((ROOT / "config/search/common.json").read_text(encoding="utf-8"))
    override_file = eval_dir / "truth_override.json"
    if override_file.is_file():
        targets = json.loads(override_file.read_text(encoding="utf-8"))["victims"]
    else:
        targets = terrain["victims"]
    true_points = np.array([t["xy"] for t in targets], dtype=float)
    report_tol = float(terrain.get("report_tolerance_m", 1.0))
    delivery_tol = float(terrain.get("delivery_tolerance_m", 1.8))
    delivery_deadline_s = float(terrain.get("delivery_deadline_after_report_s", 120.0))
    initial_cargo = int(os.environ.get("UAV_RESCUE_MAX_VICTIMS", str(common["cargo_capacity"])))
    physical_cargo = initial_cargo

    world_min, world_max = terrain.get("bounds_xy", [[-42.0, -30.0], [42.0, 30.0]])
    period_ms = 100
    dt = period_ms / 1000.0
    rover.enableContactPointsTracking(period_ms, True)
    rover_id = rover.getId()
    rover_data_field = rover.getField("customData")

    nav_budget_s = float(os.environ.get("UAV_NAV_TIMEOUT_S", "360"))
    drive_start_s = simulation.getTime()
    deadline_s = drive_start_s + nav_budget_s + 10.0
    nav_status_file = folder / "rover/nav_status.json"

    prev_xy = None
    path_length = 0.0
    stationary_s = 0.0
    collisions = 0
    chassis_contacts = 0
    in_contact = False
    in_chassis_contact = False
    collision_events = []
    unsafe_state = False
    max_abs_roll_deg = 0.0
    max_abs_pitch_deg = 0.0
    loc_errors = []
    reports = []
    deliveries = []
    events_seen = set()
    settled_after_status_s = None

    def match_current(rep_list=None):
        recs = reports if rep_list is None else rep_list
        return match_reports(recs, targets, [0.0, 0.0], 0.0, report_tol)

    record_video = os.environ.get("UAV_RECORD_VIDEO") == "1"
    viewpoint = simulation.getFromDef("SCENE_VIEW") if record_video else None
    vp_pos = viewpoint.getField("position") if viewpoint is not None else None
    vp_ori = viewpoint.getField("orientation") if viewpoint is not None else None
    cam_eye = None
    cam_target = None
    video_frames_dir = eval_dir / "video_frames"
    video_meta_file = None
    video_frame_idx = 0
    video_step = 0
    if record_video:
        video_frames_dir.mkdir(parents=True, exist_ok=True)
        video_meta_file = (eval_dir / "video_meta.jsonl").open("w", encoding="utf-8")
        if vp_pos is not None and vp_ori is not None:
            p0 = rover.getPosition()
            cam_eye = np.array([p0[0] - 4.2, p0[1] - 3.6, max(p0[2] + 3.2, height_at(terrain, p0[0] - 4.2, p0[1] - 3.6) + 2.0)], dtype=float)
            cam_target = np.array([p0[0] + 0.5, p0[1], p0[2] + 0.25], dtype=float)
            vp_pos.setSFVec3f(cam_eye.tolist())
            vp_ori.setSFRotation(look_at_rotation(cam_eye.tolist(), cam_target.tolist()))
        print(f"[Rescue Eval] Capturing follow-camera frames -> {video_frames_dir}", flush=True)

    with (eval_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "time_s", "x", "y", "z", "yaw_rad", "speed_mps", "nearest_victim_m",
            "est_x", "est_y", "loc_error_m", "obstacle_contact", "chassis_contact",
            "total_collisions", "path_length_m", "stationary_s", "roll_deg", "pitch_deg",
        ])

        while simulation.step(period_ms) != -1:
            now = simulation.getTime()
            pos = rover.getPosition()
            vel = rover.getVelocity()
            orient = rover.getOrientation()
            roll_deg, pitch_deg, yaw_rad, up_z = _orientation_rpy_deg(orient)
            max_abs_roll_deg = max(max_abs_roll_deg, abs(roll_deg))
            max_abs_pitch_deg = max(max_abs_pitch_deg, abs(pitch_deg))

            xy = np.asarray(pos[:2], dtype=float)
            if prev_xy is not None:
                path_length += float(np.linalg.norm(xy - prev_xy))
            prev_xy = xy

            speed = math.hypot(vel[0], vel[1])
            yaw_rate = abs(vel[5]) if len(vel) >= 6 else 0.0
            nearest_v = float(np.min(np.linalg.norm(true_points - xy[None, :], axis=1)))

            is_still = speed < 0.03 and yaw_rate < 0.08
            stationary_s = stationary_s + dt if is_still else 0.0

            unsafe_state = unsafe_state or up_z < 0.5 or not (
                world_min[0] < pos[0] < world_max[0] and world_min[1] < pos[1] < world_max[1]
            )

            contacts_now = rover.getContactPoints(True) if now > deployed_at + 1.0 else []
            obs_hit = None
            chassis_hit = False
            for point in contacts_now:
                gx, gy, gz = point.point
                ground_z = height_at(terrain, gx, gy)
                if gz > ground_z + 0.055:
                    obs_hit = (gx, gy, gz, ground_z)
                if getattr(point, "node_id", None) == rover_id:
                    chassis_hit = True

            contact = obs_hit is not None
            if contact and not in_contact:
                collisions += 1
                collision_events.append(dict(
                    time_s=round(now, 3),
                    xy_m=[round(obs_hit[0], 3), round(obs_hit[1], 3)],
                    contact_z_m=round(obs_hit[2], 3),
                    ground_z_m=round(obs_hit[3], 3),
                    height_above_ground_m=round(obs_hit[2] - obs_hit[3], 3),
                ))
            in_contact = contact

            if chassis_hit and not in_chassis_contact:
                chassis_contacts += 1
            in_chassis_contact = chassis_hit

            try:
                rover_msg = json.loads(rover_data_field.getSFString())
            except json.JSONDecodeError:
                rover_msg = {}
            est = rover_msg.get("pose_est_xy", [None, None])
            loc_err = float(np.linalg.norm(xy - np.asarray(est, dtype=float))) if est[0] is not None else None
            if loc_err is not None:
                loc_errors.append(loc_err)

            for event in rover_msg.get("events", []):
                key = (event.get("type"), event.get("track_id"), event.get("timestamp_s"))
                if key in events_seen:
                    continue
                events_seen.add(key)
                if event.get("type") == "report":
                    reports.append(event.copy())
                elif event.get("type") == "delivery":
                    assignment = match_current()
                    rep_idx = next((i for i, r in enumerate(reports) if r.get("track_id") == event.get("track_id")), None)
                    target_idx = assignment.get(rep_idx)
                    dist_m = (float(np.linalg.norm(xy - true_points[target_idx]))
                              if target_idx is not None else float("inf"))
                    already = {d["victim_id"] for d in deliveries if d["valid"]}
                    valid = physical_delivery_ok(
                        target_idx is not None, stationary_s, dist_m, physical_cargo, delivery_tol,
                    )
                    if target_idx is not None and targets[target_idx]["id"] in already:
                        valid = False
                    physical_cargo = max(0, physical_cargo - 1)
                    rep_ts = reports[rep_idx]["timestamp_s"] if rep_idx is not None else None
                    deliveries.append(dict(
                        event,
                        valid=bool(valid),
                        victim_id=targets[target_idx]["id"] if target_idx is not None else None,
                        true_distance_m=round(dist_m, 4) if math.isfinite(dist_m) else None,
                        stationary_s=round(stationary_s, 3),
                        actual_drop_xy=[round(float(pos[0]), 4), round(float(pos[1]), 4)],
                        report_timestamp_s=rep_ts,
                        deadline_met=bool(rep_ts is not None and now <= rep_ts + delivery_deadline_s),
                    ))
                    # Spawn visual orange rescue supply crate beside the rover.
                    bx = float(pos[0]) + 0.45 * math.cos(yaw_rad)
                    by = float(pos[1]) + 0.45 * math.sin(yaw_rad)
                    bz = float(height_at(terrain, bx, by)) + 0.055
                    simulation.getRoot().getField("children").importMFNodeFromString(
                        -1,
                        f'Pose {{ translation {bx} {by} {bz} rotation 0 0 1 {yaw_rad} children [ Shape {{ '
                        f'appearance PBRAppearance {{ baseColor 0.95 0.48 0.06 roughness 0.72 metalness 0 }} '
                        f'geometry Box {{ size 0.18 0.14 0.10 }} }} ] }}',
                    )
                    print(
                        f"[Rescue Eval] Delivery track={event.get('track_id')} valid={valid} "
                        f"victim_id={targets[target_idx]['id'] if target_idx is not None else None} "
                        f"dist={dist_m:.3f}m still={stationary_s:.2f}s",
                        flush=True,
                    )

            writer.writerow([
                round(now, 3), round(pos[0], 4), round(pos[1], 4), round(pos[2], 4),
                round(yaw_rad, 4), round(speed, 4), round(nearest_v, 4),
                est[0], est[1], round(loc_err, 4) if loc_err is not None else "",
                int(contact), int(chassis_hit), collisions, round(path_length, 4),
                round(stationary_s, 3), round(roll_deg, 2), round(pitch_deg, 2),
            ])
            csv_file.flush()

            if round(now * 10) % 10 == 0:
                valid_drops = len({d["victim_id"] for d in deliveries if d["valid"]})
                eff_reps = len(match_current())
                simulation.setLabel(
                    0,
                    f"UAV-RESCUE | reports {eff_reps}/{len(targets)} | drops {valid_drops}/{initial_cargo} | col {collisions}",
                    0.02, 0.025, 0.04, 0xFFFFFF, 0, "Arial",
                )
                simulation.setLabel(
                    1,
                    f"state {rover_msg.get('state', 'init')} | target #{rover_msg.get('active_victim_id', '-')} | cargo {rover_msg.get('cargo', '?')}",
                    0.02, 0.075, 0.032, 0xFFE0A0, 0, "Arial",
                )

            if record_video and vp_pos is not None and vp_ori is not None:
                st = rover_msg.get("state", "TRACKING")
                # Smooth 3/4 rear-top chase camera; shift slightly wider on verify/deliver to frame rover + victim + crate.
                if st in ("FACE_VICTIM", "VERIFY_AND_DELIVER", "DELIVERED"):
                    back_d, side_d, up_d, ahead_d = 2.6, -1.45, 2.05, 0.85
                else:
                    back_d, side_d, up_d, ahead_d = 3.3, -1.05, 2.25, 0.70
                cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
                des_eye = np.array([
                    pos[0] - back_d * cy - side_d * sy,
                    pos[1] - back_d * sy + side_d * cy,
                    pos[2] + up_d,
                ], dtype=float)
                des_eye[2] = max(des_eye[2], float(height_at(terrain, des_eye[0], des_eye[1])) + 1.35)
                des_tgt = np.array([
                    pos[0] + ahead_d * cy,
                    pos[1] + ahead_d * sy,
                    pos[2] + 0.25,
                ], dtype=float)
                alpha = 0.14
                cam_eye = (1.0 - alpha) * cam_eye + alpha * des_eye
                cam_target = (1.0 - alpha) * cam_target + alpha * des_tgt
                vp_pos.setSFVec3f(cam_eye.tolist())
                vp_ori.setSFRotation(look_at_rotation(cam_eye.tolist(), cam_target.tolist()))
                video_step += 1
                if video_step % 2 == 0:
                    fpath = video_frames_dir / f"{video_frame_idx:05d}.jpg"
                    try:
                        simulation.exportImage(str(fpath), 86)
                        video_meta_file.write(json.dumps(dict(
                            frame=video_frame_idx,
                            time_s=round(now, 2),
                            x=round(float(pos[0]), 3),
                            y=round(float(pos[1]), 3),
                            state=st,
                            delivered=len({d["victim_id"] for d in deliveries if d["valid"]}),
                            reports=len(match_current()),
                            speed=round(speed, 2),
                        )) + "\n")
                        video_frame_idx += 1
                    except Exception:
                        pass

            if nav_status_file.is_file():
                if settled_after_status_s is None:
                    settled_after_status_s = now
                elif now - settled_after_status_s >= (1.5 if record_video else 0.4):
                    break
            if now >= deadline_s:
                break

    if video_meta_file is not None:
        video_meta_file.close()
        print(f"[Rescue Eval] Captured {video_frame_idx} video frames.", flush=True)

    rover_status = (json.loads(nav_status_file.read_text(encoding="utf-8"))
                    if nav_status_file.is_file() else {})
    end_time_s = simulation.getTime()
    final_pos = rover.getPosition()
    matches = match_current()
    report_errors = [
        round(float(np.linalg.norm(np.asarray(reports[i]["xy"], dtype=float) - true_points[j])), 4)
        for i, j in matches.items()
    ]
    valid_deliveries = [d for d in deliveries if d["valid"]]
    delivered_ids = sorted({d["victim_id"] for d in valid_deliveries})
    target_deliveries = min(len(targets), initial_cargo)
    valid_cross_view = sum({"rover", "uav"} <= set(reports[i].get("sources", [])) for i in matches)
    false_reports = len(reports) - len(matches)
    mission_success = bool(
        len(delivered_ids) >= target_deliveries
        and len(matches) >= target_deliveries
        and false_reports == 0
        and collisions == 0
        and not unsafe_state
    )

    if mission_success:
        overall_status = "MISSION_SUCCESS"
        failure_reason = None
    elif rover_status.get("status") == "STOPPED_SAFE" and collisions == 0 and not unsafe_state:
        overall_status = "STOPPED_SAFE"
        failure_reason = rover_status.get("reason") or "STOPPED_SAFE"
    else:
        overall_status = "FAILED" if (collisions > 0 or unsafe_state) else "PARTIAL_OR_TIMEOUT"
        reasons = []
        if rover_status.get("reason"):
            reasons.append(rover_status["reason"])
        if len(delivered_ids) < target_deliveries:
            reasons.append(f"DELIVERED={len(delivered_ids)}/{target_deliveries}")
        if false_reports > 0:
            reasons.append(f"FALSE_REPORTS={false_reports}")
        if collisions > 0:
            reasons.append(f"COLLISIONS={collisions}")
        if unsafe_state:
            reasons.append("UNSAFE_STATE")
        failure_reason = "; ".join(reasons) or "INCOMPLETE_RESCUE"

    # Aerial candidate accuracy audit (evaluation-only comparison against D.json truth).
    aerial_audit = []
    aerial_file = folder / "aerial_victims.json"
    if aerial_file.is_file():
        aerial_doc = json.loads(aerial_file.read_text(encoding="utf-8"))
        for cand in aerial_doc.get("candidates", []):
            cxy = np.asarray(cand["mean_enu_m"], dtype=float)
            dists = np.linalg.norm(true_points - cxy[None, :], axis=1)
            j = int(np.argmin(dists))
            aerial_audit.append(dict(
                candidate_id=cand["id"],
                detected_xy_m=cand["mean_enu_m"],
                nearest_truth_victim_id=targets[j]["id"],
                error_m=round(float(dists[j]), 4),
                within_report_tol=bool(dists[j] <= report_tol),
                views=cand["views"],
            ))

    summary = dict(
        evaluation_only=True,
        mode="rescue",
        detector=rover_status.get("detector", os.environ.get("UAV_RESCUE_DETECTOR", "surrogate_rgbd")),
        planner=rover_status.get("planner", os.environ.get("UAV_NAV_PLANNER", "igha_vctp")),
        status=overall_status,
        mission_success=mission_success,
        all_targets_delivered=bool(len(delivered_ids) == len(targets)),
        truth_targets=len(targets),
        cargo_capacity=initial_cargo,
        aerial_candidates_total=len(aerial_audit),
        aerial_candidates_audit=aerial_audit,
        effective_reports=len(matches),
        total_reports=len(reports),
        false_reports=false_reports,
        valid_cross_view_reports=valid_cross_view,
        localization_errors_m=report_errors,
        localization_rmse_m=round(math.sqrt(float(np.mean(np.square(report_errors)))), 4) if report_errors else None,
        delivered_people=len(delivered_ids),
        delivered_victim_ids=delivered_ids,
        deliveries=deliveries,
        deployment_time_s=round(float(deployed_at), 3),
        drive_start_time_s=round(float(rover_status.get("drive_start_s", drive_start_s)), 3),
        end_time_s=round(float(end_time_s), 3),
        total_rescue_duration_s=round(float(end_time_s - rover_status.get("drive_start_s", drive_start_s)), 3),
        final_position_xy_m=[round(float(final_pos[0]), 4), round(float(final_pos[1]), 4)],
        actual_path_length_m=round(path_length, 4),
        collisions=collisions,
        chassis_contacts=chassis_contacts,
        collision_events=collision_events,
        unsafe_state=unsafe_state,
        max_abs_roll_deg=round(max_abs_roll_deg, 2),
        max_abs_pitch_deg=round(max_abs_pitch_deg, 2),
        replan_count=rover_status.get("replan_count", 0),
        lidar_detection_frames=rover_status.get("lidar_detection_frames", 0),
        rgbd_detection_frames=rover_status.get("rgbd_detection_frames", 0),
        drop_detection_frames=rover_status.get("drop_detection_frames", 0),
        failure_reason=failure_reason,
        rover_legs=rover_status.get("legs", []),
        injected_obstacles=spawned_obstacles,
        rover_pose_error_m=(dict(
            median=round(float(np.median(loc_errors)), 4),
            p90=round(float(np.quantile(loc_errors, 0.9)), 4),
            max=round(float(np.max(loc_errors)), 4),
        ) if loc_errors else None),
    )
    (eval_dir / "events.json").write_text(json.dumps(dict(reports=reports, deliveries=deliveries), indent=2, ensure_ascii=False), encoding="utf-8")
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        simulation.exportImage(str(eval_dir / "final_scene.png"), 90)
    except Exception:
        pass
    print(
        f"[Rescue Eval] status={summary['status']} success={summary['mission_success']} "
        f"reports={summary['effective_reports']}/{summary['total_reports']} "
        f"cross_view={summary['valid_cross_view_reports']} "
        f"delivered={summary['delivered_people']}/{target_deliveries} "
        f"collisions={summary['collisions']} rmse={summary['localization_rmse_m']}",
        flush=True,
    )
    return summary


def main():
    folder = ROOT / "runs" / os.environ["UAV_MAPPING_RUN"]
    config = json.loads((folder / "map_config.json").read_text(encoding="utf-8"))
    flight = config["flight"]
    end = float(os.environ["UAV_MAPPING_FLIGHT_END"])
    nav_enabled = os.environ.get("UAV_NAV_ENABLE") == "1"

    simulation = Supervisor()
    drone = simulation.getFromDef("OBSERVER_DRONE")
    translation, rotation = (drone.getField(field) for field in ("translation", "rotation"))
    rover = simulation.getFromDef("ROVER")
    rover_position = rover.getField("translation")
    rover_orientation = rover.getField("rotation")
    fixture_data = json.loads(simulation.getCustomData())
    deployment = fixture_data["deployment_xyz_m"]
    injected_spec = fixture_data.get("injected_obstacles", [])
    deployed_at = None
    spawned_obstacles = []
    survey_file = folder / "oblique_survey.json"
    supplement = None
    terrain = json.loads((ROOT / "evaluation/search/D.json").read_text(encoding="utf-8")) if nav_enabled else None

    while simulation.step(200) != -1:
        now = simulation.getTime()
        if supplement is None and survey_file.is_file():
            supplement = json.loads(survey_file.read_text(encoding="utf-8"))
        # Arrive during the first half of each second, then let the sensors
        # acquire the next sample while both the camera and GPS are stationary.
        whole = math.floor(now)
        progress = min(whole + min(2 * (now - whole), 1), end)
        xy, heading = path_pose(flight["waypoints_xy_enu_m"], flight["speed_mps"], progress)
        if supplement is not None and supplement["stations"] and now >= supplement["start_s"]:
            xyz, yaw, pitch = oblique_pose(supplement, now)
            translation.setSFVec3f(xyz.tolist())
            vector = Rotation.from_euler("ZY", [yaw, pitch]).as_rotvec()
            angle = math.sqrt(vector @ vector)
            axis = (vector / angle).tolist() if angle > 1e-9 else [0.0, 0.0, 1.0]
            rotation.setSFRotation([*axis, angle])
        else:
            translation.setSFVec3f([*map(float, xy), flight["altitude_enu_m"]])
            rotation.setSFRotation([0.0, 0.0, 1.0, heading])
        if deployed_at is None and (folder / "uav_map.npz").is_file():
            rover_position.setSFVec3f(deployment)
            rover_orientation.setSFRotation([0.0, 0.0, 1.0, 0.0])
            rover.resetPhysics()
            deployed_at = now
            if nav_enabled and injected_spec:
                spawned_obstacles = _spawn_test_obstacles(simulation, terrain, injected_spec)
                print(f"[Survey] Spawned {len(spawned_obstacles)} post-survey obstacle(s).", flush=True)
        if deployed_at is not None and now - deployed_at >= 1.6:
            rover.getField("customData").setSFString("deployed")
        if (folder / "route.json").is_file():
            if not nav_enabled:
                simulation.simulationQuit(0)
                break
            if os.environ.get("UAV_RESCUE_MODE") == "1":
                _evaluate_rescue(simulation, rover, folder, terrain, deployed_at, spawned_obstacles)
            else:
                _evaluate_navigation(simulation, rover, folder, terrain, deployed_at, spawned_obstacles)
            if os.environ.get("UAV_NAV_KEEP_OPEN") != "1":
                simulation.simulationQuit(0)
            break
        if now > (supplement["end_s"] if supplement else end) + 20:
            print("[Survey] Timed out waiting for the map/route handoff.", flush=True)
            simulation.simulationQuit(1)
            break


if __name__ == "__main__":
    main()
