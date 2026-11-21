"""Read the aerial map, plan from GPS, and optionally drive closed-loop along the route."""
from collections import deque
from pathlib import Path
import json
import math
import os
import sys

import numpy as np
from controller import Robot

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "controllers/uav_mapper"))
from survey_map import load_prior
from grid_route import direct_line_cells, plan, plan_dual
from igha_star import plan_igha_star
from route_tracker import (
    OnboardObstacleMap,
    RouteNavigationController,
    RoverPoseEstimator,
    extract_lidar_obstacles,
    extract_rgbd_obstacles,
    wheel_velocities_rad_s,
    wrap_angle,
)
from rescue_mission import confirm_victim_onboard, plan_next_victim
from vctp import FineTerrainSurface, VctpChecker


def route_result(grid, states, layers, start, goal, radius, terrain_aware,
                 vctp=None, planner_mode="astar", start_yaw=0.0):
    result = dict(status="NO_ROUTE", goal_xy_m=goal,
                  straight_line=direct_line_cells(grid, states, start, goal))
    goal_cell = grid.cell(goal)
    if grid.contains(goal_cell):
        result["goal_state"] = int(states[goal_cell])

    def measure(path):
        slope = np.asarray([layers["slope_deg"][grid.cell(xy)] for xy in path])
        known = slope[np.isfinite(slope)]
        return dict(length_m=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
                    mean_slope_deg=float(known.mean()) if known.size else None,
                    max_slope_deg=float(known.max()) if known.size else None,
                    slope_supported_waypoints=int(known.size))

    cost = layers["planning_cost"] if terrain_aware else None
    dual = plan_dual(grid, states, start, goal, radius, cost)
    result["dual_route"] = {
        key: dual[key] for key in (
            "conservative_status", "conservative_length_m",
            "optimistic_status", "optimistic_length_m",
            "optimistic_waypoint_count", "unconfirmed_clearance_waypoints",
            "direct_unknown_waypoints", "corridor_unknown_cell_count",
            "changes_reachability", "potential_saving_m",
        )
    }

    try:
        route = plan(grid, states, start, goal, radius, cost)
        metrics = measure(route)
        result.update(status="PLANNED", planner="astar", route_xy_m=route.tolist(),
                      route_length_m=metrics["length_m"], terrain=metrics)
        if terrain_aware:
            baseline = plan(grid, states, start, goal, radius)
            result["distance_only"] = measure(baseline)
            result["distance_only_route_xy_m"] = baseline.tolist()
        if vctp is not None:
            result["astar_vctp_audit"] = vctp.audit_route(route)
    except ValueError as error:
        result["reason"] = str(error)

    if planner_mode == "igha_vctp" and vctp is not None:
        igha = plan_igha_star(grid, states, vctp, start, goal, radius, start_yaw=start_yaw)
        result["igha_vctp"] = {
            key: igha[key] for key in ("status", "reason", "route_length_m", "vctp", "stats")
            if key in igha
        }
        if igha["status"] == "PLANNED":
            result.update(
                status="PLANNED",
                planner="igha_vctp",
                astar_route_xy_m=result.get("route_xy_m"),
                astar_length_m=result.get("route_length_m"),
                route_xy_m=igha["route_xy_m"],
                route_xyyaw_m=igha["route_xyyaw_m"],
                motions=igha["motions"],
                route_length_m=igha["route_length_m"],
                vctp_audit=igha["vctp"],
            )
            result.pop("reason", None)
        else:
            result.update(status="NO_ROUTE", planner="igha_vctp", reason=igha.get("reason", "VCTP_NO_ROUTE"))
            result.pop("route_xy_m", None)
            result.pop("route_length_m", None)
    return result


def _write_json_atomic(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def drive_route(robot, folder, settings, grid, states, layers, selected_route):
    """Execute closed-loop tracking of a planned route using onboard sensors only."""
    rover_dir = folder / "rover"
    rover_dir.mkdir(parents=True, exist_ok=True)
    motors = [robot.getDevice(name) for name in ("left_motor::front", "right_motor::front")]
    for motor in motors:
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)

    if selected_route["status"] != "PLANNED":
        reason = f"INITIAL_ROUTE_UNAVAILABLE: {selected_route.get('reason', 'NO_ROUTE')}"
        status = dict(
            goal_id=selected_route["id"],
            goal_label=selected_route.get("label", selected_route["id"]),
            goal_xy_m=selected_route["goal_xy_m"],
            status="STOPPED_SAFE",
            reason=reason,
            drive_start_s=robot.getTime(),
            end_time_s=robot.getTime(),
            replan_count=0,
            lidar_detection_frames=0,
            rgbd_detection_frames=0,
            drop_detection_frames=0,
            routes_history=[],
            events=[dict(time_s=round(robot.getTime(), 3), type="INITIAL_NO_ROUTE", reason=reason)],
        )
        _write_json_atomic(rover_dir / "nav_status.json", status)
        _write_json_atomic(rover_dir / "events.json", status["events"])
        print(f"[Map rover] {reason}", flush=True)
        return

    period_ms = 100
    dt = period_ms / 1000.0
    gps = robot.getDevice("deployment_gps")
    imu = robot.getDevice("attitude")
    encoders = [robot.getDevice(name) for name in ("left_encoder", "right_encoder")]
    lidar = robot.getDevice("lidar")
    depth_cam = robot.getDevice("depth")
    rgb_cam = robot.getDevice("rgb")
    chase_cam = robot.getDevice("rover_chase_cam")
    for device in (gps, imu, *encoders, lidar, depth_cam, rgb_cam):
        device.enable(period_ms)
    if chase_cam is not None:
        chase_cam.enable(200)

    # Allow 3 ticks for onboard sensors to deliver valid frames at 10 Hz.
    init_gps = []
    init_rpy = None
    for _ in range(5):
        if robot.step(period_ms) == -1:
            return
        g = np.asarray(gps.getValues(), dtype=float)
        r = np.asarray(imu.getRollPitchYaw(), dtype=float)
        if np.isfinite(g).all() and np.isfinite(r).all():
            init_gps.append(g)
            init_rpy = r
        if len(init_gps) >= 3:
            break
    if not init_gps or init_rpy is None:
        return

    start_xyz = np.median(init_gps, axis=0)
    estimator = RoverPoseEstimator(start_xyz, init_rpy[2])
    onboard_map = OnboardObstacleMap(
        grid=grid,
        prior_states=states,
        ground_z_m=layers["ground_z_m"],
        planning_cost=layers.get("planning_cost") if "max_slope_deg" in settings else None,
        rover_radius_m=settings["rover_radius_m"],
    )
    experiment = None
    policy = os.environ.get("UAV_LOCAL_POLICY", "rule")
    record = os.environ.get("UAV_LOCAL_RECORD", "0") == "1"
    if policy != "rule" or record:
        from local_policy import LocalPolicyExperiment
        from policy_features import encode_rgbd

        experiment = LocalPolicyExperiment(
            policy, checkpoint=os.environ.get("UAV_LOCAL_CHECKPOINT") or None,
            record_to=rover_dir / "imitation" if record else None,
            expert_mix=float(os.environ.get("UAV_LOCAL_EXPERT_MIX", "0")),
            seed=int(os.environ.get("UAV_LOCAL_SEED", "17")),
        )
    controller = RouteNavigationController(
        onboard_map=onboard_map,
        initial_route_xy=selected_route["route_xy_m"],
        goal_xy=selected_route["goal_xy_m"],
        local_planner=experiment,
    )

    drive_start_s = robot.getTime()
    nav_budget_s = float(os.environ.get("UAV_NAV_TIMEOUT_S", "240"))
    nav_deadline_s = drive_start_s + nav_budget_s
    depth_h, depth_w = depth_cam.getHeight(), depth_cam.getWidth()
    depth_fov = depth_cam.getFov()
    num_layers = lidar.getNumberOfLayers()
    last_replan_seen = 0
    saved_stop_image = False
    tick = 0

    print(
        f"[Map rover] Driving goal '{selected_route['id']}' ({selected_route['route_length_m']:.2f} m) "
        f"using LiDAR + RGB-D + UAV prior.",
        flush=True,
    )

    with (rover_dir / "telemetry.jsonl").open("w", encoding="utf-8") as telemetry:
        try:
            while robot.step(period_ms) != -1:
                now = robot.getTime()
                tick += 1
                g = np.asarray(gps.getValues(), dtype=float)
                rpy = np.asarray(imu.getRollPitchYaw(), dtype=float)
                wheels = np.asarray([enc.getValue() for enc in encoders], dtype=float)
                pose_xyz, yaw = estimator.update(g, rpy, wheels, dt, grid, layers["ground_z_m"])

                layer_ranges = [lidar.getLayerRangeImage(layer) for layer in range(num_layers)]
                lidar_body, lidar_enu = extract_lidar_obstacles(
                    layer_ranges, pose_xyz, (rpy[0], rpy[1], yaw),
                    grid, layers["ground_z_m"], z_bias=estimator.z_bias_m,
                )

                depth_img = np.asarray(depth_cam.getRangeImage(), dtype=np.float32).reshape(depth_h, depth_w)
                rgbd_body, rgbd_enu, drop_count = extract_rgbd_obstacles(
                    depth_img, pose_xyz, (rpy[0], rpy[1], yaw),
                    grid, layers["ground_z_m"], z_bias=estimator.z_bias_m, fov=depth_fov,
                )

                v_cmd, w_cmd, info = controller.step(
                    now=now,
                    dt=dt,
                    pose_xyz=pose_xyz,
                    rpy=(rpy[0], rpy[1], yaw),
                    speed_mps=estimator.speed_mps,
                    yaw_rate_rad_s=estimator.yaw_rate_rad_s,
                    lidar_body_xy=lidar_body,
                    lidar_enu_xy=lidar_enu,
                    rgbd_body_xy=rgbd_body,
                    rgbd_enu_xy=rgbd_enu,
                    drop_count=drop_count,
                    rgbd_frame=encode_rgbd(rgb_cam.getImage(), depth_img) if experiment else None,
                )
                left_rad, right_rad = wheel_velocities_rad_s(v_cmd, w_cmd)
                motors[0].setVelocity(left_rad)
                motors[1].setVelocity(right_rad)
                if experiment:
                    experiment.observe_executed(v_cmd, w_cmd, now)

                # Publish onboard estimate to Supervisor for truth-vs-estimate audit only.
                robot.setCustomData(json.dumps(dict(
                    nav_active=True,
                    goal_id=selected_route["id"],
                    pose_est_xy=[round(float(pose_xyz[0]), 4), round(float(pose_xyz[1]), 4)],
                    yaw_est_rad=round(float(yaw), 4),
                    state=controller.state,
                    status=controller.report.status,
                    reason=controller.report.reason,
                    replan_count=controller.report.replan_count,
                )))

                if controller.report.replan_count > last_replan_seen:
                    last_replan_seen = controller.report.replan_count
                    rgb_cam.saveImage(str(rover_dir / f"replan_{last_replan_seen:02d}.jpg"), 90)
                    print(f"[Map rover] Replan #{last_replan_seen} at t={now:.1f}s "
                          f"xy=({pose_xyz[0]:.2f}, {pose_xyz[1]:.2f})", flush=True)

                if controller.state == "STOPPING_SAFE" and not saved_stop_image:
                    saved_stop_image = True
                    rgb_cam.saveImage(str(rover_dir / "stop_view.jpg"), 90)
                    print(f"[Map rover] Safe stop triggered at t={now:.1f}s: "
                          f"{controller.pending_stop_reason}", flush=True)

                if tick % 2 == 0 or controller.report.status != "RUNNING":
                    telemetry.write(json.dumps(dict(
                        time_s=round(now, 3),
                        pose_est_xyz=[round(float(x), 4) for x in pose_xyz],
                        rpy_rad=[round(float(x), 4) for x in rpy],
                        speed_est_mps=round(float(estimator.speed_mps), 4),
                        cmd_vw=[round(float(v_cmd), 4), round(float(w_cmd), 4)],
                        **{k: v for k, v in info.items() if k != "lookahead_xy"},
                    )) + "\n")
                    telemetry.flush()

                if controller.report.status in ("ARRIVED", "STOPPED_SAFE"):
                    rgb_cam.saveImage(str(rover_dir / f"final_{controller.report.status.lower()}.jpg"), 90)
                    break

                if now >= nav_deadline_s:
                    controller.report.status = "TIMEOUT"
                    controller.report.reason = (
                        f"NAV_TIMEOUT: exceeded {nav_budget_s:.0f} s budget "
                        f"(dist_to_goal={info['dist_to_goal_m']:.2f} m)"
                    )
                    rgb_cam.saveImage(str(rover_dir / "final_timeout.jpg"), 90)
                    break
        finally:
            for motor in motors:
                motor.setVelocity(0.0)

    status_payload = dict(
        goal_id=selected_route["id"],
        goal_label=selected_route.get("label", selected_route["id"]),
        goal_xy_m=selected_route["goal_xy_m"],
        status=controller.report.status,
        reason=controller.report.reason or None,
        drive_start_s=round(float(drive_start_s), 3),
        end_time_s=round(float(robot.getTime()), 3),
        final_pose_est_xy=[round(float(estimator.xyz[0]), 4), round(float(estimator.xyz[1]), 4)],
        replan_count=controller.report.replan_count,
        lidar_detection_frames=controller.report.lidar_detection_frames,
        rgbd_detection_frames=controller.report.rgbd_detection_frames,
        drop_detection_frames=controller.report.drop_detection_frames,
        onboard_blocked_cells=[
            dict(row=int(r), col=int(c), xy_m=grid.center((int(r), int(c))).tolist(),
                 source="+".join(sorted(onboard_map.coarse_sources.get((int(r), int(c)), set()))))
            for r, c in map(tuple, np.argwhere(onboard_map.onboard_blocked))
        ],
        routes_history=controller.route_history,
    )
    if experiment:
        status_payload["local_policy"] = experiment.close(controller.report.status)
    _write_json_atomic(rover_dir / "events.json", controller.report.events)
    _write_json_atomic(rover_dir / "nav_status.json", status_payload)
    print(f"[Map rover] Navigation finished: {status_payload['status']} "
          f"(reason={status_payload['reason']}, replans={status_payload['replan_count']}).", flush=True)
    robot.step(period_ms)


def run_rescue_mission(robot, folder, settings, grid, states, layers, map_file, planner_mode):
    """Autonomous multi-victim rescue: read aerial_victims.json -> plan -> drive -> confirm -> deliver."""
    rover_dir = folder / "rover"
    rover_dir.mkdir(parents=True, exist_ok=True)
    motors = [robot.getDevice(name) for name in ("left_motor::front", "right_motor::front")]
    for motor in motors:
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)

    victims_file = folder / "aerial_victims.json"
    if not victims_file.is_file():
        status = dict(status="STOPPED_SAFE", reason="NO_AERIAL_VICTIMS_FILE", events=[])
        _write_json_atomic(rover_dir / "nav_status.json", status)
        return

    victims_doc = json.loads(victims_file.read_text(encoding="utf-8"))
    remaining = list(victims_doc.get("candidates", []))
    detector_mode = os.environ.get("UAV_RESCUE_DETECTOR", victims_doc.get("detector", "surrogate_rgbd"))
    common = json.loads((ROOT / "config/search/common.json").read_text(encoding="utf-8"))
    score_min = float(common["detection_score_min"][detector_mode])
    cargo = int(os.environ.get("UAV_RESCUE_MAX_VICTIMS", str(common["cargo_capacity"])))

    if str(ROOT / "controllers/search_agent") not in sys.path:
        sys.path.insert(0, str(ROOT / "controllers/search_agent"))
    from perception import PersonDetector
    detector = PersonDetector(detector_mode, ROOT, bias_std=common["position_bias_std_m"])

    period_ms = 100
    dt = period_ms / 1000.0
    gps = robot.getDevice("deployment_gps")
    imu = robot.getDevice("attitude")
    encoders = [robot.getDevice(name) for name in ("left_encoder", "right_encoder")]
    lidar = robot.getDevice("lidar")
    depth_cam = robot.getDevice("depth")
    rgb_cam = robot.getDevice("rgb")
    chase_cam = robot.getDevice("rover_chase_cam")
    for device in (gps, imu, *encoders, lidar, depth_cam, rgb_cam):
        device.enable(period_ms)
    if chase_cam is not None:
        chase_cam.enable(200)

    init_gps = []
    init_rpy = None
    for _ in range(5):
        if robot.step(period_ms) == -1:
            return
        g = np.asarray(gps.getValues(), dtype=float)
        r = np.asarray(imu.getRollPitchYaw(), dtype=float)
        if np.isfinite(g).all() and np.isfinite(r).all():
            init_gps.append(g)
            init_rpy = r
        if len(init_gps) >= 3:
            break
    if not init_gps or init_rpy is None:
        return

    estimator = RoverPoseEstimator(np.median(init_gps, axis=0), init_rpy[2])
    onboard_map = OnboardObstacleMap(
        grid=grid,
        prior_states=states,
        ground_z_m=layers["ground_z_m"],
        planning_cost=layers.get("planning_cost") if "max_slope_deg" in settings else None,
        rover_radius_m=settings["rover_radius_m"],
    )
    vctp = VctpChecker(FineTerrainSurface.from_uav_map(map_file))

    drive_start_s = robot.getTime()
    nav_budget_s = float(os.environ.get("UAV_NAV_TIMEOUT_S", "360"))
    nav_deadline_s = drive_start_s + nav_budget_s
    depth_h, depth_w = depth_cam.getHeight(), depth_cam.getWidth()
    rgb_h, rgb_w = rgb_cam.getHeight(), rgb_cam.getWidth()
    depth_fov = depth_cam.getFov()
    num_layers = lidar.getNumberOfLayers()
    record_video = os.environ.get("UAV_RECORD_VIDEO") == "1"
    fpv_dir = rover_dir / "fpv_frames"
    chase_dir = rover_dir / "chase_frames"
    if record_video:
        fpv_dir.mkdir(parents=True, exist_ok=True)
        if chase_cam is not None:
            chase_dir.mkdir(parents=True, exist_ok=True)
    cam_rec_idx = 0
    cam_rec_step = 0

    def maybe_record_rover_cams():
        nonlocal cam_rec_idx, cam_rec_step
        if not record_video:
            return
        cam_rec_step += 1
        if cam_rec_step % 2 == 0:
            rgb_cam.saveImage(str(fpv_dir / f"{cam_rec_idx:05d}.jpg"), 85)
            if chase_cam is not None:
                chase_cam.saveImage(str(chase_dir / f"{cam_rec_idx:05d}.jpg"), 85)
            cam_rec_idx += 1

    rescue_events = []
    leg_records = []
    routes_history = []
    total_replans = 0
    total_lidar_frames = 0
    total_rgbd_frames = 0
    total_drop_frames = 0
    overall_status = "COMPLETED"
    stop_reason = None
    tick = 0

    def publish_state(state_name, active_victim_id=None):
        robot.setCustomData(json.dumps(dict(
            nav_active=True,
            rescue_mode=True,
            detector=detector_mode,
            planner=planner_mode,
            pose_est_xy=[round(float(estimator.xyz[0]), 4), round(float(estimator.xyz[1]), 4)],
            yaw_est_rad=round(float(estimator.yaw), 4),
            state=state_name,
            active_victim_id=active_victim_id,
            cargo=cargo,
            replan_count=total_replans,
            events=rescue_events,
        )))

    print(
        f"[Map rover] Rescue mission start: {len(remaining)} aerial candidate(s), "
        f"cargo={cargo}, detector={detector_mode}, planner={planner_mode}.",
        flush=True,
    )

    with (rover_dir / "telemetry.jsonl").open("w", encoding="utf-8") as telemetry:
        try:
            while cargo > 0 and remaining and robot.getTime() < nav_deadline_s:
                chosen, all_plans = plan_next_victim(
                    grid, states,
                    layers.get("planning_cost") if "max_slope_deg" in settings else None,
                    estimator.xyz[:2], estimator.yaw,
                    remaining, settings["rover_radius_m"],
                    vctp=vctp, planner_mode=planner_mode,
                )
                if chosen is None:
                    overall_status = "STOPPED_SAFE"
                    stop_reason = "NO_REACHABLE_DETECTED_VICTIM_ON_CONFIRMED_TERRAIN"
                    break

                victim = chosen["victim_candidate"]
                vid = victim["id"]
                remaining = [v for v in remaining if v["id"] != vid]
                routes_history.append(chosen["route_xy_m"])
                print(
                    f"[Map rover] Leg -> aerial candidate #{vid} at {victim['mean_enu_m']} "
                    f"via approach {chosen['approach_xy_m']} (standoff={chosen['standoff_m']}m, "
                    f"route={chosen['route_length_m']:.2f}m, planner={chosen['planner']}).",
                    flush=True,
                )

                controller = RouteNavigationController(
                    onboard_map=onboard_map,
                    initial_route_xy=chosen["route_xy_m"],
                    goal_xy=chosen["approach_xy_m"],
                    settle_time_s=0.4,  # We perform full face-victim + 2.2 s settle right after arrival.
                    goal_tolerance_m=0.28,
                )
                target_xy = np.asarray(victim["mean_enu_m"], dtype=float)
                leg_start_s = robot.getTime()
                leg_replans_before = 0
                arrived_leg = False

                while robot.step(period_ms) != -1:
                    now = robot.getTime()
                    tick += 1
                    g = np.asarray(gps.getValues(), dtype=float)
                    rpy = np.asarray(imu.getRollPitchYaw(), dtype=float)
                    wheels = np.asarray([enc.getValue() for enc in encoders], dtype=float)
                    pose_xyz, yaw = estimator.update(g, rpy, wheels, dt, grid, layers["ground_z_m"])

                    layer_ranges = [lidar.getLayerRangeImage(layer) for layer in range(num_layers)]
                    lidar_body, lidar_enu = extract_lidar_obstacles(
                        layer_ranges, pose_xyz, (rpy[0], rpy[1], yaw),
                        grid, layers["ground_z_m"], z_bias=estimator.z_bias_m,
                    )
                    depth_img = np.asarray(depth_cam.getRangeImage(), dtype=np.float32).reshape(depth_h, depth_w)
                    rgbd_body, rgbd_enu, drop_count = extract_rgbd_obstacles(
                        depth_img, pose_xyz, (rpy[0], rpy[1], yaw),
                        grid, layers["ground_z_m"], z_bias=estimator.z_bias_m, fov=depth_fov,
                    )
                    v_cmd, w_cmd, info = controller.step(
                        now, dt, pose_xyz, (rpy[0], rpy[1], yaw),
                        estimator.speed_mps, estimator.yaw_rate_rad_s,
                        lidar_body, lidar_enu, rgbd_body, rgbd_enu, drop_count,
                    )
                    left_rad, right_rad = wheel_velocities_rad_s(v_cmd, w_cmd)
                    motors[0].setVelocity(left_rad)
                    motors[1].setVelocity(right_rad)

                    if controller.report.replan_count > leg_replans_before:
                        total_replans += controller.report.replan_count - leg_replans_before
                        leg_replans_before = controller.report.replan_count
                        routes_history.append(controller.route_xy.tolist())

                    publish_state(controller.state, vid)
                    maybe_record_rover_cams()
                    if tick % 2 == 0 or controller.report.status != "RUNNING":
                        telemetry.write(json.dumps(dict(
                            time_s=round(now, 3),
                            victim_id=vid,
                            pose_est_xyz=[round(float(x), 4) for x in pose_xyz],
                            rpy_rad=[round(float(x), 4) for x in rpy],
                            speed_est_mps=round(float(estimator.speed_mps), 4),
                            cmd_vw=[round(float(v_cmd), 4), round(float(w_cmd), 4)],
                            state=controller.state,
                            dist_to_goal_m=round(float(info["dist_to_goal_m"]), 3),
                        )) + "\n")
                        telemetry.flush()

                    if controller.report.status == "ARRIVED":
                        arrived_leg = True
                        break
                    if controller.report.status == "STOPPED_SAFE":
                        standoff_now = float(np.linalg.norm(pose_xyz[:2] - target_xy))
                        if standoff_now <= 1.55:
                            print(
                                f"[Map rover] Safe standoff reached ({standoff_now:.2f} m from detected candidate #{vid}); "
                                f"stopping outside victim footprint to verify & deliver.",
                                flush=True,
                            )
                            arrived_leg = True
                            break
                        leg_records.append(dict(
                            victim_candidate_id=vid,
                            status="STOPPED_SAFE",
                            reason=controller.report.reason,
                            approach_xy_m=chosen["approach_xy_m"],
                            planner=chosen["planner"],
                            route_length_m=round(chosen["route_length_m"], 3),
                        ))
                        stop_reason = controller.report.reason
                        break
                    if now >= nav_deadline_s:
                        overall_status = "TIMEOUT"
                        stop_reason = f"RESCUE_TIMEOUT(budget={nav_budget_s:.0f}s)"
                        break

                total_lidar_frames += controller.report.lidar_detection_frames
                total_rgbd_frames += controller.report.rgbd_detection_frames
                total_drop_frames += controller.report.drop_detection_frames

                if not arrived_leg:
                    if overall_status == "TIMEOUT":
                        break
                    continue

                # Face the detected victim, settle >= 1.8 s, and verify with onboard RGB-D.
                target_xy = np.asarray(victim["mean_enu_m"], dtype=float)
                face_deadline = robot.getTime() + 5.0
                while robot.step(period_ms) != -1 and robot.getTime() < face_deadline:
                    g = np.asarray(gps.getValues(), dtype=float)
                    rpy = np.asarray(imu.getRollPitchYaw(), dtype=float)
                    wheels = np.asarray([enc.getValue() for enc in encoders], dtype=float)
                    pose_xyz, yaw = estimator.update(g, rpy, wheels, dt, grid, layers["ground_z_m"])
                    desired_yaw = math.atan2(target_xy[1] - pose_xyz[1], target_xy[0] - pose_xyz[0])
                    err = wrap_angle(desired_yaw - yaw)
                    if abs(err) < 0.14:
                        break
                    w_turn = float(np.clip(1.6 * err, -0.75, 0.75))
                    l_r, r_r = wheel_velocities_rad_s(0.0, w_turn)
                    motors[0].setVelocity(l_r)
                    motors[1].setVelocity(r_r)
                    publish_state("FACE_VICTIM", vid)
                    maybe_record_rover_cams()

                for motor in motors:
                    motor.setVelocity(0.0)

                # Hold stationary for 2.2 s while sampling onboard RGB-D confirmation frames.
                best_confirm = None
                settle_start = robot.getTime()
                confirm_img_name = f"victim_{vid:02d}_confirm.jpg"
                while robot.step(period_ms) != -1:
                    now = robot.getTime()
                    g = np.asarray(gps.getValues(), dtype=float)
                    rpy = np.asarray(imu.getRollPitchYaw(), dtype=float)
                    wheels = np.asarray([enc.getValue() for enc in encoders], dtype=float)
                    pose_xyz, yaw = estimator.update(g, rpy, wheels, dt, grid, layers["ground_z_m"])
                    publish_state("VERIFY_AND_DELIVER", vid)
                    maybe_record_rover_cams()
                    elapsed_still = now - settle_start
                    if elapsed_still >= 0.4 and int(round(elapsed_still * 10)) % 4 == 0:
                        depth_img = np.asarray(depth_cam.getRangeImage(), dtype=np.float32).reshape(depth_h, depth_w).copy()
                        bgr = np.frombuffer(rgb_cam.getImage(), dtype=np.uint8).reshape(rgb_h, rgb_w, 4)[:, :, :3].copy()
                        chk = confirm_victim_onboard(
                            detector, bgr, depth_img, pose_xyz[:2], yaw, rpy[0], rpy[1],
                            score_min, victim["mean_enu_m"], victim["covariance_enu"],
                        )
                        if chk["confirmed"] and (best_confirm is None or chk["rover_score"] > best_confirm["rover_score"]):
                            best_confirm = chk
                            rgb_cam.saveImage(str(rover_dir / confirm_img_name), 92)
                    if elapsed_still >= 2.2:
                        if best_confirm is None:
                            rgb_cam.saveImage(str(rover_dir / confirm_img_name), 92)
                        break

                now = robot.getTime()
                sources = ["rover", "uav"] if (best_confirm and best_confirm["confirmed"]) else ["uav"]
                report_xy = best_confirm["fused_xy_m"] if best_confirm else victim["mean_enu_m"]
                report_cov = best_confirm["fused_cov_enu"] if best_confirm else victim["covariance_enu"]
                report_event = dict(
                    type="report",
                    track_id=vid,
                    timestamp_s=round(now, 3),
                    xy=report_xy,
                    covariance=report_cov,
                    score=best_confirm["rover_score"] if best_confirm else victim["max_score"],
                    sources=sources,
                    evidence=f"rover/{confirm_img_name}",
                    aerial_evidence=victim["evidence"][:2],
                )
                cargo = max(0, cargo - 1)
                delivery_event = dict(
                    type="delivery",
                    track_id=vid,
                    timestamp_s=round(now + 0.1, 3),
                    estimated_xy=[round(float(estimator.xyz[0]), 4), round(float(estimator.xyz[1]), 4)],
                    approach_xy_m=chosen["approach_xy_m"],
                    cargo_remaining=cargo,
                    sources=sources,
                )
                rescue_events.extend([report_event, delivery_event])
                publish_state("DELIVERED", vid)
                for _ in range(3):
                    robot.step(period_ms)

                leg_records.append(dict(
                    victim_candidate_id=vid,
                    status="DELIVERED",
                    planner=chosen["planner"],
                    approach_xy_m=chosen["approach_xy_m"],
                    standoff_m=chosen["standoff_m"],
                    route_length_m=round(chosen["route_length_m"], 3),
                    leg_duration_s=round(now - leg_start_s, 3),
                    reported_xy_enu_m=report_xy,
                    sources=sources,
                    onboard_confirmed=bool(best_confirm and best_confirm["confirmed"]),
                    onboard_score=best_confirm["rover_score"] if best_confirm else None,
                    vctp_audit=chosen.get("vctp_audit"),
                ))
                print(
                    f"[Map rover] Reported & delivered to candidate #{vid} at t={now:.1f}s "
                    f"(sources={sources}, cargo_left={cargo}).",
                    flush=True,
                )
        finally:
            for motor in motors:
                motor.setVelocity(0.0)

    status_payload = dict(
        mode="rescue",
        detector=detector_mode,
        planner=planner_mode,
        status=overall_status,
        reason=stop_reason,
        drive_start_s=round(float(drive_start_s), 3),
        end_time_s=round(float(robot.getTime()), 3),
        final_pose_est_xy=[round(float(estimator.xyz[0]), 4), round(float(estimator.xyz[1]), 4)],
        aerial_candidates_total=len(victims_doc.get("candidates", [])),
        delivered_count=sum(1 for r in leg_records if r["status"] == "DELIVERED"),
        cargo_remaining=cargo,
        replan_count=total_replans,
        lidar_detection_frames=total_lidar_frames,
        rgbd_detection_frames=total_rgbd_frames,
        drop_detection_frames=total_drop_frames,
        onboard_blocked_cells=[
            dict(row=int(r), col=int(c), xy_m=grid.center((int(r), int(c))).tolist(),
                 source="+".join(sorted(onboard_map.coarse_sources.get((int(r), int(c)), set()))))
            for r, c in map(tuple, np.argwhere(onboard_map.onboard_blocked))
        ],
        legs=leg_records,
        routes_history=routes_history,
    )
    _write_json_atomic(rover_dir / "events.json", rescue_events)
    _write_json_atomic(rover_dir / "nav_status.json", status_payload)
    print(f"[Map rover] Rescue mission finished: status={overall_status}, "
          f"delivered={status_payload['delivered_count']}, cargo_left={cargo}.", flush=True)
    robot.step(period_ms)


def main():
    folder = ROOT / "runs" / os.environ["UAV_MAPPING_RUN"]
    settings = json.loads((folder / "map_config.json").read_text(encoding="utf-8"))
    robot = Robot()
    gps = robot.getDevice("deployment_gps")
    gps.enable(200)
    imu = robot.getDevice("attitude")
    imu.enable(200)
    for name in ("left_motor::front", "right_motor::front"):
        motor = robot.getDevice(name)
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)

    positions = deque(maxlen=5)
    map_file = folder / "uav_map.npz"
    while robot.step(200) != -1:
        if robot.getCustomData() != "deployed":
            continue
        xy = np.asarray(gps.getValues()[:2])
        if np.isfinite(xy).all():
            positions.append(xy)
        if len(positions) == positions.maxlen and map_file.is_file():
            break
    else:
        return

    start = np.median(positions, axis=0)
    rpy0 = np.asarray(imu.getRollPitchYaw(), dtype=float)
    start_yaw = float(rpy0[2]) if np.isfinite(rpy0).all() else 0.0
    planner_mode = os.environ.get("UAV_NAV_PLANNER", "astar")
    radius = settings["rover_radius_m"]
    goals = settings.get("route_goals")
    if goals is None:
        goals = [{"id": "local", "label": "局部测试路点", "xy_enu_m": settings["waypoint_xy_enu_m"]}]
    grid, states = load_prior(map_file)
    with np.load(map_file, allow_pickle=False) as data:
        layers = {name: data[name] for name in ("slope_deg", "planning_cost", "ground_z_m")}
    vctp = None
    if planner_mode == "igha_vctp" or os.environ.get("UAV_NAV_ENABLE") == "1":
        vctp = VctpChecker(FineTerrainSurface.from_uav_map(map_file))
    routes = [dict(id=case["id"], label=case["label"],
                   **route_result(grid, states, layers, start, case["xy_enu_m"], radius,
                                  "max_slope_deg" in settings, vctp=vctp,
                                  planner_mode=planner_mode, start_yaw=start_yaw))
              for case in goals]
    baseline_file = folder / "uav_map_nadir.npz"
    if baseline_file.is_file():
        base_grid, base_states = load_prior(baseline_file)
        with np.load(baseline_file, allow_pickle=False) as data:
            base_layers = {name: data[name] for name in ("slope_deg", "planning_cost")}
        for case, current in zip(goals, routes):
            initial = route_result(base_grid, base_states, base_layers, start, case["xy_enu_m"], radius, True)
            current["nadir"] = {key: initial[key] for key in
                                ("status", "goal_state", "route_length_m", "terrain", "dual_route") if key in initial}
    result = dict(**routes[0], frame="ENU", start_xy_m=start.tolist(), start_yaw_rad=start_yaw,
                  rover_radius_m=radius, planner_mode=planner_mode,
                  timestamp_s=robot.getTime(), routes=routes)
    _write_json_atomic(folder / "route.json", result)
    print(f"[Map rover] {sum(r['status'] == 'PLANNED' for r in routes)}/{len(routes)} routes planned ({planner_mode}).",
          flush=True)
    robot.step(200)

    if os.environ.get("UAV_NAV_ENABLE") == "1":
        if os.environ.get("UAV_RESCUE_MODE") == "1":
            run_rescue_mission(robot, folder, settings, grid, states, layers, map_file, planner_mode)
        else:
            target_id = os.environ.get("UAV_NAV_GOAL")
            selected = next((r for r in routes if r["id"] == target_id), routes[0]) if target_id else routes[0]
            drive_route(robot, folder, settings, grid, states, layers, selected)


if __name__ == "__main__":
    main()
