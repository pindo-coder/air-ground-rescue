#!/usr/bin/env python3
"""ROS 2 Jazzy rover global planner, 10 Hz closed-loop tracker, neural/rule local policy, and rescue node.

Consumes ONLY onboard rover sensors (`/rover/gps`, `/rover/imu`, `/rover/joint_states`,
`/rover/lidar/points` or `/rover/lidar`, `/rover/rgb`, `/rover/depth`) and UAV sensor-derived
artifacts (`uav_map.npz`, `aerial_victims.json`). Zero access to simulator ground truth.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import threading

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for rel in ("controllers/uav_mapper", "controllers/uav_map_rover", "controllers/search_agent", "gazebo/ros2_rescue"):
    p = str(ROOT / rel)
    if p not in sys.path:
        sys.path.insert(0, p)

from grid_route import direct_line_cells, plan, plan_dual
from igha_star import plan_igha_star
from policy_features import encode_rgbd
from rescue_mission import confirm_victim_onboard, plan_next_victim
from route_tracker import (
    OnboardObstacleMap,
    RouteNavigationController,
    RoverPoseEstimator,
    extract_lidar_obstacles,
    extract_pointcloud_obstacles,
    extract_rgbd_obstacles,
    wrap_angle,
)
from sensor_adapters import (
    NavSatToEnu,
    extract_wheel_positions_rad,
    quat_xyzw_to_rpy,
    ros_depth_to_meters,
    ros_image_to_bgr,
    ros_laserscan_to_layers,
    ros_pointcloud2_to_xyz,
    rpy_to_quat_xyzw,
    static_tf_specs,
)
from survey_map import load_prior
from vctp import FineTerrainSurface, VctpChecker


def _write_json_atomic(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def plan_goal_route(grid, states, layers, start_xy, goal_xy, radius, terrain_aware,
                    vctp=None, planner_mode="igha_vctp", start_yaw=0.0):
    """Plan dual-route, 2D A*, and optionally VCTP+IGHA* (mirrors Webots uav_map_rover.route_result)."""
    result = dict(
        status="NO_ROUTE",
        goal_xy_m=list(map(float, goal_xy)),
        straight_line=direct_line_cells(grid, states, start_xy, goal_xy),
    )
    goal_cell = grid.cell(goal_xy)
    if grid.contains(goal_cell):
        result["goal_state"] = int(states[goal_cell])

    def measure(path):
        slope = np.asarray([layers["slope_deg"][grid.cell(xy)] for xy in path])
        known = slope[np.isfinite(slope)]
        return dict(
            length_m=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
            mean_slope_deg=float(known.mean()) if known.size else None,
            max_slope_deg=float(known.max()) if known.size else None,
            slope_supported_waypoints=int(known.size),
        )

    cost = layers["planning_cost"] if terrain_aware else None
    dual = plan_dual(grid, states, start_xy, goal_xy, radius, cost)
    result["dual_route"] = {
        k: dual[k]
        for k in (
            "conservative_status", "conservative_length_m",
            "optimistic_status", "optimistic_length_m",
            "optimistic_waypoint_count", "unconfirmed_clearance_waypoints",
            "direct_unknown_waypoints", "corridor_unknown_cell_count",
            "changes_reachability", "potential_saving_m",
        )
    }

    try:
        route = plan(grid, states, start_xy, goal_xy, radius, cost)
        metrics = measure(route)
        result.update(
            status="PLANNED",
            planner="astar",
            route_xy_m=route.tolist(),
            route_length_m=metrics["length_m"],
            terrain=metrics,
        )
        if terrain_aware:
            baseline = plan(grid, states, start_xy, goal_xy, radius)
            result["distance_only"] = measure(baseline)
            result["distance_only_route_xy_m"] = baseline.tolist()
        if vctp is not None:
            result["astar_vctp_audit"] = vctp.audit_route(route)
    except ValueError as err:
        result["reason"] = str(err)

    if planner_mode == "igha_vctp" and vctp is not None:
        igha = plan_igha_star(grid, states, vctp, start_xy, goal_xy, radius, start_yaw=start_yaw)
        result["igha_vctp"] = {
            k: igha[k] for k in ("status", "reason", "route_length_m", "vctp", "stats") if k in igha
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


class RoverMissionCore:
    """Simulator-agnostic rover navigation & rescue state machine for ROS 2 Jazzy."""

    def __init__(self, run_dir, mode="navigation", goal_id=None, planner_mode="igha_vctp",
                 local_policy="rule", local_checkpoint=None, local_record=False,
                 local_expert_mix=0.0, local_seed=17, detector_mode="yolo",
                 max_victims=2, timeout_s=240.0, gps_noise_m=0.04,
                 spherical_origin=None, depth_fov_rad=math.pi / 2.0):
        self.folder = Path(run_dir)
        self.rover_dir = self.folder / "rover"
        self.rover_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.goal_id = goal_id
        self.planner_mode = planner_mode
        self.local_policy = local_policy
        self.local_checkpoint = local_checkpoint
        self.local_record = bool(local_record)
        self.local_expert_mix = float(local_expert_mix)
        self.local_seed = int(local_seed)
        self.detector_mode = detector_mode
        self.max_victims = int(max_victims)
        self.timeout_s = float(timeout_s)
        self.depth_fov = float(depth_fov_rad)

        origin = spherical_origin or {}
        self.navsat = NavSatToEnu(
            lat0_deg=float(origin.get("latitude_deg", 0.0)),
            lon0_deg=float(origin.get("longitude_deg", 0.0)),
            alt0_m=float(origin.get("elevation_m", 0.0)),
            noise_std_m=float(gps_noise_m),
            noise_correlation=0.2,
            seed=local_seed,
        )

        self.settings = json.loads((self.folder / "map_config.json").read_text(encoding="utf-8"))
        self.grid, self.states, self.layers = load_prior(self.folder / "uav_map.npz")
        self.fine_surface = FineTerrainSurface.from_uav_map(self.folder / "uav_map.npz")
        self.vctp = VctpChecker(self.fine_surface)
        self.terrain_aware = "max_slope_deg" in self.settings
        self.planning_cost = self.layers.get("planning_cost") if self.terrain_aware else None
        self.rover_radius = float(self.settings["rover_radius_m"])

        self.experiment = None
        if self.local_policy != "rule" or self.local_record:
            from local_policy import LocalPolicyExperiment

            self.experiment = LocalPolicyExperiment(
                self.local_policy,
                checkpoint=self.local_checkpoint,
                record_to=(self.rover_dir / "imitation") if self.local_record else None,
                expert_mix=self.local_expert_mix,
                seed=self.local_seed,
            )

        self.onboard_detector = None
        self.score_min = 0.35
        if self.mode == "rescue":
            from perception import PersonDetector

            common = json.loads((ROOT / "config/search/common.json").read_text(encoding="utf-8"))
            self.score_min = float(common["detection_score_min"][self.detector_mode])
            self.onboard_detector = PersonDetector(
                self.detector_mode, ROOT, max_depth=8.0, field_of_view=self.depth_fov, view="ground"
            )

        self.estimator = None
        self.onboard_map = None
        self.controller = None
        self.initialized = False
        self.finished = False
        self.drive_start_s = None
        self.deadline_s = None
        self.selected_route = None
        self.telemetry_file = None

        # Rescue mission state
        self.remaining_victims = []
        self.current_leg = None
        self.completed_legs = []
        self.rescue_events = []
        self.confirm_phase = None  # None | "turn" | "settle"
        self.confirm_settle_s = 0.0
        self.confirm_frames = 0
        self.confirm_best = None
        self.tick = 0

    def initialize(self, now_s, init_gps_enu, init_rpy):
        if self.initialized:
            return
        self.initialized = True
        self.drive_start_s = float(now_s)
        self.deadline_s = self.drive_start_s + self.timeout_s
        start_xyz = np.asarray(init_gps_enu, dtype=float)
        start_yaw = float(init_rpy[2])
        self.estimator = RoverPoseEstimator(start_xyz, start_yaw)
        self.onboard_map = OnboardObstacleMap(
            grid=self.grid,
            prior_states=self.states,
            ground_z_m=self.layers["ground_z_m"],
            planning_cost=self.planning_cost,
            rover_radius_m=self.rover_radius,
        )
        self.telemetry_file = (self.rover_dir / "telemetry.jsonl").open("w", encoding="utf-8")

        if self.mode == "rescue":
            self._init_rescue(start_xyz, start_yaw)
        else:
            self._init_navigation(start_xyz, start_yaw)

    def _init_navigation(self, start_xyz, start_yaw):
        goals = self.settings["route_goals"]
        routes = []
        for g in goals:
            r = plan_goal_route(
                self.grid, self.states, self.layers, start_xyz[:2], g["xy_enu_m"],
                self.rover_radius, self.terrain_aware, vctp=self.vctp,
                planner_mode=self.planner_mode, start_yaw=start_yaw,
            )
            r["id"] = g["id"]
            r["label"] = g.get("label", g["id"])
            routes.append(r)

        selected = (
            next((r for r in routes if r["id"] == self.goal_id), routes[0])
            if self.goal_id else routes[0]
        )
        self.selected_route = selected
        _write_json_atomic(
            self.folder / "route.json",
            dict(
                planner=self.planner_mode,
                start_xy_m=start_xyz[:2].tolist(),
                start_yaw_rad=round(start_yaw, 4),
                selected_goal_id=selected["id"],
                routes=routes,
            ),
        )
        if selected["status"] != "PLANNED":
            self._finish_navigation("STOPPED_SAFE", f"INITIAL_ROUTE_UNAVAILABLE: {selected.get('reason', 'NO_ROUTE')}", self.drive_start_s)
            return

        self.controller = RouteNavigationController(
            onboard_map=self.onboard_map,
            initial_route_xy=selected["route_xy_m"],
            goal_xy=selected["goal_xy_m"],
            local_planner=self.experiment,
        )
        print(
            f"[ROS2 Rover] Planned goal '{selected['id']}' ({selected['route_length_m']:.2f} m, "
            f"planner={selected.get('planner')}, local={self.local_policy}).",
            flush=True,
        )

    def _init_rescue(self, start_xyz, start_yaw):
        victims_path = self.folder / "aerial_victims.json"
        if not victims_path.is_file():
            self._finish_rescue("FAILED", "MISSING_AERIAL_VICTIMS_JSON", self.drive_start_s)
            return
        doc = json.loads(victims_path.read_text(encoding="utf-8"))
        self.remaining_victims = list(doc.get("candidates", []))
        if not self.remaining_victims:
            self._finish_rescue("STOPPED_SAFE", "NO_AERIAL_VICTIM_CANDIDATES", self.drive_start_s)
            return
        self._start_next_rescue_leg(start_xyz[:2], start_yaw, self.drive_start_s)

    def _start_next_rescue_leg(self, rover_xy, rover_yaw, now_s):
        if len(self.completed_legs) >= self.max_victims or not self.remaining_victims:
            status = "MISSION_SUCCESS" if self.completed_legs else "STOPPED_SAFE"
            self._finish_rescue(status, "ALL_ASSIGNED_VICTIMS_PROCESSED", now_s)
            return

        chosen, all_plans = plan_next_victim(
            self.grid, self.states, self.planning_cost, rover_xy, rover_yaw,
            self.remaining_victims, self.rover_radius, vctp=self.vctp,
            planner_mode=self.planner_mode,
        )
        if chosen is None:
            status = "MISSION_PARTIAL" if self.completed_legs else "STOPPED_SAFE"
            self._finish_rescue(status, "NO_REACHABLE_REMAINING_VICTIM", now_s)
            return

        cand = chosen["victim_candidate"]
        self.remaining_victims = [v for v in self.remaining_victims if v["id"] != cand["id"]]
        self.current_leg = chosen
        self.confirm_phase = None
        self.confirm_settle_s = 0.0
        self.confirm_frames = 0
        self.confirm_best = None
        self.controller = RouteNavigationController(
            onboard_map=self.onboard_map,
            initial_route_xy=chosen["route_xy_m"],
            goal_xy=chosen["approach_xy_m"],
            local_planner=self.experiment,
        )
        _write_json_atomic(
            self.folder / "route.json",
            dict(
                mode="rescue",
                planner=chosen.get("planner", self.planner_mode),
                active_victim_id=cand["id"],
                goal_xy_m=chosen["approach_xy_m"],
                victim_xy_m=chosen["victim_xy_m"],
                route_xy_m=chosen["route_xy_m"],
                route_length_m=chosen["route_length_m"],
                all_plans=[{k: v for k, v in p.items() if k != "route_xy_m"} for p in all_plans],
            ),
        )
        print(
            f"[ROS2 Rover] Rescue leg -> victim #{cand['id']} at {cand['mean_enu_m']} "
            f"(approach={chosen['approach_xy_m']}, len={chosen['route_length_m']:.2f} m).",
            flush=True,
        )

    def step(self, now_s, dt, gps_enu, rpy, wheel_rad, lidar_xyz=None, lidar_scan=None,
             bgr=None, depth=None):
        """Run one 10 Hz perception + control tick; returns (v_cmd, w_cmd, finished)."""
        if self.finished:
            return 0.0, 0.0, True
        if not self.initialized:
            self.initialize(now_s, gps_enu, rpy)
            if self.finished:
                return 0.0, 0.0, True

        self.tick += 1
        pose_xyz, yaw = self.estimator.update(
            gps_enu, rpy, wheel_rad, dt, self.grid, self.layers["ground_z_m"]
        )
        rpy_est = (float(rpy[0]), float(rpy[1]), float(yaw))

        if lidar_xyz is not None and len(lidar_xyz) > 0:
            lidar_body, lidar_enu = extract_pointcloud_obstacles(
                lidar_xyz, pose_xyz, rpy_est, self.grid, self.layers["ground_z_m"],
                z_bias=self.estimator.z_bias_m,
            )
        elif lidar_scan is not None:
            layers_r, elevs, az_order = lidar_scan
            lidar_body, lidar_enu = extract_lidar_obstacles(
                layers_r, pose_xyz, rpy_est, self.grid, self.layers["ground_z_m"],
                z_bias=self.estimator.z_bias_m, elevations=elevs, azimuth_order=az_order,
            )
        else:
            lidar_body = np.empty((0, 2), dtype=float)
            lidar_enu = np.empty((0, 2), dtype=float)

        if depth is not None:
            rgbd_body, rgbd_enu, drop_count = extract_rgbd_obstacles(
                depth, pose_xyz, rpy_est, self.grid, self.layers["ground_z_m"],
                z_bias=self.estimator.z_bias_m, fov=self.depth_fov,
            )
        else:
            rgbd_body = np.empty((0, 2), dtype=float)
            rgbd_enu = np.empty((0, 2), dtype=float)
            drop_count = 0

        if now_s >= self.deadline_s:
            if self.mode == "rescue":
                status = "MISSION_PARTIAL" if self.completed_legs else "TIMEOUT"
                self._finish_rescue(status, "RESCUE_TIMEOUT", now_s)
            else:
                self._finish_navigation("TIMEOUT", "NAVIGATION_TIMEOUT", now_s)
            return 0.0, 0.0, True

        # Rescue confirmation & supply delivery phase once arrived at victim standoff.
        if self.mode == "rescue" and self.confirm_phase is not None:
            return self._step_rescue_confirm(now_s, dt, pose_xyz, rpy_est, bgr, depth)

        rgbd_frame = None
        if self.experiment is not None and bgr is not None and depth is not None:
            rgbd_frame = encode_rgbd(bgr, depth, encoding="bgr8")

        v_cmd, w_cmd, info = self.controller.step(
            now=now_s,
            dt=dt,
            pose_xyz=pose_xyz,
            rpy=rpy_est,
            speed_mps=self.estimator.speed_mps,
            yaw_rate_rad_s=self.estimator.yaw_rate_rad_s,
            lidar_body_xy=lidar_body,
            lidar_enu_xy=lidar_enu,
            rgbd_body_xy=rgbd_body,
            rgbd_enu_xy=rgbd_enu,
            drop_count=drop_count,
            rgbd_frame=rgbd_frame,
        )
        if self.experiment is not None:
            self.experiment.observe_executed(v_cmd, w_cmd, now_s)

        if self.telemetry_file is not None:
            self.telemetry_file.write(json.dumps(dict(
                time_s=round(float(now_s), 3),
                pose_xy=[round(float(pose_xyz[0]), 4), round(float(pose_xyz[1]), 4)],
                pose_z=round(float(pose_xyz[2]), 4),
                yaw_rad=round(float(yaw), 4),
                speed_mps=round(float(self.estimator.speed_mps), 4),
                v_cmd=round(float(v_cmd), 4),
                w_cmd=round(float(w_cmd), 4),
                state=self.controller.state,
                status=self.controller.report.status,
                lidar_obs=int(len(lidar_body)),
                rgbd_obs=int(len(rgbd_body)),
            )) + "\n")

        # Rescue proximity stop: if close to victim (< 1.55 m) and controller paused or arrived, start confirmation.
        if self.mode == "rescue" and self.current_leg is not None:
            victim_xy = np.asarray(self.current_leg["victim_xy_m"], dtype=float)
            dist_victim = float(np.linalg.norm(pose_xyz[:2] - victim_xy))
            if self.controller.report.status == "ARRIVED" or (
                dist_victim <= 1.55 and self.controller.report.status in ("STOPPED_SAFE", "BLOCKED")
            ):
                self.confirm_phase = "turn"
                self.confirm_settle_s = 0.0
                return 0.0, 0.0, False
            if self.controller.report.status in ("STOPPED_SAFE", "FAILED"):
                self.rescue_events.append(dict(
                    time_s=round(float(now_s), 3),
                    type="LEG_ABORTED",
                    victim_id=self.current_leg["victim_candidate"]["id"],
                    reason=self.controller.report.reason,
                ))
                self._start_next_rescue_leg(pose_xyz[:2], yaw, now_s)
                return 0.0, 0.0, self.finished
        else:
            if self.controller.report.status in ("ARRIVED", "STOPPED_SAFE", "FAILED"):
                self._finish_navigation(
                    self.controller.report.status, self.controller.report.reason, now_s
                )
                return 0.0, 0.0, True

        return float(v_cmd), float(w_cmd), False

    def _step_rescue_confirm(self, now_s, dt, pose_xyz, rpy_est, bgr, depth):
        cand = self.current_leg["victim_candidate"]
        victim_xy = np.asarray(cand["mean_enu_m"], dtype=float)
        desired_yaw = math.atan2(victim_xy[1] - pose_xyz[1], victim_xy[0] - pose_xyz[0])
        yaw_err = wrap_angle(desired_yaw - rpy_est[2])

        if self.confirm_phase == "turn":
            if abs(yaw_err) <= 0.14:
                self.confirm_phase = "settle"
                self.confirm_settle_s = 0.0
                return 0.0, 0.0, False
            w_turn = float(np.clip(1.6 * yaw_err, -0.55, 0.55))
            return 0.0, w_turn, False

        # Settle stationary for 2.2 s and run onboard RGB-D YOLO confirmation.
        if abs(self.estimator.speed_mps) < 0.05 and abs(self.estimator.yaw_rate_rad_s) < 0.12:
            self.confirm_settle_s += dt
        if bgr is not None and depth is not None and self.onboard_detector is not None:
            res = confirm_victim_onboard(
                self.onboard_detector, bgr, depth, pose_xyz[:2], rpy_est[2],
                rpy_est[0], rpy_est[1], self.score_min,
                cand["mean_enu_m"], cand["covariance_enu"],
            )
            self.confirm_frames += 1
            if res.get("confirmed") and (
                self.confirm_best is None or res["rover_score"] > self.confirm_best.get("rover_score", 0.0)
            ):
                self.confirm_best = res
                cv2.imwrite(
                    str(self.rover_dir / f"victim_{cand['id']:02d}_confirm.jpg"),
                    bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                )

        if self.confirm_settle_s < 2.2:
            return 0.0, 0.0, False

        fused = self.confirm_best or dict(
            confirmed=False,
            fused_xy_m=cand["mean_enu_m"],
            fused_cov_enu=cand["covariance_enu"],
            rover_detections=0,
        )
        cross_view = bool(fused.get("confirmed"))
        report_evt = dict(
            time_s=round(float(now_s), 3),
            type="report",
            victim_id=cand["id"],
            xy_enu_m=fused["fused_xy_m"],
            cross_view=cross_view,
            aerial_score=cand.get("mean_score"),
            rover_score=fused.get("rover_score"),
            rover_pose_xy_m=[round(float(pose_xyz[0]), 4), round(float(pose_xyz[1]), 4)],
            settled_s=round(float(self.confirm_settle_s), 2),
        )
        delivery_evt = dict(
            time_s=round(float(now_s), 3),
            type="delivery",
            victim_id=cand["id"],
            victim_xy_m=fused["fused_xy_m"],
            rover_xy_m=[round(float(pose_xyz[0]), 4), round(float(pose_xyz[1]), 4)],
            rover_yaw_rad=round(float(rpy_est[2]), 4),
            settled_s=round(float(self.confirm_settle_s), 2),
        )
        self.rescue_events.extend([report_evt, delivery_evt])
        self.completed_legs.append(dict(
            victim_id=cand["id"],
            planner=self.current_leg.get("planner"),
            route_length_m=round(float(self.current_leg["route_length_m"]), 3),
            cross_view=cross_view,
            fused_xy_m=fused["fused_xy_m"],
            settled_s=round(float(self.confirm_settle_s), 2),
        ))
        _write_json_atomic(self.rover_dir / "events.json", self.rescue_events)
        print(
            f"[ROS2 Rover] Victim #{cand['id']} delivered! cross_view={cross_view} "
            f"fused={fused['fused_xy_m']} settled={self.confirm_settle_s:.2f}s",
            flush=True,
        )
        self._start_next_rescue_leg(pose_xyz[:2], rpy_est[2], now_s)
        return 0.0, 0.0, self.finished

    def _close_common(self, status_str):
        if self.telemetry_file is not None:
            self.telemetry_file.close()
            self.telemetry_file = None
        if self.experiment is not None and self.experiment.recorder is not None:
            meta = dict(
                simulator="gazebo_harmonic_ros2",
                mode=self.mode,
                planner=self.planner_mode,
                policy=self.local_policy,
                checkpoint=self.local_checkpoint,
                expert_mix=self.local_expert_mix,
                seed=self.local_seed,
            )
            self.experiment.recorder.close(meta, status_str)

    def _finish_navigation(self, status, reason, now_s):
        if self.finished:
            return
        self.finished = True
        self._close_common(status)
        report = self.controller.report if self.controller is not None else None
        payload = dict(
            goal_id=self.selected_route["id"],
            goal_label=self.selected_route.get("label", self.selected_route["id"]),
            goal_xy_m=self.selected_route["goal_xy_m"],
            planner=self.selected_route.get("planner", self.planner_mode),
            local_policy=self.local_policy,
            status=status,
            reason=reason,
            drive_start_s=round(float(self.drive_start_s), 3),
            end_time_s=round(float(now_s), 3),
            replan_count=report.replan_count if report else 0,
            events=report.events if report else [],
        )
        if self.experiment is not None:
            payload["local_policy_stats"] = self.experiment.summary()
        _write_json_atomic(self.rover_dir / "nav_status.json", payload)
        _write_json_atomic(self.rover_dir / "events.json", payload["events"])
        print(f"[ROS2 Rover] Navigation finished: status={status} reason={reason}", flush=True)

    def _finish_rescue(self, status, reason, now_s):
        if self.finished:
            return
        self.finished = True
        self._close_common(status)
        payload = dict(
            mode="rescue",
            planner=self.planner_mode,
            local_policy=self.local_policy,
            detector=self.detector_mode,
            status=status,
            reason=reason,
            drive_start_s=round(float(self.drive_start_s), 3),
            end_time_s=round(float(now_s), 3),
            delivered_count=len(self.completed_legs),
            legs=self.completed_legs,
            events=self.rescue_events,
        )
        if self.experiment is not None:
            payload["local_policy_stats"] = self.experiment.summary()
        _write_json_atomic(self.rover_dir / "rescue_status.json", payload)
        _write_json_atomic(self.rover_dir / "nav_status.json", payload)
        _write_json_atomic(self.rover_dir / "events.json", self.rescue_events)
        print(
            f"[ROS2 Rover] Rescue finished: status={status} delivered={len(self.completed_legs)} reason={reason}",
            flush=True,
        )


def run_ros2_rover(args):
    import rclpy
    from geometry_msgs.msg import TransformStamped, Twist
    from rclpy.node import Node
    from rclpy.parameter import Parameter as RosParam
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, Imu, JointState, LaserScan, NavSatFix, PointCloud2
    from tf2_ros import StaticTransformBroadcaster

    rclpy.init()

    class _RoverNode(Node):
        def __init__(self):
            super().__init__(
                "rover_nav_node",
                parameter_overrides=[RosParam("use_sim_time", RosParam.Type.BOOL, True)],
            )
            env_meta = {}
            if args.environment_json and Path(args.environment_json).is_file():
                env_meta = json.loads(Path(args.environment_json).read_text(encoding="utf-8"))
            self.core = RoverMissionCore(
                run_dir=args.run_dir,
                mode=args.mode,
                goal_id=args.goal,
                planner_mode=args.planner,
                local_policy=args.local_policy,
                local_checkpoint=args.local_checkpoint or None,
                local_record=args.local_record,
                local_expert_mix=args.local_expert_mix,
                local_seed=args.local_seed,
                detector_mode=args.detector,
                max_victims=args.max_victims,
                timeout_s=args.timeout_s,
                gps_noise_m=args.gps_noise_m,
                spherical_origin=env_meta.get("spherical_origin"),
            )
            self._publish_static_tfs()
            self.cmd_pub = self.create_publisher(Twist, "/rover/cmd_vel", 10)
            self._lock = threading.Lock()
            self._gps_samples = []
            self._latest_gps = None
            self._latest_rpy = None
            self._latest_wheels = None
            self._latest_pc = None
            self._latest_scan = None
            self._latest_bgr = None
            self._latest_depth = None
            self._last_tick_s = None
            self._fpv_dir = self.core.rover_dir / "fpv_frames"
            self._chase_dir = self.core.rover_dir / "chase_frames"
            self._fpv_idx = 0
            self._chase_idx = 0
            if args.record_cameras:
                self._fpv_dir.mkdir(exist_ok=True)
                self._chase_dir.mkdir(exist_ok=True)

            self.create_subscription(NavSatFix, "/rover/gps", self._on_gps, qos_profile_sensor_data)
            self.create_subscription(Imu, "/rover/imu", self._on_imu, qos_profile_sensor_data)
            self.create_subscription(JointState, "/rover/joint_states", self._on_joints, qos_profile_sensor_data)
            self.create_subscription(PointCloud2, "/rover/lidar/points", self._on_pc, qos_profile_sensor_data)
            self.create_subscription(LaserScan, "/rover/lidar", self._on_scan, qos_profile_sensor_data)
            self.create_subscription(Image, "/rover/rgb", self._on_rgb, qos_profile_sensor_data)
            self.create_subscription(Image, "/rover/depth", self._on_depth, qos_profile_sensor_data)
            if args.record_cameras:
                self.create_subscription(Image, "/rover/chase_cam", self._on_chase, qos_profile_sensor_data)
            self.create_timer(0.1, self._tick)

        def _publish_static_tfs(self):
            broadcaster = StaticTransformBroadcaster(self)
            now_msg = self.get_clock().now().to_msg()
            transforms = []
            for parent, child, xyz, rpy in static_tf_specs():
                t = TransformStamped()
                t.header.stamp = now_msg
                t.header.frame_id = parent
                t.child_frame_id = child
                t.transform.translation.x = float(xyz[0])
                t.transform.translation.y = float(xyz[1])
                t.transform.translation.z = float(xyz[2])
                qx, qy, qz, qw = rpy_to_quat_xyzw(*rpy)
                t.transform.rotation.x = qx
                t.transform.rotation.y = qy
                t.transform.rotation.z = qz
                t.transform.rotation.w = qw
                transforms.append(t)
            broadcaster.sendTransform(transforms)
            self._static_tf = broadcaster

        def _on_gps(self, msg):
            enu = self.core.navsat.convert(msg.latitude, msg.longitude, msg.altitude, apply_noise=True)
            with self._lock:
                self._latest_gps = enu
                if len(self._gps_samples) < 3:
                    self._gps_samples.append(enu)

        def _on_imu(self, msg):
            rpy = quat_xyzw_to_rpy(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
            with self._lock:
                self._latest_rpy = rpy

        def _on_joints(self, msg):
            wheels = extract_wheel_positions_rad(msg.name, msg.position)
            if wheels is not None:
                with self._lock:
                    self._latest_wheels = wheels

        def _on_pc(self, msg):
            xyz = ros_pointcloud2_to_xyz(
                msg.data, msg.width, msg.height, msg.point_step, msg.row_step, msg.fields, msg.is_bigendian
            )
            with self._lock:
                self._latest_pc = xyz

        def _on_scan(self, msg):
            scan = ros_laserscan_to_layers(msg.ranges, msg.angle_min, msg.angle_max)
            with self._lock:
                self._latest_scan = scan

        def _on_rgb(self, msg):
            bgr = ros_image_to_bgr(msg.data, msg.width, msg.height, msg.encoding, msg.step)
            with self._lock:
                self._latest_bgr = bgr
            if args.record_cameras and not self.core.finished and self._fpv_idx % 2 == 0:
                cv2.imwrite(
                    str(self._fpv_dir / f"{self._fpv_idx // 2:05d}.jpg"),
                    bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88],
                )
            self._fpv_idx += 1

        def _on_chase(self, msg):
            if self.core.finished:
                return
            bgr = ros_image_to_bgr(msg.data, msg.width, msg.height, msg.encoding, msg.step)
            cv2.imwrite(
                str(self._chase_dir / f"{self._chase_idx:05d}.jpg"),
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 88],
            )
            self._chase_idx += 1

        def _on_depth(self, msg):
            depth = ros_depth_to_meters(msg.data, msg.width, msg.height, msg.encoding, msg.step)
            with self._lock:
                self._latest_depth = depth

        def _send_twist(self, v, w):
            msg = Twist()
            msg.linear.x = float(v)
            msg.angular.z = float(w)
            self.cmd_pub.publish(msg)

        def _tick(self):
            now_s = self.get_clock().now().nanoseconds * 1e-9
            if now_s <= 0.0:
                return
            with self._lock:
                gps = self._latest_gps
                gps_init = list(self._gps_samples)
                rpy = self._latest_rpy
                wheels = self._latest_wheels
                pc = self._latest_pc
                scan = self._latest_scan
                bgr = self._latest_bgr
                depth = self._latest_depth

            if gps is None or rpy is None or wheels is None:
                return
            if not self.core.initialized:
                if len(gps_init) < 3:
                    return
                self.core.initialize(now_s, np.median(gps_init, axis=0), rpy)
                self._last_tick_s = now_s
                if self.core.finished:
                    self._send_twist(0.0, 0.0)
                    rclpy.shutdown()
                return

            dt = max(now_s - (self._last_tick_s or now_s), 0.02)
            self._last_tick_s = now_s
            v_cmd, w_cmd, done = self.core.step(
                now_s=now_s, dt=dt, gps_enu=gps, rpy=rpy, wheel_rad=wheels,
                lidar_xyz=pc, lidar_scan=scan, bgr=bgr, depth=depth,
            )
            self._send_twist(v_cmd, w_cmd)
            if done:
                self._send_twist(0.0, 0.0)
                rclpy.shutdown()

    node = _RoverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._send_twist(0.0, 0.0)
    finally:
        node._send_twist(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--mode", default="navigation", choices=("navigation", "rescue"))
    parser.add_argument("--goal", default=None)
    parser.add_argument("--planner", default="igha_vctp", choices=("astar", "igha_vctp"))
    parser.add_argument("--local-policy", default="rule", choices=("rule", "neural"))
    parser.add_argument("--local-checkpoint", default="")
    parser.add_argument("--local-record", action="store_true")
    parser.add_argument("--local-expert-mix", type=float, default=0.0)
    parser.add_argument("--local-seed", type=int, default=17)
    parser.add_argument("--detector", default="yolo", choices=("yolo", "surrogate_rgbd"))
    parser.add_argument("--max-victims", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--gps-noise-m", type=float, default=0.04)
    parser.add_argument("--environment-json", type=Path, default=None)
    parser.add_argument("--record-cameras", action="store_true")
    args = parser.parse_args()
    run_ros2_rover(args)


if __name__ == "__main__":
    main()
