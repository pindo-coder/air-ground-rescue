#!/usr/bin/env python3
"""ROS 2 Jazzy UAV aerial mapping, DUCPP route-value oblique refinement, and YOLO11n victim search node.

Subscribes ONLY to onboard Gazebo/ROS 2 drone sensors (`/drone/rgb`, `/drone/depth`,
`/drone/gps`, `/drone/imu`, optional `/drone/fpv_cam`). Zero access to ground truth.
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

from multiview import aerial_candidates, aerial_pixels_enu, attitude_matrix
from sensor_adapters import (
    NavSatToEnu,
    quat_xyzw_to_matrix,
    quat_xyzw_to_rpy,
    ros_depth_to_meters,
    ros_image_to_bgr,
)
from survey_map import (
    Grid,
    binned_fine_surface,
    binned_surfaces,
    classify,
    compose_fine_ground,
    sampled_pixels,
    save_prior,
)
from survey_refinement import refine_ground, schedule_oblique


def _stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _write_json_atomic(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


class UavMapperRos2Node:
    """Core state machine usable both inside rclpy.Node and in offline unit tests."""

    def __init__(self, run_dir, flight_end_s, detector_mode=None, gps_noise_m=0.04,
                 fov_rad=1.1, min_range_m=0.2, max_range_m=24.0,
                 spherical_origin=None, record_fpv=False):
        self.folder = Path(run_dir)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.settings = json.loads((self.folder / "map_config.json").read_text(encoding="utf-8"))
        self.grid = Grid.from_bounds(self.settings["bounds_enu_m"], self.settings["resolution_m"])
        self.deadline = float(flight_end_s)
        self.fov = float(fov_rad)
        self.min_range = float(min_range_m)
        self.max_range = float(max_range_m)
        self.stride = int(self.settings.get("sample_stride_px", 4))
        self.record_fpv = bool(record_fpv)

        origin = spherical_origin or {}
        self.navsat = NavSatToEnu(
            lat0_deg=float(origin.get("latitude_deg", 0.0)),
            lon0_deg=float(origin.get("longitude_deg", 0.0)),
            alt0_m=float(origin.get("elevation_m", 0.0)),
            noise_std_m=float(gps_noise_m),
            noise_correlation=0.2,
            seed=17,
        )

        self.images_dir = self.folder / "frames"
        self.images_dir.mkdir(exist_ok=True)
        self.fpv_dir = self.folder / "drone_fpv_frames"
        if self.record_fpv:
            self.fpv_dir.mkdir(exist_ok=True)

        self.timestamps = []
        self.origins = []
        self.attitudes = []
        self.depths = []
        self.colors = []
        self.rotations = []
        self.phases = []
        self.lower_views = []
        self.upper_views = []
        self.fine_views = []
        self.oblique_points = []

        self.baseline = None
        self.baseline_count = 0
        self.schedule = None
        self.capture_times = set()
        self.captured_secs = set()
        self.finished = False
        self.width = 640
        self.height = 480

        self.detector_mode = detector_mode
        self.victim_detector = None
        self.victim_tracker = None
        self.raw_victim_hits = 0
        if detector_mode:
            from aerial_victims import AerialVictimTracker
            from perception import PersonDetector

            common = json.loads((ROOT / "config/search/common.json").read_text(encoding="utf-8"))
            score_min = float(common["detection_score_min"][detector_mode])
            self.victim_detector = PersonDetector(detector_mode, ROOT, max_depth=self.max_range, field_of_view=self.fov)
            self.victim_tracker = AerialVictimTracker(association_radius_m=1.6, min_score=score_min)
            print(f"[ROS2 UAV mapper] Aerial victim detector enabled: {detector_mode} (score>={score_min}).", flush=True)

    def ingest_frame(self, now_s, bgr_full, raw_depth, gps_enu, quat_xyzw):
        """Process one synchronized RGB-D + GPS + ENU IMU observation."""
        if self.finished:
            return True

        sec_bucket = int(round(now_s))
        should_capture = False
        if self.baseline is None:
            if sec_bucket >= 1 and sec_bucket not in self.captured_secs and now_s <= self.deadline + 0.6:
                should_capture = True
        elif sec_bucket in self.capture_times and sec_bucket not in self.captured_secs:
            should_capture = True

        if should_capture:
            origin = np.asarray(gps_enu, dtype=float)
            roll, pitch, yaw = quat_xyzw_to_rpy(*quat_xyzw)
            attitude = np.array([roll, pitch, yaw], dtype=float)
            if np.isfinite(origin).all() and np.isfinite(attitude).all():
                self.captured_secs.add(sec_bucket)
                self.height, self.width = raw_depth.shape
                orientation = (attitude_matrix(*attitude) if self.baseline is None
                               else quat_xyzw_to_matrix(*quat_xyzw))
                z = raw_depth[::self.stride, ::self.stride].copy()
                rgb_sub = bgr_full[::self.stride, ::self.stride, ::-1].copy()
                pixels, _ = sampled_pixels(z, self.stride, self.min_range, self.max_range)
                points = aerial_pixels_enu(
                    pixels, self.width, self.height, self.fov, origin, attitude,
                    body_rotation=orientation,
                )
                self.fine_views.append(binned_fine_surface(self.grid, points, subdiv=4))
                if self.baseline is None:
                    low, high = binned_surfaces(self.grid, points)
                    self.lower_views.append(low)
                    self.upper_views.append(high)
                    if self.victim_detector is not None:
                        cands = aerial_candidates(
                            self.victim_detector, bgr_full, raw_depth, origin, attitude, self.fov
                        )
                        self.raw_victim_hits += len(cands)
                        if cands:
                            self.victim_tracker.ingest_candidates(
                                cands, len(self.timestamps), now_s, origin,
                                f"frames/{len(self.timestamps):03d}.jpg",
                            )
                else:
                    self.oblique_points.append(points)

                cv2.imwrite(str(self.images_dir / f"{len(self.timestamps):03d}.jpg"), bgr_full,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                self.timestamps.append(float(now_s))
                self.origins.append(origin)
                self.attitudes.append(attitude)
                self.rotations.append(orientation)
                self.phases.append(int(self.baseline is not None))
                self.depths.append(z)
                self.colors.append(rgb_sub)

        if self.baseline is None and now_s >= self.deadline + 0.8:
            if len(self.timestamps) < 2:
                raise RuntimeError("Gazebo UAV survey did not collect enough usable observations.")
            self.baseline_count = len(self.timestamps)
            self.baseline = classify(
                self.grid, self.lower_views, self.upper_views,
                self.settings["ground_seed_xy_enu_m"],
                self.settings.get("max_slope_deg"),
            )
            if "refinement" not in self.settings:
                self.finalize()
                return True
            save_prior(self.folder / "uav_map_nadir.npz", self.grid, self.baseline, self.timestamps[-1])
            self.schedule = schedule_oblique(self.grid, self.baseline, self.settings, now_s)
            self.capture_times = {int(item["capture_s"]) for item in self.schedule["stations"]}
            _write_json_atomic(self.folder / "oblique_survey.json", self.schedule)
            print(
                f"[ROS2 UAV mapper] {self.baseline_count} nadir frames; "
                f"planned {len(self.capture_times)} DUCPP oblique views "
                f"(strategy={self.schedule.get('strategy')}).",
                flush=True,
            )
            if not self.capture_times:
                self.finalize()
                return True

        if self.schedule is not None and now_s >= float(self.schedule["end_s"]) + 0.8:
            self.finalize()
            return True

        return False

    def finalize(self):
        if self.finished:
            return
        self.finished = True
        np.savez_compressed(
            self.folder / "observations.npz",
            timestamp_s=self.timestamps,
            gps_enu_m=self.origins,
            attitude_rpy_rad=self.attitudes,
            depth_m=self.depths,
            rgb=self.colors,
            size_wh=[self.width, self.height],
            fov_rad=self.fov,
            stride_px=self.stride,
            range_m=[self.min_range, self.max_range],
            body_rotation_enu=self.rotations,
            survey_phase=self.phases,
        )
        print(f"[ROS2 UAV mapper] Captured {len(self.timestamps)} synchronized observations.", flush=True)

        result = self.baseline
        if self.schedule is not None:
            result = refine_ground(
                self.grid,
                self.baseline,
                self.oblique_points,
                self.origins[self.baseline_count:],
                self.settings["max_slope_deg"],
                self.schedule,
            )
            print(f"[ROS2 UAV mapper] Oblique views recovered {int(result['refined_ground'].sum())} cells.", flush=True)

        fine_ground = compose_fine_ground(
            self.grid, self.fine_views, result["states"], result["ground_z_m"], subdiv=4
        )
        result["fine_ground_z_m"] = fine_ground
        result["fine_resolution_m"] = np.float32(self.grid.resolution / 4.0)
        print(
            f"[ROS2 UAV mapper] Exported {int(np.isfinite(fine_ground).sum())} fine ground cells "
            f"({self.grid.resolution / 4.0:.3f} m).",
            flush=True,
        )
        save_prior(self.folder / "uav_map.npz", self.grid, result, self.timestamps[-1])

        if self.victim_tracker is not None:
            confirmed = self.victim_tracker.export(min_views=2)
            payload = dict(
                detector=self.detector_mode,
                raw_hits=self.raw_victim_hits,
                candidate_count=len(confirmed),
                candidates=confirmed,
            )
            _write_json_atomic(self.folder / "aerial_victims.json", payload)
            print(
                f"[ROS2 UAV mapper] Aerial victim candidates: {len(confirmed)} confirmed "
                f"(from {self.raw_victim_hits} raw hits).",
                flush=True,
            )


def run_ros2_mapper(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter as RosParam
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import Image, Imu, NavSatFix

    rclpy.init()

    class _MapperNode(Node):
        def __init__(self):
            super().__init__(
                "uav_mapper_node",
                parameter_overrides=[RosParam("use_sim_time", RosParam.Type.BOOL, True)],
            )
            env_meta = {}
            if args.environment_json and Path(args.environment_json).is_file():
                env_meta = json.loads(Path(args.environment_json).read_text(encoding="utf-8"))
            self.core = UavMapperRos2Node(
                run_dir=args.run_dir,
                flight_end_s=args.flight_end,
                detector_mode=args.detector or None,
                gps_noise_m=args.gps_noise_m,
                spherical_origin=env_meta.get("spherical_origin"),
                record_fpv=args.record_fpv,
            )
            self._lock = threading.Lock()
            self._latest_rgb = None
            self._latest_depth = None
            self._latest_gps = None
            self._latest_imu = None
            self._fpv_count = 0

            # Each low-rate survey frame is used for a paired RGB-D observation.
            survey_qos = QoSProfile(depth=3, reliability=ReliabilityPolicy.RELIABLE)
            self.create_subscription(Image, "/drone/rgb", self._on_rgb, survey_qos)
            self.create_subscription(Image, "/drone/depth", self._on_depth, survey_qos)
            self.create_subscription(NavSatFix, "/drone/gps", self._on_gps, qos_profile_sensor_data)
            self.create_subscription(Imu, "/drone/imu", self._on_imu, qos_profile_sensor_data)
            if args.record_fpv:
                self.create_subscription(Image, "/drone/fpv_cam", self._on_fpv, qos_profile_sensor_data)
            self.create_timer(0.2, self._tick)

        def _on_rgb(self, msg):
            bgr = ros_image_to_bgr(msg.data, msg.width, msg.height, msg.encoding, msg.step)
            with self._lock:
                self._latest_rgb = (_stamp_to_sec(msg.header.stamp), bgr)

        def _on_depth(self, msg):
            depth = ros_depth_to_meters(msg.data, msg.width, msg.height, msg.encoding, msg.step)
            with self._lock:
                self._latest_depth = (_stamp_to_sec(msg.header.stamp), depth)

        def _on_gps(self, msg):
            enu = self.core.navsat.convert(msg.latitude, msg.longitude, msg.altitude, apply_noise=True)
            with self._lock:
                self._latest_gps = (_stamp_to_sec(msg.header.stamp), enu)

        def _on_imu(self, msg):
            q = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
            with self._lock:
                self._latest_imu = (_stamp_to_sec(msg.header.stamp), q)

        def _on_fpv(self, msg):
            if self.core.finished:
                return
            bgr = ros_image_to_bgr(msg.data, msg.width, msg.height, msg.encoding, msg.step)
            cv2.imwrite(
                str(self.core.fpv_dir / f"{self._fpv_count:05d}.jpg"),
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 88],
            )
            self._fpv_count += 1

        def _tick(self):
            now_s = self.get_clock().now().nanoseconds * 1e-9
            if now_s <= 0.0:
                return
            with self._lock:
                rgb = self._latest_rgb
                depth = self._latest_depth
                gps = self._latest_gps
                imu = self._latest_imu
            if rgb is None or depth is None or gps is None or imu is None:
                return
            # Check loose synchronization within 0.6 s.
            t_ref = depth[0] if depth[0] > 0 else now_s
            if abs(rgb[0] - t_ref) > 0.6 or abs(gps[0] - t_ref) > 0.6 or abs(imu[0] - t_ref) > 0.6:
                return
            done = self.core.ingest_frame(now_s, rgb[1], depth[1], gps[1], imu[1])
            if done:
                rclpy.shutdown()

    node = _MapperNode()
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
    parser.add_argument("--flight-end", required=True, type=float)
    parser.add_argument("--detector", default="", choices=("", "yolo", "surrogate_rgbd"))
    parser.add_argument("--gps-noise-m", type=float, default=0.04)
    parser.add_argument("--environment-json", type=Path, default=None)
    parser.add_argument("--record-fpv", action="store_true")
    args = parser.parse_args()
    run_ros2_mapper(args)


if __name__ == "__main__":
    main()
