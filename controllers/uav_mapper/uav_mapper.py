"""Webots survey capture. Mapping reads only the drone's on-board devices."""
from pathlib import Path
import json
import os
import sys

import numpy as np
from controller import Robot
from scipy.spatial.transform import Rotation

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "controllers/search_agent"))
from multiview import aerial_candidates, aerial_pixels_enu, attitude_matrix


def main():
    folder = ROOT / "runs" / os.environ["UAV_MAPPING_RUN"]
    deadline = float(os.environ["UAV_MAPPING_FLIGHT_END"])
    settings = json.loads((folder / "map_config.json").read_text(encoding="utf-8"))
    grid = Grid.from_bounds(settings["bounds_enu_m"], settings["resolution_m"])
    images = folder / "frames"
    images.mkdir(exist_ok=True)

    robot = Robot()
    camera, depth, gps, imu = (robot.getDevice(name) for name in
                             ("aerial_rgb", "aerial_depth", "aerial_gps", "aerial_attitude"))
    period_ms = 200
    for sensor in (camera, depth, gps, imu):
        sensor.enable(1000)
    fpv_cam = robot.getDevice("drone_fpv_cam")
    if fpv_cam is not None:
        fpv_cam.enable(period_ms)
    record_video = os.environ.get("UAV_RECORD_VIDEO") == "1"
    drone_video_dir = folder / "evaluation/drone_chase_frames"
    if record_video and fpv_cam is not None:
        drone_video_dir.mkdir(parents=True, exist_ok=True)
    drone_video_idx = 0
    width, height = depth.getWidth(), depth.getHeight()
    fov, min_range, max_range = depth.getFov(), depth.getMinRange(), depth.getMaxRange()
    stride = settings.get("sample_stride_px", 4)
    timestamps, origins, attitudes, depths, colors = [], [], [], [], []
    rotations, phases = [], []
    lower_views, upper_views, fine_views = [], [], []
    oblique_points = []
    baseline, schedule = None, None
    baseline_count = 0
    capture_times = set()
    detector_mode = os.environ.get("UAV_RESCUE_DETECTOR")
    victim_detector = None
    victim_tracker = None
    raw_victim_hits = 0
    if detector_mode:
        from aerial_victims import AerialVictimTracker
        from perception import PersonDetector
        common = json.loads((ROOT / "config/search/common.json").read_text(encoding="utf-8"))
        score_min = float(common["detection_score_min"][detector_mode])
        victim_detector = PersonDetector(detector_mode, ROOT, max_depth=24.0, field_of_view=fov)
        victim_tracker = AerialVictimTracker(association_radius_m=1.6, min_score=score_min)
        print(f"[UAV mapper] Aerial victim detector enabled: {detector_mode} (score>={score_min}).", flush=True)
    tick = 0
    while robot.step(period_ms) != -1:
        tick += 1
        now = robot.getTime()
        if record_video and fpv_cam is not None:
            fpv_cam.saveImage(str(drone_video_dir / f"{drone_video_idx:05d}.jpg"), 85)
            drone_video_idx += 1
        if tick % 5 == 0 and (baseline is None or tick // 5 in capture_times):
            origin = np.asarray(gps.getValues())
            attitude = np.asarray(imu.getRollPitchYaw())
            if np.isfinite(origin).all() and np.isfinite(attitude).all():
                orientation = (attitude_matrix(*attitude) if baseline is None
                               else Rotation.from_quat(imu.getQuaternion()).as_matrix())
                raw_depth = np.asarray(depth.getRangeImage(), dtype=np.float32).reshape(height, width)
                z = raw_depth[::stride, ::stride].copy()
                bgr_full = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape(height, width, 4)
                color = bgr_full[::stride, ::stride, 2::-1].copy()
                pixels, _ = sampled_pixels(z, stride, min_range, max_range)
                points = aerial_pixels_enu(pixels, width, height, fov, origin, attitude,
                                          body_rotation=orientation)
                fine_views.append(binned_fine_surface(grid, points, subdiv=4))
                if baseline is None:
                    low, high = binned_surfaces(grid, points)
                    lower_views.append(low)
                    upper_views.append(high)
                    if victim_detector is not None:
                        cands = aerial_candidates(victim_detector, bgr_full[:, :, :3].copy(), raw_depth, origin, attitude, fov)
                        raw_victim_hits += len(cands)
                        if cands:
                            victim_tracker.ingest_candidates(
                                cands, len(timestamps), now, origin, f"frames/{len(timestamps):03d}.jpg"
                            )
                else:
                    oblique_points.append(points)
                camera.saveImage(str(images / f"{len(timestamps):03d}.jpg"), 90)
                timestamps.append(now)
                origins.append(origin)
                attitudes.append(attitude)
                rotations.append(orientation)
                phases.append(int(baseline is not None))
                depths.append(z)
                colors.append(color)
        if baseline is None and now >= deadline + 0.8:
            if len(timestamps) < 2:
                raise RuntimeError("The survey did not collect enough usable observations.")
            baseline_count = len(timestamps)
            baseline = classify(grid, lower_views, upper_views, settings["ground_seed_xy_enu_m"],
                                settings.get("max_slope_deg"))
            if "refinement" not in settings:
                break
            save_prior(folder / "uav_map_nadir.npz", grid, baseline, timestamps[-1])
            schedule = schedule_oblique(grid, baseline, settings, now)
            capture_times = {item["capture_s"] for item in schedule["stations"]}
            temporary = folder / "oblique_survey.tmp"
            temporary.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
            temporary.replace(folder / "oblique_survey.json")
            print(f"[UAV mapper] {baseline_count} nadir frames; planned {len(capture_times)} oblique views.", flush=True)
            if not capture_times:
                break
        if schedule is not None and now >= schedule["end_s"] + 0.8:
            break
    else:
        return
    np.savez_compressed(folder / "observations.npz", timestamp_s=timestamps,
                        gps_enu_m=origins, attitude_rpy_rad=attitudes, depth_m=depths,
                        rgb=colors, size_wh=[width, height], fov_rad=fov, stride_px=stride,
                        range_m=[min_range, max_range], body_rotation_enu=rotations,
                        survey_phase=phases)
    print(f"[UAV mapper] Captured {len(timestamps)} synchronized observations.", flush=True)
    result = baseline
    if schedule is not None:
        result = refine_ground(grid, baseline, oblique_points, origins[baseline_count:],
                               settings["max_slope_deg"], schedule)
    fine_z = compose_fine_ground(grid, fine_views, result["states"], result["ground_z_m"], subdiv=4)
    result["fine_ground_z_m"] = fine_z
    result["fine_resolution_m"] = np.float32(grid.resolution / 4.0)
    save_prior(folder / "uav_map.npz", grid, result, timestamps[-1])
    if victim_tracker is not None:
        victims_payload = dict(
            detector=detector_mode,
            scanned_nadir_frames=baseline_count,
            raw_detections=raw_victim_hits,
            candidates=victim_tracker.export(min_views=2),
        )
        tmp_v = folder / "aerial_victims.tmp"
        tmp_v.write_text(json.dumps(victims_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_v.replace(folder / "aerial_victims.json")
        print(f"[UAV mapper] Published aerial_victims.json ({len(victims_payload['candidates'])} candidate(s)).",
              flush=True)
    print("[UAV mapper] Published uav_map.npz in ENU (with 0.125 m fine_ground_z_m).", flush=True)


if __name__ == "__main__":
    main()
