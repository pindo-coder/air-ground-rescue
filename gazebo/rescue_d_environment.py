"""Drive the D scene's prescribed flight, DUCPP oblique survey, rover staging, and obstacle motion in Gazebo simulation time."""

import json
import math
import os
from pathlib import Path

from motion import path_pose

try:
    from gz.math7 import Pose3d
    from gz.sim8 import Model, World
except ImportError:  # Allows unit-testing motion/survey logic outside Gazebo.
    Pose3d = None
    Model = None
    World = None


def oblique_station_pose(schedule, now):
    """Interpolate 6-DoF drone pose (xyz, roll, pitch, yaw) along a DUCPP oblique schedule."""
    stations = schedule.get("stations") or []
    if not stations:
        return None
    position = list(map(float, schedule["start_xyz_m"]))
    last_time = float(schedule["start_s"])
    for station in stations:
        target = list(map(float, station["xyz_m"]))
        capture_s = float(station["capture_s"])
        arrival_s = float(station["arrival_s"])
        if now <= capture_s:
            denom = max(arrival_s - last_time, 1e-6)
            progress = min(1.0, max(0.0, (now - last_time) / denom))
            xyz = [p + progress * (t - p) for p, t in zip(position, target)]
            return xyz, 0.0, float(station["pitch_rad"]), float(station["yaw_rad"])
        position, last_time = target, capture_s
    last = stations[-1]
    return list(map(float, last["xyz_m"])), 0.0, float(last["pitch_rad"]), float(last["yaw_rad"])


def compute_survey_drone_pose(mission, oblique_schedule, now):
    """Compute exact ENU 6-DoF drone pose matching Webots uav_survey.py cadence."""
    if oblique_schedule and oblique_schedule.get("stations") and now >= float(oblique_schedule["start_s"]):
        pose = oblique_station_pose(oblique_schedule, now)
        if pose is not None:
            return pose
    flight = mission["flight"]
    end_s = float(mission["flight_end_s"])
    whole = math.floor(now)
    progress = min(whole + min(2.0 * (now - whole), 1.0), end_s)
    xy, heading = path_pose(flight["waypoints_xy_enu_m"], float(flight["speed_mps"]), progress)
    return [float(xy[0]), float(xy[1]), float(flight["altitude_enu_m"])], 0.0, 0.0, float(heading)


class RescueDEnvironment:
    def configure(self, entity, sdf, ecm, event_mgr):
        self.world = World(entity)
        config_path = Path(__file__).with_name("environment.json")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.last_step = None
        run_env = os.environ.get("GZ_UAV_SURVEY_RUN", "").strip()
        self.run_dir = Path(run_env) if run_env else None
        self.survey_mission = None
        self.oblique_schedule = None
        self.rover_parked = False
        self.rover_deployed = False
        if self.run_dir is not None:
            mission_file = self.run_dir / "map_config.json"
            if mission_file.is_file():
                mission = json.loads(mission_file.read_text(encoding="utf-8"))
                flight_end = os.environ.get("GZ_UAV_FLIGHT_END")
                if flight_end:
                    mission["flight_end_s"] = float(flight_end)
                elif "flight_end_s" not in mission:
                    pts = mission["flight"]["waypoints_xy_enu_m"]
                    dist = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
                    mission["flight_end_s"] = dist / max(float(mission["flight"]["speed_mps"]), 0.1)
                self.survey_mission = mission

    def ground_height(self, x, y):
        grid = self.config["heights"]
        dx, dy = self.config["spacing"]
        x0, y0 = self.config["origin"]
        u, v = (x - x0) / dx, (y - y0) / dy
        i = min(len(grid[0]) - 2, max(0, math.floor(u)))
        j = min(len(grid) - 2, max(0, math.floor(v)))
        a = min(1.0, max(0.0, u - i))
        b = min(1.0, max(0.0, v - j))
        return ((1.0 - b) * ((1.0 - a) * grid[j][i] + a * grid[j][i + 1]) +
                b * ((1.0 - a) * grid[j + 1][i] + a * grid[j + 1][i + 1]))

    def _maybe_read_json(self, path):
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _update_rover_staging(self, ecm):
        if self.run_dir is None or self.survey_mission is None:
            return
        rover = self.world.model_by_name(ecm, "rover")
        if not rover:
            return
        map_ready = (self.run_dir / "uav_map.npz").is_file()
        deploy_xy = self.survey_mission.get("deployment_xy_enu_m")
        if not map_ready and os.environ.get("GZ_PARK_ROVER_DURING_SURVEY", "1") == "1":
            if not self.rover_parked:
                Model(rover).set_world_pose_cmd(ecm, Pose3d(-60.0, -45.0, 2.0, 0.0, 0.0, 0.0))
                self.rover_parked = True
        elif map_ready and not self.rover_deployed and deploy_xy is not None:
            dx, dy = float(deploy_xy[0]), float(deploy_xy[1])
            dz = self.ground_height(dx, dy) + 0.02
            yaw = float(self.survey_mission.get("deployment_yaw_rad", 0.0))
            Model(rover).set_world_pose_cmd(ecm, Pose3d(dx, dy, dz, 0.0, 0.0, yaw))
            self.rover_deployed = True

    def pre_update(self, info, ecm):
        if info.paused:
            return
        seconds = info.sim_time.total_seconds()
        step = math.floor(seconds * 10)
        if step == self.last_step:
            return
        self.last_step = step

        self._update_rover_staging(ecm)

        # Optional live pose override from external ROS 2 controller (e.g., interactive re-survey).
        drone_override = None
        if self.run_dir is not None:
            drone_override = self._maybe_read_json(self.run_dir / "drone_pose_override.json")
            if self.oblique_schedule is None:
                self.oblique_schedule = self._maybe_read_json(self.run_dir / "oblique_survey.json")

        for route in self.config["routes"]:
            entity = self.world.model_by_name(ecm, route["model"])
            if not entity:
                continue
            if route["model"] == "drone":
                if drone_override is not None:
                    xyz = drone_override["xyz_m"]
                    rpy = drone_override.get("rpy_rad", [0.0, 0.0, 0.0])
                    Model(entity).set_world_pose_cmd(
                        ecm, Pose3d(xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2])
                    )
                    continue
                if self.survey_mission is not None:
                    xyz, roll, pitch, yaw = compute_survey_drone_pose(
                        self.survey_mission, self.oblique_schedule, seconds
                    )
                    Model(entity).set_world_pose_cmd(
                        ecm, Pose3d(xyz[0], xyz[1], xyz[2], roll, pitch, yaw)
                    )
                    continue
            (x, y), yaw = path_pose(route["path"], route["speed_mps"], seconds, route["phase_s"])
            altitude = route.get("altitude_enu_m")
            if altitude is None:
                altitude = self.ground_height(x, y) + route["height_m"] / 2
            Model(entity).set_world_pose_cmd(ecm, Pose3d(x, y, altitude, 0.0, 0.0, yaw))


def get_system():
    return RescueDEnvironment()
