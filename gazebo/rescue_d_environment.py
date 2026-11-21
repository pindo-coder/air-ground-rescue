"""Drive the D scene's prescribed flight and obstacle motion in simulation time."""

import json
from math import floor
from pathlib import Path

from gz.math7 import Pose3d
from gz.sim8 import Model, World

from motion import path_pose


class RescueDEnvironment:
    def configure(self, entity, sdf, ecm, event_mgr):
        self.world = World(entity)
        config_path = Path(__file__).with_name("environment.json")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.last_step = None

    def ground_height(self, x, y):
        grid = self.config["heights"]
        dx, dy = self.config["spacing"]
        x0, y0 = self.config["origin"]
        u, v = (x - x0) / dx, (y - y0) / dy
        i = min(len(grid[0]) - 2, max(0, floor(u)))
        j = min(len(grid) - 2, max(0, floor(v)))
        a = min(1, max(0, u - i))
        b = min(1, max(0, v - j))
        return ((1-b) * ((1-a)*grid[j][i] + a*grid[j][i+1]) +
                b * ((1-a)*grid[j+1][i] + a*grid[j+1][i+1]))

    def pre_update(self, info, ecm):
        if info.paused:
            return
        seconds = info.sim_time.total_seconds()
        step = floor(seconds * 10)
        if step == self.last_step:
            return
        self.last_step = step
        for route in self.config["routes"]:
            entity = self.world.model_by_name(ecm, route["model"])
            if not entity:
                continue
            (x, y), yaw = path_pose(route["path"], route["speed_mps"], seconds, route["phase_s"])
            altitude = route.get("altitude_enu_m")
            if altitude is None:
                altitude = self.ground_height(x, y) + route["height_m"] / 2
            Model(entity).set_world_pose_cmd(ecm, Pose3d(x, y, altitude, 0, 0, yaw))


def get_system():
    return RescueDEnvironment()
