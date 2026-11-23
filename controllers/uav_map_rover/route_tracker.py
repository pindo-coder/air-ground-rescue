"""Closed-loop route tracking and onboard LiDAR/RGB-D obstacle avoidance.

This module consumes ONLY onboard rover sensors (GPS, IMU, wheel encoders,
3-layer LiDAR, RGB-D depth) and the UAV sensor-derived prior map (`uav_map.npz`).
It has zero access to Webots simulator ground truth.
"""
from collections import deque
from dataclasses import dataclass, field
import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from survey_map import BLOCKED, FREE, NEIGHBORS, UNKNOWN, Grid


WHEEL_RADIUS_M = 0.09
TRACK_WIDTH_M = 0.41
MAX_WHEEL_RAD_S = 9.5  # Slightly below ResearchRoverWheel maxVelocity (10 rad/s)

LIDAR_OFFSET_BODY = np.array([0.0, 0.0, 0.37], dtype=float)
LIDAR_LAYER_ELEVATIONS_RAD = (0.075, 0.0, -0.075)  # top, middle, bottom (vfov=0.15)

DEPTH_OFFSET_BODY = np.array([0.26, 0.0, 0.30], dtype=float)
DEPTH_PITCH_RAD = 0.08
DEPTH_FOV_RAD = math.pi / 2.0


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rotation_2d(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=float)


def gravity_rotation(roll, pitch):
    """Rotate body-frame vectors to gravity-aligned (level) rover frame."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    return ry @ rx


def wheel_velocities_rad_s(v_mps, w_rad_s):
    """Differential/skid-steer wheel angular speeds with proportional saturation."""
    left = (v_mps - 0.5 * TRACK_WIDTH_M * w_rad_s) / WHEEL_RADIUS_M
    right = (v_mps + 0.5 * TRACK_WIDTH_M * w_rad_s) / WHEEL_RADIUS_M
    peak = max(abs(left), abs(right), MAX_WHEEL_RAD_S)
    scale = MAX_WHEEL_RAD_S / peak
    return float(left * scale), float(right * scale)


def sample_ground_z(grid, ground_z_m, xy):
    """Bilinearly sample UAV sensor-derived ground elevation at ENU xy points."""
    pts = np.atleast_2d(np.asarray(xy, dtype=float))
    out = np.full(len(pts), np.nan, dtype=float)
    if len(pts) == 0:
        return out
    rows_n, cols_n = grid.shape
    c_f = (pts[:, 0] - grid.origin[0]) / grid.resolution - 0.5
    r_f = (pts[:, 1] - grid.origin[1]) / grid.resolution - 0.5

    r0 = np.floor(r_f).astype(int)
    c0 = np.floor(c_f).astype(int)
    dr = r_f - r0
    dc = c_f - c0

    in_bilin = (r0 >= 0) & (r0 + 1 < rows_n) & (c0 >= 0) & (c0 + 1 < cols_n)
    if np.any(in_bilin):
        rr, cc = r0[in_bilin], c0[in_bilin]
        z00 = ground_z_m[rr, cc]
        z01 = ground_z_m[rr, cc + 1]
        z10 = ground_z_m[rr + 1, cc]
        z11 = ground_z_m[rr + 1, cc + 1]
        valid = np.isfinite(z00) & np.isfinite(z01) & np.isfinite(z10) & np.isfinite(z11)
        idx = np.flatnonzero(in_bilin)[valid]
        if idx.size:
            a = dc[idx]
            b = dr[idx]
            out[idx] = ((1.0 - b) * ((1.0 - a) * z00[valid] + a * z01[valid])
                        + b * ((1.0 - a) * z10[valid] + a * z11[valid]))

    missing = ~np.isfinite(out)
    if np.any(missing):
        rn = np.floor((pts[missing, 1] - grid.origin[1]) / grid.resolution).astype(int)
        cn = np.floor((pts[missing, 0] - grid.origin[0]) / grid.resolution).astype(int)
        inside = (rn >= 0) & (rn < rows_n) & (cn >= 0) & (cn < cols_n)
        idx = np.flatnonzero(missing)[inside]
        if idx.size:
            out[idx] = ground_z_m[rn[inside], cn[inside]]
    return out


def _bin_points_2d(body_xy, enu_xy, cell_size=0.10, min_count=2):
    """Filter sparse sensor noise by requiring multiple hits per fine spatial bin."""
    if len(body_xy) == 0:
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float)
    keys = np.floor(body_xy / cell_size).astype(int)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    keep_bins = np.flatnonzero(counts >= min_count)
    if keep_bins.size == 0:
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float)
    b_out = np.empty((keep_bins.size, 2), dtype=float)
    e_out = np.empty((keep_bins.size, 2), dtype=float)
    for k, bin_id in enumerate(keep_bins):
        mask = inverse == bin_id
        b_out[k] = body_xy[mask].mean(axis=0)
        e_out[k] = enu_xy[mask].mean(axis=0)
    return b_out, e_out


def extract_lidar_obstacles(layer_ranges, pose_xyz, rpy, grid, ground_z_m,
                            z_bias=0.0, max_range=5.0,
                            elevations=LIDAR_LAYER_ELEVATIONS_RAD):
    """Extract obstacle points from multi-layer LiDAR, rejecting smooth sloped ground.

    Returns (body_xy, enu_xy) arrays of shape (N, 2).
    """
    pose_xyz = np.asarray(pose_xyz, dtype=float)
    roll, pitch, yaw = map(float, rpy)
    r_grav = gravity_rotation(roll, pitch)
    r_yaw = rotation_2d(yaw)

    body_chunks = []
    for ranges, el in zip(layer_ranges, elevations):
        arr = np.asarray(ranges, dtype=float).ravel()
        n = arr.size
        if n == 0:
            continue
        angles = np.linspace(math.pi, -math.pi, n, endpoint=False)
        valid = np.isfinite(arr) & (arr > 0.28) & (arr < min(max_range, 7.8))
        if not np.any(valid):
            continue
        r = arr[valid]
        az = angles[valid]
        ce, se = math.cos(el), math.sin(el)
        bx = r * ce * np.cos(az)
        by = r * ce * np.sin(az)
        bz = LIDAR_OFFSET_BODY[2] + r * se
        not_self = ~((np.abs(bx) < 0.32) & (np.abs(by) < 0.27))
        if np.any(not_self):
            body_chunks.append(np.column_stack((bx[not_self], by[not_self], bz[not_self])))

    if not body_chunks:
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float)

    body = np.vstack(body_chunks)
    grav = body @ r_grav.T
    enu_xy = pose_xyz[:2] + grav[:, :2] @ r_yaw.T
    enu_z = pose_xyz[2] + grav[:, 2]
    horiz = np.hypot(body[:, 0], body[:, 1])

    prior_z = sample_ground_z(grid, ground_z_m, enu_xy)
    has_prior = np.isfinite(prior_z)
    h_above = (enu_z - z_bias) - prior_z

    obs_known = has_prior & (h_above >= 0.16) & (h_above <= 1.50)
    obs_unknown = (~has_prior) & (body[:, 2] >= 0.18 + 0.26 * horiz) & (body[:, 2] <= 1.50)
    keep = obs_known | obs_unknown
    if not np.any(keep):
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float)

    body_obs = grav[keep, :2]
    enu_obs = enu_xy[keep]
    close = np.hypot(body_obs[:, 0], body_obs[:, 1]) <= 1.15
    far_b, far_e = _bin_points_2d(body_obs[~close], enu_obs[~close], cell_size=0.10, min_count=2)
    if np.any(close):
        close_b, close_e = _bin_points_2d(body_obs[close], enu_obs[close], cell_size=0.08, min_count=1)
        if len(far_b):
            return np.vstack((close_b, far_b)), np.vstack((close_e, far_e))
        return close_b, close_e
    return far_b, far_e


def extract_rgbd_obstacles(depth_image, pose_xyz, rpy, grid, ground_z_m,
                           z_bias=0.0, fov=DEPTH_FOV_RAD, stride=6, max_range=4.0):
    """Extract low obstacles and drop hazards from onboard RGB-D depth.

    Returns (body_xy, enu_xy, drop_count).
    """
    depth = np.asarray(depth_image, dtype=float)
    if depth.ndim != 2 or depth.size == 0:
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float), 0
    height, width = depth.shape
    sub = depth[::stride, ::stride]
    valid = np.isfinite(sub) & (sub > 0.25) & (sub < max_range)
    if not np.any(valid):
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float), 0

    rows, cols = np.nonzero(valid)
    u = cols * stride
    v = rows * stride
    d = sub[valid]

    focal = width / (2.0 * math.tan(fov / 2.0))
    cam_pts = np.column_stack((
        d,
        -(u - (width - 1) / 2.0) * d / focal,
        -(v - (height - 1) / 2.0) * d / focal,
    ))
    cp, sp = math.cos(DEPTH_PITCH_RAD), math.sin(DEPTH_PITCH_RAD)
    ry_cam = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    body = cam_pts @ ry_cam.T + DEPTH_OFFSET_BODY

    not_self = ~((body[:, 0] < 0.34) & (np.abs(body[:, 1]) < 0.28) & (body[:, 2] < 0.36))
    body = body[not_self]
    if len(body) == 0:
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float), 0

    pose_xyz = np.asarray(pose_xyz, dtype=float)
    roll, pitch, yaw = map(float, rpy)
    grav = body @ gravity_rotation(roll, pitch).T
    enu_xy = pose_xyz[:2] + grav[:, :2] @ rotation_2d(yaw).T
    enu_z = pose_xyz[2] + grav[:, 2]
    horiz = np.hypot(grav[:, 0], grav[:, 1])

    prior_z = sample_ground_z(grid, ground_z_m, enu_xy)
    has_prior = np.isfinite(prior_z)
    h_above = (enu_z - z_bias) - prior_z

    raised_known = has_prior & (horiz <= 3.8) & (h_above >= 0.15) & (h_above <= 1.45)
    drop_known = has_prior & (horiz <= 2.8) & (h_above <= -0.28)
    raised_unknown = (~has_prior) & (horiz <= 3.2) & (body[:, 2] >= 0.18 + 0.28 * horiz) & (body[:, 2] <= 1.45)
    drop_unknown = (~has_prior) & (horiz <= 2.4) & (body[:, 2] <= -0.28 - 0.32 * horiz)

    drops = drop_known | drop_unknown
    keep = raised_known | raised_unknown | drops
    if not np.any(keep):
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float), 0

    body_obs, enu_obs = _bin_points_2d(grav[keep, :2], enu_xy[keep], cell_size=0.10, min_count=3)
    drop_bins, _ = _bin_points_2d(grav[drops, :2], enu_xy[drops], cell_size=0.14, min_count=4)
    return body_obs, enu_obs, int(len(drop_bins))


class RoverPoseEstimator:
    """Fuse onboard wheel encoders, IMU attitude, and noisy GPS in ENU."""

    def __init__(self, initial_xyz, initial_yaw):
        self.xyz = np.asarray(initial_xyz, dtype=float).copy()
        self.yaw = float(initial_yaw)
        self.prev_wheels = None
        self.prev_yaw = float(initial_yaw)
        self.gps_window = deque(maxlen=12)
        self.gps_window.append(self.xyz[:2].copy())
        self.yaw_window = deque(maxlen=6)
        self.yaw_window.append(float(initial_yaw))
        self.speed_mps = 0.0
        self.yaw_rate_rad_s = 0.0
        self.z_bias_m = 0.0

    def update(self, gps_xyz, rpy, wheel_rad, dt, grid=None, ground_z_m=None):
        gps_xyz = np.asarray(gps_xyz, dtype=float)
        wheel_rad = np.asarray(wheel_rad, dtype=float)
        roll, pitch, yaw = map(float, rpy)
        dt = max(float(dt), 1e-3)

        if self.prev_wheels is None:
            self.prev_wheels = wheel_rad.copy()
        dl, dr = (wheel_rad - self.prev_wheels) * WHEEL_RADIUS_M
        self.prev_wheels = wheel_rad.copy()

        ds = 0.5 * (dl + dr) * math.cos(pitch)
        mid_yaw = wrap_angle(self.prev_yaw + 0.5 * wrap_angle(yaw - self.prev_yaw))
        pred_xy = self.xyz[:2] + np.array([ds * math.cos(mid_yaw), ds * math.sin(mid_yaw)])

        if np.isfinite(gps_xyz).all():
            self.gps_window.append(gps_xyz[:2].copy())
            self.xyz[:2] = 0.75 * pred_xy + 0.25 * gps_xyz[:2]
            self.xyz[2] = 0.72 * self.xyz[2] + 0.28 * gps_xyz[2]
        else:
            self.xyz[:2] = pred_xy

        self.yaw_window.append(yaw)
        wheel_speed = abs(ds) / dt
        wheel_yaw_rate = abs(dr - dl) / (TRACK_WIDTH_M * dt)

        # Use multi-step averaged GPS/IMU windows to detect genuine slope slip or rotation
        # without being tripped by single-step 4 cm GPS or 3 mrad IMU noise.
        gps_slide_speed = 0.0
        if len(self.gps_window) >= 12:
            pts = np.asarray(self.gps_window)
            early = pts[:5].mean(axis=0)
            late = pts[-5:].mean(axis=0)
            drift = float(np.linalg.norm(late - early))
            gps_slide_speed = max(0.0, drift - 0.06) / (7.0 * dt)

        imu_yaw_rate = 0.0
        if len(self.yaw_window) >= 6:
            dyaw = abs(wrap_angle(self.yaw_window[-1] - self.yaw_window[0]))
            imu_yaw_rate = max(0.0, dyaw - 0.018) / ((len(self.yaw_window) - 1) * dt)

        raw_speed = max(wheel_speed, gps_slide_speed)
        raw_yaw_rate = max(wheel_yaw_rate, imu_yaw_rate)
        self.speed_mps = 0.50 * self.speed_mps + 0.50 * raw_speed
        self.yaw_rate_rad_s = 0.50 * self.yaw_rate_rad_s + 0.50 * raw_yaw_rate
        self.yaw = yaw
        self.prev_yaw = yaw

        if grid is not None and ground_z_m is not None:
            gz = sample_ground_z(grid, ground_z_m, self.xyz[:2])[0]
            if np.isfinite(gz):
                sample_bias = float(np.clip(self.xyz[2] - gz, -0.15, 0.15))
                self.z_bias_m = 0.90 * self.z_bias_m + 0.10 * sample_bias

        return self.xyz.copy(), self.yaw


def _astar_on_mask(grid, usable, start_xy, goal_xy, planning_cost=None):
    """8-connected A* on an arbitrary boolean traversability mask."""
    start, goal = grid.cell(start_xy), grid.cell(goal_xy)
    for label, cell in (("Start", start), ("Goal", goal)):
        if not grid.contains(cell) or not usable[cell]:
            raise ValueError(f"{label} has no confirmed space for the rover footprint.")
    moves = (*NEIGHBORS, (-1, -1), (-1, 1), (1, -1), (1, 1))
    queue = [(math.dist(start, goal), 0.0, start)]
    costs = {start: 0.0}
    parents = {start: None}
    while queue:
        _, cost, cell = heapq.heappop(queue)
        if cost > costs[cell]:
            continue
        if cell == goal:
            path = []
            while cell is not None:
                path.append(grid.center(cell))
                cell = parents[cell]
            return np.asarray(path[::-1], dtype=float)
        for dr, dc in moves:
            other = (cell[0] + dr, cell[1] + dc)
            if not grid.contains(other) or not usable[other]:
                continue
            if dr and dc and not (usable[other[0], cell[1]] and usable[cell[0], other[1]]):
                continue
            weight = 1.0 if planning_cost is None else 0.5 * (planning_cost[cell] + planning_cost[other])
            candidate = cost + weight * math.hypot(dr, dc)
            if candidate >= costs.get(other, math.inf):
                continue
            costs[other] = candidate
            parents[other] = cell
            heapq.heappush(queue, (candidate + math.dist(other, goal), candidate, other))
    raise ValueError("No route on currently confirmed terrain.")


class OnboardObstacleMap:
    """Fine-resolution (0.125 m) onboard obstacle overlay on top of the UAV prior.

    UAV `UNKNOWN` and `BLOCKED` cells are strictly preserved; onboard LiDAR and
    RGB-D detections stamp obstacles only at fine resolution so localized rocks
    in narrow corridors do not artificially seal a passable confirmed gap, while
    true corridor-spanning barriers cleanly invalidate all confirmed routes.
    """

    def __init__(self, grid, prior_states, ground_z_m, planning_cost, rover_radius_m,
                 confirm_frames=2, subdiv=4, prior_margin_m=0.25, obstacle_margin_m=0.33):
        self.grid = grid
        self.prior_states = np.asarray(prior_states, dtype=np.int8).copy()
        self.ground_z_m = np.asarray(ground_z_m, dtype=float).copy()
        self.planning_cost = (None if planning_cost is None
                              else np.asarray(planning_cost, dtype=float).copy())
        self.rover_radius_m = float(rover_radius_m)
        self.confirm_frames = int(confirm_frames)
        self.subdiv = int(subdiv)
        self.prior_margin_m = float(prior_margin_m)
        self.obstacle_margin_m = float(obstacle_margin_m)

        self.fine_grid = Grid(
            grid.origin,
            grid.resolution / self.subdiv,
            (grid.shape[0] * self.subdiv, grid.shape[1] * self.subdiv),
        )
        self.fine_prior = np.repeat(np.repeat(self.prior_states, self.subdiv, axis=0), self.subdiv, axis=1)
        if self.planning_cost is not None:
            raw_cost = np.where(np.isfinite(self.planning_cost), self.planning_cost, 1.0)
            self.fine_cost = np.repeat(np.repeat(raw_cost, self.subdiv, axis=0), self.subdiv, axis=1).astype(float)
        else:
            self.fine_cost = None

        self.confirm_counts = np.zeros(self.fine_grid.shape, dtype=np.int16)
        self.miss_counts = np.zeros(self.fine_grid.shape, dtype=np.int16)
        self.fine_onboard_blocked = np.zeros(self.fine_grid.shape, dtype=bool)
        self.stale_clear_ticks = 18  # 1.8 s of clear close-range LiDAR view frees vacated dynamic cells.
        self.cell_sources = {}  # fine (r, c) -> set of {"lidar", "rgbd"}
        self.coarse_sources = {}  # coarse (r, c) -> set of {"lidar", "rgbd"}
        self._dirty = True
        self._effective_cache = None
        self._usable_cache = None
        self._clearance_cache = None

    @property
    def onboard_blocked(self):
        """Coarse-grid view of cells containing confirmed onboard obstacles."""
        r_n, c_n = self.grid.shape
        return self.fine_onboard_blocked.reshape(r_n, self.subdiv, c_n, self.subdiv).any(axis=(1, 3))

    @property
    def effective_states(self):
        if self._dirty or self._effective_cache is None:
            self._effective_cache = np.where(
                self.fine_onboard_blocked, BLOCKED, self.fine_prior
            ).astype(np.int8)
            prior_open = self.fine_prior == FREE
            prior_dist = distance_transform_edt(np.pad(prior_open, 1))[1:-1, 1:-1]
            prior_clear = self.fine_grid.resolution * (prior_dist - 0.5)

            obs_open = ~self.fine_onboard_blocked
            obs_dist = distance_transform_edt(np.pad(obs_open, 1))[1:-1, 1:-1]
            obs_clear = self.fine_grid.resolution * (obs_dist - 0.5)

            self._clearance_cache = np.minimum(prior_clear, obs_clear)
            self._usable_cache = (
                prior_open
                & obs_open
                & (prior_clear >= self.prior_margin_m)
                & (obs_clear >= self.obstacle_margin_m)
            )
            self._dirty = False
        return self._effective_cache

    @property
    def usable_mask(self):
        _ = self.effective_states
        return self._usable_cache

    def _count_in_fine_cells(self, points_enu, rover_xy, max_dist_m=3.6):
        counts = {}
        if len(points_enu) == 0:
            return counts
        dists = np.linalg.norm(points_enu - rover_xy, axis=1)
        pts = points_enu[dists <= max_dist_m]
        if len(pts) == 0:
            return counts
        cols = np.floor((pts[:, 0] - self.fine_grid.origin[0]) / self.fine_grid.resolution).astype(int)
        rows = np.floor((pts[:, 1] - self.fine_grid.origin[1]) / self.fine_grid.resolution).astype(int)
        inside = ((rows >= 0) & (rows < self.fine_grid.shape[0])
                  & (cols >= 0) & (cols < self.fine_grid.shape[1]))
        for r, c in zip(rows[inside], cols[inside]):
            key = (int(r), int(c))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def update(self, rover_xy, lidar_enu_xy, rgbd_enu_xy, max_dist_m=3.6):
        """Integrate onboard detections into the fine grid on UAV-FREE cells only.

        Returns newly blocked fine cells [(fr, fc, source), ...].
        """
        rover_xy = np.asarray(rover_xy, dtype=float)
        lidar_counts = self._count_in_fine_cells(lidar_enu_xy, rover_xy, max_dist_m)
        rgbd_counts = self._count_in_fine_cells(rgbd_enu_xy, rover_xy, max_dist_m)

        active = {}
        for cell, cnt in lidar_counts.items():
            if cnt >= 1:
                active.setdefault(cell, set()).add("lidar")
        for cell, cnt in rgbd_counts.items():
            if cnt >= 1:
                active.setdefault(cell, set()).add("rgbd")

        # Keep only cells the UAV prior marked FREE and outside the rover's immediate footprint.
        filtered = {}
        for (r, c), sources in active.items():
            if self.fine_prior[r, c] != FREE:
                continue
            center = self.fine_grid.center((r, c))
            if float(np.linalg.norm(center - rover_xy)) < 0.30:
                continue
            filtered[(r, c)] = sources

        unlatched = (~self.fine_onboard_blocked) & (self.confirm_counts > 0)
        if np.any(unlatched):
            for r, c in map(tuple, np.argwhere(unlatched)):
                if (r, c) not in filtered:
                    self.confirm_counts[r, c] = max(0, int(self.confirm_counts[r, c]) - 1)

        # Dynamic obstacle clearing: if a previously latched onboard cell is within close 360 deg
        # LiDAR range (0.45..2.6 m) and gets zero hits for `stale_clear_ticks`, un-latch it so a
        # moving obstacle does not leave a permanent ghost wall across a passable corridor.
        latched = np.argwhere(self.fine_onboard_blocked)
        if len(latched):
            for r, c in latched:
                key = (int(r), int(c))
                if key in filtered:
                    self.miss_counts[r, c] = 0
                    continue
                center = self.fine_grid.center(key)
                dist = float(np.linalg.norm(center - rover_xy))
                if 0.45 <= dist <= 2.6:
                    self.miss_counts[r, c] = int(self.miss_counts[r, c]) + 1
                    if self.miss_counts[r, c] >= self.stale_clear_ticks:
                        self.fine_onboard_blocked[r, c] = False
                        self.confirm_counts[r, c] = 0
                        self.miss_counts[r, c] = 0
                        self._dirty = True

        newly_blocked = []
        for (r, c), sources in filtered.items():
            self.miss_counts[r, c] = 0
            self.confirm_counts[r, c] = min(10, int(self.confirm_counts[r, c]) + 1)
            self.cell_sources.setdefault((r, c), set()).update(sources)
            self.coarse_sources.setdefault((r // self.subdiv, c // self.subdiv), set()).update(sources)
            if not self.fine_onboard_blocked[r, c] and self.confirm_counts[r, c] >= self.confirm_frames:
                self.fine_onboard_blocked[r, c] = True
                self._dirty = True
                newly_blocked.append((r, c, "+".join(sorted(self.cell_sources[(r, c)]))))
        return newly_blocked

    def continuous_clearance(self, xy):
        """Continuous clearance to the nearest UAV non-FREE cell boundary."""
        xy = np.asarray(xy, dtype=float)
        cell = self.grid.cell(xy)
        if not self.grid.contains(cell) or self.prior_states[cell] != FREE:
            return 0.0
        res = self.grid.resolution
        ox, oy = self.grid.origin
        r0, c0 = cell
        min_d = res
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r0 + dr, c0 + dc
                if not self.grid.contains((nr, nc)) or self.prior_states[nr, nc] != FREE:
                    x0 = ox + nc * res
                    y0 = oy + nr * res
                    dx = max(x0 - xy[0], 0.0, xy[0] - (x0 + res))
                    dy = max(y0 - xy[1], 0.0, xy[1] - (y0 + res))
                    d = math.hypot(dx, dy)
                    if d < min_d:
                        min_d = d
        return float(min_d)

    def _segment_confirmed_free(self, a, b, obstacles_enu=None, min_obs_dist=0.33):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        dist = float(np.linalg.norm(b - a))
        steps = max(2, int(math.ceil(dist / 0.08)) + 1)
        states = self.effective_states
        for t in np.linspace(0.0, 1.0, steps):
            p = a + t * (b - a)
            fcell = self.fine_grid.cell(p)
            if not self.fine_grid.contains(fcell) or states[fcell] != FREE:
                return False
            if obstacles_enu is not None and len(obstacles_enu):
                if float(np.min(np.linalg.norm(obstacles_enu - p, axis=1))) < min_obs_dist:
                    return False
        return True

    def route_needs_replan(self, route_xy, waypoint_idx, rover_xy, local_obstacles_enu):
        """Check whether onboard obstacles invalidate the upcoming route."""
        if len(route_xy) == 0:
            return True, "EMPTY_ROUTE"
        usable = self.usable_mask
        start_i = max(0, min(waypoint_idx, len(route_xy) - 1))

        # Sample along upcoming segments (not just sparse vertices) so obstacles between
        # two 0.5 m waypoints are caught immediately.
        pts = np.asarray(route_xy, dtype=float)
        for i in range(start_i, len(pts)):
            fcell = self.fine_grid.cell(pts[i])
            if not self.fine_grid.contains(fcell) or not usable[fcell]:
                return True, f"ROUTE_CELL_INVALIDATED@{fcell}"
            if i + 1 < len(pts):
                mid = 0.5 * (pts[i] + pts[i + 1])
                mcell = self.fine_grid.cell(mid)
                if not self.fine_grid.contains(mcell) or not usable[mcell]:
                    return True, f"ROUTE_SEG_INVALIDATED@{mcell}"

        if len(local_obstacles_enu):
            obs = np.asarray(local_obstacles_enu, dtype=float)
            near = obs[np.linalg.norm(obs - rover_xy, axis=1) <= 3.2]
            if len(near):
                # Only trigger on obstacles sitting inside UAV-FREE space near upcoming path.
                look_end = min(len(pts), start_i + 10)
                for i in range(start_i, look_end):
                    d_arr = np.linalg.norm(near - pts[i], axis=1)
                    j = int(np.argmin(d_arr))
                    if d_arr[j] < 0.34:
                        ocell = self.grid.cell(near[j])
                        if self.grid.contains(ocell) and self.prior_states[ocell] == FREE:
                            return True, f"ONBOARD_OBSTACLE_ON_ROUTE@wp{i}"
        return False, ""

    def replan(self, rover_xy, goal_xy, local_obstacles_enu=None):
        """Replan on the fine 0.125 m confirmed-FREE grid excluding onboard obstacles.

        Returns (route_xy or None, reason_str).
        """
        rover_xy = np.asarray(rover_xy, dtype=float)
        goal_xy = np.asarray(goal_xy, dtype=float)
        usable = self.usable_mask
        # Add soft repulsion near onboard obstacles and prior boundaries so detours stay centered.
        base_cost = np.ones(self.fine_grid.shape, dtype=float) if self.fine_cost is None else self.fine_cost.copy()
        edge_pen = np.where(self._clearance_cache < 0.50, 0.8 * (0.50 - self._clearance_cache) / 0.25, 0.0)
        plan_cost = base_cost + edge_pen

        try:
            raw = _astar_on_mask(self.fine_grid, usable, rover_xy, goal_xy, plan_cost)
            return self._thin_route(raw, spacing_m=0.25), "REPLANNED_DIRECT"
        except ValueError as direct_err:
            rover_fcell = self.fine_grid.cell(rover_xy)
            if not (self.fine_grid.contains(rover_fcell) and usable[rover_fcell]):
                candidates = []
                for dr in range(-5, 6):
                    for dc in range(-5, 6):
                        cand = (rover_fcell[0] + dr, rover_fcell[1] + dc)
                        if self.fine_grid.contains(cand) and usable[cand]:
                            center = self.fine_grid.center(cand)
                            d = float(np.linalg.norm(center - rover_xy))
                            if d <= 0.70 and self._segment_confirmed_free(rover_xy, center, local_obstacles_enu):
                                candidates.append((d, center))
                for _, anchor in sorted(candidates, key=lambda item: item[0]):
                    try:
                        raw = _astar_on_mask(self.fine_grid, usable, anchor, goal_xy, plan_cost)
                        thinned = self._thin_route(raw, spacing_m=0.25)
                        return np.vstack((rover_xy[None, :], thinned)), "REPLANNED_FROM_NEAREST_CONFIRMED"
                    except ValueError:
                        continue

            has_onboard = bool(np.any(self.fine_onboard_blocked))
            near_unknown = False
            for r, c in map(tuple, np.argwhere(self.onboard_blocked)):
                r0, r1 = max(0, r - 3), min(self.grid.shape[0], r + 4)
                c0, c1 = max(0, c - 3), min(self.grid.shape[1], c + 4)
                if np.any(self.prior_states[r0:r1, c0:c1] == UNKNOWN):
                    near_unknown = True
                    break
            if has_onboard and near_unknown:
                return None, ("OBSTACLE_BLOCKED_NO_CONFIRMED_DETOUR: onboard obstacle blocks "
                              "corridor; detour would enter unconfirmed (unknown) terrain")
            if has_onboard:
                return None, ("OBSTACLE_BLOCKED_NO_CONFIRMED_DETOUR: onboard obstacle blocks "
                              "all confirmed traversable routes to goal")
            return None, f"UNCONFIRMED_OR_BLOCKED_ROUTE: {direct_err}"

    @staticmethod
    def _thin_route(route, spacing_m=0.25):
        if len(route) <= 2:
            return route
        kept = [route[0]]
        for pt in route[1:-1]:
            if float(np.linalg.norm(pt - kept[-1])) >= spacing_m:
                kept.append(pt)
        kept.append(route[-1])
        return np.asarray(kept, dtype=float)


@dataclass
class LocalPlannerInput:
    """Standardized input interface for rule-based and future neural local planners."""
    pose_xy: np.ndarray
    yaw: float
    attitude_rp: tuple
    speed_mps: float
    prev_cmd: tuple
    lookahead_xy: np.ndarray
    goal_xy: np.ndarray
    dist_to_goal: float
    obstacles_body_xy: np.ndarray
    lidar_obstacles_body_xy: np.ndarray
    rgbd_obstacles_body_xy: np.ndarray
    onboard_map: OnboardObstacleMap
    rgbd_frame: np.ndarray | None = None


class RuleLocalPlanner:
    """Rule-based dynamic-window velocity-rollout controller (baseline for future NN)."""

    def __init__(self, goal_tolerance_m=0.32, max_speed_mps=0.42, max_yaw_rate=0.85,
                 footprint_radius_m=0.33, min_map_clearance_m=0.20,
                 max_roll_rad=0.42, max_pitch_rad=0.38):
        self.goal_tolerance_m = float(goal_tolerance_m)
        self.max_speed_mps = float(max_speed_mps)
        self.max_yaw_rate = float(max_yaw_rate)
        self.footprint_radius_m = float(footprint_radius_m)
        self.min_map_clearance_m = float(min_map_clearance_m)
        self.max_roll_rad = float(max_roll_rad)
        self.max_pitch_rad = float(max_pitch_rad)

    @staticmethod
    def _nearest_obstacle(inp):
        points = inp.obstacles_body_xy
        return float(np.min(np.linalg.norm(points, axis=1))) if len(points) else float("inf")

    def _speed_cap(self, inp, min_obs_now):
        roll, pitch = inp.attitude_rp
        v_cap = self.max_speed_mps
        if inp.dist_to_goal < 1.0:
            v_cap = min(v_cap, max(0.14, 0.48 * inp.dist_to_goal))
        if min_obs_now < 1.25:
            v_cap = min(v_cap, 0.22)
        if max(abs(roll), abs(pitch)) > 0.20:
            v_cap = min(v_cap, 0.28)
        return v_cap

    def _rollout(self, inp, v, w, min_obs_now, now_map_clear, r_world_to_body):
        """Apply the same six-step clearance test to expert or neural commands."""
        stop_margin = (v * v) / (2.0 * 0.70)
        req_obs = (0.27 if v == 0.0 else self.footprint_radius_m) + stop_margin
        x, y, th = float(inp.pose_xy[0]), float(inp.pose_xy[1]), float(inp.yaw)
        rollout_min_obs, rollout_min_map = min_obs_now, now_map_clear
        for _ in range(6):
            x += v * math.cos(th) * 0.20
            y += v * math.sin(th) * 0.20
            th = wrap_angle(th + w * 0.20)

            if v > 0.0:
                clearance = inp.onboard_map.continuous_clearance((x, y))
                rollout_min_map = min(rollout_min_map, clearance)
                if clearance <= 0.0 or (clearance < self.min_map_clearance_m
                                        and clearance < now_map_clear - 0.02):
                    return None

            points = inp.obstacles_body_xy
            if len(points):
                local = (np.array([x, y]) - inp.pose_xy) @ r_world_to_body.T
                distance = float(np.min(np.linalg.norm(points - local, axis=1)))
                rollout_min_obs = min(rollout_min_obs, distance)
                if distance < req_obs and (v > 0.0 or distance < min_obs_now - 0.01):
                    return None
        return x, y, th, rollout_min_obs, rollout_min_map

    def allow_action(self, inp, action):
        """Validate an external velocity suggestion against onboard safety limits."""
        v, w = map(float, action)
        if not np.isfinite([v, w]).all():
            return False
        roll, pitch = inp.attitude_rp
        if abs(roll) > self.max_roll_rad or abs(pitch) > self.max_pitch_rad:
            return False
        if inp.dist_to_goal <= self.goal_tolerance_m:
            return v == 0.0 and w == 0.0
        nearest = self._nearest_obstacle(inp)
        if not 0.0 <= v <= self._speed_cap(inp, nearest) + 0.02 or abs(w) > self.max_yaw_rate + 1e-6:
            return False
        if v > 0.0 and abs(v - inp.prev_cmd[0]) > 0.18 + 1e-6:
            return False
        if v == 0.0 and abs(w) > 0.0 and nearest < 0.28:
            return False
        if v == 0.0 and w == 0.0:
            return True
        return self._rollout(inp, v, w, nearest, inp.onboard_map.continuous_clearance(inp.pose_xy),
                             rotation_2d(-inp.yaw)) is not None

    def compute(self, inp: LocalPlannerInput):
        roll, pitch = inp.attitude_rp
        if abs(roll) > self.max_roll_rad or abs(pitch) > self.max_pitch_rad:
            return 0.0, 0.0, {"mode": "UNSAFE_ATTITUDE", "blocked_locally": True}

        if inp.dist_to_goal <= self.goal_tolerance_m:
            return 0.0, 0.0, {"mode": "SETTLING_AT_GOAL", "blocked_locally": False}

        min_obs_now = self._nearest_obstacle(inp)
        now_map_clear = inp.onboard_map.continuous_clearance(inp.pose_xy)

        delta = inp.lookahead_xy - inp.pose_xy
        desired_heading = math.atan2(delta[1], delta[0])
        heading_err = wrap_angle(desired_heading - inp.yaw)

        if abs(heading_err) > 0.62:
            if min_obs_now >= 0.28:
                w_cmd = float(np.clip(1.65 * heading_err, -self.max_yaw_rate, self.max_yaw_rate))
                return 0.0, w_cmd, {
                    "mode": "ALIGN_HEADING",
                    "blocked_locally": False,
                    "min_obs_m": min_obs_now,
                    "map_clearance_m": now_map_clear,
                }

        v_cap = self._speed_cap(inp, min_obs_now)

        prev_v, prev_w = inp.prev_cmd
        raw_vs = (0.0, 0.08, 0.16, 0.26, 0.34, self.max_speed_mps)
        v_candidates = [0.0] + [v for v in raw_vs[1:] if v <= v_cap + 0.02 and abs(v - prev_v) <= 0.18]
        if len(v_candidates) == 1 and v_cap >= 0.08:
            v_candidates.append(min(0.12, v_cap))
        w_grid = np.linspace(-self.max_yaw_rate, self.max_yaw_rate, 17)
        w_candidates = [0.0] + [float(w) for w in w_grid if abs(w - prev_w) <= 0.65]

        r_world_to_body = rotation_2d(-inp.yaw)
        best = None
        for v in v_candidates:
            for w in w_candidates:
                if v == 0.0 and abs(w) < 0.05:
                    continue
                rollout = self._rollout(inp, v, w, min_obs_now, now_map_clear, r_world_to_body)
                if rollout is None:
                    continue
                x, y, th, rollout_min_obs, rollout_min_map = rollout

                end_dist = math.hypot(inp.lookahead_xy[0] - x, inp.lookahead_xy[1] - y)
                end_heading = math.atan2(inp.lookahead_xy[1] - y, inp.lookahead_xy[0] - x)
                head_pen = abs(wrap_angle(end_heading - th))
                obs_bonus = min(rollout_min_obs, 1.0)
                map_bonus = min(rollout_min_map, 0.65)
                score = (-3.6 * end_dist - 1.15 * head_pen + 0.95 * v
                         + 0.70 * obs_bonus + 0.40 * map_bonus
                         - 0.16 * abs(w) - 0.10 * abs(w - prev_w))
                if v == 0.0:
                    score -= 0.45
                if best is None or score > best[0]:
                    best = (score, float(v), float(w), rollout_min_obs, rollout_min_map)

        if best is None:
            return 0.0, 0.0, {
                "mode": "NO_SAFE_ROLLOUT",
                "blocked_locally": True,
                "min_obs_m": min_obs_now,
                "map_clearance_m": now_map_clear,
            }
        return best[1], best[2], {
            "mode": "TRACKING",
            "blocked_locally": best[1] == 0.0 and abs(heading_err) < 0.40,
            "min_obs_m": best[3],
            "map_clearance_m": best[4],
        }


def compute_lookahead(route_xy, pose_xy, waypoint_idx, lookahead_m=0.80):
    """Advance along polyline monotonically and return (new_idx, lookahead_xy)."""
    pts = np.asarray(route_xy, dtype=float)
    if len(pts) == 0:
        return 0, np.asarray(pose_xy, dtype=float)
    if len(pts) == 1:
        return 0, pts[0].copy()

    window_end = min(len(pts), waypoint_idx + 8)
    dists = np.linalg.norm(pts[waypoint_idx:window_end] - pose_xy, axis=1)
    nearest_local = int(np.argmin(dists))
    idx = waypoint_idx + nearest_local
    while idx < len(pts) - 1 and float(np.linalg.norm(pts[idx] - pose_xy)) < 0.28:
        idx += 1

    cursor = pts[idx].copy()
    d0 = float(np.linalg.norm(cursor - pose_xy))
    if d0 >= lookahead_m:
        return idx, cursor
    accum = d0
    for k in range(idx, len(pts) - 1):
        seg = pts[k + 1] - pts[k]
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6:
            continue
        if accum + seg_len >= lookahead_m:
            frac = (lookahead_m - accum) / seg_len
            return idx, pts[k] + frac * seg
        accum += seg_len
    return idx, pts[-1].copy()


@dataclass
class NavigationReport:
    status: str  # "RUNNING" | "ARRIVED" | "STOPPED_SAFE"
    reason: str = ""
    replan_count: int = 0
    lidar_detection_frames: int = 0
    rgbd_detection_frames: int = 0
    drop_detection_frames: int = 0
    events: list = field(default_factory=list)


class RouteNavigationController:
    """High-level route-following state machine with obstacle replanning and safe stop."""

    def __init__(self, onboard_map: OnboardObstacleMap, initial_route_xy, goal_xy,
                 local_planner=None, settle_time_s=1.5, settle_speed_mps=0.03,
                 settle_yaw_rate=0.08, goal_tolerance_m=0.32, lookahead_m=0.80):
        self.onboard_map = onboard_map
        self.route_xy = np.asarray(initial_route_xy, dtype=float).copy()
        self.route_history = [self.route_xy.tolist()]
        self.goal_xy = np.asarray(goal_xy, dtype=float).copy()
        self.local_planner = local_planner or RuleLocalPlanner(goal_tolerance_m=goal_tolerance_m)
        self.settle_time_s = float(settle_time_s)
        self.settle_speed_mps = float(settle_speed_mps)
        self.settle_yaw_rate = float(settle_yaw_rate)
        self.goal_tolerance_m = float(goal_tolerance_m)
        self.lookahead_m = float(lookahead_m)

        self.waypoint_idx = 0
        self.prev_cmd = (0.0, 0.0)
        self.report = NavigationReport(status="RUNNING")
        self.state = "TRACKING"
        self.pending_stop_reason = ""

        self.stationary_s = 0.0
        self.blocked_s = 0.0
        self.last_replan_t = -99.0
        self.progress_anchor_xy = None
        self.progress_anchor_t = None

    def _record_event(self, now, kind, **details):
        event = dict(time_s=round(float(now), 3), type=kind, **details)
        self.report.events.append(event)

    def _attempt_replan(self, now, pose_xy, local_obstacles_enu, trigger_reason):
        self.last_replan_t = now
        new_route, replan_reason = self.onboard_map.replan(pose_xy, self.goal_xy, local_obstacles_enu)
        if new_route is not None:
            self.route_xy = np.asarray(new_route, dtype=float)
            self.route_history.append(self.route_xy.tolist())
            self.waypoint_idx = 0
            self.blocked_s = 0.0
            self.report.replan_count += 1
            length_m = float(np.linalg.norm(np.diff(self.route_xy, axis=0), axis=1).sum()) if len(self.route_xy) > 1 else 0.0
            self._record_event(now, "REPLAN_SUCCESS", trigger=trigger_reason,
                               detail=replan_reason, new_length_m=round(length_m, 3),
                               waypoints=len(self.route_xy), rover_xy=pose_xy.tolist())
            return True, replan_reason

        self._record_event(now, "REPLAN_FAILED", trigger=trigger_reason,
                           reason=replan_reason, rover_xy=pose_xy.tolist())
        return False, replan_reason

    def step(self, now, dt, pose_xyz, rpy, speed_mps, yaw_rate_rad_s,
             lidar_body_xy, lidar_enu_xy, rgbd_body_xy, rgbd_enu_xy, drop_count=0, rgbd_frame=None):
        """Run one navigation control tick. Returns (v_cmd, w_cmd, debug_dict)."""
        pose_xy = np.asarray(pose_xyz[:2], dtype=float)
        roll, pitch, yaw = map(float, rpy)
        dist_to_goal = float(np.linalg.norm(self.goal_xy - pose_xy))

        if len(lidar_body_xy):
            self.report.lidar_detection_frames += 1
        if len(rgbd_body_xy):
            self.report.rgbd_detection_frames += 1
        if drop_count > 0:
            self.report.drop_detection_frames += 1

        if len(lidar_body_xy) and len(rgbd_body_xy):
            obs_body = np.vstack((lidar_body_xy, rgbd_body_xy))
            obs_enu = np.vstack((lidar_enu_xy, rgbd_enu_xy))
        elif len(lidar_body_xy):
            obs_body, obs_enu = lidar_body_xy, lidar_enu_xy
        elif len(rgbd_body_xy):
            obs_body, obs_enu = rgbd_body_xy, rgbd_enu_xy
        else:
            obs_body = np.empty((0, 2), dtype=float)
            obs_enu = np.empty((0, 2), dtype=float)

        newly_blocked = self.onboard_map.update(pose_xy, lidar_enu_xy, rgbd_enu_xy)
        if newly_blocked:
            self._record_event(
                now, "OBSTACLE_CELLS_BLOCKED",
                cells=[dict(row=r, col=c, xy_m=self.onboard_map.fine_grid.center((r, c)).tolist(), source=src)
                       for r, c, src in newly_blocked[:12]],
                total_new=len(newly_blocked),
                rover_xy=pose_xy.tolist(),
            )

        is_still = abs(speed_mps) < self.settle_speed_mps and abs(yaw_rate_rad_s) < self.settle_yaw_rate
        self.stationary_s = self.stationary_s + dt if is_still else 0.0

        if self.report.status in ("ARRIVED", "STOPPED_SAFE"):
            self.prev_cmd = (0.0, 0.0)
            return 0.0, 0.0, dict(state=self.state, status=self.report.status,
                                  dist_to_goal_m=dist_to_goal, stationary_s=self.stationary_s)

        if self.state == "STOPPING_SAFE":
            self.prev_cmd = (0.0, 0.0)
            if self.stationary_s >= self.settle_time_s:
                self.state = "STOPPED_SAFE"
                self.report.status = "STOPPED_SAFE"
                self.report.reason = self.pending_stop_reason
                self._record_event(now, "STOPPED_SAFE", reason=self.pending_stop_reason,
                                   rover_xy=pose_xy.tolist(), dist_to_goal_m=round(dist_to_goal, 3))
            return 0.0, 0.0, dict(state=self.state, status=self.report.status,
                                  reason=self.pending_stop_reason,
                                  dist_to_goal_m=dist_to_goal, stationary_s=self.stationary_s)

        if dist_to_goal <= self.goal_tolerance_m or self.state == "SETTLING_AT_GOAL":
            if dist_to_goal <= self.goal_tolerance_m + 0.10:
                self.state = "SETTLING_AT_GOAL"
                self.prev_cmd = (0.0, 0.0)
                if self.stationary_s >= self.settle_time_s:
                    self.state = "ARRIVED"
                    self.report.status = "ARRIVED"
                    self._record_event(now, "ARRIVED_SETTLED", rover_xy=pose_xy.tolist(),
                                       dist_to_goal_m=round(dist_to_goal, 3),
                                       stationary_s=round(self.stationary_s, 2))
                return 0.0, 0.0, dict(state=self.state, status=self.report.status,
                                      dist_to_goal_m=dist_to_goal, stationary_s=self.stationary_s)
            else:
                self.state = "TRACKING"

        needs_replan, invalid_reason = self.onboard_map.route_needs_replan(
            self.route_xy, self.waypoint_idx, pose_xy, obs_enu)
        if needs_replan and (now - self.last_replan_t >= 0.8 or newly_blocked):
            ok, detail = self._attempt_replan(now, pose_xy, obs_enu, invalid_reason)
            if not ok:
                self.state = "STOPPING_SAFE"
                self.pending_stop_reason = detail
                self.prev_cmd = (0.0, 0.0)
                return 0.0, 0.0, dict(state=self.state, status=self.report.status,
                                      reason=detail, dist_to_goal_m=dist_to_goal)

        self.waypoint_idx, lookahead_xy = compute_lookahead(
            self.route_xy, pose_xy, self.waypoint_idx, self.lookahead_m)

        inp = LocalPlannerInput(
            pose_xy=pose_xy,
            yaw=yaw,
            attitude_rp=(roll, pitch),
            speed_mps=speed_mps,
            prev_cmd=self.prev_cmd,
            lookahead_xy=lookahead_xy,
            goal_xy=self.goal_xy,
            dist_to_goal=dist_to_goal,
            obstacles_body_xy=obs_body,
            lidar_obstacles_body_xy=lidar_body_xy,
            rgbd_obstacles_body_xy=rgbd_body_xy,
            onboard_map=self.onboard_map,
            rgbd_frame=rgbd_frame,
        )
        v_cmd, w_cmd, info = self.local_planner.compute(inp)

        if info.get("mode") == "UNSAFE_ATTITUDE":
            self.state = "STOPPING_SAFE"
            self.pending_stop_reason = (f"UNSAFE_ATTITUDE: roll={math.degrees(roll):.1f}deg, "
                                        f"pitch={math.degrees(pitch):.1f}deg")
            self._record_event(now, "UNSAFE_ATTITUDE_STOP", reason=self.pending_stop_reason,
                               rover_xy=pose_xy.tolist())
            self.prev_cmd = (0.0, 0.0)
            return 0.0, 0.0, dict(state=self.state, status=self.report.status,
                                  reason=self.pending_stop_reason, dist_to_goal_m=dist_to_goal)

        if info.get("blocked_locally"):
            self.blocked_s += dt
            if self.blocked_s >= 1.2 and now - self.last_replan_t >= 1.2:
                ok, detail = self._attempt_replan(now, pose_xy, obs_enu, "LOCAL_PLANNER_BLOCKED")
                if not ok:
                    self.state = "STOPPING_SAFE"
                    self.pending_stop_reason = detail
                    self.prev_cmd = (0.0, 0.0)
                    return 0.0, 0.0, dict(state=self.state, status=self.report.status,
                                          reason=detail, dist_to_goal_m=dist_to_goal)
            if self.blocked_s >= 3.0:
                min_obs = info.get("min_obs_m", float("inf"))
                map_clear = info.get("map_clearance_m", float("inf"))
                if min_obs < 1.5:
                    reason = (f"OBSTACLE_BLOCKED_LOCALLY: nearby obstacle at {min_obs:.2f} m "
                              f"leaves no safe rollout on confirmed terrain")
                elif map_clear < 0.35:
                    reason = (f"UNCONFIRMED_TERRAIN_AHEAD: confirmed corridor clearance "
                              f"{map_clear:.2f} m insufficient to continue safely")
                else:
                    reason = "NO_SAFE_LOCAL_ROLLOUT: unable to find safe control command"
                self.state = "STOPPING_SAFE"
                self.pending_stop_reason = reason
                self._record_event(now, "LOCAL_STOP", reason=reason, rover_xy=pose_xy.tolist())
                self.prev_cmd = (0.0, 0.0)
                return 0.0, 0.0, dict(state=self.state, status=self.report.status,
                                      reason=reason, dist_to_goal_m=dist_to_goal)
        else:
            self.blocked_s = max(0.0, self.blocked_s - 0.5 * dt)

        if self.progress_anchor_xy is None:
            self.progress_anchor_xy = pose_xy.copy()
            self.progress_anchor_t = now
        elif float(np.linalg.norm(pose_xy - self.progress_anchor_xy)) >= 0.25:
            self.progress_anchor_xy = pose_xy.copy()
            self.progress_anchor_t = now
        elif now - self.progress_anchor_t >= 16.0:
            reason = (f"STUCK_NO_PROGRESS: net movement < 0.25 m over 16.0 s "
                      f"(dist_to_goal={dist_to_goal:.2f} m)")
            self.state = "STOPPING_SAFE"
            self.pending_stop_reason = reason
            self._record_event(now, "STUCK_STOP", reason=reason, rover_xy=pose_xy.tolist())
            self.prev_cmd = (0.0, 0.0)
            return 0.0, 0.0, dict(state=self.state, status=self.report.status,
                                  reason=reason, dist_to_goal_m=dist_to_goal)

        self.prev_cmd = (v_cmd, w_cmd)
        min_lidar = float(np.min(np.linalg.norm(lidar_body_xy, axis=1))) if len(lidar_body_xy) else None
        min_rgbd = float(np.min(np.linalg.norm(rgbd_body_xy, axis=1))) if len(rgbd_body_xy) else None
        return v_cmd, w_cmd, dict(
            state=self.state,
            mode=info.get("mode", self.state),
            status=self.report.status,
            waypoint_idx=self.waypoint_idx,
            lookahead_xy=lookahead_xy.tolist(),
            dist_to_goal_m=dist_to_goal,
            stationary_s=self.stationary_s,
            min_lidar_obs_m=min_lidar,
            min_rgbd_obs_m=min_rgbd,
            lidar_obs_count=int(len(lidar_body_xy)),
            rgbd_obs_count=int(len(rgbd_body_xy)),
            drop_count=int(drop_count),
            replan_count=self.report.replan_count,
            **{key: info[key] for key in ("action_source", "network_proposal_vw") if key in info},
        )
