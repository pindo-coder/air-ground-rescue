"""Pure-Python sensor & coordinate adapters between ROS 2 Jazzy / Gazebo Harmonic and core algorithms.

This module intentionally has zero runtime dependency on `rclpy` so every conversion
can be exhaustively unit-tested on any host (macOS, Linux, Windows, WSL2).
"""

from dataclasses import dataclass
import math
import struct

import numpy as np


# WGS84 ellipsoid constants matching gz::math::SphericalCoordinates (EARTH_WGS84).
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@dataclass
class NavSatToEnu:
    """Convert NavSatFix (lat_deg, lon_deg, alt_m) to local ENU meters matching Gazebo Harmonic.

    Optional AR(1) correlated Gaussian noise matches Webots GPS (accuracy=0.04 m,
    noiseCorrelation=0.2) without hitting gz-sensors horizontal degree-noise ambiguity.
    """

    lat0_deg: float = 0.0
    lon0_deg: float = 0.0
    alt0_m: float = 0.0
    noise_std_m: float = 0.04
    noise_correlation: float = 0.2
    seed: int = 17

    def __post_init__(self):
        lat0 = math.radians(self.lat0_deg)
        sin_lat = math.sin(lat0)
        denom = math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        self._rn = WGS84_A / denom
        self._rm = WGS84_A * (1.0 - WGS84_E2) / (denom ** 3)
        self._cos_lat0 = math.cos(lat0)
        self._rng = np.random.default_rng(self.seed)
        self._noise_state = np.zeros(3, dtype=float)

    def convert(self, lat_deg, lon_deg, alt_m, apply_noise=True):
        dlat = math.radians(float(lat_deg) - self.lat0_deg)
        dlon = math.radians(float(lon_deg) - self.lon0_deg)
        east = dlon * self._rn * self._cos_lat0
        north = dlat * self._rm
        up = float(alt_m) - self.alt0_m
        enu = np.array([east, north, up], dtype=float)
        if apply_noise and self.noise_std_m > 0.0:
            rho = min(max(float(self.noise_correlation), 0.0), 0.99)
            innov_scale = self.noise_std_m * math.sqrt(max(1.0 - rho * rho, 0.0))
            self._noise_state = rho * self._noise_state + self._rng.normal(0.0, innov_scale, size=3)
            enu = enu + self._noise_state
        return enu

    def enu_to_navsat(self, enu_xyz):
        """Inverse helper for testing and spherical coordinate verification."""
        east, north, up = map(float, enu_xyz)
        lat_deg = self.lat0_deg + math.degrees(north / self._rm)
        lon_deg = self.lon0_deg + math.degrees(east / max(self._rn * self._cos_lat0, 1e-9))
        alt_m = self.alt0_m + up
        return lat_deg, lon_deg, alt_m


def quat_xyzw_to_matrix(qx, qy, qz, qw):
    """Convert ROS 2 / Gazebo ENU quaternion (x, y, z, w) to 3x3 rotation matrix."""
    x, y, z, w = float(qx), float(qy), float(qz), float(qw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3, dtype=float)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def quat_xyzw_to_rpy(qx, qy, qz, qw):
    """Convert ENU quaternion (x, y, z, w) to intrinsic ZYX (roll, pitch, yaw) radians."""
    m = quat_xyzw_to_matrix(qx, qy, qz, qw)
    sin_p = max(-1.0, min(1.0, -float(m[2, 0])))
    pitch = math.asin(sin_p)
    if abs(sin_p) < 0.999999:
        roll = math.atan2(float(m[2, 1]), float(m[2, 2]))
        yaw = math.atan2(float(m[1, 0]), float(m[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(m[0, 1]), float(m[1, 1]))
    return roll, pitch, yaw


def ros_image_to_bgr(data, width, height, encoding="rgb8", step=None):
    """Decode ROS 2 sensor_msgs/Image bytes to contiguous (H, W, 3) uint8 BGR array."""
    enc = encoding.lower()
    channels = 4 if enc in ("rgba8", "bgra8") else 3
    row_bytes = int(step) if step else width * channels
    raw = np.frombuffer(data, dtype=np.uint8)
    rows = raw.reshape(height, row_bytes)[:, :width * channels].reshape(height, width, channels)
    if enc in ("rgb8", "rgba8"):
        return np.ascontiguousarray(rows[:, :, 2::-1])
    if enc in ("bgr8", "bgra8"):
        return np.ascontiguousarray(rows[:, :, :3])
    raise ValueError(f"Unsupported ROS image encoding: {encoding}")


def ros_depth_to_meters(data, width, height, encoding="32FC1", step=None):
    """Decode ROS 2 depth Image (32FC1 meters or 16UC1 mm) to (H, W) float32 meters with NaN invalids."""
    enc = encoding.upper()
    raw = np.frombuffer(data, dtype=np.uint8)
    if enc in ("32FC1", "R_FLOAT32"):
        row_bytes = int(step) if step else width * 4
        arr = raw.reshape(height, row_bytes)[:, :width * 4].copy().view(np.float32).reshape(height, width)
    elif enc in ("16UC1", "MONO16"):
        row_bytes = int(step) if step else width * 2
        arr = raw.reshape(height, row_bytes)[:, :width * 2].copy().view(np.uint16).reshape(height, width).astype(np.float32) * 0.001
    else:
        raise ValueError(f"Unsupported ROS depth encoding: {encoding}")
    invalid = (~np.isfinite(arr)) | (arr <= 0.0)
    out = arr.astype(np.float32, copy=True)
    out[invalid] = np.nan
    return out


# ROS 2 PointField datatype codes -> (struct format char, byte size)
_POINTFIELD_TYPES = {
    1: ("b", 1),
    2: ("B", 1),
    3: ("h", 2),
    4: ("H", 2),
    5: ("i", 4),
    6: ("I", 4),
    7: ("f", 4),
    8: ("d", 8),
}


def ros_pointcloud2_to_xyz(data, width, height, point_step, row_step, fields, is_bigendian=False):
    """Decode ROS 2 sensor_msgs/PointCloud2 into an (N, 3) float32 array in sensor FLU frame.

    `fields` is an iterable of objects or dicts with `name`, `offset`, `datatype`.
    """
    field_map = {}
    for f in fields:
        if isinstance(f, dict):
            field_map[f["name"]] = (int(f["offset"]), int(f["datatype"]))
        else:
            field_map[f.name] = (int(f.offset), int(f.datatype))
    for axis in ("x", "y", "z"):
        if axis not in field_map:
            raise ValueError(f"PointCloud2 missing required '{axis}' field")

    n_points = int(width) * int(height)
    if n_points == 0 or not data:
        return np.empty((0, 3), dtype=np.float32)

    raw = np.frombuffer(data, dtype=np.uint8)
    expected_row = max(int(row_step), int(width) * int(point_step))
    buf = raw[:int(height) * expected_row].reshape(int(height), expected_row)
    pts_bytes = np.ascontiguousarray(buf[:, :int(width) * int(point_step)]).reshape(n_points, int(point_step))

    endian = ">" if is_bigendian else "<"
    out = np.empty((n_points, 3), dtype=np.float32)
    for idx, axis in enumerate(("x", "y", "z")):
        offset, dtype_code = field_map[axis]
        fmt_char, byte_len = _POINTFIELD_TYPES[dtype_code]
        np_dtype = np.dtype(endian + fmt_char)
        col = np.ascontiguousarray(pts_bytes[:, offset:offset + byte_len]).view(np_dtype).reshape(-1)
        out[:, idx] = col.astype(np.float32)
    return out


def ros_laserscan_to_layers(ranges, angle_min, angle_max, n_layers=1, elevations=(0.0,)):
    """Reshape flat LaserScan ranges into per-layer arrays and return azimuth direction."""
    arr = np.asarray(ranges, dtype=float).ravel()
    if n_layers <= 1:
        layers = [arr]
    else:
        per_layer = arr.size // n_layers
        layers = [arr[i * per_layer:(i + 1) * per_layer] for i in range(n_layers)]
    azimuth_order = "ccw" if float(angle_max) >= float(angle_min) else "cw"
    return layers, tuple(elevations), azimuth_order


def extract_wheel_positions_rad(joint_names, joint_positions):
    """Return (left_rad, right_rad) wheel encoder angles for 4WD or 2WD Gazebo rover."""
    by_name = {name: float(pos) for name, pos in zip(joint_names, joint_positions)}
    left_vals = [by_name[k] for k in ("front_left_wheel_joint", "rear_left_wheel_joint", "left_wheel_joint") if k in by_name]
    right_vals = [by_name[k] for k in ("front_right_wheel_joint", "rear_right_wheel_joint", "right_wheel_joint") if k in by_name]
    if not left_vals or not right_vals:
        return None
    return np.array([sum(left_vals) / len(left_vals), sum(right_vals) / len(right_vals)], dtype=float)


def static_tf_specs():
    """Static TF transforms (parent, child, xyz_m, rpy_rad) for ROS 2 Jazzy visualization & tooling."""
    return (
        ("map", "odom", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("base_link", "rover/chassis", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("base_link", "rover/lidar", (0.0, 0.0, 0.37), (0.0, 0.0, 0.0)),
        ("base_link", "rover/rgb", (0.26, 0.0, 0.30), (0.0, 0.08, 0.0)),
        ("base_link", "rover/depth", (0.26, 0.0, 0.30), (0.0, 0.08, 0.0)),
        ("base_link", "rover/chase_cam", (-1.35, 0.0, 0.92), (0.0, 0.42, 0.0)),
        ("base_link", "rover/imu", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("base_link", "rover/gps", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("drone/body", "drone/rgb", (0.0, 0.0, -0.20), (0.0, math.pi / 2.0, 0.0)),
        ("drone/body", "drone/depth", (0.0, 0.0, -0.20), (0.0, math.pi / 2.0, 0.0)),
        ("drone/body", "drone/fpv_cam", (-1.10, 0.0, 0.48), (0.0, 0.46, 0.0)),
        ("drone/body", "drone/gps", (0.0, 0.0, -0.20), (0.0, 0.0, 0.0)),
        ("drone/body", "drone/imu", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )


def rpy_to_quat_xyzw(roll, pitch, yaw):
    """Convert intrinsic ZYX (roll, pitch, yaw) radians to quaternion (x, y, z, w)."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )
