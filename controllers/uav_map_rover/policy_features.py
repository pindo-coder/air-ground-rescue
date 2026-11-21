"""Deterministic onboard features shared by collection, training and inference."""
import math

import numpy as np

from route_tracker import rotation_2d

RGBD_SHAPE = (5, 24, 32)
SECTORS = 36
RANGE_SHAPE = (4, SECTORS)
NAV_SIZE = 19
SCHEMA_VERSION = 1
MAX_SPEED = 0.42
MAX_ANGULAR_SPEED = 0.85
ACTION_SCALE = np.array([MAX_SPEED, MAX_ANGULAR_SPEED], dtype=np.float32)
LOOK_ANGLES = np.linspace(-math.pi / 3.0, math.pi / 3.0, 5)


def encode_rgbd(image, depth, encoding="bgra"):
    """Sample aligned R, G, B, optical-axis depth and validity.

    Supports Webots 4-channel BGRA buffers (default) as well as ROS 2 / Gazebo
    3-channel RGB/BGR and 4-channel RGBA images.
    """
    height, width = depth.shape
    enc = encoding.lower()
    channels = 3 if enc in ("rgb", "rgb8", "bgr", "bgr8") else 4
    if isinstance(image, np.ndarray):
        pixels = image.reshape(height, width, -1)
        channels = pixels.shape[2]
    else:
        raw = np.frombuffer(image, dtype=np.uint8)
        channels = raw.size // (height * width)
        pixels = raw.reshape(height, width, channels)
    rows = np.linspace(0, height - 1, RGBD_SHAPE[1]).round().astype(int)
    cols = np.linspace(0, width - 1, RGBD_SHAPE[2]).round().astype(int)
    sampled = pixels[rows[:, None], cols[None, :]]
    if enc in ("rgb", "rgb8", "rgba", "rgba8"):
        rgb = sampled[..., :3]
    else:
        rgb = sampled[..., 2::-1]
    d = np.asarray(depth[rows[:, None], cols[None, :]], dtype=np.float32)
    valid = np.isfinite(d) & (d >= 0.20)
    frame = np.empty(RGBD_SHAPE, dtype=np.float32)
    frame[:3] = rgb.transpose(2, 0, 1) / 255.0
    frame[3] = np.where(valid, np.clip(d / 4.0, 0.0, 1.0), 1.0)
    frame[4] = valid
    return frame


def obstacle_sectors(points):
    """Nearest filtered obstacle and a hit mask per body-frame angular sector."""
    result = np.zeros((2, SECTORS), dtype=np.float32)
    result[0] = 1.0
    xy = np.asarray(points, dtype=float).reshape(-1, 2)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy):
        angles = np.arctan2(xy[:, 1], xy[:, 0])
        bins = np.floor((angles + math.pi) * SECTORS / (2 * math.pi)).astype(int) % SECTORS
        distance = np.linalg.norm(xy, axis=1)
        np.minimum.at(result[0], bins, np.minimum(distance / 5.0, 1.0))
        result[1, bins] = 1.0
    return result


def planner_features(inp):
    if inp.rgbd_frame is None or inp.rgbd_frame.shape != RGBD_SHAPE:
        raise ValueError("Local policy needs a synchronized encoded RGB-D frame.")
    rays = np.vstack((obstacle_sectors(inp.lidar_obstacles_body_xy),
                      obstacle_sectors(inp.rgbd_obstacles_body_xy)))
    to_body = rotation_2d(-inp.yaw)
    target = (inp.lookahead_xy - inp.pose_xy) @ to_body.T
    state = [
        *np.clip(target / 2.0, -1.0, 1.0),
        min(inp.dist_to_goal / 6.0, 1.0),
        np.clip(inp.speed_mps / MAX_SPEED, -1.0, 1.0),
        *np.clip(np.asarray(inp.prev_cmd) / ACTION_SCALE, -1.0, 1.0),
        *np.clip(np.asarray(inp.attitude_rp) / 0.42, -1.0, 1.0),
    ]
    state.append(inp.onboard_map.continuous_clearance(inp.pose_xy))
    for distance in (0.35, 0.80):
        for angle in inp.yaw + LOOK_ANGLES:
            xy = inp.pose_xy + distance * np.array([math.cos(angle), math.sin(angle)])
            state.append(inp.onboard_map.continuous_clearance(xy))
    state = np.asarray(state, dtype=np.float32)
    state[8:] = np.clip(state[8:], 0.0, 1.0)
    if not np.isfinite(state).all():
        raise ValueError("Local planner pose or map features are invalid.")
    return inp.rgbd_frame, rays, state


def expert_mode(action):
    """Class labels correspond to stopping, turning in place and driving."""
    if action[0] > 1e-4:
        return 2
    return int(abs(action[1]) > 1e-4)
