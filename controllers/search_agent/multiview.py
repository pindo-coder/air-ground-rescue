"""Sensor-only RGB-D projection and time/frame checks for ground-air fusion."""
import json
import math
import re
import numpy as np
from mapping import rotation


def attitude_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array(
        [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array(
        [[1, 0, 0], [0, cr, -sr], [0, sr, cr]])


def aerial_camera_rotation(attitude, body_rotation=None):
    """Camera-to-ENU rotation for the drone's downward FLU camera."""
    body = attitude_matrix(*attitude) if body_rotation is None else body_rotation
    return body @ attitude_matrix(0, math.pi / 2, 0)


def aerial_pixels_enu(pixels, width, height, fov, gps, attitude, *, body_rotation=None):
    """Project (..., 3) image coordinates and optical-axis depths to ENU."""
    u, v, depth = np.moveaxis(np.asarray(pixels, dtype=float), -1, 0)
    focal = width / (2 * math.tan(fov / 2))
    rays = np.stack((depth, -(u - (width - 1) / 2) * depth / focal,
                     -(v - (height - 1) / 2) * depth / focal), axis=-1)
    # GPS is mounted at the camera optical centre; its readings include noise.
    orientation = aerial_camera_rotation(attitude, body_rotation)
    return rays @ orientation.T + np.asarray(gps)


def aerial_pixel_enu(pixel, width, height, fov, gps, attitude):
    return aerial_pixels_enu(pixel, width, height, fov, gps, attitude)


def aerial_candidates(detector, bgr, depth, gps, attitude, fov=1.1):
    height, width = depth.shape
    result = []
    for bbox, score, supplied_mask in detector.detect(bgr, depth):
        x1, y1, x2, y2 = map(int, bbox)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        mask = np.zeros(depth.shape, bool)
        if supplied_mask is not None:
            mask[:] = supplied_mask
        else:
            dx, dy = max(1, (x2-x1)//5), max(1, (y2-y1)//5)
            mask[y1+dy:y2-dy, x1+dx:x2-dx] = True
        use = mask & np.isfinite(depth) & (depth > .3) & (depth < 24)
        if np.count_nonzero(use) < 5:
            continue
        rows, cols = np.nonzero(use)
        distance = float(np.median(depth[use]))
        if np.median(np.abs(depth[use] - distance)) > .45:
            continue
        pixel = np.array([np.median(cols), np.median(rows), distance])
        project = lambda p: aerial_pixel_enu(p, width, height, fov, gps, attitude)
        point = project(pixel)
        jac = np.empty((2, 3))
        for axis, step in enumerate((1.0, 1.0, .01)):
            changed = pixel.copy(); changed[axis] += step
            jac[:, axis] = (project(changed)[:2] - point[:2]) / step
        covariance = jac @ np.diag([4., 4., (.035+.005*distance**2)**2]) @ jac.T
        covariance += np.eye(2) * (.04**2 + (.006*distance)**2 + .18**2)
        result.append(dict(mean_enu=point[:2].tolist(), covariance_enu=covariance.tolist(),
                           score=float(score), bbox=[x1,y1,x2,y2], depth=distance))
    return result


class AerialInbox:
    """Converts measured ENU coordinates to the initial noisy UGV deployment frame."""
    def __init__(self, gps_origin, heading, max_age=2.5):
        self.origin = np.asarray(gps_origin, dtype=float)
        self.heading = float(heading)
        self.max_age = max_age
        self.last_sequence = -1
        self.accepted = 0
        self.rejected = 0

    def ingest(self, text, now):
        try:
            if not isinstance(text, str) or len(text) > 32768:
                raise ValueError("packet size")
            packet = json.loads(text)
            stamp, sequence = float(packet['timestamp_s']), packet['sequence']
            if packet.get('version') != 1 or packet.get('source') != 'uav_rgbd':
                raise ValueError("packet source")
            if type(sequence) is not int or sequence <= self.last_sequence:
                raise ValueError("replayed packet")
            if not math.isfinite(stamp) or not -.2 <= now-stamp <= self.max_age:
                raise ValueError("packet timestamp")
            observer = np.asarray(packet['camera_origin_enu'], dtype=float)
            if observer.shape != (3,) or not np.isfinite(observer).all():
                raise ValueError("camera origin")
            evidence = packet['evidence']
            if not re.fullmatch(r'aerial/[A-Za-z0-9_.-]+\.jpg', evidence):
                raise ValueError("evidence path")
            candidates = packet['candidates']
            if not isinstance(candidates, list) or len(candidates) > 20:
                raise ValueError("candidate count")
            transform = rotation(-self.heading)
            observer_xy = transform @ (observer[:2]-self.origin[:2])
            result = []
            for candidate in candidates:
                mean = np.asarray(candidate['mean_enu'], dtype=float)
                covariance = np.asarray(candidate['covariance_enu'], dtype=float)
                score = float(candidate['score']); depth = float(candidate['depth'])
                bbox = np.asarray(candidate['bbox'], dtype=float)
                if (mean.shape != (2,) or covariance.shape != (2, 2) or
                        not np.isfinite(mean).all() or not np.isfinite(covariance).all() or
                        not np.allclose(covariance, covariance.T) or
                        np.linalg.eigvalsh(covariance).min() <= 0 or
                        not math.isfinite(score) or not .35 <= score <= 1 or not .3 < depth < 24):
                    raise ValueError("invalid candidate")
                if (bbox.shape != (4,) or not np.isfinite(bbox).all() or
                        not 0 <= bbox[0] < bbox[2] <= 640 or
                        not 0 <= bbox[1] < bbox[3] <= 480):
                    raise ValueError("invalid image box")
                local_mean = transform @ (mean-self.origin[:2])
                if np.linalg.norm(mean-observer[:2]) > 25:
                    raise ValueError("candidate outside sensor range")
                local_cov = transform @ covariance @ transform.T + np.eye(2)*.04**2
                yaw_jac = np.array([-local_mean[1], local_mean[0]])
                local_cov += np.outer(yaw_jac, yaw_jac)*.006**2
                result.append(dict(mean=local_mean, cov=local_cov, score=score,
                    bbox=bbox.tolist(), depth=depth, source='uav', evidence=evidence,
                    observer_xy=observer_xy.copy(), observer_z=float(observer[2]-self.origin[2]),
                    observation_time=stamp))
            self.last_sequence = sequence
            self.accepted += 1
            return result
        except (ValueError, TypeError, KeyError, OverflowError):
            self.rejected += 1
            return []
