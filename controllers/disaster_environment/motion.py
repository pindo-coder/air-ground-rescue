"""Deterministic environment trajectories, independent of the search policy."""
import math
import numpy as np


def path_pose(path, speed, seconds, phase=0.0):
    """Constant-speed, continuous ping-pong motion along a polyline."""
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("A trajectory needs at least two 2-D points")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(lengths.sum())
    if total <= 0 or speed <= 0:
        raise ValueError("Trajectory and speed must be positive")
    distance = ((max(0.0, seconds) + phase) * speed) % (2 * total)
    backwards = distance > total
    if backwards:
        distance = 2 * total - distance
    for index, length in enumerate(lengths):
        if distance <= length or index == len(lengths) - 1:
            vector = points[index + 1] - points[index]
            xy = points[index] + vector * min(1.0, distance / max(length, 1e-9))
            heading = math.atan2(vector[1], vector[0]) + (math.pi if backwards else 0)
            return xy, heading
        distance -= length
    raise AssertionError("Unreachable trajectory segment")
