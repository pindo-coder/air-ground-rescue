"""Make a north-up preview and verify cross-heading projection on the saved frames."""
from pathlib import Path
import math
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
for controller in ("uav_mapper", "search_agent"):
    sys.path.insert(0, str(ROOT / "controllers" / controller))

from multiview import aerial_pixels_enu
from survey_map import BLOCKED, FREE, UNKNOWN, load_prior, sampled_pixels


def project(observations, frame, pixels):
    width, height = observations["size_wh"]
    orientations = observations.get("body_rotation_enu")
    return aerial_pixels_enu(pixels, width, height, float(observations["fov_rad"]),
                            observations["gps_enu_m"][frame],
                            observations["attitude_rpy_rad"][frame],
                            body_rotation=None if orientations is None else orientations[frame])


def aerial_mosaic(grid, measurements):
    scale = 5
    shape = tuple(size * scale for size in grid.shape)
    count = np.zeros(math.prod(shape))
    color_sums = np.zeros((math.prod(shape), 3))
    stride = int(measurements["stride_px"])
    depths, colors = measurements["depth_m"], measurements["rgb"]
    min_range, max_range = measurements["range_m"]
    phase = measurements.get("survey_phase", np.zeros(len(depths), dtype=int))
    for frame, (depth, rgb) in enumerate(zip(depths, colors)):
        if phase[frame] != 0:
            continue
        pixels, valid = sampled_pixels(depth, stride, min_range, max_range)
        points = project(measurements, frame, pixels)
        indices = np.floor((points[:, :2] - grid.origin) * scale / grid.resolution).astype(int)
        cols, rows = indices.T
        inside = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
        cells = rows[inside] * shape[1] + cols[inside]
        np.add.at(count, cells, 1)
        np.add.at(color_sums, cells, rgb[valid][inside])
    mosaic = color_sums / np.maximum(count[:, None], 1)
    mosaic[count == 0] = [30, 38, 43]
    return mosaic.reshape(*shape, 3).astype(np.uint8)


def draw_preview(folder, route, observations):
    grid, states = load_prior(folder / "uav_map.npz")
    palette = {UNKNOWN: (84, 94, 107), FREE: (139, 178, 144), BLOCKED: (223, 113, 79)}
    cells = np.zeros((*states.shape, 3), dtype=np.uint8)
    for state, color in palette.items():
        cells[states == state] = color
    w = 520
    h = round(w * states.shape[0] / states.shape[1])
    gap, top = 24, 91
    panel = Image.new("RGB", (2 * w + 3 * gap, h + top + 86), "#111923")
    draw = ImageDraw.Draw(panel)
    title = ImageFont.load_default(size=27)
    body = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=15)
    draw.text((gap, 16), "D  /  Aerial survey and rover route", font=title, fill="#edf4f8")
    draw.text((gap, 61), "Aerial RGB samples projected into ENU", font=body, fill="#b8c9d7")
    x_map = w + 2 * gap
    draw.text((x_map, 61), f"Coarse terrain prior  |  {grid.resolution:.2f} m cells",
              font=body, fill="#b8c9d7")
    mosaic = aerial_mosaic(grid, observations)
    mosaic = Image.fromarray(np.flipud(mosaic)).resize((w, h), Image.Resampling.BILINEAR)
    panel.paste(mosaic, (gap, top))
    panel.paste(Image.fromarray(np.flipud(cells)).resize((w, h), Image.Resampling.NEAREST),
                (x_map, top))

    def coord(xy):
        east, north = (np.asarray(xy) - grid.origin) / grid.resolution
        return (x_map + w * east / states.shape[1], top + h * (1 - north / states.shape[0]))

    start, goal = coord(route["start_xy_m"]), coord(route["goal_xy_m"])
    for step in range(0, 20, 2):
        first = tuple(a + (b - a) * step / 20 for a, b in zip(start, goal))
        last = tuple(a + (b - a) * (step + 1) / 20 for a, b in zip(start, goal))
        draw.line([first, last], fill="#e6e9e0", width=2)
    if route["status"] == "PLANNED":
        path = [coord(point) for point in route["route_xy_m"]]
        if len(path) > 1:
            draw.line(path, fill="#0d2634", width=9, joint="curve")
            draw.line(path, fill="#50ddfa", width=5, joint="curve")
    for (x, y), label in ((start, "S"), (goal, "G")):
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill="#142230", outline="#f1f7f9", width=2)
        draw.text((x, y), label, font=small, fill="#f1f7f9", anchor="mm")
    scale_px = (w * 5) / (states.shape[1] * grid.resolution)
    draw.line([(gap + 14, top + h - 20), (gap + 14 + scale_px, top + h - 20)], fill="white", width=3)
    draw.text((gap + 14, top + h - 42), "5 m", font=small, fill="white", stroke_width=1,
              stroke_fill="#141c24")
    draw.text((x_map + 12, top + 10), "N  ^", font=body, fill="white", stroke_width=1,
              stroke_fill="#141c24")
    y_legend = top + h + 15
    for offset, (state, label) in enumerate(((FREE, "Candidate free"), (BLOCKED, "Hazard"),
                                             (UNKNOWN, "Unknown"))):
        x = gap + 167 * offset
        draw.rectangle((x, y_legend, x + 15, y_legend + 15), fill=palette[state])
        draw.text((x + 22, y_legend - 2), label, font=small, fill="#d1dee8")
    description = (f"A*  {route['route_length_m']:.1f} m   |   dashed: direct line"
                   if route["status"] == "PLANNED" else route["reason"])
    draw.text((x_map, y_legend - 2), description, font=body, fill="#50ddfa")
    draw.text((gap, y_legend + 32), f"{len(observations['timestamp_s'])} synchronized observations",
              font=small, fill="#b8c9d7")
    draw.text((x_map, y_legend + 32), "S  Rover GPS position    G  Test waypoint",
              font=small, fill="#b8c9d7")
    panel.save(folder / "preview.png")
    return {key: int(np.count_nonzero(states == state)) for key, state in
            (("candidate_free", FREE), ("blocked", BLOCKED), ("unknown", UNKNOWN))}


def audit_alignment(folder, observations):
    """Match actual image features before comparing their independently projected locations."""
    yaw = observations["attitude_rpy_rad"][:, 2]
    phase = observations.get("survey_phase", np.zeros(len(yaw), dtype=int))
    gps = observations["gps_enu_m"][:, :2]
    candidates = sorted((math.dist(gps[i], gps[j]), i, j)
                        for i in range(len(yaw)) for j in range(i + 1, len(yaw))
                        if phase[i] == phase[j] == 0 and math.cos(yaw[j] - yaw[i]) < -0.8
                        and math.dist(gps[i], gps[j]) < 9.0)
    if not candidates:
        return {"status": "NO_OPPOSING_OVERLAP"}
    sift = cv2.SIFT_create(nfeatures=650)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    features = {}
    depths = observations["depth_m"]
    stride = int(observations["stride_px"])
    min_range, max_range = observations["range_m"]
    for _, first, second in candidates[:12]:
        for index in (first, second):
            if index not in features:
                photo = cv2.imread(str(folder / "frames" / f"{index:03d}.jpg"), cv2.IMREAD_GRAYSCALE)
                features[index] = (photo, *sift.detectAndCompute(photo, None))
        image_a, points_a, desc_a = features[first]
        image_b, points_b, desc_b = features[second]
        if desc_a is None or desc_b is None or len(desc_b) < 2:
            continue
        matches = [a for pair in matcher.knnMatch(desc_a, desc_b, k=2) if len(pair) == 2
                   for a, b in (pair,) if a.distance < 0.72 * b.distance]
        if len(matches) < 10:
            continue
        xy_a = np.array([points_a[m.queryIdx].pt for m in matches])
        xy_b = np.array([points_b[m.trainIdx].pt for m in matches])
        _, inliers = cv2.findHomography(xy_a, xy_b, cv2.RANSAC, 3.0)
        if inliers is None:
            continue
        pixels_a, pixels_b, keep = [], [], []
        for k in np.flatnonzero(inliers):
            entries = []
            for index, (u, v) in ((first, xy_a[k]), (second, xy_b[k])):
                r, c = round(v / stride), round(u / stride)
                depth = depths[index]
                if not 1 <= r < depth.shape[0] - 1 or not 1 <= c < depth.shape[1] - 1:
                    break
                neighborhood = depth[r - 1:r + 2, c - 1:c + 2]
                if (not np.isfinite(neighborhood).all() or np.ptp(neighborhood) > 0.24
                        or not min_range < depth[r, c] < max_range):
                    break
                entries.append((u, v, float(depth[r, c])))
            if len(entries) == 2:
                pixels_a.append(entries[0])
                pixels_b.append(entries[1])
                keep.append(matches[k])
        if len(keep) < 8:
            continue
        errors = np.linalg.norm(project(observations, first, pixels_a)[:, :2]
                                - project(observations, second, pixels_b)[:, :2], axis=1)
        visual = cv2.drawMatches(image_a, points_a, image_b, points_b, keep[:30], None,
                                 matchColor=(104, 224, 240), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(str(folder / "alignment.jpg"), visual)
        return dict(status="CHECKED", frames=[first, second], matched_features=len(keep),
                    median_error_m=float(np.median(errors)), p90_error_m=float(np.quantile(errors, 0.9)),
                    within_one_cell_fraction=float(np.mean(errors <= 0.5)))
    return {"status": "NO_RELIABLE_IMAGE_MATCHES"}
