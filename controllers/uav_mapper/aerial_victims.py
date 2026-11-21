"""Sensor-only aerial victim candidate detection and multi-view ENU clustering."""
from pathlib import Path
import json
import math
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for name in ("search_agent",):
    p = str(ROOT / "controllers" / name)
    if p not in sys.path:
        sys.path.insert(0, p)

from multiview import aerial_candidates
from perception import PersonDetector, covariance_intersection


class AerialVictimTracker:
    """Cluster multi-frame UAV RGB-D person detections into ENU victim candidates."""

    def __init__(self, association_radius_m=1.6, min_score=0.35):
        self.association_radius_m = float(association_radius_m)
        self.min_score = float(min_score)
        self.clusters = []

    def ingest_candidates(self, candidates, frame_index, timestamp_s, observer_xy, evidence_path):
        for cand in candidates:
            score = float(cand["score"])
            if score < self.min_score:
                continue
            mean = np.asarray(cand["mean_enu"], dtype=float)
            cov = np.asarray(cand["covariance_enu"], dtype=float)
            if not (np.isfinite(mean).all() and np.isfinite(cov).all()):
                continue
            obs_xy = np.asarray(observer_xy[:2], dtype=float)

            best_idx = None
            best_dist = self.association_radius_m
            for idx, cluster in enumerate(self.clusters):
                d = float(np.linalg.norm(mean - cluster["mean"]))
                if d < best_dist:
                    best_dist = d
                    best_idx = idx

            if best_idx is None:
                self.clusters.append(dict(
                    id=len(self.clusters) + 1,
                    mean=mean.copy(),
                    cov=cov.copy(),
                    scores=[score],
                    frames=[int(frame_index)],
                    timestamps=[float(timestamp_s)],
                    observers=[obs_xy.tolist()],
                    evidences=[str(evidence_path)],
                    depths=[float(cand["depth"])],
                ))
            else:
                cluster = self.clusters[best_idx]
                fused_mean, fused_cov = covariance_intersection(cluster["mean"], cluster["cov"], mean, cov)
                cluster["mean"] = fused_mean
                cluster["cov"] = fused_cov
                cluster["scores"].append(score)
                cluster["frames"].append(int(frame_index))
                cluster["timestamps"].append(float(timestamp_s))
                cluster["observers"].append(obs_xy.tolist())
                if len(cluster["evidences"]) < 6:
                    cluster["evidences"].append(str(evidence_path))
                cluster["depths"].append(float(cand["depth"]))

    def export(self, min_views=2, high_score_single=0.88):
        """Return confirmed multi-view ENU victim candidate list (sorted by support)."""
        confirmed = []
        for cluster in self.clusters:
            n_views = len(cluster["frames"])
            max_score = max(cluster["scores"])
            if n_views < min_views and max_score < high_score_single:
                continue
            obs = np.asarray(cluster["observers"], dtype=float)
            baseline = (float(np.max(np.linalg.norm(obs[:, None] - obs[None, :], axis=-1)))
                        if len(obs) >= 2 else 0.0)
            eigvals = np.linalg.eigvalsh(cluster["cov"])
            radius_95 = math.sqrt(5.991 * max(float(np.max(eigvals)), 1e-6))
            confirmed.append(dict(
                id=cluster["id"],
                mean_enu_m=[round(float(cluster["mean"][0]), 4), round(float(cluster["mean"][1]), 4)],
                covariance_enu=[[round(float(v), 5) for v in row] for row in cluster["cov"]],
                radius_95_m=round(radius_95, 4),
                views=n_views,
                baseline_m=round(baseline, 3),
                mean_score=round(float(np.mean(cluster["scores"])), 4),
                max_score=round(float(max_score), 4),
                first_timestamp_s=round(min(cluster["timestamps"]), 3),
                evidence=cluster["evidences"],
            ))
        confirmed.sort(key=lambda item: (-item["views"], -item["mean_score"], item["id"]))
        for new_id, item in enumerate(confirmed, start=1):
            item["id"] = new_id
        return confirmed


def detect_victims_from_saved_survey(run_folder, detector_mode="surrogate_rgbd", min_views=2, transform_bgr=None):
    """Run sensor-only aerial person detection on saved observations.npz + frames/."""
    run_folder = Path(run_folder)
    common = json.loads((ROOT / "config/search/common.json").read_text(encoding="utf-8"))
    score_min = float(common["detection_score_min"][detector_mode])

    with np.load(run_folder / "observations.npz", allow_pickle=False) as obs:
        gps = obs["gps_enu_m"]
        att = obs["attitude_rpy_rad"]
        depths = obs["depth_m"]
        timestamps = obs["timestamp_s"]
        fov = float(obs["fov_rad"])
        phase = obs["survey_phase"] if "survey_phase" in obs else np.zeros(len(gps), dtype=int)
        width, height = map(int, obs["size_wh"])

    detector = PersonDetector(detector_mode, ROOT, max_depth=24.0, field_of_view=fov)
    tracker = AerialVictimTracker(association_radius_m=1.6, min_score=score_min)
    raw_count = 0
    scanned_frames = 0

    for idx in range(len(gps)):
        if int(phase[idx]) != 0:
            continue
        frame_rel = f"frames/{idx:03d}.jpg"
        frame_path = run_folder / frame_rel
        if not frame_path.is_file():
            continue
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            continue
        if transform_bgr is not None:
            bgr = transform_bgr(bgr)
        scanned_frames += 1
        full_depth = cv2.resize(depths[idx], (width, height), interpolation=cv2.INTER_NEAREST)
        cands = aerial_candidates(detector, bgr, full_depth, gps[idx], att[idx], fov)
        raw_count += len(cands)
        if cands:
            tracker.ingest_candidates(cands, idx, float(timestamps[idx]), gps[idx], frame_rel)

    victims = tracker.export(min_views=min_views)
    return dict(
        detector=detector_mode,
        score_min=score_min,
        scanned_nadir_frames=scanned_frames,
        raw_detections=raw_count,
        candidates=victims,
    )
