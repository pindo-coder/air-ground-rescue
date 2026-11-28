"""End-to-end training & head-to-head benchmark: ICTIS 2025 UDDF vs Our Aerial YOLO Method.

Evaluates 6 detectors under strictly zero Scene D leakage (trained on Scenes A+B,
validated/calibrated on Scene C, benchmarked held-out on Scene D 8 suites = 1,248 frames):
  1. COCO-YOLOv8n        (off-the-shelf models/yolov8n.pt, 4-rotation pass)
  2. YOLOv8n-FT          (fine-tuned on v3 Scenes A+B, single pass)
  3. UDDF (T=1, No MC)   (UGDIP + UQN + YOLOv8n, single pass ablation)
  4. UDDF (Full, T=10)   (ICTIS 2025 full pipeline: UGDIP + UQN + YOLOv8n + 10-pass MC-Dropout)
  5. Ours-v2             (supine-only YOLO11n baseline)
  6. Ours-v3             (depth-aware multi-posture + partial-burial + all-weather YOLO11n)
"""
from pathlib import Path
import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
import urllib.request

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO, settings as ultra_settings

ROOT = Path(__file__).resolve().parents[1]
for rel in ("controllers/uav_mapper", "controllers/search_agent", "tools"):
    p = str(ROOT / rel)
    if p not in sys.path:
        sys.path.insert(0, p)

from aerial_victims import AerialVictimTracker
from multiview import aerial_candidates
from perception import unrotate_box
from reproduce_uddf_ictis2025 import (
    INPUT_HW,
    UDDFModel,
    UDDFPersonDetector,
    UDDFVictimDataset,
    _xywh_norm_to_xyxy_px,
    apply_paper_eq3_fog,
    compute_box_metrics,
    train_uddf,
)
from train_yolo_victim import _hue_shift, _to_grayscale

OUT_DIR = ROOT / "runs/uddf_ictis2025"
CN_FIG_DIR = ROOT / "runs/cn_figures_20261121"
DATASET_YAML = ROOT / "runs/yolo_victim_dataset_v3/dataset.yaml"
DATASET_ROOT = ROOT / "runs/yolo_victim_dataset_v3"
YOLOV8N_BASE = ROOT / "models/yolov8n.pt"
YOLOV8N_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"

SUITES = {
    "postures_5": {
        "cn": "5类非仰卧姿态\n(含蜷缩/半掩埋)",
        "en": "5 Non-Supine\nPostures",
        "dir": ROOT / "runs/uav_rescue_D_full_yolo_igha_vctp_clear_randv53_20261123_174245_32236",
        "truth": ROOT / "runs/uav_rescue_D_full_yolo_igha_vctp_clear_randv53_20261123_174245_32236/evaluation/truth_override.json",
        "variant": "orig",
    },
    "orig": {
        "cn": "标准仰卧\n(留出D)",
        "en": "Standard\nSupine",
        "dir": ROOT / "runs/uav_mapping_D_full_route_value_20261121_133722_31008",
        "truth": ROOT / "evaluation/search/D.json",
        "variant": "orig",
    },
    "hue60": {
        "cn": "全局Hue+60°\n(全色域)",
        "en": "Hue +60°\nShift",
        "dir": ROOT / "runs/uav_mapping_D_full_route_value_20261121_133722_31008",
        "truth": ROOT / "evaluation/search/D.json",
        "variant": "hue+60",
    },
    "grayscale": {
        "cn": "纯灰度去色\n(纯轮廓)",
        "en": "Pure\nGrayscale",
        "dir": ROOT / "runs/uav_mapping_D_full_route_value_20261121_133722_31008",
        "truth": ROOT / "evaluation/search/D.json",
        "variant": "grayscale",
    },
    "lowlight": {
        "cn": "黄昏低照度\n(Webots实测)",
        "en": "Low-Light\nDusk",
        "dir": ROOT / "runs/uav_rescue_D_full_yolo_igha_vctp_clear_lowlight_20261123_170102_8096",
        "truth": ROOT / "evaluation/search/D.json",
        "variant": "orig",
    },
    "fog": {
        "cn": "38m三维浓雾\n(Webots实测)",
        "en": "Webots 3D\nDense Fog",
        "dir": ROOT / "runs/uav_rescue_D_full_yolo_igha_vctp_clear_fog_20261123_171203_25240",
        "truth": ROOT / "evaluation/search/D.json",
        "variant": "orig",
    },
    "fog_eq3": {
        "cn": "公式(3)合成雾\n(β=0.10对照)",
        "en": "Eq.(3) Fog\n(β=0.10)",
        "dir": ROOT / "runs/uav_mapping_D_full_route_value_20261121_133722_31008",
        "truth": ROOT / "evaluation/search/D.json",
        "variant": "fog_eq3",
    },
    "randv17": {
        "cn": "随机全非蓝衣\n(seed=17)",
        "en": "Non-Blue\nRandom (s17)",
        "dir": ROOT / "runs/uav_rescue_D_full_yolo_igha_vctp_clear_randv17_20261122_200923_13096",
        "truth": ROOT / "runs/uav_rescue_D_full_yolo_igha_vctp_clear_randv17_20261122_200923_13096/evaluation/truth_override.json",
        "variant": "orig",
    },
}


class UltralyticsDetectAdapter:
    """Consistent Ultralytics detector wrapper supporting raw COCO (4-rot) or single-class (1-pass)."""

    def __init__(self, weights_path, conf_thr=0.50, four_rotations=False):
        ultra_settings.update({"sync": False})
        self.model = YOLO(str(weights_path))
        self.conf_thr = float(conf_thr)
        self.four_rotations = bool(four_rotations)
        self.param_count = sum(p.numel() for p in self.model.model.parameters())

    def detect(self, bgr, depth=None):
        boxes, scores = [], []
        rotations = range(4) if self.four_rotations else (0,)
        for turns in rotations:
            view = bgr if turns == 0 else np.ascontiguousarray(np.rot90(bgr, -turns))
            output = self.model.predict(
                view, conf=self.conf_thr, classes=[0], imgsz=640, device="cpu", verbose=False,
            )[0]
            for box, sc in zip(output.boxes.xyxy.cpu().numpy(), output.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = unrotate_box(box, turns, bgr.shape[1], bgr.shape[0])
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(float(sc))
        if not boxes:
            return []
        keep = np.asarray(cv2.dnn.NMSBoxes(boxes, scores, self.conf_thr, 0.45)).reshape(-1)
        return [
            ([int(round(boxes[i][0])), int(round(boxes[i][1])),
              int(round(boxes[i][0] + boxes[i][2])), int(round(boxes[i][1] + boxes[i][3]))],
             float(scores[i]), None)
            for i in keep
        ]


def ensure_yolov8n_base():
    if not YOLOV8N_BASE.is_file():
        YOLOV8N_BASE.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Setup] Downloading {YOLOV8N_URL} -> {YOLOV8N_BASE}", flush=True)
        with urllib.request.urlopen(YOLOV8N_URL, timeout=60) as resp:
            YOLOV8N_BASE.write_bytes(resp.read())
    return YOLOV8N_BASE


def train_yolov8n_ft(out_pt, epochs=18):
    ultra_settings.update({"sync": False})
    model = YOLO(str(ensure_yolov8n_base()))
    train_name = f"yolov8n_ft_{time.strftime('%Y%m%d_%H%M%S')}"
    model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=640,
        batch=16,
        device="cpu",
        workers=4,
        project=str(OUT_DIR),
        name=train_name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.0018,
        lrf=0.05,
        weight_decay=5e-4,
        warmup_epochs=2.0,
        patience=12,
        hsv_h=0.35,
        hsv_s=0.70,
        hsv_v=0.45,
        degrees=180.0,
        translate=0.15,
        scale=0.45,
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.8,
        mixup=0.08,
        erasing=0.20,
        close_mosaic=5,
        seed=71,
        verbose=True,
    )
    run_dir = OUT_DIR / train_name
    best_pt = run_dir / "weights/best.pt"
    if not best_pt.is_file():
        best_pt = run_dir / "weights/last.pt"
    out_pt = Path(out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, out_pt)
    print(f"[YOLOv8n-FT] Exported -> {out_pt}", flush=True)
    return out_pt


def evaluate_on_scene_c(detector_obj, is_uddf=False, uddf_conf=0.45, uddf_mc=1, uddf_max_unc=0.80):
    val_ds = UDDFVictimDataset(DATASET_ROOT, split="val", augment=False)
    preds, gts = [], []
    for img_path in val_ds.images:
        bgr = cv2.imread(str(img_path))
        h, w = bgr.shape[:2]
        lbl_path = val_ds.root / "labels/val" / f"{img_path.stem}.txt"
        gt_norm = []
        if lbl_path.is_file():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    gt_norm.append(list(map(float, parts[1:])))
        gts.append(_xywh_norm_to_xyxy_px(gt_norm, w, h))

        if is_uddf:
            det_items, _ = detector_obj.model.predict_bgr(
                bgr, conf_thr=uddf_conf, mc_passes=uddf_mc, max_unc=uddf_max_unc,
            )
            preds.append(det_items)
        else:
            raw = detector_obj.detect(bgr, None)
            preds.append([dict(bbox=[float(x) for x in b], score=float(s)) for b, s, _ in raw])
    return compute_box_metrics(preds, gts, iou_thr=0.5)


def profile_latency(detector_obj, n_frames=50, warmup=3):
    """Benchmark actual single-frame CPU latency (median / p95 ms and FPS) on Scene D frames."""
    sample_dir = ROOT / "runs/uav_mapping_D_full_route_value_20261121_133722_31008/frames"
    imgs = [cv2.imread(str(p)) for p in sorted(sample_dir.glob("*.jpg"))[: (n_frames + warmup)]]
    for im in imgs[:warmup]:
        _ = detector_obj.detect(im, None)
    times_ms = []
    for im in imgs[warmup:]:
        t0 = time.perf_counter()
        _ = detector_obj.detect(im, None)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    med = float(np.median(times_ms))
    p95 = float(np.percentile(times_ms, 95))
    return dict(
        median_ms=round(med, 2),
        p95_ms=round(p95, 2),
        fps=round(1000.0 / max(med, 1e-3), 2),
        frames=len(times_ms),
    )


def eval_scene_d_suite(detector_obj, suite_cfg, score_min=0.35, min_views=2):
    variant = suite_cfg["variant"]
    transform = None
    if variant == "hue+60":
        transform = lambda img: _hue_shift(img, 60)
    elif variant == "grayscale":
        transform = _to_grayscale
    elif variant == "fog_eq3":
        transform = lambda img: apply_paper_eq3_fog(img, level_i=5, A=0.5)

    run_dir = suite_cfg["dir"]
    with np.load(run_dir / "observations.npz", allow_pickle=False) as obs:
        gps = obs["gps_enu_m"]
        att = obs["attitude_rpy_rad"]
        depths = obs["depth_m"]
        timestamps = obs["timestamp_s"]
        fov = float(obs["fov_rad"])
        phase = obs["survey_phase"] if "survey_phase" in obs else np.zeros(len(gps), dtype=int)
        width, height = map(int, obs["size_wh"])

    tracker = AerialVictimTracker(association_radius_m=1.6, min_score=score_min)
    raw_count = 0
    scanned_frames = 0
    t0 = time.perf_counter()

    for idx in range(len(gps)):
        if int(phase[idx]) != 0:
            continue
        frame_rel = f"frames/{idx:03d}.jpg"
        frame_path = run_dir / frame_rel
        if not frame_path.is_file():
            continue
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            continue
        if transform is not None:
            bgr = transform(bgr)
        scanned_frames += 1
        full_depth = cv2.resize(depths[idx], (width, height), interpolation=cv2.INTER_NEAREST)
        cands = aerial_candidates(detector_obj, bgr, full_depth, gps[idx], att[idx], fov)
        raw_count += len(cands)
        if cands:
            tracker.ingest_candidates(cands, idx, float(timestamps[idx]), gps[idx], frame_rel)

    elapsed_s = time.perf_counter() - t0
    confirmed = tracker.export(min_views=min_views)

    truth = json.loads(suite_cfg["truth"].read_text(encoding="utf-8"))
    targets = truth["victims"]
    true_pts = np.array([v["xy"] for v in targets], dtype=float)
    tol = float(truth.get("report_tolerance_m", 1.0))

    per_victim = {
        v["id"]: dict(
            id=v["id"], xy=v["xy"], posture=v.get("posture", "supine"),
            detected=False, error_m=None, views=0, max_score=0.0,
        )
        for v in targets
    }
    fps = []
    for cand in confirmed:
        cxy = np.asarray(cand["mean_enu_m"], dtype=float)
        dists = np.linalg.norm(true_pts - cxy[None, :], axis=1)
        j = int(np.argmin(dists))
        err = float(dists[j])
        tid = targets[j]["id"]
        if err <= tol:
            rec = per_victim[tid]
            if (not rec["detected"]) or err < rec["error_m"]:
                rec.update(detected=True, error_m=round(err, 4), views=int(cand["views"]), max_score=round(float(cand["max_score"]), 4))
        else:
            fps.append(dict(xy=cand["mean_enu_m"], views=cand["views"], max_score=cand["max_score"], nearest_err_m=round(err, 3)))

    errs = [r["error_m"] for r in per_victim.values() if r["detected"]]
    matched = len(errs)
    total = len(targets)
    views_list = [r["views"] for r in per_victim.values() if r["detected"]]
    return dict(
        elapsed_s=round(elapsed_s, 2),
        scanned_frames=scanned_frames,
        raw_detections=raw_count,
        matched_victims=matched,
        truth_victims=total,
        recall=round(matched / max(1, total), 4),
        false_positives=len(fps),
        fp_details=fps,
        mean_error_m=round(float(np.mean(errs)), 4) if errs else None,
        rmse_m=round(float(np.sqrt(np.mean(np.square(errs)))), 4) if errs else None,
        mean_views=round(float(np.mean(views_list)), 2) if views_list else 0.0,
        per_victim=[per_victim[v["id"]] for v in targets],
    )


def extract_qualitative_strip(uddf_det, ours_det):
    """Extract held-out dense-fog & half-buried crops showing UGDIP, UQN heatmap, UDDF vs Ours."""
    # Case 1: Webots 3D dense fog (frame 061 of fog run -> Victim #2 at (-12.5, -9.0))
    fog_dir = SUITES["fog"]["dir"]
    fog_bgr = cv2.imread(str(fog_dir / "frames/061.jpg"))
    # Case 2: Held-out half_buried victim (frame 113 of randv53 -> Victim #5)
    buried_dir = SUITES["postures_5"]["dir"]
    buried_bgr = cv2.imread(str(buried_dir / "frames/113.jpg"))

    strip_data = []
    for tag_cn, tag_en, bgr, center_hint in [
        ("38 m 三维浓雾（Webots）", "Webots 3D Fog", fog_bgr, (285, 255)),
        ("废墟半掩埋伤员（留出 D）", "Half-Buried Casualty", buried_bgr, (320, 240)),
    ]:
        uddf_preds, meta = uddf_det.model.predict_bgr(
            bgr, conf_thr=uddf_det.conf_thr, mc_passes=uddf_det.mc_passes, max_unc=uddf_det.max_unc,
        )
        ours_preds = ours_det.detect(bgr, None)

        # Center crop 160x120 around target (using Ours-v3 box if available, else hint)
        cx, cy = center_hint
        if ours_preds:
            bx1, by1, bx2, by2 = ours_preds[0][0]
            cx = int(round(0.5 * (bx1 + bx2)))
            cy = int(round(0.5 * (by1 + by2)))
        hw, hh = 80, 60
        h_i, w_i = bgr.shape[:2]
        x0 = max(0, min(w_i - 2 * hw, cx - hw))
        y0 = max(0, min(h_i - 2 * hh, cy - hh))

        def crop_rgb(img_bgr):
            return cv2.cvtColor(img_bgr[y0:y0 + 2 * hh, x0:x0 + 2 * hw], cv2.COLOR_BGR2RGB)

        sig_src = meta.get("sigma_map", meta["aleatoric_map"])
        ale_crop = sig_src[y0:y0 + 2 * hh, x0:x0 + 2 * hw]
        strip_data.append(dict(
            tag_cn=tag_cn,
            tag_en=tag_en,
            raw_rgb=crop_rgb(bgr),
            enh_rgb=crop_rgb(meta["enhanced_bgr"]),
            ale_map=ale_crop,
            uddf_boxes=[([b["bbox"][0] - x0, b["bbox"][1] - y0, b["bbox"][2] - x0, b["bbox"][3] - y0], b["score"], b["u_total"]) for b in uddf_preds],
            ours_boxes=[([b[0][0] - x0, b[0][1] - y0, b[0][2] - x0, b[0][3] - y0], b[1]) for b in ours_preds],
        ))
    return strip_data


def _render_figure(report, strip_data, lang, out_paths):
    is_cn = (lang == "cn")
    if is_cn:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["Times New Roman", "DejaVu Serif", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    methods_plot = [
        ("yolov8n_ft", "YOLOv8n-FT" if not is_cn else "YOLOv8n-FT (同数据)", "#64748b"),
        ("uddf_t1", "UDDF (T=1)" if not is_cn else "UDDF (T=1 无MC)", "#f59e0b"),
        ("uddf_t10", "UDDF (T=10, ICTIS'25)" if not is_cn else "UDDF (T=10 论文完整版)", "#ea580c"),
        ("ours_v2", "Ours-v2 (Supine)" if not is_cn else "Ours-v2 (仅仰卧基线)", "#94a3b8"),
        ("ours_v3", "Ours-v3 (YOLO11n)" if not is_cn else "Ours-v3 (本文完整版)", "#16a34a"),
    ]

    fig = plt.figure(figsize=(18.0, 10.2), dpi=180, facecolor="#ffffff")
    gs = fig.add_gridspec(2, 1, height_ratios=[0.95, 1.30], hspace=0.28, left=0.048, right=0.985, top=0.88, bottom=0.08)

    res_uddf = report["methods"]["uddf_t10"]
    res_ours = report["methods"]["ours_v3"]
    u_hit, o_hit = res_uddf["scene_d_summary"]["total_matched"], res_ours["scene_d_summary"]["total_matched"]
    u_tot = res_uddf["scene_d_summary"]["total_truth"]
    u_fp, o_fp = res_uddf["scene_d_summary"]["total_fp"], res_ours["scene_d_summary"]["total_fp"]
    u_lat, o_lat = res_uddf["latency"]["median_ms"], res_ours["latency"]["median_ms"]
    speedup = u_lat / max(o_lat, 1e-3)

    if is_cn:
        fig.suptitle(
            f"ICTIS 2025 UDDF (UGDIP + UQN + YOLOv8n + MC-Dropout) 复现与本文方法全面对比（严格留出 D 场景 8 大工况）",
            fontsize=14.6, fontweight="bold", y=0.97, color="#0f172a",
        )
        fig.text(
            0.5, 0.93,
            f"同源零泄漏 A+B 训练 (784 图) / C 验证 (175 图) → D 全程留出 (1,248 帧, 26 人次) | "
            f"本文 v3: {o_hit}/{u_tot} 召回 · {o_fp} 误报 · {o_lat:.1f} ms/帧 vs UDDF(T=10): {u_hit}/{u_tot} 召回 · {u_fp} 误报 · {u_lat:.1f} ms/帧 ({speedup:.1f}× 提速)",
            ha="center", va="center", fontsize=10.1, color="#334155",
        )
    else:
        fig.suptitle(
            "Head-to-Head Comparison: Reproduced ICTIS 2025 UDDF vs. Proposed Physics-Aware Aerial YOLO11n (Held-Out Scene D)",
            fontsize=14.0, fontweight="bold", y=0.97, color="#0f172a",
        )
        fig.text(
            0.5, 0.93,
            f"Zero-Leakage Protocol (Train: A+B 784 imgs, Calibrate: C 175 imgs, Held-Out Test: D 8 Suites / 1,248 Frames / 26 Targets) | "
            f"Ours-v3: {o_hit}/{u_tot} Recall, {o_fp} FP, {o_lat:.1f} ms vs. UDDF (T=10): {u_hit}/{u_tot} Recall, {u_fp} FP, {u_lat:.1f} ms ({speedup:.1f}x faster)",
            ha="center", va="center", fontsize=10.0, color="#334155",
        )

    # Top row: 2 rows x 4 columns = 8 panels? Use 2 cases x 4 columns side by side (1x8 grid)!
    gs_top = gs[0].subgridspec(1, 8, wspace=0.10)
    col_idx = 0
    for item in strip_data:
        case_lbl = item["tag_cn"] if is_cn else item["tag_en"]
        panels = [
            (item["raw_rgb"], f"{case_lbl}\n" + ("原始输入帧" if is_cn else "Raw Input"), None, None),
            (item["enh_rgb"], "UGDIP " + ("增强输出" if is_cn else "Enhanced"), None, None),
            (item["ale_map"], "UQN " + ("偶然不确定图" if is_cn else "Uncertainty Map"), "jet", None),
            (item["raw_rgb"], ("UDDF vs 本文v3" if is_cn else "UDDF vs Ours-v3"), None, "boxes"),
        ]
        for img_arr, title, cmap, mode in panels:
            ax = fig.add_subplot(gs_top[0, col_idx])
            col_idx += 1
            if cmap is not None:
                ax.imshow(img_arr, cmap=cmap, vmin=0.0, vmax=1.0)
            else:
                ax.imshow(img_arr)
            if mode == "boxes":
                for box, sc, ut in item["uddf_boxes"]:
                    x1, y1, x2, y2 = box
                    if -10 <= x1 <= 170 and -10 <= y1 <= 130:
                        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#f97316", lw=1.8, ls="--"))
                        ax.text(max(2, x1), min(114, y2 + 9), f"UDDF {sc:.2f}", color="#fff", fontsize=6.8, fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.14", fc="#ea580c", ec="none", alpha=0.9))
                for box, sc in item["ours_boxes"]:
                    x1, y1, x2, y2 = box
                    if -10 <= x1 <= 170 and -10 <= y1 <= 130:
                        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#22c55e", lw=2.2))
                        ax.text(max(2, x1), max(10, y1 - 3), f"v3 {sc:.2f}", color="#fff", fontsize=7.0, fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.14", fc="#16a34a", ec="none", alpha=0.92))
            ax.set_title(title, fontsize=8.8, fontweight="bold", pad=4, color="#0f172a")
            ax.set_xticks([])
            ax.set_yticks([])

    # Bottom row: 3 quantitative panels
    gs_bot = gs[1].subgridspec(1, 3, width_ratios=[1.50, 0.98, 0.82], wspace=0.34)

    # Panel A: 8-Suite Recall Heatmap / Grouped Bar on UDDF(T=10) vs YOLOv8n-FT vs Ours-v3
    ax_a = fig.add_subplot(gs_bot[0, 0])
    s_keys = ["postures_5", "orig", "hue60", "grayscale", "lowlight", "fog", "fog_eq3", "randv17"]
    s_labels = [SUITES[k]["cn" if is_cn else "en"] for k in s_keys]
    xs = np.arange(len(s_keys))
    bar_specs = [
        ("yolov8n_ft", -0.26, "#64748b", "YOLOv8n-FT"),
        ("uddf_t10", 0.0, "#ea580c", "UDDF (T=10)" if not is_cn else "UDDF (T=10 论文完整版)"),
        ("ours_v3", 0.26, "#16a34a", "Ours-v3" if not is_cn else "Ours-v3 (本文)"),
    ]
    for m_key, offset, col, lbl in bar_specs:
        vals = [report["methods"][m_key]["scene_d_suites"][k]["recall"] * 100.0 for k in s_keys]
        bs = ax_a.bar(xs + offset, vals, 0.25, color=col, edgecolor="#1e293b", lw=0.6, label=lbl)
        for idx_k, (b, k) in enumerate(zip(bs, s_keys)):
            sr = report["methods"][m_key]["scene_d_suites"][k]
            txt = f"{sr['matched_victims']}/{sr['truth_victims']}"
            if sr["false_positives"] > 0:
                txt += f"\n+{sr['false_positives']}FP"
            ax_a.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, txt,
                      ha="center", va="bottom", fontsize=7.0, fontweight="bold",
                      color="#b91c1c" if (sr["matched_victims"] < sr["truth_victims"] or sr["false_positives"] > 0) else "#0f172a")
    ax_a.set_ylim(0, 128)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(s_labels, fontsize=8.0)
    ax_a.set_ylabel("高空多视角召回率 (%)" if is_cn else "Aerial Multi-View Recall (%)", fontsize=9.4)
    ax_a.set_title(
        "留出 D 场景 8 大工况全召回与误报对比（1,248 帧）" if is_cn else "Held-Out Scene D 8-Suite Recall & FP (1,248 Frames)",
        fontsize=10.4, fontweight="bold", pad=8,
    )
    ax_a.legend(loc="upper center", ncol=3, fontsize=8.0, framealpha=0.95)
    ax_a.grid(axis="y", alpha=0.3)

    # Panel B: Aggregate 26-Victim Recall, FP, and 3D ENU Localization RMSE across all 5 fine-tuned models
    ax_b = fig.add_subplot(gs_bot[0, 1])
    xm = np.arange(len(methods_plot))
    recalls_all = [report["methods"][k]["scene_d_summary"]["total_recall_pct"] for k, _, _ in methods_plot]
    rmses_all = [report["methods"][k]["scene_d_summary"]["overall_rmse_m"] or 0.0 for k, _, _ in methods_plot]
    ax_b2 = ax_b.twinx()
    b_rec = ax_b.bar(xm - 0.18, recalls_all, 0.34, color=[c for _, _, c in methods_plot], edgecolor="#1e293b", lw=0.6,
                     label="总召回率 (%)" if is_cn else "Total Recall (%)")
    b_err = ax_b2.bar(xm + 0.18, rmses_all, 0.34, color="#3b82f6", alpha=0.75, edgecolor="#1e3a8a", lw=0.6,
                      label="3D定位RMSE (m)" if is_cn else "3D Loc RMSE (m)")
    for b, (k, _, _) in zip(b_rec, methods_plot):
        sm = report["methods"][k]["scene_d_summary"]
        ax_b.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                  f"{sm['total_matched']}/26\n{sm['total_fp']}FP",
                  ha="center", va="bottom", fontsize=7.2, fontweight="bold",
                  color="#15803d" if (sm["total_matched"] == 26 and sm["total_fp"] == 0) else "#334155")
    for b in b_err:
        ax_b2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                   f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#1e40af")
    ax_b.set_ylim(0, 128)
    ax_b2.set_ylim(0, max(max(rmses_all, default=0.3) * 1.45, 0.38))
    ax_b.set_xticks(xm)
    ax_b.set_xticklabels([lbl.replace(" ", "\n") for _, lbl, _ in methods_plot], fontsize=7.4)
    ax_b.set_ylabel("总召回率 (%)" if is_cn else "Aggregate Recall (%)", fontsize=9.2)
    ax_b2.set_ylabel("三维定位 RMSE (m)" if is_cn else "3D ENU RMSE (m)", fontsize=9.0, color="#1e40af", labelpad=6)
    ax_b.set_title("总召回 / 误报 / 三维定位精度" if is_cn else "Aggregate Recall, FP & 3D RMSE", fontsize=10.4, fontweight="bold", pad=8)
    h1, l1 = ax_b.get_legend_handles_labels()
    h2, l2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.6, framealpha=0.95)
    ax_b.grid(axis="y", alpha=0.3)

    # Panel C: CPU Per-Frame Inference Latency & Speedup
    ax_c = fig.add_subplot(gs_bot[0, 2])
    lats = [report["methods"][k]["latency"]["median_ms"] for k, _, _ in methods_plot]
    fps_vals = [report["methods"][k]["latency"]["fps"] for k, _, _ in methods_plot]
    bc = ax_c.bar(xm, lats, 0.50, color=[c for _, _, c in methods_plot], edgecolor="#1e293b", lw=0.6)
    for b, lat, fps in zip(bc, lats, fps_vals):
        ax_c.text(b.get_x() + b.get_width() / 2, b.get_height() + max(lats) * 0.02,
                  f"{lat:.1f} ms\n({fps:.1f} FPS)", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#0f172a")
    ax_c.set_ylim(0, max(lats) * 1.24)
    ax_c.set_xticks(xm)
    ax_c.set_xticklabels([lbl.replace(" ", "\n") for _, lbl, _ in methods_plot], fontsize=7.4)
    ax_c.set_ylabel("单帧 CPU 推理耗时 (ms)" if is_cn else "CPU Latency (ms / frame)", fontsize=9.2, labelpad=6)
    ax_c.set_title("单帧推理延迟与实时帧率" if is_cn else "Onboard CPU Latency & FPS", fontsize=10.4, fontweight="bold", pad=8)
    ax_c.grid(axis="y", alpha=0.3)

    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=180)
        print(f"[Figure] Saved {p}", flush=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Reproduce ICTIS 2025 UDDF & benchmark vs Ours.")
    parser.add_argument("--yolov8-epochs", type=int, default=18)
    parser.add_argument("--uddf-epochs", type=int, default=7)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CN_FIG_DIR.mkdir(parents=True, exist_ok=True)

    yolov8_base = ensure_yolov8n_base()
    yolov8_ft_pt = OUT_DIR / "yolov8n_ft.pt"
    uddf_ckpt_pt = OUT_DIR / "uddf_yolov8n.pt"
    json_path = OUT_DIR / "uddf_vs_ours_benchmark.json"

    if args.plot_only:
        report = json.loads(json_path.read_text(encoding="utf-8"))
        uddf_ckpt = torch.load(str(uddf_ckpt_pt), map_location="cpu", weights_only=False)
        cal_conf = float(uddf_ckpt.get("calibrated_conf_thr", 0.45))
        cal_max_u = float(uddf_ckpt.get("calibrated_max_unc", 0.75))
        uddf_t10 = UDDFPersonDetector(uddf_ckpt_pt, conf_thr=cal_conf, mc_passes=10, max_unc=cal_max_u)
        ours_v3 = UltralyticsDetectAdapter(ROOT / "models/yolo11n_victim.pt", conf_thr=0.50, four_rotations=False)
        strip = extract_qualitative_strip(uddf_t10, ours_v3)
        _render_figure(report, strip, "cn", [CN_FIG_DIR / "29_ICTIS2025_UDDF复现与本文方法全面对比图.png"])
        _render_figure(report, strip, "en", [OUT_DIR / "fig_uddf_vs_ours_ictis2027.png", OUT_DIR / "fig_uddf_vs_ours_ictis2027.pdf"])
        return

    if not (args.skip_train and yolov8_ft_pt.is_file()):
        print("=== Step 1: Training YOLOv8n-FT on zero-leakage v3 dataset (Scenes A+B -> Scene C) ===", flush=True)
        train_yolov8n_ft(yolov8_ft_pt, epochs=args.yolov8_epochs)
    else:
        print(f"[Skip] Reusing existing {yolov8_ft_pt}", flush=True)

    if not (args.skip_train and uddf_ckpt_pt.is_file()):
        print("=== Step 2: Jointly training UDDF (UGDIP + UQN + YOLOv8n) with Eq. (2) loss ===", flush=True)
        train_uddf(DATASET_ROOT, yolov8_ft_pt, uddf_ckpt_pt, epochs=args.uddf_epochs, batch_size=8, lr=2.5e-4)
    else:
        print(f"[Skip] Reusing existing {uddf_ckpt_pt}", flush=True)

    uddf_ckpt = torch.load(str(uddf_ckpt_pt), map_location="cpu", weights_only=False)
    cal_conf = float(uddf_ckpt.get("calibrated_conf_thr", 0.45))
    cal_max_u = float(uddf_ckpt.get("calibrated_max_unc", 0.75))

    ours_v2_pt = ROOT / "runs/yolo_victim_v2_backup.pt"
    if not ours_v2_pt.is_file():
        ours_v2_pt = ROOT / "models/yolo11n_victim.pt"

    detectors = {
        "coco_yolov8n": ("COCO-YOLOv8n", UltralyticsDetectAdapter(yolov8_base, conf_thr=0.35, four_rotations=True), False, {}),
        "yolov8n_ft": ("YOLOv8n-FT", UltralyticsDetectAdapter(yolov8_ft_pt, conf_thr=0.50, four_rotations=False), False, {}),
        "uddf_t1": ("UDDF (T=1, No MC-Dropout)", UDDFPersonDetector(uddf_ckpt_pt, conf_thr=cal_conf, mc_passes=1, max_unc=1.0), True, dict(mc=1, max_u=1.0)),
        "uddf_t10": ("UDDF (Full ICTIS'25, T=10)", UDDFPersonDetector(uddf_ckpt_pt, conf_thr=cal_conf, mc_passes=10, max_unc=cal_max_u), True, dict(mc=10, max_u=cal_max_u)),
        "ours_v2": ("Ours-v2 (Supine YOLO11n)", UltralyticsDetectAdapter(ours_v2_pt, conf_thr=0.50, four_rotations=False), False, {}),
        "ours_v3": ("Ours-v3 (Depth-Aware Multi-Posture YOLO11n)", UltralyticsDetectAdapter(ROOT / "models/yolo11n_victim.pt", conf_thr=0.50, four_rotations=False), False, {}),
    }

    report = {
        "protocol": {
            "train": "Scenes A+B (784 images, 6 postures + partial burial + grayscale + rangefinder fog)",
            "val_and_calibration": "Scene C (175 images)",
            "held_out_test": "Scene D (8 suites, 1,248 aerial frames, 26 victim instances)",
            "uddf_calibrated_conf_thr": cal_conf,
            "uddf_calibrated_max_unc": cal_max_u,
            "uddf_unc_logits_weights": torch.softmax(uddf_ckpt["model_state"]["unc_logits"].detach().clone(), dim=0).tolist(),
        },
        "methods": {},
    }

    for key, (disp_name, det_obj, is_uddf, u_cfg) in detectors.items():
        print(f"\n=== Evaluating {disp_name} ===", flush=True)
        if is_uddf:
            params_m = round(sum(p.numel() for p in det_obj.model.parameters()) / 1e6, 3)
            scene_c = evaluate_on_scene_c(det_obj, is_uddf=True, uddf_conf=cal_conf, uddf_mc=u_cfg["mc"], uddf_max_unc=u_cfg["max_u"])
        else:
            params_m = round(det_obj.param_count / 1e6, 3)
            scene_c = evaluate_on_scene_c(det_obj, is_uddf=False)

        latency = profile_latency(det_obj, n_frames=50, warmup=3)
        print(f"  [Scene C Val] P={scene_c['precision']:.3f} R={scene_c['recall']:.3f} F1={scene_c['f1']:.3f} AP50={scene_c['ap50']:.3f} unc_corr={scene_c['unc_err_corr']} | latency={latency['median_ms']:.2f}ms ({latency['fps']:.1f} FPS)", flush=True)

        suites_res = {}
        tot_matched, tot_truth, tot_fp = 0, 0, 0
        all_errs = []
        for skey, scfg in SUITES.items():
            sr = eval_scene_d_suite(det_obj, scfg, score_min=0.35, min_views=2)
            suites_res[skey] = sr
            tot_matched += sr["matched_victims"]
            tot_truth += sr["truth_victims"]
            tot_fp += sr["false_positives"]
            for pv in sr["per_victim"]:
                if pv["detected"] and pv["error_m"] is not None:
                    all_errs.append(pv["error_m"])
            print(f"  [Scene D][{skey:10s}] {sr['matched_victims']}/{sr['truth_victims']} FP={sr['false_positives']} rmse={sr['rmse_m']} ({sr['elapsed_s']:.1f}s)", flush=True)

        overall_rmse = round(float(np.sqrt(np.mean(np.square(all_errs)))), 4) if all_errs else None
        mean_err = round(float(np.mean(all_errs)), 4) if all_errs else None
        report["methods"][key] = dict(
            name=disp_name,
            params_m=params_m,
            scene_c=scene_c,
            latency=latency,
            scene_d_suites=suites_res,
            scene_d_summary=dict(
                total_matched=tot_matched,
                total_truth=tot_truth,
                total_recall_pct=round(100.0 * tot_matched / max(1, tot_truth), 2),
                total_fp=tot_fp,
                overall_mean_error_m=mean_err,
                overall_rmse_m=overall_rmse,
            ),
        )

    json_path = OUT_DIR / "uddf_vs_ours_benchmark.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = OUT_DIR / "uddf_vs_ours_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Params_M", "SceneC_P", "SceneC_R", "SceneC_F1", "SceneC_AP50", "Unc_Err_Corr",
                    "SceneD_Matched", "SceneD_Truth", "SceneD_Recall_Pct", "SceneD_FP", "SceneD_RMSE_m",
                    "Postures5_Recall", "WebotsFog_Recall", "Eq3Fog_Recall", "CPU_Med_ms", "CPU_P95_ms", "FPS"])
        for key, m in report["methods"].items():
            sc = m["scene_c"]
            sd = m["scene_d_summary"]
            ss = m["scene_d_suites"]
            lat = m["latency"]
            w.writerow([
                m["name"], m["params_m"], sc["precision"], sc["recall"], sc["f1"], sc["ap50"], sc["unc_err_corr"],
                sd["total_matched"], sd["total_truth"], sd["total_recall_pct"], sd["total_fp"], sd["overall_rmse_m"],
                f"{ss['postures_5']['matched_victims']}/{ss['postures_5']['truth_victims']}",
                f"{ss['fog']['matched_victims']}/{ss['fog']['truth_victims']}",
                f"{ss['fog_eq3']['matched_victims']}/{ss['fog_eq3']['truth_victims']}",
                lat["median_ms"], lat["p95_ms"], lat["fps"],
            ])

    print(f"\n[Saved] {json_path}\n[Saved] {csv_path}", flush=True)

    strip = extract_qualitative_strip(detectors["uddf_t10"][1], detectors["ours_v3"][1])
    _render_figure(report, strip, "cn", [CN_FIG_DIR / "29_ICTIS2025_UDDF复现与本文方法全面对比图.png"])
    _render_figure(report, strip, "en", [OUT_DIR / "fig_uddf_vs_ours_ictis2027.png", OUT_DIR / "fig_uddf_vs_ours_ictis2027.pdf"])


if __name__ == "__main__":
    main()
