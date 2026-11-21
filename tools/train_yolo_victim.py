"""Fine-tune YOLO11n on Webots Scenes A/B/C (UAV + Rover) and benchmark held-out on Scene D."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for rel in ("controllers/uav_mapper", "controllers/search_agent"):
    p = str(ROOT / rel)
    if p not in sys.path:
        sys.path.insert(0, p)


def benchmark_on_scene_d(weights_path, survey_dir, conf_min=0.35, min_views=2):
    """Evaluate a YOLO weights file on strictly held-out Scene D aerial survey frames."""
    from aerial_victims import detect_victims_from_saved_survey

    prev_env = os.environ.get("UAV_YOLO_WEIGHTS")
    os.environ["UAV_YOLO_WEIGHTS"] = str(weights_path)
    try:
        t0 = time.time()
        doc = detect_victims_from_saved_survey(
            survey_dir, detector_mode="yolo", min_views=min_views,
        )
        elapsed_s = time.time() - t0
    finally:
        if prev_env is None:
            os.environ.pop("UAV_YOLO_WEIGHTS", None)
        else:
            os.environ["UAV_YOLO_WEIGHTS"] = prev_env

    truth = json.loads((ROOT / "evaluation/search/D.json").read_text(encoding="utf-8"))
    targets = truth["victims"]
    true_pts = np.array([v["xy"] for v in targets], dtype=float)
    tol = float(truth.get("report_tolerance_m", 1.0))

    matched_truth = {}
    false_positives = []
    audits = []
    for cand in doc["candidates"]:
        cxy = np.asarray(cand["mean_enu_m"], dtype=float)
        dists = np.linalg.norm(true_pts - cxy[None, :], axis=1)
        j = int(np.argmin(dists))
        err = float(dists[j])
        tid = targets[j]["id"]
        hit = err <= tol
        audits.append(dict(
            candidate_id=cand["id"],
            detected_xy_m=cand["mean_enu_m"],
            nearest_truth_victim_id=tid,
            error_m=round(err, 4),
            within_tol=hit,
            views=cand["views"],
            max_score=cand["max_score"],
        ))
        if hit and (tid not in matched_truth or err < matched_truth[tid]):
            matched_truth[tid] = err
        elif not hit:
            false_positives.append(cand)

    errors = list(matched_truth.values())
    return dict(
        weights=str(weights_path),
        survey_dir=str(survey_dir),
        elapsed_s=round(elapsed_s, 2),
        raw_frame_detections=doc["raw_detections"],
        candidates_total=len(doc["candidates"]),
        matched_victims=len(matched_truth),
        truth_victims=len(targets),
        matched_victim_ids=sorted(matched_truth.keys()),
        false_positives=len(false_positives),
        localization_errors_m=[round(e, 4) for e in errors],
        localization_rmse_m=round(float(np.sqrt(np.mean(np.square(errors)))), 4) if errors else None,
        audits=audits,
    )


def train(dataset_yaml, epochs=35, imgsz=640, batch=16, lr0=0.0025,
          output_weights="models/yolo11n_victim.pt", survey_dir=None):
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    dataset_yaml = Path(dataset_yaml)
    if not dataset_yaml.is_absolute():
        dataset_yaml = ROOT / dataset_yaml
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset yaml not found: {dataset_yaml}")

    base_weights = ROOT / "models/yolo11n.pt"
    if not base_weights.is_file():
        raise FileNotFoundError(f"Base weights missing: {base_weights}")

    run_name = f"yolo_victim_train_{time.strftime('%Y%m%d_%H%M%S')}"
    project_dir = ROOT / "runs"

    model = YOLO(str(base_weights))
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        workers=4,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=lr0,
        lrf=0.05,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        patience=15,
        # Strong hue & full 360-deg rotation augmentation so the model learns body silhouette, not blue shirts.
        hsv_h=0.18,
        hsv_s=0.65,
        hsv_v=0.45,
        degrees=180.0,
        translate=0.15,
        scale=0.45,
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.8,
        mixup=0.08,
        close_mosaic=8,
        seed=71,
        verbose=True,
    )

    train_dir = project_dir / run_name
    best_pt = train_dir / "weights/best.pt"
    if not best_pt.is_file():
        best_pt = train_dir / "weights/last.pt"
    out_path = Path(output_weights)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, out_path)
    print(f"[Train YOLO] Exported fine-tuned weights -> {out_path}", flush=True)

    val_metrics = dict(
        val_precision=round(float(results.results_dict.get("metrics/precision(B)", 0.0)), 4),
        val_recall=round(float(results.results_dict.get("metrics/recall(B)", 0.0)), 4),
        val_map50=round(float(results.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
        val_map50_95=round(float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
    )

    report = dict(
        train_dir=str(train_dir),
        base_weights=str(base_weights),
        finetuned_weights=str(out_path),
        epochs=epochs,
        imgsz=imgsz,
        val_metrics_scene_C=val_metrics,
    )

    if survey_dir:
        sdir = Path(survey_dir)
        if not sdir.is_absolute():
            sdir = ROOT / sdir
        if sdir.is_dir():
            print("[Train YOLO] Running held-out Scene D A/B benchmark (upstream COCO vs fine-tuned)...", flush=True)
            report["held_out_scene_D_baseline_coco"] = benchmark_on_scene_d(base_weights, sdir)
            report["held_out_scene_D_finetuned"] = benchmark_on_scene_d(out_path, sdir)

    report_file = train_dir / "benchmark_scene_D.json"
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return out_path, report


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11n on Webots Scenes A/B/C and test held-out on Scene D.")
    parser.add_argument("--data", default="runs/yolo_victim_dataset/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.0025)
    parser.add_argument("--output", default="models/yolo11n_victim.pt")
    parser.add_argument("--benchmark-survey", default="runs/uav_mapping_D_full_route_value_20261121_133722_31008")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    if args.benchmark_only:
        sdir = Path(args.benchmark_survey)
        if not sdir.is_absolute():
            sdir = ROOT / sdir
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        report = dict(
            held_out_scene_D_baseline_coco=benchmark_on_scene_d(ROOT / "models/yolo11n.pt", sdir),
            held_out_scene_D_finetuned=benchmark_on_scene_d(out_path, sdir),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    train(
        dataset_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        output_weights=args.output,
        survey_dir=args.benchmark_survey,
    )


if __name__ == "__main__":
    main()
