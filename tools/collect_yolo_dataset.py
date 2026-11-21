"""Collect a zero-leakage Webots UAV + Rover victim detection dataset from Scenes A, B (train) and C (val).

Scene D is strictly held out and NEVER used during dataset generation or training.
"""
from pathlib import Path
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from build_search_worlds import build, replace_node
from run_search import ROOT, executable_path, simulator_has_errors
from scene_elements import RescueElements

sys.path.insert(0, str(ROOT / "evaluation"))
from metrics import height_at


# Extra casualty placements (x, y, yaw) on Scenes A, B, C (strictly outside Scene D).
EXTRA_CASUALTIES = {
    "A": [
        (-11.5, -6.5, 0.35),
        (-3.0, 8.0, 1.95),
        (5.5, -5.5, -2.10),
        (11.0, 6.2, 0.85),
        (-13.5, 5.5, -0.90),
    ],
    "B": [
        (-12.0, -5.0, 2.40),
        (-4.5, 7.5, -0.30),
        (4.0, -6.8, 1.25),
        (12.2, 4.5, -1.60),
        (-2.0, -2.5, 0.55),
    ],
    "C": [
        (-11.0, -7.2, -0.75),
        (-5.5, 6.8, 2.10),
        (6.0, -4.5, 0.15),
        (10.5, 7.0, -2.45),
        (1.0, 1.5, 1.40),
    ],
}


def build_collector_world(world_path, scene_key):
    """Build a temporary Webots world based on Scene A, B, or C with extra varied casualties."""
    assert scene_key in ("A", "B", "C"), "Scene D must remain strictly held out!"
    source = ROOT / "worlds" / f"search_{scene_key}.wbt"
    if not source.is_file():
        source = build(scene_key)
    scene = source.read_text(encoding="utf-8")
    meta = json.loads((ROOT / "evaluation/scenarios" / f"{scene_key}.json").read_text(encoding="utf-8"))

    # Remove rover & referee so the collector runs super fast and cleanly.
    scene = replace_node(scene, r"\bDEF\s+ROVER\s+ResearchRover", "")
    scene = replace_node(scene, r"\bDEF\s+SEARCH_REFEREE\s+Robot", "")

    victim_defs = ["CASUALTY"]
    for idx, (vx, vy, vyaw) in enumerate(EXTRA_CASUALTIES[scene_key], 1):
        # Cycle visual model keys A/B/C so shirt/pants/skin variation is diverse.
        vkey = ("A", "B", "C")[(idx + ord(scene_key)) % 3]
        elements = RescueElements(dict(key=vkey, seed=500 + idx * 17 + ord(scene_key), goal=[vx, vy]),
                                  lambda x, y: height_at(meta, x, y))
        elements.casualty()
        node = elements.nodes[0]
        def_name = f"TRAIN_VICTIM_{idx}"
        node = node.replace("DEF CASUALTY ", f"DEF {def_name} ", 1)
        node = node.replace('name "casualty"', f'name "train_victim_{idx}"', 1)
        # Override yaw by wrapping/replacing the top-level rotation? Easier: ground_frame already baked yaw,
        # so vary yaw by cycling vkey + let UAV random yaw cover full 360 deg orientation!
        scene += "\n" + node
        victim_defs.append(def_name)

    collector_node = '''
DEF YOLO_COLLECTOR Robot {
  name "YOLO dataset collector"
  supervisor TRUE
  controller "yolo_dataset_collector"
  translation 0 0 18
  rotation 0 1 0 1.5707963
  window "<none>"
  children [
    Camera {
      name "uav_rgb"
      width 640
      height 480
      fieldOfView 1.1
      near 0.2
      far 80
    }
    RangeFinder {
      name "uav_depth"
      width 640
      height 480
      fieldOfView 1.1
      minRange 0.25
      maxRange 35
    }
    Camera {
      name "rover_rgb"
      width 160
      height 120
      fieldOfView 1.5707963
      near 0.1
      far 30
    }
    RangeFinder {
      name "rover_depth"
      width 160
      height 120
      fieldOfView 1.5707963
      minRange 0.25
      maxRange 12
    }
  ]
}
'''
    scene += "\n" + collector_node
    world_path.write_text(scene, encoding="utf-8")
    return victim_defs


def draw_label_preview(dataset_dir):
    """Draw a 4x4 contact sheet of labeled UAV and Rover samples for visual quality audit."""
    samples = []
    for split in ("train", "val"):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        for img_path in sorted(img_dir.glob("*_pos_*.jpg")):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.is_file() and lbl_path.read_text(encoding="utf-8").strip():
                samples.append((split, img_path, lbl_path))
    if not samples:
        return
    uav_s = [s for s in samples if "_uav_" in s[1].stem]
    rov_s = [s for s in samples if "_rover_" in s[1].stem]
    rng = random.Random(42)
    rng.shuffle(uav_s)
    rng.shuffle(rov_s)
    picks = uav_s[:10] + rov_s[:6]
    while len(picks) < 16 and samples:
        picks.append(samples[len(picks) % len(samples)])

    tiles = []
    for split, img_path, lbl_path in picks[:16]:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        canvas = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            x1 = int(round((cx - bw / 2) * 320))
            y1 = int(round((cy - bh / 2) * 240))
            x2 = int(round((cx + bw / 2) * 320))
            y2 = int(round((cy + bh / 2) * 240))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 240, 80), 2)
        cv2.putText(canvas, f"{split}:{img_path.stem[:20]}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)

    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, 16, 4)]
    sheet = np.vstack(rows)
    cv2.imwrite(str(dataset_dir / "preview_labels.jpg"), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def collect(out_dir, scale=1.0, port=18590):
    dataset_dir = Path(out_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    executable = executable_path()
    runtime = f"[python]\nCOMMAND = {Path(sys.executable).as_posix()}\n"
    (ROOT / "controllers/yolo_dataset_collector/runtime.ini").write_text(runtime, encoding="utf-8")

    # Train on Scenes A + B; Val on Scene C; Scene D strictly held out!
    plan = [
        ("A", "train", int(round(130 * scale)), int(round(45 * scale)), int(round(55 * scale)), int(round(18 * scale)), 101),
        ("B", "train", int(round(130 * scale)), int(round(45 * scale)), int(round(55 * scale)), int(round(18 * scale)), 202),
        ("C", "val",   int(round(90 * scale)),  int(round(30 * scale)), int(round(40 * scale)), int(round(15 * scale)), 303),
    ]

    manifest = dict(
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        held_out_test_scenes=["D"],
        train_scenes=["A", "B"],
        val_scenes=["C"],
        per_scene={},
    )

    for idx, (scene_key, split, uav_pos, uav_neg, rov_pos, rov_neg, seed) in enumerate(plan):
        world_path = ROOT / "worlds" / f"_tmp_yolo_collect_{scene_key}.wbt"
        victim_defs = build_collector_world(world_path, scene_key)
        report_path = dataset_dir / f"report_{scene_key}.json"
        spec_path = dataset_dir / f"spec_{scene_key}.json"
        spec = dict(
            scene=scene_key,
            prefix=f"scene{scene_key}",
            seed=seed,
            victim_defs=victim_defs,
            uav_pos=uav_pos,
            uav_neg=uav_neg,
            rover_pos=rov_pos,
            rover_neg=rov_neg,
            out_images=str(dataset_dir / "images" / split),
            out_labels=str(dataset_dir / "labels" / split),
            report_path=str(report_path),
        )
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        env = dict(os.environ, YOLO_COLLECT_SPEC=str(spec_path), PYTHONDONTWRITEBYTECODE="1")
        cmd = [str(executable), "--batch", "--mode=fast", f"--port={port + idx}",
               "--stdout", "--stderr", "--no-rendering", str(world_path)]
        print(f"[Collect] Scene {scene_key} ({split}) -> target {uav_pos}+{uav_neg} UAV, {rov_pos}+{rov_neg} Rover...",
              flush=True)
        try:
            res = subprocess.run(
                cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=300, text=True, encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        finally:
            if world_path.is_file():
                world_path.unlink(missing_ok=True)
        if res.returncode or simulator_has_errors(res.stdout):
            raise RuntimeError(f"Collector failed on Scene {scene_key} (code {res.returncode}):\n{res.stdout[-2000:]}")
        manifest["per_scene"][scene_key] = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"[Collect] Scene {scene_key} done: {manifest['per_scene'][scene_key]}", flush=True)

    yaml_text = (
        f"path: {dataset_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: person\n"
    )
    (dataset_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    draw_label_preview(dataset_dir)
    print(f"[Collect] Dataset ready at {dataset_dir}", flush=True)
    return dataset_dir


def main():
    parser = argparse.ArgumentParser(description="Collect Webots UAV+Rover YOLO victim dataset (Scenes A/B/C only).")
    parser.add_argument("--out", default="runs/yolo_victim_dataset")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--port", type=int, default=18590)
    args = parser.parse_args()
    collect(args.out, scale=args.scale, port=args.port)


if __name__ == "__main__":
    main()
