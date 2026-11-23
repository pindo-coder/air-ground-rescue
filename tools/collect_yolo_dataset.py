"""Collect a zero-leakage, color- & position-randomized Webots aerial YOLO dataset from Scenes A, B (train) and C (val).

Scene D is strictly held out and NEVER used during dataset generation or training.
"""
from pathlib import Path
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from build_search_worlds import build, replace_node
from posture_casualty import ALL_POSTURES, NON_SUPINE_POSTURES, build_posture_casualty_vrml
from run_search import ROOT, executable_path, simulator_has_errors
from scene_elements import RescueElements

sys.path.insert(0, str(ROOT / "evaluation"))
from metrics import height_at


EXTRA_CASUALTIES = {
    "A": [(-11.5, -6.5), (-3.0, 8.0), (5.5, -5.5), (11.0, 6.2), (-13.5, 5.5)],
    "B": [(-12.0, -5.0), (-4.5, 7.5), (4.0, -6.8), (12.2, 4.5), (-2.0, -2.5)],
    "C": [(-11.0, -7.2), (-5.5, 6.8), (6.0, -4.5), (10.5, 7.0), (1.0, 1.5)],
}


def _tag_clothing_appearances(text, tag_index):
    """Attach unique DEF names to shirt and pants PBRAppearance nodes for live color randomization."""
    shirt_def = f"SHIRT_APP_{tag_index}"
    pants_def = f"PANTS_APP_{tag_index}"
    out_lines = []
    shirt_tagged = False
    pants_tagged = False
    for line in text.splitlines():
        if not shirt_tagged and "_shirt.obj" in line and "appearance PBRAppearance {" in line:
            line = line.replace("appearance PBRAppearance {", f"appearance DEF {shirt_def} PBRAppearance {{", 1)
            shirt_tagged = True
        elif not pants_tagged and "_pants.obj" in line and "appearance PBRAppearance {" in line:
            line = line.replace("appearance PBRAppearance {", f"appearance DEF {pants_def} PBRAppearance {{", 1)
            pants_tagged = True
        out_lines.append(line)
    return "\n".join(out_lines), (shirt_def if shirt_tagged else None), (pants_def if pants_tagged else None)


def build_collector_world(world_path, scene_key, multi_posture=False):
    """Build a temporary Webots world based on Scene A, B, or C with taggable multicolored casualties."""
    assert scene_key in ("A", "B", "C"), "Scene D must remain strictly held out!"
    source = ROOT / "worlds" / f"search_{scene_key}.wbt"
    if not source.is_file():
        source = build(scene_key)
    scene = source.read_text(encoding="utf-8")
    meta = json.loads((ROOT / "evaluation/scenarios" / f"{scene_key}.json").read_text(encoding="utf-8"))

    scene = replace_node(scene, r"\bDEF\s+ROVER\s+ResearchRover", "")
    scene = replace_node(scene, r"\bDEF\s+SEARCH_REFEREE\s+Robot", "")

    # Tag the existing scene casualty's clothing appearances as index 0 (supine).
    scene, s0, p0 = _tag_clothing_appearances(scene, 0)
    victim_defs = ["CASUALTY"]
    victim_postures = ["supine"]
    shirt_defs = [s0] if s0 else []
    pants_defs = [p0] if p0 else []

    for idx, (vx, vy) in enumerate(EXTRA_CASUALTIES[scene_key], 1):
        vkey = ("A", "B", "C")[(idx + ord(scene_key)) % 3]
        posture = NON_SUPINE_POSTURES[(idx - 1) % len(NON_SUPINE_POSTURES)] if multi_posture else "supine"
        def_name = f"TRAIN_VICTIM_{idx}"
        s_def = f"SHIRT_APP_{idx}"
        p_def = f"PANTS_APP_{idx}"
        if multi_posture:
            node = build_posture_casualty_vrml(
                def_name, f"train_victim_{idx}_{posture}",
                [vx, vy], 0.3 * idx, lambda x, y: height_at(meta, x, y),
                posture=posture, mesh_key=vkey,
                shirt_def=s_def, pants_def=p_def,
            )
        else:
            elements = RescueElements(dict(key=vkey, seed=500 + idx * 17 + ord(scene_key), goal=[vx, vy]),
                                      lambda x, y: height_at(meta, x, y))
            elements.casualty()
            node = elements.nodes[0]
            node = node.replace("DEF CASUALTY ", f"DEF {def_name} ", 1)
            node = node.replace('name "casualty"', f'name "train_victim_{idx}"', 1)
            node, s_def, p_def = _tag_clothing_appearances(node, idx)
        scene += "\n" + node
        victim_defs.append(def_name)
        victim_postures.append(posture)
        if s_def:
            shirt_defs.append(s_def)
        if p_def:
            pants_defs.append(p_def)

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
      width 320
      height 240
      fieldOfView 1.5707963
      near 0.1
      far 30
    }
    RangeFinder {
      name "rover_depth"
      width 320
      height 240
      fieldOfView 1.5707963
      minRange 0.25
      maxRange 12
    }
  ]
}
'''
    scene += "\n" + collector_node
    world_path.write_text(scene, encoding="utf-8")
    return victim_defs, victim_postures, shirt_defs, pants_defs


def add_grayscale_train_copies(dataset_dir, pos_count=55, neg_count=20, seed=71):
    """Add pure 3-channel grayscale copies to train/ to enforce color-agnostic silhouette learning."""
    rng = random.Random(seed)
    img_dir = dataset_dir / "images/train"
    lbl_dir = dataset_dir / "labels/train"
    pos_imgs = sorted(img_dir.glob("*_pos_*.jpg"))
    neg_imgs = sorted(img_dir.glob("*_neg_*.jpg"))
    rng.shuffle(pos_imgs)
    rng.shuffle(neg_imgs)
    added = 0
    for src in pos_imgs[:pos_count] + neg_imgs[:neg_count]:
        bgr = cv2.imread(str(src))
        if bgr is None:
            continue
        gray = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        dst_stem = f"{src.stem}_gray"
        cv2.imwrite(str(img_dir / f"{dst_stem}.jpg"), gray, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        src_lbl = lbl_dir / f"{src.stem}.txt"
        if src_lbl.is_file():
            shutil.copy2(src_lbl, lbl_dir / f"{dst_stem}.txt")
        added += 1
    return added


def add_fog_train_copies(dataset_dir, pos_count=90, neg_count=30, seed=108):
    """Add Koschmieder atmospheric haze/fog copies to train/ (Scenes A+B only) for weather-degraded robustness."""
    rng = random.Random(seed)
    img_dir = dataset_dir / "images/train"
    lbl_dir = dataset_dir / "labels/train"
    pos_imgs = sorted(p for p in img_dir.glob("*_pos_*.jpg") if "_gray" not in p.stem and "_fog" not in p.stem)
    neg_imgs = sorted(p for p in img_dir.glob("*_neg_*.jpg") if "_gray" not in p.stem and "_fog" not in p.stem)
    rng.shuffle(pos_imgs)
    rng.shuffle(neg_imgs)
    added = 0
    for src in pos_imgs[:pos_count] + neg_imgs[:neg_count]:
        bgr = cv2.imread(str(src))
        if bgr is None:
            continue
        t = rng.uniform(0.36, 0.66)
        base_veil = rng.uniform(135.0, 192.0)
        veil = np.array([
            base_veil + rng.uniform(-6.0, 8.0),
            base_veil + rng.uniform(-5.0, 6.0),
            base_veil + rng.uniform(-6.0, 6.0),
        ], dtype=np.float32).reshape(1, 1, 3)
        fogged = np.clip(bgr.astype(np.float32) * t + veil * (1.0 - t), 0.0, 255.0).astype(np.uint8)
        dst_stem = f"{src.stem}_fog"
        cv2.imwrite(str(img_dir / f"{dst_stem}.jpg"), fogged, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        src_lbl = lbl_dir / f"{src.stem}.txt"
        if src_lbl.is_file():
            shutil.copy2(src_lbl, lbl_dir / f"{dst_stem}.txt")
        added += 1
    return added


def draw_label_preview(dataset_dir):
    """Draw a 4x4 contact sheet of labeled multi-color aerial samples for visual audit."""
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
    rng = random.Random(42)
    rng.shuffle(samples)
    picks = samples[:16]
    while len(picks) < 16:
        picks.append(samples[len(picks) % len(samples)])

    tiles = []
    for split, img_path, lbl_path in picks[:16]:
        img = cv2.imread(str(img_path))
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
        cv2.putText(canvas, f"{split}:{img_path.stem[:22]}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)

    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 16, 4)])
    cv2.imwrite(str(dataset_dir / "preview_labels.jpg"), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def collect(out_dir, scale=1.0, port=18590, multi_posture=False):
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
    # 100% dedicated to 640x480 aerial frames with randomized victim positions & 16 clothing colors.
    plan = [
        ("A", "train", int(round(170 * scale)), int(round(55 * scale)), 0, 0, 1011),
        ("B", "train", int(round(170 * scale)), int(round(55 * scale)), 0, 0, 2022),
        ("C", "val",   int(round(105 * scale)), int(round(35 * scale)), 0, 0, 3033),
    ]

    manifest = dict(
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        held_out_test_scenes=["D"],
        train_scenes=["A", "B"],
        val_scenes=["C"],
        randomized_victim_positions=True,
        randomized_clothing_palette=16,
        multi_posture=bool(multi_posture),
        postures=list(ALL_POSTURES) if multi_posture else ["supine"],
        per_scene={},
    )

    for idx, (scene_key, split, uav_pos, uav_neg, rov_pos, rov_neg, seed) in enumerate(plan):
        world_path = ROOT / "worlds" / f"_tmp_yolo_collect_{scene_key}.wbt"
        wbproj_path = ROOT / "worlds" / f"._tmp_yolo_collect_{scene_key}.wbproj"
        victim_defs, victim_postures, shirt_defs, pants_defs = build_collector_world(
            world_path, scene_key, multi_posture=multi_posture
        )
        report_path = dataset_dir / f"report_{scene_key}.json"
        spec_path = dataset_dir / f"spec_{scene_key}.json"
        spec = dict(
            scene=scene_key,
            prefix=f"scene{scene_key}",
            seed=seed,
            victim_defs=victim_defs,
            victim_postures=victim_postures,
            shirt_app_defs=shirt_defs,
            pants_app_defs=pants_defs,
            refresh_every=4,
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
        print(f"[Collect] Scene {scene_key} ({split}) -> {uav_pos} pos + {uav_neg} hard-neg UAV (random pos + 16 colors)...",
              flush=True)
        try:
            res = subprocess.run(
                cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=300, text=True, encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        finally:
            world_path.unlink(missing_ok=True)
            wbproj_path.unlink(missing_ok=True)
        if res.returncode or simulator_has_errors(res.stdout):
            raise RuntimeError(f"Collector failed on Scene {scene_key} (code {res.returncode}):\n{res.stdout[-2000:]}")
        manifest["per_scene"][scene_key] = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"[Collect] Scene {scene_key} done: {manifest['per_scene'][scene_key]}", flush=True)

    gray_added = add_grayscale_train_copies(dataset_dir, pos_count=int(round(55 * scale)), neg_count=int(round(20 * scale)))
    manifest["grayscale_train_copies"] = gray_added
    print(f"[Collect] Added {gray_added} pure-grayscale training copies for color-agnostic learning.", flush=True)
    fog_added = add_fog_train_copies(dataset_dir, pos_count=int(round(75 * scale)), neg_count=int(round(25 * scale)))
    manifest["fog_train_copies"] = fog_added
    print(f"[Collect] Added {fog_added} Koschmieder atmospheric-haze training copies for fog robustness.", flush=True)

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
    parser = argparse.ArgumentParser(description="Collect color- & position-randomized Webots aerial YOLO dataset (A/B/C only).")
    parser.add_argument("--out", default="runs/yolo_victim_dataset_v2")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--port", type=int, default=18590)
    parser.add_argument("--multi-posture", action="store_true",
                        help="Include all 6 disaster postures (supine/prone/side/curled/seated/half_buried).")
    args = parser.parse_args()
    collect(args.out, scale=args.scale, port=args.port, multi_posture=args.multi_posture)


if __name__ == "__main__":
    main()
