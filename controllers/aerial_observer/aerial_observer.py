"""Actual airborne RGB-D/GPS/IMU acquisition; no scene-tree access."""
from pathlib import Path
import json
import os
import sys
import time
import cv2
import numpy as np
from controller import Robot

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'controllers/search_agent'))
from perception import PersonDetector
from multiview import aerial_candidates

robot = Robot()
run_id = os.environ.get('RESCUE_RUN_ID',time.strftime('manual_D_%Y%m%d_%H%M%S'))
out = ROOT/'runs'/run_id/'aerial';out.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR',str(out/'yolo_config'))
settings = json.loads((ROOT/'config/search/public/D.json').read_text())['aerial']
detector = PersonDetector(os.environ.get('RESCUE_DETECTOR','yolo'),ROOT,
                          max_depth=24,field_of_view=settings['camera_fov_rad'])
camera = robot.getDevice('aerial_rgb');depth_camera = robot.getDevice('aerial_depth')
gps = robot.getDevice('aerial_gps');imu = robot.getDevice('aerial_attitude')
emitter = robot.getDevice('aerial_radio')
for sensor in (camera,depth_camera,gps,imu):sensor.enable(200)
sequence = 0;tick = 0
with (out/'frames.jsonl').open('w',encoding='utf-8') as log:
    while robot.step(200) != -1:
        tick += 1
        if tick % 5:continue
        now = robot.getTime()
        if now>float(os.environ.get('RESCUE_DURATION',600)):break
        h,w = camera.getHeight(),camera.getWidth()
        depth = np.asarray(depth_camera.getRangeImage(),np.float32).reshape(h,w)
        rgb = np.frombuffer(camera.getImage(),np.uint8).reshape(h,w,4)[:,:,:3].copy()
        origin,attitude = gps.getValues(),imu.getRollPitchYaw()
        if not np.isfinite(origin+attitude).all():continue
        candidates = aerial_candidates(detector,rgb,depth,origin,attitude,settings['camera_fov_rad'])
        name = f'aerial_{sequence:06d}.jpg';cv2.imwrite(str(out/name),rgb)
        packet = dict(version=1,source='uav_rgbd',sequence=sequence,timestamp_s=now,
                      camera_origin_enu=origin,attitude_rpy=attitude,evidence=f'aerial/{name}',candidates=candidates)
        emitter.send(json.dumps(packet).encode('utf-8'))
        log.write(json.dumps(packet)+'\n');log.flush()
        if sequence==0:print('[Aerial observer] RGB-D/GPS/IMU frames and radio stream active.',flush=True)
        if candidates:print(f'[Aerial observer] {len(candidates)} candidate(s) at {now:.1f}s.',flush=True)
        sequence += 1
