"""Webots agent: only on-board sensors and the public mission configuration."""
from pathlib import Path
import json
import math
import os
import sys
import time
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from controller import Robot

from mapping import LaserOdometry, OccupancyMap, local_control
from perception import PersonDetector, TrackManager, project_depth
from mapping import rotation
from mission import METHODS, MissionScheduler, delivery_guidance

root=Path(__file__).resolve().parents[2]
configuration=json.loads((root/"config/search/common.json").read_text(encoding="utf-8"))
robot=Robot()
period=100
scenario=json.loads(robot.getCustomData())["search_case"]
method=os.environ.get("RESCUE_METHOD","full")
detector_mode=os.environ.get("RESCUE_DETECTOR",configuration["detector"])
score_min=configuration["detection_score_min"][detector_mode]
run_id=os.environ.get("RESCUE_RUN_ID",time.strftime("%Y%m%d_%H%M%S")+f"_{os.getpid()}")
out=root/"runs"/run_id/"agent"
out.mkdir(parents=True,exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR",str(out/"yolo_config"))
detector=PersonDetector(detector_mode,root,bias_std=configuration["position_bias_std_m"])
odometry=LaserOdometry()
grid=OccupancyMap()
public_task=json.loads((root/"config/search/public"/f"{scenario}.json").read_text(encoding="utf-8"))
boundary=np.array([grid.cell(p)[::-1] for p in public_task["search_boundary_xy"]],np.int32)
allowed=np.zeros(grid.logodds.shape,np.uint8)
cv2.fillPoly(allowed,[boundary],1)
grid.logodds[allowed==0]=4
grid.seen[allowed==0]=True
tracks=TrackManager(independent_groups=METHODS[method]["independent"],score_min=score_min,
                    radius_95=configuration["report_radius_95_m"])
scheduler=MissionScheduler(method,configuration["cargo_capacity"],configuration["distance_budget_m"])
motors=[robot.getDevice(n) for n in ("left_motor","right_motor")]
wheels=[robot.getDevice(n) for n in ("left_encoder","right_encoder")]
for motor in motors:
    motor.setPosition(float("inf"));motor.setVelocity(0.)
for sensor in wheels:sensor.enable(period)
camera=robot.getDevice("rgb")
depth_camera=robot.getDevice("depth")
lidar=robot.getDevice("lidar")
imu=robot.getDevice("attitude")
for sensor in (camera,depth_camera,lidar,imu):sensor.enable(period)
height,width=camera.getHeight(),camera.getWidth()
events=[]
reports=[]
last_control=(0.,0.)
last_time=0.
last_mode=None
service_start=None
service_key=None
depth_obstacles=np.empty((0,2))
tick=0
report_file=(out/"reports.jsonl").open("w",encoding="utf-8")
log=(out/"state.jsonl").open("w",encoding="utf-8")
detection_log=(out/"detections.jsonl").open("w",encoding="utf-8")
print(f"[Search Agent] {method}; {detector_mode}; LiDAR + RGB-D; local frame starts at deployment.",flush=True)
try:
    while robot.step(period)!=-1:
        now=robot.getTime();tick+=1
        try:message=json.loads(robot.getCustomData())
        except json.JSONDecodeError:message={}
        if message.get("episode_end"):
            break
        ranges=np.asarray(lidar.getLayerRangeImage(1),np.float64)
        ranges=np.nan_to_num(ranges,nan=8.,posinf=8.,neginf=8.)
        ranges=np.clip(ranges,.1,8.)
        angles=np.linspace(math.pi,-math.pi,len(ranges),endpoint=False)
        laser=np.column_stack((ranges*np.cos(angles),ranges*np.sin(angles)))
        roll,pitch,heading=imu.getRollPitchYaw()
        pose,pose_cov=odometry.update([sensor.getValue() for sensor in wheels],laser[ranges<7.9][::3],heading)
        scheduler.consume(pose,now-last_time);last_time=now
        depth=np.asarray(depth_camera.getRangeImage(),np.float32).reshape(height,width).copy()
        rgb=np.frombuffer(camera.getImage(),np.uint8).reshape(height,width,4)[:,:,:3].copy()
        if tick%4==0:
            grid.update(pose,laser)
            # Depth adds low obstacles below the horizontal LiDAR plane.
            ys,xs=np.nonzero(np.isfinite(depth)&(depth>.35)&(depth<3.5))
            ys,xs=ys[::14],xs[::14]
            points=project_depth(np.column_stack((xs,ys,depth[ys,xs])),width,height,roll,pitch)
            horizontal=np.linalg.norm(points[:,:2],axis=1)
            raised=(points[:,2]>.065+.025*horizontal)&(points[:,2]<1.5)
            drop=(points[:,2]<-.24)&(horizontal<1.75)
            depth_obstacles=points[raised|drop,:2]
            for local in depth_obstacles:
                point=local @ rotation(pose[2]).T+pose[:2]
                cell=grid.cell(point)
                if grid.inside(cell):
                    grid.logodds[cell]=min(4,grid.logodds[cell]+.9)
                    grid.seen[cell]=True
            grid.logodds[allowed==0]=4
            grid.clearance=distance_transform_edt(grid.logodds<.7)*grid.resolution
        local_obstacles=np.vstack((laser[ranges<7.9],depth_obstacles))
        if tick%10==0:
            detected=[d for d in detector.localize(rgb,depth,pose,pose_cov,(roll,pitch)) if d["score"]>=score_min]
            for detection in detected:
                c=grid.cell(detection["mean"])
                if grid.inside(c):
                    grid.logodds[c]=4
                    grid.seen[c]=True
            grid.clearance=distance_transform_edt(grid.logodds<.7)*grid.resolution
            prior=len(tracks.tracks)
            tracks.update(detected,pose,now)
            detection_log.write(json.dumps(dict(t=now,candidates=[dict(xy=d["mean"].tolist(),
                                      score=d["score"],depth=d["depth"]) for d in detected]))+"\n")
            detection_log.flush()
            if len(tracks.tracks)>prior:
                scheduler.current=None
            for track in tracks.tracks:
                if tracks.reportable(track,scheduler.settings["covariance"]):
                    track.reported=True
                    track.report_time=now
                    track.deadline=now+public_task.get("delivery_deadline_after_report_s",configuration["delivery_deadline_after_report_s"])
                    evidence=f"report_{track.id}_{int(now)}.jpg"
                    cv2.imwrite(str(out/evidence),rgb)
                    report=dict(type="report",track_id=track.id,timestamp_s=now,
                                xy=track.mean.tolist(),covariance=track.covariance.tolist(),
                                score=track.confidence,view_groups=len(track.groups),evidence=evidence)
                    reports.append(report);events.append(report)
                    report_file.write(json.dumps(report)+"\n");report_file.flush()
                    scheduler.current=None
                    print(f"[Search Agent] report {track.id}, groups {len(track.groups)}, xy={track.mean}",flush=True)
        action=scheduler.choose(pose,grid,tracks.tracks,now)
        mode=action["kind"]
        key=(mode,action["track"],tuple(np.round(action["point"],2)))
        if key!=service_key:
            service_start=None;service_key=key
        if mode!=last_mode:
            print(f"[Search Agent] {mode} at {now:.1f}s, tracks={len(tracks.tracks)}, cargo={scheduler.cargo}",flush=True)
            last_mode=mode
        terminal_yaw=pose[2] if mode in ("deliver","reload","return") else action["yaw"]
        tolerance=.15 if mode=="confirm" else None
        v,w,arrived=local_control(pose,action["point"],terminal_yaw,local_obstacles,grid,last_control,tolerance)
        if mode=="deliver":
            target=next(t for t in tracks.tracks if t.id==action["track"])
            guidance=delivery_guidance(target,pose,now)
            if guidance is None:
                v,w,_=local_control(pose,action["point"],action["yaw"],local_obstacles,grid,last_control)
                arrived=False
            else:
                point,yaw,ready=guidance
                v,w,_=local_control(pose,point,yaw,local_obstacles,grid,last_control,.05)
                arrived=ready
                if ready:v=w=0.
        if arrived:
            service_start=now if service_start is None else service_start
            v=w=0.
            duration=6 if mode=="reload" else 2
            if now-service_start>=duration:
                event=scheduler.finish(tracks.tracks,now)
                if event:
                    event["estimated_xy"]=pose[:2].tolist()
                    events.append(event)
                if mode=="return":
                    break
                service_start=None
        else:
            service_start=None
        roll,pitch,_=imu.getRollPitchYaw()
        if abs(roll)>.35 or abs(pitch)>.25:
            v=w=0.
        left=(v-.41*w/2)/.09;right=(v+.41*w/2)/.09
        motors[0].setVelocity(left);motors[1].setVelocity(right)
        last_control=v,w
        payload=dict(run_id=run_id,scenario=scenario,method=method,detector=detector_mode,
                     pose_estimate=pose.tolist(),cargo=scheduler.cargo,events=events,mode=mode)
        robot.setCustomData(json.dumps(payload))
        if tick%10==0:
            log.write(json.dumps(dict(t=now,pose=pose.tolist(),mode=mode,
                                      targets=len(tracks.tracks),reports=len(reports),cargo=scheduler.cargo,
                                      command=[v,w],goal=action["point"].tolist()))+"\n")
            log.flush()
        if tick in (20,80):
            camera.saveImage(str(out/f"rgb_{int(now):02d}s.png"),100)
        if now>float(os.environ.get("RESCUE_DURATION",configuration["max_duration_s"]))+1:
            break
finally:
    for motor in motors:motor.setVelocity(0.)
    grid.save(out)
    (out/"decisions.json").write_text(json.dumps(scheduler.decisions,indent=2),encoding="utf-8")
    report_file.close();log.close();detection_log.close()
