"""Separate referee. Reads truth for metrics, never sends direction/pose to policy."""
from pathlib import Path
import csv
import json
import math
import os
import sys
import time
from collections import deque

from controller import Supervisor

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "evaluation"))
from metrics import height_at, target_in_view, look_at_rotation

supervisor = Supervisor()
period = 100
scenario_id = supervisor.getCustomData()
meta = json.loads((root / "evaluation" / "scenarios" / f"{scenario_id}.json").read_text(encoding="utf-8"))
public = json.loads((root / "config" / "public" / f"{scenario_id}_task.json").read_text(encoding="utf-8"))
rover = supervisor.getFromDef("ROVER")
rover.enableContactPointsTracking(period, True)
data_field = rover.getField("customData")
validation = os.environ.get("VISION_VALIDATE") == "1"
inspection_view = supervisor.getFromDef("SCENE_VIEW") if validation else None
original_view = (inspection_view.getField("position").getSFVec3f(), inspection_view.getField("orientation").getSFRotation()) if inspection_view else None
started = None
status = "PREVIEW"
previous = None
path_length = 0.0
collisions = 0
was_contact = False
stopped_seconds = 0.0
cut_seen_at = None
first_goal_time = None
ended_at = None
recent_heights = deque()
distance_to_goal = None
out = None
file = None
writer = None
snapshots = set()
last_summary = -1

print(f"[Evaluator] {scenario_id}: truth is isolated from the policy observation API.", flush=True)


def write_summary(now, position, visible):
    summary = {"scenario_id": scenario_id, "status": status, "surface_mode": meta["surface_mode"], "navigation_policy_supplied": started is not None, "simulation_seconds": round(now,3), "task_seconds": round((ended_at if ended_at is not None else now)-started,3) if started is not None else None, "path_length_m": round(path_length,4), "collision_events": collisions, "target_distance_m": round(distance_to_goal,3), "target_in_view_approx": visible, "visibility_method": "horizontal frustum + terrain + obstacle boxes + shrub leaf spheres; grass and tree crowns omitted", "goal_claim_required": True, "first_washout_approach_task_s": round(cut_seen_at-started,2) if cut_seen_at is not None and started is not None else None, "position_ground_truth": position, "camera_resolution": [640,480], "navigation_input": "stereo_RGB_and_initial_direction_sector_only"}
    temporary = out / "summary.tmp"
    temporary.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    try:
        temporary.replace(out / "summary.json")
    except PermissionError:
        pass


try:
    while supervisor.step(period) != -1:
        now = supervisor.getTime()
        try:
            telemetry = json.loads(data_field.getSFString() or "{}")
        except json.JSONDecodeError:
            telemetry = {}
        if not telemetry.get("run_id"):
            continue
        if out is None:
            out = root / "runs" / telemetry["run_id"] / scenario_id / "evaluation"
            out.mkdir(parents=True,exist_ok=True)
            file = (out / "trajectory.csv").open("w",newline="",encoding="utf-8")
            writer = csv.writer(file)
            writer.writerow(["sim_time_s","x_m","y_m","z_m","yaw_rad","path_length_m","contact_events","target_distance_m","status"])
        position = rover.getPosition()
        recent_heights.append((now,position[2]))
        while recent_heights and now-recent_heights[0][0]>1.0:
            recent_heights.popleft()
        rotation = rover.getOrientation()
        yaw = math.atan2(rotation[3],rotation[0])
        if previous is not None and now > 1:
            path_length += math.dist(previous[:2],position[:2])
        previous = position
        contact = now>1 and any(cp.point[2] > height_at(meta,cp.point[0],cp.point[1])+.055 for cp in rover.getContactPoints(True))
        if contact and not was_contact:
            collisions += 1
        was_contact = contact
        distance_to_goal = math.dist(position[:2],meta["goal"])
        if telemetry.get("active") and started is None:
            started = now
            status = "RUNNING"
        speed = math.hypot(*rover.getVelocity()[:2])
        stopped_seconds = stopped_seconds + period/1000 if speed<.03 else 0
        visible = target_in_view(meta,position,rotation,public["camera"]) if telemetry.get("claim_arrival") else None
        if status == "RUNNING":
            if rotation[8] < .5:
                status = "ROLLOVER"
            elif meta["cut"] and math.dist(position[:2],meta["cut"])<4 and max(z for _,z in recent_heights)-position[2]>.35:
                status = "DROPOFF"
            elif not (-20.6<position[0]<20.6 and -14.6<position[1]<14.6):
                status = "LEFT_TERRAIN"
            elif now-started >= meta["episode_timeout_s"]:
                status = "TIMEOUT"
            elif telemetry.get("claim_arrival") and distance_to_goal<=meta["success_distance_m"] and stopped_seconds>=meta["required_stop_duration_s"] and visible:
                status = "SUCCESS"
                first_goal_time = now
            if status != "RUNNING":
                ended_at = now
                telemetry["episode_end"] = True
                data_field.setSFString(json.dumps(telemetry))
                print(f"[Evaluator] {status}; path={path_length:.2f} m, contacts={collisions}",flush=True)
        if meta["cut"] and started is not None:
            distance_to_cut = math.dist(position[:2],meta["cut"])
            if distance_to_cut<2.6 and cut_seen_at is None:
                cut_seen_at=now
        writer.writerow([f"{now:.3f}",*position,f"{yaw:.5f}",f"{path_length:.5f}",collisions,f"{distance_to_goal:.4f}",status])
        file.flush()
        if now-last_summary>=1:
            write_summary(now,position,visible)
            supervisor.setLabel(0,f"VISION RESCUE | {scenario_id} | {status}",.025,.025,.055,0xF3F5E8,0,"Arial")
            supervisor.setLabel(1,"Initial coarse direction only | Stereo RGB | No navigation policy supplied" if status=="PREVIEW" else f"{status} | contacts {collisions} | path {path_length:.1f} m",.025,.085,.032,0xFFDF9A,0,"Arial")
            last_summary=now
        if validation and now>=3 and 3 not in snapshots:
            supervisor.exportImage(str(out / "scene_overview.png"),100)
            snapshots.add(3)
            view=meta["inspection_views"]["casualty"]
            inspection_view.getField("position").setSFVec3f(view["eye"])
            inspection_view.getField("orientation").setSFRotation(look_at_rotation(view["eye"],view["target"]))
        if validation and now>=4 and 4 not in snapshots:
            supervisor.exportImage(str(out / "casualty_detail.png"),100)
            snapshots.add(4)
            view=meta["inspection_views"]["obstacles"]
            inspection_view.getField("position").setSFVec3f(view["eye"])
            inspection_view.getField("orientation").setSFRotation(look_at_rotation(view["eye"],view["target"]))
        if validation and now>=5 and 5 not in snapshots:
            supervisor.exportImage(str(out / "obstacles_detail.png"),100)
            snapshots.add(5)
            inspection_view.getField("position").setSFVec3f(original_view[0])
            inspection_view.getField("orientation").setSFRotation(original_view[1])
        if validation and now>=12:
            write_summary(now,position,visible)
            check={"scene_loaded":True,"simulation_seconds":now,"rover_clearance_m":position[2]-height_at(meta,*position[:2]),"upright":rotation[8]>.95,"stationary_speed_m_s":speed,"public_task_has_no_world_coordinates":not any(k in public for k in ("goal","roads","heights","spawn")),"status":status}
            (out / "smoke_test.json").write_text(json.dumps(check,indent=2),encoding="utf-8")
            file.flush()
            supervisor.simulationQuit(0 if check["upright"] and speed<.03 else 2)
finally:
    if file:
        file.close()
