"""Independent Webots mission evaluation. No truth is sent to the agent."""
from pathlib import Path
import csv
import json
import math
import os
import sys
import numpy as np
from controller import Supervisor

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/"evaluation"))
from metrics import height_at
from search_metrics import match_reports, physical_delivery_ok, world_xy

robot=Supervisor()
period=100
scenario=robot.getCustomData()
private=json.loads((root/"evaluation/search"/f"{scenario}.json").read_text())
truth=dict(json.loads((root/"evaluation/scenarios"/f"{private['base_scenario']}.json").read_text()),**private)
rover=robot.getFromDef("ROVER")
rover.enableContactPointsTracking(period,True)
field=rover.getField("customData")
duration=float(os.environ.get("RESCUE_DURATION",truth["duration_s"]))
run_id=os.environ.get("RESCUE_RUN_ID")
out=None
targets=truth["victims"]
true_points=np.array([target["xy"] for target in targets])
reports=[];deliveries=[];events_seen=set();contacts=0;in_contact=False
method_name=None;detector_name=None
length=0.;previous=None;stationary=0.;path=[];ended=False
mission_config=json.loads((root/"config/search/common.json").read_text())
physical_cargo=mission_config["cargo_capacity"]
unsafe_state=False
world_min,world_max=truth.get('bounds_xy',[[-20.6,-14.6],[20.6,14.6]])
time_file=None


def global_xy(local):
    return world_xy(local,truth["spawn"],truth["spawn_yaw"])


def matching(report_list=None):
    records=reports if report_list is None else report_list
    return match_reports(records,targets,truth["spawn"],truth["spawn_yaw"],truth["report_tolerance_m"])


def summary(now):
    matches=matching()
    effective=[reports[i]["timestamp_s"] for i in matches]
    errors=[float(np.linalg.norm(global_xy(reports[i]["xy"])-true_points[j])) for i,j in matches.items()]
    valid_delivery=[d for d in deliveries if d["valid"]]
    delivered_ids={d["victim_id"] for d in valid_delivery}
    elapsed=[d["timestamp_s"]-d["report_timestamp_s"] for d in valid_delivery]
    inside=[]
    yaw=truth["spawn_yaw"];c,s=math.cos(yaw),math.sin(yaw)
    transform=np.array([[c,-s],[s,c]])
    for i,j in matches.items():
        error=global_xy(reports[i]["xy"])-true_points[j]
        covariance=transform @ np.array(reports[i]["covariance"]) @ transform.T
        inside.append(float(error @ np.linalg.solve(covariance,error))<=5.991)
    return dict(
        scenario=scenario,method=method_name,detector=detector_name,
        simulation_seconds=round(now,2),first_effective_report_s=min(effective,default=None),
        first_report_censored=not bool(effective),effective_reports=len(matches),
        total_reports=len(reports),false_reports=len(reports)-len(matches),
        miss_rate=(len(targets)-len(matches))/max(1,len(targets)),
        false_discovery_rate=(len(reports)-len(matches))/len(reports) if reports else None,
        localization_rmse_m=math.sqrt(np.mean(np.square(errors))) if errors else None,
        localization_errors_m=errors,delivered_people=len(delivered_ids),
        reported_95pct_ellipse_coverage=float(np.mean(inside)) if inside else None,
        effective_report_counts={str(t):len(matching([r for r in reports if r["timestamp_s"]<=t])) for t in sorted(set([t for t in (30,60,120,300,600) if t<=min(now,duration)]+[min(now,duration)]))},
        on_time_delivered_people=len({d["victim_id"] for d in valid_delivery if d["deadline_met"]}),
        material_arrival_times_s=[d["timestamp_s"] for d in valid_delivery],
        report_to_delivery_times_s=elapsed,
        mission_success=(len(matches)==len(targets) and
                         len({d["victim_id"] for d in valid_delivery if d["deadline_met"]})==len(targets) and contacts==0 and not unsafe_state),
        collisions=contacts,path_length_m=round(length,3),truth_targets=len(targets),
        fusion_enabled=message.get('fusion_enabled',False),aerial_packets=message.get('aerial_packets',0),
        cross_view_tracks=message.get('cross_view_tracks',0),
        valid_cross_view_reports=sum({'rover','uav'}<=set(reports[i].get('sources',[])) for i in matches),
        evaluation_only=True,status="FINISHED" if now>=duration else "RUNNING",unsafe_state=unsafe_state,
        limitations=["prototype detector calibration pending","2D local odometry without loop closure",
                     "delivery is a stationary drop event, not a simulated manipulator"],
    )


try:
    while robot.step(period)!=-1:
        now=robot.getTime()
        try:message=json.loads(field.getSFString())
        except json.JSONDecodeError:message={}
        if out is None:
            run_id=run_id or message.get("run_id")
            if not run_id:continue
            out=root/"runs"/run_id/"evaluation"
            out.mkdir(parents=True,exist_ok=True)
            time_file=(out/"trajectory.csv").open("w",newline="",encoding="utf-8")
            writer=csv.writer(time_file)
            writer.writerow(["time","x","y","z","estimated_x","estimated_y","contacts","path_length"])
        method_name=message.get("method",method_name)
        detector_name=message.get("detector",detector_name)
        position=rover.getPosition()
        unsafe_state=unsafe_state or rover.getOrientation()[8]<.5 or not (
            world_min[0]<position[0]<world_max[0] and world_min[1]<position[1]<world_max[1])
        if previous is not None:length+=math.dist(previous[:2],position[:2])
        previous=position
        speed=math.hypot(*rover.getVelocity()[:2])
        stationary=stationary+.1 if speed<.03 else 0.
        contact=now>1 and any(p.point[2]>height_at(truth,*p.point[:2])+.055 for p in rover.getContactPoints(True))
        if contact and not in_contact:contacts+=1
        in_contact=contact
        estimate=message.get("pose_estimate",[None,None])
        global_estimate=global_xy(estimate[:2]) if estimate[0] is not None else [None,None]
        writer.writerow([now,*position,*global_estimate,contacts,length]);time_file.flush()
        path.append(position)
        for event in message.get("events",[]):
            key=(event["type"],event.get("track_id"),event["timestamp_s"])
            if key in events_seen:continue
            events_seen.add(key)
            if event["type"]=="report":
                reports.append(event.copy())
            elif event["type"]=="delivery":
                assignment=matching()
                report_index=next((i for i,r in enumerate(reports) if r["track_id"]==event["track_id"]),None)
                target=assignment.get(report_index)
                distance=np.linalg.norm(np.asarray(position[:2])-true_points[target]) if target is not None else float("inf")
                already={d["victim_id"] for d in deliveries if d["valid"]}
                valid=physical_delivery_ok(target is not None,stationary,distance,physical_cargo,truth["delivery_tolerance_m"])
                if target is not None and targets[target]["id"] in already:valid=False
                physical_cargo=max(0,physical_cargo-1)
                result=dict(event,valid=bool(valid),victim_id=targets[target]["id"] if target is not None else None,
                            actual_drop_xy=position[:2],report_timestamp_s=reports[report_index]["timestamp_s"] if report_index is not None else None,
                            deadline_met=bool(report_index is not None and now<=reports[report_index]["timestamp_s"]+truth.get("delivery_deadline_after_report_s",120)))
                deliveries.append(result)
                x,y,z=position
                robot.getRoot().getField("children").importMFNodeFromString(-1,
                    f'Pose {{ translation {x} {y} {z+.09} children [ Shape {{ appearance PBRAppearance {{ baseColor .95 .52 .06 roughness .8 metalness 0 }} geometry Box {{ size .16 .13 .09 }} }} ] }}')
                # Delivery correctness is logged only in this private evaluator.
            elif event["type"]=="reload":
                if stationary>=5 and np.linalg.norm(np.asarray(position[:2])-truth["spawn"])<.65:
                    physical_cargo=mission_config["cargo_capacity"]
        if round(now*10)%10==0 and not ended:
            result=summary(now)
            (out/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
            (out/"events.json").write_text(json.dumps(dict(reports=reports,deliveries=deliveries),indent=2),encoding="utf-8")
            robot.setLabel(0,f"RESCUE RESEARCH | {message.get('method','starting')} | reports {result['effective_reports']} | drops {result['delivered_people']}",
                           .02,.025,.045,0xFFFFFF,0,"Arial")
            robot.setLabel(1,f"{message.get('mode','starting')} | cargo {message.get('cargo','?')} | {message.get('detector','starting')}",
                           .02,.08,.03,0xFFE0A0,0,"Arial")
        if now>=duration and not ended:
            ended=True
            field.setSFString(json.dumps(dict(episode_end=True)))
            robot.exportImage(str(out/"final_scene.png"),100)
            result=summary(now)
            (out/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        if now>=duration+1:
            time_file.flush()
            if os.environ.get("RESCUE_AUTO_QUIT")=="1":
                robot.simulationQuit(0)
                break
finally:
    if time_file:time_file.close()
