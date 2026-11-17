"""Offline audit of actual simulator motion and synchronized camera evidence."""
from pathlib import Path
import argparse
import json
import math
import cv2
import numpy as np
from run_search import simulator_has_errors


def records(path):
    if not path.is_file():return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def extent(points):
    data=np.asarray(points,float)
    return float(np.linalg.norm(np.ptp(data[:,:2],axis=0))) if len(data)>1 else 0.


def audit(folder):
    folder=Path(folder).resolve()
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['case']!='D':raise ValueError('Use a D scene run')
    aerial=records(folder/'aerial/frames.jsonl')
    ground=records(folder/'agent/frames.jsonl')
    motion=records(folder/'environment/motion.jsonl')
    fusion=records(folder/'agent/fusion.jsonl')
    summary_path=folder/'evaluation/summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    pairs=[]
    for frame in ground:
        if not aerial:break
        other=min(aerial,key=lambda a:abs(a['timestamp_s']-frame['timestamp_s']))
        delta=abs(other['timestamp_s']-frame['timestamp_s'])
        if delta<=.25 and (folder/frame['evidence']).is_file() and (folder/other['evidence']).is_file():
            pairs.append(dict(ground=frame,aerial=other,delta_s=delta))
    names=motion[0]['obstacles'].keys() if motion else []
    travel={name:extent([m['obstacles'][name] for m in motion]) for name in names}
    drone_extent=extent([m['drone'] for m in motion])
    mixed={t['id'] for frame in fusion for t in frame['tracks'] if {'rover','uav'}<=set(t['sources'])}
    output=folder/'validation';output.mkdir(exist_ok=True)
    # Choose a temporally matched pair with most aerial proposals for inspection.
    # This is a preview selection, not a performance metric or a ground-truth label.
    if pairs:
        pair=max(pairs,key=lambda p:len(p['aerial']['candidates']))
        panes=[]
        for key,title in (('ground','ROVER RGB'),('aerial','UAV RGB')):
            frame=pair[key]
            picture=cv2.imread(str(folder/frame['evidence']))
            if picture is None:raise ValueError(f"Unreadable image: {frame['evidence']}")
            picture=cv2.resize(picture,(640,480))
            canvas=np.full((540,640,3),22,np.uint8);canvas[60:]=picture
            cv2.putText(canvas,f"{title} | t = {frame['timestamp_s']:.1f} s",(16,26),
                        cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1,cv2.LINE_AA)
            cv2.putText(canvas,'Actual sensor image | no truth overlay',(16,49),
                        cv2.FONT_HERSHEY_SIMPLEX,.43,(175,190,205),1,cv2.LINE_AA)
            panes.append(canvas)
        cv2.imwrite(str(output/'paired_views.jpg'),np.hstack(panes))
        (output/'paired_views.json').write_text(json.dumps(pair,indent=2),encoding='utf-8')
    expected_dynamic=manifest.get('dynamic','on')=='on'
    motion_ok=(len(travel)==2 and all(value>.25 for value in travel.values())) if expected_dynamic else (
        len(travel)==2 and all(value<.03 for value in travel.values()))
    errors=(folder/'webots.log').read_text(encoding='utf-8',errors='replace') if (folder/'webots.log').is_file() else ''
    clean=not simulator_has_errors(errors)
    result=dict(detector=manifest['detector'],fusion=manifest.get('fusion'),dynamic=manifest.get('dynamic'),
        simulator_status=summary.get('status'),controller_errors_absent=clean,
        obstacle_xy_extent_m=travel,physical_motion_check=motion_ok,drone_xy_extent_m=drone_extent,
        ground_frames=len(ground),aerial_frames=len(aerial),synchronized_pairs=len(pairs),
        max_pair_delta_s=max((p['delta_s'] for p in pairs),default=None),
        accepted_aerial_packets=max((f['accepted_packets'] for f in fusion),default=0),
        rejected_aerial_packets=max((f['rejected_packets'] for f in fusion),default=0),
        cross_view_track_ids=sorted(mixed),valid_cross_view_reports=summary.get('valid_cross_view_reports',0),
        effective_reports=summary.get('effective_reports',0),delivered_people=summary.get('delivered_people',0),
        mission_success=summary.get('mission_success',False),
        infrastructure_pass=bool(clean and motion_ok and drone_extent>.25 and len(pairs)>=2 and
                                 max((f['accepted_packets'] for f in fusion),default=0)>0),
        note='Infrastructure pass is not detection accuracy, navigation success, or proof of algorithm improvement.')
    (output/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    args=parser.parse_args();audit(args.run)
