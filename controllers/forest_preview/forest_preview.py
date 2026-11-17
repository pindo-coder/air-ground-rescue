"""Isolated visual audit: static rover and reversible tree removal, never scored."""
from pathlib import Path
import json
import os
import cv2
import numpy as np
from controller import Supervisor

ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'runs'/os.environ['FOREST_RENDER_RUN'];out.mkdir(exist_ok=True)
robot=Supervisor()
camera=robot.getDevice('rgb');depth=robot.getDevice('depth')
stills={name:robot.getDevice('render_'+name) for name in ('overview','rescue_core','forest_detail','hills')}
for device in (camera,depth,*stills.values()):device.enable(100)
private=json.loads((ROOT/'evaluation/search/D.json').read_text())
tree_records=private['forest']['trees']
bush_records=private['forest'].get('bushes',[])
trees=[robot.getFromDef(record['def_name']) for record in tree_records+bush_records]
positions=[node.getField('translation').getSFVec3f() for node in trees]
h,w=camera.getHeight(),camera.getWidth()
with_forest=without_forest=None
for tick in range(1,21):
    if robot.step(100)==-1:break
    if tick==5:
        for name,sensor in stills.items():sensor.saveImage(str(out/f'{name}.png'),100)
        camera.saveImage(str(out/'rover_with_trees.png'),100)
        with_forest=np.array(depth.getRangeImage(),np.float32).reshape(h,w).copy()
        np.save(out/'depth_with_trees.npy',with_forest)
        for node,position in zip(trees,positions):
            node.getField('translation').setSFVec3f([*position[:2],position[2]-100])
    if tick==10:
        camera.saveImage(str(out/'rover_without_trees.png'),100)
        without_forest=np.array(depth.getRangeImage(),np.float32).reshape(h,w).copy()
        np.save(out/'depth_without_trees.npy',without_forest)
        for node,position in zip(trees,positions):node.getField('translation').setSFVec3f(position)
    if tick==15:
        camera.saveImage(str(out/'rover_restored.png'),100)
        visible=np.isfinite(with_forest)&(with_forest>.2)&(with_forest<12)
        blocked=visible & (np.nan_to_num(without_forest,nan=12,posinf=12)-with_forest>.25)
        center=blocked[h//4:3*h//4,w//4:3*w//4]
        cv2.imwrite(str(out/'tree_occlusion_mask.png'),blocked.astype(np.uint8)*255)
        panels=[]
        for filename,label in (('rover_without_trees.png','SAME POSE | TREES REMOVED FOR AUDIT'),
                               ('rover_with_trees.png','SAME POSE | TREES AND UNDERSTORY')):
            picture=cv2.imread(str(out/filename))
            panel=np.full((h+30,w,3),22,np.uint8);panel[30:]=picture
            cv2.putText(panel,label,(7,19),cv2.FONT_HERSHEY_SIMPLEX,.36,(240,240,240),1,cv2.LINE_AA)
            panels.append(panel)
        cv2.imwrite(str(out/'rover_occlusion_pair.png'),np.hstack(panels))
        heights=np.array(private['heights'])
        result=dict(status='FINISHED',tree_count=len(tree_records),bush_count=len(bush_records),
            terrain_extent_m=[84,60],terrain_elevation_range_m=[float(heights.min()),float(heights.max())],
            rover_height_m=.30,camera_size=[w,h],
            method='identical static rover pose, noise disabled only in audit; remove and restore tree nodes',
            pixels_with_tree_in_front_fraction=float(blocked.mean()),
            central_view_blocked_fraction=float(center.mean()),
            occlusion_pass=bool(blocked.mean()>=.45 and center.mean()>=.50),
            evaluation_only=True,not_a_navigation_or_detection_result=True)
        (out/'render_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('[Forest render]',json.dumps(result),flush=True)
    if tick==18:
        if os.environ.get('FOREST_KEEP_OPEN')=='1':
            robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
        else:robot.simulationQuit(0)
        break
