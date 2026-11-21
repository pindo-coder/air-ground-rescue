"""RGB-D candidates, correlated-evidence tracks and active observation poses."""
from dataclasses import dataclass, field
from pathlib import Path
import math
import os
import cv2
import numpy as np
from mapping import rotation, wrap
from scipy.optimize import linear_sum_assignment


def project_depth(uvd, width, height, roll=0., pitch=0.):
    """Aligned depth pixels -> gravity-aligned body coordinates, without yaw."""
    uvd=np.asarray(uvd,dtype=float).reshape(-1,3)
    u,v,d=uvd.T;focal=width/2
    points=np.column_stack((d,-(u-(width-1)/2)*d/focal,-(v-(height-1)/2)*d/focal))
    def ry(angle):
        c,s=math.cos(angle),math.sin(angle)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    cr,sr=math.cos(roll),math.sin(roll)
    rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    points=points @ ry(.08).T+np.array([.26,0,.30])
    return points @ (ry(pitch) @ rx).T


def unrotate_box(box, turns, width, height):
    """Undo clockwise 90-degree image rotations, using continuous box edges."""
    x1,y1,x2,y2=map(float,box)
    if turns==1:return [y1,height-x2,y2,height-x1]
    if turns==2:return [width-x2,height-y2,width-x1,height-y1]
    if turns==3:return [width-y2,x1,width-y1,x2]
    return [x1,y1,x2,y2]


def covariance_intersection(a, pa, b, pb):
    ia, ib = np.linalg.inv(pa), np.linalg.inv(pb)
    candidates = []
    for weight in np.linspace(.2, .8, 7):
        p = np.linalg.inv(weight*ia+(1-weight)*ib)
        candidates.append((np.linalg.slogdet(p)[1], weight, p))
    _, weight, p = min(candidates, key=lambda item: item[0])
    return p @ (weight*ia @ a+(1-weight)*ib @ b), p


class PersonDetector:
    def __init__(self, mode, root, bias_std=.18, max_depth=8., field_of_view=math.pi/2, view=None):
        if mode not in ("yolo","surrogate_rgbd"):
            raise ValueError(f"Unknown detector: {mode}")
        self.mode=mode
        self.bias_std=bias_std
        self.max_depth=max_depth
        self.field_of_view=field_of_view
        self.view = view or ("aerial" if max_depth >= 12.0 else "ground")
        self.model = None
        self.yolo_single_pass = False
        if mode == "yolo":
            from ultralytics import YOLO,settings
            settings.update({"sync":False})
            override = os.environ.get("UAV_YOLO_WEIGHTS")
            if override:
                weights = Path(override)
                if not weights.is_absolute():
                    weights = Path(root) / weights
            elif self.view == "aerial":
                finetuned = Path(root) / "models/yolo11n_victim.pt"
                weights = finetuned if finetuned.is_file() else Path(root) / "models/yolo11n.pt"
            else:
                weights = Path(root) / "models/yolo11n.pt"
            if not weights.is_file():
                raise FileNotFoundError(f"YOLO weights not found: {weights} (run tools/fetch_search_model.py or tools/train_yolo_victim.py)")
            self.weights_path = weights
            self.model = YOLO(str(weights))
            # Fine-tuned single-class aerial victim model is trained with 360-deg rotation augmentation.
            self.yolo_single_pass = len(getattr(self.model, "names", {})) == 1

    def detect(self, bgr, depth):
        if self.model is not None:
            boxes=[];scores=[]
            conf_thr = .50 if self.yolo_single_pass else .35
            rotations = (0,) if self.yolo_single_pass else range(4)
            for turns in rotations:
                view=bgr if turns==0 else np.ascontiguousarray(np.rot90(bgr,-turns))
                output=self.model.predict(view,conf=conf_thr,classes=[0],imgsz=640,device="cpu",verbose=False)[0]
                for box,score in zip(output.boxes.xyxy.cpu().numpy(),output.boxes.conf.cpu().numpy()):
                    x1,y1,x2,y2=unrotate_box(box,turns,bgr.shape[1],bgr.shape[0])
                    boxes.append([x1,y1,x2-x1,y2-y1]);scores.append(float(score))
            if not boxes:return []
            keep=np.asarray(cv2.dnn.NMSBoxes(boxes,scores,conf_thr,.45)).reshape(-1)
            return [([int(boxes[i][0]),int(boxes[i][1]),int(boxes[i][0]+boxes[i][2]),
                      int(boxes[i][1]+boxes[i][3])],scores[i],None) for i in keep]
        # Controlled pipeline test. This colour/size filter is not a general person detector.
        hue, saturation, value = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
        blue, green, red = cv2.split(bgr.astype(np.float32))
        colour = (hue>88)&(hue<126)&(blue>1.30*red)&(blue>1.20*green)
        mask = (colour & (saturation>85) & (value>30) & (value<225) &
                np.isfinite(depth) & (depth>.3) & (depth<self.max_depth-1)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        choices = []
        for label in range(1, count):
            x,y,w,h,area = stats[label]
            if area<20 or area>.16*mask.size or min(w,h)<2:
                continue
            pixels = labels == label
            z = float(np.median(depth[pixels]))
            dimension = max(w,h)*z/(bgr.shape[1]/(2*math.tan(self.field_of_view/2)))
            if .35<dimension<.78:
                score = .65+.15*min(1,area/90)+.1*max(0,1-abs(dimension-.48)/.5)
                if x<4 or y<4 or x+w>bgr.shape[1]-4 or y+h>bgr.shape[0]-4:
                    score=min(score,.65)
                choices.append(([int(x),int(y),int(x+w),int(y+h)],min(.91,score),pixels))
        return choices

    def localize(self, bgr, depth, pose, pose_cov, attitude=(0.,0.)):
        detections = []
        height, width = depth.shape
        focal = width/2
        for bbox, score, mask in self.detect(bgr, depth):
            x1,y1,x2,y2 = bbox
            x1,x2,y1,y2 = max(0,x1),min(width,x2),max(0,y1),min(height,y2)
            if mask is None:
                mask = np.zeros(depth.shape, bool)
                dx,dy = max(1,(x2-x1)//5),max(1,(y2-y1)//5)
                mask[y1+dy:y2-dy,x1+dx:x2-dx] = True
            use = mask & np.isfinite(depth) & (depth>.3) & (depth<8)
            if np.count_nonzero(use)<5:
                continue
            rows,cols = np.nonzero(use)
            z = float(np.median(depth[use]))
            if np.median(np.abs(depth[use]-z))>.45:
                continue
            pixel=np.array([float(np.median(cols)),float(np.median(rows)),z])
            local=project_depth(pixel,width,height,*attitude)[0,:2]
            mean = local @ rotation(pose[2]).T+pose[:2]
            jac=np.empty((2,3))
            for axis,step in enumerate((1.,1.,.01)):
                changed=pixel.copy();changed[axis]+=step
                jac[:,axis]=(project_depth(changed,width,height,*attitude)[0,:2]-local)/step
            r=jac @ np.diag([2.**2,2.**2,(.035+.005*z*z)**2]) @ jac.T
            world_ray = rotation(pose[2]) @ local
            j = np.array([[1,0,-world_ray[1]],[0,1,world_ray[0]]])
            transform = rotation(pose[2])
            cov = transform @ r @ transform.T+j @ pose_cov @ j.T+np.eye(2)*self.bias_std**2
            detections.append(dict(mean=mean,cov=cov,score=score,bbox=bbox,depth=z,local_xy=local))
        return detections


@dataclass
class Track:
    id: int
    mean: np.ndarray
    covariance: np.ndarray
    confidence: float = .45
    groups: list = field(default_factory=list)
    reported: bool = False
    delivered: bool = False
    last_seen: float = 0
    report_time: float | None = None
    deadline: float | None = None
    latest_bbox: list | None = None
    latest_local: np.ndarray | None = None
    observation_pose: np.ndarray | None = None
    local_last_seen: float | None = None
    sources: set = field(default_factory=set)
    group_origins: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


class TrackManager:
    def __init__(self, single_frame=False, independent_groups=True, score_min=.70, radius_95=.80,
                 require_cross_source=False):
        self.tracks = []
        self.single_frame = single_frame
        self.independent_groups = independent_groups
        self.score_min=score_min
        self.radius_95=radius_95
        self.require_cross_source=require_cross_source

    @staticmethod
    def independent_camera(track, xy, height, source):
        if source=='rover' and track.sources<=set(['rover']):
            return TrackManager.independent(track,xy)
        origin=np.r_[xy,height]
        vector=origin-np.r_[track.mean,0.]
        for old in track.group_origins:
            other=old-np.r_[track.mean,0.]
            denominator=np.linalg.norm(vector)*np.linalg.norm(other)
            angle=math.acos(np.clip(vector @ other/max(denominator,1e-9),-1,1))
            if np.linalg.norm(origin-old)<=.50 or angle<=math.radians(12):return False
        return True

    @staticmethod
    def independent(track, xy):
        if not track.groups:
            return True
        vector = np.asarray(xy)-track.mean
        angle = math.atan2(vector[1],vector[0])
        return all(np.linalg.norm(np.asarray(xy)-group)>.50 and
                   abs(wrap(angle-math.atan2(group[1]-track.mean[1],group[0]-track.mean[0])))>math.radians(12)
                   for group in track.groups)

    def update(self, detections, pose, now):
        assignments={}
        if self.tracks and detections:
            costs=np.full((len(self.tracks),len(detections)),1e6)
            for i,track in enumerate(self.tracks):
                for j,detection in enumerate(detections):
                    delta=detection["mean"]-track.mean
                    mahalanobis=float(delta @ np.linalg.solve(track.covariance+detection["cov"]+np.eye(2)*.20**2,delta))
                    if mahalanobis<9.21 and np.linalg.norm(delta)<.90:costs[i,j]=mahalanobis
            rows,cols=linear_sum_assignment(costs)
            assignments={int(j):self.tracks[i] for i,j in zip(rows,cols) if costs[i,j]<1e5}
        for index,detection in enumerate(detections):
            track=assignments.get(index)
            if track is None:
                # Suppress duplicate frame proposals instead of creating sibling tracks.
                if any(np.linalg.norm(detection["mean"]-t.mean)<.65 for t in self.tracks):
                    continue
                track = Track(len(self.tracks)+1,detection["mean"].copy(),detection["cov"].copy())
                self.tracks.append(track)
            source=detection.get('source','rover')
            observer=np.asarray(detection.get('observer_xy',pose[:2]))
            observer_z=detection.get('observer_z',.30)
            independent=self.independent_camera(track,observer,observer_z,source)
            track.mean,track.covariance = covariance_intersection(
                track.mean,track.covariance,detection["mean"],detection["cov"])
            track.last_seen,track.latest_bbox = now,detection["bbox"]
            if detection.get("local_xy") is not None:
                track.latest_local=np.asarray(detection["local_xy"]).copy()
                track.observation_pose=pose.copy()
                track.local_last_seen=now
            if detection.get('evidence'):track.evidence[source]=detection['evidence']
            track.sources.add(source)
            if detection["score"]>=self.score_min and (not self.independent_groups or independent):
                track.groups.append(observer.copy())
                track.group_origins.append(np.r_[observer,observer_z])
                odds = 4*track.confidence/max(1e-6,1-track.confidence)
                track.confidence = min(.999,odds/(1+odds))

    def reportable(self, track, covariance_gate=True):
        radius = math.sqrt(5.991*float(np.linalg.eigvalsh(track.covariance).max()))
        count = 1 if self.single_frame else 2
        threshold = .70 if self.single_frame else .90
        duplicate=any(t.reported and np.linalg.norm(t.mean-track.mean)<1.0 for t in self.tracks if t.id!=track.id)
        source_ok=not self.require_cross_source or {'rover','uav'}<=track.sources
        return (source_ok and not duplicate and not track.reported and len(track.groups)>=count and track.confidence>=threshold and
                (radius<=self.radius_95 or not covariance_gate))


def observation_views(track, pose, grid, active=True):
    result = []
    angles = np.linspace(-math.pi,math.pi,16,endpoint=False) if active else [
        math.atan2(pose[1]-track.mean[1],pose[0]-track.mean[0])+.45]
    for radius in (1.25,1.9,2.6):
        for angle in angles:
            point = track.mean+radius*np.array([math.cos(angle),math.sin(angle)])
            if not grid.free(point,.50):
                continue
            yaw = math.atan2(track.mean[1]-point[1],track.mean[0]-point[0])
            offsets = [(0,0),(-.25,0),(.25,0),(0,-.25),(0,.25)]
            visibility = sum(grid.visible(point,track.mean+offset,endpoint_margin=.95) for offset in offsets)/len(offsets)
            if visibility<.6:
                continue
            new_group = TrackManager.independent(track,point)
            if track.groups:
                # Reserve position/yaw margins for imperfect arrival, rather than
                # predicting independent evidence at the exact geometric threshold.
                bearing=math.atan2(point[1]-track.mean[1],point[0]-track.mean[0])
                new_group=all(np.linalg.norm(point-group)>.75 and
                    abs(wrap(bearing-math.atan2(group[1]-track.mean[1],group[0]-track.mean[0])))>math.radians(18)
                    for group in track.groups)
                if not new_group:continue
            r = np.diag([(.035+.005*radius**2)**2,(radius*.02)**2])+np.eye(2)*.18**2
            _,future = covariance_intersection(track.mean,track.covariance,track.mean,r)
            information = max(0,float(np.linalg.slogdet(track.covariance)[1]-np.linalg.slogdet(future)[1]))
            probability = visibility*math.exp(-radius**2/45)*(1.0 if new_group else .08)
            value = probability*(.6+information)/(.8+np.linalg.norm(point-pose[:2])/.23)
            result.append(dict(kind="confirm",point=point,yaw=yaw,track=track.id,
                               gain=probability*track.confidence,view_value=value,new_group=new_group))
    return sorted(result,key=lambda item:item["view_value"],reverse=True)[:4]
