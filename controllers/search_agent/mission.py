"""Search/confirmation/delivery scheduling using only the online belief."""
import math
from itertools import permutations
import numpy as np
from perception import observation_views
from mapping import rotation


def delivery_guidance(track,pose,now):
    """Reacquire locally before a drop; map arrival alone is insufficient."""
    if track.latest_local is None or track.observation_pose is None or now-track.last_seen>2.:
        return None
    seen=track.observation_pose
    target=seen[:2]+rotation(seen[2]) @ track.latest_local
    delta=target-pose[:2]
    distance=float(np.linalg.norm(delta))
    ready=distance<=1.45
    point=pose[:2] if ready else target-1.25*delta/max(distance,.01)
    return point,math.atan2(delta[1],delta[0]),ready


METHODS = {
    "frontier": dict(active=False,schedule=False,person_search=False,covariance=True,independent=True),
    "weighted": dict(active=False,schedule=False,person_search=True,covariance=True,independent=True),
    "active_only": dict(active=True,schedule=False,person_search=False,covariance=True,independent=True),
    "schedule_only": dict(active=False,schedule=True,person_search=True,covariance=True,independent=True),
    "full": dict(active=True,schedule=True,person_search=True,covariance=True,independent=True),
    "no_prior": dict(active=True,schedule=True,person_search=False,covariance=True,independent=True),
    "no_covariance": dict(active=True,schedule=True,person_search=True,covariance=False,independent=True),
    "no_view_groups": dict(active=True,schedule=True,person_search=True,covariance=True,independent=False),
}


class MissionScheduler:
    def __init__(self, method, capacity=2, budget=80):
        self.settings=METHODS[method]
        self.method=method
        self.capacity=capacity
        self.cargo=capacity
        self.budget=budget
        self.spent=0.0
        self.previous_xy=None
        self.current=None
        self.current_since=0.0
        self.completed=[]
        self.blacklist=[]
        self.decisions=[]

    def consume(self, pose, dt):
        if self.previous_xy is not None:
            self.spent+=float(np.linalg.norm(pose[:2]-self.previous_xy))
        self.spent+=dt*.002
        self.previous_xy=pose[:2].copy()

    @staticmethod
    def delivery_pose(track,pose,grid):
        for radius in (1.20,1.45):
            options=[]
            for angle in np.linspace(-math.pi,math.pi,20,endpoint=False):
                point=track.mean+radius*np.array([math.cos(angle),math.sin(angle)])
                if grid.free(point,.50):
                    options.append((np.linalg.norm(point-pose[:2]),point))
            if options:
                _,point=min(options,key=lambda item:item[0])
                return dict(kind="deliver",point=point,
                            yaw=math.atan2(track.mean[1]-point[1],track.mean[0]-point[0]),
                            track=track.id,gain=0.0)
        return None

    def candidates(self, pose, grid, tracks, now):
        actions=[]
        for track in tracks:
            if not track.reported and now-track.last_seen<50:
                views=observation_views(track,pose,grid,self.settings["active"])
                if views:
                    actions.append(views[0])
            elif track.reported and not track.delivered and self.cargo:
                deliver=self.delivery_pose(track,pose,grid)
                if deliver:actions.append(deliver)
        for point,size,distance,revisit in grid.frontiers(pose)[:6]:
            delta=point-pose[:2]
            yaw=math.atan2(delta[1],delta[0])
            if any(np.linalg.norm(point-old)<.9 and now-expiry<35 for old,expiry in self.blacklist):
                continue
            if self.settings["person_search"]:
                gain=grid.search_gain(point,yaw,tracks)
            else:
                gain=min(.8,.015*size)
            if revisit<.7:
                gain*=.35
            actions.append(dict(kind="explore",point=point,yaw=yaw,track=None,gain=gain))
        if self.cargo<self.capacity:
            actions.append(dict(kind="reload",point=np.zeros(2),yaw=0.0,track=None,gain=0.0))
        return [action for action in actions if action["kind"] in ("reload","deliver") or not
                any(np.linalg.norm(action["point"]-old)<.8 and now-expiry<35 for old,expiry in self.blacklist)]

    def choose(self,pose,grid,tracks,now,force=False):
        urgent=[t for t in tracks if self.settings["schedule"] and t.reported and not t.delivered and
                t.deadline-now < 16+1.35*np.linalg.norm(t.mean-pose[:2])/.18]
        if urgent and self.current is not None and self.current["kind"] not in ("deliver","reload"):
            force=True
        if self.current is not None and not force:
            if now-self.current_since<4:
                return self.current
            if now-self.current_since>32:
                self.blacklist.append((self.current["point"].copy(),now))
                self.current=None
            else:
                return self.current
        actions=self.candidates(pose,grid,tracks,now)
        pending=[t for t in tracks if t.reported and not t.delivered]
        distance_home=float(np.linalg.norm(pose[:2]))
        if self.budget-self.spent < 1.25*distance_home+6:
            chosen=dict(kind="return",point=np.zeros(2),yaw=0.0,track=None,gain=0.0)
        elif urgent:
            requested=min(urgent,key=lambda t:t.deadline)
            chosen=self.delivery_pose(requested,pose,grid) if self.cargo else dict(
                kind="reload",point=np.zeros(2),yaw=0.0,track=None,gain=0.0)
            if chosen is None:
                chosen=dict(kind="scan",point=pose[:2].copy(),yaw=pose[2]+.5,track=None,gain=0.0)
        elif not actions:
            chosen=dict(kind="scan",point=pose[:2].copy(),yaw=pose[2]+math.pi/2,track=None,gain=0.0)
        elif not self.settings["schedule"]:
            deliveries=[a for a in actions if a["kind"]=="deliver"]
            confirms=[a for a in actions if a["kind"]=="confirm"]
            reloads=[a for a in actions if a["kind"]=="reload"]
            if pending and self.cargo==0 and reloads:
                chosen=reloads[0]
            elif deliveries:
                first=min(t.report_time for t in pending)
                ids={t.id for t in pending if t.report_time==first}
                chosen=next((a for a in deliveries if a["track"] in ids),deliveries[0])
            elif confirms:
                chosen=max(confirms,key=lambda a:a.get("view_value",0))
            else:
                chosen=(max(actions,key=lambda a:a["gain"]/(1+np.linalg.norm(a["point"]-pose[:2])))
                        if self.method=="weighted" else min(actions,key=lambda a:np.linalg.norm(a["point"]-pose[:2])))
        else:
            best=None
            pool=sorted(actions,key=lambda a:a["gain"],reverse=True)[:10]
            for length in (1,2):
                for sequence in permutations(pool,length):
                    if length==2 and sequence[0]["track"] is not None and sequence[0]["track"]==sequence[1]["track"]:
                        continue
                    xy=pose[:2].copy();elapsed=0.;cargo=self.cargo;travel=0.;gain=0.;serviced=set();valid=True
                    prior=[];completion={}
                    for action in sequence:
                        leg=1.35*float(np.linalg.norm(action["point"]-xy))
                        travel+=leg
                        elapsed+=leg/.18+(6 if action["kind"]=="reload" else 2)
                        if action["kind"]=="deliver":
                            if not cargo:valid=False;break
                            cargo-=1;serviced.add(action["track"]);completion[action["track"]]=now+elapsed
                        if action["kind"]=="reload":cargo=self.capacity
                        value=action["gain"]
                        if any(np.linalg.norm(action["point"]-old)<2.4 for old in prior):
                            value*=.35
                        gain+=value
                        prior.append(action["point"]);xy=action["point"]
                    if not valid or elapsed>52 or travel+1.25*np.linalg.norm(xy)+3>self.budget-self.spent:
                        continue
                    missed=0
                    for track in pending:
                        if track.id in serviced:
                            missed+=completion[track.id]>track.deadline
                            continue
                        earliest=now+elapsed+1.35*np.linalg.norm(track.mean-xy)/.18+2
                        if cargo==0:
                            earliest+=np.linalg.norm(xy)/.18+6+np.linalg.norm(track.mean)/.18
                        missed+=earliest>track.deadline
                    # Deadline feasibility precedes expected qualified reports.
                    rank=(-missed,gain,-elapsed,-travel)
                    if best is None or rank>best[0]:
                        best=rank,sequence[0]
            chosen=best[1] if best else min(actions,key=lambda a:np.linalg.norm(a["point"]-pose[:2]))
        self.current=chosen
        self.current_since=now
        self.decisions.append(dict(time=now,kind=chosen["kind"],track=chosen["track"],
                                   point=chosen["point"].tolist(),gain=chosen["gain"]))
        return chosen

    def finish(self,tracks,now):
        current=self.current
        if current is None:return None
        event=None
        if current["kind"]=="deliver" and self.cargo:
            track=next(t for t in tracks if t.id==current["track"])
            track.delivered=True
            self.cargo-=1
            event=dict(type="delivery",track_id=track.id,timestamp_s=now)
        elif current["kind"]=="reload":
            self.cargo=self.capacity
            event=dict(type="reload",timestamp_s=now)
        self.completed.append(current)
        self.current=None
        return event
