"""Online LiDAR mapping and odometry. This module has no simulator truth access."""

import math
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


class LaserOdometry:
    def __init__(self):
        self.pose = np.zeros(3)
        self.cov = np.diag([.04**2, .04**2, .015**2])
        self.previous_wheels = None
        self.reference = None
        self.initial_heading = None

    def update(self, wheels, scan, heading=None):
        wheels = np.asarray(wheels)
        if self.previous_wheels is None:
            self.previous_wheels = wheels.copy()
        dl, dr = (wheels - self.previous_wheels) * .09
        self.previous_wheels = wheels.copy()
        travel, turn = (dl + dr) / 2, (dr - dl) / .41
        midpoint = self.pose[2] + turn / 2
        self.pose += [travel * math.cos(midpoint), travel * math.sin(midpoint), turn]
        self.pose[2] = wrap(self.pose[2])
        if heading is not None:
            if self.initial_heading is None:
                self.initial_heading = heading
            self.pose[2] = wrap(heading-self.initial_heading)
        self.cov += np.diag([.002 + abs(travel)*.01, .002 + abs(travel)*.01, .0003 + abs(turn)*.003])**2
        if self.reference is not None and len(scan) >= 18:
            predicted = scan @ rotation(self.pose[2]).T + self.pose[:2]
            initial = predicted.copy()
            delta_yaw = 0.0
            residual = 1.0
            tree = cKDTree(self.reference)
            for _ in range(5):
                distances, indices = tree.query(predicted)
                keep = distances < .38
                if np.count_nonzero(keep) < 16:
                    break
                a, b = predicted[keep], self.reference[indices[keep]]
                center_a, center_b = a.mean(0), b.mean(0)
                u, _, vt = np.linalg.svd((a-center_a).T @ (b-center_b))
                r = vt.T @ u.T
                if np.linalg.det(r) < 0:
                    vt[-1] *= -1
                    r = vt.T @ u.T
                translation = center_b - r @ center_a
                predicted = predicted @ r.T + translation
                delta_yaw += math.atan2(r[1,0], r[0,0])
                residual = float(np.median(distances[keep]))
            shift = predicted.mean(0) - initial.mean(0)
            if residual < .16 and np.linalg.norm(shift) < .08 and abs(delta_yaw) < .10:
                self.pose[:2] += .20*shift
                if heading is None:
                    self.pose[2] = wrap(self.pose[2] + delta_yaw)
                floor = np.diag([.055**2, .055**2, .018**2])
                self.cov = .92*self.cov + .08*floor
        if len(scan) > 12:
            self.reference = scan @ rotation(self.pose[2]).T + self.pose[:2]
        return self.pose.copy(), self.cov.copy()


class OccupancyMap:
    def __init__(self, resolution=.18, side=66):
        self.resolution = resolution
        self.n = round(side/resolution)
        self.origin = np.array([-side/2, -side/2])
        self.logodds = np.zeros((self.n,self.n), np.float32)
        self.seen = np.zeros_like(self.logodds, bool)
        self.clearance = np.full_like(self.logodds, 20)
        self.last_frontiers = []
        self.visited = []

    def cell(self, point):
        return tuple(np.floor((np.asarray(point)-self.origin)/self.resolution).astype(int)[::-1])

    def xy(self, cell):
        return self.origin + (np.array(cell[::-1])+.5)*self.resolution

    def inside(self, cell):
        return 0<=cell[0]<self.n and 0<=cell[1]<self.n

    def update(self, pose, points, max_range=8):
        start=np.asarray(self.cell(pose[:2]))
        for p in points[::2]:
            length=np.linalg.norm(p)
            if length<.30:
                continue
            end=np.asarray(self.cell(p @ rotation(pose[2]).T + pose[:2]))
            count=max(abs(end-start))+1
            cells=np.rint(np.linspace(start,end,int(count))).astype(int)
            valid=(cells[:,0]>=0)&(cells[:,0]<self.n)&(cells[:,1]>=0)&(cells[:,1]<self.n)
            cells=cells[valid]
            if len(cells)<2:
                continue
            self.seen[cells[:,0],cells[:,1]]=True
            np.add.at(self.logodds,(cells[:-1,0],cells[:-1,1]),-.26)
            if length<max_range-.1:
                self.logodds[tuple(cells[-1])]+=.85
        self.logodds=np.clip(self.logodds,-4,4)
        j,i=start
        self.logodds[max(0,j-1):j+2,max(0,i-1):i+2]=-3
        self.seen[max(0,j-1):j+2,max(0,i-1):i+2]=True
        self.clearance=distance_transform_edt(self.logodds<.7)*self.resolution
        if not self.visited or np.linalg.norm(np.array(self.visited[-1])-pose[:2])>.45:
            self.visited.append(pose[:2].tolist())

    def free(self, point, margin=.36):
        c=self.cell(point)
        return self.inside(c) and self.seen[c] and self.logodds[c]<.5 and self.clearance[c]>=margin

    def visible(self, a, b, endpoint_margin=.55):
        delta=np.asarray(b)-a
        distance=np.linalg.norm(delta)
        if distance<endpoint_margin:
            return True
        for t in np.arange(.18,distance-endpoint_margin,.18):
            c=self.cell(np.asarray(a)+delta*(t/distance))
            if not self.inside(c) or self.logodds[c]>.7:
                return False
        return True

    def frontiers(self, pose):
        unknown=(~self.seen).astype(np.uint8)
        neighbor=cv2.dilate(unknown,np.ones((3,3),np.uint8))>0
        mask=neighbor & self.seen & (self.logodds<-.2) & (self.clearance>.40)
        count, labels, stats, centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        choices=[]
        for label in range(1,count):
            if stats[label,cv2.CC_STAT_AREA]<3:
                continue
            pixels=np.argwhere(labels==label)
            center=np.array([centers[label,1],centers[label,0]])
            point=self.xy(pixels[np.argmin(np.sum((pixels-center)**2,axis=1))])
            d=np.linalg.norm(point-pose[:2])
            if d<.8 or d>12:
                continue
            revisit=min((np.linalg.norm(point-v) for v in self.visited[:-3]),default=9)
            choices.append((point, int(len(pixels)), d, revisit))
        self.last_frontiers=sorted(choices,key=lambda a:a[2])[:10]
        return self.last_frontiers

    def search_gain(self, point, yaw, tracks):
        # Uniform unknown-person density, augmented only by online image hypotheses.
        total=0.0
        c0=self.cell(point)
        r=round(5/self.resolution)
        for dj in range(-r,r+1,4):
            for di in range(-r,r+1,4):
                c=(c0[0]+dj,c0[1]+di)
                if not self.inside(c) or self.seen[c]:
                    continue
                q=self.xy(c)
                d=np.linalg.norm(q-point)
                angle=wrap(math.atan2(q[1]-point[1],q[0]-point[0])-yaw)
                if d<.5 or d>5 or abs(angle)>.68 or not self.visible(point,q):
                    continue
                density=.005
                for track in tracks:
                    if not track.reported:
                        density += .025*track.confidence*math.exp(-np.sum((q-track.mean)**2)/8)
                pd=math.exp(-d*d/36)
                total += density*pd*(4*self.resolution)**2
        return 1-math.exp(-total)

    def save(self, folder):
        image=np.full(self.logodds.shape,125,np.uint8)
        image[self.seen & (self.logodds<.5)]=235
        image[self.logodds>.7]=25
        cv2.imwrite(str(folder/"occupancy.png"),np.flipud(image))


def local_control(pose, target, target_yaw, laser, grid, previous=(0,0),arrival_tolerance=None):
    distance=np.linalg.norm(np.asarray(target)-pose[:2])
    tolerance=2*grid.resolution+.04 if arrival_tolerance is None else arrival_tolerance
    if distance<tolerance:
        error=wrap(target_yaw-pose[2])
        return 0.0,max(-.65,min(.65,2.0*error)),abs(error)<.12
    # Only a rolling 2 m local window is used; there is no global path planner.
    goal_heading=math.atan2(target[1]-pose[1],target[0]-pose[0])
    alternatives=[]
    for offset in np.linspace(-1.5,1.5,21):
        heading=goal_heading+offset
        reach=min(2.,distance)
        safe=True
        for d in np.arange(.45,reach+.01,.18):
            point=pose[:2]+d*np.array([math.cos(heading),math.sin(heading)])
            if not grid.free(point,.43):
                safe=False;break
        if safe:
            point=pose[:2]+reach*np.array([math.cos(heading),math.sin(heading)])
            alternatives.append((np.linalg.norm(point-target)+.15*abs(wrap(heading-pose[2])),heading))
    heading=min(alternatives)[1] if alternatives else goal_heading
    error=wrap(heading-pose[2])
    # Acceleration-limited velocity rollouts with a stopping-distance check.
    best=None
    for v in (0.0,.08,.14,.18):
        if abs(v-previous[0])>.12:continue
        footprint=.34 if v==0 else .43
        for w in np.linspace(-.8,.8,13):
            if abs(w-previous[1])>.45:continue
            x,y,theta=pose.copy()
            safe=True
            for _ in range(9):
                x+=v*math.cos(theta)*.2; y+=v*math.sin(theta)*.2;theta+=w*.2
                if not grid.inside(grid.cell((x,y))) or grid.clearance[grid.cell((x,y))]<footprint:
                    safe=False;break
                local=np.array([x,y])-pose[:2]
                local=local @ rotation(pose[2])
                if len(laser) and np.min(np.linalg.norm(laser-local,axis=1))<footprint+v*v/(2*.5):
                    safe=False;break
            if not safe:continue
            score=-2.0*abs(wrap(heading-theta))+.7*v-.15*abs(w)
            if best is None or score>best[0]:
                best=score,v,float(w)
    if best is None:
        return 0.0,0.0,False
    return best[1],best[2],False
