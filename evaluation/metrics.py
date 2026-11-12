"""Privileged evaluation helpers. Never imported by the vision agent."""
import math


def look_at_rotation(eye, target):
    """Webots ENU view: local +X is forward and local +Z is up."""
    def unit(v):
        length=math.sqrt(sum(q*q for q in v))
        return [q/length for q in v]
    def cross(a,b):
        return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    forward=unit([b-a for a,b in zip(eye,target)])
    left=unit(cross([0,0,1],forward))
    up=cross(forward,left)
    m=[[forward[i],left[i],up[i]] for i in range(3)]
    angle=math.acos(max(-1,min(1,(m[0][0]+m[1][1]+m[2][2]-1)/2)))
    if angle<1e-8:
        return [0,0,1,0]
    axis=unit([m[2][1]-m[1][2],m[0][2]-m[2][0],m[1][0]-m[0][1]])
    return [*axis,angle]


def height_at(meta, x, y):
    grid = meta["heights"]
    u = (x - meta["origin"][0]) / meta["spacing"]
    v = (y - meta["origin"][1]) / meta["spacing"]
    i = min(len(grid[0])-2, max(0, math.floor(u)))
    j = min(len(grid)-2, max(0, math.floor(v)))
    a, b = min(1, max(0, u-i)), min(1, max(0, v-j))
    return (1-b)*((1-a)*grid[j][i]+a*grid[j][i+1])+b*((1-a)*grid[j+1][i]+a*grid[j+1][i+1])


def target_in_view(meta, position, orientation, camera):
    # A geometry-only visibility approximation, not an object-recognition score.
    yaw = math.atan2(orientation[3], orientation[0])
    gx, gy = meta["goal"]
    delta = (math.atan2(gy-position[1],gx-position[0])-yaw+math.pi)%(2*math.pi)-math.pi
    if abs(delta) > camera["fov_rad"]/2:
        return False
    start = (position[0]+.26*math.cos(yaw), position[1]+.26*math.sin(yaw), position[2]+camera["height_m"])
    end = (gx, gy, height_at(meta,gx,gy)+.20)
    distance = math.dist(start[:2],end[:2])
    for k in range(1, max(2, int(distance/.10))):
        t = k/max(2,int(distance/.10))
        x,y,z = [a+t*(b-a) for a,b in zip(start,end)]
        if height_at(meta,x,y) > z-.015:
            return False
        for foliage in meta.get("foliage_spheres", []):
            if math.dist((x,y,z),foliage["center"]) < foliage["radius"]:
                return False
        for obstacle in meta["obstacles"]:
            ox,oy = obstacle["xy"]
            c,s = math.cos(obstacle["yaw"]),math.sin(obstacle["yaw"])
            lx,ly = c*(x-ox)+s*(y-oy),-s*(x-ox)+c*(y-oy)
            if abs(lx)<obstacle["size"][0]/2 and abs(ly)<obstacle["size"][1]/2 and z < height_at(meta,ox,oy)+obstacle["size"][2]:
                return False
    return True
