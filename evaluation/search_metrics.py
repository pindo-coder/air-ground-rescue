"""Pure evaluator helpers; never imported by the Robot controller."""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment


def world_xy(local,spawn,yaw):
    value=np.asarray(local,dtype=float)
    if value.shape!=(2,) or not np.isfinite(value).all():return np.full(2,np.inf)
    c,s=math.cos(yaw),math.sin(yaw)
    return np.asarray(spawn)+np.array([[c,-s],[s,c]]) @ value


def match_reports(reports,targets,spawn,yaw,tolerance):
    if not reports or not targets:return {}
    predictions=np.array([world_xy(record.get("xy",[]),spawn,yaw) for record in reports])
    points=np.array([target["xy"] for target in targets])
    distances=np.linalg.norm(predictions[:,None,:]-points[None,:,:],axis=2)
    # Maximize cardinality first, then prefer earlier reports over closer duplicates.
    time_order=np.argsort(np.argsort([r["timestamp_s"] for r in reports],kind="stable"),kind="stable")
    priority=time_order*(len(targets)+1)*(tolerance+1)
    costs=np.where(distances<=tolerance,distances+priority[:,None],1e12)
    rows,cols=linear_sum_assignment(costs)
    return {int(i):int(j) for i,j in zip(rows,cols) if costs[i,j]<1e11}


def physical_delivery_ok(matched,stationary_s,distance_m,cargo,tolerance=1.8):
    return bool(matched and stationary_s>=1.5 and distance_m<=tolerance and cargo>0)
