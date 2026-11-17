"""Expand physical D terrain, reanchor props, and keep mountain visuals outside it."""
import re
import numpy as np
from build_scenarios import terrain,parameters
from upgrade_casualties import node_span


def expanded_terrain(meta,config):
    result=dict(meta)
    width,height=config['extent_m'];spacing=meta['spacing']
    xs=np.arange(-width/2,width/2+spacing/2,spacing)
    ys=np.arange(-height/2,height/2+spacing/2,spacing)
    x,y=np.meshgrid(xs,ys)
    heights=terrain(x,y,parameters('B',meta['seed']))
    # Preserve the already committed core, then add explicit rolling landforms.
    original=np.asarray(meta['heights'])
    col=round((meta['origin'][0]+width/2)/spacing)
    row=round((meta['origin'][1]+height/2)/spacing)
    heights[row:row+original.shape[0],col:col+original.shape[1]]=original
    for hx,hy,amplitude,sx,sy in config['terrain_hills']:
        heights+=amplitude*np.exp(-((x-hx)/sx)**2-((y-hy)/sy)**2)
    result.update(origin=[-width/2,-height/2],heights=heights.tolist(),
                  bounds_xy=[[-width/2+.5,-height/2+.5],[width/2-.5,height/2-.5]])
    return result


def reanchor_props(scene,before,after,height_at):
    # Named root solids in the source are ground props; nested human pieces are Pose nodes.
    for match in reversed(list(re.finditer(r'\bDEF\s+(\w+)\s+Solid\s*\{',scene))):
        name=match[1]
        if name=='TERRAIN':continue
        a,b=node_span(scene,rf'\bDEF\s+{name}\s+Solid')
        node=scene[a:b];prefix=node.split('children',1)[0]
        position=re.search(r'\btranslation\s+([-+.\deE]+)\s+([-+.\deE]+)\s+([-+.\deE]+)',prefix)
        if position is None:continue
        x,y,z=map(float,position.groups())
        z+=height_at(after,x,y)-height_at(before,x,y)
        updated=node[:position.start()]+f'translation {x:.5f} {y:.5f} {z:.5f}'+node[position.end():]
        scene=scene[:a]+updated+scene[b:]
    return scene


def outer_mountains(meta,height_at):
    # Coarse distant mesh starts outside the full playable 84x60 m height grid.
    half_x,half_y=abs(meta['origin'][0]),abs(meta['origin'][1])
    xs=[-165,-125,-90,-65,-half_x,*np.arange(-half_x+3,half_x,3),half_x,65,90,125,165]
    ys=[-145,-110,-80,-50,-half_y,*np.arange(-half_y+3,half_y,3),half_y,50,80,110,145]
    points=[]
    for y in ys:
        for x in xs:
            dx=max(abs(x)-half_x,0);dy=max(abs(y)-half_y,0)
            fade=min(1,np.hypot(dx,dy)/20)
            peaks=(17*np.exp(-((x+63)/38)**2-((y-58)/32)**2)+
                   21*np.exp(-((x-64)/34)**2-((y-72)/38)**2)+
                   12*np.exp(-((x-78)/40)**2-((y+53)/32)**2))
            points.append([x,y,height_at(meta,x,y)+fade*peaks])
    faces=[];n=len(xs)
    for j in range(len(ys)-1):
        for i in range(n-1):
            if xs[i]>=-half_x and xs[i+1]<=half_x and ys[j]>=-half_y and ys[j+1]<=half_y:continue
            a=j*n+i;faces.append(f'{a} {a+1} {a+1+n} {a+n} -1')
    coordinates=', '.join(' '.join(f'{q:.4f}' for q in p) for p in points)
    uv=', '.join(f'{(p[0]+half_x)/(2*half_x):.5f} {(p[1]+half_y)/(2*half_y):.5f}' for p in points)
    return f'''DEF DISTANT_MOUNTAINS Shape {{ appearance PBRAppearance {{
 baseColorMap ImageTexture {{ url ["textures/D_ground.jpg"] }}
 normalMap ImageTexture {{ url ["textures/D_ground_normal.jpg"] }} normalMapFactor .7
 roughnessMap ImageTexture {{ url ["textures/D_ground_rough.jpg"] }}
 roughness .84 IBLStrength .85 metalness 0 }} geometry IndexedFaceSet {{
 coord Coordinate {{ point [{coordinates}] }} texCoord TextureCoordinate {{ point [{uv}] }}
 coordIndex [{' '.join(faces)}] texCoordIndex [{' '.join(faces)}] creaseAngle 1 }} }}'''
