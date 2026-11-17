"""Baked terrain maps and irregular twig/leaf geometry from bundled photo textures."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def ground_maps(config,size=8192):
    destination=ROOT/'worlds/textures'
    signature=hashlib.sha256(json.dumps(dict(extent=config['extent_m'],seed=config['seed'],size=size,revision=3),
                                      sort_keys=True).encode()).hexdigest()
    stamp=destination/'D_ground_signature.json'
    files=[destination/f'D_ground{suffix}.jpg' for suffix in ('','_normal','_rough')]
    if stamp.is_file() and json.loads(stamp.read_text()).get('sha256')==signature and all(p.is_file() for p in files):return
    def tile(name):return np.asarray(Image.open(destination/'materials'/name).convert('RGB'))
    mud=tile('brown_mud_02_Diffuse.jpg');litter=tile('forest_floor_Diffuse.jpg')
    normals=tile('brown_mud_02_nor_gl.jpg');rough=tile('brown_mud_02_Rough.jpg')
    result=[np.empty((size,size,3),np.uint8) for _ in range(3)]
    width,height=config['extent_m'];x=np.linspace(-width/2,width/2,size,dtype=np.float32)[None,:]
    all_y=np.linspace(height/2,-height/2,size,dtype=np.float32)
    for start in range(0,size,96):
        y=all_y[start:start+96,None]
        macro=np.clip(.50+.26*np.sin(.24*x+.19*y)*np.cos(.27*y-.11*x),.15,.85)
        u=np.floor((x/2.4+.12*np.sin(y*.27))*1024).astype(np.int32)%1024
        v=np.floor((-y/2.4+.10*np.sin(x*.21))*1024).astype(np.int32)%1024
        dirt=mud[v,u].astype(np.float32)
        leaves=litter[(v+137)%1024,(u+349)%1024].astype(np.float32)*[.64,.76,.60]
        colour=dirt*macro[...,None]+leaves*(1-macro[...,None])
        # Damp silt near the central gully; other patches retain leaf/soil detail.
        damp=.13*np.exp(-((x+4)/2.2)**2-((y+1.5)/8)**4)
        colour*=((1-damp)*1.31)[...,None]
        result[0][start:start+len(y)]=np.clip(colour,0,255).astype(np.uint8)
        result[1][start:start+len(y)]=normals[v,u]
        result[2][start:start+len(y)]=np.clip(rough[v,u]*.62+48,70,242).astype(np.uint8)
    Image.fromarray(result[0]).save(files[0],quality=90,subsampling=2)
    Image.fromarray(result[1]).resize((4096,4096),Image.Resampling.LANCZOS).save(files[1],quality=90,subsampling=0)
    Image.fromarray(result[2]).convert('L').resize((4096,4096),Image.Resampling.LANCZOS).save(files[2],quality=90)
    stamp.write_text(json.dumps(dict(sha256=signature,resolution=[size,size],extent_m=config['extent_m'],
        source='bundled Poly Haven CC0 mud and forest-floor maps'),indent=2),encoding='utf-8')


def bush_proto():
    """Static mesh reused by instances: curved leaf-spray cards, no green spheres."""
    destination=ROOT/'protos/ForestBush.proto'
    if destination.is_file() and '# generated version: 2' in destination.read_text()[:400]:return
    rng=np.random.default_rng(731)
    vertices=[];uv=[];faces=[];branches=[]
    for shoot in range(9):
        angle=shoot*2.39996+rng.uniform(-.25,.25)
        length=rng.uniform(.75,1.20)
        end=np.array([math.cos(angle)*length*.49,math.sin(angle)*length*.49,length])
        midpoint=end*.46
        branches.append((np.array([0.,0.,.02]),end,rng.uniform(.008,.017)))
        for layer in range(12):
            t=.20+.80*layer/11
            position=t*end+np.array([rng.normal(0,.11),rng.normal(0,.11),rng.normal(0,.055)])
            azimuth=angle+layer*2.4
            size=rng.uniform(.18,.29)*(1.15-.22*t)
            tangent=np.array([math.cos(azimuth),math.sin(azimuth),rng.uniform(-.25,.25)])*size
            up=np.array([-math.sin(azimuth)*.35,math.cos(azimuth)*.35,rng.uniform(.65,1)])*size
            # Creased, two-sided textured branch spray rather than a flat silhouette.
            points=[position-tangent-up,position+tangent-up,position+tangent+up,position-tangent+up,
                    position+np.array([.025,-.018,.045])]
            start=len(vertices);vertices.extend(points);uv.extend([(0,0),(1,0),(1,1),(0,1),(.5,.5)])
            for i in range(4):
                faces.append(f'{start+i} {start+(i+1)%4} {start+4} -1')
                faces.append(f'{start+4} {start+(i+1)%4} {start+i} -1')
    branch_points=[];branch_faces=[]
    for start,end,radius in branches:
        vector=end-start;vector/=np.linalg.norm(vector)
        left=np.cross(vector,[0,0,1]);left/=np.linalg.norm(left)
        up=np.cross(vector,left);base=len(branch_points)
        for p,r in ((start,radius),(end,radius*.24)):
            for i in range(7):branch_points.append(p+r*(math.cos(i*2*math.pi/7)*left+math.sin(i*2*math.pi/7)*up))
        for i in range(7):branch_faces.append(f'{base+i} {base+(i+1)%7} {base+7+(i+1)%7} {base+7+i} -1')
    def points(data):return ', '.join(' '.join(f'{float(v):.5f}' for v in p) for p in data)
    destination.write_text(f'''#VRML_SIM R2025a utf8
# Project-generated irregular understory; leaf/bark photos: Cyberbotics CC BY 4.0.
# generated version: 2
# Source textures and attribution: MATERIAL_SOURCES.md.
PROTO ForestBush [ field SFVec3f translation 0 0 0 field SFRotation rotation 0 0 1 0
 field SFVec3f scale 1 1 1 field SFString name "textured understory" ] {{
 Solid {{ translation IS translation rotation IS rotation name IS name
 children [ Transform {{ scale IS scale children [
 Shape {{ appearance PBRAppearance {{ baseColor .88 .94 .79
 baseColorMap ImageTexture {{ url ["textures/sassafras_leaves.png"] repeatS FALSE repeatT FALSE }}
 normalMap ImageTexture {{ url ["textures/sassafras_leaves_normal.jpg"] }}
 roughness .95 metalness 0 IBLStrength 1.2 }}
 geometry IndexedFaceSet {{ coord Coordinate {{ point [{points(vertices)}] }}
 texCoord TextureCoordinate {{ point [{points(uv)}] }}
 coordIndex [{' '.join(faces)}] texCoordIndex [{' '.join(faces)}] creaseAngle .35 }} }}
 Shape {{ appearance PBRAppearance {{ baseColor .47 .39 .25
 baseColorMap ImageTexture {{ url ["textures/sassafras_trunk.jpg"] }} roughness 1 metalness 0 }}
 geometry IndexedFaceSet {{ coord Coordinate {{ point [{points(branch_points)}] }}
 coordIndex [{' '.join(branch_faces)}] creaseAngle .8 }} }}
 ] }} ] boundingObject Pose {{ translation 0 0 .38 children [ Cylinder {{ radius .12 height .76 subdivision 8 }} ] }}
 }} }}
''',encoding='utf-8')
