"""Build-only D scene: real terrain geometry, solid vegetation, moving hazards."""
import json
import math
import re
from pathlib import Path
import numpy as np
from scene_elements import RescueElements
from forest_render import add_forest
from render_style import appearance
from forest_terrain import expanded_terrain,reanchor_props,outer_mountains
from build_forest_materials import ground_maps,bush_proto
from upgrade_casualties import node_span

ROOT = Path(__file__).resolve().parents[1]


def add_complex_scene(scene, meta, height_at):
    config = json.loads((ROOT/'config/scenes/D.json').read_text(encoding='utf-8'))
    original_meta=meta
    meta=expanded_terrain(meta,config)
    ground_maps(config);bush_proto()
    heights = np.asarray(meta['heights'], float)
    ys, xs = np.indices(heights.shape)
    x = meta['origin'][0] + xs*meta['spacing']
    y = meta['origin'][1] + ys*meta['spacing']
    gully = config['gully']; gx, gy = gully['center']
    edge = .5*(1-np.tanh((np.abs(y-gy)-gully['length_m']/2)/.35))
    depression = gully['depth_m']*np.exp(-((x-gx)/(gully['width_m']*.43))**8)*edge
    rough = config['rough_patch']; rx, ry = rough['center']; sx, sy = rough['size']
    envelope = np.exp(-((x-rx)/(sx/2))**8-((y-ry)/(sy/2))**8)
    ripples = rough['amplitude_m']*np.sin(2*math.pi*x/rough['wavelength_m'])*np.cos(2*math.pi*y/3.4)*envelope
    heights = np.round(heights-depression+ripples, 5)
    meta['heights'] = heights.tolist()
    scene=reanchor_props(scene,original_meta,meta,height_at)
    # Expand the actual collider, texture coordinates and world translation together.
    scene=scene.replace('translation -21 -15 0',f"translation {meta['origin'][0]} {meta['origin'][1]} 0",1)
    scene=re.sub(r'\bxDimension\s+337\b',f'xDimension {heights.shape[1]}',scene,count=1)
    scene=re.sub(r'\byDimension\s+241\b',f'yDimension {heights.shape[0]}',scene,count=1)
    a,b=node_span(scene,r'\bDEF\s+DISTANT_MOUNTAINS\s+Shape')
    scene=scene[:a]+outer_mountains(meta,height_at)+scene[b:]
    scene=scene.replace('textures/B_ground.png','textures/D_ground.jpg')
    scene=scene.replace('textures/B_ground_normal.jpg','textures/D_ground_normal.jpg')
    scene=scene.replace('textures/B_ground_rough.jpg','textures/D_ground_rough.jpg')
    # The visual grid is also the USE GROUND_GRID collision object.
    values = ', '.join(f'{v:.5f}' for v in heights.ravel())
    scene, count = re.subn(r'(geometry DEF GROUND_GRID ElevationGrid\s*\{\s*height\s*\[)[^\]]*',
                           lambda m: m[1]+'\n'+values+'\n', scene, count=1)
    if count != 1:
        raise ValueError('Terrain collision grid not found')
    if 'EXTERNPROTO "../protos/Pine.proto"' not in scene:
        scene = scene.replace('#VRML_SIM R2025a utf8', '#VRML_SIM R2025a utf8\nEXTERNPROTO "../protos/Pine.proto"', 1)
    scene = scene.replace('#VRML_SIM R2025a utf8',
        '#VRML_SIM R2025a utf8\nEXTERNPROTO "../protos/ObserverDrone.proto"',1)
    water=[];water_faces=[]
    for i,py in enumerate(np.linspace(gy-gully['length_m']/2+.8,gy+gully['length_m']/2-.8,65)):
        middle=gx+.075*math.sin(py*1.8);width=.27+.065*math.sin(py*2.1)
        level=height_at(meta,middle,py)+.055
        water.extend([[middle-width,py,level],[middle+width,py,level]])
        if i:water_faces.append(f'{2*i-2} {2*i-1} {2*i+1} {2*i} -1')
    water_points=', '.join(' '.join(f'{float(v):.5f}' for v in p) for p in water)
    scene += f'''\nDEF GULLY_WATER Shape {{
 appearance PBRAppearance {{ baseColor .20 .19 .145 transparency .20 roughness .23 metalness 0 }}
 geometry IndexedFaceSet {{ coord Coordinate {{ point [{water_points}] }}
 coordIndex [{' '.join(water_faces)}] creaseAngle .8 }} }}\n'''
    victims = [dict(id=1, xy=meta['goal'])]
    for index, xy in enumerate(config['additional_people_xy'], 2):
        elements = RescueElements(dict(key='B', seed=202+index, goal=xy), lambda a,b:height_at(meta,a,b))
        elements.casualty()
        scene += '\n'+elements.nodes[0].replace('CASUALTY', f'SEARCH_VICTIM_{index}').replace(
            'name "casualty"', f'name "search victim {index}"')
        victims.append(dict(id=index, xy=xy))
    for obstacle in config['moving_obstacles']:
        px, py = obstacle['path'][0]; w, d, h = obstacle['size']
        if obstacle['kind']=='log':
            visual=f'''Pose {{ rotation 0 1 0 1.5707963268 children [ Shape {{
 appearance {appearance('wood','.8 .77 .69',(.8,1.8))}
 geometry Cylinder {{ radius {d*.48} height {w} subdivision 18 }} }} ] }}
 Pose {{ translation .20 .13 .10 rotation 1 0 0 -.9 children [ Shape {{
 appearance {appearance('wood','.75 .71 .61')} geometry Cylinder {{ radius .035 height .39 subdivision 9 }} }} ] }}'''
            for side in (-1,1):
                visual+=f'''Pose {{ translation {side*(w/2+.001)} 0 0 rotation 0 1 0 1.5707963268 children [ Shape {{
 appearance PBRAppearance {{ baseColor .48 .35 .20 roughness 1 metalness 0 }}
 geometry Cylinder {{ radius {d*.44} height .005 subdivision 18 }} }} ] }}'''
        else:
            points=[[-w/2,-d/2,-h/2],[w/2,-d/2,-h/2],[w/2,d/2,-h/2],[-w/2,d/2,-h/2],
                    [-w*.43,-d*.42,h*.35],[w*.29,-d*.37,h/2],[w*.47,d*.28,h*.1],[-w*.3,d*.48,h*.28]]
            coords=', '.join(' '.join(map(str,p)) for p in points)
            visual=f'''Shape {{ appearance {appearance('plaster','.75 .71 .64',(.55,.55))}
 geometry IndexedFaceSet {{ coord Coordinate {{ point [{coords}] }}
 coordIndex [0 3 2 1 -1, 0 1 5 4 -1, 1 2 6 5 -1, 2 3 7 6 -1, 3 0 4 7 -1, 4 5 6 7 -1] creaseAngle .10 }} }}
 Pose {{ translation .02 .04 {h*.43} rotation 0 0 1 .37 children [ Shape {{
 appearance {appearance('brick','.8 .7 .62')} geometry Box {{ size {w*.6} {d*.33} .09 }} }} ] }}'''
        scene += f'''\nDEF {obstacle['def']} Solid {{ translation {px} {py} {height_at(meta,px,py)+h/2:.5f}
 name "{obstacle['def'].lower()}" children [ {visual} ] boundingObject Box {{ size {w} {d} {h} }} }}\n'''
    scene,meta=add_forest(scene,meta,config,victims,lambda a,b:height_at(meta,a,b))
    # Balanced skylight and softer contrast expose bark/leaf texture in the rover camera.
    scene=scene.replace('luminosity 0.8','luminosity 1.05',1)
    scene=scene.replace('ambientIntensity 0.18','ambientIntensity 0.30',1)
    scene=scene.replace('color 1 0.94 0.83','color 1 0.97 0.91',1)
    scene=scene.replace('intensity 1.65','intensity 2.1',1)
    scene=scene.replace('intensity 0.4','intensity 0.70',1)
    scene=scene.replace('visibilityRange 350','visibilityRange 650',1)
    scene=re.sub(r'\btitle\s+"[^"]+"','title "Rescue D | Forested flood valley | 84 x 60 m"',scene,count=1)
    from metrics import look_at_rotation
    a,b=node_span(scene,r'\bDEF\s+SCENE_VIEW\s+Viewpoint')
    view_rotation=' '.join(map(str,look_at_rotation([-58,-69,48],[0,0,2])))
    scene=scene[:a]+f'DEF SCENE_VIEW Viewpoint {{ position -58 -69 48 orientation {view_rotation} fieldOfView .95 far 330 exposure 1.28 ambientOcclusionRadius .8 }}'+scene[b:]
    ax, ay = config['aerial']['path'][0]
    scene += f'''\nDEF OBSERVER_DRONE ObserverDrone {{ translation {ax} {ay} {config['aerial']['altitude_enu_m']} }}
DEF DISASTER_ENVIRONMENT Robot {{ name "Disaster environment" supervisor TRUE
 controller "disaster_environment" customData "D" window "<none>" }}\n'''
    return scene, meta, victims, config
