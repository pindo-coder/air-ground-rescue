"""Textured 3-D forest placement for D, including rover-height branch screens."""
import math
import re
import numpy as np
from upgrade_casualties import node_span

SPECIES_HEIGHT = {'Oak':5.38, 'Sassafras':12.79, 'Cypress':4.94}


def tree_node(name, xy, terrain, rng, young=False, screen=False, species=None):
    species = species or rng.choice(['Oak','Sassafras','Cypress'],p=[.53,.32,.15])
    height = float(rng.uniform(1.8,2.9) if young else rng.uniform(4.1,6.7))
    if screen:species='Cypress';height=float(rng.uniform(2.3,3.2))
    x,y=map(float,xy);ground=float(terrain(x,y))
    height=min(height,15.4-ground)  # Clear the expanded scene's 16 m survey altitude.
    scale=height/SPECIES_HEIGHT[species]
    spread=scale*float(rng.uniform(.86,1.14))
    yaw=float(rng.uniform(-math.pi,math.pi))
    radius=max(.045,(.09 if species=='Oak' else .15)*spread)
    scaling=f'{scale:.6f}' if species=='Cypress' else f'{spread:.6f} {spread:.6f} {scale:.6f}'
    # Only trunks collide. Alpha-cut leaf meshes affect actual RGB/depth views.
    node=f'''DEF {name} Solid {{ translation {x:.4f} {y:.4f} {ground-.035:.5f}
 name "{name.lower()}" model "{species} forest tree"
 children [ {species} {{ scale {scaling} rotation 0 0 1 {yaw:.6f} enableBoundingObject FALSE }} ]
 boundingObject Pose {{ translation 0 0 {height*.40:.5f} children [
 Cylinder {{ radius {radius:.5f} height {height*.80:.5f} subdivision 12 }} ] }} }}'''
    return node,dict(def_name=name,xy=[x,y],height_m=height,trunk_radius_m=radius,
                     species=species,young=bool(young),screen=bool(screen))


def in_obstacle(point, obstacle, margin=.45):
    delta=np.asarray(point)-obstacle['xy'];yaw=obstacle.get('yaw',0.)
    c,s=math.cos(yaw),math.sin(yaw)
    local=np.array([[c,s],[-s,c]])@delta
    return bool(np.all(np.abs(local)<np.asarray(obstacle['size'][:2])/2+margin))


def add_forest(scene,meta,config,people,terrain):
    rng=np.random.default_rng(config['seed']+73)
    settings=config['vegetation'];trees=[];nodes=[];bushes=[]
    for species in SPECIES_HEIGHT:
        declaration=f'EXTERNPROTO "../protos/{species}.proto"'
        if declaration not in scene:
            scene=scene.replace('#VRML_SIM R2025a utf8','#VRML_SIM R2025a utf8\n'+declaration,1)
    # Replace the old sparse, bare-looking pines in the derived world only.
    for match in reversed(list(re.finditer(r'\bDEF\s+(TREE_\d+)\s+Solid',scene))):
        name=match[1];a,b=node_span(scene,rf'\bDEF\s+{name}\s+Solid')
        xy=[float(v) for v in re.search(r'translation\s+([^\n]+)',scene[a:b])[1].split()[:2]]
        node,record=tree_node(name,xy,terrain,rng,young=False)
        scene=scene[:a]+node+scene[b:];trees.append(record)
    original_count=len(trees)
    scene=scene.replace('#VRML_SIM R2025a utf8','#VRML_SIM R2025a utf8\nEXTERNPROTO "../protos/ForestBush.proto"',1)
    reserved=[meta['spawn']]+[v['xy'] for v in people]
    sites=[np.asarray(item['xy']) for item in trees]
    gx,gy=config['gully']['center']
    def available(point,screen=False):
        if any(np.linalg.norm(point-p)<(1.4 if screen else 1.6) for p in reserved):return False
        if abs(point[0]-gx)<1.4 and abs(point[1]-gy)<config['gully']['length_m']/2+.6:return False
        if any(np.linalg.norm(point-p)<settings['minimum_trunk_spacing_m'] for p in sites):return False
        if any(in_obstacle(point,o) for o in meta['obstacles'] if not o['def'].startswith('TREE_')):return False
        return True
    for xy in settings['low_branch_screens']:
        point=np.asarray(xy,float)
        if not available(point,True):
            raise ValueError(f'Low branch tree intersects existing obstacle at {xy}')
        node,record=tree_node(f'D_TREE_{len(nodes)}',point,terrain,rng,young=True,screen=True)
        nodes.append(node);trees.append(record);sites.append(point)
    for _ in range(30000):
        if len(nodes)==settings['trees']:break
        cx,cy,w,h=settings['patches'][int(rng.integers(len(settings['patches'])))]
        point=np.array([rng.uniform(cx-w/2,cx+w/2),rng.uniform(cy-h/2,cy+h/2)])
        if not available(point):continue
        node,record=tree_node(f'D_TREE_{len(nodes)}',point,terrain,rng,
                              young=rng.random()<settings['sapling_fraction'])
        nodes.append(node);trees.append(record);sites.append(point)
    if len(nodes)!=settings['trees']:raise ValueError('Not enough clear forest sites')
    added_trees=len(nodes)
    def add_bush(point,screen=False):
        scale=float(rng.uniform(1.0,1.4) if screen else rng.uniform(.65,1.25))
        spread=scale*float(rng.uniform(.8,1.15));yaw=float(rng.uniform(-math.pi,math.pi))
        x,y=map(float,point);name=f'D_BUSH_{len(bushes)}'
        nodes.append(f'''DEF {name} ForestBush {{ name "understory {len(bushes)}"
 translation {x:.5f} {y:.5f} {terrain(x,y)-.03:.5f} rotation 0 0 1 {yaw:.5f}
 scale {spread:.5f} {spread:.5f} {scale:.5f} }}''')
        bushes.append(dict(def_name=name,xy=[x,y],screen=screen,height_m=1.48*scale))
    for xy in settings['understory_screens']:add_bush(np.array(xy,float),True)
    for _ in range(30000):
        if len(bushes)>=settings['bushes']:break
        # Bush clusters sit near tree cover, leaving irregular walk/drive gaps.
        center=trees[int(rng.integers(len(trees)))]['xy']
        point=np.asarray(center)+rng.normal(0,1.15,2)
        if not (-41<point[0]<41 and -29<point[1]<29):continue
        if any(np.linalg.norm(point-p)<1.35 for p in reserved):continue
        if any(np.linalg.norm(point-p['xy'])<.7 for p in bushes):continue
        if abs(point[0]-gx)<1.4 and abs(point[1]-gy)<config['gully']['length_m']/2+.6:continue
        if any(in_obstacle(point,o,.25) for o in meta['obstacles'] if not o['def'].startswith('TREE_')):continue
        add_bush(point)
    if len(bushes)!=settings['bushes']:raise ValueError('Not enough understory sites')
    meta['forest']=dict(original_replaced=original_count,added=added_trees,trees=trees,bushes=bushes)
    # A single named group makes the real-camera occlusion audit reversible.
    scene+='\nDEF D_FOREST Group { children [\n'+'\n'.join(nodes)+'\n] }\n'
    return scene,meta
