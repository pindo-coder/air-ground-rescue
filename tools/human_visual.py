"""Bake a neutral supine body from a licensed mesh, with relaxed arm poses."""
import json
import math
import numpy as np


def human_assets(up,render,write_obj):
    folder=up/'cesium_man/glTF'
    model=json.loads((folder/'CesiumMan.gltf').read_text())
    data=(folder/model['buffers'][0]['uri']).read_bytes()
    def accessor(index):
        a=model['accessors'][index];view=model['bufferViews'][a['bufferView']]
        dtype={5126:'<f4',5123:'<u2',5121:'u1',5125:'<u4'}[a['componentType']]
        width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]
        size=np.dtype(dtype).itemsize
        return np.ndarray((a['count'],width),dtype=dtype,buffer=data,
                          offset=view.get('byteOffset',0)+a.get('byteOffset',0),
                          strides=(view.get('byteStride',width*size),size)).copy()
    primitive=model['meshes'][0]['primitives'][0];attributes=primitive['attributes']
    source=accessor(attributes['POSITION']);source_normal=accessor(attributes['NORMAL'])
    weights=accessor(attributes['WEIGHTS_0']);joint_indices=accessor(attributes['JOINTS_0']).astype(int)
    actual_joints=np.asarray(model['skins'][0]['joints'])[joint_indices]
    indices=accessor(primitive['indices']).reshape(-1,3)
    categories={'shirt':[3,12,13,17,14],'skin':[20,21,18,15,19,16],
                'pants':[8,4,9,5],'shoes':[10,6,11,7]}
    scores=np.stack([np.sum(weights*np.isin(actual_joints,bones),axis=1) for bones in categories.values()],axis=1)
    labels=np.argmax(scores[indices].mean(axis=1),axis=1)
    head_score=np.sum(weights*(actual_joints==21),axis=1)
    keep=head_score[indices].mean(axis=1)<.30
    result={}
    for key,angle in zip('ABC',(46,55,50)):
        vertex=source.copy();normal=source_normal.copy()
        for side,bones in ((1,[17,18,19]),(-1,[14,15,16])):
            amount=np.sum(weights*np.isin(actual_joints,bones),axis=1)
            radians=-side*math.radians(angle);c,s=math.cos(radians),math.sin(radians)
            rotation=np.array([[1,0,0],[0,c,-s],[0,s,c]])
            anchor=np.array([-.01,side*.13,1.10])
            transformed=(source-anchor)@rotation.T+anchor
            vertex+=amount[:,None]*(transformed-source)
            normal+=amount[:,None]*(source_normal@rotation.T-source_normal)
        # Original Z-up body, forward +X. Lie supine: head -X, front +Z.
        vertex=vertex[:,[2,1,0]]*[-1,1,1];normal=normal[:,[2,1,0]]*[-1,1,1]
        size=np.array([1.22,1.22,.90])
        vertex*=size
        vertex[:,0]+=.94
        vertex[:,2]-=vertex[indices[keep]].reshape(-1,3)[:,2].min()-.02
        normal/=size;normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        uv=np.column_stack((np.arctan2(vertex[:,2]-.16,vertex[:,1])/(2*math.pi)*3,vertex[:,0]*4))
        parts=[]
        for i,group in enumerate(categories):
            name=f'casualty_{key}_{group}.obj'
            faces=indices[keep&(labels==i)]
            if group=='skin':
                # The replacement scan supplies the neck. Remove the old neck
                # skin inside the collar to avoid two overlapping skin surfaces.
                points=vertex[faces]
                old_neck=np.any(points[:,:,0]<-.425,axis=1)&np.all(np.abs(points[:,:,1])<.11,axis=1)
                faces=faces[~old_neck]
            assert len(faces)>0
            write_obj(render/name,vertex,uv,faces,normals=normal)
            shirt={'A':'.48 .26 .20','B':'.19 .29 .44','C':'.46 .43 .29'}[key]
            color={'shirt':shirt,'skin':'.64 .46 .35','pants':'.19 .21 .24','shoes':'.07 .065 .06'}[group]
            cloth=group in ('shirt','pants')
            parts.append({'mesh':name,'albedo':'upstream/webots/projects/appearances/protos/textures/fabric/fabric_base_color.jpg' if cloth else None,
                          'normal':'upstream/webots/projects/appearances/protos/textures/fabric/fabric_normal.jpg' if cloth else None,'color':color})
        parts.append({'mesh':'head_scan.obj','albedo':'upstream/head_scan/Map-COL.jpg',
                      'normal':'upstream/head_scan/Infinite-Level_02_Tangent_SmoothUV.jpg'})
        used=vertex[indices[keep]].reshape(-1,3)
        result[key]={'parts':parts,'bounds':[used.min(0).tolist(),used.max(0).tolist()]}
    return result
