"""Physical scene and measured ground-air fusion contracts."""
from pathlib import Path
import ast
import json
import math
import re
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for directory in ('controllers/search_agent','controllers/disaster_environment','tools'):
    sys.path.insert(0,str(ROOT/directory))
from multiview import aerial_pixel_enu, AerialInbox
from perception import TrackManager
from mission import delivery_guidance
from motion import path_pose
from build_search_worlds import build
from run_search import simulator_has_errors


class ComplexSceneContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = build('D').read_text(encoding='utf-8')
        cls.private = json.loads((ROOT/'evaluation/search/D.json').read_text())

    def test_gully_is_in_shared_visual_and_collision_height_grid(self):
        original = json.loads((ROOT/'evaluation/scenarios/B.json').read_text())
        before, expanded = np.array(original['heights']), np.array(self.private['heights'])
        row_offset=round((original['origin'][1]-self.private['origin'][1])/original['spacing'])
        col_offset=round((original['origin'][0]-self.private['origin'][0])/original['spacing'])
        after=expanded[row_offset:row_offset+before.shape[0],col_offset:col_offset+before.shape[1]]
        row = round((-1.5-original['origin'][1])/original['spacing'])
        col = round((-4-original['origin'][0])/original['spacing'])
        self.assertGreater(before[row,col]-after[row,col],1.0)
        visual = re.search(r'geometry DEF GROUND_GRID ElevationGrid\s*\{\s*height\s*\[([^\]]+)',self.world)[1]
        np.testing.assert_allclose(np.fromstring(visual,sep=','),expanded.ravel(),atol=1e-6)
        self.assertIn('boundingObject USE GROUND_GRID',self.world)
        # Rough patch contains both raised and lowered areas, not just a texture.
        from scipy.ndimage import gaussian_filter
        difference = after-before
        patch = (difference-gaussian_filter(difference,10))[40:80,44:100]
        self.assertGreater(patch.max(),.025)
        self.assertLess(patch.min(),-.025)

    def test_expanded_area_is_physical_and_fits_online_map(self):
        from mapping import OccupancyMap
        grid=np.array(self.private['heights'])
        self.assertEqual(grid.shape,(481,673))
        self.assertEqual(self.private['origin'],[-42,-30])
        # Rolling hills must have meaningful relief outside the old 42x30 m core.
        self.assertGreater(np.ptp(grid[:120,:]),3.)
        self.assertGreater(np.ptp(grid[:,:165]),3.)
        public=json.loads((ROOT/'config/search/public/D.json').read_text())
        online=OccupancyMap(side=public['mapping_side_m'])
        for corner in public['search_boundary_xy']:self.assertTrue(online.inside(online.cell(corner)))
        self.assertIn('xDimension 673',self.world)
        self.assertIn('yDimension 481',self.world)

    def test_added_hazards_have_collision_geometry(self):
        from upgrade_casualties import node_span
        for name in ('MOVING_LOG','MOVING_DEBRIS'):
            a,b = node_span(self.world,rf'\bDEF\s+{name}\s+Solid')
            self.assertIn('boundingObject Box',self.world[a:b])
        for name in re.findall(r'DEF (D_TREE_\d+) Solid',self.world):
            a,b = node_span(self.world,rf'\bDEF\s+{name}\s+Solid')
            self.assertIn('boundingObject Pose',self.world[a:b])
        self.assertNotIn('D_SHRUB_',self.world)
        self.assertEqual(len(re.findall(r'DEF D_TREE_\d+',self.world)),230)
        self.assertEqual(len(re.findall(r'DEF D_BUSH_\d+ ForestBush',self.world)),140)
        bush=(ROOT/'protos/ForestBush.proto').read_text()
        self.assertNotIn('Sphere',bush)
        self.assertIn('sassafras_leaves.png',bush)
        self.assertIn('boundingObject',bush)

    def test_aerial_perception_has_no_scene_tree_or_true_target_inputs(self):
        source = (ROOT/'controllers/aerial_observer/aerial_observer.py').read_text()
        ast.parse(source)
        for forbidden in ('Supervisor','getFromDef','getPosition(',"'evaluation'",'getRecognition'):
            self.assertNotIn(forbidden,source)
        public = json.loads((ROOT/'config/search/public/D.json').read_text())
        self.assertNotIn('path',public['aerial'])
        self.assertNotIn('additional_people_xy',public)
        self.assertFalse(public['require_cross_source_confirmation'])

    def test_pingpong_motion_is_continuous_at_turn_and_bounded(self):
        path = [[0,0],[2,0],[2,3]]
        before,_ = path_pose(path,1,5-.001)
        after,_ = path_pose(path,1,5+.001)
        np.testing.assert_allclose(before,after,atol=1e-10)
        samples = np.array([path_pose(path,1,t)[0] for t in np.linspace(0,25,251)])
        self.assertTrue(((samples>=0)&(samples<=[2,3])).all())
        self.assertLessEqual(np.linalg.norm(np.diff(samples,axis=0),axis=1).max(),.10000001)

    def test_parser_error_on_first_log_line_is_not_a_success(self):
        self.assertTrue(simulator_has_errors("ERROR: Missing declaration\nINFO: Finished"))
        self.assertTrue(simulator_has_errors("WARNING: Transform: Skipped unexpected Solid node in 'children'"))
        self.assertFalse(simulator_has_errors("INFO: controller exited successfully.\n"))

    def test_proto_declarations_precede_scene_nodes(self):
        last_declaration=max(m.end() for m in re.finditer(r'^EXTERNPROTO[^\n]+',self.world,re.M))
        first_world_node=re.search(r'\bWorldInfo\s*\{',self.world).start()
        self.assertLess(last_declaration,first_world_node)

    def test_downward_projection_and_yaw_sign(self):
        gps = [10,20,9]
        center = aerial_pixel_enu([319.5,239.5,8],640,480,1.1,gps,[0,0,0])
        np.testing.assert_allclose(center,[10,20,1],atol=1e-9)
        focal = 640/(2*math.tan(1.1/2))
        right = aerial_pixel_enu([319.5+focal/8,239.5,8],640,480,1.1,gps,[0,0,math.pi/2])
        np.testing.assert_allclose(right,[11,20,1],atol=1e-9)

    def packet(self):
        return dict(version=1,source='uav_rgbd',sequence=0,timestamp_s=1,
                    camera_origin_enu=[10,20,9],evidence='aerial/frame_0.jpg',
                    candidates=[dict(mean_enu=[10,22],covariance_enu=[[.04,0],[0,.09]],
                                     score=.9,depth=8,bbox=[10,20,50,80])])

    def test_radio_transforms_measured_frame_and_rejects_replay(self):
        inbox = AerialInbox([10,20,0],math.pi/2)
        result = inbox.ingest(json.dumps(self.packet()),1.1)
        np.testing.assert_allclose(result[0]['mean'],[2,0],atol=1e-9)
        self.assertGreater(np.linalg.eigvalsh(result[0]['cov']).min(),.04)
        self.assertEqual(inbox.ingest(json.dumps(self.packet()),1.2),[])
        self.assertEqual((inbox.accepted,inbox.rejected),(1,1))

    def test_radio_rejects_stale_future_nonfinite_and_invalid_boxes(self):
        variants=[]
        for key,value in (('timestamp_s',-5),('timestamp_s',4),('source','truth'),('evidence','../x.jpg')):
            packet=self.packet();packet[key]=value;variants.append(packet)
        for key,value in (('mean_enu',[float('nan'),0]),('covariance_enu',[[1,2],[2,1]]),
                          ('bbox',[40,20,10,80]),('bbox',[10,20,800,80])):
            packet=self.packet();packet['candidates'][0][key]=value;variants.append(packet)
        for packet in variants:
            inbox=AerialInbox([0,0,0],0)
            self.assertEqual(inbox.ingest(json.dumps(packet),1),[])
            self.assertEqual(inbox.accepted,0)

    def test_cross_view_fusion_does_not_refresh_stale_ground_delivery(self):
        manager=TrackManager(require_cross_source=True)
        ground=dict(mean=np.array([2.,0.]),cov=np.eye(2)*.04,score=.9,bbox=[1,1,30,50],
                    source='rover',local_xy=np.array([2.,0.]),evidence='agent/a.jpg')
        pose=np.zeros(3)
        manager.update([ground],pose,1)
        aerial=dict(ground,source='uav',observer_xy=np.array([2.,0.]),observer_z=8,
                    evidence='aerial/a.jpg')
        aerial.pop('local_xy')
        manager.update([aerial],pose,5)
        track=manager.tracks[0]
        self.assertTrue(manager.reportable(track))
        self.assertEqual(track.sources,{'rover','uav'})
        self.assertEqual(len(track.groups),2)
        self.assertEqual(track.local_last_seen,1)
        self.assertIsNone(delivery_guidance(track,pose,5))
        for t in range(6,20):manager.update([aerial],pose,t)
        self.assertEqual(len(track.groups),2)
        np.testing.assert_allclose(track.covariance,np.eye(2)*.04,atol=1e-8)


if __name__=='__main__':unittest.main()
