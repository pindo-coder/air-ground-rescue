"""Safety/research contracts that are meaningful without starting Webots."""
from pathlib import Path
import ast
import json
import math
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"controllers/search_agent"))
try:
    from mapping import OccupancyMap, local_control
    from perception import Track, TrackManager, covariance_intersection, observation_views, project_depth, unrotate_box
    from mission import MissionScheduler,delivery_guidance
except ModuleNotFoundError as error:
    raise unittest.SkipTest("Install requirements-search.txt to run optional algorithm checks") from error
sys.path.insert(0,str(ROOT/"evaluation"))
from search_metrics import match_reports,physical_delivery_ok,world_xy
sys.path.insert(0,str(ROOT/"tools"))
from build_search_worlds import build


class ResearchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for case in ("A","B","C","smoke","D"):build(case)

    def free_map(self):
        grid=OccupancyMap(side=20)
        grid.seen[:]=True;grid.logodds[:]=-3;grid.clearance[:]=5
        return grid

    def test_controller_has_no_truth_or_supervisor_access(self):
        for file in (ROOT/"controllers/search_agent").glob("*.py"):
            source=file.read_text(encoding="utf-8-sig")
            ast.parse(source)
            for forbidden in ("Supervisor", "getFromDef", "getPosition(", "getRecognition", '"evaluation"', "'evaluation'"):
                self.assertNotIn(forbidden,source,file.name)

    def test_public_configs_do_not_supply_people(self):
        files=list((ROOT/"config/search/public").glob("*.json"))
        self.assertEqual(len(files),5)
        for file in files:
            public=json.loads(file.read_text())
            self.assertFalse(public["target_positions_provided"])
            for key in ("victims","goal","spawn","target_xy"):
                self.assertNotIn(key,public)

    def test_correlated_identical_measurements_cannot_collapse_covariance(self):
        mean=np.array([2.,1.]);cov=np.diag([.4,.2])
        for _ in range(40):
            mean,cov=covariance_intersection(mean,cov,mean,np.diag([.4,.2]))
        np.testing.assert_allclose(cov,np.diag([.4,.2]),rtol=1e-9)
        self.assertGreater(np.linalg.eigvalsh(cov).min(),0)

    def test_repeated_frames_at_one_pose_cannot_create_valid_report(self):
        manager=TrackManager()
        detection=dict(mean=np.array([2.,0.]),cov=np.eye(2)*.02,score=.9,bbox=[20,20,50,60])
        for t in range(30):manager.update([detection],np.zeros(3),t)
        self.assertEqual(len(manager.tracks),1)
        self.assertEqual(len(manager.tracks[0].groups),1)
        self.assertFalse(manager.reportable(manager.tracks[0]))

    def test_distinct_observations_can_report_and_not_repeat(self):
        manager=TrackManager()
        detection=dict(mean=np.array([2.,0.]),cov=np.eye(2)*.02,score=.9,bbox=[20,20,50,60])
        manager.update([detection],np.zeros(3),0)
        manager.update([detection],np.array([0.,1.,0.]),5)
        self.assertTrue(manager.reportable(manager.tracks[0]))
        manager.tracks[0].reported=True
        self.assertFalse(manager.reportable(manager.tracks[0]))

    def test_uncertain_position_rejected_despite_high_person_score(self):
        track=Track(1,np.array([2.,0.]),np.eye(2),confidence=.999,
                    groups=[np.zeros(2),np.ones(2)])
        self.assertFalse(TrackManager().reportable(track))
        self.assertTrue(TrackManager().reportable(track,covariance_gate=False))

    def test_active_views_look_at_target_and_stay_free(self):
        grid=self.free_map();track=Track(1,np.array([2.,0.]),np.eye(2)*.08,groups=[np.zeros(2)])
        views=observation_views(track,np.zeros(3),grid)
        self.assertTrue(views)
        for view in views:
            self.assertTrue(grid.free(view["point"],.4))
            self.assertAlmostEqual(view["yaw"],math.atan2(*(track.mean-view["point"])[::-1]))

    def test_urgent_job_preempts_exploration(self):
        grid=self.free_map();scheduler=MissionScheduler("full")
        scheduler.current=dict(kind="explore",point=np.array([6.,0.]),yaw=0.,track=None,gain=1.)
        scheduler.current_since=0
        target=Track(1,np.array([2.,0.]),np.eye(2)*.02,reported=True,report_time=0,deadline=8)
        self.assertEqual(scheduler.choose(np.zeros(3),grid,[target],2)["kind"],"deliver")

    def test_empty_payload_requires_reload_not_phantom_delivery(self):
        grid=self.free_map();scheduler=MissionScheduler("full");scheduler.cargo=0
        target=Track(1,np.array([2.,0.]),np.eye(2)*.02,reported=True,report_time=0,deadline=8)
        self.assertEqual(scheduler.choose(np.zeros(3),grid,[target],2)["kind"],"reload")

    def test_reserve_forces_return(self):
        scheduler=MissionScheduler("full",budget=10);scheduler.spent=7
        self.assertEqual(scheduler.choose(np.array([3.,0.,0.]),self.free_map(),[],0)["kind"],"return")

    def test_local_control_does_not_drive_through_wall(self):
        grid=self.free_map()
        for y in np.linspace(-3,3,50):
            grid.clearance[grid.cell([.5,y])]=0
        laser=np.array([[.5,y] for y in np.linspace(-3,3,100)])
        v,w,_=local_control(np.zeros(3),np.array([3.,0.]),0,laser,grid)
        self.assertEqual(v,0.)

    def test_one_frame_cannot_update_one_track_twice(self):
        manager=TrackManager()
        proposal=dict(mean=np.array([2.,0.]),cov=np.eye(2)*.02,score=.9,bbox=[20,20,50,60])
        manager.update([proposal,dict(proposal,mean=np.array([2.1,0.]))],np.zeros(3),1)
        self.assertEqual(len(manager.tracks),1)
        self.assertEqual(len(manager.tracks[0].groups),1)

    def test_duplicate_report_is_suppressed(self):
        manager=TrackManager()
        first=Track(1,np.array([2.,0.]),np.eye(2)*.02,reported=True)
        duplicate=Track(2,np.array([2.7,0.]),np.eye(2)*.02,confidence=.99,groups=[np.zeros(2),np.ones(2)])
        manager.tracks=[first,duplicate]
        self.assertFalse(manager.reportable(duplicate))

    def test_rgbd_projection_compensates_camera_mount_and_body_pitch(self):
        center=[159.5,119.5,2.]
        point=project_depth(center,320,240)[0]
        np.testing.assert_allclose(point,[.26+2*math.cos(.08),0,.30-2*math.sin(.08)],atol=1e-9)
        corrected=project_depth(center,320,240,pitch=-.08)[0]
        self.assertGreater(corrected[2],point[2])

    def test_rotated_person_box_returns_to_original_pixels(self):
        np.testing.assert_allclose(unrotate_box([20,30,60,80],1,320,240),[30,180,80,220])
        np.testing.assert_allclose(unrotate_box([20,30,60,80],2,320,240),[260,160,300,210])

    def test_first_valid_report_cannot_be_replaced_by_a_later_duplicate(self):
        reports=[dict(timestamp_s=2,xy=[.9,0]),dict(timestamp_s=5,xy=[.1,0]),dict(timestamp_s=7,xy=[50,50])]
        matches=match_reports(reports,[dict(id=1,xy=[0,0])],[0,0],0,1.)
        self.assertEqual(matches,{0:0})

    def test_delivery_message_alone_does_not_count_as_delivery(self):
        self.assertFalse(physical_delivery_ok(True,2.,5.,2))
        self.assertFalse(physical_delivery_ok(True,.2,1.,2))
        self.assertFalse(physical_delivery_ok(True,2.,1.,0))
        self.assertFalse(physical_delivery_ok(False,2.,1.,2))
        self.assertTrue(physical_delivery_ok(True,2.,1.,2))

    def test_map_arrival_is_not_enough_without_recent_local_target_evidence(self):
        track=Track(1,np.array([3.,0.]),np.eye(2)*.04,reported=True,last_seen=1)
        self.assertIsNone(delivery_guidance(track,np.array([2.,0.,0.]),1))
        track.latest_local=np.array([1.4,0.]);track.observation_pose=np.zeros(3)
        self.assertTrue(delivery_guidance(track,np.zeros(3),1.5)[2])
        self.assertIsNone(delivery_guidance(track,np.zeros(3),5))

    def test_main_worlds_preserve_the_original_people_and_geometry(self):
        from upgrade_casualties import node_span
        for key,name in (("A","A_trail.wbt"),("B","B_washout.wbt"),("C","C_generalization.wbt")):
            base=(ROOT/"worlds"/name).read_text(encoding="utf-8")
            derived=(ROOT/"worlds"/f"search_{key}.wbt").read_text(encoding="utf-8")
            for pattern in (r"\bDEF\s+CASUALTY\s+Solid",r"\bDEF\s+TERRAIN\s+Solid"):
                a,b=node_span(base,pattern);c,d=node_span(derived,pattern)
                self.assertEqual(base[a:b],derived[c:d])
            private=json.loads((ROOT/"evaluation/search"/f"{key}.json").read_text())
            original=json.loads((ROOT/"evaluation/scenarios"/f"{key}.json").read_text())
            self.assertEqual(private["victims"],[dict(id=1,xy=original["goal"])])


if __name__=="__main__":unittest.main()
