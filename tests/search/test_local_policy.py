"""Regression checks for the observation schema and DAgger action provenance."""
from pathlib import Path
import importlib.util
import json
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for name in ("uav_mapper", "uav_map_rover"):
    sys.path.insert(0, str(ROOT / "controllers" / name))

from policy_features import RANGE_SHAPE, RGBD_SHAPE, NAV_SIZE, SCHEMA_VERSION, encode_rgbd, planner_features
from route_tracker import LocalPlannerInput, RuleLocalPlanner


def make_input(obstacles=()):
    points = np.asarray(obstacles, dtype=float).reshape(-1, 2)
    return LocalPlannerInput(
        pose_xy=np.zeros(2), yaw=0.0, attitude_rp=(0.0, 0.0),
        speed_mps=0.10, prev_cmd=(0.10, 0.0), lookahead_xy=np.array([0.80, 0.35]),
        goal_xy=np.array([3.0, 1.0]), dist_to_goal=3.0,
        obstacles_body_xy=points, lidar_obstacles_body_xy=points,
        rgbd_obstacles_body_xy=np.empty((0, 2)),
        onboard_map=SimpleNamespace(continuous_clearance=lambda _: 0.80),
        rgbd_frame=np.zeros(RGBD_SHAPE, dtype=np.float32),
    )


class LocalPolicyInputTests(unittest.TestCase):
    def test_rgbd_preserves_channel_order_and_invalid_depth(self):
        image = np.zeros((24, 32, 4), dtype=np.uint8)
        image[..., :3] = [15, 20, 240]
        depth = np.full((24, 32), 2.0)
        depth[0, 0] = np.inf
        frame = encode_rgbd(image.tobytes(), depth)
        self.assertEqual(frame.shape, RGBD_SHAPE)
        np.testing.assert_allclose(frame[:, 1, 0], [240/255, 20/255, 15/255, 0.5, 1.0])
        np.testing.assert_array_equal(frame[3:, 0, 0], [1.0, 0.0])
        rgbd, obstacles, state = planner_features(make_input([(0.42, 0.0)]))
        self.assertEqual((rgbd.shape, obstacles.shape, state.shape), (RGBD_SHAPE, RANGE_SHAPE, (NAV_SIZE,)))
        self.assertAlmostEqual(float(obstacles[0, 18]), 0.42/5)
        self.assertEqual(obstacles[1, 18], 1.0)
        self.assertFalse(obstacles[3].any())

    def test_shared_guard_checks_rollout_clearance(self):
        guard = RuleLocalPlanner()
        self.assertTrue(guard.allow_action(make_input(), (0.22, 0.0)))
        self.assertFalse(guard.allow_action(make_input([(0.50, 0.0)]), (0.22, 0.0)))


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional local learning dependency")
class DaggerProvenanceTests(unittest.TestCase):
    def test_queries_expert_on_learner_state_and_records_outer_stop(self):
        import torch
        from local_policy import LocalPolicyExperiment, MultimodalPolicy

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "bc.pt"
            model = MultimodalPolicy()
            torch.save(dict(model_state_dict=model.state_dict(), feature_schema=SCHEMA_VERSION, stage="bc"), checkpoint)
            experiment = LocalPolicyExperiment("neural", checkpoint=checkpoint, record_to=root / "capture")
            inp = make_input()
            expert_vw = (0.16, -0.26)
            with patch.object(experiment.expert, "compute", return_value=(*expert_vw, {})) as label:
                proposal = torch.tensor([[0.20, 0.18]])
                with patch("local_policy.decode_actions", return_value=proposal):
                    v, w, info = experiment.compute(inp)
                    label.assert_called_once_with(inp)
                    self.assertEqual(info["action_source"], "learner")
                    experiment.observe_executed(v, w, time_s=1.0)
                    experiment.compute(inp)
                    experiment.observe_executed(0.0, 0.0, time_s=1.1)
            summary = experiment.close("STOPPED_SAFE")
            self.assertEqual(summary["decisions"]["learner_nonzero_applied"], 1)
            manifest = json.loads((root / "capture/manifest.json").read_text())
            self.assertEqual(manifest["samples"], 2)
            with np.load(root / "capture" / manifest["shards"][0], allow_pickle=False) as shard:
                np.testing.assert_allclose(shard["expert_vw"], [expert_vw, expert_vw])
                np.testing.assert_allclose(shard["executed_vw"], [[0.20, 0.18], [0, 0]])
                self.assertEqual(shard["action_source"].tolist(), ["learner", "controller_override"])
