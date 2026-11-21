"""Small multimodal policy, safety-filtered Webots execution and rollout capture."""
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from policy_features import ACTION_SCALE, NAV_SIZE, RANGE_SHAPE, RGBD_SHAPE, SCHEMA_VERSION
from policy_features import expert_mode, planner_features
from route_tracker import RuleLocalPlanner, wrap_angle


class MultimodalPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(RGBD_SHAPE[0], 12, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(12, 24, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 3)), nn.Flatten(), nn.Linear(24 * 6, 32), nn.ReLU(),
        )
        self.obstacles = nn.Sequential(
            nn.Flatten(), nn.Linear(RANGE_SHAPE[0] * RANGE_SHAPE[1], 32), nn.ReLU(),
        )
        self.navigation = nn.Sequential(nn.Linear(NAV_SIZE, 24), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(32 + 32 + 24, 64), nn.ReLU())
        self.action = nn.Linear(64, 2)
        self.mode = nn.Linear(64, 3)

    def forward(self, rgbd, ranges, state):
        fused = self.fusion(torch.cat((self.vision(rgbd), self.obstacles(ranges), self.navigation(state)), dim=1))
        return self.action(fused), self.mode(fused)


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("feature_schema") != SCHEMA_VERSION:
        raise ValueError(f"Incompatible local learning checkpoint: {path}")
    model = MultimodalPolicy()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval(), checkpoint


def decode_actions(normalized, mode_logits):
    """Decode meters per second and radians per second, honoring the predicted mode."""
    modes = mode_logits.argmax(dim=-1)
    velocity = normalized[..., 0].clamp(0.0, 1.0) * float(ACTION_SCALE[0])
    turn = normalized[..., 1].clamp(-1.0, 1.0) * float(ACTION_SCALE[1])
    return torch.stack((velocity * (modes == 2), turn * (modes != 0)), dim=-1)


class RolloutRecorder:
    """One finalized manifest per episode and bounded compressed NumPy shards."""
    SHARD_SIZE = 128

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.rows = []
        self.shards = []
        self.samples = 0

    def append(self, features, expert_vw, proposal, selected, executed, source, time_s):
        self.rows.append((*features, expert_vw, expert_mode(expert_vw), proposal or (0.0, 0.0),
                          proposal is not None, selected, executed, source, time_s))
        self.samples += 1
        if len(self.rows) >= self.SHARD_SIZE:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        rgbd, ranges, state, labels, mode, proposal, has_proposal, selected, executed, source, times = zip(*self.rows)
        name = f"part_{len(self.shards):04d}.npz"
        np.savez_compressed(
            self.path / name, rgbd=np.asarray(rgbd, dtype=np.float16),
            ranges=np.asarray(ranges, dtype=np.float32), state=np.asarray(state, dtype=np.float32),
            expert_vw=np.asarray(labels, dtype=np.float32), expert_mode=np.asarray(mode, dtype=np.uint8),
            proposal_vw=np.asarray(proposal, dtype=np.float32), has_proposal=np.asarray(has_proposal, dtype=bool),
            selected_vw=np.asarray(selected, dtype=np.float32), executed_vw=np.asarray(executed, dtype=np.float32),
            action_source=np.asarray(source), time_s=np.asarray(times, dtype=np.float64),
        )
        self.shards.append(name)
        self.rows.clear()

    def close(self, metadata, status):
        self.flush()
        manifest = dict(feature_schema=SCHEMA_VERSION, samples=self.samples, shards=self.shards,
                        status=status, **metadata)
        pending = self.path / "manifest.tmp"
        pending.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        pending.replace(self.path / "manifest.json")


class LocalPolicyExperiment:
    """Run the expert, a guarded learner, or a DAgger mixture at a decision tick."""

    def __init__(self, policy, checkpoint=None, record_to=None, expert_mix=0.0, seed=17):
        self.policy = policy
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.expert_mix = expert_mix
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.expert = RuleLocalPlanner()
        self.model = None
        self.checkpoint_stage = None
        if policy == "neural":
            torch.set_num_threads(1)
            self.model, metadata = load_policy(checkpoint)
            self.checkpoint_stage = metadata["stage"]
        self.recorder = RolloutRecorder(record_to) if record_to is not None else None
        self.counts = Counter()
        self.latencies_ms = []
        self.pending = None

    def compute(self, inp):
        features = planner_features(inp) if self.recorder is not None or self.model is not None else None
        expert = self.expert.compute(inp) if self.model is None or self.recorder is not None else None
        proposal = None
        if self.model is not None:
            start = perf_counter()
            with torch.inference_mode():
                inputs = (torch.from_numpy(array).unsqueeze(0) for array in features)
                predicted, logits = self.model(*inputs)
                action = decode_actions(predicted, logits)[0].tolist()
            self.latencies_ms.append((perf_counter() - start) * 1000.0)
            proposal = tuple(action)

        if self.model is None:
            source = "expert"
        elif self.recorder is not None and self.rng.random() < self.expert_mix:
            source = "mixture_expert"
        elif not self.expert.allow_action(inp, proposal):
            source = "safety_fallback"
            if expert is None:
                expert = self.expert.compute(inp)
        else:
            source = "learner"

        if source == "learner":
            v, w = proposal
            delta = inp.lookahead_xy - inp.pose_xy
            heading_error = wrap_angle(np.arctan2(delta[1], delta[0]) - inp.yaw)
            info = dict(mode="NEURAL", blocked_locally=v < 1e-4 and (abs(w) < 1e-4 or abs(heading_error) < 0.40),
                        min_obs_m=self.expert._nearest_obstacle(inp),
                        map_clearance_m=inp.onboard_map.continuous_clearance(inp.pose_xy))
        else:
            v, w, info = expert
        self.pending = (features, expert, proposal, (v, w), source)
        return v, w, dict(info, action_source=source, network_proposal_vw=proposal)

    def observe_executed(self, v, w, time_s):
        """Called after supervisory stop logic and motor command application."""
        if self.pending is None:
            return
        features, expert, proposal, selected, source = self.pending
        self.pending = None
        self.counts[source] += 1
        executed = (v, w)
        actuated_source = source if executed == selected else "controller_override"
        if actuated_source != source:
            self.counts["controller_override"] += 1
        elif source == "learner":
            self.counts["learner_applied"] += 1
            if abs(v) > 1e-4 or abs(w) > 1e-4:
                self.counts["learner_nonzero_applied"] += 1
        if self.recorder is not None:
            self.recorder.append(features, expert[:2], proposal, selected, executed, actuated_source, time_s)

    def close(self, status):
        inference = None
        if self.latencies_ms:
            median, p95 = np.percentile(self.latencies_ms, [50, 95])
            inference = dict(median=float(median), p95=float(p95))
        stats = dict(policy=self.policy, checkpoint=self.checkpoint, checkpoint_stage=self.checkpoint_stage,
                     expert_mix=self.expert_mix, seed=self.seed, decisions=dict(self.counts),
                     model_inference_ms=inference)
        if self.recorder is not None:
            self.recorder.close(stats, status)
            stats["recorded_samples"] = self.recorder.samples
        return stats
