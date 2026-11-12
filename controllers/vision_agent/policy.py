"""Stereo-only, low-speed exploration. Set VISION_DRIVE=1 to enable movement."""
from dataclasses import dataclass
import math
import os


@dataclass
class Action:
    linear_m_s: float = 0.0
    angular_rad_s: float = 0.0
    claim_arrival: bool = False
    active: bool = False


class Policy:
    def reset(self, task):
        self.enabled = os.environ.get("VISION_DRIVE") == "1"
        self.mode = "preview"
        if not self.enabled:
            return
        import cv2
        import numpy as np
        self.cv2, self.np = cv2, np
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=64, blockSize=5,
            P1=8 * 25, P2=32 * 25, disp12MaxDiff=1,
            uniquenessRatio=8, speckleWindowSize=60, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.baseline = task["camera"]["baseline_m"]
        self.pitch = task["camera"]["pitch_rad"]
        self.height = task["camera"]["height_m"]
        self.fov = task["camera"]["fov_rad"]
        self.initial_heading = math.radians(task["goal_hint"]["center_deg_left_positive"])
        self.yaw_estimate = 0.0
        self.previous_time = None
        self.previous_turn = 0.0
        self.turn_until = 0.0
        self.turn_sign = 1.0

    def command(self, linear, angular, mode):
        self.mode = mode
        self.previous_turn = angular
        return Action(linear, angular, active=True)

    def act(self, observation):
        if not self.enabled:
            return Action()
        np, cv2 = self.np, self.cv2
        now = observation["timestamp_s"]
        elapsed = 0 if self.previous_time is None else max(0, min(.2, now - self.previous_time))
        self.previous_time = now
        self.yaw_estimate += self.previous_turn * elapsed
        if now < 1.5:
            return self.command(0, 0, "settling")

        shape = observation["height"], observation["width"], 4
        images = []
        for name in ("left_bgra", "right_bgra"):
            frame = np.frombuffer(observation[name], dtype=np.uint8).reshape(shape)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            small = cv2.resize(gray, (320, 240), interpolation=cv2.INTER_AREA)
            images.append(self.contrast.apply(small))
        disparity = self.stereo.compute(*images).astype(np.float32) / 16.0
        valid = disparity > 1.0
        focal = 320 / (2 * math.tan(self.fov / 2))
        depth = focal * self.baseline / np.maximum(disparity, .1)
        rows, cols = np.indices(depth.shape)
        down = (rows - 119.5) * depth / focal
        lateral = (159.5 - cols) * depth / focal
        forward = depth * math.cos(self.pitch) - down * math.sin(self.pitch) + .26
        height = self.height - depth * math.sin(self.pitch) - down * math.cos(self.pitch)
        range_valid = valid & (forward > .38) & (forward < 4.0)
        raised = height > (.075 + .025 * forward)
        drop = (height < -.27) & (forward < 1.75) & (rows > 120)
        obstacles = range_valid & (raised | drop)
        x, y = forward[obstacles], lateral[obstacles]

        def gap(angle):
            along = x * math.cos(angle) + y * math.sin(angle)
            side = -x * math.sin(angle) + y * math.cos(angle)
            corridor = (along > .1) & (np.abs(side) < .52)
            if np.count_nonzero(corridor) < 8:
                return 4.0
            return float(np.percentile(along[corridor], 8))

        front = gap(0.0)
        if now < self.turn_until:
            return self.command(0.0, self.turn_sign * .48, "turning away from obstacle")
        if front < 1.35:
            left, right = gap(.65), gap(-.65)
            self.turn_sign = 1.0 if left >= right else -1.0
            self.turn_until = now + 1.4
            return self.command(0.0, self.turn_sign * .48, "turning away from obstacle")

        desired = (self.initial_heading - self.yaw_estimate + math.pi) % (2 * math.pi) - math.pi
        candidates = np.linspace(-.60, .60, 9)
        scores = []
        for angle in candidates:
            opening = gap(float(angle))
            alignment = abs(float(angle) - max(-.60, min(.60, desired)))
            scores.append(min(opening, 2.8) - .52 * alignment - .16 * abs(float(angle)))
        steering = float(candidates[int(np.argmax(scores))])
        angular = max(-.40, min(.40, 1.15 * steering))
        coverage = float(np.mean(valid[155:225, 100:220]))
        if coverage < .22:
            return self.command(0.0, .22, "checking stereo visibility")
        speed = .085 * min(1.0, max(0, (front - 1.20) / 1.2))
        speed *= max(.30, 1.0 - abs(angular))
        return self.command(speed, angular, "stereo obstacle avoidance")
