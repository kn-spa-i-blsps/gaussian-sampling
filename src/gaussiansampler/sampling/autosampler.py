import math
import random

import numpy as np
import torch


class AutoSampler:
    _default_dist: float
    _learning_rate: float
    _distance_vol: float
    _max_constant: float
    _min_constant: float
    _up_constant: float
    _k_samples: int  # controls the number of gaussians used to compute the target point

    def __init__(
        self,
        renderer,
        hole_scorer,
        gaussians,
        k_samples=3,
        granularity=5,
        *,
        width=800,
        height=600,
        fov_deg=60.0,
        score_threshold=0.0,
        learning_rate=0.1,
        distance_vol=0.5,
        min_constant=0.5,
    ):
        self.gaussian_means = torch.as_tensor(gaussians["means"]).detach().cpu()
        self.renderer = renderer
        self.hole_scorer = hole_scorer
        self.granularity = granularity
        self._k_samples = k_samples
        self.width = width
        self.height = height
        self.fov_deg = fov_deg
        self.score_threshold = score_threshold
        self._learning_rate = learning_rate
        self._distance_vol = distance_vol
        self._min_constant = min_constant

        if self.granularity <= 0:
            raise ValueError("granularity must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("render dimensions must be positive")
        if not 0 < self.fov_deg < 180:
            raise ValueError("fov_deg must be between 0 and 180")
        if not 0 <= self._learning_rate < 1:
            raise ValueError("learning_rate must be in the range [0, 1)")

        # Safe init with a large value from the scene
        self._default_dist = self.get_scene_diameter()

    def get_scene_diameter(self):
        """Computes the diameter of the bounding box for all gaussians"""
        bounds = self.gaussian_means.amax(dim=0) - self.gaussian_means.amin(dim=0)
        return torch.linalg.vector_norm(bounds).item()

    def get_random_spot(self):
        indices = torch.randint(
            self.gaussian_means.shape[0],
            (self._k_samples,),
            device="cpu",
        )
        return self.gaussian_means[indices].mean(dim=0)

    def _camera_pose(self, target, direction, distance):
        target = target.detach().cpu().numpy()
        direction = np.asarray(direction, dtype=np.float32)
        camera_position = target + direction * distance
        forward = target - camera_position
        forward /= np.linalg.norm(forward)

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-8:
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.column_stack((right, down, forward))
        c2w[:3, 3] = camera_position
        return c2w

    def _score_pose(self, c2w):
        focal_length = 0.5 * self.width / math.tan(math.radians(self.fov_deg) / 2)
        intrinsics = np.array(
            [
                [focal_length, 0, self.width / 2],
                [0, focal_length, self.height / 2],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        image = self.renderer.render(c2w, intrinsics, self.width, self.height)
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating) and image.max(initial=0) <= 1:
            image = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return float(self.hole_scorer.score_image(image))

    def next_step(self):
        target = self.get_random_spot()

        # Sample one direction uniformly on the unit sphere.
        z = random.uniform(-1.0, 1.0)
        azimuth = random.uniform(0.0, 2 * math.pi)
        radial = math.sqrt(1 - z * z)
        direction = np.array(
            [radial * math.cos(azimuth), radial * math.sin(azimuth), z],
            dtype=np.float32,
        )

        minimum_distance = self._min_constant * self._default_dist
        minimum_pose = self._camera_pose(target, direction, minimum_distance)
        if self._score_pose(minimum_pose) < self.score_threshold:
            self._default_dist *= 1 - self._learning_rate
            return []

        # The randomized distance starts above the learned average.
        distance = self._default_dist * (1 + self._distance_vol * random.random())
        lower_bound = self._min_constant * distance
        distances = np.linspace(distance, lower_bound, num=self.granularity)

        successful_poses = []
        highest_successful_distance = None
        for candidate_distance in distances:
            pose = self._camera_pose(target, direction, candidate_distance)
            if self._score_pose(pose) >= self.score_threshold:
                successful_poses.append(pose)
                if highest_successful_distance is None:
                    highest_successful_distance = candidate_distance

        if successful_poses:
            # Move the learned distance only toward the farthest successful view.
            self._default_dist += self._learning_rate * (
                highest_successful_distance - self._default_dist
            )
            return successful_poses

        self._default_dist *= 1 - self._learning_rate
        return []
