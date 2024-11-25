import copy
from typing import List

import dlimp as dl
import gym
import jax.numpy as jnp
import numpy as np

# need to put https://github.com/tonyzhaozh/act in your PATH for this import to work
from sim_env import BOX_POSE, make_sim_env
import cv2


def hit_or_miss_sample(circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h):
    rect_min_x = rect_center_x-rect_w/2
    rect_max_x = rect_center_x+rect_w/2
    rect_min_y = rect_center_y-rect_h/2
    rect_max_y = rect_center_y+rect_h/2
    
    min_x = min(circle_x-circle_r, rect_min_x)
    max_x = max(circle_x+circle_r, rect_max_x)    
    min_y = min(circle_y-circle_r, rect_min_y)
    max_y = max(circle_y+circle_r, rect_max_y)        

    in_rect, in_circle = False, False

    while not (in_rect and in_circle):    
        sample = np.random.uniform(low=[min_x, min_y], high=[max_x, max_y])
        sample_x, sample_y = sample
        
        in_rect = (rect_min_x < sample_x) & (sample_x < rect_max_x) & (rect_min_y < sample_y) & (sample_y < rect_max_y)
        in_circle = np.linalg.norm(sample - np.stack([circle_x, circle_y])) < circle_r

    return sample


class AlohaGymEnv(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        camera_names: List[str],
        im_size: int = 256,
        seed: int = 1234,
    ):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in ["primary", "wrist"][: len(camera_names)]
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        ts = self._env.step(action)
        obs, images = self.get_obs(ts)
        reward = ts.reward
        info = {"images": images}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        return obs, reward, False, False, info

    def reset(self, **kwargs):
        # sample new box pose
        x_range = [0.0, 0.2]
        y_range = [0.4, 0.6]
        z_range = [0.05, 0.05]
        ranges = np.vstack([x_range, y_range, z_range])
        cube_position = self._rng.uniform(ranges[:, 0], ranges[:, 1])
        cube_quat = np.array([1, 0, 0, 0])
        BOX_POSE[0] = np.concatenate([cube_position, cube_quat])

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0

        return obs, info

    def get_obs(self, ts):
        curr_obs = {}
        vis_images = []

        obs_img_names = ["primary", "wrist"]
        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation["images"][cam_name]
            vis_images.append(copy.deepcopy(curr_image))
            curr_image = jnp.array(curr_image)
            curr_obs[f"image_{obs_img_names[i]}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        qpos_numpy = np.array(ts.observation["qpos"])
        qpos = jnp.array(qpos_numpy)
        curr_obs["proprio"] = qpos

        return curr_obs, np.concatenate(vis_images, axis=-2)

    def get_task(self):
        return {
            "language_instruction": ["pick up the cube and hand it over"],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }
    
class AlohaCubeGymEnv(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        camera_names: List[str],
        im_size: int = 256,
        seed: int = 1234,
    ):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in ["primary", "secondary", "left_wrist", "right_wrist"][: len(camera_names)]
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        ts = self._env.step(action)
        obs, images = self.get_obs(ts)
        reward = ts.reward
        info = {"images": images}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        return obs, reward, False, False, info

    def reset(self, **kwargs):
        # sample new box pose
        obj_x_range = [-0.1, 0.3]
        dst_x_range = [-0.1, 0.3]
        y_range = [0.3, 0.8] # 0.3
        # y_range = [-0.1, 0.1]    
        z_range = [0.015, 0.015]
        obj_angle_range = [0, 180]
        obj_dst_interval = 0.1    

        obj_ranges = np.vstack([obj_x_range, y_range, z_range])
        obj_position = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1])
        obj_angle = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        obj_quat = np.array([np.cos(np.deg2rad(obj_angle)/2), 0, 0, np.sin(np.deg2rad(obj_angle)/2)])    
        dst_ranges = np.vstack([dst_x_range, y_range, z_range])
        dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])

        while np.linalg.norm(dst_position[:2] - obj_position[:2]) < 0.055+0.002+obj_dst_interval:
            dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
        dst_quat = np.array([1, 0, 0, 0])
    
        BOX_POSE[0] = np.concatenate([obj_position, obj_quat, dst_position, dst_quat])

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0

        return obs, info

    def get_obs(self, ts):
        curr_obs = {}
        vis_images = []

        obs_img_names = ["primary", "secondary", "left_wrist", "right_wrist"]
        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation["images"][cam_name]
            vis_images.append(copy.deepcopy(curr_image))
            curr_image = jnp.array(curr_image)
            curr_obs[f"image_{obs_img_names[i]}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        qpos_numpy = np.array(ts.observation["qpos"])
        qpos = jnp.array(qpos_numpy)
        curr_obs["proprio"] = qpos

        return curr_obs, np.concatenate(vis_images, axis=-2)

    def get_task(self):
        return {
            "language_instruction": ["pick up the red cube and put it in the bucket"],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }


class AlohaCucumberGymEnv(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        camera_names: List[str],
        im_size: int = 256,
        seed: int = 1234,
    ):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in ["primary", "secondary", "left_wrist", "right_wrist"][: len(camera_names)]
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        try:
            ts = self._env.step(action)
        except:
            curr_obs = {}
            vis_images = []

            black = np.zeros((self._im_size, self._im_size, 3), dtype=np.uint8)
            obs_img_names = ["primary", "secondary", "left_wrist", "right_wrist"]
            for i, cam_name in enumerate(self.camera_names):
                curr_image = jnp.array(black.copy())
                curr_obs[f"image_{obs_img_names[i]}"] = curr_image
                vis_images.append(copy.deepcopy(curr_image))

            curr_obs = dl.transforms.resize_images(
                curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
            )

            images = np.concatenate(vis_images, axis=-2)
            info = {"images": images}
            return {"image_primary":black, "image_secondary":black, "image_left_wrist":black, "image_right_wrist":black, "proprio": np.nan}, 0, True, False, info
        obs, images = self.get_obs(ts)
        reward = ts.reward
        info = {"images": images}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        return obs, reward, False, False, info

    def reset(self, **kwargs):
        # sample new box pose
        obj_x_range = [-0.1, 0.3]
        dst_x_range = [-0.1, 0.3]
        y_range = [0.3, 0.8] # 0.3
        # y_range = [-0.1, 0.1]    
        z_range = [0.015, 0.015]
        obj_angle_range = [0, 180]
        obj_dst_interval = 0.1    

        obj_ranges = np.vstack([obj_x_range, y_range, z_range])
        obj_position = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1])
        obj_angle = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        obj_quat = np.array([np.cos(np.deg2rad(obj_angle)/2), 0, 0, np.sin(np.deg2rad(obj_angle)/2)])    
        dst_ranges = np.vstack([dst_x_range, y_range, z_range])
        dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])

        while np.linalg.norm(dst_position[:2] - obj_position[:2]) < 0.055+0.002+obj_dst_interval:
            dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
        dst_quat = np.array([1, 0, 0, 0])
    
        BOX_POSE[0] = np.concatenate([obj_position, obj_quat, dst_position, dst_quat])

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0

        return obs, info

    def get_obs(self, ts):
        curr_obs = {}
        vis_images = []

        obs_img_names = ["primary", "secondary", "left_wrist", "right_wrist"]
        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation["images"][cam_name]
            vis_images.append(copy.deepcopy(curr_image))
            curr_image = jnp.array(curr_image)
            curr_obs[f"image_{obs_img_names[i]}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        qpos_numpy = np.array(ts.observation["qpos"])
        qpos = jnp.array(qpos_numpy)
        curr_obs["proprio"] = qpos

        return curr_obs, np.concatenate(vis_images, axis=-2)

    def get_task(self):
        return {
            "language_instruction": ["pick up the cucumber and put it in the bucket"],
            # "language_instruction": ["pick up the cucumber and put it on the table"],
            # "language_instruction": ["pick up the cucumber and put it outside the bucket"],
            # "language_instruction": [""],
            # "language_instruction": ["pick up the red cube and put it in the bucket"],
            # "language_instruction": ["cucumber"],
            # "language_instruction": ["move the cucumber into the bucket"],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }
    

class MobileAlohaCucumberGymEnv(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        camera_names: List[str],
        im_size: int = 256,
        seed: int = 1234,
    ):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in ["primary", "secondary", "tertiary", "quaternary", "left_wrist", "right_wrist"][: len(camera_names)]
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        try:
            ts = self._env.step(action)
        except:
            curr_obs = {}
            vis_images = []

            black = np.zeros((self._im_size, self._im_size, 3), dtype=np.uint8)
            obs_img_names = ["primary", "secondary", "tertiary", "quaternary", "left_wrist", "right_wrist"]
            for i, cam_name in enumerate(self.camera_names):
                curr_image = jnp.array(black.copy())
                curr_obs[f"image_{obs_img_names[i]}"] = curr_image
                vis_images.append(copy.deepcopy(curr_image))

            curr_obs = dl.transforms.resize_images(
                curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
            )

            images = np.concatenate(vis_images, axis=-2)
            info = {"images": images}
            return {"image_primary":black, "image_secondary":black, "image_tertiary":black, "image_quaternary":black, "image_left_wrist":black, "image_right_wrist":black, "proprio": np.nan}, 0, True, False, info
        obs, images = self.get_obs(ts)
        reward = ts.reward
        info = {"images": images}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        return obs, reward, False, False, info

    def reset(self, **kwargs):
        circle_x = 0.295
        circle_y = 0.414
        obj_circle_r = 0.564095655480825
        dst_circle_r = 0.7767498780487515

        rect_center_x = -0.0235
        rect_center_y = 0.9845
        rect_offset = 0.05
        rect_w = 0.272*2 - rect_offset*2
        rect_h = 0.1785*2 - rect_offset*2

        obj_z = 0.842
        dst_z = 0.827
        obj_angle_range = [-90, 90] # -90, 90
        obj_dst_interval = 0.1            
        obj_xy = hit_or_miss_sample(obj_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
        dst_xy = hit_or_miss_sample(dst_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)

        obj_angle = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        obj_quat = np.array([np.cos(np.deg2rad(obj_angle)/2), 0, 0, np.sin(np.deg2rad(obj_angle)/2)])    

        while np.linalg.norm(dst_xy - obj_xy) < 0.055+0.002+obj_dst_interval:
            dst_xy = hit_or_miss_sample(dst_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
        
        obj_position = np.hstack([obj_xy, obj_z])
        dst_position = np.hstack([dst_xy, dst_z])
        dst_quat = np.array([1, 0, 0, 0])
    
        BOX_POSE[0] = np.concatenate([obj_position, obj_quat, dst_position, dst_quat])

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0

        return obs, info

    def get_obs(self, ts):
        curr_obs = {}
        self.vis_images = {}

        obs_img_names = ["primary", "secondary", "tertiary", "quaternary", "left_wrist", "right_wrist"]
        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation["images"][cam_name]
            self.vis_images[f"image_{obs_img_names[i]}"] = copy.deepcopy(curr_image)
            curr_image = jnp.array(curr_image)
            curr_obs[f"image_{obs_img_names[i]}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        qpos_numpy = np.array(ts.observation["qpos"])
        qpos = jnp.array(qpos_numpy)
        curr_obs["proprio"] = qpos

        return curr_obs, self.vis_images

    def get_task(self):
        return {
            "language_instruction": ["pick up the cucumber and put it in the bucket"],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }


class AlohaCoupleCucumberGymEnv(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        camera_names: List[str],
        im_size: int = 256,
        seed: int = 1234,
    ):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in ["primary", "secondary", "left_wrist", "right_wrist"][: len(camera_names)]
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        ts = self._env.step(action)
        obs, images = self.get_obs(ts)
        reward = int(self._env.task.cucumber_in_bucket)
        info = {"images": images}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        return obs, reward, False, False, info

    def reset(self, **kwargs):
        # sample new box pose
        obj_x_range = [-0.1, 0.3]
        dst_x_range = [-0.1, 0.3]
        y_range = [0.3, 0.8] # 0.3
        # y_range = [-0.1, 0.1]    
        z_range = [0.015, 0.015]
        obj_angle_range = [0, 180]
        obj_dst_interval = 0.1    

        obj_ranges = np.vstack([obj_x_range, y_range, z_range])
        obj_position = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1])
        obj_angle = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        obj_quat = np.array([np.cos(np.deg2rad(obj_angle)/2), 0, 0, np.sin(np.deg2rad(obj_angle)/2)])    

        box_position = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1])
        while np.linalg.norm(box_position[:2] - obj_position[:2]) < 0.055+0.002+obj_dst_interval:
            box_position = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1])
        box_angle = np.random.uniform(obj_angle_range[0], obj_angle_range[1])
        box_quat = np.array([np.cos(np.deg2rad(box_angle)/2), 0, 0, np.sin(np.deg2rad(box_angle)/2)]) 

        dst_ranges = np.vstack([dst_x_range, y_range, z_range])
        dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
        while np.linalg.norm(dst_position[:2] - obj_position[:2]) < 0.055+0.002+obj_dst_interval or np.linalg.norm(dst_position[:2] - box_position[:2]) < 0.055+0.002+obj_dst_interval:
            dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
        dst_quat = np.array([1, 0, 0, 0])
    
        BOX_POSE[0] = np.concatenate([obj_position, obj_quat, box_position, box_quat, dst_position, dst_quat])

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0

        return obs, info

    def get_obs(self, ts):
        curr_obs = {}
        self.vis_images = []

        obs_img_names = ["primary", "secondary", "left_wrist", "right_wrist"]
        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation["images"][cam_name]
            self.vis_images.append(copy.deepcopy(curr_image))
            curr_image = jnp.array(curr_image)
            curr_obs[f"image_{obs_img_names[i]}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        qpos_numpy = np.array(ts.observation["qpos"])
        qpos = jnp.array(qpos_numpy)
        curr_obs["proprio"] = qpos

        return curr_obs, np.concatenate(self.vis_images, axis=-2)

    def get_task(self):
        return {
            "language_instruction": ["pick up the cucumber and put it in the bucket"],
            # "language_instruction": ["pick up the cucumber and put it on the table"],
            # "language_instruction": ["pick up the cucumber and put it outside the bucket"],
            # "language_instruction": [""],
            # "language_instruction": ["pick up the red cube and put it in the bucket"],
            # "language_instruction": ["cucumber"],
            # "language_instruction": ["red cube"],
            # "language_instruction": ["move the cucumber into the bucket"],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }
    

class MobileAlohaCoupleCucumberGymEnv(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        camera_names: List[str],
        im_size: int = 256,
        seed: int = 1234,
    ):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in ["primary", "secondary", "tertiary", "quaternary", "left_wrist", "right_wrist"][: len(camera_names)]
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        try:
            ts = self._env.step(action)
        except:
            curr_obs = {}
            vis_images = []

            black = np.zeros((self._im_size, self._im_size, 3), dtype=np.uint8)
            obs_img_names = ["primary", "secondary", "tertiary", "quaternary", "left_wrist", "right_wrist"]
            for i, cam_name in enumerate(self.camera_names):
                curr_image = jnp.array(black.copy())
                curr_obs[f"image_{obs_img_names[i]}"] = curr_image
                vis_images.append(copy.deepcopy(curr_image))

            curr_obs = dl.transforms.resize_images(
                curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
            )

            images = np.concatenate(vis_images, axis=-2)
            info = {"images": images}
            return {"image_primary":black, "image_secondary":black, "image_tertiary":black, "image_quaternary":black, "image_left_wrist":black, "image_right_wrist":black, "proprio": np.nan}, 0, True, False, info
        obs, images = self.get_obs(ts)
        reward = int(self._env.task.cucumber_in_bucket)
        info = {"images": images}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        return obs, reward, False, False, info

    def reset(self, **kwargs):
        circle_x = 0.295
        circle_y = 0.414
        obj_circle_r = 0.564095655480825
        dst_circle_r = 0.7767498780487515

        rect_center_x = -0.0235
        rect_center_y = 0.9845
        rect_offset = 0.05
        rect_w = 0.272*2 - rect_offset*2
        rect_h = 0.1785*2 - rect_offset*2

        obj_z = 0.842
        dst_z = 0.827
        box_z = 0.842
        obj_angle_range = [-90, 90] # -90, 90
        obj_dst_interval = 0.055 + 0.002  + 0.1            
        box_dst_interval = 0.055 + 0.002 + 0.02*np.sqrt(2)
        obj_box_interval = 0.1 + 0.02*np.sqrt(2)

        obj_xy = hit_or_miss_sample(obj_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
        box_xy = hit_or_miss_sample(obj_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
        dst_xy = hit_or_miss_sample(dst_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
        while np.linalg.norm(dst_xy - obj_xy) < obj_dst_interval or np.linalg.norm(dst_xy - box_xy) < box_dst_interval or np.linalg.norm(obj_xy - box_xy) < obj_box_interval:
            obj_xy = hit_or_miss_sample(obj_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
            box_xy = hit_or_miss_sample(obj_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
            dst_xy = hit_or_miss_sample(dst_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)

        obj_angle = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        obj_quat = np.array([np.cos(np.deg2rad(obj_angle)/2), 0, 0, np.sin(np.deg2rad(obj_angle)/2)])    

        box_angle = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        box_quat = np.array([np.cos(np.deg2rad(box_angle)/2), 0, 0, np.sin(np.deg2rad(box_angle)/2)])            
        
        obj_position = np.hstack([obj_xy, obj_z])
        box_position = np.hstack([box_xy, box_z])
        dst_position = np.hstack([dst_xy, dst_z])
        dst_quat = np.array([1, 0, 0, 0])
    
        BOX_POSE[0] = np.concatenate([obj_position, obj_quat, box_position, box_quat, dst_position, dst_quat])

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0

        return obs, info

    def get_obs(self, ts):
        curr_obs = {}
        self.vis_images = {}

        obs_img_names = ["primary", "secondary", "tertiary", "quaternary", "left_wrist", "right_wrist"]
        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation["images"][cam_name]
            self.vis_images[f"image_{obs_img_names[i]}"] = copy.deepcopy(curr_image)
            curr_image = jnp.array(curr_image)
            curr_obs[f"image_{obs_img_names[i]}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        qpos_numpy = np.array(ts.observation["qpos"])
        qpos = jnp.array(qpos_numpy)
        curr_obs["proprio"] = qpos

        return curr_obs, self.vis_images

    def get_task(self):
        return {
            "language_instruction": ["pick up the cucumber and put it in the bucket"],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }
    



class AlohaCoupleCubeGymEnv(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        camera_names: List[str],
        im_size: int = 256,
        seed: int = 1234,
    ):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in ["primary", "secondary", "left_wrist", "right_wrist"][: len(camera_names)]
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        ts = self._env.step(action)
        obs, images = self.get_obs(ts)
        # reward = ts.reward
        reward = int(self._env.task.cube_in_bucket)        
        info = {"images": images}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        return obs, reward, False, False, info

    def reset(self, **kwargs):
        # sample new box pose
        obj_x_range = [-0.1, 0.3]
        dst_x_range = [-0.1, 0.3]
        y_range = [0.3, 0.8]    
        z_range = [0.015, 0.015]
        obj_angle_range = [0, 180]
        obj_dst_interval = 0.1    

        obj_ranges = np.vstack([obj_x_range, y_range, z_range])
        obj_position = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1])
        obj_angle = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        obj_quat = np.array([np.cos(np.deg2rad(obj_angle)/2), 0, 0, np.sin(np.deg2rad(obj_angle)/2)])    

        box_position = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1])
        while np.linalg.norm(box_position[:2] - obj_position[:2]) < 0.055+0.002+obj_dst_interval:
            box_position = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1])
        box_angle = np.random.uniform(obj_angle_range[0], obj_angle_range[1])
        box_quat = np.array([np.cos(np.deg2rad(box_angle)/2), 0, 0, np.sin(np.deg2rad(box_angle)/2)]) 

        dst_ranges = np.vstack([dst_x_range, y_range, z_range])
        dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
        while np.linalg.norm(dst_position[:2] - obj_position[:2]) < 0.055+0.002+obj_dst_interval or np.linalg.norm(dst_position[:2] - box_position[:2]) < 0.055+0.002+obj_dst_interval:
            dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
        dst_quat = np.array([1, 0, 0, 0])
    
        BOX_POSE[0] = np.concatenate([obj_position, obj_quat, box_position, box_quat, dst_position, dst_quat])

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0

        return obs, info

    def get_obs(self, ts):
        curr_obs = {}
        self.vis_images = []

        obs_img_names = ["primary", "secondary", "left_wrist", "right_wrist"]
        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation["images"][cam_name]
            self.vis_images.append(copy.deepcopy(curr_image))
            curr_image = jnp.array(curr_image)
            curr_obs[f"image_{obs_img_names[i]}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        qpos_numpy = np.array(ts.observation["qpos"])
        qpos = jnp.array(qpos_numpy)
        curr_obs["proprio"] = qpos

        return curr_obs, np.concatenate(self.vis_images, axis=-2)

    def get_task(self):
        return {
            "language_instruction": ["pick up the red cube and put it in the bucket"],
            # "language_instruction": ["red cube"],
            # "language_instruction": [""],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }    
    

class MobileAlohaCoupleCubeGymEnv(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        camera_names: List[str],
        im_size: int = 256,
        seed: int = 1234,
    ):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in ["primary", "secondary", "tertiary", "quaternary", "left_wrist", "right_wrist"][: len(camera_names)]
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        try:
            ts = self._env.step(action)
        except:
            curr_obs = {}
            vis_images = []

            black = np.zeros((self._im_size, self._im_size, 3), dtype=np.uint8)
            obs_img_names = ["primary", "secondary", "tertiary", "quaternary", "left_wrist", "right_wrist"]
            for i, cam_name in enumerate(self.camera_names):
                curr_image = jnp.array(black.copy())
                curr_obs[f"image_{obs_img_names[i]}"] = curr_image
                vis_images.append(copy.deepcopy(curr_image))

            curr_obs = dl.transforms.resize_images(
                curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
            )

            images = np.concatenate(vis_images, axis=-2)
            info = {"images": images}
            return {"image_primary":black, "image_secondary":black, "image_tertiary":black, "image_quaternary":black, "image_left_wrist":black, "image_right_wrist":black, "proprio": np.nan}, 0, True, False, info
        obs, images = self.get_obs(ts)
        reward = int(self._env.task.cube_in_bucket)        
        info = {"images": images}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        return obs, reward, False, False, info

    def reset(self, **kwargs):
        circle_x = 0.295
        circle_y = 0.414
        obj_circle_r = 0.564095655480825
        dst_circle_r = 0.7767498780487515

        rect_center_x = -0.0235
        rect_center_y = 0.9845
        rect_offset = 0.05
        rect_w = 0.272*2 - rect_offset*2
        rect_h = 0.1785*2 - rect_offset*2

        obj_z = 0.842
        dst_z = 0.827
        box_z = 0.842
        obj_angle_range = [-90, 90] # -90, 90
        obj_dst_interval = 0.055 + 0.002  + 0.1            
        box_dst_interval = 0.055 + 0.002 + 0.02*np.sqrt(2)
        obj_box_interval = 0.1 + 0.02*np.sqrt(2)

        obj_xy = hit_or_miss_sample(obj_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
        box_xy = hit_or_miss_sample(obj_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
        dst_xy = hit_or_miss_sample(dst_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
        while np.linalg.norm(dst_xy - obj_xy) < obj_dst_interval or np.linalg.norm(dst_xy - box_xy) < box_dst_interval or np.linalg.norm(obj_xy - box_xy) < obj_box_interval:
            obj_xy = hit_or_miss_sample(obj_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
            box_xy = hit_or_miss_sample(obj_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
            dst_xy = hit_or_miss_sample(dst_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)

        obj_angle = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        obj_quat = np.array([np.cos(np.deg2rad(obj_angle)/2), 0, 0, np.sin(np.deg2rad(obj_angle)/2)])    

        box_angle = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        box_quat = np.array([np.cos(np.deg2rad(box_angle)/2), 0, 0, np.sin(np.deg2rad(box_angle)/2)])            
        
        obj_position = np.hstack([obj_xy, obj_z])
        box_position = np.hstack([box_xy, box_z])
        dst_position = np.hstack([dst_xy, dst_z])
        dst_quat = np.array([1, 0, 0, 0])
    
        BOX_POSE[0] = np.concatenate([obj_position, obj_quat, box_position, box_quat, dst_position, dst_quat])

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0

        return obs, info

    def get_obs(self, ts):
        curr_obs = {}
        self.vis_images = {}

        obs_img_names = ["primary", "secondary", "tertiary", "quaternary", "left_wrist", "right_wrist"]
        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation["images"][cam_name]
            self.vis_images[f"image_{obs_img_names[i]}"] = copy.deepcopy(curr_image)
            curr_image = jnp.array(curr_image)
            curr_obs[f"image_{obs_img_names[i]}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        qpos_numpy = np.array(ts.observation["qpos"])
        qpos = jnp.array(qpos_numpy)
        curr_obs["proprio"] = qpos

        return curr_obs, self.vis_images

    def get_task(self):
        return {
            "language_instruction": ["pick up the red cube and put it in the bucket"],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }    


class Aloha1of4CucumberGymEnv(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        camera_names: List[str],
        im_size: int = 256,
        seed: int = 1234,
        num_obj: int = 4
    ):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in ["primary", "secondary", "left_wrist", "right_wrist"][: len(camera_names)]
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)
        self.pickup_num = 1
        self.num_obj = num_obj

    def step(self, action):
        try:
            ts = self._env.step(action)
        except:
            curr_obs = {}
            vis_images = []

            black = np.zeros((self._im_size, self._im_size, 3), dtype=np.uint8)
            obs_img_names = ["primary", "secondary", "left_wrist", "right_wrist"]
            for i, cam_name in enumerate(self.camera_names):
                curr_image = jnp.array(black.copy())
                curr_obs[f"image_{obs_img_names[i]}"] = curr_image
                vis_images.append(copy.deepcopy(curr_image))

            curr_obs = dl.transforms.resize_images(
                curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
            )

            images = np.concatenate(vis_images, axis=-2)
            info = {"images": images}
            return {"image_primary":black, "image_secondary":black, "image_left_wrist":black, "image_right_wrist":black, "proprio": np.nan}, 0, True, False, info
        obs, images = self.get_obs(ts)
        reward = ts.reward
        info = {"images": images}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        return obs, reward, False, False, info

    def reset(self, **kwargs):
        BUCKET_THICK = 0.002
        BUCKET_RADIUS = 0.055
        OBJ_DST_INTERVAL = 0.1 + BUCKET_RADIUS + BUCKET_THICK
        OBJ_OBJ_INTERVAL = 0.1 + 0.015

        obj_x_range = [-0.1, 0.3]
        dst_x_range = [-0.1, 0.3]
        y_range = [0.3, 0.8] # 0.3
        z_range = [0.015, 0.015]
        dst_ranges = np.vstack([dst_x_range, y_range, z_range])
        obj_angle_range = [0, 180]
        obj_ranges = np.vstack([obj_x_range, y_range, z_range])

        indices = []
        for i in range(self.num_obj):
            for j in range(i):
                indices.append([i, j])
        y_indices = np.array(indices)[:, 0]
        x_indices = np.array(indices)[:, 1]

        dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
        obj_positions = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1], (self.num_obj, 3))
        obj_obj_dist_table = np.linalg.norm(obj_positions[:, None, :2] - obj_positions[:, :2], axis=2) # (num_obj, num_obj)
        obj_dst_dist_table = np.linalg.norm(obj_positions[:, :2] - dst_position[:2], axis=1) # (num_obj)

        obj_obj_mask = obj_obj_dist_table < OBJ_OBJ_INTERVAL
        obj_dst_mask = obj_dst_dist_table < OBJ_DST_INTERVAL

        while np.any(obj_obj_mask[y_indices, x_indices]) or np.any(obj_dst_mask):
            dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
            obj_positions = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1], (self.num_obj, 3))
            obj_obj_dist_table = np.linalg.norm(obj_positions[:, None, :2] - obj_positions[:, :2], axis=2) # (num_obj, num_obj)
            obj_dst_dist_table = np.linalg.norm(obj_positions[:, :2] - dst_position[:2], axis=1) # (num_obj)

            obj_obj_mask = obj_obj_dist_table < OBJ_OBJ_INTERVAL
            obj_dst_mask = obj_dst_dist_table < OBJ_DST_INTERVAL

        obj_angles = np.random.uniform(obj_angle_range[0], obj_angle_range[1], self.num_obj)    
        obj_quats = np.array([np.cos(np.deg2rad(obj_angles)/2), np.zeros(self.num_obj), np.zeros(self.num_obj), np.sin(np.deg2rad(obj_angles)/2)]).T

        obj_poses = []
        for i in range(self.num_obj):
            obj_poses.append(obj_positions[i])
            obj_poses.append(obj_quats[i])

        dst_quat = np.array([1, 0, 0, 0])

        obj_poses.append(dst_position)
        obj_poses.append(dst_quat)

        BOX_POSE[0] = np.concatenate(obj_poses)

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0

        return obs, info

    def get_obs(self, ts):
        curr_obs = {}
        self.vis_images = []

        obs_img_names = ["primary", "secondary", "left_wrist", "right_wrist"]
        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation["images"][cam_name]
            self.vis_images.append(copy.deepcopy(curr_image))
            curr_image = jnp.array(curr_image)
            curr_obs[f"image_{obs_img_names[i]}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        qpos_numpy = np.array(ts.observation["qpos"])
        qpos = jnp.array(qpos_numpy)
        curr_obs["proprio"] = qpos

        return curr_obs, np.concatenate(self.vis_images, axis=-2)

    def get_task(self):
        strings = "pick up one cucumber and put it in the bucket"
        
        return {
            "language_instruction": [strings],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }
    
class Aloha2of4CucumberGymEnv(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        camera_names: List[str],
        im_size: int = 256,
        seed: int = 1234,
        num_obj: int = 4
    ):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in ["primary", "secondary", "left_wrist", "right_wrist"][: len(camera_names)]
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)
        self.pickup_num = 2
        self.num_obj = num_obj

    def step(self, action):
        try:
            ts = self._env.step(action)
        except:
            curr_obs = {}
            vis_images = []

            black = np.zeros((self._im_size, self._im_size, 3), dtype=np.uint8)
            obs_img_names = ["primary", "secondary", "left_wrist", "right_wrist"]
            for i, cam_name in enumerate(self.camera_names):
                curr_image = jnp.array(black.copy())
                curr_obs[f"image_{obs_img_names[i]}"] = curr_image
                vis_images.append(copy.deepcopy(curr_image))

            curr_obs = dl.transforms.resize_images(
                curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
            )

            images = np.concatenate(vis_images, axis=-2)
            info = {"images": images}
            return {"image_primary":black, "image_secondary":black, "image_left_wrist":black, "image_right_wrist":black, "proprio": np.nan}, 0, True, False, info
        obs, images = self.get_obs(ts)
        reward = ts.reward
        info = {"images": images}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        return obs, reward, False, False, info

    def reset(self, **kwargs):
        BUCKET_THICK = 0.002
        BUCKET_RADIUS = 0.055
        OBJ_DST_INTERVAL = 0.1 + BUCKET_RADIUS + BUCKET_THICK
        OBJ_OBJ_INTERVAL = 0.1 + 0.015

        obj_x_range = [-0.1, 0.3]
        dst_x_range = [-0.1, 0.3]
        y_range = [0.3, 0.8] # 0.3
        z_range = [0.015, 0.015]
        dst_ranges = np.vstack([dst_x_range, y_range, z_range])
        obj_angle_range = [0, 180]
        obj_ranges = np.vstack([obj_x_range, y_range, z_range])

        indices = []
        for i in range(self.num_obj):
            for j in range(i):
                indices.append([i, j])
        y_indices = np.array(indices)[:, 0]
        x_indices = np.array(indices)[:, 1]

        dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
        obj_positions = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1], (self.num_obj, 3))
        obj_obj_dist_table = np.linalg.norm(obj_positions[:, None, :2] - obj_positions[:, :2], axis=2) # (num_obj, num_obj)
        obj_dst_dist_table = np.linalg.norm(obj_positions[:, :2] - dst_position[:2], axis=1) # (num_obj)

        obj_obj_mask = obj_obj_dist_table < OBJ_OBJ_INTERVAL
        obj_dst_mask = obj_dst_dist_table < OBJ_DST_INTERVAL

        while np.any(obj_obj_mask[y_indices, x_indices]) or np.any(obj_dst_mask):
            dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
            obj_positions = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1], (self.num_obj, 3))
            obj_obj_dist_table = np.linalg.norm(obj_positions[:, None, :2] - obj_positions[:, :2], axis=2) # (num_obj, num_obj)
            obj_dst_dist_table = np.linalg.norm(obj_positions[:, :2] - dst_position[:2], axis=1) # (num_obj)

            obj_obj_mask = obj_obj_dist_table < OBJ_OBJ_INTERVAL
            obj_dst_mask = obj_dst_dist_table < OBJ_DST_INTERVAL

        obj_angles = np.random.uniform(obj_angle_range[0], obj_angle_range[1], self.num_obj)    
        obj_quats = np.array([np.cos(np.deg2rad(obj_angles)/2), np.zeros(self.num_obj), np.zeros(self.num_obj), np.sin(np.deg2rad(obj_angles)/2)]).T

        obj_poses = []
        for i in range(self.num_obj):
            obj_poses.append(obj_positions[i])
            obj_poses.append(obj_quats[i])

        dst_quat = np.array([1, 0, 0, 0])

        obj_poses.append(dst_position)
        obj_poses.append(dst_quat)

        BOX_POSE[0] = np.concatenate(obj_poses)

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0

        return obs, info

    def get_obs(self, ts):
        curr_obs = {}
        self.vis_images = []

        obs_img_names = ["primary", "secondary", "left_wrist", "right_wrist"]
        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation["images"][cam_name]
            self.vis_images.append(copy.deepcopy(curr_image))
            curr_image = jnp.array(curr_image)
            curr_obs[f"image_{obs_img_names[i]}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        qpos_numpy = np.array(ts.observation["qpos"])
        qpos = jnp.array(qpos_numpy)
        curr_obs["proprio"] = qpos

        return curr_obs, np.concatenate(self.vis_images, axis=-2)

    def get_task(self):
        strings = "pick up two cucumbers and put them in the bucket"
        
        return {
            "language_instruction": [strings],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }
    
class Aloha3of4CucumberGymEnv(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        camera_names: List[str],
        im_size: int = 256,
        seed: int = 1234,
        num_obj: int = 4
    ):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in ["primary", "secondary", "left_wrist", "right_wrist"][: len(camera_names)]
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)
        self.pickup_num = 3
        self.num_obj = num_obj

    def step(self, action):
        try:
            ts = self._env.step(action)
        except:
            curr_obs = {}
            vis_images = []

            black = np.zeros((self._im_size, self._im_size, 3), dtype=np.uint8)
            obs_img_names = ["primary", "secondary", "left_wrist", "right_wrist"]
            for i, cam_name in enumerate(self.camera_names):
                curr_image = jnp.array(black.copy())
                curr_obs[f"image_{obs_img_names[i]}"] = curr_image
                vis_images.append(copy.deepcopy(curr_image))

            curr_obs = dl.transforms.resize_images(
                curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
            )

            images = np.concatenate(vis_images, axis=-2)
            info = {"images": images}
            return {"image_primary":black, "image_secondary":black, "image_left_wrist":black, "image_right_wrist":black, "proprio": np.nan}, 0, True, False, info
        obs, images = self.get_obs(ts)
        reward = ts.reward
        info = {"images": images}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        return obs, reward, False, False, info

    def reset(self, **kwargs):
        BUCKET_THICK = 0.002
        BUCKET_RADIUS = 0.055
        OBJ_DST_INTERVAL = 0.1 + BUCKET_RADIUS + BUCKET_THICK
        OBJ_OBJ_INTERVAL = 0.1 + 0.015

        obj_x_range = [-0.1, 0.3]
        dst_x_range = [-0.1, 0.3]
        y_range = [0.3, 0.8] # 0.3
        z_range = [0.015, 0.015]
        dst_ranges = np.vstack([dst_x_range, y_range, z_range])
        obj_angle_range = [0, 180]
        obj_ranges = np.vstack([obj_x_range, y_range, z_range])

        indices = []
        for i in range(self.num_obj):
            for j in range(i):
                indices.append([i, j])
        y_indices = np.array(indices)[:, 0]
        x_indices = np.array(indices)[:, 1]

        dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
        obj_positions = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1], (self.num_obj, 3))
        obj_obj_dist_table = np.linalg.norm(obj_positions[:, None, :2] - obj_positions[:, :2], axis=2) # (num_obj, num_obj)
        obj_dst_dist_table = np.linalg.norm(obj_positions[:, :2] - dst_position[:2], axis=1) # (num_obj)

        obj_obj_mask = obj_obj_dist_table < OBJ_OBJ_INTERVAL
        obj_dst_mask = obj_dst_dist_table < OBJ_DST_INTERVAL

        while np.any(obj_obj_mask[y_indices, x_indices]) or np.any(obj_dst_mask):
            dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
            obj_positions = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1], (self.num_obj, 3))
            obj_obj_dist_table = np.linalg.norm(obj_positions[:, None, :2] - obj_positions[:, :2], axis=2) # (num_obj, num_obj)
            obj_dst_dist_table = np.linalg.norm(obj_positions[:, :2] - dst_position[:2], axis=1) # (num_obj)

            obj_obj_mask = obj_obj_dist_table < OBJ_OBJ_INTERVAL
            obj_dst_mask = obj_dst_dist_table < OBJ_DST_INTERVAL

        obj_angles = np.random.uniform(obj_angle_range[0], obj_angle_range[1], self.num_obj)    
        obj_quats = np.array([np.cos(np.deg2rad(obj_angles)/2), np.zeros(self.num_obj), np.zeros(self.num_obj), np.sin(np.deg2rad(obj_angles)/2)]).T

        obj_poses = []
        for i in range(self.num_obj):
            obj_poses.append(obj_positions[i])
            obj_poses.append(obj_quats[i])

        dst_quat = np.array([1, 0, 0, 0])

        obj_poses.append(dst_position)
        obj_poses.append(dst_quat)

        BOX_POSE[0] = np.concatenate(obj_poses)

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0

        return obs, info

    def get_obs(self, ts):
        curr_obs = {}
        self.vis_images = []

        obs_img_names = ["primary", "secondary", "left_wrist", "right_wrist"]
        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation["images"][cam_name]
            self.vis_images.append(copy.deepcopy(curr_image))
            curr_image = jnp.array(curr_image)
            curr_obs[f"image_{obs_img_names[i]}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        qpos_numpy = np.array(ts.observation["qpos"])
        qpos = jnp.array(qpos_numpy)
        curr_obs["proprio"] = qpos

        return curr_obs, np.concatenate(self.vis_images, axis=-2)

    def get_task(self):
        strings = "pick up three cucumbers and put them in the bucket"
        
        return {
            "language_instruction": [strings],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }
    
class Aloha4of4CucumberGymEnv(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        camera_names: List[str],
        im_size: int = 256,
        seed: int = 1234,
        num_obj: int = 4
    ):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in ["primary", "secondary", "left_wrist", "right_wrist"][: len(camera_names)]
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)
        self.pickup_num = 4
        self.num_obj = num_obj

    def step(self, action):
        try:
            ts = self._env.step(action)
        except:
            curr_obs = {}
            vis_images = []

            black = np.zeros((self._im_size, self._im_size, 3), dtype=np.uint8)
            obs_img_names = ["primary", "secondary", "left_wrist", "right_wrist"]
            for i, cam_name in enumerate(self.camera_names):
                curr_image = jnp.array(black.copy())
                curr_obs[f"image_{obs_img_names[i]}"] = curr_image
                vis_images.append(copy.deepcopy(curr_image))

            curr_obs = dl.transforms.resize_images(
                curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
            )

            images = np.concatenate(vis_images, axis=-2)
            info = {"images": images}
            return {"image_primary":black, "image_secondary":black, "image_left_wrist":black, "image_right_wrist":black, "proprio": np.nan}, 0, True, False, info
        obs, images = self.get_obs(ts)
        reward = ts.reward
        info = {"images": images}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        return obs, reward, False, False, info

    def reset(self, **kwargs):
        BUCKET_THICK = 0.002
        BUCKET_RADIUS = 0.055
        OBJ_DST_INTERVAL = 0.1 + BUCKET_RADIUS + BUCKET_THICK
        OBJ_OBJ_INTERVAL = 0.1 + 0.015

        obj_x_range = [-0.1, 0.3]
        dst_x_range = [-0.1, 0.3]
        y_range = [0.3, 0.8] # 0.3
        z_range = [0.015, 0.015]
        dst_ranges = np.vstack([dst_x_range, y_range, z_range])
        obj_angle_range = [0, 180]
        obj_ranges = np.vstack([obj_x_range, y_range, z_range])

        indices = []
        for i in range(self.num_obj):
            for j in range(i):
                indices.append([i, j])
        y_indices = np.array(indices)[:, 0]
        x_indices = np.array(indices)[:, 1]

        dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
        obj_positions = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1], (self.num_obj, 3))
        obj_obj_dist_table = np.linalg.norm(obj_positions[:, None, :2] - obj_positions[:, :2], axis=2) # (num_obj, num_obj)
        obj_dst_dist_table = np.linalg.norm(obj_positions[:, :2] - dst_position[:2], axis=1) # (num_obj)

        obj_obj_mask = obj_obj_dist_table < OBJ_OBJ_INTERVAL
        obj_dst_mask = obj_dst_dist_table < OBJ_DST_INTERVAL

        while np.any(obj_obj_mask[y_indices, x_indices]) or np.any(obj_dst_mask):
            dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
            obj_positions = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1], (self.num_obj, 3))
            obj_obj_dist_table = np.linalg.norm(obj_positions[:, None, :2] - obj_positions[:, :2], axis=2) # (num_obj, num_obj)
            obj_dst_dist_table = np.linalg.norm(obj_positions[:, :2] - dst_position[:2], axis=1) # (num_obj)

            obj_obj_mask = obj_obj_dist_table < OBJ_OBJ_INTERVAL
            obj_dst_mask = obj_dst_dist_table < OBJ_DST_INTERVAL

        obj_angles = np.random.uniform(obj_angle_range[0], obj_angle_range[1], self.num_obj)    
        obj_quats = np.array([np.cos(np.deg2rad(obj_angles)/2), np.zeros(self.num_obj), np.zeros(self.num_obj), np.sin(np.deg2rad(obj_angles)/2)]).T

        obj_poses = []
        for i in range(self.num_obj):
            obj_poses.append(obj_positions[i])
            obj_poses.append(obj_quats[i])

        dst_quat = np.array([1, 0, 0, 0])

        obj_poses.append(dst_position)
        obj_poses.append(dst_quat)

        BOX_POSE[0] = np.concatenate(obj_poses)

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0

        return obs, info

    def get_obs(self, ts):
        curr_obs = {}
        self.vis_images = []

        obs_img_names = ["primary", "secondary", "left_wrist", "right_wrist"]
        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation["images"][cam_name]
            self.vis_images.append(copy.deepcopy(curr_image))
            curr_image = jnp.array(curr_image)
            curr_obs[f"image_{obs_img_names[i]}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        qpos_numpy = np.array(ts.observation["qpos"])
        qpos = jnp.array(qpos_numpy)
        curr_obs["proprio"] = qpos

        return curr_obs, np.concatenate(self.vis_images, axis=-2)

    def get_task(self):
        strings = "pick up four cucumbers and put them in the bucket"
        
        return {
            "language_instruction": [strings],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }



class FrankaDualCucumberGymEnv(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        camera_names: List[str],
        im_size: int = 256,
        seed: int = 1234,
    ):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in ["primary", "secondary", "tertiary", "quaternary", "left_wrist", "right_wrist"][: len(camera_names)]
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((16,)) * -1, high=np.ones((16,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((16,)) * -1, high=np.ones((16,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        try:
            ts = self._env.step(action)
        except:
            curr_obs = {}
            vis_images = []

            black = np.zeros((self._im_size, self._im_size, 3), dtype=np.uint8)
            obs_img_names = ["primary", "secondary", "tertiary", "quaternary", "left_wrist", "right_wrist"]
            for i, cam_name in enumerate(self.camera_names):
                curr_image = jnp.array(black.copy())
                curr_obs[f"image_{obs_img_names[i]}"] = curr_image
                vis_images.append(copy.deepcopy(curr_image))

            curr_obs = dl.transforms.resize_images(
                curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
            )

            images = np.concatenate(vis_images, axis=-2)
            info = {"images": images}
            return {"image_primary":black, "image_secondary":black, "image_tertiary":black, "image_quaternary":black, "image_left_wrist":black, "image_right_wrist":black, "proprio": np.nan}, 0, True, False, info
        obs, images = self.get_obs(ts)
        reward = ts.reward
        info = {"images": images}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        return obs, reward, False, False, info

    def reset(self, **kwargs):
        circle_x = 0.295
        circle_y = 0.414
        obj_circle_r = 0.564095655480825
        dst_circle_r = 0.7767498780487515

        rect_center_x = -0.0235
        rect_center_y = 0.9845
        rect_offset = 0.05
        rect_w = 0.272*2 - rect_offset*2
        rect_h = 0.1785*2 - rect_offset*2

        obj_z = 0.842
        dst_z = 0.827
        obj_angle_range = [-90, 90] # -90, 90
        obj_dst_interval = 0.1            
        obj_xy = hit_or_miss_sample(obj_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
        dst_xy = hit_or_miss_sample(dst_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)

        obj_angle = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        obj_quat = np.array([np.cos(np.deg2rad(obj_angle)/2), 0, 0, np.sin(np.deg2rad(obj_angle)/2)])    

        while np.linalg.norm(dst_xy - obj_xy) < 0.055+0.002+obj_dst_interval:
            dst_xy = hit_or_miss_sample(dst_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
        
        obj_position = np.hstack([obj_xy, obj_z])
        dst_position = np.hstack([dst_xy, dst_z])
        dst_quat = np.array([1, 0, 0, 0])
    
        BOX_POSE[0] = np.concatenate([obj_position, obj_quat, dst_position, dst_quat])

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0

        return obs, info

    def get_obs(self, ts):
        curr_obs = {}
        self.vis_images = {}

        obs_img_names = ["primary", "secondary", "tertiary", "quaternary", "left_wrist", "right_wrist"]
        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation["images"][cam_name]
            self.vis_images[f"image_{obs_img_names[i]}"] = copy.deepcopy(curr_image)
            curr_image = jnp.array(curr_image)
            curr_obs[f"image_{obs_img_names[i]}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        qpos_numpy = np.array(ts.observation["qpos"])
        qpos = jnp.array(qpos_numpy)
        curr_obs["proprio"] = qpos

        return curr_obs, self.vis_images

    def get_task(self):
        return {
            "language_instruction": ["pick up the cucumber and put it in the bucket"],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }


class FrankaDualCushionGymEnv(FrankaDualCucumberGymEnv):
    def reset(self, **kwargs):
        angle_range = [-180, 180] #-180, 180
        x_range = [0.1, 0.3] # 0.1, 0.3
        y_range = [0.88, 0.98] # 0.88, 0.98
        z_range = [0.7, 0.7]
        obj_pose = np.array([0.4, 1.25, 1.201, 1.0, 0.0, 0.0, 0.0])

        ranges = np.vstack([x_range, y_range, z_range])
        pos = np.random.uniform(ranges[:, 0], ranges[:, 1])
        angle = np.random.uniform(angle_range[0], angle_range[1])    
        quat = np.array([np.cos(np.deg2rad(angle)/2), 0, 0, np.sin(np.deg2rad(angle)/2)])    
    
        BOX_POSE[0] = np.concatenate([obj_pose, pos, quat])

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0

        return obs, info
    def get_task(self):
        return {
            "language_instruction": ["put the cushion in the empty space"],
        }


class FrankaDualCushionRecoveryGymEnv(FrankaDualCucumberGymEnv):
    def reset(self, **kwargs):
        cardboard_angle_range = [-180, 180] #-180, 180
        cardboard_x_range = [0.1, 0.3] # 0.1, 0.3
        cardboard_y_range = [0.88, 0.98] # 0.88, 0.98
        cardboard_z_range = [0.7, 0.7]

        cushion_angle_range = [-180, 180] #-180, 180
        cushion_x_range = [0.71, 0.81] # 0.1, 0.3
        cushion_y_range = [0.88, 0.98] # 0.88, 0.98
        cushion_z_range = [0.717, 0.717]

        cardboard_ranges = np.vstack([cardboard_x_range, cardboard_y_range, cardboard_z_range])
        cardboard_pos = np.random.uniform(cardboard_ranges[:, 0], cardboard_ranges[:, 1])
        cardboard_angle = np.random.uniform(cardboard_angle_range[0], cardboard_angle_range[1])    
        cardboard_quat = np.array([np.cos(np.deg2rad(cardboard_angle)/2), 0, 0, np.sin(np.deg2rad(cardboard_angle)/2)])    

        cushion_ranges = np.vstack([cushion_x_range, cushion_y_range, cushion_z_range])
        cushion_pos = np.random.uniform(cushion_ranges[:, 0], cushion_ranges[:, 1])
        cushion_angle = np.random.uniform(cushion_angle_range[0], cushion_angle_range[1])    
        cushion_quat = np.array([np.cos(np.deg2rad(cushion_angle)/2), 0, 0, np.sin(np.deg2rad(cushion_angle)/2)])            
    
        obj_pose = np.array([0.4, 1.25, 1.201, 1.0, 0.0, 0.0, 0.0])
        BOX_POSE[0] = np.concatenate([obj_pose, cushion_pos, cushion_quat, cardboard_pos, cardboard_quat])

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0

        return obs, info
    def get_task(self):
        return {
            "language_instruction": ["put the cushion in the empty space"],
        }


class FrankaDualBimanualLeftCucumberGymEnv(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        camera_names: List[str],
        im_size: int = 256,
        seed: int = 1234,
    ):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in ["primary", "secondary", "tertiary", "quaternary", "left_wrist", "right_wrist"][: len(camera_names)]
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        try:
            ts = self._env.step(action)
        except:
            curr_obs = {}
            vis_images = []

            black = np.zeros((self._im_size, self._im_size, 3), dtype=np.uint8)
            obs_img_names = ["primary", "secondary", "tertiary", "quaternary", "left_wrist", "right_wrist"]
            for i, cam_name in enumerate(self.camera_names):
                curr_image = jnp.array(black.copy())
                curr_obs[f"image_{obs_img_names[i]}"] = curr_image
                vis_images.append(copy.deepcopy(curr_image))

            curr_obs = dl.transforms.resize_images(
                curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
            )

            images = np.concatenate(vis_images, axis=-2)
            info = {"images": images}
            return {"image_primary":black, "image_secondary":black, "image_tertiary":black, "image_quaternary":black, "image_left_wrist":black, "image_right_wrist":black, "proprio": np.nan}, 0, True, False, info
        obs, images = self.get_obs(ts)
        reward = ts.reward
        info = {"images": images}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        return obs, reward, False, False, info

    def reset(self, **kwargs):
        rect_center_x = -0.0235
        rect_center_y = 0.9845
        rect_half_w = 0.272
        rect_half_h = 0.1785    
        rect_offset = 0.05
        valid_w = (rect_half_w - rect_offset) * 2 
        valid_h = (rect_half_h - rect_offset) * 2 
        valid_x_start = rect_offset + rect_center_x - rect_half_w
        valid_x_end = rect_center_x + rect_half_w - rect_offset
        valid_y_start = rect_offset + rect_center_y - rect_half_h
        valid_y_end = rect_center_y + rect_half_h - rect_offset

        obj_x_range_left = [valid_x_start, valid_x_start+valid_w/3]
        obj_x_range_right = [valid_x_start+valid_w*2/3, valid_x_end]    
        dst_x_range = [valid_x_start+valid_w/3, valid_x_start+valid_w*2/3]

        obj_y_range_left = [valid_y_start, valid_y_end]
        obj_y_range_right = [valid_y_start, valid_y_end]
        dst_y_range = [valid_y_start, valid_y_end]

        obj_z_left = 0.842
        obj_z_right = 0.842
        dst_z = 0.827

        obj_angle_range = [-90, 90] # -90, 90
        obj_dst_interval = 0.055 + 0.002  + 0.1            

        obj_xy_ranges_left = np.vstack([obj_x_range_left, obj_y_range_left])
        obj_xy_ranges_right = np.vstack([obj_x_range_right, obj_y_range_right])
        dst_xy_ranges = np.vstack([dst_x_range, dst_y_range])

        dst_xy = np.random.uniform(dst_xy_ranges[:, 0], dst_xy_ranges[:, 1])    
        obj_xy_left = np.random.uniform(obj_xy_ranges_left[:, 0], obj_xy_ranges_left[:, 1])
        obj_xy_right = np.random.uniform(obj_xy_ranges_right[:, 0], obj_xy_ranges_right[:, 1])

        while np.linalg.norm(dst_xy - obj_xy_left) < obj_dst_interval:
            obj_xy_left = np.random.uniform(obj_xy_ranges_left[:, 0], obj_xy_ranges_left[:, 1])
        while np.linalg.norm(dst_xy - obj_xy_right) < obj_dst_interval:
            obj_xy_right = np.random.uniform(obj_xy_ranges_right[:, 0], obj_xy_ranges_right[:, 1])

        obj_angle_left = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        obj_quat_left = np.array([np.cos(np.deg2rad(obj_angle_left)/2), 0, 0, np.sin(np.deg2rad(obj_angle_left)/2)])    
        obj_angle_right = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        obj_quat_right = np.array([np.cos(np.deg2rad(obj_angle_right)/2), 0, 0, np.sin(np.deg2rad(obj_angle_right)/2)])    
        dst_quat = np.array([1, 0, 0, 0])

        obj_position_left = np.hstack([obj_xy_left, obj_z_left])
        obj_position_right = np.hstack([obj_xy_right, obj_z_right])
        dst_position = np.hstack([dst_xy, dst_z])    

        BOX_POSE[0] = np.concatenate([obj_position_left, obj_quat_left, obj_position_right, obj_quat_right, dst_position, dst_quat])

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0

        return obs, info

    def get_obs(self, ts):
        curr_obs = {}
        self.vis_images = {}

        obs_img_names = ["primary", "secondary", "tertiary", "quaternary", "left_wrist", "right_wrist"]
        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation["images"][cam_name]
            self.vis_images[f"image_{obs_img_names[i]}"] = copy.deepcopy(curr_image)
            curr_image = jnp.array(curr_image)
            curr_obs[f"image_{obs_img_names[i]}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        qpos_numpy = np.array(ts.observation["qpos"])
        qpos = jnp.array(qpos_numpy)
        curr_obs["proprio"] = qpos

        return curr_obs, self.vis_images

    def get_task(self):
        return {
            "language_instruction": ["pick up the cucumber and the red cube and put them in the bucket"],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }
    

class FrankaDualBimanualQuadrupleLeftCucumberFirstGymEnv(FrankaDualBimanualLeftCucumberGymEnv):
    def get_task(self):
        return {
            "language_instruction": ["pick up objects and put them in a bucket cucumber first"],
        }
    
class FrankaDualBimanualQuadrupleRightCubeFirstGymEnv(FrankaDualBimanualLeftCucumberGymEnv):
    def get_task(self):
        return {
            "language_instruction": ["pick up objects and put them in a bucket cube first"],
        }


class FrankaDualBimanualLeftCubeGymEnv(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        camera_names: List[str],
        im_size: int = 256,
        seed: int = 1234,
    ):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in ["primary", "secondary", "tertiary", "quaternary", "left_wrist", "right_wrist"][: len(camera_names)]
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        try:
            ts = self._env.step(action)
        except:
            curr_obs = {}
            vis_images = []

            black = np.zeros((self._im_size, self._im_size, 3), dtype=np.uint8)
            obs_img_names = ["primary", "secondary", "tertiary", "quaternary", "left_wrist", "right_wrist"]
            for i, cam_name in enumerate(self.camera_names):
                curr_image = jnp.array(black.copy())
                curr_obs[f"image_{obs_img_names[i]}"] = curr_image
                vis_images.append(copy.deepcopy(curr_image))

            curr_obs = dl.transforms.resize_images(
                curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
            )

            images = np.concatenate(vis_images, axis=-2)
            info = {"images": images}
            return {"image_primary":black, "image_secondary":black, "image_tertiary":black, "image_quaternary":black, "image_left_wrist":black, "image_right_wrist":black, "proprio": np.nan}, 0, True, False, info
        obs, images = self.get_obs(ts)
        reward = ts.reward
        info = {"images": images}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        return obs, reward, False, False, info

    def reset(self, **kwargs):
        rect_center_x = -0.0235
        rect_center_y = 0.9845
        rect_half_w = 0.272
        rect_half_h = 0.1785    
        rect_offset = 0.05
        valid_w = (rect_half_w - rect_offset) * 2 
        valid_h = (rect_half_h - rect_offset) * 2 
        valid_x_start = rect_offset + rect_center_x - rect_half_w
        valid_x_end = rect_center_x + rect_half_w - rect_offset
        valid_y_start = rect_offset + rect_center_y - rect_half_h
        valid_y_end = rect_center_y + rect_half_h - rect_offset

        obj_x_range_left = [valid_x_start, valid_x_start+valid_w/3]
        obj_x_range_right = [valid_x_start+valid_w*2/3, valid_x_end]    
        dst_x_range = [valid_x_start+valid_w/3, valid_x_start+valid_w*2/3]

        obj_y_range_left = [valid_y_start, valid_y_end]
        obj_y_range_right = [valid_y_start, valid_y_end]
        dst_y_range = [valid_y_start, valid_y_end]

        obj_z_left = 0.842
        obj_z_right = 0.842
        dst_z = 0.827

        obj_angle_range = [-90, 90] # -90, 90
        obj_dst_interval = 0.055 + 0.002  + 0.1            

        obj_xy_ranges_left = np.vstack([obj_x_range_left, obj_y_range_left])
        obj_xy_ranges_right = np.vstack([obj_x_range_right, obj_y_range_right])
        dst_xy_ranges = np.vstack([dst_x_range, dst_y_range])

        dst_xy = np.random.uniform(dst_xy_ranges[:, 0], dst_xy_ranges[:, 1])    
        obj_xy_left = np.random.uniform(obj_xy_ranges_left[:, 0], obj_xy_ranges_left[:, 1])
        obj_xy_right = np.random.uniform(obj_xy_ranges_right[:, 0], obj_xy_ranges_right[:, 1])

        while np.linalg.norm(dst_xy - obj_xy_left) < obj_dst_interval:
            obj_xy_left = np.random.uniform(obj_xy_ranges_left[:, 0], obj_xy_ranges_left[:, 1])
        while np.linalg.norm(dst_xy - obj_xy_right) < obj_dst_interval:
            obj_xy_right = np.random.uniform(obj_xy_ranges_right[:, 0], obj_xy_ranges_right[:, 1])

        obj_angle_left = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        obj_quat_left = np.array([np.cos(np.deg2rad(obj_angle_left)/2), 0, 0, np.sin(np.deg2rad(obj_angle_left)/2)])    
        obj_angle_right = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        obj_quat_right = np.array([np.cos(np.deg2rad(obj_angle_right)/2), 0, 0, np.sin(np.deg2rad(obj_angle_right)/2)])    
        dst_quat = np.array([1, 0, 0, 0])

        obj_position_left = np.hstack([obj_xy_left, obj_z_left])
        obj_position_right = np.hstack([obj_xy_right, obj_z_right])
        dst_position = np.hstack([dst_xy, dst_z])    
    
        BOX_POSE[0] = np.concatenate([obj_position_right, obj_quat_right, obj_position_left, obj_quat_left, dst_position, dst_quat])

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0

        return obs, info

    def get_obs(self, ts):
        curr_obs = {}
        self.vis_images = {}

        obs_img_names = ["primary", "secondary", "tertiary", "quaternary", "left_wrist", "right_wrist"]
        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation["images"][cam_name]
            self.vis_images[f"image_{obs_img_names[i]}"] = copy.deepcopy(curr_image)
            curr_image = jnp.array(curr_image)
            curr_obs[f"image_{obs_img_names[i]}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        qpos_numpy = np.array(ts.observation["qpos"])
        qpos = jnp.array(qpos_numpy)
        curr_obs["proprio"] = qpos

        return curr_obs, self.vis_images

    def get_task(self):
        return {
            "language_instruction": ["pick up the cucumber and the red cube and put them in the bucket"],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }    


class FrankaDualBimanualQuadrupleLeftCubeFirst(FrankaDualBimanualLeftCubeGymEnv):
    def get_task(self):
        return {
            "language_instruction": ["pick up objects and put them in a bucket cube first"],
        }


class FrankaDualBimanualQuadrupleRightCucumberFirst(FrankaDualBimanualLeftCubeGymEnv):
    def get_task(self):
        return {
            "language_instruction": ["pick up objects and put them in a bucket cucumber first"],
        }


# register gym environments
gym.register(
    "aloha-sim-cube-v0",
    entry_point=lambda: AlohaGymEnv(
        make_sim_env("sim_transfer_cube"), camera_names=["top"]
    ),
)

gym.register(
    "aloha-sim-cube-v1",
    entry_point=lambda: AlohaCubeGymEnv(
        make_sim_env("sim_put_cube_in_bucket_on_static_aloha"), camera_names=["top", "angle", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "aloha-sim-cucumber-v0",
    entry_point=lambda: AlohaCucumberGymEnv(
        make_sim_env("sim_put_cucumber_in_bucket_on_static_aloha"), camera_names=["top", "angle", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "aloha-sim-couple-cucumber-v0",
    entry_point=lambda: AlohaCoupleCucumberGymEnv(
        make_sim_env("sim_put_couple_in_bucket_on_static_aloha"), camera_names=["top", "angle", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "aloha-sim-couple-cube-v0",
    entry_point=lambda: AlohaCoupleCubeGymEnv(
        make_sim_env("sim_put_couple_in_bucket_on_static_aloha"), camera_names=["top", "angle", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "aloha-sim-1of4-cucumber-v0",
    entry_point=lambda: Aloha1of4CucumberGymEnv(
        make_sim_env("sim_put_multiple_cucumber_in_bucket_on_static_aloha", pickup_num=1), camera_names=["top", "angle", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "aloha-sim-2of4-cucumber-v0",
    entry_point=lambda: Aloha2of4CucumberGymEnv(
        make_sim_env("sim_put_multiple_cucumber_in_bucket_on_static_aloha", pickup_num=2), camera_names=["top", "angle", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "aloha-sim-3of4-cucumber-v0",
    entry_point=lambda: Aloha3of4CucumberGymEnv(
        make_sim_env("sim_put_multiple_cucumber_in_bucket_on_static_aloha", pickup_num=3), camera_names=["top", "angle", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "aloha-sim-4of4-cucumber-v0",
    entry_point=lambda: Aloha4of4CucumberGymEnv(
        make_sim_env("sim_put_multiple_cucumber_in_bucket_on_static_aloha", pickup_num=4), camera_names=["top", "angle", "left_wrist", "right_wrist"]
    ),
)





gym.register(
    "mobile-aloha-sim-cucumber-v0",
    entry_point=lambda: MobileAlohaCucumberGymEnv(
        make_sim_env("sim_put_cucumber_in_bucket_on_mobile_aloha"), camera_names=["angle", "top", "side", "back", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "mobile-aloha-sim-couple-cucumber-v0",
    entry_point=lambda: MobileAlohaCoupleCucumberGymEnv(
        make_sim_env("sim_put_couple_in_bucket_on_mobile_aloha"), camera_names=["angle", "top", "side", "back", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "mobile-aloha-sim-couple-cube-v0",
    entry_point=lambda: MobileAlohaCoupleCubeGymEnv(
        make_sim_env("sim_put_couple_in_bucket_on_mobile_aloha"), camera_names=["angle", "top", "side", "back", "left_wrist", "right_wrist"]
    ),
)





gym.register(
    "franka-dual-sim-cucumber-v0",
    entry_point=lambda: FrankaDualCucumberGymEnv(
        make_sim_env("sim_put_cucumber_in_bucket_on_franka_dual"), camera_names=["angle", "top", "side", "back", "left_wrist", "right_wrist"]
        # make_sim_env("sim_put_cucumber_in_bucket_on_franka_dual"), camera_names=["left_wrist", "right_wrist"]
    ),
)

gym.register(
    "franka-dual-sim-cushion-v-v0",
    entry_point=lambda: FrankaDualCushionGymEnv(
        make_sim_env("sim_put_cushion_in_cardboard_v_on_franka_dual"), camera_names=["angle", "top", "side", "back", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "franka-dual-sim-cushion-h-v0",
    entry_point=lambda: FrankaDualCushionGymEnv(
        make_sim_env("sim_put_cushion_in_cardboard_h_on_franka_dual"), camera_names=["angle", "top", "side", "back", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "franka-dual-sim-cushion-v-eval-v0",
    entry_point=lambda: FrankaDualCushionGymEnv(
        make_sim_env("sim_put_cushion_in_cardboard_v_eval_on_franka_dual"), camera_names=["angle", "top", "side", "back", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "franka-dual-sim-cushion-h-eval-v0",
    entry_point=lambda: FrankaDualCushionGymEnv(
        make_sim_env("sim_put_cushion_in_cardboard_h_eval_on_franka_dual"), camera_names=["angle", "top", "side", "back", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "franka-dual-sim-cushion-v-recovery-v0",
    entry_point=lambda: FrankaDualCushionRecoveryGymEnv(
        make_sim_env("sim_put_cushion_in_cardboard_v_recovery_on_franka_dual"), camera_names=["angle", "top", "side", "back", "left_wrist", "right_wrist"]
    ),
)




gym.register(
    "franka-dual-sim-couple-cucumber-v0",
    entry_point=lambda: MobileAlohaCoupleCucumberGymEnv(
        make_sim_env("sim_put_couple_in_bucket_on_franka_dual"), camera_names=["angle", "top", "side", "back", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "franka-dual-sim-couple-cube-v0",
    entry_point=lambda: MobileAlohaCoupleCubeGymEnv(
        make_sim_env("sim_put_couple_in_bucket_on_franka_dual"), camera_names=["angle", "top", "side", "back", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "franka-dual-bimanual-sim-left-cucumber-v0",
    entry_point=lambda: FrankaDualBimanualLeftCucumberGymEnv(
        make_sim_env("sim_put_quadruple_in_bucket_on_franka_dual_bimanual"), camera_names=["angle", "top", "side", "back", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "franka-dual-bimanual-sim-left-cucumber-first-v0",
    entry_point=lambda: FrankaDualBimanualQuadrupleLeftCucumberFirstGymEnv(
        make_sim_env("sim_put_quadruple_in_bucket_on_franka_dual_bimanual"), camera_names=["angle", "top", "side", "back", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "franka-dual-bimanual-sim-right-cube-first-v0",
    entry_point=lambda: FrankaDualBimanualQuadrupleRightCubeFirstGymEnv(
        make_sim_env("sim_put_quadruple_in_bucket_on_franka_dual_bimanual"), camera_names=["angle", "top", "side", "back", "left_wrist", "right_wrist"]
    ),
)


gym.register(
    "franka-dual-bimanual-sim-left-cube-v0",
    entry_point=lambda: FrankaDualBimanualLeftCubeGymEnv(
        make_sim_env("sim_put_quadruple_in_bucket_on_franka_dual_bimanual"), camera_names=["angle", "top", "side", "back", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "franka-dual-bimanual-sim-left-cube-first-v0",
    entry_point=lambda: FrankaDualBimanualQuadrupleLeftCubeFirst(
        make_sim_env("sim_put_quadruple_in_bucket_on_franka_dual_bimanual"), camera_names=["angle", "top", "side", "back", "left_wrist", "right_wrist"]
    ),
)

gym.register(
    "franka-dual-bimanual-sim-right-cucumber-first-v0",
    entry_point=lambda: FrankaDualBimanualQuadrupleRightCucumberFirst(
        make_sim_env("sim_put_quadruple_in_bucket_on_franka_dual_bimanual"), camera_names=["angle", "top", "side", "back", "left_wrist", "right_wrist"]
    ),
)