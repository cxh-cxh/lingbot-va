
from typing import List, Optional  # noqa: UP035

import einops
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override



from typing import Optional, List
import dm_env
import numpy as np

#this is a ROS package
import cv2
import torch

from threading import Thread
import copy
#this is  a camera name list for config
CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']

ros_config = {
    "img_front_topic": "/camera_f/color/image_raw",
    "img_left_topic": "/camera_l/color/image_raw",
    "img_right_topic": "/camera_r/color/image_raw",

    "img_front_depth_topic": "/camera_f/depth/image_raw",
    "img_left_depth_topic": "/camera_l/depth/image_raw",
    "img_right_depth_topic": "/camera_r/depth/image_raw",

    "puppet_arm_left_topic": "/puppet/joint_left",
    "puppet_arm_right_topic": "/puppet/joint_right",
    "puppet_eef_left_topic":"/puppet/end_pose_left",
    "puppet_eef_right_topic":"/puppet/end_pose_right",

    "puppet_arm_left_cmd_topic": "/master/joint_left",
    "puppet_arm_right_cmd_topic": "/master/joint_right",
    "puppet_eef_left_cmd_topic": "pos_cmd_left",
    "puppet_eef_right_cmd_topic": "pos_cmd_right",

    "robot_base_topic": "/odom_raw",
    "robot_base_cmd_topic": "/cmd_vel",
    "use_robot_base": False,

    "publish_rate": 30,
    "ctrl_freq": 25,
    "state_dim": 14,
    "chunk_size": 64,
    "arm_steps_length": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2],

    "use_actions_interpolation": False,
    "use_depth_image": False,

    "use_eef": False,
    "disable_puppet_arm": False,
    "disable_robot_base": False,
}

# Interpolate the actions to make the robot move smoothly
def interpolate_action(args, prev_action, cur_action):
    steps = np.concatenate((np.array(args["arm_steps_length"]), np.array(args["arm_steps_length"])), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]

class PiperRealEnv:
    """
    Environment for real robot bi-manual manipulation
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
    """

    def __init__(self, init_node, *, reset_pos:Optional[List[float]] = None, setup_robots: bool = False):
        # self.action = None
        self.pre_action = np.zeros(ros_config['state_dim'])

    def spin(self):
        pass
    def setup_robots(self):
        pass

    def reset(self,*, fake=False):
        if not fake:
            left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156,
                     -0.00286102294921875, 0.00095367431640625, 0.00001] #3.557830810546875
            right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656,
                      -0.00476837158203125, -0.00209808349609375, 1.543]
            left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156,
                     -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
            right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156,
                      -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]
            print("reset")

            # Initialize the previous action to be the initial robot state

            self.pre_action[:14] = np.array(
                [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156,
                 -0.00286102294921875,
                 0.00095367431640625, -0.3393220901489258] +
                [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156,
                 -0.00286102294921875,
                 0.00095367431640625, -0.3397035598754883]
            )

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation()
        )

    def get_reward(self):
        return 0

    def get_observation(self):

        obs = {
                    'qpos': np.ones(14),
                    'eef': np.ones(14),
                    'images':
                        {
                            "cam_high": np.ones((480,640,3),dtype='uint8'),
                            "cam_right_wrist":  np.ones((480,640,3),dtype='uint8'),
                            "cam_left_wrist":  np.ones((480,640,3),dtype='uint8'),
                        },
                }
            
        return obs


    def step(self, action, STOP=False):
        interp_actions = None

        # 如果STOP的话直接跳过动作发布
        if STOP:
            # 可以选择直接返回当前观测，或执行一个空动作
            print("[STOP] skipping action publish.")
            return dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=self.get_reward(),
                discount=None,
                observation=self.get_observation()
            )

        if ros_config["use_actions_interpolation"]:
            print(f"use_actions_interpolation")
            interp_actions = interpolate_action(ros_config, self.pre_action, action)
        else:
            interp_actions = action[np.newaxis, :]

        # Execute the interpolated actions one by one
        for act in interp_actions:
            state_len = int(len(act) / 2)
            left_action = act[:state_len]
            right_action = act[state_len:]

            if  not ros_config["disable_puppet_arm"]:
                print(left_action, right_action)  # puppet_arm_publish_continuous_thread

            if ros_config["use_robot_base"]:
                vel_action = act[14:16]
                print(vel_action)

        self.pre_action = action.copy()

        # get next frame obs
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID, 
            reward=self.get_reward(), 
            discount=None, 
            observation=self.get_observation()
        )




def make_real_env(init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True) -> PiperRealEnv:
    return PiperRealEnv(init_node, reset_pos=reset_position, setup_robots=setup_robots)
class PiperRealEnvironment(_environment.Environment):
    """An environment for an Aloha robot on real hardware."""

    def __init__(
        self,
        reset_position: Optional[List[float]] = None,  # noqa: UP006,UP007
        render_height: int = 224,
        render_width: int = 224,
        prompt: str = "",
    ) -> None:
        self._env = make_real_env(init_node=True, reset_position=reset_position)
        self._prompt = prompt
        self._render_height = render_height
        self._render_width = render_width
        self._ts = None
        self.save_obs = True
        self.frame_cnt = 0

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = copy.deepcopy(self._ts.observation)
        
        for k in list(obs["images"].keys()):
            if "_depth" in k:
                del obs["images"][k]

        for cam_name in obs["images"]:
            print(cam_name,obs["images"][cam_name].shape)
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
            )
            # obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")
            
        #normalization for qpos puppet gript: TODO

        # 保存观察结果
        if self.save_obs:
            self.frame_cnt = self.frame_cnt+1
        print("main obs")
        return {
            "state": obs["qpos"],
            "eef":obs["eef"],
            "images": obs["images"],
            "prompt": self._prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        # 如果actions在字典中，则保存动作
        print("main action apply")
        stop_flag = action.get("STOP", False)
        print(f"STOP_SIGNAL: {stop_flag}")
        if "actions" in action:
            self._ts = self._env.step(action["actions"], STOP=stop_flag)
        else:
            self._ts = self._env.step(None, STOP=stop_flag)
        

