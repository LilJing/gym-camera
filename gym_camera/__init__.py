from gym.envs.registration import register

from gym_camera.multi_camera_env import Multi_Camera_Env
from gym_camera.multi_camera_env1 import Multi_Camera_Env_v1

register(
    id='MultiCamEnv-v0',
    entry_point='gym_camera:Multi_Camera_Env',
    max_episode_steps=500,
)

register(
    id='MultiCamEnv-v1',
    entry_point='gym_camera:Multi_Camera_Env_v1',
    max_episode_steps=500,
)
