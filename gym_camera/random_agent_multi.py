import argparse
import gym_camera
import gym
from gym import wrappers
import cv2
import time
import numpy as np

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='MultiCamEnv-v1',
                        help='Select the environment to run')
    parser.add_argument("-r", "--render", default=True, metavar='G', help='show env using cv2')
    args = parser.parse_args()
    env = gym.make(args.env_id)

    agents_num = len(env.action_space)
    agents = [RandomAgent(env.action_space[i]) for i in range(agents_num)]
    # agent_0 = RandomAgent(env.action_space[0])
    # agent_1 = RandomAgent(env.action_space[1])

    episode_count = 1

    done = False
    Total_rewards = np.zeros(agents_num)
    for epi in range(episode_count):
        env.seed(epi)
        obs = env.reset()
        count_step = 0
        step = 0
        t0 = time.time()
        C_rewards = np.zeros(agents_num)
        while True:

            actions = [agents[i].act(obs[i]) for i in range(agents_num)]
            obs, rewards, done, _ = env.step(actions)
            step += 1
            C_rewards += rewards
            count_step += 1
            if args.render:
                env.render()
                # img = env.render(mode='rgb_array')
                # #  img = img[..., ::-1]  # bgr->rgb
                # cv2.imshow('show', img)
                # cv2.waitKey(1)
            if done:
                # print(step)
                fps = count_step / (time.time() - t0)
                Total_rewards += C_rewards
                print ('Fps:' + str(fps), 'R:'+str(C_rewards), 'R_ave:'+str(Total_rewards/(epi+1)))
                C_rewards = np.zeros(agents_num)
                break

    # Close the env and write monitor result info to disk
    env.close()
    plt.close('all')


