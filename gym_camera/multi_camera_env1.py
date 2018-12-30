'''multi env for recall reward'''

import gym
from gym import spaces
from gym.utils import seeding

import cv2
import math
import numpy as np

class Multi_Camera_Env_v1(gym.Env):
    def __init__(self,
                 cameras_number = 3,
                 targets_number = 5,
                 get_trace = False,
                 traces_len = 3,
                 length = 80,
                 angle = 90):

        self.targets_number = targets_number
        self.cameras_number = cameras_number

        self.get_trace = get_trace
        self.traces = []
        self.traces_len = traces_len

        self.init_states = []
        self.goal_states = []
        self.camera_positions = []

        self.num_actions = 3
        self.angle = angle
        self.rotate_angle = 30
        self.length = length
        self.dmax = self.length * 1.4

        self.action_space = spaces.Discrete(self.num_actions)
        self.all_actions = list(range(self.action_space.n))

        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.cameras_number, self.targets_number, 2))  # 2 : distance and angle

        self.total_steps = 100
        self.current_step = 0
        self.done = False

        for i in range(self.targets_number):
            self.init_states.append([np.random.randint(1,self.length), np.random.randint(1,self.length)])
            self.goal_states.append([np.random.randint(1,self.length), np.random.randint(1,self.length)])

        for i in range(self.cameras_number):
            if i%1 == 0: #first edge
                self.camera_positions.append([0, np.random.randint(1,self.length)])
            elif i%2 == 0: #second edge
                self.camera_positions.append([np.random.randint(1, self.length), 0])
            elif i%3 == 0: #third edge
                self.camera_positions.append([np.random.randint(1, self.length), np.random.randint(1, self.length)])

        self.target_states = self.init_states

        self.abs_angles = []
        for i in range(self.cameras_number):
            self.abs_angles.append(np.random.randint(0, self.angle))

    def reset(self):
        self.observations = np.zeros((self.cameras_number, self.targets_number, 2))
        for i in range(self.cameras_number):
            for j in range(self.targets_number):
                dy = self.init_states[j][1] - self.camera_positions[i][1]
                dx = self.init_states[j][0] - self.camera_positions[i][0]
                if self.camera_positions[i][0] == 0:
                   target_angle = 180 * math.atan2(dx, dy)/math.pi
                else:
                    target_angle = 180 * math.atan2(dy, dx) / math.pi
                if self.abs_angles[i] <= target_angle <= self.abs_angles[i] + self.angle:
                    d = np.sqrt(dx * dx + dy * dy)
                    mu = d / self.length
                    noise = np.random.normal(mu, 0.001)
                    d_norm = (d + noise - self.dmax / 2) / self.dmax / 2
                    self.observations[i][j][0] = d_norm
                    target_angle_norm = (target_angle - self.angle) / self.angle
                    self.observations[i][j][1] = target_angle_norm

        return self.observations


    def step(self, actions):
        self.current_step += 1
        # if self.current_step % self.total_steps == 0:
        #     done = True
        # else:
        #     done = False

        for i in range(self.targets_number):
            target_dy = self.goal_states[i][1] - self.target_states[i][1]
            target_dx = self.goal_states[i][0] - self.target_states[i][0]
            # print('dx', dx)  # target_state (x, y)
            # print('dy', dy)
            if target_dy != 0:
                self.target_states[i][1] += int(target_dy / abs(target_dy))
            elif target_dx != 0:
                self.target_states[i][0] += int(target_dx / abs(target_dx))
            else:
                self.target_states[i][0] = np.random.randint(1, self.length)
                self.target_states[i][1] = np.random.randint(1, self.length)
                self.goal_states[i][0] = np.random.randint(1, self.length)
                self.goal_states[i][1] = np.random.randint(1, self.length)

        for i in range(self.cameras_number):
            if actions[i] == -1 :
                self.abs_angles[i] -= self.rotate_angle
                if self.abs_angles[i] < 0 or self.abs_angles[i] > 180:
                    self.abs_angles[i] += self.rotate_angle
            elif actions[i] == 1:
                self.abs_angles[i] += self.rotate_angle
                if self.abs_angles[i] < 0 or self.abs_angles[i] > 180:
                    self.abs_angles[i] -= self.rotate_angle

        self.targets_observed = np.zeros((self.cameras_number, self.targets_number))
        self.observations = np.zeros((self.cameras_number, self.targets_number, 2)) - 2
        for i in range(self.cameras_number):
            for j in range(self.targets_number):
                dy = self.init_states[j][1] - self.camera_positions[i][1]
                dx = self.init_states[j][0] - self.camera_positions[i][0]
                if self.camera_positions[i][0] == 0:
                   target_angle = 180 * math.atan2(dx, dy)/math.pi
                else:
                    target_angle = 180 * math.atan2(dy, dx) / math.pi
                if self.abs_angles[i] <= target_angle <= self.abs_angles[i] + self.angle:
                    d = np.sqrt(dx*dx + dy*dy)
                    mu = d/self.length
                    noise = np.random.normal(mu, 0.001)
                    d_norm = (d + noise - self.dmax/2)/self.dmax/2
                    self.observations[i][j][0] = d_norm
                    target_angle_norm = (target_angle - self.angle) / self.angle
                    self.observations[i][j][1] = target_angle_norm

                    self.targets_observed[i][j] = 1

        observed_num = np.sum(self.targets_observed, 1)
        self.done = any([observed_num[k] < 1 for k in range(len(observed_num))])  # done is False when each camera can see at least one target
        observed_times = np.sum(self.targets_observed, 0)
        observed_ = [1 / observed_times[i] if observed_times[i] != 0 else 0 for i in range(len(observed_times))]

        for i in range(self.cameras_number):
            self.targets_observed[i] *= observed_
        r1 = np.sum(self.targets_observed, 1)/self.targets_number
        r2 = observed_num / self.targets_number
        observed_reward = r1 + r2
        return self.observations, observed_reward, self.done


class One_Camera_Env_v1(gym.Env):
    def __init__(self,
                 cameras_number = 1,
                 targets_number = 5,
                 get_trace = False,
                 traces_len = 3,
                 length = 80,
                 angle = 90):

        self.targets_number = targets_number
        self.cameras_number = cameras_number

        self.get_trace = get_trace
        self.traces = []
        self.traces_len = traces_len

        self.init_states = []
        self.goal_states = []
        self.camera_positions = []

        self.num_actions = 3
        self.angle = angle
        self.rotate_angle = 30
        self.length = length
        self.dmax = self.length * 1.4

        self.action_space = spaces.Discrete(self.num_actions)
        self.all_actions = list(range(self.action_space.n))

        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.cameras_number, self.targets_number, 2))  # 2 : distance and angle

        self.total_steps = 100
        self.current_step = 0
        self.done = False

        for i in range(self.targets_number):
            self.init_states.append([np.random.randint(1,self.length), np.random.randint(1,self.length)])
            self.goal_states.append([np.random.randint(1,self.length), np.random.randint(1,self.length)])

        for i in range(self.cameras_number):
            if i%1 == 0: #first edge
                self.camera_positions.append([0, np.random.randint(1,self.length)])
            elif i%2 == 0: #second edge
                self.camera_positions.append([np.random.randint(1, self.length), 0])
            elif i%3 == 0: #third edge
                self.camera_positions.append([np.random.randint(1, self.length), np.random.randint(1, self.length)])

        self.target_states = self.init_states

        self.abs_angles = []
        for i in range(self.cameras_number):
            self.abs_angles.append(np.random.randint(0, self.angle))

    def reset(self):
        self.observations = np.zeros((self.cameras_number, self.targets_number, 2))
        for i in range(self.cameras_number):
            for j in range(self.targets_number):
                dy = self.init_states[j][1] - self.camera_positions[i][1]
                dx = self.init_states[j][0] - self.camera_positions[i][0]
                if self.camera_positions[i][0] == 0:
                   target_angle = 180 * math.atan2(dx, dy)/math.pi
                else:
                    target_angle = 180 * math.atan2(dy, dx) / math.pi
                if self.abs_angles[i] <= target_angle <= self.abs_angles[i] + self.angle:
                    d = np.sqrt(dx * dx + dy * dy)
                    mu = d / self.length
                    noise = np.random.normal(mu, 0.001)
                    d_norm = (d + noise - self.dmax / 2) / self.dmax / 2
                    self.observations[i][j][0] = d_norm
                    target_angle_norm = (target_angle - self.angle) / self.angle
                    self.observations[i][j][1] = target_angle_norm

        return self.observations


    def step(self, actions):
        self.current_step += 1
        # if self.current_step % self.total_steps == 0:
        #     done = True
        # else:
        #     done = False

        for i in range(self.targets_number):
            target_dy = self.goal_states[i][1] - self.target_states[i][1]
            target_dx = self.goal_states[i][0] - self.target_states[i][0]
            # print('dx', dx)  # target_state (x, y)
            # print('dy', dy)
            if target_dy != 0:
                self.target_states[i][1] += int(target_dy / abs(target_dy))
            elif target_dx != 0:
                self.target_states[i][0] += int(target_dx / abs(target_dx))
            else:
                self.target_states[i][0] = np.random.randint(1, self.length)
                self.target_states[i][1] = np.random.randint(1, self.length)
                self.goal_states[i][0] = np.random.randint(1, self.length)
                self.goal_states[i][1] = np.random.randint(1, self.length)

        for i in range(self.cameras_number):
            if actions[i] == -1 :
                self.abs_angles[i] -= self.rotate_angle
                if self.abs_angles[i] < 0 or self.abs_angles[i] > 180:
                    self.abs_angles[i] += self.rotate_angle
            elif actions[i] == 1:
                self.abs_angles[i] += self.rotate_angle
                if self.abs_angles[i] < 0 or self.abs_angles[i] > 180:
                    self.abs_angles[i] -= self.rotate_angle

        self.targets_observed = np.zeros((self.cameras_number, self.targets_number))
        self.observations = np.zeros((self.cameras_number, self.targets_number, 2)) - 2
        for i in range(self.cameras_number):
            for j in range(self.targets_number):
                dy = self.init_states[j][1] - self.camera_positions[i][1]
                dx = self.init_states[j][0] - self.camera_positions[i][0]
                if self.camera_positions[i][0] == 0:
                   target_angle = 180 * math.atan2(dx, dy)/math.pi
                else:
                    target_angle = 180 * math.atan2(dy, dx) / math.pi
                if self.abs_angles[i] <= target_angle <= self.abs_angles[i] + self.angle:
                    d = np.sqrt(dx*dx + dy*dy)
                    mu = d/self.length
                    noise = np.random.normal(mu, 0.001)
                    d_norm = (d + noise - self.dmax/2)/self.dmax/2
                    self.observations[i][j][0] = d_norm
                    target_angle_norm = (target_angle - self.angle) / self.angle
                    self.observations[i][j][1] = target_angle_norm

                    self.targets_observed[i][j] = 1

        observed_num = np.sum(self.targets_observed, 1)
        self.done = any([observed_num[k] < 1 for k in range(len(observed_num))])
        observed_times = np.sum(self.targets_observed, 0)
        observed_ = [1 / observed_times[i] if observed_times[i] != 0 else 0 for i in range(len(observed_times))]

        for i in range(self.cameras_number):
            self.targets_observed[i] *= observed_
        r1 = np.sum(self.targets_observed, 1)/self.targets_number
        r2 = observed_num / self.targets_number
        observed_reward = r1 + r2

        return self.observations, observed_reward, self.done