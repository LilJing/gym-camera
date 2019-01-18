<<<<<<< HEAD
'''multi env for recall reward'''
=======
'''multi env with reward: observe number with distance'''
>>>>>>> change reward

import gym
from gym import spaces
from gym.utils import seeding

import cv2
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors

class Multi_Camera_Env_v1(gym.Env):
    def __init__(self,
                 cameras_number=4,
<<<<<<< HEAD
                 targets_number=5,
                 get_trace=False,
                 traces_len=3,
                 length=80,
                 angle=30):
=======
                 targets_number=10,
                 get_trace=False,
                 traces_len=3,
                 length=80,
                 angle=50):
>>>>>>> change reward

        self.targets_number = targets_number
        self.cameras_number = cameras_number

        self.get_trace = get_trace
        self.traces = []
        self.traces_len = traces_len

        self.init_states = []
        self.goal_states = []

        self.num_actions = 3
        self.angle = angle
        self.rotate_angle = 5
        self.length = length
        self.dmax = self.length * 1.4

        # define action space
        self.action_space = [spaces.Discrete(self.num_actions) for i in range(self.cameras_number)]
        # self.all_actions = list(range(self.action_space.n))

        # define observation space
        self.observation_space = [spaces.Box(low=-np.ones(self.targets_number * 3), high=np.ones(self.targets_number * 3))
                                  for i in range(self.cameras_number)]  #2: distance and angle

        self.total_steps = 100
        self.current_step = 0
        self.done = False

        # render
        self.color_dict = {'red': [255, 0, 0], 'black': [0, 0, 0], 'blue': [0, 0, 255], 'green': [0, 255, 0],
                           'darkred': [128, 0, 0], 'yellow':[255, 255, 0], 'deeppink':[255,20,147]}
        self.color_index ={0:'red', 1:'yellow', 2:'blue', 3:'green', 4:'darkred', 5:'deeppink', 6:'black'}

<<<<<<< HEAD
=======
        self.rand_cam = False
        self.radius = 30

>>>>>>> change reward
    def reset(self):
        # random target position
        self.current_step = 0
        self.init_pos = self.random_target(self.targets_number, self.length)
        self.goal_pos = self.random_target(self.targets_number, self.length)
        self.target_pos = self.init_pos.copy()
        # random camera
<<<<<<< HEAD
        self.camera_position, self.abs_angles = self.random_camera(self.cameras_number, self.length, self.angle)

        # need check, and delete for loop
        self.observations, self.targets_observed = self.get_observation(self.target_pos,
=======
        self.camera_position, self.abs_angles = self.random_camera(self.cameras_number, self.length, self.angle, self.rand_cam)

        # need check, and delete for loop

        self.observations, self.targets_observed, self.targets_reward = self.get_observation(self.target_pos,
>>>>>>> change reward
                                                                        self.camera_position, self.abs_angles)

        return self.observations

    def step(self, actions):
        self.current_step += 1
        # if self.current_step % self.total_steps == 0:
        #     done = True
        # else:
        #     done = False
        info = dict()
        # move target
        for i in range(self.targets_number):
            target_dy = self.goal_pos[i][1] - self.target_pos[i][1]
            target_dx = self.goal_pos[i][0] - self.target_pos[i][0]
            # print('dx', dx)  # target_state (x, y)
            # print('dy', dy)
            # not shortest path
            if target_dy != 0:
                self.target_pos[i][1] += int(target_dy / abs(target_dy))
            elif target_dx != 0:
                self.target_pos[i][0] += int(target_dx / abs(target_dx))
            else:  # reach goal, and generate a new goal
                # self.target_pos[i][0] = np.random.randint(1, self.length)
                # self.target_pos[i][1] = np.random.randint(1, self.length)
                self.goal_pos[i] = np.random.randint(1, self.length, size=2)

        # move camera
        for i in range(self.cameras_number):
            if actions[i] == 1:
                self.abs_angles[i] -= self.rotate_angle
                if self.abs_angles[i] < 0 or self.abs_angles[i] > 90:
                    self.abs_angles[i] += self.rotate_angle
            elif actions[i] == 2:
                self.abs_angles[i] += self.rotate_angle
                if self.abs_angles[i] < 0 or self.abs_angles[i] > 90:
                    self.abs_angles[i] -= self.rotate_angle

<<<<<<< HEAD
        self.observations, self.targets_observed = self.get_observation(self.target_pos,
                                                                        self.camera_position, self.abs_angles)

        observed_num = np.sum(self.targets_observed, 1)
        self.done = False
        # self.done = any([observed_num[k] < 1 for k in range(len(observed_num))])  # done is False when each camera can see at least one target
        observed_times = np.sum(self.targets_observed, 0)
        observed_ = [1 / observed_times[i] if observed_times[i] != 0 else 0 for i in range(len(observed_times))]

        for i in range(self.cameras_number):
            self.targets_observed[i] *= observed_
        r1 = np.sum(self.targets_observed, 1)/self.targets_number
        r2 = observed_num / self.targets_number
        # observed_reward = r1 + r2
        observed_reward = r1 + r2
        # print (observed_num)
        return self.observations, observed_reward, self.done, info

    def random_camera(self, num_cam, length, view_angle):
        camera_positions = []
        for i in range(num_cam):
            if i % 4 == 0:  # first edge
                camera_positions.append([0, np.random.randint(1, length-1)])
            elif i % 4 == 1:  # second edge
                camera_positions.append([np.random.randint(1, length-1), 0])
            elif i % 4 == 2:  # third edge
                camera_positions.append([length-1, np.random.randint(1, length-1)])
            elif i % 4 == 3:  # fourth edge
                camera_positions.append([np.random.randint(1, length-1), length-1])

        abs_angles = np.random.randint(0, view_angle, size=num_cam)
=======
        self.observations, self.targets_observed, self.targets_reward, = self.get_observation(self.target_pos,
                                                                        self.camera_position, self.abs_angles)

        # print 'reward', self.reward
        # self.done = False
        '''reward with observe number and observe times'''
        # observed_num = np.sum(self.targets_observed, 1)
        # observed_times = np.sum(self.targets_observed, 0)
        # observed_ = [1 / observed_times[i] if observed_times[i] != 0 else 0 for i in range(len(observed_times))]

        # for i in range(self.cameras_number):
            # self.targets_observed[i] *= observed_
        # r1 = np.sum(self.targets_observed, 1)/self.targets_number
        # r2 = observed_num / self.targets_number
        # observed_reward = r1 + r2
        '''reward with observe number'''
        observed_num = np.sum(self.targets_observed, 1)
        self.reward = observed_num
        # observed_reward = observed_num / self.targets_number
        # print (observed_reward)
        '''reward with observe number with distance'''
        observed_reward = np.sum(self.targets_reward, 1)
        #print observed_reward
        return self.observations, observed_reward, self.done, info

    def random_camera(self, num_cam, length, view_angle, random):
        if random == True:
            camera_positions = []
            for i in range(num_cam):
                if i % 4 == 0:  # first edge
                    camera_positions.append([0, np.random.randint(1, length-1)])
                elif i % 4 == 1:  # second edge
                    camera_positions.append([np.random.randint(1, length-1), 0])
                elif i % 4 == 2:  # third edge
                    camera_positions.append([length-1, np.random.randint(1, length-1)])
                elif i % 4 == 3:  # fourth edge
                    camera_positions.append([np.random.randint(1, length-1), length-1])

            abs_angles = np.random.randint(0, view_angle, size=num_cam)
        else:
            camera_positions = []
            for i in range(num_cam):
                if i % 4 == 0:  # first edge
                    camera_positions.append([0, self.length / 2])
                elif i % 4 == 1:  # second edge
                    camera_positions.append([self.length / 2, 0])
                elif i % 4 == 2:  # third edge
                    camera_positions.append(([length - 1, self.length / 2]))
                elif i % 4 == 3:  # fourth edge
                    camera_positions.append([self.length / 2, length-1])
            abs_angles = [90 - self.angle / 2 for i in range(num_cam)]
>>>>>>> change reward

        return camera_positions, abs_angles

    def random_target(self, num_target, length):
        target_positions = np.random.randint(1, length-1, size=(num_target, 2))
        return target_positions

    def get_observation(self, target_pos, camera_pos, abs_angles):
        targets_observed = np.zeros((self.cameras_number, self.targets_number))
<<<<<<< HEAD
=======
        targets_reward = np.zeros((self.cameras_number, self.targets_number))
>>>>>>> change reward
        obs = np.zeros((self.cameras_number, self.targets_number*3)) - 2
        for i in range(self.cameras_number):
            for j in range(self.targets_number):
                dy = target_pos[j][1] - camera_pos[i][1]
                dx = target_pos[j][0] - camera_pos[i][0]
                if camera_pos[i][0] == 0:
<<<<<<< HEAD
                    if dx == 0 and dy < 0: #how if dy = dy =0
=======
                    if dx == 0 and dy < 0:
>>>>>>> change reward
                        target_angle = 0
                    elif dx == 0 and dy > 0:
                        target_angle = 180
                    elif dy == 0:
                        target_angle = 90
                    elif dx * dy < 0:
                        ref_angle = 180 * math.atan2(dy, dx)/math.pi
                        target_angle = 90 + ref_angle
                    elif dx * dy > 0:
                        ref_angle = 180 * math.atan2(dy, dx) / math.pi
                        target_angle = 180 - ref_angle

                elif camera_pos[i][1] == self.length-1:
                    if dy == 0 and dx > 0:
                        target_angle = 180
                    elif dy == 0 and dx < 0:
                        target_angle = 0
                    elif dx == 0:
                        target_angle = 90
                    else:
                        ref_angle = 180 * math.atan2(dy, dx) / math.pi
                        target_angle = 180 + ref_angle

                elif camera_pos[i][0] == self.length-1:
                    if dx == 0 and dy < 0:
                        target_angle = 180
                    elif dx == 0 and dy > 0:
                        target_angle = 0
                    elif dy == 0:
                        target_angle = 90
                    elif dx * dy < 0:
                        ref_angle = 180 * math.atan2(dy, dx) / math.pi
                        target_angle = ref_angle - 90
                    else:
                        ref_angle = 180 * math.atan2(dy, dx) / math.pi
                        target_angle = ref_angle + 270

                else:
                    if dy == 0 and dx < 0:
                        target_angle = 180
                    elif dy == 0 and dx > 0:
                        target_angle = 0
                    elif dx == 0:
                        target_angle = 90
                    else:
                        target_angle = 180 * math.atan2(dy, dx) / math.pi
                    if target_angle < 0 or target_angle > 180:
                        print('cam id', i, 'tar id', j, 'target angle', target_angle)

                # print('cam id', i, 'tar id', j, 'target angle', target_angle)
                # if target_angle < 0 or target_angle > 180:  # check for invalid target angle
                #     print('cam id', i, 'tar id', j, 'target angle', target_angle)
<<<<<<< HEAD
                if abs_angles[i] <= target_angle <= abs_angles[i] + self.angle:
                    d = np.sqrt(dx*dx + dy*dy)
=======
                d = np.sqrt(dx * dx + dy * dy)
                if abs_angles[i] <= target_angle <= abs_angles[i] + self.angle and d <= self.radius:

>>>>>>> change reward
                    mu = d/self.length
                    noise = np.random.normal(mu, 0.001)
                    d_norm = (d + noise - self.dmax/2)/self.dmax/2
                    obs[i][j*3] = d_norm
                    target_angle_norm = (target_angle - self.angle) / self.angle
                    obs[i][j*3 + 1] = target_angle_norm
                    obs[i][j*3 + 2] = abs_angles[i] / 180.0
                    targets_observed[i][j] = 1
<<<<<<< HEAD

        return obs, targets_observed
=======
                    targets_reward[i][j] = 1 - d_norm / self.dmax

        return obs, targets_observed, targets_reward
>>>>>>> change reward

    def seed(self, seed=None):
        np.random.seed()

    def render(self, mode='rgb'):
<<<<<<< HEAD
=======

>>>>>>> change reward
        img = np.zeros((self.length+1, self.length+1, 3)) + 255

        # plot camera
        for i in range(self.cameras_number):

            img[self.camera_position[i][1]][self.camera_position[i][0]][0] = self.color_dict["black"][0]
            img[self.camera_position[i][1]][self.camera_position[i][0]][1] = self.color_dict["black"][1]
            img[self.camera_position[i][1]][self.camera_position[i][0]][2] = self.color_dict["black"][2]

        # plot target
        for i in range(self.targets_number):

<<<<<<< HEAD
            img[self.target_pos[i][1]][self.target_pos[i][0]][0] = self.color_dict[self.color_index[i]][0]
            img[self.target_pos[i][1]][self.target_pos[i][0]][1] = self.color_dict[self.color_index[i]][1]
            img[self.target_pos[i][1]][self.target_pos[i][0]][2] = self.color_dict[self.color_index[i]][2]

        plt.figure(figsize=(5, 5))
        plt.imshow(img.astype(np.uint8))

        # get camera's view space positions
        visua_len = 20 # length of arrow
        for i in range(self.cameras_number):
            if self.camera_position[i][0] == 0:
                for theta in [self.abs_angles[i], self.abs_angles[i] + 90]:
=======
            # img[self.target_pos[i][1]][self.target_pos[i][0]][0] = self.color_dict[self.color_index[i]][0]
            # img[self.target_pos[i][1]][self.target_pos[i][0]][1] = self.color_dict[self.color_index[i]][1]
            # img[self.target_pos[i][1]][self.target_pos[i][0]][2] = self.color_dict[self.color_index[i]][2]
            img[self.target_pos[i][1]][self.target_pos[i][0]][0] = self.color_dict['blue'][0]
            img[self.target_pos[i][1]][self.target_pos[i][0]][1] = self.color_dict['blue'][1]
            img[self.target_pos[i][1]][self.target_pos[i][0]][2] = self.color_dict['blue'][2]

        plt.cla()
        plt.imshow(img.astype(np.uint8))

        # get camera's view space positions
        visua_len = self.radius # length of arrow
        for i in range(self.cameras_number):
            if self.camera_position[i][0] == 0:
                for theta in [self.abs_angles[i], self.abs_angles[i] + self.angle]:
>>>>>>> change reward
                    dx = visua_len * math.sin(theta * math.pi / 180)
                    dy = - visua_len * math.cos(theta * math.pi / 180)
                    plt.arrow(self.camera_position[i][0], self.camera_position[i][1], dx, dy, width=0.1, head_width=1,
                              length_includes_head=True)
<<<<<<< HEAD
            elif self.camera_position[i][1] == 0:
                for theta in [self.abs_angles[i], self.abs_angles[i] + 90]:
=======
                plt.annotate('num%s' % int(self.reward[i]), xy=(self.camera_position[i][0], self.camera_position[i][1] + 2),
                             color='green')

            elif self.camera_position[i][1] == 0:
                for theta in [self.abs_angles[i], self.abs_angles[i] + self.angle]:
>>>>>>> change reward
                    dx = visua_len * math.cos(theta * math.pi / 180)
                    dy = visua_len * math.sin(theta * math.pi / 180)
                    plt.arrow(self.camera_position[i][0], self.camera_position[i][1], dx, dy, width=0.1, head_width=1,
                              length_includes_head=True)
<<<<<<< HEAD
            elif self.camera_position[i][0] == self.length - 1:
                for theta in [self.abs_angles[i], self.abs_angles[i] + 90]:
=======
                plt.annotate('num%s' % int(self.reward[i]), xy=(self.camera_position[i][0], self.camera_position[i][1]+2),
                             color='green')

            elif self.camera_position[i][0] == self.length - 1:
                for theta in [self.abs_angles[i], self.abs_angles[i] + self.angle]:
>>>>>>> change reward
                    dx = - visua_len * math.sin(theta * math.pi / 180)
                    dy = visua_len * math.cos(theta * math.pi / 180)
                    plt.arrow(self.camera_position[i][0], self.camera_position[i][1], dx, dy, width=0.1, head_width=1,
                              length_includes_head=True)
<<<<<<< HEAD
            else:
                for theta in [self.abs_angles[i], self.abs_angles[i] + 90]:
=======
                plt.annotate('num%s' % int(self.reward[i]), xy=(self.camera_position[i][0], self.camera_position[i][1]+2),
                             color='green')

            else:
                for theta in [self.abs_angles[i], self.abs_angles[i] + self.angle]:
>>>>>>> change reward
                    dx = - visua_len * math.cos(theta * math.pi / 180)
                    dy = - visua_len * math.sin(theta * math.pi / 180)
                    plt.arrow(self.camera_position[i][0], self.camera_position[i][1], dx, dy, width=0.1, head_width=1,
                              length_includes_head=True)
<<<<<<< HEAD
            plt.annotate('cam%s'%i, xy=(self.camera_position[i][0], self.camera_position[i][1]),
                         xytext=(self.camera_position[i][0], self.camera_position[i][1]), fontsize=10, color='red')
        plt.show()
=======
                plt.annotate('num%s' % int(self.reward[i]), xy=(self.camera_position[i][0], self.camera_position[i][1]-2),
                             color='green')
            plt.annotate('cam%s'%i, xy=(self.camera_position[i][0], self.camera_position[i][1]),
                         xytext=(self.camera_position[i][0], self.camera_position[i][1]), fontsize=10, color='red')


        plt.pause(0.01)

>>>>>>> change reward
