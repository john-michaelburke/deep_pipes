
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
import torch
import numpy as np
# import cv2

class MarioGame:
    def __init__(self, world=1, stage=1, seed=1331):
        env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0')
        # self.env = JoypadSpace(env, RIGHT_ONLY)
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space.n
        # import pdb;pdb.set_trace()
        self.height_, self.width_, self.channels = self.observation_space.shape
        self.dim_reduc = int(1)#int(2)
        self.height = int(self.height_ / self.dim_reduc)
        self.width = int(self.width_ / self.dim_reduc)
        self.device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
        self.total_steps = 0
        self.actual_total_steps = 0
        self.game_steps = 0
        self.actual_game_steps = 0
        self.n_frame_per_step = 1
        self.seed = seed
        self.env.seed(self.seed)

    def set_n_frame_per_step(self, n_frame_per_step):
        self.n_frame_per_step = n_frame_per_step

    def step(self, action):
        # state, reward, done, info
        self.game_steps += 1
        self.total_steps += 1
        return self.preprocess(action)

    def preprocess(self, action):
        step = 0
        done = False
        reward = 0
        # action_ = action
        cur_state = np.zeros((self.n_frame_per_step, self.height_, self.width_, 3))
        while not done and step < self.n_frame_per_step:
            state, reward_, done, info = self.env.step(action)
            reward += reward_
            cur_state[step, :, :]= state
            step += 1
            self.actual_total_steps += 1
            self.actual_game_steps += 1
            # if last_state is not None:
            #     avg_state = state - last_state
            #     last_state = state
            # action_ = 0
        while step < self.n_frame_per_step:
            cur_state[step, :, :]= state.copy()
            step += 1
        # if avg_state is not None:
        #     state = avg_state
        #/self.n_frame_per_step
        # state = np.mean(state, 0)
        # import pdb;pdb.set_trace()
        # backtorgb = cv2.cvtColor(cur_state[0],cv2.COLOR_GRAY2RGB)
        
        return self.process_state(cur_state), self.process_reward(reward), torch.BoolTensor([done]).to(self.device), info

    def process_state(self, state):
        state = state.copy()
        state = torch.from_numpy(state).to(device=self.device, dtype=torch.float)
        # state = state / 128.0 - 1.0
        state = state / 255.
        # import pdb;pdb.set_trace()
        state = torch.mean(state, 3)[None, :, ::self.dim_reduc, ::self.dim_reduc]
        # cv2.imshow("im", state[0,0].clone().cpu().detach().numpy())
        # cv2.waitKey(1)
        return state

    def process_reward(self, reward):
        return torch.FloatTensor([reward]).to(device=self.device, dtype=torch.float)#/15.

    def reset(self):
        self.game_steps = 0
        self.actual_game_steps = 0
        # cur_state = np.zeros((self.n_frame_per_step, self.height, self.width, 3))
        state = self.env.reset()
        cur_state = np.repeat(state[None, :], (4), axis=0)
        return self.process_state(cur_state)

    def render(self):
        self.env.render()

    def stop(self):
        self.env.close()


    