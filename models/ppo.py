
from math import sqrt
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
import torch.optim.lr_scheduler as lr
import torch.nn.functional as F
import random
import numpy as np
import os
from datetime import datetime
import csv
import time

STATE = 0
PROB = 1
ACTION = 2
CRIT = 3
REWARD = 4
DONE = 5
ENTROPY = 6

ACTOR = "Actor"
CRITIC = "Critic"
SCHEDULER = "Scheduler"
OPTIMIZER = "Optimizer"
BEST_AVG_REWARD = "BestAvgReward"

# Actions
NOOP = 0
RIGHT = 1
RIGHT_A = 2
RIGHT_B = 3
RIGHT_A_B = 4
JUMP = 5
LEFT = 6

# Csv Keys
AVG_10_REW = "AVG_10_REW"
LAST_REW = "LAST_REW"
GAME = "GAME"
LOSS = "LOSS"
LOSS_STD = "LOSS_STD"
LOSS_CLIP = "LOSS_CLIP"
LOSS_CLIP_STD = "LOSS_CLIP_STD"
LOSS_VHS = "LOSS_VHS"
LOSS_VHS_STD = "LOSS_VHS_STD"
LOSS_ENTROPY = "LOSS_ENTROPY"
LOSS_ENTROPY_STD = "LOSS_ENTROPY_STD"
TOTAL_STEPS = "TOTAL_STEPS"
GAME_STEPS = "GAME_STEPS"
TOTAL_TIME = "TOTAL_TIME"
GAME_TIME = "GAME_TIME"
FLAG_GET = "FLAG_GET"

class PPO():
    def __init__(self, env, height, width, channels, n_actions, load_model="", render=False, config=None):
        """
        """
        self.validate = False
        self.training = False
        if config is not None:
            self.training = config['train']
            self.validate = config['test']
        if not self.validate and not self.training:
            return

        self.env = env
        self.device_str = "cpu" if not torch.cuda.is_available() else "cuda"
        self.device = torch.device(self.device_str)

        self.channels = channels
        self.height, self.width, self.n_actions = (height, width, n_actions)
        self.actor_alpha = 1e-4#5e-4#1e-4
        self.critic_alpha = 1e-4#5e-4#1e-4
        self.epsilon_clip = 0.2
        self.c_1 = 0.5 # Value function coeff
        self.c_2 = 0.01 # Enropy regularization coeff
        self.gamma = 0.99
        self.gae = 0.95
        self.iterations = 20#15
        self.n_traj = 32#16
        
        self.batch_size = 8#16#1024
        self.mem_size = 512 #2048#256#4096#self.batch_size * 4
        self.render = render

        self.n_frame_per_step = 4
        self.max_game_steps = 1000
        self.max_action_repeat = 200
        self.best_avg_rew = np.float('-inf')
        self.name = "PPO"
        self.checkpoint_dir = f"checkpoints/{self.name}"
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.name + "latest.pth")
        self.checkpoint_csv_path = os.path.join(self.checkpoint_dir, self.name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + f"_seed{self.env.seed}.csv")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.actor = Actor(self.n_frame_per_step, height, width, n_actions, self.device, self.checkpoint_dir).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_alpha)#, weight_decay=10e-2, amsgrad=True)
        self.critic = Critic(self.n_frame_per_step, height, width, self.device, self.checkpoint_dir).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_alpha)#, weight_decay=10e-2, amsgrad=True)

        self.memory = ReplayBuffer(state_shape=(self.n_frame_per_step, self.height, self.width), num_actions=self.n_actions, batch_size=self.batch_size, size=self.mem_size, n_traj=self.n_traj, device=self.device_str)

        if self.validate and load_model != "":
            self.best_avg_rew = float(load_model.split("_")[1])
            self.load_model(load_model)

        self.env.set_n_frame_per_step(self.n_frame_per_step)
        self.action_count = 0
        self.last_action = -1
        self.last_100_total_rew = []
        self.losses = [0]
        self.l_clips = [0]
        self.l_vhs = [0]
        self.l_entropies = [0]
        self.total_time = None
        self.game_time = None

    def save_model(self, avg_rew):
        checkpoint = {
            f"{BEST_AVG_REWARD}": avg_rew,
            f"{ACTOR}": self.actor.state_dict(),
            f"{ACTOR}{OPTIMIZER}": self.actor_optimizer.state_dict(),
            f"{CRITIC}": self.critic.state_dict(),
            f"{CRITIC}{OPTIMIZER}": self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_path)
        checkpoint_path = os.path.join(self.checkpoint_dir, self.name + f"_{round(avg_rew,3)}_" + ".pth")
        torch.save(checkpoint, checkpoint_path)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint[ACTOR])
        self.actor_optimizer.load_state_dict(checkpoint[f"{ACTOR}{OPTIMIZER}"])
        self.critic.load_state_dict(checkpoint[CRITIC])
        self.critic_optimizer.load_state_dict(checkpoint[f"{CRITIC}{OPTIMIZER}"])
        self.best_avg_rew = checkpoint[BEST_AVG_REWARD]

    def get_action(self, state):
        with torch.no_grad():
            val = self.actor(state)
            prob = 0.5
            crit = 0.5
            if self.action_count >= self.max_action_repeat:
                action = torch.randint(self.n_actions, (1,)).to(self.device)
            elif not self.validate or self.action_count >= self.max_action_repeat:
                cat = torch.distributions.Categorical(val)
                action = cat.sample()
                prob = cat.log_prob(action)
                crit = self.critic(state)
            else:
                action = torch.argmax(val)
            if action == self.last_action:
                self.action_count += 1
            else:
                self.action_count = 0
            self.last_action = action
        return action, prob, crit
            
    def game(self, validate=False):
        done = False
        reward = 0
        action = 0
        state = self.env.reset()
        self.action_count = 0
        rewards = [0.]*100
        total_reward = 0
        step = 0
        self.flag_get = False
        self.finished_successfully = False
        if self.validate:
            self.max_game_steps = np.float('inf')
        while not done and step <= self.max_game_steps:
            action, prob, crit = self.get_action(state)
            n_state, reward, done, info = self.env.step(action.item())
            self.flag_get = info['flag_get']
            total_reward += reward
            if not self.validate:
                self.memory.append(n_state, state, prob, action, crit, reward/15., done, 0, info)
                if self.memory.full_enough():
                    self.optimize(n_state)
            rewards = reward.detach().clone().tolist() + rewards[:-1]
            state = n_state
            if self.render or self.epoch % 20 == 0 or self.validate:
                self.env.render()
            step += 1
        self.last_100_total_rew = total_reward.detach().clone().tolist() + self.last_100_total_rew[:100]    

    def optimize(self, state):
        with torch.no_grad():
            next_value = self.critic(state)
        batches = self.memory.get_batch()
        a_t, ret = self.batch_gen_adv_estimation(batches, next_value)
        self.losses = []
        self.l_clips = []
        self.l_vhs = []
        self.l_entropies = []
        for i in range(self.iterations):
            indices_ = self.memory.get_rand_indices()
            l_entropy_ = 0
            l_clip_ = 0
            loss = 0
            l_vh_ = 0
            for idx, indices in enumerate(indices_):
                a_ts = a_t[indices]
                rets = ret[indices]
                batch = [x[indices] for x in batches]
                val = self.actor(batch[STATE])
                cat = torch.distributions.Categorical(val)
                prob = cat.log_prob(batch[ACTION][:,0])
                crit = self.critic(batch[STATE])
                ratio = torch.exp(prob - batch[PROB][:,0])                
                obj = ratio * a_ts
                clipped_obj = a_ts * torch.clamp(ratio, 1. - self.epsilon_clip, 1. + self.epsilon_clip)
                l_entropy = self.c_2*cat.entropy()
                l_clip = -1*torch.min(obj, clipped_obj)
                l_vh = (self.c_1*(rets - crit.T)**2)[0,:]
                loss += torch.mean(l_clip + l_vh - l_entropy)
                l_entropy_ += l_entropy.cpu().clone().detach().mean()
                l_clip_ += l_clip.cpu().clone().detach().mean()
                l_vh_ += l_vh.cpu().clone().detach().mean()
            loss /= len(indices_)
            l_clip = l_clip_/len(indices_)
            l_vh = l_vh_ / len(indices_)
            l_entropy = l_entropy_/len(indices_)
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            self.l_clips.append(l_clip.cpu().clone().detach().mean().item())
            self.l_vhs.append(l_vh.cpu().clone().detach().mean().item())
            self.l_entropies.append(l_entropy.cpu().clone().detach().mean().item())
            self.losses.append(loss.cpu().clone().detach().item())
        print(f"Loss:{np.mean(self.losses)},L_clips:{np.mean(self.l_clips)},L_vhs:{np.mean(self.l_vhs)},L_entropies:{np.mean(self.l_entropies)}")
        self.memory.clear()

    def batch_gen_adv_estimation(self, batches, next_value):
        values = torch.cat((batches[CRIT], next_value))[:, 0]
        dones = batches[DONE][:, 0]
        rewards = batches[REWARD][:, 0]
        mem_size = batches[REWARD].shape[0]
        returns = []
        ret = 0
        for t in range(mem_size):
            n_t_r = t_r = mem_size - t - 1
            n_t_r += 1
            done = 1. - dones[t_r]
            discount = rewards[t_r] + (self.gamma * done * values[n_t_r]).detach() - values[t_r].detach()
            ret = discount + self.gamma * self.gae * done * ret
            returns.append(ret + values[t_r])
        returns = torch.tensor(returns[::-1]).to(self.device)
        adv = returns - values[:-1]
        return adv, returns

    def train(self):
        avg_rew = self.best_avg_rew
        self.epoch = 1
        
        with open(self.checkpoint_csv_path, 'w', newline='') as csvfile:
            fieldnames = [GAME, GAME_STEPS, TOTAL_STEPS, AVG_10_REW,LAST_REW,LOSS,LOSS_STD,LOSS_CLIP,LOSS_CLIP_STD,LOSS_VHS,LOSS_VHS_STD,LOSS_ENTROPY,LOSS_ENTROPY_STD, TOTAL_TIME, GAME_TIME, FLAG_GET]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            self.total_time = time.time()
            while True and (self.epoch <= 10 if self.validate else True):
                self.game_time = time.time()
                self.game()
                row = {
                    AVG_10_REW: round(np.mean(self.last_100_total_rew[:10]),3),
                    LAST_REW: round(self.last_100_total_rew[:1][0],3),
                    GAME: self.epoch,
                    LOSS: round(np.mean(self.losses),3),
                    LOSS_STD: round(np.std(self.losses),3),
                    LOSS_CLIP: round(np.mean(self.l_clips),3),
                    LOSS_CLIP_STD: round(np.std(self.l_clips),3),
                    LOSS_VHS: round(np.mean(self.l_vhs),3),
                    LOSS_VHS_STD: round(np.std(self.l_vhs),3),
                    LOSS_ENTROPY: round(np.mean(self.l_entropies),3),
                    LOSS_ENTROPY_STD: round(np.std(self.l_entropies),3),
                    GAME_STEPS: self.env.actual_game_steps,
                    TOTAL_STEPS: self.env.actual_total_steps,
                    TOTAL_TIME: round(time.time() - self.total_time, 3),
                    GAME_TIME: round(time.time() - self.game_time, 3),
                    FLAG_GET: self.flag_get
                }
                writer.writerow(row)
                avg_rew = np.mean(self.last_100_total_rew[:10])
                print(f"Game:{self.epoch},LastRew:{self.last_100_total_rew[:1][0]},Avg10Rew:{avg_rew}")
                
                if not self.validate and self.epoch >= 10 and avg_rew > self.best_avg_rew:
                    self.best_avg_rew = avg_rew
                    self.save_model(avg_rew)
                self.epoch += 1
                csvfile.flush()


class ReplayBuffer:
    def __init__(self, state_shape, num_actions, batch_size=32, size=1000, n_traj=10, device="cpu"):
        self.size = int(size)
        self.n_traj = int(n_traj)
        self.device=device
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.traj_idx = 0
        self.idx = 0
        self.states = []
        self.crits = []
        self.probs = []
        self.entropies = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.clear()

    def get_rand_indices(self):
        lenn = self.len()
        x = lenn-self.n_traj
        starts = np.random.choice(np.arange(x), self.batch_size, replace=False).astype(int)
        indices = [range(y, y+self.n_traj) for y in starts]
        return indices
    
    def get_batch(self):
        return (self.states, self.probs, self.actions, self.crits, self.rewards, self.dones, self.entropies)

    def clear(self):
        del self.states
        del self.crits
        del self.probs
        del self.entropies
        del self.actions
        del self.rewards
        del self.dones
        self.states = torch.empty((self.size, self.state_shape[0], self.state_shape[1], self.state_shape[2], ), dtype=torch.float, device = self.device )
        self.crits = torch.empty((self.size, 1, ), dtype=torch.float, device = self.device )
        self.probs = torch.empty((self.size, 1, ), dtype=torch.float, device = self.device )
        self.entropies = torch.empty((self.size, 1, ), dtype=torch.float, device = self.device )
        self.actions = torch.empty((self.size, 1, ), dtype=torch.float, device = self.device )
        self.rewards = torch.empty((self.size, 1, ), dtype=torch.float, device = self.device )
        self.dones = torch.empty((self.size, 1, ), dtype=torch.float, device = self.device )
        self.idx = 0
        
    def full_enough(self):
        return self.idx == self.size

    def len(self):
        return min(self.idx, self.size)
        
    def append(self, next_state, state, prob, action, crit, reward, done, entropy, info):
        idx = self.idx % self.size
        self.states[idx, :] = state
        self.crits[idx, :] = crit
        self.probs[idx, :] = prob
        self.actions[idx, :] = action
        self.rewards[idx, :] = reward
        self.dones[idx, :] = done
        self.entropies[idx, :] = entropy
        self.idx += 1
# INP_DIM = 32*26*28
# INP_DIM = 32*13*14
# INP_DIM = 32*120*128
# INP_DIM = 32*30*32
# INP_DIM = 32*60*64
INP_DIM = 16*29*31
FC_DIM = 512#400
class Actor(nn.Module):
    def __init__(self, channels, height, width, n_actions, device, checkpoint_dir, name="Actor", postfix=""):
        super(Actor, self).__init__()
        
        self.n_actions = n_actions
        self.device = device
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.linear1 = nn.Linear(INP_DIM, FC_DIM)
        self.final = nn.Linear(FC_DIM, n_actions)

    def forward(self, X):
        val = F.relu(self.conv1(X))
        val = F.relu(self.conv2(val))
        val = F.relu(self.conv3(val))
        val = torch.flatten(val, 1)
        val = F.relu(self.linear1(val))
        val = F.softmax(self.final(val), 1)
        return val

class Critic(nn.Module):
    def __init__(self, channels, height, width, device, checkpoint_dir, name="Critic", postfix=""):
        super(Critic, self).__init__()
        
        self.device = device
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.linear1 = nn.Linear(INP_DIM, FC_DIM)
        self.final = nn.Linear(FC_DIM, 1)

    def forward(self, X):
        val = X.to(self.device)
        val = F.relu(self.conv1(val))
        val = F.relu(self.conv2(val))
        val = F.relu(self.conv3(val))
        val = torch.flatten(val, 1)
        val = F.relu(self.linear1(val))
        return self.final(val)
