"""
Main Function.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import yaml
import argparse
import time
import copy

import numpy as np
import torch

import torch
import torch.nn as nn

from game import MarioGame

from models import PPO

from utils import plot as p 

parser = argparse.ArgumentParser(description='CS7643 deep_pipes')
parser.add_argument('--config', default='./configs/ppo.yaml')


def main():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.SafeLoader)

    env = MarioGame()
    
    if config['model'] == 'PPO':
        if config['train_csv_path'] != "":
            p.plot_loss_and_reward(config['train_csv_path'])
        if config['test_csv_path'] != "":
            p.plot_test_reward(config['test_csv_path'])
        model = PPO(env, env.height, env.width, env.channels, env.action_space, load_model=config['load_model'], render=config['render'], config=config)    
        if config['test'] or config['train']:
            model.train()

if __name__ == '__main__':
    main()
