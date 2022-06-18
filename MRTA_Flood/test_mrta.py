"""
Author: Steve Paul 
Date: 5/19/22 """
from torch import nn
from collections import defaultdict
import warnings
import math
import numpy as np
import gym
from stable_baselines_al import PPO, A2C
# from stable_baselines.common import make_vec_env
from MRTA_Flood_Env import MRTAENV
import json
import datetime as dt
import torch
from utils import *
from topology import *
import scipy.sparse as sp
from persim import wasserstein, bottleneck
import ot
from CustomPolicies import ActorCriticGCAPSPolicy
# from CustomPolicies import ActorCriticGCAPSPolicy
from stable_baselines_al.common.utils import set_random_seed


from stable_baselines_al.common.vec_env import DummyVecEnv, SubprocVecEnv

def as_tensor(observation):
    for key, obs in observation.items():
        observation[key] = torch.tensor(obs)
    return observation

env = DummyVecEnv([lambda: MRTAENV(
        n_locations = 41,
        n_agents = 6,
        enable_dynamic_tasks=False,
        display = False,
        enable_topological_features = False
)])
log_dir = "../Trained_models/."
model = PPO.load(log_dir + "r1_simple_nn", env=env)
env = DummyVecEnv([lambda: MRTAENV(
        n_locations = 20,
        n_agents = 3,
        enable_dynamic_tasks=False,
        display = True,
        enable_topological_features = False
)])
model.env = env
obs = env.reset()
obs = as_tensor(obs)
total_rewards_list = []
for i in range(1000000):
        model.policy.set_training_mode(False)
        action = model.policy._predict(obs)
        obs, reward, done, _ = env.step(action)
        obs = as_tensor(obs)
        if done:
                total_rewards_list.append(reward)
                env = DummyVecEnv([lambda: MRTAENV(
                        n_locations=20,
                        n_agents=3,
                        enable_dynamic_tasks=False,
                        display=True,
                        enable_topological_features=False
                )])
                obs = as_tensor(obs)

        if len(total_rewards_list) == 100:
                break
total_rewards_array = np.array(total_rewards_list)
print(total_rewards_array.mean(), total_rewards_array.std())

