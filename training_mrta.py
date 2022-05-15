"""
Author: Steve Paul 
Date: 4/15/22 """
""
from torch import nn
from collections import defaultdict
import warnings
import math
import numpy as np
import gym
from stable_baselines_al import PPO, A2C
# from stable_baselines.common import make_vec_env
from mrta_flood_env import MRTAENV
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

warnings.filterwarnings('ignore')

class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input



env = DummyVecEnv([lambda: MRTAENV(
        n_locations = 41,
        n_agents = 6,
        enable_dynamic_tasks=False,
        display = False,
        enable_topological_features = True
)])

# n_envs = 4
# env = make_vec_env(mTSPEnv, n_envs=n_envs, env_kwargs={"n_locations":21, "n_agents":5})
# num_cpu = 6
# env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

# model = PPO('MlpPolicy',
#     env,
#     gamma=1.00,
#     verbose=1,
#     n_epochs=10,
#     batch_size=10000,
#     tensorboard_log="logger/",
#     create_eval_env=True,
#     n_steps=20000)

policy_kwargs=dict(
    # features_extractor_class=GCAPCNFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128,node_dim=2),
    # activation_fn=torch.nn.LeakyReLU,
    # net_arch=[dict(vf=[128,128])]
)

model = PPO(
    ActorCriticGCAPSPolicy,
    env,
    gamma=1.00,
    verbose=1,
    n_epochs=100,
    batch_size=10000,
    tensorboard_log="logger/",
    # create_eval_env=True,
    n_steps=20000,
    learning_rate=0.000001,
    policy_kwargs = policy_kwargs,
    ent_coef=0.001,
    vf_coef=0.5
)
#
# model = A2C(
#     ActorCriticGCAPSPolicy,
#     env,
#     gamma=1.00,
#     verbose=1,
#     tensorboard_log="logger/",
#     create_eval_env=True,
#     n_steps=20000
# )

model.learn(total_timesteps=2000000)

obs = env.reset()

log_dir = "."
model.save(log_dir + "r1")
model = PPO.load(log_dir + "r1", env=env)