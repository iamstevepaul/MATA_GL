"""
Author: Steve Paul 
Date: 6/2/22 """

from torch import nn
from collections import defaultdict
import warnings
import math
import numpy as np
import gym
from stable_baselines_al import PPO, A2C
# from stable_baselines.common import make_vec_env
from MRTA_Flood_PO_Env import MRTA_Flood_PO_Env
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
from training_config_PO import get_config

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

config = get_config()


env = DummyVecEnv([lambda: MRTA_Flood_PO_Env(
        n_locations = config.n_locations,
        n_agents = config.n_robots,
        max_capacity = config.max_capacity,
        max_range = config.max_range,
        enable_dynamic_tasks=config.enable_dynamic_tasks,
        display = False,
        enable_topological_features = config.enable_topological_features,
        agents_info_exchange_distance_threshold = config.agents_info_exchange_distance_threshold
)])

# n_envs = 4
# env = make_vec_env(mTSPEnv, n_envs=n_envs, env_kwargs={"n_locations":21, "n_agents":5})
# num_cpu = 6
# env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])


policy_kwargs=dict(
    # features_extractor_class=GCAPCNFeatureExtractor,
    features_extractor_kwargs=dict(
        feature_extractor=config.node_encoder,
        features_dim=config.features_dim,
        K=config.K,
        Le=config.Le,
        P=config.P,
        node_dim=env.envs[0].task_graph_node_dim,
        agent_node_dim=env.envs[0].agent_node_dim,
        n_heads=config.n_heads,
        tanh_clipping=config.tanh_clipping,
        mask_logits=config.mask_logits,
        temp=config.temp
))

if config.enable_dynamic_tasks:
    task_type = "D"
else:
    task_type = "ND"

if config.node_encoder == "CAPAM" or config.node_encoder == "MLP":
    tb_logger_location = config.logger+config.problem\
                     + "/" + config.node_encoder + "/" \
                    + config.problem\
                          + "_nloc_" + str(config.n_locations)\
                         + "_nrob_" + str(config.n_robots) + "_" + task_type + "_"\
                         + config.node_encoder\
                         + "_K_" + str(config.K) \
                         + "_P_" + str(config.P) + "_Le_" + str(config.Le) \
                         + "_h_" + str(config.features_dim)
    save_model_loc = config.model_save+config.problem\
                     + "/" + config.node_encoder + "/" \
                    + config.problem\
                          + "_nloc_" + str(config.n_locations)\
                         + "_nrob_" + str(config.n_robots) + "_" + task_type + "_"\
                         + config.node_encoder\
                         + "_K_" + str(config.K) \
                         + "_P_" + str(config.P) + "_Le_" + str(config.Le) \
                         + "_h_" + str(config.features_dim)

model = PPO(
    ActorCriticGCAPSPolicy,
    env,
    gamma=config.gamma,
    verbose=1,
    n_epochs=config.n_epochs,
    batch_size=config.batch_size,
    tensorboard_log=tb_logger_location,
    # create_eval_env=True,
    n_steps=config.n_steps,
    learning_rate=config.learning_rate,
    policy_kwargs = policy_kwargs,
    ent_coef=config.ent_coef,
    vf_coef=config.val_coef
)
model.learn(total_timesteps=config.total_steps)

obs = env.reset()
model.save(save_model_loc)